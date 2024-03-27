import os
import torch
import numpy as np
from torch import nn
import json
import torch.distributed as dist
from transformers.trainer import Trainer
from arguments import WrappedTrainingArguments
from transformers.utils import logging
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.trainer_utils import PredictionOutput, speed_metrics
from typing import Optional, Dict, Union, List, Tuple, Any, NamedTuple
from torch.utils.data import Dataset, DataLoader
from transformers.trainer_callback import TrainerState
import time

skip_first_batches = None
# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
SAMPLER_WEIGHTS_NAME = "sampler_weights.pt"

logger = logging.get_logger(__name__)

B_INST, E_INST = "[INST]", "[/INST]"

class EvalPrediction(NamedTuple):
    predictions: List[str]
    items: List[dict]

class EvaluateFriendlyTrainer(Trainer):
    def __init__(
        self,
        evaluator,
        *args: WrappedTrainingArguments,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.evaluator = evaluator
        self.compute_metrics = self._compute_metrics
    
    def predict(
        self,
        test_dataset: Optional[Dataset],
        metric_key_prefix: str = "predict"
    ) -> PredictionOutput:
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.evaluation_loop(
                test_dataloader,
                description="Inference",
                metric_key_prefix=metric_key_prefix,
            )
            output.predictions[output.predictions==-100] = self.tokenizer.pad_token_id
            
        finally:
            self.compute_metrics = compute_metrics
        
        eval_preds = self._post_process_function(test_dataset, output.predictions, metric_key_prefix)
        if self.args.should_evaluate:
            if self.compute_metrics is not None:
                output.metrics.update(self.compute_metrics(eval_preds, section="inference"))
            
            #if not self.args.predict_without_evaluation_on_test_dataset:
            output.metrics.update(speed_metrics(metric_key_prefix, start_time, len(test_dataset)))

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(output.metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    output.metrics[f"{metric_key_prefix}_{key}"] = output.metrics.pop(key)

            self.log(output.metrics)

            self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output       
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Customized prediction_step
        Override the original prediction_step to enable predict with generate
        """
        # fj: modify src transformers/trainer.py/Trainer _prepare_input referring to latest version: https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py
        inputs = self._prepare_inputs(inputs)
        gen_kwargs = {
            "max_new_tokens": self.args.max_new_tokens,
            "do_sample" : self.args.do_sample,
            "top_p": self.args.top_p,
            "temperature": self.args.temperature,
            "use_cache": self.args.use_cache,
            "top_k" : self.args.top_k,
            "repetition_penalty": self.args.repetition_penalty,
            "length_penalty": self.args.length_penalty
        }
        with torch.no_grad():
            generated_tokens = self.model.model.generate(
                inputs["input_ids"],
                **gen_kwargs,
            )
            # # print(generated_tokens)
            # temp = generated_tokens >= len(self.tokenizer.get_vocab())
            # print(temp.nonzero())
            # exit()

            # in case the batch is shorter than max length, the output should be padded
            # if generated_tokens.shape[-1] < gen_kwargs["max_new_tokens"]:
            #     generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_new_tokens"])
            # predicti with generate, thereby
            loss, labels = None, None
            return (loss, generated_tokens, labels)

    def _post_process_function(
        self, 
        dataset: Dataset, 
        predictions: np.ndarray, 
        stage: str
    ) -> EvalPrediction:
        def extract_last_output(pred):
            return " ".join(pred.split(E_INST)[-1].strip().replace("\n", " ").split())
            
        assert isinstance(dataset, Dataset)
        predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # for pred in predictions:
        #     print(f"{pred}\n")
        # exit()
        # Save locally.
        if (stage == 'predict' or stage.startswith('eval_')) and (self.args.local_rank <= 0) and (not self.args.do_not_save_results):
            with open(f"{self.args.output_dir}/predictions.json", "w") as f:
                json.dump(
                    [dict(**{"prediction": extract_last_output(predictions[idx])}, **dataset.examples[idx].attrs) for idx in range(len(predictions))],
                    f,
                    indent=4,
                )
        return EvalPrediction(predictions=predictions, items=[dataset.examples[idx].attrs for idx in range(len(predictions))])

    def _compute_metrics(self, eval_prediction: EvalPrediction, section='predict') -> dict:
        return self.evaluator.evaluate(preds=eval_prediction.predictions, golds=eval_prediction.items, section=section)
    