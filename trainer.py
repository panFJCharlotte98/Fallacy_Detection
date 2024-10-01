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
from transformers.deepspeed import is_deepspeed_zero3_enabled
from typing import Optional, Dict, Union, List, Tuple, Any, NamedTuple
from torch.utils.data import Dataset, DataLoader
from transformers.trainer_callback import TrainerState
import time
from torch.cuda.amp import autocast
import math

skip_first_batches = None
# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
SAMPLER_WEIGHTS_NAME = "sampler_weights.pt"

logger = logging.get_logger(__name__)

# For Llama2
B_INST, E_INST = "[INST]", "[/INST]"
# For Llama3
END_OF_TURN, END_INST = "<|eot_id|>", "<|start_header_id|> assistant <|end_header_id|>"

class EvalPrediction(NamedTuple):
    predictions: List[str]
    items: List[dict]

class EvaluateFriendlyTrainer(Trainer):
    def __init__(
        self,
        evaluator,
        *args: WrappedTrainingArguments,
        ignore_pad_token_for_loss: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.evaluator = evaluator
        self.compute_metrics = self._compute_metrics
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.mtask_args = None
        if self.args.exp_args.model.model_tag.startswith("t5"):
            self.mtask_args = self.args.exp_args

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_time: Optional[int] = None,
    ) -> Dict[str, float]:
        self._max_length, self._num_beams  = None, None
        self._max_time = max_time
        if self.mtask_args is not None:
            self._max_length = self.mtask_args.model.generation_max_length
            self._num_beams = self.mtask_args.model.generation_num_beams
            
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()
        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.evaluation_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics
        if eval_dataset is not None and self.compute_metrics is not None:
            eval_preds = self._post_process_function(
                eval_dataset,
                output.predictions,
                "eval_{}".format(math.ceil(self.state.epoch)),
            )
            summary = self.compute_metrics(eval_preds, section="dev", epoch=self.state.epoch)
            output.metrics.update(summary)

        n_samples = len(eval_dataset if eval_dataset is not None else self.eval_dataset)
        output.metrics.update(speed_metrics(metric_key_prefix, start_time, n_samples))
        # Prefix all keys with metric_key_prefix + '_'
        for key in list(output.metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                output.metrics[f"{metric_key_prefix}_{key}"] = output.metrics.pop(key)
        self.log(output.metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)
        return output.metrics


    def predict(
        self,
        test_dataset: Optional[Dataset],
        metric_key_prefix: str = "predict"
    ) -> PredictionOutput:
        self._max_length, self._num_beams  = None, None
        if self.mtask_args is not None:
            self._max_length = self.mtask_args.model.generation_max_length
            self._num_beams = self.mtask_args.model.generation_num_beams
            
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
        eval_preds = self._post_process_function(
            test_dataset, 
            output.predictions, 
            metric_key_prefix
        )
        if self.args.should_evaluate:
            if self.compute_metrics is not None:
                output.metrics.update(self.compute_metrics(eval_preds, section="inference", epoch=None))
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
        if self.args.exp_args.model.model_tag.startswith("t5"):
            gen_kwargs = {
                "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
                "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
                "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
                "no_repeat_ngram_size": 0,  # FIXME: hard coding the no_repeat_ngram_size
            }
            generated_tokens = self.model.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=True,
                **gen_kwargs,
            )
            # # in case the batch is shorter than max length, the output should be padded
            # if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            #     generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])
            with torch.no_grad():
                outputs = model(**inputs)
                if 'labels' in inputs:
                    if self.label_smoother is not None:
                        loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                    else:
                        loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
                else:
                    loss = None
            labels = inputs["labels"]
            # if labels.shape[-1] < gen_kwargs["max_length"]:
            #     labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
            return (loss, generated_tokens, labels)
        else:   
            if self.args.exp_args.model.model_tag.startswith("qwen2.5"):
                gen_kwargs = {
                    "max_new_tokens": self.args.max_new_tokens,
                } 
            if self.args.exp_args.model.model_tag.startswith("mistral"):
                gen_kwargs = {
                    "max_new_tokens": self.args.max_new_tokens,
                    "do_sample" : self.args.do_sample,
                }
            if self.args.exp_args.model.model_tag.startswith("llama2"):
                gen_kwargs = {
                    "max_new_tokens": self.args.max_new_tokens,
                    "do_sample" : self.args.do_sample,
                    "top_p": self.args.top_p,
                    "temperature": self.args.temperature,
                    "use_cache": self.args.use_cache,
                    "top_k" : self.args.top_k,
                    "repetition_penalty": self.args.repetition_penalty,
                    "length_penalty": self.args.length_penalty,
                    "pad_token_id": self.model.tokenizer.pad_token_id
                }
            if self.args.exp_args.model.model_tag.startswith("llama3"):
                terminators = [
                    self.model.tokenizer.eos_token_id,
                    self.model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]
                gen_kwargs = {
                    "max_new_tokens": self.args.max_new_tokens,
                    "eos_token_id": terminators,
                    "do_sample" : self.args.do_sample,
                    "top_p": self.args.top_p,
                    "temperature": self.args.temperature,
                }
            with torch.no_grad():
                generated_tokens = self.model.model.generate(
                    **inputs,
                    **gen_kwargs,
                )
                # extract response in batch
                generated_tokens = generated_tokens[:, inputs.input_ids.shape[1]:]
                
                loss, labels = None, None
                return (loss, generated_tokens, labels)

    def _post_process_function(
        self, 
        dataset: Dataset, 
        predictions: np.ndarray, 
        stage: str
    ) -> EvalPrediction:        
        assert isinstance(dataset, Dataset)
        # # Extract generative LLM's response from the output
        # if not self.args.exp_args.model.model_tag.startswith("t5"):
        #     if self.args.exp_args.model.model_tag.startswith("qwen2.5"):
        #         extracted_predictions = predictions
        #     else:
        #         extracted_predictions = []
        #         for i, one_pred in enumerate(predictions):
        #             input_prompt_len = dataset.examples[i].attrs['input_prompt_len']
        #             if self.args.exp_args.model.model_tag.startswith("llama3"):
        #                 for j, token_id in enumerate(one_pred):
        #                     if (token_id == self.tokenizer.bos_token_id) and (one_pred[j+1] == self.tokenizer.encode('<|start_header_id|>')):
        #                         input_start_index = j
        #                         break
        #             if self.args.exp_args.model.model_tag.split('-')[0] in ["llama2", "mistral"]: # 
        #                 for j, token_id in enumerate(one_pred):
        #                     if (token_id == self.tokenizer.bos_token_id) and (one_pred[j+1] != self.tokenizer.bos_token_id):
        #                         input_start_index = j
        #                         break
        #             response_start_index = input_start_index + input_prompt_len
        #             extracted_predictions.append(one_pred[response_start_index:].tolist())
        # else:
        #     extracted_predictions = predictions
        predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Save locally.
        if (stage == 'predict' or stage.startswith('eval_')) and (self.args.local_rank <= 0) and (not self.args.do_not_save_results):
            if self.args.exp_args.model.model_tag.startswith("t5"):
                save_fn = stage + '_predictions.json'
            else:
                save_fn = 'predictions.json'
            with open(f"{self.args.output_dir}/{save_fn}", "w") as f:
                json.dump(
                    [dict(**{"prediction": " ".join(predictions[idx].split())}, **dataset.examples[idx].attrs) for idx in range(len(predictions))],
                    f,
                    indent=4,
                )
        return EvalPrediction(predictions=predictions, items=[dataset.examples[idx].attrs for idx in range(len(predictions))])

    def _compute_metrics(self, eval_prediction: EvalPrediction, section='predict', epoch=None) -> dict:
        return self.evaluator.evaluate(preds=eval_prediction.predictions, golds=eval_prediction.items, section=section, epoch=epoch)
