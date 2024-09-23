from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments

@dataclass
class WrappedTrainingArguments(TrainingArguments):
    cfg: str = None
    regen_results_to:Optional[str] = field(
        default="", 
        metadata={"help": ""}
    )
    which_task:str = field(
        default="all", 
        metadata={"help":"what tasks (datasets) to run.",}
    )
    scheme:str = field(
        default="", 
        metadata={"help":"experiment scheme, used for the selection prompt formats and result saving",}
    )
    split: str = field(
        default="test", 
        metadata={"help":"which dataset split to run inference on",}
    )
    context_window: int = field(
        default=2, 
        metadata={"help":"context window for propaganda."}
    )
    context_window_elecdebate: int = field(
        default=2, 
        metadata={"help":"context window for elecdebate."}
    )
    load_weights_from: Optional[str] = field(
        default=None, 
        metadata={"help": "The directory to load the model weights from."}
    )
    do_not_save_results: Optional[bool] = field(
        default=False, 
        metadata={"help": ""}
    )
    use_dataset_cache: bool = field(
        default=False, 
        metadata={"help":"use cached dataset."}
    )
    quantization:Optional[bool] = field(
        default=True, 
        metadata={"help":"whehter to load quantized model."}
    ) 
    # --------------------- Generation-specific args ---------------------- #
    max_new_tokens: Optional[int] = field(
        default=128, 
        metadata={"help":"The maximum numbers of tokens to generate"}
    )
    do_sample: Optional[bool] = field(
        default=True, 
        metadata={"help": "Whether or not to use sampling ; use greedy decoding otherwise."}
    )
    use_cache: Optional[bool] = field(
        default=True, 
        metadata={"help": "[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding."}
    )
    top_p: Optional[float] = field(
        default=0.9, 
        metadata={"help": "[optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.\
                  When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens"}
    )
    temperature : Optional[float] = field(
        default=0.75, 
        metadata={"help": "[optional] The value used to modulate the next token probabilities.\
                  Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value."}
    )
    top_k : Optional[int] = field(
        default=50, 
        metadata={"help": "[optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.\
                  When decoding text, samples from the top k most likely tokens; lower to ignore less likely tokens"}
    )
    repetition_penalty: Optional[float] = field(
        default=1.0, 
        metadata={"help": "The parameter for repetition penalty. 1.0 means no penalty."}
    )
    length_penalty: Optional[int] = field(
        default=1, 
        metadata={"help": "[optional] Exponential penalty to the length that is used with beam-based generation."}
    )
    # --------------------- Generation-specific args ---------------------- #
    cache_dir: Optional[str] = field(
        default="",
        metadata={"help": "Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)"},
    )
    
