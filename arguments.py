from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments

@dataclass
class WrappedTrainingArguments(TrainingArguments):
    task:str = field(
        default="argotario",
        metadata={
            "help":"which dataset to use?", 
            "choices": ["argotario", "logic"]}
    )
    model_tag:str = field(
        default="llama2-13bf", 
        metadata={
            "help":"which model to use?", 
            "choices": ['llama2-13b','llama2-13bf', 'gpt-3.5-turbo']}
    )
    quantization:Optional[bool] = field(
        default=True, 
        metadata={"help":"whehter to load quantized model."}
    ) 
    which_gpu:str = field(
        default="4,5",
        metadata={"help":"which gpus to use?"}
    )
    use_dataset_cache: bool = field(default=True, metadata={"help":"use cached dataset."})
    use_argotario_gold: bool = field(default=False, metadata={"help":"Use only the gold data of argotario."})
    max_input_length : Optional[int] = field(default=600, metadata={"help":""})
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
    run_baseline: bool = field(default=False, metadata={"help":""})
    run_multiprompt: bool = field(default=False, metadata={"help":""})
    multipt_start_from: Optional[int] = field(
        default=0, metadata={"help": ""}
    )
    cache_dir: Optional[str] = field(
        default="",
        metadata={"help": "Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)"},
    )
    do_not_save_results: Optional[bool] = field(
        default=False, 
        metadata={"help": ""}
    )
    # n_gpu_layers : Optional[int] = field(
    #     default=10, 
    #     metadata={"help": ""}
    # )
