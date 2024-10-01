import os
import torch
import torch.nn as nn    
from transformers import (
    AutoConfig, T5Config,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig, LlamaForCausalLM, LlamaTokenizer,
    BitsAndBytesConfig,
    T5Config, T5ForConditionalGeneration, T5Tokenizer
)
from transformers.models.llama.convert_llama_weights_to_hf import write_model
from utils.configure import Configure

MODEL_CLASSES = {
    'llama2': (LlamaConfig, LlamaForCausalLM, LlamaTokenizer),
    'llama3': (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
    'mistral': (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
    'qwen2.5': (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
    #'t5': (AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer),
    't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer)
}
LLMs = ['llama2', 'llama3', 'mistral', 'qwen2.5']

class Model(nn.Module):   
    def __init__(self, args):
        super(Model, self).__init__()
        args.model_type = args.exp_args.model.model_tag.split("-")[0]
        tokenizer_kwargs = {}
        kwargs = {}
        if args.model_type in LLMs:
            args.model_size = args.exp_args.model.model_tag.split("-")[1]
            model_id = f"./models/{args.model_type}_hf/{args.model_size}"
        if args.model_type == "t5":
            model_size = args.exp_args.model.model_tag.split("-")[1]
            t5_mtask_args = args.exp_args
            model_id = t5_mtask_args.bert.location
            tokenizer_kwargs = {"model_max_length":t5_mtask_args.model.max_input_length, "use_fast": False}
        
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        config = config_class.from_pretrained(model_id, cache_dir=args.cache_dir if args.cache_dir else None)
        tokenizer = tokenizer_class.from_pretrained(model_id, **tokenizer_kwargs)
        
        if args.model_type in LLMs:
            model_size = int(args.model_size.strip('bf'))
            quantization_config = BitsAndBytesConfig(load_in_8bit = True if model_size >= 13 else False) #llm_int8_threshold=200.0
            # "device_map": accelerator.process_index; 'device_map': 'auto'
            kwargs = {'torch_dtype': torch.bfloat16, 'low_cpu_mem_usage':True} #'quantization_config': quantization_config
  
            if not args.model_type.startswith('qwen'):
                # 1. add a new pad token and resize, increase vocab size from 32K to 32K+1, need to resize the embeddings
                # tokenizer.add_special_tokens({"pad_token": "<PAD>",})
                # 2. more common practice
                tokenizer.pad_token = tokenizer.bos_token
            tokenizer.padding_side = "left"
                      
        model = model_class.from_pretrained(model_id, **kwargs)
        #model.resize_token_embeddings(len(tokenizer))
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        if args.model_type in LLMs:
            self.model.config.pad_token_id = tokenizer.pad_token_id
            self.model.generation_config.pad_token_id = tokenizer.pad_token_id
            self.model.eval()
        # print(model.config)
        print(f"{args.model_type} model from {model_id} loaded successfully...")
        
    def forward(
        self, 
        input_ids, 
        attention_mask, 
        labels, 
    ):
        loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        ).loss
        return {'loss': loss}


