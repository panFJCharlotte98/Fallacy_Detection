import os
import torch
import torch.nn as nn    
from transformers import (
    AutoConfig, T5Config,
    AutoModelForSeq2SeqLM,
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
    #'t5': (AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer),
    't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer)
}
LLMs = ['llama2', 'llama3', 'mistral']

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
            # if model_size == '3b':
            #     model_id = "local model path"
            # else:
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
            if model_size < 13: 
                # for llama2-7bf
                # quantization_config = BitsAndBytesConfig(
                #     load_in_4bit=True,
                #     bnb_4bit_use_double_quant=True,
                #     bnb_4bit_quant_type="nf4",
                #     bnb_4bit_compute_dtype=torch.bfloat16
                # )  
                kwargs = {'torch_dtype':torch.bfloat16, 'low_cpu_mem_usage':True}
            # 1. add a new pad token and resize, increase vocab size from 32K to 32K+1, need to resize the embeddings
            # tokenizer.add_special_tokens({"pad_token": "<PAD>",})
            # 2. more common practice
            tokenizer.pad_token = tokenizer.bos_token
            tokenizer.padding_side = "left"
            # print(repr(tokenizer.pad_token))
            # print(repr(tokenizer.bos_token))
            # print(repr(tokenizer.eos_token))
            # print(f"pad_token_id={tokenizer.pad_token_id}")
            # print(f"vocab length={len(tokenizer.get_vocab())}")
            # print(tokenizer)
            # exit()        
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
        # exit()
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
    
# Function to load the main model for text generation
def load_model(model_name, quantization):
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        load_in_8bit=quantization,
        device_map="auto",
        low_cpu_mem_usage=True,

    )
    return model

class BaseModel:
    def predict(self, x, max_length: int = 4096, device: str = "cpu"):
        with torch.no_grad():
            if self.device is None:
                self.device = device
                self.model.to(device)

            tokenized = self.tokenizer(x, return_tensors="pt")
            generate_ids = self.model.generate(
                tokenized.input_ids.to(device), max_length=max_length
            )
            output = self.tokenizer.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            return output


class LLaMAModel(BaseModel):
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        model_size: str = "7B",
    ) -> None:
        if not os.path.exists(output_dir):
            write_model(
                output_dir, os.path.join(input_dir, model_size), model_size=model_size
            )
        self.model_name = "LLaMA"
        self.device = None
        self.tokenizer = LlamaTokenizer.from_pretrained(f"{input_dir}/tokenizer.model")
        self.model = LlamaForCausalLM.from_pretrained(output_dir)
        self.model.eval()


class LLaMAModelQuantized(BaseModel):
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        model_size: str = "7B",
    ) -> None:
        if not os.path.exists(output_dir):
            write_model(
                output_dir, os.path.join(input_dir, model_size), model_size=model_size
            )
        self.model_name = "LLaMA"
        self.device = None
        self.tokenizer = LlamaTokenizer.from_pretrained(f"{input_dir}/tokenizer.model")
        self.model = LlamaForCausalLM.from_pretrained(output_dir, load_in_8bit=True)
        self.model.eval()

    def predict(self, x, max_length: int = 4096):
        with torch.no_grad():
            tokenized = self.tokenizer(x, return_tensors="pt")
            generate_ids = self.model.generate(
                tokenized.input_ids.to("cuda:0"), max_length=max_length
            )
            output = self.tokenizer.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            return output


class VicunaModel(BaseModel):
    def __init__(self, model_size: str = "7B") -> None:
        self.model_name = "Vicuna"
        self.device = None
        if model_size == "7B":
            self.tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.3")
            self.model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.3")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-13b-v1.3")
            self.model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-13b-v1.3")
        self.model.eval()


class AlpacaModel(BaseModel):
    def __init__(self, model_size: str = "7B") -> None:
        self.model_name = "Alpaca"
        self.device = None
        if model_size == "7B":
            self.tokenizer = AutoTokenizer.from_pretrained("chavinlo/alpaca-native")
            self.model = AutoModelForCausalLM.from_pretrained("chavinlo/alpaca-native")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("chavinlo/alpaca-13b")
            self.model = AutoModelForCausalLM.from_pretrained("chavinlo/alpaca-13b")
        self.model.eval()


class MPTModel(BaseModel):
    def __init__(self, model_size: str = "7B") -> None:
        self.model_name = "MPT"
        self.device = None
        if model_size == "7B":
            mpt_model = "mosaicml/mpt-7b"
        else:
            mpt_model = "mosaicml/mpt-13b"

        config = AutoConfig.from_pretrained(mpt_model, trust_remote_code=True)
        config.max_seq_len = 4096
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.model = AutoModelForCausalLM.from_pretrained(
            mpt_model, trust_remote_code=True
        )
        self.model.eval()


class DollyModel(BaseModel):
    def __init__(self, model_size: str = "7B") -> None:
        self.model_name = "Dolly"
        self.device = None
        if model_size == "7B":
            self.tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-7b")
            self.model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-7b")
        elif model_size == "6B":
            self.tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v1-6b")
            self.model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v1-6b")
        elif model_size == "3B":
            self.tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-3b")
            self.model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-3b")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-12b")
            self.model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-12b")
        self.model.eval()


if __name__ == "__main__":
    model = LLaMAModel("LLaMA/", "Converted_Llama/", model_size="7B")

