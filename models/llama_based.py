import os
import torch
import torch.nn as nn    
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig, LlamaForCausalLM, LlamaTokenizer,
    BitsAndBytesConfig
)
from transformers.models.llama.convert_llama_weights_to_hf import write_model



MODEL_CLASSES = {
    # 'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    # 'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    # 'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    # 'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    # 'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    # 'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
    'llama2': (LlamaConfig, LlamaForCausalLM, LlamaTokenizer)
}
class Model(nn.Module):   
    def __init__(self, args):
        super(Model, self).__init__()
        args.model_type = args.model_tag.split("-")[0]
        args.model_size = args.model_tag.split("-")[1]
        if args.model_type == "llama2":
            model_id = f"./models/llama2_hf/{args.model_size}"
            
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        config = config_class.from_pretrained(
            model_id,
            cache_dir=args.cache_dir if args.cache_dir else None)
        tokenizer = tokenizer_class.from_pretrained(model_id)
        
        if args.model_type == "llama2":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True) #llm_int8_threshold=200.0
            #{"": accelerator.process_index} 'device_map': 'auto'
            kwargs = {'torch_dtype':torch.float16, 'low_cpu_mem_usage':True, 'quantization_config': quantization_config}
            # 1. add a new pad token and resize, increase vocab size from 32K to 32K+1
            #tokenizer.add_special_tokens({"pad_token": "<PAD>",})
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
        model.config.pad_token_id = tokenizer.pad_token_id
        # print(model.config)
        # exit()
        self.model = model
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
        self.model.eval()
        print(f"{args.model_type} model from {model_id} loaded successfully...")
    
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

