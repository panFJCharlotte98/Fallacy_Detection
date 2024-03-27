from torch.utils.data import Dataset
from collections import Counter
import torch
import json
import math
import os
import numpy as np
import copy
import random
from tqdm import tqdm
import logging
import pandas as pd
from utils.argotario_prompt import prompt_argo
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import numpy as np
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


PROMPTS_MAP = {
    "argotario": prompt_argo
}
logger = logging.getLogger(__name__)
    
class InputFeatures(object):
    def __init__(
        self,
        idx,
        input_ids,
        label=None,
        attrs=None
    ):
        self.idx = idx
        self.input_ids = input_ids
        self.label=label
        self.attrs=attrs

def convert_examples_to_features(idx, js, tokenizer, args):
    # convert the js string (one data sample) to the encoded token
    source_ids, js = PROMPTS_MAP[args.task](args, js, tokenizer)
    label = js['label']
    return InputFeatures(idx, source_ids, label, js)

class TokenizedDataset(Dataset):
    def __init__(
        self, 
        split='eval', 
        args=None, 
        cache_root=None, # cache/argotario
        tokenizer=None,
        examples=None
    ):
        # "use_corrected_data" means using human-labeled data
        # Retrieve data files and set up cache path
        self.split = split
        self.tokenizer = tokenizer
        self.examples = examples
        
        if self.examples is None:
            cache_file = "tokenized_data.cache"
            if args.use_argotario_gold:
                cache_file = "tokenized_data_gold.cache"
            cache_folder = os.path.join(cache_root, args.model_tag) # cache/argotario/llama2-13bf
            if args.run_multiprompt:
                cache_folder = os.path.join(cache_folder, "multi") # cache/argotario/llama2-13bf/multi
                if args.use_argotario_gold:
                    cache_file = f"tokenized_data_gold_round{args.current_round}.cache"
                else:
                    cache_file = f"tokenized_data_round{args.current_round}.cache"
            else:
                if args.run_baseline:
                    cache_folder = os.path.join(cache_folder, "baseline")
                elif args.run_multiprompt:
                    cache_folder = os.path.join(cache_folder, "multi")
                else:
                    cache_folder = os.path.join(cache_folder, "single")
            os.makedirs(cache_folder, exist_ok=True)
            cache_path = os.path.join(cache_folder, cache_file)
            if args.run_multiprompt:
                if args.current_round == 0:
                    self.load_and_cache_data(args, cache_path, args.raw_data_file)
                else:
                    read_data_from = os.path.join(args.last_output_dir, "predictions.json")
                    self.load_and_cache_data(args, cache_path, read_data_from)
            else:
                self.load_and_cache_data(args, cache_path, args.raw_data_file)
    
    def load_and_cache_data(self, args, cache_path, file_path):
        split = self.split
        if args.use_dataset_cache and os.path.exists(cache_path):
            print(f"**** Loading tokenized data from cache in {cache_path} ****")
            self.examples = torch.load(cache_path)#[:16]
            # print(self.examples[0].input_ids)
            # print(self.examples[0].attrs)
        else:
            print(f"**** Generating tokenized {args.task} dataset for {args.model_type} from raw data in {file_path} ****")
            self.examples = []
            i = 0
            data = json.load(open(file_path))
            for js in data:
                if args.use_argotario_gold:
                    if js['is_gold'] == 1:
                        self.examples.append(convert_examples_to_features(i, js, self.tokenizer, args))
                        i += 1
                else:
                    self.examples.append(convert_examples_to_features(i, js, self.tokenizer, args))
                    i += 1
            #self.examples = self.examples[:16]
            if args.use_dataset_cache:
                cached_data = self.examples
                torch.save(cached_data, cache_path)
                print(f"**** Cache tokenized {args.task} dataset to {cache_path} ****")
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        item = {
            'input_ids': torch.tensor(self.examples[i].input_ids),
            #'labels': self.examples[i].label,
            'sample_idx': torch.tensor(i)
        }   
        return item
    


class FormattedDataset(Dataset):
    def __init__(
        self,
        args=None, 
        cache_root=None,
        examples=None
    ):
        self.examples = examples
        
        if self.examples is None:
            cache_file = "formatted_data.json"
            if args.use_argotario_gold:
                cache_file = "formatted_data_gold.json"
            cache_folder = os.path.join(cache_root, args.model_tag) # cache/argotario/gpt3.5-turbo
            if args.run_multiprompt:
                cache_folder = os.path.join(cache_folder, "multi") # cache/argotario/gpt3.5-turbo/multi
                if args.use_argotario_gold:
                    cache_file = f"formatted_data_gold_round{args.current_round}.json"
                else:
                    cache_file = f"formatted_data_round{args.current_round}.json"
            else:
                if args.run_baseline:
                    cache_folder = os.path.join(cache_folder, "baseline")
                else:
                    cache_folder = os.path.join(cache_folder, "single")
            os.makedirs(cache_folder, exist_ok=True)
            cache_path = os.path.join(cache_folder, cache_file)
            if args.run_multiprompt:
                if args.current_round == 0:
                    self.load_and_cache_data(args, cache_path, args.raw_data_file)
                else:
                    read_data_from = os.path.join(args.last_output_dir, "predictions.json")
                    self.load_and_cache_data(args, cache_path, read_data_from)
            else:
                self.load_and_cache_data(args, cache_path, args.raw_data_file)

    def load_and_cache_data(self, args, cache_path, file_path):
        if args.use_dataset_cache and os.path.exists(cache_path):
            print(f"**** Loading formatted data from cache in {cache_path} ****")
            self.examples = json.load(open(cache_path))
        else:
            print(f"**** Generating formatted {args.task} dataset for {args.model_type} from raw data in {file_path} ****")
            self.examples = []
            i = 0
            data = json.load(open(file_path))
            for js in data:
                js['id'] = i
                if args.use_argotario_gold:
                    if js['is_gold'] == 1:
                        self.examples.append(PROMPTS_MAP[args.task](args, js))
                        i += 1
                else:
                    self.examples.append(PROMPTS_MAP[args.task](args, js))
                    i += 1
            if args.use_dataset_cache:
                cached_data = self.examples
                out_file = open(cache_path, "w") 
                json.dump(cached_data, out_file, indent = 4) 
                out_file.close() 
                print(f"**** Cache formatted {args.task} dataset to {cache_path} ****")
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):  
        return self.examples[i]