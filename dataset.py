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
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import numpy as np
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from utils.configure import Configure
import datasets
import utils
from utils.format_utils import format_tokens
from utils.prompt_templates import argotario_prompt, logic_prompt, elecdebate_prompt, propaganda_prompt, mafalda_prompt, covid_prompt, reddit_prompt
PROMPTS_MAP = {
    "argotario": argotario_prompt.prompt_argotario,
    "logic": logic_prompt.prompt_logic,
    "elecdebate": elecdebate_prompt.prompt_elecdebate,
    "propaganda": propaganda_prompt.prompt_propaganda,
    "mafalda": mafalda_prompt.prompt_mafalda,
    "covid": covid_prompt.prompt_covid,
    "reddit": reddit_prompt.prompt_reddit
}

logger = logging.getLogger(__name__)
    
class DataItem(object):
    def __init__(
        self,
        idx,
        input_ids,
        attention_mask=None,
        label=None,
        attrs=None
    ):
        self.idx = idx
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.label=label
        self.attrs=attrs

def convert_examples_to_features(idx, js, tokenizer, args):
    # convert the js string (one data sample) to the encoded token
    source_ids, js = PROMPTS_MAP[args.task](args, js, tokenizer)
    label = js['label']
    return DataItem(idx, source_ids, label, js)

def tokenize_input(args, idx, js, tokenizer):
    if args.exp_args.model.model_tag.startswith("t5"):
        t5_mtask_args = args.exp_args
        seq_in = js['seq_in']
        seq_out = js['label'][0].lower()
        # Concatenate description if any.
        if t5_mtask_args.model.use_description and t5_mtask_args.model.concatenate_description:
            seq_in = "{} ; {}".format(js["description"], seq_in)
        tokenized_input = tokenizer(
            text=seq_in,
            #padding="max_length",
            #truncation=True,
            #max_length=t5_mtask_args.model.max_input_length,
            # We found that set it as large as possible can boost the performance significantly
            # , meanwhile, due to the t5 uses a relative position coding, we need to manually
            # assign the max input length into some large numbers, instead of using the "max_model_length"
            # ,which the default is 512, which will hurt the performance a lot.
        )
        tokenized_inferred = tokenizer(
            text=seq_out,
            padding="max_length",
            truncation=True,
            max_length=t5_mtask_args.model.generation_max_length,
            # We set the max_length of "seq_out" during training is the same with the one in inference.
        )
        tokenized_inferred_input_ids = torch.LongTensor(tokenized_inferred.data["input_ids"])
        # Here -100 will let the model not to compute the loss of the padding tokens.
        # "In addition, we must make sure that padding token id’s of the labels (input_ids of target sequence) 
        # are not taken into account by the loss function. In PyTorch and Tensorflow, this can be done 
        # by replacing them with -100, which is the ignore_index of the CrossEntropyLoss." ---hugging face illustration
        tokenized_inferred_input_ids[tokenized_inferred_input_ids == tokenizer.pad_token_id] = -100
        
        label = tokenized_inferred_input_ids
        input_ids = tokenized_input.data["input_ids"]
        attention_mask = tokenized_input.data["attention_mask"]  
    else:
        attention_mask=None
        label = js['label']
        dialog = js['chat_history'] # list of messages
        input_ids = format_tokens(args, [dialog], tokenizer)[0]
        js['input_prompt_len'] = len(input_ids)
        
    return DataItem(idx, input_ids, attention_mask, label, js)

def generate_t5_seq2seq_datasets(args, split, cache_folder):
    #seq2seq_cache_path = os.path.join("/".join(cache_path.split("/")[:-1]),  "seq2seq_" + cache_path.split("/")[-1])
    seq2seq_cache_path = os.path.join(cache_folder, 'seq2seq_datasets.cache')
    
    if args.use_dataset_cache and os.path.exists(seq2seq_cache_path):
        seq2seq_datasets = torch.load(seq2seq_cache_path)
        return seq2seq_datasets[split]
    else:
        meta_tuning_data = {}
        if args.exp_args.model.do_multitask:
            for task, cfg_path in args.exp_args.arg_paths:
                if task in args.active_task_list:
                    task_args = Configure.Get(cfg_path)
                    task_args.bert = args.exp_args.bert
                    data_files = {sp: task_args.dataset.load_from + f"{sp}.json" for sp in ['train', 'dev', 'test']}
                    task_raw_datasets_split: datasets.DatasetDict = datasets.load_dataset('json', data_files=data_files)
                    task_seq2seq_dataset_split: tuple = utils.tool.get_constructor(task_args.seq2seq.constructor)(task_args, args).\
                        to_seq2seq(task_raw_datasets_split) 
                    meta_tuning_data[cfg_path] = task_seq2seq_dataset_split
        else:
            task_args = Configure.Get(args.task_arg_path)
            task_args.bert = args.exp_args.bert
            data_files = {sp: task_args.dataset.load_from + f"{sp}.json" for sp in ['train', 'dev', 'test']}
            task_raw_datasets_split: datasets.DatasetDict = datasets.load_dataset('json', data_files=data_files)
            task_seq2seq_dataset_split: tuple = utils.tool.get_constructor(task_args.seq2seq.constructor)(task_args, args).\
                to_seq2seq(task_raw_datasets_split)
            meta_tuning_data[args.task_arg_path] = task_seq2seq_dataset_split
        seq2seq_dataset_split: tuple = utils.tool.get_constructor(args.exp_args.seq2seq.constructor)(args.exp_args).to_seq2seq(meta_tuning_data)
        seq2seq_train_dataset, seq2seq_dev_dataset, seq2seq_test_dataset = None, None, None 
        if len(seq2seq_dataset_split) == 2:
            seq2seq_train_dataset, seq2seq_dev_dataset = seq2seq_dataset_split
        elif len(seq2seq_dataset_split) == 3:
            seq2seq_train_dataset, seq2seq_dev_dataset, seq2seq_test_dataset = seq2seq_dataset_split
        else:
            raise ValueError("Other split not support yet.")
        seq2seq_datasets = {'train': seq2seq_train_dataset, 'dev': seq2seq_dev_dataset, 'test': seq2seq_test_dataset}
        torch.save(seq2seq_datasets, seq2seq_cache_path)
        print(f"**** Cache seq2seq {args.task}-{split} dataset to {seq2seq_cache_path} ****")
        return seq2seq_datasets[split]
    
def generate_llm_seq2seq_datasets(args, cache_path):
    seq2seq_cache_path = os.path.join("/".join(cache_path.split("/")[:-1]),  "seq2seq_" + cache_path.split("/")[-1])
    if args.use_dataset_cache and os.path.exists(seq2seq_cache_path):
        return torch.load(seq2seq_cache_path)
    else:
        def prepare_task_seq2seq_datasets(args):
            task_args = args.task_args
            data_files = {args.split: task_args.dataset.load_from + f"{args.split}.json"}
            #data_files = {'test': task_args.dataset.load_from + "test_toy.json"}
            task_raw_datasets_split: datasets.DatasetDict = datasets.load_dataset('json', data_files=data_files)
            task_seq2seq_dataset_split: tuple = utils.tool.get_constructor(task_args.seq2seq.constructor)(task_args, args).to_seq2seq(task_raw_datasets_split) 
            seq2seq_dataset_split = {'train':task_seq2seq_dataset_split[0], 'dev':task_seq2seq_dataset_split[1], 'test':task_seq2seq_dataset_split[2]}
            return seq2seq_dataset_split
        if args.exp_args.model.run_multiprompt:
            if args.current_round == 0:
                seq2seq_test_dataset = prepare_task_seq2seq_datasets(args)[args.split]
            else:
                data_files = {'test': os.path.join(args.last_output_dir, "predictions.json")}
                seq2seq_dataset_split: tuple = utils.tool.get_constructor(args.task_args.seq2seq.constructor)(args.task_args, args).\
                    to_seq2seq(datasets.load_dataset('json', data_files=data_files)) 
                _, _, seq2seq_test_dataset = seq2seq_dataset_split
                seq2seq_test_dataset = seq2seq_test_dataset
        else:
            seq2seq_test_dataset = prepare_task_seq2seq_datasets(args)[args.split]
        if args.use_dataset_cache:
            torch.save(seq2seq_test_dataset, seq2seq_cache_path)
            print(f"**** Cache seq2seq {args.task}-{args.split} dataset to {seq2seq_cache_path} ****")
        return seq2seq_test_dataset

class TokenizedDataset(Dataset):
    def __init__(
        self,
        args,
        tokenizer,
        split='test',
    ):
        self.model_tag = args.exp_args.model.model_tag
        self.tokenizer = tokenizer
        self.split = split
        #self.seq2seq_dataset = seq2seq_dataset
        cache_file = f"{split}_tokenized_cwindow{args.context_window}_data.cache"
        cache_folder = os.path.join(args.cache_root, self.model_tag) # cache/argotario/llama2-13bf
        if args.exp_args.model.run_multiprompt:
            cache_folder = os.path.join(cache_folder, "multi") # cache/argotario/llama2-13bf/multi
            cache_file = f"tokenized_cwindow{args.context_window}_data_round{args.current_round}.cache"
        else:
            if args.exp_args.model.run_baseline:
                cache_folder = os.path.join(cache_folder, "baseline")
            elif args.exp_args.model.run_multiprompt:
                cache_folder = os.path.join(cache_folder, "multi")
            else:
                cache_folder = os.path.join(cache_folder, "single")
        cache_folder = os.path.join(cache_folder, args.scheme)
        os.makedirs(cache_folder, exist_ok=True)
        cache_path = os.path.join(cache_folder, cache_file)

        self.load_and_cache_data(args, cache_folder, cache_path)
    
    def load_and_cache_data(self, args, cache_folder, cache_path):
        if args.use_dataset_cache and os.path.exists(cache_path):
            print(f"**** Loading tokenized {self.split} data from cache in {cache_path} ****")
            self.examples = torch.load(cache_path)#[:16]
            # print(self.examples[0].input_ids)
            # print(self.examples[0].attrs)
        else:
            print(f"**** Generating tokenized {args.task}-{self.split} dataset for {args.model_type} from prompted seq2seq data ****")
            if args.exp_args.model.model_tag.startswith("t5"):
                seq2seq_dataset = generate_t5_seq2seq_datasets(args, self.split, cache_folder)
            else:
                seq2seq_dataset = generate_llm_seq2seq_datasets(args, cache_path)
            
            self.examples = []
            i = 0
            for item in seq2seq_dataset:
                self.examples.append(tokenize_input(args, i, item, self.tokenizer))
                i += 1
            #self.examples = self.examples[:50]
            if args.use_dataset_cache:
                cached_data = self.examples
                torch.save(cached_data, cache_path)
                print(f"**** Cache tokenized {args.task}-{self.split} dataset to {cache_path} ****")
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        if self.model_tag.startswith("t5"):
            item = {
                'input_ids': torch.LongTensor(self.examples[i].input_ids),
                'attention_mask': torch.LongTensor(self.examples[i].attention_mask),
                'labels': self.examples[i].label
            }   
        else:
            item = {
                'input_ids': torch.LongTensor(self.examples[i].input_ids),
            } 
        return item
    

class FormattedDataset(Dataset):
    def __init__(
        self,
        args,
    ):
        cache_root = args.cache_root
        cache_file = f"formatted_cwindow{args.context_window}_data.json"
        task_args = Configure.Get(args.task_arg_path)
        args.raw_data_file = os.path.join(task_args.dataset.load_from, "test.json")
        cache_folder = os.path.join(cache_root, args.exp_args.model.model_tag) # cache/argotario/gpt3.5-turbo
        
        if args.exp_args.model.run_multiprompt:
            cache_folder = os.path.join(cache_folder, "multi") # cache/argotario/gpt3.5-turbo/multi
            cache_file = f"formatted_cwindow{args.context_window}_data_round{args.current_round}.json"
        else:
            if args.exp_args.model.run_baseline:
                cache_folder = os.path.join(cache_folder, "baseline")
            else:
                cache_folder = os.path.join(cache_folder, "single")
        cache_folder = os.path.join(cache_folder, args.scheme)
        os.makedirs(cache_folder, exist_ok=True)
        cache_path = os.path.join(cache_folder, cache_file)

        if args.exp_args.model.run_multiprompt:
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
            print(f"**** Generating formatted {args.task} dataset for {args.exp_args.model.model_tag} from raw data in {file_path} ****")
            self.examples = []
            data = json.load(open(file_path))
            for js in data:
                self.examples.append(PROMPTS_MAP[args.task](args, js))
            #self.examples = self.examples[:50]
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