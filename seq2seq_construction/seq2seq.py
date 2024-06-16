import copy
import os
import torch
from datasets import DatasetDict
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from tqdm import tqdm
from utils.prompt_templates.argotario_prompt import prompt_argotario
from utils.prompt_templates.logic_prompt import prompt_logic
from utils.prompt_templates.elecdebate_prompt import prompt_elecdebate
from utils.prompt_templates.propaganda_prompt import prompt_propaganda
from utils.prompt_templates.mafalda_prompt import prompt_mafalda
from utils.prompt_templates.covid_prompt import prompt_covid
from utils.prompt_templates.reddit_prompt import prompt_reddit

class Constructor(object):
    def __init__(self, args, global_args):
        self.args = args
        self.global_args = global_args
        
    def to_seq2seq(self, raw_datasets: DatasetDict):
        """
        We do not do cache right here but cache after the datasets have been tokenized.
        Here we just format the input by adding prompts to each data instance.
        """
        train_dataset, dev_dataset, test_dataset = None, None, None
        if 'train' in raw_datasets:
            train_dataset = Seq2SeqDataset(self.args, self.global_args, raw_datasets['train'])
        if 'dev' in raw_datasets:
            dev_dataset = Seq2SeqDataset(self.args, self.global_args, raw_datasets['dev'])
        if 'test' in raw_datasets:
            test_dataset = Seq2SeqDataset(self.args, self.global_args, raw_datasets['test'])

        return train_dataset, dev_dataset, test_dataset

class Seq2SeqDataset(Dataset):
    def __init__(self, args, global_args, raw_datasets):
        if global_args.task != 'multi-task':
            task = global_args.task
        else:
            task = args.dataset.name
        if task == 'argotario':
            prompt_func = prompt_argotario
        elif task == 'logic':
            prompt_func = prompt_logic
        elif task == 'elecdebate':
            prompt_func = prompt_elecdebate
        elif task == 'propaganda':
            prompt_func = prompt_propaganda
        elif task == 'mafalda':
            prompt_func = prompt_mafalda
        elif task == 'covid':
            prompt_func = prompt_covid
        elif task == 'reddit':
            prompt_func = prompt_reddit
        self.raw_datasets = raw_datasets
        self.prompted_data = []
        expansion = args.seq2seq.expansion if args.seq2seq.expansion else 1
        
        for expand_id in range(expansion):
            for example in tqdm(self.raw_datasets):
                one_data = copy.deepcopy(example)
                p_example = prompt_func(global_args, one_data)
                self.prompted_data.append(p_example)

    def __getitem__(self, index) -> T_co:
        return self.prompted_data[index]

    def __len__(self):
        return len(self.prompted_data)
