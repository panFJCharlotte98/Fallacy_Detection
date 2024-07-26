import os
import sys
import copy
#os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
import datasets
import importlib
import time
import random
import logging
import numpy as np
import json
from tqdm import tqdm
import multiprocessing
from typing import Dict
from datetime import datetime
from arguments import WrappedTrainingArguments
import torch
torch.autograd.set_detect_anomaly(True)
import transformers
from transformers.trainer_utils import get_last_checkpoint
#torch.cuda.set_sync_debug_mode(1)
#torch.backends.cudnn.benchmark = True
from transformers import (
    HfArgumentParser,
    EarlyStoppingCallback,
    WEIGHTS_NAME,
)
from transformers.data.data_collator import DataCollatorWithPadding
# local import
from models.transformers_based import Model
from models.gpt_based import do_inference
from dataset import TokenizedDataset, FormattedDataset
from trainer import EvaluateFriendlyTrainer
from utils.configure import Configure
import utils.tool
# try:
#     from torch.utils.tensorboard import SummaryWriter
# except:
#     from tensorboardX import SummaryWriter

TASK_LIST = ["argotario", "logic", "reddit", "elecdebate", "propaganda", "mafalda", "covid"]
from utils.prompt_templates.argotario_prompt import argotario_multiround_prompts
from utils.prompt_templates.logic_prompt import logic_multiround_prompts
from utils.prompt_templates.elecdebate_prompt import elecdebate_multiround_prompts
from utils.prompt_templates.propaganda_prompt import propaganda_multiround_prompts
from utils.prompt_templates.mafalda_prompt import mafalda_multiround_prompts
from utils.prompt_templates.covid_prompt import covid_multiround_prompts
from utils.prompt_templates.reddit_prompt import reddit_multiround_prompts

TASK_N_ROUNDS = {
    "argotario": argotario_multiround_prompts,
    "logic": logic_multiround_prompts,
    "elecdebate": elecdebate_multiround_prompts,
    "propaganda": propaganda_multiround_prompts,
    "mafalda": mafalda_multiround_prompts,
    "covid": covid_multiround_prompts,
    "reddit": reddit_multiround_prompts
}

logger = logging.getLogger(__name__)
cpu_cont = multiprocessing.cpu_count()

# EVALUATOR_TOOL = {
#     "argotario": 'evaluate.argotario_evaluator',
#     "logic": 'evaluate.logic_evaluator',
#     "elecdebate": 'evaluate.elecdebate_evaluator',
#     "propaganda": 'evaluate.propaganda_evaluator',
#     "multi-task": 'evaluate.meta_evaluator'
# }
# def get_evaluator(evaluate_tool):
#     EvaluateTool = importlib.import_module('{}'.format(evaluate_tool)).EvaluateTool
#     return EvaluateTool

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
                   
def prepare_output_dir(args):
    gen_output_dir = True
    if args.exp_args.model.multipt_start_from > 0:
        gen_output_dir = False
    if gen_output_dir:
        # Set up result reporting dir 
        if args.local_rank <= 0:
            dict_args = vars(args)
            if not args.exp_args.model.model_tag.startswith('t5'):
                args_used_in_name = [
                    ['max_new_tokens','len'],
                    ['seed', 'seed'],
                    ['per_device_eval_batch_size','ebs'],
                ]
                setting = 'fewshot'
                if args.exp_args.model.run_multiprompt:
                    setting = "multipt"
                if args.exp_args.model.run_baseline:
                    setting = 'baseline'
                folder_name = [f"{setting}_{args.scheme}_GPU-{args.world_size}"]
                if args.task in ['propaganda', 'elecdebate']:
                    folder_name = [f"{setting}_{args.scheme}_{args.context_window}_GPU-{args.world_size}"]
            else:
                if not args.do_train:
                    args_used_in_name = [
                        ['seed', 'seed'],
                        ['per_device_eval_batch_size','ebs']]
                else:
                    args_used_in_name = [
                        ['seed', 'seed'],
                        ['optim','optim'],
                        ['learning_rate','lr'],
                        ['num_train_epochs', 'ep'],
                        ['gradient_accumulation_steps', 'gas'],
                        ['per_device_train_batch_size','tbs'],
                        ['per_device_eval_batch_size','ebs'],
                    ]
                folder_name = [f"GPU-{args.world_size}"]
                setting = 'baseline'
            for arg_name, rename in args_used_in_name:
                folder_name.append('{}-{}'.format(rename, dict_args[arg_name]))
            sys_dt = datetime.now().strftime("%Y%m%d%H%M%S")
            folder_name_no_date = copy.deepcopy(folder_name[:-1])
            folder_name.append(sys_dt)
            folder_name = '_'.join(folder_name)
            output_dir = os.path.join(args.output_dir, args.exp_args.model.model_tag, args.task, folder_name)
            if args.exp_args.model.run_multiprompt:
                output_dir = os.path.join(output_dir, "round_0")
        else:
            output_dir, setting, folder_name_no_date = "", "", []
        
        output_dir, setting, folder_name_no_date = [output_dir], [setting], [folder_name_no_date]
        torch.distributed.broadcast_object_list(folder_name_no_date, src=0, device=args.device)#dist.send(param.data, dst=sibling)
        folder_name_no_date = folder_name_no_date[0]
        
        if (args.regen_results_to == "") and (not args.exp_args.model.model_tag.startswith('t5')) and (args.output_dir != "./results/test/"):
            folder_name_no_date = '_'.join(folder_name_no_date)
            run_dir = os.path.join(args.output_dir, args.exp_args.model.model_tag, args.task)
            os.makedirs(run_dir,exist_ok=True)
            for fd in os.listdir(run_dir):
                if "_".join(fd.split("_")[:-2]) == folder_name_no_date:
                    run_fd = os.path.join(run_dir, fd)
                    if ('result.json' in os.listdir(run_fd)) or ('predict_results.json' in os.listdir(run_fd)):
                        print("Detected existing experiment records, skip this run.")
                        sys.exit(0)
        
        torch.distributed.broadcast_object_list(output_dir, src=0, device=args.device)#dist.send(param.data, dst=sibling)
        torch.distributed.broadcast_object_list(setting, src=0, device=args.device)#dist.send(param.data, dst=sibling)
        args.output_dir, args.setting = output_dir[0], setting[0]
    else:
        args.output_dir = os.path.join(args.output_dir, "round_"+str(args.exp_args.model.multipt_start_from))
    
    if args.local_rank > 0: ## Barrier to make sure only the first process in distributed training download model & vocab
        torch.distributed.barrier()
    a = 1
    if args.local_rank == 0:
        torch.distributed.barrier()  
    if args.regen_results_to == "":
        os.makedirs(args.output_dir, exist_ok=True)
    # by default
    args.log_dir = args.output_dir
    return args

def setup_wandb(args):
    args.run_name = "_".join([args.task, args.exp_args.model.model_tag])
    if "wandb" in args.report_to and args.local_rank <= 0:
        print("start wandb...")
        import wandb
        init_args = {}
        if "MLFLOW_EXPERIMENT_ID" in os.environ:
            init_args["group"] = os.environ["MLFLOW_EXPERIMENT_ID"]
        # Get system's datetime
        sys_dt = datetime.now().strftime("%Y%m%d%H%M%S")
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "fallacy"),
            name='{}_{}'.format(args.run_name, sys_dt),
            entity=os.getenv("WANDB_ENTITY", 'fengjunp-nus'),
            **init_args,
        )
        wandb.config.update(args, allow_val_change=True)

def setup_logging(args):
    #------------------------------- Set up logging --------------------------------#
    # Initialize the logger
    if args.local_rank > 0: ## Barrier to make sure only the first process in distributed training download model & vocab
        torch.distributed.barrier()
    setup_wandb(args)  
    # Reset logging handler
    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    #log_file_name = '{}-{}.log'.format('run',datetime.now().strftime("%Y%m%d%H%M%S"))
    log_file_name = 'run.log'
    logging.basicConfig(filename=os.path.join(args.log_dir, log_file_name),
                        filemode='a',
                        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank <= 0 else logging.WARN)
    logger.info("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16)
    if args.local_rank == 0:
        torch.distributed.barrier()  
    # Set seed
    set_seed(args.seed)    

# def prepare_t5_datasets(args, tokenizer):
#     meta_tuning_data = {}
#     if args.exp_args.model.do_multitask:
#         for task, cfg_path in args.exp_args.arg_paths:
#             task_args = Configure.Get(cfg_path)
#             task_args.bert = args.exp_args.bert
#             data_files = {sp: task_args.dataset.load_from + f"{sp}.json" for sp in ['train', 'dev', 'test']}
#             task_raw_datasets_split: datasets.DatasetDict = datasets.load_dataset('json', data_files=data_files)
#             task_seq2seq_dataset_split: tuple = utils.tool.get_constructor(task_args.seq2seq.constructor)(task_args, args).\
#                 to_seq2seq(task_raw_datasets_split) 
#             meta_tuning_data[cfg_path] = task_seq2seq_dataset_split
#     else:
#         task_args = Configure.Get(args.task_arg_path)
#         task_args.bert = args.exp_args.bert
#         data_files = {sp: task_args.dataset.load_from + f"{sp}.json" for sp in ['train', 'dev', 'test']}
#         task_raw_datasets_split: datasets.DatasetDict = datasets.load_dataset('json', data_files=data_files)
#         task_seq2seq_dataset_split: tuple = utils.tool.get_constructor(task_args.seq2seq.constructor)(task_args, args).\
#             to_seq2seq(task_raw_datasets_split)
#         meta_tuning_data[args.task_arg_path] = task_seq2seq_dataset_split
#     seq2seq_dataset_split: tuple = utils.tool.get_constructor(args.exp_args.seq2seq.constructor)(args.exp_args).to_seq2seq(meta_tuning_data)
#     seq2seq_train_dataset, seq2seq_dev_dataset, seq2seq_test_dataset = None, None, None 
#     if len(seq2seq_dataset_split) == 2:
#         seq2seq_train_dataset, seq2seq_dev_dataset = seq2seq_dataset_split
#     elif len(seq2seq_dataset_split) == 3:
#         seq2seq_train_dataset, seq2seq_dev_dataset, seq2seq_test_dataset = seq2seq_dataset_split
#     else:
#         raise ValueError("Other split not support yet.")
#     train_dataset = TokenizedDataset(args, tokenizer, seq2seq_train_dataset, split='train') if seq2seq_train_dataset else None
#     eval_dataset = TokenizedDataset(args, tokenizer, seq2seq_dev_dataset, split='dev') if seq2seq_dev_dataset else None
#     test_dataset = TokenizedDataset(args, tokenizer, seq2seq_test_dataset, split='test') if seq2seq_test_dataset else None
#     return train_dataset, eval_dataset, test_dataset

# def prepare_llm_dataset(args, tokenizer):
#     def prepare_task_seq2seq_datasets(args):
#         task_args = Configure.Get(args.task_arg_path)
#         args.task_args = task_args
#         data_files = {sp: task_args.dataset.load_from + f"{sp}.json" for sp in ['train', 'dev', 'test']}
#         #data_files = {'test': task_args.dataset.load_from + "test_toy.json"}
#         task_raw_datasets_split: datasets.DatasetDict = datasets.load_dataset('json', data_files=data_files)
#         task_seq2seq_dataset_split: tuple = utils.tool.get_constructor(task_args.seq2seq.constructor)(task_args, args).to_seq2seq(task_raw_datasets_split) 
#         seq2seq_dataset_split = {'train':task_seq2seq_dataset_split[0], 'dev':task_seq2seq_dataset_split[1], 'test':task_seq2seq_dataset_split[2]}
#         return seq2seq_dataset_split, args
#     if args.exp_args.model.run_multiprompt:
#         if args.current_round == 0:
#             seq2seq_test_dataset, args = prepare_task_seq2seq_datasets(args)[args.split]
#         else:
#             data_files = {'test': os.path.join(args.last_output_dir, "predictions.json")}
#             seq2seq_dataset_split: tuple = utils.tool.get_constructor(args.task_args.seq2seq.constructor)(args.task_args, args).\
#                 to_seq2seq(datasets.load_dataset('json', data_files=data_files)) 
#             _, _, seq2seq_test_dataset = seq2seq_dataset_split
#             seq2seq_test_dataset = seq2seq_test_dataset
#     else:
#         seq2seq_test_dataset, args = prepare_task_seq2seq_datasets(args)[args.split]
            
#     test_dataset = TokenizedDataset(args, tokenizer, seq2seq_test_dataset, split=args.split)
#     return test_dataset

def run(args, model=None, evaluator=None):
    if not args.logging_set:
        setup_logging(args)
    if (evaluator is None) and (args.should_evaluate):
        evaluator = utils.tool.get_evaluator(args.exp_args.evaluate.tool)(args)
    if model is None:
        model = Model(args)
    
    if args.local_rank > 0: ## Barrier to make sure only the first process in distributed training download model & vocab
        torch.distributed.barrier()  
    
    if args.exp_args.model.model_tag.startswith("t5"):
        train_dataset, eval_dataset = None, None
        if args.do_train:
            train_dataset = TokenizedDataset(args, model.tokenizer, split='train')
            eval_dataset = TokenizedDataset(args, model.tokenizer, split='dev')
        test_dataset = TokenizedDataset(args, model.tokenizer, split='test')
        early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=args.exp_args.seq2seq.patience if args.exp_args.seq2seq.patience else 5)
    else:
        train_dataset, eval_dataset, early_stopping_callback = None, None, None
        args.task_args = Configure.Get(args.task_arg_path)
        #test_dataset = prepare_llm_dataset(args, model.tokenizer)
        test_dataset = TokenizedDataset(args, model.tokenizer, split=args.split)
    #exit()    
    #------------------------------- Create Trainer --------------------------------#
    data_collator = DataCollatorWithPadding(model.tokenizer, padding="longest")
    trainer = EvaluateFriendlyTrainer(
        args=args,
        evaluator=evaluator,
        model=model,
        tokenizer=model.tokenizer,
        data_collator = data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[early_stopping_callback] if early_stopping_callback is not None else None,
    )
    logger.info('***** Trainer built successfully. ***** \n')
    if args.local_rank == 0:
        torch.distributed.barrier()  
    # End of barrier to make sure only the first process in distributed training download model & vocab
    
    if args.exp_args.model.model_tag.startswith("t5"):
        # Load model weights (for --do_train=False or post finetuning).
        if args.load_weights_from:
            state_dict = torch.load(os.path.join(args.load_weights_from, transformers.WEIGHTS_NAME), map_location="cpu")
            trainer.model.load_state_dict(state_dict, strict=True)
            print("***** Load the previous checkpoint. *****\n")
            logger.info("***** Load the previous checkpoint. *****\n")
            # release memory
            del state_dict
    # Training
    if args.do_train:
        # Detect last checkpoint and check whether to train from scratch or to train from last checkpoint
        last_checkpoint = None
        if os.path.isdir(args.output_dir) and not args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(args.output_dir)
            if last_checkpoint is None and len(os.listdir(args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None and args.resume_from_checkpoint is None:
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )
        checkpoint = None
        if args.resume_from_checkpoint is not None:
            checkpoint = args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        start_time = time.time()
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics
        max_train_samples = len(train_dataset)
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        train_time = start_time - time.time()
        logger.info(f"train_time = {train_time}")
    # Evaluation
    if args.do_eval:
        start_time = time.time()
        logger.info("***** Evaluate *****")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        max_eval_samples = len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        eval_time = start_time - time.time()
        logger.info(f"eval_time = {eval_time}")
    # Predict
    if args.do_predict:
        logger.info("***** Predict *****")
        test_outputs = trainer.predict(
            test_dataset=test_dataset #if test_dataset else eval_dataset
        )
        if args.should_evaluate:
            metrics = test_outputs.metrics
            metrics["predict_samples"] = len(test_dataset)
            trainer.log_metrics("predict", metrics)
            if not args.do_not_save_results:
                trainer.save_metrics("predict", metrics)
    
    if not args.exp_args.model.model_tag.startswith('llama'):
        del model
    del evaluator
    return

def run_gpt(args):
    args.model_type = 'gpt'
    args.task_args = Configure.Get(args.task_arg_path)
    if args.exp_args.model.run_multiprompt:
        n_rounds = len(TASK_N_ROUNDS[args.task][args.scheme])
        args.last_output_dir = "" if args.exp_args.model.multipt_start_from == 0 else os.path.join(args.output_dir.split("round_")[0], "round_"+str(args.exp_args.model.multipt_start_from-1))
        args.log_dir = args.output_dir.split("round_")[0]
        setup_logging(args)
        for round in tqdm(range(args.exp_args.model.multipt_start_from, n_rounds)):
            args.current_round = round
            args.should_evaluate = False
            os.makedirs(args.output_dir, exist_ok=True)
            if args.current_round == (n_rounds - 1):
                args.should_evaluate = True
            test_dataset = FormattedDataset(args)
            do_inference(args=args, dataset=test_dataset)
            args.last_output_dir = args.output_dir
            args.output_dir = os.path.join(args.output_dir.split("round_")[0], "round_"+str(round+1))
            #break
    else:
        args.should_evaluate = True
        setup_logging(args)
        test_dataset = FormattedDataset(args)
        do_inference(args=args, dataset=test_dataset)

def run_llama(args, model):
    if args.exp_args.model.run_multiprompt:
        n_rounds = len(TASK_N_ROUNDS[args.task][args.scheme])
        args.last_output_dir = "" if args.exp_args.model.multipt_start_from == 0 else os.path.join(args.output_dir.split("round_")[0], "round_"+str(args.exp_args.model.multipt_start_from-1))
        #print("here here here")
        args.log_dir = args.output_dir.split("round_")[0]
        setup_logging(args)
        args.logging_set = True
        for round in tqdm(range(args.exp_args.model.multipt_start_from, n_rounds)):
            print(args.last_output_dir)
            print(args.output_dir)
            args.current_round = round
            args.should_evaluate = False
            os.makedirs(args.output_dir, exist_ok=True)
            if args.current_round == (n_rounds - 1):
                args.per_device_eval_batch_size = args.per_device_eval_batch_size - 2 if args.task == 'mafalda' else args.per_device_eval_batch_size - 1
                args.should_evaluate = True
            run(args, model)
            args.last_output_dir = args.output_dir
            args.output_dir = os.path.join(args.output_dir.split("round_")[0], "round_"+str(round+1))
    else:
        args.should_evaluate = True
        args.logging_set = True
        setup_logging(args)
        run(args, model)

def make_cache_root(args, task):
    args.task = task
    args.cache_root = os.path.join('cache', args.task)
    os.makedirs(args.cache_root, exist_ok=True)
    return args

def main():
    # Get args
    parser = HfArgumentParser((WrappedTrainingArguments,))
    args, = parser.parse_args_into_dataclasses()
    args.ddp_find_unused_parameters = False
    #args.local_rank = int(os.environ["LOCAL_RANK"])
    # if args.use_dp: args.local_rank = -1
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        print("!!!! Use multi-GPU training with Data Parallel !!!!")
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"device = {args.device}, local_rank={args.local_rank}, n_gpu={args.n_gpu}")
    args.exp_args = Configure.Get(args.cfg)

    if args.regen_results_to != "":
        if not os.path.exists(os.path.join(args.regen_results_to, 'result_updated.json')):
            #print("here")
            #print(args.cfg)
            #print(args.exp_args)
            task = args.regen_results_to.split("results/")[1].split("/")[1]
            
            if (args.exp_args.model.model_tag.startswith('t5')) and (args.exp_args.model.do_multitask):
                args = prepare_output_dir(make_cache_root(args, "multi-task"))
            else:
                args = prepare_output_dir(make_cache_root(args, task))

            args.output_dir = args.regen_results_to
            args.log_dir = args.output_dir
            evaluator = utils.tool.get_evaluator(args.exp_args.evaluate.tool)(args)
            if args.exp_args.model.model_tag.startswith('t5'):
                file_name = 'predict_predictions.json'
            else:
                file_name = 'predictions.json'
            if args.scheme in ["", "w_def", "wo_def", "w_logic_def"]:
                predictions_dir = os.path.join(args.regen_results_to)
            else:
                n_rounds = len(TASK_N_ROUNDS[args.task][args.scheme])
                predictions_dir = os.path.join(args.regen_results_to, f'round_{n_rounds-1}')
            predictions_file = os.path.join(predictions_dir, file_name)
            # output_dir = os.path.join(args.output_dir, 'updated_results')
            # os.makedirs(output_dir, exist_ok=True)
            predictions, golds = [], []
            for one_data in json.load(open(predictions_file)):
                one_pred = one_data.pop('prediction')
                predictions.append(one_pred)
                golds.append(one_data)
            evaluator.evaluate(preds=predictions, golds=golds, section='predict', epoch=None)
        else:
            print("Already got updated result, skip......")
        return
    
    #print(".....")
    args.logging_set = False
    ori_output_dir = args.output_dir
    ori_per_device_eval_batch_size= args.per_device_eval_batch_size
    ori_max_new_tokens = args.max_new_tokens
    active_task_list = TASK_LIST if args.which_task == 'all' else [t.strip() for t in args.which_task.split(',')] 
    args.active_task_list = active_task_list
    print(args.active_task_list)
    if args.exp_args.model.model_tag.startswith('t5'):
        args.should_evaluate = True
        if args.exp_args.model.do_multitask:
            args = prepare_output_dir(make_cache_root(args, "multi-task"))
            run(args)
        else:
            for task, task_arg_path in args.exp_args.arg_paths:
                if task in active_task_list:
                    print(task)
                    if task in ['argotario', 'elecdebate']:
                        args.gradient_accumulation_steps = 8
                    elif task in ['logic', 'propaganda', 'reddit']:
                        args.gradient_accumulation_steps = 16
                    args.output_dir = ori_output_dir
                    args.task_arg_path = task_arg_path
                    args = prepare_output_dir(make_cache_root(args, task))
                    run(args)
    else: 
        model_size = ""
        #if args.exp_args.model.model_tag.startswith('llama'):
        if args.exp_args.model.model_tag.split('-')[0] in ['llama2', 'llama3', 'mistral']:
            model = Model(args)
            model_size = args.exp_args.model.model_tag.split("-")[-1]
        for task, task_arg_path in args.exp_args.arg_paths:
            #print(task)
            if task in active_task_list:
                args.per_device_eval_batch_size = ori_per_device_eval_batch_size
                args.max_new_tokens = ori_max_new_tokens
                args.output_dir = ori_output_dir
                ## allow large output window
                if task in ['logic', 'propaganda', 'mafalda', 'covid']:
                    if args.exp_args.model.model_tag.startswith("llama2"):
                        args.per_device_eval_batch_size = 2 if model_size.startswith("13") else 16
                        #args.per_device_eval_batch_size = 4 if model_size.startswith("13") else 16
                    if args.exp_args.model.model_tag.startswith("mistral"):
                        args.per_device_eval_batch_size = 16
                    if args.scheme in ["v2_gen_def", "v21_gen_def", "v4_wo_def"]:
                        args.max_new_tokens = 1536
                    if args.scheme == "v21_gen_def" and task == 'propaganda':
                        #print("here")
                        if args.exp_args.model.model_tag.startswith("llama2"):
                            args.per_device_eval_batch_size = 2 if model_size.startswith("13") else 12
                        if args.exp_args.model.model_tag.startswith("mistral") or args.exp_args.model.model_tag.startswith("llama3"):
                            args.per_device_eval_batch_size = 12
                #print(args.per_device_eval_batch_size)
                args.task_arg_path = task_arg_path
                #print(args.task_arg_path)
                args = prepare_output_dir(make_cache_root(args, task))
                if args.exp_args.model.model_tag.startswith('gpt'):
                    run_gpt(args)
                else:
                    run_llama(args, model)

if __name__ == "__main__":
    main()