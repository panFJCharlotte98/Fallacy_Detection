
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
import importlib
import random
import logging
import numpy as np
from tqdm import tqdm
import multiprocessing
from typing import Dict
from datetime import datetime
from arguments import WrappedTrainingArguments
import torch
torch.autograd.set_detect_anomaly(True)
#torch.cuda.set_sync_debug_mode(1)
#torch.backends.cudnn.benchmark = True
from transformers import (
    HfArgumentParser,
    EarlyStoppingCallback,
    WEIGHTS_NAME,
    # BertConfig, BertTokenizer, BertForSequenceClassification,
    # GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
    # OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
    # RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
    # T5Config, T5ForConditionalGeneration, T5Tokenizer,
    # DistilBertConfig, DistilBertForMaskedLM, DistilBertForSequenceClassification, DistilBertTokenizer,
    LlamaConfig,LlamaForCausalLM, LlamaTokenizer
    )
from transformers.data.data_collator import DataCollatorWithPadding
# local import
from models.llama_based import Model
from models.gpt_based import do_inference
from dataset import TokenizedDataset, FormattedDataset
from trainer import EvaluateFriendlyTrainer
from utils.argotario_prompt import USER_PROMPTS

# try:
#     from torch.utils.tensorboard import SummaryWriter
# except:
#     from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)
cpu_cont = multiprocessing.cpu_count()

EVALUATOR_TOOL = {
    "argotario": 'evaluate.argotario_evaluator',
}
def get_evaluator(evaluate_tool):
    EvaluateTool = importlib.import_module('{}'.format(evaluate_tool)).EvaluateTool
    return EvaluateTool

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
                   
def prepare_output_dir(args):
    dict_args = vars(args)
    # format: [arg name, name used in path]
    args_used_in_name = [
        #['per_device_train_batch_size','tbs'],
        ['per_device_eval_batch_size','ebs'],
        ['max_new_tokens','len'],
        # ['optimizer','optim'],
        # ['meta_lr','mlr'],
        # ['meta_weight_decay','mwd'],
        # ['meta_net_num_layers','mnlayers'],
        # ['meta_net_hidden_size','mhidden'],
        # ['learning_rate','lr'],
        # ['num_train_epochs', 'ep'],
        # ['gradient_accumulation_steps', 'gas']
    ]
    setting = 'single'
    if args.run_multiprompt:
        setting = "multi"
    if args.run_baseline:
        setting = 'baseline'
    folder_name = [f"GPU-{args.world_size}_{setting}"]
    for arg_name, rename in args_used_in_name:
        folder_name.append('{}-{}'.format(rename, dict_args[arg_name]))
    sys_dt = datetime.now().strftime("%Y%m%d%H%M%S")
    folder_name.append(sys_dt)
    folder_name = '_'.join(folder_name)
    output_dir = os.path.join(args.output_dir, args.task, args.model_tag, folder_name)
    if args.run_multiprompt:
        output_dir = os.path.join(output_dir, "round_0")
    return output_dir
  
MULTIPROMPT_N_ROUNDS = {'argotario': len(USER_PROMPTS)}


def run(args, model, evaluator=None):
    if args.local_rank > 0: ## Barrier to make sure only the first process in distributed training download model & vocab
        torch.distributed.barrier()  
    #----------------------- Generate and cache datasets ----------------------#
    if args.do_predict:
        test_dataset = TokenizedDataset("inference", args, args.cache_root, model.tokenizer)
    #------------------------------- Create Trainer --------------------------------#
    data_collator = DataCollatorWithPadding(model.tokenizer, padding="longest")
    trainer = EvaluateFriendlyTrainer(
        args=args,
        evaluator=evaluator,
        model=model,
        tokenizer=model.tokenizer,
        data_collator = data_collator,
        eval_dataset=test_dataset
    )
    logger.info('***** Trainer built successfully. ***** \n')
    if args.local_rank == 0:
        torch.distributed.barrier()  
    # End of barrier to make sure only the first process in distributed training download model & vocab
    
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
    return

def main():
    # Get args
    parser = HfArgumentParser((WrappedTrainingArguments,))
    args, = parser.parse_args_into_dataclasses()
    ori_output_dir = args.output_dir
    args.ddp_find_unused_parameters = False
    #args.local_rank = int(os.environ["LOCAL_RANK"])
    # if args.use_dp: args.local_rank = -1
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        print("!!!! Use multi-GPU training with Data Parallel !!!!")
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"device = {args.device}, local_rank={args.local_rank}, n_gpu={args.n_gpu}")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.which_gpu
    #------------------------------- Set up Output directory --------------------------------#
    gen_output_dir = True
    if args.multipt_start_from > 0:
        gen_output_dir = False
    if gen_output_dir:
        # Set up result reporting dir 
        if args.local_rank <= 0:
            output_dir = [prepare_output_dir(args)]
        else:
            output_dir = [""]
        torch.distributed.broadcast_object_list(output_dir, src=0, device=args.device)#dist.send(param.data, dst=sibling)
        args.output_dir = output_dir[0]
    else:
        args.output_dir = os.path.join(args.output_dir, "round_"+str(args.multipt_start_from))
    os.makedirs(args.output_dir, exist_ok=True)
    #------------------------------- Set up WanDB --------------------------------#
    args.run_name = "_".join([args.task, args.model_tag])
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
    #------------------------------- Set up logging --------------------------------#
    # Initialize the logger
    if args.local_rank > 0: ## Barrier to make sure only the first process in distributed training download model & vocab
        torch.distributed.barrier()  
    #log_file_name = '{}-{}.log'.format('run',datetime.now().strftime("%Y%m%d%H%M%S"))
    log_file_name = 'run.log'
    logging.basicConfig(filename=os.path.join(args.output_dir, log_file_name),
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
    
    # Load pretrained model and tokenizer
    cache_root = os.path.join('cache', args.task)
    args.cache_root = cache_root
    os.makedirs(cache_root, exist_ok=True)
    # ------------------------- Choose task dataset -------------------------#
    data_folder = f"./data/{args.task}/"   
    if args.task == "argotario":
        args.raw_data_file = data_folder + "data.json"
    elif args.task == "logic":
        args.raw_data_file = data_folder + "data/data.json"
    # ------------------------- Load pre-trained model -------------------------#
    evaluator = get_evaluator(EVALUATOR_TOOL[args.task])(args)
    if args.model_tag.startswith('gpt'):
        args.model_type = 'gpt'
        if args.run_multiprompt:
            n_rounds = MULTIPROMPT_N_ROUNDS[args.task]
            args.last_output_dir = "" if args.multipt_start_from == 0 else os.path.join(args.output_dir.split("round_")[0], "round_"+str(args.multipt_start_from-1))
            for round in tqdm(range(args.multipt_start_from, n_rounds)):
                args.current_round = round
                args.should_evaluate = False
                if args.current_round == (n_rounds - 1):
                    args.should_evaluate = True
                test_dataset = FormattedDataset(args, cache_root)
                #exit()
                do_inference(args=args, dataset=test_dataset, evaluator=evaluator)
                args.last_output_dir = args.output_dir
                args.output_dir = os.path.join(args.output_dir.split("round_")[0], "round_"+str(round+1))
        else:
            args.should_evaluate = True
            test_dataset = FormattedDataset(args, cache_root)
            do_inference(args=args, dataset=test_dataset, evaluator=evaluator)
    else:    
        model= Model(args)
        if args.run_multiprompt:
            args.last_output_dir = "" if args.multipt_start_from == 0 else os.path.join(args.output_dir.split("round_")[0], "round_"+str(args.multipt_start_from-1))
            print(args.last_output_dir)
            print(args.output_dir)
            for round in tqdm(range(args.multipt_start_from, MULTIPROMPT_N_ROUNDS[args.task])):
                args.current_round = round
                args.should_evaluate = False
                if args.current_round == (MULTIPROMPT_N_ROUNDS[args.task] - 1):
                    args.per_device_eval_batch_size = 6
                    args.should_evaluate = True
                run(args, model, evaluator)
                args.last_output_dir = args.output_dir
                args.output_dir = os.path.join(args.output_dir.split("round_")[0], "round_"+str(round+1))
        else:
            args.should_evaluate = True
            run(args, model, evaluator)

if __name__ == "__main__":
    main()


