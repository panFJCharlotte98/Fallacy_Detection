from openai import OpenAI
from tqdm import tqdm
from typing import List, NamedTuple
import copy
import json
import os
import logging
import utils
logger = logging.getLogger(__name__)

class EvalPrediction(NamedTuple):
    predictions: List[str]
    items: List[dict]

client = OpenAI()
def do_inference(
    args, 
    dataset
):
    model_tag = args.exp_args.model.model_tag
    if args.local_rank <= 0:
        print(f"Model --- Run inference with {model_tag}.")

    total_n_gen_tks, total_n_prompt_tks = 0, 0
    predictions, gold, failed = [], [], []
    for i in tqdm(range(len(dataset))):
        js = dataset[i] # js is a dict
        assert isinstance(js['chat_history'], list), 'Input Message must be a list of dicts.'
        try:
            # Create a chat completion using the question and context
            response = client.chat.completions.create(
                model=model_tag,
                messages=js['chat_history'],
                temperature=args.temperature,
                max_tokens=args.max_new_tokens,
                top_p=args.top_p,
                frequency_penalty=0,
                presence_penalty=0
            )
            prediction = response.choices[0].message.content.strip()
            total_n_gen_tks += response.usage.completion_tokens
            total_n_prompt_tks += response.usage.prompt_tokens
            print(f"Token usage --- input={total_n_prompt_tks} gen={total_n_gen_tks}")
            
            predictions.append(prediction)
            gold.append(copy.deepcopy(js))
            js['prediction'] = prediction

        except Exception as e:
            logger.info(f"Error --- Failed to infer on Example {js['id']} because {e}")
            failed.append(js)
    
    total_cost = compute_cost(model_tag, total_n_gen_tks, total_n_prompt_tks)

    if (args.local_rank <= 0) and (not args.do_not_save_results):
        os.makedirs(args.output_dir, exist_ok=True)
        with open(f"{args.output_dir}/predictions.json", "w") as f:
            json.dump(dataset.examples, f, indent=4)
            f.close()
        if failed:
            logger.info(f"{len(failed)} examples failed due to generation error...")
            with open(f"{args.output_dir}/gen_error.json", "w") as f:
                json.dump(failed, f, indent=4)
                f.close()
        
    if args.should_evaluate: 
        evaluator = utils.tool.get_evaluator(args.exp_args.evaluate.tool)(args)
        eval_prediction = EvalPrediction(predictions=predictions, items=gold)
        evaluator.evaluate(preds=eval_prediction.predictions, golds=eval_prediction.items, section='inference')
    
    if args.local_rank <= 0:
        print(f"Spending --- Spent ${total_cost} after iterating over {args.task} once.")
    return total_cost

def compute_cost(model_type, n_gen, n_input):
    # pricing in dollars per 1M (1e6) tokens
    pricing = {
        'gpt-3.5-turbo': {'input': 0.5, 'output':1.5},
        'gpt-4-turbo':{'input': 10, 'output': 30}
    }
    for model, price in pricing.items():
        if model_type.startswith(model):
            return (n_input / 1e6) * price['input'] + (n_gen / 1e6) * price['output']
            
