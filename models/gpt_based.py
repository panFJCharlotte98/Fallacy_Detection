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
            predictions.append(prediction)
            gold.append(copy.deepcopy(js))
            js['prediction'] = prediction
        except Exception as e:
            logger.info(f"Failed to infer on Example {js['id']} because {e}")
            failed.append(js)
        
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

