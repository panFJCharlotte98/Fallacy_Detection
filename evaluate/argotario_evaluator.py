import os
import copy
import numpy as np
import json
from utils.format_utils import E_INST
import regex
from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score, accuracy_score, confusion_matrix, classification_report
import pprint
from utils.fallacy_utils import FALLACY_DEFINITIONS

import logging
logger = logging.getLogger(__name__)

fallacy_names = [n.lower() for n, d in FALLACY_DEFINITIONS.items()] + ['no fallacy']
#emotional language | appeal to false authority | appeal to emotion
FALLACY_ID = dict(zip(fallacy_names, list(range(len(fallacy_names)))))


class EvaluateTool(object):
    """
    The Argotario evaluator
    """
    def __init__(self, args):
        self.args = args
    
    def convert_to_name(self, t):
        if 'authority' in t:
            #return 'appeal to opinion of false authority'
            return 'appeal to false authority'
        elif ('emotion' in t) or ('emotional' in t) or ('emoti' in t):
            return 'appeal to emotion'
        elif ('hasty' in t) or ('generalization' in t):
            return 'hasty generalization'
        elif 'ad hominem' in t:
            return 'ad hominem'
        elif ('red' in t) or ('herring' in t):
            return 'red herring'
        elif 'no fallacy' in t:
            return 'no fallacy'
        else:
            return ""
            
    def extract_answer(self, text):
        #print(text)
        last_reply = text.split(E_INST)[-1].replace("\n", " ")
        last_reply = regex.sub(r"\{.*\{", "{", last_reply)
        last_reply = regex.sub(r"\}.*\}", "}", last_reply).strip()
        if ("\"explanation\":" in last_reply):
            if ("\"explanation\": \"" not in last_reply):
                last_reply = last_reply.replace("\"explanation\":", "\"explanation\": \"")
            if ("}" in last_reply):
                if ("\" }" not in last_reply) and ("\"}" not in last_reply): 
                    last_reply = last_reply.replace("}", "\" }")
                if ("\" }" not in last_reply) and ("\"}" in last_reply):
                    last_reply = last_reply.replace("\"}", "\" }") 
            exp_content = last_reply.split("\"explanation\": \"")[1].split("\" }")[0]
            exp_content_clean = exp_content.replace("\"", "\'")
            last_reply = last_reply.replace(exp_content, exp_content_clean)
            if "'}" in last_reply:
                last_reply = last_reply.replace("'}", "\" }")
        # extract json
        try:
            ans = json.loads(regex.findall(r"\{.*\}", last_reply)[0])['fallacy']
        except:
            return "", last_reply
        if isinstance(ans, str):
            ans = self.convert_to_name(ans.lower().replace("\n",""))
        if isinstance(ans, list):
            return [self.convert_to_name(" ".join(a.lower().replace("\n","").split())) for a in ans], last_reply
        
        return ans, last_reply
        
        
    def evaluate(self, preds, golds, section='predict'):
        """
        pred: 
            1.baseline: {'fallacy': 'Red herring'}
        gold:
            a dictionary in format like
            {'text':xxx,
            'label': 'Red Herring',
            'is_gold': 1,
            'stance': "pro"
            }
        """
        args = self.args
        summary = {}
        predictions, labels, passed, failed, incorrect = [], [], [],  [], []

        for pred, gold in zip(preds, golds):
            #self.extract_answer(pred)
            pred_ans, last_reply = self.extract_answer(pred)
            gold_fn = self.convert_to_name(gold['label'].lower())
            gold_id = FALLACY_ID[gold_fn]
            this_rec = dict(**{'prediction': last_reply}, **gold)
            if (isinstance(pred_ans, list)):
                if (gold_fn in pred_ans):
                    pred_id = gold_id
                    predictions.append(pred_id)  
                else:
                    pred_id = FALLACY_ID[pred_ans[0]]
                    predictions.append(pred_id)
                labels.append(gold_id)    
                passed.append(this_rec)
                if gold_id != pred_id:
                    incorrect.append(this_rec)
            else:
                if pred_ans: # != ""
                    pred_id = FALLACY_ID[pred_ans]
                    predictions.append(pred_id)
                    labels.append(gold_id)
                    passed.append(this_rec)
                    if gold_id != pred_id:
                        incorrect.append(this_rec)
                else:
                    failed.append(this_rec)
          
        summary['acc'] = round(accuracy_score(labels, predictions),4)
        summary['f1'] = round(f1_score(labels, predictions, average = 'macro'),4)
        summary['precision'] = round(precision_score(labels, predictions, average= 'macro', zero_division=0.),4)
        summary['recall'] = round(recall_score(labels, predictions, average= 'macro'),4)
        summary['balanced_acc'] = round(balanced_accuracy_score(labels, predictions),4)
        summary['overall_acc'] = round(int(len(passed)*summary['acc']) / len(preds), 4)
        #summary['avr'] = round(float(np.mean([f1, balanced_acc, recall, precision])),4)
        
        summary_print = pprint.pformat(summary)
        if (int(os.environ["LOCAL_RANK"]) <= 0):
            if not args.do_not_save_results:
                with open(f"{self.args.output_dir}/passed.json", "w") as f:
                    json.dump(passed,f,indent=4,)
                    f.close()
                with open(f"{self.args.output_dir}/failed.json", "w") as f:
                    json.dump(failed,f,indent=4,)
                    f.close()
                with open(f"{self.args.output_dir}/incorrect.json", "w") as f:
                    json.dump(incorrect,f,indent=4,)
                    f.close()
            logger.info(f'\n**********report metrics***********\n{summary_print}\n#total:{len(preds)}\n#passed examples:{len(passed)}\n#failed examples:{len(failed)}\n**********report metrics***********\n')
        
        try:
            classification_rpt = classification_report(labels, predictions, target_names=list(FALLACY_ID.keys()), zero_division=0., digits=4)
            if int(os.environ["LOCAL_RANK"]) <= 0:
                logger.info("\n**********classification report***********\n" + classification_rpt)
        except:
            return summary
        conf_matrix = confusion_matrix(labels, predictions, labels=list(FALLACY_ID.values()))
        if int(os.environ["LOCAL_RANK"]) <= 0:
            logger.info("\n**********confusion matrix***********\n")
            logger.info(conf_matrix)
        return summary
