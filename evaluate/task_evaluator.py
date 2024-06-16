import os
import copy
import numpy as np
import json
from utils.format_utils import E_INST
import regex
from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score, accuracy_score, confusion_matrix, classification_report
import pprint
from utils.fallacy_utils import *
from evaluate.convert import *
import random 
import logging
logger = logging.getLogger(__name__)
         
class EvaluateTool(object):
    """
    The task evaluator
    """
    def __init__(self, args):
        self.args = args
        if args.task == "multi-task":
            self.task = self.args.this_task.split(".cfg")[0].split("/")[-1]
        else:
            self.task = self.args.task
        

        if args.exp_args.model.model_tag == 't5':
            self.output_dir = os.path.join(args.output_dir, self.task)
            os.makedirs(self.output_dir, exist_ok=True)
        else:
            self.output_dir = args.log_dir
        
        if self.task == "argotario":
            fallacy_names = [n.lower() for n, d in ARGOTARIO_FALLACY_DEFINITIONS.items()] + ['no fallacy']
            #emotional language | appeal to false authority | appeal to emotion
            self.FALLACY_ID = dict(zip(fallacy_names, list(range(len(fallacy_names)))))
            self.convert_to_name = argotario_convert_to_name
        elif self.task == "logic":
            fallacy_names = [n.lower() for n, d in LOGIC_FALLACY_DEFINITIONS.items()] #+ ['failed']
            #emotional language | appeal to false authority | appeal to emotion
            self.FALLACY_ID = dict(zip(fallacy_names, list(range(len(fallacy_names)))))
            self.convert_to_name = logic_convert_to_name
        elif self.task == "elecdebate":
            fallacy_names = [n.lower() for n, d in ELECDEBATE_FALLACY_DEFINITIONS.items()] #+ ['failed']
            #emotional language | appeal to false authority | appeal to emotion
            self.FALLACY_ID = dict(zip(fallacy_names, list(range(len(fallacy_names)))))
            self.convert_to_name = elecdebate_convert_to_name
        elif self.task == "propaganda":
            fallacy_names = [n.lower() for n, d in PROPAGANDA_FALLACY_DEFINITIONS.items()] #+ ['failed']
            #emotional language | appeal to false authority | appeal to emotion
            self.FALLACY_ID = dict(zip(fallacy_names, list(range(len(fallacy_names)))))
            self.convert_to_name = propaganda_convert_to_name
        elif self.task == "mafalda":
            fallacy_names = [n.lower() for n, d in MAFALDA_FALLACY_DEFINITIONS.items() if n != 'Appeal to Pity'] + ['no fallacy']
            #emotional language | appeal to false authority | appeal to emotion
            self.fallacy_names = fallacy_names
            self.FALLACY_ID = dict(zip(fallacy_names, list(range(len(fallacy_names)))))
            self.convert_to_name = mafalda_convert_to_name
            self.fallacy_categories = MAFALDA_FALLACY_CATEGORIES
        elif self.task == 'covid':
            fallacy_names = [n.lower() for n, d in COVID_FALLACY_DEFINITIONS.items()] + ['no fallacy']
            self.FALLACY_ID = dict(zip(fallacy_names, list(range(len(fallacy_names)))))
            self.convert_to_name = covid_convert_to_name
        elif self.task == 'reddit':
            fallacy_names = [n.lower() for n, d in REDDIT_FALLACY_DEFINITIONS.items()]
            self.FALLACY_ID = dict(zip(fallacy_names, list(range(len(fallacy_names)))))
            self.convert_to_name = reddit_convert_to_name
        self.FALLACY_NAME = {v:k for k, v in self.FALLACY_ID.items()}
    
    def extract_answer(self, text):
        #print(text)
        #last_reply = text.split(E_INST)[-1].replace("\n", " ")
        last_reply = text.strip("\n``` ").replace("\n", " ")
        last_reply = " ".join(last_reply.split())
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
        if ('''"explanation": "''') in last_reply:
            if (not last_reply.endswith('''}''')):
                last_reply = last_reply.rstrip("'}\"")
                last_reply = last_reply + "\" }"
                exp_content = last_reply.split("\"explanation\": \"")[1].split("\" }")[0]
                exp_content_clean = exp_content.replace("\"", "'")
                last_reply = last_reply.replace(exp_content, exp_content_clean)
        # extract json
        try:
            ans = json.loads(regex.findall(r"\{.*\}", last_reply)[0])['fallacy']
        except:
            try: 
                ans = json.loads(regex.findall(r"\{.*\}", last_reply)[0])['propaganda_technique']
            except:
                try:
                    ans = json.loads(regex.findall(r"\{.*\}", last_reply)[0])['propaganda technique']
                except:
                    # return "", last_reply
                    return 'failed', last_reply
    
        if isinstance(ans, list):
            return [self.convert_to_name(" ".join(str(a).lower().replace("\n","").split())) for a in ans], last_reply
        else:
            ans = self.convert_to_name(str(ans).lower().replace("\n",""))
            return ans, last_reply
        
        
    def evaluate(self, preds, golds, section='predict', epoch=None):
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
        predictions, labels, failed, incorrect = [], [], [], []

        for pred, gold in zip(preds, golds):
            #self.extract_answer(pred)
            if self.args.exp_args.model.model_tag.startswith("t5"):
                pred_ans = self.convert_to_name(pred)
                last_reply = pred_ans
            else:
                pred_ans, last_reply = self.extract_answer(pred)

            gold_fals = [self.convert_to_name(f.lower()) for f in gold['label']]
            # print(gold['label'])
            # print(gold_fals)
            false_fals = list(set(list(self.FALLACY_ID.keys())) - set(gold_fals))
            if self.task == "mafalda":
                other_gold_fals = [f_ for f_ in [self.convert_to_name(f.lower()) for f in gold['other_labels']] if f_]
                false_fals = list(set(list(self.FALLACY_ID.keys())) - set(gold_fals + other_gold_fals))
            #print(gold_fals)
            gold_ids = {f: self.FALLACY_ID[f] for f in gold_fals}
            this_rec = dict(**{'prediction': last_reply}, **gold)
            gold_id = list(gold_ids.values())[0]
            if (isinstance(pred_ans, list)):
                if (self.task == 'mafalda'):
                    if set(pred_ans) == set(gold_fals + other_gold_fals):
                        is_found = True
                    else:
                        is_found = False
                else:    
                    is_found = False
                    for gf in gold_fals:
                        if gf in pred_ans:
                            is_found = True
                            gold_id = gold_ids[gf]
                            break
                if is_found:
                    pred_id = gold_id
                    predictions.append(pred_id)  
                else:
                    if pred_ans[0] == 'failed':
                        failed.append(this_rec)
                        if self.task == "argotario":
                            pred_ans[0] = 'no fallacy'
                        else:
                            pred_ans[0] = random.choice(false_fals)
                    else:
                        pred_ans = [ans for ans in pred_ans if ans in self.FALLACY_ID]
                        if len(pred_ans) == 0:
                            this_rec['prediction_error'] = "Wronly predicted to be appeal to pity"
                            failed.append(this_rec)
                            pred_ans[0] = random.choice(false_fals)
                    pred_id = self.FALLACY_ID[pred_ans[0]]
                    predictions.append(pred_id)
                labels.append(gold_id)
                if gold_id != pred_id:
                    incorrect.append(this_rec)
            else:
                if pred_ans: # != ""
                    # failed examples counted as 'no fallacy' for result report
                    if pred_ans == 'failed':
                        failed.append(this_rec)
                        if self.task == "argotario":
                            pred_ans = 'no fallacy'
                        else:
                            pred_ans = random.choice(false_fals)
                    
                    # #-------------------------for Mafalda------------------------# #
                    if (self.task == 'mafalda') and (pred_ans == 'appeal to emotion'):
                        if any([gf in [f.lower() for f in self.fallacy_categories['Appeal to Emotion']] for gf in gold_fals]):
                            # if gold fallacies contain any category that belongs to 'appeal to emotion' and pred_ans == 'appeal to emotion'
                            # count as correct prediction 
                            # but need to take note of these examples as failed
                            # if 'appeal to emotion' not in self.fallacy_names:
                            #     fallacy_names = self.fallacy_names + ['appeal to emotion']
                            #     self.fallacy_names = fallacy_names
                            #     #emotional language | appeal to false authority | appeal to emotion
                            #     self.FALLACY_ID = dict(zip(fallacy_names, list(range(len(fallacy_names)))))
                            # gold_ids = {f: self.FALLACY_ID[f] for f in gold_fals + ['appeal to emotion']}
                            emo_gfs = [gf for gf in gold_fals if gf in [f.lower() for f in self.fallacy_categories['Appeal to Emotion']]]
                            pred_ans = random.choice(emo_gfs)
                            this_rec['prediction_error'] = "Failed to predict the exact emotion type."
                            failed.append(this_rec)
                        else:
                            failed.append(this_rec)
                            pred_ans = random.choice(false_fals)
                    if (self.task == 'mafalda') and (pred_ans == 'appeal to pity'):
                        assert pred_ans not in gold_fals
                        if (pred_ans in other_gold_fals):
                            if 'appeal to pity' not in self.fallacy_names:
                                fallacy_names = self.fallacy_names + ['appeal to pity']
                                self.fallacy_names = fallacy_names
                                #emotional language | appeal to false authority | appeal to emotion
                                self.FALLACY_ID = dict(zip(fallacy_names, list(range(len(fallacy_names)))))
                            gold_ids = {f: self.FALLACY_ID[f] for f in gold_fals + other_gold_fals}
                        else:
                            failed.append(this_rec)
                            pred_ans = random.choice(false_fals)
                    # #-------------------------for Mafalda------------------------# #
                    pred_id = self.FALLACY_ID[pred_ans]
                    predictions.append(pred_id)
                    if pred_id in list(gold_ids.values()):
                        gold_id = pred_id
                    labels.append(gold_id)
                    if gold_id != pred_id:
                        incorrect.append(this_rec)
                # else:
                #     failed.append(this_rec)
          
        summary['acc'] = round(accuracy_score(labels, predictions),4)
        summary['f1'] = round(f1_score(labels, predictions, average = 'macro'),4)
        summary['precision'] = round(precision_score(labels, predictions, average= 'macro', zero_division=0.),4)
        summary['recall'] = round(recall_score(labels, predictions, average= 'macro'),4)
        summary['balanced_acc'] = round(balanced_accuracy_score(labels, predictions),4)
        summary_report = copy.deepcopy(summary)
        summary['n_failed'] = len(failed)
        summary['model'] = self.args.exp_args.model.model_tag
        summary['task'] = self.task
        summary['setting'] = self.args.setting
        summary['scheme'] = self.args.scheme
        summary['max_new_tokens'] = self.args.max_new_tokens
        summary['context_window'] = self.args.context_window
        #summary['avr'] = round(float(np.mean([f1, balanced_acc, recall, precision])),4)
        
        summary_print = pprint.pformat(summary)
        # if (int(os.environ["LOCAL_RANK"]) <= 0):
        #     if not args.do_not_save_results:
        #         if args.exp_args.model.model_tag == 't5':
        #             if epoch is not None:
        #                 self.output_dir = os.path.join(self.output_dir, section+'_', 'epoch_' + str(int(epoch)))
        #             else:
        #                 self.output_dir = os.path.join(self.output_dir, section)
        #             os.makedirs(self.output_dir, exist_ok=True)
        #         #print(self.output_dir)
        #         with open(os.path.join(self.output_dir, "failed.json"), "w") as f:
        #             json.dump(failed,f,indent=4,)
        #             f.close()
        #         with open(os.path.join(self.output_dir, "incorrect.json"), "w") as f:
        #             json.dump(incorrect,f,indent=4,)
        #             f.close()
        #         with open(os.path.join(self.output_dir, "result.json"), "w") as f:
        #             json.dump(summary, f, indent=4)
        #             f.close()
        #     logger.info(f'\n**********report metrics***********\n{summary_print}\n#total:{len(preds)}\n#failed examples:{len(failed)}\n**********report metrics***********\n')
        
        if (int(os.environ["LOCAL_RANK"]) <= 0):
            if not args.do_not_save_results:
                if args.exp_args.model.model_tag == 't5':
                    if epoch is not None:
                        self.output_dir = os.path.join(self.output_dir, section+'_', 'epoch_' + str(int(epoch)))
                    else:
                        self.output_dir = os.path.join(self.output_dir, section)
                    os.makedirs(self.output_dir, exist_ok=True)
                #print(self.output_dir)
                gen_file_names = ['failed', 'incorrect', 'result']
                gen_files = {}
                for f_name in gen_file_names:
                    gen_files[f_name] = f"{f_name}.json" if args.regen_results_to == "" else f"{f_name}_updated.json"
                with open(os.path.join(self.output_dir, gen_files['failed']), "w") as f:
                    json.dump(failed,f,indent=4,)
                    f.close()
                with open(os.path.join(self.output_dir, gen_files['incorrect']), "w") as f:
                    json.dump(incorrect,f,indent=4,)
                    f.close()
                with open(os.path.join(self.output_dir, gen_files['result']), "w") as f:
                    json.dump(summary, f, indent=4)
                    f.close()
            logger.info(f'\n**********report metrics***********\n{summary_print}\n#total:{len(preds)}\n#failed examples:{len(failed)}\n**********report metrics***********\n')
        # pred_classes = sorted(list(set(predictions)))
        # if args.exp_args.model.model_tag == 't5':
        #     target_names = [self.FALLACY_NAME[lid] for lid in sorted(list(set(labels)))]
        #     cm_labels = sorted(list(set(labels)))
        #     assert len(target_names) == len(cm_labels)
        #     if len(pred_classes) > len(target_names):
        #         target_names = [self.FALLACY_NAME[lid] for lid in sorted(list(set(predictions)))]
        #         cm_labels = sorted(list(set(predictions)))
        # else: 
        target_names = list(self.FALLACY_ID.keys())
        cm_labels = sorted(list(self.FALLACY_ID.values()))
        # n_label_classes == n_target_names & pred_classes belongs to label_classes 
        classification_rpt = classification_report(labels, predictions, target_names=target_names, zero_division=0., digits=4)
        if int(os.environ["LOCAL_RANK"]) <= 0:
            logger.info("\n**********classification report***********\n" + classification_rpt)
        
        conf_matrix = confusion_matrix(labels, predictions, labels=cm_labels)
        if int(os.environ["LOCAL_RANK"]) <= 0:
            logger.info("\n**********confusion matrix***********\n")
            logger.info(conf_matrix)
        
        return summary_report
