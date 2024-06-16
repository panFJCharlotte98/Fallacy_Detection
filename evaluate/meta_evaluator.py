#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf8
import os
import copy
import numpy as np
import utils.tool
from utils.configure import Configure
from collections.abc import Mapping

class EvaluateTool(object):
    """
    The meta evaluator
    """
    def __init__(self, args):
        self.args = args

    def evaluate(self, preds, golds, section, epoch=None):
        meta_args = self.args.exp_args
        summary = {}
        wait_for_eval = {}

        for pred, gold in zip(preds, golds):
            if gold['arg_path'] not in wait_for_eval.keys():
                wait_for_eval[gold['arg_path']] = {'preds': [], "golds":[]}
            wait_for_eval[gold['arg_path']]['preds'].append(pred)
            wait_for_eval[gold['arg_path']]['golds'].append(gold)

        for arg_path, preds_golds in wait_for_eval.items():
            task_args = Configure.refresh_args_by_file_cfg(os.path.join(meta_args.dir.configure, arg_path), meta_args)
            if self.args.task == 'multi-task':
                self.args.this_task = arg_path
            evaluator = utils.tool.get_evaluator(task_args.evaluate.tool)(self.args)
            summary_tmp = evaluator.evaluate(preds_golds['preds'], preds_golds['golds'], section, epoch)
            for key, metric in summary_tmp.items():  # TODO
                summary[os.path.join(arg_path, key)] = metric
        all_metrics, all_f1, all_acc = [], [], []
        for k, v in summary.items():
            if 'f1' in k:
                all_f1.append(float(v))
            elif k.split("/")[-1] == 'acc':
                all_acc.append(float(v))
            all_metrics.append(float(v))
        summary['avr_f1'] = float(np.mean(all_f1))
        summary['avr_acc'] = float(np.mean(all_acc))
        summary['avr'] = float(np.mean(all_metrics))
        return summary
