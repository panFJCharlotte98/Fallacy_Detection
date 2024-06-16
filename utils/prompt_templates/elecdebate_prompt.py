
from utils.fallacy_utils import *
import regex
import json
import random
FALDEFS = ELECDEBATE_FALLACY_DEFINITIONS
elec_fals = list(FALDEFS.keys())
fal_def_str = "\n".join([f"{str(i+1)}. {fal} is {FALDEFS[fal]}" for i, fal in enumerate(elec_fals)])    
fal_name_str = ", ".join([f"{str(i+1)}. {fal}" for i, fal in enumerate(elec_fals)])

T5_PROMPTS = [
'''Given the following fallacy definitions and a part of a political speech extracted from one of the US presidential campaign debates, determine which of the fallacies defined below occurs in the focused segment.
Definitions:
{definitions}
Speech:
{argument}
Focused segment:
{segment}'''
]

SYSTEM_PROMPT = [
'''You are a knowledgable expert in analysing fallacies in speeches for political debates. 
Please ensure that your responses are socially unbiased in nature.
Your response should not be lengthy.
Answer the last question.''',
]

BASELINE_PROMPTS = {
    'w_def':'''Based on the following definitions of fallacies,
{fallacies}
Given a part of a political speech extracted from a US presidential campaign debate as below,
{argument}
Determine which of the fallacies defined above is present in the speech's argument highlighted by '<>'?
Output your answer in JSON format {{"fallacy": name_of_the_fallacy, "explanation": in_a_sentence_or_two}}. Only output JSON.''',
    'wo_def':'''Given five types of fallacies, namely, {fallacies}, and a part of a political speech extracted from a US presidential campaign debate as below,
{argument}
Determine which of the fallacies given is present in the speech's argument highlighted by '<>'?
Output your answer in JSON format {{"fallacy": name_of_the_fallacy, "explanation": in_a_sentence_or_two}}. Only output JSON.'''
}

# Version 1: GPT-3.5 Multi-prompt
v1_wo_def = {
0:"""Given the following part of a political speech extracted from one of the US presidential campaign debates:\n{argument}\nsummarize the possible topic in debate and the central argument of the part highlighted by '<>'.""",
1:"""Is the speech's argument highlighted by '<>' logically reasonable or potentially fallacious? Give your analysis.""",
2: """According to your previous analysis, given five types of fallacies, namely, {fallacies}, determine which of these fallacies is present in the speech's argument highlighted by '<>'? Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. DO NOT output fallacy that is not in the given types. Only output JSON.""".format(fallacies=fal_name_str),
}

v12_wo_def = {
0:"""Given the following part of a political speech extracted from one of the US presidential campaign debates: {argument}\nIs the speech's argument highlighted by '<>' logically reasonable or potentially fallacious? Give your analysis.""",
1: """According to your previous analysis, considering five types of fallacies: {fallacies}, determine which of these listed fallacies is present in the speech's argument highlighted by '<>'? Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. Only output JSON.""".format(fallacies=fal_name_str),
}

v13_wo_def = {
0:"""Given the following part of a political speech extracted from one of the US presidential campaign debates: {argument}\nSummarize the speech's argument highlighted by '<>' by pointing out the premise(s) and conclusion of the argument.""",
1:"""Whether or not the premise(s) of the speech's argument highlighted by '<>' entail the conclusion? Give your analysis.""",
2: """A fallacy is an argument where the premises do not entail the conclusion. According to your previous analysis, considering five types of fallacies: {fallacies}, determine which of these listed fallacies is present in the speech's argument highlighted by '<>'. Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. Only output JSON.""".format(fallacies=fal_name_str),
}

v14_wo_def = {
0:"""Given the following part of a political speech extracted from one of the US presidential campaign debates: {argument}\nExtract and summarize the speech's argument highlighted by '<>' by pointing out the premise(s) and conclusion of the argument.""",
1:"""A fallacy is defined as an argument where the premises do not entail the conclusion. According to this definition, is the speech's argument highlighted by '<>' logically reasonable or potentially fallacious? Give your analysis.""",
2: """According to your previous analysis, considering five types of fallacies: {fallacies}, determine which of these listed fallacies is present in the speech's argument highlighted by '<>'. Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. Only output JSON.""".format(fallacies=fal_name_str),
}

v2_gen_def = {
0:'''Give a definition to each of the following five types of fallacies in {fallacies}.'''.format(fallacies=fal_name_str),
1:'''Based on the definitions you provided, given the following part of a political speech extracted from one of the US presidential campaign debates:\n{argument}\ndetermine which of the defined fallacies is present in the speech's argument highlighted by '<>'? Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. Only output JSON.'''
}

v21_gen_def = {
0:'''Give a definition to each of the following types of fallacies: {fallacies}.'''.format(fallacies=fal_name_str),
1:'''Given the following part of a political speech extracted from one of the US presidential campaign debates:\n{argument}\nContrast each of the fallacy definitions you provided with the speech's argument highlighted by '<>' and determine whether or not that fallacy is present in this highlighted argument.''',
2:'''According to your previous analysis, summarize whether any of these given fallacies is present in the speech's argument highlighted by '<>', output your final decision in JSON format {"fallacy": name_of_the_fallacy}. Only output JSON.'''
}

v3_cot_wo_def = {
0:'''Given the following part of a political speech extracted from one of the US presidential campaign debates,
[]
and the following five types of fallacies, namely, {fallacies}, which of the listed fallacies is present in the speech's argument highlighted by '<>'? Now, let's think step by step.'''.format(fallacies=fal_name_str),
1: '''Output your previous conclusion in JSON format {"fallacy": name_of_the_fallacy}. Only output JSON.'''
}

v4_wo_def = {
0:'''Given the following five types of fallacies, namely, {fallacies},
and given the following part of a political speech extracted from one of the US presidential campaign debates,
[]
contrast each of the listed fallacies with the speech's argument highlighted by '<>' and determine whether that fallacy is present in the this argument highlighted by '<>'.'''.format(fallacies=fal_name_str),
1: '''According to your previous analysis, summarize which of these fallacies is present in the speech's argument highlighted by '<>', output your final conclusion in JSON format {"fallacy": name_of_the_fallacy}. Only output JSON.'''
}

elecdebate_multiround_prompts = {
    'v1_wo_def' : v1_wo_def,
    'v12_wo_def': v12_wo_def,
    'v13_wo_def': v13_wo_def,
    'v14_wo_def': v14_wo_def,
    'v2_gen_def': v2_gen_def,
    'v21_gen_def': v21_gen_def,
    'v3_cot_wo_def': v3_cot_wo_def,
    'v4_wo_def': v4_wo_def
}

#{'Appeal to Emotion': 777, 'Appeal to False Authority': 180, 'Ad Hominem': 171, 'False Causality (Post Hoc Fallacy)': 50, 'Slippery Slope': 44}
def prompt_w_few_shot_examples(args, text):
    train_data = json.load(open(args.task_args.dataset.load_from + "train.json"))
    fal_examples = {fal.lower(): [] for fal in elec_fals}
    for data in train_data:
        assert data['label'][0].lower() in fal_examples
        fal_examples[data['label'][0].lower()].append(data)
    n_shots_per_class = {"Appeal to Emotion": 1, "Ad Hominem": 2, "Appeal to False Authority": 2, "False Causality (Post Hoc Fallacy)": 3, "Slippery Slope": 3}
    few_shots = []
    for name, num in n_shots_per_class.items():
        for js in random.sample(fal_examples[name.lower()], num):
            argument = " ".join([js['pre-text'], js['text'], js['post-text']]).strip()
            few_shots.append((name, argument))
    prompt = '''Given five types of fallacies, namely, {fallacies}, and a part of a political speech extracted from a US presidential campaign debate, determine which of the fallacies given is present in the speech's argument highlighted by '<>'? Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. Only output JSON.\n'''.format(fallacies = fal_name_str) + \
    "\n".join(['''Political speech snippet: "{example}" | Fallacy: {{ "fallacy": "{fal_name}" }}'''.format(example=one_example[1], fal_name=one_example[0]) for one_example in few_shots]) + \
    '''\nPolitical speech snippet: "{argument}" | Fallacy: '''.format(argument=text)
    return prompt

def prompt_elecdebate(args, js):
    """
    input:
        js is one data sample in the format of dictionary
        js['text'] is a string of QA in a format like : "A: ....\nB: ..."
    """
    if args.context_window > 0:
        argument = " ".join([js['pre-text'], js['text'], js['post-text']]).strip()
    else:
        argument = js['text']
    if args.exp_args.model.model_tag.startswith('t5'):
        text = argument.lower()
        js['seq_in'] = T5_PROMPTS[0].format(argument=text, segment=js['text'], definitions=fal_def_str.lower())
    else:
        dialog = []
        sys_pt = {"role": "system", "content": SYSTEM_PROMPT[0]}
        if args.exp_args.model.run_multiprompt:
            USER_PROMPTS = elecdebate_multiround_prompts[args.scheme]
            tp = USER_PROMPTS[args.current_round]
            if "[]" in tp:
                tp = tp.replace("[]", "{argument}")
            usr_ct = tp.format(argument=argument) if '{argument}' in tp else tp
            usr_prompt = {"role": "user", "content": usr_ct}
            if args.current_round == 0: # initial round
                dialog = [sys_pt, usr_prompt]
                #js.pop('text')
            else:
                last_prediction = js.pop('prediction')
                dialog = js['chat_history'] + [{"role":"assistant", "content":last_prediction}] + [usr_prompt]  
        else:
            text = argument
            if args.exp_args.model.run_baseline:
                fallacies = fal_def_str if args.scheme == 'w_def' else fal_name_str
                content = BASELINE_PROMPTS[args.scheme].format(fallacies=fallacies, argument=text)
            else:
                #content = SINGLE_PROMPT_TEMPLATE[0].format(argument=text, definitions=fal_def_str)
                content = prompt_w_few_shot_examples(args, text)
            usr_pt = {"role": "user", "content": content}
            dialog = [sys_pt, usr_pt]
        js['chat_history'] = dialog
        js['arg_path'] = args.task_arg_path
    return js
    
SINGLE_PROMPT_TEMPLATE = {}

