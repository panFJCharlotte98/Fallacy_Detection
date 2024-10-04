from utils.fallacy_utils import *
import json
import random

FALDEFS = LOGIC_FALLACY_DEFINITIONS
logic_fals = list(FALDEFS.keys())
fal_def_str = "\n".join([f"{str(i+1)}. {fal} is {FALDEFS[fal]}" for i, fal in enumerate(logic_fals)])    
fal_name_str = ", ".join([f"{str(i+1)}. {fal}" for i, fal in enumerate(logic_fals)])

T5_PROMPTS = [
'''Given the following segment and definitions, determine which of the fallacies defined below occurs in the segment.
Definitions:
{definitions}
Segment:
{segment}''',
'''Given the following segment and definitions, determine which of the fallacies defined below occurs in the segment.
Fallacies: {fallacies}
Definitions:
{definitions}
Segment:
{segment}'''
]

SYSTEM_PROMPT = [
'''You are a knowledgable expert in analysing fallacies in discourses. 
Please ensure that your responses are socially unbiased in nature.
Your response should not be lengthy.
Answer the last question.''',
]

BASELINE_PROMPTS = {
    'w_def':'''Based on the following definitions of fallacies,
{fallacies}
Given a segment of discourse below, determine which of the fallacies defined above is present in the argument?
Segment:\n{segment}
Output your answer in JSON format {{"fallacy": name_of_the_fallacy, "explanation": in_a_sentence_or_two}}. Only output JSON.''',
    'wo_def':'''Given 13 types of fallacies, namely, {fallacies}, and a segment of discourse below, determine which of the fallacies given is present in the argument?
Segment:\n{segment}
Output your answer in JSON format {{"fallacy": name_of_the_fallacy, "explanation": in_a_sentence_or_two}}. Only output JSON.''',

    'w_def_cf':'''Based on the following definitions of fallacies,
{fallacies}
Given a segment of discourse below,
Discourse: {segment}
Determine which of the fallacies defined above is present in the argument of the discourse? Output your answer in JSON format {{"fallacy": name_of_the_fallacy, "explanation": in_a_sentence_or_two}}. Only output JSON.''',
    'wo_def_cf':'''Given 13 types of fallacies, namely, {fallacies}, and given a segment of discourse below,
Discourse: {segment}
Determine which of the fallacies given is present in the argument of the discourse? Output your answer in JSON format {{"fallacy": name_of_the_fallacy, "explanation": in_a_sentence_or_two}}. Only output JSON.'''
}

# Version 1: GPT-3.5 Multi-prompt
v1_wo_def = {
0:"""Extract and summarize the focal argument in the following segment of discourse:\n{segment}""",
1:"""Is the argument logically reasonable or potentially fallacious? Explain why.""",
2: """Reflecting on your previous answers, given 13 types of fallacies, namely, {fallacies}, determine which of these fallacies is present in the argument? Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. DO NOT output fallacy that is not in the given types. Only output JSON.""".format(fallacies=fal_name_str),
}

v12_wo_def = {
0:"""Given the following segment of discourse,\n{segment}\nIs the focal argument of the discourse logically reasonable or potentially fallacious? Give your analysis.""",
1: """According to your previous analysis, considering 13 types of fallacies: {fallacies}, determine which of these listed fallacies is present in this focal argument of the discourse. Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. Only output JSON.""".format(fallacies=fal_name_str),
}

v12_wo_def_qf = {
0:"""Given the following segment of discourse, is the focal argument of the discourse logically reasonable or potentially fallacious? Segment: {segment}\nGive your analysis.""",
1: """According to your previous analysis, considering 13 types of fallacies: {fallacies}, determine which of these listed fallacies is present in this focal argument of the discourse. Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. Only output JSON.""".format(fallacies=fal_name_str),
}

v13_wo_def = {
0:"""Given the following discourse,\n{segment}\nExtract and summarize the focal argument by pointing out the premise(s) and conclusion of the argument.""",
1:"""Whether or not the argument's premise(s) entail the conclusion? Give your analysis.""",
2: """A fallacy is an argument where the premises do not entail the conclusion. According to your previous analysis, considering 13 types of fallacies: {fallacies}, determine which of these fallacies is present in this focal argument. Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. Only output JSON.""".format(fallacies=fal_name_str),
}

v14_wo_def = {
0:"""Given the following discourse,\n{segment}\nExtract and summarize the focal argument by pointing out the premise(s) and conclusion of the argument.""",
1:"""A fallacy is defined as an argument where the premises do not entail the conclusion. According to this definition, is this focal argument logically reasonable or potentially fallacious? Give your analysis.""",
2: """According to your previous analysis, considering 13 types of fallacies: {fallacies}, determine which of these listed fallacies is present in the focal argument of the discourse. Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. Only output JSON.""".format(fallacies=fal_name_str),
}

v2_gen_def = {
0:'''Give a definition to each of the following types of fallacies in {fallacies}.'''.format(fallacies=fal_name_str),
1:'''Based on the definitions you provided, given the following segment of discourse,\n{segment}\ndetermine which of the defined fallacies is present in the focal argument of the discourse. Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. Only output JSON.'''
}

v2_gen_def_qf = {
0:'''Give a definition to each of the following types of fallacies in {fallacies}.'''.format(fallacies=fal_name_str),
1:'''Based on the definitions you provided, given the following segment of discourse, determine which of the defined fallacies is present in the focal argument of the discourse. Discourse: {segment}\nOutput your answer in JSON format {{"fallacy": name_of_the_fallacy}}. Only output JSON.'''
}

v21_gen_def = {
0:'''Give a definition to each of the following types of fallacies: {fallacies}.'''.format(fallacies=fal_name_str),
1:'''Given the following segment of discourse,\n{segment}\nContrast each of the fallacy definitions you provided with the focal argument of the discourse and determine whether or not that fallacy is present in the focal argument.''',
2:'''According to your previous analysis, summarize whether any of these given fallacies is present in the focal argument of the discourse, output your final decision in JSON format {"fallacy": name_of_the_fallacy}. Only output JSON.'''
}

v3_cot_wo_def = {
0:'''Given a segment of discourse below,
[]
and the following 13 types of fallacies, namely, {fallacies}. Which of the listed fallacies is present in the focal argument of the discourse? Now, let's think step by step.'''.format(fallacies=fal_name_str),
1: '''Output your previous conclusion in JSON format {"fallacy": name_of_the_fallacy}. Only output JSON.'''
}

v3_cot_w_def = {
0:'''Based on the following definitions of 13 types of fallacies,
{fallacies}
Given a segment of discourse below,
[]
Which of the listed fallacies is present in the focal argument of the discourse? Now, let's think step by step.'''.format(fallacies=fal_def_str),
1: '''Output your previous conclusion in JSON format {"fallacy": name_of_the_fallacy}. Only output JSON.'''
}

v3_cot_wo_def_ff = {
0:'''Given the following 13 types of fallacies, namely, {fallacies} and given a segment of discourse below,
[]
Which of the listed fallacies is present in the focal argument of the discourse? Now, let's think step by step.'''.format(fallacies=fal_name_str),
1: '''Output your previous conclusion in JSON format {"fallacy": name_of_the_fallacy}. Only output JSON.'''
}

v4_wo_def = {
0:'''Given the following 13 types of fallacies, namely, {fallacies},
and given a segment of discourse below,
[]
contrast each of the listed fallacies with the focal argument of the discourse and determine whether that fallacy is present in the discourse.'''.format(fallacies=fal_name_str),
1: '''According to your previous analysis, summarize which of these fallacies is present in the focal argument of the discourse, output your final conclusion in JSON format {"fallacy": name_of_the_fallacy}. Only output JSON.'''
}

logic_multiround_prompts = {
    'v1_wo_def' : v1_wo_def,
    'v12_wo_def': v12_wo_def,
    'v13_wo_def': v13_wo_def,
    'v14_wo_def': v14_wo_def,
    'v2_gen_def': v2_gen_def,
    'v2_gen_def_qf': v2_gen_def_qf,
    'v21_gen_def': v21_gen_def,
    'v3_cot_wo_def': v3_cot_wo_def,
    'v3_cot_w_def': v3_cot_w_def,
    'v3_cot_wo_def_ff': v3_cot_wo_def_ff,
    'v4_wo_def': v4_wo_def,
    'v12_wo_def_qf':v12_wo_def_qf
}


def prompt_w_few_shot_examples(args, text):
    train_data = json.load(open(args.task_args.dataset.load_from + "train.json"))
    fal_examples = {fal.lower(): [] for fal in logic_fals}
    for data in train_data:
        assert data['label'][0].lower() in fal_examples
        fal_examples[data['label'][0].lower()].append(data)
    n_shots_per_class = {
        "Appeal to Emotion": 0,
        "Red Herring": 0,
        "Faulty Generalization": 0,
        "Ad Hominem": 0,
        "False Causality (Post Hoc Fallacy)": 0,
        "Circular Reasoning":0,
        "Ad Populum":0,
        "Fallacy of Converse (Affirming the Consequent)": 2,
        "False Dilemma": 0,
        "Equivocation":2,
        "Straw Man": 0,
        "Doubt Credibility": 2,
        "Intentional (Intentionally Wrong Argument)":2,
    }
    fs_classes = list(n_shots_per_class.keys())
    few_shots = []
    #for name, num in n_shots_per_class.items():
    for name in fs_classes:
        num = args.n_fewshots
        for js in random.sample(fal_examples[name.lower()], num):
            few_shots.append((name, js['text']))
    prompt = '''Given 13 types of fallacies, namely, {fallacies}, and a segment of discourse, determine which of the fallacies given is present in the discourse's argument? Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. Only output JSON.\n'''.format(fallacies = fal_name_str) + \
    "\n".join(['''Discourse segment: {example} | Fallacy: {{ "fallacy": "{fal_name}" }}'''.format(example=one_example[1], fal_name=one_example[0]) for one_example in few_shots]) + \
    '''\nDiscourse segment: {segment} | Fallacy: '''.format(segment=text)
    return prompt

def prompt_logic(args, js):
    """
    input:
        js is one data sample in the format of dictionary
        js['text'] is a string of QA in a format like : "A: ....\nB: ..."
    """
    if args.exp_args.model.model_tag.startswith('t5'):
        text = js['text'].lower()
        js['seq_in'] = T5_PROMPTS[0].format(segment=text, definitions=fal_def_str.lower())
        #js['seq_in'] = T5_PROMPTS[0].format(segment=text, definitions=fal_def_str.lower(), fallacies=fal_name_str.lower())
    else:
        dialog = []
        sys_pt = {"role": "system", "content": SYSTEM_PROMPT[0]}
        if args.exp_args.model.run_multiprompt:
            USER_PROMPTS = logic_multiround_prompts[args.scheme]
            tp = USER_PROMPTS[args.current_round]
            if "[]" in tp:
                tp = tp.replace("[]", "{segment}")
            usr_ct = tp.format(segment=js['text']) if '{segment}' in tp else tp
            usr_prompt = {"role": "user", "content": usr_ct}
            if args.current_round == 0: # initial round
                dialog = [sys_pt, usr_prompt]
                #js.pop('text')
            else:
                last_prediction = js.pop('prediction')
                dialog = js['chat_history'] + [{"role":"assistant", "content":last_prediction}] + [usr_prompt]  
        else:
            text = js['text']
            if args.exp_args.model.run_baseline:
                fallacies = fal_def_str if args.scheme.startswith('w_def') else fal_name_str
                content = BASELINE_PROMPTS[args.scheme].format(fallacies=fallacies, segment=text)
            else:
                #content = SINGLE_FEWSHOT_PROMPTS[args.scheme] + '''\nSegment: {segment}. Fallacy: '''.format(segment=text)
                content = prompt_w_few_shot_examples(args, text)
            usr_pt = {"role": "user", "content": content}
            dialog = [sys_pt, usr_pt]
        js['chat_history'] = dialog
        js['arg_path'] = args.task_arg_path
    return js

  
SINGLE_FEWSHOT_PROMPTS = {
    'wo_def': '''Given 13 types of fallacies, namely, {fallacies}, and a segment of discourse, determine which of the fallacies given is present in the segment's argument? Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. Only output JSON.\n'''.format(fallacies = fal_name_str) + \
    "\n".join(['''Segment: {example}. Fallacy: {{"fallacy": "{fal_name}" }}'''.format(example=fal_exp, fal_name=fal_n) for fal_n, fal_exp in LOGIC_FEWSHOTS.items()]),
    'w_def': '''Based on the following definitions of fallacies,
{fallacies}
Given a segment of discourse, determine which of the fallacies defined above is present in the segment's argument? Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. Only output JSON.\n'''.format(fallacies = fal_def_str) + \
    "\n".join(['''Segment: {example}. Fallacy: {{"fallacy": "{fal_name}" }}'''.format(example=fal_exp, fal_name=fal_n) for fal_n, fal_exp in LOGIC_FEWSHOTS.items()])
}