
from utils.fallacy_utils import *
import regex
import json
import random

FALDEFS = PROPAGANDA_FALLACY_DEFINITIONS
propaganda_fals = list(FALDEFS.keys())
fal_def_str = "\n".join([f"{str(i+1)}. {fal} is {FALDEFS[fal]}" for i, fal in enumerate(propaganda_fals)])    
fal_name_str = ", ".join([f"{str(i+1)}. {fal}" for i, fal in enumerate(propaganda_fals)])

T5_PROMPTS = [
'''Given the following segment of a news article with a focused fragment and some definitions of fallacies, determine which of the fallacies defined below occurs in the focused fragment of the news.
Definitions:
{definitions}
News article segment:
{segment}
Focused fragment: 
{fragment}'''
]

SYSTEM_PROMPT = [
'''You are a knowledgable expert in analysing fallacies in news articles. 
Please ensure that your responses are socially unbiased in nature.
Your response should not be lengthy.
Answer the last question.''',
]

BASELINE_PROMPTS = {
    'w_def':'''Based on the following definitions of fallacies,
{fallacies}
Given a segment of a news article below,
{segment}
Determine which of the fallacies defined above is present in the news fragment highlighted by '<>'?
Output your answer in JSON format {{"fallacy": name_of_the_fallacy, "explanation": in_a_sentence_or_two}}. Only output JSON.''',
    
    'wo_def':'''Given 13 types of fallacies, namely, {fallacies}, and a segment of a news article below,
{segment}
Determine which of the fallacies given is present in the news fragment highlighted by '<>'?
Output your answer in JSON format {{"fallacy": name_of_the_fallacy, "explanation": in_a_sentence_or_two}}. Only output JSON.''',
    
    'w_def_qf':'''Based on the following definitions of fallacies,
{fallacies}
Given a segment of a news article below, determine which of the fallacies defined above is present in the news fragment highlighted by '<>'?
News article segment:
{segment}
Output your answer in JSON format {{"fallacy": name_of_the_fallacy, "explanation": in_a_sentence_or_two}}. Only output JSON.''',
    
    'wo_def_qf':'''Given 13 types of fallacies, namely, {fallacies}, and a segment of a news article below, determine which of the fallacies given is present in the news fragment highlighted by '<>'?
News article segment:
{segment}
Output your answer in JSON format {{"fallacy": name_of_the_fallacy, "explanation": in_a_sentence_or_two}}. Only output JSON.
'''
}

# Version 1: GPT-3.5 Multi-prompt
v1_wo_def = {
0:"""Given the following segment of a news article,\n{segment}\nSummarize 1) what the news means by the fragment highlighted by '<>' and 2) the focal argument of the fragment highlighted by '<>'.""",
1:"""Is this fragment highlighted by '<>' logically reasonable or potentially fallacious? Explain why.""",
2: """Reflecting on your previous analysis, given 13 types of fallacies, namely, {fallacies}, determine which of these listed fallacies is present in the news fragment highlighted by '<>'? Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. DO NOT output fallacy that is not in the given types. Only output JSON.""".format(fallacies=fal_name_str),
}

v12_wo_def = {
0:"""Given the following segment of a news article,\n{segment}\nIs the fragment highlighted by '<>' logically reasonable or potentially fallacious? Give your analysis.""",
1: """According to your previous analysis, considering 13 types of fallacies: {fallacies}, determine which of these listed fallacies is present in the news fragment highlighted by '<>'? Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. Only output JSON.""".format(fallacies=fal_name_str),
}

v13_wo_def = {
0:"""Given the following segment of a news article,\n{segment}\nExtract and summarize the focal argument of the fragment highlighted by '<>' by pointing out its premise(s) and conclusion""",
1:"""Whether or not the highlighted fragment's premise(s) entail the conclusion? Give your analysis.""",
2: """A fallacy is an argument where the premises do not entail the conclusion. According to your previous analysis, considering 13 types of fallacies: {fallacies}, determine which of these fallacies is present in this news fragment highlighted by '<>'. Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. Only output JSON.""".format(fallacies=fal_name_str),
}

v14_wo_def = {
0:"""Given the following segment of a news article,\n{segment}\nExtract and summarize the focal argument of the fragment highlighted by '<>' by pointing out its premise(s) and conclusion""",
1:"""A fallacy is defined as an argument where the premises do not entail the conclusion. According to this definition, is the fragment highlighted by '<>' logically reasonable or potentially fallacious? Give your analysis.""",
2: """According to your previous analysis, considering 13 types of fallacies: {fallacies}, determine which of these listed fallacies is present in this news fragment highlighted by '<>'. Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. Only output JSON.""".format(fallacies=fal_name_str),
}

v2_gen_def = {
0:'''Give a definition to each of the following types of fallacies in {fallacies}.'''.format(fallacies=fal_name_str),
1:'''Based on the definitions you provided, given the following segment of a news article,\n{segment}\ndetermine which of the defined fallacies is present in the news fragment highlighted by '<>'? Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. Only output JSON.'''
}

v21_gen_def = {
0:'''Give a definition to each of the following types of fallacies: {fallacies}.'''.format(fallacies=fal_name_str),
1:'''Given the following segment of a news article with a fragment highlighted by '<>',\n{segment}\nContrast each of the fallacy definitions you provided with the news fragment highlighted by '<>' and determine whether or not that fallacy is present in the fragment.''',
2:'''According to your previous analysis, summarize which of these fallacies is present in the news fragment highlighted by '<>', output your final decision in JSON format {"fallacy": name_of_the_fallacy}. Only output JSON.'''
}

v3_cot_wo_def = {
0:'''Given the following segment of a news article below,
[]
and the following 13 types of fallacies, namely, {fallacies}, which of the listed fallacies is present in the news fragment highlighted by '<>'? Now, let's think step by step.'''.format(fallacies=fal_name_str),
1: '''Output your previous conclusion in JSON format {"fallacy": name_of_the_fallacy}. Only output JSON.'''
}

v4_wo_def = {
0:'''Given the following 13 types of fallacies, namely, {fallacies},
and given a segment of a news article below, where a fragment is highlighted by '<>',
[]
contrast each of the listed fallacies with the news fragment highlighted by '<>' and determine whether that fallacy is present in the news fragment.'''.format(fallacies=fal_name_str),
1: '''According to your previous analysis, summarize which of these fallacies is present in the news fragment highlighted by '<>', output your final conclusion in JSON format {"fallacy": name_of_the_fallacy}. Only output JSON.'''
}

propaganda_multiround_prompts = {
    'v1_wo_def' : v1_wo_def,
    'v12_wo_def': v12_wo_def,
    'v13_wo_def': v13_wo_def,
    'v14_wo_def': v14_wo_def,
    'v2_gen_def': v2_gen_def,
    'v21_gen_def': v21_gen_def,
    'v3_cot_wo_def': v3_cot_wo_def,
    'v4_wo_def': v4_wo_def
}

def prompt_propaganda(args, js):
    """
    input:
        js is one data sample in the format of dictionary
        js['text'] is a string of QA in a format like : "A: ....\nB: ..."
    """
    segment = f"News title: {js['title']}. Segment: "
    if args.context_window == 0:
        segment += js['text']
    elif args.context_window == 1:
        context = []
        if js['pre_text']:
            context.append(js['pre_text'][-1])
        context.append(js['text'])
        if js['post_text']:
            context.append(js['post_text'][0])
        segment += " ".join(context)
    else:
        segment += " ".join(js['pre_text'] + [js['text']] + js['post_text'])

    if args.exp_args.model.model_tag.startswith('t5'):
        text = segment.lower()
        fragment = js['text'].lower()
        js['seq_in'] = T5_PROMPTS[0].format(segment=text, fragment=fragment, definitions=fal_def_str.lower())
    else:
        dialog = []
        sys_pt = {"role": "system", "content": SYSTEM_PROMPT[0]}
        if args.exp_args.model.run_multiprompt:
            USER_PROMPTS = propaganda_multiround_prompts[args.scheme]
            tp = USER_PROMPTS[args.current_round]
            if "[]" in tp:
                tp = tp.replace("[]", "{segment}")
            usr_ct = tp.format(segment=segment) if '{segment}' in tp else tp
            usr_prompt = {"role": "user", "content": usr_ct}
            if args.current_round == 0: # initial round
                dialog = [sys_pt, usr_prompt]
                #js.pop('text')
            else:
                last_prediction = js.pop('prediction')
                dialog = js['chat_history'] + [{"role":"assistant", "content":last_prediction}] + [usr_prompt]  
        else:
            text = segment
            if args.exp_args.model.run_baseline:
                fallacies = fal_def_str if args.scheme == 'w_def' else fal_name_str
                content = BASELINE_PROMPTS[args.scheme].format(fallacies=fallacies, segment=text)
            else:
                content = SINGLE_FEWSHOT_PROMPTS[args.scheme] + '''\nNews Article Segment: {segment}. Fallacy: '''.format(segment=text)
            usr_pt = {"role": "user", "content": content}
            dialog = [sys_pt, usr_pt]
        js['chat_history'] = dialog
        js['arg_path'] = args.task_arg_path
    return js
    
SINGLE_FEWSHOT_PROMPTS = {
    'wo_def': '''Given 18 types of propaganda techniques or fallacies, namely, {fallacies}, and a segment of a news article, determine which of the propaganda techniques or fallacies given is present in the news fragment highlighted by '<>'? Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. Only output JSON.\n'''.format(fallacies = fal_name_str) + \
    "\n".join(['''News Article Segment: {example}. Fallacy: {{"fallacy": "{fal_name}" }}'''.format(example=fal_exp, fal_name=fal_n) for fal_n, fal_exp in PROPAGANDA_FEWSHOTS.items()]),
}

def prompt_w_few_shot_examples(args, text):
    def format_segment(js):
        segment = f"News title: {js['title']}. Segment: "
        if args.context_window == 0:
            segment += js['text']
        elif args.context_window == 1:
            context = []
            if js['pre_text']:
                context.append(js['pre_text'][-1])
            context.append(js['text'])
            if js['post_text']:
                context.append(js['post_text'][0])
            segment += " ".join(context)
        else:
            segment += " ".join(js['pre_text'] + [js['text']] + js['post_text'])
        return segment
    train_data = json.load(open(args.task_args.dataset.load_from + "train.json"))
    fal_examples = {fal.lower(): [] for fal in propaganda_fals}
    for data in train_data:
        assert data['label'][0].lower() in fal_examples
        fal_examples[data['label'][0].lower()].append(data)
    n_shots_per_class = {
        'Name-calling': 0,
        'Doubt Credibility': 2,
        'Appeal to Fear': 0,
        'Flag-waving': 0,
        'Causal Oversimplification': 0,
        'Appeal to False Authority': 2,
        'False Dilemma': 2,
        'Whataboutism': 0,
        'Reductio Ad Hitlerum': 0,
        'Red Herring': 2,
        'Straw Man': 2,
        'Ad Populum': 2,
        'Equivocation': 2,
    }
    few_shots = []
    for name, num in n_shots_per_class.items():
        for js in random.sample(fal_examples[name.lower()], num):
            few_shots.append((name, format_segment(js)))
    prompt = '''Given 13 types of fallacies, namely, {fallacies}, and a segment of a news article, determine which of the fallacies given is present in the news fragment highlighted by '<>'? Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. Only output JSON.\n'''.format(fallacies = fal_name_str) + \
    "\n".join(['''News article segment: {example} | Fallacy: {{ "fallacy": "{fal_name}" }}'''.format(example=one_example[1], fal_name=one_example[0]) for one_example in few_shots]) + \
    '''\nNews article segment: {segment} | Fallacy: '''.format(segment=text)
    return prompt