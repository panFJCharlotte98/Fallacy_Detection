
from utils.fallacy_utils import *
import regex
FALDEFS = REDDIT_FALLACY_DEFINITIONS
reddit_fals = list(FALDEFS.keys())
fal_def_str = "\n".join([f"{str(i+1)}. {fal} is {FALDEFS[fal]}" for i, fal in enumerate(reddit_fals)])    
fal_name_str = ", ".join([f"{str(i+1)}. {fal}" for i, fal in enumerate(reddit_fals)])

T5_PROMPTS = [
'''Given the following fallacy definitions and a snippet of reddit comments, determine which of the fallacies defined below occurs in the focused comment.
Definitions:
{definitions}
Snippet:
{snippet}
Focused comment:
{comment}''',
'''Given the following fallacy definitions and a snippet of reddit comments, determine which of the fallacies defined below occurs in the focused comment.
Fallacies: {fallacies}
Definitions:
{definitions}
Snippet:
{snippet}
Focused comment:
{comment}'''
]

SYSTEM_PROMPT = [
'''You are a knowledgable expert in analysing fallacies in social media comments. 
Please ensure that your responses are socially unbiased in nature.
Your response should not be lengthy.
Answer the last question.''',
]

BASELINE_PROMPTS = {
    'w_def':'''Based on the following definitions of fallacies,
{fallacies}
Given a snippet of reddit comments for a certain topic as below,
{snippet}
Determine which of the fallacies defined above is present in the part(s) of comment highlighted by '<>'?
Output your answer in JSON format {{"fallacy": name_of_the_fallacy, "explanation": in_a_sentence_or_two}}. Only output JSON.''',
    'wo_def':'''Given eight types of fallacies, namely, {fallacies}, and a snippet of reddit comments for a certain topic as below,
{snippet}
Determine which of the fallacies given is present in the part(s) of comment highlighted by '<>'?
Output your answer in JSON format {{"fallacy": name_of_the_fallacy, "explanation": in_a_sentence_or_two}}. Only output JSON.''',

    'w_def_qf':'''Based on the following definitions of fallacies,
{fallacies}
Given a snippet of reddit comments for a certain topic as below, determine which of the fallacies defined above is present in the part(s) of comment highlighted by '<>'?
Snippet: {snippet}
Output your answer in JSON format {{"fallacy": name_of_the_fallacy, "explanation": in_a_sentence_or_two}}. Only output JSON.''',
    'wo_def_qf':'''Given eight types of fallacies, namely, {fallacies}, and given a snippet of reddit comments for a certain topic as below, determine which of the fallacies given is present in the part(s) of comment highlighted by '<>'?
Snippet: {snippet}
Output your answer in JSON format {{"fallacy": name_of_the_fallacy, "explanation": in_a_sentence_or_two}}. Only output JSON.'''
}

# Version 1: GPT-3.5 Multi-prompt
v1_wo_def = {
0:"""Given the following snippet of reddit comments for a certain topic:\n{snippet}\nsummarize the central argument of the part(s) of comment highlighted by '<>'.""",
1:"""Is the comment's argument highlighted by '<>' logically reasonable or potentially fallacious? Give your analysis.""",
2: """According to your previous analysis, given eight types of fallacies, namely, {fallacies}, determine which of these fallacies is present in the comment's argument highlighted by '<>'? Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. DO NOT output fallacy that is not in the given types. Only output JSON.""".format(fallacies=fal_name_str),
}

v1_wo_def_qf = {
0:"""Given the following snippet of reddit comments for a certain topic, summarize the central argument of the part(s) of comment highlighted by '<>'. Snippet: {snippet}""",
1:"""Is the comment's argument highlighted by '<>' logically reasonable or potentially fallacious? Give your analysis.""",
2: """According to your previous analysis, given eight types of fallacies, namely, {fallacies}, determine which of these fallacies is present in the comment's argument highlighted by '<>'? Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. DO NOT output fallacy that is not in the given types. Only output JSON.""".format(fallacies=fal_name_str),
}

v12_wo_def = {
0:"""Given the following snippet of reddit comments for a certain topic:\n{snippet}\nIs the comment's argument highlighted by '<>' logically reasonable or potentially fallacious? Give your analysis.""",
1: """According to your previous analysis, considering eight types of fallacies: {fallacies}, determine which of these listed fallacies is present in the comment's argument highlighted by '<>'? Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. Only output JSON.""".format(fallacies=fal_name_str),
}

v12_wo_def_qf = {
0:"""Given the following snippet of reddit comments for a certain topic, is the comment's argument highlighted by '<>' logically reasonable or potentially fallacious? Snippet: {snippet}\nGive your analysis.""",
1: """According to your previous analysis, considering eight types of fallacies: {fallacies}, determine which of these listed fallacies is present in the comment's argument highlighted by '<>'? Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. Only output JSON.""".format(fallacies=fal_name_str),
}

v13_wo_def = {
0:"""Given the following snippet of reddit comments for a certain topic: {snippet}\nSummarize the comment's argument highlighted by '<>' by pointing out the premise(s) and conclusion of the argument.""",
1:"""Whether or not the premise(s) of the comment's argument highlighted by '<>' entail the conclusion? Give your analysis.""",
2: """A fallacy is an argument where the premises do not entail the conclusion. According to your previous analysis, considering eight types of fallacies: {fallacies}, determine which of these listed fallacies is present in the comment's argument highlighted by '<>'. Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. Only output JSON.""".format(fallacies=fal_name_str),
}

v14_wo_def = {
0:"""Given the following snippet of reddit comments for a certain topic: {snippet}\nExtract and summarize the comment's argument highlighted by '<>' by pointing out the premise(s) and conclusion of the argument.""",
1:"""A fallacy is defined as an argument where the premises do not entail the conclusion. According to this definition, is the comment's argument highlighted by '<>' logically reasonable or potentially fallacious? Give your analysis.""",
2: """According to your previous analysis, considering eight types of fallacies: {fallacies}, determine which of these listed fallacies is present in the comment's argument highlighted by '<>'. Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. Only output JSON.""".format(fallacies=fal_name_str),
}

v2_gen_def = {
0:'''Give a definition to each of the following eight types of fallacies in {fallacies}.'''.format(fallacies=fal_name_str),
1:'''Based on the definitions you provided, given the following snippet of reddit comments for a certain topic:\n{snippet}\ndetermine which of the defined fallacies is present in the comment's argument highlighted by '<>'? Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. Only output JSON.'''
}

# Snippet:
v2_gen_def_qf = {
0:'''Give a definition to each of the following eight types of fallacies in {fallacies}.'''.format(fallacies=fal_name_str),
1:'''Based on the definitions you provided, given the following snippet of reddit comments for a certain topic, determine which of the defined fallacies is present in the comment's argument highlighted by '<>'? Snippet: {snippet}\nOutput your answer in JSON format {{"fallacy": name_of_the_fallacy}}. Only output JSON.'''
}

v21_gen_def = {
0:'''Give a definition to each of the following types of fallacies: {fallacies}.'''.format(fallacies=fal_name_str),
1:'''Given the following snippet of reddit comments for a certain topic:\n{snippet}\nContrast each of the fallacy definitions you provided with the comment's argument highlighted by '<>' and determine whether or not that fallacy is present in this highlighted argument.''',
2:'''According to your previous analysis, summarize whether any of these given fallacies is present in the comment's argument highlighted by '<>', output your final decision in JSON format {"fallacy": name_of_the_fallacy}. Only output JSON.'''
}

v3_cot_wo_def = {
0:'''Given the following snippet of reddit comments for a certain topic,
[]
and the following eight types of fallacies, namely, {fallacies}, which of the listed fallacies is present in the comment's argument highlighted by '<>'? Now, let's think step by step.'''.format(fallacies=fal_name_str),
1: '''Output your previous conclusion in JSON format {"fallacy": name_of_the_fallacy}. Only output JSON.'''
}

v3_cot_w_def = {
0:'''Based on the following definitions of 8 types of fallacies,
{fallacies}
Given the following snippet of reddit comments for a certain topic,
[]
Which of the listed fallacies is present in the comment's argument highlighted by '<>'? Now, let's think step by step.'''.format(fallacies=fal_def_str),
1: '''Output your previous conclusion in JSON format {"fallacy": name_of_the_fallacy}. Only output JSON.'''
}

v3_cot_wo_def_ff = {
0:'''Given the following eight types of fallacies, namely, {fallacies}, and given a snippet of reddit comments for a certain topic as below,
[]
Which of the listed fallacies is present in the comment's argument highlighted by '<>'? Now, let's think step by step.'''.format(fallacies=fal_name_str),
1: '''Output your previous conclusion in JSON format {"fallacy": name_of_the_fallacy}. Only output JSON.'''
}

v4_wo_def = {
0:'''Given the following eight types of fallacies, namely, {fallacies},
and given the following snippet of reddit comments for a certain topic,
[]
contrast each of the listed fallacies with the comment's argument highlighted by '<>' and determine whether that fallacy is present in the this argument highlighted by '<>'.'''.format(fallacies=fal_name_str),
1: '''According to your previous analysis, summarize which of these fallacies is present in the comment's argument highlighted by '<>', output your final conclusion in JSON format {"fallacy": name_of_the_fallacy}. Only output JSON.'''
}

reddit_multiround_prompts = {
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
    'v1_wo_def_qf' : v1_wo_def_qf,
    'v12_wo_def_qf': v12_wo_def_qf
}

def prompt_reddit(args, js):
    """
    input:
        js is one data sample in the format of dictionary
        js['text'] is a string of QA in a format like : "A: ....\nB: ..."
    """
    snippet = js['text']
    if args.exp_args.model.model_tag.startswith('t5'):
        text = snippet.lower()
        js['seq_in'] = T5_PROMPTS[0].format(snippet=text, comment=js['fal_text'].lower(), definitions=fal_def_str.lower())
        #js['seq_in'] = T5_PROMPTS[0].format(snippet=text, comment=js['fal_text'].lower(), definitions=fal_def_str.lower(), fallacies=fal_name_str.lower())
    else:
        dialog = []
        sys_pt = {"role": "system", "content": SYSTEM_PROMPT[0]}
        if args.exp_args.model.run_multiprompt:
            USER_PROMPTS = reddit_multiround_prompts[args.scheme]
            tp = USER_PROMPTS[args.current_round]
            if "[]" in tp:
                tp = tp.replace("[]", "{snippet}")
            usr_ct = tp.format(snippet=snippet) if '{snippet}' in tp else tp
            usr_prompt = {"role": "user", "content": usr_ct}
            if args.current_round == 0: # initial round
                dialog = [sys_pt, usr_prompt]
                #js.pop('text')
            else:
                last_prediction = js.pop('prediction')
                dialog = js['chat_history'] + [{"role":"assistant", "content":last_prediction}] + [usr_prompt]  
        else:
            text = snippet
            if args.exp_args.model.run_baseline:
                fallacies = fal_def_str if args.scheme.startswith('w_def') else fal_name_str
                content = BASELINE_PROMPTS[args.scheme].format(fallacies=fallacies, snippet=text)
            else:
                content = SINGLE_PROMPT_TEMPLATE[0].format(argument=text, definitions=fal_def_str)
            usr_pt = {"role": "user", "content": content}
            dialog = [sys_pt, usr_pt]
        js['chat_history'] = dialog
        js['arg_path'] = args.task_arg_path
    return js
    
SINGLE_PROMPT_TEMPLATE = {}

