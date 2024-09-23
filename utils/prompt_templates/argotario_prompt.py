
# from ..fallacy_utils import *
# from ..format_utils import format_tokens
from utils.fallacy_utils import *
from utils.format_utils import format_tokens
import regex
FALDEFS = ARGOTARIO_FALLACY_DEFINITIONS
argo_fals = list(FALDEFS.keys())
fal_def_str = "\n".join([f"{str(i+1)}. {fal} is {FALDEFS[fal]}" for i, fal in enumerate(argo_fals)])    
fal_name_str = ", ".join([f"{str(i+1)}. {fal}" for i, fal in enumerate(argo_fals)])

# We only include fallacy definitions in T5 prompts, meaning that fallacy names only occur once
# each fallacy name appears once and is followed by its definition.
T5_PROMPTS = [
'''Given the question and answer below, which of the fallacies defined below occurs in the answer?
Definitions:
{definitions}
Question and Answer:
{dialog}''',
'''Given the question and answer below, which of the fallacies defined below occurs in the answer?
Fallacies: {fallacies}
Definitions:
{definitions}
Question and Answer:
{dialog}'''
]

SYSTEM_PROMPT = [
'''You are a knowledgable expert in analysing fallacies in arguments. 
Please ensure that your responses are socially unbiased in nature.
Your response should not be lengthy.
Answer the last question.''',
]

BASELINE_PROMPTS = {
    'w_def':'''Based on the following definitions of fallacies,
{fallacies}
Given the conversation below,
{dialog}
Determine whether any of the fallacies defined above is present in B's argument replied to A?
Output your answer in JSON format {{"fallacy": name_of_the_fallacy, "explanation": in_a_sentence_or_two}}. If none of the fallacies is found, output {{"fallacy": "No Fallacy", "explanation": in_a_sentence_or_two}}. Only output JSON.''',
    'wo_def':'''Given five types of fallacies, namely, {fallacies}, and the following conversation,
{dialog}
Determine whether any of the fallacies given is present in B's argument replied to A?
Output your answer in JSON format {{"fallacy": name_of_the_fallacy, "explanation": in_a_sentence_or_two}}. If none of the fallacies is found, output {{"fallacy": "No Fallacy", "explanation": in_a_sentence_or_two}}. Only output JSON.''',
    'wo_def_tf':'''Given the following conversation,
{dialog}
determine whether any of the fallacies listed below is present in B's argument replied to A? Fallacies: {fallacies}. 
Output your answer in JSON format {{"fallacy": name_of_the_fallacy, "explanation": in_a_sentence_or_two}}. If none of the fallacies is found, output {{"fallacy": "No Fallacy", "explanation": in_a_sentence_or_two}}. Only output JSON.''',
    'w_ls_def':'''Based on the definitions of fallacies {fal_list} as below,
{fallacies}
Given the conversation below,
{dialog}
Determine whether any of the fallacies defined above is present in B's argument replied to A?
Output your answer in JSON format {{"fallacy": name_of_the_fallacy, "explanation": in_a_sentence_or_two}}. If none of the fallacies is found, output {{"fallacy": "No Fallacy", "explanation": in_a_sentence_or_two}}. Only output JSON.''',
}

SINGLE_PROMPTS = {
    'cot_w_def': '''Based on the following definitions of five fallacies,
{fallacies}
Given the conversation below,
{dialog}
Is any of the fallacies defined above present in B's argument replied to A? Output your final answer in JSON format {{"fallacy": name_of_the_fallacy}}. If none of the fallacies is present, output {{"fallacy": "No Fallacy"}}.
Now, let's think step by step.
''',
}

# Version 1: GPT-3.5 Multi-prompt, new tokens=128
v1_wo_def = {
0:"""In the following conversation,\n{dialog}\nIs B in favor of or against the topic proposed by A?""",
1:"""Considering B's stance, is B's argument logically reasonable or potentially fallacious? Explain why.""",
2: """Reflecting on your previous answers, given five types of fallacies, namely, {fallacies}, determine whether any of these fallacies is present in B's argument replied to A? Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. DO NOT output fallacy that is not in the five types. If none of these fallacies is found, output {{"fallacy": "No Fallacy"}}. Only output JSON.""".format(fallacies=fal_name_str),
}

v11_wo_def = {
0:"""In the following conversation,\n{dialog}\nIs B in favor of or against the topic proposed by A?""",
1:"""Considering B's stance, is B's argument logically reasonable or potentially fallacious? Explain why.""",
2: """If you find B's argument to be potentially fallacious, select the fallacy that is cloeset to your analysis from the following list: {fallacies}. Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. DO NOT output fallacy that is not in the list. If none of the listed fallacies is close to your analysis or you find B's argument to be logically reasonable, output {{"fallacy": "No Fallacy"}}. Only output JSON.""".format(fallacies=fal_name_str),
}

v12_wo_def = {
0:"""Given the following conversation,\n{dialog}\nConsidering B's stance towards the topic proposed by A, is B's argument logically reasonable or potentially fallacious? Give your analysis.""",
1: """According to your previous analysis, considering five types of fallacies: {fallacies}, determine whether any of these listed fallacies is present in B's argument replied to A? Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. If none of these fallacies is present, output {{"fallacy": "No Fallacy"}}. Only output JSON.""".format(fallacies=fal_name_str),
}

v13_wo_def = {
0:"""Given the following conversation,\n{dialog}\nSummarize B's stance towards the topic proposed by A and point out the premise(s) and conclusion of B's argument.""",
1:"""Whether or not the premise(s) of B's argument entail B's conclusion? Give your analysis.""",
2: """A fallacy is an argument where the premises do not entail the conclusion. According to your previous analysis, considering five types of fallacies: {fallacies}, determine whether any of these fallacies is present in B's argument replied to A? Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. If none of these fallacies is present, output {{"fallacy": "No Fallacy"}}. Only output JSON.""".format(fallacies=fal_name_str),
}

v14_wo_def = {
0:"""Given the following conversation,\n{dialog}\nSummarize the premise(s) and conclusion of B's argument towards the topic proposed by A.""",
1:"""A fallacy is defined as an argument where the premises do not entail the conclusion. According to this definition, is B's argument logically reasonable or potentially fallacious? Give your analysis.""",
2: """According to your previous analysis, considering five types of fallacies: {fallacies}, determine whether any of these fallacies is present in B's argument replied to A? Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. If none of these fallacies is present, output {{"fallacy": "No Fallacy"}}. Only output JSON.""".format(fallacies=fal_name_str),
}


v2_gen_def = {
0:'''Give a definition to each of the following types of fallacies in {fallacies}.'''.format(fallacies=fal_name_str),
1:'''Based on the definitions you provided, given the following conversation,\n{dialog}\ndetermine whether any of the defined fallacies is present in B's argument replied to A? Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. If none of these fallacies is present, output {{"fallacy": "No Fallacy"}}. Only output JSON.'''
}

v21_gen_def = {
0:'''Give a definition to each of the following types of fallacies: {fallacies}.'''.format(fallacies=fal_name_str),
1:'''Given the following conversation,\n{dialog}\nContrast each of the fallacy definitions you provided with B's argument replied to A and determine whether or not that fallacy is present in B's argument.''',
2:'''According to your previous analysis, summarize whether any of these given fallacies is present in B's argument, output your final decision in JSON format {"fallacy": name_of_the_fallacy}. If none of the listed fallacies is present, output {"fallacy": "No Fallacy"}. Only output JSON.'''
}
# # Version 2: Llama2 Multi-prompt: Ask Llama to give definitions and finally give the answer
# v2_gen_def = {
# 5: '''Based on your previous definitions of these fallacies,
# given the following conversation,
# {dialog}
# Determine whether any of the fallacies defined is present in B's argument replied to A?
# Output your answer in JSON format {{"fallacy": name_of_the_fallacy, "explanation": in_a_sentence_or_two}}. If none of the fallacies is found, output {{"fallacy": "No Fallacy", "explanation": in_a_sentence_or_two}}. Only output JSON.'''
# }
# ASK_SINGLE_FALLACY = """Give a definition to the fallacy of {name}."""
# OUTPUT_FORMAT = ''' Output in JSON format {"fallacy": name_of_the_fallacy, "definition": in_a_sentence_or_two}. Only output JSON.'''
# def_questions = {}
# for i, name in enumerate(argo_fals):
#     def_questions[i] = ASK_SINGLE_FALLACY.format(name=name) + OUTPUT_FORMAT
# v2_gen_def.update(def_questions)

# Version 3: Think step by step, #max_new_tokens = 512
v3_cot_w_def = {
0:'''Based on the following definitions of five fallacies,
{fallacies}
Given the conversation below,
[]
Is any of the fallacies defined above present in B's argument replied to A? Now, let's think step by step.'''.format(fallacies=fal_def_str),
1: '''Output your previous conclusion in JSON format {"fallacy": name_of_the_fallacy_defined}. If none of the defined fallacies is present, output {"fallacy": "No Fallacy"}. Only output JSON.'''
}

v3_cot_wo_def = {
0:'''Given the conversation below,
[]
and the following five types of fallacies, namely, {fallacies}. Is any of the fallacies listed present in B's argument replied to A? Now, let's think step by step.'''.format(fallacies=fal_name_str),
1: '''Output your previous conclusion in JSON format {"fallacy": name_of_the_fallacy}. If none of the listed fallacies is present, output {"fallacy": "No Fallacy"}. Only output JSON.'''
}

v4_wo_def = {
0:'''Given the following five types of fallacies, namely, {fallacies},
and given the conversation below,
[]
contrast each of the listed fallacies with B's argument replied to A and determine whether that fallacy is present in B's argument.'''.format(fallacies=fal_name_str),
1: '''According to your previous analysis, summarize whether any of these given fallacies is present in B's argument, output your final conclusion in JSON format {"fallacy": name_of_the_fallacy}. If none of the listed fallacies is present, output {"fallacy": "No Fallacy"}. Only output JSON.'''
}

v5_wo_def = {
0:'''Give a definition to each of the following types of fallacies in {fallacies}.'''.format(fallacies=fal_name_str),
1:"""Given the following conversation,\n{dialog}\nConsidering B's stance towards the topic proposed by A, is B's argument logically reasonable or potentially fallacious? Give your analysis.""",
2:'''Based on the definitions you provided and your previous analysis, determine whether any of the defined fallacies is present in B's argument replied to A? Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. If none of these fallacies is present, output {{"fallacy": "No Fallacy"}}. Only output JSON.'''
}

v6_wo_def = {
0:"""Given the following conversation,\n{dialog}\nConsidering B's stance towards the topic proposed by A, is B's argument logically reasonable or potentially fallacious? Give your analysis.""",   
1:'''Give a definition to each of the following types of fallacies in {fallacies}.'''.format(fallacies=fal_name_str),
2:'''Based on the definitions you provided and your previous analysis, determine whether any of the defined fallacies is present in B's argument replied to A? Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. If none of these fallacies is present, output {{"fallacy": "No Fallacy"}}. Only output JSON.'''
}

# Version 5: GPT-3.5 Multi-prompt
USER_PROMPTS5 = {
0:"""Given the following conversation,\n{dialog}\nSummarize what is B's point of view on the topic proposed by A?""",
1:"""Is B in favor of or against the topic proposed by A?""",
2:"""Considering B's stance, is B's argument logically reasonable or potentially fallacious? Explain why.""",
3: """Reflecting on all your previous answers, consider five types of fallacies, namely, {fallacies}, determine whether any of these fallacies is present in B's argument replied to A? Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. DO NOT output fallacy that's not in the given five types. If none of these fallacies is found, output {{"fallacy": "No Fallacy"}}. Only output JSON.""".format(fallacies=fal_name_str),
}

argotario_multiround_prompts = {
    'v1_wo_def' : v1_wo_def,
    'v11_wo_def': v11_wo_def,
    'v12_wo_def': v12_wo_def,
    'v13_wo_def': v13_wo_def,
    'v14_wo_def': v14_wo_def,
    'v2_gen_def': v2_gen_def,
    'v21_gen_def': v21_gen_def,
    'v3_cot_w_def': v3_cot_w_def,
    'v3_cot_wo_def': v3_cot_wo_def,
    'v4_wo_def': v4_wo_def,
    'v5_wo_def': v5_wo_def,
    'v6_wo_def': v6_wo_def,
}

def prompt_argotario(args, js):
    """
    input:
        js is one data sample in the format of dictionary
        js['text'] is a string of QA in a format like : "A: ....\nB: ..."
    """
    if args.exp_args.model.model_tag.startswith('t5'):
        assert len(regex.findall('A: ', js['text'])) == 1
        assert len(regex.findall('\nB: ', js['text'])) == 1
        text = js['text'].replace("A: ", "Question: ").replace("\nB: ", "\nAnswer: ").lower()
        js['seq_in'] = T5_PROMPTS[0].format(dialog=text, definitions=fal_def_str.lower())
        #js['seq_in'] = T5_PROMPTS[0].format(dialog=text, definitions=fal_def_str.lower(), fallacies=fal_name_str.lower())
    else:
        dialog = []
        sys_pt = {"role": "system", "content": SYSTEM_PROMPT[0]}
        if args.exp_args.model.run_multiprompt:
            USER_PROMPTS = argotario_multiround_prompts[args.scheme]
            tp = USER_PROMPTS[args.current_round]
            if "[]" in tp:
                tp = tp.replace("[]", "{dialog}")
            usr_ct = tp.format(dialog=js['text']) if '{dialog}' in tp else tp
            usr_prompt = {"role": "user", "content": usr_ct}
            if args.current_round == 0: # initial round
                dialog = [sys_pt, usr_prompt]
                #js.pop('text')
            else:
                last_prediction = js.pop('prediction')
                dialog = js['chat_history'] + [{"role":"assistant", "content":last_prediction}] + [usr_prompt]  
        else:
            text = js['text']
            fallacies = fal_def_str if args.scheme == 'w_def' else fal_name_str
            if args.exp_args.model.run_baseline:
                if args.scheme == 'w_ls_def':
                    usr_pt = {"role": "user", "content": BASELINE_PROMPTS[args.scheme].format(fallacies=fal_def_str, dialog=text, fal_list=fal_name_str)}
                else:
                    usr_pt = {"role": "user", "content": BASELINE_PROMPTS[args.scheme].format(fallacies=fallacies, dialog=text)}
            else:
                content = SINGLE_PROMPTS[args.scheme].format(dialog=text, fallacies=fallacies)
                usr_pt = {"role": "user", "content": content}
            dialog = [sys_pt, usr_pt]
        js['chat_history'] = dialog
        js['arg_path'] = args.task_arg_path
    return js
    