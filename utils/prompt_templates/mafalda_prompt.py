from utils.fallacy_utils import *
from utils.format_utils import format_tokens
import regex
FALDEFS = MAFALDA_FALLACY_DEFINITIONS
LOGICAL_FALDEFS = MAFALDA_FALLACY_LOGICAL_DEFINITIONS
mafalda_fals = list(FALDEFS.keys())
fal_def_str = "\n".join([f"{str(i+1)}. {fal}: {FALDEFS[fal]}" for i, fal in enumerate(mafalda_fals)])    
fal_name_str = ", ".join([f"{str(i+1)}. {fal}" for i, fal in enumerate(mafalda_fals)])
logical_fal_def_str = "\n".join([f"{str(i+1)}. {fal}: {LOGICAL_FALDEFS[fal]}" for i, fal in enumerate(mafalda_fals)])

T5_PROMPTS = [
'''Given the segment of text below, which of the fallacies defined below occurs in the segment?
Definitions:
{definitions}
Segment of text:
{segment}''',
'''Given the segment of text below, which of the fallacies defined below occurs in the segment?
Fallacies: {fallacies}
Definitions:
{definitions}
Segment of text:
{segment}'''
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
Given the segment of text below,
{segment}
Determine whether or not any of the fallacies defined above is present in the argument of the text?
Output your answer in JSON format {{"fallacy": name_of_the_fallacy, "explanation": in_a_sentence_or_two}}. If none of the fallacies is found, output {{"fallacy": "No Fallacy", "explanation": in_a_sentence_or_two}}. Only output JSON.''',
    
    'wo_def':'''Given the following types of fallacies, namely, {fallacies}, and the following segment of text,
{segment}
Determine whether or not any of the fallacies given is present in the argument of the text?
Output your answer in JSON format {{"fallacy": name_of_the_fallacy, "explanation": in_a_sentence_or_two}}. If none of the fallacies is found, output {{"fallacy": "No Fallacy", "explanation": in_a_sentence_or_two}}. Only output JSON.''',
    
    'wo_def_tf':'''Given the following segment of text,
{segment}
determine whether or not any of the fallacies listed below is present in the argument of the text? Fallacies: {fallacies}. 
Output your answer in JSON format {{"fallacy": name_of_the_fallacy, "explanation": in_a_sentence_or_two}}. If none of the fallacies is found, output {{"fallacy": "No Fallacy", "explanation": in_a_sentence_or_two}}. Only output JSON.''',
    
    'w_logic_def': '''An argument consists of an assertion called the conclusion and one or more assertions called premises, where the premises are intended to establish the truth of the conclusion. Premises or conclusions can be implicit in an argument. A fallacy is an argument where the premises do not entail the conclusion. Following the notations:
Let E stand for entity (persons, organizations) or group of entities; Let P or P1, P2, P3, etc. stand for premises, properties, or possibilities; Let A stand for attack; Let C stand for conclusion.
The formal definitions of various types of fallacies are given below:
{fallacies}
Based on the above definitions of fallacies, given the segment of text below,
{segment}
Determine whether or not any of the fallacies defined above is present in the argument of the text?
Output your answer in JSON format {{"fallacy": name_of_the_fallacy, "explanation": in_a_sentence_or_two}}. If none of the fallacies is found, output {{"fallacy": "No Fallacy", "explanation": in_a_sentence_or_two}}. Only output JSON.'''
}

SINGLE_PROMPTS = {
}

# Version 1: GPT-3.5 Multi-prompt, new tokens=128
v1_wo_def = {
0:"""Extract and summarize the main argument made in the following segment of text,\n{segment}.""",
1:"""Is the argument of the text logically reasonable or potentially fallacious? Explain why.""",
2: """Reflecting on your previous analysis, given the following types of fallacies, namely, {fallacies}, determine whether or not any of these fallacies is present in the argument of the text? Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. DO NOT output fallacy that is not in the given types. If none of these fallacies is found, output {{"fallacy": "No Fallacy"}}. Only output JSON.""".format(fallacies=fal_name_str),
}

v12_wo_def = {
0:"""Given the following segment of text,\n{segment}\nIs the argument of the text logically reasonable or potentially fallacious? Give your analysis.""",
1: """According to your previous analysis, considering the following types of fallacies: {fallacies}, determine whether or not any of these listed fallacies is present in the argument of the text? Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. If none of these fallacies is present, output {{"fallacy": "No Fallacy"}}. Only output JSON.""".format(fallacies=fal_name_str),
}

v13_wo_def = {
0:"""Given the following segment of text,\n{segment}\nSummarize the argument of the text by pointing out the premise(s) and conclusion of the argument.""",
1:"""Whether or not the premise(s) of the text's argument entail the conclusion? Give your analysis.""",
2: """A fallacy is an argument where the premises do not entail the conclusion. According to your previous analysis, considering the following types of fallacies: {fallacies}, determine whether or not any of these fallacies is present in the argument of the text? Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. If none of these fallacies is present, output {{"fallacy": "No Fallacy"}}. Only output JSON.""".format(fallacies=fal_name_str),
}

v14_wo_def = {
0:"""Given the following segment of text,\n{segment}\nSummarize the premise(s) and conclusion of the text's argument.""",
1:"""A fallacy is defined as an argument where the premises do not entail the conclusion. According to this definition, is the argument of the text logically reasonable or potentially fallacious? Give your analysis.""",
2: """According to your previous analysis, considering the following types of fallacies: {fallacies}, determine whether any of these fallacies is present in the argument of the text? Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. If none of these fallacies is present, output {{"fallacy": "No Fallacy"}}. Only output JSON.""".format(fallacies=fal_name_str),
}

v2_gen_def = {
0:'''Give a definition to each of the following types of fallacies: {fallacies}.'''.format(fallacies=fal_name_str),
1:'''Based on the definitions you provided, given the following segment of text,\n{segment}\ndetermine whether or not any of the defined fallacies is present in the argument of the text? Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. If none of these fallacies is present, output {{"fallacy": "No Fallacy"}}. Only output JSON.'''
}

v21_gen_def = {
0:'''Give a definition to each of the following types of fallacies: {fallacies}.'''.format(fallacies=fal_name_str),
1:'''Given the following segment of text,\n{segment}\nContrast each of the fallacy definitions you provided with the argument of the text and determine whether or not that fallacy is present in the text.''',
2:'''According to your previous analysis, determine whether any of these fallacies is present in the text, output your final decision in JSON format {"fallacy": name_of_the_fallacy}. If none of the listed fallacies is present, output {"fallacy": "No Fallacy"}. Only output JSON.'''
}

# Version 3: Think step by step, #max_new_tokens = 512
v3_cot_w_def = {
0:'''Based on the following definitions of various types of fallacies,
{fallacies}
Given the segment of text below,
[]
Is any of the fallacies defined above is present in the argument of the text? Now, let's think step by step.'''.format(fallacies=fal_def_str),
1: '''Output your previous conclusion in JSON format {"fallacy": name_of_the_fallacy}. If none of the defined fallacies is present, output {"fallacy": "No Fallacy"}. Only output JSON.'''
}

v3_cot_wo_def = {
0:'''Given the segment of text below,
[]
and given the following types of fallacies, namely, {fallacies}. Is any of the fallacies listed present in the argument of the text? Now, let's think step by step.'''.format(fallacies=fal_name_str),
1: '''Output your previous conclusion in JSON format {"fallacy": name_of_the_fallacy_listed}. If none of the listed fallacies is present, output {"fallacy": "No Fallacy"}. Only output JSON.'''
}

v4_wo_def = {
0:'''Given the following types of fallacies, namely, {fallacies},
and a segment of text below,
[]
contrast each of the listed fallacies with the argument of the text and determine whether that fallacy is present in the text.'''.format(fallacies=fal_name_str),
1: ''''According to your previous analysis, summarize whether any of these fallacies is present in the text, output your final decision in JSON format {"fallacy": name_of_the_fallacy}. If none of the listed fallacies is present, output {"fallacy": "No Fallacy"}. Only output JSON.'''
}

hierarchical_gen_def = {
0:'''Based on Aristotle's categories of persuasive strategies, fallacies found in common communication can be categorized into three main categories:
1. Fallacy of Emotion (Pathos), which refers to the types of fallacies that persuade an argument by appealing to the audience's emotions rather than using logical reasoning.
2. Fallacy of Credibility (Ethos), which refers to the types of fallacies that attempt to establish specific credibility or character of the speaker/argument to gain the audience's trust or influence their decision.
3. Fallacy of Logic (Logos), which refers to the types of fallacies that persuade an argument by appealing to flawed or problematic logic, reason, relevance and evidence.
According to the definitions above, given the segment of text below,
[]
Determine whether or not any of the three fallacies defined (Fallacy of Emotion, Fallacy of Credibility and Fallacy of Logic) is present in the argument of the text?
Output your answer in JSON format {{"fallacy": name_of_the_fallacy, "explanation": in_a_sentence_or_two}}. If none of the fallacies is found, output {{"fallacy": "No Fallacy", "explanation": in_a_sentence_or_two}}. Only output JSON.''',
1:'''Based on your previous decision, given the following sub-types of {l1_fallacy}, namely, {fallacies},
provide a definitionto each of these sub-fallacies in the category of {l1_fallacy}.''',
2:'''Based on the definitions you provided, determine which of the defined sub-fallacies is present in the argument of the text? Output your answer in JSON format {{"fallacy": name_of_the_sub-fallacy}}. Only output JSON.'''
}

mafalda_multiround_prompts = {
    'v1_wo_def' : v1_wo_def,
    'v12_wo_def': v12_wo_def,
    'v13_wo_def': v13_wo_def,
    'v14_wo_def': v14_wo_def,
    'v2_gen_def': v2_gen_def,
    'v21_gen_def': v21_gen_def,
    'v3_cot_w_def': v3_cot_w_def,
    'v3_cot_wo_def': v3_cot_wo_def,
    'v4_wo_def': v4_wo_def
}

def prompt_mafalda(args, js):
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
            USER_PROMPTS = mafalda_multiround_prompts[args.scheme]
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
                pt = BASELINE_PROMPTS[args.scheme]
                if args.scheme == 'w_def':
                    content = pt.format(segment=text, fallacies=fal_def_str)
                elif args.scheme == 'w_logic_def':
                    content = pt.format(segment=text, fallacies=logical_fal_def_str)
                else:
                    content = pt.format(segment=text, fallacies=fal_name_str)
                usr_pt = {"role": "user", "content": content}
            else:
                fallacies = fal_def_str if args.scheme == 'w_def' else fal_name_str
                content = SINGLE_PROMPTS[args.scheme].format(segment=text, fallacies=fallacies)
                usr_pt = {"role": "user", "content": content}
            dialog = [sys_pt, usr_pt]
        js['chat_history'] = dialog
        js['arg_path'] = args.task_arg_path
    return js
    