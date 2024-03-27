
from .fallacy_utils import *
from .format_utils import format_tokens

argo_fal = list(FALLACY_DEFINITIONS.keys())
def_str = "\n".join([f"{str(i+1)}. {fal} is {FALLACY_DEFINITIONS[fal]}" for i, fal in enumerate(argo_fal)])    
f_names = ", ".join([f"{str(i+1)}. {fal}" for i, fal in enumerate(argo_fal)])

SYSTEM_PROMPT = [
'''You are a knowledgable expert in analysing fallacies in arguments. 
Please ensure that your responses are socially unbiased in nature.
Your response should not be lengthy.
Answer the last question.''',
'''You are a knowledgable expert in detecting fallacies in arguments. 
Please ensure that your responses are socially unbiased in nature.
Your response should not be lengthy.
Answer the last question.''',
]

# do you think
BASELINE_TEMPLATES = [
'''Based on the following definitions of fallacies,
{definitions}
Given the conversation below,
{dialog}
Determine whether any of the fallacies defined above is present in B's argument replied to A?
Output your answer in JSON format {{"fallacy": name_of_the_fallacy, "explanation": in_a_sentence_or_two}}. If none of the fallacies is found, output {{"fallacy": "No Fallacy", "explanation": in_a_sentence_or_two}}. Only output JSON.
''',# I'm now using this one, with definitions
'''Given five types of fallacies, namely, {fallacies}, and the following conversation,
{dialog}
Determine whether any of the fallacies is present in B's argument replied to A?
Output your answer in JSON format {{"fallacy": name_of_the_fallacy, "explanation": in_a_sentence_or_two}}. If none of the fallacies is found, output {{"fallacy": "No Fallacy", "explanation": in_a_sentence_or_two}}. Only output JSON.
''', # no definition
# '''Based on the following definitions of fallacies,
# {definitions}
# Given the conversation below,
# {dialog}
# Determine whether any of the fallacies defined above occurs in the reply by B?
# Output your answer in JSON format {{"fallacy": name_of_the_fallacy, "explanation": in_a_sentence_or_two}}. If none of the fallacies is found, output {{"fallacy": "No Fallacy", "explanation": in_a_sentence_or_two}}. Only output JSON.
# ''',
# '''Given the conversation below,
# {dialog}
# According to the following definitions, do you think any of the fallacies defined occurs in the reply by B?
# Output your answer in JSON format {{"fallacy": name_of_the_fallacy, "explanation": in_a_sentence_or_two}}. If none of the fallacies is found, output {{"fallacy": "No Fallacy", "explanation": in_a_sentence_or_two}}. Only output JSON.
# Fallacies:
# {definitions}
# ''',
# '''Given the following definitions of fallacies and a conversation, determine whether any of the fallacies defined occurs in the reply by B? Output your answer in JSON format {{"fallacy": name_of_the_fallacy, "explanation": in_a_sentence_or_two}}. If none of the fallacies is found, output {{"fallacy": "No Fallacy", "explanation": in_a_sentence_or_two}}. Only output JSON.
# Fallacies:
# {definitions}
# Conversation:
# {dialog}
# ''',
# ('''Given the question and answer below, which of the following fallacies occurs in the answer: {fallacies}, or {last_fallacy}?\nQuestion: {question}\nAnswer: {answer}''', '{fallacy}'),
# ('''Given the question, answer and context below, which of the following fallacies occurs in the answer: {fallacies}, or {last_fallacy}?\nQuestion: {question}\nAnswer: {answer}\nContext: {context}''', '{fallacy}'),
# ('''Given the following question and answer and definitions, determine which of the fallacies defined below occurs in the answer.\nDefinitions:\n{definitions}\n\nQuestion: {question}\nAnswer: {answer}''',
#          '{fallacy}'),
# ('Which fallacy does the following answer to the question have: "{question}" "{answer}"?\n{fallacies}', '{fallacy}')
]

SINGLE_PROMPT_TEMPLATE = [
'''Based on the following definitions of fallacies,
{definitions}
Given the conversation below,
{dialog}
Think step by step to determine whether any of the fallacies defined above is present in B's argument replied to A? Output your final answer in JSON format {{"fallacy": name_of_the_fallacy}}. If none of the fallacies is found, output {{"fallacy": "No Fallacy"}}.
Now, let's think step by step.
'''
]

MULTI_PROMPTS = {}
# Version 1: Include our own definitions
ASK_SINGLE_FALLACY = """According to the definition of {name}: "{defi}", given the following conversation, \n[]\nIs the fallacy defined above absent or present in B's argument?"""
OUTPUT_FORMAT = ''' Output your answer in JSON format {"answer": 'present' or 'absent', "explanation": in_a_sentence_or_two}. Only output JSON.'''
REVIEW_ALL_FALLACIES = '''Review the definitions of these fallacies,
{definitions}
Review the conversation,
{dialog}
Based on all your previous analysis, determine whether any of the fallacies defined above is dominantly present in B's argument replied to A?
Output your answer in JSON format {{"fallacy": name_of_the_fallacy, "explanation": in_a_sentence_or_two}}. If none of the fallacies is found, output {{"fallacy": "No Fallacy", "explanation": in_a_sentence_or_two}}. Only output JSON.
'''
USER_PROMPTS1 = {
#0:"""Given the conversation below,\n{dialog}\nIs B in favor of or against the topic proposed by A?""",
#5:"""Based on all your previous analysis, review the definitions of all the fallacies, summarize the most dominant fallacy/fallacies present in B's argument in JSON format {{"fallacy": [name(s)_of_the_fallacy]}}. If none of the fallacies is present, output {{"fallacy": "No Fallacy"}}. Only output JSON."""
#0:'''Give a definition to fallacy.''',
5: REVIEW_ALL_FALLACIES
}#1:"""Considering B's stance, is it logically sound for B to draw an argument like that?""",
def_questions = {}
for i, (name, def_) in enumerate(FALLACY_DEFINITIONS.items()):
    def_questions[i] = ASK_SINGLE_FALLACY.format(name=name, defi=def_)
USER_PROMPTS1.update(def_questions)
MULTI_PROMPTS['v1'] = USER_PROMPTS1


# Version 2: Ask Llama to give definitions and finally give the answer
ASK_SINGLE_FALLACY = """Give a definition to the fallacy of {name}."""
OUTPUT_FORMAT = ''' Output your answer in JSON format {"fallacy": name_of_the_fallacy, "definition": in_a_sentence_or_two}. Only output JSON.'''
REVIEW_AND_ANSWER = '''Based on your previous definitions of these fallacies,
given the following conversation,
{dialog}
Determine whether any of the fallacies defined above is present in B's argument replied to A?
Output your answer in JSON format {{"fallacy": name_of_the_fallacy, "explanation": in_a_sentence_or_two}}. If none of the fallacies is found, output {{"fallacy": "No Fallacy", "explanation": in_a_sentence_or_two}}. Only output JSON.
'''
USER_PROMPTS2 = {
#0:"""Given the conversation below,\n{dialog}\nIs B in favor of or against the topic proposed by A?""",
#5:"""Based on all your previous analysis, review the definitions of all the fallacies, summarize the most dominant fallacy/fallacies present in B's argument in JSON format {{"fallacy": [name(s)_of_the_fallacy]}}. If none of the fallacies is present, output {{"fallacy": "No Fallacy"}}. Only output JSON."""
#0:'''Give a definition to fallacy.''',
5: REVIEW_AND_ANSWER
}#1:"""Considering B's stance, is it logically sound for B to draw an argument like that?""",
def_questions = {}
for i, (name, def_) in enumerate(FALLACY_DEFINITIONS.items()):
    def_questions[i] = ASK_SINGLE_FALLACY.format(name=name) + OUTPUT_FORMAT
USER_PROMPTS2.update(def_questions)
MULTI_PROMPTS['v2'] = USER_PROMPTS2


# # Version 3: Ask Llama to give definitions
# ASK_SINGLE_FALLACY = """Give a definition to the fallacy of {name}."""
# OUTPUT_FORMAT = ''' Output your answer in JSON format {"fallacy": name_of_the_fallacy, "definition": in_a_sentence_or_two}. Only output JSON.'''
# REVIEW_AND_ANSWER = '''Based on your previous definitions of these fallacies,
# given the following conversation,
# {dialog}
# Determine whether any of the fallacies defined above is present in B's argument replied to A?
# Output your answer in JSON format {{"fallacy": name_of_the_fallacy, "explanation": in_a_sentence_or_two}}. If none of the fallacies is found, output {{"fallacy": "No Fallacy", "explanation": in_a_sentence_or_two}}. Only output JSON.
# '''
# USER_PROMPTS = {
# #0:"""Given the conversation below,\n{dialog}\nIs B in favor of or against the topic proposed by A?""",
# #5:"""Based on all your previous analysis, review the definitions of all the fallacies, summarize the most dominant fallacy/fallacies present in B's argument in JSON format {{"fallacy": [name(s)_of_the_fallacy]}}. If none of the fallacies is present, output {{"fallacy": "No Fallacy"}}. Only output JSON."""
# #0:'''Give a definition to fallacy.''',
# 5: REVIEW_AND_ANSWER
# }#1:"""Considering B's stance, is it logically sound for B to draw an argument like that?""",
# def_questions = {}
# for i, (name, def_) in enumerate(FALLACY_DEFINITIONS.items()):
#     def_questions[i] = ASK_SINGLE_FALLACY.format(name=name) + OUTPUT_FORMAT
# USER_PROMPTS.update(def_questions)


# Version 4: GPT-3.5 Multi-prompt
USER_PROMPTS4 = {
0:"""In the following conversation,\n{dialog}\nIs B in favor of or against the topic proposed by A?""",
1:"""Considering B's stance, is B's argument logically reasonable or potentially fallacious? Explain why.""",
# a
#2:"""Based on all your previous analysis, if no fallacy is found in B's argument, output {{"fallacy": "No Fallacy"}}. Otherwise, reflect on your previous answers, within five types of fallacies, namely, {fallacies}, determine which of these fallacies is most likely present in B's argument. Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. Only output JSON.""".format(fallacies=f_names)
# b
#2: """Reflecting on your previous answers, given five types of fallacies, namely, {fallacies}, determine whether any of these fallacies is present in B's argument replied to A? Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. If none of these fallacies is found, output {{"fallacy": "No Fallacy"}}. Only output JSON.""".format(fallacies=f_names)
# c
#2: """Reflecting on your previous answers, given five types of fallacies, namely, {fallacies}, determine whether any of these fallacies is present in B's argument replied to A? Output your answer in JSON format {{"fallacy": name_of_one_of_the_five_fallacies}}. If none of these fallacies is found, output {{"fallacy": "No Fallacy"}}. Only output JSON.""".format(fallacies=f_names),
# d
2: """Reflecting on your previous answers, given five types of fallacies, namely, {fallacies}, determine whether any of these fallacies is present in B's argument replied to A? Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. DO NOT output fallacy that's not in the five types. If none of these fallacies is found, output {{"fallacy": "No Fallacy"}}. Only output JSON.""".format(fallacies=f_names),
# e
#2: """Reflecting on your previous answers, determine whether any of the following five fallacies, namely, {fallacies}, is present in B's argument replied to A? Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. If none of these fallacies is found, output {{"fallacy": "No Fallacy"}}. DO NOT output fallacy that is not mentioned in the five. Only output JSON.""".format(fallacies=f_names),
# f
#2: """Reflecting on your previous answers, given five types of fallacies, namely, {fallacies}, determine whether any of these fallacies is present in B's argument replied to A? Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. If none of these fallacies is found, output {{"fallacy": "No Fallacy"}}. DO NOT output fallacy that is not mentioned in the five types. Only output JSON.""".format(fallacies=f_names)
}
MULTI_PROMPTS['v4'] = USER_PROMPTS4

# Version 5: GPT-3.5 Multi-prompt
USER_PROMPTS5 = {
0:"""Given the following conversation,\n{dialog}\nSummarize what is B's point of view on the topic proposed by A?""",
1:"""Is B in favor of or against the topic proposed by A?""",
2:"""Considering B's stance, is B's argument logically reasonable or potentially fallacious? Explain why.""",
3: """Reflecting on all your previous answers, consider five types of fallacies, namely, {fallacies}, determine whether any of these fallacies is present in B's argument replied to A? Output your answer in JSON format {{"fallacy": name_of_the_fallacy}}. DO NOT output fallacy that's not in the given five types. If none of these fallacies is found, output {{"fallacy": "No Fallacy"}}. Only output JSON.""".format(fallacies=f_names),
}
MULTI_PROMPTS['v5'] = USER_PROMPTS5

def extract_stance(text):
    if "not in favor of" in text:
        stance = "against"
    else:
        if "in favor of" in text:
            stance = "in favor of"
        else:
            stance = "against"
    return stance

p_version = 'v4'
USER_PROMPTS = MULTI_PROMPTS[p_version]

def prompt_argo(args, js, tokenizer=None):
    """
    input:
        js is one data sample in the format of dictionary
        js['text'] is a string of QA in a format like : "A: ....\nB: ..."
    """
    
    dialog = []
    sys_pt = {"role": "system", "content": SYSTEM_PROMPT[0]}
    if args.run_multiprompt:
        
        tp = USER_PROMPTS[args.current_round]
        if args.current_round < 5:
            if "[]" in tp:
                tp = tp.replace("[]", "{dialog}")
            usr_ct = tp.format(dialog=js['text']) if '{dialog}' in tp else tp
            # if OUTPUT_FORMAT not in usr_ct:
            #     usr_ct = usr_ct + OUTPUT_FORMAT
        else:
            if '{definitions}' in tp:
                usr_ct = tp.format(definitions=def_str, dialog=js['text'])
            else:
                usr_ct = tp.format(dialog=js['text'])
        usr_prompt = {"role": "user", "content": usr_ct}
        if args.current_round == 0: # initial round
            dialog = [sys_pt, usr_prompt]
            #js.pop('text')
        else:
            last_prediction = js.pop('prediction')
            dialog = js['chat_history'] + [{"role":"assistant", "content":last_prediction}] + [usr_prompt]
        js['chat_history'] = dialog  
    else:
        text = js['text']
        if args.run_baseline:
            #usr_pt = {"role": "user", "content": BASELINE_TEMPLATES[0].format(dialog=text, definitions=def_str)}
            usr_pt = {"role": "user", "content": BASELINE_TEMPLATES[1].format(fallacies=f_names, dialog=text)}
        else:
            content = SINGLE_PROMPT_TEMPLATE[0].format(dialog=text, definitions=def_str)
            usr_pt = {"role": "user", "content": content}
        dialog = [sys_pt, usr_pt]
        js['chat_history'] = dialog
        #js.pop('text')
    if tokenizer is not None:
        return format_tokens(args, [dialog], tokenizer)[0], js
    else:
        return js



    

SINGLE_PROMPT_TEMPLATE_=[
'''
Given the conversation below,
{dialog}
Answer the following questions step-by-step:
1. According to the following definition: "Appeal to Emotion is a fallacy that attempts to arouse sentiments within the intended audience to persuade.", do you think this fallacy is present in B's argument?
2. According to the following definition: "Red Herring is a fallacy that introduces irrelevant information or arguments to diverge attention from the main topic being discussed to irrelevant issues.", do you think this fallacy is present in B's argument?
3. According to the following definition: "Hasty Generalization is a fallacy that occurs when a conclusion is drawn from unrepresentative observations or evidence from a small sample that cannot represent the whole population.", do you think this fallacy is present in B's argument?
4. According to the following definition: "Ad Hominem is a fallacy that involves attacking a person’s character or motives instead of addressing the substance of their argument.", do you think this fallacy is present in B's argument?
5. According to the following definition: "Appeal to False Authority is a fallacy that occurs when an argument relies on the opinion or endorsement of an authority figure who lacks substantial credibility in the discussed matter because his expertise is questionable or irrelevant, or because he is attributed a statement which has been tweaked. When applicable, a scientific consensus is not an appeal to authority.", do you think this fallacy is present in B's argument?
6. According to your previous answers, output the fallacy/fallacies B has committed in JSON format {{"fallacy": [name(s)_of_the_fallacy]}}. If none of the fallacies is present, output {{"fallacy": "No Fallacy"}}. Only output JSON.
''',#44.88
'''
Given the conversation below,
{dialog}
Answer the following questions one by one:
1. Is B in favor of or against the topic proposed by A?
2. According to the definition of Emotional Language: "A fallacy when someone tries to to persuade his stance using emotive language to arouse particular sentiments within the intended audience.", considering B's stance, is this fallacy present in B's argument?
3. According to the definition of Red Herring: "A fallacy when someone introduces irrelevant or confusing information or arguments to diverge attention from the main topic being discussed to irrelevant issues.", considering B's stance, is this fallacy present in B's argument?
4. According to the definition of Hasty Generalization: "A fallacy when someone draws hasty conclusions from unrepresentative observations from a small sample that cannot represent the whole population or from a small aspect of all factors.", considering B's stance, is this fallacy present in B's argument?
5. According to the definition of Ad Hominem: "A fallacy when someone attacks the others' character or motives instead of addressing the substance of their arguments.", considering B's stance, is this fallacy present in B's argument?
6. According to the definition of Appeal to False Authority: "A fallacy when someone's argument relies on the opinion or endorsement of an authority figure who lacks acknowledged credibility in the discussed matter because his/her expertise is questionable/irrelevant or he/she is attributed a statement which has been tweaked. When applicable, a scientific consensus is not an appeal to authority.", considering B's stance, is this fallacy present in B's argument?
7. According to your previous answers, output the fallacy/fallacies B has committed in JSON format {{"fallacy": [name(s)_of_the_fallacy]}}. If none of the fallacies is present, output {{"fallacy": "No Fallacy"}}. Only output JSON.
Answer the above questions from 1 to 7.
''',# replace Appeal to Emotion with Emotional Language
"""
Based on the following definitions of fallacies,
{definitions}
Given the conversation below,
{dialog}
First, contrast the definitions of fallacies with B's reply one by one and evaluate whether that fallacy is present or absent in B's reply.
Second, output the present fallacy in JSON format {{"fallacy": name_of_the_fallacy}}. If you think there're multiple fallacies, only output the one that is most dominantly present. If none of the fallacies is present, output {{"fallacy": "No Fallacy"}}. Only output JSON.
""",
"""
Based on the following definitions of fallacies,
{definitions}
Given the conversation below,
{dialog}
Answer the following questions step-by-step:
1. What is the stance of B? Is B in favor of or against the argument proposed by A?
2. Considering B's stance, do you think it's logically sound for B to draw a conclusion like that?
3. Contrast the definitions of fallacies with B's reply one by one and evaluate whether that fallacy is present or absent in B's reply.
4. Output the fallacy that is most dominantly present in JSON format {{"fallacy": name_of_the_fallacy}}. If none of the fallacies is present, output {{"fallacy": "No Fallacy"}}. Only output JSON.
""",
]

MULTI_PROMPTS_TEMPLATES_ = {
0:[
'''Given the conversation below,
{input}
Is B in favor of or against the argument proposed by A?
'''],
1:[
'''
Although B is {input} A's argument, do you think it's logically sound for B to draw a conclusion like that?
''',
'''
Taking B's stance into account, do you think it's logically sound for B to draw a conclusion like that?{input}
''',
'''
Taking B's stance into account, do you think it's logically sound for B to draw a conclusion like that? If not, why?{input}
''',
'''
Considering your opinion about B's stance, do you think it's logically sound for B to draw a conclusion like that? If not, why?{input}
''',
'''
Given that you think B {input}, do you think it's logically sound for B to draw a conclusion like that? If not, why?
'''
],
2:[
'''
Below are the definitions of five commonly seen fallacies in daily argumentation:
{input}
Based on what you mentioned for the previous questions, contrast the definitions of fallacies with B's reply one by one and evaluate whether that fallacy is present or absent in B's reply.
'''
],
3:[
'''
If you think B has committed multiple fallacies in last question, choose the one that is most dominantly present. Output your final answer in JSON format {{"fallacy": name_of_the_fallacy}}. If none of the fallacies is present, output {{"fallacy": "No Fallacy"}}. Only output JSON.
'''
]
}