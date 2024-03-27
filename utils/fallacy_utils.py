# version 1, consistent with gpt3.5
FALLACY_DEFINITIONS = {
"Appeal to Emotion": "a fallacy when someone attempts to argue or persuade by using emotive language to arouse non-rational sentiments within the intended audience.",
"Red Herring": "a fallacy when someone introduces irrelevant or confusing information in arguments to diverge attention from the main topic being discussed to irrelevant issues.",
"Hasty Generalization": "a fallacy when someone makes generalizations based on incomplete observations that cannot represent or generalize to other situations if other relevant factors are taken into account.",
"Ad Hominem": "a fallacy when someone attacks the others' characters or motives instead of addressing the substance of their arguments.",
"Appeal to False Authority": "a fallacy when someone attempts to argue or persuade by referring to the opinions or statements of a questionable authority who lacks sufficient credibility in the discussed matter because the authority's expertise may be inadequate/irrelevant or the authority is attributed a statement that has been tweaked.",
}

# change order, change definitions
# # version 2
# FALLACY_DEFINITIONS = {
# "Ad Hominem": "a fallacy when someone attacks the others' characters or motives instead of addressing the substance of their arguments.",
# "Appeal to Opinion of False Authority": "a fallacy when someone attempts to argue or persuade by referring to the opinions or statements of another questionable authority who lacks sufficient credibility in the discussed matter because the authority's expertise may be inadequate/irrelevant or the authority is attributed a statement that has been tweaked.",
# "Red Herring": "a fallacy when someone introduces irrelevant or confusing information in arguments thus diverge the audience's attention from the main topic being discussed to irrelevant issues.",
# "Appeal to Emotion": "a fallacy when someone attempts to argue or persuade by using emotionally charged language to arouse non-rational sentiments within the intended audience.",
# "Hasty Generalization": "a fallacy when someone makes generalizations based on partial/incomplete observations on a small sample of the whole populations that cannot represent or generalize to other situations legitimately.",
# }

FALLACY_EXAMPLES = {
"Appeal to Emotion": [
'''
For example,
Conversation:
A: Is the legality of abortion desirable?
B: If you abort your child, you miss a smiling face, whenever you could cuddle him/her. DONT legalize it for more little happy faces.
Answer:
{"answer": 'present', "explanation": "B's argument opposed against abortion legality commits the fallacy of Appeal to Emotion by using emotional language such as 'miss a smiling face', 'cuddle' and 'little happy faces' to arouse audience's sympathy and affection towards fetuses."}
Conversation:
A: Is the legality of abortion desirable?
B: No, it has been illegal since the Supreme Court overturned Roe v. Wade.
Answer:
{"answer": 'absent', "explanation": "B's argument does not contain any emotional language but just state a fact."}
'''
],
"Red Herring": [
'''
For example,
Conversation:
A: Should Christians accept same sex marriage?
B: Yes, they should, because the courts have made a decision, and need to move on to other things.
Answer:
{"answer": 'present', "explanation": "B's argument in favor of A's proposal commits the fallacy of Red Herring by bringing up 'the courts have made a decision' which is irrelevant to the discussed matter focusing on Christians and diverts people's attention to elsewhere."}
Conversation:
A: Should Christians accept same sex marriage?
B: Yes, they should, becuse homosexuality is also part of human nature.
Answer:
{"answer": 'absent', "explanation": "B's argument does not contain irrelevant information that may divert people's attention but directly address the discuessed matter."}
'''
],
"Hasty Generalization": [
'''
For example,
Conversation:
A: Should musicians be payed for streaming music? (by Apple)
B: All streaming music should be for free. A friend of mine is a musician and he earns a lot from playing concerts.
Answer:
{"answer": 'present', "explanation": "B's argument commits the fallacy of Hasty Generalization because his conclusion against the idea of paying musicians for streaming music is based on a partial observation of his friend which is just a small sample that cannot represent and generalize to all the musicians."}
Conversation:
A: Should musicians be paied for streaming music? (by Apple)
B: Sure, everyone deserves to be paied for his or her intellectual work.
Answer:
{"answer": 'absent', "explanation": "B's argument does not contain "}
'''
],
"Ad Hominem" : [

],
"Appeal to False Authority": [

]
}