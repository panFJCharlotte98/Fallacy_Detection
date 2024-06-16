
def argotario_convert_to_name(t):
    if 'authority' in t:
        #return 'appeal to opinion of false authority'
        return 'appeal to false authority'
    elif ('emotion' in t) or ('emotional' in t) or ('emoti' in t):
        return 'appeal to emotion'
    elif ('hasty' in t) or ('generalization' in t):
        return 'hasty generalization'
    elif 'ad hominem' in t:
        return 'ad hominem'
    elif ('red' in t) or ('herring' in t):
        return 'red herring'
    elif 'no fallacy' in t:
        return 'no fallacy'
    else:
        return 'failed'

def elecdebate_convert_to_name(t):
    if 'ad hominem' in t:
        return 'ad hominem'
    elif ('emotion' in t) or ('emotional' in t) or ('emoti' in t):
        return 'appeal to emotion'
    elif 'authority' in t:
        return 'appeal to false authority'
    # elif ('slogan' in t):
    #     return 'slogans'
    elif ('slippery' in t) or ('slope' in t):
        return 'slippery slope'
    elif ('causal' in t) or ('post hoc' in t):
        return 'false causality (post hoc fallacy)'
    else:
        return 'failed'
    
def logic_convert_to_name(t):
    if 'generali' in t:
        return 'faulty generalization'
    elif ('emotion' in t) or ('emotional' in t) or ('emoti' in t):
        return 'appeal to emotion'
    elif 'ad hominem' in t:
        return 'ad hominem'
    elif ('herring' in t):
        return 'red herring'
    elif ('causal' in t):
        return 'false causality (post hoc fallacy)'
    elif ('circular' in t) or ('reasoning' in t):
        return "circular reasoning"
    elif ("ad populum" in t) or ('popularity' in t) or ('populum' in t):
        return "ad populum"
    elif ('converse' in t) or ('consequent' in t) or ('affirming' in t):
        return 'fallacy of converse (affirming the consequent)'
    elif ('dilemma' in t):
        return "false dilemma"
    elif ('equivocation' in t):
        return 'equivocation'
    elif ("straw" in t):
        return "straw man"
    elif ('credibility' in t) or ('doubt' in t):
        return 'doubt credibility'
    elif ('intent' in t):
        return 'intentional (intentionally wrong argument)'
    else:
        return 'failed'

def propaganda_convert_to_name(t):
    # if 'loaded language' in t:
    #     return 'loaded language'
    if ('calling' in t):
        return 'name-calling'
    # elif ('exaggeration' in t) or ('minimisation' in t):
    #     return 'exaggeration or minimisation'
    elif ('credibility' in t) or ('doubt' in t):
        return 'doubt credibility'
    elif ('fear' in t) and ('appeal' in t):
        return 'appeal to fear'
    elif ('flag' in t) or ('waving' in t) or ('flag-waving' in t):
        return 'flag-waving'
    elif ('causal' in t) or ('oversimplification' in t):
        return 'causal oversimplification'
    elif 'authority' in t:
        #return 'appeal to opinion of false authority'
        return 'appeal to false authority'
    elif ('dilemma' in t):
        return "false dilemma"
    # elif ('cliche' in t) or ('terminat' in t):
    #     return 'thought-terminating cliches'
    elif ('whatabout' in t):
        return 'whataboutism'
    elif ('ad hitlerum' in t) or ('hitler' in t):
        return 'reductio ad hitlerum'
    elif ('herring' in t):
        return 'red herring'
    elif ("straw" in t):
        return "straw man"
    # elif ('slogan' in t):
    #     return 'slogans'
    # elif ('repetition' in t) or ('repeat' in t):
    #     return 'repetition'
    elif ("ad populum" in t) or ('popularity' in t) or ('populum' in t):
        return "ad populum"
    elif ('equivocation' in t):
        return 'equivocation'
    else:
        return 'failed'

def covid_convert_to_name(t):
    if 'authority' in t:
        #return 'appeal to opinion of false authority'
        return 'appeal to false authority'
    elif ('causal' in t) or ('post hoc' in t):
        return 'false causality (post hoc fallacy)'
    elif ('hasty' in t) or ('general' in t):
        return 'hasty generalization'
    elif ('red ' in t) or ('herring' in t):
        return 'red herring'
    elif ('analogy' in t):
        return 'false analogy'
    elif 'equivocation' in t:
        return 'equivocation'
    elif ('straw man' in t) or ('strawman' in t) or ('straw' in t):
        return 'straw man'
    elif ('burden ' in t) or ('proof' in t):
        return 'evading the burden of proof'
    elif ('cherry' in t) or ('picking' in t):
        return 'cherry picking'
    elif 'no fallacy' in t:
        return 'no fallacy'
    else:
        return 'failed'

def reddit_convert_to_name(t):
    if 'authority' in t:
        #return 'appeal to opinion of false authority'
        return 'appeal to authority'
    elif 'dilemma' in t:
        return 'false dilemma'
    elif ('hasty' in t) or ('general' in t):
        return 'hasty generalization'
    elif (' nature' in t):
        return 'appeal to nature'
    elif ('populum' in t) or ('popularity' in t) or ('populum' in t):
        return 'ad populum'
    elif 'slippery slope' in t:
        return 'slippery slope'
    elif ('tradition' in t):
        return 'appeal to tradition'
    elif 'worse problems' in t:
        return 'appeal to worse problems'
    else:
        return 'failed'
    

def mafalda_l1_convert_to_name(t):
    if 'credibility' in t:
        return 
    return 


def mafalda_convert_to_name(t):
    if 'hominem' in t:
        return 'ad hominem'
    elif 'quoque' in t:
        return 'tu quoque'
    elif ('guilt' in t) and ('association' in t):
        return 'guilt by association'
    elif ('populum' in t) or ('popularity' in t) or ('populum' in t):
        return 'ad populum'
    elif ('appeal to nature' in t):
        return 'appeal to nature'
    elif ('appeal to tradition' in t):
        return 'appeal to tradition'
    elif ('authority' in t):
        return 'appeal to false authority'
    elif ('causal oversimplification' in t):
        return 'causal oversimplification'
    elif ('hasty generalization' in t):
        return 'hasty generalization'
    elif ('false causality' in t):
        return 'false causality'
    elif ('false analogy' in t):
        return 'false analogy'
    elif 'false dilemma' in t:
        return 'false dilemma'
    elif 'slippery slope' in t:
        return 'slippery slope'
    elif ('fallacy' in t) and ('division' in t):
        return 'fallacy of division'
    elif ('straw man' in t) or ('strawman' in t) or ('straw' in t):
        return 'straw man'
    elif 'circular reasoning' in t:
        return 'circular reasoning'
    elif 'equivocation' in t:
        return 'equivocation'
    elif 'positive emotion' in t:
        return 'appeal to positive emotion'
    elif 'anger' in t:
        return 'appeal to anger'
    elif 'fear' in t:
        return 'appeal to fear'
    elif 'pity' in t:
        return 'appeal to pity'
    elif 'ridicule' in t:
        return 'appeal to ridicule'
    elif 'worse problems' in t:
        return 'appeal to worse problems'
    elif 'no fallacy' in t:
        return 'no fallacy'
    elif 'appeal to emotion' in t:
        return 'appeal to emotion'
    else:
        return 'failed'
    
def confusion_convert_to_name(t):
    if 'authority' in t:
        #return 'appeal to opinion of false authority'
        return 'appeal to false authority'
    elif ('emotion' in t) or ('emotional' in t) or ('emoti' in t):
        return 'appeal to emotion'
    elif ('hasty' in t) or ('general' in t):
        return 'hasty generalization'
    elif 'ad hominem' in t:
        return 'ad hominem'
    elif ('red' in t) or ('herring' in t):
        return 'red herring'
    elif ('no fallacy' in t) or ('none' in t):
        return 'no fallacy'
    elif ('populum' in t) or ('popularity' in t) or ('populum' in t):
        return 'ad populum'
    elif ('dilemma' in t):
        return "false dilemma"
    elif (('false' in t) and ('causal' in t)) or ('post hoc' in t):
        return 'false causality (post hoc fallacy)'
    elif ('equivocation' in t) or ('vagueness' in t):
        return 'equivocation'
    elif ('straw man' in t) or ('strawman' in t) or ('straw' in t):
        return 'straw man'
    elif ('slippery' in t) or ('slope' in t):
        return 'slippery slope'
    elif ('circular reasoning' in t):
        return "circular reasoning"
    elif ('appeal' in t) and (' nature' in t):
        return 'appeal to nature'
    elif 'worse problem' in t:
        return 'appeal to worse problems'
    elif ('credibility' in t) or ('doubt' in t):
        return 'doubt credibility'
    elif ('fear' in t) and ('appeal' in t):
        return 'appeal to fear'
    elif ('causal oversimplification' in t):
        return 'causal oversimplification'
    elif ('analogy' in t):
        return 'false analogy'
    else:
        return t