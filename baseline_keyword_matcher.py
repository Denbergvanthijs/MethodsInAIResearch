import pandas as pd

df = pd.read_csv('./data/dialog_acts.dat', header=None, sep='\s\s+', engine='python')

# Split intent from utterances, rename column
df['intent'] = df[0].str.split(" ", 1).str[0]
df[0] = df[0].str.split(n=1).str[1]
df.rename(columns={df.columns[0]: "utterance"}, inplace=True)

names = ['ack', 'affirm', 'bye', 'confirm', 'deny', 'hello', 'inform',
         'negate', 'null', 'repeat', 'reqalts', 'reqmore', 'request',
         'restart', 'thankyou']
keys = [['kay', 'okay', 'okay ah', 'okay um', 'okay uh', 'im good', 'fine', 'okay can you', 'thatll do'],
        ['yes', 'yea', 'right', 'yeah', 'correct'],
        ['goodbye', 'good bye', 'bye'],
        ['is it', 'is there', 'does it', 'okay and', 'is that', 'does that'],
        ['i dont', 'wrong', 'no i dont', 'not any more', 'no not', 'can you change', 'dont want', 'not'],
        ['hi', 'hello', 'hey', 'howdy', 'hello welcome'],
        ['i dont care', 'any', 'dont care', 'south', 'east', 'west', 'north', 'moderate', 'food', 'im looking for', 'priced', 'any',
            'expensive', 'restaurant', 'part of town', 'i need a', 'doesnt matter', 'price range', 'i need', 'looking for', 'cheap', 'is okay'],
        ['no im sorry', 'no', 'no id like', 'no dont care', 'no im looking for'],
        ['child', 'worthless system', 'noise', 'sil', 'unintelligible', 'cough'],
        ['repeat', 'repeat that', 'go back', 'back', 'again please', 'try this again', 'cant repeat'],
        ['is there anything else', 'anything else', 'how about', 'what about'],
        ['more'],
        ['phone number', 'address', 'whats the adress', 'what is the address', 'what is the phone number',
            'whats the phone number', 'post code', 'price range', 'type of', 'area'],
        ['start over', 'reset', 'restart', 'start again'],
        ['thank you good bye', 'thank you goodbye', 'thank you', 'good bye']]

intents = ['ack', 'affirm', 'bye', 'confirm', 'deny', 'hello', 'inform',
           'negate', 'null', 'repeat', 'reqalts', 'reqmore', 'request',
           'restart', 'thankyou']

named = dict(zip(names, keys))

# "Rules", most common keywords per intent. Note: some intents are
# more common that others.
count = 0
z = 0
utt = df["utterance"]
inte = df["intent"]

for w in utt:
    if any(i in w for i in named["ack"]):
        if inte[z] == "ack":
            count += 1
    if any(i in w for i in named["affirm"]):
        if inte[z] == "affirm":
            count += 1
    if any(i in w for i in named["bye"]):
        if inte[z] == "bye":
            count += 1
    if any(i in w for i in named["confirm"]):
        if inte[z] == "confirm":
            count += 1
    if any(i in w for i in named["deny"]):
        if inte[z] == "deny":
            count += 1
    if any(i in w for i in named["hello"]):
        if inte[z] == "hello":
            count += 1
    if any(i in w for i in named["inform"]):
        if inte[z] == "inform":
            count += 1
    if any(i in w for i in named["negate"]):
        if inte[z] == "negate":
            count += 1
    if any(i in w for i in named["null"]):
        if inte[z] == "null":
            count += 1
    if any(i in w for i in named["repeat"]):
        if inte[z] == "repeat":
            count += 1
    if any(i in w for i in named["reqalts"]):
        if inte[z] == "reqalts":
            count += 1
    if any(i in w for i in named["reqmore"]):
        if inte[z] == "reqmore":
            count += 1
    if any(i in w for i in named["request"]):
        if inte[z] == "request":
            count += 1
    if any(i in w for i in named["restart"]):
        if inte[z] == "restart":
            count += 1
    if any(i in w for i in named["thankyou"]):
        if inte[z] == "thankyou":
            count += 1
    z += 1
print("This many counted correctly:", count)
print("Accuracy = ", count/len(utt))


def userprompt():
    w = input("Type new sentence here: ")
    if any(i in w for i in named["ack"]):
        print("ack")
    if any(i in w for i in named["affirm"]):
        print("affirm")
    if any(i in w for i in named["bye"]):
        print("bye")
    if any(i in w for i in named["confirm"]):
        print("confirm")
    if any(i in w for i in named["deny"]):
        print("deny")
    if any(i in w for i in named["hello"]):
        print("hello")
    if any(i in w for i in named["inform"]):
        print("inform")
    if any(i in w for i in named["negate"]):
        print("negate")
    if any(i in w for i in named["null"]):
        print("null")
    if any(i in w for i in named["repeat"]):
        print("repeat")
    if any(i in w for i in named["reqalts"]):
        print("reqalts")
    if any(i in w for i in named["reqmore"]):
        print("reqmore")
    if any(i in w for i in named["request"]):
        print("request")
    if any(i in w for i in named["restart"]):
        print("restart")
    if any(i in w for i in named["thankyou"]):
        print("thankyou")


userprompt()
