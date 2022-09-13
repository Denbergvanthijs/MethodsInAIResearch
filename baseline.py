from typing import Counter

with open("./data/dialog_acts.dat") as file:
    dialog_acts = file.read().splitlines()
    dialog_acts_split = [dialog_act.split(" ", 1) for dialog_act in dialog_acts]

# print("Dialog Acts:", dialog_acts_split)

counter = Counter([da[0] for da in dialog_acts_split])

print("Counter:", counter)
print("Most common:", counter.most_common(1))
