from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split

with open("./data/dialog_acts.dat") as file:
    data = file.read()

dataset = [i.lower() for i in data.splitlines()]    
train_set, test_set = train_test_split(dataset, train_size=0.85, test_size=0.15)

test_set[:2], train_set[:2], len(test_set), len(train_set)

dialog_act = [x.split()[0] for x in train_set]
most_frequent = max(set(dialog_act), key=dialog_act.count)
y_pred = [most_frequent] * len(dialog_act)

accuracy = accuracy_score(dialog_act, y_pred)

precision_macro = precision_score(dialog_act, y_pred, average='macro')
precision_micro = precision_score(dialog_act, y_pred, average='micro')

f_measure_macro = f1_score(dialog_act, y_pred, average='macro')
f_measure_micro = f1_score(dialog_act, y_pred, average='micro')

recall_macro = recall_score(dialog_act, y_pred, average='macro')
recall_micro = recall_score(dialog_act, y_pred, average='micro')

print('accuracy: '+str(accuracy))

print('macro precision: ' + str(precision_macro))
print('micro precision: ' + str(precision_micro))

print('macro f-measure: ' + str(f_measure_macro))
print('micro f-measure: ' + str(f_measure_micro))

print('macro recall: ' + str(recall_macro))
print('micro recall: ' + str(recall_micro))
