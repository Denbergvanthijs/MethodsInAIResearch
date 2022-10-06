import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from baseline_majority import preprocess_data

x_train, x_test, y_train, y_test, n_classes = preprocess_data("./data/dialog_acts.dat")

vec = CountVectorizer()
vec.fit(x_train)

X_train_vec = vec.transform(x_train)
X_test_vec = vec.transform(x_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

score_train = model.score(X_train_vec, y_train)
print("Accuracy train: ", score_train)

score_test = model.score(X_test_vec, y_test)
print("Accuracy test: ", score_test)
