import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("./data/dialog_acts.dat", header=None, sep="\s\s+", engine="python")

# Split intent from utterances, rename column
df["intent"] = df[0].str.split(" ", 1).str[0]
df[0] = df[0].str.split(n=1).str[1]
df.rename(columns={df.columns[0]: "utterance"}, inplace=True)

x = df["utterance"].values
y = df["intent"].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=1000)

vec = CountVectorizer()
vec.fit(x_train)

X_train_vec = vec.transform(x_train)
X_test_vec = vec.transform(x_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

score = model.score(X_test_vec, y_test)
print("Accuracy: ", score)
