from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import spacy

with open('dialog_acts.dat') as file:
    data = file.read().splitlines()  # Split the dataset into lines

# Split each line into two parts: the intent and the utterance
# The 1 as the second argument means that we only split the line once
dataset = [row.split(" ", 1) for row in data]
y_data, x_data = zip(*dataset)

# Convert texts to vectors

nlp = spacy.load('en_core_web_sm')
docs = [nlp(sentence) for sentence in x_data]
x_vectors = [doc.vector for doc in docs]

# Split training:testing data by 85:15 ratio

x_train, x_test, y_train, y_test = train_test_split(x_vectors, y_data, test_size=0.15)

# Train and test model

model = LinearSVC()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
score = accuracy_score(y_test, predictions)
print("ACCURACY SCORE:", score)