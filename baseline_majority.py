from typing import Counter

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def preprocess_data(filepath_dataset: str, test_size: float = 0.15, random_state: int = 42) -> tuple:
    """Preprocess the data."""
    with open(filepath_dataset) as file:

        data = file.read().splitlines()  # Split the dataset into lines

    # Split each line into two parts: the intent and the utterance
    # The 1 as the second argument means that we only split the line once
    dataset = [row.split(" ", 1) for row in data]
    y_data, x_data = zip(*dataset)
    y_data, _ = pd.factorize(y_data)  # Convert the labels to numbers

    n_classes = len(set(y_data))

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test, n_classes


class MajorityBaseline:
    def __init__(self, y_train: list):
        """A baseline model that always predicts the most common intent."""
        self.frequencies = Counter(y_train)
        self.mode = self.frequencies.most_common(1)[0][0]

    def predict(self, input_sentence) -> str:
        """Predict the intent of the input sentence."""
        return self.mode

    def evaluate(self, y_true: list) -> float:
        """Evaluate the model."""
        return accuracy_score(y_true, [self.mode] * len(y_true))


if __name__ == "__main__":
    # Preprocess the data
    x_train, x_test, y_train, y_test, n_classes = preprocess_data("./data/dialog_acts.dat")

    # Create a MajorityBaseline object
    model = MajorityBaseline(y_train)

    results_train = model.evaluate(y_train)
    results_test = model.evaluate(y_test)

    print(f"{results_train=}, {results_test=}")

    # Predict the intent of an input sentence
    prediction = model.predict("I want to book a flight from London to Paris")
    print(prediction)
