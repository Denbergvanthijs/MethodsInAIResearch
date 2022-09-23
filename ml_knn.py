import spacy
from sklearn.neighbors import KNeighborsClassifier

from baseline_majority import preprocess_data


class KNNModel:
    def __init__(self, spacy_model: str = "en_core_web_sm", n_neighbors: int = 5):
        """A Transformer model that uses word embeddings to do text classification."""
        self.nlp = spacy.load(spacy_model)
        # self.nlp.vocab.prune_vectors(20)

        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def _data_to_vectors(self, x_data: list) -> list:
        """Convert the data to vectors."""
        docs = [self.nlp(sentence) for sentence in x_data]
        x_vectors = [doc.vector for doc in docs]
        return x_vectors

    def train(self, x_train: list, y_train: list):
        """Train the model."""
        vectors = self._data_to_vectors(x_train)
        self.model.fit(vectors, y_train)

    def evaluate(self, x_test: list, y_test: list) -> float:
        """Evaluate the model."""
        vectors = self._data_to_vectors(x_test)
        return self.model.score(vectors, y_test)

    def predict(self, input_sentence: str) -> str:
        """Predict the intent of the input sentence."""
        return self.model.predict([self.nlp(input_sentence).vector])


if __name__ == "__main__":
    # Preprocess the data
    x_train, x_test, y_train, y_test, n_classes = preprocess_data("./data/dialog_acts.dat")

    # Create a KNNModel object
    model = KNNModel(n_neighbors=7)

    # Predict the intent of an input sentence
    model.train(x_train, y_train)
    results_train = model.evaluate(x_train, y_train)
    results_test = model.evaluate(x_test, y_test)

    print(f"{results_train=}, {results_test=}")

    # Predict the intent of an input sentence
    prediction = model.predict("I want to book a flight from London to Paris")
    print(prediction)
