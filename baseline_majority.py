from typing import Counter


class MajorityBaseline:
    def __init__(self, filepath_dataset: str):
        """A baseline model that always predicts the most common intent."""
        # Read the dataset from local folder
        with open(filepath_dataset) as file:
            data = file.read().splitlines()  # Split the dataset into lines

        # Split each line into two parts: the intent and the utterance
        # The 1 as the second argument means that we only split the line once
        self.dataset = [row.split(" ", 1) for row in data]  

        # Make a dictionary of number of intents
        # Key is the intent, value is the number of occurrences of the intent
        self.frequencies = Counter([row[0] for row in self.dataset])
        self.most_common = self.frequencies.most_common(1)[0][0]

    def predict(self, input_sentence: str) -> str:
        """Predict the intent of the input sentence."""
        return self.most_common



if __name__ == "__main__":
    # Create a MajorityBaseline object
    baseline = MajorityBaseline("./data/dialog_acts.dat")

    # Predict the intent of an input sentence
    print(baseline.predict("I want to book a flight from London to Paris"))
