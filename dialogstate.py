from typing import List

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


class DialogState:
    def __init__(self, fp_restaurant_info: str = "./data/restaurant_info.csv") -> None:
        self.history_utterances = []
        self.history_intents = [None]
        self.history_states = ["1"]
        self.slots = {"area": None, "food": None, "pricerange": None}

        self.states = ["1", "2", "3", "3.1", "4", "5", "6", "7", "8"]
        self.intents = ["ack", "affirm", "bye", "confirm", "deny", "hello", "inform",
                        "negate", "null", "repeat", "reqalts", "reqmore", "request", "restart", "thankyou"]

        self.area = ["north", "south", "west", "east", "centre"]
        self.food = ['jamaican', 'chinese', 'cuban', 'portuguese', 'australasian', 'moroccan', 'traditional', 'international', 'seafood', 'steakhouse', 'japanese', 'gastropub', 'asian oriental', 'catalan', 'north american', 'polynesian', 'french', 'european', 'vietnamese', 'tuscan', 'romanian', 'swiss', 'thai', 'british', 'modern european', 'fusion', 'african', 'indian', 'turkish', 'italian', 'korean', 'lebanese', 'persian', 'mediterranean', 'bistro', 'spanish', 'indonesian']
        self.pricerange = ["expensive", "cheap", "moderate"]
        
        self.dontcare = ["any", "doesnt matter", 'dont care', 'dontcare']
        self.previous = {}
        self.restaurants = []

        dialog_acts = pd.read_csv("./data/dialog_acts.dat", header=None, sep="\s\s+", engine="python")
        dialog_acts["intent"] = dialog_acts[0].str.split(" ", 1).str[0]
        dialog_acts[0] = dialog_acts[0].str.split(n=1).str[1]
        dialog_acts.rename(columns={dialog_acts.columns[0]: "utterance"}, inplace=True)

        X = dialog_acts["utterance"].values
        y = dialog_acts["intent"].values

        self.vec = CountVectorizer()
        self.vec.fit(X)
        X_vec = self.vec.transform(X)

        self.intent_model = LogisticRegression()
        self.intent_model.fit(X_vec, y)
        self.restaurant_info = pd.read_csv(fp_restaurant_info)


    def act(self, user_utterance: str) -> None:
        """Determines the intent of current user utterance, fills slots and determines the next state of the dialog."""
        self.history_utterances.append(user_utterance)

        self.history_intents.append(self.classify_intent(user_utterance))

        self.fill_slots(user_utterance)

        self.history_states.append(self.determine_next_state(user_utterance))

    def classify_intent(self, user_utterance: str) -> str:
        """Classifies the intent of the user utterance using a logistic regression model."""
        return self.intent_model.predict(self.vec.transform([user_utterance]))[0]

    def fill_slots(self, user_utterance: str) -> None:
        """Fills the slots with the information from the user utterance."""
        slots, keys = {}, ["area", "food", "price"]
        
        for i in user_utterance.split():
            # Dont care --> Return slot based of previous computer message
            # next_word = s[s.index(i) + 1]
            if i in self.dontcare and self.previous in keys:
                slots[self.previous] = "dontcare"

            # Return intent for area, price, food
            elif i in self.area:
                slots['area'] = i
            elif i in self.pricerange:
                slots['pricerange'] = i
            elif i in self.food:
                slots['food'] = i
        
        self.previous = slots
        self.slots.update(slots)

    def determine_next_state(self, user_utterance: str) -> str:
        """Determines the next state of the dialog based on the current state, filled slots and the intent of the current utterance."""
        if self.history_states[-1] in ("1", "2", "3.1"):
            if self.slots["area"] is None:
                return "2"
        if self.history_states[-1] in ("1", "2", "3.1", "3"):
            if self.slots["food"] is None:
                return "3"
            # If no restaurant in DB matches the user's preferences, go to state 3.1
            if self.lookup() is None:
                return "3.1"
            if self.slots["pricerange"] is None:
                return "4"
        if self.history_states[-1] in ("1", "2", "3.1", "3", "4"):
            if self.lookup() is None:
                return "6"
            else:
                return "5"
        if self.history_states[-1] in ("5", "6", "7"):
            if self.history_intents[-1] == "request":
                return "7"
            if self.history_intents[-1] == "bye":
                return "8"

        return "undefined"

    def lookup(self) -> List[str]:
        """Looks up all restaurants in the database that matches the user's preferences."""
        query = "ilevel_0 in ilevel_0"

        for key, value in self.slots.items():
            if value != "dontcare":
                query += f" and {key} == '{value}'"

        df_output = self.restaurant_info.query(query)
        self.restaurants = [restaurant for restaurant in df_output['restaurantname']]

    def __str__(self) -> str:
        return f"slots={self.slots}; intent={self.intents[-1]}; state={self.states[-1]}; history_intents={self.history_intents}; history_states={self.history_states}; lookup={self.restaurants}"


if __name__ == "__main__":
    dialog_state = DialogState()
    print(dialog_state)
    dialog_state.act("I'm looking for british food")
    print(dialog_state)
    dialog_state.act("I'm looking for a restaurant in the centre")
    print(dialog_state)
    dialog_state.act("Can I have a cheap restaurant")
    print(dialog_state)
    dialog_state.act("Goodbye")
    print(dialog_state)
