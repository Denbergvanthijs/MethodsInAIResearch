import os
import pickle
import random
from typing import List

import Levenshtein as lev
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')


class DialogState:
    def __init__(self, fp_restaurant_info: str = "./data/restaurant_info.csv", fp_dialog_acts: str = "./data/dialog_acts.dat", fp_pickle: str = "./data/logreg.pkl", max_lev_distance: int = 3) -> None:
        self.history_utterances = []
        self.history_intents = [None]
        self.history_states = ["1"]
        self.slots = {"area": None, "food": None, "pricerange": None}

        self.states = ("1", "2", "3", "3.1", "4", "5", "6", "7", "8")
        self.intents = ("ack", "affirm", "bye", "confirm", "deny", "hello", "inform",
                        "negate", "null", "repeat", "reqalts", "reqmore", "request", "restart", "thankyou")

        self.area = ("north", "south", "west", "east", "centre")
        self.food = ('jamaican', 'chinese', 'cuban', 'portuguese', 'australasian', 'moroccan', 'traditional',
                     'international', 'seafood', 'steakhouse', 'japanese', 'gastropub', 'asian oriental', 'catalan',
                     'north american', 'polynesian', 'french', 'european', 'vietnamese', 'tuscan', 'romanian', 'swiss',
                     'thai', 'british', 'modern european', 'fusion', 'african', 'indian', 'turkish', 'italian', 'korean',
                     'lebanese', 'persian', 'mediterranean', 'bistro', 'spanish', 'indonesian')
        self.pricerange = ("expensive", "cheap", "moderate")
        self.dontcare = ("any", "doesnt matter", 'dont care', 'dontcare')

        self.state_to_slot = {"2": "area", "3": "food", "4": "pricerange"}
        self.restaurants = []
        self.restaurant_info = pd.read_csv(fp_restaurant_info)

        self.max_lev_distance = max_lev_distance
        self.stopwords = stopwords.words('english')

        # Save the model to a pickle file to speedup the loading process
        if not os.path.exists(fp_pickle):
            with open(fp_dialog_acts) as file:
                data = file.read().splitlines()  # Split the dataset into lines

            # Split each line into two parts: the intent and the utterance
            # The 1 as the second argument means that we only split the line once
            dataset = [row.split(" ", 1) for row in data]
            y_data, x_data = zip(*dataset)

            self.vec = CountVectorizer()
            self.vec.fit(x_data)
            X_vec = self.vec.transform(x_data)

            self.intent_model = LogisticRegression()
            self.intent_model.fit(X_vec, y_data)

            pickle.dump([self.vec, self.intent_model], open(fp_pickle, "wb"))
        else:
            self.vec, self.intent_model = pickle.load(open(fp_pickle, "rb"))

    def act(self, user_utterance: str) -> None:
        """Determines the intent of current user utterance, fills slots and determines the next state of the dialog."""
        user_utterance_processed = self.preprocessing(user_utterance)
        self.history_utterances.append(user_utterance_processed)
        self.history_intents.append(self.classify_intent(user_utterance_processed))

        self.run_state()
        print(user_utterance)
        self.fill_slots(user_utterance_processed)

        self.history_states.append(self.determine_next_state())
        print(f"intend={self.history_intents[-1]}; slots={self.slots}")

    def preprocessing(self, user_utterance: str):
        """Preprocesses the user utterance by tokenizing and removing stopwords."""
        user_utterance = user_utterance.lower()
        user_utterance = word_tokenize(user_utterance)
        return " ".join(word for word in user_utterance if word not in self.stopwords)

    def run_state(self) -> None:
        """Runs the current state of the dialog."""
        if self.history_states[-1] == "1":
            print("1.  Welcome to the UU restaurant system! You can ask for restaurants by area, price range or food type. How may I help you?")
        elif self.history_states[-1] == "2":
            print("2. What part of town do you have in mind?")
        elif self.history_states[-1] == "3":
            print("3. What kind of food would you like?")
        elif self.history_states[-1] == "3.1":
            print(f"3.1. ")
        elif self.history_states[-1] == "4":
            print("4.  Would you like something in the cheap, moderate, or expensive price range?")
        elif self.history_states[-1] == "5":
            self.restaurant_chosen = random.choice(self.restaurants)
            print(f"5. {self.restaurant_chosen} is a great restaurant in the {self.slots.get('area')}, it is a {self.slots.get('pricerange')} restaurant and it serves a {self.slots.get('food')} cuisine.")
        elif self.history_states[-1] == "6":
            print(
                f"6. I'm sorry but there is no {self.slots.get('pricerange')} place serving {self.slots.get('food')} cuisine in the {self.slots.get('area')}. What else can I help you with?")
        elif self.history_states[-1] == "7":
            print(f"7. Would you like the phone number, adress or postal code of {self.restaurant_chosen}?")
        elif self.history_states[-1] == "8":
            print(f"8. Goodbye and have a nice day!")
            exit()

    def classify_intent(self, user_utterance: str) -> str:
        """Classifies the intent of the user utterance using a logistic regression model."""
        return self.intent_model.predict(self.vec.transform([user_utterance]))[0]

    def fill_slots(self, user_utterance: str) -> None:
        """Fills the slots with the information from the user utterance."""
        slots_strict = {}  # Dict with slots that directly match the user utterance
        slots_lev = {}  # Dict with slots that match the user utterance with a maximum Levenshtein distance

        # Dict indicating the Levenshtein distance for each slot
        slots_dist = {'area': self.max_lev_distance + 1,
                      'pricerange': self.max_lev_distance + 1,
                      'food': self.max_lev_distance + 1
                      }

        for word in user_utterance.split():
            # Dont care --> Return slot based of previous state
            previous_state = self.history_states[-1]

            # If previous question was about {area, pricerange or food}
            # Only then we can use the dontcare word
            if word in self.dontcare and previous_state in ("2", "3", "4"):
                slots_strict[self.state_to_slot.get(previous_state)] = "dontcare"

            # If direct match is found, set the slot to the found word
            for category, category_name in zip((self.area, self.pricerange, self.food), ("area", "pricerange", "food")):
                if word in category:
                    slots_strict[category_name] = word

            # Find what category the word belongs to
            # Some errors will be tolerated, e.g. "for" and "north", "the" and "thai"
            # But Levenshtein by itself is probably not enough
            match_word, match_dist, match_category = self.recognize_keyword(word)

            # Check if for that category, there's already a better match in the lev slots
            # For each slot, we only want to keep the best match
            if match_word and match_dist < slots_dist.get(match_category):
                slots_dist[match_category] = match_dist  # Update the distance with the new distance
                slots_lev[match_category] = match_word  # Update the best match for that category

        # Priority to strict slots, overwrite Levenshtein slots with strict slots
        slots_lev.update(slots_strict)
        self.slots.update(slots_lev)  # Update the user preferences based on the user's current utterance

    def determine_next_state(self) -> str:
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
            # If no restaurant in DB matches the user's preferences, go to state 6
            if self.lookup() is None:
                return "6"
            else:
                return "5"
        if self.history_states[-1] in ("5", "6", "7"):
            # If the user wants to know more about the restaurant, go to state 7
            if self.history_intents[-1] == "request":
                return "7"
            # If the user wants to end the dialog, go to state 8
            if self.history_intents[-1] == "bye":
                print(f"8. Goodbye and have a nice day!")
                exit()

        return "undefined"  # This should never happen

    def lookup(self) -> List[str]:
        """Looks up all restaurants in the database that matches the user's preferences."""
        query_text = "ilevel_0 in ilevel_0"

        for key, value in self.slots.items():
            if value != "dontcare":
                query_text += f" and {key} == '{value}'"

        df_output = self.restaurant_info.query(query_text)

        self.restaurants = df_output["restaurantname"].values
        return self.restaurants

    def recognize_keyword(self, key: str) -> str:
        """Recognizes the keyword that is closest to the key in terms of Levenshtein distance."""
        res_word = None  # Resulting word
        res_dist = self.max_lev_distance  # Resulting distance
        res_cat = None  # Resulting category

        for category, category_name in zip((self.area, self.pricerange, self.food), ("area", "pricerange", "food")):
            for word in category:
                dist = lev.distance(word, key)  # Levenshtein distance
                if dist < res_dist:  # If the distance is smaller than the current best match
                    res_word = word
                    res_dist = dist
                    res_cat = category_name

        return res_word, res_dist, res_cat

    def __str__(self) -> str:
        return f"slots={self.slots}; intent={self.intents[-1]}; state={self.states[-1]}; history_intents={self.history_intents}; history_states={self.history_states}; lookup={self.restaurants}"


if __name__ == "__main__":
    dialog_state = DialogState()
    # print(dialog_state)
    dialog_state.act("I'm looking for a cheap brimish food")
    # print(dialog_state)
    dialog_state.act("I'm looking for a restaurant in the centre")
    # print(dialog_state)
    # dialog_state.act("Can I have a cheap restaurant")
    dialog_state.act("dontcare")
    # print(dialog_state)
    dialog_state.act("Goodbye")
    # print(dialog_state)
