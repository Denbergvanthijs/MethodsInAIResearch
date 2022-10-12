import os
import pickle
import random
import ssl
import time
from itertools import cycle
from typing import List

import Levenshtein as lev
import nltk
import pandas as pd
from dotenv import dotenv_values
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

try:
    # For Mac machines, bypass SSL checking for NLTK
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
finally:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)


class DialogState:
    def __init__(self, fp_restaurant_info: str = "./data/restaurant_data.csv", fp_dialog_acts: str = "./data/dialog_acts.dat",
                 fp_pickle: str = "./data/logreg.pkl", configurability: dict = {}) -> None:
        """Initializes the Dialog State Manager."""
        self.history_utterances = []
        self.history_states = ["1"]  # Start with state 1
        self.history_intents = [None]  # Start with no intent for state 1
        self.slots = {"area": None, "food": None, "pricerange": None}
        self.slots_preferences = {"preference": None, "touristic": None, "romantic": None, "child": None}

        self.states = ("1", "2", "3", "3.1", "4", "5", "6", "7", "8")
        self.intents = ("ack", "affirm", "bye", "confirm", "deny", "hello", "inform",
                        "negate", "null", "repeat", "reqalts", "reqmore", "request", "restart", "thankyou")

        self.area = ("north", "east", "south", "west", "centre")
        self.food = ("jamaican", "chinese", "cuban", "portuguese", "australasian", "moroccan", "traditional",
                     "international", "seafood", "steakhouse", "japanese", "gastropub", "asian oriental", "catalan",
                     "north american", "polynesian", "french", "european", "vietnamese", "tuscan", "romanian", "swiss",
                     "thai", "british", "modern european", "fusion", "african", "indian", "turkish", "italian", "korean",
                     "lebanese", "persian", "mediterranean", "bistro", "spanish", "indonesian", "world", "swedish")
        self.pricerange = ("cheap", "moderate", "expensive")
        self.dontcare = ("any", "doesnt matter", "dont care", "dontcare")
        self.yes = ("yes", "y", "yeah", "sure", "agree", "absolutely", "aye", "certainly", "ok", "yep", "yup")

        # Maps states to dictionairy keys
        self.state_to_slot = {"2": "area", "3": "food", "4": "pricerange",
                              "9": "preference", "9.1": "touristic", "9.2": "romantic", "9.3": "child"}

        self.restaurant_info = pd.read_csv(fp_restaurant_info)
        self.restaurants = None  # List of restaurants that match the current slots

        self.stopwords = stopwords.words("english")

        self.configurability = configurability
        self.formal = self.configurability.get("formal") == 'True'
        self.output_in_caps = self.configurability.get("output_in_caps") == 'True'
        self.print_info = self.configurability.get("print_info") == 'True'

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

    def act(self, user_utterance: str = None) -> None:
        """Determines the intent of current user utterance, fills slots and determines the next state of the dialog."""
        self.execute_state()

        if user_utterance is None:  # Ask the user for an utterance via CLI
            user_utterance = input("User: ")
        else:
            print(user_utterance)  # We only print the user utterance if the user is not asked for an input

        user_utterance_processed = self.preprocessing(user_utterance)
        self.history_utterances.append(user_utterance_processed)

        current_intent = self.classify_intent(user_utterance_processed)
        self.history_intents.append(current_intent)

        if self.history_states[-1] in ("9", "9.1", "9.2", "9.3"):
            self.fill_slots_preferences(user_utterance_processed)
        elif self.history_states[-1] in ("1", "2", "3", "3.1", "4"):
            self.fill_slots(user_utterance_processed)

        next_state = self.determine_next_state()
        self.history_states.append(next_state)

        if self.print_info:
            self.print_w_option(f"{current_intent=}; {next_state=}; slots={self.slots}; slots_preferences={self.slots_preferences}")

    def preprocessing(self, user_utterance: str):
        """Preprocesses the user utterance by tokenizing and removing stopwords."""
        user_utterance = user_utterance.lower()
        user_utterance = word_tokenize(user_utterance)
        return " ".join(word for word in user_utterance if word not in self.stopwords)

    def execute_state(self) -> None:
        """Runs the current state of the dialog."""
        delay_time = float(self.configurability.get("delay", 0)) / 1000

        if delay_time > 0:
            self.print_w_option("Please wait ...")
            time.sleep(delay_time)

        if self.history_states[-1] == "1":
            if self.formal:
                self.print_w_option("1.  Welcome to the UU restaurant system! "
                                    "You can ask for restaurants by area, price range or food type. How may I help you?")
            else:
                self.print_w_option("1. Yarr matey, I be recommending you the best taverns! Tell me your price range, area, "
                                    "and food wishes or I'll throw you off my ship!")

        elif self.history_states[-1] == "2":
            if self.formal:
                self.print_w_option("2. What part of town do you have in mind? Choose from {north, south, east, west, centre}.")
            else:
                self.print_w_option("2. Look at your compass landlubber, be it pointing north, south, east, west or centre?")

        elif self.history_states[-1] == "3":
            if self.formal:
                self.print_w_option("3. What kind of food would you like? Choose any cuisine!")
            else:
                self.print_w_option("3. Yer don't want scurvy right? Pick a food!")

        elif self.history_states[-1] == "3.1":
            if self.formal:
                self.print_w_option(f"3.1. There are no restaurants in the {self.slots['area']} area "
                                    f"that serve {self.slots['food']}. What else can I help you with?")
            else:
                self.print_w_option(f"3.1. There no be taverns in the {self.slots['area']} area "
                                    f"that serve {self.slots['food']} loot. Any alternatives?")

        elif self.history_states[-1] == "4":
            if self.formal:
                self.print_w_option("4.  Would you like the restaurant to be in the cheap, moderate, or expensive price range?")
            else:
                self.print_w_option("4.  I hope ye got some doubloons, pick a cheap, moderate or expensive tavern.",)

        elif self.history_states[-1] == "9":
            if self.formal:
                self.print_w_option("9. Do you have additional requirements? Yes or no?")
            else:
                self.print_w_option("9. Do you have additional requirements? Yes or no?")  # TODO: change to piratespeak
        elif self.history_states[-1] == "9.1":
            if self.formal:
                self.print_w_option("9.1. Would you like a touristic place?")
            else:
                self.print_w_option("9.1. Would you like a touristic place?")  # TODO: change to piratespeak
        elif self.history_states[-1] == "9.2":
            if self.formal:
                self.print_w_option("9.2. Is it for a romantic occasion?")
            else:
                self.print_w_option("9.2. Is it for a romantic occasion?")  # TODO: change to piratespeak
        elif self.history_states[-1] == "9.3":
            if self.formal:
                self.print_w_option("9.3. Does the place have to be child-friendly?")
            else:
                self.print_w_option("9.3. Does the place have to be child-friendly?")  # TODO: change to piratespeak

        elif self.history_states[-1] == "5":
            self.restaurant_chosen = next(self.restaurants)  # if not isinstance(self.restaurants, type(None)) else None
            if self.formal:
                self.print_w_option(f"5. I recommend {self.restaurant_chosen}, it is a {self.slots.get('pricerange')} "
                                    f"{self.slots.get('food')} restaurant"
                                    f" in the {self.slots.get('area')} of town.")
            else:
                self.print_w_option(f"5. {self.restaurant_chosen} is a jolly tavern in the {self.slots.get('area')}, "
                                    f"it be a {self.slots.get('pricerange')} tavern serving {self.slots.get('food')} loot.")
            if self.slots_preferences["preference"]:
                self.print_w_option(self.create_reasoning_sentence())  # TODO: implement informal for reasoning sentence

        elif self.history_states[-1] == "6":
            if self.formal:
                self.print_w_option(f"6. I'm sorry but there is no {self.slots.get('pricerange')} place "
                                    f"serving {self.slots.get('food')} cuisine in the {self.slots.get('area')}."
                                    "What else can I help you with?")
            else:
                self.print_w_option(f"6. Sink me, but there no be a {self.slots.get('pricerange')} tavern "
                                    f"serving {self.slots.get('food')} loot in the {self.slots.get('area')}. What else do ye want?")

        elif self.history_states[-1] == "7":
            if self.formal:
                self.print_w_option(f"7. Would you like the phone number, address or postal code of {self.restaurant_chosen}?")
            else:
                self.print_w_option(f"7. Ye want the phone number, address or postal code of {self.restaurant_chosen}?")

        elif self.history_states[-1] == "8":
            if self.configurability.get("formal") == 'True':
                self.print_w_option("8. Goodbye and have a nice day!")
            else:
                self.print_w_option("8. Ahoy landlubber!")
            exit()

        elif self.history_states[-1] == "10":
            if self.formal:
                self.print_w_option("10. Sorry, I cannot find a place that matches your criteria."
                                    " Would you like to try searching without the additional preferences?")
            else:
                self.print_w_option("10. Sorry, I cannot find a place that matches your criteria."
                                    " Would you like to try searching without the additional preferences?")   # TODO: change to piratespeak

        elif self.history_states[-1] == "11":
            if self.formal:
                self.print_w_option("11. Sorry, I cannot find a place that matches your criteria"
                                    " Would you like me to try broadening your search?")
            else:
                self.print_w_option("11. Sorry, I cannot find a place that matches your criteria"
                                    " Would you like me to try broadening your search?")   # TODO: change to piratespeak

    def classify_intent(self, user_utterance: str) -> str:
        """Classifies the intent of the user utterance using a logistic regression model."""
        return self.intent_model.predict(self.vec.transform([user_utterance]))[0]

    def fill_slots(self, user_utterance: str) -> None:
        """Fills the slots with the information from the user utterance."""
        slots_strict = {}  # Dict with slots that directly match the user utterance
        slots_lev = {}  # Dict with slots that match the user utterance with a maximum Levenshtein distance

        # Dict indicating the Levenshtein distance for each slot
        slots_dist = {"area": int(self.configurability.get("max_lev_distance", 3)) + 1,
                      "pricerange": int(self.configurability.get("max_lev_distance", 3)) + 1,
                      "food": int(self.configurability.get("max_lev_distance", 3)) + 1}

        for word in user_utterance.split():
            # Dont care --> Return slot based of most recent state
            current_state = self.history_states[-1]

            # If most recent state was about {area, pricerange or food}
            # Only then we can use the dontcare word
            if word in self.dontcare and current_state in ("2", "3", "4"):
                slots_strict[self.state_to_slot.get(current_state)] = "dontcare"

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

    def fill_slots_preferences(self, user_utterance: str) -> None:
        """Fills the preference slots with the information from the user utterance."""
        slot_preference = self.state_to_slot.get(self.history_states[-1])

        # If the user affirms the preference, set the slot to "yes"
        self.slots_preferences[slot_preference] = self.determine_yes_or_no(user_utterance)

    def determine_next_state(self) -> str:
        """Determines the next state of the dialog based on the current state, filled slots and the intent of the current utterance."""
        # Always be able to exit
        if self.history_intents[-1] in ("bye", "thankyou"):
            if self.formal:
                self.print_w_option("8. Goodbye and have a nice day!")
            else:
                self.print_w_option("8. Ahoy landlubber!")
            exit()

        if self.history_states[-1] in ("1", "2", "3.1"):
            if self.slots["area"] is None:
                return "2"

        if self.history_states[-1] in ("1", "2", "3.1", "3"):
            if self.slots["food"] is None:
                return "3"
            # If no restaurant in DB matches the user's preferences, go to state 3.1
            if not self.lookup():
                return "3.1"
            if self.slots["pricerange"] is None:
                return "4"

        if self.history_states[-1] in ("1", "2", "3.1", "3", "4"):
            # If no restaurant in DB matches the user's preferences, go to state 6
            if not self.lookup():
                return "11"
            else:
                return "9"

        if self.history_states[-1] in ("9"):
            # If the user has additional requirements, go to state 9.1
            if self.slots_preferences["preference"]:
                return "9.1"
            # If the user has no additional requirements, go to state 5
            else:
                return "5"

        if self.history_states[-1] in ("9.1"):
            return "9.2"

        if self.history_states[-1] in ("9.2"):
            return "9.3"

        if self.history_states[-1] in ("9.3"):
            if not self.lookup():
                return "10"
            else:
                return "5"

        if self.history_states[-1] == "10":
            if self.determine_yes_or_no(self.history_utterances[-1]):
                # Drop the preferences
                self.slots_preferences = {"preference": None, "touristic": None, "romantic": None, "child": None}
                if not self.lookup():
                    return "11"
                else:
                    return "5"
            else:
                return "6"

        if self.history_states[-1] == "11":
            if self.determine_yes_or_no(self.history_utterances[-1]):
                self.slots["pricerange"] = None
                self.slots["area"] = None
                if not self.lookup():
                    return "6"
                else:
                    return "5"
            else:
                return "6"

        if self.history_states[-1] in ("5", "6", "7"):
            # If the user wants to know more about the restaurant, go to state 7
            if self.history_intents[-1] == "request":
                return "7"
            # If the user wants to end the dialog, go to state 8
            if self.history_intents[-1] in ("bye", "thankyou"):
                if self.formal:
                    self.print_w_option("8. Goodbye and have a nice day!")
                else:
                    self.print_w_option("8. Ahoy landlubber!")

        if self.history_states[-1] == "5" and self.history_intents[-1] == "reqalts":
            # In state 5 (self.restaurants is not empty), so user can loop through the findings.
            return "5"

        if self.history_states[-1] == "6" and self.history_intents[-1] == "reqalts":
            # In state 6 (self.restaurants is empty), so cannot call next() on self.restaurants.
            return "6"

        return "undefined"  # This should never happen

    def filter_based_on_preferences(self) -> str:
        """Filters a list of restaurants based on the user's preferences (touristic, romantic, and child-friendliness).

        Returns
            - A query string with the filtered restaurants
        """
        insert_errors = self.configurability.get('insert_errors') == 'True'

        query_text = ''
        if self.slots_preferences['touristic']:  # touristic
            if insert_errors:  # intentional mistake
                query_text += " and (pricerange == 'expensive' or (pricerange == 'cheap' and foodquality == 'acceptable'))"
            else:
                query_text += " and (pricerange == 'cheap' or (pricerange == 'moderate' and foodquality == 'good'))"
        if self.slots_preferences['romantic']:  # romantic
            if insert_errors:  # intentional mistake
                query_text += " and crowdedness == 'busy' and lengthofstay == 'long'"
            else:
                query_text += " and crowdedness == 'not busy' and lengthofstay == 'long'"
        if self.slots_preferences['child']:  # child-friendly
            if insert_errors:  # intentional mistake
                query_text += " and lengthofstay == 'long'"
            else:
                query_text += " and lengthofstay == 'short'"

        return query_text

    def create_reasoning_sentence(self):
        """Creates a reasoning sentence based on the user's preferences."""
        reasonstr = ''

        # TODO: Add piratespeak
        if self.slots_preferences["touristic"]:  # touristic
            reasonstr += f" it serves quality food with affordable price"
        if self.slots_preferences["romantic"]:  # romantic
            reasonstr += f"{', and' if len(reasonstr) > 0 else ''} the restaurant is not too crowded and suitable for long stay"
        if self.slots_preferences["child"]:  # child-friendly
            reasonstr += f"{', and' if len(reasonstr) > 0 else ''} the place is good for a short visit"

        reasonstr = "Reasoning: The restaurant matches your preference because" + reasonstr
        return f"{reasonstr}."

    def lookup(self) -> List[str]:
        """Looks up all restaurants in the database that matches the user's preferences."""
        query_text = "ilevel_0 in ilevel_0"

        for key, value in self.slots.items():
            if value != "dontcare" and value is not None:
                query_text += f" and {key} == '{value}'"

        if self.slots_preferences["preference"]:
            query_text += self.filter_based_on_preferences()

        df_output = self.restaurant_info.query(query_text)

        recommendations = df_output["restaurantname"].values.tolist()
        random.shuffle(recommendations)

        self.restaurants = cycle(recommendations) if recommendations else None
        return self.restaurants

    def recognize_keyword(self, key: str) -> str:
        """Recognizes the keyword that is closest to the key in terms of Levenshtein distance."""
        res_word = None  # Resulting word
        res_dist = int(self.configurability.get("max_lev_distance", 3))  # Resulting distance
        res_cat = None  # Resulting category

        for category, category_name in zip((self.area, self.pricerange, self.food), ("area", "pricerange", "food")):
            for word in category:
                dist = lev.distance(word, key)  # Levenshtein distance
                if dist < res_dist:  # If the distance is smaller than the current best match
                    res_word = word
                    res_dist = dist
                    res_cat = category_name

        return res_word, res_dist, res_cat

    def determine_yes_or_no(self, user_utterance: str) -> str:
        """Determines whether the user's utterance is a yes or no."""
        # Main checks on intent
        if self.history_intents[-1] == "affirm":
            return True
        elif self.history_intents[-1] in ("negate", "deny"):
            return False

        # Backup checks based on user utterance
        for word in self.yes:
            if word in user_utterance:
                return True

        # If no yes word is found, then it is a no
        return False

    def print_w_option(self, input_utterance: str):
        """Prints the input utterance and applies the configurability options to it.

        This function can be extended to apply more configurability options."""
        if self.output_in_caps:
            print(input_utterance.upper())
        else:
            print(input_utterance)


if __name__ == "__main__":
    configurability = dotenv_values(".env")

    dialog_state = DialogState(configurability=configurability)
    dialog_state.act("I'm looking for cheap brimish food in the north of town")
    dialog_state.act("I'm looking for a restaurant in the center")
    dialog_state.act("Yes, I would like to provide some preferences")
    dialog_state.act("Yes please!")
    dialog_state.act("No, it is not")
    dialog_state.act("Nope")
    dialog_state.act("Thank you!")
