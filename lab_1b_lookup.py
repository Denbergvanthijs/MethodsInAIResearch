import pandas as pd

df = pd.read_csv("./data/restaurant_info.csv")

pref_dict = {"area": "centre", "pricerange": "expensive", "food": "british"}


def lookup(dictionary):
    string = "ilevel_0 in ilevel_0"

    for i in dictionary:
        if dictionary[i] != "dontcare":
            string += f" and {i} == '{dictionary[i]}'"

    df2 = df.query(string)
    rest_list = [i for i in df2['restaurantname']]

    return rest_list


restaurants = lookup(pref_dict)
print(restaurants)
