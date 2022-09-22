import pandas as pd

df = pd.read_csv("./data/restaurant_info.csv")

pref_dict = {"area": "centre", "pricerange": "expensive", "food": "british"}


def lookup(dictionary):
    query = "ilevel_0 in ilevel_0"

    for key, value in dictionary.items():
        if value != "dontcare":
            query += f" and {key} == '{value}'"

    df_output = df.query(query)
    results = [restaurant for restaurant in df_output['restaurantname']]

    return results


restaurants = lookup(pref_dict)
print(restaurants)
