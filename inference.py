import pandas as pd


def filter_based_on_preferences(restaurants, preferences: dict):
    """Filters a list of restaurants based on the user's preferences (touristic, romantic,
    and child-friendliness).

    Parameters
        - restaurants: (list) A list of Restaurant objects.
        - preferences: (list) An array of int denoting the user's preferences. Contains 3 elements, which are
        mapped to: [0] touristic, [1] romantic, [2] child-friendliness. Each element can only be 1 (for true) or 0 (for false).
    Returns
        - An array of Restaurant objects that match the preferences (a subset of the input param),
        or None if no match is found.
    """
    querystr = "ilevel_0 in ilevel_0"

    if preferences['touristic']:  # touristic
        querystr += " and (pricerange == 'cheap' or (pricerange == 'moderate' and foodquality == 'good'))"
    if preferences['romantic']:  # romantic
        querystr += " and crowdedness == 'not busy' and lengthofstay == 'long'"
    if preferences['child']:  # child-friendly
        querystr += " and lengthofstay == 'short'"

    res = restaurants.query(querystr)
    names = res['restaurantname'].values.tolist()
    reasonstr = ''

    if preferences['touristic']:  # touristic
        reasonstr += f"{', and' if len(reasonstr) > 0 else ''} {'it serves' if len(names) == 1 else 'they serve'} quality food with affordable price"
    if preferences['romantic']:  # romantic
        reasonstr += f"{', and' if len(reasonstr) > 0 else ''} {'it is' if len(names) == 1 else 'they are'} not too crowded and suitable for long stay"
    if preferences['child']:  # child-friendly
        reasonstr += f"{', and' if len(reasonstr) > 0 else ''} {'it is' if len(names) == 1 else 'they are'} good for a short visit"

    if len(names) > 0:
        resultstr = ' and '.join(names)
        resultstr += f" {'matches' if len(names) == 1 else 'match'} your preference because{reasonstr}."
    else:
        resultstr = "Sorry, we cannot find any place that match your preference."

    return res, resultstr


restos = pd.read_csv('./data/dummy_restaurants.csv')
prefs = {
    "touristic": True,
    "romantic": False,
    "child": True
}

filtered_restos, reason_string = filter_based_on_preferences(restos, prefs)
print(filtered_restos)
print(reason_string)