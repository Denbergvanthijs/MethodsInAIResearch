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
    return res


def create_reasoning_sentence(preferences):
    reasonstr = ''

    if preferences['touristic']:  # touristic
        reasonstr += f"{', and' if len(reasonstr) > 0 else ''} it serves quality food with affordable price"
    if preferences['romantic']:  # romantic
        reasonstr += f"{', and' if len(reasonstr) > 0 else ''} the restaurant is not too crowded and suitable for long stay"
    if preferences['child']:  # child-friendly
        reasonstr += f"{', and' if len(reasonstr) > 0 else ''} the place is good for a short visit"

    reasonstr = 'The restaurant matches your preference because' + reasonstr
    return f"{reasonstr}."


in_tou = input("Would you like a touristic place? ").lower()
in_rom = input("Is it for a romantic occassion? ").lower()
in_chi = input("Does the place have to be child-friendly? ").lower()

restos = pd.read_csv('./data/dummy_restaurants.csv')
prefs = {}
prefs['touristic'] = True if in_tou == 'y' or in_tou == 'yes' else False
prefs['romantic'] = True if in_rom == 'y' or in_rom == 'yes' else False
prefs['child'] = True if in_chi == 'y' or in_chi == 'yes' else False

restos = filter_based_on_preferences(restos, prefs).to_dict()
reasoning = create_reasoning_sentence(prefs)

print(restos)

if len(restos['area']) < 1:
    print("Sorry, we cannot find any place that match your preference.")
else:
    satisfied = False
    for key, val in restos['restaurantname'].items():
        print(
            f"I recommend { val }, it is a { restos['pricerange'][key] } { restos['food'][key] } restaurant in the { restos['area'][key] } of town.\n{ reasoning }")

        response = input("Are you happy with this recommendation? ")
        if response.lower() == 'y' or response.lower() == 'yes':
            satisfied = True
            break

    if satisfied:
        print("Thank you and goedendag!")
    else:
        print("Sorry, that is all the recommendations we have.")
