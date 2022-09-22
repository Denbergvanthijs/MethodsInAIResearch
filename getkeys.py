area = ["north", "south", "west", "east", "centre"]
pricerange = ["expensive", "cheap", "moderate"]
dontcare = ["any", "doesnt matter", 'dont care', 'dontcare']
food = ['jamaican', 'chinese', 'cuban', 'portuguese', 'australasian', 'moroccan', 'traditional', 'international', 'seafood', 'steakhouse', 'japanese', 'gastropub', 'asian oriental', 'catalan', 'north american', 'polynesian', 'french', 'european', 'vietnamese', 'tuscan', 'romanian', 'swiss', 'thai', 'british', 'modern european', 'fusion', 'african', 'indian', 'turkish', 'italian', 'korean', 'lebanese', 'persian', 'mediterranean', 'bistro', 'spanish']


def lookup(sentence, previous):
    """ lookup checks if words in the sentence fit the slots based on lists,
     and returns a dictionary with all the filled lists.
     it fills "dontcare" for when the user has no preference. """

    slots, keys = {}, ["area", "food", "price"]

    for i in sentence.split():
        # Dont care --> Return slot based of previous computer message
        # next_word = s[s.index(i) + 1]
        if i in dontcare and previous in keys:
            slots[previous] = "dontcare"

        # Return intent for area, price, food
        elif i in area:
            slots['area'] = i
        elif i in pricerange:
            slots['pricerange'] = i
        elif i in food:
            slots['food'] = i

    return slots

p = "chinese food in the west"
print(lookup(p, "area"))
