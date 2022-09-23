import Levenshtein as lev


def search_restaurant(utterance, err_tolerance):
    restaurants = ['Italian', 'Indian', 'Korean', 'Indonesian', 'American']
    words = utterance.split()

    found = False
    for w in words:
        for res in restaurants:
            if lev.distance(w, res) <= err_tolerance:
                print(f"Understood that \"{w}\" is \"{res}\"")
                found = True

    if not found: print(f"Nothing's found")


ut1 = "I am looking for moderately priced Indonesian food in the centre of town."  # Found Indonesian
ut2 = "Do you have any recommendation for a Australian restaurant?"  # Not found
ut3 = "Have you got any recommendation on americen or Koreyan food around here?"  # Found American and Korean

ms1 = "I am looking for moderately priced Indonesien food in the centre of town."  # Found
ms2 = "I am looking for moderately priced Indoenesien food in the centre of town."  # Found
ms3 = "I am looking for moderately priced Inoesieen food in the centre of town."  # Not found

search_restaurant(ut1, 3)
search_restaurant(ut2, 3)
search_restaurant(ut3, 3)
print("======")
search_restaurant(ms1, 3)
search_restaurant(ms2, 3)
search_restaurant(ms3, 3)
print("======")
search_restaurant(ms1, 4)  # error tolerance 4 is too loose, 3 still makes sense.
search_restaurant(ms2, 4)
search_restaurant(ms3, 4)
