from dialogstate import DialogState

if __name__ == "__main__":
    # Test cases
    dialog_state = DialogState()
    dialog_state.act("I'm looking for world food")
    assert dialog_state.slots["food"] == "world"

    dialog_state = DialogState()
    dialog_state.act("I want a restaurant that serves world food")
    assert dialog_state.slots["food"] == "world"

    dialog_state = DialogState()
    dialog_state.act("I want a restaurant serving Swedish food")
    assert dialog_state.slots["food"] == "swedish"

    dialog_state = DialogState()
    dialog_state.act("I'm looking for a restaurant in the center")
    assert dialog_state.slots["area"] == "centre"

    dialog_state = DialogState()
    dialog_state.act("I would like a cheap restaurant in the west part of town")
    assert dialog_state.slots["pricerange"] == "cheap"
    assert dialog_state.slots["area"] == "west"

    dialog_state = DialogState()
    dialog_state.act("I'm looking for a moderately priced restaurant in the west part of town")
    assert dialog_state.slots["pricerange"] == "moderate"
    assert dialog_state.slots["area"] == "west"

    dialog_state = DialogState()
    dialog_state.act("I'm looking for a restaurant in any area that serves Tuscan food")
    # assert dialog_state.slots["area"] == "dontcare"
    assert dialog_state.slots["food"] == "tuscan"

    dialog_state = DialogState()
    dialog_state.act("Can I have an expensive restaurant")
    assert dialog_state.slots["pricerange"] == "expensive"

    dialog_state = DialogState()
    dialog_state.act("I'm looking for an expensive restaurant and it should serve international food")
    assert dialog_state.slots["pricerange"] == "expensive"
    assert dialog_state.slots["food"] == "international"

    dialog_state = DialogState()
    dialog_state.act("I need a Cuban restaurant that is moderately priced")
    assert dialog_state.slots["food"] == "cuban"
    assert dialog_state.slots["pricerange"] == "moderate"

    dialog_state = DialogState()
    dialog_state.act("I'm looking for a moderately priced restaurant with Catalan food")
    assert dialog_state.slots["pricerange"] == "moderate"
    assert dialog_state.slots["food"] == "catalan"

    dialog_state = DialogState()
    dialog_state.act("What is a cheap restaurant in the south part of town")
    assert dialog_state.slots["pricerange"] == "cheap"
    assert dialog_state.slots["area"] == "south"

    dialog_state = DialogState()
    dialog_state.act("What about Chinese food")
    assert dialog_state.slots["food"] == "chinese"

    dialog_state = DialogState()
    dialog_state.act("I wanna find a cheap restaurant")
    assert dialog_state.slots["pricerange"] == "cheap"

    dialog_state = DialogState()
    dialog_state.act("I'm looking for Persian food please")
    assert dialog_state.slots["food"] == "persian"

    dialog_state = DialogState()
    dialog_state.act("Find a Cuban restaurant in the center")
    assert dialog_state.slots["food"] == "cuban"

    print("All tests passed!")
