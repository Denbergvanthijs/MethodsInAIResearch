import random

import pandas as pd


def add_random_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add columns to the dataframe based on the consequents.
    In our case, this is food quliaty, crowdedness and lengthofstay.
    The variables are added randomly."""
    food_quality, crowded, lengthstay = [], [], []
    food, crowd, length = ["good", "acceptable"], ["busy", "not busy"], ["long", "short"]

    for _ in range(len(dataframe)):
        food_quality.append(random.choice(food))
        crowded.append(random.choice(crowd))
        lengthstay.append(random.choice(length))

    dataframe = dataframe.assign(foodquality=food_quality, crowdedness=crowded, lengthofstay=lengthstay)

    return dataframe


if __name__ == "__main__":
    # Add columns and add to empty csv file
    df = pd.read_csv("./data/restaurant_info.csv")
    df_plus = add_random_columns(df)
    df_plus.to_csv("./data/restaurant_data.csv")
