import csv
import pandas as pd
import random
"""Add columns to the dataframe based on the consequents.
In our case, this is food quliaty, crowdedness and lengthofstay.
The variables are added randomly."""

df = pd.read_csv('restaurant_info.csv')
food_quality, crowded, lengthstay = [], [], []
food, crowd, length = ['good', 'acceptable'],['busy', 'not busy'],['long', 'short']

def add_random_columns(dataframe):
  for i in range(len(dataframe)):
    food_quality.append(random.choice(food))
    crowded.append(random.choice(crowd))
    lengthstay.append(random.choice(length))
  dataframe =dataframe.assign(foodquality=food_quality,crowdedness=crowded,lengthofstay=lengthstay)
  return(dataframe)

# Add columns and add to empty csv file
df_plus = add_random_columns(df)
df_plus.to_csv("/restaurant_data.csv")