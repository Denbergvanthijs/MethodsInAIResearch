import csv
import pandas as pd
import random
"""Add 3 columns based on handwritten rules for if the restaurant
is romantic, touristic and child friendly."""

df = pd.read_csv('restaurant_data.csv')
t,c,r = [], [], []

def touristic(price, food):
  if price == 'cheap':
    return True
  elif price == 'moderate' and food == 'good':
    return True
  return False

def child_friendly(stay):
  if stay == 'short':
    return True
  return False

def romantic(stay, food, price, crowdedness):
  if stay == 'long' and crowdedness == 'not busy':
    return True
  elif stay == 'long' and crowdedness == 'not busy':
    return True
  return False

def add_rules(dataframe):
  for index, row in dataframe.iterrows():
    t.append(touristic(row['pricerange'],row['foodquality']))
    c.append(child_friendly(row['lengthofstay']))
    r.append(romantic(row['lengthofstay'],row['foodquality'],row['pricerange'],row['crowdedness']))
  dataframe =dataframe.assign(touristic=t,child_friendly=c,romantic=r)
  return(dataframe)

if __name__ == "__main__":
    # Add consequents and add to restaurant_data csv file
    df = pd.read_csv("./data/restaurant_data.csv")
    df_add = add_rules(df)
    df_add.to_csv("./data/restaurantdata.csv")

# Check if there are romantic restaurants
# df_add.loc[(df_add['romantic'] == True)]
