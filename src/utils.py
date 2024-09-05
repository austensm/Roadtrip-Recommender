import pandas as pd

# Read in file as panda dataframe
def read_data(filename):
  currentDF = pd.read_csv("data\\" + filename, encoding="utf-16", sep="\t")
  return currentDF

# utility values into 5 categories
    # ‘great’ is equivalent to [0.8, 1.0]
    # ‘good’ is equivalent to [0.6, 0.8)
    # ‘ok’ is equivalent to [0.4, 0.6)
    # ‘low’ is equivalent to [0.2, 0.4)
    # ‘very low’ is equivalent to [0.0, 0.2)
def construct_categories_from_DF(df):
  df["utility"] = pd.cut(df["utility"], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                         include_lowest=True, labels=['very low', 'low', 'ok', 'high', 'great'])
  return df
