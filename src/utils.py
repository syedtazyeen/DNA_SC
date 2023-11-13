import pandas as pd
from sklearn.model_selection import train_test_split

def load_txt(file,x_name ,y_name):
  data = pd.read_table(file)
  return data[x_name], data[y_name]


def split_data(x, y):
  return train_test_split(x, y, test_size=0.2, random_state=42)


def get_dataframe(x,y):
  return pd.DataFrame({
    'sequence': x,
    'class' : y
  })
