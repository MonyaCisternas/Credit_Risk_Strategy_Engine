import pandas as pd

def load_data(path = "Data/raw/give_me_some_credit.csv"):
    df = pd.read_csv(path, index_col = 0)
    return df

if __name__ == "__main__":
    df = load_data()
    print(df.head())
