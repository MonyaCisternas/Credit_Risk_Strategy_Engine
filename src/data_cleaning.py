import pandas as pd

def clean_data(df):
    df = df.loc[:, ~df.columns.str.contains("^Unnamed: 0")]
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(df["NumberOfDependents"].median())
    df["MissingIncome"] = df["MonthlyIncome"].isnull().astype(int)
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
    return df

if __name__ == "__main__":
    from data_loader import load_data
    df = load_data()
    df = clean_data(df)
    print(df.head())
    print(df.isnull().sum())
