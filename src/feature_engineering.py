import pandas as pd
import numpy as np

def engineer_features(df):
    df["DebtBurden"] = df["DebtRatio"] * df["MonthlyIncome"]
    df["IncomePerDependent"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)
    df["TotalLatePayments"] = df["NumberOfTime30-59DaysPastDueNotWorse"] + df["NumberOfTime60-89DaysPastDueNotWorse"] + df["NumberOfTimes90DaysLate"]
    df["CreditLinesPerAge"] = df["NumberOfOpenCreditLinesAndLoans"] / (df["age"] + 1)
    df["HighUtilization"] = (df["RevolvingUtilizationOfUnsecuredLines"] > 0.8).astype(int)
    df["LogIncome"] = np.log1p(df["MonthlyIncome"])
    df["LogDebtBurden"] = np.log1p(df["DebtBurden"])
    df["Utilization_x_Late"] = (
        df["RevolvingUtilizationOfUnsecuredLines"] * df["TotalLatePayments"]
    )
    if "NumberOfDependents" not in df.columns:
        df["NumberOfDependents"] = 0
    return df


if __name__ == "__main__":
    from src.data_loader import load_data
    from src.data_cleaning import clean_data
    df = load_data()
    df = clean_data(df)
    df = engineer_features(df)
    print(df.head())
