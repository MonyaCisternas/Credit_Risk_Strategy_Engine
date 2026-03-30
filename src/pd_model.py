import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

def train_pd_model(df):

    feature_cols = [
        "RevolvingUtilizationOfUnsecuredLines",
        "DebtRatio",
        "MonthlyIncome",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberOfTimes90DaysLate",
        "TotalLatePayments",
        "age",
        "NumberOfDependents",
        "LogIncome",
        "LogDebtBurden",
        "HighUtilization"
    ]

    if "SeriousDlqin2yrs" not in df.columns:
        raise ValueError("Target column missing")

    df = df.copy()

    # Clean features
    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    y = df["SeriousDlqin2yrs"].fillna(0)

    # Ensure binary target
    y = y[y.isin([0, 1])]
    X = X.loc[y.index]

    print("Target distribution:")
    print(y.value_counts())

    if y.nunique() < 2:
        raise ValueError("Target variable has only one class")

    # 🔥 Gradient Boosting Model
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    model.fit(X, y)

    return model, feature_cols


def predict_pd(model, df, feature_cols):
    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    pd_values = model.predict_proba(X)[:, 1]

    # Clip for stability
    pd_values = np.clip(pd_values, 0.001, 0.999)

    return pd_values

def explain_prediction(model, X, feature_names, top_n=5):
    """
    Explain prediction using feature importance (tree-based models)
    """
    import pandas as pd
    import numpy as np

    # Get feature importance
    importances = model.feature_importances_

    # Get row values
    row = X.iloc[0]

    df = pd.DataFrame({
        "feature": feature_names,
        "value": row.values,
        "importance": importances
    })

    # Estimate directional contribution
    df["contribution"] = df["value"] * df["importance"]

    # Sort by impact
    df["abs_contribution"] = np.abs(df["contribution"])
    df = df.sort_values(by="abs_contribution", ascending=False)

    return df.head(top_n)
