from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def segment_customers(df):
    feature_cols = [
        "RevolvingUtilizationOfUnsecuredLines",
        "DebtRatio",
        "TotalLatePayments",
        "NumberOfTimes90DaysLate",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse"
    ]
    df[feature_cols] = df[feature_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    kmeans = KMeans(n_clusters = 3, random_state = 42, n_init = 10)
    df["Cluster"] = kmeans.fit_predict(X_scaled)
    return df, kmeans, scaler, feature_cols
