import os
import joblib
import pandas as pd
import sys

sys.path.append(os.path.abspath("."))

from src.data_loader import load_data
from src.data_cleaning import clean_data
from src.feature_engineering import engineer_features
from src.segmentation import segment_customers
from src.strategy import assign_strategy
from src.risk import assign_risk_bucket
from src.pd_model import train_pd_model, predict_pd
from src.loss import calculate_expected_loss
from src.scorecard import calculate_score

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

df = load_data()
df = clean_data(df)
df = engineer_features(df)
df, kmeans, scaler, feature_cols = segment_customers(df)
pd_model, pd_features = train_pd_model(df)
df["PD"] = predict_pd(pd_model, df, pd_features)
df["RiskBucket"] = df["PD"].apply(assign_risk_bucket)
df["Score"] = df["PD"].apply(calculate_score)
df["Strategy"] = df.apply(assign_strategy, axis=1)
df["EL"] = df.apply(calculate_expected_loss, axis=1)

df.to_csv("Data/processed.csv", index=False)

joblib.dump(pd_model, "models/pd_model.pkl")
joblib.dump(pd_features, "models/pd_features.pkl")
joblib.dump(kmeans, "models/kmeans.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(feature_cols, "models/feature_cols.pkl")
