import sys
import os

sys.path.append(os.path.abspath("."))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from src.feature_engineering import engineer_features
from src.strategy import assign_strategy, generate_recommendations
from src.risk import assign_risk_bucket
from src.pd_model import train_pd_model, predict_pd, explain_prediction
from src.loss import calculate_expected_loss
from src.scorecard import calculate_score

@st.cache_data
def load_data():
    return pd.read_csv("Data/processed.csv")

def load_assets():
    pd_model = joblib.load("models/pd_model.pkl")
    pd_features = joblib.load("models/pd_features.pkl")
    kmeans = joblib.load("models/kmeans.pkl")
    scaler = joblib.load("models/scaler.pkl")
    feature_cols = joblib.load("models/feature_cols.pkl")
    return pd_model, pd_features, kmeans, scaler, feature_cols

st.set_page_config(page_title = "Customer Risk Segmentation & Strategy Engine", layout = "wide")
st.title("Customer Risk & Strategy Engine")
st.info("""
**How to use this tool:**
1. Explore portfolio risk in *Overview*
2. Analyze segments in *Segment Insights*
3. Evaluate individual customers
4. Test new applicants in *Credit Decision Tool*
5. Use the simulator to reduce risk
""")

df = load_data()
pd_model, pd_features, kmeans, scaler, feature_cols = load_assets()

st.sidebar.markdown("Credit Risk App")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["Overview", "Segment Insights", "Strategy Distribution", "Customer Explorer", "Credit Decision Tool"])
st.sidebar.markdown("---")
st.sidebar.write("Built by Monya Cisternas")

if page == "Overview":
    st.subheader("Dataset Overview")
    st.write(df.head())
    st.write("Target Distribution:")
    st.write(df["SeriousDlqin2yrs"].value_counts())
    st.subheader("Portfolio Metrics")
    approval_rate = (df["Strategy"] != "Decline / Collections").mean()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Approval Rate", f"{approval_rate:.2f}")
    with col2:
        st.metric("Total Expected Loss (R)", f"{df["EL"].sum():,.0f}")
    with col3:
        st.metric("Average PD", f"{df["PD"].mean():.2%}")
    st.write("High Risk %:", (df["PD"] > 0.25).mean())
    st.write("Decline %:", (df["Strategy"] == "Decline / Collections").mean())
    high_risk = df[df["PD"] > 0.25]
    potential_loss_reduction = high_risk["EL"].sum()
    st.markdown("---")
    st.metric(
        "Potential Loss Reduction (High-Risk Rejection)",
        f"R{potential_loss_reduction:,.0f}"
    )
    st.subheader("PD Distribution")
    fig, ax = plt.subplots()
    ax.hist(df["PD"], bins = 30)
    st.pyplot(fig)

elif page == "Segment Insights":
    st.subheader("Segment Insights")
    cluster_summary = df.groupby("Cluster").agg({
        "PD": "mean",
        "EL": "mean",
        "RevolvingUtilizationOfUnsecuredLines": "mean",
        "DebtRatio": "mean",
        "TotalLatePayments": "mean",
        "MonthlyIncome": "mean",
        "age": "mean"
    }).reset_index()
    st.dataframe(cluster_summary)
    st.subheader("Average PD by Segment")
    fig, ax = plt.subplots()
    cluster_summary.plot(x = "Cluster", y = "PD", kind = "bar", ax = ax)
    st.pyplot(fig)

elif page == "Strategy Distribution":
    st.subheader("Strategy Breakdown")
    strategy_counts = df["Strategy"].value_counts()
    fig, ax = plt.subplots()
    strategy_counts.plot(kind = "bar", ax = ax)
    st.pyplot(fig)

elif page == "Customer Explorer":
    st.subheader("Explore Individual Customer")
    idx = st.number_input("Enter Customer Index", min_value = 0, max_value = len(df) - 1, value = 0)
    customer = df.iloc[int(idx)]
    st.subheader("Customer Profile")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Financials**")
        st.write({"Income (R)": f"{customer['MonthlyIncome']:.0f}", "Debt Burden (R)": f"{customer['DebtBurden']:.2f}"})
    with col2:
        st.write("**Behaviour**")
        st.write({"Late Payments": int(customer["TotalLatePayments"]), "Utilization": customer["RevolvingUtilizationOfUnsecuredLines"]})
    st.subheader("Decision")
    st.write(f"Cluster: {customer['Cluster']}")
    st.write(f"Risk Level: {customer['RiskBucket']}")
    st.write(f"PD: {customer['PD']:.2%}")
    st.write(f"Credit Score: {customer['Score']}")
    st.write(f"Expected Loss (R): {customer['EL']:.2f}")
    st.success(customer["Strategy"])

elif page == "Credit Decision Tool":
    st.subheader("Evaluate New Customer")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 100, 30)
        income = st.number_input("Monthly Income (R)", 0, 100000, 35000)
        debt_ratio = st.number_input("Debt Ratio", 0.0, 5.0, 0.68)
        utilization = st.number_input("Credit Utilization", 0.0, 1.5, 0.75)
        dependents = st.number_input("Number of Dependents", 0, 10, 2)
    with col2:
        late_30 = st.number_input("30-59 Days Late", 0, 50, 2)
        late_60 = st.number_input("60-89 Days Late", 0, 50, 1)
        late_90 = st.number_input("90 Days Late", 0, 50, 0)
        open_lines = st.number_input("Open Credit Lines", 0, 50, 8)
        real_estate = st.number_input("Real Estate Loans", 0, 20, 1)
        
    if st.button("Evaluate Customer"):
        st.session_state["customer_data"] = pd.DataFrame([{
            "RevolvingUtilizationOfUnsecuredLines": utilization,
            "DebtRatio": debt_ratio,
            "NumberOfTimes90DaysLate": late_90,
            "NumberOfTime30-59DaysPastDueNotWorse": late_30,
            "NumberOfTime60-89DaysPastDueNotWorse": late_60,
            "age": age,
            "MonthlyIncome": income,
            "NumberOfOpenCreditLinesAndLoans": open_lines,
            "NumberRealEstateLoansOrLines": real_estate,
            "NumberOfDependents": dependents
        }])
        
    if "customer_data" in st.session_state:
        new_customer = st.session_state["customer_data"].copy()
        new_customer = engineer_features(new_customer)
        new_customer["PD"] = predict_pd(pd_model, new_customer, pd_features)
        new_customer["RiskBucket"] = new_customer["PD"].apply(assign_risk_bucket)
        new_customer["Score"] = new_customer["PD"].apply(calculate_score)
        new_scaled = scaler.transform(new_customer[feature_cols])
        new_customer["Cluster"] = kmeans.predict(new_scaled)[0]
        new_customer["Strategy"] = new_customer.apply(assign_strategy, axis = 1)
        new_customer["EL"] = new_customer.apply(calculate_expected_loss, axis = 1)
        row = new_customer.iloc[0]
        decision = row["Strategy"]
        sim_customer = new_customer.copy()
        recommendations = generate_recommendations(row)
        explanation = explain_prediction(pd_model, new_customer[pd_features], pd_features)
        feature_group_map = {
            "LogIncome": "MonthlyIncome",
            "IncomePerDependent": "MonthlyIncome",
            "LogDebt": "DebtRatio",
            "DebtPerDependent": "DebtRatio"
        }
        explanation["base_feature"] = explanation["feature"].replace(feature_group_map)
        explanation = (
            explanation
            .sort_values("abs_contribution", ascending=False)
            .drop_duplicates("base_feature")
        )
        max_contrib = explanation["abs_contribution"].max()

        if max_contrib > 0:
            explanation["impact_score"] = explanation["abs_contribution"] / max_contrib
        else:
            explanation["impact_score"] = 0

        def get_impact_label(score):
            if score > 0.66:
                return "High"
            elif score > 0.33:
                return "Medium"
            else:
                return "Low"

        explanation["impact_level"] = explanation["impact_score"].apply(get_impact_label)
                                                      
        def display_decision(row, decision):
            risk = row["RiskBucket"]
            pd_val = row["PD"]
            el = row["EL"]
            score = row["Score"]
            if "Decline" in decision:
                st.error("REJECT")
            elif "High Interest" in decision:
                st.warning("REVIEW / HIGH RISK")
            elif "Premium" in decision:
                st.success("APPROVED")
            else:
                st.info("CONDITIONAL")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Risk Level", risk)
            with col2:
                st.metric("PD", f"{pd_val:.2%}")
            with col3:
                st.metric("Expected Loss (R)", f"{el:,.2f}")
            with col4:
                st.metric("Score", f"{score:.0f}")

            st.markdown("---")
            st.markdown("### Recommended Strategy")
            st.info(decision)

            if "Decline" in decision:
                st.error("This customer has a high likelihood of default and may result in financial loss.")
            elif "High Interest" in decision:
                st.warning("This customer presents moderate risk. Consider adjusting terms or reviewing manually.")
            else:
                st.success("This customer is low risk and suitable for approval.")

        st.subheader("Decision")
        display_decision(row, decision)
        st.markdown("---")
        
        st.subheader("Top Risk Drivers")
        feature_names_map = {
            "RevolvingUtilizationOfUnsecuredLines": "Credit Utilization",
            "DebtRatio": "Debt Ratio",
            "MonthlyIncome": "Income",
            "age": "Age",
            "TotalLatePayments": "Late Payments",
            "LogDebtBurden": "Debt Burden",
            "NumberOfOpenCreditLinesAndLoans": "Amount of credit lines"
        }

        for _, row_exp in explanation.iterrows():
            feature = row_exp["feature"]
            value = row_exp["value"]
            impact = row_exp["impact_level"]
            name = feature_names_map.get(feature, feature)

            if feature == "RevolvingUtilizationOfUnsecuredLines":
                is_risky = value > 0.7
            elif feature == "DebtRatio":
                is_risky = value > 0.6
            elif feature == "MonthlyIncome":
                is_risky = value < 7000
            elif feature == "TotalLatePayments":
                is_risky = value > 0
            elif feature == "age":
                is_risky = value < 30
            else:
                is_risky = True

            if is_risky:
                st.error(f"{name} increases risk ({impact} Impact)")
            else:
                st.success(f"{name} reduces risk ({impact} Impact)")

        st.subheader("Improvement Recommendations")
        if len(recommendations) == 0:
            st.success("Customer profile is already strong")
        else:
            for rec in recommendations:
                st.write(f"* {rec}")


        st.subheader("How Customer Can Improve Risk Level")
        st.caption("Ajust the variables below to see how the customer's risk profile improves.")
        col1, col2 = st.columns(2)
        with col1:
            sim_util = st.slider("Simulate Credit Utilization", 0.0, 1.0, float(row["RevolvingUtilizationOfUnsecuredLines"]))
            sim_debt = st.slider("Simulate Debt Ratio", 0.0, 2.0, float(row["DebtRatio"]))
            sim_income = st.slider("Simulate Monthly Income (R)", 0, 50000, int(row["MonthlyIncome"]))
        with col2:
            sim_30 = st.slider("30-59 Days Late", 0, 10, int(row["NumberOfTime30-59DaysPastDueNotWorse"]))
            sim_60 = st.slider("60-89 Days Late", 0, 10, int(row["NumberOfTime60-89DaysPastDueNotWorse"]))
            sim_90 = st.slider("90+ Days Late", 0, 10, int(row["NumberOfTimes90DaysLate"]))
        sim_customer["RevolvingUtilizationOfUnsecuredLines"] = sim_util
        sim_customer["DebtRatio"] = sim_debt
        sim_customer["MonthlyIncome"] = sim_income
        sim_customer["NumberOfTime30-59DaysPastDueNotWorse"] = sim_30
        sim_customer["NumberOfTime60-89DaysPastDueNotWorse"] = sim_60
        sim_customer["NumberOfTimes90DaysLate"] = sim_90
        sim_customer = engineer_features(sim_customer)
        sim_customer["PD"] = predict_pd(pd_model, sim_customer, pd_features)
        sim_customer["RiskBucket"] = sim_customer["PD"].apply(assign_risk_bucket)
        sim_customer["Strategy"] = sim_customer.apply(assign_strategy, axis = 1)
        sim_customer["EL"] = sim_customer.apply(calculate_expected_loss, axis = 1)
        sim_row = sim_customer.iloc[0]
        sim_decision = sim_row["Strategy"]
        
        st.write("### Simulation Results")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Current Profile")
            st.metric("PD", f"{row['PD']:.2%}")
            st.metric("Expected Loss (R)", f"{row['EL']:,.2f}")
            st.write(f"Strategy: {decision}")
        with col2:
            st.write("### Improved Profile")
            st.metric("PD", f"{sim_row['PD']:.2%}")
            st.metric("Expected Loss (R)", f"{sim_row['EL']:,.2f}")
            st.write(f"Strategy: {sim_decision}")
            
        st.markdown("---")
        st.markdown("### Impact Summary")
        delta_pd = sim_row["PD"] - row["PD"]
        delta_el = sim_row["EL"] - row["EL"]
        if delta_pd < 0:
            st.success(f"Risk reduced by {abs(delta_pd):.2%}")
        elif delta_pd > 0:
            st.error(f"Risk increased by {abs(delta_pd):.2%}")
        else:
            st.info("Risk remains unchanged")

        if delta_el < 0:
            st.success(f"Expected loss reduced by R{abs(delta_el):,.2f}")
        elif delta_el > 0:
            st.error(f"Expected loss increased by R{abs(delta_el):,.2f}")
        else:
            st.info("Expected loss remains unchanged")

    if st.button("Reset Customer"):
        st.session_state.clear()
        st.rerun()
