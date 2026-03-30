def assign_risk_bucket(pd):
    if pd >= 0.3:
        return "High Risk"
    elif pd >= 0.12:
        return "Medium Risk"
    else:
        return "Low Risk"
