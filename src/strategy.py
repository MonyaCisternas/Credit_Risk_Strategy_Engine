def assign_strategy(row):
    pd = row["PD"]

    if pd >= 0.30:
        return "Decline / Collections"
    elif pd >= 0.20:
        return "Approve with High Interest + Low Limit"
    elif pd < 0.05:
        return "Premium - Increase Limit + Lower Interest"
    else:
        return "Low Risk - Standard Terms"

def generate_recommendations(row):
    recs = []

    # Credit Utilization
    if row["RevolvingUtilizationOfUnsecuredLines"] > 0.7:
        recs.append("Reduce credit utilization below 50%")

    # Late Payments
    if row.get("TotalLatePayments", 0) > 0:
        recs.append("Avoid late payments for at least 3–6 months")

    # Debt Ratio
    if row["DebtRatio"] > 0.6:
        recs.append("Reduce debt ratio below 40%")

    # Income
    if row["MonthlyIncome"] < 4000:
        recs.append("Increase stable monthly income")

    # Age (thin file proxy)
    if row["age"] < 30:
        recs.append("Build longer credit history (keep accounts active)")

    # Open lines (optional)
    if row.get("NumberOfOpenCreditLinesAndLoans", 0) < 3:
        recs.append("Maintain at least 3 active credit lines")

    return recs
