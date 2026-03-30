def calculate_expected_loss(row):
    PD = row["PD"]

    # Proxy for exposure
    EAD = row["NumberOfOpenCreditLinesAndLoans"] * 1000

    # LGD based on utilization
    if row["RevolvingUtilizationOfUnsecuredLines"] > 0.8:
        LGD = 0.7
    elif row["RevolvingUtilizationOfUnsecuredLines"] > 0.5:
        LGD = 0.6
    else:
        LGD = 0.5

    return PD * LGD * EAD
