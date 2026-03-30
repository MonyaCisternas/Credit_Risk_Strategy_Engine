import numpy as np

def calculate_score(pd):
    # Clamp PD to avoid infinity
    pd = max(min(pd, 0.999), 0.001)

    odds = (1 - pd) / pd
    score = 600 + 50 * np.log(odds)

    # Bound final score (industry style)
    score = max(300, min(850, score))

    return int(score)
