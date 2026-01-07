import pandas as pd
import numpy as np

import os

data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'clustered_data.csv'))
data = pd.read_csv(data_path)


def recommend_portfolio(risk_level, cluster):
    portfolios = {
        1: {"Equity": 0.1, "Debt": 0.6, "Gold": 0.1, "Crypto": 0.0, "Cash": 0.2},
        2: {"Equity": 0.4, "Debt": 0.4, "Gold": 0.1, "Crypto": 0.05, "Cash": 0.05},
        3: {"Equity": 0.7, "Debt": 0.2, "Gold": 0.05, "Crypto": 0.05, "Cash": 0.0},
    }

    adjustments = {
        0: {"Crypto": +0.02, "Cash": -0.02},  
        1: {"Debt": +0.05, "Equity": -0.05},   
        2: {"Gold": +0.03, "Cash": -0.03},     
    }

    base = portfolios.get(risk_level, portfolios[2]).copy()
    if cluster in adjustments:
        for k, v in adjustments[cluster].items():
            base[k] += v

    total = sum(base.values())
    for k in base:
        base[k] = round(base[k] / total, 2)

    return base
