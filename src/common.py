import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import statsmodels.tsa.api as tsa

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf


def Stationarity_analysis(y, yDesc):
    print(
        "===================================================================================================="
    )
    print("\nSTATIONARITY ANALYSIS FOR: {}".format(yDesc))

    print(
        "\n------------------------------- (1) Time series plot -------------------------------"
    )

    plt.rcParams["figure.figsize"] = (6, 3)
    plt.plot(y)
    plt.title(f"Time Series with {yDesc}")
    plt.ylabel("Variable")
    plt.xlabel("Date")
    plt.show()

    print(
        "------------------------------- (2) ACF and PACF plot -------------------------------"
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=80)
    plot_acf(y, ax=ax1, lags=50)
    plot_pacf(y, method="ywm", ax=ax2, lags=20)

    ax1.spines["top"].set_alpha(0.3)
    ax2.spines["top"].set_alpha(0.3)
    ax1.spines["bottom"].set_alpha(0.3)
    ax2.spines["bottom"].set_alpha(0.3)
    ax1.spines["right"].set_alpha(0.3)
    ax2.spines["right"].set_alpha(0.3)
    ax1.spines["left"].set_alpha(0.3)
    ax2.spines["left"].set_alpha(0.3)

    ax1.tick_params(axis="both", labelsize=12)
    ax2.tick_params(axis="both", labelsize=12)
    plt.show()

    print(
        "------------------------------- (3) ADF test -------------------------------"
    )
    print("ADF test - various underlying models:")
    res = tsa.adfuller(y, regression="c", autolag="AIC")
    print("    (1) ADF p-value (const):", res[1])
    res = tsa.adfuller(y, regression="ct", autolag="AIC")
    print("    (2) ADF p-value (const + trend):", res[1])
    res = tsa.adfuller(y, regression="ctt", autolag="AIC")
    print("    (3) ADF p-value (const + trend + quad trend):", res[1])
    res = tsa.adfuller(y, regression="n", autolag="AIC")
    print("    (4) ADF p-value (no const + no trend):", res[1])
    print(
        "===================================================================================================="
    )
