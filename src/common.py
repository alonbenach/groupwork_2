import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels
import statsmodels.tsa.api as tsa
import statsmodels.api as sm

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA


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


def test_het(results):
    """
    Testing for heteroskedasticity

    :param results: Univariate time series (panda series).

    :print: Goldfeld-Quandt test stat and p-value (null hypothesis - no heteroskedasticity)
    """
    test_results = results.test_heteroskedasticity(method="breakvar")
    test_results = test_results[0]
    printout = print(
        f"""
        -------------------------------------------------------------------
        Testing for heteroskedasticity in time series: Goldfeld-Quandt test
            H0: no heteroskedasticity
            Goldfeld-Quandt test stat:{test_results[0]}
                              p-value:{test_results[1]}
        -------------------------------------------------------------------
        """
    )

    return printout


def test_norm(results):
    """
    Testing for normality of errors - Jarque Bera test:

    :param results: Univariate time series (panda series).

    :print: Jarque Bera test test stat and p-value (null hypothesis - normal destribution)
    """
    test_results = results.test_normality(method="jarquebera")
    test_results = test_results[0]
    printout = print(
        f"""
        -------------------------------------------------------------------
        Testing normality - Jarque Bera test:
            H0: normal destribution
            Jarque Bera test stat:{test_results[0]}
                              p-value:{test_results[1]}
        -------------------------------------------------------------------
        """
    )

    return printout


def serial_corr(results, lagnum):
    """
    Testing correlation for lags in ARMA model results

    :param results: model results of ARMA model (statsmodels.tsa.arima.model).

    :lagnum: int, number of lags to be used for individual correlation test
    """
    BPres = results.test_serial_correlation("ljungbox", lags=lagnum)
    BPres = BPres[0]

    BPstat = pd.DataFrame(
        {
            "lag": list(range(1, lagnum + 1)),
            "ljungbox stat": BPres[0],
            "p-value": BPres[1],
        }
    )
    printout = print(
        f"""
    ----------------------------------------------------------------------------------------------
    Testing serial correlation - Ljung-Box Q test:
                H0: no serial correlation for a given lag
    {BPstat}
    ----------------------------------------------------------------------------------------------
    """
    )
    return printout


def serial_corr_all_lags(results, lagnum):
    """
    Testing serial correlation for all lags up to certain lag together in ARMA model results

    :param results: model results of ARMA model (statsmodels.tsa.arima.model).

    :lagnum: int, number of lags to be used for individual correlation test
    """
    df = sm.stats.acorr_ljungbox(
        results.resid, lags=[lagnum], return_df=False, model_df=4, boxpierce=True
    )

    printout = print(
        f"""
    ----------------------------------------------------------------------------------------------
    Testing serial correlation - Ljung-Box Q test:
                H0: no serial correlation all lags up to lag {lagnum} together

            Ljungbox test stat: {df[0][0]}
            p-value: {df[1][0]}

    ----------------------------------------------------------------------------------------------
    """
    )
    return printout
