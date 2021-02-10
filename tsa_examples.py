# Relevant imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def rmse(y, t):
    return np.sqrt(np.mean(y - t)**2)


def mae(y, t):
    return np.mean(np.abs(y - t))


def sma():
    close = pd.read_csv('sp500_close.csv', index_col=0, parse_dates=True)
    goog = close[['GOOG']].copy().dropna()

    goog.plot()
    plt.show()

    goog_ret = np.log(goog.pct_change(1) + 1)
    goog_ret.plot()
    plt.show()

    goog['SMA-10'] = goog['GOOG'].rolling(10).mean()
    goog.plot(figsize=(10, 5))
    plt.show()

    goog['SMA-50'] = goog['GOOG'].rolling(50).mean()
    goog.plot(figsize=(10, 5))
    plt.show()

    goog_aapl = close[['GOOG', 'AAPL']].copy().dropna()
    goog_aapl_ret = np.log(1 + goog_aapl.pct_change(1))
    goog_aapl_ret['GOOG-SMA-50'] = goog_aapl_ret['GOOG'].rolling(50).mean()
    goog_aapl_ret['AAPL-SMA-50'] = goog_aapl_ret['AAPL'].rolling(50).mean()

    goog_aapl_ret.plot(figsize=(10, 5))
    plt.show()


def ewma():
    df = pd.read_csv('airline_passengers.csv', index_col='Month', parse_dates=True)
    df.plot()
    plt.show()

    alpha = 0.2
    df['EWMA'] = df['Passengers'].ewm(alpha=alpha, adjust=False).mean()
    df.plot()
    plt.show()

    manual_ewma = []
    for x in df['Passengers'].to_numpy():
        if len(manual_ewma) > 0:
            xhat = alpha * x + (1 - alpha) * manual_ewma[-1]
        else:
            xhat = x
    df['Manual'] = manual_ewma
    df.plot()
    plt.show()


def simple_exponential_smoothing(alpha=0.2):
    df = pd.read_csv('airline_passengers.csv', index_col='Month', parse_dates=True)
    df['EWMA'] = df['Passengers'].ewm(alpha=alpha, adjust=False).mean()
    df.index.freq = 'MS'

    ses = SimpleExpSmoothing(df['Passengers'])
    res = ses.fit(smoothing_level=alpha, optimized=False)
    print(res.predict(start=df.index[0], end=df.index[-1]))

    df['SES'] = res.predict(start=df.index[0], end=df.index[-1])
    df.plot()
    plt.show()

    N_test = 12
    train = df.iloc[:-N_test]
    test = df.iloc[-N_test:]

    ses = SimpleExpSmoothing(train['Passengers'])
    res = ses.fit()

    # Boolean series to index df rows
    train_idx = df.index <= train.index[-1]
    test_idx = df.index > train.index[-1]

    df.loc[train_idx, 'SESfitted'] = res.fittedvalues
    df.loc[test_idx, 'SESfitted'] = res.forecast(N_test)
    df[['Passengers', 'SESfitted']].plot()
    plt.show()

    print(f"The Model parameters: {res.params}")


def holt(alpha=0.2):
    df = pd.read_csv('airline_passengers.csv', index_col='Month', parse_dates=True)
    df['EWMA'] = df['Passengers'].ewm(alpha=alpha, adjust=False).mean()
    df.index.freq = 'MS'

    N_test = 12
    train = df.iloc[:-N_test]
    test = df.iloc[-N_test:]

    train_idx = df.index <= train.index[-1]
    test_idx = df.index > train.index[-1]

    holt = Holt(df['Passengers'])
    res_h = holt.fit()
    df['Holt'] = res_h.fittedvalues
    df[['Passengers', 'Holt']].plot()
    plt.show()

    holt = Holt(train['Passengers'])
    res_h = holt.fit()
    df.loc[train_idx, 'Holt'] = res_h.fittedvalues
    df.loc[test_idx, 'Holt'] = res_h.forecast(N_test)

    df[['Passengers', 'Holt']].plot()
    plt.show()


def holt_winters(alpha=0.2):
    df = pd.read_csv('airline_passengers.csv', index_col='Month', parse_dates=True)
    df['EWMA'] = df['Passengers'].ewm(alpha=alpha, adjust=False).mean()
    df.index.freq = 'MS'

    N_test = 12
    train = df.iloc[:-N_test]
    test = df.iloc[-N_test:]

    train_idx = df.index <= train.index[-1]
    test_idx = df.index > train.index[-1]

    hw = ExponentialSmoothing(train['Passengers'], trend='add', seasonal='add', seasonal_periods=12)
    res_hw = hw.fit()

    df.loc[train_idx, 'HoltWinters'] = res_hw.fittedvalues
    df.loc[test_idx, 'HoltWinters'] = res_hw.forecast(N_test)
    df[['Passengers', 'HoltWinters']].plot()
    plt.show()

    print(f"Train RMSE: {rmse(train['Passengers'], res_hw.fittedvalues)}")
    print(f"Test RMSE: {rmse(test['Passengers'], res_hw.forecast(N_test))}")

    print(f"Train MAE: {mae(train['Passengers'], res_hw.fittedvalues)}")
    print(f"Test MAE: {mae(test['Passengers'], res_hw.forecast(N_test))}")

    hw = ExponentialSmoothing(train['Passengers'], trend='add', seasonal='mul', seasonal_periods=12)
    res_hw = hw.fit()

    df.loc[train_idx, 'HoltWinters'] = res_hw.fittedvalues
    df.loc[test_idx, 'HoltWinters'] = res_hw.forecast(N_test)
    df[['Passengers', 'HoltWinters']].plot()
    plt.show()

    print(f"Train RMSE: {rmse(train['Passengers'], res_hw.fittedvalues)}")
    print(f"Test RMSE: {rmse(test['Passengers'], res_hw.forecast(N_test))}")

    print(f"Train MAE: {mae(train['Passengers'], res_hw.fittedvalues)}")
    print(f"Test MAE: {mae(test['Passengers'], res_hw.forecast(N_test))}")

    hw = ExponentialSmoothing(train['Passengers'], trend='mul', seasonal='mul', seasonal_periods=12)
    res_hw = hw.fit()

    df.loc[train_idx, 'HoltWinters'] = res_hw.fittedvalues
    df.loc[test_idx, 'HoltWinters'] = res_hw.forecast(N_test)
    df[['Passengers', 'HoltWinters']].plot()
    plt.show()

    print(f"Train RMSE: {rmse(train['Passengers'], res_hw.fittedvalues)}")
    print(f"Test RMSE: {rmse(test['Passengers'], res_hw.forecast(N_test))}")

    print(f"Train MAE: {mae(train['Passengers'], res_hw.fittedvalues)}")
    print(f"Test MAE: {mae(test['Passengers'], res_hw.forecast(N_test))}")


if __name__ == "__main__":
    sma()
