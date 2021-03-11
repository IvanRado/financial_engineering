import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def random_ex():
    x0 = np.random.randn(1000)
    plt.plot(x0)
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_pacf(x0, ax=ax)
    plt.show()

    x1 = [0]
    for i in range(1000):
        x = 0.5 * x1[-1] + 0.1 * np.random.randn()
        x1.append(x)
    x1 = np.array(x1)
    plt.plot(x1)
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_pacf(x1, ax=ax)
    plt.show()

    x1 = [0]
    for i in range(1000):
        x = -0.5 * x1[-1] + 0.1 * np.random.randn()
        x1.append(x)
    x1 = np.array(x1)
    plt.plot(x1)
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_pacf(x1, ax=ax)
    plt.show()

    x2 = [0, 0]
    for i in range(1000):
        x = 0.5 * x2[-1] - 0.3 * x2[-2] + 0.1 * np.random.randn()
        x2.append(x)
    x2 = np.array(x2)
    plt.plot(x2)
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_pacf(x2, ax=ax)
    plt.show()

    x5 = [0, 0, 0, 0, 0]
    for i in range(1000):
        x = 0.5 * x5[-1] - 0.3 * x5[-2] - 0.6 * x5[-5] + 0.1 * np.random.randn()
        x5.append(x)
    x5 = np.array(x5)
    plt.plot(x5)
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_pacf(x5, ax=ax)
    plt.show()

    # IID Noise
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_acf(np.random.randn(1000), ax=ax)
    plt.show()

    # First order
    errors = 0.1 * np.random.randn(1000)
    ma1 = []
    for i in range(1000):
        if i >= 1:
            x = 0.5 * errors[i-1] + errors[i]
        else:
            x = errors[i]
        ma1.append(x)
    ma1 = np.array(ma1)
    plt.plot(ma1)
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_acf(ma1, ax=ax)
    plt.show()

    # Second order
    errors = 0.1 * np.random.randn(1000)
    ma2 = []
    for i in range(1000):
        x = 0.5 * errors[i - 1] - 0.3 * errors[i-2] + errors[i]
        ma2.append(x)
    ma2 = np.array(ma2)
    plt.plot(ma2)
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_acf(ma2, ax=ax)
    plt.show()

    # Third order
    errors = 0.1 * np.random.randn(1000)
    ma3 = []
    for i in range(1000):
        x = 0.5 * errors[i - 1] - 0.3 * errors[i - 2] + 0.7 * errors[i-3] + errors[i]
        ma3.append(x)
    ma3 = np.array(ma3)
    plt.plot(ma3)
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_acf(ma3, ax=ax)
    plt.show()

    # Sixth order
    errors = 0.1 * np.random.randn(1000)
    ma6 = []
    for i in range(1000):
        x = 0.5 * errors[i - 1] - 0.3 * errors[i - 2] + 0.7 * errors[i - 3] + \
            0.2 * errors[i - 4] - 0.8 * errors[i - 5] - 0.9 * errors[i - 6] +  errors[i]
        ma6.append(x)
    ma6 = np.array(ma6)
    plt.plot(ma6)
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_acf(ma6, ax=ax)
    plt.show()

    return


def stock_ex():
    df = pd.read_csv('sp500sub.csv', index_col='Date', parse_dates=True)
    goog = df[df['Name'] == 'GOOG'][['Close']].copy()
    goog['LogRet'] = np.log(goog).diff()

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_pacf(goog['LogRet'].dropna(), ax=ax)
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_acf(goog['LogRet'].dropna(), ax=ax)
    plt.show()

    # Repeat with Apple
    aapl = df[df['Name'] == 'AAPL'][['Close']].copy()
    aapl['LogRet'] = np.log(aapl).diff()

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_pacf(aapl['LogRet'].dropna(), ax=ax)
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_acf(aapl['LogRet'].dropna(), ax=ax)
    plt.show()

    # Repeat with IBM
    ibm = df[df['Name'] == 'IBM'][['Close']].copy()
    ibm['LogRet'] = np.log(ibm).diff()

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_pacf(ibm['LogRet'].dropna(), ax=ax)
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_acf(ibm['LogRet'].dropna(), ax=ax)
    plt.show()

    # Repeat with Starbucks
    sbux = df[df['Name'] == 'SBUX'][['Close']].copy()
    sbux['LogRet'] = np.log(sbux).diff()

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_pacf(sbux['LogRet'].dropna(), ax=ax)
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_acf(sbux['LogRet'].dropna(), ax=ax)
    plt.show()


if __name__ == "__main__":
    stock_ex()