import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

df = pd.read_csv('airline_passengers.csv')
df['LogPassengers'] = np.log(df['Passengers'])
df['Diff'] = df['Passengers'].diff()
df['LogDiff'] = df['LogPassengers'].diff()


def adf(x):
    res = adfuller(x)
    print(f"Test-Statistic: {res[0]}")
    print(f"P-value: {res[1]}")
    if res[1] < 0.05:
        print("Stationary")
    else:
        print("Non-Stationary")


print(f"Passengers: {adf(df['Passengers'])}")
print(f"Log Passengers: {adf(df['LogPassengers'])}")
print(f"Passengers Diff: {adf(df['Diff'].dropna())}")
print(f"Log Passengers Diff: {adf(df['LogDiff'].dropna())}")
print(f"Gamma dist: {adf(np.random.gamma(1, 1, 100))}")

df['Diff'].plot()
plt.title("Difference from first")
plt.show()

df['LogDiff'].plot()
plt.title("Log difference from first")
plt.show()

# With Google stock
stocks = pd.read_csv('sp500sub.csv', index_col='Date', parse_dates=True)
goog = stocks[stocks['Name'] == 'GOOG'][['Close']]
goog['LogPrice'] = np.log(goog['Close'])
goog['LogRet'] = goog['LogPrice'].diff()

goog['LogPrice'].plot()
plt.title('Log of Google Prices')
plt.show()

goog['LogRet'].plot()
plt.title('Log of Google Returns')
plt.show()

print(f"Google Log Price: {adf(goog['LogPrice'].dropna())}")
print(f"Google Log Returns: {adf(goog['LogRet'].dropna())}")

# With Starbucks stock
sbux = stocks[stocks['Name'] == 'SBUX'][['Close']]
sbux['LogPrice'] = np.log(sbux['Close'])
sbux['LogRet'] = sbux['LogPrice'].diff()

sbux['LogPrice'].plot()
plt.title('Log of Starbucks Prices')
plt.show()

sbux['LogRet'].plot()
plt.title('Log of Starbucks Returns')
plt.show()

print(f"Starbucks Log Price: {adf(sbux['LogPrice'].dropna())}")
print(f"Starbucks Log Returns: {adf(sbux['LogRet'].dropna())}")