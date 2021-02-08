import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('all_stocks_5yr.csv', parse_dates=True)

# Returns
sbux = data[data['Name'] == 'SBUX'].copy()
sbux['prev_close'] = sbux['close'].shift(1)

sbux['return'] = sbux['close']/sbux['prev_close'] - 1
sbux['return2'] = sbux['close'].pct_change(1)

sbux['return'].hist(bins=100)
plt.show()

print(f"The mean of the returns: {sbux['return'].mean()}")
print(f"The standard deviation of the returns: {sbux['return'].std()}")

sbux['log_return'] = np.log(sbux['return'] + 1)
sbux['log_return'].hist(bins=100)
plt.show()

print(f"The mean of the log returns: {sbux['return'].mean()}")
print(f"The standard deviation of the log returns: {sbux['return'].std()}")

# Normal Distribution
from scipy.stats import norm

x_list = np.linspace(
    sbux['return'].min(),
    sbux['return'].max(),
    100)

y_list = norm.pdf(x_list, loc=sbux['return'].mean(), scale=sbux['return'].std())

plt.plot(x_list, y_list)
sbux['return'].hist(bins=100, density=True)
plt.show()

from scipy.stats import probplot

probplot(sbux['return'].dropna(), dist='norm', fit=True, plot=plt)
plt.show()

import statsmodels.api as sm

sm.qqplot(sbux['return'].dropna(), line='s')
plt.show()

x_list = np.linspace(
    sbux['log_return'].min(),
    sbux['log_return'].max(),
    100)

y_list = norm.pdf(x_list,
                  loc=sbux['log_return'].mean(),
                  scale=sbux['log_return'].std())

plt.plot(x_list, y_list)
sbux['log_return'].hist(bins=100, density=True)
plt.show()

sm.qqplot(sbux['log_return'].dropna(), line='s')
plt.show()

# t-distribution
from scipy.stats import t

x_list = np.linspace(
    sbux['return'].min(),
    sbux['return'].max(),
    100)

params = t.fit(sbux['return'].dropna())
print(f"t distribution parameters are: {params}")

df, loc, scale = params
y_list = t.pdf(x_list, df, loc, scale)

plt.plot(x_list, y_list)
sbux['return'].hist(bins=100, density=True)


class myt:
    def __init__(self, df):
        self.df = df

    def fit(self, x):
        return t.fit(x)

    def ppf(self, x, loc=0, scale=1):
        return t.ppf(x, self.df, loc, scale)


sm.qqplot(sbux['return'].dropna(), dist=myt(df), line='s')
plt.show()

x_list = np.linspace(
    sbux['log_return'].min(),
    sbux['log_return'].max(),
    100)

params = t.fit(sbux['log_return'].dropna())
df, loc, scale = params
y_list = t.pdf(x_list, df, loc, scale)

plt.plot(x_list, y_list)
sbux['log_return'].hist(bins=100, density=True)
plt.show()

sm.qqplot(sbux['log_return'].dropna(), dist=myt(df), line='s')
plt.show()

# Skewness and Kurtosis
print(f"The skewness of Starbucks returns: {sbux['return'].skew()}")
print(f"The excess kurtosis of Starbucks returns: {sbux['return'].kurtosis()}")

print(f"The skewness of Starbucks log returns: {sbux['log_return'].skew()}")
print(f"The excess kurtosis of Starbucks log returns: {sbux['log_return'].kurtosis()}")

samp = pd.Series(np.random.randn(10000))
print(samp.skew(), samp.kurtosis())

# Confidence Intervals
values = sbux['return'].dropna().to_numpy()

m = values.mean()
s = values.std(ddof=1)

low = m - 1.96 * s / np.sqrt(len(values))
high = m + 1.96 * s / np.sqrt(len(values))

sbux['return'].hist(bins=100, density=True)
plt.axvline(m, label='mean', color='red')
plt.axvline(low, label='low', color='green')
plt.axvline(high, label='high', color='green')
plt.legend()
plt.show()

plt.axvline(m, label='mean', color='red')
plt.axvline(low, label='low', color='green')
plt.axvline(high, label='high', color='green')
plt.axvline(0, label='zero', color='blue')
plt.legend()
plt.show()

# Statistical Tests
from scipy.stats import jarque_bera, normaltest

# Testing for the normal distribution
print(f"Tests for normality of returns; Jarque-bera: {jarque_bera(values)}")
print(f"Tests for normality of returns; normaltest: {normaltest(values)}")

print(f"Tests for normality of log returns; Jarque-bera: {jarque_bera(sbux['log_return'].dropna())}")
print(f"Tests for normality of log returns; normaltest: {normaltest(sbux['log_return'].dropna())}")

from scipy.stats import kstest

df, loc, scale = t.fit(values)


def cdf(x):
    return t.cdf(x, df, loc, scale)


print(f"Kolmogorov-Smirnov Test: {kstest(values, cdf)}")

df, loc, scale = t.fit(sbux['log_return'].dropna())
print(f"Kolmogorov-Smirnov test for log returns: {kstest(sbux['log_return'].dropna(), cdf)}")

from scipy.stats import ttest_1samp

print(f"1 Sample t-test for SBUX returns: {ttest_1samp(values, 0)}")
print(f"1 Sample t-test for SBUX log returns: {ttest_1samp(sbux['log_return'].dropna(), 0)}")

# Covariance and Correlation
close = pd.read_csv('sp500_close.csv')
goog = data[data['Name'] == 'GOOG']
goog['close'].plot()
plt.show()

symbols = ['AAPL', 'GOOG', 'IBM', 'NFLX', 'SBUX']
sub = close[symbols].copy()
sub.dropna(axis=0, how='all', inplace=True)

for symbol in symbols:
    sub[symbol + '_prev'] = sub[symbol].shift(1)
    sub[symbol + '_ret'] = sub[symbol] / sub[symbol + '_prev'] - 1

rets = sub[[symbol + '_ret' for symbol in symbols]].copy()

import seaborn as sns
sns.pairplot(rets)
plt.show()

print(f"The correlation matrix: {rets.corr()}")
print(f"The covariance matrix: {rets.cov()}")

# Mixture of Gaussians
x_list = np.linspace(-0.1, 0.1, 500)
p = 0.5
fx = p * norm.pdf(x_list, loc=0, scale=0.01) + (1 - p) * norm.pdf(x_list, loc=0, scale=0.002)

plt.plot(x_list, fx)
plt.show()

# Generate samples from our model
samples = []
m0, s0 = 0, 0.01
m1, s1 = 0, 0.002
for _ in range(5000):
    if np.random.random() < p:
        # Choose Gaussian 0
        x = norm.rvs(m0, s0)
    else:
        # Choose Gaussian 1
        x = norm.rvs(m1, s1)
    samples.append(x)

series = pd.Series(samples)
print(f"The kurtosis of the mixture: {series.kurtosis()}")

from sklearn.mixture import GaussianMixture

data = sbux['log_return'].dropna().to_numpy().reshape(-1, 1)
model = GaussianMixture(n_components=2)
model.fit(data)

weights = model.weights_
means = model.means_
cov = model.covariances_
print("Weights:", weights)
print("Means:", means)
print("Variances", cov)

means = means.flatten()
var = cov.flatten()

x_list = np.linspace(data.min(), data.max(), 100)
fx0 = norm.pdf(x_list, means[0], np.sqrt(var[0]))
fx1 = norm.pdf(x_list, means[1], np.sqrt(var[1]))
fx = weights[0] * fx0 + weights[1] * fx1

sbux['log_return'].hist(bins=100, density=True)
plt.plot(x_list, fx, label='mixture model')
plt.legend()
plt.show()

# Volatility Clustering
for i, symbol in enumerate(rets.columns):
    plt.subplot(len(rets.columns), 1, i+1)
    plt.title(symbol)
    rets[symbol].plot(figsize=(12,18))
plt.show()

# Price Simulation
p0 = sbux.iloc[-1]['close']
prices = [p0]
returns = sbux['return'].dropna()
for _ in range(100):
    r = np.random.choice(returns)
    p = prices[-1] * (1 + r)
    prices.append(p)

plt.plot(prices)
plt.show()

df, loc, scale = t.fit(sbux['return'].dropna())
p0 = sbux.iloc[-1]['close']
prices = [p0]
for _ in range(100):
    r = t.rvs(df, loc, scale)
    p = prices[-1] * (1 + r)
    prices.append(p)
