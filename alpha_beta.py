import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

spy = pd.read_csv('SPY.csv', index_col=0, parse_dates=True)
spy['SPY'] = spy['Close'].pct_change(1)
index = pd.read_csv('sp500sub.csv', index_col=0, parse_dates=True)
aapl = index[index['Name'] == 'AAPL'].copy()
aapl['AAPL'] = aapl['Close'].pct_change(1)
joined = aapl[['AAPL']].join(spy['SPY'])

joined.iloc[100:150].plot(figsize=(10,5))
plt.show()

joined.plot.scatter('SPY', 'AAPL')
plt.show()

# Make the dataset
joined.dropna(inplace=True)
X = joined[['SPY']].to_numpy()
Y = joined[['AAPL']].to_numpy()

plt.scatter(X, Y)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, Y)

X_predict = np.linspace(X.min(), X.max(), 5).reshape(-1,1)
Y_predict = model.predict(X_predict)

plt.scatter(X, Y)
plt.xlabel('SPY')
plt.ylabel('AAPL')
plt.plot(X_predict, Y_predict)
plt.show()

beta = model.coef_
alpha = model.intercept_
print(f"alpha value: {alpha}, beta value: {beta}")

Y_predict = beta * X_predict + alpha
plt.scatter(X, Y)
plt.xlabel('SPY')
plt.ylabel('AAPL')
plt.plot(X_predict, Y_predict)
plt.show()

print(f"Standard deviations: {joined.std()}")
print(f"Relative standard deviation: {joined['AAPL'].std() / joined['SPY'].std()}")


