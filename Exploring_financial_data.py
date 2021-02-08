import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('all_stocks_5yr.csv')

print(df.head())
print(df.info())

print(df['Name'].unique().shape)

ibm = df[df['Name'] == 'IBM']
ibm['close'].plot()
# plt.show()

dates = pd.date_range(df['date'].min(), df['date'].max())
close_prices = pd.DataFrame(index=dates)

df2 = pd.DataFrame(data=ibm['close'].to_numpy(), index=ibm['date'], columns=['IBM'])
print(df2.head())

symbols = df['Name'].unique()
for symbol in symbols:
    df_sym = df[df['Name'] == symbol]
    df_tmp = pd.DataFrame(data=df_sym['close'].to_numpy(),
                          index=pd.to_datetime(df_sym['date']), columns=[symbol])
    close_prices = close_prices.join(df_tmp)


print(close_prices.head())

close_prices.to_csv('sp500_close.csv')

close_prices.dropna(axis=0, how='all', inplace=True)
close_prices.fillna(method='ffill', inplace=True)
close_prices.fillna(method='bfill', inplace=True)

close_prices.plot(legend=False, figsize=(10,10))
plt.show()

close_prices_normalized = close_prices / close_prices.iloc[0]
close_prices_normalized.plot(legend=False, figsize=(10,10))
plt.show()

