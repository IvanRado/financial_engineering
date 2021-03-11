import warnings
import numpy as np
import matplotlib.pyplot as plt
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd


def trend_following_strategy():
    df = pd.read_csv('SPY.csv', index_col='Date', parse_dates=True)
    df['LogReturn'] = np.log(df['Close']).diff()
    df['LogReturn'] = df['LogReturn'].shift(-1)

    df['SlowSMA'] = df['Close'].rolling(30).mean()
    df['FastSMA'] = df['Close'].rolling(10).mean()

    # Looked at the data
    df[['Close', 'FastSMA', 'SlowSMA']].plot(figsize=(10, 5))
    plt.show()

    # A closer look
    df[['Close', 'FastSMA', 'SlowSMA']].iloc[:300].plot(figsize=(10, 5))
    plt.show()

    df['Signal'] = np.where(df['FastSMA'] >= df['SlowSMA'], 1, 0)
    df['PrevSignal'] = df['Signal'].shift(1)
    buy = []
    sell = []
    for idx, row in df.iterrows():
        if row['PrevSignal'] == 0 and row['Signal'] == 1:
            buy.append(True)
            sell.append(False)
        elif row['PrevSignal'] == 1 and row['Signal'] == 0:
            buy.append(False)
            sell.append(True)
        else:
            buy.append(False)
            sell.append(False)

    df['Buy'] = buy
    df['Sell'] = sell


    # df['Buy'] = (df['PrevSignal' == 0]) & (df['Signal'] == 1) # Fast < Slow --> Fast > Slow
    # df['Sell'] = (df['PrevSignal'] == 1) & (df['Signal'] == 0) # Fast > Slow --> Fast < Slow
    is_invested = False

    def assign_is_invested(row, is_invested):
        if is_invested and row['Sell']:
            is_invested = False
        if not is_invested and row['Buy']:
            is_invested = True

        # Otherwise return the same
        return is_invested

    df['IsInvested'] = df.apply(assign_is_invested, axis=1, is_invested=is_invested)
    df['AlgoLogReturn'] = df['IsInvested'] * df['LogReturn']

    print(f"Total algo log return: {df['AlgoLogReturn'].sum()}")
    print(f"Total return buy-and-hold {df['LogReturn'].sum()}")

    print(f"Algorithm Standard Deviation: {df['AlgoLogReturn'].std()}")
    print(f"Algorithm Mean/STD: {df['AlgoLogReturn'].mean()/df['AlgoLogReturn'].std()}")

    print(f"Buy-and-hold standard deviation: {df['LogReturn'].std()}")
    print(f"Buy-and-hold Mean/STD: {df['LogReturn'].mean()/df['LogReturn'].std()}")

    # Searching for Fast and Slow Hyperparameters - Using Grid Search
    Ntest = 1000

    def trend_following(df, fast, slow):
        global is_invested
        df['SlowSMA'] = df['Close'].rolling(slow).mean()
        df['FastSMA'] = df['Close'].rolling(fast).mean()
        df['Signal'] = np.where(df['FastSMA'] >= df['SlowSMA'], 1, 0)
        df['PrevSignal'] = df['Signal'].shift(1)
        df['Buy'] = (df['PrevSignal'] == 0) & (df['Signal'] == 1)
        df['Sell'] = (df['PrevSignal'] == 1) & (df['Signal'] == 0)

        # Split into train and test
        train = df.iloc[:-Ntest]
        test = df.iloc[-Ntest:]

        is_invested = False
        df.loc[:-Ntest, 'IsInversted'] = train.apply(assign_is_invested, axis=1, is_invested=is_invested)
        df.loc[:-Ntest, 'AlgoLogReturn'] = train['IsInvested'] * train['LogReturn']

        is_invested = False
        df.loc[-Ntest:, 'IsInvested'] = test.apply(assign_is_invested, axis=1, is_invested=is_invested)
        df.loc[-Ntest:, 'AlgoLogReturn'] = test['IsInvested'] * test['LogReturn']

        return train['AlgoLogReturn'][:-1].sum(), test['AlgoLogReturn'][:-1].sum()

    best_fast = None
    best_slow = None
    best_score = float('-inf')
    for fast in range(3, 30):
        for slow in range(fast + 5, 50):
            score, _ = trend_following(df, fast, slow)
            if score > best_score:
                best_fast = fast
                best_slow = slow
                best_score = score

    print(f"Best fast window: {best_fast}")
    print(f"Best slow window: {best_slow}")
    print(f"Train and test returns: {trend_following(df, best_fast, best_slow, Ntest)}")

    train = df.iloc[:-Ntest].copy()
    test = df.iloc[-Ntest:].copy()

    print(f"Total return buy-and-hold train: {train['LogReturn'][:-1].sum()}")
    print(f"Total return buy-and-hold test: {test['LogReturn'][:-1].sum()}")

    print(f"Standard return algorithm train: {train['AlgoLogReturn'].mean()/train['AlgoLogReturn'].std()}")
    print(f"Standard return buy-and-hold train: {train['LogReturn'].mean()/train['LogReturn'].std()}")

    print(f"Standard return algorithm test: {test['AlgoLogReturn'].mean() / test['AlgoLogReturn'].std()}")
    print(f"Standard return buy-and-hold test: {test['LogReturn'].mean() / test['LogReturn'].std()}")

    # Wealth over time
    train['CumLogReturn'] = train['AlgoLogReturn'].cumsum().shift(1)
    train['CumWealth'] = train.iloc[0]['Close'] * np.exp(train['CumLogReturn'])
    train[['Close', 'SlowSMA', 'FastSMA', 'CumWealth']].plot(figsize=(20, 10))
    plt.show()


if __name__ == "__main__":
    trend_following_strategy()
