from hmmlearn import hmm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)


def hmm_model():
    df = pd.read_csv('SPY.csv', index_col='Date', parse_dates=True)
    returns = np.log(df['Close']).diff()
    returns.dropna(inplace=True)

    returns.hist(bins=50)
    plt.title("Histogram of returns")
    plt.show()

    model = hmm.GaussianHMM(n_components=2, covariance_type='diag')
    X = returns.to_numpy().reshape(-1, 1)
    model.fit(X)
    Z = model.predict(X)

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.subplot(211)
    plt.plot(Z)
    plt.subplot(212)
    plt.plot(returns)
    plt.show()

    # Draw different segments in different colors according to state
    fig, ax = plt.subplots(figsize=(10, 5))

    # First create arrays with nan
    returns0 = np.empty(len(Z))
    returns1 = np.empty(len(Z))
    returns0[:] = np.nan
    returns1[:] = np.nan

    # Fill in the values only if the state is the one corresponding to the array
    returns0[Z == 0] = returns[Z == 0]
    returns1[Z == 1] = returns[Z == 1]
    plt.plot(returns0, label='state 0')
    plt.plot(returns1, label='state 1')
    plt.legend()
    plt.show()

    print(f"Transition matrix: {model.transmat_}")

    # Try to set the transition matrix intuitively
    model.transmat_ = np.array([
        [0.999, 0.001],
        [0.001, 0.999],
    ])

    Z = model.predict(X)
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.subplot(211)
    plt.plot(Z)
    plt.subplot(212)
    plt.plot(returns)
    plt.show()

    # Draw different segments in different colors according to state
    fig, ax = plt.subplots(figsize=(10, 5))

    # First create arrays with nan
    returns0 = np.empty(len(Z))
    returns1 = np.empty(len(Z))
    returns0[:] = np.nan
    returns1[:] = np.nan

    # Fill in the values only if the state is the one corresponding to the array
    returns0[Z == 0] = returns[Z == 0]
    returns1[Z == 1] = returns[Z == 1]
    plt.plot(returns0, label='state 0')
    plt.plot(returns1, label='state 1')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    hmm_model()
