import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pmdarima as pm
from statsmodels.tsa.arima_model import ARIMA


def plot_fit_and_forecast(df, train, test, N_test, result):
    fix, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['Passengers'], label='data')

    # Plot the curve fitted on the train set
    train_pred = result.fittedvalues
    ax.plot(train.index ,train_pred, color='green', label='fitted')

    # Forecast the test set
    forecast, stderr, confint = result.forecast(N_test)
    ax.plot(test.index, forecast, label='forecast')
    ax.fill_between(test.index,
                    confint[:,0], confint[:,1],
                    color='red', alpha=0.3)
    ax.legend()
    plt.show()


def plot_fit_and_forecast_int(df, train, test, N_test, result, d, col='Passengers'):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df[col], label='data')

    # Plot the curve fitted on the train set
    train_pred = result.predict(start=train.index[d], end=train.index[-1], type='levels')
    ax.plot(train.index[d:], train_pred, color='green', label='fitted')

    # Forecast the test set
    forecast, stderr, confint = result.forecast(N_test)
    ax.plot(test.index, forecast, label='forecast')
    ax.fill_between(test.index,
                    confint[:,0], confint[:,1],
                    color='red', alpha=0.3)
    ax.legend()
    plt.show()


def plot_difference(df, train, result, d, col='Passengers'):
    train_pred = result.predict(start=train.index[d], end=train.index[-1])
    diff = df[col].diff()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(diff, label='True 1st difference')
    ax.plot(train_pred, label='Fitted 1st difference')
    plt.show()


def simple_rmse(t, y):
    return np.sqrt(np.mean((t - y)**2))


def rmse(test, N_test, result, is_logged):
    forecast, stderr, confint = result.forecast(N_test)
    if is_logged:
        forecast = np.exp(forecast)

    t = test['Passengers']
    y = forecast
    return np.sqrt(np.mean((t - y)**2))


def plot_result(model, fulldata, train, test, N_test):
    params = model.get_params()
    d = params['order'][1]

    train_pred = model.predict_in_sample(start=d, end=-1)
    test_pred, confint = model.predict(n_periods=N_test, return_conf_int=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(fulldata.index, fulldata, label='data')
    ax.plot(train.index[d:], train_pred, label='fitted')
    ax.plot(test.index, test_pred, label='forecast')
    ax.fill_between(test.index,
                    confint[:,0], confint[:,1],
                    color='red', alpha=0.3)
    ax.legend()
    plt.show()


def plot_test(model, test, N_test):
    test_pred, confint = model.predict(n_periods=N_test, return_conf_int=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(test.index, test, label='true')
    ax.plot(test.index, test_pred, label='forecast')
    ax.fill_between(test.index,
                    confint[:,0], confint[:,1],
                    color='red', alpha=0.3)
    ax.legend()
    plt.show()


def arima1():
    df = pd.read_csv('airline_passengers.csv')
    df.plot()
    plt.show()

    df['1stdiff'] = df['Passengers'].diff()
    df.plot()
    plt.show()

    df['LogPassengers'] = np.log(df['Passengers'])
    df['LogPassengers'].plot()
    plt.show()

    df.index.freq = 'MS'

    N_test = 12
    train = df.iloc[:-N_test]
    test = df.iloc[-N_test:]

    train_idx = df.index <= train.index[-1]
    test_idx = df.index > train.index[-1]

    arima = ARIMA(train['Passengers'], order=(1,0,0))
    arima_result = arima.fit()

    df.loc[test_idx, 'AR(1)'] = arima_result.predict(start=train.index[0], end=train.index[-1])
    df[['Passengers', 'AR(1)']].plot()
    plt.show()

    forecast, stderr, confint = arima_result.forecast(N_test)
    df.loc[test_idx, 'AR(1)'] = forecast
    df[['Passengers', 'AR(1)']].plot()
    plt.show()

    plot_fit_and_forecast(df, train, test, N_test, arima_result)

    arima = ARIMA(train['Passengers'], order=(10,0,0))
    arima_result = arima.fit()
    plot_fit_and_forecast(df, train, test, N_test, arima_result)

    arima = ARIMA(train['Passengers'], order=(0,0,1))
    arima_result = arima.fit()
    plot_fit_and_forecast(df, train, test, N_test, arima_result)

    df['Log1stDiff'] = df['LogPassengers'].diff()
    df['Log1stDiff'].plot()
    plt.show()

    arima = ARIMA(train['Passengers'], order=(8, 1, 1))
    arima_result_811 = arima.fit()
    plot_fit_and_forecast_int(df, train, test, N_test, arima_result_811, 1)
    plot_difference(df, train, arima_result_811, 1)

    arima = ARIMA(train['LogPassengers'], order=(8, 1, 1))
    arima_result_log811 = arima.fit()
    plot_fit_and_forecast_int(df, train, test, N_test, arima_result_log811, 1, col='LogPassengers')
    plot_difference(df, train, arima_result_log811, 1, col='LogPassengers')

    arima = ARIMA(train['LogPassengers'], order=(12, 1, 0))
    arima_result_log1210 = arima.fit()
    plot_fit_and_forecast_int(df, train, test, N_test, arima_result_log1210, 1, col='LogPassengers')

    print(f"RMSE ARIMA(8,1,1): {rmse(arima_result_811, False)}")
    print(f"RMSE ARIMA(8,1,1) logged: {rmse(arima_result_log811, False)}")
    print(f"RMSE ARIMA(12,1,0) logged: {rmse(arima_result_log1210, False)}")


def arima2():
    df = pd.read_csv('airline_passengers.csv', index_col='Month', parse_dates=True)
    df['LogPassengers'] = np.log(df['Passengers'])

    N_test = 12
    train = df.iloc[:-N_test]
    test = df.iloc[-N_test:]

    model = pm.auto_arima(train['Passengers'],
                          trace=True,
                          suppress_warnings=True,
                          seasonal=True, m=12)

    print(f"Model summary: {model.summary()}")

    test_pred, confint = model.predict(n_periods=N_test, return_conf_int=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(test.index, test['Passengers'], label='data')
    ax.plot(test.index, test_pred, label='forecast')
    ax.fill_between(test.index,
                    confint[:,0], confint[:,1],
                    color='red', alpha=0.3)
    ax.legend()
    plt.show()

    train_pred = model.predict_in_sample(start=0, end=-1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df['Passengers'], label='data')
    ax.plot(train.index, train_pred, label='fitted')
    ax.plot(test.index, test_pred, label='forecast')
    ax.fill_between(test.index,
                    confint[:, 0], confint[:, 1],
                    color='red', alpha=0.3)
    ax.legend()
    plt.show()

    logmodel = pm.auto_arima(train['LogPassengers'],
                          trace=True,
                          suppress_warnings=True,
                          seasonal=True, m=12)

    print(f"Log Model Summary: {logmodel.summary()}")

    test_pred_log, confint = logmodel.predict(n_periods=N_test, return_conf_int=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(test.index, test['LogPassengers'], label='data')
    ax.plot(test.index, test_pred_log, label='forecast')
    ax.fill_between(test.index,
                    confint[:, 0], confint[:, 1],
                    color='red', alpha=0.3)
    ax.legend()
    plt.show()

    train_pred_log = logmodel.predict_in_sample(start=0, end=-1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df['LogPassengers'], label='data')
    ax.plot(train.index, train_pred_log, label='fitted')
    ax.plot(test.index, test_pred_log, label='forecast')
    ax.fill_between(test.index,
                    confint[:, 0], confint[:, 1],
                    color='red', alpha=0.3)
    ax.legend()
    plt.show()

    print(f"Non-logged RMSE: {simple_rmse(test['Passengers'], test_pred)}")
    print(f"Logged RMSE: {simple_rmse(test['Passengers'], np.exp(test_pred_log))}")

    # Non-seasonal model
    model = pm.auto_arima(train['LogPassengers'],
                          trace=True,
                          suppress_warnings=True,
                          max_p=12, max_q=2, max_order=14,
                          stepwise=False,
                          seasonal=False)

    print(f"Model Summary: {model.summary()}")

    test_pred, confint = model.predict(n_periods=N_test, return_conf_int=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(test.index, test['LogPassengers'], label='data')
    ax.plot(test.index, test_pred, label='forecast')
    ax.fill_between(test.index,
                    confint[:, 0], confint[:, 1],
                    color='red', alpha=0.3)
    ax.legend()
    plt.show()

    train_pred = model.predict_in_sample(start=0, end=-1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df['LogPassengers'], label='data')
    ax.plot(train.index, train_pred, label='fitted')
    ax.plot(test.index, test_pred, label='forecast')
    ax.fill_between(test.index,
                    confint[:, 0], confint[:, 1],
                    color='red', alpha=0.3)
    ax.legend()
    plt.show()

    print(f"logged RMSE: {simple_rmse(test['Passengers'], np.exp(test_pred))}")

    # Non-seasonal non-logged model
    model = pm.auto_arima(train['Passengers'],
                          trace=True,
                          suppress_warnings=True,
                          max_p=12, max_q=2, max_order=14,
                          stepwise=False,
                          seasonal=False)

    print(f"Model Summary: {model.summary()}")

    test_pred_log, confint = model.predict(n_periods=N_test, return_conf_int=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(test.index, test['LogPassengers'], label='data')
    ax.plot(test.index, test_pred_log, label='forecast')
    ax.fill_between(test.index,
                    confint[:, 0], confint[:, 1],
                    color='red', alpha=0.3)
    ax.legend()
    plt.show()

    train_pred = model.predict_in_sample(start=0, end=-1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df['LogPassengers'], label='data')
    ax.plot(train.index, train_pred, label='fitted')
    ax.plot(test.index, test_pred, label='forecast')
    ax.fill_between(test.index,
                    confint[:, 0], confint[:, 1],
                    color='red', alpha=0.3)
    ax.legend()
    plt.show()

    print(f"Non-logged RMSE: {simple_rmse(test['Passengers'], test_pred)}")


    df.plot()
    plt.show()

    df['1stdiff'] = df['Passengers'].diff()
    df.plot()
    plt.show()


    df['LogPassengers'].plot()
    plt.show()

    df.index.freq = 'MS'



    train_idx = df.index <= train.index[-1]
    test_idx = df.index > train.index[-1]


def arima3():
    df = pd.read_csv('sp500sub.csv', index_col='Date', parse_dates=True)
    goog = df[df['Name'] == 'GOOG']['Close']
    goog.plot()
    plt.show()

    N_test = 30
    train = goog.iloc[:-N_test]
    test = goog.iloc[-N_test:]

    model = pm.auto_arima(train,
                          error_action='ignore', trace=True,
                          suppress_warnings=True, maxiter=10,
                          seasonal=False)

    print(f"Model Summary: {model.summary()}")
    print(f"Model Parameters: {model.get_params()}")

    plot_result(model, goog, train, test, N_test)
    plot_test(model, test, N_test)

    print(f"RMSE ARIMA: {simple_rmse(model.predict(N_test), test)}")
    print(f"RMSE Naive: {simple_rmse(train.iloc[-1], test)}")

    # Done with Apple
    aapl = df[df['Name'] == 'AAPL']['Close']
    aapl.plot()
    plt.show()

    train = aapl.iloc[:-N_test]
    test = aapl.iloc[-N_test:]

    model = pm.auto_arima(train,
                          error_action='ignore', trace=True,
                          suppress_warnings=True, maxiter=10,
                          seasonal=False)

    print(f"Model Summary: {model.summary()}")
    print(f"Model Parameters: {model.get_params()}")

    plot_result(model, aapl, train, test, N_test)
    plot_test(model, test, N_test)

    print(f"RMSE ARIMA: {simple_rmse(model.predict(N_test), test)}")
    print(f"RMSE Naive: {simple_rmse(train.iloc[-1], test)}")

    # Done with IBM
    ibm = df[df['Name'] == 'IBM']['Close']
    ibm.plot()
    plt.show()

    train = ibm.iloc[:-N_test]
    test = ibm.iloc[-N_test:]

    model = pm.auto_arima(train,
                          error_action='ignore', trace=True,
                          suppress_warnings=True, maxiter=10,
                          seasonal=False)

    print(f"Model Summary: {model.summary()}")
    print(f"Model Parameters: {model.get_params()}")

    plot_result(model, ibm, train, test, N_test)
    plot_test(model, test, N_test)

    print(f"RMSE ARIMA: {simple_rmse(model.predict(N_test), test)}")
    print(f"RMSE Naive: {simple_rmse(train.iloc[-1], test)}")

    # Done with IBM
    sbux = df[df['Name'] == 'SBUX']['Close']
    sbux.plot()
    plt.show()

    train = sbux.iloc[:-N_test]
    test = sbux.iloc[-N_test:]

    model = pm.auto_arima(train,
                          error_action='ignore', trace=True,
                          suppress_warnings=True, maxiter=10,
                          seasonal=False)

    print(f"Model Summary: {model.summary()}")
    print(f"Model Parameters: {model.get_params()}")

    plot_result(model, sbux, train, test, N_test)
    plot_test(model, test, N_test)

    print(f"RMSE ARIMA: {simple_rmse(model.predict(N_test), test)}")
    print(f"RMSE Naive: {simple_rmse(train.iloc[-1], test)}")


if __name__ == "__main__":
    arima1()



