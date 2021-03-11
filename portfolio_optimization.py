import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from scipy.optimize import minimize


def get_portfolio_variance(weights, cov):
    return weights.dot(cov).dot(weights)


def target_return_constraint(weights, target, mean_return):
    return weights.dot(mean_return) - target


def portfolio_constraint(weights):
    return weights.sum() - 1


def neg_sharpe_ratio(weights, mean_return, cov):
    mean = weights.dot(mean_return)
    sd = np.sqrt(weights.dot(cov).dot(weights))
    return -mean / sd


def main():
    df = pd.read_csv('sp500sub.csv', index_col='Date', parse_dates=True)
    names = ['GOOG', 'SBUX', 'KSS', 'NEM']
    print(df['Name'].unique())
    all_dates = df.index.unique().sort_values()

    start = all_dates.get_loc('2014-01-02')
    end = all_dates.get_loc('2016-06-30')
    dates = all_dates[start:end+1]

    close_prices = pd.DataFrame(index=dates)
    tmp1 = df.loc[dates]
    for name in names:
        df_sym = tmp1[tmp1['Name'] == name]
        df_tmp = pd.DataFrame(data=df_sym['Close'].to_numpy(),
                              index=df_sym.index, columns=[name])
        close_prices = close_prices.join(df_tmp)

    close_prices.fillna(method='ffill', inplace=True)
    returns = pd.DataFrame(index=dates[1:])
    for name in names:
        current_returns = close_prices[name].pct_change()
        returns[name] = current_returns.iloc[1:] * 100

    mean_return = returns.mean()
    print(f"Mean Returns: {mean_return}")
    cov = returns.cov()
    print(f"Covariances {cov}")
    cov_np = cov.to_numpy()

    N = 10000
    D = len(mean_return)
    returns = np.zeros(N)
    risks = np.zeros(N)
    random_weights = []
    for i in range(N):
        rand_range = 1.0
        w = np.random.random(D)*rand_range - rand_range
        w[-1] = 1 - w[:-1].sum()
        np.random.shuffle(w)
        random_weights.append(w)
        ret = mean_return.dot(w)
        risk = np.sqrt(w.dot(cov_np).dot(w))
        returns[i] = ret
        risks[i] = risk

    single_asset_returns = np.zeros(D)
    single_asset_risks = np.zeros(D)
    for i in range(D):
        ret = mean_return[i]
        risk = np.sqrt(cov_np[i,i])

        single_asset_risks[i] = risk
        single_asset_returns[i] = ret

    plt.scatter(risks, returns, alpha=0.1)
    plt.scatter(single_asset_risks, single_asset_returns, c='red')
    plt.xlabel("Risk")
    plt.ylabel("Return")
    # plt.show()

    for idx, val in cov['GOOG'].iteritems():
        print(idx, val)

    D = len(mean_return)
    A_eq = np.ones((1, D))
    b_eq = np.ones(1)
    bounds = [(-0.5, None)]*D

    res = linprog(mean_return, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    min_return = res.fun
    res = linprog(-mean_return, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    max_return = -res.fun
    print(f"Min return: {min_return}, max return: {max_return}")

    # Mean-Variance Optimal Portfolios
    N = 100
    target_returns = np.linspace(min_return, max_return, num=N)

    constraints = [
        {
            'type': 'eq',
            'fun': target_return_constraint,
            'args': (target_returns[0], mean_return)
        },
        {
            'type': 'eq',
            'fun': portfolio_constraint
        }
    ]

    res = minimize(
        fun=get_portfolio_variance,
        args=cov_np,
        x0=np.ones(D) / D,
        method='SLSQP',
        constraints=constraints
    )
    print(f"The result of minimization: {res}")

    # Limit the magnitude of the weights
    res = minimize(
        fun=get_portfolio_variance,
        args=cov_np,
        x0=np.ones(D) / D,
        method='SLSQP',
        constraints=constraints,
        bounds=bounds
    )
    print(f"The result of minimization, bounded: {res}")

    optimized_risks = []
    for target in target_returns:
        # Set target return constraint
        constraints[0]['args'] = [target, mean_return]
        res = minimize(
            fun=get_portfolio_variance,
            x0=np.ones(D) / D,
            args=cov_np,
            method='SLSQP',
            constraints=constraints,
            bounds=bounds
        )
        optimized_risks.append(np.sqrt(res.fun))
        if res.status != 0:
            print(res)

    plt.scatter(risks, returns, alpha=0.1)
    plt.plot(optimized_risks, target_returns, c='black')
    plt.xlabel("Risk")
    plt.ylabel("Return")
    plt.show()

    # Min Variance Portfolio
    res = minimize(
        fun=get_portfolio_variance,
        x0=np.ones(D) / D,
        args=cov,
        method='SLSQP',
        constraints={
            'type': 'eq',
            'fun': portfolio_constraint,
        },
        bounds=bounds
    )
    print(f"Min Variance Portfolio: {res}")

    mv_risk = np.sqrt(res.fun)
    mv_weights = res.x
    mv_ret = mv_weights.dot(mean_return)

    plt.scatter(risks, returns, alpha=0.1)
    plt.plot(optimized_risks, target_returns, c='black')
    plt.scatter([mv_risk], [mv_ret], c='red')
    plt.xlabel("Risk")
    plt.ylabel("Return")
    plt.show()

    # Sharpe Ratio
    # https://fred.stlouisfed.org/series/TB3MS
    risk_free_rate = 0.03 / 252

    res = minimize(
        fun=neg_sharpe_ratio,
        x0=np.ones(D) / D,
        args=(mean_return, cov),
        method='SLSQP',
        constraints={
            'type': 'eq',
            'fun': portfolio_constraint,
        },
        bounds=bounds
    )
    print(f"Optimal Sharpe Ratio Portfolio: {res}")

    best_sr, best_w = -res.fun, res.x
    mc_best_w = None
    mc_best_sr = float('-inf')
    for i, (risk, ret) in enumerate(zip(risks, returns)):
        sr = (ret - risk_free_rate) / risk
        if sr > mc_best_sr:
            mc_best_sr = sr
            mc_best_w = random_weights[i]
    print(f"The best weights: {mc_best_w}")
    print(f"The best sharpe ratio: {mc_best_sr}")

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.scatter(risks, returns, alpha=0.1)
    plt.plot(optimized_risks, target_returns, c='black')

    # Found by optimization
    opt_risk = np.sqrt(best_w.dot(cov).dot(best_w))
    opt_ret = mean_return.dot(best_w)
    plt.scatter([opt_risk], [opt_ret], c='red')

    # Found by Monte Carlo Simulation
    mc_risk = np.sqrt(mc_best_w.dot(cov).dot(mc_best_w))
    mc_ret = mean_return.dot(mc_best_w)
    plt.scatter([mc_risk], [mc_ret], c='pink')

    plt.xlabel('Risk')
    plt.ylabel('Return')
    plt.show()

    # Risk-free asset with tangency portfolio
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.scatter(risks, returns, alpha=0.1)
    plt.plot(optimized_risks, target_returns, c='black')

    # Found by optimization
    opt_risk = np.sqrt(best_w.dot(cov).dot(best_w))
    opt_ret = mean_return.dot(best_w)
    plt.scatter([opt_risk], [opt_ret], c='red')

    # Tangent line
    x1 = 0
    y1 = risk_free_rate
    x2 = opt_risk
    y2 = opt_ret
    plt.plot([x1, x2], [y1, y2])

    plt.xlabel('Risk')
    plt.ylabel('Return')
    plt.show()


if __name__ == "__main__":
    main()
