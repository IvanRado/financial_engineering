import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from scipy.optimize import minimize
from cvxopt import matrix
from cvxopt.blas import dot
from cvxopt.solvers import qp, options
np.random.seed(11111111)


def softmax(w):
    a = np.exp(w)
    return a / a.sum()


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


def my_minimize(mu, cov, target_return, D):
    mu2 = matrix(mu)
    cov2 = matrix(2*cov)
    q = matrix(0.0, (D,1))

    G = matrix(-np.eye(D))
    h = matrix(0.5, (D,1))

    A = np.vstack((
        np.ones(D),
        mu,
    ))
    A = matrix(A)
    b = matrix([1.0, target_return])

    res = qp(cov2, q, G, h, A, b)
    if res['status'] != 'optimal':
        print(res)
    return np.sqrt(res['primal objective']), np.array(res['x'])


def my_minimize2(mu, cov, target_return, D):
    mu2 = matrix(mu)
    cov2 = matrix(2*cov)
    q = matrix(0.0, (D,1))

    I = -np.eye(D)
    G = np.vstack((I, -mu))
    h = np.concatenate(([0.5]*D, [-target_return])).reshape(D+1, 1)
    G = matrix(G)
    h = matrix(h)

    A = np.vstack((
        np.ones(D),
    ))
    A = matrix(A)
    b = matrix([1.0])

    res = qp(cov2, q, G, h, A, b)
    if res['status'] != 'optimal':
        print(res)
    return np.sqrt(res['primal objective']), np.array(res['x'])


def two_asset_portfolio():
    mean_return = 0.01 * np.random.randn(2)
    print(f"Mean returns: {mean_return}")

    rho = 0.01 * np.random.randn()
    print(f"Rho: {rho}")

    sigmas = np.exp(np.random.randn(2))
    print(f"The standard deviations: {sigmas}")

    cov = np.diag(sigmas**2)
    print(f"The covariance matrix: {cov}")

    sigma12 = sigmas[0] * sigmas[1] * rho
    cov[0, 1] = sigma12
    cov[1, 0] = sigma12
    print(f"Updated covariance matrix: {cov}")

    # Simulate returns
    N = 1000
    returns = np.zeros(N)
    risks = np.zeros(N)
    for i in range(N):
        w = softmax(np.random.randn(2))
        ret = mean_return.dot(w)
        risk = np.sqrt(w.dot(cov).dot(w))
        returns[i] = ret
        risks[i] = risk

    plt.scatter(risks, returns, alpha=0.1)
    plt.xlabel('Risk')
    plt.ylabel('Return')
    plt.show()

    # Force w to only be positive (no short selling)
    for i in range(N):
        x = np.random.random()
        w = np.array([x, 1-x])
        ret = mean_return.dot(w)
        risk = np.sqrt(w.dot(cov).dot(w))
        returns[i] = ret
        risks[i] = risk

    plt.scatter(risks, returns, alpha=0.1)
    plt.xlabel('Risk')
    plt.ylabel('Return')
    plt.show()

    # Short selling allowed
    for i in range(N):
        x = np.random.random() - 0.5
        w = np.array([x, 1-x])
        ret = mean_return.dot(w)
        risk = np.sqrt(w.dot(cov).dot(w))
        returns[i] = ret
        risks[i] = risk

    plt.scatter(risks, returns, alpha=0.1)
    plt.xlabel('Risk')
    plt.ylabel('Return')
    plt.show()


def three_asset_portfolio():
    mean_return = 0.01 * np.random.randn(3)
    print(f"Mean returns: {mean_return}")

    rhos = 0.01 * np.random.randn(3)
    print(f"Rho: {rhos}")

    sigmas = np.exp(np.random.randn(3))
    print(f"The standard deviations: {sigmas}")

    cov = np.array([
        [sigmas[0] ** 2, rhos[0] * sigmas[0] * sigmas[1], rhos[1] * sigmas[0] * sigmas[2]],
        [rhos[0] * sigmas[0] * sigmas[1], sigmas[1] ** 2, rhos[2] * sigmas[1] * sigmas[2]],
        [rhos[1] * sigmas[0] * sigmas[2], rhos[2] * sigmas[1] * sigmas[2], sigmas[2] ** 2],
    ])
    print(f"The covariance matrix: {cov}")

    N = 1000
    returns = np.zeros(N)
    risks = np.zeros(N)

    # Short selling allowed
    for i in range(N):
        x1, x2 = np.random.random(2) - 0.5
        w = np.array([x1, x2, 1 - x1 - x2])
        np.random.shuffle(w)
        ret = mean_return.dot(w)
        risk = np.sqrt(w.dot(cov).dot(w))
        returns[i] = ret
        risks[i] = risk

    plt.scatter(risks, returns, alpha=0.1)
    plt.xlabel('Risk')
    plt.ylabel('Return')
    plt.show()

    # Min-Max return
    D = len(mean_return)
    A_eq = np.ones((1, D))
    b_eq = np.ones(1)

    bounds = [(-0.5, None)]*D
    print(f"Bounds of the optimization: {bounds}")

    # Minimize
    res = linprog(mean_return, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    min_return = res.fun

    # Maximize
    res = linprog(-mean_return, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    max_return = -res.fun

    print(f"Min and Max returns: {min_return} and {max_return}")

    # Mean-Variance Optimal Portfolios
    N = 100
    target_returns = np.linspace(min_return, max_return, num=N)

    constraints = [
        {
            'type': 'eq',
            'fun': target_return_constraint,
            'args': [target_returns[0], mean_return]
        },
        {
            'type': 'eq',
            'fun': portfolio_constraint
        }
    ]

    res = minimize(
        fun=get_portfolio_variance,
        args=cov,
        x0=np.ones(D) / D,
        method='SLSQP',
        constraints=constraints
    )
    print(f"The result of minimization: {res}")

    # Limit the magnitude of the weights
    res = minimize(
        fun=get_portfolio_variance,
        args=cov,
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
            args=cov,
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

    # Repeat but with >= constraint on target return
    constraints_ineq = [
        {
            'type': 'ineq',
            'fun': target_return_constraint,
            'args': [target_returns[0], mean_return]
        },
        {
            'type': 'eq',
            'fun': portfolio_constraint
        }
    ]

    optimized_risks = []
    optimized_returns = []
    for target in target_returns:
        # Set target return constraint
        constraints_ineq[0]['args'] = [target, mean_return]
        res = minimize(
            fun=get_portfolio_variance,
            x0=np.ones(D) / D,
            args=cov,
            method='SLSQP',
            constraints=constraints_ineq,
            bounds=bounds
        )
        optimized_risks.append(np.sqrt(res.fun))
        optimized_returns.append(mean_return.dot(res.x))
        if res.status != 0:
            print(res)

    plt.scatter(risks, returns, alpha=0.1)
    plt.plot(optimized_risks, optimized_returns, c='black')
    plt.xlabel("Risk")
    plt.ylabel("Return")
    plt.show()

    # Sharpe Ratio
    max_sr = float('-inf')
    max_sr_weights = None
    for target in target_returns:
        constraints[0]['args'] = [target, mean_return]
        res = minimize(
            fun=neg_sharpe_ratio,
            x0=np.ones(D) / D,
            args=(mean_return, cov),
            method='SLSQP',
            constraints={
                'type':'eq',
                'fun':portfolio_constraint,
            },
            bounds=bounds,
        )
        if -res.fun > max_sr:
            max_sr = -res.fun
            max_sr_weights = res.x

    print(f"Max Sharpe Ratio: {max_sr} and weights: {max_sr_weights}")

    res = minimize(
        fun=neg_sharpe_ratio,
        x0=np.ones(D) / D,
        args=(mean_return, cov),
        method='SLSQP',
        constraints={
            'type': 'eq',
            'fun': portfolio_constraint,
        },
        bounds=bounds,
    )
    print(f"The result of the optimization: {res}")
    print(f"Another statistic: {-res.fun * np.sqrt(252)}")

    w = res.x
    sr_return = w.dot(mean_return)
    sr_risk = np.sqrt(w.dot(cov).dot(w))

    plt.scatter(risks, returns, alpha=0.1)
    plt.plot(optimized_risks, optimized_returns, c='black')
    plt.scatter([sr_risk], [sr_return], c='red')
    plt.xlabel("Risk")
    plt.ylabel("Return")
    plt.show()

    # Using CVXOPT instead
    G = matrix(0.0, (3,3))
    G[::4] = -1.0
    print(G)

    print(f"My minimization: {my_minimize(mean_return, cov, 0, D)}")

    cvx_risks = []
    for target in target_returns:
        risk, w = my_minimize(mean_return, cov, target, D)
        cvx_risks.append(risk)

    plt.scatter(risks, returns, alpha=0.1)
    plt.plot(cvx_risks, target_returns, c='black')
    plt.scatter([sr_risk], [sr_return], c='red')
    plt.xlabel("Risk")
    plt.ylabel("Return")
    plt.show()


    cvx_risks = []
    cvx_returns = []
    for target in target_returns:
        risk, w = my_minimize2(mean_return, cov, target, D)
        cvx_risks.append(risk)
        cvx_returns.append(mean_return.dot(w))

    plt.scatter(risks, returns, alpha=0.1)
    plt.plot(cvx_risks, cvx_returns, c='black')
    plt.scatter([sr_risk], [sr_return], c='red')
    plt.xlabel("Risk")
    plt.ylabel("Return")
    plt.show()

    # Repeat with tangential line
    plt.scatter(risks, returns, alpha=0.1)
    plt.plot(cvx_risks, cvx_returns, c='black')
    plt.scatter([sr_risk], [sr_return], c='red')

    x1 = 0
    y1 = 0
    x2 = sr_risk
    y2 = sr_return
    x3 = 3
    y3 = 3 * sr_return / sr_risk
    plt.plot([x1, x2, x3], [y1, y2, y3])

    plt.xlabel("Risk")
    plt.ylabel("Return")
    plt.show()


if __name__ == "__main__":
    three_asset_portfolio()
