# -*- coding: utf-8 -*-
"""
Strong rules for coordinate descent
Author: Fabian Pedregosa <fabian@fseoane.net>
"""

import numpy as np
from scipy import linalg

MAX_ITER = 100


def l1_coordinate_descent(X, y, alpha, warm_start=None, max_iter=MAX_ITER):

    if warm_start is not None:
        beta = warm_start
    else:
        beta = np.zeros(X.shape[1], dtype=np.float)
    alpha = alpha * X.shape[0]

    for _ in range(max_iter):
        for i in range(X.shape[1]):
            bb = beta.copy()
            bb[i] = 0.
            residual = np.dot(X[:, i], y - np.dot(X, bb).T)
            beta[i] = np.sign(residual) * np.fmax(np.abs(residual) - alpha, 0) \
                / np.dot(X[:, i], X[:, i])
    return beta


def shrinkage(X, y, alpha, beta, active_set, max_iter):
    bb = beta.copy()
    for _ in range(max_iter):
        for i in active_set:

            bb[i] = 0
            residual = np.dot(X[:, i], y - np.dot(X, bb).T)
            bb[i] = np.sign(residual) * np.fmax(np.abs(residual) - alpha, 0) \
                / np.dot(X[:, i], X[:, i])
    return bb


def l1_path(X, y, alphas, max_iter=MAX_ITER, verbose=False):
    """
    The strategy is described in "Strong rules for discarding predictors in lasso-type problems"
    alphas must be a decreasing sequence of regularization parameters
    WARNING: does not compute intercept
    """
    beta = np.zeros((len(alphas), X.shape[1]), dtype=np.float)

    alphas_scaled = np.array(alphas) * X.shape[0]

    active_set = np.arange(X.shape[1]).tolist()
    for k, a in enumerate(alphas_scaled):

        if k > 0:
        # .. Strong rules for discarding predictors in lasso-type ..
            tmp = np.dot(X.T, y - np.dot(X, beta[k - 1]))
            tmp = np.abs(tmp)
            strong_active_set = tmp >= 2 * alphas_scaled[k] - alphas_scaled[k - 1]
            strong_active_set = np.where(strong_active_set)[0]
        else:
            strong_active_set = np.arange(X.shape[1])

        if verbose:
	    print 'Strong active set ', strong_active_set
	    print 'Active set ', active_set
        # solve for the current active set
        beta[k] = shrinkage(X, y, a, beta[k], active_set, max_iter)

        # check KKT in the strong active set
        kkt_violations = False
        for i in strong_active_set:
            tmp = np.dot(X[:, i], y - np.dot(X, beta[k]))
            if beta[k, i] != 0 and not np.allclose(tmp, np.sign(beta[k, i]) * alphas_scaled[k]):
                if i not in active_set:
                    active_set.append(i)
                kkt_violations = True
            if beta[k, i] == 0 and abs(tmp) >= np.abs(alphas_scaled[k]):
                if i not in active_set:
                    active_set.append(i)
                kkt_violations = True

        if not kkt_violations:
            # passed KKT for all variables in strong active set, we're done
            active_set = np.where(beta[k] != 0)[0].tolist()
            if verbose:
                print 'no KKT violated on strong active set'

        # .. recompute with new active set ..
        else:
            if verbose:
                print 'KKT violated on strong active set'
            beta[k] = shrinkage(X, y, a, beta[k], active_set, max_iter)

        # .. check KKT on all predictors ..
        kkt_violations = False
        for i in range(X.shape[1]):
            tmp = np.dot(X[:, i], y - np.dot(X, beta[k]))
            if beta[k, i] != 0 and not np.allclose(tmp, np.sign(beta[k, i]) * alphas_scaled[k]):
                if i not in active_set:
                    active_set.append(i)
                kkt_violations = True
            if beta[k, i] == 0 and abs(tmp) >= np.abs(alphas_scaled[k]):
                if i not in active_set:
                    active_set.append(i)
                kkt_violations = True

        if not kkt_violations:
            # passed KKT for all variables, we're done
            active_set = np.where(beta[k] != 0)[0].tolist()
            print 'no KKT violated on full active set'
        else:
            if verbose:
                print 'KKT violated on full active set'
            beta[k] = shrinkage(X, y, a, beta[k], active_set, max_iter)


    return beta

def check_kkt_lasso(xr, coef, penalty, tol=1e-3):
    """
    Check KKT conditions for Lasso
    xr : X'(y - X coef)
    """
    nonzero = (coef != 0)
    return np.all(np.abs(xr[nonzero] - np.sign(coef[nonzero]) * penalty) < tol) \
        and np.all(np.abs(xr[~nonzero] / penalty) <= 1)


if __name__ == '__main__':
    np.random.seed(0)
    from sklearn import datasets
    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target

#    X = np.random.randn(20, 100)
#    y = np.random.randn(20)
#    X = (X / np.sqrt((X ** 2).sum(0)))    

    #print l1_coordinate_descent(X, y, .001)
    alphas = np.linspace(.1, 2., 10)[::-1]
    coef = l1_path(X, y, alphas, verbose=True)

    for i in range(len(alphas)):
        Xr = np.dot(X.T, y - np.dot(X, coef[i]))
        assert check_kkt_lasso(Xr, coef[i], X.shape[0] * alphas[i])
