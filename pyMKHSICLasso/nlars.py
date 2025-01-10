#!/usr/bin/env python
# coding: utf-8

"""
Python implementation of the Nonnegative LARS (Least Angle Regression and Selection) solver.
This script solves the problem:
    argmin_beta (1/2) ||y - X*beta||_2^2  subject to beta >= 0.

It also computes the entire regularization path for the LASSO problem:
    min (1/2) ||y - X*beta||_2^2 + lambda * |beta|_1  subject to beta >= 0.

"""

from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import range
from future import standard_library
import numpy as np
from scipy.sparse import lil_matrix
import logging

# Enable Python 3 compatibility
standard_library.install_aliases()

def nlars(X, X_ty, num_feat, max_neighbors):
    """
    Nonnegative LARS solver for feature selection.

    Parameters:
    - X: (n, d) matrix, feature matrix.
    - X_ty: (d, 1) vector, X^T * y.
    - num_feat: Number of features to extract.
    - max_neighbors: Maximum number of neighbors for each selected feature.

    Returns:
    - A_all: List of active feature sets at each step.
    - path_final: Full regularization path for beta.
    - beta: (d, 1) vector, final solution for beta.
    - A_sorted: Selected features, sorted by their beta values.
    - lam_final: Regularization values at each step.
    - A_neighbors: Neighbors of the selected features.
    - A_neighbors_score: Scores of the neighbors.
    """
    n, d = X.shape  # Number of samples (n) and features (d)

    # Initialization
    A = []  # Active set of selected features
    A_all = []  # Records active features at each step
    A_neighbors = []  # Neighbors of selected features
    A_neighbors_score = []  # Scores of neighbors
    beta = np.zeros((d, 1), dtype=np.float32) 
    path_all = lil_matrix((d, 4 * d))  # Path matrix
    lam = np.zeros((1, 4 * d))  # Regularization values

    I = list(range(d))  # inactive set
    XtXbeta = np.dot(X.T, np.dot(X, beta))
    c = X_ty - XtXbeta  # Residual correlation
    j = c.argmax()
    C = c[j]
    A.append(I[j])  # Add feature with max correlation to active set
    I.remove(I[j])

    if len(C) == 0:
        lam[0] = 0
    else:
        lam[0, 0] = C[0]

    k = 0
    while sum(c[A]) / len(A) >= 1e-16 and len(A) < num_feat + 1:
        s = np.ones((len(A), 1), dtype=np.float32)

        try:
            # Solve for direction vector w
            w = np.linalg.solve(np.dot(X[:, A].T, X[:, A]), s)
        except np.linalg.LinAlgError:
            # Add noise if matrix is singular
            X_noisy = X[:, A] + np.random.normal(0, 1e-10, X[:, A].shape)
            w = np.linalg.solve(np.dot(X_noisy.T, X_noisy), s)

        XtXw = np.dot(X.T, np.dot(X[:, A], w))

        # Compute step sizes
        gamma1 = (C - c[I]) / (XtXw[A[0]] - XtXw[I])
        gamma2 = -beta[A] / w
        gamma3 = c[A[0]] / XtXw[A[0]]
        gamma = np.concatenate((np.concatenate((gamma1, gamma2)), [gamma3]))

        gamma[gamma <= 1e-16] = np.inf
        t = gamma.argmin()
        mu = min(gamma)

        beta[A] = beta[A] + mu * w

        # Check if we need to remove a feature
        if t >= len(gamma1) and t < (len(gamma1) + len(gamma2)):
            lasso_cond = True
            j = t - len(gamma1)
            I.append(A[j])
            A.remove(A[j])
        else:
            lasso_cond = False

        # Update residuals and active set
        XtXbeta = np.dot(X.T, np.dot(X, beta))
        c = X_ty - XtXbeta
        j = np.argmax(c[I])
        C = max(c[I])

        k += 1
        path_all[:, k] = beta

        lam[0, k] = C[0] if len(C) != 0 else 0

        if not lasso_cond:
            A.append(I[j])
            I.remove(I[j])

        A_all.append(A[:])

    # Truncate to desired number of features
    if len(A) > num_feat:
        A.pop()

    # Sort active set by beta values
    A_sorted = sorted(A, key=lambda i: beta[i], reverse=True)

    # Find neighbors of selected features
    XtXA = np.dot(X.T, X[:, A_sorted])
    num_neighbors = max_neighbors + 1
    for i in range(len(A_sorted)):
        tmp = XtXA[:, i]
        sort_index = np.argsort(tmp)[::-1]
        A_neighbors.append(sort_index[:num_neighbors])
        A_neighbors_score.append(tmp[sort_index[:num_neighbors]])

    path_final = path_all[:, :k + 1].toarray()
    lam_final = lam[:, :k + 1]

    return A_all, path_final, beta, A_sorted, lam_final, A_neighbors, A_neighbors_score


def nlars_SSR_KKT(X, X_ty, num_feat, max_neighbors):
    """
    Nonnegative LARS solver for feature selection with Sequential Strong Rule (SSR) and KKT condition check.

    Parameters:
    - X: (n, d) matrix, feature matrix.
    - X_ty: (d, 1) vector, X^T * y.
    - num_feat: Number of features to extract.
    - max_neighbors: Maximum number of neighbors for each selected feature.

    Returns:
    - A_all: List of active feature sets at each step.
    - path_final: Full regularization path for beta.
    - beta: (d, 1) vector, final solution for beta.
    - A_sorted: Selected features, sorted by their beta values.
    - lam_final: Regularization values at each step.
    - A_neighbors: Neighbors of the selected features.
    - A_neighbors_score: Scores of the neighbors.
    """
    import numpy as np
    from scipy.sparse import lil_matrix

    n, d = X.shape  # Number of samples (n) and features (d)

    # Initialization
    A = []  # Active set of selected features
    A_all = []  # Records active features at each step
    A_neighbors = []  # Neighbors of selected features
    A_neighbors_score = []  # Scores of neighbors
    beta = np.zeros((d, 1), dtype=np.float32) 
    path_all = lil_matrix((d, 4 * d))  # Path matrix
    lam = np.zeros((1, 4 * d))  # Regularization values

    I = list(range(d))  # Inactive set
    XtXbeta = np.dot(X.T, np.dot(X, beta))
    c = X_ty - XtXbeta  # Residual correlation

    # Sequential Strong Rule (SSR) Initialization
    SSR_threshold = np.inf

    while len(A) < num_feat:
        # Apply SSR to filter inactive features
        filtered_I = [j for j in I if abs(c[j]) >= SSR_threshold]
        if not filtered_I:
            break  # Stop if no eligible features remain

        j = np.argmax([c[idx] for idx in filtered_I])
        selected_feature = filtered_I[j]

        # Update SSR threshold for the next iteration
        SSR_threshold = 2 * abs(c[selected_feature]) - lam[0, len(A)]

        # Add selected feature to active set
        A.append(selected_feature)
        I.remove(selected_feature)

        # Solve for direction vector w
        s = np.ones((len(A), 1), dtype=np.float32)
        try:
            w = np.linalg.solve(np.dot(X[:, A].T, X[:, A]), s)
        except np.linalg.LinAlgError:
            X_noisy = X[:, A] + np.random.normal(0, 1e-10, X[:, A].shape)
            w = np.linalg.solve(np.dot(X_noisy.T, X_noisy), s)

        # Compute step size
        XtXw = np.dot(X.T, np.dot(X[:, A], w))
        gamma = np.min([
            (c[j] - c[I]) / (XtXw[j] - XtXw[I]) for j in A if XtXw[j] != 0
        ] + [np.inf])

        # Update beta and residuals
        beta[A] += gamma * w
        XtXbeta = np.dot(X.T, np.dot(X, beta))
        c = X_ty - XtXbeta

        # KKT condition check
        for idx in I:
            if c[idx] > max(c[A]):
                raise ValueError(f"KKT condition violated: Inactive feature {idx} has larger correlation than active set.")

        # Store active set and regularization path
        A_all.append(A[:])
        path_all[:, len(A)] = beta.flatten()
        lam[0, len(A)] = max(abs(c[A]))

    # Sort active set by beta values
    A_sorted = sorted(A, key=lambda i: beta[i], reverse=True)

    # Find neighbors of selected features
    XtXA = np.dot(X.T, X[:, A_sorted])
    num_neighbors = max_neighbors + 1
    for i in range(len(A_sorted)):
        tmp = XtXA[:, i]
        sort_index = np.argsort(tmp)[::-1]
        A_neighbors.append(sort_index[:num_neighbors])
        A_neighbors_score.append(tmp[sort_index[:num_neighbors]])

    path_final = path_all[:, :len(A) + 1].toarray()
    lam_final = lam[:, :len(A) + 1]

    return A_all, path_final, beta, A_sorted, lam_final, A_neighbors, A_neighbors_score


