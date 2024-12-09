#!/usr/bin/env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from builtins import dict, range

from future import standard_library

import numpy as np
from joblib import Parallel, delayed

from .kernel_tools import kernel_delta_norm, kernel_gaussian,kernel_linear,kernel_polynomial

standard_library.install_aliases()

def MKhsic_lasso(X, Y, y_kernel, n_jobs=-1, discarded=0, B=0, M=1):
    """
    Input:
        X      input_data
        Y      target_data
        y_kernel  We employ the Gaussian kernel for inputs. For output kernels,
                  we use the Gaussian kernel for regression cases and
                  the delta kernel for classification problems.
    Output:
        X         matrix of size d x (n * B (or n) * M)
        X_ty      vector of size d x 1
    """
    d, n = X.shape
    dy = Y.shape[0]

    L = compute_kernel(Y, y_kernel, B, M, discarded)
    L = np.reshape(L,(n * B * M,1))

    # Preparing design matrix for HSIC Lars
    result1 = Parallel(n_jobs=n_jobs)([delayed(parallel_compute_kernel)(
        np.reshape(X[k,:],(1,n)), 'Gaussian', k, B, M, n, discarded) for k in range(d)])
    result2 = Parallel(n_jobs=n_jobs)([delayed(parallel_compute_kernel)(
        np.reshape(X[k,:],(1,n)), 'Poly_2', k, B, M, n, discarded) for k in range(d)])
    result3 = Parallel(n_jobs=n_jobs)([delayed(parallel_compute_kernel)(
        np.reshape(X[k,:],(1,n)), 'Linear', k, B, M, n, discarded) for k in range(d)])

    # non-parallel version for debugging purposes
    # result = []
    # for k in range(d):
    #     X = parallel_compute_kernel(X[k, :], x_kernel, k, B, M, n, discarded)
    #     result.append(X)

    result1 = dict(result1)
    result2 = dict(result2)
    result3 = dict(result3)

     # combine column matrices horizontally
    K1 = np.array([result1[k] for k in range(d)]).T
    K2 = np.array([result2[k] for k in range(d)]).T
    K3 = np.array([result3[k] for k in range(d)]).T
    
    K = np.hstack((K1,K2,K3))
    KtL = np.dot(K.T, L)

    return K, KtL, L

def compute_kernel(x, kernel, B = 20, M = 3, discarded = 0):

    d,n = x.shape

    H = np.eye(B, dtype=np.float32) - 1 / B * np.ones(B, dtype=np.float32)
    K = np.zeros(n * B * M, dtype=np.float32)

    # Normalize data
    if kernel != 'Delta' :
        x = (x / (x.std() + 10e-20)).astype(np.float32)

    st = 0
    ed = B ** 2
    index = np.arange(n)
    for m in range(M):
        np.random.seed(m)
        index = np.random.permutation(index)

        for i in range(0, n - discarded, B):
            j = min(n, i + B)

            if kernel == 'Gaussian':
                k = kernel_gaussian(x[:,index[i:j]], x[:,index[i:j]], np.sqrt(d))   
            elif kernel == 'Linear':
                k = kernel_linear(x[:,index[i:j]], x[:,index[i:j]])
            elif kernel == 'Poly_2':
                k = kernel_polynomial(x[:,index[i:j]], x[:,index[i:j]], 2)
            elif kernel == 'Delta':
                k = kernel_delta_norm(x[:,index[i:j]], x[:, index[i:j]])

            k = np.dot(np.dot(H, k), H)

            # Normalize HSIC tr(k*k) = 1
            k = k / (np.linalg.norm(k, 'fro') + 10e-10)
            K[st:ed] = k.flatten()
            st += B ** 2
            ed += B ** 2

    return K


def parallel_compute_kernel(x, kernel, feature_idx, B, M, n, discarded):

    return (feature_idx, compute_kernel(x, kernel, B, M, discarded))
