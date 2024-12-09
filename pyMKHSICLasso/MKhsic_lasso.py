#!/usr/bin/env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function, unicode_literals)
from builtins import dict, range
from future import standard_library
import numpy as np
from joblib import Parallel, delayed
from .kernel_tools import kernel_delta_norm, kernel_gaussian, kernel_linear, kernel_polynomial

# Enable standard library aliases for Python 2/3 compatibility
standard_library.install_aliases()

def MKhsic_lasso(X, Y, y_kernel, n_jobs=-1, discarded=0, B=0, M=1):
    """
    Performs Multi-Kernel HSIC LASSO for feature selection.
    """
    d, n = X.shape

    # Compute kernel for the target data Y
    L = compute_kernel(Y, y_kernel, B, M, discarded)
    L = L.reshape((n * B * M, 1))

    # Prepare design matrices using parallel processing
    kernel_types = ['Gaussian', 'Poly_2', 'Linear']
    results = [
        Parallel(n_jobs=n_jobs)(
            delayed(parallel_compute_kernel)(X[k, :].reshape(1, n), kernel, k, B, M, n, discarded)
            for k in range(d)
        )
        for kernel in kernel_types
    ]

    # Convert results to dictionaries
    kernels = [dict(res) for res in results]

    # Combine column matrices horizontally
    K = np.hstack([np.array([kernels[i][k] for k in range(d)]).T for i in range(len(kernel_types))])

    # Compute K.T @ L
    KtL = np.dot(K.T, L)

    return K, KtL, L


def compute_kernel(x, kernel, B=20, M=3, discarded=0):
    """
    Computes the kernel matrix for a given input.
    """
    d, n = x.shape
    H = np.eye(B, dtype=np.float32) - np.ones((B, B), dtype=np.float32) / B
    K = np.zeros(n * B * M, dtype=np.float32)

    # Normalize data for non-delta kernels
    if kernel != 'Delta':
        x = (x / (x.std() + 1e-20)).astype(np.float32)

    start, end = 0, B ** 2

    for m in range(M):
        np.random.seed(m)
        indices = np.random.permutation(n)

        for i in range(0, n - discarded, B):
            j = min(n, i + B)
            x_batch = x[:, indices[i:j]]

            if kernel == 'Gaussian':
                k = kernel_gaussian(x_batch, x_batch, np.sqrt(d))
            elif kernel == 'Linear':
                k = kernel_linear(x_batch, x_batch)
            elif kernel == 'Poly_2':
                k = kernel_polynomial(x_batch, x_batch, 2)
            elif kernel == 'Delta':
                k = kernel_delta_norm(x_batch, x_batch)
            else:
                raise ValueError(f"Invalid kernel type: {kernel}")

            # Center and normalize kernel matrix
            k = np.dot(H, np.dot(k, H))
            k /= (np.linalg.norm(k, 'fro') + 1e-10)

            K[start:end] = k.flatten()
            start += B ** 2
            end += B ** 2

    return K


def parallel_compute_kernel(x, kernel, feature_idx, B, M, n, discarded):
    """
    Computes the kernel matrix for a specific feature in parallel.
    """
    return feature_idx, compute_kernel(x, kernel, B, M, discarded)
