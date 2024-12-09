#!/usr/bin/env python
# coding: utf-8

"""
Kernel functions for machine learning tasks.

This script implements various kernel functions, including delta norm, delta, Gaussian, linear, 
and polynomial kernels, which are commonly used in machine learning for computing similarity matrices.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
from future import standard_library
import numpy as np

# Enable Python 3 compatibility
standard_library.install_aliases()

def kernel_delta_norm(X_in_1, X_in_2):
    """
    Computes the normalized delta kernel between two datasets.
    """
    n_1, n_2 = X_in_1.shape[1], X_in_2.shape[1]
    K = np.zeros((n_1, n_2))
    unique_values = np.unique(X_in_1)
    
    for value in unique_values:
        # Compute normalization constants for each unique value
        c_1 = np.sqrt(np.sum(X_in_1 == value))
        c_2 = np.sqrt(np.sum(X_in_2 == value))
        
        # Find indices where the value appears in both inputs
        ind_1 = np.where(X_in_1 == value)[1]
        ind_2 = np.where(X_in_2 == value)[1]
        
        # Update kernel matrix
        K[np.ix_(ind_1, ind_2)] = 1 / (c_1 * c_2)
    
    return K


def kernel_delta(X_in_1, X_in_2):
    """
    Computes the unnormalized delta kernel between two datasets.
    """
    n_1, n_2 = X_in_1.shape[1], X_in_2.shape[1]
    K = np.zeros((n_1, n_2))
    unique_values = np.unique(X_in_1)
    
    for value in unique_values:
        # Find indices where the value appears in both inputs
        ind_1 = np.where(X_in_1 == value)[1]
        ind_2 = np.where(X_in_2 == value)[1]
        
        # Update kernel matrix
        K[np.ix_(ind_1, ind_2)] = 1
    
    return K


def kernel_gaussian(X_in_1, X_in_2, sigma):
    """
    Computes the Gaussian (RBF) kernel between two datasets.
    """
    n_1, n_2 = X_in_1.shape[1], X_in_2.shape[1]
    X_in_12 = np.sum(np.power(X_in_1, 2), axis=0)
    X_in_22 = np.sum(np.power(X_in_2, 2), axis=0)
    
    # Compute squared Euclidean distance matrix
    dist_2 = (
        np.tile(X_in_22, (n_1, 1)) + 
        np.tile(X_in_12, (n_2, 1)).T - 
        2 * np.dot(X_in_1.T, X_in_2)
    )
    
    # Apply Gaussian kernel
    K = np.exp(-dist_2 / (2 * np.power(sigma, 2)))
    return K


def kernel_linear(X_in_1, X_in_2):
    """
    Computes the linear kernel between two datasets.
    """
    return X_in_1.T.dot(X_in_2)


def kernel_polynomial(X_in_1, X_in_2, degree):
    """
    Computes the polynomial kernel between two datasets.

    """
    return np.power(X_in_1.T.dot(X_in_2), degree)
