#!/usr/bin/env python
# coding: utf-8

"""
Script for creating custom plots such as dendrograms and MKHSICLasso paths.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import range
from future import standard_library

# Install Python 3 compatibility
standard_library.install_aliases()

# Import necessary libraries
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import dendrogram

# Define a custom colormap for microarray visualization
microarray_cmap = LinearSegmentedColormap(
    'microarray',
    {
        'red': [(0.0, 1.0, 1.0), (0.5, 0.2, 0.2), (1.0, 0.0, 0.0)],
        'green': [(0.0, 0.0, 0.0), (0.5, 0.2, 0.2), (1.0, 1.0, 1.0)],
        'blue': [(0.0, 0.0, 0.0), (0.5, 0.2, 0.2), (1.0, 0.0, 0.0)],
    }
)

def plot_dendrogram(linkage, feature_names, filepath):
    """
    Plots a dendrogram based on hierarchical clustering.

    Parameters:
    - linkage: The linkage matrix for hierarchical clustering.
    - feature_names: A list of feature names to label the dendrogram leaves.
    - filepath: Path to save the dendrogram plot.
    """
    plt.figure(figsize=(10, 7))
    dendrogram(linkage, labels=feature_names)
    plt.title("Dendrogram")
    plt.xlabel("Feature Index")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.clf()

def plot_path(path, beta, active_indices, filepath):
    """
    Plots the MKHSICLasso path showing coefficients for different values of lambda.

    Parameters:
    - path: A matrix where each row represents the path of a coefficient.
    - beta: Coefficient values (not used in the current plot implementation).
    - active_indices: List of active feature indices (zero-based).
    - filepath: Path to save the plot.
    """
    t = path.sum(axis=0)
    plt.figure(figsize=(10, 7))
    plt.title("MKHSICLasso Result")
    plt.xlabel("Lambda")
    plt.ylabel("Coefficients")
    
    for ind in active_indices:
        plt.plot(t, path[ind, :], label=f"Feature {ind + 1}")
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath)
    plt.clf()
