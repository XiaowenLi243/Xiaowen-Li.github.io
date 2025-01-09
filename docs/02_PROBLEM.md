# Problem Statement

This project aims at re-implementing the Block HSIC Lasso algorithm described in [Climente-González et al., 2019](https://doi.org/10.1093/bioinformatics/btz333).
An official implementation is available in the [pyHSICLasso](https://github.com/riken-aip/pyHSICLasso) Python package.
However this implementation is no suited for very large-scale datasets, with million observations and million features.

The following figure showcases the memory requirement of the reference implementation for a varying number of samples and features (columns in the present case).
The memory requirement seems linear in the number of samples and features, which matches the claims of the original paper.

![](imgs/memory_per_ncols.png)

![](imgs/memory_per_nsamples.png)

**Note:** These results has been created using the Slurm script `slurm/benchmark.sl` and CLI tools provided in the {mod}`hsic_optimization.benchmark` module, available in the project repository.
`nfeats` is the number of active features, `ncols` the total number of features.

Using a linear regression with `ncols` x `nsamples` as predictor, we can try to extrapolate the memory requirements for 1 millions samples and 1 million features.
This would represent 448 TB of memory.

To address this challenge, this project provides tools to trade memory for computation.
The core of the Block HSIC Lasso algorithm is a Lasso problem on very large feature vectors.
As the whole regularization path is computed, we start from no active feature and gradually add newly activate features as the penalization parameter decreases.

To keep the memory requirement low, the following steps are used [^book]:

1. At the beginning of each step of the regularization path, only a subset of all features is selected using the *sequential strong rule*.
   It consists in comparing the dot-product of each feature vector with the current residuals to a threshold:
   
   $$X_i . r > 2 \lambda_l - \lambda_{l-1}$$

   where $X_i$ is a feature vector, $r$ the residuals, $\lambda_l$ the regularization penalty at the step $l$ of the regularization path.

2. Then the Lasso problem is solved using a non-negative coordinate descent solver with covariance updates.

The first step, the *sequential strong rule*, is an embarrassingly parallel task.
Each feature can be computed on-the-fly and multiply with residuals.
This is achieved using [numba](https://numba.pydata.org/) to compute the Block HSIC feature vector quickly and [Dask](https://dask.org/) to run the computation in parallel on many nodes on the HPC platform.

The second step, solving the Lasso problem, only requires the dot-products between the feature vectors of the candidate features.
Thus, a cache of these dot-products is maintained and updated with the newly selected features.
In addition, to accelerate the convergence of the Lasso solver, results from the previous problem is used to warm-start the solver.

[^book]: Section 5.4.2 of *Statistical Learning with Sparsity: The Lasso and Generalizations*, 2015, by T. Hastie, R. Tibshirani and M. Wainwright (ISBN 1498712169).