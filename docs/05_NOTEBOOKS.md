# Notebooks

The repository contains a set of notebooks exploring options to accelerate the
HSIC lasso algorithm:

- [01_hsic_explore.ipynb](notebooks/01_hsic_explore.ipynb) is a basic exploration of the official implementation with synthetic data,
- [02_hsic_profile.ipynb](notebooks/02_hsic_profile.ipynb) contains an analysis of the memory usage of the official implementation,
- [03a_feature_optimization.ipynb](notebooks/03a_feature_optimization.ipynb) explores how fast computing columns of the HSIC lasso feature matrix can be,
- [03b_feature_optimization.ipynb](notebooks/03b_feature_optimization.ipynb) tests additional strategies to accelerate the HSIC lasso feature matrix creation,
- [04_coordinate_descent.ipynb](notebooks/04_coordinate_descent.ipynb) details the characteristics of coordinate descent solver for HSIC lasso,
- [05_dask_parallelization.ipynb](notebooks/05_dask_parallelization.ipynb) presents the performances of the distributed version of the CD solver for HSIC lasso,
- [06_dask_graphs.ipynb](notebooks/06_dask_graphs.ipynb) explores different patterns to project the HSIC lasso features on the target data using Dask,
- [07_large_datasets.ipynb](notebooks/07_large_datasets.ipynb) profiles the code when using a larger dataset loaded from disk,
- [08_one_iteration_timings.ipynb](notebooks/08_one_iteration_timings.ipynb) analyses timings results from running one iteration of HSIC lasso on a very large dataset (using [test_one_iteration.sl](slurm/test_one_iteration.sl)),
- [09_multikernel.ipynb](notebooks/09_multikernel.ipynb) illustrates how to use the multi-kernel feature, combining a Gaussian kernel with a linear kernel and a polynomial kernel (order 2).
