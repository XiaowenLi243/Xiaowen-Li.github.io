# User Guide

The project repository provides multiple ways to call the Block HSIC Lasso implementation.
This section details how these different levels of abstraction can be used and what is the purpose of each of them.


## Command line interface tool

As exposed in the [main instructions](01_README.md), the `hsic_tools` CLI tool provides the simplest way to run the Block HSIC Lasso solver.
More precisely, the solver is accessible using the `hsic_tools fit` command.
The complete list of available options can be obtained using the `-h` / `--help` option:
```
$ hsic_tools fit --help
usage: hsic_tools fit [-h] [-f {vanilla,accelerated,multikernel}] [-b BLOCK_SIZE] [-r REPEATS]
                      [-n NUM_FEATS] [-e EPS] [--max-outer-iter MAX_OUTER_ITER]
                      [--max-inner-iter MAX_INNER_ITER] [-a ABSTOL] [-d DASK_CONFIGFILE] [-w WORKERS]
                      [--chunksize CHUNKSIZE] [-v | --verbose | --no-verbose]
                      [--checkpoint CHECKPOINT]
                      dataset_path result_path

Fit a block HSIC lasso model to a dataset

positional arguments:
  dataset_path          input dataset as a .h5 file
  result_path           result dataset as a .h5 file

options:
  -h, --help            show this help message and exit
  -f {vanilla,accelerated,multikernel}, --feature-type {vanilla,accelerated,multikernel}
                        function used to compute HSIC features
                        (default: accelerated)
  -b BLOCK_SIZE, --block-size BLOCK_SIZE
                        block size
                        (default: 20)
  -r REPEATS, --repeats REPEATS
                        number of repeats for bagging
                        (default: 3)
  -n NUM_FEATS, --num-feats NUM_FEATS
                        maximum number of active features
                        (default: 1)
  -e EPS, --eps EPS     factor between values in the regularization path
                        (default: 0.95)
  --max-outer-iter MAX_OUTER_ITER
                        maximum number of outer loops for lasso solver
                        (default: 100)
  --max-inner-iter MAX_INNER_ITER
                        maximum number of inner loops for lasso solver
                        (default: 2000)
  -a ABSTOL, --abstol ABSTOL
                        tolerance to assess lasso solver inner loop convergence
                        (default: 1e-06)
  -d DASK_CONFIGFILE, --dask-configfile DASK_CONFIGFILE
                        Dask configuration as a .yaml file
                        (default: None)
  -w WORKERS, --workers WORKERS
                        number of Dask workers
                        (default: 1)
  --chunksize CHUNKSIZE
                        size of chunks used to load the dataset
                        (default: 1)
  -v, --verbose, --no-verbose
                        verbose mode
                        (default: False)
  --checkpoint CHECKPOINT
                        checkpoint .h5 file to save/restore intermediate results
                        (default: None)
```

Most of these options come from the {func}`~hsic_optimization.hsic_lasso.cd_hsic_lasso` and {func}`~hsic_optimization.optimization.cd_hsic_lasso` functions detailed below.
However the `--dask-configfile` and `--workers` options are specific to this interface.

The `--dask-configfile` option let the user provide a `.yaml` file containing the configuration used to start a Dask cluster.
A [LocalCluster](https://docs.dask.org/en/stable/how-to/deploy-dask/single-distributed.html#localcluster) instance is used if this configuration file is not provided.
This file can contain any standard key from the [standard Dask configuration](https://docs.dask.org/en/stable/configuration.html#configuration-reference).

In addition, it needs to contain the information about which Python class to use to start the Dask cluster.
This information is configured using the `hsic_optimization` configuration key.
For example, to start a {class}`~hsic_optimization.cluster.PatchedSLURMCluster` cluster, use the following snippet:
```
hsic_optimization:
    module: hsic_optimization.cluster
    class: PatchedSLURMCluster
```

The `--workers` option of `hsic_tools fit` is used defines the target number of Dask workers.
Computations will start as soon as at least one worker is available.
An [adaptive scheme](https://docs.dask.org/en/stable/how-to/adaptive.html) is used to force the scheduler to maintained alive the target number of workers, respawning new workers when some expire.
This behaviour is particularly desirable with clusters like the [SlurmCluster](https://jobqueue.dask.org/en/latest/generated/dask_jobqueue.SLURMCluster.html), which uses Slurm jobs with a limited lifetime to host the workers.

**Note:** Currently (2021-12-13), the implementation of [SlurmCluster](https://jobqueue.dask.org/en/latest/generated/dask_jobqueue.SLURMCluster.html) contains a bug that prevent it from properly respawning workers when more than one worker is started per job.
The provided {class}`~hsic_optimization.cluster.PatchedSLURMCluster` class fixes this issue.

The `hsic_tools` CLI tool also provides the following tools, useful for testing and debugging:

- The `hsic_tools generate` tool generates synthetic datasets (see the [small-scale](01_README.md#small-scale-example) and [large-scale](01_README.md#large-scale-example) examples).
- The `hsic_tools count-workers` tool counts the number of active workers available to a Dask scheduler.
  This tool is used in the test scripts `slurm/test_one_iteration.sl` and `slurm/test_resilience.sl` to correlate the behaviour of the solver with the number of workers over time. 

The source code of these tools can be found in the {mod}`hsic_optimization.hsic_tools` module.


### Submit a Slurm script

TODO


### Track progress using Dask dashboard

TODO


## Block HSIC Lasso function

The Block HSIC Lasso solver is implemented in the {func}`~hsic_optimization.hsic_lasso.cd_hsic_lasso` function.
This function is a thin wrapper around the {func}`~hsic_optimization.optimization.cd_nn_lasso_path` function (detailed [below](#coordinate-descent-lasso-function)) which solves the Lasso problem using a non-negative coordinate descent solver.

The main role of {func}`~hsic_optimization.hsic_lasso.cd_hsic_lasso` is to prepare the input data `X`.
Depending on the size of the input dataset, different strategies can be used:

- for small datasets, it is advised to pass `X` as a numpy array

TODO example

- for medium datasets, that can fit into memory of the main script, pass `X` as a [Dask array](https://docs.dask.org/en/stable/array.html) with proper chunking on the first dimension

TODO example

- for large datasets, save the dataset on disk and pass a function used to load each raw feature

TODO example

If `X` is a Dask array or a function and a Dask client is passed as an input to {func}`~hsic_optimization.hsic_lasso.cd_hsic_lasso`, computations will be distributed on the corresponding Dask cluster. 

**Note:** Under the hood, the input dataset is transformed as an object implementing the {class}`~hsic_optimization.optimization.Operator` protocol: {class}`~hsic_optimization.optimization.FunctionOp`, {class}`~hsic_optimization.hsic_lasso.DaskArrayOp` or {class}`~hsic_optimization.hsic_lasso.DaskFunctionOp`.
These objects (re-)compute the Block HSIC feature vectors on-the-fly everytime the Lasso solver needs them.


### Implement a new kernel

TODO


## Coordinate descent Lasso function

The {func}`~hsic_optimization.optimization.cd_nn_lasso_path` implements a non-negative coordinate descent solver using covariance updates.

To keep the implementation generic for different types of inputs, this function expecst the feature matrix to be provided as an object following the {class}`~hsic_optimization.optimization.Operator` protocol.
Any object implementing this protocol must provide:

- a `__len__` method to fetch the number features,
- a `matvec` method to compute the dot product of each feature vector with a given vector,
- a `rmatvec` method to compute the dot product of the transposed feature matrix with a given vector,
- a `products` method to compute the dot product between pairs of feature vectors.

TODO explain regularization path and warm-start

TODO explain checkpointing

**Note:** This designs is heavily inspired by Scipy's [LinearOperator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html#scipy.sparse.linalg.LinearOperator) class.


## Known issues and shortcomings

TODO numba speed

TODO small block size discrepancies

TODO discared param

TODO empty set issue?