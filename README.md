# HSIC Lasso optimization

This project aims at optimizing the memory consumption of HSIC Lasso to scale it to very large datasets.


## Installation

First make sure you have [Git](https://git-scm.com/downloads), [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and `make` installed on your computer.

*On NeSI, make sure to first load the Miniconda module using:*

```
module purge && module load Miniconda3/23.10.0-1
export PYTHONNOUSERSITE=1
source $(conda info --base)/etc/profile.d/conda.sh
```

Then open a terminal and clone this repository:

```
git clone https://github.com/XiaowenLi243/MKHSICLasso.git
```

Use `make` to create a conda environment and install all dependencies:

```
cd hsic_optimization
make venv
```

The conda environment is created in the local `venv` folder.

Now you can run the provided notebooks using the `hsic_optimization` kernel, or use the scripts detailed in the next section.

**Note:** You can delete the conda environment and jupyter kernel using `make clean`.


## Getting Started

The provided `hsic_tools` script can be used to generate fake data and fit a HSIC lasso model for data saved on disk.

*On NeSI, make sure to first load the Miniconda module using:*

```
module purge && module load Miniconda3/23.10.0-1
export PYTHONNOUSERSITE=1
source $(conda info --base)/etc/profile.d/conda.sh
```

Before using the script, activate the conda environment:

```
conda activate ./venv
```

All options of the script can be accessed using the `-h`/`--help` options:

```
hsic_tools generate --help
hsic_tools fit --help
```


## Small-scale example

For a small-scale test, create a test dataset with 100 features and 1000 samples, saved in a HDF5 file, as follow:

```
hsic_tools generate --feats 100 --samples 1000 data/test_dataset_100_by_1000.h5
```

then fit the HSIC lasso model to it:

```
hsic_tools fit --num-feats 10 --workers 4 --verbose data/test_dataset_100_by_1000.h5 results.h5
```

using 4 Dask workers and stopping when 10 active features have been found.

**Note:** The `--checkpoint` option let you define a hdf5 file to save intermediate results and restart your computations from the last save state.


## Large-scale example

For more realistic results, you can generate a larger test dataset with 10,000 features and 1,000,000 samples (default value):

```
hsic_tools generate --feats 10000 data/test_dataset_10000_by_1000000.h5
```

and fit it using a Slurm-based Dask cluster:

```
hsic_tools fit data/test_dataset_10000_by_1000000.h5 results.h5 \
    --dask-configfile config/dask_slurm_default.yaml \
    --verbose --workers 100 --chunksize 10 --num-feats 10
```

The `config/dask_slurm_default.yaml` file configures the Dask cluster, using the [SlurmCluster](https://jobqueue.dask.org/en/latest/generated/dask_jobqueue.SLURMCluster.html) backend.
Using this configuration, supporting Slurm jobs contain 4 workers, each one using 2GB of memory and lasting 2 hours.


## Continuous features and target example

The `hsic_tools` script also supports continuous data for features and/or target data.

To illustrate this, let's generate a small dataset with continuous features and target data:

```
hsic_tools generate --samples 100 --feats 1000 \
    --continuous-features --continuous-target \
    data/test_dataset_100_by_1000_continuous.h5
```

then fit the HSIC lasso model using:

```
hsic_tools fit data/test_dataset_100_by_1000_continuous.h5 results.h5 \
    --verbose --num-feats 10 --workers 4 --eps 0.99 \
    --y-kernel Gaussian --feature-type gaussian_accel
```

where

- `--y-kernel Gaussian` enables a Gaussian kernel for the target data,
- `--feature-type gaussian_accel` selects an accelerated Gaussian kernel supporting continuous features.

**Note:** `accelerated` and `multikernel` feature types do not support continuous data, only discrete values.

**Note:** `--eps 0.99` set the scale factor between each value of the regularisation path to 0.99.


## File formats

The `hsic_tools fit` script expects input data to be saved as a HDF5 file with the following datasets:

- `X`, the input feature matrix as a 2D array of size `n_features` x `n_samples`,
- `Y`, the target data as a 1D array of size `n_samples`.

*Note: Save `X` as a compressed array of `uint8` to save disk space (see the [h5py package documentation](https://docs.h5py.org/en/stable/high/dataset.html#filter-pipeline) for an example).*

The results from `hsic_tools fit` are also save in a HDF5 file containing:

- `lams`, values of $\lambda$ in the regularization path, as a 1D array of size `n_lams`,
- `path`, coefficients estimated for each `lams` value, as a 2D array of size `n_lams` x `n_features`,
- `beta`, coefficients values for the final penalization value, as a 1D array of size `n_features`.


## Documentation

The complete documentation can be generated using [Sphinx](https://www.sphinx-doc.org).

Use `make` to generate an html version:

```
make doc
```

or a pdf version:

```
make latexpdf
```

**Note:** You need to install LaTeX to generate the pdf version.


## References

- [pyHSICLasso](https://github.com/riken-aip/pyHSICLasso), Python package used as reference implementation of the HSIC Lasso algorithm
- Climente-González, H., Azencott, C-A., Kaski, S., & Yamada, M., [Block HSIC Lasso: model-free biomarker detection for ultra-high dimensional data.](https://doi.org/10.1093/bioinformatics/btz333) Bioinformatics, Volume 35, Issue 14, July 2019, Pages i427–i435
- Hastie, T., Tibshirani, R., & Wainwright, M., [Statistical Learning with Sparsity: The Lasso and Generalizations](https://hastie.su.domains/StatLearnSparsity_files/SLS_corrected_1.4.16.pdf). Chapman & Hall/CRC. *Sections 5.4 "Coordinate Descent" and 5.10 "Screening Rules"*
- Tibshirani, R., Bien, J., Friedman, J., Hastie, T., Simon, N., Taylor, J., & Tibshirani, R. J. (2012). Strong rules for discarding predictors in lasso-type problems. Journal of the Royal Statistical Society. Series B, Statistical methodology, 74(2), 245–266. https://doi.org/10.1111/j.1467-9868.2011.01004.x
