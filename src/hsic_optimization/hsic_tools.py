import logging
import time
import datetime
from pathlib import Path
from importlib import import_module
from typing import Optional, Literal

import yaml
import defopt
import numpy as np
import scipy.special as sp
import h5py
import tqdm
import dask
from dask.distributed import Client, LocalCluster

from hsic_optimization.hsic_lasso import (
    cd_hsic_lasso,
    feature_accel,
    feature_vanilla,
    feature_multikernel,
    feature_accel_gaussian,
    feature_accel_multikernel,
)


def generate(
    result_file: Path,
    *,
    samples: int = 1_000_000,
    feats: int = 1_000,
    active: int = 20,
    coeff_value: float = 5.0,
    continuous_features: bool = False,
    continuous_target: bool = False,
    std_noise: float = 1.0,
    seed: int = 12345,
):
    """Generate a fake dataset and save it in a compressed .h5 file

    :param result_file: filename of the resulting .hdf5 file
    :param samples: number of samples
    :param feats: number of features
    :param active: number of active features (non-zero coefficient)
    :param coeff_value: coefficient value for active features
    :param continuous_features: use continuous, or categorical, feature data
    :param continuous_target: generate continous, or categorical, target data
    :param std_noise: Gaussian noise standard deviation for continuous target data
    :param seed: random generator seed
    """

    rng = np.random.default_rng(seed)

    result_file.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(result_file, "w") as fd:
        coeffs = np.zeros(feats)
        coeffs[:active] = coeff_value
        fd["coeffs"] = coeffs

        X = fd.create_dataset(
            "X",
            (feats, samples),
            chunks=(1, samples),
            dtype=np.float32 if continuous_features else np.uint8,
            compression="gzip",
            fillvalue=0,
        )

        yint = np.zeros(samples)
        for i in tqdm.trange(feats):
            if continuous_features:
                x_i = rng.normal(loc=0, scale=1, size=(samples,))
            else:
                x_i = rng.choice(
                    a=[0, 1, 2], p=[0.81, 0.18, 0.01], replace=True, size=(samples,)
                )
            X[i] = x_i
            yint += coeffs[i] * x_i

        if continuous_target:
            Y = yint + rng.normal(loc=0, scale=std_noise, size=(samples,))
        else:
            p1 = sp.expit(yint - np.mean(yint))
            Y = rng.binomial(1, p1)
        fd["Y"] = Y


def count_workers(scheduler_address: str, period: float):
    """Print the number of active Dask workers associated with a scheduler

    :param scheduler_address: address of the Dask sceduler
    :param period: sampling period, in seconds
    """
    client = Client(scheduler_address)
    print("date,n_workers", flush=True)
    while True:
        n_workers = len(client.scheduler_info()["workers"])
        print(datetime.datetime.now(), n_workers, sep=",", flush=True)
        time.sleep(period)


FeatureMap = {
    "vanilla": feature_vanilla,
    "accelerated": feature_accel,
    "multikernel": feature_multikernel,
    "gaussian_accel": feature_accel_gaussian,
    "multikernel_accel": feature_accel_multikernel,
}


def fit(
    dataset_path: Path,
    result_path: Path,
    *,
    feature_type: Literal[
        "vanilla", "accelerated", "multikernel", "gaussian_accel", "multikernel_accel"
    ] = "accelerated",
    y_kernel: Literal["Delta", "Gaussian"] = "Delta",
    block_size: int = 20,
    repeats: int = 3,
    num_feats: int = 1,
    eps: float = 0.95,
    max_outer_iter: int = 100,
    max_inner_iter: int = 2000,
    abstol: float = 1e-6,
    dask_configfile: Optional[Path] = None,
    workers: int = 1,
    chunksize: int = 1,
    verbose: bool = False,
    checkpoint: Optional[Path] = None,
):
    """Fit a block HSIC lasso model to a dataset

    :param dataset_path: input dataset as a .h5 file
    :param result_path: result dataset as a .h5 file
    :param feature_type: function used to compute HSIC features
    :param y_kernel: kernel used for target data
    :param block_size: block size
    :param repeats: number of repeats for bagging
    :param num_feats: maximum number of active features
    :param eps: factor between values in the regularization path
    :param max_outer_iter: maximum number of outer loops for lasso solver
    :param max_inner_iter: maximum number of inner loops for lasso solver
    :param abstol: tolerance to assess lasso solver inner loop convergence
    :param dask_configfile: Dask configuration as a .yaml file
    :param workers: number of Dask workers
    :param chunksize: size of chunks used to load the dataset
    :param verbose: verbose mode
    :param checkpoint: checkpoint .h5 file to save/restore intermediate results
    """
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s :: %(name)s :: %(levelname)s :: %(message)s",
    )
    logger = logging.getLogger(__name__)

    # select the feature function
    feature = FeatureMap[feature_type]
    logger.info(f"use {feature} function to compute HSIC features")

    # load Dask configuration
    if dask_configfile is not None:
        dask_config = yaml.safe_load(dask_configfile.read_text())
        dask.config.update(dask.config.config, dask_config)
        logger.info("Dask configuration loaded")

    # find the cluster class, if any configured
    try:
        module = import_module(dask.config.get("hsic_optimization.module"))
        Cluster = getattr(module, dask.config.get("hsic_optimization.class"))
    except KeyError:
        logger.warning(
            "Cluster class not found in Dask configuration, using LocalCluster."
        )
        Cluster = LocalCluster

    with Cluster() as cluster, Client(cluster) as client:
        logger.info(f"Scheduler address: {cluster.scheduler_address}")
        logger.info(f"Dashboard address: {cluster.dashboard_link}")

        cluster.adapt(minimum=workers, maximum=workers)
        logger.info("Dask cluster started, waiting for workers")
        client.wait_for_workers(1)

        logger.info("loading target data")
        with h5py.File(dataset_path, "r") as fd:
            Y = np.array(fd["Y"])
            n_features = len(fd["X"])

        def raw_feature(i):
            with h5py.File(dataset_path, "r") as fd:
                return fd["X"][i]

        # replicate features for the multi-kernel
        n_repeat = 3 if feature_type.startswith("multikernel") else 1

        logger.info("HSIC lasso started")
        path, beta, lams = cd_hsic_lasso(
            raw_feature,
            Y,
            B=block_size,
            M=repeats,
            num_feat=num_feats,
            feature=feature,
            n_repeat=n_repeat,
            n_features=n_features,
            client=client,
            chunksize=chunksize,
            eps=eps,
            max_outer_iter=max_outer_iter,
            max_inner_iter=max_inner_iter,
            abstol=abstol,
            checkpoint=checkpoint,
        )
        logger.info("HSIC lasso finished")

    logger.info(f"saving results to {result_path}")
    with h5py.File(result_path, "w") as fd:
        fd["beta"] = beta
        fd["lams"] = lams

        fd.create_dataset(
            "path", path.shape, dtype=path.dtype, fillvalue=0, compression="gzip"
        )
        for i, row in enumerate(path):
            fd["path"][i] = row.toarray()
    logger.info("results saved")


def main():
    defopt.run([generate, fit, count_workers])
