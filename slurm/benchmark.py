import os
import contextlib
import itertools as it
import typing as T
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import defopt
import numpy as np
import pandas as pd
import scipy.special as sp
import seaborn as sb
from pyHSICLasso import HSICLasso
from memory_profiler import memory_usage
from tqdm import tqdm


def generate_data(nsamples, ncols, nfeats, seed, transpose_X=False):
    rng = np.random.default_rng(seed)
    X = rng.choice(
        a=[0, 1, 2], p=[0.81, 0.18, 0.01], replace=True, size=(nsamples, ncols)
    )

    coeffs = np.full(nfeats, 5.0)

    yint = X[:, : len(coeffs)] @ coeffs
    p1 = sp.expit(yint - np.mean(yint))
    Y = rng.binomial(1, p1)

    if transpose_X:
        X = X.T.copy()

    return X, Y


def test_model(nsamples, ncols, nfeats, seed, **kwargs):
    X, Y = generate_data(nsamples, ncols, nfeats, seed)
    hsic_lasso = HSICLasso()
    hsic_lasso.input(X, Y)
    with open(os.devnull, "w") as fd, contextlib.redirect_stdout(fd):
        hsic_lasso.classification(num_feat=nfeats, **kwargs)
    return np.mean(hsic_lasso.get_index() == np.arange(nfeats))


def run(
    result_path: Path,
    *,
    nsamples: T.Iterable[int] = (100, 200, 400, 800, 1600),
    ncols: T.Iterable[int] = (100, 200, 400, 800, 1600),
    nfeats: T.Iterable[int] = (2, 4, 8, 16, 32),
    seeds: T.Iterable[int] = (12345,),
    verbose: bool = False,
    n_jobs: int = -1,
):
    """Run a benchmark of HSIC Lasso

    :param result_path: output .csv filename
    :param nsamples: number samples to try
    :param ncols: number of features to try
    :param nfeats: number of relevant features to try
    :param seeds: random seeds to try
    :param verbose: display progress information
    :param n_jobs: number of jobs, use -1 for all available cores
    """

    configs = [
        {
            "nsamples": nsample,
            "ncols": ncol,
            "nfeats": nfeat,
            "seed": seed,
            "n_jobs": 1,
        }
        for nsample, ncol, nfeat, seed in it.product(nsamples, ncols, nfeats, seeds)
    ]

    random.seed(42)
    random.shuffle(configs)

    mem_usage = partial(
        memory_usage, max_usage=True, include_children=True, retval=True
    )

    n_workers = None if n_jobs == -1 else n_jobs

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = executor.map(
            mem_usage, [(test_model, (), params) for params in configs]
        )

        if verbose:
            results = tqdm(results, total=len(configs))

        dset = pd.DataFrame(
            [
                {"mem_mib": mem_mib, "percent_correct": percent_correct, **params}
                for (mem_mib, percent_correct), params in zip(results, configs)
            ]
        )

    result_path.parent.mkdir(exist_ok=True, parents=True)
    dset.to_csv(result_path, index=False)


def plot(input_path: Path, result_path: Path):
    """Plot the results from the benchmark

    :param input_path: input .csv filename
    :param result_path: figures folder name
    """
    dset = pd.read_csv(input_path)
    result_path.mkdir(exist_ok=True, parents=True)

    sb.set()

    grid = sb.relplot(
        x="nsamples",
        y="mem_mib",
        hue="nfeats",
        col="ncols",
        data=dset,
        kind="line",
        col_wrap=3,
        marker="o",
    )
    grid.fig.savefig(result_path / "memory_per_nsamples.png", bbox_inches="tight")

    grid = sb.relplot(
        x="ncols",
        y="mem_mib",
        hue="nfeats",
        col="nsamples",
        data=dset,
        kind="line",
        col_wrap=3,
        marker="o",
    )
    grid.fig.savefig(result_path / "memory_per_ncols.png", bbox_inches="tight")


def main():
    defopt.run([run, plot])
