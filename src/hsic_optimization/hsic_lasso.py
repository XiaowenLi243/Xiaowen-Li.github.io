import warnings
from typing import Callable, Iterable

import numpy as np
import scipy.sparse as sp
from numba import njit
from pyHSICLasso.hsic_lasso import compute_kernel, hsic_lasso
from pyHSICLasso.kernel_tools import kernel_gaussian
from pyHSICLasso.nlars import nlars
import dask.array as da
from dask.distributed import Client, as_completed
from tlz import partition_all

from hsic_optimization.optimization import cd_nn_lasso_path, FunctionOp


def vanilla_hsic_lasso(X, Y, B, M, num_feat, y_kernel="Delta"):
    K, KtL, L = hsic_lasso(X, Y[None, :], y_kernel, B=B, M=M)
    scale = B / (Y.size * M)
    K *= np.sqrt(scale)
    KtL *= scale
    path, beta, A, lams, _, _ = nlars(K, KtL, num_feat, 0)
    return path, beta, A, lams


def feature_vanilla(i, x, B, M, discarded, scale):
    x = x.reshape(1, -1)
    return compute_kernel(x, "Gaussian", B=B, M=M, discarded=discarded) * scale


@njit(nogil=True, cache=True)
def kernel_lookup_(x, lookup):
    n = len(x)
    out = np.empty((n, n), dtype=lookup.dtype)
    for i in range(n):
        for j in range(i, n):
            res = lookup[x[i], x[j]]
            out[i, j] = res
            out[j, i] = res
    return out


@njit(nogil=True, cache=True)
def compute_lookup_kernel_(x, lookup, B=0, M=1, discarded=0):
    n = len(x)
    H = np.eye(B, dtype=np.float32) - np.full((B, B), 1 / B, dtype=np.float32)
    K = np.zeros(n * B * M, dtype=np.float32)

    st = 0
    ed = B**2
    index = np.arange(n)
    for m in range(M):
        np.random.seed(m)
        index = np.random.permutation(index)
        X_k = x[index]

        for i in range(0, n - discarded, B):
            j = min(n, i + B)

            k = kernel_lookup_(X_k[i:j], lookup)
            k = (H @ k) @ H
            k = k / (np.sqrt(np.sum(k**2)) + 1e-9)

            K[st:ed] = k.ravel()
            st += B**2
            ed += B**2

    return K


def feature_accel(i, x, B, M, discarded, scale):
    arr = np.array([[0, 1, 2]])
    x_std = x.std() + 1e-19
    lookup = kernel_gaussian(arr / x_std, arr / x_std, 1.0).astype(np.float32)
    return compute_lookup_kernel_(x, lookup, B=B, M=M, discarded=discarded) * scale


def kernel_linear(X_in_1, X_in_2):
    K = X_in_1.T.dot(X_in_2)
    return K


def kernel_polynomial(X_in_1, X_in_2, degree):
    K = X_in_1.T.dot(X_in_2) ** degree
    return K


def feature_multikernel(i, x, B, M, discarded, scale):
    arr = np.array([[0, 1, 2]])
    x_std = x.std() + 1e-19

    kernel_index = i % 3

    if kernel_index == 0:
        lookup = kernel_gaussian(arr / x_std, arr / x_std, 1.0).astype(np.float32)

    if kernel_index == 1:
        lookup = kernel_linear(arr / x_std, arr / x_std).astype(np.float32)

    if kernel_index == 2:
        lookup = kernel_polynomial(arr / x_std, arr / x_std, 2).astype(np.float32)

    return compute_lookup_kernel_(x, lookup, B=B, M=M, discarded=discarded) * scale


@njit(nogil=True, cache=True)
def kernel_gaussian_(x):
    n = len(x)
    out = np.empty((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i, n):
            res = np.exp(-((x[i] - x[j]) ** 2) * 0.5)
            out[i, j] = res
            out[j, i] = res
    return out


@njit(nogil=True, cache=True)
def kernel_linear_(x):
    K = x[:, None] * x
    return K


@njit(nogil=True, cache=True)
def kernel_quadratic_(x):
    K = kernel_linear_(x) ** 2
    return K


@njit(nogil=True, cache=True)
def compute_kernel_(x, kernel, B=0, M=1, discarded=0):
    n = len(x)
    H = np.eye(B, dtype=np.float32) - np.full((B, B), 1 / B, dtype=np.float32)
    K = np.zeros(n * B * M, dtype=np.float32)

    st = 0
    ed = B**2
    index = np.arange(n)
    for m in range(M):
        np.random.seed(m)
        index = np.random.permutation(index)
        X_k = x[index]

        for i in range(0, n - discarded, B):
            j = min(n, i + B)

            k = kernel(X_k[i:j]).astype(np.float32)
            k = (H @ k) @ H
            k = k / (np.sqrt(np.sum(k**2)) + 1e-9)

            K[st:ed] = k.ravel()
            st += B**2
            ed += B**2

    return K


def feature_accel_gaussian(i, x, B, M, discarded, scale):
    x_std = x.std() + 1e-19
    K = compute_kernel_(x / x_std, kernel_gaussian_, B=B, M=M, discarded=discarded)
    return K * scale


def feature_accel_multikernel(i, x, B, M, discarded, scale):
    x_std = x.std() + 1e-19

    kernel_index = i % 3

    if kernel_index == 0:
        K = compute_kernel_(x / x_std, kernel_gaussian_, B=B, M=M, discarded=discarded)

    elif kernel_index == 1:
        K = compute_kernel_(x / x_std, kernel_linear_, B=B, M=M, discarded=discarded)

    elif kernel_index == 2:
        K = compute_kernel_(x / x_std, kernel_quadratic_, B=B, M=M, discarded=discarded)

    return K * scale


def project_block(block, y, features):
    return np.array([features(x) @ y for x in block])


def feature_product(coef, i, features):
    return features(i) * coef


def feature_dotproduct(j, i, features):
    return j, i, features(j) @ features(i)


class DaskFunctionOp:
    def __init__(
        self,
        features: Callable[[int], np.ndarray],
        n_features: int,
        client: Client,
        chunksize: int,
    ):
        self.features = features
        self.n_features = n_features
        self.client = client
        self.chunksize = chunksize

        self.indices_cache = set()
        dtype = features(0).dtype
        self.products_cache = sp.dok_matrix((n_features, n_features), dtype=dtype)

    def __len__(self) -> int:
        return self.n_features

    def matvec(self, y: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)

            # Ensure resilience in case all workers die, as the task (hosting the
            # data) is saved in scheduler memory and will be rerun if needed.
            #
            # This is not the case of "client.scatter" that doesn't save the
            # original data after scattering: if all workers die, the corresponding
            # task becomes invalid and the whole tasks graphs fails.
            #
            # Note that "client.submit(lambda: y)" would trigger a memory leak,
            # whereas this version doesn't (but triggers a warning, silenced here).

            y_dist = self.client.submit(lambda x: x, y)

        self.client.replicate([y_dist])

        chunks_idx = list(partition_all(self.chunksize, range(self.n_features)))
        futures = self.client.map(
            project_block, chunks_idx, y=y_dist, features=self.features
        )

        results = {}
        for future, result in as_completed(futures, with_results=True):
            results[future.key] = result
            self.client.cancel(future)

        inner_products = [results[future.key] for future in futures]
        return np.concatenate(inner_products)

    def rmatvec(self, y: np.ndarray, indices: Iterable[int]) -> np.ndarray:
        futures = self.client.map(feature_product, y, indices, features=self.features)
        result = 0
        for future in as_completed(futures):
            result += future.result()
            self.client.cancel(future)
        return result

    def _update_caches(self, indices: Iterable[int]):
        futures = []

        for i in indices:
            if i in self.indices_cache:
                continue

            self.indices_cache.add(i)

            new_futures = self.client.map(
                feature_dotproduct, self.indices_cache, i=i, features=self.features
            )
            futures.extend(new_futures)

        for future, (i, j, xi_xj) in as_completed(futures, with_results=True):
            self.products_cache[i, j] = xi_xj
            self.products_cache[j, i] = xi_xj
            self.client.cancel(future)

    def products(self, indices: Iterable[int]) -> np.ndarray:
        self._update_caches(indices)
        return self.products_cache[indices][:, indices].toarray()


class DaskArrayOp(FunctionOp):
    def __init__(
        self,
        X: da.Array,
        hsic_feature: Callable[[np.ndarray], np.ndarray],
        client: Client,
    ):
        def features(i):
            return hsic_feature(X[i].compute())

        super().__init__(features, len(X))

        if len(X.shape) != 2 or X.chunksize[1] != X.shape[1]:
            raise ValueError(
                "Only 2D Dask arrays with non-chunked 2nd axis are supported."
            )

        self.X = X
        self.hsic_feature = hsic_feature
        self.client = client

    def matvec(self, y: np.ndarray) -> np.ndarray:
        y = self.client.scatter(y, broadcast=True)
        inner_products = self.X.map_blocks(
            project_block,
            dtype=self.X.dtype,
            drop_axis=1,
            y=y,
            features=self.hsic_feature,
        )
        return inner_products.compute()


FeatureFunction = Callable[[np.ndarray, int, int, int, float], np.ndarray]


def cd_hsic_lasso(
    X: np.ndarray | Callable[[int], np.ndarray] | da.Array,
    Y: np.ndarray,
    B: int,
    M: int,
    num_feat: int,
    feature: FeatureFunction = feature_vanilla,
    y_kernel: str = "Delta",
    n_repeat: int = 1,
    n_features: int | None = None,
    client=None,
    chunksize: int = 1,
    **lasso_kwarg,
) -> tuple[sp.lil_matrix, np.ndarray, np.ndarray]:
    if isinstance(X, Callable) and n_features is None:
        raise ValueError("'n_features' needs to be defined if X is a function.")

    if isinstance(X, da.Array) and client is None:
        raise ValueError("'client' needs to be defined if X is a Dask array.")

    # TODO add a "discarded" parameter (and warning about it)
    assert Y.size % B == 0
    discarded = 0

    scale = np.sqrt(B / (Y.size * M))
    L = compute_kernel(Y[None, :], y_kernel, B, M, discarded=discarded).ravel() * scale

    if isinstance(X, Callable):

        def hsic_feature(i):
            return feature(i, X(i // n_repeat), B, M, discarded, scale)

        if client is None:
            operator = FunctionOp(hsic_feature, n_features * n_repeat)
        else:
            operator = DaskFunctionOp(
                hsic_feature, n_features * n_repeat, client, chunksize
            )

    elif isinstance(X, da.Array):
        # TODO add support for feature that requires index as input (e.g. multikernel)
        assert n_repeat == 1

        def hsic_feature(x):
            return feature(None, x, B, M, discarded, scale)

        operator = DaskArrayOp(X, hsic_feature, client)

    elif isinstance(X, np.ndarray):
        assert n_features is None or n_features == len(X)

        def hsic_feature(i):
            return feature(i, X[i // n_repeat], B, M, discarded, scale)

        operator = FunctionOp(hsic_feature, len(X) * n_repeat)

    else:
        raise ValueError(f"Unsupported type for 'X': {type(X)}.")

    return cd_nn_lasso_path(operator, y=L, max_features=num_feat, **lasso_kwarg)
