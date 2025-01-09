import logging
import typing as T
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp


def cd_nn_lasso_(
    Xy: np.ndarray,
    XiXj: np.ndarray,
    beta: np.ndarray,
    lam: float,
    max_iter: int,
    abstol: float,
) -> np.ndarray:
    beta = beta.copy()

    for i in range(max_iter):
        old_beta = beta.copy()

        for j in range(len(Xy)):
            r_j = Xy[j] - XiXj[j] @ beta + XiXj[j, j] * beta[j]
            beta[j] = max(r_j - lam, 0) / XiXj[j, j]

        if np.max(np.abs(beta - old_beta)) < abstol:
            break

    return beta, i


class Operator(T.Protocol):
    def __len__(self) -> int:
        pass

    def matvec(self, y: np.ndarray) -> np.ndarray:
        pass

    def rmatvec(self, y: np.ndarray, indices: T.Iterable[int]) -> np.ndarray:
        pass

    def products(self, indices) -> np.ndarray:
        pass


class ArrayOp:
    def __init__(self, arr: np.ndarray):
        self.arr = arr

    def __len__(self) -> int:
        return len(self.arr)

    def matvec(self, y: np.ndarray) -> np.ndarray:
        return self.arr @ y

    def rmatvec(self, y: np.ndarray, indices: T.Iterable[int]) -> np.ndarray:
        return self.arr[indices].T @ y

    def products(self, indices: T.Iterable[int]) -> np.ndarray:
        arr = self.arr[indices]
        return arr @ arr.T


class FunctionOp:
    def __init__(self, features: T.Callable[[int], np.ndarray], n_features: int):
        self.features = features
        self.n_features = n_features

        self.features_cache = {}
        dtype = features(0).dtype
        self.products_cache = sp.dok_matrix((n_features, n_features), dtype=dtype)

    def __len__(self) -> int:
        return self.n_features

    def matvec(self, y: np.ndarray) -> np.ndarray:
        return np.array([self.features(i) @ y for i in range(self.n_features)])

    def _update_caches(self, indices: T.Iterable[int]):
        for i in indices:
            if i in self.features_cache:
                continue

            self.features_cache[i] = self.features(i)

            for j in self.features_cache:
                xi_xj = self.features_cache[i] @ self.features_cache[j]
                self.products_cache[i, j] = xi_xj
                self.products_cache[j, i] = xi_xj

    def rmatvec(self, y: np.ndarray, indices: T.Iterable[int]) -> np.ndarray:
        self._update_caches(indices)
        result = 0
        for idx, coeff in zip(indices, y):
            result += self.features_cache[idx] * coeff
        return result

    def products(self, indices: T.Iterable[int]) -> np.ndarray:
        self._update_caches(indices)
        return self.products_cache[indices][:, indices].toarray()


def load_state(
    checkpoint: Path,
) -> T.Tuple[np.ndarray, sp.lil_matrix, np.ndarray, np.ndarray, np.ndarray, int]:
    with h5py.File(checkpoint, "r") as fd:
        Xy = np.array(fd["Xy"])

        path = sp.lil_matrix(fd["path"].shape, dtype=fd["path"].dtype)
        for i, row in enumerate(fd["path"]):
            idx = row.nonzero()
            path[i, idx] = row[idx]

        lams = np.array(fd["lams"])
        inner_products = np.array(fd["inner_products"])
        beta = np.array(fd["beta"])
        idx_outer = np.array(fd["idx_outer"]).item()

    return Xy, path, lams, inner_products, beta, idx_outer


def save_state(
    checkpoint: Path,
    Xy: np.ndarray,
    path: sp.lil_matrix,
    lams: np.ndarray,
    inner_products: np.ndarray,
    beta: np.ndarray,
    idx_outer: int,
):
    with h5py.File(checkpoint, "w") as fd:
        fd["Xy"] = Xy

        fd.create_dataset(
            "path", path.shape, dtype=path.dtype, fillvalue=0, compression="gzip"
        )
        for i, row in enumerate(path):
            fd["path"][i] = row.toarray()

        fd["lams"] = lams
        fd["inner_products"] = inner_products
        fd["beta"] = beta
        fd["idx_outer"] = idx_outer


def cd_nn_lasso_path(
    features: T.Union[np.ndarray, Operator],
    y: np.ndarray,
    max_features: int,
    eps: float = 0.95,
    max_outer_iter: int = 100,
    max_inner_iter: int = 2000,
    abstol: float = 1e-6,
    checkpoint: T.Optional[Path] = None,
) -> T.Tuple[sp.lil_matrix, np.ndarray, np.ndarray]:
    assert max_features <= len(features)
    logger = logging.getLogger(__name__)

    if isinstance(features, np.ndarray):
        features = ArrayOp(features)

    if checkpoint is not None and checkpoint.is_file():
        logger.info(f"loading saved intermediate results from {checkpoint}")
        Xy, path, lams, inner_products, beta, idx_outer = load_state(checkpoint)

    else:
        logger.info(f"outer iteration started - 1/{max_outer_iter}")
        Xy = features.matvec(y)
        lams_max = np.abs(Xy).max()
        logger.info(f"outer iteration finished - largest penality: {lams_max}")

        path = sp.lil_matrix((max_outer_iter, len(features)), dtype=y.dtype)
        lams = np.geomspace(
            lams_max, lams_max * eps ** (max_outer_iter - 1), max_outer_iter
        )

        inner_products = Xy
        beta = np.zeros(len(features), dtype=y.dtype)

        idx_outer = 0

    if checkpoint is not None:
        save_state(checkpoint, Xy, path, lams, inner_products, beta, idx_outer)

    for idx_outer in range(idx_outer + 1, max_outer_iter):
        logger.info(f"outer iteration started - {idx_outer + 1}/{max_outer_iter}")

        # screening using sequential strong rule
        gap = 2 * lams[idx_outer] - lams[idx_outer - 1]
        active_mask = np.abs(inner_products) > gap
        active_set = np.where(active_mask)[0]

        # TODO add fallback solution if all predictors are discarded (& message)
        assert len(active_set) > 0

        while True:
            XiXj = features.products(active_set)
            beta[active_set], n_iter = cd_nn_lasso_(
                Xy[active_set],
                XiXj,
                beta[active_set],
                lams[idx_outer],
                max_inner_iter,
                abstol,
            )
            logger.info(f"inner iteration finished - {n_iter}/{max_inner_iter} steps")

            residuals = y - features.rmatvec(beta[active_set], active_set)
            inner_products = features.matvec(residuals)

            old_active_mask = active_mask.copy()
            active_mask |= np.abs(inner_products) > lams[idx_outer]
            active_set = np.where(active_mask)[0]

            if np.all(active_mask == old_active_mask):
                break

        path[idx_outer, active_set] = beta[active_set]
        n_active = path[idx_outer].count_nonzero()
        logger.info(f"outer iteration finished - {n_active} active features")
        if n_active > max_features:
            break

        if checkpoint is not None:
            save_state(checkpoint, Xy, path, lams, inner_products, beta, idx_outer)

    if checkpoint is not None:
        save_state(checkpoint, Xy, path, lams, inner_products, beta, idx_outer)

    # remove the last iteration to keep only max_features active, if needed
    if path[idx_outer].count_nonzero() > max_features:
        path = path[:idx_outer]
        lams = lams[:idx_outer]
    else:
        assert idx_outer == max_outer_iter - 1

    beta = path[-1].toarray().squeeze()
    return path, beta, lams
