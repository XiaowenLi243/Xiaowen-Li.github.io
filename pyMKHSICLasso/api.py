# coding: utf-8

from __future__ import (absolute_import, division, print_function, unicode_literals)

import warnings
from builtins import int, open, range, str
from future import standard_library
import numpy as np
import scipy.spatial.distance as distance
from scipy.cluster.hierarchy import linkage
from six import string_types

from .MKhsic_lasso import MKhsic_lasso, compute_kernel
from .input_data import input_csv_file, input_matlab_file, input_tsv_file
from .nlars import nlars
from .plot_figure import plot_dendrogram, plot_path

standard_library.install_aliases()


class MKHSICLasso:
    """
    A class to perform Multivariate Kernel HSIC Lasso for feature selection and hierarchical clustering.
    """

    def __init__(self):
        self.input_file = None
        self.X_in = None
        self.Y_in = None
        self.X = None
        self.Xty = None
        self.A_all = None
        self.path = None
        self.beta = None
        self.A = None
        self.A_neighbors = None
        self.A_neighbors_score = None
        self.lam = None
        self.featname = None
        self.linkage_dist = None
        self.hclust_featname = None
        self.hclust_featnameindex = None
        self.max_neighbors = 10

    def input(self, *args, output_list=['class'], featname=None):
      
        self._check_args(args)

        if isinstance(args[0], string_types):
            self._input_data_file(args[0], output_list)
        elif isinstance(args[0], np.ndarray):
            self._input_data_ndarray(*args, featname)
        else:
            raise ValueError("Unsupported input type.")

        if self.X_in is None or self.Y_in is None:
            raise ValueError("Invalid input data.")
        
        self._check_shape()
        return True

    def regression(self, num_feat=5, B=20, M=3, discrete_x=False, max_neighbors=10, n_jobs=-1, covars=np.array([]), covars_kernel="Gaussian"):
        """
        Runs regression using MKHSIC Lasso.
        """
        self._run_MKhsic_lasso(
            num_feat=num_feat, y_kernel="Gaussian", B=B, M=M,
            discrete_x=discrete_x, max_neighbors=max_neighbors,
            n_jobs=n_jobs, covars=covars, covars_kernel=covars_kernel
        )
        return True

    def classification(self, num_feat=5, B=20, M=3, discrete_x=False, max_neighbors=10, n_jobs=-1, covars=np.array([]), covars_kernel="Gaussian"):
        """
        Runs classification using MKHSIC Lasso.
        """
        self._run_MKhsic_lasso(
            num_feat=num_feat, y_kernel="Delta", B=B, M=M,
            discrete_x=discrete_x, max_neighbors=max_neighbors,
            n_jobs=n_jobs, covars=covars, covars_kernel=covars_kernel
        )
        return True

    def _run_MKhsic_lasso(self, y_kernel, num_feat, B, M, discrete_x, max_neighbors, n_jobs, covars, covars_kernel):
        """
        Internal method to run the MKHSIC Lasso algorithm.
        """
        if self.X_in is None or self.Y_in is None:
            raise UnboundLocalError("Input data is missing.")

        self.max_neighbors = max_neighbors
        n = self.X_in.shape[1]
        B = B or n
        numblocks = n // B
        discarded = n % B

        print(f'Block MK-HSIC Lasso B = {B}.')
        if discarded:
            warnings.warn(
                f"B ({B}) must divide the number of samples ({n}). Adjusted blocks to {numblocks}.",
                RuntimeWarning
            )
            numblocks = int(numblocks)

        M = 1 + (numblocks > 1) * (M - 1)
        print(f'M set to {M}.')
        print(f'{y_kernel} kernel for outcomes {"and Gaussian kernel for covariates" if covars.size else ""}.')

        X, Xty, Ky = MKhsic_lasso(self.X_in, self.Y_in, y_kernel, n_jobs=n_jobs, discarded=discarded, B=B, M=M)
        self.X = X * np.sqrt(1 / (numblocks * M))
        self.Xty = Xty * (1 / (numblocks * M))

        # Handle covariates
        if covars.size:
            if self.X_in.shape[1] != covars.shape[0]:
                raise ValueError("Mismatch between samples in covariates and input data.")
            Kc = compute_kernel(covars.T, covars_kernel, B, M, discarded)
            Kc = Kc.reshape(n * B * M, 1) * np.sqrt(1 / (numblocks * M))
            Ky *= np.sqrt(1 / (numblocks * M))
            betas = np.dot(Ky.T, Kc) / np.trace(np.dot(Kc.T, Kc))
            self.Xty -= betas * np.dot(self.X.T, Kc)

        # Run NLARS
        self.A_all, self.path, self.beta, self.A, self.lam, self.A_neighbors, self.A_neighbors_score = nlars(
            self.X, self.Xty, num_feat, self.max_neighbors
        )
        return True

    def linkage(self, method="ward"):
        """
        Performs hierarchical clustering on selected features.
        """
        if self.A is None:
            raise UnboundLocalError("Run regression/classification first.")
        
        featname_index, featname_selected = self._get_selected_features()
        self.hclust_featname = featname_selected
        self.hclust_featnameindex = featname_index

        sim = np.dot(self.X[:, featname_index].T, self.X[:, featname_index])
        dist = 1 - sim
        dist = np.maximum(0, dist - np.diag(np.diag(dist)))
        dist_sym = (dist + dist.T) / 2.0
        self.linkage_dist = linkage(distance.squareform(dist_sym), method)
        return True

    def dump(self):
        maxval = self.beta[self.A[0]][0]
        results = []
        results.append(" MKHSICLasso : Result ")
        results.append("| Order | Feature      | Score | Top-{} Related Feature (Relatedness Score)".format(min(5, len(self.beta) - 1)))
        for i in range(len(self.A)):
            ofs = "| {:<5} | {:<12} | {:.3f} |".format(i + 1, self.featname[self.A[i]], self.beta[self.A[i]][0] / maxval)
            rf = [" {:<12} ({:.3f})".format(self.featname[nn], ns) for nn, ns, _ in zip(self.A_neighbors[i][1:], self.A_neighbors_score[i][1:], range(5))]
            row = ofs + ",".join(rf)
            results.append(row + " " * max(0, len(results[1]) - len(row)) + "|")

        results[1] = results[1] + " " * max(0, len(row) - len(results[1])) + "|"
        deco = "=" * ((len(results[1]) - len(results[0])) // 2)
        results[0] = deco + results[0] + deco

        print("\n".join(results))

    def plot_dendrogram(self, filepath = 'dendrogram.png'):
        if self.linkage_dist is None or self.hclust_featname is None:
            raise UnboundLocalError("Input your data")
        plot_dendrogram(self.linkage_dist, self.hclust_featname, filepath)
        return True

    def plot_path(self, filepath = 'path.png'):
        if self.path is None or self.beta is None or self.A is None:
            raise UnboundLocalError("Input your data")
        plot_path(self.path, self.beta, self.A, filepath)
        return True

    def get_features(self):
        index = self.get_index()

        return [self.featname[i] for i in index]

    def get_features_neighbors(self, feat_index=0, num_neighbors=5):
        index = self.get_index_neighbors(
            feat_index=feat_index, num_neighbors=num_neighbors)

        return [self.featname[i] for i in index]

    def get_index(self):
        return self.A

    def get_index_score(self):
        return self.beta[self.A, -1]

    def get_index_neighbors(self, feat_index=0, num_neighbors=5):
        if feat_index > len(self.A) - 1:
            raise IndexError("Index does not exist")

        num_neighbors = min(num_neighbors, self.max_neighbors)

        return self.A_neighbors[feat_index][1:(num_neighbors + 1)]

    def get_index_neighbors_score(self, feat_index=0, num_neighbors=5):
        if feat_index > len(self.A) - 1:
            raise IndexError("Index does not exist")

        num_neighbors = min(num_neighbors, self.max_neighbors)

        return self.A_neighbors_score[feat_index][1:(num_neighbors + 1)]

    def save_HSICmatrix(self, filename='HSICmatrix.csv'):
        if self.X_in is None or self.Y_in is None:
            raise UnboundLocalError("Input your data")

        self.X, self.X_ty = MKhsic_lasso(self.X_in, self.Y_in, "Gaussian")

        K = np.dot(self.X.transpose(), self.X)

        np.savetxt(filename, K, delimiter=',', fmt='%.7f')

        return True

    def save_score(self, filename='aggregated_score.csv'):
        maxval = self.beta[self.A[0]][0]

        #print(maxval + ' ' + maxval_)
        fout = open(filename, 'w')
        featscore = {}
        featcorrcoeff = {}
        for i in range(len(self.A)):
            HSIC_XY = (self.beta[self.A[i]][0] / maxval)

            if self.featname[self.A[i]] not in featscore:
                featscore[self.featname[self.A[i]]] = HSIC_XY

                corrcoeff = np.corrcoef(self.X_in[self.A[i]], self.Y_in)[0][1]

                featcorrcoeff[self.featname[self.A[i]]] = corrcoeff

            else:
                featscore[self.featname[self.A[i]]] += HSIC_XY

            for j in range(1, self.max_neighbors + 1):
                HSIC_XX = self.A_neighbors_score[i][j]
                if self.featname[self.A_neighbors[i][j]] not in featscore:
                    featscore[self.featname[self.A_neighbors[i][j]]
                              ] = HSIC_XY * HSIC_XX

                    corrcoeff = np.corrcoef(
                        self.X_in[self.A_neighbors[i][j]], self.Y_in)[0][1]

                    featcorrcoeff[self.featname[self.A_neighbors[i]
                                                [j]]] = corrcoeff
                else:
                    featscore[self.featname[self.A_neighbors[i][j]]
                              ] += HSIC_XY * HSIC_XX

        # Sorting decending order
        featscore_sorted = sorted(
            featscore.items(), key=lambda x: x[1], reverse=True)

        # Add Pearson correlation for comparison
        fout.write('Feature,Score,Pearson Corr\n')
        for (key, val) in featscore_sorted:
            fout.write(key + ',' + str(val) + ',' +
                       str(featcorrcoeff[key]) + '\n')

        fout.close()

    def save_param(self, filename='param.csv'):
        # Save parameters
        maxval = self.beta[self.A[0]][0]

        fout = open(filename, 'w')
        sstr = 'Feature,Score,'
        for j in range(1, self.max_neighbors + 1):
            sstr = sstr + 'Neighbor %d, Neighbor %d score,' % (j, j)

        sstr = sstr + '\n'
        fout.write(sstr)
        for i in range(len(self.A)):
            tmp = []
            tmp.append(self.featname[self.A[i]])
            tmp.append(str(self.beta[self.A[i]][0] / maxval))
            for j in range(1, self.max_neighbors + 1):
                tmp.append(str(self.featname[self.A_neighbors[i][j]]))
                tmp.append(str(self.A_neighbors_score[i][j]))

            sstr = ','.join(tmp) + '\n'
            fout.write(sstr)

        fout.close()
        
    def save_param_path(self, filename='param_path.npy'):
        # Convert the list to a NumPy array
        my_array = np.array(self.A_all, dtype=object)

        # Save the array to an npy file
        np.save(filename, my_array)
    

    def _check_args(self, args):
        if len(args) == 0 or len(args) >= 4:
            raise SyntaxError("Input as input_data(file_name) or \
                input_data(X_in, Y_in)")
        elif len(args) == 1:
            if isinstance(args[0], string_types):
                if len(args[0]) <= 4:
                    raise ValueError("Check your file name")
                else:
                    ext = args[0][-4:]
                    if ext == ".csv" or ext == ".tsv" or ext == ".mat":
                        pass
                    else:
                        raise TypeError("Input file is only .csv, .tsv .mat")
            else:
                raise TypeError("File name is only str")
        elif len(args) == 2:
            if isinstance(args[0], string_types):
                raise TypeError("Check arg type")
            elif isinstance(args[0], list):
                if isinstance(args[1], list):
                    pass
                else:
                    raise TypeError("Check arg type")
            elif isinstance(args[0], np.ndarray):
                if isinstance(args[1], np.ndarray):
                    pass
                else:
                    raise TypeError("Check arg type")
            else:
                raise TypeError("Check arg type")
        elif len(args) == 3:
            if isinstance(args[0], np.ndarray) and isinstance(args[1], np.ndarray) and isinstance(args[2], list):
                pass
            else:
                raise TypeError("Check arg type")

        return True

    def _input_data_file(self, file_name, output_list):
        ext = file_name[-4:]
        if ext == ".csv":
            self.X_in, self.Y_in, self.featname = input_csv_file(
                file_name, output_list=output_list)
        elif ext == ".tsv":
            self.X_in, self.Y_in, self.featname = input_tsv_file(
                file_name, output_list=output_list)
        elif ext == ".mat":
            self.X_in, self.Y_in, self.featname = input_matlab_file(file_name)
        return True

    def _input_data_list(self, X_in, Y_in):
        if isinstance(Y_in[0], list):
            raise ValueError("Check your input data")
        self.X_in = np.array(X_in).T
        self.Y_in = np.array(Y_in).reshape(1, len(Y_in))
        return True

    def _input_data_ndarray(self, X_in, Y_in, featname = None):
        if len(Y_in.shape) == 2:
            raise ValueError("Check your input data")
        self.X_in = X_in.T
        self.Y_in = Y_in.reshape(1, len(Y_in))
        self.featname = featname
        return True

    def _check_shape(self):
        _, x_col_len = self.X_in.shape
        y_row_len, y_col_len = self.Y_in.shape
        # if y_row_len != 1:
        #    raise ValueError("Check your input data")
        if x_col_len != y_col_len:
            raise ValueError(
                "The number of samples in input and output should be same")
        return True

    def _permute_data(self, seed=None):
        np.random.seed(seed)
        n = self.X_in.shape[1]

        perm = np.random.permutation(n)
        self.X_in = self.X_in[:, perm]
        self.Y_in = self.Y_in[:, perm]

