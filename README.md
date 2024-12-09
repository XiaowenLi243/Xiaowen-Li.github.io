# MK-HSIC-LASSO

We developed the Multi-Kernel Hilbert-Schmidt Independence Criterion Lasso (MK-HSIC-Lasso)
for feature selection. Our approach integrates multiple kernel learning into the HSIC-Lasso framework,
enabling the selection of appropriate kernels in a data-driven manner. Multi-kernel learning enables MK-
HSIC-Lasso to model diverse types of relationships between features and outcomes effectively, capturing
features with both linear and non-linear effects. Our extensive simulation studies have demonstrated that
MK-HSIC-Lasso outperforms traditional methods in efficiently selecting features related to outcomes. Our
analyses of Alzheimer’s disease-related datasets revealed that the features selected by MK-HSIC-Lasso not
only exhibit biological relevance to Alzheimer’s disease but also have better predictive performance.


## Installation


First make sure you have [Git](https://git-scm.com/downloads) installed on your computer.

Then open a terminal and clone this repository:

```
git clone https://github.com/XiaowenLi243/MKHSICLasso.git
```

create a virtual environment MKHSIC and activate it:

```
cd MKHSICLasso

module purge
module load Python/3.10.5-gimkl-2022a
python -m venv MKHSIC 
source ./MKHSIC/bin/activate

```
After activating your virtual environment, install all dependencies:

```sh
$ pip install -r requirements.txt
$ python setup.py install
```

or  

```sh
$ pip install MKHSICLasso
```


## Getting Started

In this paper, we considered the most commonly used kernels, including the Gaussian (i.e., $K(x,x^{\prime}) = \exp\left(-\frac{\|x - x^{\prime}\|^2}{2\sigma^2}\right)$ with the bandwidth $\sigma^2 = 1$), linear (i.e., $K(x,x^{\prime}) = x^\top x^{\prime}$) and polynomial kernels with 2 degrees of freedom ($K(x,x^{\prime}) = (x^\top x^{\prime})^d$ with $d=2$) for inputs. We only used one kernel for the outcome, where Gaussian kernel is used if the outcome is continuous and the delta kernel \citep{song2012feature, yamada2014high} defined in equation (\ref{eq: Gaussia_y}) is used when the outcome is categorical. 


# run HSIC-lasso
hsic_lasso = MKHSICLasso()
hsic_lasso.input(x_train_norm,y_train_norm)

# regression continuous y 
hsic_lasso.regression(desire_number,B=B, M=M, max_neighbors=max_neighbors)
# save selected results.
hsic_lasso.save_param('/nesi/project/uoa03056/MultiOmics/ADNI/Gene_expression/MKparams_GM/gene_ex_GM_%d.csv'%ind)
#active_ind = hsic_lasso.get_index() 


## Save results to a csv file
If you want to save the feature selection results in csv file, please call the following function:

```
>>> MKhsic_lasso.save_param()
```

```py
>>> hsic_lasso.regression(5,covars=X)

>>> hsic_lasso.classification(10,covars=X)
```


## Example

```py
>>> from pyHSICLasso import HSICLasso
>>> hsic_lasso = HSICLasso()

>>> hsic_lasso.input("data.mat")

>>> hsic_lasso.input("data.csv")

>>> hsic_lasso.input("data.tsv")

>>> hsic_lasso.input(np.array([[1, 1, 1], [2, 2, 2]]), np.array([0, 1]))
```

You can specify the number of subset of feature selections with arguments `regression` and` classification`.

```py
>>> hsic_lasso.regression(5)

>>> hsic_lasso.classification(10)
```

About output method, it is possible to select plots on the graph, details of the analysis result, output of the feature index. Note, to run the dump() function, it needs at least 5 features in the dataset.

```py
>>> hsic_lasso.plot()
# plot the graph

>>> hsic_lasso.dump()
============================================== HSICLasso : Result ==================================================
| Order | Feature      | Score | Top-5 Related Feature (Relatedness Score)                                          |
| 1     | 1100         | 1.000 | 100          (0.979), 385          (0.104), 1762         (0.098), 762          (0.098), 1385         (0.097)|
| 2     | 100          | 0.537 | 1100         (0.979), 385          (0.100), 1762         (0.095), 762          (0.094), 1385         (0.092)|
| 3     | 200          | 0.336 | 1200         (0.979), 264          (0.094), 1482         (0.094), 1264         (0.093), 482          (0.091)|
| 4     | 1300         | 0.140 | 300          (0.984), 1041         (0.107), 1450         (0.104), 1869         (0.102), 41           (0.101)|
| 5     | 300          | 0.033 | 1300         (0.984), 1041         (0.110), 41           (0.106), 1450         (0.100), 1869         (0.099)|
>>> hsic_lasso.get_index()
[1099, 99, 199, 1299, 299]

>>> hsic_lasso.get_index_score()
array([0.09723658, 0.05218047, 0.03264885, 0.01360242, 0.00319763])

>>> hsic_lasso.get_features()
['1100', '100', '200', '1300', '300']

>>> hsic_lasso.get_index_neighbors(feat_index=0,num_neighbors=5)
[99, 384, 1761, 761, 1384]

>>> hsic_lasso.get_features_neighbors(feat_index=0,num_neighbors=5)
['100', '385', '1762', '762', '1385']

>>> hsic_lasso.get_index_neighbors_score(feat_index=0,num_neighbors=5)
array([0.9789888 , 0.10350618, 0.09757666, 0.09751763, 0.09678892])

>>> hsic_lasso.save_param() #Save selected features and its neighbors 

```


