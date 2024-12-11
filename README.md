# MK-HSIC-Lasso

We developed the Multi-Kernel Hilbert-Schmidt Independence Criterion Lasso (MK-HSIC-Lasso)
for feature selection. Our approach integrates multiple kernel learning into the HSIC-Lasso framework,
enabling the selection of appropriate kernels in a data-driven manner. Multi-kernel learning enables MK-
HSIC-Lasso to model diverse types of relationships between features and outcomes effectively, capturing
features with both linear and non-linear effects. Our extensive simulation studies have demonstrated that
MK-HSIC-Lasso outperforms traditional methods in efficiently selecting features related to outcomes. Our
analyses of Alzheimer’s disease-related datasets revealed that the features selected by MK-HSIC-Lasso not
only exhibit biological relevance to Alzheimer’s disease but also have better predictive performance.

In this paper, we considered the most commonly used kernels, including the Gaussian (i.e., $K(x,x^{\prime}) = \exp\left(-\frac{\|x - x^{\prime}\|^2}{2\sigma^2}\right)$ with the bandwidth $\sigma^2 = 1$), linear (i.e., $K(x,x^{\prime}) = x^\top x^{\prime}$) and polynomial kernels with 2 degrees of freedom ($K(x,x^{\prime}) = (x^\top x^{\prime})^d$ with $d=2$) for inputs. We only used one kernel for the outcome, where Gaussian kernel is used if the outcome is continuous and the delta kernel is used when the outcome is categorical. 


## Getting Started

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
```

## Customise kernel functions

see 'kernel_tool` file, add the kernel you want to apply.

details see:[https://github.com/riken-aip/pyHSICLasso]



