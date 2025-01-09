# ensure conda environment do not use user site-packages
export PYTHONNOUSERSITE=1  # TODO check if useful in the Makefile or outside

PYTHON_VERSION ?= 3.10
CONDA_VENV_PATH ?= $(PWD)/venv
KERNEL_NAME ?= $(shell basename $(CURDIR))

CONDA_RUN := conda run -p $(CONDA_VENV_PATH) --live-stream
KERNEL_DIR := $(HOME)/.local/share/jupyter/kernels/$(KERNEL_NAME)

all: help

## Display this help message
help: Makefile
	@echo 'Available targets:'
	@echo ''
	@tac Makefile | \
	    awk '/^##/ { sub(/## /,"#"); print "    " prev $$0 } { FS=":"; prev = $$1 }' | \
	    column -t -s '#' | \
	    tac
	@echo ''
	@echo 'Configurable variables (current value):'
	@echo ''
	@echo '    PYTHON_VERSION   Version of Python to install in the conda environment ($(PYTHON_VERSION))'
	@echo '    CONDA_VENV_PATH  Path of the conda environment ($(CONDA_VENV_PATH))'
	@echo '    KERNEL_NAME      Jupyter kernel name ($(KERNEL_NAME))'
	@echo ''

# create a conda environment with bare minimum packages
venv/.canary:
	conda create -p $(CONDA_VENV_PATH) -c conda-forge -y python=$(PYTHON_VERSION) pip
	$(CONDA_RUN) python -m pip install pip-tools
	touch "$@"

# compile dependencies for reproducibility
requirements.txt: pyproject.toml venv/.canary
	$(CONDA_RUN) pip-compile --all-extras -o "$@" "$<"

## Create a conda environment and register it as a Jupyter kernel
venv: requirements.txt venv/.canary
	$(CONDA_RUN) pip-sync requirements.txt
	$(CONDA_RUN) pip install --no-deps -e .
	$(CONDA_RUN) python -m ipykernel install --user --name $(KERNEL_NAME)

## Remove the conda environment and associated Jupyter kernel
clean:
	rm -rf $(KERNEL_DIR)
	conda env remove -p $(CONDA_VENV_PATH)
	rm -rf src/*.egg-info

## Format Python scripts and notebooks
format: venv/.canary
	$(CONDA_RUN) ruff format src
	$(CONDA_RUN) ruff format notebooks

## Lint Python scripts and notebooks
lint: venv/.canary
	$(CONDA_RUN) ruff check src
	$(CONDA_RUN) ruff check --ignore=E402,E501 notebooks

## Run the tests
test:
	$(CONDA_RUN) pytest

## Build the html documentation
doc:
	cd docs; $(CONDA_RUN) make html

## Watch and rebuild html documentation on changes
livedoc:
	$(CONDA_RUN) sphinx-autobuild docs docs/_build

## Build the pdf documentation
report:
	cd docs; $(CONDA_RUN) make latexpdf

.PHONY: help venv clean format lint test doc report
