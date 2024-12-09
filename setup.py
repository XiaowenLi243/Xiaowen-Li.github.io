#!/usr/bin/env python
# coding: utf-8
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
from setuptools import setup, find_packages


here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.rst")) as f:
    long_description = f.read()


def _load_requires_from_file(filepath):
    return [pkg_name.rstrip('\r\n') for pkg_name in open(filepath).readlines()]


def _install_requires():
    return _load_requires_from_file('requirements.txt')


def _packages():
    return find_packages(
        exclude=[
            '*.tests',
            '*.tests.*',
            'tests.*',
            'tests'
        ]
    )


setup(
    name="pyMKHSICLasso",
    version="1.4.2",
    download_url="https://github.com/XiaowenLi243/pyHSICLasso.git",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
    ],
    platforms=[ "python3.8", "python3.10", "python3.11"],
    license="MIT",
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    install_requires=_install_requires(),
    packages=_packages()
)
