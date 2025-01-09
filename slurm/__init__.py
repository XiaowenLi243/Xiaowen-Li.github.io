from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("hsic_optimization")
except PackageNotFoundError:
    # package is not installed
    pass
