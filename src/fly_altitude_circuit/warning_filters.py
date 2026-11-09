"""Warning policy for figure and analysis execution."""
from contextlib import contextmanager
import warnings


@contextmanager
def silence_empty_mean():
    """Quiet empty-mean RuntimeWarnings"""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message=r'^Mean of empty slice\.?$',
                                category=RuntimeWarning)
        yield
