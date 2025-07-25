import sys

if sys.version_info[:2] != (3, 11):
    import warnings
    warnings.warn(
        "Quantreo has only been tested on Python 3.11. "
        "You are using Python {}.{}. Use at your own risk.".format(*sys.version_info[:2]),
        RuntimeWarning
    )
