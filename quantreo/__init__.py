import sys
import warnings

# === Supported versions ===
supported_versions = [(3, 10), (3, 11), (3, 12), (3, 13)]

if sys.version_info[:2] not in supported_versions:
    warnings.warn(
        f"⚠️ Quantreo has only been tested on Python 3.10, 3.11, 3.12, and 3.13. "
        f"You are using Python {sys.version_info[0]}.{sys.version_info[1]}. "
        f"Use at your own risk.",
        RuntimeWarning,
    )
