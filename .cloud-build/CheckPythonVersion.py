import sys

MINIMUM_MAJOR_VERSION = 3
MINIMUM_MINOR_VERSION = 3

if (
    sys.version_info.major < MINIMUM_MAJOR_VERSION
    and sys.version_info.minor < MINIMUM_MINOR_VERSION
):
    print("Error: Python version less than 3.5")
    exit(1)
else:
    print(f"Python version acceptable: {sys.version}")
    exit(0)
