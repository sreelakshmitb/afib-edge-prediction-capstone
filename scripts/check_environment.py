"""Print the local Python and dependency versions used by the capstone."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import platform
import sys

REQUIRED_DISTRIBUTIONS = (
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "PyYAML",
    "wfdb",
    "torch",
    "onnx",
    "onnxruntime",
)


def main() -> int:
    """Print environment details and return non-zero when a required package is missing."""
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")

    if sys.version_info < (3, 11):
        print("ERROR: Python 3.11 or newer is required.")
        return 1

    missing: list[str] = []
    for distribution in REQUIRED_DISTRIBUTIONS:
        try:
            installed_version = version(distribution)
        except PackageNotFoundError:
            missing.append(distribution)
            print(f"{distribution}: MISSING")
        else:
            print(f"{distribution}: {installed_version}")

    if missing:
        print("ERROR: Missing required distributions: " + ", ".join(missing))
        return 1

    print("Environment dependency check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
