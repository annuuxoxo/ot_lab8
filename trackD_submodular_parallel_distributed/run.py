"""Reproduce all Track D results: python trackD_submodular_parallel_distributed/run.py"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fed  # noqa: E402
import multitask  # noqa: E402
import parallel  # noqa: E402
import parallelism_demo  # noqa: E402
import submodular  # noqa: E402

if __name__ == "__main__":
    pd.set_option("display.width", 220)
    print(submodular.experiment().to_string(index=False))
    parallel.main()
    print(parallelism_demo.main().to_string(index=False))
    print(fed.main().to_string(index=False))
    print(multitask.main().to_string(index=False))
