"""Reproduce all Track A results: python trackA_continuous/run.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a1_experiments  # noqa: E402
import a2_experiments  # noqa: E402

if __name__ == "__main__":
    a1_experiments.main()
    a2_experiments.main()
