"""Reproduce all Track B results: python trackB_discrete/run.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b_experiments  # noqa: E402

if __name__ == "__main__":
    b_experiments.main()
