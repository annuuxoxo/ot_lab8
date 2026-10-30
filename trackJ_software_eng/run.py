"""Reproduce all Track J results: python trackJ_software_eng/run.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import j_experiments  # noqa: E402

if __name__ == "__main__":
    j_experiments.main()
