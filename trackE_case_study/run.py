"""Reproduce all Track E results: python trackE_case_study/run.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e_experiments  # noqa: E402

if __name__ == "__main__":
    e_experiments.main()
