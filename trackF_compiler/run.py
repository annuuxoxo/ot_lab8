"""Reproduce all Track F results: python trackF_compiler/run.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import f_experiments  # noqa: E402

if __name__ == "__main__":
    f_experiments.main()
