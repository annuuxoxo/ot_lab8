"""Reproduce all Track C results (needs Track B results first): python trackC_quantum_opt/run.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pqc  # noqa: E402
import qaoa_vqe  # noqa: E402

if __name__ == "__main__":
    qaoa_vqe.main()
    pqc.main()
