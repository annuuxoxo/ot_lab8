"""Reproduce all Track I results: python trackI_quantum_algos_qml/run.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import algorithms  # noqa: E402
import qml_vs_classical  # noqa: E402

if __name__ == "__main__":
    algorithms.main()
    qml_vs_classical.main()
