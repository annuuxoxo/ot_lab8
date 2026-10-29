"""Reproduce Track H: python trackH_dbms/run.py   (generates TPC-H SF 0.1 into trackH_dbms/data/ if missing, ~5 min total)"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data  # noqa: E402
import index_selection  # noqa: E402

if __name__ == "__main__":
    if not os.path.exists(data.DB):
        data.generate()
    print(data.row_counts())
    index_selection.main()
