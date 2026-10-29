"""Generate TPC-H at scale factor 0.1 with DuckDB's tpch extension and export it to a SQLite file.
The SQLite copy has NO indexes at all (no primary keys either) - that is the starting point for index selection."""
import os
import sqlite3

import duckdb

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "data", "tpch_sf0.1.db")
TABLES = ["region", "nation", "supplier", "customer", "part", "partsupp", "orders", "lineitem"]


def generate(sf=0.1):
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    if os.path.exists(DB):
        os.remove(DB)
    con = duckdb.connect()
    con.execute("INSTALL tpch; LOAD tpch;")
    con.execute(f"CALL dbgen(sf={sf})")
    lite = sqlite3.connect(DB)
    for t in TABLES:
        rel = con.execute(f"SELECT * FROM {t}")
        cols = [d[0] for d in rel.description]
        rows = rel.fetchall()
        # dates -> ISO text (SQLite compares them correctly as strings), decimals -> REAL
        conv = [tuple(x.isoformat() if hasattr(x, "isoformat") else (float(x) if type(x).__name__ == "Decimal" else x)
                      for x in r) for r in rows]
        lite.execute(f"CREATE TABLE {t} ({', '.join(cols)})")
        lite.executemany(f"INSERT INTO {t} VALUES ({', '.join('?' * len(cols))})", conv)
        lite.commit()
        print(f"{t}: {len(rows)} rows", flush=True)
    lite.execute("VACUUM")
    lite.close()


def row_counts():
    lite = sqlite3.connect(DB)
    out = {t: lite.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES}
    lite.close()
    return out


if __name__ == "__main__":
    generate()
    print(row_counts(), f"{os.path.getsize(DB) / 1e6:.1f} MB")
