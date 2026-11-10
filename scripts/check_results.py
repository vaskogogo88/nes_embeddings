"""Check B's output/results.csv against what the edge workers sent (data/worker_XX.csv).

Reports rows received per worker (catches NebulaStream's silent failures: header-only output),
whether embeddings arrived unchanged, and whether B's SCORE matches the same dot product in Python.

Example:
  python scripts/check_results.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--results", default="output/results.csv")
    p.add_argument("--topology", default="compose-topology.yaml")
    a = p.parse_args()

    schema = json.loads((Path(a.data_dir) / "schema.json").read_text())
    names = [c["name"] for c in schema["columns"]]
    emb_cols = names[2:]
    ref = np.array(schema["reference"], dtype=np.float64)

    out = pd.read_csv(a.results)
    out.columns = [c.split(":")[0] for c in out.columns]  # "SCORE:FLOAT32:NOT_NULLABLE" -> "SCORE"
    if out.empty:
        raise SystemExit("results.csv has a header but no rows: the query probably failed (check worker logs)")

    # expected workers = File sources in the generated topology (so a worker that sent 0 rows is caught)
    import re
    expected = sorted(int(m) for m in re.findall(r"/data/worker_(\d+)\.csv", Path(a.topology).read_text()))

    ok = True
    for wid in expected:
        sent = pd.read_csv(Path(a.data_dir) / f"worker_{wid:02d}.csv", header=None, names=names)
        got = out[out.WORKER_ID == wid]
        merged = sent.merge(got, on=["WORKER_ID", "TS"], suffixes=("_sent", "_got"))
        if merged.empty:
            ok = False
            print(f"worker {wid:2d}:    0/{len(sent)} rows  <-- NOTHING RECEIVED")
            continue
        emb_err = np.abs(merged[[f"{c}_sent" for c in emb_cols]].values
                         - merged[[f"{c}_got" for c in emb_cols]].values).max()
        score_err = np.abs(merged[[f"{c}_sent" for c in emb_cols]].values @ ref - merged.SCORE.values).max()
        complete = len(got) == len(sent) == len(merged)
        ok &= complete and emb_err < 1e-6 and score_err < 1e-4
        print(f"worker {wid:2d}: {len(got):4d}/{len(sent)} rows | max emb diff {emb_err:.1e} | "
              f"max score diff {score_err:.1e} | mean score {got.SCORE.mean():+.3f}"
              f"{'' if complete else '  <-- INCOMPLETE'}")
    print("OK" if ok else "PROBLEMS FOUND")


if __name__ == "__main__":
    main()
