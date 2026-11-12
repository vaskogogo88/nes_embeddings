"""TPCx-IoT records (SQLite) -> one embedding per substation per time window -> one CSV per worker.

Counterpart of embed.py for the TPCx-IoT data from scripts/iot_to_sqlite.py, writing the same three
outputs, so gen_topology.py, check_results.py and the dashboard work unchanged:

  worker_01.csv ... worker_NN.csv   WORKER_ID,TS,EMB_0..EMB_{d-1}   (no header; client i -> worker i)
  schema.json                       columns, dims, reference vector, worker -> client mapping
  meta.csv                          WORKER_ID,TS,GROUP,RAW_BYTES,CONTENT_BYTES,N_RECORDS,SNIPPET (dashboard only)

One embedding = one client (substation) x one tumbling window of --window-s seconds:
  1. features: mean reading of each sensor in the window (160 sensors); a sensor without a reading
     in that window gets the window's overall mean, so every vector is complete;
  2. z-score each feature over all windows, one PCA (shared by all workers) -> --dims, unit length.
B's reference vector is the normalised centroid of all windows.

Note: TPCx-IoT readings are uniform random numbers, so these embeddings carry no meaning beyond
noise. The demo shows delivery and byte savings; it cannot show preserved meaning.

  python scripts/embed_iot.py --window-s 60 --dims 32
"""
import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent


def load_records(db):
    """The only data-source-specific part: TPCx-IoT records from iot_to_sqlite.py."""
    con = sqlite3.connect(db)
    df = pd.read_sql("SELECT client, sensor, ts_ms, reading, raw_bytes, "
                     "length(raw_key) + length(printf('%.4f', reading)) AS content_bytes "
                     "FROM records WHERE reading IS NOT NULL", con)
    con.close()
    return df


def unit_rows(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=str(ROOT / "data" / "iot" / "tpcx_iot.db"))
    p.add_argument("--window-s", type=int, default=60, help="tumbling window length in seconds")
    p.add_argument("--dims", type=int, default=32, help="PCA dims (keep <= 64: NebulaStream SQL limit)")
    p.add_argument("--min-records", type=int, default=100, help="drop windows with fewer records (partial)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=str(ROOT / "data" / "iot_emb"))
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_records(a.db)
    clients = sorted(df.client.unique(), key=lambda c: int(c.removeprefix("client")))
    sensors = sorted(df.sensor.unique())
    window_ms = a.window_s * 1000

    # 1) window index per record, relative to each client's first record (tumbling windows)
    df["window"] = (df.ts_ms - df.groupby("client").ts_ms.transform("min")) // window_ms
    windows = df.groupby(["client", "window"]).agg(
        n_records=("reading", "size"), raw_bytes=("raw_bytes", "sum"),
        content_bytes=("content_bytes", "sum"), window_mean=("reading", "mean"),
        sensors_seen=("sensor", "nunique")).reset_index()
    windows = windows[windows.n_records >= a.min_records]
    # equal number of windows per worker (the dashboard expects rows_per_worker for every worker)
    per_client = int(windows.groupby("client").size().min())
    windows = windows.sort_values(["client", "window"]).groupby("client").head(per_client).reset_index(drop=True)

    # 2) 160-dim feature vector per window: mean reading per sensor, gaps filled with the window mean
    feats = df.pivot_table(index=["client", "window"], columns="sensor", values="reading", aggfunc="mean")
    feats = feats.reindex(columns=sensors).loc[list(zip(windows.client, windows.window))]
    X = feats.to_numpy(copy=True)
    fill = np.repeat(windows.window_mean.to_numpy()[:, None], X.shape[1], axis=1)
    missing = np.isnan(X)
    X[missing] = fill[missing]

    # 3) standardise, one shared PCA, unit length -> dot product at B == cosine similarity
    Z = (X - X.mean(axis=0)) / X.std(axis=0)
    pca = PCA(n_components=a.dims, random_state=a.seed).fit(Z)
    emb = unit_rows(pca.transform(Z)).astype(np.float32)
    explained = float(pca.explained_variance_ratio_.sum())
    reference = unit_rows(emb.mean(axis=0, keepdims=True))[0]

    emb_cols = [f"EMB_{i}" for i in range(a.dims)]
    windows["WORKER_ID"] = windows.client.map({c: i + 1 for i, c in enumerate(clients)})
    windows["TS"] = (windows.window * window_ms).astype(np.int64)
    for wid in range(1, len(clients) + 1):
        sel = (windows.WORKER_ID == wid).to_numpy()
        part = pd.DataFrame(emb[sel], columns=emb_cols)
        part.insert(0, "TS", windows.TS[sel].to_numpy())
        part.insert(0, "WORKER_ID", wid)
        part.to_csv(out / f"worker_{wid:02d}.csv", header=False, index=False, float_format="%.6f")

    meta = pd.DataFrame({
        "WORKER_ID": windows.WORKER_ID, "TS": windows.TS, "GROUP": windows.client,
        "RAW_BYTES": windows.raw_bytes, "CONTENT_BYTES": windows.content_bytes, "N_RECORDS": windows.n_records,
        "SNIPPET": [f"{c} window {int(w)} ({w * a.window_s}-{(w + 1) * a.window_s} s): {n} records, "
                    f"{s}/{len(sensors)} sensors" for c, w, n, s in
                    zip(windows.client, windows.window, windows.n_records, windows.sensors_seen)],
    })
    meta.to_csv(out / "meta.csv", index=False)

    schema = {
        "columns": [{"name": "WORKER_ID", "type": "UINT64"}, {"name": "TS", "type": "UINT64"}]
                   + [{"name": c, "type": "FLOAT32"} for c in emb_cols],
        "dims": a.dims,
        "dataset": "TPCx-IoT",
        "reference_group": "all-windows centroid",
        "reference": [round(float(x), 6) for x in reference],
        "workers": {str(i + 1): [c] for i, c in enumerate(clients)},
        "window_s": a.window_s,
        "features": len(sensors),
        "pca_explained_variance": round(explained, 4),
        "rows_per_worker": per_client,
    }
    (out / "schema.json").write_text(json.dumps(schema, indent=2))

    row_bytes = 16 + 4 * a.dims
    n = len(windows)
    print(f"{len(df):,} records, {len(clients)} clients, {len(sensors)} sensors -> {n} windows of {a.window_s}s "
          f"({per_client} per worker, median {int(windows.n_records.median())} records/window, "
          f"median {int(windows.sensors_seen.median())}/{len(sensors)} sensors seen)")
    # baseline: the same PCA on pure Gaussian noise of the same shape (n windows x n sensors)
    noise = np.random.default_rng(a.seed).standard_normal(Z.shape)
    noise_explained = PCA(n_components=a.dims, random_state=a.seed).fit(noise).explained_variance_ratio_.sum()
    print(f"PCA {len(sensors)} -> {a.dims} dims keeps {explained:.1%} of the variance "
          f"(pure noise of the same shape: {noise_explained:.1%})")
    print(f"bytes: embeddings {n * row_bytes / 1024:,.0f} KB | raw records as generated "
          f"{windows.raw_bytes.sum() / 1024:,.0f} KB ({windows.raw_bytes.sum() / (n * row_bytes):,.0f}x) | "
          f"content only (key + reading) {windows.content_bytes.sum() / 1024:,.0f} KB "
          f"({windows.content_bytes.sum() / (n * row_bytes):,.0f}x)")
    print(f"wrote {len(clients)} worker CSVs, schema.json, meta.csv to {out}/")


if __name__ == "__main__":
    main()
