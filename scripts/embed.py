"""Dataset -> sentence embeddings -> PCA-reduced unit vectors -> one CSV per worker.

Outputs (in --out, default data/):
  worker_01.csv ... worker_NN.csv   WORKER_ID,TS,EMB_0..EMB_{d-1}  (no header; read by NebulaStream File sources)
  schema.json                       column list, dims, reference vector, worker->group mapping (read by gen_topology.py)
  meta.csv                          WORKER_ID,TS,GROUP,RAW_BYTES,SNIPPET  (dashboard only; NOT sent through NebulaStream,
                                    joined with B's output on (WORKER_ID, TS))

All vectors (and the reference) are unit length, so B's dot product == cosine similarity.
One PCA is fitted on the whole dataset and shared by all workers ("shared codebook").

Example:
  python scripts/embed.py --workers 2 --dims 32 --rows-per-worker 200
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def load_dataset():
    """Return (texts, labels): the only dataset-specific part of this script.

    To switch datasets, replace this function; it must return two equal-length lists of str.
    """
    from sklearn.datasets import fetch_20newsgroups

    # Strip headers/footers/quotes so posts cluster by content, not by e-mail addresses and
    # subject lines, and so RAW_BYTES measures the actual message.
    ds = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    texts, labels = [], []
    for text, target in zip(ds.data, ds.target):
        text = text.strip()
        if len(text) >= 50:  # drop empty / near-empty posts
            texts.append(text)
            labels.append(ds.target_names[target])
    return texts, labels


def embed_all(texts, model_name, cache_dir):
    """Full-size, unit-normalised sentence embeddings; cached so changing --dims doesn't re-embed."""
    cache = cache_dir / f"emb_{model_name.replace('/', '_')}_{len(texts)}.npy"
    if cache.exists():
        print(f"using cached embeddings {cache}")
        return np.load(cache)
    import torch
    from sentence_transformers import SentenceTransformer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"embedding {len(texts)} texts with {model_name} on {device} ...")
    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache, emb.astype(np.float32))
    return emb


def assign_groups(all_groups, workers, groups_per_worker, explicit):
    """Map worker id (1..N) -> list of groups. Default: alphabetical, consecutive groups per worker."""
    groups = explicit or sorted(all_groups)[: workers * groups_per_worker]
    unknown = set(groups) - set(all_groups)
    if unknown:
        raise SystemExit(f"unknown groups: {sorted(unknown)}")
    if len(groups) < workers * groups_per_worker:
        raise SystemExit(f"need {workers * groups_per_worker} groups, have {len(groups)}")
    return {w + 1: groups[w * groups_per_worker:(w + 1) * groups_per_worker] for w in range(workers)}


def unit_rows(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--dims", type=int, default=32, help="PCA dims (keep <= 64: NebulaStream SQL limit)")
    p.add_argument("--rows-per-worker", type=int, default=200)
    p.add_argument("--groups-per-worker", type=int, default=2)
    p.add_argument("--groups", default=None,
                   help="comma-separated groups to use instead of the alphabetical default "
                        "(workers*groups_per_worker of them, in worker order)")
    p.add_argument("--reference-group", default="sci.space",
                   help="B's score = cosine similarity to this group's centroid")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--ts-step-ms", type=int, default=100, help="TS spacing between a worker's rows")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    texts, labels = load_dataset()
    labels = np.array(labels)
    if a.reference_group not in set(labels):
        raise SystemExit(f"unknown reference group {a.reference_group!r}")
    mapping = assign_groups(set(labels), a.workers, a.groups_per_worker,
                            a.groups.split(",") if a.groups else None)

    # 1) embed everything, 2) one PCA for all workers, 3) back to unit length so dot == cosine
    full = embed_all(texts, a.model, out / "cache")
    pca = PCA(n_components=a.dims, random_state=a.seed).fit(full)
    reduced = unit_rows(pca.transform(full)).astype(np.float32)
    explained = float(pca.explained_variance_ratio_.sum())
    reference = unit_rows(reduced[labels == a.reference_group].mean(axis=0, keepdims=True))[0]

    # sample each worker's slice (balanced over its groups) and write the CSVs
    emb_cols = [f"EMB_{i}" for i in range(a.dims)]
    meta_rows = []
    for wid, groups in mapping.items():
        per_group = a.rows_per_worker // len(groups)
        idx = np.concatenate([
            rng.choice(np.flatnonzero(labels == g), size=min(per_group, int((labels == g).sum())), replace=False)
            for g in groups
        ])
        rng.shuffle(idx)  # interleave the worker's groups over time
        ts = np.arange(len(idx), dtype=np.int64) * a.ts_step_ms
        df = pd.DataFrame(reduced[idx], columns=emb_cols)
        df.insert(0, "TS", ts)
        df.insert(0, "WORKER_ID", wid)
        df.to_csv(out / f"worker_{wid:02d}.csv", header=False, index=False, float_format="%.6f")
        for i, t in zip(idx, ts):
            meta_rows.append({"WORKER_ID": wid, "TS": int(t), "GROUP": labels[i],
                              "RAW_BYTES": len(texts[i].encode("utf-8")),
                              "SNIPPET": " ".join(texts[i].split())[:120]})
    meta = pd.DataFrame(meta_rows)
    meta.to_csv(out / "meta.csv", index=False)

    schema = {
        "columns": [{"name": "WORKER_ID", "type": "UINT64"}, {"name": "TS", "type": "UINT64"}]
                   + [{"name": c, "type": "FLOAT32"} for c in emb_cols],
        "dims": a.dims,
        "reference_group": a.reference_group,
        "reference": [round(float(x), 6) for x in reference],
        "workers": {str(k): v for k, v in mapping.items()},
        "model": a.model,
        "pca_explained_variance": round(explained, 4),
        "rows_per_worker": a.rows_per_worker,
    }
    (out / "schema.json").write_text(json.dumps(schema, indent=2))

    # summary: compression + a sanity check of the score B will compute
    row_bytes = 2 * 8 + a.dims * 4  # WORKER_ID, TS as UINT64 + FLOAT32 per dim
    print(f"\n{len(texts)} documents, {len(set(labels))} groups; PCA {full.shape[1]} -> {a.dims} dims "
          f"keeps {explained:.1%} of the variance")
    print(f"raw text per row: median {int(meta.RAW_BYTES.median())} B, mean {int(meta.RAW_BYTES.mean())} B; "
          f"embedding row: {row_bytes} B -> {meta.RAW_BYTES.sum() / (row_bytes * len(meta)):.1f}x fewer bytes")
    for wid, groups in mapping.items():
        print(f"  worker {wid:2d}: {', '.join(groups)}")
    print(f"mean cosine similarity to {a.reference_group} centroid, per group:")
    for g in sorted(set(meta.GROUP)):
        sims = reduced[labels == g] @ reference
        print(f"  {g:28s} {sims.mean():+.3f}")
    print(f"wrote {a.workers} worker CSVs, schema.json, meta.csv to {out}/")


if __name__ == "__main__":
    main()
