# nes_embeddings

Prototype for "LLMs in Hierarchical Aggregation Topologies": NebulaStream nodes exchange
**embeddings instead of raw data**.

In this first demo, 10 edge workers (A1–A10) each send sentence embeddings of their slice of the
20 Newsgroups dataset to one sink worker (B). B computes a similarity score on the embeddings it
receives, and a live dashboard shows that

1. the embeddings stay meaningful after travelling through the topology, and
2. they need far fewer bytes than the raw text (8.4× fewer in the default run).

```
20 Newsgroups ──> scripts/embed.py ──> data/worker_01.csv … worker_10.csv   (offline, Python)
                                              │
          worker-a1 … worker-a10   (NebulaStream, File source each, send downstream)
                                              │
          worker-b   (SQL: SCORE = cosine similarity to the sci.space centroid → output/results.csv)
                                              │
          dashboard/app.py   (Streamlit, reads output/results.csv, refreshes every 2 s)
```

Embeddings are 32-dim (MiniLM sentence embeddings reduced with PCA) and unit length, so B's
dot product is a cosine similarity.

## Requirements

- Docker (Docker Desktop on macOS: give it ≥ 8 GB RAM). The NebulaStream images are pinned by digest
  in `compose.yaml` and run natively on Apple Silicon (arm64) and x86-64.
- Python 3.11+ for the scripts and the dashboard.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To only run the pipeline and check its output, `pandas` and `numpy` are enough. The pre-generated
data in `data/` is committed, so PyTorch / sentence-transformers are only needed to regenerate it.

## Run the demo (10 workers, pre-generated data)

```bash
mkdir -p output && chmod a+rwX output    # the workers write as uid 10000
docker compose down
rm -f output/results.csv                 # avoid looking at stale results
docker compose up -d                     # starts A1–A10 and B; nes-cli submits the query and exits 0
python scripts/check_results.py          # rows per worker, embeddings unchanged, SCORE correct
streamlit run dashboard/app.py           # open http://localhost:8501
```

All 2000 rows arrive within a few seconds. `check_results.py` should end with `OK` and show
`200/200 rows` for every worker.

Stop everything with `docker compose down`.

## Change the setup

The pipeline is generated, never hand-edited:

```bash
# 1. data: embeddings for N workers (downloads the dataset and model once, ~1 min on an M3)
python scripts/embed.py --workers 10 --dims 32 --rows-per-worker 200

# 2. config: compose.yaml + compose-topology.yaml for the first N workers of that data
python scripts/gen_topology.py --workers 3

# 3. run as above
```

Useful options:

| Script | Option | Meaning |
|---|---|---|
| `embed.py` | `--workers N` | number of edge workers (max 10 with 2 newsgroups each, 20 with `--groups-per-worker 1`) |
| `embed.py` | `--dims D` | embedding size after PCA. **Keep ≤ 64** (see limits below) |
| `embed.py` | `--rows-per-worker R` | rows each worker sends |
| `embed.py` | `--reference-group G` | B scores similarity to this newsgroup (default `sci.space`) |
| `embed.py` | `--groups g1,g2,…` | pick the newsgroups explicitly, e.g. so a 2-worker run includes `sci.space` |
| `gen_topology.py` | `--workers N` | use the first N workers of the generated data |
| `gen_topology.py` | `--score-at-edge` | let NebulaStream compute the score on the edge workers instead of B |

To try another dataset, replace `load_dataset()` in `scripts/embed.py`. It is the only
dataset-specific function and returns `(texts, labels)`.

## Dashboard

`streamlit run dashboard/app.py` shows:

- rows received at B, workers reporting, and bytes sent as embeddings vs. the same posts as raw text;
- a 2D PCA projection of the embeddings B received, coloured by B's score (hover for the post);
- B's mean score per newsgroup and the highest-scoring posts;
- one small plot per worker (its cluster vs. everything else) and rows received per worker.

It only listens on `localhost` (`.streamlit/config.toml`). On a remote server, use an SSH tunnel:
`ssh -L 8501:localhost:8501 <server>`, then open http://localhost:8501 locally.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `results.csv` has a header but no rows | The query failed on a worker. NebulaStream fails **silently**: `nes-cli` still prints a query id. Check `docker compose logs worker-b` for `Query Failed`. |
| `nes-cli` cannot connect | Workers weren't ready yet. Rerun `docker compose run --rm nes-cli`. |
| `Channel ... was rejected`, `max_delay ... reached` warnings at startup | Harmless race while workers start. |
| `[E] ... validateFinalState` / `~SequenceShredder` after a run | Harmless teardown messages; the data is complete. |
| `Port 8501 is not available` | Another Streamlit is running. Stop it, or use `--server.port 8502`. |
| Old results on the dashboard | Delete `output/results.csv` before each run. |

Query status / stop (the query id is printed by `docker compose logs nes-cli`):

```bash
docker compose run --rm nes-cli -t /catalog/compose-topology.yaml status <query-id>
docker compose run --rm nes-cli -t /catalog/compose-topology.yaml stop <query-id>
```

## Known NebulaStream limits (pinned build, 2026-11-09)

- Embeddings of 32, 64, 128 and 256 dims all travel A → B bit-exactly.
- The SQL score is one long expression. It compiles up to **64** terms but fails at 128+
  (`Stack is too deep`). Details: `experiments/schema_width/`.
- Operators are placed as close to the source as possible. `gen_topology.py` sets
  `max_operators: 0` on the edge workers so the score is computed on B.

## Repository layout

```
compose.yaml, compose-topology.yaml   generated by scripts/gen_topology.py
scripts/embed.py                      dataset -> embeddings -> data/
scripts/gen_topology.py               data/schema.json -> compose files
scripts/check_results.py              verify B's output against what the workers sent
dashboard/app.py                      Streamlit dashboard
data/                                 generated data (committed); data/cache/ is ignored
experiments/schema_width/             schema-width test results (66/130/258 columns)
output/                               B's output (ignored)
```
