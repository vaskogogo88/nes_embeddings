# nes_embeddings

Prototype for "LLMs in Hierarchical Aggregation Topologies": NebulaStream nodes exchange
**embeddings instead of raw data**.

Edge workers (A1–A10) each send embeddings of their own data to one sink worker (B). B computes a
similarity score on the embeddings it receives, and a live dashboard shows what arrived, how many
bytes the embeddings saved, and whether they kept their meaning.

The same pipeline runs on two datasets; only the data step differs:

| Dataset | One embedding = | Result |
|---|---|---|
| **TPCx-IoT** (default) | one substation × one 60 s window of its 160 sensors | 350/350 windows delivered exactly; **182× fewer bytes** than the windows' content (4,284× vs. the generated, padded records). The generator's readings are uniform random, so the embeddings carry no meaning (see `reports/iot_report.pdf`). |
| **20 Newsgroups** | one forum post (MiniLM sentence embedding) | 2,000/2,000 posts delivered exactly; **8.4× fewer bytes**; B's score recognises the reference topic (`reports/demo_report.pdf`). |

```
data (offline, Python) ──> data/<dataset>/worker_01.csv … worker_10.csv
                                     │
        worker-a1 … worker-a10   (NebulaStream, File source each, send downstream)
                                     │
        worker-b   (SQL: SCORE = cosine similarity to a reference vector → output/results.csv)
                                     │
        dashboard/app.py   (Streamlit, reads output/results.csv, refreshes every 2 s)
```

Embeddings are 32-dim and unit length (PCA-reduced), so B's dot product is a cosine similarity.
The edge workers have `max_operators: 0`, which makes NebulaStream compute the score on B.

## Requirements

- Docker (Docker Desktop on macOS: give it ≥ 8 GB RAM). The NebulaStream images are pinned by digest
  in `compose.yaml` and run natively on Apple Silicon (arm64) and x86-64.
- Python 3.11+:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The generated embeddings for both datasets are committed (`data/iot_emb/`, `data/`), so running the
demo needs neither the TPCx-IoT kit nor PyTorch.

## Run the demo (TPCx-IoT, the committed default)

```bash
mkdir -p output && chmod a+rwX output    # the workers write as uid 10000
docker compose down
rm -f output/results.csv                 # avoid looking at stale results
docker compose up -d                     # starts A1–A10 and B; nes-cli submits the query and exits 0
python scripts/check_results.py --data-dir data/iot_emb       # should end with OK, 35/35 per worker
streamlit run dashboard/app.py -- --data-dir data/iot_emb     # open http://localhost:8501
```

## Run the demo on 20 Newsgroups

```bash
python scripts/gen_topology.py --data-dir data   # regenerate compose files for the Newsgroups data
docker compose down && rm -f output/results.csv && docker compose up -d
python scripts/check_results.py                  # should end with OK, 200/200 per worker
streamlit run dashboard/app.py                   # default --data-dir is data
```

The dashboard, `check_results.py` and the reports must use the same data folder as the running
pipeline. The dashboard shows an error, and the report scripts stop, if they do not match.

## TPCx-IoT data from scratch

The TPCx-IoT kit is **not in git** (its licence forbids redistribution). Put `TPCx-IoT_Tools_v2.1.2/`
in the repository root (it is git-ignored); Java is required.

```bash
# 1. run the kit's generator (print mode, no database) and store its records in SQLite (~2 s, ~300 MB)
python scripts/iot_to_sqlite.py --clients 10 --records-per-client 20000 --fresh   # -> data/iot/tpcx_iot.db

# 2. one embedding per substation per window -> data/iot_emb/
python scripts/embed_iot.py --window-s 60 --dims 32

# 3. compose files for it, then run as above
python scripts/gen_topology.py --data-dir data/iot_emb
```

What the generator produces (kit v2.1.2): keys `client:sensor:timestamp` (100 ms apart, one random
sensor per record, 160 sensors), values of 1,000 characters that repeat a 4-decimal reading as filler.
The readings are uniform random numbers: no differences between substations, sensors or time.

## 20 Newsgroups data from scratch

```bash
python scripts/embed.py --workers 10 --dims 32 --rows-per-worker 200   # downloads dataset + model once
python scripts/gen_topology.py --data-dir data
```

To use another text dataset, replace `load_dataset()` in `scripts/embed.py`.

## Options

| Script | Option | Meaning |
|---|---|---|
| `gen_topology.py` | `--data-dir D` | which embeddings to run (`data` or `data/iot_emb`) |
| `gen_topology.py` | `--workers N` | use the first N workers of that data |
| `gen_topology.py` | `--score-at-edge` | let NebulaStream compute the score on the edge workers instead of B |
| `embed_iot.py` | `--window-s S` | window length per embedding |
| `embed.py` / `embed_iot.py` | `--dims D` | embedding size after PCA. **Keep ≤ 64** (see limits below) |
| `embed.py` | `--reference-group G` | B scores similarity to this newsgroup (default `sci.space`) |
| `iot_to_sqlite.py` | `--clients`, `--records-per-client` | how much TPCx-IoT data to generate |

## Reports

```bash
python reports/make_report_iot.py   # -> reports/iot_report.pdf   (needs the TPCx-IoT run + data/iot/tpcx_iot.db)
python reports/make_report.py       # -> reports/demo_report.pdf  (needs the Newsgroups run)
cd reports && latexmk -pdf demo_report.tex   # LaTeX version of the Newsgroups report
```

Every number in the reports is computed from `output/results.csv` and the data folder.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `results.csv` has a header but no rows | The query failed on a worker. NebulaStream fails **silently**: `nes-cli` still prints a query id. Check `docker compose logs worker-b` for `Query Failed`. |
| Dashboard: "the running pipeline was generated from a different data folder" | Start it with the `--data-dir` that `gen_topology.py` used. |
| `nes-cli` cannot connect | Workers weren't ready yet. Rerun `docker compose run --rm nes-cli`. |
| `Channel ... was rejected`, `max_delay ... reached` warnings at startup | Harmless race while workers start. |
| `[E] ... validateFinalState` / `~SequenceShredder` after a run | Harmless teardown messages; the data is complete. |
| `Port 8501 is not available` | Another Streamlit is running. Stop it, or use `--server.port 8502`. |

Query status / stop (the query id is printed by `docker compose logs nes-cli`):

```bash
docker compose run --rm nes-cli -t /catalog/compose-topology.yaml status <query-id>
docker compose run --rm nes-cli -t /catalog/compose-topology.yaml stop <query-id>
```

## Known NebulaStream limits (pinned build, 2026-11-09)

- Embeddings of 32, 64, 128 and 256 dims all travel A → B bit-exactly.
- The SQL score is one long expression. It compiles up to **64** terms but fails at 128+
  (`Stack is too deep`). Details: `experiments/schema_width/`.
- Operators are placed as close to the source as possible; `max_operators: 0` on the edge workers
  moves the score to B.
- One query per worker (each scoring against its own centroid on B) works for 10 concurrent queries:
  `experiments/multi_query/`.

## Repository layout

```
compose.yaml, compose-topology.yaml   generated by scripts/gen_topology.py
scripts/iot_to_sqlite.py              TPCx-IoT generator -> data/iot/tpcx_iot.db
scripts/embed_iot.py                  TPCx-IoT windows -> data/iot_emb/
scripts/embed.py                      20 Newsgroups -> data/
scripts/gen_topology.py               <data-dir>/schema.json -> compose files
scripts/check_results.py              verify B's output against what the workers sent
dashboard/app.py                      Streamlit dashboard (--data-dir)
reports/                              report scripts, PDFs, LaTeX version, figures
data/                                 Newsgroups embeddings (committed); data/cache/ ignored
data/iot_emb/                         TPCx-IoT embeddings (committed)
data/iot/                             TPCx-IoT records in SQLite (ignored, regenerate)
experiments/                          schema-width and multi-query test logs
output/                               B's output (ignored)
TPCx-IoT_Tools_v2.1.2/                TPC kit (ignored, licence)
```
