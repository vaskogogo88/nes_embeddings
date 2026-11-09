# CLAUDE.md — Embeddings over NebulaStream (MSc thesis demo)

## Project in one paragraph

MSc thesis: "LLMs in Hierarchical Aggregation Topologies". Core idea (from my professor): instead of
nodes in a NebulaStream topology exchanging **raw data**, they exchange **embeddings**. This repo holds
the first demo: 10 edge workers (A1–A10) each send embeddings of their slice of a dataset downstream to
one sink worker (B). B does a simple computation on them (e.g. cosine similarity to a reference vector),
and a live dashboard visualises the result. There is **no aggregation** in this first version.

Goal of the demo: show that (1) embeddings stay meaningful after travelling through the topology, and
(2) how many bytes they save compared to the raw data.

## Working setup

- Everything runs **locally on my MacBook Pro (M3 Pro, Apple Silicon)**: NebulaStream in Docker
  Desktop, the Python scripts, and the dashboard.
- Claude Code runs in this folder and **may run commands itself**: `docker compose up/down/logs/ps`,
  `python`, `pip`, `streamlit`. Run things and read the logs instead of guessing.
- The university server (DMSL) is a **later deployment target only**. Code reaches it through git
  (`git push` here, `git pull` there). Never hardcode server hostnames, usernames, passwords or IPs.
- Only touch Docker resources that belong to this project (the compose project in this folder). No
  `docker system prune`, and don't remove other containers, images or volumes.
- Use a Python virtualenv in `.venv/` (gitignored). Don't install packages globally.

## Current state

Working (on the server, to be re-created locally first): 2 workers via Docker Compose. Worker A has a
Generator source, sends data `downstream` to Worker B, and B writes `/output/results.csv`. Verified
that data really travels A → B (only B has the output volume mounted, and changing the query changed
the output).

**First task locally:** reproduce this 2-worker setup on the Mac before building anything new.

## Target architecture (v1 demo)

```
dataset ──> embed.py (Python, offline) ──> 10 CSV files (one per worker)
                                              │
        A1 ... A10  (NebulaStream workers, File source each, downstream → B)
                                              │
        B  (sink worker: query computes similarity score, File sink → CSV)
                                              │
        dashboard (Streamlit, reads B's output CSV, refreshes live)
```

Embeddings are computed **outside** NebulaStream for v1. NebulaStream only transports them and runs
the SQL on B. Running the model inside NebulaStream is out of scope until the professor decides.

## Planned file layout

```
compose.yaml               # generated: docker services (worker-a1..aN, worker-b, nes-cli)
compose-topology.yaml      # generated: NebulaStream query, sources, sinks, workers
scripts/
  embed.py                 # dataset -> reduced embeddings -> data/worker_XX.csv
  gen_topology.py          # N workers + dims -> compose.yaml + compose-topology.yaml
dashboard/
  app.py                   # Streamlit dashboard
data/                      # generated CSVs (gitignored if large)
output/                    # B's sink output (mounted into worker-b only, gitignored)
requirements.txt
.gitignore                 # .venv/, output/, data/ (if large), .env
```

Generate files that repeat per worker with `gen_topology.py`. Never hand-write N copies of a block.

## Docker on the Mac

- `nebulastream/nes-worker` and `nebulastream/nes-cli` are published for **linux/arm64**, so they run
  natively on Apple Silicon (no emulation). Don't add `platform: linux/amd64`.
- `:latest` is rebuilt very often (sometimes daily). **Pin one version** in `compose.yaml` (a dated tag
  or an image digest) once the 2-worker setup works, so the Mac and the server run the same build.
- Docker Desktop resource limits: give it at least 8 GB RAM and 6+ CPUs (Settings → Resources).
- Develop with 2–3 workers. Scale to 10 only after the pipeline works end to end.
- If a container dies with no clear error, check `docker stats` and the Docker Desktop memory limit.

## NebulaStream facts learned so far (trust these over guesses)

Docs: https://nebulastream.github.io/nebulastream/ (quick start, distributed, API concepts) and
`docs/guide/nebulastream-frontend.md` in https://github.com/nebulastream/nebulastream.
Ignore the `nebulastream-tutorial` repo, which is outdated (old coordinator/UI architecture).

- Images: `nebulastream/nes-worker` and `nebulastream/nes-cli`. `nes-repl-embedded` is **single-node only**.
- A worker = a node. Sources and sinks are *attached to* workers via `host:`. A worker can have a
  source, a sink, both, or neither (relay).
- Topology is **declared**, not wired by hand: each worker entry has `host`, `data_address`,
  `max_operators`, and optionally `downstream: [ "worker-b:8080" ]`. NebulaStream creates the network
  channels itself.
- Ports: 8080 = gRPC/control, 9090 = data between workers. Each worker is started with
  `-- --grpc=0.0.0.0:8080 --data_address=<hostname>:9090`.
- The compose service name/hostname must match the names used in the topology file.
- Only the sink worker needs `./output:/output` mounted.
- `nes-cli` exits with code 0 after submitting the query. That's normal; workers keep running.
  `depends_on` doesn't wait for workers to be ready. If the CLI can't connect, rerun it with
  `docker compose run --rm nes-cli`.
- Warnings like `Channel '...' was rejected` right after start are a benign race (the receiver isn't
  ready yet). They're only a problem if they never stop.
- Schema types: INT8–INT64, UINT8–UINT64, FLOAT32, FLOAT64, CHAR, BOOLEAN, VARSIZED.
  **No array/vector type.** Embeddings = one `FLOAT32` column per dimension (`EMB_0..EMB_{d-1}`).
- Incoming data must match the schema exactly, **including column order**. A mismatch terminates
  the query. Generate schema and CSV from the same code.
- Source types: File, TCP, MQTT, Generator. Input formats: CSV, JSON. Use File for v1; TCP later.
- Multiple physical sources of one logical source are treated as one stream (verify row counts).
- When checking results, delete `output/results.csv` first or change the query. Otherwise you may be
  looking at stale output from a previous run.
- YAML error "invalid node; ... map iterator as a sequence iterator" usually means a key the parser
  expected is missing, or the wrong file is mounted. It's not necessarily a syntax error.
- The docs mention ML model inference inside queries (FLOAT32 tensor inputs). Not for v1, but note it.

## Scope rules for this week (demo on Friday)

- Embedding dims: reduce to **16–32** (PCA or a model whose embeddings can be truncated). Wide
  schemas are the biggest NebulaStream risk.
- Every embedding row carries `WORKER_ID` and `TS` (plus the embedding columns).
- Give each worker a distinguishable data slice so the dashboard can show per-worker clusters.
- B's computation is a SQL expression (e.g. dot product / cosine similarity against a fixed reference
  vector, generated by `gen_topology.py`).
- Dashboard reads B's output CSV. No custom sinks.
- Bring things up incrementally: 2 workers (A1 → B) → 3 → 10. If 10 doesn't work by Tuesday
  evening, demo with 3.
- Deploy to the server and run it there once by **Thursday**, so server-only problems don't show up
  on Friday.
- Not this week: TCP live streaming, model inference inside NebulaStream, aggregation, multi-tier
  (A → B → C) topologies, node add/remove at runtime.

## Dashboard must show

1. Live 2D projection (PCA/UMAP) of received embeddings, coloured by `WORKER_ID`.
2. The similarity/anomaly score computed at B.
3. Bytes sent as embeddings vs. the equivalent raw data (the thesis argument).
4. Rows received per worker (doubles as a health check).

## Conventions

- Python 3.11+, deps in `requirements.txt`, Streamlit for the dashboard.
- Embedding model: small and CPU/Apple-GPU friendly (e.g. a sentence-transformers model); download
  once and cache it.
- Scripts take CLI args (`--workers 10 --dims 32`) instead of hardcoded constants.
- Keep it simple and readable. This is a demo and a thesis artifact, not production code.
- Commit working steps to git with clear messages (they double as a progress log for the thesis).

## Standard run loop

```bash
python scripts/gen_topology.py --workers 3 --dims 32
docker compose down
rm -f output/results.csv
docker compose up -d
docker compose logs -f          # check workers start and the query deploys
head output/results.csv
streamlit run dashboard/app.py
```

## Deploy to the server (later)

On the server: `git pull`, then the same run loop (no Streamlit needed there unless the demo is shown
from the server). Use the same pinned image version as on the Mac.
