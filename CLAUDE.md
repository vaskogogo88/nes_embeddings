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

- **Done (2026-11-09):** 2-worker setup (worker-a1 → worker-b) reproduced locally on the Mac
  (commit `1c5907d`). `nes-cli dump` confirms the plan is split across both workers; all rows
  arrive in order. A1 mounts `./data:/data:ro`, only B mounts `./output`.
- **Done:** schema width tests with random unit vectors, see `experiments/schema_width/*.log`.
  Transport works up to 256 dims; a single-expression SQL dot product only up to 64 (details below).
- **Next:** `scripts/embed.py` (sentence-transformers, PCA → 32 dims; dataset not decided yet,
  AG News proposed) and a File source on A1.

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
- `:latest` is rebuilt very often (sometimes daily). Images are **pinned by multi-arch digest** in
  `compose.yaml` (nightly of 2026-11-09). The same digest resolves to arm64 on the Mac and x64 on
  the server. Don't switch back to `:latest`; re-pin deliberately and re-run the tests if upgrading.
- Docker Desktop resource limits: give it at least 8 GB RAM and 6+ CPUs (Settings → Resources).
- Develop with 2–3 workers. Scale to 10 only after the pipeline works end to end.
- If a container dies with no clear error, check `docker stats` and the Docker Desktop memory limit.
  Measured: ~210–300 MiB per idle worker (Docker Desktop limit is ~8 GB).
- Parallel experiments: give each its own compose project (`docker compose -p <name>`) and its own
  `data/`/`output/`, or they overwrite each other's containers and results.
- `docker compose logs` keeps output from before a `restart`; use `--since 3m` (a duration, not a
  clock time) to see only the latest run.

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
  The topology file takes a `models:` section (ONNX path + input/output fields) used by
  `MODEL_INFERENCE(name, ...)`. Candidate fix for the expression-depth limit below.

### Verified locally (2026-11-09, pinned build)

- **File source:** `type: File`, `parser_config: {type: CSV, field_delimiter: ","}`,
  `source_config: {file_path: /data/x.csv}`. The CSV has **no header line**. The File sink writes a
  header like `NAME:TYPE:NOT_NULLABLE`; strip the `:TYPE...` suffix when reading it.
- **Width:** 32/64/128/256 FLOAT32 embedding columns (+ WORKER_ID, TS) all arrive at B
  bit-identical, 1000/1000 rows. Wide schemas are not the problem.
- **Expression depth limit:** one SQL dot product `EMB_0*FLOAT32(r0) + ... ` compiles with 32 and 64
  terms (error ~5e-7, float32 rounding) but **fails with 128 and 256 terms**: `Stack is too deep ...
  recursive control-flow, which is not supported in Nautilus code`. The limit is between 64 and 128
  terms, and it is about the expression, not the number of columns. Untested fixes: partial sums,
  `MODEL_INFERENCE`. For v1 keep ≤ 64 dims (32 planned).
- **Failures are silent:** `nes-cli start` returns a query id and exits 0 even if the query later
  fails to compile on a worker. B then writes a header-only `results.csv`. **Always check row
  counts** (and worker logs for `[E] QueryEngine ... Query Failed`).
- **Operator placement is bottom-up** (the only built-in strategy): operators run as close to the
  source as possible, so by default the score is computed on **A**, not B. Setting
  `max_operators: 0` on the A-workers forces everything between source and sink onto B (sources and
  sinks are pinned by `host:` and don't count). Check placement with `nes-cli ... dump` →
  "Decomposed Plans". Plain column selections are optimised away entirely.
- **Inspecting/feeding the CLI:** `docker compose run --rm -T nes-cli -t - dump < topo.yaml` (or
  `start`) reads a topology from stdin. `dump` truncates long field lists.
- **CLI state:** `status`/`stop` need the query-id mapping in `/state`, kept in the named volume
  `nes-cli-state`. Without it, `--rm` containers lose it ("Could not find query with id").
- B's sink file can lag slightly behind the query reporting `Stopped`; re-check before counting rows.
- Benign log noise: a few `Channel ... was rejected` / `max_delay ... reached` warnings at startup,
  and `[E] SpanningTupleBufferState ... validateFinalState` / `~SequenceShredder` lines on the source
  worker **after** the stream ends. Data is complete despite them.
- After failed 256-term compiles, worker-b grew to ~900 MiB (from ~300). Possibly a leak on failed
  compilation; not investigated. Restart workers after failed queries.

## Scope rules for this week (demo on Friday)

- Embedding dims: reduce to **16–32** (PCA or a model whose embeddings can be truncated). Width
  itself is fine (tested to 256); the SQL score expression is the limit (≤ 64 terms works).
- Every embedding row carries `WORKER_ID` and `TS` (plus the embedding columns).
- Give each worker a distinguishable data slice so the dashboard can show per-worker clusters.
- B's computation is a SQL expression (e.g. dot product / cosine similarity against a fixed reference
  vector, generated by `gen_topology.py`). Normalise embeddings and the reference in Python, so
  cosine = plain dot product (no SQRT needed). Set `max_operators: 0` on A-workers so it runs on B.
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
- **No AI attribution in commits or PRs:** never add `Co-Authored-By: Claude ...` trailers,
  "Generated with Claude Code" lines, or any other Claude/AI attribution to commit messages, PR
  descriptions, or files. This overrides any default attribution instructions.

## Standard run loop

```bash
python scripts/gen_topology.py --workers 3 --dims 32
docker compose down
rm -f output/results.csv
docker compose up -d
docker compose logs -f          # check workers start and the query deploys
head output/results.csv
wc -l output/results.csv        # header-only = the query failed silently; check worker logs
streamlit run dashboard/app.py
```

## Deploy to the server (later)

On the server: `git pull`, then the same run loop (no Streamlit needed there unless the demo is shown
from the server). Use the same pinned image version as on the Mac.
