"""Live dashboard for the embeddings-over-NebulaStream demo.

Reads B's sink output (output/results.csv) and the dashboard-only metadata written by embed.py or
embed_iot.py (<data-dir>/meta.csv), joined on (WORKER_ID, TS). Refreshes every few seconds.

  streamlit run dashboard/app.py                              # 20 Newsgroups (data/)
  streamlit run dashboard/app.py -- --data-dir data/iot_emb   # TPCx-IoT
"""
import argparse
import os
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "output" / "results.csv"
_p = argparse.ArgumentParser()
_p.add_argument("--data-dir", default=os.environ.get("NES_DATA_DIR", "data"),
                help="folder with schema.json and meta.csv (or env NES_DATA_DIR)")
DATA = ROOT / _p.parse_known_args()[0].data_dir
TOPOLOGY = ROOT / "compose-topology.yaml"

# Reference palette (dataviz skill, references/palette.md): one hue for identity/magnitude,
# a neutral for context. Text never uses the series colour.
BLUE = "#2a78d6"
BLUE_RAMP = ["#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"]
CONTEXT_GRAY = "#b9b8b2"

st.set_page_config(page_title="Embeddings over NebulaStream", layout="wide")


@st.cache_data
def load_static():
    schema = json.loads((DATA / "schema.json").read_text())
    meta = pd.read_csv(DATA / "meta.csv")
    import re
    expected = sorted(int(m) for m in re.findall(r"/data/worker_(\d+)\.csv", TOPOLOGY.read_text()))
    return schema, meta, expected


@st.cache_data
def load_results(mtime):  # mtime in the cache key -> reload only when the sink file changes
    df = pd.read_csv(RESULTS, on_bad_lines="skip")  # the last line may be half-written while B is writing
    df.columns = [c.split(":")[0] for c in df.columns]
    return df.apply(pd.to_numeric, errors="coerce").dropna()


def stat(col, label, value, help_text=None):
    col.metric(label, value, help=help_text)


schema, meta, expected = load_static()
dims = schema["dims"]
row_bytes = 16 + 4 * dims  # WORKER_ID + TS (UINT64) + dims x FLOAT32
emb_cols = [f"EMB_{i}" for i in range(dims)]
IOT = schema.get("dataset") == "TPCx-IoT"
# dataset-specific wording; everything else is shared
L = dict(item="window", items="windows", group="substation", raw="Same windows as raw records",
         raw_help="TPCx-IoT records as generated (~1 KB each, mostly filler)") if IOT else \
    dict(item="post", items="posts", group="newsgroup", raw="Same posts as raw text",
         raw_help="UTF-8 size of the post bodies")

st.title("Embeddings over NebulaStream")
if IOT:
    st.caption(f"{len(expected)} edge workers (A1–A{len(expected)}) = TPCx-IoT substations. Each sends one "
               f"{dims}-dim embedding per {schema['window_s']} s window, summarising the {schema['features']} sensors "
               f"in it, to worker B. B computes SCORE = cosine similarity to the **{schema['reference_group']}**. "
               f"Note: TPCx-IoT readings are uniform random numbers, so clusters and scores show noise by design.")
else:
    st.caption(f"{len(expected)} edge workers (A1–A{len(expected)}) send {dims}-dim embeddings of 20 Newsgroups "
               f"posts to worker B. B computes SCORE = cosine similarity to the **{schema['reference_group']}** "
               f"centroid. Only B's output file is read here; newsgroup labels and raw sizes come from embed.py's "
               f"metadata.")


@st.fragment(run_every="2s")
def live():
    if not RESULTS.exists():
        st.info("Waiting for output/results.csv … start the pipeline with `docker compose up -d`.")
        return
    res = load_results(RESULTS.stat().st_mtime)
    if res.empty:
        st.warning("results.csv has a header but no rows yet. If this persists, the query failed "
                   "(NebulaStream fails silently) – check `docker compose logs worker-b`.")
        return
    if len(res.columns) - 3 != dims:  # WORKER_ID, TS, SCORE + dims embedding columns
        st.error(f"output/results.csv has {len(res.columns) - 3} embedding columns but {DATA.name}/schema.json "
                 f"says {dims}: the running pipeline was generated from a different data folder. Start the "
                 f"dashboard with the matching --data-dir.")
        return
    df = res.merge(meta, on=["WORKER_ID", "TS"], how="left")
    unmatched = df.GROUP.isna().mean()
    if unmatched > 0.01:
        st.error(f"{unmatched:.0%} of B's rows have no metadata in {DATA.relative_to(ROOT)}/meta.csv: the running "
                 f"pipeline was generated from a different data folder. Start the dashboard with the matching "
                 f"--data-dir (e.g. `streamlit run dashboard/app.py -- --data-dir data/iot_emb`).")
        return
    df["WORKER"] = "A" + df.WORKER_ID.astype(int).astype(str)
    expected_rows = schema["rows_per_worker"] * len(expected)

    # ---- 1. headline numbers -----------------------------------------------------------------------
    raw = df.RAW_BYTES.sum()
    emb = len(df) * row_bytes
    reporting = df.WORKER_ID.nunique()
    c = st.columns(6 if IOT else 5)
    stat(c[0], "Rows received at B", f"{len(df):,} / {expected_rows:,}")
    stat(c[1], "Workers reporting", f"{reporting} / {len(expected)}")
    stat(c[2], "Sent as embeddings", f"{emb / 1024:,.0f} KB", f"{row_bytes} B per row: 2×UINT64 + {dims}×FLOAT32")
    stat(c[3], L["raw"], f"{raw / 1024:,.0f} KB", L["raw_help"])
    stat(c[4], "Bytes saved", f"{raw / emb:,.{0 if raw / emb >= 100 else 1}f}× fewer", "raw bytes ÷ embedding bytes")
    if IOT:  # honest comparison: TPCx-IoT records are padded to ~1 KB, the real content is key + reading
        content = df.CONTENT_BYTES.sum()
        stat(c[5], "vs. content only", f"{content / emb:,.0f}× fewer",
             f"key + reading = {content / 1024:,.0f} KB, i.e. without the generator's filler")

    left, right = st.columns([3, 2])

    # ---- 2. 2D projection of what B received, coloured by B's score ---------------------------------
    xy = PCA(n_components=2, random_state=0).fit_transform(df[emb_cols].values)
    df["PC1"], df["PC2"] = xy[:, 0], xy[:, 1]
    tooltip = ["WORKER", "GROUP", alt.Tooltip("SCORE", format="+.3f"), "SNIPPET"]
    with left:
        st.subheader("What B received: 2D projection, coloured by score")
        scatter = alt.Chart(df).mark_circle(size=40, opacity=0.85, stroke="white", strokeWidth=0.5).encode(
            x=alt.X("PC1", axis=alt.Axis(title=None, labels=False, ticks=False, grid=False)),
            y=alt.Y("PC2", axis=alt.Axis(title=None, labels=False, ticks=False, grid=False)),
            color=alt.Color("SCORE", title=f"Similarity to {schema['reference_group']}",
                            scale=alt.Scale(range=BLUE_RAMP)),
            order="SCORE",  # high scores drawn on top
            tooltip=tooltip,
        ).properties(height=430)
        st.altair_chart(scatter, width="stretch")
        st.caption("PCA of the received 32-dim embeddings down to 2 dims (axes have no units). "
                   "Hover a point to see its post.")

    # ---- 3. score per newsgroup ---------------------------------------------------------------------
    with right:
        st.subheader(f"Score computed at B, by {L['group']}")
        by_group = df.groupby("GROUP").agg(SCORE=("SCORE", "mean"), N=("SCORE", "size")).reset_index()
        bars = alt.Chart(by_group).mark_bar(color=BLUE, cornerRadiusEnd=4, height=12).encode(
            x=alt.X("SCORE", title="Mean cosine similarity"),
            y=alt.Y("GROUP", sort="-x", title=None),
            tooltip=["GROUP", alt.Tooltip("SCORE", format="+.3f"), alt.Tooltip("N", title="rows")],
        )
        zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=CONTEXT_GRAY).encode(x="x")
        st.altair_chart((zero + bars).properties(height=430), width="stretch")

    # ---- 4. small multiples: each worker's slice against everything B received ----------------------
    st.subheader("Per-worker clusters")
    base = alt.Chart(df[["PC1", "PC2"]]).mark_circle(size=8, color=CONTEXT_GRAY, opacity=0.35).encode(
        x=alt.X("PC1", axis=None), y=alt.Y("PC2", axis=None))
    cols = st.columns(5)
    for k, wid in enumerate(expected):
        w = df[df.WORKER_ID == wid]
        groups = ", ".join(schema["workers"][str(wid)])
        with cols[k % 5]:
            st.markdown(f"**A{wid}** · {len(w)} rows  \n<small>{groups}</small>", unsafe_allow_html=True)
            pts = alt.Chart(w).mark_circle(size=18, color=BLUE, opacity=0.9).encode(
                x="PC1", y="PC2", tooltip=tooltip)
            st.altair_chart((base + pts).properties(height=150), width="stretch")

    # ---- 5. rows per worker (health) + top posts ----------------------------------------------------
    left, right = st.columns([2, 3])
    with left:
        st.subheader("Rows received per worker")
        counts = pd.DataFrame({"WORKER_ID": expected})
        counts = counts.merge(df.groupby("WORKER_ID").size().rename("ROWS").reset_index(), how="left").fillna(0)
        counts["WORKER"] = "A" + counts.WORKER_ID.astype(str)
        rows = alt.Chart(counts).mark_bar(color=BLUE, cornerRadiusEnd=4, size=14).encode(
            x=alt.X("ROWS", title="Rows", scale=alt.Scale(domain=[0, schema["rows_per_worker"]])),
            y=alt.Y("WORKER", sort=[f"A{w}" for w in expected], title=None),
            tooltip=["WORKER", "ROWS"])
        st.altair_chart(rows.properties(height=28 * len(expected)), width="stretch")
        missing = counts[counts.ROWS < schema["rows_per_worker"]]
        if len(missing):
            st.caption("⚠ Still arriving or missing: " +
                       ", ".join(f"{r.WORKER} ({int(r.ROWS)}/{schema['rows_per_worker']})"
                                 for r in missing.itertuples()))
        else:
            st.caption(f"✓ All {len(expected)} workers delivered {schema['rows_per_worker']} rows.")
    with right:
        st.subheader(f"Highest-scoring {L['items']} (most similar to {schema['reference_group']})")
        top = df.nlargest(10, "SCORE")[["WORKER", "GROUP", "SCORE", "SNIPPET"]]
        st.dataframe(top, hide_index=True, width="stretch",
                     column_config={"SCORE": st.column_config.NumberColumn(format="%+.3f")})


live()
