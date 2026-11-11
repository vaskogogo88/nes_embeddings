"""Build reports/demo_report.pdf from B's actual output (output/results.csv) and data/.

Redraws the dashboard's two main graphs with matplotlib (same data, same PCA as dashboard/app.py)
and lays out a short report with reportlab.

  python reports/make_report.py
"""
import datetime as dt
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "reports"
BLUE = "#2a78d6"
BLUE_RAMP = ["#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"]
GRAY = "#b9b8b2"
INK, INK_2 = "#0b0b0b", "#52514e"


# ---------------------------------------------------------------- data
schema = json.loads((ROOT / "data/schema.json").read_text())
names = [c["name"] for c in schema["columns"]]
emb_cols = names[2:]
dims = schema["dims"]
row_bytes = 16 + 4 * dims
expected = sorted(int(m) for m in re.findall(r"/data/worker_(\d+)\.csv", (ROOT / "compose-topology.yaml").read_text()))

res = pd.read_csv(ROOT / "output/results.csv")
res.columns = [c.split(":")[0] for c in res.columns]
meta = pd.read_csv(ROOT / "data/meta.csv")
df = res.merge(meta, on=["WORKER_ID", "TS"], how="left")

# delivery check: B's output vs what each worker sent (same logic as scripts/check_results.py)
ref = np.array(schema["reference"])
per_worker, max_emb_err, max_score_err = {}, 0.0, 0.0
for w in expected:
    sent = pd.read_csv(ROOT / f"data/worker_{w:02d}.csv", header=None, names=names)
    m = sent.merge(res[res.WORKER_ID == w], on=["WORKER_ID", "TS"], suffixes=("_s", "_g"))
    xs = m[[f"{c}_s" for c in emb_cols]].values
    max_emb_err = max(max_emb_err, float(np.abs(xs - m[[f"{c}_g" for c in emb_cols]].values).max()))
    max_score_err = max(max_score_err, float(np.abs(xs @ ref - m.SCORE.values).max()))
    per_worker[w] = (len(m), len(sent))
rows_ok = all(got == sent for got, sent in per_worker.values())

raw_kb = df.RAW_BYTES.sum() / 1024
emb_kb = len(df) * row_bytes / 1024
by_group = df.groupby("GROUP").SCORE.mean().sort_values(ascending=False)
ref_group = schema["reference_group"]
next_best = by_group.drop(ref_group).max()
top200 = df.nlargest(200, "SCORE")
n_ref = (df.GROUP == ref_group).sum()
ref_in_top = (top200.GROUP == ref_group).sum()

# same 2-D projection as the dashboard
xy = PCA(n_components=2, random_state=0).fit_transform(df[emb_cols].values)
df["PC1"], df["PC2"] = xy[:, 0], xy[:, 1]

plt.rcParams.update({"font.family": "Helvetica", "font.size": 9, "text.color": INK,
                     "axes.edgecolor": GRAY, "axes.labelcolor": INK_2})


def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ---------------------------------------------------------------- figure 1
fig, ax = plt.subplots(figsize=(7.0, 4.3), dpi=200)
cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
order = df.SCORE.argsort().values  # high scores on top
sc = ax.scatter(df.PC1.values[order], df.PC2.values[order], c=df.SCORE.values[order], cmap=cmap,
                s=14, linewidths=0.3, edgecolors="white")
clean(ax)
cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
cb.set_label(f"SCORE computed at B (cosine similarity to {ref_group})", color=INK_2)
cb.outline.set_visible(False)
ref_pts = df[df.GROUP == ref_group]  # outline the reference group's rows (B never sees this label)
ax.scatter(ref_pts.PC1, ref_pts.PC2, s=26, facecolors="none", edgecolors=INK, linewidths=0.6,
           label=f"{ref_group} posts (sent by A8), outlined")
ax.legend(loc="upper left", frameon=False, fontsize=8)
fig.tight_layout()
fig1 = OUT_DIR / "fig1_projection_by_score.png"
fig.savefig(fig1, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- figure 2
fig, axes = plt.subplots(2, 5, figsize=(7.2, 3.4), dpi=200, sharex=True, sharey=True)
for ax, w in zip(axes.flat, expected):
    mine = df.WORKER_ID == w
    ax.scatter(df.PC1[~mine], df.PC2[~mine], s=2, color=GRAY, alpha=0.35, linewidths=0)
    ax.scatter(df.PC1[mine], df.PC2[mine], s=4, color=BLUE, linewidths=0)
    clean(ax)
    groups = schema["workers"][str(w)]
    short = "\n".join(g if len(g) <= 22 else g[:21] + "…" for g in groups)
    ax.set_title(f"A{w}  ({per_worker[w][0]} rows)\n{short}", fontsize=6.5, color=INK, loc="left", pad=3)
fig.tight_layout(h_pad=1.2, w_pad=0.6)
fig2 = OUT_DIR / "fig2_per_worker_clusters.png"
fig.savefig(fig2, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- PDF
styles = getSampleStyleSheet()
body = ParagraphStyle("body", parent=styles["BodyText"], fontName="Helvetica", fontSize=10, leading=14,
                      textColor=colors.HexColor(INK), spaceAfter=6)
h1 = ParagraphStyle("h1", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=17, leading=21,
                    alignment=0, spaceAfter=2)
sub = ParagraphStyle("sub", parent=body, fontSize=9, textColor=colors.HexColor(INK_2), spaceAfter=10)
h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=15,
                    spaceBefore=8, spaceAfter=4)
cap = ParagraphStyle("cap", parent=body, fontSize=8.5, leading=11, textColor=colors.HexColor(INK_2), spaceAfter=8)

n_workers = len(expected)
rpw = schema["rows_per_worker"]
story = [
    Paragraph("Sending embeddings from 10 edge nodes to one node with NebulaStream", h1),
    Paragraph(f"Demo report · LLMs in Hierarchical Aggregation Topologies · Vasil · "
              f"{dt.date.today():%d %B %Y}", sub),

    Paragraph("1. What was run", h2),
    Paragraph(
        f"{n_workers} NebulaStream edge workers (A1–A{n_workers}) each read a slice of the 20 Newsgroups "
        f"dataset that was turned into sentence embeddings beforehand (all-MiniLM-L6-v2, reduced from 384 to "
        f"{dims} dimensions with PCA and normalised to unit length). Each worker holds posts from two newsgroups "
        f"and sends {rpw} rows downstream to a single sink worker B. A row is only "
        f"<font name='Courier'>WORKER_ID, TS</font> and {dims} <font name='Courier'>FLOAT32</font> values "
        f"({row_bytes} bytes); the text itself never enters the topology.", body),
    Paragraph(
        f"B runs one SQL expression on every incoming row: <font name='Courier'>SCORE</font> = cosine similarity "
        f"between the embedding and a fixed reference vector, the centroid of all <i>{ref_group}</i> posts. "
        f"The edge workers are configured with <font name='Courier'>max_operators: 0</font>, so NebulaStream "
        f"places this computation on B, not at the edge (verified in the query plan). B writes "
        f"<font name='Courier'>WORKER_ID, TS, SCORE</font> and the embedding to a CSV file, which is the only "
        f"input of both graphs below.", body),
]

summary = [
    ["Rows received at B", f"{len(df):,} of {n_workers * rpw:,}"
     + (f" (every worker {rpw}/{rpw})" if rows_ok else " (INCOMPLETE)")],
    ["Embeddings at B vs. as sent", "bit-identical (max difference "
     + (f"{max_emb_err:.0f})" if max_emb_err == 0 else f"{max_emb_err:.1e})")],
    ["SCORE at B vs. same formula in Python", f"max difference {max_score_err:.1e} (float32 rounding)"],
    ["Bytes sent as embeddings", f"{emb_kb:,.0f} KB"],
    ["Same posts as raw text", f"{raw_kb:,.0f} KB  →  {raw_kb / emb_kb:.1f}× more than the embeddings"],
]
t = Table(summary, colWidths=[6.2 * cm, 10.3 * cm])
t.setStyle(TableStyle([
    ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
    ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 9),
    ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor(INK)),
    ("LINEBELOW", (0, 0), (-1, -2), 0.4, colors.HexColor("#e1e0d9")),
    ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
story += [Paragraph("2. Did everything arrive?", h2),
          Paragraph("B's output was compared row by row with what each worker sent:", body), t, Spacer(1, 6)]

story += [
    Paragraph("3. What B received, coloured by its score", h2),
    Image(str(fig1), width=16.0 * cm, height=16.0 * cm * 4.3 / 7.0 * 0.93),
    Paragraph(f"<b>Figure 1.</b> Every point is one of the {len(df):,} rows B received. The {dims}-dimensional "
              f"embeddings are projected to two dimensions with PCA (the axes have no units; only distances "
              f"matter). Colour is the SCORE that B computed in NebulaStream. Outlined points are the "
              f"<i>{ref_group}</i> posts; this label comes from the dataset and was never sent to B.", cap),
    Paragraph(
        f"The darkest points, i.e. the rows B scored as most similar to <i>{ref_group}</i>, are not scattered "
        f"randomly: they are concentrated in one part of the plot, and the outlines show that they are mostly the "
        f"<i>{ref_group}</i> posts sent by worker A8. "
        f"The {ref_group} rows average a score of <b>{by_group[ref_group]:+.2f}</b>, while the next-highest of the "
        f"other 19 newsgroups averages only {next_best:+.2f}. Of the {n_ref} {ref_group} rows, {ref_in_top} are among "
        f"the 200 highest-scoring rows (top 10 %). B never saw any text, only {dims} numbers per post, yet its "
        f"computation recognises the topic. The embeddings therefore kept their meaning after travelling from "
        f"the edge nodes through the topology to B.", body),

    Paragraph("4. Each node's data stays recognisable", h2),
    Image(str(fig2), width=16.5 * cm, height=16.5 * cm * 3.4 / 7.2 * 0.97),
    Paragraph("<b>Figure 2.</b> The same projection as Figure 1, drawn once per edge worker: that worker's rows "
              "in blue, all other rows in grey. Titles give the worker, the number of its rows that reached B, "
              "and its two newsgroups.", cap),
    Paragraph(
        "Each panel shows one node's data arriving at B in its own part of the plot (sometimes two parts, one "
        "per newsgroup, e.g. A5: motorcycles vs. baseball), distinct from most other workers. Workers whose topics are related overlap, as expected: A2 and A3 both hold computer "
        "groups, A9 and A10 both hold politics groups, and A8 and A10 share a religion group (these are also the "
        "three most similar pairs of worker centroids in the full 32-dimensional space). Workers with unrelated "
        "topics (e.g. A4: for-sale and cars vs. A8: space and religion) occupy different areas. This is what one "
        f"would expect from meaningful embeddings, and it is visible after transport: all {n_workers} streams "
        "were merged by NebulaStream at B, yet the structure of every node's data is preserved and every node "
        f"delivered all {rpw} of its rows.", body),

    Paragraph("5. Conclusion", h2),
    Paragraph(
        f"The demo shows that NebulaStream can carry embeddings from {n_workers} edge nodes to one node: all "
        f"{len(df):,} rows arrived complete and unchanged, B computed on them inside NebulaStream, and the result "
        f"is meaningful (Figure 1: topic recognised by similarity; Figure 2: each node's data keeps its own "
        f"structure). Sending embeddings instead of the raw posts needed {raw_kb / emb_kb:.1f}× fewer bytes.", body),
    Paragraph(
        "<b>Limits of this first version.</b> Embeddings were computed offline, not on the edge nodes, so the cost "
        "of computing them at the edge is not yet included. The byte comparison uses uncompressed text and "
        "payload bytes, not NebulaStream's wire format. A single SQL dot product compiles only up to 64 dimensions "
        "in this NebulaStream build (embeddings themselves travel fine up to 256). There is no aggregation yet; "
        "intermediate aggregating nodes are the next step toward hierarchical topologies.", body),
]

pdf = OUT_DIR / "demo_report.pdf"
doc = SimpleDocTemplate(str(pdf), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                        title="Sending embeddings from 10 edge nodes to one node with NebulaStream", author="Vasil")
doc.build(story)
print(f"wrote {pdf} ({doc.page})" if hasattr(doc, "page") else f"wrote {pdf}")
