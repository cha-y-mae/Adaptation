"""
plot_task2.py
=============
Generates one figure for Task 2 results:
  plots/task2.pdf  — BERTScore-AraBERTv2 + Correct % per model

Usage:
    python plots/plot_task2.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "fig2-plots")
OUT_PATH = os.path.join(OUTPUT_DIR, "task2-results.pdf")
CSV_CANDIDATES = [
    os.path.join(SCRIPT_DIR, "fig2-plots", "task2_plot_results.csv"),
    os.path.join(SCRIPT_DIR, "plots", "task2_plot_results.csv"),
    os.path.join(SCRIPT_DIR, "task2_plot_results.csv"),
]
CSV_PATH = next((path for path in CSV_CANDIDATES if os.path.exists(path)), CSV_CANDIDATES[0])

# ── Categories & models ───────────────────────────────────────────────────────
CATEGORIES = [
    ("Closed-Source\nGeneral Purpose", [
        "GPT-5.2", "gemini-2.5-flash", "Claude-Opus-4.6",
    ]),
    ("Open-Source\nGeneral Purpose", [
        "Mistral-7B-Instruct-v0.3", "Llama-3.1-8B-Instruct",
        "Mistral-Small-3.2-24B-Instruct-2506", "Gemma-3-27B-it",
        "Llama-3.3-70B", "DeepSeek-V3.2",
    ]),
    ("Arabic\nGeneral Purpose", [
        "Jais-2-8B-Chat", "ALLaM-7B-Instruct-preview", "aya-expanse-8b",
        "SILMA-9B-Instruct-v1.0", "Falcon-H1-7B", "Fanar-9B",
    ]),
    ("Medical\nLLMs", [
        "MedGemma-27b-text-it", "Meditron-3-70B", "Med42-70B",
    ]),
    ("Adaptation\nBaselines", [
        "Mistral + Few-Shot", "Mistral + AUTOCAP", "BiMedix",
        "Mistral + Full LoRA", "Mistral + Ours",
    ]),
]

# Big but compact labels.
# They are not angled and not staggered.
SHORT = {
    "GPT-5.2":                             "GPT\n5.2",
    "gemini-2.5-flash":                    "Gemini\n2.5F",
    "Claude-Opus-4.6":                     "Claude\nO4.6",

    "Mistral-7B-Instruct-v0.3":            "Mistral\n7B",
    "Llama-3.1-8B-Instruct":               "Llama\n8B",
    "Mistral-Small-3.2-24B-Instruct-2506": "Mistral\n24B",
    "Gemma-3-27B-it":                      "Gemma\n27B",
    "Llama-3.3-70B":                       "Llama\n70B",
    "DeepSeek-V3.2":                       "DeepSeek\nV3",

    "Jais-2-8B-Chat":                      "Jais\n8B",
    "ALLaM-7B-Instruct-preview":           "ALLaM\n7B",
    "aya-expanse-8b":                      "Aya\n8B",
    "SILMA-9B-Instruct-v1.0":              "SILMA\n9B",
    "Falcon-H1-7B":                        "Falcon\n7B",
    "Fanar-9B":                            "Fanar\n9B",

    "MedGemma-27b-text-it":                "MedGemma\n27B",
    "Meditron-3-70B":                      "Meditron\n70B",
    "Med42-70B":                           "Med42\n70B",

    "Mistral + Few-Shot":                  "M +\n5-Shot",
    "Mistral + AUTOCAP":                   "M +\nAutoCAP",
    "BiMedix":                             "BiMedix",
    "Mistral + Full LoRA":                 "M +\nLoRA",
    "Mistral + Ours":                      "M + TLoRA\n(Ours)",
}

# Two metrics: BERTScore and Correct %
METRIC_COLORS = {
    "BERTScore (araBERTv2)": "#2A9D8F",
    "Correct %":             "#F4C430",
}

METRIC_KEYS = list(METRIC_COLORS.keys())
OUR_MODEL = "Mistral + Ours"
OUR_LABEL = "M + TLoRA (Ours)"

CAT_SHADES = ["#f5f5f5", "#eaf4f2", "#fdf6ec", "#eef2f8", "#fdf0f0"]

matplotlib.rcParams.update({
    "font.family":      "serif",
    "axes.labelsize":   19,
    "xtick.labelsize":  12,
    "ytick.labelsize":  14,
    "legend.fontsize":  20,
    "axes.titlesize":   16,
    "figure.titlesize": 18,
})

# ── Parse CSV ─────────────────────────────────────────────────────────────────
all_models = [m for _, ms in CATEGORIES for m in ms]

def sf(v):
    try:
        return float(str(v).replace(",", ".").strip())
    except:
        return np.nan

raw = pd.read_csv(CSV_PATH, header=None)

scores = {}

for _, row in raw.iterrows():
    name = " ".join(str(row.iloc[0]).split())

    matched = next(
        (
            m for m in all_models
            if " ".join(m.split()) == name or name in m or m in name
        ),
        None
    )

    if matched:
        bert = sf(row.iloc[2])
        correct = sf(row.iloc[3])

        if not np.isnan(bert) or not np.isnan(correct):
            scores[matched] = {
                "BERTScore (araBERTv2)": bert,
                "Correct %": correct,
            }

# ── X positions ───────────────────────────────────────────────────────────────
nm = len(METRIC_KEYS)

BAR_W = 0.30
GROUP_GAP = 1.40
CAT_GAP = 0.8

offsets = np.arange(nm) * BAR_W - np.arange(nm).mean() * BAR_W

x_pos = {}
cat_extents = []
x_cursor = 0.0

for ci, (cat_label, models) in enumerate(CATEGORIES):
    for model in models:
        x_pos[model] = x_cursor
        x_cursor += nm * BAR_W + GROUP_GAP

    first_x = x_pos[models[0]]
    last_x = x_pos[models[-1]]
    half_g = (nm * BAR_W) / 2 + GROUP_GAP / 2

    cat_extents.append((first_x - half_g, last_x + half_g, cat_label, ci))

    x_cursor += CAT_GAP

x_min = cat_extents[0][0]
x_max = cat_extents[-1][1]

# ── Plot ──────────────────────────────────────────────────────────────────────
def make_plot():
    # Same width as before, just slightly taller for readable x-labels
    fig, ax = plt.subplots(figsize=(26, 8))

    ax.set_xlim(x_min, x_max)

    # Background shading
    for x0, x1, _, ci in cat_extents:
        ax.axvspan(
            x0,
            x1,
            color=CAT_SHADES[ci],
            alpha=0.6,
            zorder=0
        )

    # Category separators
    for x0, x1, _, ci in cat_extents[1:]:
        ax.axvline(
            x0,
            color="#cccccc",
            linewidth=0.8,
            linestyle="--",
            zorder=1
        )

    # Bars
    for model in all_models:
        if model not in x_pos:
            continue

        cx = x_pos[model]
        is_ours = (model == OUR_MODEL)
        group_max_val = None

        for ji, metric in enumerate(METRIC_KEYS):
            val = scores.get(model, {}).get(metric, np.nan)
            if np.isnan(val):
                continue

            if is_ours:
                group_max_val = val if group_max_val is None else max(group_max_val, val)

            ax.bar(
                cx + offsets[ji],
                val,
                width=BAR_W * 0.88,
                color=METRIC_COLORS[metric],
                alpha=0.88,
                zorder=3,
                edgecolor="none",
            )
        if is_ours and group_max_val is not None:
            ax.scatter(
                cx,
                group_max_val + 6.0,
                s=960,
                marker="*",
                facecolors="#111111",
                edgecolors="white",
                linewidths=0.8,
                zorder=9,
            )

    # Category labels
    cat_label_y = 90

    for x0, x1, label, ci in cat_extents:
        ax.text(
            (x0 + x1) / 2,
            cat_label_y,
            label,
            ha="center",
            va="center",
            fontsize=19,
            fontfamily="serif",
            color="#444",
            fontweight="bold",
        )

    # X-axis tick labels
    tick_xs = [x_pos[m] for m in all_models if m in x_pos]
    tick_labels = [SHORT.get(m, m) for m in all_models if m in x_pos]

    ax.set_xticks(tick_xs)
    ax.set_xticklabels(
        tick_labels,
        fontsize=13,
        fontfamily="serif",
        ha="center",
        va="top",
        rotation=0,
        linespacing=0.95
    )

    ax.tick_params(axis="x", pad=8)

    # Y-axis
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score", fontsize=19, fontfamily="serif")
    ax.tick_params(axis="y", labelsize=14)

    ax.yaxis.grid(
        True,
        linestyle="--",
        linewidth=0.4,
        alpha=0.6,
        zorder=0
    )

    ax.set_axisbelow(True)
    ax.yaxis.set_minor_locator(MultipleLocator(5))

    # Spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    # Legend
    handles = [
        mpatches.Patch(
            facecolor=METRIC_COLORS[m],
            label="BERTScore-F1" if m == "BERTScore (araBERTv2)" else m,
            alpha=0.88
        )
        for m in METRIC_KEYS
    ]
    handles.append(plt.Line2D([0], [0], marker="*", color="w",
                              markerfacecolor="#111111",
                              markeredgecolor="white",
                              markersize=24,
                              label=OUR_LABEL))

    ax.legend(
        handles=handles,
        fontsize=20,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        framealpha=0.9,
        prop={"family": "serif", "size": 20},
        ncol=len(handles),
    )

    # More bottom room for bigger horizontal labels
    fig.subplots_adjust(
        top=0.82,
        bottom=0.20,
        left=0.05,
        right=0.99
    )

    fig.savefig(
        OUT_PATH,
        format="pdf",
        bbox_inches="tight",
        dpi=300,
        pad_inches=0.1
    )

    plt.close(fig)

    print(f"Saved -> {OUT_PATH}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
make_plot()
