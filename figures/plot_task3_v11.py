"""
plot_task3.py
=============
Generates a figure for Task 3 results:
    1. plots/task3_avg.pdf - LLM-as-Judge % Correct with BERTScore overlay

Usage:
        python plot_task3.py
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
CSV_PATH = os.path.join(SCRIPT_DIR, "task3_new.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "plots")

DIALECT_COLORS = {
    "MSA":       "#2A9D8F",
    "Emirati":   "#F4C430",
    "Moroccan":  "#F07C00",
    "Jordanian": "#0D3349",
    "Egyptian":  "#C0392B",
}
DIALECTS = list(DIALECT_COLORS.keys())

CATEGORIES = [
    ("Closed-Source\nGeneral Purpose", [
        "gemini-2.5-flash", "Claude-Opus-4.6",
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
        "Mistral + 5-Shot", "Mistral + AUTOCAP", "BiMedix",
        "Mistral + Full LoRA", "Mistral + Ours",
    ]),
]

SHORT = {
    "gemini-2.5-flash":                   "Gemini\n2.5F",
    "Claude-Opus-4.6":                    "Claude\nO4.6",
    "Mistral-7B-Instruct-v0.3":           "Mistral\n7B",
    "Llama-3.1-8B-Instruct":              "Llama3.1\n8B",
    "Mistral-Small-3.2-24B-Instruct-2506":"Mistral\n24B",
    "Gemma-3-27B-it":                     "Gemma3\n27B",
    "Llama-3.3-70B":                      "Llama3.3\n70B",
    "DeepSeek-V3.2":                      "DeepSeek\nV3",
    "Jais-2-8B-Chat":                     "Jais\n8B",
    "ALLaM-7B-Instruct-preview":          "ALLaM\n7B",
    "aya-expanse-8b":                     "Aya\n8B",
    "SILMA-9B-Instruct-v1.0":             "SILMA\n9B",
    "Falcon-H1-7B":                       "Falcon\n7B",
    "Fanar-9B":                           "Fanar\n9B",
    "MedGemma-27b-text-it":               "MedGemma\n27B",
    "Meditron-3-70B":                     "Meditron\n70B",
    "Med42-70B":                          "Med42\n70B",
    "Mistral + 5-Shot":                   "M+\n5-Shot",
    "Mistral + AUTOCAP":                  "M+\nAutoCAP",
    "BiMedix":                            "BiMedix",
    "Mistral + Full LoRA":                "M+\nLoRA",
    "Mistral + Ours":                     "M+\nOurs",
}

CAT_SHADES = ["#f5f5f5", "#eaf4f2", "#fdf6ec", "#eef2f8", "#fdf0f0"]

matplotlib.rcParams.update({
    "font.family":       "serif",
    "axes.labelsize":    13,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "legend.fontsize":   12,
    "axes.titlesize":    12,
    "figure.titlesize":  13,
    "hatch.linewidth":   0.4,
})

# ── Parse CSV ─────────────────────────────────────────────────────────────────
raw = pd.read_csv(CSV_PATH, header=None)
all_models = [m for _, ms in CATEGORIES for m in ms]

def sf(v):
    try: return float(str(v).replace(",", ".").strip())
    except: return np.nan

def _norm_cell(v):
    if pd.isna(v):
        return ""
    return " ".join(str(v).split())

def _find_header_row(df, match_fn):
    for i, row in df.iterrows():
        if any(match_fn(str(cell).lower()) for cell in row):
            return i
    raise ValueError("Could not find a header row in the CSV.")

def _find_dialect_row(df, header_row):
    for i in range(header_row - 1, -1, -1):
        row = df.iloc[i]
        if any(_norm_cell(cell) in DIALECTS for cell in row):
            return i
    raise ValueError("Could not find a dialect header row in the CSV.")

def _build_col_map(df, match_fn):
    header_row = _find_header_row(df, match_fn)
    dialect_row = _find_dialect_row(df, header_row)
    col_map = {}
    header = df.iloc[header_row]
    dialects = df.iloc[dialect_row]
    for col_idx, cell in enumerate(header):
        if match_fn(str(cell).lower()):
            for left in range(col_idx, -1, -1):
                label = _norm_cell(dialects.iloc[left])
                if label in DIALECTS:
                    col_map[label] = col_idx
                    break
    missing = [d for d in DIALECTS if d not in col_map]
    if missing:
        raise ValueError(f"Missing columns for: {', '.join(missing)}")
    return col_map, header_row

DIALECT_CORRECT_COLS, correct_header_row = _build_col_map(
    raw, lambda s: "%correct" in s
)
DIALECT_BERT_COLS, _ = _build_col_map(
    raw, lambda s: "bertscore" in s
)
DATA_START_ROW = correct_header_row + 1

METRICS = {
    "avg": {
        "cols":    DIALECT_CORRECT_COLS,
        "ylabel":  "LLM-as-Judge % Correct",
        "title":   "Multi-Turn Clinical Dialogue: LLM-as-Judge % Correct across Dialects and Model Categories",
        "outpath": os.path.join(OUTPUT_DIR, "task3_avg.pdf"),
        "ylim":    (0, 100),
        "cat_label_y": 86,
        "plot_type": "bar",
    },
}

def get_scores(col_map):
    scores = {}
    for _, row in raw.iloc[DATA_START_ROW:].iterrows():
        name = " ".join(str(row.iloc[0]).split())
        matched = next((m for m in all_models if
                        " ".join(m.split()) == name or name in m or m in name), None)
        if matched:
            scores[matched] = {d: sf(row.iloc[c]) for d, c in col_map.items()}
    return scores

def _lighten_hex(hex_color, factor=0.55):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"

# ── Compute x positions (shared across both plots) ────────────────────────────
BAR_W     = 0.32
GROUP_GAP = 1.10
CAT_GAP   = 0.70
nd = len(DIALECTS)
offsets = np.arange(nd) * BAR_W - np.arange(nd).mean() * BAR_W

x_pos = {}
cat_extents = []
x_cursor = 0.0
for ci, (cat_label, models) in enumerate(CATEGORIES):
    for mi, model in enumerate(models):
        x_pos[model] = x_cursor
        x_cursor += BAR_W * nd + GROUP_GAP
    first_x = x_pos[models[0]]
    last_x  = x_pos[models[-1]]
    half_group = (BAR_W * nd) / 2 + GROUP_GAP / 4
    cat_extents.append((first_x - half_group, last_x + half_group, cat_label, ci))
    x_cursor += CAT_GAP

# ── Plot function ─────────────────────────────────────────────────────────────
def make_plot(metric_key):
    cfg          = METRICS[metric_key]
    correct_scores = get_scores(cfg["cols"])
    bert_scores    = get_scores(DIALECT_BERT_COLS)
    ylo, yhi     = cfg["ylim"]

    fig, ax = plt.subplots(figsize=(20, 6))

    # Category shading & separators
    for x0, x1, label, ci in cat_extents:
        ax.axvspan(x0, x1, color=CAT_SHADES[ci], alpha=0.6, zorder=0)
    for x0, x1, label, ci in cat_extents[1:]:
        ax.axvline(x0, color="#cccccc", linewidth=0.8, linestyle="--", zorder=1)

    # ── Bar plot (Percent Correct) ─────────────────────────────────────────
    bar_width = BAR_W * 0.88
    for model in all_models:
        if model not in x_pos:
            continue
        cx      = x_pos[model]
        scores  = correct_scores.get(model, {})
        b_scores = bert_scores.get(model, {})
        is_ours = (model == "Mistral + Ours")
        group_max_val = None
        for j, d in enumerate(DIALECTS):
            val_correct = scores.get(d, np.nan)
            val_bert    = b_scores.get(d, np.nan)
            if np.isnan(val_correct) and np.isnan(val_bert):
                continue
            base_color = DIALECT_COLORS[d]
            if is_ours:
                local_max = np.nanmax([val_correct, val_bert])
                if not np.isnan(local_max):
                    group_max_val = local_max if group_max_val is None else max(group_max_val, local_max)
            is_bert_front = (
                not np.isnan(val_correct)
                and not np.isnan(val_bert)
                and val_bert <= val_correct
            )
            light_factor = 0.55
            light_color = _lighten_hex(base_color, factor=light_factor)
            bars = []
            if not np.isnan(val_correct):
                bars.append((val_correct, base_color))
            if not np.isnan(val_bert):
                bars.append((val_bert, light_color))
            bars.sort(key=lambda item: item[0], reverse=True)
            for idx, (val, color) in enumerate(bars):
                x_center = cx + offsets[j]
                ax.bar(x_center, val, width=bar_width,
                       color=color, alpha=1.0, zorder=2 + idx)
                if idx == len(bars) - 1:
                    ax.hlines(val, x_center - bar_width / 2, x_center + bar_width / 2,
                              colors="#111111", linewidth=1.0, zorder=6)
        if is_ours and group_max_val is not None:
            ax.scatter(
                cx,
                group_max_val + 6.0,
                s=480,
                marker="*",
                facecolors="#111111",
                edgecolors="white",
                linewidths=0.8,
                zorder=9,
            )

    tick_xs     = [x_pos[m] for m in all_models if m in x_pos]
    tick_labels = [SHORT.get(m, m) for m in all_models if m in x_pos]
    d_handles = [mpatches.Patch(facecolor=DIALECT_COLORS[d], label=d, alpha=0.88)
                 for d in DIALECTS]
    d_handles.append(mpatches.Patch(facecolor="#666666", alpha=1.0,
                                    label="LLM-as-Judge % Correct (dark)"))
    d_handles.append(mpatches.Patch(facecolor="#dddddd", alpha=1.0,
                                    label="BERTScore (light)"))
    d_handles.append(plt.Line2D([0], [0], marker="*", color="w",
                                markerfacecolor="#111111",
                                markeredgecolor="white",
                                markersize=10,
                                label="Mistral + Ours"))
    fig.legend(handles=d_handles, title="Dialect", title_fontsize=13,
               fontsize=12, loc="upper left", bbox_to_anchor=(0.01, 1.02),
               framealpha=0.9, prop={"family": "serif"}, ncol=7)

    # Category labels
    for x0, x1, label, ci in cat_extents:
        ax.text((x0 + x1) / 2, cfg["cat_label_y"], label,
                ha="center", va="bottom", fontsize=15,
                fontfamily="serif", fontweight="bold", color="#333333")

    # X-axis
    ax.set_xticks(tick_xs)
    ax.set_xticklabels(tick_labels, fontsize=9, fontfamily="serif",
                       ha="center", va="top")
    ax.tick_params(axis="x", pad=4)

    # Y-axis
    ax.set_ylim(ylo, yhi)
    ax.set_ylabel(cfg["ylabel"], fontsize=15, fontfamily="serif")
    ax.yaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.86])
    fig.savefig(cfg["outpath"], format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved -> {cfg['outpath']}")
# ── Run ───────────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
make_plot("avg")