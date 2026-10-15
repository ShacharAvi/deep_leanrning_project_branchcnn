"""Figures used in the final report (sections 5-6).

    python tools/report_figures.py      # -> results/figures/fig_*.png

Every number is read from a result file under results/cbp2025/, except the few
BranchCNN-without-TAGE values quoted inline from results/branchcnn_published/.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results" / "cbp2025"
OUT = REPO / "results" / "figures"

BLUE, ORANGE, GRAY, DARK = "#2B7BD6", "#EB6834", "#8A8A87", "#1F3E66"
BG, INK, MUTED, GRID = "#FCFCFA", "#111111", "#555555", "#E6E6E3"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#BBBBBB",
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.spines.top": False, "axes.spines.right": False,
})


def red(base, sys_):
    return 100.0 * (base - sys_) / base


def load(name):
    return json.loads((RES / "combined" / f"{name}.json").read_text())


def label_bars(ax, bars, fmt="{:+.2f}%", pad=0.6, size=7.5):
    for b in bars:
        w = b.get_width()
        x = w + pad if w >= 0 else w - pad
        ax.text(x, b.get_y() + b.get_height() / 2, fmt.format(w), va="center",
                ha="left" if w >= 0 else "right", fontsize=size, color=INK)


def finish(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


# --- 5.1: TAGE-SC-L state is what makes either network useful ---------------
def fig_tage_state():
    pilot = json.loads((RES / "crosstrace" / "web_g0_learned.json").read_text())["variants"]
    rows = [  # (label, without TAGE state, with TAGE state)
        ("Specialists, attach every model\n(web_g0 pilot)",
         100 * pilot["tokens"]["attach_all"]["mpki_reduction"],
         100 * pilot["tage_raw"]["attach_all"]["mpki_reduction"]),
        # BranchCNN: results/branchcnn_published/data/branchcnn_all_runs.csv
        ("Generalist BranchCNN\n(fold 1, web_32+33)", -72.2, 10.3),
        ("Generalist BranchCNN\n(fold 2, web_31)", -108.4, 8.3),
    ]
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    y = range(len(rows))
    h = 0.36
    b1 = ax.barh([i - h / 2 for i in y], [r[1] for r in rows], h, color=GRAY,
                 label="without TAGE-SC-L state")
    b2 = ax.barh([i + h / 2 for i in y], [r[2] for r in rows], h, color=BLUE,
                 label="with TAGE-SC-L state")
    label_bars(ax, list(b1) + list(b2), fmt="{:+.1f}%", pad=1.5)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(list(y), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(-125, 25)
    ax.set_xlabel("MPKI reduction vs TAGE-SC-L (%)  (positive = fewer mispredictions)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False, fontsize=8)
    finish(fig, "fig_5_1_tage_state.png")


# --- 5.2: data volume helps the specialists, not the generalist --------------
def fig_data_scaling():
    f1, big = load("fold1_tage_raw"), load("big16_tage_raw")
    cnn = [red(d["tage"]["mpki"], d["branchcnn"]["mpki"]) for d in (f1, big)]
    h2h_f1 = json.loads((RES / "crosstrace" / "bcnn_web_f1.json").read_text())
    h2h_big = json.loads((RES / "crosstrace" / "bcnn_web_big.json").read_text())
    spec = [100 * h["variants"]["tage_raw"]["learned_lr_fb"]["mpki_reduction"]
            for h in (h2h_f1, h2h_big)]

    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.5),
                               gridspec_kw={"width_ratios": [1, 1]})
    x = [0, 1]
    a.plot(x, cnn, "-o", color=ORANGE, lw=2, ms=6, label="generalist BranchCNN")
    a.plot(x, spec, "-o", color=BLUE, lw=2, ms=6, label="specialists + arbiter")
    for xi, v in zip(x, cnn):
        a.annotate(f"{v:+.2f}%", (xi, v), textcoords="offset points", xytext=(0, 7),
                   ha="center", fontsize=7.5)
    for xi, v in zip(x, spec):
        a.annotate(f"{v:+.2f}%", (xi, v), textcoords="offset points", xytext=(0, 7),
                   ha="center", fontsize=7.5)
    a.set_xticks(x, ["4 training traces", "16 training traces"])
    a.set_xlim(-0.4, 1.4)
    a.set_ylim(-1, 14)
    a.set_ylabel("MPKI reduction vs TAGE-SC-L (%)")
    a.set_title("(a) whole program, same test traces", fontsize=8.5, color=INK, loc="left")
    a.legend(frameon=False, fontsize=7.5, loc="center right")

    # Specialists on their 16 hard branches, vs TAGE on those branches.
    # Source: per-branch rows of results/cbp2025/crosstrace/bcnn_web_{f1,big}.json.
    alone, arb = [-10.9, 10.0], [2.5, 11.5]
    w = 0.36
    b1 = b.bar([i - w / 2 for i in x], alone, w, color=GRAY, label="model alone")
    b2 = b.bar([i + w / 2 for i in x], arb, w, color=BLUE, label="model + arbiter")
    for bar in list(b1) + list(b2):
        v = bar.get_height()
        b.text(bar.get_x() + bar.get_width() / 2, v + (0.6 if v >= 0 else -0.6),
               f"{v:+.1f}%", ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5)
    b.axhline(0, color=MUTED, lw=1)
    b.set_xticks(x, ["4 training traces", "16 training traces"])
    b.set_ylim(-15, 15)
    b.set_ylabel("misprediction reduction vs TAGE (%)")
    b.set_title("(b) specialists on their 16 hard branches", fontsize=8.5, color=INK, loc="left")
    b.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout(w_pad=2.5)
    finish(fig, "fig_5_2_data_scaling.png")


# --- 5.4: combining the generalist and the specialists -----------------------
def fig_combination():
    runs = [("fold 1 (4 traces)", load("fold1_tage_raw")),
            ("fold 2 (5 traces)", load("fold2_tage_raw")),
            ("16 training traces", load("big16_tage_raw"))]
    systems = [("BranchCNN alone", "branchcnn"), ("A: arbiter(TAGE, BranchCNN)", "A_tage|cnn"),
               ("B: arbiter(BranchCNN, specialists)", "B_cnn|spec"), ("C: three-way", "C_full")]
    colors = ["#C9C9C6", GRAY, BLUE, DARK]
    fig, ax = plt.subplots(figsize=(6.3, 2.7))
    w = 0.2
    for j, (lab, key) in enumerate(systems):
        vals = [red(d["tage"]["mpki"], d[key]["mpki"]) for _, d in runs]
        xs = [i + (j - 1.5) * w for i in range(len(runs))]
        bars = ax.bar(xs, vals, w * 0.92, color=colors[j], label=lab)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.15, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(range(len(runs)), [r[0] for r in runs])
    ax.set_ylim(0, 14)
    ax.set_ylabel("MPKI reduction vs TAGE-SC-L (%)")
    ax.legend(frameon=False, fontsize=7.2, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.12))
    finish(fig, "fig_5_4_combination.png")


# --- 6.1: per branch, the combination beats both models ----------------------
def fig_per_branch():
    d = load("big16_tage_raw")["B_cnn|spec"]["per_branch"]
    rows = sorted(d.items(), key=lambda kv: -kv[1]["incumbent_misses"])
    labels = [f"…{pc[-4:]}" for pc, _ in rows]
    spec = [red(v["incumbent_misses"], v["specialist_alone_misses"]) for _, v in rows]
    comb = [red(v["incumbent_misses"], v["combined_misses"]) for _, v in rows]
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    x = range(len(rows))
    w = 0.4
    ax.bar([i - w / 2 for i in x], spec, w * 0.92, color=GRAY, label="specialist alone")
    ax.bar([i + w / 2 for i in x], comb, w * 0.92, color=BLUE,
           label="BranchCNN + specialist via arbiter")
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(list(x), labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("fewer mispredictions\nthan BranchCNN (%)")
    ax.set_xlabel("hard branch (sorted by BranchCNN's mispredictions, largest first)")
    ax.legend(frameon=False, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    finish(fig, "fig_6_1_per_branch.png")


if __name__ == "__main__":
    fig_tage_state()
    fig_data_scaling()
    fig_combination()
    fig_per_branch()
