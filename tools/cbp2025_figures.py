#!/usr/bin/env python3
"""Figures for the final report, from saved result files only.

    python tools/cbp2025_figures.py      # -> results/cbp2025/figures/*.png

Palette: the reference categorical slots 1-3 (validated all-pairs with the
dataviz validator: CVD dE 9.2, normal-vision dE 27.6) plus neutral gray for
baselines and controls. Slot 3 (aqua) is below 3:1 contrast on white, so
every mark carries a direct value label.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results" / "cbp2025"
XT = RES / "crosstrace"
OUT = RES / "figures"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, LIGHTGRAY = "#8a8985", "#c9c8c3"
INK, INK2 = "#0b0b0b", "#52514e"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": LIGHTGRAY, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#ebeae6", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 11.5,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False,
})


def load(name):
    p = XT / name
    return json.loads(p.read_text()) if p.exists() else None


def pct(x):
    return "0.00%" if abs(x) < 0.005 else f"{x:+.2f}%"


def save(fig, name, note=None):
    if note:
        # As a subtitle under the title: never collides with axis labels.
        ax = fig.axes[0]
        ax.set_title(ax.get_title(loc="left"), loc="left", pad=20)
        ax.text(0, 1.015, note, transform=ax.transAxes, fontsize=8,
                color=INK2, ha="left", va="bottom")
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


# ---------------------------------------------------------------- 1. volume

def fig_volume():
    base, pool = load("web_g0.json"), load("web_g0_pool.json")
    if not base:
        return
    ns = [1, 2, 4, 8, 16, 29]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for variant, color, label in (("tokens", BLUE, "BranchNet-big (history only)"),
                                  ("tage_raw", ORANGE, "BranchNet-big + TAGE state + raw")):
        ys = []
        for n in ns:
            v = variant if n == 29 else f"{variant}#t{n}"
            ys.append(100 * base["variants"][v]["confidence"]["mpki_reduction"])
        xs = list(ns)
        if pool and variant in pool["variants"]:
            xs.append(43)
            ys.append(100 * pool["variants"][variant]["confidence"]["mpki_reduction"])
        ax.plot(xs, ys, color=color, lw=2, marker="o", ms=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        ax.annotate(pct(ys[-1]), (xs[-1], ys[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=INK, fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 29, 43])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "29", "43*"])
    ax.set_xlabel("training traces (inputs of the same program)")
    ax.set_ylabel("MPKI reduction vs TAGE-SC-L (%)")
    ax.set_title("More inputs keep helping: no plateau at 43 traces")
    ax.legend(loc="upper left")
    ax.set_xlim(0.8, 60)
    save(fig, "fig1_data_volume.png",
         "web_g0, same 9 unseen test traces at every point; threshold arbiter. "
         "*43 = 29 + 14 pooled traces of the same program.")


# -------------------------------------------------------------- 2. policies

def policy_values(report, shared, variant):
    v = report["variants"][variant]
    out = [("BranchNet rule (per branch)", v["branch"]["mpki_reduction"]),
           ("Threshold arbiter (ours)", v["confidence"]["mpki_reduction"])]
    key = "learned_lr_fb" if "learned_lr_fb" in v else "learned_lr"
    if key in v:
        out.append(("Learned arbiter (ours)", v[key]["mpki_reduction"]))
    if shared and variant in shared["variants"]:
        out.append(("Shared + per-branch correction (ours)",
                    shared["variants"][variant]["shared_lr_corrected"]["mpki_reduction"]))
    return out


def fig_policies():
    progs = [("web_g0 (pooled, 43 train traces)", load("web_g0_pool.json"),
              load("web_g0_pool_shared.json"), BLUE),
             ("infra_g0 (fold 0 only; see Fig. 6)", load("infra_g0.json"),
              load("infra_g0_shared.json"), ORANGE)]
    progs = [p for p in progs if p[1]]
    rows = [name for name, _ in policy_values(progs[0][1], progs[0][2], "tage_raw")]
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    h = 0.36
    for i, (label, rep, sh, color) in enumerate(progs):
        vals = dict(policy_values(rep, sh, "tage_raw"))
        ys = np.arange(len(rows)) + (i - 0.5) * (h + 0.04)
        xs = [100 * vals.get(r, np.nan) for r in rows]
        ax.barh(ys, xs, height=h, color=color, label=label)
        for y, x in zip(ys, xs):
            if x == x:
                ax.text(x + (0.05 if x >= 0 else -0.05), y, pct(x), va="center",
                        ha="left" if x >= 0 else "right", fontsize=8.5, color=INK)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows)
    ax.invert_yaxis()
    ax.axvline(0, color=GRAY, lw=1)
    ax.set_xlabel("MPKI reduction vs TAGE-SC-L (%)")
    ax.set_title("Deciding per execution beats attaching per branch")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    save(fig, "fig2_policies.png",
         "BranchNet-big + TAGE state + raw features; unseen test traces; "
         "all arbiters fitted on validation traces only.")


# ------------------------------------------------------------------- 3. gap

GAP = {  # from tools/cbp2025_gap_analysis.py (variant tage_raw)
    "web_g0": dict(total=155501, both=97201, recoverable=58300, recovered=45228),
    "infra_g0": dict(total=274500, both=212053, recoverable=62447, recovered=19856),
}


def fig_gap():
    fig, ax = plt.subplots(figsize=(7.4, 2.8))
    names = list(GAP)
    for i, n in enumerate(names):
        g = GAP[n]
        parts = [("arbiter recovers", g["recovered"], BLUE),
                 ("recoverable, missed", g["recoverable"] - g["recovered"], ORANGE),
                 ("both wrong (no arbiter can fix)", g["both"], LIGHTGRAY)]
        left = 0
        for label, val, color in parts:
            w = 100 * val / g["total"]
            ax.barh(i, w - 0.3, left=left, height=0.55, color=color,
                    label=label if i == 0 else None)
            if w > 6:
                ax.text(left + w / 2, i, f"{w:.0f}%", ha="center", va="center",
                        fontsize=9, color=INK)
            left += w
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of TAGE-SC-L mispredictions on the modelled branches (%)")
    ax.set_title("What limits the hybrid: mostly cases where both are wrong")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.38), ncol=3)
    save(fig, "fig3_headroom.png")


# -------------------------------------------------------------- 4. features

def fig_features():
    rep = load("web_g0_learned.json")
    if not rep:
        return
    order = [("tokens", "history only (BranchNet-big)", BLUE),
             ("tage", "+ TAGE internal state", BLUE),
             ("raw_path", "+ raw path features", BLUE),
             ("raw_committed", "+ committed register values", BLUE),
             ("tage_raw", "+ TAGE state + all raw", BLUE),
             ("tage_raw@a3", "+ TAGE-unsure loss weighting", BLUE),
             ("tage_raw_shuffled", "control: features shuffled", GRAY),
             ("tokens_labelshuffle", "control: labels shuffled", GRAY)]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ys = np.arange(len(order))
    for y, (key, label, color) in zip(ys, order):
        x = 100 * rep["variants"][key]["learned_lr"]["mpki_reduction"]
        ax.barh(y, x, height=0.6, color=color)
        ax.text(x + 0.04, y, pct(x), va="center", fontsize=8.5, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([o[1] for o in order])
    ax.invert_yaxis()
    ax.set_xlabel("MPKI reduction vs TAGE-SC-L (%), learned arbiter")
    ax.set_title("Which inputs help BranchNet-big (web_g0 pilot)")
    save(fig, "fig4_features.png",
         "29 training traces, 9 unseen test traces. Gray = controls.")


# ------------------------------------------------------- 5. producer screen

def fig_screen():
    p = RES / "screen" / "web_g0.json"
    if not p.exists():
        return
    t = json.loads(p.read_text())["totals"]
    base = t["base_misses"]
    groups = [("instruction-type history", "inst_hist"),
              ("producers, lag 1024 (safe)", "producers_L1024"),
              ("producers, lag 256 (assumed)", "producers_L256"),
              ("producers, lag 64 (assumed)", "producers_L64")]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    h = 0.36
    for i, (label, key) in enumerate(groups):
        real = 100 * (base - t[key]) / base
        shuf = 100 * (base - t[key + "_shuffled"]) / base
        ax.barh(i - h / 2 - 0.02, real, height=h, color=BLUE,
                label="real feature" if i == 0 else None)
        ax.barh(i + h / 2 + 0.02, shuf, height=h, color=GRAY,
                label="same feature, rows shuffled" if i == 0 else None)
        ax.text(real + 0.1, i - h / 2 - 0.02, pct(real), va="center", fontsize=8.5)
        ax.text(max(shuf, 0) + 0.1, i + h / 2 + 0.02, pct(shuf), va="center",
                fontsize=8.5)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([g[0] for g in groups])
    ax.invert_yaxis()
    ax.axvline(0, color=GRAY, lw=1)
    ax.set_xlabel("validation mispredictions removed vs TAGE-state baseline (%)")
    ax.set_title("Screening: dataflow producers carry real information")
    ax.legend(loc="upper right")
    save(fig, "fig5_producer_screen.png",
         "Tree model without branch history (a proxy). In BranchNet itself the "
         "lag-1024 gain vanishes: history already holds it.")


# ---------------------------------------------------------------- 6. k-fold

def kfold_table():
    out = {}
    for g in ("web_g0", "infra_g0"):
        rows = []
        for k in range(5):
            name = g if k == 0 else f"{g}_f{k}"
            rep, sh = load(f"{name}.json"), load(f"{name}_shared.json")
            # web_g0 fold 0's main file was rewritten by the data-volume run's
            # eval (no learned arbiter); its full evaluation is _learned.json.
            if k == 0 and load(f"{name}_learned.json"):
                rep = load(f"{name}_learned.json")
            if not rep or "tage_raw" not in rep["variants"]:
                continue
            v = rep["variants"]["tage_raw"]
            key = "learned_lr_fb" if "learned_lr_fb" in v else "learned_lr"
            rows.append({
                "fold": k, "tage_mpki": rep["tage_mpki"],
                "branch": v["branch"]["mpki_reduction"],
                "threshold": v["confidence"]["mpki_reduction"],
                "learned": v.get(key, {}).get("mpki_reduction", np.nan),
                "shared_corrected": (sh["variants"]["tage_raw"]["shared_lr_corrected"]
                                     ["mpki_reduction"] if sh and "tage_raw" in sh["variants"]
                                     else np.nan),
                "tokens_branch": rep["variants"]["tokens"]["branch"]["mpki_reduction"]
                if "tokens" in rep["variants"] else np.nan,
                "labelshuffle": rep["variants"].get("tokens_labelshuffle", {})
                .get("confidence", {}).get("mpki_reduction", np.nan),
            })
        out[g] = rows
    return out


def fig_kfold(table):
    pols = [("branch", "BranchNet rule"), ("threshold", "threshold arbiter"),
            ("learned", "learned arbiter"), ("shared_corrected", "shared + correction")]
    groups = [(g, rows) for g, rows in table.items() if len(rows) >= 2]
    if not groups:
        return
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    h = 0.36
    colors = [BLUE, ORANGE]
    for i, (g, rows) in enumerate(groups):
        ys = np.arange(len(pols)) + (i - 0.5) * (h + 0.04)
        for y, (key, _) in zip(ys, pols):
            vals = 100 * np.array([r[key] for r in rows], float)
            vals = vals[~np.isnan(vals)]
            if not vals.size:
                continue
            m, s = vals.mean(), vals.std(ddof=1) if vals.size > 1 else 0.0
            ax.barh(y, m, height=h, color=colors[i],
                    label=f"{g} ({len(rows)} folds)" if key == "branch" else None)
            ax.errorbar(m, y, xerr=s, color=INK, capsize=3, lw=1)
            ax.text(m + s + 0.08, y, f"{m:+.2f} ± {s:.2f}%", va="center", fontsize=8.5)
    ax.set_yticks(range(len(pols)))
    ax.set_yticklabels([p[1] for p in pols])
    ax.invert_yaxis()
    ax.axvline(0, color=GRAY, lw=1)
    ax.set_xlabel("MPKI reduction vs TAGE-SC-L (%), mean ± std over folds")
    ax.set_title("k-fold over traces: mean ± std over folds")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    save(fig, "fig6_kfold.png", "BranchNet-big + TAGE state + raw features. "
         "Each fold: disjoint unseen test traces, branches re-selected on its train traces.")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_volume()
    fig_policies()
    fig_gap()
    fig_features()
    fig_screen()
    table = kfold_table()
    (RES / "kfold_table.json").write_text(json.dumps(table, indent=2))
    fig_kfold(table)


if __name__ == "__main__":
    main()
