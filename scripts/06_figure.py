#!/usr/bin/env python
"""Publication figure matching the reference layout in the task docx.

Reference style (Benchmark任务说明 docx, embedded image):
  2x4 facet grid, one panel per regulatory element (gray banner titles),
  per-facet "Model" x axis (DeepCompARE | Enformer | + AlphaGenome),
  per-facet independent y ("Pearson Correlation") scales, no gridlines.
  Shape encodes prediction method, colour encodes model; a short black
  horizontal line marks each model's median within a facet. Right-side
  legend in two blocks: Prediction method (shapes) + Model (colours).

Task-spec point counts per element (cell-line-matched tracks only):
  DeepCompARE 4 (CAGE/DNase/STARR/SuRE), Enformer 2 (CAGE/DNase means,
  cell-type-agnostic tracks dropped), AlphaGenome 3 (CAGE/DNase/ATAC).

Output: results/figure_correlations.png (300 dpi) + .pdf
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mpra_bench import REPO_ROOT, load_config  # noqa: E402

MODEL_COLOR = {"DeepCompARE": "#F8766D", "Enformer": "#00BFC4", "AlphaGenome": "#B983FF"}
MODEL_ORDER = ["DeepCompARE", "Enformer", "AlphaGenome"]
# shape = prediction method (reference legend; ATAC added for AlphaGenome)
ASSAY_SHAPE = {"CAGE": "o", "DNASE": "D", "STARR": "X", "SURE": "*", "ATAC": "v"}
ASSAY_LABEL = {"CAGE": "CAGE", "DNASE": "DNase",
               "STARR": "STARR", "SURE": "SuRE",
               "ATAC": "ATAC"}
ASSAY_PER_MODEL = {"DeepCompARE": ["CAGE", "DNASE", "STARR", "SURE"],
                   "Enformer": ["CAGE", "DNASE"],
                   "AlphaGenome": ["CAGE", "DNASE", "ATAC"]}
ELEMENT_ORDER = ["BCL11A", "F9", "GP1BA", "HBB", "HBG1", "LDLR", "PKLR-24h", "SORT1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--correlations", default=str(REPO_ROOT / "results" / "correlations.csv"))
    ap.add_argument("--outdir", default=str(REPO_ROOT / "results"))
    args = ap.parse_args()

    cfg = load_config()
    corr = pd.read_csv(args.correlations)
    corr["assay"] = corr["Track"].str.split().str[0].str.upper()
    corr["track_cell"] = corr["Track"].str.split().str[-1]
    corr = corr[corr.apply(lambda r: r["track_cell"] == cfg["cell_line_map"][r["Element"]]
                           and r["assay"] in ASSAY_SHAPE
                           and r["assay"] in ASSAY_PER_MODEL[r["Method"]], axis=1)]

    elements = [el for el in ELEMENT_ORDER if el in set(corr["Element"])]
    fig, axes = plt.subplots(2, 4, figsize=(11.5, 6.0))

    for k, el in enumerate(elements):
        ax = axes[k // 4][k % 4]
        sub = corr[corr["Element"] == el]
        for m_i, model in enumerate(MODEL_ORDER):
            pts = sub[sub["Method"] == model]
            med_vals = []
            for assay in ASSAY_PER_MODEL[model]:
                v = pts[pts["assay"] == assay]["Correlation"]
                if not len(v):
                    continue
                n = len(v)
                jit = np.linspace(-0.05, 0.05, n) if n > 1 else np.array([0.0])
                y = v.to_numpy() if n > 1 else np.array([float(v.iloc[0])])
                ax.scatter(m_i + jit, y, s=52 if assay == "SURE" else 30,
                           c=MODEL_COLOR[model], marker=ASSAY_SHAPE[assay],
                           edgecolors="white", linewidths=0.4, zorder=3)
                med_vals.extend(y)
            if med_vals:
                med = float(np.median(med_vals))
                ax.hlines(med, m_i - 0.13, m_i + 0.13, colors="black", linewidths=1.8, zorder=4)

        ax.set_title(f"{el} ({cfg['cell_line_map'][el]})", fontsize=9.5,
                     pad=3, backgroundcolor="#E8E8E8")
        ax.set_xlim(-0.55, 2.55)
        ax.set_xticks(range(3))
        if k // 4 == 1:
            ax.set_xticklabels(MODEL_ORDER, rotation=45, ha="right", fontsize=8.5)
        else:
            ax.set_xticklabels([])
        ax.tick_params(labelsize=8)
        if k % 4 == 0:
            ax.set_ylabel("Pearson Correlation", fontsize=9)
        ax.set_facecolor("#F7F7F7")
        for s in ax.spines.values():
            s.set_visible(False)

    # right-side two-block legend
    handles = [plt.Line2D([], [], color="black", marker=ASSAY_SHAPE[a], linestyle="",
                          markersize=7 if a != "SURE" else 10, label=ASSAY_LABEL[a])
               for a in ["CAGE", "DNASE", "STARR", "SURE", "ATAC"]]
    handles += [plt.Line2D([], [], color=MODEL_COLOR[m], marker="o", linestyle="",
                           markersize=7, label=m) for m in MODEL_ORDER]
    fig.legend(handles=handles, loc="center right", fontsize=8.5, frameon=False,
               title="Prediction method / Model", labelspacing=0.9,
               bbox_to_anchor=(0.995, 0.5))
    fig.subplots_adjust(left=0.06, right=0.8, top=0.93, bottom=0.14, wspace=0.28, hspace=0.32)
    fig.suptitle("MPRA variant effect size prediction", fontsize=11.5, y=0.985)

    outdir = Path(args.outdir)
    fig.savefig(outdir / "figure_correlations.png", dpi=300)
    fig.savefig(outdir / "figure_correlations.pdf")
    print(f"wrote {outdir / 'figure_correlations.png'} and .pdf")
    print("points per method:", corr.groupby("Method").size().to_dict(), "| total:", len(corr))
    print("global medians:", corr.groupby("Method")["Correlation"].median()
          .reindex(MODEL_ORDER).round(3).to_dict())


if __name__ == "__main__":
    main()
