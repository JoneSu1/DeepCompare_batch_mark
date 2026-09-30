#!/usr/bin/env python
"""Publication figure: per-element Pearson r of predicted ref->mut differences.

Per the task spec, for each regulatory element only its cell line's tracks:
  DeepCompARE  4 points (CAGE / DNase / STARR / SuRE of the cell line)
  Enformer     2 points (CAGE mean, DNASE mean)
  AlphaGenome  3 points (CAGE mean, DNASE mean, ATAC)
Black tick under each (element, method) cluster = that group's median.

Output: results/figure_correlations.png (300 dpi) + .pdf
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mpra_bench import REPO_ROOT, load_config  # noqa: E402

METHOD_ASSAYS = {  # compared against upper-cased assay names from the Track column
    "DeepCompARE": {"CAGE", "DNASE", "STARR", "SURE"},
    "Enformer":    {"CAGE", "DNASE"},
    "AlphaGenome": {"CAGE", "DNASE", "ATAC"},
}
STYLE = {  # one colour per method; all of a method's points share it
    "DeepCompARE": ("#1f77b4", "o"),
    "Enformer":    ("#ff7f0e", "s"),
    "AlphaGenome": ("#2ca02c", "^"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--correlations", default=str(REPO_ROOT / "results" / "correlations.csv"))
    ap.add_argument("--outdir", default=str(REPO_ROOT / "results"))
    args = ap.parse_args()

    cfg = load_config()
    corr = pd.read_csv(args.correlations)
    corr["assay"] = corr["Track"].str.split().str[0].str.upper()
    corr["track_cell"] = corr["Track"].str.split().str[-1]
    # keep only rows whose track cell line matches the element's cell line
    keep = corr.apply(lambda r: r["track_cell"] == cfg["cell_line_map"][r["Element"]], axis=1)
    corr = corr[keep & corr["assay"].isin({a for ms in METHOD_ASSAYS.values() for a in ms})]
    corr = corr[corr.apply(lambda r: r["assay"] in METHOD_ASSAYS[r["Method"]], axis=1)]

    elements = [el for el in ["BCL11A", "F9", "GP1BA", "HBB", "HBG1", "LDLR", "PKLR-24h", "SORT1"]
                if el in set(corr["Element"])]
    offsets = {"DeepCompARE": -0.26, "Enformer": 0.0, "AlphaGenome": 0.26}

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for method in ["DeepCompARE", "Enformer", "AlphaGenome"]:
        color, marker = STYLE[method]
        for i, el in enumerate(elements):
            pts = corr[(corr["Method"] == method) & (corr["Element"] == el)]["Correlation"]
            if not len(pts):
                continue
            x = i + offsets[method]
            ax.scatter([x] * len(pts), pts, s=34, c=color, marker=marker,
                       edgecolors="white", linewidths=0.5, zorder=3, alpha=0.85,
                       label=method if i == 0 else None)
            med = pts.median()
            ax.hlines(med, x - 0.09, x + 0.09, colors="black", linewidths=2.0, zorder=4)

    ax.set_xticks(range(len(elements)))
    ax.set_xticklabels([f"{el}\n({cfg['cell_line_map'][el]})" for el in elements], fontsize=9)
    ax.set_ylabel("Pearson r  (predicted vs measured mutation effect)")
    ax.set_ylim(-1, 1)
    ax.axhline(0, color="grey", linewidth=0.6, linestyle=":")
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=1.4)

    outdir = Path(args.outdir)
    fig.savefig(outdir / "figure_correlations.png", dpi=300)
    fig.savefig(outdir / "figure_correlations.pdf")
    print(f"wrote {outdir / 'figure_correlations.png'} and .pdf")

    med = corr.groupby("Method")["Correlation"].median().reindex(list(METHOD_ASSAYS))
    print("\nGlobal per-method medians across all element-track points:")
    print(med.round(3).to_string())
    n_pts = corr.groupby("Method").size()
    print("\nPoints per method:", n_pts.to_dict(), "| total:", len(corr))


if __name__ == "__main__":
    main()
