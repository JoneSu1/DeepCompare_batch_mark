#!/usr/bin/env python
"""Merge per-method predictions -> per (element x method-track) Pearson r -> correlations.csv.

Track handling per the task spec:
  DeepCompARE  4 points/element: the cell line's CAGE/DNase/STARR/SuRE tracks
  Enformer     2 points/element: mean over all CAGE tracks and all DNase tracks of the cell line
  AlphaGenome  3 points/element: CAGE mean, DNase mean, cell-type-specific ATAC

Group definitions come from configs/benchmark.json (not from parsing column names).
Schema: Element, Cell, Method, Track, Correlation, N
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mpra_bench import REPO_ROOT, load_config  # noqa: E402

DC_ASSAY_BY_TRACK = {0: "CAGE", 1: "CAGE", 2: "DNase", 3: "DNase",
                     4: "STARR", 5: "STARR", 6: "SuRE", 7: "SuRE"}
DC_CELL_BY_PARITY = {0: "HepG2", 1: "K562"}


def correlate(df, pred, method, track_of_col, rows):
    merged = df[["row_id", "Element", "Cell", "Value"]].merge(pred, on="row_id")
    for (el, cell), g in merged.groupby(["Element", "Cell"]):
        for col, (assay, track_cell) in track_of_col.items():
            r = g[col].corr(g["Value"])
            rows.append({"Element": el, "Cell": cell, "Method": method,
                         "Track": f"{assay} {track_cell}", "Correlation": r, "N": len(g)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=str(REPO_ROOT / "data" / "processed" / "variants.csv"))
    ap.add_argument("--predictions-dir", default=str(REPO_ROOT / "results" / "predictions"))
    ap.add_argument("--outfile", default=str(REPO_ROOT / "results" / "correlations.csv"))
    args = ap.parse_args()

    cfg = load_config()
    df = pd.read_csv(args.variants)
    rows = []
    pdir = Path(args.predictions_dir)

    # ---- DeepCompARE: one row per raw track t0..t7 ----
    dc_path = pdir / "deepcompare.csv"
    if dc_path.exists():
        dc = pd.read_csv(dc_path)
        track_of_col = {f"dc_t{t}": (DC_ASSAY_BY_TRACK[t], DC_CELL_BY_PARITY[t % 2]) for t in range(8)}
        correlate(df, dc, "DeepCompARE", track_of_col, rows)

    # ---- Enformer: average raw tracks per (assay, cell) from config ----
    enf_path = pdir / "enformer.csv"
    if enf_path.exists():
        enf = pd.read_csv(enf_path)
        sel = cfg["enformer"]["track_selection"]
        track_of_col = {}
        for group, idxs in cfg["enformer"]["track_indices"].items():
            assay, cell = group.split("_", 1)
            cols = [f"enf_{group}_{i}" for i in idxs[sel]]
            missing = [c for c in cols if c not in enf.columns]
            if missing:
                raise ValueError(f"enformer.csv is missing expected columns: {missing}")
            mean_col = f"enfmean_{group}"
            enf[mean_col] = enf[cols].mean(axis=1)
            track_of_col[mean_col] = (assay, cell)
        correlate(df, enf[["row_id"] + list(track_of_col)], "Enformer", track_of_col, rows)

    # ---- AlphaGenome: average raw tracks per (assay, cell) from config ----
    ag_path = pdir / "alphagenome.csv"
    if ag_path.exists():
        ag = pd.read_csv(ag_path)
        track_of_col = {}
        for cell, groups in cfg["alphagenome"]["tracks_by_cell_line"].items():
            for gname in groups:
                assay, gcell = gname.split(":", 1)
                cols = [c for c in ag.columns if c.startswith(f"ag_{assay}_{gcell}_")]
                if not cols:
                    raise ValueError(f"alphagenome.csv has no columns for {gname}")
                mean_col = f"agmean_{assay}_{gcell}"
                ag[mean_col] = ag[cols].mean(axis=1)
                track_of_col[mean_col] = (assay, gcell)
        correlate(df, ag[["row_id"] + list(track_of_col)], "AlphaGenome", track_of_col, rows)

    if not rows:
        raise SystemExit("no prediction files found in " + str(pdir))
    out = pd.DataFrame(rows)
    Path(args.outfile).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.outfile, index=False, float_format="%.6f")
    print(out.to_string(index=False))
    print(f"\nwrote {args.outfile}")


if __name__ == "__main__":
    main()
