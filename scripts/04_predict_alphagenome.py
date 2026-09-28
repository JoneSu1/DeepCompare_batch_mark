#!/usr/bin/env python
"""AlphaGenome predictions via deepISA's AlphaGenomeAdapter (API backend).

The adapter is a drop-in nn.Module: (N, 4, 600) one-hot -> (N, n_tracks).
Anchoring matches the other methods: the element's 600 bp window is centered
in 16,384 bp of N padding, per-track values are aggregated over the central
600 bp window (config: aggregation, default sum) and log1p-transformed.

Unique ref sequences (one per element) are predicted once, then subtracted
per variant. The adapter's SQLite cache makes interrupted runs resumable.

Output: results/predictions/alphagenome.csv (row_id, ag_{OT}_{CELL}_{k}, ...)
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mpra_bench import REPO_ROOT, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=str(REPO_ROOT / "data" / "processed" / "variants.csv"))
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "alphagenome.yaml"))
    ap.add_argument("--outdir", default=str(REPO_ROOT / "results" / "predictions"))
    ap.add_argument("--batch-size", type=int, default=8, help="sequences per predict() call group")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: only first N variants")
    args = ap.parse_args()

    import numpy as np
    import pandas as pd
    from deepISA.model.alpha_genome_adapter import AlphaGenomeAdapter
    from deepISA.model.predict import compute_predictions

    cfg = load_config()
    df = pd.read_csv(args.variants)
    if args.limit:
        df = df.head(args.limit).copy()

    adapter = AlphaGenomeAdapter(args.config)

    # Column names from the adapter's extraction plan, aligned with its track config
    names = []
    for track, (attr, col_idx) in zip(adapter._cfg["tracks"], adapter._extraction_plan):
        ot = track["output_type"]
        names.extend(f"ag_{ot}_{track['biosample_name']}_{k}" for k in range(len(col_idx)))
    assert len(names) == adapter.n_tracks

    def predict(seqs):
        return compute_predictions(adapter, seqs, device="cpu",
                                   batch_size=args.batch_size, tracks=list(range(adapter.n_tracks)))

    # 1) unique per-element reference sequences (cache makes repeats free)
    ref_seqs = df.groupby("Element")["seq_ref"].first().tolist()
    ref_els = df.groupby("Element")["seq_ref"].first().index.tolist()
    t0 = time.time()
    ref_pred = predict(ref_seqs)
    print(f"refs done: {len(ref_seqs)} seqs in {time.time() - t0:.0f}s")

    # 2) mutated sequences
    t0 = time.time()
    mut_pred = predict(df["seq_mut"].tolist())
    print(f"muts done: {len(df)} seqs in {time.time() - t0:.0f}s | stats: {adapter.stats}")

    diff = mut_pred - ref_pred[[ref_els.index(el) for el in df["Element"]]]
    out = pd.DataFrame(diff, columns=names)
    out.insert(0, "row_id", df["row_id"].values)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(outdir / "alphagenome.csv", index=False, float_format="%.6f")
    print(f"wrote {outdir / 'alphagenome.csv'}")


if __name__ == "__main__":
    main()
