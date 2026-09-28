#!/usr/bin/env python
"""DeepCompARE predictions: 8 tracks, ref->mut difference per variant.

model.h5 is a torch.save() pickle (NOT HDF5) of multitasking_models.AstigCRConv5D;
it unpickles with weights_only=False and needs multitasking_models.py importable,
hence the models/ dir on sys.path. Forward returns 16 heads (8 regression + 8
classification); we keep the first 8 = the benchmark tracks.

Output: results/predictions/deepcompare.csv  (row_id, dc_t0..dc_t7 = mut - ref)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mpra_bench import REPO_ROOT, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=str(REPO_ROOT / "data" / "processed" / "variants.csv"))
    ap.add_argument("--models-dir", default=str(REPO_ROOT / "models"))
    ap.add_argument("--outdir", default=str(REPO_ROOT / "results" / "predictions"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=2048)
    args = ap.parse_args()

    import pandas as pd
    import torch
    sys.path.insert(0, args.models_dir)  # for unpickling multitasking_models classes
    from deepISA.model.predict import compute_predictions

    cfg = load_config()
    df = pd.read_csv(args.variants)
    model = torch.load(Path(args.models_dir) / "model.h5", map_location=args.device, weights_only=False)
    model.eval()

    preds = {}
    for tag in ["seq_ref", "seq_mut"]:
        out = compute_predictions(model, df[tag].tolist(), device=args.device,
                                  batch_size=args.batch_size, tracks=list(range(8)))
        preds[tag] = out
    diff = preds["seq_mut"] - preds["seq_ref"]

    out_df = pd.DataFrame(diff, columns=[f"dc_t{i}" for i in range(8)])
    out_df.insert(0, "row_id", df["row_id"])
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(outdir / "deepcompare.csv", index=False, float_format="%.6f")

    # Immediate feedback: per-element Pearson r on the cell line's 4 tracks
    for el, g in df.groupby("Element"):
        cell = cfg["cell_line_map"][el]
        tracks = cfg["deepcompare"]["tracks_by_cell_line"][cell]
        sub = diff[g.index]
        for t in tracks:
            r = pd.Series(sub[:, t]).corr(g["Value"].reset_index(drop=True))
            print(f"{el:10s} {cell:6s} dc_t{t} ({cfg['deepcompare']['track_names'][str(t)]}): r = {r:+.3f}  (n={len(g)})")


if __name__ == "__main__":
    main()
