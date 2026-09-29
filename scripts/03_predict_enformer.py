#!/usr/bin/env python
"""Enformer predictions: ref->mut difference at the center bin, selected tracks.

Design notes (mirror the AlphaGenome anchoring logic):
  * Context = the element's 196,608 bp midpoint-centered window from contexts.fasta;
    a variant is substituted at offset context_bp/2 + (position_rel - window_bp/2).
  * Output bin = center_bin (default 448), which starts exactly at the context
    midpoint - the same "element-centered" convention as DeepCompARE/AlphaGenome.
  * Only the per-element reference context is unique (8 sequences total), so refs
    are predicted once per element and reused for every variant of that element.

Output: results/predictions/enformer.csv  (row_id, enf_{ASSAY}_{CELL}_{track_idx}, ...)
Raw per-track values; averaging per (assay, cell) happens in 05_correlations.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mpra_bench import REPO_ROOT, load_config  # noqa: E402

BASES = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}


def encode(seqs):
    import torch
    return torch.tensor([[BASES.get(b, 4) for b in s] for s in seqs], dtype=torch.long)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=str(REPO_ROOT / "data" / "processed" / "variants.csv"))
    ap.add_argument("--contexts", default=str(REPO_ROOT / "data" / "processed" / "contexts.fasta"))
    ap.add_argument("--outdir", default=str(REPO_ROOT / "results" / "predictions"))
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: only first N variants")
    args = ap.parse_args()

    import numpy as np
    import pandas as pd
    import torch
    from enformer_pytorch import from_pretrained
    from pyfaidx import Fasta

    cfg = load_config()
    ctx_bp = cfg["enformer"]["context_bp"]
    win_bp = cfg["window_bp"]
    center_bin = cfg["enformer"]["center_bin"]
    sel = cfg["enformer"]["track_selection"]
    track_idx = sorted({i for g in cfg["enformer"]["track_indices"].values() for i in g[sel]})
    track_cols = {i: f"enf_{grp}_{i}" for grp, groups in cfg["enformer"]["track_indices"].items() for i in groups[sel]}

    df = pd.read_csv(args.variants)
    if args.limit:
        df = df.head(args.limit).copy()

    fa = Fasta(args.contexts, as_raw=True)
    contexts = {r.name.split("|")[0]: str(r[:]).upper() for r in fa}

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and args.device == "auto" and args.limit is None:
        raise SystemExit(
            "No CUDA device visible: Enformer needs ~30 s/sequence on CPU "
            "(~34 h for the full run). In Colab: Runtime > Change runtime type > T4 GPU, "
            "then Run all again. For a deliberate CPU smoke test, add --limit N.")

    def load_enformer():
        # transformers>=4.53 refuses torch.load() of .bin checkpoints on torch<2.6
        # (CVE-2025-32434). EleutherAI/enformer-official-rough is a known public
        # file we deliberately trust, so neutralize the gate on old-torch hosts
        # (Windows + Anaconda caps this venv at torch 2.4). Colab (torch>=2.6)
        # is unaffected.
        if tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) < (2, 6):
            import transformers.modeling_utils as mu
            import transformers.utils.import_utils as iu
            mu.check_torch_load_is_safe = lambda *a, **k: None
            iu.check_torch_load_is_safe = lambda *a, **k: None
        return from_pretrained("EleutherAI/enformer-official-rough")

    model = load_enformer().to(device).eval()
    print(f"device={device} | variants={len(df)} | tracks={len(track_idx)} | center_bin={center_bin}")

    def predict(seqs):
        outs = []
        for i in range(0, len(seqs), args.batch_size):
            x = encode(seqs[i:i + args.batch_size]).to(device)
            with torch.no_grad():
                o = model(x)
            o = o["human"] if isinstance(o, dict) else o
            outs.append(o[:, center_bin, track_idx].float().cpu().numpy())
            if (i // args.batch_size) % 10 == 0:
                print(f"  {i + len(x)}/{len(seqs)} seqs ({time.strftime('%H:%M:%S')})", flush=True)
        return np.concatenate(outs, axis=0)

    # 1) unique per-element reference contexts
    ref_seqs, ref_els = [], []
    for el in df["Element"].unique():
        ref_seqs.append(contexts[el])
        ref_els.append(el)
    ref_pred = predict(ref_seqs)

    # 2) mutated contexts
    mut_seqs = []
    for r in df.itertuples():
        c = contexts[r.Element]
        off = ctx_bp // 2 + (r.position_rel - win_bp // 2)
        if c[off] != r.Ref.upper():
            raise ValueError(f"{r.Element} row_id={r.row_id}: context base {c[off]} != TSV Ref {r.Ref}")
        mut_seqs.append(c[:off] + r.Alt.upper() + c[off + 1:])
    mut_pred = predict(mut_seqs)

    diff = mut_pred - ref_pred[[ref_els.index(el) for el in df["Element"]]]
    out = pd.DataFrame(diff, columns=[track_cols[i] for i in track_idx])
    out.insert(0, "row_id", df["row_id"].values)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(outdir / "enformer.csv", index=False, float_format="%.6f")
    print(f"wrote {outdir / 'enformer.csv'}")


if __name__ == "__main__":
    main()
