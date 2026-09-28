#!/usr/bin/env python
"""Build element-centered 600 bp ref/mut sequences + Enformer 196,608 bp contexts.

Adapted from the first 70 lines of the reference benchmark.py, with three fixes:
  * dynamic row count (no hardcoded reshape),
  * cell_line_map keyed by the real element name 'PKLR-24h',
  * sequences uppercased before leaving this step (UCSC hg38 is soft-masked and
    deepISA's one_hot_encode is case-sensitive; lowercase would encode as zeros).

Outputs (in --outdir, default data/processed):
  variants.csv    row_id, Element, Cell, Position (1-based), position_rel, seq_ref, seq_mut ...
  contexts.fasta  one 196,608 bp uppercase record per element (Enformer input reference)
  manifest.json   filter + per-element counts for provenance
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mpra_bench import REPO_ROOT, load_config, load_variants, require_in_window


def load_fastas(paths):
    from pyfaidx import Fasta
    return [Fasta(str(p), as_raw=True, sequence_always_upper=True) for p in paths]


def fetch(fastas, chrom, start, end):
    for fa in fastas:
        if chrom in fa:
            return str(fa[chrom][start:end]).upper()
    raise KeyError(f"chromosome {chrom} not found in any provided FASTA")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", nargs="+", required=True, help="hg38 FASTA file(s), e.g. data/reference/chr11.fa")
    ap.add_argument("--tsv", default=str(REPO_ROOT / "data" / "input" / "GRCh38_F9_GP1BA_HBB_HBG1_LDLR_PKLR-24h_BCL11A_SORT1.tsv"))
    ap.add_argument("--elements", default=None, help="comma-separated subset, e.g. HBB,HBG1 (default: all)")
    ap.add_argument("--outdir", default=str(REPO_ROOT / "data" / "processed"))
    args = ap.parse_args()

    cfg = load_config()
    elements = args.elements.split(",") if args.elements else None
    df = load_variants(cfg, args.tsv, elements=elements)
    require_in_window(df, cfg["window_bp"])
    fastas = load_fastas(args.fasta)

    seqs_ref = [fetch(fastas, r.Chromosome, r.window_start, r.window_start + cfg["window_bp"])
                for r in df.itertuples()]
    df["seq_ref"] = seqs_ref

    ref_check = [s[r.position_rel] for s, r in zip(df["seq_ref"], df.itertuples())]
    bad = df[[c != ref.upper() for c, ref in zip(ref_check, df["Ref"])]]
    if len(bad):
        raise ValueError(f"{len(bad)} rows where TSV Ref disagrees with the reference genome, e.g.:\n"
                         f"{bad[['Element', 'Position', 'Ref']].head()}")
    df["seq_mut"] = [s[:r.position_rel] + a.upper() + s[r.position_rel + 1:]
                     for s, r, a in zip(df["seq_ref"], df.itertuples(), df["Alt"])]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "variants.csv", index=False)

    # Enformer contexts: one per element, midpoint-centered
    ctx = cfg["enformer"]["context_bp"]
    half = ctx // 2
    with open(outdir / "contexts.fasta", "w") as f:
        for el in sorted(df["Element"].unique()):
            sub = cfg["elements"][el]
            mid = (sub["start"] + sub["end"]) // 2
            seq = fetch(fastas, sub["chrom"], mid - half, mid - half + ctx)
            f.write(f">{el}|{cfg['cell_line_map'][el]}|{sub['chrom']}:{mid - half}-{mid - half + ctx}\n")
            for i in range(0, len(seq), 60):
                f.write(seq[i:i + 60] + "\n")

    manifest = {
        "tsv": Path(args.tsv).name,
        "filters": cfg["filters"],
        "n_variants": len(df),
        "per_element": df["Element"].value_counts().to_dict(),
        "elements_written": sorted(df["Element"].unique()),
    }
    with open(outdir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
