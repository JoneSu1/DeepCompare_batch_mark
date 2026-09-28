"""Shared helpers for the MPRA benchmark pipeline (DeepCompARE / Enformer / AlphaGenome)."""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "benchmark.json"


def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def element_midpoint(cfg, element):
    e = cfg["elements"][element]
    return (e["start"] + e["end"]) // 2


def load_variants(cfg, tsv_path, elements=None):
    """Load + filter the satMutMPRA TSV. Mirrors benchmark.py: Alt != '-' and P-Value < 0.05.

    Returns a DataFrame with stable 0-based row_id (TSV order preserved) and the
    element-centered 600 bp window coordinates attached.
    """
    import pandas as pd

    df = pd.read_csv(tsv_path, sep="\t")
    df = df[df["Alt"] != "-"]
    df = df[df["P-Value"] < cfg["filters"]["p_value_max"]]
    df = df[df["Ref"].str.upper() != df["Alt"].str.upper()].copy()
    if elements:
        df = df[df["Element"].isin(elements)]
    df = df.reset_index(drop=True)
    df.insert(0, "row_id", df.index)

    df["Chromosome"] = "chr" + df["Chromosome"].astype(str)
    df["mid"] = df["Element"].map({el: element_midpoint(cfg, el) for el in df["Element"].unique()})
    half = cfg["window_bp"] // 2
    df["window_start"] = df["mid"] - half
    df["position_rel"] = (df["Position"] - 1) - df["window_start"]
    df["Cell"] = df["Element"].map(cfg["cell_line_map"])
    return df


def require_in_window(df, window_bp):
    """Sanity gate: every variant must fall inside its element-centered window."""
    bad = df[(df["position_rel"] < 0) | (df["position_rel"] >= window_bp)]
    if len(bad):
        raise ValueError(f"{len(bad)} variants fall outside the {window_bp} bp window, e.g.:\n{bad.head()}")
