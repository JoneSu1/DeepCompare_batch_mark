# MPRA mutation-effect benchmark: DeepCompARE vs Enformer vs AlphaGenome

Predicts satMutMPRA mutation effect sizes (kircherlab bihealth, GRCh38 TSV of
F9 / GP1BA / HBB / HBG1 / LDLR / PKLR-24h / BCL11A / SORT1) with three models and
compares per-regulatory-element Pearson correlations between the predicted
ref->mut difference and the measured effect size (`Value`).

```
data/input/*.tsv ─► 01_prepare_sequences ─► data/processed/variants.csv (600 bp ref/mut, element-centered)
                                            data/processed/contexts.fasta (196,608 bp Enformer contexts)
                       │
       ┌───────────────┼─────────────────────┐
  02 DeepCompARE   03 Enformer          04 AlphaGenome (API)
  (models/model.h5) (HF weights, GPU rec.)  (deepISA adapter, needs key)
       └───────────────┼─────────────────────┘
                       ▼
              05_correlations ─► results/correlations.csv
```

## Quickstart (local, minimal viable test)

```bash
python -m venv .venv && .venv/Scripts/pip install "torch>=2.0,<2.5" --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/pip install -r requirements.txt   # torch deliberately not listed there
.venv/Scripts/pip install --no-deps "git+https://github.com/anderssonlab/deepISA"

# reference genome (only chromosomes actually used; UCSC hg38)
mkdir -p data/reference
for c in 1 2 11 19 22 X; do
  curl -sL "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr$c.fa.gz" \
    | gunzip > data/reference/chr$c.fa
done

python scripts/01_prepare_sequences.py --fasta data/reference/chr11.fa --elements HBB,HBG1
python scripts/02_predict_deepcompare.py
python scripts/05_correlations.py
```

## Full run on Colab (GPU for Enformer)

Open `notebooks/run_colab.ipynb` in Colab, paste your repo URL, and run all cells.
The repo ships `data/processed/` artifacts, so the genome download above is only
needed when re-running step 01 from scratch.

AlphaGenome needs a free non-commercial API key
(https://deepmind.google.com/science/alphagenome):

```bash
cp configs/alphagenome.yaml.example configs/alphagenome.yaml   # then set api_key
python scripts/04_predict_alphagenome.py
```

## Method decisions (do not improvise; source: task spec + reference benchmark.py)

- **Element-centered windows.** 600 bp = regulatory-element midpoint ± 300 bp
  (hardcoded coordinates in `configs/benchmark.json`), never variant-centered.
- **DeepCompARE tracks** 0..7 = CAGE/DNase/STARR/SuRE × HepG2/K562; per element use
  its cell line's four tracks (`cell_line_map`; PKLR-24h is the real element name —
  the reference script's map key "PKLR" was a latent bug, fixed here).
- **Enformer**: input = element midpoint ± 98,304 bp; read the center output bin
  (index 448, 128 bp) — the same element-centered anchoring as the other models.
  Per cell line: mean over all CAGE tracks and all DNase tracks
  (`enformer.track_indices` in the config; set `track_selection` to `untreated` to
  drop K562's treated/sorted DNase replicates).
- **AlphaGenome**: 600 bp window centered in 16,384 bp N-padded context, per-track
  aggregation over the central 600 bp + log1p (deepISA adapter default `sum`);
  CAGE mean, DNase mean, and cell-type-specific ATAC per element (both K562 and
  HepG2 have all three).
- **Metric**: Pearson r per (element × method-track) vs `Value`; N reported per row.
- **Sequences are uppercased** before any encoding: UCSC hg38 is soft-masked and
  deepISA's one-hot is case-sensitive (lowercase encodes as all-zeros).
- `model.h5` is a `torch.save()` pickle of `AstigCRConv5D` (16 heads: 8 regression
  + 8 classification; we keep the first 8). It needs `models/multitasking_models.py`
  importable — `02` handles that.

## Environment notes

- **Windows + Anaconda**: torch must stay `<2.5`; newer wheels abort with
  `WinError 1114` on `c10.dll` when Anaconda's old `msvcp140.dll` (14.27) shadows
  the system one. A clean venv (as above) with `torch>=2.0,<2.5` is verified working.
- Enformer CPU costs ~30 s/sequence → ~34 h for the full 3,877 variants; on a T4
  (Colab) the same run is minutes. DeepCompARE CPU is seconds.
- AlphaGenome costs ≈ n_variants + n_elements API calls (≈3,885 for the full set):
  per-element ref sequences repeat, and the adapter's SQLite cache deduplicates
  and makes runs resumable.

## Status

| step | state |
|---|---|
| 01 sequence preparation | verified (full run: 3,877 variants, 8 elements, on Colab) |
| 02 DeepCompARE | verified (full run; local-MVP cross-check matched to 4e-6) |
| 03 Enformer | verified (full run on Colab T4) |
| 04 AlphaGenome | verified (full run, ~3,885 API calls) |
| 05 correlations | verified (full `correlations.csv`, 144 rows) |
| 06 figure | done (`results/figure_correlations.png/.pdf`; medians DC 0.446 / Enf 0.472 / AG 0.613) |
