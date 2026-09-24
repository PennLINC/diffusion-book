# Offline data pipeline

Everything here runs in WSL, in the `dwibook` environment, and produces the datasets the
notebooks load. The book itself never runs any of this: notebooks fetch finished outputs from a
data release (or from `$DWIBOOK_DATA` while developing).

```bash
wsl -e bash -lc "cd /mnt/c/Users/tsalo/Documents/linc/diffusion-book/pipelines && micromamba run -n dwibook snakemake -c 8 ref-clean"
```

| Piece | Purpose | Status |
|---|---|---|
| `config/datasets.yaml` | single source of truth: tools, phantoms, schemes, every dataset's flags | written |
| `Snakefile` | `prepare_grid` → `truth` → `simulate` rules | grid + truth wired; `simulate` stubbed |
| `scripts/run_dataset.py` | expands one dataset entry into `trxscan` runs (sweeps, variants, PE pairs), writes provenance JSON | to do |
| `scripts/make_schemes.py` | writes `config/schemes/*.bval/.bvec` from `dwibook.schemes` | to do |
| `scripts/normalize_grid_names.py` | renames `prepare_acquisition_grid.py` outputs to the fixed names the rules expect | to do |
| `scripts/precompute_*.py` | topup / eddy / MRtrix / qsiprep inside `docker run pennlinc/qsiprep:<tag>` | to do |
| `scripts/package_release.py` | tar per dataset, sha256 registry → `dwibook/registry.txt`, upload as release assets | to do |

Tools the rules expect on the WSL PATH: `trxscan`, `trxscan-microstructure` (from the TRXScan
`book` branch once T1–T3 land), `docker`. See `docs/implementation-plan.md` §4–5.
