# diffusion-book
Diffusion MRI executable book, built with Jupyter Book 2 (mystmd). Figures come from small
in-notebook simulations and from pre-simulated TRXScan phantom data; every model fit is scored
against the phantom's analytic ground truth.

## Quickstart

All commands run inside the `dwibook` environment (WSL on this machine):

```bash
micromamba create -n dwibook -f environment.yml
micromamba run -n dwibook pip install -e ".[test]"
micromamba run -n dwibook pytest                       # helper-package unit tests
micromamba run -n dwibook myst start --execute         # live preview at http://localhost:3000
micromamba run -n dwibook myst build --html --execute  # static site in _build/html
```

Set `DWIBOOK_DATA=/path/to/data` to build against local pipeline output instead of the data
release. The execution cache is keyed on notebook text only, so after editing anything in
`dwibook/` delete `_build/execute` (or the cached outputs will silently reflect the old code). The offline data pipeline lives in [pipelines/](pipelines/README.md).

## Layout

| Path | What |
|---|---|
| `myst.yml` | book configuration and table of contents |
| `book/` | chapters as MyST Markdown notebooks, `references.bib` |
| `dwibook/` | helper package: data loader, toy simulators, k-space helpers, schemes, truth metrics, plotting |
| `pipelines/` | Snakemake pipeline that runs TRXScan and the QSIPrep Docker tools offline |
| `docs/` | planning documents |

## Planning documents

- [docs/outline.md](docs/outline.md) - chapter-by-chapter outline and the dataset dependency table
- [docs/implementation-plan.md](docs/implementation-plan.md) - toolchain, repository layout, TRXScan work items, data pipeline, phases, open decisions
