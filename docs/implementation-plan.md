# Implementation plan

Status: stage 1, 2026-09-24. Companion: [outline.md](outline.md) (chapter contents and dataset table).

## 1. Goals and non-goals

**Goals**
- An executable book (web first, PDF second) where every figure is regenerated from code.
- One phantom threaded through the whole book, with quantitative fit-vs-truth results from
  TRXScan's analytic ground truth wherever a chapter fits a model or corrects an artifact.
- A build that runs on CI in under ~20 minutes, so contributors can edit prose and light code
  without a workstation-scale simulation.

**Non-goals (for the first edition)**
- Re-implementing preprocessing tools. The book *calls* dipy/FSL/MRtrix where they exist and
  writes small didactic versions only where the step is the point (e.g., a GRAPPA kernel fit,
  a fieldmap unwarp, a Kellner unringing loop).
- Real-scanner data. Everything is simulated; real data are mentioned only as motivation.
- Restricted-diffusion (time-dependent) *phantom* simulation. TRXScan's compartments are
  Gaussian, so Chapter 22 is toy-only until a restricted compartment exists.

## 2. Toolchain

| Concern | Decision | Why |
|---|---|---|
| Book engine | **Jupyter Book 2** (`mystmd` engine; decided 2026-09-24, see §2a). Content is MyST Markdown notebooks (`.md` with ` ```{code-cell} ` fences), config in `myst.yml` | live incremental preview, maintained engine, rich cross-references; the content format is shared with Jupyter Book 1, so the exit cost is a config swap if ever needed |
| Python env | **`dwibook`**, a dedicated micromamba env defined in [`environment.yml`](../environment.yml) (WSL side), created 2026-09-24 | no existing env had both dipy and Jupyter Book; the `trxscan` env is the NIBS pipeline env and is left untouched |
| Core Python deps | python 3.11, numpy, scipy, nibabel, dipy ≥ 1.10, matplotlib, pandas, scikit-image, pooch, jupyter-book 2, jupyterlab, jupytext, ipywidgets (static fallbacks only) | dipy covers DTI/DKI/MAP-MRI/SHORE/CSD/MSMT/QBI/DSI/free-water/IVIM/tracking/denoising/unringing; pooch fetches the data release |
| Optional Python deps | sigpy (CS k-space recon), dmri-amico (NODDI), scilpy or tractometer-style scoring code (tractography evaluation), fury (3-D renders; off on CI) | each is used in one or two chapters; keep them optional extras so the core build never depends on them |
| External tools (offline pipeline only) | TRXScan + `trxscan-microstructure` (built, on the WSL PATH), TRXViz CLI (tractogram/ODF renders to PNG), **FSL and MRtrix via the `pennlinc/qsiprep` Docker image** (Docker 29 is installed in WSL; `pennlinc/qsiprep:test` and `:unstable` are already pulled; the pipeline pins a released tag) | heavy or non-Python steps run in the data pipeline and ship results; the book never shells out at build time. Using the QSIPrep image also lets Ch. 14 run the real assembled pipeline (`qsiprep` itself) on the kitchen-sink dataset. |
| Simulation host | WSL (`wsl -e bash -lc "..."`), per the machine's toolchain layout | TRXScan, cargo, micromamba all live there |
| Hosting | GitHub Pages via Actions; data on a GitHub release (or OSF/Zenodo if > 2 GB) with a pooch registry | zero-infrastructure, versioned data with checksums |

### 2a. Jupyter Book 1 vs. Jupyter Book 2

Versions on PyPI on 2026-09-24: `jupyter-book` 1.0.4 (Sphinx line) and 2.1.7 (a thin CLI over
`mystmd` 1.11). Both read the same MyST Markdown notebooks, so the content never has to change;
what differs is the build tool, its extensions, and its outputs.

| | Jupyter Book 1 (Sphinx + myst-nb) | Jupyter Book 2 (mystmd) |
|---|---|---|
| Maturity | mature; effectively in maintenance | active development; 2.x stable since 2025, minor releases still change config keys occasionally |
| Execution | `jupyter-cache`; well understood, robust on CI | built-in execution with cache; equally usable, younger |
| Authoring loop | full rebuild per change (slow on a 20+ chapter book) | `myst start` live preview with incremental rebuilds; the fastest way to iterate on figures |
| Extensions | the whole Sphinx ecosystem: `autodoc` for the `dwibook` package, `sphinx-proof`, `sphinx-exercise`, `sphinx-design`, `sphinxcontrib-bibtex`, mermaid | built-in equivalents for proofs/exercises/cards/bibtex/mermaid; **no autodoc** (API docs of `dwibook` would be hand-written or generated separately) |
| Cross-referencing | Sphinx roles; works | richer: hover previews of figures/equations, cross-project links |
| PDF | LaTeX via Sphinx; battle-tested | Typst or LaTeX export; good but less exercised on long books |
| In-browser execution | Thebe (works, dated) | Thebe/JupyterLite integration is first-class |
| Dependencies | Python only | needs Node ≥ 20 (in the env via conda-forge) |
| Exit cost | to JB2: replace `_config.yml`/`_toc.yml` with `myst.yml`, fix a few directive spellings | to JB1: the reverse |

**Decision: Jupyter Book 2** (2026-09-24). The book is long, figure-heavy, and will be iterated
on for years; live preview and a maintained engine matter more than Sphinx extensions. The one
real loss is `autodoc`, and the helper package is small enough that its reference page can be
hand-written. The exit cost is a config swap, so the decision is low-risk. Jupyter Book 1 was
removed from the environment once the decision was made.

## 3. Repository layout

```
diffusion-book/
  myst.yml                    # JB2 config: toc, execution, exports
  environment.yml             # the dwibook env (Python side)
  README.md
  docs/                       # planning docs (this file, outline)
  book/
    index.md
    00-frontmatter/  01-mri-physics/  02-diffusion-encoding/
    03-preprocessing/  04-modeling/  05-advanced/  appendices/
      *.md                    # MyST notebooks; one file per chapter section
    _static/                  # css, logos
    references.bib
  dwibook/                    # small importable helper package (pip -e .)
    data.py                   # pooch registry, load_dataset("noise-sweep") → paths/arrays
    phantoms.py               # toy phantoms: Shepp–Logan, crossing-fiber block, Bloch sim, random walks
    kspace.py                 # toy k-space/EPI/GRAPPA/PF/CS helpers used in Ch. 2–3
    schemes.py                # scheme generators + sphere/q-space plots (Ch. 6)
    plotting.py               # house style: slice mosaics, error maps, fit-vs-truth scatter
    truth.py                  # load the 27 truth maps + truth peaks; angular error metrics
    tests/
  pipelines/                  # OFFLINE data generation, run in WSL
    Snakefile                 # one rule per dataset id in the outline's table
    config/
      datasets.yaml           # dataset id → phantom, grid, scheme, trxscan flags
      schemes/                # DTI-30, HARDI-64, HBCD, DSI-257 bval/bvec (+ generator script)
    scripts/                  # prepare grids (wraps TRXScan's prepare_acquisition_grid.py),
                              # run trxscan, run truth maps, precompute heavy fits/tracking,
                              # render TRXViz PNGs, build manifest + checksums, package release
  data/                       # gitignored; pooch cache target
  .github/workflows/
    build.yml                 # fetch data → jupyter book build (execution cache) → deploy
    pipeline-smoke.yml        # optional: tiny-grid end-to-end TRXScan run to catch drift
```

Design rules:
- **Two tiers, hard boundary.** Anything that calls TRXScan, FSL, MRtrix, TRXViz, or takes more
  than ~60 s lives in `pipelines/` and ships its result in the data release. Notebooks only load,
  compute light things, and plot. This keeps CI green and makes the book editable without WSL.
- **One loader.** Every notebook obtains files through `dwibook.data.load_dataset(id)`, which
  resolves to the pooch cache and asserts checksums, so the manifest (Appendix B) is generated
  from the same registry, never hand-maintained.
- **Truth is a first-class input.** `dwibook.truth` exposes the 27 maps and truth peaks with the
  dipy conventions TRXScan validated against, so "fit vs. truth" is one function call per chapter.

## 4. TRXScan and mrsim-acq work items

What the book needs that the simulator does not expose today. Ordered by how many chapters
block on each. Sizes are rough. All are changes to the TRXScan/mrsim-acq repos, developed on a
branch there and pinned by commit in `pipelines/config`.

| # | Item | Blocks | Size | Notes |
|---|---|---|---|---|
| T1 | **Acquisition parameters from the CLI (or the planned TOML config)**: `--te`, `--partial-fourier`, `--ghost`, `--spikes`/`--spike-amplitude`, `--window {none,hann,tukey,fermi}`, `--readout-time` (or `--t-line`), `--acs-lines`, `--t-inhom` | Ch. 7, 9, 14, 20, `te-sweep`, `gibbs` | small–medium | Today `TE_MS = 88.0` is a `const` and the `Acquisition` literal in `src/bin/trxscan.rs` is hard-coded (t_line pinned to HBCD's 91.7 ms readout, PF 0.75, ghost 0.015). The struct already has every field; this is plumbing. The `config` feature's TOML is the cleaner home; a flag set is the faster one. |
| T2 | **Raw k-space export**: per-slice, per-coil, pre-GRAPPA/pre-combination complex k-space (and the sampling mask) as NIfTI complex64 or `.npy`, opt-in flag, ideally for a slice subset | Ch. 2, 3, `slab-kspace` | small–medium | The book's reconstruction chapter must start from *acquired* k-space (undersampled, PF-cropped, per coil), not from an FFT of the reconstructed image. Needs a hook in `mrsim-acq::kspace` before reconstruction; write once per volume × slice × coil. Keep to a slab (`--slices a:b`) to bound size. |
| T3 | **Per-volume TE** (a TE column, e.g. `--te-list` or a BIDS-style TSV) so a single run yields a multi-TE series with shared noise realization | Ch. 20 | small once T1 exists | Per-compartment T2 is already applied in k-space per volume; TE just needs to vary per volume. |
| T4 | **CS-style irregular undersampling mask** (random / Poisson-disc PE lines, or a user-supplied mask) as an alternative to regular GRAPPA `accel`, with the mask exported | Ch. 3 (k-space CS), 7 | medium | Optional if T2 lands: the book can undersample the exported *fully sampled* k-space retrospectively in Python, which is how retrospective k-space CS studies are done anyway. Unrelated to CS-DSI (Ch. 6.4), which undersamples q-space and needs no simulator change. So T4 is a nice-to-have. |
| T5 | **Multi-echo readout** (N EPI echoes per excitation, per-echo TE, T2* decay and distortion per echo) | Ch. 21 phantom figures | large | Stretch. mrsim-acq's readout model is single-shot SE-EPI; multi-echo needs a readout-time offset per echo and per-echo k-space. Ch. 21 ships toy-only if this slips. |
| T6 | **Restricted compartment** (cylinder/sphere with Δ/δ dependence) | Ch. 22 phantom figures | large | Out of scope for edition 1; Ch. 22 is toy-only. Note the truth maps already accept `--big-delta/--small-delta` for MAP-MRI units. |
| T7 | **b-tensor (LTE/PTE/STE) encoding** in `scheme` + signal stage | Ch. 15 QTI, 22 | medium–large | Truth for µFA/k_bulk/k_shear already exists, which makes this tempting, but it is a second-edition item. |
| T8 | **`--slices a:b` / small-FOV slab mode** for cheap runs | all k-space chapters, CI smoke test | small | Cropping the tissue maps upstream with nibabel achieves most of this without touching Rust; keep as a convenience. |

Decision for stage 2: implement **T1 and T2 first** (both small, both unblock several chapters);
T3 right after T1; treat T4 as retrospective-in-Python unless T2 turns out awkward; defer T5–T7.

## 5. Phantoms and datasets

**Primary phantom `sub-0001a`** (`rust-trx/data/trxscan_truth_data`): 1 mm probsegs, atlas
fieldmap, ACT+SIFT2 tractogram. **Secondary `sub-60501`** (`trxscan-nibs-reference`): the same
kind of inputs plus a measured DRBUDDI fieldmap and real AP/PA motion traces; its `simulate.sh`
is the template for our pipeline scripts.

**Working resolution.** Default book resolution is **2.5 mm isotropic** (not HBCD's 1.7 mm):
~3.6× fewer voxels than 1.7 mm, so the whole dataset table simulates in hours rather than days
and the release stays in the low GB. Chapter 7's voxel sweep is the one place 1.5–3.0 mm all
appear. Streamlines: `--subsample 200000 --seed 1` with SIFT2 weights, identical in `trxscan`
and `trxscan-microstructure` so truth and data match. Preset: `adult` (3T literature values)
everywhere except the Chapter 1 preset comparison.

**Schemes** (generated once by `pipelines/scripts/make_schemes.py`, checked in):
- `dti30`: 30 dirs b=1000 + 3 b0 (electrostatic repulsion)
- `hardi64`: 64 dirs b=2000 + 4 b0
- `hbcd`: the bundled HBCD AP/PA 4-shell scheme (as-is, it is the "real protocol" reference)
- `dsi257`: 257-point Cartesian grid to b≈4000 (lattice points with |q|² ≤ 16, i.e. radius 4; radius 5 is the 515-point grid)
- `csdsi64`: a random 64-point subset of `dsi257` (Ch. 6.4). Not simulated separately: the pipeline subsets the `dsi257` volumes so the CS-DSI and full-DSI comparisons share one noise realization, which is exactly how retrospective CS-DSI studies are done.
- `msmt`: HCP-like 1000/2000/3000 × 30 (only if `hbcd` proves an awkward teaching shell set; otherwise `hbcd` serves)

**Datasets** are those in the outline's dependency table. Each Snakemake rule produces
`data/<id>/` containing the BIDS outputs, the JSON provenance (full command line, TRXScan commit,
seed), and any precomputed heavy results for that chapter. Rough compute per run at 2.5 mm,
`kspace` + `par` build: signal stage minutes, acquisition stage tens of seconds; the motion
dataset (per-volume re-simulation) is the expensive one (× volumes). `--oversample 2` for
Gibbs doubles in-plane voxels ×4 and needs the memory headroom TRXScan's preflight warns about.

**Precomputed heavy results** (shipped with the dataset, loaded by notebooks):
FSL topup/eddy outputs (Ch. 10–12), gradwarp fields from HCP `gradunwarp` and from TORTOISE's
`CreateNonlinearityDisplacementMap` plus the `CreateGradientNonlinearityBMatrix` graddev image
(Ch. 13; both TORTOISE tools ship in the qsiprep image, `gradunwarp` is a pip install), MRtrix `dwi2response`/`dwi2fod`/`tckgen`/`tcksift2` outputs
and tractography scores (Ch. 16, 18), a full `qsiprep` run on the kitchen-sink dataset (Ch. 14),
NODDI/AMICO fits (Ch. 17), TRXViz renders (Ch. 0, 18). FSL, MRtrix, and qsiprep itself all come
from the `pennlinc/qsiprep` Docker image, invoked by Snakemake rules as
`docker run --rm -v <data>:/data pennlinc/qsiprep:<tag> <command>`; the image tag is pinned in
`pipelines/config/datasets.yaml` and recorded in every provenance JSON. Everything else (DTI/DKI/MAP-MRI fits,
denoising, unringing, k-space recon on a slab) is fast enough to run in the notebook, which is
the point of an executable book.

**Release packaging:** `pipelines/scripts/package_release.py` tars each dataset, writes
`registry.txt` (pooch format: path, sha256), and uploads as a GitHub release asset tagged
`data-vX.Y`. `dwibook.data` pins the tag.

## 6. Notebook standard

Every chapter section is one MyST notebook with this skeleton, enforced by a template and a
lint script (`dwibook/tests/test_structure.py`):

1. Front cell: learning goals (3–5 bullets), datasets used (`load_dataset` calls), estimated run time.
2. Physics/prose with equations (MyST math, numbered, glossary links).
3. "See it" code cells: minimal, readable, top-level; helpers imported from `dwibook`. No cell over ~30 lines and none over ~60 s.
4. "Measure it": at least one number or error map against truth where a truth exists.
5. "What this means for acquisition": 3–6 bullets.
6. Further reading (bibtex keys).

Figure conventions live in `dwibook.plotting`: same axial/coronal slice indices across the
book, fixed intensity windows per contrast, colorblind-safe diverging maps for error maps,
identical layout for every fit-vs-truth panel (map | truth | difference | scatter).

## 7. Build and CI

- `myst.yml` sets `execute: cache` so unchanged notebooks are not re-run; CI restores the
  execution cache and the pooch data cache with `actions/cache` keyed on the data tag.
- `build.yml`: create env from `environment.yml` (micromamba action) → `pip install -e .` →
  `jupyter book build --html` → deploy to `gh-pages` on `main`; PRs get a build check only.
- Failure policy: notebook execution errors fail the build. Optional-dependency chapters
  (sigpy, AMICO, fury) guard imports and render a static fallback figure from the data release
  if the extra is absent, so the core build never depends on them.
- `pipeline-smoke.yml` (optional, self-hosted or manual): runs one tiny-grid TRXScan rule to
  catch simulator drift against the pinned commit. Not required for edition 1.
- Local build in WSL: `micromamba run -n dwibook jupyter book start` for live preview.

## 8. Testing

- `dwibook/tests`: unit tests for toy simulators (Bloch sim against analytic decay, FFT
  round trips, scheme generators produce the requested shells, angular-error metric on known
  peaks, loader checksum failure path).
- Notebook execution in CI is the integration test; `nbmake`-style per-notebook timing is
  reported so slow cells are caught before they creep.
- Pipeline: each Snakemake rule ends with a validation script (dims, volume count, bval
  shells, no NaNs, truth/data phantom identity via matching `--subsample`/`--seed` in
  provenance JSON).

## 9. Phases and milestones

| Phase | Deliverable | Depends on |
|---|---|---|
| **0** (done) | outline + this plan | — |
| **1 Scaffold** | `dwibook` env; JB2 skeleton with the full TOC as stubs; `dwibook` package with loader + plotting; CI builds and deploys the empty book; `pipelines/` with grid prep and one working rule (`ref-clean`) run end to end in WSL; data release v0.1 | env approval (§10) |
| **1b Simulator PRs** | T1 (acquisition flags/TOML), T2 (k-space export), T3 (per-volume TE) on a TRXScan/mrsim-acq branch, pinned by commit | runs in parallel with 1 |
| **2 Physics (Part I, Ch. 4–5)** | Ch. 1–5 complete; all toy-tier, plus `slab-kspace` once T2 lands | 1, T2 for the phantom figures in Ch. 2–3 |
| **3 Encoding (Ch. 6–7)** | schemes checked in; `ref-schemes`, `voxel-sweep`, `te-sweep` datasets; Table 6.1 first version | 1, T1, T3 |
| **4 Preprocessing (Part III)** | `noise-sweep`, `gibbs`, `sdc-pair`, `eddy`, `motion-mb`, `kitchen-sink` datasets with FSL precomputes; Ch. 8–14 | 1, T1; `fsl` env for topup/eddy |
| **5 Modeling (Part IV)** | truth loader; DTI/DKI/MAP-MRI/CSD/NODDI/tractography chapters with fit-vs-truth; Ch. 19 matrix | 3 |
| **6 Advanced + polish** | Ch. 20 (needs `te-sweep`), 20–22 (toy), appendices auto-generated from the registry and pipeline config, glossary, PDF export; Codex adversarial review of the helper package and of each chapter's fit-vs-truth code; final Codex code review | all |

Each chapter is a PR. A chapter is "done" when its notebook executes in CI, its figures follow
the plotting conventions, and any truth comparison reports a number in the text.

**Phase 1 progress (2026-09-24).** Done: `myst.yml` with the full TOC, 31 chapter stubs that
each execute a cell, the `dwibook` package (loader with `$DWIBOOK_DATA` override and pooch
registry, toy phantoms, k-space helpers, scheme generators, truth metrics, plotting) with 22
passing unit tests, the CI workflow, `pipelines/config/datasets.yaml` encoding every dataset,
and a Snakefile with the grid-preparation and truth rules. Remaining: `pipelines/scripts/`
(`run_dataset.py`, `make_schemes.py`, `normalize_grid_names.py`, `package_release.py`), the
first `ref-clean` run end to end, the `data-v0.1` release, and the TRXScan `book` branch
(T1–T3).

**Phase 2 progress (2026-09-24).** Chapters 1–3 are written at the toy tier and execute in
the build: a Bloch isochromat simulator (FID, spin echo, T2*), the k-space/EPI chapter
(Fourier relationship, FOV/resolution/aliasing, blipped-EPI timing, partial Fourier, Gibbs
ringing, multi-shot phase ghosts), and the reconstruction chapter (coil combination, SENSE,
GRAPPA with an empirical g-factor, zero-fill/homodyne/POCS, Haar-FISTA compressed sensing,
Rician and non-central chi noise with the bias curve). The `dwibook.kspace` and
`dwibook.phantoms` helpers behind them have unit tests. Each chapter's phantom section
(`slab-kspace`) and the Chapter 1 preset comparison are marked pending on T2 and on a
`presets` dataset respectively.

## 10. Decisions

Decided (2026-09-24):
1. **Python environment:** `dwibook`, created from `environment.yml` (Python 3.11, dipy 1.12.1, mystmd 1.11.0 on Node 26, sigpy, snakemake 9).
2. **Simulator changes:** T1–T3 are developed on a branch of TRXScan/mrsim-acq and pinned by commit in the pipeline config.
3. **q-space families:** single-shell, multi-shell, DSI, and CS-DSI (the "DTI"/"CS-DTI" wording in the original brief meant DSI/CS-DSI).
4. **FSL/MRtrix:** used freely in the offline pipeline through the `pennlinc/qsiprep` Docker image, which also provides `qsiprep` itself for Ch. 14.
5. **Book engine:** Jupyter Book 2 (§2a).

Still open (defaults assumed until changed):
6. **Data hosting.** Default: GitHub release assets (2 GB per-file cap, so datasets are split per id); OSF/Zenodo if the release grows past that.
7. **Resolution.** Default: 2.5 mm for the book, HBCD's 1.7 mm only in the Ch. 7 voxel sweep.

## 11. Risks

- **Memory/time of oversampled runs.** `--oversample 2` at 2.5 mm is fine; at 1.7 mm it is the
  11+ GB case. Keep Gibbs on the 2.5 mm grid or on a slab.
- **Gaussian-compartment phantom vs. biophysical models.** NODDI/SMT fits to a stick+tensor+ball
  phantom will not look like in vivo fits. The book turns this into a teaching point (Ch. 17), but
  reviewers may read it as "the simulator is wrong". Say it up front in Ch. 0.2.
- **Simulator drift.** Pin TRXScan/mrsim-acq commits in the pipeline config and store the full
  command line in every dataset's provenance JSON.
- **CI time creep.** The 60-s cell rule and the execution cache are the guard; the timing report
  in §8 makes creep visible.
- **Headless 3-D rendering** (fury/vtk) is fragile on CI. Ship TRXViz PNGs from the pipeline
  instead and keep fury optional.
