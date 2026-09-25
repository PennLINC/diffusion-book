---
title: "Appendix A: TRXScan cookbook"
subtitle: Every flag and command behind the book's datasets
kernelspec:
  name: python3
  display_name: Python 3
---

This appendix is generated from the pipeline configuration
(`pipelines/config/datasets.yaml`), the one file that describes what the offline pipeline
runs. The command lines below are the expansion of that file into `trxscan` and
`trxscan-microstructure` invocations, produced by the same rules the pipeline driver uses,
so the cookbook cannot drift from the pipeline. Paths are relative to the pipeline's working
directory; the tissue maps under `work/` come from TRXScan's `prepare_acquisition_grid.py`
(Chapter 0.2), and the schemes under `schemes/` from `dwibook.schemes`.

```{code-cell} python
:tags: [hide-cell]
from dwibook import cookbook

cfg = cookbook.load_config()
```

## The simulator in one paragraph

TRXScan takes a tractogram, tissue volume-fraction maps, a gradient scheme, and a fieldmap,
and writes a BIDS complex 4-D diffusion series (`part-mag`, `part-phase`, `.bval`, `.bvec`,
JSON sidecars). The signal stage rasterizes the streamlines into a per-voxel orientation
mixture and evaluates stick, tensor, and ball compartments per gradient; the acquisition
stage simulates each slice's k-space with EPI distortion from the fieldmap, T2 and T2*
decay during the readout, eddy currents, Nyquist ghosting, partial Fourier, Gibbs ringing
from a finer object grid, multi-coil reception, GRAPPA, and k-space noise; motion and
multiband dropout act per volume or per shot. The companion `trxscan-microstructure`
writes 27 analytic ground-truth maps (Appendix E) from the same mixture. Its physics is a
port of MITK Fiberfox {cite:p}`neher2014`.

## Flags used in this book

```{code-cell} python
:tags: [hide-input]
used = set(cookbook.flags_used(cfg))
print(f"{'flag':<28} what it controls")
for flag, what in cookbook.FLAG_GLOSSARY.items():
    if any(f in used for f in flag.split(" / ")):
        print(f"{flag:<28} {what}")
```

Flags marked *planned* correspond to simulator changes listed in the implementation plan;
the datasets that need them are generated once those changes land.

## Shared settings

```{code-cell} python
:tags: [hide-input]
d = cfg["defaults"]
print(f"acquisition voxel {d['voxel_mm']} mm; oversampling {d['oversample']}; {d['subsample']} streamlines sampled by SIFT2 weight with seed {d['seed']}; preset {d['params']}; default scheme {d['scheme']}")
print(f"tools: TRXScan branch {cfg['tools']['trxscan_branch']}; container image {cfg['tools']['qsiprep_image']}")
```

## Commands per dataset

```{code-cell} python
:tags: [hide-input]
for ds_id, ds in cfg["datasets"].items():
    print(f"\n### {ds_id}: {ds['description'].strip()}")
    if ds.get("requires"):
        print(f"    (waits on simulator items {', '.join(ds['requires'])})")
    for cmd in cookbook.render_commands(cfg, ds_id):
        print("   ", cmd)
```

## Regenerating a dataset

Every dataset directory carries a `provenance.json` with the exact command lines, the
TRXScan commit, and the container image tag. To regenerate one:

```bash
cd pipelines
micromamba run -n dwibook snakemake -c 8 <dataset-id>
```

The Snakefile prepares the acquisition grids for the phantom and voxel size the dataset
needs, runs the commands above, runs the precomputations the chapter needs inside the
QSIPrep container, validates the outputs, and writes the provenance file.
