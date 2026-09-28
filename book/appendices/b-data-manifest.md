---
title: "Appendix B: Data manifest"
subtitle: Every file the notebooks load, with checksums
kernelspec:
  name: python3
  display_name: Python 3
---

The notebooks obtain every simulated file through one loader, `dwibook.data.load_dataset`,
which resolves a dataset id to a directory: a local pipeline output if `DWIBOOK_DATA` points
at one, otherwise a download from the data release verified against the checksums in the
package's registry. The registry is written by the pipeline's release script and never
edited by hand, and this appendix is rendered from it, so the manifest is the registry.

```{code-cell} python
:tags: [hide-cell]
from pathlib import Path

from dwibook import cookbook, data

registry_path = Path(data.__file__).with_name("registry.txt")
cfg = cookbook.load_config()
```

## The current release

```{code-cell} python
:tags: [hide-input]
entries = [line.split() for line in registry_path.read_text().splitlines() if line.strip() and not line.startswith("#")]
print(f"release tag: {data.DATA_TAG}")
if not entries:
    print("no data release has been published yet; the registry is empty")
else:
    print(f"{len(entries)} files in {len(data.registered_datasets())} datasets\n")
    print(f"{'path':<60} sha256")
    for path, digest in entries:
        print(f"{path:<60} {digest[:16]}...")
```

## Planned datasets

Until the release exists, the table lists what the pipeline configuration defines. Each dataset is described in [Appendix A](./a-trxscan-cookbook.md#app-a-datasets). Each
row becomes a BIDS dataset: one subject per source anatomy with a complex diffusion series
per run (`part-mag`, `part-phase`, `.bval`, `.bvec`, JSON sidecars), the ground truth and
the `provenance.json` in `derivatives/trxscan`, and the chapter's precomputed results in
one derivative dataset per tool. Appendix A shows the file tree of each.

```{code-cell} python
:tags: [hide-input]
print(f"{'dataset':<14} {'source anatomy':<22} {'runs':>5}   description")
for ds_id, ds in cfg["datasets"].items():
    phantoms = ds["phantom"] if isinstance(ds["phantom"], str) else ", ".join(ds["phantom"])
    n_runs = len(cookbook.render_commands(cfg, ds_id))
    print(f"{ds_id:<14} {phantoms:<22} {n_runs:>5}   {ds['description'].strip()[:80]}")
```

(app-b-package-data)=
## Package data

Three files ship inside the `dwibook` package so that the toy tier of the book builds
without any download: one axial slice of the simulated brain's tissue fractions at 2 mm
(`brain_slice.npz`), the same slice at 1 mm (`brain_slice_1mm.npz`), a 3 mm tissue volume
(`brain_volume.npz`), and the HBCD gradient scheme (`schemes/hbcd_ap.bval`, `.bvec`). Each
NPZ records its provenance:

```{code-cell} python
:tags: [hide-input]
from dwibook import phantoms

for name, loader in [("brain_slice.npz", lambda: phantoms.brain_slice(2.0)), ("brain_slice_1mm.npz", lambda: phantoms.brain_slice(1.0)), ("brain_volume.npz", phantoms.brain_volume)]:
    t = loader()
    print(f"{name}: shape {t['wm'].shape}, {t['voxel_mm']:g} mm\n    {t['provenance']}")
```

## Verifying a download

`load_dataset` raises if a file's checksum does not match the registry, and re-downloads
it. To verify a copy obtained by other means, compare its SHA-256 digests with the
registry lines above.
