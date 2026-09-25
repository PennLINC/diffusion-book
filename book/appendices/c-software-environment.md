---
title: "Appendix C: Software environment"
subtitle: Versions used to build this edition
kernelspec:
  name: python3
  display_name: Python 3
---

The book is built by executing every notebook in one Python environment, described by
`environment.yml` at the root of the repository. The versions below are read from the
environment that built the page you are reading, so they are the versions behind every
figure and number in it.

```{code-cell} python
:tags: [hide-input]
import importlib.metadata as md
import platform
import sys

print(f"Python {platform.python_version()} on {platform.system()} {platform.machine()}")
for pkg in ["numpy", "scipy", "nibabel", "dipy", "matplotlib", "scikit-image", "pooch", "cvxpy", "sigpy", "mystmd", "dwibook"]:
    try:
        print(f"{pkg:<14} {md.version(pkg)}")
    except md.PackageNotFoundError:
        print(f"{pkg:<14} not installed")
```

## Creating the environment

```bash
micromamba create -n dwibook -f environment.yml
micromamba run -n dwibook pip install -e .
```

The environment file:

```{code-cell} python
:tags: [hide-input]
from pathlib import Path

import dwibook

env = Path(dwibook.__file__).resolve().parents[1] / "environment.yml"
print(env.read_text() if env.exists() else "environment.yml not found next to the package")
```

## Building the book

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 myst build --html --execute
```

The notebooks execute in parallel, one kernel each; one BLAS thread per kernel keeps the
build at about two minutes. `myst start --execute` serves a live preview that rebuilds on
every edit. The execution cache in `_build/execute` is keyed on the notebook text, so after
a change to the `dwibook` package the cache must be cleared for the outputs to reflect it.

## Offline tools

The simulated datasets are produced by a separate pipeline that is not part of the book
build ([Appendix A](./a-trxscan-cookbook.md)):

| Tool | Role | Where it runs |
|---|---|---|
| TRXScan, `trxscan-microstructure` | the simulator and its ground truth | native binary; pinned to a commit in the pipeline configuration |
| FSL (topup, eddy, dtifit, bedpostx), MRtrix (dwi2response, dwi2fod, tckgen, tcksift2), TORTOISE, qsiprep | the reference preprocessing and reconstruction tools the chapters compare against | the `pennlinc/qsiprep` container image, tag pinned in the pipeline configuration |
| gradunwarp | HCP gradient-nonlinearity correction ([Chapter 13](../03-preprocessing/13-gradient-nonlinearity.md)) | pip, in the pipeline environment |
| Snakemake | pipeline driver | the `dwibook` environment |

## Reproducibility

Every number in the book comes from a seeded random number generator, so the toy-tier
results are identical between builds on the same versions. Small differences between
platforms are possible in the iterative fits (registration, free-water elimination,
compressed sensing) and are reported to the precision printed.
