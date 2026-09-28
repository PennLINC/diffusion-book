"""The book's phantom, simulated live: one slice of the hosted NIBS subject through TRXScan.

Every phantom-tier cell in the book goes through :func:`run`, so the protocol, the slice and
the seed are the same everywhere and a figure is regenerated in seconds at build time.
Full-brain runs (motion over all volumes, the kitchen sink, tractography) stay in the offline
pipeline (``pipelines/``) and are loaded with :func:`dwibook.data.load_dataset`.

Set ``DWIBOOK_PHANTOM=sub-60501`` (with ``TRXSCAN_DATA`` or a network connection) to run the
cells on the full phantom instead of the 20 MB slab bundle used on CI.
"""

from __future__ import annotations

import functools
import os
from typing import Any

import numpy as np
import trxscan as ts

#: The book's acquisition: the HBCD protocol at the book's working resolution.
PROTO = ts.Protocol.HBCD.replace(voxel_mm=2.5, coils=8, accel=2, name="book")

#: The book's artifact baseline for "realistic" figures (the CLI's subtle ghost plus noise).
ARTIFACTS = ts.Artifacts(noise=2e-4, ghost=0.015, seed=1)

#: World z (mm) of the book's axial slice: through the lateral ventricles.
SLICE_Z_MM = 10.0


@functools.lru_cache(maxsize=1)
def phantom() -> ts.Phantom:
    """The phantom (the ``slab`` bundle by default; ``$DWIBOOK_PHANTOM`` overrides)."""
    return ts.Phantom.load(os.environ.get("DWIBOOK_PHANTOM", "slab"))


@functools.lru_cache(maxsize=4)
def grid(protocol: ts.Protocol = PROTO) -> ts.Object:
    return phantom().grid(protocol)


def slice_index(protocol: ts.Protocol = PROTO) -> int:
    """The book's slice as a local index on the protocol's acquisition grid."""
    return grid(protocol).z_of(SLICE_Z_MM)


@functools.lru_cache(maxsize=1)
def gtab(n_vol: int | None = None):
    """The phantom's own HBCD acquisition as a dipy ``GradientTable`` (all 76 volumes, or the
    first ``n_vol``: b0s come first, then the shells interleaved)."""
    from dipy.core.gradients import gradient_table
    from dipy.io.gradients import read_bvals_bvecs

    bval, bvec = ts.data.scheme_files(os.environ.get("DWIBOOK_PHANTOM", "slab"), "AP")
    bvals, bvecs = read_bvals_bvecs(str(bval), str(bvec))
    if n_vol is not None:
        bvals, bvecs = bvals[:n_vol], bvecs[:n_vol]
    return gradient_table(bvals, bvecs=bvecs)


def run(gradients: Any = None, protocol: ts.Protocol = PROTO, artifacts: ts.Artifacts = ARTIFACTS, **kw: Any) -> ts.Simulation:
    """Simulate the book's slice: ``phantom().simulate(gradients, protocol, artifacts,
    slices=<book slice>, **kw)``."""
    g = gtab() if gradients is None else gradients
    kw.setdefault("slices", slice_index(protocol))
    return phantom().simulate(g, protocol, artifacts, **kw)


def truth(gradients: Any = None, protocol: ts.Protocol = PROTO, **kw: Any) -> dict[str, Any]:
    """The 27 ground-truth maps on the book's slice."""
    g = gtab() if gradients is None else gradients
    kw.setdefault("slices", slice_index(protocol))
    return phantom().microstructure(protocol, g, **kw)


def axial(img: Any, volume: int | None = None) -> np.ndarray:
    """A ``(nx, ny, 1[, n])`` simulation image as a 2-D array oriented anterior-up, as the
    book's plotting helpers expect."""
    a = np.asarray(img.dataobj if hasattr(img, "dataobj") else img)
    if a.ndim == 4:
        a = a[:, :, 0, :] if volume is None else a[:, :, 0, volume]
    else:
        a = a[:, :, 0]
    return np.rot90(a)
