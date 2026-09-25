---
title: Gibbs ringing
subtitle: Chapter 9
kernelspec:
  name: python3
  display_name: Python 3
---

## Learning goals

After this chapter you can:

- explain why every acquisition rings at sharp edges and why diffusion data ring worst at
  the ventricles
- show how the ringing changes with b-value and what that does to fitted diffusivities
- apply sub-voxel-shift unringing and measure what it removes
- compare correcting the ringing after the fact with apodizing at acquisition

**Datasets used:** `gibbs`, `truth` (pending); the toy tier uses a synthetic series
**Simulation tier:** toy + phantom

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt
from dipy.denoise.gibbs import gibbs_removal
from scipy import ndimage

from dwibook import kspace, phantoms, schemes, synth
from dwibook.plotting import PALETTE, set_style, show_image

set_style()
```

## The physics

An acquisition samples k-space out to a finite frequency, and the image is reconstructed
from that limited set. A sharp edge in the object needs all frequencies to be represented;
with the high ones missing, the reconstruction overshoots on both sides of the edge by
about 9 % of the step, with ripples that decay away from it (Chapter 2). The ripple spacing
is one voxel, so the artifact looks like fine stripes parallel to every sharp boundary.

In the brain the sharpest boundary is between CSF and tissue. In diffusion data this
boundary has a property that makes the ringing worse than in an anatomical image: its
contrast changes sign with b-value. At b = 0, CSF is much brighter than tissue; at
b ≥ 1000, CSF has decayed to nothing and is much darker. The ripples therefore flip sign
between the b=0 and the diffusion-weighted volumes. A voxel next to a ventricle that rings
upward in the b=0 image and downward in the high-b images shows a steeper apparent decay
than the tissue has, and a voxel one ripple farther out shows a shallower one. The result is
a striped pattern of over- and underestimated diffusivity along every CSF boundary, which
propagates into FA and into every model fitted downstream.

## The TRXScan flags

TRXScan simulates the object on a finer grid than the acquisition (`--oversample 2`) and
acquires only the nominal k-space band, so ringing arises exactly as it does on a scanner.
With `--oversample 1` the object sits on the acquisition grid and there is no ringing, which
provides the artifact-free reference. `--window hann|tukey|fermi` applies apodization at
reconstruction (implementation plan item T1).

## The artifact-free reference

The toy version follows the same design. The synthetic series is built on the 1 mm slice of
the phantom, and the acquisition keeps only the k-space band of a 2 mm matrix. The reference
is the same series block-averaged to 2 mm, which is what a ring-free 2 mm image of the object
would be.

```{code-cell} python
:tags: [hide-input]
fine_t = phantoms.brain_slice(1.0)
bvals, bvecs = schemes.multi_shell({1000: 6, 2000: 6}, n_b0=2)
fine = synth.synthetic_dwi(fine_t, bvals, bvecs)  # 256 x 256 x 14 at 1 mm

def block2(a):
    return a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2, *a.shape[2:]).mean(axis=(1, 3))

def acquire_2mm(series, window=None):
    k = kspace.fft2c(np.moveaxis(series, -1, 0))[:, 64:192, 64:192]  # keep the central 128 x 128 band
    if window is not None:
        k = k * window
    return np.moveaxis(np.abs(kspace.ifft2c(k)), 0, -1) * (128 / 256)

reference = block2(fine)
acquired = acquire_2mm(fine)
tissue2 = {k: block2(fine_t[k]) for k in ("wm", "gm", "csf")}
mask = (tissue2["wm"] + tissue2["gm"] + tissue2["csf"]) > 0.5
csf = tissue2["csf"] > 0.5
rim = ndimage.binary_dilation(csf, iterations=2) & ~csf & mask  # the two voxels of tissue next to CSF
print(f"series {acquired.shape}; rim voxels next to CSF: {rim.sum()}")
```

## See it: the ringing and its sign change

```{code-cell} python
:tags: [hide-input]
row = 58  # through the frontal horns of the lateral ventricles
v0, v1 = 0, 2  # a b=0 and a b=1000 volume
fig, axes = plt.subplots(2, 3, figsize=(10, 6), gridspec_kw={"width_ratios": [1, 1, 1.6]})
for r, (v, label, vmax) in enumerate([(v0, "b = 0", 0.5), (v1, "b = 1000", 0.2)]):
    show_image(axes[r, 0], reference[..., v], f"{label}, reference", vmin=0, vmax=vmax)
    show_image(axes[r, 1], acquired[..., v], f"{label}, acquired at 2 mm", vmin=0, vmax=vmax)
    axes[r, 1].axhline(row, color=PALETTE[1], lw=1)
    axes[r, 2].plot(reference[row, 40:90, v], color="0.3", lw=1, ls="--", label="reference")
    axes[r, 2].plot(acquired[row, 40:90, v], color=PALETTE[1], lw=1.5, label="acquired")
    axes[r, 2].set(title=f"{label}: profile through the ventricles", xlabel="column (voxels from 40)")
    axes[r, 2].legend()
fig.tight_layout()
```

At b = 0 the ripples next to the ventricle dip below the tissue value on the tissue side;
at b = 1000 the same voxels rise above it. The fitted diffusivity of those voxels reads the
ratio of the two, and is wrong in opposite directions in alternating voxels.

## Correction step by step: sub-voxel shifts

The unringing method of {cite:t}`kellner2016` uses a property of the ripples: they are
sampled at the worst possible positions, the voxel centers, where the truncated Fourier
series oscillates most. Shifting the sampling grid by a fraction of a voxel (which is a
phase ramp in k-space) changes the ripple amplitude, and for each voxel there is a sub-voxel
shift at which the local oscillation is smallest. The method reconstructs the image at many
sub-voxel shifts, measures the local total variation at each, and takes the value from the
shift that minimizes it. It removes the ripples without blurring the edge, and it is applied
per 2-D slice, volume by volume. Because it is a local operation on the magnitude image, it
belongs immediately after denoising and before anything that resamples the data.

```{code-cell} python
:tags: [hide-input]
unrung = gibbs_removal(acquired.copy(), slice_axis=2, n_points=3, inplace=False)
hann = np.outer(np.hanning(128), np.hanning(128))
apodized = acquire_2mm(fine, window=hann)

def rim_error(series):
    return np.abs(series - reference)[rim].mean()

def ripple(series):
    """Voxel-to-voxel oscillation along the phase-encode axis in the rim: the ringing itself."""
    second_diff = series[:-2] - 2 * series[1:-1] + series[2:]
    return np.abs(second_diff)[rim[1:-1]].mean()

print("in the rim next to CSF, all volumes:   error vs reference   ripple amplitude")
for label, s in [("reference", reference), ("acquired", acquired), ("unringed", unrung), ("Hann apodized", apodized)]:
    print(f"  {label:>14}:                     {rim_error(s):.4f}               {ripple(s):.4f}")

fig, axes = plt.subplots(1, 4, figsize=(11, 3))
show_image(axes[0], reference[..., v0], "reference", vmin=0, vmax=0.5)
show_image(axes[1], acquired[..., v0], "acquired", vmin=0, vmax=0.5)
show_image(axes[2], unrung[..., v0], "unringed", vmin=0, vmax=0.5)
show_image(axes[3], apodized[..., v0], "Hann apodized", vmin=0, vmax=0.5)
fig.tight_layout()
```

Apodization, the acquisition-side alternative, multiplies k-space by a window that goes
smoothly to zero at the edge. It removes the ripples because the truncation is no longer
abrupt, and it costs resolution: the edges are blurred. Unringing keeps the resolution.
Most scanners apply a mild filter by default; check the reconstruction settings, because
unringing data that were already apodized does nothing useful.

## Residual error versus truth

```{code-cell} python
:tags: [hide-input]
ref_fit = synth.dti_maps(reference, bvals, bvecs, mask=mask)
rows = []
for label, s in [("acquired", acquired), ("unringed", unrung), ("Hann apodized", apodized)]:
    m = synth.dti_maps(s, bvals, bvecs, mask=mask)
    rows.append((label, m))
    md_err = (m["md"] - ref_fit["md"])[rim] * 1e3
    fa_err = (m["fa"] - ref_fit["fa"])[rim]
    print(f"{label:>14}: rim MD error {md_err.mean():+.3f} ± {md_err.std():.3f} (x10^-3 mm^2/s)   rim FA error {fa_err.mean():+.3f} ± {fa_err.std():.3f}")

fig, axes = plt.subplots(1, 4, figsize=(11, 3))
show_image(axes[0], ref_fit["md"] * 1e3, "MD, reference", kind="scalar", vmin=0.5, vmax=3.0)
for ax, (label, m) in zip(axes[1:], rows):
    show_image(ax, (m["md"] - ref_fit["md"]) * 1e3 * mask, f"MD error, {label}", kind="diff", vmin=-0.5, vmax=0.5)
fig.tight_layout()
```

The error map of the acquired data shows the striped over- and underestimation along the
ventricles and the cortex. Unringing removes the stripes, which the ripple measure above
confirms; the error against the block-averaged reference falls less, because part of that
error is not ringing at all but the difference between a band-limited image and a
box-averaged one, which no unringing method addresses. What remains of the ringing sits at
corners and where two boundaries are within a few voxels of each other. Apodization removes
the stripes too, but replaces them with a blur that biases every boundary voxel toward its
neighbor, which the MD error shows as a systematic offset rather than a spread.

## Measure it: the phantom

:::{admonition} Phantom figure pending
:class: note
This section will load the `gibbs` dataset (`--oversample 2` against `--oversample 1`, plus
the Hann-windowed variant) and the `truth` maps, and repeat the rim measurements on the
simulated acquisition, where the ringing also interacts with the partial-Fourier
reconstruction of Chapter 3.
:::

## What acquisition choices reduce it

- **Ringing scales with voxel size relative to the anatomy**: the ripple is one voxel wide,
  so larger voxels ring over a larger distance. Higher resolution helps.
- **Partial Fourier changes the ringing** along the phase-encode axis, because the
  reconstruction fills in half of k-space from the other half; the unringing method has a
  partial-Fourier-aware variant.
- **Do not apodize at the scanner** if the data will be unrung; do check whether the
  scanner already did.
- **Order matters:** denoise, then unring, then everything that resamples.

## Further reading

Sub-voxel-shift unringing {cite:p}`kellner2016` and its implementation in dipy and MRtrix
{cite:p}`garyfallidis2014,tournier2019`.
