---
title: Gradient nonlinearity
subtitle: Chapter 13
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Why this chapter comes last among the artifacts
:class: note
Gradient nonlinearity is a fixed property of the scanner, like the receive coils and the
gradient strength of Chapter 7, and could be introduced before the noise and motion of
the preceding chapters. It is placed here because its corrections are the last ones
applied: the geometric part is composed with the susceptibility, eddy, and motion
corrections of Chapters 10 through 12 into one resampling, and the encoding part is
handed to the model fits of Part IV. Reading it after those chapters keeps the order of
the pipeline (Chapter 14) and the order of the book the same.
:::

## Learning goals

After this chapter you can:

- explain why a gradient coil's field is not linear and name the two consequences: a
  spatial warp of the image and a per-voxel error in the diffusion encoding
- predict how both grow with distance from the isocenter
- apply the geometric correction (gradwarp) and the encoding correction (a per-voxel
  b-matrix) and show that each removes a different error
- state which acquisition choices reduce the problem

**Datasets used:** `gnl`, `truth` (pending); the toy tier uses a radial warp
**Simulation tier:** toy + phantom

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt

from dwibook import phantoms, presets, schemes, signal, synth
from dwibook.plotting import PALETTE, set_style, show_image

set_style()
```

## The physics

Spatial encoding assumes that each gradient coil produces a field that varies linearly
with position (Chapter 2). A real coil does so only near the isocenter of the magnet.
Farther out, the field falls away from the linear ramp, by a few percent at 10 cm from the
isocenter on a whole-body system and more on head-only or high-performance gradient
systems, where strength is bought at the expense of linearity. The deviation is a smooth,
fixed property of the hardware, described by a set of spherical-harmonic coefficients that
the vendor can supply.

Two things follow from one field:

1. **Spatial encoding error.** A spin at true position $r$ is assigned the position
   $\phi(r)$ where the field says it is. The image is warped, compressed toward the
   isocenter, and its intensity is scaled by the local volume change. The warp is the same
   in every volume and in every acquisition on that scanner, including the anatomical
   images, so it is not visible as a misalignment; it is visible as a size error and as a
   mismatch with images from other scanners.
2. **Diffusion encoding error.** The diffusion gradient a voxel experiences is the local
   derivative of the field, $J(r)^\top g$, not the nominal $g$. Both the b-value and the
   direction differ from voxel to voxel. Diffusivities fitted with the nominal gradient
   table are wrong by the local deviation of $b$, and fiber directions are rotated. This
   error survives the image unwarp, because unwarping moves voxels but does not change the
   gradient they were encoded with {cite:p}`bammer2003`.

## The phantom dataset

The simulated dataset for this chapter is `gnl`: the phantom on a whole-body and a
Connectom-class gradient system, a severity sweep, and runs with only the spatial warp or
only the encoding deviation, each written with its true coefficient file, displacement
field, and gradient-deviation image, scored against `truth`. The simulator settings that
produce it are listed under its name in [Appendix A](#app-a-datasets), and its files in
Appendix B.

## The artifact-free reference

The toy version uses a radial model, apparent radius $r\,(1 + a\,r^2/R^2)$ with $R$ = 100 mm,
on the 2 mm slice, with the isocenter placed 90 mm posterior to the anterior edge of the
brain so that the frontal lobe sits where the nonlinearity is strongest. Both effects are
computed from that one field: the images are warped, and each voxel's signal is computed
with its own effective b-value and direction.

```{code-cell} python
:tags: [hide-input]
t = phantoms.brain_slice()
mask = t["mask"]
vox = t["voxel_mm"]
A = 0.03  # 3 % apparent displacement at 100 mm from the isocenter (whole-body class)
center = (110.0, 63.5)  # isocenter, in (row, column) voxels: posterior of the brain
bvals, bvecs = schemes.multi_shell({1000: 12, 2000: 12}, n_b0=2)
orientation = synth.orientation_field(t["wm"])
reference = synth.synthetic_dwi(t, bvals, bvecs, orientation=orientation)

_, jac = synth.gnl_warp(np.zeros(mask.shape), A, vox, center)  # per-voxel gradient deviation (2 x 2, in-plane)
yy, xx = np.indices(mask.shape, dtype=float)
dist_mm = np.hypot(yy - center[0], xx - center[1]) * vox

# encoding error: each voxel sees g_eff = J g (in-plane part), so b and direction change
def encoded_series():
    out = np.zeros(mask.shape + (len(bvals),))
    for v in range(len(bvals)):
        g_in = jac @ bvecs[v, :2]                       # (ny, nx, 2)
        g_eff = np.concatenate([g_in, np.broadcast_to(bvecs[v, 2], mask.shape)[..., None]], axis=-1)
        norm = np.linalg.norm(g_eff, axis=-1)
        b_eff = bvals[v] * norm**2
        cos = np.sum(orientation * g_eff, axis=-1) / np.maximum(norm, 1e-12)
        s0 = {k: phantoms.PROTON_DENSITY[k] * np.exp(-presets.TE_HBCD_MS / presets.T2_MS["adult"][k]) for k in ("WM", "GM", "CSF")}
        out[..., v] = (t["wm"] * s0["WM"] * signal.white_matter(b_eff, cos) + t["gm"] * s0["GM"] * signal.gray_matter(b_eff) + t["csf"] * s0["CSF"] * signal.csf(b_eff))
    return out

encoded = encoded_series()                       # encoding error only
acquired, _ = synth.gnl_warp(encoded, A, vox, center)  # plus the spatial warp
b_scale = np.linalg.norm(jac @ np.array([0.0, 1.0]), axis=-1) ** 2
print(f"effective b / nominal b for a left-right gradient: {b_scale[mask].min():.2f} to {b_scale[mask].max():.2f} across the brain")
```

## See it: the warp and the encoding deviation

```{code-cell} python
:tags: [hide-input]
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
show_image(axes[0], reference[..., 0], "true geometry", vmin=0, vmax=0.5)
axes[0].contour(mask, levels=[0.5], colors=[PALETTE[1]], linewidths=0.8)
ax = axes[1]; ax.imshow(acquired[..., 0], cmap="gray", vmin=0, vmax=0.5); ax.contour(mask, levels=[0.5], colors=[PALETTE[1]], linewidths=0.8); ax.set_axis_off(); ax.set_title("acquired: warped toward the isocenter")
im = axes[2].imshow(np.where(mask, dist_mm, np.nan), cmap="viridis"); axes[2].set_axis_off(); axes[2].set_title("distance from isocenter (mm)"); fig.colorbar(im, ax=axes[2], shrink=0.7)
im = axes[3].imshow(np.where(mask, b_scale, np.nan), cmap="RdBu_r", vmin=0.8, vmax=1.2); axes[3].set_axis_off(); axes[3].set_title("effective b / nominal b (L-R gradient)"); fig.colorbar(im, ax=axes[3], shrink=0.7)
fig.tight_layout()
```

The orange outline is the true brain edge; the acquired image falls inside it, more so at
the front. The effective b-value is too high at the same locations, by up to 20 % for a
gradient along the direction of the warp.

## Correction step by step: gradwarp

The geometric correction resamples the image from the apparent grid back to the true one,
using the displacement field computed from the coefficient file, and multiplies by the
Jacobian to restore the intensity. The HCP pipelines do this with `gradunwarp`
{cite:p}`glasser2013`; TORTOISE's `CreateNonlinearityDisplacementMap` produces the same
field, and qsiprep applies it when given the coefficient file. The displacement is composed
with the susceptibility and eddy corrections so that the data are resampled once
(Chapter 14). Note that the anatomical images are warped by the same field and are
corrected on the scanner by default on most systems; the diffusion data usually are not.

```{code-cell} python
:tags: [hide-input]
unwarped = synth.gnl_unwarp(acquired, A, vox, center)
print(f"mean absolute error of the b=0 image inside the brain: acquired {np.abs(acquired[..., 0] - reference[..., 0])[mask].mean():.4f}, "
      f"after gradwarp {np.abs(unwarped[..., 0] - reference[..., 0])[mask].mean():.4f}, "
      f"encoding error alone {np.abs(encoded[..., 0] - reference[..., 0])[mask].mean():.4f}")
```

The b=0 image is fully recovered by gradwarp; the encoding error does not touch it, because
b = 0 has no encoding to get wrong.

## Correction step by step: per-voxel b-matrix

The encoding error is corrected in the fit, not in the images. The gradient-deviation
tensor $J$ at each voxel (`CreateGradientNonlinearityBMatrix`, or qsiprep's `graddev`
output) converts the nominal gradient table into the effective one for that voxel, and the
model is fitted with a different design matrix in every voxel. FSL's `dtifit --gradnonlin`
and `bedpostx -g` accept the deviation image directly; for other models it is applied by
looping over voxels, as below, or by tools such as `odx graddev` for orientation
distributions. The deviation must be evaluated in the frame the data are in after
gradwarp, on the true grid, because it is a property of the true position.

```{code-cell} python
:tags: [hide-input]
def fit_per_voxel(series, use_graddev):
    """Log-linear tensor fit, with the nominal table or with each voxel's effective table."""
    md = np.full(mask.shape, np.nan); fa = np.full(mask.shape, np.nan)
    dw = bvals > 0
    logS = np.log(np.clip(series, 1e-6, None))
    for (i, j) in zip(*np.nonzero(mask)):
        if use_graddev:
            g_in = jac[i, j] @ bvecs[:, :2].T          # (2, n)
            g = np.column_stack([g_in.T, bvecs[:, 2]])
            n = np.linalg.norm(g, axis=1)
            b = bvals * n**2; g = g / np.maximum(n[:, None], 1e-12)
        else:
            b, g = bvals, bvecs
        X = np.column_stack([np.ones(len(b)), signal.dti_design_matrix(b, g)])
        beta = np.linalg.lstsq(X, logS[i, j], rcond=None)[0]
        D = np.array([[beta[1], beta[4], beta[5]], [beta[4], beta[2], beta[6]], [beta[5], beta[6], beta[3]]])
        ev = np.clip(np.linalg.eigvalsh(D), 0, None)
        md[i, j] = ev.mean()
        fa[i, j] = np.sqrt(1.5 * np.sum((ev - ev.mean()) ** 2) / max(np.sum(ev**2), 1e-20))
    return md, fa

md_ref, fa_ref = fit_per_voxel(reference, False)
md_nom, fa_nom = fit_per_voxel(encoded, False)    # encoding error, nominal gradient table
md_cor, fa_cor = fit_per_voxel(encoded, True)     # encoding error, per-voxel b-matrix
```

## Residual error versus truth

The encoding error is isolated first: the fits below use the series with the encoding
deviation but without the spatial warp, so that no interpolation is involved.

```{code-cell} python
:tags: [hide-input]
wm = t["wm"] > 0.9
bins = np.arange(20, 130, 10)
def radial(err):
    return [np.nanmean(np.abs(err)[wm & (dist_mm >= lo) & (dist_mm < lo + 10)]) for lo in bins]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.4))
ax1.plot(bins + 5, radial(100 * (md_nom - md_ref) / md_ref), label="nominal b-matrix")
ax1.plot(bins + 5, radial(100 * (md_cor - md_ref) / md_ref), label="per-voxel b-matrix")
ax1.set(xlabel="distance from isocenter (mm)", ylabel="|MD error| in WM (%)", title="mean diffusivity"); ax1.legend()
ax2.plot(bins + 5, radial(fa_nom - fa_ref), label="nominal b-matrix")
ax2.plot(bins + 5, radial(fa_cor - fa_ref), label="per-voxel b-matrix")
ax2.set(xlabel="distance from isocenter (mm)", ylabel="|FA error| in WM", title="fractional anisotropy"); ax2.legend()
fig.tight_layout()
far = wm & (dist_mm > 80)
rel = lambda md: 100 * np.nanmean(np.abs(md - md_ref)[far] / md_ref[far])
print(f"white matter more than 80 mm from the isocenter, encoding error only: MD error {rel(md_nom):.1f} % with the nominal table, {rel(md_cor):.1f} % with the per-voxel table")
md_full_nom, _ = fit_per_voxel(unwarped, False)
md_full_cor, _ = fit_per_voxel(unwarped, True)
print(f"same region, full pipeline (warp, gradwarp, then fit): {rel(md_full_nom):.1f} % with the nominal table, {rel(md_full_cor):.1f} % with the per-voxel table")
```

The error with the nominal gradient table grows with distance from the isocenter, as the
effective b-value departs from the nominal one; it reaches tens of percent of MD in the
frontal white matter of this example, far larger than the group differences most studies
report. The per-voxel b-matrix removes it. On the full pipeline a residual of several
percent remains that has nothing to do with the encoding: it is the interpolation error of
warping and unwarping the diffusion-weighted images at the edges of white matter, the same
cost every resampling carries (Chapter 14). Neither correction is applied by most default
pipelines unless the coefficient file is provided.

## Where it sits in the pipeline

Gradwarp is a resampling and belongs with the other geometric corrections (susceptibility,
eddy, motion), composed into one transform. The gradient-deviation image is computed on
the final grid and passed to the model fit; it is the last thing the preprocessing produces
and the first thing the reconstruction consumes. Applying only gradwarp leaves the
encoding error; applying only the b-matrix correction leaves the geometry wrong, which
matters wherever the diffusion data meet an anatomical image or an atlas.

## Measure it: the phantom

:::{admonition} Phantom figure pending
:class: note
This section will load the `gnl` dataset (whole-body and Connectom presets, a severity
sweep, and warp-only and encoding-only runs) with the truth displacement field and
gradient-deviation image TRXScan wrote, and score `gradunwarp`, TORTOISE, and the
per-voxel fits against them.
:::

## What acquisition choices reduce it

- **Position the head at the isocenter.** The deviation grows with the cube of the
  distance; a few centimeters matter.
- **Obtain the coefficient file** for the scanner before the study. Without it neither
  correction can be applied, and it is not part of the standard export.
- **Know the gradient system.** Head-insert and high-performance gradients are less linear
  than whole-body gradients; the gain in TE (Chapter 5) comes with a larger correction.
- **Do not compare uncorrected diffusivities across scanners** or across head positions.

## Further reading

The encoding error and its correction {cite:p}`bammer2003`, the HCP implementation
{cite:p}`glasser2013`, and the design note for the simulator's implementation in the
TRXScan repository.
