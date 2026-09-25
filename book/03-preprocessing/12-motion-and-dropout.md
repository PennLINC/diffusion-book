---
title: Head motion, multiband, and slice dropout
subtitle: Chapter 12
kernelspec:
  name: python3
  display_name: Python 3
---

## Learning goals

After this chapter you can:

- separate motion between volumes from motion during a volume and say what each does to
  the data
- correct between-volume motion by registration, rotate the b-vectors accordingly, and
  measure the residual
- recognize slice dropout, explain why multiband spreads it over several slices, detect it
  from the model residuals, and replace it
- report motion with the quality measures pipelines use

**Datasets used:** `motion-mb`, `truth` (pending); the toy tier uses the 3 mm volume
**Simulation tier:** toy + phantom

```{code-cell} python
:tags: [hide-cell]
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # one BLAS thread: registration is many small operations
import numpy as np
import matplotlib.pyplot as plt
from dipy.align.imaffine import AffineRegistration, MutualInformationMetric
from dipy.align.transforms import RigidTransform3D
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel

from dwibook import phantoms, schemes, synth
from dwibook.plotting import PALETTE, set_style, show_image

set_style()
```

## The physics

A diffusion acquisition lasts minutes and consists of dozens of volumes, each acquired
slice by slice within a few seconds. Head motion affects it on two time scales:

- **Between volumes.** The head is in a different position for different volumes. Each
  volume is internally consistent but misaligned with the others, so a voxel does not
  contain the same tissue throughout the series. The fix is registration, with one
  addition specific to diffusion: a rotation of the head is a rotation of the gradient
  direction relative to the tissue, so the b-vectors must be rotated by the same amount
  {cite:p}`leemans2009`.
- **During the diffusion encoding of one slice.** The encoding gradients make the signal
  phase proportional to displacement (Chapter 5). A small movement during the 40 ms
  between the pulses gives all spins in a slice a large, spatially varying phase, and the
  signal of that slice is partly or completely lost. This is **slice dropout**: a dark
  slice, or a dark band of slices, in one volume. It scales with b, so it is most frequent
  in the highest shell, and it is the dominant motion artifact in children and clinical
  populations.

**Multiband** acquisition excites several slices at once and separates them using the
coils (Chapter 7). A movement during one excitation therefore affects all slices of that
group, spread across the brain at regular intervals, rather than one slice.

## The TRXScan flags

`--motion <tsv>` replays a measured per-volume head trajectory: for each volume the
tractogram and tissue maps are moved to that pose and the signal is re-simulated, so the
fiber-gradient angles change correctly. `--mb 3 --dropout-rate 0.1` simulates a multiband
acquisition in which 10 % of the diffusion-weighted volumes suffer a motion event during
one shot, attenuating that shot's slices; the affected volumes, shots, and slices are
written to a TSV as ground truth for outlier detection.

## The artifact-free reference

The toy demonstrations use the 3 mm volume with a single-shell scheme (12 directions at
b = 1000 plus 2 b=0), 14 volumes in all. Three volumes are acquired with the head rotated
and translated; the series is computed with the gradients seen from the rotated head, then
the images are moved.

```{code-cell} python
:tags: [hide-input]
vol = phantoms.brain_volume()
mask = vol["mask"]
K = phantoms.VOLUME_VENTRICLE_SLICE
bvals, bvecs = schemes.single_shell(1000, 12, n_b0=2)
still = [((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))] * len(bvals)
poses = list(still)
poses[5] = ((4.0, 0.0, 0.0), (0.0, 0.5, 0.0))    # 4 degrees about the row axis, 1.5 mm shift
poses[8] = ((0.0, -3.0, 2.0), (0.7, 0.0, 0.0))   # 3 and 2 degrees about two axes, 2 mm shift
poses[11] = ((0.0, 0.0, 6.0), (0.0, 0.0, 0.5))   # 6 degrees in-plane
reference, _ = synth.moving_series(vol, bvals, bvecs, still)
moved, bvecs_head = synth.moving_series(vol, bvals, bvecs, poses)
print(f"series {moved.shape}; volumes 5, 8, 11 moved")
```

## See it: between-volume motion

```{code-cell} python
:tags: [hide-input]
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
show_image(axes[0], reference[:, :, K, 5], "volume 5, head still", vmin=0, vmax=0.2)
for ax, v in zip(axes[1:], [5, 8, 11]):
    ax.imshow(moved[:, :, K, v], cmap="gray", vmin=0, vmax=0.2)
    ax.contour(mask[:, :, K], levels=[0.5], colors=[PALETTE[1]], linewidths=0.8)
    ax.set_axis_off(); ax.set_title(f"volume {v}, moved")
fig.tight_layout()
```

## Correction step by step: registration and b-vector rotation

Each volume is registered rigidly to the first b=0 volume (six parameters: three rotations,
three translations). For diffusion-weighted volumes, FSL eddy registers to a predicted
image rather than to the b=0 (Chapter 11); the toy version registers to the b=0 directly.
The rotation recovered by the registration is then applied to the b-vector of that volume.

```{code-cell} python
:tags: [hide-input]
affreg = AffineRegistration(metric=MutualInformationMetric(nbins=32, sampling_proportion=None),
                            level_iters=[100, 30], sigmas=[1.0, 0.0], factors=[2, 1], verbosity=0)
registered = moved.copy()
bvecs_rot = bvecs.copy()
static = moved[..., 0]
for v in [5, 8, 11]:
    mapping = affreg.optimize(static, moved[..., v], RigidTransform3D(), None)
    registered[..., v] = mapping.transform(moved[..., v])
    R_est = mapping.affine[:3, :3]  # maps the moved volume's grid onto the static grid
    bvecs_rot[v] = R_est @ bvecs[v]
    err_deg = np.degrees(np.arccos(np.clip(abs(bvecs_rot[v] @ bvecs_head[v]), 0, 1)))
    print(f"volume {v:>2}: rotation recovered to within {err_deg:.1f}° of the true gradient direction in the head's frame")
```

## Residual error versus truth

Three fits show what each step buys: the moved series with the nominal b-vectors
(uncorrected), the registered series with the nominal b-vectors (images aligned, gradients
not rotated), and the registered series with the rotated b-vectors.

```{code-cell} python
:tags: [hide-input]
ref_fit = synth.dti_maps(reference, bvals, bvecs, mask=mask)
cases = [("uncorrected", moved, bvecs), ("registered, b-vectors not rotated", registered, bvecs), ("registered, b-vectors rotated", registered, bvecs_rot)]
wm = vol["wm"] > 0.9
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
show_image(axes[0], ref_fit["fa"][:, :, K], "FA, reference", kind="scalar", vmin=0, vmax=0.9)
for ax, (label, s, bv) in zip(axes[1:], cases):
    m = synth.dti_maps(s, bvals, bv, mask=mask)
    fa_err = np.abs(m["fa"] - ref_fit["fa"])[wm].mean()
    ang = np.degrees(np.arccos(np.clip(np.abs(np.sum(m["evecs"][..., 0] * ref_fit["evecs"][..., 0], axis=-1)), 0, 1)))
    show_image(ax, (m["fa"] - ref_fit["fa"])[:, :, K] * mask[:, :, K], f"FA error, {label}", kind="diff", vmin=-0.3, vmax=0.3)
    print(f"{label:>34}: WM FA error {fa_err:.3f}, principal-direction error {np.median(ang[wm]):.1f}°")
fig.tight_layout()
```

Registration removes the large errors at edges. Rotating the b-vectors then removes a
smaller, distributed error in the fiber directions; with rotations of a few degrees the
direction error is a few degrees, which matters for tractography more than for FA.

## Slice dropout

A within-volume event is simulated in a single volume: the head moves during one multiband
excitation, and the three slices of that shot lose 60 % of their signal.

```{code-cell} python
:tags: [hide-input]
MB = 3
n_slices = mask.shape[2]
shot = 12
dropped = [shot + k * (n_slices // MB) for k in range(MB)]
dwi_drop = synth.dropout(reference, volume=4, slices=dropped, attenuation=0.4)

fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.4))
show_image(axes[0], dwi_drop[:, :, dropped[1], 4], f"volume 4, slice {dropped[1]}: dropped", vmin=0, vmax=0.2)
show_image(axes[1], dwi_drop[:, :, dropped[1] + 1, 4], f"volume 4, slice {dropped[1] + 1}: normal", vmin=0, vmax=0.2)
axes[2].imshow(dwi_drop[:, 26, :, 4].T, cmap="gray", vmin=0, vmax=0.2, origin="lower", aspect="auto"); axes[2].set_axis_off()
axes[2].set_title(f"coronal view: multiband {MB} drops slices {dropped}")
fig.tight_layout()
```

### Detection and replacement

Dropout is detected from the model: fit the tensor to all volumes, predict every volume
from the fit, and compare each slice of each volume with its prediction. A dropped slice
has a residual far outside the distribution of that slice across volumes. The affected
slice is then replaced by the prediction, and the fit is repeated without it
{cite:p}`andersson2016b`. The same principle is used by FSL eddy (`--repol`) and by
SHORELine in QSIPrep.

```{code-cell} python
:tags: [hide-input]
gtab = gradient_table(bvals, bvecs=bvecs)
fit = TensorModel(gtab, fit_method="WLS", return_S0_hat=True).fit(dwi_drop, mask=mask)
pred = fit.predict(gtab, S0=fit.S0_hat)
dw = np.flatnonzero(bvals > 0)  # only the diffusion-weighted volumes are candidates for dropout
resid = np.zeros((n_slices, len(dw)))
for k in range(n_slices):
    m = mask[:, :, k]
    if m.sum() > 20:
        # relative residual: (measured - predicted) / predicted, averaged over the slice
        resid[k] = np.mean(((dwi_drop[:, :, k, :][..., dw] - pred[:, :, k, :][..., dw]) / np.maximum(pred[:, :, k, :][..., dw], 1e-3))[m], axis=0)
# a robust z-score: median and median absolute deviation, so that the outlier itself does not
# inflate the scale it is judged against (with 12 volumes, one outlier would otherwise
# dominate the standard deviation and hide itself)
med = np.median(resid, axis=1, keepdims=True)
mad = 1.4826 * np.median(np.abs(resid - med), axis=1, keepdims=True) + 1e-6
z = (resid - med) / mad
flagged = [(int(k), int(dw[j])) for k, j in np.argwhere(z < -4)]
print("slices with residual more than 4 robust SD below their own median (slice, volume):", flagged)

fig, ax = plt.subplots(figsize=(7, 3.2))
im = ax.imshow(z, cmap="RdBu_r", vmin=-8, vmax=8, aspect="auto", origin="lower", extent=(dw[0] - 0.5, dw[-1] + 0.5, -0.5, n_slices - 0.5))
ax.set(xlabel="volume", ylabel="slice", title="standardized residual of each slice in each diffusion-weighted volume")
fig.colorbar(im, ax=ax, shrink=0.8, label="robust z")
fig.tight_layout()

repaired = dwi_drop.copy()
for k, v in flagged:
    repaired[:, :, k, v] = pred[:, :, k, v]
fit_drop = synth.dti_maps(dwi_drop, bvals, bvecs, mask=mask)
fit_rep = synth.dti_maps(repaired, bvals, bvecs, mask=mask)
k = dropped[1]
print(f"FA error in WM of slice {k}: with dropout {np.abs(fit_drop['fa'] - ref_fit['fa'])[:, :, k][wm[:, :, k]].mean():.3f}, "
      f"after replacement {np.abs(fit_rep['fa'] - ref_fit['fa'])[:, :, k][wm[:, :, k]].mean():.3f}")
```

Replacement uses the other volumes' information to fill the gap; it cannot recover the
measurement, and a dataset with many dropped slices in the same shell is effectively a
dataset with fewer directions. Pipelines report the number of replaced slices per volume
as a quality measure, and studies set a threshold above which a scan is excluded.

## Reporting motion

Pipelines summarize motion as the framewise displacement between successive volumes, the
translation plus the rotation converted to millimeters at the head's surface, and as the
count of outlier slices. Both should be inspected before any group analysis, because
motion correlates with age and with clinical status, and residual motion effects bias FA
downward and MD upward in a way that can masquerade as a group difference.

## Measure it: the phantom

:::{admonition} Phantom figure pending
:class: note
This section will load the `motion-mb` dataset, in which the phantom is re-simulated for
each volume at the head pose measured in a real subject (sub-60501) and 10 % of the volumes
carry a multiband dropout event, and score FSL eddy's motion estimates and outlier
detection against the poses and the dropout TSV that TRXScan wrote.
:::

## What acquisition choices reduce it

- **Shorter scans move less.** Multiband and short TR reduce the time per volume and the
  total time.
- **Interleaved shells and spread b=0 volumes** limit the damage of a movement to a few
  directions of every shell rather than a whole shell.
- **Higher multiband factors spread dropout** over more slices per event.
- **Padding, instruction, and, for children, mock scanning** reduce motion more than any
  sequence parameter.
- **Acquire enough directions that replacing some slices leaves a usable scheme.**

## Further reading

Rotating the b-matrix {cite:p}`leemans2009`, integrated motion and eddy correction
{cite:p}`andersson2016`, and outlier detection and replacement {cite:p}`andersson2016b`.
