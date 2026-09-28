---
title: "11. Eddy currents"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** two-shell synthetic series on the packaged 2 mm slice and 3 mm volume with a linear eddy model ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
- **`eddy`** (pending): modeled and replayed eddy currents, geometric and phase ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-eddy)).
- **`truth`** (pending): the 27 analytic ground-truth maps and the true fiber orientations ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-truth), [Appendix E](../appendices/e-truth-map-catalogue.md)).

Pipeline-tier datasets are simulated offline by TRXScan ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)) and are marked *pending* until their release; the figures that need them say so where they will appear.
:::

## Learning goals

After this chapter you can:

- recognize the direction- and b-dependent shear, scale, and translation that eddy currents
  produce, and tell them apart from susceptibility distortion
- correct them by registering the diffusion-weighted volumes and measure what remains
- state why the b-vectors must be rotated with the registration
- state which acquisition choices reduce eddy currents at the source

```{code-cell} python
:tags: [hide-cell]
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # one BLAS thread: the registration loops are many small operations
import numpy as np
import matplotlib.pyplot as plt
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel

from dwibook import phantoms, schemes, synth
from dwibook.plotting import PALETTE, set_style, show_image

set_style()
```

## The physics

The diffusion gradients are strong and switch quickly. A changing magnetic field induces
currents in every conductor nearby, the cryostat, the gradient coil former, the RF shield,
and those currents produce a magnetic field of their own that decays over tens of
milliseconds. Part of it is still present during the EPI readout. Like any unwanted field
during the readout, it displaces signal along the phase-encode axis ([Chapter 10](./10-susceptibility-distortion.md)), but
with two differences from the susceptibility field:

- **It is different in every volume.** The eddy field is proportional to the diffusion
  gradient, so it changes with the gradient direction and strength. The b=0 volumes are
  unaffected; each diffusion-weighted volume is distorted in its own way.
- **It is spatially simple.** To a good approximation the residual field is linear in
  space, so the displacement it produces is a combination of a shear, a scale, and a
  translation along the phase-encode axis, the pattern depending on which gradient axis
  was used.

A series with uncorrected eddy currents therefore consists of volumes that do not line up
with one another. A voxel at the edge of the brain contains tissue in some volumes and
background in others, and any fit across volumes reads that as a signal change. The
errors concentrate at edges and along the phase-encode axis, and they are largest at the
highest b-values.

## The artifact-free reference

The synthetic series uses two shells (12 directions each at b = 1000 and 2000) on the 2 mm
slice. The eddy model is linear: the displacement along the phase-encode axis grows with
the column coordinate for a gradient along columns (shear), with the row coordinate for a
gradient along rows (scale), and is uniform for a gradient along the slice axis
(translation), all in proportion to b.

```{code-cell} python
:tags: [hide-input]
t = phantoms.brain_slice()
mask = t["mask"]
bvals, bvecs = schemes.multi_shell({1000: 12, 2000: 12}, n_b0=2)
series = synth.synthetic_dwi(t, bvals, bvecs)
shifts = synth.eddy_shift(bvals, bvecs, mask.shape, strength=0.03)
distorted = np.stack([synth.displace_along_pe(series[..., v], shifts[..., v]) for v in range(len(bvals))], axis=-1)
print(f"largest displacement in any volume: {np.abs(shifts).max():.1f} voxels")
```

## See it: volumes that do not line up

The displacement is along the phase-encode axis, anterior-posterior, and it varies with
position in all three directions: with the left-right coordinate for a gradient along that
axis (a shear), with the anterior-posterior coordinate (a scale), and with the slice
position for the gradient component along the slice axis, which shifts each slice by a
different amount. The axial view shows the first two; the sagittal view shows the third, as
a brain outline that is displaced by a different amount on every slice. Radiologists and
quality-control tools look at sagittal reformats of diffusion-weighted volumes for this
reason. The 3 mm volume shows both views of one volume whose gradient has a large
slice-axis component:

```{code-cell} python
:tags: [hide-input]
vol = phantoms.brain_volume()
mask3 = vol["mask"]
b_v, g_v = schemes.single_shell(2000, 6, n_b0=1)
series3 = synth.synthetic_dwi(vol, b_v, g_v)
shifts3 = synth.eddy_shift(b_v, g_v, mask3.shape, strength=0.06)  # twice the chapter's strength, for visibility at 3 mm
v = 1 + int(np.argmax(np.abs(g_v[1:, 2])))  # the direction with the largest slice-axis component
dist3 = synth.displace_along_pe(series3[..., v], shifts3[..., v])
K, C = phantoms.VOLUME_VENTRICLE_SLICE, 26

fig, axes = plt.subplots(1, 4, figsize=(12, 3.6))
for ax, img_, m_, title in [
    (axes[0], series3[:, :, K, v], mask3[:, :, K], "axial, undistorted"),
    (axes[1], dist3[:, :, K], mask3[:, :, K], "axial, eddy-distorted: shear and scale"),
    (axes[2], series3[:, C, :, v].T[::-1], mask3[:, C, :].T[::-1], "sagittal, undistorted"),
    (axes[3], dist3[:, C, :].T[::-1], mask3[:, C, :].T[::-1], "sagittal, eddy-distorted: slice-dependent shift"),
]:
    ax.imshow(img_, cmap="gray", vmin=0, vmax=0.12)
    ax.contour(m_, levels=[0.5], colors=[PALETTE[1]], linewidths=0.8)
    ax.set_axis_off(); ax.set_title(title, fontsize=9)
fig.tight_layout()
print(f"gradient of volume {v}: ({g_v[v, 0]:+.2f}, {g_v[v, 1]:+.2f}, {g_v[v, 2]:+.2f}) in (anterior-posterior, left-right, slice); shift ranges from {shifts3[..., v][mask3].min():+.1f} to {shifts3[..., v][mask3].max():+.1f} voxels across the brain")
```

The rest of the chapter works on the single 2 mm slice, where the shear and scale are
visible in the axial view and the correction can be scored voxel by voxel:

```{code-cell} python
:tags: [hide-input]
edge = mask & ~np.roll(mask, 3, axis=0)  # the anterior brain edge
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
show_image(axes[0], series[..., 0], "b = 0 (undistorted)", vmin=0, vmax=0.5)
for ax, v in zip(axes[1:], [3, 9, 20]):
    d = distorted[..., v] / distorted[..., v].max()
    ax.imshow(d, cmap="gray", vmin=0, vmax=1)
    ax.contour(mask, levels=[0.5], colors=[PALETTE[1]], linewidths=0.8)
    ax.set_axis_off()
    ax.set_title(f"volume {v}, b = {bvals[v]:.0f}\ng = ({bvecs[v, 0]:+.2f}, {bvecs[v, 1]:+.2f}, {bvecs[v, 2]:+.2f})")
fig.tight_layout()
```

The orange outline is the true brain edge. Each diffusion-weighted volume overhangs or
falls short of it in a different pattern: a sheared brain, a stretched brain, a shifted
brain, according to the gradient direction of that volume.

## Correction step by step

Because the displacement is a low-order geometric transform per volume, the standard
correction is registration: each diffusion-weighted volume is aligned to a reference, and
the transform is restricted to the shear, scale, and translation along the phase-encode
axis. Two details decide the quality:

- **The reference.** Registering every volume to the b=0 image is simple but unreliable,
  because a b = 2000 image and a b=0 image have different contrast, and a similarity
  metric responds to the contrast difference as if it were a geometric one (on this
  synthetic series the result is a worse alignment than no correction).
  FSL eddy instead predicts what each volume should look like from all the others, using a
  Gaussian process over the sphere of directions, and registers each volume to its own
  prediction {cite:p}`andersson2016`. That is why eddy needs a full set of directions and
  works better with more of them. The version below does the same with a tensor model as
  the predictor and three rounds of prediction and registration.
- **The b-vectors.** If the registration includes a rotation, the gradient direction in
  the head's frame has rotated too, and the b-vector table must be rotated with it
  {cite:p}`leemans2009`. The eddy shear is not a rotation, but head motion ([Chapter 12](./12-motion-and-dropout.md)) is,
  and the two are estimated together.

```{code-cell} python
:tags: [hide-input]
gtab = gradient_table(bvals, bvecs=bvecs)
corrected = distorted.copy()
estimated = np.zeros((len(bvals), 3))
for iteration in range(3):
    # predict every volume from a tensor fit of the current estimate, then register each
    # acquired volume to its own prediction with the three-parameter eddy model
    fit = TensorModel(gtab, fit_method="WLS", return_S0_hat=True).fit(corrected, mask=mask)
    pred = fit.predict(gtab, S0=fit.S0_hat)
    for v in range(len(bvals)):
        if bvals[v] == 0:
            continue
        estimated[v], corrected[..., v] = synth.register_pe_affine(distorted[..., v], pred[..., v], mask=mask)

def edge_error(s):
    rim = (mask & ~np.roll(mask, 2, axis=0)) | (mask & ~np.roll(mask, -2, axis=0))
    return np.abs(s - series)[rim][:, bvals > 0].mean()

truth_params = np.column_stack([0.03 * bvals / 1000 * bvecs[:, 1], 0.03 * bvals / 1000 * bvecs[:, 0], 4.0 * 0.03 * bvals / 1000 * bvecs[:, 2]])
print("mean absolute error at the anterior/posterior brain edges, diffusion-weighted volumes:")
print(f"  distorted {edge_error(distorted):.4f}   corrected {edge_error(corrected):.4f}")
print(f"estimated eddy parameters vs truth (RMS over volumes): shear {np.sqrt(np.mean((estimated[:, 0] - truth_params[:, 0])**2)):.4f}, "
      f"scale {np.sqrt(np.mean((estimated[:, 1] - truth_params[:, 1])**2)):.4f}, translation {np.sqrt(np.mean((estimated[:, 2] - truth_params[:, 2])**2)):.3f} voxels")
```

## Residual error versus truth

```{code-cell} python
:tags: [hide-input]
ref_fit = synth.dti_maps(series, bvals, bvecs, mask=mask)
rows = [(label, synth.dti_maps(s, bvals, bvecs, mask=mask)) for label, s in [("uncorrected", distorted), ("registered", corrected)]]
fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2))
show_image(axes[0], ref_fit["fa"], "FA, reference", kind="scalar", vmin=0, vmax=0.9)
for ax, (label, m) in zip(axes[1:], rows):
    show_image(ax, (m["fa"] - ref_fit["fa"]) * mask, f"FA error, {label}", kind="diff", vmin=-0.3, vmax=0.3)
fig.tight_layout()
wm = t["wm"] > 0.9
for label, m in rows:
    e = (m["fa"] - ref_fit["fa"])
    print(f"{label:>12}: FA error in WM {np.abs(e)[wm].mean():.3f}; at the brain edge {np.abs(e)[mask & ~np.roll(mask, 2, axis=0)].mean():.3f}")
```

Uncorrected, the FA error forms a rim around the brain and along the ventricles, where
volumes disagree about where the tissue is, and a fainter texture inside, where the shear
moved tissue by a fraction of a voxel. Registration to the predictions removes the rim; what
remains comes from the interpolation of each resampled volume and from the parameters the
prediction did not pin down exactly.

## The phase view

:::{admonition} Simulated dataset pending
:class: note
This section will load the `eddy` dataset: the modeled field (`--eddy`, `--eddy-quad`), a
measured per-volume field replayed from sub-60501, and the eddy phase ramp in the complex
output, which shows the per-volume field directly in the phase image before any correction
is attempted. FSL eddy's output from the pipeline will be scored against the `truth` maps.
:::

## What acquisition choices reduce it

- **Twice-refocused spin echo** ([Chapter 5](../02-diffusion-encoding/05-diffusion-encoding.md)) splits each diffusion lobe into two with
  opposite polarity so that the eddy fields cancel at the readout, at the cost of a longer
  TE. Many vendors offer it; most current high-b protocols use the single-refocused
  sequence and rely on correction.
- **A full set of well-distributed directions** makes the prediction-based correction
  work; a scheme with few directions or with all directions in one hemisphere gives the
  Gaussian process little to work with.
- **Interleave the shells and spread the b=0 volumes** so that the reference is available
  throughout the acquisition.
- **Save the phase**: the eddy phase ramp is visible per volume and provides a direct check
  on the correction.

## Further reading

The twice-refocused sequence {cite:p}`reese2003`, the integrated eddy and motion
correction of FSL eddy {cite:p}`andersson2016`, and why the b-matrix must be rotated
{cite:p}`leemans2009`.
