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

- recognize the direction- and b-dependent shear, stretch, and shift that eddy currents
  produce, and tell them apart from susceptibility distortion
- correct them by registering each diffusion-weighted volume to a prediction of itself, and
  measure what remains
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
from dwibook.plotting import INK, PALETTE, animate, set_style, show_image

set_style()
```

## The physics

The diffusion gradients are strong and switch on and off quickly. A changing magnetic field
induces electric currents in any metal nearby (the vessel that holds the superconducting
magnet, the structure the gradient coils are wound on, the shield around the radio-frequency
coil), and those currents, called eddy currents, produce a magnetic field of their own that
decays over tens of milliseconds. Part of it is still present during the EPI readout. Like
any unwanted field during the readout, it displaces signal along the phase-encode axis
([Chapter 10](./10-susceptibility-distortion.md)), but with two differences from the
susceptibility field:

- **It is different in every volume.** The eddy field is proportional to the diffusion
  gradient, so it changes with the gradient direction and strength. The b=0 volumes are
  unaffected; each diffusion-weighted volume is distorted in its own way.
- **It is spatially simple.** To a first approximation the residual field changes linearly
  across the head, so the displacement it produces is one of three simple patterns along the
  phase-encode axis, or a mix of them, depending on which gradient axis was used. (Real eddy
  fields also have curved components, so FSL eddy's default model of the field is quadratic
  in position, plus a constant term that shifts the whole volume, `--flm=quadratic`; this
  chapter's toy keeps only the linear part.)

The three patterns are easiest to see on a square grid. The phase-encode axis here runs
anterior-posterior (up-down in the pictures), and every point moves only along it:

```{code-cell} python
:tags: [hide-input]
lines = np.linspace(-1, 1, 7)
fine = np.linspace(-1, 1, 200)
theta = np.linspace(0, 2 * np.pi, 200)
head = (0.62 * np.cos(theta), 0.82 * np.sin(theta))  # a head-shaped ellipse, (left-right, anterior-posterior)
patterns = [("gradient along left-right\n→ shear", lambda x, y: 0.25 * x),
            ("gradient along anterior-posterior\n→ stretch", lambda x, y: 0.25 * y),
            ("gradient along the slice axis\n→ shift", lambda x, y: 0.25 + 0 * x)]
fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.6))
for ax, (title, s) in zip(axes, patterns):
    for c in lines:
        ax.plot(fine, c + 0 * fine, color="0.82", lw=1)   # undistorted grid
        ax.plot(c + 0 * fine, fine, color="0.82", lw=1)
        ax.plot(fine, c + s(fine, c + 0 * fine), color=PALETTE[0], lw=1.2)  # displaced grid: each point moves along A-P only
        ax.plot(c + 0 * fine, fine + s(c + 0 * fine, fine), color=PALETTE[0], lw=1.2)
    ax.plot(*head, color=INK["secondary"], lw=1, ls="--")
    ax.plot(head[0], head[1] + s(*head), color=PALETTE[1], lw=2)
    ax.set(xlim=(-1.3, 1.3), ylim=(1.45, -1.45), aspect="equal", title=title)
    ax.set_axis_off()
axes[0].annotate("", xy=(-1.25, 1.2), xytext=(-1.25, -1.2), arrowprops=dict(arrowstyle="<->", color=INK["secondary"]))
axes[0].text(-1.25, -1.33, "A", ha="center", fontsize=8, color=INK["secondary"])
axes[0].text(-1.25, 1.42, "P", ha="center", fontsize=8, color=INK["secondary"])
fig.tight_layout()
fig.subplots_adjust(top=0.84)
```

Gray is the grid and the head (dashed) as they are; blue and orange are where the eddy
field puts them. A gradient along left-right moves each column by an amount that grows
across the head, so horizontal lines tilt: a **shear**. A gradient along anterior-posterior
moves each row by an amount that grows from the center outward, so the head gets longer (or,
with the opposite gradient sign, shorter): a **stretch**. A gradient along the slice axis
moves everything in the slice by the same amount: a **shift**, which in a 3-D volume is
different on each slice. A real gradient direction has components along all three axes, so
each volume gets a mix, in proportions set by its direction.

A series with uncorrected eddy currents therefore consists of volumes that do not line up
with one another. A voxel at the edge of the brain contains tissue in some volumes and
background in others, and any fit across volumes reads that as a signal change. The
errors concentrate at edges and along the phase-encode axis, and they grow with b.

Both eddy currents and susceptibility ([Chapter 10](./10-susceptibility-distortion.md)) move
signal along the phase-encode axis, so the question when looking at a distorted series is
which one you are seeing:

| | Susceptibility distortion | Eddy-current distortion |
|---|---|---|
| In the b=0 volumes? | yes | no |
| Changes from volume to volume? | no, the same in every volume | yes, with the gradient direction |
| Changes with b? | no | yes, larger at higher b |
| Shape | local: pile-ups and stretches near air-filled spaces (sinuses, ear canals) | global: shear, stretch, or shift of the whole slice |

A distortion that is already in the b=0 image and does not change across the series is
susceptibility; a distortion that differs between diffusion-weighted volumes is eddy
currents (or head motion, [Chapter 12](./12-motion-and-dropout.md), which is also different
in every volume but is a rigid rotation and translation rather than a shear or stretch).

## The artifact-free reference

The synthetic series uses two shells (12 directions each at b = 1000 and 2000) on the 2 mm
slice. The eddy model is the linear one in the grid figure: each volume's displacement is
the shear, stretch, and shift above, weighted by the left-right, anterior-posterior, and
slice components of its gradient direction and scaled with b. (The model makes the
displacement proportional to b. On a scanner that reaches the higher shell by raising the
gradient amplitude, the eddy field is proportional to the amplitude, which grows as √b; the
direction dependence is the same.)

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

The animation steps through a b = 2000 series on the 3 mm volume, one b=0 volume and six
gradient directions, in an axial view and a sagittal view (the head seen from the side). The
orange outline is the true brain edge. In the axial view, watch the shear and stretch. In
the sagittal view, watch the slice-axis shift: the outline moves by a different amount on
every slice, which is why quality-control tools show sagittal views of diffusion-weighted
volumes. The label under the images names the gradient axis that dominates each volume.

```{code-cell} python
:tags: [hide-input]
vol = phantoms.brain_volume()
mask3 = vol["mask"]
b_v, g_v = schemes.single_shell(2000, 6, n_b0=1)
series3 = synth.synthetic_dwi(vol, b_v, g_v)
shifts3 = synth.eddy_shift(b_v, g_v, mask3.shape, strength=0.06)  # twice the chapter's strength, for visibility at 3 mm
dist3 = np.stack([synth.displace_along_pe(series3[..., v], shifts3[..., v]) for v in range(len(b_v))], axis=-1)
K, C = phantoms.VOLUME_VENTRICLE_SLICE, 26
AXIS_WORDS = ["anterior-posterior: stretch", "left-right: shear", "slice axis: shift"]  # gradient components are (A-P, L-R, slice)

def views(v):
    scale = np.percentile(dist3[..., v][mask3], 99)  # each volume on its own scale
    return dist3[:, :, K, v] / scale, dist3[:, C, :, v].T[::-1] / scale

fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.9))
ims = []
for ax, img_, m_, title in zip(axes, views(0), [mask3[:, :, K], mask3[:, C, :].T[::-1]], ["axial: shear and stretch", "sagittal: slice-dependent shift"]):
    ims.append(ax.imshow(img_, cmap="gray", vmin=0, vmax=1))
    ax.contour(m_, levels=[0.5], colors=[PALETTE[1]], linewidths=0.8)
    ax.set_axis_off(); ax.set_title(title, fontsize=9)
fig.tight_layout(rect=(0, 0.06, 1, 0.97))
stamp = fig.text(0.5, 0.02, "", ha="center", fontsize=10)

def frame(v):
    for im, img_ in zip(ims, views(v)):
        im.set_data(img_)
    stamp.set_text("b = 0: no eddy field" if b_v[v] == 0 else
                   f"volume {v}: gradient mostly {AXIS_WORDS[int(np.argmax(np.abs(g_v[v])))]}")

animate(fig, frame, range(len(b_v)), fps=1, width=620,
        alt="axial and sagittal views of each volume of a b = 2000 series in turn; the axial brain shears and stretches and the sagittal brain outline shifts by a different amount on each slice, differently for each gradient direction")
```

The rest of the chapter works on the single 2 mm slice, where the shear and stretch are
visible in the axial view and the correction can be scored voxel by voxel. Below are four
volumes of that series: the b=0 volume and the three b = 2000 volumes whose gradient lies
closest to each axis. The top row is what a viewer shows; compare each brain with the orange
outline of the true brain edge. The displacements are a few voxels at most, so the bottom row
subtracts the undistorted volume to make them visible: red where the distorted volume is
brighter than it should be (tissue has moved in), blue where it is darker (tissue has moved
out).

```{code-cell} python
:tags: [hide-input]
hi = np.flatnonzero(bvals == 2000)
picks = [(0, "b = 0: no eddy field")] + [
    (int(hi[np.argmax(np.abs(bvecs[hi, a]))]), f"b = 2000, gradient mostly\n{AXIS_WORDS[a]}") for a in (1, 0, 2)]
fig, axes = plt.subplots(2, 4, figsize=(10, 6.2))
for j, (v, title) in enumerate(picks):
    top = np.percentile(series[..., v][mask], 99)  # each volume on its own scale
    axes[0, j].imshow(distorted[..., v] / top, cmap="gray", vmin=0, vmax=1)
    show_image(axes[1, j], (distorted[..., v] - series[..., v]) / top, "distorted − true" if j == 0 else None, kind="diff", vmin=-0.6, vmax=0.6)
    for ax in axes[:, j]:
        ax.contour(mask, levels=[0.5], colors=[PALETTE[1]], linewidths=0.9)
        ax.set_axis_off()
    axes[0, j].set_title(title, fontsize=9)
fig.tight_layout()
print("largest displacement inside the brain: " + ", ".join(f"volume {v} {np.abs(shifts[..., v][mask]).max():.1f} voxels" for v, _ in picks[1:]))
```

The b=0 difference is blank: no eddy field. In the others the red-blue banding marks tissue
that moved, and it is strongest where the displacement is largest. For the shear volume
that is the left and right sides of the brain, which moved in opposite directions, with
little change down the middle. For the stretch volume it is the front and the back. The
slice-axis volume changes least: in a single slice its main term is a small uniform shift
(in the 3-D volume above it moves each slice by a different amount), and what banding there
is comes from the smaller left-right and anterior-posterior parts of its direction. In this
model the b = 2000 volumes move twice as far as the b = 1000 volumes with the same
direction.

## Correction step by step

Because the displacement in each volume is one of these simple patterns, the standard
correction is registration: each diffusion-weighted volume is aligned to a reference, and
the transform is restricted to the shear, stretch, and shift along the phase-encode axis.
The choice of reference decides the quality.

The obvious reference, the b=0 image, does not work well, because a b = 2000 image does not
look like a b=0 image even when the two are perfectly aligned: fluid is bright at b=0 and
dark at b = 2000, and white matter is bright or dark depending on the direction. A
registration that compares intensities directly, as this chapter's toy does (it minimizes
the sum of squared differences), treats those contrast differences as misalignment and
moves the volume to reduce them (the figure after the next cell shows the result). A metric
built for images of different contrast, such as mutual information, fails less badly, but
it still has little to lock onto in a high-b image whose edges are faint.

FSL eddy instead predicts what each volume should look like from all the others and
registers each volume to its own prediction {cite:p}`andersson2016`. The predictor is a
**Gaussian process**: a smooth guess of the signal for any gradient direction, learned from
how the signal varies across the neighboring directions that were measured. The prediction
has the contrast of the volume being corrected, so the only difference left between them is
geometric. That is why eddy needs a full set of directions and works better with more of
them. The version below does the same with a tensor model as the predictor
([Chapter 15](../04-modeling/15-signal-representations.md)): fit the tensor to the current
volumes, predict each volume from the fit, register each volume to its prediction, and
repeat three times, each round starting from better-aligned volumes.

Head motion is corrected in the same step in practice, and for motion the b-vectors must be
rotated along with the images; [Chapter 12](./12-motion-and-dropout.md) shows why. The eddy
shear and stretch are not rotations and need no b-vector change.

```{code-cell} python
:tags: [hide-input]
gtab = gradient_table(bvals, bvecs=bvecs)
corrected = distorted.copy()
estimated = np.zeros((len(bvals), 3))
V_SHOW = picks[1][0]  # the shear-dominated b = 2000 volume, followed through the iterations
preds_show, regs_show = [], []
for iteration in range(3):
    # predict every volume from a tensor fit of the current estimate, then register each
    # acquired volume to its own prediction with the three-parameter eddy model
    fit = TensorModel(gtab, fit_method="WLS", return_S0_hat=True).fit(corrected, mask=mask)
    pred = fit.predict(gtab, S0=fit.S0_hat)
    for v in range(len(bvals)):
        if bvals[v] == 0:
            continue
        estimated[v], corrected[..., v] = synth.register_pe_affine(distorted[..., v], pred[..., v], mask=mask)
    preds_show.append(pred[..., V_SHOW] * mask)
    regs_show.append(corrected[..., V_SHOW].copy())

def edge_error(s):
    rim = (mask & ~np.roll(mask, 2, axis=0)) | (mask & ~np.roll(mask, -2, axis=0))
    return np.abs(s - series)[rim][:, bvals > 0].mean()

# the naive alternative: register every diffusion-weighted volume straight to the b=0 image
to_b0 = distorted.copy()
for v in np.flatnonzero(bvals > 0):
    _, to_b0[..., v] = synth.register_pe_affine(distorted[..., v], series[..., 0], mask=mask)

errors = {"uncorrected": edge_error(distorted), "registered to b=0": edge_error(to_b0), "registered to prediction": edge_error(corrected)}
print("mean absolute error at the anterior/posterior brain edges, diffusion-weighted volumes:")
for name, e in errors.items():
    print(f"  {name:>25}: {e:.4f}")
```

```{code-cell} python
:tags: [hide-input]
fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), gridspec_kw=dict(width_ratios=[1, 1, 1.3]))
show_image(axes[0], series[..., 0], "b = 0 (the naive reference)", vmin=0, vmax=np.percentile(series[..., 0][mask], 99))
show_image(axes[1], series[..., V_SHOW], "b = 2000, perfectly aligned", vmin=0, vmax=np.percentile(series[..., V_SHOW][mask], 99))
axes[2].barh(list(errors)[::-1], list(errors.values())[::-1], color=[PALETTE[0], PALETTE[7], INK["secondary"]])
axes[2].set(xlabel="mean absolute error at the brain edge", title="edge error, diffusion-weighted volumes")
axes[2].grid(axis="y", visible=False)
fig.tight_layout()
```

The two images on the left are aligned exactly; they differ only in contrast. The fluid in
the ventricles and around the brain is bright at b=0 and nearly black at b = 2000. A
registration to the b=0 image tries to make them match anyway, and the bars show the
result: with the toy's sum-of-squares metric, the edge error after registering to b=0 is
larger than with no correction at all. Registering each volume to its own prediction
reduces it to less than half.

The prediction does not need to be good at the start. Here is the volume dominated by the
left-right gradient through the three rounds:

```{code-cell} python
:tags: [hide-input]
fig, axes = plt.subplots(2, 4, figsize=(10, 5.6))
top = np.percentile(series[..., V_SHOW][mask], 99)
acq = distorted[..., V_SHOW]
diff_kw = dict(kind="diff", vmin=-0.3 * top, vmax=0.3 * top)
show_image(axes[0, 0], acq, "acquired (distorted)", vmin=0, vmax=top)
show_image(axes[1, 0], (acq - preds_show[0]) * mask, "acquired − prediction 1", **diff_kw)
for i, (p, r) in enumerate(zip(preds_show, regs_show)):
    show_image(axes[0, i + 1], p, f"prediction, round {i + 1}", vmin=0, vmax=top)
    show_image(axes[1, i + 1], (r - p) * mask, f"registered − prediction, round {i + 1}", **diff_kw)
for ax in axes.flat:
    ax.contour(mask, levels=[0.5], colors=[PALETTE[1]], linewidths=0.6)
fig.tight_layout()
truth_v = series[..., V_SHOW]
print(f"before registration: mean |acquired - prediction| {np.abs(acq - preds_show[0])[mask].mean():.4f}")
for i, (p, r) in enumerate(zip(preds_show, regs_show)):
    print(f"round {i + 1}: mean |registered - prediction| {np.abs(r - p)[mask].mean():.4f}; "
          f"prediction vs truth {np.abs(p - truth_v)[mask].mean():.4f}; registered vs truth {np.abs(r - truth_v)[mask].mean():.4f}")
```

The top row is the acquired volume and the prediction made for it in each round. The bottom
row compares them: first the acquired volume minus the round-1 prediction, then, for each
round, the volume after registration minus that round's prediction (red where the volume is
brighter than the prediction, blue where it is darker). In round 1 the tensor is fitted to
the distorted volumes, so the prediction is an average of misaligned brains, but it already
has the right contrast, and because the volumes are displaced in different directions, the
average sits close to the true position. Before registration the difference is strong red
and blue banding at the left and right sides of the brain, where the shear moved tissue
farthest. Registering to the round-1 prediction removes most of it; each later round fits
the tensor to better-aligned volumes, so the prediction sharpens and the registration
improves. In numbers, the mean difference falls from 0.0103 before registration to 0.0077,
0.0057, and 0.0055 after rounds 1 to 3, and the prediction itself moves closer to the true
volume (from 0.0097 to 0.0070).

How well did the registration recover each volume's eddy parameters? Each point is one
diffusion-weighted volume; a perfect estimate falls on the diagonal line:

```{code-cell} python
:tags: [hide-input]
w = 0.03 * bvals / 1000
truth_params = np.column_stack([w * bvecs[:, 1], w * bvecs[:, 0], 4.0 * w * bvecs[:, 2]])  # shear, stretch, shift in the 2-D model
names = ["shear (voxels of shift\nper voxel left-right)", "stretch (voxels of shift\nper voxel anterior-posterior)", "shift (voxels)"]
dw = bvals > 0
fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
for j, (ax, name) in enumerate(zip(axes, names)):
    lim = 1.15 * np.abs(truth_params[dw, j]).max()
    ax.plot([-lim, lim], [-lim, lim], color=INK["secondary"], lw=1, ls="--")
    for b, colr in [(1000, PALETTE[0]), (2000, PALETTE[1])]:
        sel = bvals == b
        ax.scatter(truth_params[sel, j], estimated[sel, j], s=18, color=colr, label=f"b = {b}")
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal", xlabel="true", ylabel="estimated", title=name)
axes[0].legend(loc="upper left")
fig.tight_layout()
offset = (estimated - truth_params)[dw].mean(axis=0)
print(f"mean (estimated - true): shear {offset[0]:+.4f}, stretch {offset[1]:+.4f}, shift {offset[2]:+.3f} voxels")
```

For the shear and the stretch the points lie close to the diagonal in both shells: the
registration recovers each volume's pattern with small errors. The shift points follow the
diagonal's slope but sit below it by a nearly constant amount. Every gradient direction in
this scheme has a positive slice component, so every volume was shifted the same way; the
prediction, built from those volumes, is shifted too, and registering to it cannot see the
part of the shift that all volumes share. A shift common to the whole series looks exactly
like the head having moved. FSL eddy can separate the two with its *second-level model*
(`--slm`), which ties each volume's eddy parameters to its gradient direction
{cite:p}`andersson2016`; this toy correction does not do that. The default is no such
model, and the scheme here, with every direction in one hemisphere, is exactly the case in
which FSL's documentation recommends turning it on (`--slm=linear`), as it does for schemes
with few directions. The shifts here are a fraction of a voxel, so the effect on the maps
below is small.

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
  opposite polarity and chooses their timing so that eddy fields decaying with one chosen
  time constant cancel at the readout {cite:p}`reese2003`. Eddy currents that decay faster
  or slower than that are only partly cancelled, so the correction step is still needed,
  and the sequence costs a longer TE. Many vendors offer it; most current high-b protocols
  use the single-refocused sequence and rely on correction.
- **A full set of well-distributed directions** makes the prediction-based correction
  work; a scheme with few directions or with all directions in one hemisphere gives the
  Gaussian process little to work with and cannot tell a shift common to all volumes from
  head motion (for such data FSL recommends `--slm=linear`, above). Spreading the
  directions over the whole sphere, which costs nothing for the diffusion models, helps
  eddy.
- **Interleave the shells and spread the b=0 volumes through the series.** Long runs of
  strong gradients warm the gradient coils, and the eddy fields and the overall signal
  level drift as they do. With the shells interleaved, that drift is shared by every shell
  instead of being mistaken for a difference between b-values, and b=0 volumes throughout
  the run let a pipeline measure and remove the signal drift.
- **Save the phase**: the eddy field shows up in each volume's phase image as a ramp, which
  gives a direct check on the correction (the pending `eddy` dataset above will show it).

## Further reading

The twice-refocused sequence {cite:p}`reese2003`, the integrated eddy and motion
correction of FSL eddy {cite:p}`andersson2016`, and why the b-matrix must be rotated
{cite:p}`leemans2009`.
