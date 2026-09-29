---
title: "10. Susceptibility distortion"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** a synthetic series on the packaged 2 mm slice with a synthetic field ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
- **`sdc-pair`** (pending): AP/PA pairs for two source anatomies, one with an atlas field and one with a measured field, plus synthetic gradient-echo fieldmaps ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-sdc-pair)).
- **`truth`** (pending): the 27 analytic ground-truth maps and the true fiber orientations ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-truth), [Appendix E](../appendices/e-truth-map-catalogue.md)).

Pipeline-tier datasets are simulated offline by TRXScan ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)) and are marked *pending* until their release; the figures that need them say so where they will appear.
:::

## Learning goals

After this chapter you can:

- predict where and how far EPI images are displaced from a field map and the readout time
- recognize the signature of the artifact in a blip-up/blip-down pair
- correct the distortion when the field is known and explain how it is estimated when it
  is not
- state which acquisition choices reduce it and which metadata the correction requires

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from dwibook import phantoms, presets, schemes, synth
from dwibook.plotting import INK, PALETTE, animate, set_style, show_image

set_style()
```

## The physics

The static field is not uniform inside the head. Tissue, bone, and air have different
magnetic susceptibilities, and near the interfaces between them, above the sinuses, next to
the ear canals, and at the temporal poles, the field deviates from its nominal value by up
to a few hundred hertz at 3 T. Spins in those regions precess at a shifted frequency.

Diffusion images are acquired with **echo-planar imaging** (EPI), which reads a whole
slice's k-space after a single excitation ([Chapter 2](../01-mri-physics/02-spatial-encoding-kspace.md)). The readout gradient sweeps back
and forth, one k-space line per sweep, and between sweeps a brief gradient pulse, a
**blip**, steps to the next line. The axis the blips step along is the **phase-encode**
axis. Along it, position is not read from a frequency within one line; it is inferred from
how fast each spin's phase advances from one line to the next. A spin that precesses a
little too fast gains a little extra phase at every line, exactly as if it sat farther
along the axis, and the reconstruction puts its signal there. Because the lines follow one
another for the whole length of the readout, tens of milliseconds, the small extra phase
adds up to a large displacement. Along the other axis each line takes well under a
millisecond, and the same offset moves signal by a small fraction of a voxel.

The size of the error is the frequency offset times the total readout time of the EPI train:

$$\text{displacement (voxels)} = \Delta f\ (\text{Hz}) \times \text{TotalReadoutTime}\ (\text{s}).$$

For the HBCD readout of 92 ms, 100 Hz moves signal by nine voxels, close to 2 cm at the
2 mm voxels used on this page. The direction of the displacement follows the sign of the
field offset and the polarity of the phase-encode blips, which set whether the lines are
traversed front to back (AP) or back to front (PA): reversing the blips reverses the
displacement. Where the displacement varies across neighboring voxels, signal from several
true locations lands in one voxel (pile-up, bright) or is spread over several (stretching,
dark), so the intensity is altered as well as the geometry. Anatomical images such as the
T1-weighted image are acquired one k-space line per excitation, so the same field offset
moves their signal by only a fraction of a voxel, which is why diffusion data do not line
up with them without correction.

## The artifact-free reference

The toy demonstration applies a plausible synthetic field to the 2 mm slice: a focus of
120 Hz behind the frontal sinuses plus a smooth gradient across the brain. The field is
not a physical simulation; its shape and magnitude are typical of measured 3 T fieldmaps.
The undistorted series is the reference.

```{code-cell} python
:tags: [hide-input]
t = phantoms.brain_slice()
mask = t["mask"]
bvals, bvecs = schemes.single_shell(1000, 6, n_b0=2)
series = synth.synthetic_dwi(t, bvals, bvecs)
fmap = synth.synthetic_fieldmap(mask, t["voxel_mm"], amplitude_hz=120.0)
TRT = presets.READOUT_HBCD_MS / 1000  # s
shift = fmap * TRT  # voxels, positive = toward posterior for the AP polarity

ap = synth.displace_along_pe(series, shift)
pa = synth.displace_along_pe(series, -shift)
print(f"field: max {fmap.max():.0f} Hz; displacement up to {np.abs(shift).max():.1f} voxels ({np.abs(shift).max() * t['voxel_mm']:.0f} mm)")
```

## See it: pile-up and stretching

The animation draws a grid of evenly spaced horizontal lines on the true anatomy (the
phase-encode axis runs top to bottom, front of the head at the top) and then draws each
line where the reconstruction puts it, as the field around the frontal sinuses is turned up
from zero to its full strength (a peak of 131 Hz, printed above). Watch the front of the
brain. With AP blips (left) the lines there are pushed back, toward the bottom of the
image; with PA blips (middle) the same lines are pushed forward. Where neighboring lines are pushed by different amounts they crowd together
or spread apart, and the image follows them: signal piles up into a bright band where the
lines crowd and thins out where they spread. The two polarities swap which is which. On the
right, the same distortion is applied to a brain of uniform brightness and plotted along
the dashed column, so that the only changes in intensity are the ones the distortion makes:
above 1 where signal piles up, below 1 where it is stretched.

```{code-cell} python
:tags: [hide-input]
b0_img = series[..., 0]
uniform = mask.astype(float)   # a brain of uniform brightness, for the profile
col = int(np.argmax(fmap.max(axis=0) * mask.any(axis=0)))       # the column through the field focus
rows_in = np.flatnonzero(mask[:, col])
grid_rows = np.arange(4, mask.shape[0], 6)
cols_in = np.flatnonzero(mask.any(axis=0))
peaks = np.concatenate([np.zeros(3), np.linspace(0, 1, 17), np.ones(6)]) * fmap.max()   # field peak, Hz
fig, (ax_ap, ax_pa, ax_pr) = plt.subplots(1, 3, figsize=(10.5, 3.9), gridspec_kw={"width_ratios": [1, 1, 1.25]})
y_prof = np.arange(rows_in.min() - 4, rows_in.min() + 45)

def sdc_frame(i):
    s = shift * peaks[i] / fmap.max()
    for ax, sign, name in [(ax_ap, 1, "AP blips"), (ax_pa, -1, "PA blips")]:
        ax.clear()
        ax.imshow(synth.displace_along_pe(b0_img, sign * s), cmap="gray", vmin=0, vmax=0.5)
        for r in grid_rows:
            ax.plot(cols_in, r + sign * s[r, cols_in], color=PALETTE[3], lw=0.8)
        ax.axvline(col, color=PALETTE[0], lw=0.8, ls="--")
        ax.set_axis_off(); ax.set_title(name)
    ax_pr.clear()
    ax_pr.plot(y_prof, uniform[y_prof, col], color=INK["secondary"], lw=1.5, ls=":", label="true")
    ax_pr.plot(y_prof, synth.displace_along_pe(uniform, s)[y_prof, col], color=PALETTE[1], label="AP")
    ax_pr.plot(y_prof, synth.displace_along_pe(uniform, -s)[y_prof, col], color=PALETTE[0], label="PA")
    ax_pr.set(ylim=(0, 4), xlabel="row (front of head → back)", ylabel="intensity (true = 1)",
              title=f"uniform tissue, dashed column; field peak {peaks[i]:.0f} Hz")
    ax_pr.legend(loc="upper right")
    fig.tight_layout()

animate(fig, sdc_frame, range(len(peaks)), fps=5, width=760, dpi=70,
        alt="two axial brain slices with evenly spaced horizontal grid lines, one reconstructed with AP phase-encode blips and one with PA blips, as the field focus behind the frontal sinuses grows from 0 to its full peak of about 130 Hz. In the AP image the frontal grid lines move toward the back of the head, in the PA image toward the front; where lines crowd together the image brightens into a band, where they spread apart it darkens, and the two polarities swap which region is bright and which is dark. A third panel plots, for a brain of uniform brightness, the intensity along one front-to-back column: the flat true profile, and AP and PA profiles that develop a peak and a dip at opposite places")
```

## See it: the blip-up/blip-down pair

The full-strength result, for one b = 0 volume. The first panel is the field, with a second
scale on its color bar giving the displacement it causes in voxels. The yellow outline on
the other three panels is the true edge of the brain, which is where the brain appears in
the undistorted T1-weighted image. Compare the front of the brain with the outline in each
distorted panel: in the AP image the frontal edge falls behind the outline, in the PA image
it bulges beyond it.

```{code-cell} python
:tags: [hide-input]
def outline(ax):
    ax.contour(mask.astype(float), levels=[0.5], colors=[PALETTE[3]], linewidths=0.9)

fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.3))
im = axes[0].imshow(np.where(mask, fmap, np.nan), cmap="RdBu_r", vmin=-120, vmax=120); axes[0].set_axis_off(); axes[0].set_title("field offset")
cb = fig.colorbar(im, ax=axes[0], shrink=0.75)
cb.set_label("Hz")
cb.ax.yaxis.set_label_position("right")
sec = cb.ax.secondary_yaxis("left", functions=(lambda hz: hz * TRT, lambda vox: vox / TRT))
sec.set_ylabel("displacement (voxels)")
show_image(axes[1], series[..., 0], "undistorted (reference)", vmin=0, vmax=0.5); outline(axes[1])
show_image(axes[2], ap[..., 0], "AP polarity: frontal signal pushed back", vmin=0, vmax=0.5); outline(axes[2])
show_image(axes[3], pa[..., 0], "PA polarity: pushed forward", vmin=0, vmax=0.5); outline(axes[3])
fig.tight_layout()
```

The two polarities distort the same region in opposite directions, and the intensity
changes with the geometry: where the displacement converges the signal piles up into a
bright band, where it diverges the tissue is stretched and dimmed. A single distorted image
cannot be corrected from its own content, because a compressed region and a genuinely small
region look alike. The pair can: the true image lies between the two, and the field that
maps one onto the other is the one that produced both.

## Correction step by step

Three approaches are in use. They differ in how they find out the displacement, and all end
with the same operation: move each voxel's signal back along the phase-encode axis by its
displacement, and rescale its intensity by how much that part of the image was stretched or
compressed, to undo pile-up. The diagram lays them side by side.

```{code-cell} python
:tags: [hide-input]
fig, ax = plt.subplots(figsize=(10, 4.1))
ax.set(xlim=(0, 10), ylim=(0.35, 4.55)); ax.set_axis_off()

def box(x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08", fc=color, ec=INK["secondary"], lw=0.8, alpha=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, color=INK["primary"], wrap=True)

def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=INK["secondary"], lw=1))

rows = [
    (3.2, "measured fieldmap\n(gradient echo at two echo times)", "phase difference ÷ echo-time difference\n→ unwrap, register to the EPI\n→ field (Hz) × readout time"),
    (1.85, "blip-up/blip-down pair\n(a few b = 0 volumes, AP and PA)", "find the smooth field that makes the\nAP and PA images agree\n(FSL topup, TORTOISE DRBUDDI)"),
    (0.5, "b = 0 image + T1-weighted image\n(no extra acquisition)", "warp the b = 0 image along phase-encode\nuntil it matches the T1-weighted image\n(fieldmap-less, SyN)"),
]
for y, src, how in rows:
    box(0.1, y, 2.6, 0.95, src, "#dbe8f8")
    box(3.2, y, 3.2, 0.95, how, "#f7e3cf")
    arrow(2.75, y + 0.47, 3.15, y + 0.47)
    arrow(6.45, y + 0.47, 7.05, 2.3)
box(7.1, 1.6, 2.8, 1.4, "displacement map (voxels)\n↓\nmove signal back along\nphase-encode, rescale by\nthe local stretch", "#d8f0e5")
ax.text(1.4, 4.35, "what is acquired", ha="center", fontsize=9, color=INK["secondary"])
ax.text(4.8, 4.35, "how the displacement is found", ha="center", fontsize=9, color=INK["secondary"])
ax.text(8.5, 4.35, "the shared last step", ha="center", fontsize=9, color=INK["secondary"])
fig.tight_layout()
```

- **Measured fieldmap** {cite:p}`jezzard1995`: a short gradient-echo acquisition at two
  echo times. Between the two echoes each voxel's phase advances in proportion to its field
  offset, so the phase difference divided by the echo-time difference is the field (TRXScan
  writes such a fieldmap with its `--gre-out` option). Phase is only known up to whole
  turns, so wherever it passes 180° it jumps to −180°, and the jumps must first be removed,
  which is called **unwrapping**. The fieldmap must also be registered (aligned) to the EPI
  data, since it is a separate scan. This works with a single polarity.
- **Blip-up/blip-down estimation** (FSL topup, TORTOISE DRBUDDI): the field is estimated
  as the smooth displacement that makes the two polarities agree {cite:p}`andersson2003`.
  The corrected image combines both, so signal lost to stretching in one polarity is
  recovered from the other. This is the method of most current pipelines and needs only a
  few extra b=0 volumes of the opposite polarity.
- **Fieldmap-less** (registration to the anatomical image, with a nonlinear registration
  method such as SyN): the b=0 image is warped to match the T1- or T2-weighted image, with
  the warp allowed only along the phase-encode axis. It works when no fieldmap or
  reverse-polarity data were acquired and is the least accurate, because the two images
  have different contrast.

With the field known, the correction is the inverse of the forward model. With both
polarities, one more step helps. Each corrected polarity is good where it was stretched
during acquisition, since stretched signal was spread out but kept, and poor where it was
compressed, since compressed signal was summed and cannot be separated again. And where one
polarity was compressed, the other was stretched. So the two are averaged with weights that
trust each polarity where it was stretched, not where it was compressed. The weight is the
local stretch factor, called the **Jacobian** of the displacement: above 1 where the image
was stretched, below 1 where it was compressed.

The printed errors compare four versions with the undistorted series. The figure shows the
AP image before and after correction, the share of the combined image taken from the AP
polarity (yellow where AP was stretched and is trusted, dark where PA is trusted), the
combined result, and what remains of the error.

```{code-cell} python
:tags: [hide-input]
ap_corr = synth.undistort_along_pe(ap, shift)
pa_corr = synth.undistort_along_pe(pa, -shift)
# combine the two polarities weighted by the local stretch of each: where AP was compressed
# (and lost information), PA was stretched (and kept it), and vice versa; topup does the same
w_ap = np.clip(1.0 + np.gradient(shift, axis=0), 0.05, None)[..., None]
w_pa = np.clip(1.0 - np.gradient(shift, axis=0), 0.05, None)[..., None]
combined = (w_ap * ap_corr + w_pa * pa_corr) / (w_ap + w_pa)

def err(s):
    return np.abs(s - series)[mask].mean()

plain_avg = (ap_corr + pa_corr) / 2
print("mean absolute error inside the brain, all volumes (b = 0 white matter signal is about 0.2):")
for label, s in [("AP distorted", ap), ("AP corrected from the known field", ap_corr), ("PA corrected from the known field", pa_corr),
                 ("AP and PA, plain average", plain_avg), ("AP and PA, weighted toward the stretched one", combined)]:
    print(f"  {label:>45}: {err(s):.4f}")

fig, axes = plt.subplots(1, 5, figsize=(14, 3.2))
show_image(axes[0], ap[..., 0], "AP distorted", vmin=0, vmax=0.5)
show_image(axes[1], ap_corr[..., 0], "AP corrected", vmin=0, vmax=0.5)
show_image(axes[2], np.where(mask, (w_ap / (w_ap + w_pa))[..., 0], np.nan), "share taken from AP", kind="scalar", vmin=0, vmax=1)
fig.colorbar(axes[2].images[0], ax=axes[2], shrink=0.7)
show_image(axes[3], combined[..., 0], "AP + PA corrected, combined", vmin=0, vmax=0.5)
show_image(axes[4], np.abs(combined[..., 0] - series[..., 0]) * mask, "remaining error", vmin=0, vmax=0.1)
fig.tight_layout()
```

The first line is the error of doing nothing; correcting either polarity alone removes most
of it, and the weighted combination is the best of the five. The remaining error sits at
tissue boundaries, where any resampling is least accurate, and is largest where the
displacement converged most: there, several voxels of tissue were summed into one during
acquisition, and no resampling can separate them again. The share map shows how the
weighting handles this: just in front of the frontal focus, where AP was stretched, the
combination leans on AP, and just behind it, where AP piled up, it leans on PA. A plain
average mixes the good estimate with the bad one everywhere.

What happens when the correction is applied with the wrong sign, as when the
`PhaseEncodingDirection` in the image metadata is recorded backward, is shown next: the
software moves the signal the same way the field did, instead of back, and the displacement
doubles. The yellow outline is again the true edge of the brain.

```{code-cell} python
:tags: [hide-input]
wrong = synth.undistort_along_pe(ap, -shift)
fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
for ax, img, title in [(axes[0], ap, "AP distorted"), (axes[1], ap_corr, "corrected, right sign"), (axes[2], wrong, "corrected, wrong sign")]:
    show_image(ax, img[..., 0], title, vmin=0, vmax=0.5); outline(ax)
fig.tight_layout()
print(f"mean absolute error: distorted {err(ap):.4f}; right sign {err(ap_corr):.4f}; wrong sign {err(wrong):.4f}")
```

## Residual error versus truth

Because the same field displaces every volume of the series equally, the tensor fit itself
is not much affected: FA and MD are computed from ratios between volumes of the same voxel,
and those ratios survive a displacement that is common to all volumes. What is wrong is
where the values are. The FA map of the distorted series is a distorted FA map. The first
panel is the reference FA; the other two are the error, FA minus reference FA, for the
distorted and the corrected series, with the true outline of the white matter drawn in
black.

```{code-cell} python
:tags: [hide-input]
ref_fit = synth.dti_maps(series, bvals, bvecs, mask=mask)
fits = {label: synth.dti_maps(s, bvals, bvecs, mask=mask) for label, s in [("AP distorted", ap), ("corrected", combined)]}
wm_ref = t["wm"] > 0.5
fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
show_image(axes[0], ref_fit["fa"], "FA, reference", kind="scalar", vmin=0, vmax=0.9)
for ax, (label, m) in zip(axes[1:], fits.items()):
    show_image(ax, np.where(mask, m["fa"] - ref_fit["fa"], 0), f"FA error, {label}", kind="diff", vmin=-0.6, vmax=0.6)
for ax in axes[1:]:
    ax.contour(wm_ref.astype(float), levels=[0.5], colors=[INK["primary"]], linewidths=0.5)
fig.tight_layout()
fig.colorbar(axes[2].images[0], ax=axes[1:], shrink=0.8, label="FA − reference FA")
for label, m in fits.items():
    print(f"{label:>14}: mean absolute FA error in WM {np.abs(m['fa'] - ref_fit['fa'])[t['wm'] > 0.9].mean():.3f}")
```

The errors of the distorted series come in red and blue pairs along the edges of the white
matter, most of all at the front of the brain: white matter values have landed on gray
matter (red, FA too high there) and gray matter values on white matter (blue). The mean
error in white matter of 0.21 is therefore almost all misplacement, a correct value in the
wrong voxel, rather than a wrong value, with some blending where pile-up summed several
voxels into one. After correction the error drops to 0.045, most of
it a thin fringe at tissue edges left by resampling.

Two caveats limit that reassurance. Eddy currents ([Chapter 11](./11-eddy-currents.md)) add a displacement that
differs per volume, which a correction estimated from the b=0 images does not remove.
And any analysis that compares the diffusion data with another image, an atlas, a
segmentation, a tractography target, depends on the geometry being right.

## Measure it: the simulated datasets

:::{admonition} Simulated dataset pending
:class: note
This section will load the `sdc-pair` dataset (AP/PA pairs with the atlas field for
sub-0001a and the measured field for sub-60501, plus synthetic GRE fieldmaps) and the
`truth` maps, run FSL topup and a fieldmap-based correction from the pipeline's precomputed
outputs, and compare the estimated field with the field TRXScan used.
:::

## What acquisition choices reduce it

- **Shorten the readout.** The displacement is proportional to the total readout time, so
  in-plane acceleration reduces it directly: it skips lines, so each remaining step
  between lines covers more of k-space in the same time ([Chapter 7](../02-diffusion-encoding/07-acquisition-parameters.md)). Partial Fourier
  shortens the echo time but not the displacement, because the lines it keeps are as far
  apart in time as before.
- **Acquire the opposite polarity.** A few b=0 volumes with reversed blips are enough for
  topup; a full second copy of the scheme also doubles the directions.
- **Record the metadata.** `PhaseEncodingDirection` and `TotalReadoutTime` in the JSON
  sidecar are what every correction reads; a wrong sign applies the correction backward and
  doubles the displacement, as the wrong-sign panel above shows.
- **Shim.** Better shimming lowers the field offsets at the source; it is set at the
  scanner and rarely revisited.

## Further reading

Fieldmap-based correction {cite:p}`jezzard1995`, blip-up/blip-down estimation
{cite:p}`andersson2003`, and its integration with eddy and motion correction
{cite:p}`andersson2016`.
