---
title: "14. Remaining artifacts and the assembled pipeline"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** a synthetic series on the packaged 2 mm slice with several artifacts applied together ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
- **`kitchen-sink`** (pending): every artifact on at once, corrected end to end by QSIPrep ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-kitchen-sink)).
- **`truth`** (pending): the 27 analytic ground-truth maps and the true fiber orientations ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-truth), [Appendix E](../appendices/e-truth-map-catalogue.md)).

Pipeline-tier datasets are simulated offline by TRXScan ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)) and are marked *pending* until their release; the figures that need them say so where they will appear.
:::

## Learning goals

After this chapter you can:

- recognize Nyquist ghosts, k-space spikes, and receive-field bias, and say what to do
  about each
- order the preprocessing steps of Chapters [8](./08-noise.md) through [13](./13-gradient-nonlinearity.md) and justify the order
- explain why the geometric corrections must be composed into a single resampling
- read a pipeline's quality report and decide whether a dataset is usable

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt

from dwibook import kspace, phantoms, presets, schemes, synth
from dwibook.plotting import PALETTE, set_style, show_image

set_style()
img = phantoms.brain_image()
mask = phantoms.brain_slice()["mask"]
```

## Nyquist ghosting

EPI reads alternate lines of k-space in opposite directions ([Chapter 2](../01-mri-physics/02-spatial-encoding-kspace.md)). Any difference
between the two, in timing, in eddy currents, or in the gradient waveform, gives the odd
and even lines a relative phase or shift. In the image that appears as a faint copy of the
object displaced by half the field of view along the phase-encode axis. Scanners calibrate
it out with a reference scan, and a residual ghost of a percent or two is normal. A larger
ghost indicates a hardware or calibration problem. It cannot be removed after the fact
without the raw k-space data, so the remedy is to detect it (a signal in the background
outside the head, aligned with the head) and to exclude the affected volumes or rescan.

```{code-cell} python
:tags: [hide-input]
fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2))
show_image(axes[0], img, "no ghost", vmin=0, vmax=0.5)
for ax, ph in zip(axes[1:], [0.15, 0.6]):
    g = np.abs(synth.nyquist_ghost(img, ph))
    show_image(ax, g, f"odd/even phase error {np.degrees(ph):.0f}°: ghost {g[~mask].max() / img.max():.0%} of peak", vmin=0, vmax=0.5)
fig.tight_layout()
```

## k-space spikes

A discharge in the scanner room or a loose connection during the readout adds one bright
sample to k-space. Because every k-space sample is a plane wave across the whole image, a
single spike appears as a stripe pattern (a "herringbone" or "corduroy" artifact) covering
the entire slice, with a spacing set by the spike's position in k-space. It affects only
the volume and slice in which it occurred. With raw data the sample can be replaced; with
magnitude images the slice is treated as an outlier and replaced from the model
([Chapter 12](./12-motion-and-dropout.md)), which is what the outlier detection of eddy and SHORELine does with it.

```{code-cell} python
:tags: [hide-input]
fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2))
show_image(axes[0], img, "clean", vmin=0, vmax=0.5)
show_image(axes[1], np.abs(synth.kspace_spike(img, 70, 84, 1.0)), "spike far from the k-space center", vmin=0, vmax=0.5)
show_image(axes[2], np.abs(synth.kspace_spike(img, 66, 66, 1.0)), "spike near the center", vmin=0, vmax=0.5)
fig.tight_layout()
```

## Receive-field bias

The sensitivity of the coil array varies smoothly across the head ([Chapter 3](../01-mri-physics/03-reconstruction.md)), so the
images are brighter near the coils and darker in the center. This does not affect
diffusion measures, which are ratios between volumes of the same voxel, but it affects
everything that uses intensity across voxels: brain masking, tissue segmentation of the
b=0 image, and the response-function estimation of spherical deconvolution. Pipelines
estimate the field from the b=0 image (N4) and divide it out. Because the correction is a
smooth multiplicative field applied identically to all volumes, its placement in the
pipeline does not matter.

## The order of operations

The corrections of the preceding chapters have an order, and the order is not arbitrary:

1. **Denoising** ([Chapter 8](./08-noise.md)) first, because it relies on the noise being independent
   between voxels and volumes, and every later step (interpolation, averaging) correlates
   it.
2. **Unringing** ([Chapter 9](./09-gibbs-ringing.md)) second, because it operates on the acquired grid and the
   ripple structure it relies on is destroyed by resampling.
3. **The geometric corrections together**: susceptibility ([Chapter 10](./10-susceptibility-distortion.md)), eddy currents and
   motion (Chapters [11](./11-eddy-currents.md) and [12](./12-motion-and-dropout.md)), and gradient nonlinearity ([Chapter 13](./13-gradient-nonlinearity.md)). Each is a
   displacement field or a transform; they are composed into one map and the data are
   resampled once. Outlier detection and replacement happen inside this step, because
   the model that predicts each volume is fitted to the aligned data.
4. **Bias-field correction and masking** on the corrected b=0 image.
5. **The gradient-deviation image and the rotated b-vectors** are written out with the
   data for the model fit.

The reason for composing the transforms is that every resampling blurs. Applying two
corrections in sequence interpolates twice; composing them interpolates once.

```{code-cell} python
:tags: [hide-input]
fmap = synth.synthetic_fieldmap(mask, 2.0, amplitude_hz=80.0)
sdc_shift = fmap * presets.READOUT_HBCD_MS / 1000
yy, xx = np.indices(mask.shape, dtype=float)
eddy_shear = 0.04 * (xx - 63.5)  # a b = 2000 volume with a gradient along the columns
acquired = synth.displace_along_pe(synth.displace_along_pe(img, sdc_shift), eddy_shear)

# Pipelines resample with linear (trilinear) interpolation, which is used here for both routes.
sequential = synth.undistort_along_pe(synth.undistort_along_pe(acquired, eddy_shear, order=1), sdc_shift, order=1)
# composed: one map from the acquired grid to the true grid. The eddy shift acts on the
# already-distorted image, so the total shift at a true position is the susceptibility shift
# there plus the eddy shift evaluated where the susceptibility shift moved it.
y_after_sdc = yy + sdc_shift
idx = np.array([np.clip(y_after_sdc, 0, mask.shape[0] - 1), xx])
from scipy import ndimage
total_shift = sdc_shift + ndimage.map_coordinates(eddy_shear, idx, order=1, mode="nearest")
composed = synth.undistort_along_pe(acquired, total_shift, order=1)

print(f"error vs truth: two sequential resamplings {np.abs(sequential - img)[mask].mean():.4f}, one composed resampling {np.abs(composed - img)[mask].mean():.4f}")
fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2))
show_image(axes[0], acquired, "acquired: distortion + eddy shear", vmin=0, vmax=0.5)
show_image(axes[1], np.abs(sequential - img) * mask, "error, two resamplings", vmin=0, vmax=0.1)
show_image(axes[2], np.abs(composed - img) * mask, "error, one composed resampling", vmin=0, vmax=0.1)
fig.tight_layout()
```

The difference for two smooth corrections on one slice is about 10 % of the error.
It grows with the number of steps (susceptibility, eddy, motion, gradient nonlinearity,
and the final resampling to the anatomical grid would be five) and with the roughness of
the fields. The second reason to compose is that the corrections are not independent: the
eddy and motion parameters are estimated from images whose susceptibility distortion has
already been accounted for, so the tools that estimate them also apply them together.

## How the standard pipelines order these

FSL's sequence is topup for the susceptibility field, then eddy for eddy currents, motion,
outlier replacement, and the application of the topup field in one resampling
{cite:p}`andersson2016,andersson2016b`. MRtrix wraps the same tools after its own denoising
and unringing {cite:p}`tournier2019`. QSIPrep {cite:p}`cieslak2021` runs denoising,
unringing, and bias correction, then either the FSL route or the TORTOISE route (DRBUDDI
for the field, DIFFPREP for eddy and motion), and composes gradient nonlinearity into the
same transform when a coefficient file is given. The HCP pipelines add gradwarp explicitly
{cite:p}`glasser2013`. All of them write the rotated b-vectors and a quality report.

## Quality control

A pipeline's report is the first thing to read, before any map. The measures that decide
whether a dataset is usable:

| Measure | What it flags | Typical action |
|---|---|---|
| Framewise displacement per volume | between-volume motion | exclude subjects above a threshold (often 1–2 mm mean) |
| Outlier slices per volume | dropout, spikes | exclude subjects with more than a few percent of slices replaced |
| Residual distortion at the frontal lobe | wrong metadata or failed topup | check `PhaseEncodingDirection` and `TotalReadoutTime` |
| b=0 SNR in white matter | noise level | denoise; question high-b results below SNR 3 |
| Ghost level in the background | readout calibration | rescan or exclude the volume |
| Model residuals across the brain | anything not modeled | inspect the residual map |

The measures are also covariates: motion and SNR differ systematically between groups
(children, patients) and can produce apparent group differences in every diffusion
measure. Report them with the results.

## Measure it: the simulated datasets

:::{admonition} Simulated dataset pending
:class: note
This section will load the `kitchen-sink` dataset, the simulated brain with every
artifact on (noise, ringing, distortion, eddy currents, multiband dropout, gradient
nonlinearity, 8-coil GRAPPA), corrected end to end by qsiprep with the coefficient file,
and score each stage of the pipeline against the `truth` maps: the error that remains
after each step, and the error that returns if a step is skipped.
:::

## What this implies for acquisition

- **A pipeline can correct only what the acquisition recorded.** Reverse-polarity
  volumes, spread b=0 volumes, a full direction set, the phase, and the gradient
  coefficient file are acquisition-side decisions that determine which corrections are
  possible.
- **Check the metadata before processing.** Most catastrophic failures are a wrong
  phase-encode direction or readout time.
- **Plan for exclusions.** Motion and dropout thresholds remove subjects; the sample size
  should allow for it.

## Further reading

The FSL tools {cite:p}`andersson2003,andersson2016,andersson2016b`, MRtrix
{cite:p}`tournier2019`, QSIPrep {cite:p}`cieslak2021`, and the HCP pipelines
{cite:p}`glasser2013`.
