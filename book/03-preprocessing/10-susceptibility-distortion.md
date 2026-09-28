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

from dwibook import phantoms, presets, schemes, synth
from dwibook.plotting import PALETTE, set_style, show_image

set_style()
```

## The physics

The static field is not uniform inside the head. Tissue, bone, and air have different
magnetic susceptibilities, and near the interfaces between them, above the sinuses, next to
the ear canals, and at the temporal poles, the field deviates from its nominal value by up
to a few hundred hertz at 3 T. Spins in those regions precess at a shifted frequency, and
along the phase-encode axis frequency is what encodes position ([Chapter 2](../01-mri-physics/02-spatial-encoding-kspace.md)), so the
reconstruction places their signal at a shifted location.

The size of the error is the frequency offset times the total readout time of the EPI train:

$$\text{displacement (voxels)} = \Delta f\ (\text{Hz}) \times \text{TotalReadoutTime}\ (\text{s}).$$

For the HBCD readout of 92 ms, 100 Hz moves signal by nine voxels. The direction of the
displacement follows the sign of the field offset and the polarity of the phase-encode
blips: reversing the blips reverses the displacement. Where the displacement varies across
neighboring voxels, signal from several true locations lands in one voxel (pile-up, bright)
or is spread over several (stretching, dark), so the intensity is altered as well as the
geometry. Anatomical images acquired with conventional readouts do not show this, which is
why diffusion data do not line up with the T1-weighted image without correction.

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

## See it: the blip-up/blip-down pair

```{code-cell} python
:tags: [hide-input]
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
im = axes[0].imshow(np.where(mask, fmap, np.nan), cmap="RdBu_r", vmin=-120, vmax=120); axes[0].set_axis_off(); axes[0].set_title("field offset (Hz)")
fig.colorbar(im, ax=axes[0], shrink=0.7)
show_image(axes[1], series[..., 0], "undistorted (reference)", vmin=0, vmax=0.5)
show_image(axes[2], ap[..., 0], "AP polarity: frontal signal pushed back", vmin=0, vmax=0.5)
show_image(axes[3], pa[..., 0], "PA polarity: pushed forward", vmin=0, vmax=0.5)
fig.tight_layout()
```

The two polarities distort the same region in opposite directions, and the intensity
changes with the geometry: where the displacement converges the signal piles up into a
bright band, where it diverges the tissue is stretched and dimmed. A single distorted image
cannot be corrected from its own content, because a compressed region and a genuinely small
region look alike. The pair can: the true image lies between the two, and the field that
maps one onto the other is the one that produced both.

## Correction step by step

Three approaches are in use, and all end with the same operation, resampling the images
along the phase-encode axis by the negative of the displacement and rescaling by the local
stretch to undo pile-up.

- **Measured fieldmap** {cite:p}`jezzard1995`: a short gradient-echo acquisition at two
  echo times gives the field from the phase difference (the `--gre-out` output). The
  displacement follows from the field and the readout time. This works with a single
  polarity but needs the fieldmap to be registered to the EPI data and unwrapped.
- **Blip-up/blip-down estimation** (FSL topup, TORTOISE DRBUDDI): the field is estimated
  as the smooth displacement that makes the two polarities agree {cite:p}`andersson2003`.
  The corrected image combines both, so signal lost to stretching in one polarity is
  recovered from the other. This is the method of most current pipelines and needs only a
  few extra b=0 volumes of the opposite polarity.
- **Fieldmap-less** (registration to the anatomical image, SyN): a nonlinear registration
  of the b=0 image to the T1- or T2-weighted image, constrained to the phase-encode axis.
  It works when no fieldmap or reverse-polarity data were acquired and is the least
  accurate.

With the field known, the correction is the inverse of the forward model:

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

print("mean absolute error inside the brain, all volumes:")
for label, s in [("AP distorted", ap), ("AP corrected from the known field", ap_corr), ("PA corrected from the known field", pa_corr), ("AP and PA combined, Jacobian-weighted", combined)]:
    print(f"  {label:>40}: {err(s):.4f}")

fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
show_image(axes[0], ap[..., 0], "AP distorted", vmin=0, vmax=0.5)
show_image(axes[1], ap_corr[..., 0], "AP corrected", vmin=0, vmax=0.5)
show_image(axes[2], combined[..., 0], "AP + PA corrected, combined", vmin=0, vmax=0.5)
show_image(axes[3], np.abs(combined[..., 0] - series[..., 0]) * mask, "residual error", vmin=0, vmax=0.1)
fig.tight_layout()
```

The residual after correction sits at tissue boundaries, where any resampling is least
accurate, and is largest where the displacement converged most: there, several voxels of
tissue were summed into one during acquisition, and no resampling can separate them again.
Combining the two corrected polarities reduces it, provided each is weighted by how much it
was stretched at that location: the region that piled up in one polarity was stretched, and
therefore preserved, in the other. A plain average helps less, because it mixes the good
estimate with the bad one everywhere.

## Residual error versus truth

Because the same field displaces every volume of the series equally, the tensor fit itself
is not much affected: FA and MD are computed from ratios between volumes of the same voxel,
and those ratios survive a displacement that is common to all volumes. What is wrong is
where the values are. The FA map of the distorted series is a distorted FA map:

```{code-cell} python
:tags: [hide-input]
ref_fit = synth.dti_maps(series, bvals, bvecs, mask=mask)
fits = {label: synth.dti_maps(s, bvals, bvecs, mask=mask) for label, s in [("AP distorted", ap), ("corrected", combined)]}
fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
show_image(axes[0], ref_fit["fa"], "FA, reference", kind="scalar", vmin=0, vmax=0.9)
for ax, (label, m) in zip(axes[1:], fits.items()):
    show_image(ax, m["fa"], f"FA, {label}", kind="scalar", vmin=0, vmax=0.9)
fig.tight_layout()
for label, m in fits.items():
    print(f"{label:>14}: FA error in WM {np.abs(m['fa'] - ref_fit['fa'])[t['wm'] > 0.9].mean():.3f}")
```

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
  partial Fourier and in-plane acceleration reduce it directly ([Chapter 7](../02-diffusion-encoding/07-acquisition-parameters.md)).
- **Acquire the opposite polarity.** A few b=0 volumes with reversed blips are enough for
  topup; a full second copy of the scheme also doubles the directions.
- **Record the metadata.** `PhaseEncodingDirection` and `TotalReadoutTime` in the JSON
  sidecar are what every correction reads; a wrong sign applies the correction backward and
  doubles the distortion.
- **Shim.** Better shimming lowers the field offsets at the source; it is set at the
  scanner and rarely revisited.

## Further reading

Fieldmap-based correction {cite:p}`jezzard1995`, blip-up/blip-down estimation
{cite:p}`andersson2003`, and its integration with eddy and motion correction
{cite:p}`andersson2016`.
