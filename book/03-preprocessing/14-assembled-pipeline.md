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

The chapter has two parts. The first covers three artifacts that have no chapter of their
own: Nyquist ghosts, k-space spikes, and receive-field bias. The second assembles the
corrections of Chapters [8](./08-noise.md) through [13](./13-gradient-nonlinearity.md) into one pipeline and shows how to read its quality
report.

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from scipy import ndimage

from dwibook import kspace, phantoms, presets, schemes, synth
from dwibook.plotting import INK, PALETTE, set_style, show_image, show_kspace

set_style()
img = phantoms.brain_image()
tissue = phantoms.brain_slice()
mask = tissue["mask"]
```

## Nyquist ghosting

EPI reads alternate lines of k-space in opposite directions ([Chapter 2](../01-mri-physics/02-spatial-encoding-kspace.md)). Any difference
between the two, in timing, in eddy currents, or in the gradient waveform, gives the odd
and even lines a relative phase or shift. In the image that appears as a faint copy of the
object displaced by half the field of view along the phase-encode axis. Scanners calibrate
it out with a reference scan, and a residual ghost of a percent or two is normal. A larger
ghost indicates a hardware or calibration problem. It cannot be removed after the fact
without the raw k-space data, so the remedy is to detect it and to exclude the affected
volumes or rescan.

To detect it, look outside the head, half a field of view away along the phase-encode
direction (up and down in these images). At normal display settings a small ghost is
invisible, so the second row of the figure turns the brightness up until the background
shows.

```{code-cell} python
:tags: [hide-input]
ny = img.shape[0]
ghost_zone = np.roll(mask, ny // 2, axis=0) & ~ndimage.binary_dilation(mask, iterations=3)

def ghost_level(image):
    """Mean signal where the ghost lands (half a FOV away, outside the head), relative to the brain."""
    return np.abs(image)[ghost_zone].mean() / img[mask].mean()

phases = [0.0, 0.04, 0.3]
fig, axes = plt.subplots(2, 3, figsize=(9.5, 6.6))
for col, ph in enumerate(phases):
    g = np.abs(synth.nyquist_ghost(img, ph))
    lvl = ghost_level(g)
    show_image(axes[0, col], g, f"odd/even phase error {np.degrees(ph):.1f}°", vmin=0, vmax=0.5)
    show_image(axes[1, col], g, f"background window: ghost {100 * lvl:.1f} %", vmin=0, vmax=0.03)
    axes[1, col].contour(ghost_zone, levels=[0.5], colors=[PALETTE[1]], linewidths=0.6)
    print(f"phase error {np.degrees(ph):4.1f} deg: ghost level {100 * lvl:.1f} % of the mean brain signal")
axes[0, 0].set_ylabel("normal window")
fig.tight_layout()
```

In the top row the 2.3° ghost is barely visible and even the 17° one is faint. In the
bottom row, windowed to 3 % of the top row's range, the brain itself is saturated white
and the orange outlines mark where a ghost would land: the brain shifted by half the field
of view, wrapping from bottom to top. The 2.3° error puts a 2.0 % ghost there, the normal
residual; the 17° error puts a 15.1 % ghost there, which calls for excluding the volume. The ghost level grows with the size of the
phase error between odd and even lines.

## k-space spikes

A discharge in the scanner room or a loose connection during the readout adds one bright
sample to k-space. Because every k-space sample is a plane wave across the whole image, a
single spike appears as a stripe pattern (a "herringbone" or "corduroy" artifact) covering
the entire slice. It affects only the volume and slice in which it occurred.

```{code-cell} python
:tags: [hide-input]
spikes = {"clean": None, "spike far from the k-space center": (70, 84), "spike near the center": (66, 66)}
fig, axes = plt.subplots(2, 3, figsize=(9.5, 6.4))
c0 = ny // 2
fov_mm = ny * tissue["voxel_mm"]
for col, (name, pos) in enumerate(spikes.items()):
    im_s = img if pos is None else synth.kspace_spike(img, pos[0], pos[1], 1.0)
    show_image(axes[0, col], im_s, name, vmin=0, vmax=0.5)
    show_kspace(axes[1, col], kspace.fft2c(np.asarray(im_s, complex)), "k-space (central half)")
    axes[1, col].set(xlim=(c0 - 32, c0 + 32), ylim=(c0 + 32, c0 - 32))
    if pos is not None:
        axes[1, col].add_patch(Circle((pos[1], pos[0]), 6, fill=False, color=PALETTE[2], lw=1.5))
        cycles = np.hypot(pos[0] - c0, pos[1] - c0)
        print(f"{name}: {cycles:.1f} cycles across the {fov_mm:.0f} mm field of view -> stripes every {fov_mm / cycles:.0f} mm")
fig.tight_layout()
```

The bottom row is k-space, where the spike is a single bright point (circled). Its
distance from the center sets the stripe spacing: a spike far from the center makes fine
stripes a centimeter apart, and a spike near the center makes broad bands that look like
a smooth shading error. With raw data the sample can be replaced; with magnitude images the
slice is treated as an outlier and replaced from the model ([Chapter 12](./12-motion-and-dropout.md)), which is what the
outlier detection of eddy and SHORELine does with it.

## Receive-field bias

The sensitivity of the coil array varies smoothly across the head ([Chapter 3](../01-mri-physics/03-reconstruction.md)), so the
images are brighter near the coils and darker in the center. This does not affect
diffusion measures, which are ratios between volumes of the same voxel, but it affects
everything that uses intensity across voxels: brain masking, tissue segmentation of the
b=0 image, and the estimation of the single-fiber response that spherical deconvolution
needs ([Chapter 16](../04-modeling/16-fiber-orientation.md)).

Pipelines estimate the field from the b=0 image and divide it out. The usual tool is N4, an
algorithm that finds the smoothest multiplicative field that makes each tissue as uniform
in brightness as possible. The toy below does the same thing more simply: it fits a smooth
surface to the log intensity of white matter, which should be uniform.

```{code-cell} python
:tags: [hide-input]
sens = kspace.ring_coil_sensitivities(8, *img.shape, with_phase=False, width=0.3, floor=0.02)
field = np.sqrt(np.sum(np.abs(sens) ** 2, axis=0))
field /= field[mask].mean()
biased = img * field

# estimate: a smooth (3rd-order polynomial) surface fitted to log intensity in white matter
wm_core = tissue["wm"] > 0.95
yy, xx = np.indices(img.shape, dtype=float) / img.shape[0] - 0.5
terms = np.stack([yy**i * xx**j for i in range(4) for j in range(4 - i)], axis=-1)
coef = np.linalg.lstsq(terms[wm_core], np.log(biased[wm_core]), rcond=None)[0]
field_est = np.exp(terms @ coef)
field_est /= field_est[mask].mean()
corrected = biased / field_est

cv = lambda a: 100 * a[wm_core].std() / a[wm_core].mean()
print(f"white matter brightness varies by {cv(biased):.0f} % (SD / mean) before correction, {cv(corrected):.1f} % after")

# FA does not care: it is computed from ratios between volumes of the same voxel
bv, bvec = schemes.multi_shell({1000: 30}, n_b0=2)
series = synth.synthetic_dwi(tissue, bv, bvec, orientation=synth.orientation_field(tissue["wm"]))
fa_clean = synth.dti_maps(series, bv, bvec, mask)["fa"]
fa_biased = synth.dti_maps(series * field[..., None], bv, bvec, mask)["fa"]
print(f"largest FA change from the bias field anywhere in the brain: {np.abs(fa_biased - fa_clean)[mask].max():.1e}")

fig, axes = plt.subplots(1, 4, figsize=(12, 3.3))
show_image(axes[0], biased, "b=0 with receive bias", vmin=0, vmax=0.45)
im = axes[1].imshow(np.where(mask, field_est, np.nan), cmap="viridis"); axes[1].set_axis_off(); axes[1].set_title("estimated field")
fig.colorbar(im, ax=axes[1], shrink=0.7)
show_image(axes[2], corrected, "corrected", vmin=0, vmax=0.45)
show_image(axes[3], fa_biased, "FA from the biased series\n(same as without bias)", kind="scalar", vmin=0, vmax=1)
fig.tight_layout()
```

The biased image is bright at the edges of the head, near the coils, and dim in the middle;
the same white matter is darker in the center than at the edge. The estimated field
captures that pattern, and dividing it out leaves white matter nearly uniform: its
brightness varies by 15 % across the slice before correction and 2.5 % after. The FA map
from the biased series is identical to the unbiased one (the largest difference is at the
level of rounding error): the field multiplies every volume of a voxel by the same number,
and FA depends only on ratios between those volumes. Because the correction is a smooth multiplicative field applied identically to all
volumes, its placement in the pipeline does not matter.

## The order of operations

The corrections of the preceding chapters have an order, and the order is not arbitrary.
The flowchart shows the whole pipeline: the main chain in the middle, the extra inputs
that make each correction possible on the left, and what the pipeline hands on at the
bottom.

```{code-cell} python
:tags: [hide-input]
fig, ax = plt.subplots(figsize=(9.5, 7.4))
ax.set(xlim=(0, 10), ylim=(0, 10)); ax.set_axis_off()

def box(x, y, w, h, text, fc, ec=INK["secondary"], fs=9, weight="normal"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.08", fc=fc, ec=ec, lw=1))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK["primary"], weight=weight)

def arrow(x0, y0, x1, y1, color=INK["secondary"], ls="-"):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2, ls=ls, shrinkA=2, shrinkB=2))

main_x, light = 5.6, "#f3f2ee"
box(main_x, 9.5, 3.4, 0.5, "raw DWI series + b-values, b-vectors", "white", weight="bold")
box(main_x, 8.6, 3.4, 0.5, "denoise (Ch 8)", light)
box(main_x, 7.7, 3.4, 0.5, "remove Gibbs ringing (Ch 9)", light)
ax.add_patch(FancyBboxPatch((main_x - 2.1, 3.9), 4.2, 3.05, boxstyle="round,pad=0.08", fc="#e8f0fb", ec=PALETTE[0], lw=1.4))
ax.text(main_x, 6.7, "geometric corrections: estimate each,\nthen compose into ONE resampling", ha="center", va="center", fontsize=9, weight="bold")
for i, s in enumerate(["susceptibility field (Ch 10)", "eddy currents (Ch 11)", "head motion (Ch 12)",
                       "gradient nonlinearity (Ch 13)", "outlier slices detected and replaced (Ch 12)"]):
    box(main_x, 6.0 - 0.47 * i, 3.6, 0.36, s, "white", fs=8.5)
box(main_x, 3.1, 3.4, 0.5, "bias field + brain mask (this chapter)", light)
outs = ["corrected\ndata", "rotated\nb-vectors", "gradient-\ndeviation image", "QC report"]
for i, s in enumerate(outs):
    x = 2.6 + 2.0 * i
    box(x, 1.2, 1.7, 0.75, s, "#e6f5ee", ec=PALETTE[2], fs=8.5)
    arrow(main_x, 2.8, x, 1.6, color=PALETTE[2])
for y0, y1 in [(9.25, 8.85), (8.35, 7.95), (7.45, 7.05), (3.85, 3.35)]:
    arrow(main_x, y0, main_x, y1)
ax.text(5.6, 0.35, "data, b-vectors, and deviation image go on to the model fits (Part IV)",
        ha="center", va="center", fontsize=8.5, color=INK["secondary"])
sides = [(6.0, "reverse phase-encode\nb=0 volumes"), (5.3, "metadata: phase-encode\ndirection, readout time"),
         (4.59, "gradient coefficient\nfile (from the vendor)")]
for y, s in sides:
    box(1.2, y, 2.0, 0.52, s, "#fdf0e6", ec=PALETTE[1], fs=8)
    arrow(2.3, y, main_x - 1.85, y, color=PALETTE[1])
ax.text(1.2, 7.0, "side inputs", ha="center", fontsize=9, color=PALETTE[1], weight="bold")
fig.tight_layout()
```

In words:

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

The reason for composing the transforms is that every resampling blurs. To move an image
by a fraction of a voxel, a pipeline must invent values between the samples it has, by
interpolation, and linear interpolation averages neighbors. Each round of averaging
softens edges a little more. The figure shows one sharp edge moved by a third of a voxel
at a time, as a pipeline that applied each correction separately would do.

```{code-cell} python
:tags: [hide-input]
x = np.arange(40, dtype=float)
edge = lambda pos: 0.5 * (1 + np.tanh((pos - 20.0) / 0.5))   # a sharp, known edge
step = 0.37                                                   # voxels moved per correction

def resample_n(n):
    prof = edge(x)
    for _ in range(n):
        prof = np.interp(x - step, x, prof)                   # one linear-interpolation resampling
    return prof

def width_10_90(prof):
    xf = np.linspace(0, 39, 3901); p = np.interp(xf, x, prof)
    return xf[np.argmax(p >= 0.9)] - xf[np.argmax(p >= 0.1)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.4))
xf = np.linspace(14, 30, 400)
ns = [1, 2, 3, 5]
for i, n in enumerate(ns):
    ax1.plot(x - 20 - n * step, resample_n(n), "-o", ms=3, color=PALETTE[i], label=f"{n} resampling{'s' if n > 1 else ''}")
ax1.plot(xf - 20, edge(xf), color=INK["primary"], lw=1, ls="--", label="true edge")
ax1.set(xlim=(-4, 4), xlabel="position relative to the edge (voxels)", ylabel="intensity", title="the edge after n resamplings")
ax1.legend(loc="upper left")
n_all = np.arange(1, 7)
w_seq = [width_10_90(resample_n(n)) for n in n_all]
w_one = [width_10_90(np.interp(x - n * step, x, edge(x))) for n in n_all]
ax2.axhline(width_10_90(edge(x)), color=INK["primary"], lw=1, ls="--", label="true edge (sampled)")
ax2.plot(n_all, w_seq, "-o", label="one resampling per correction")
ax2.plot(n_all, w_one, "-o", label="all corrections composed, one resampling")
ax2.set(xlabel="number of corrections", ylabel="edge width, 10–90 % (voxels)", title="blur vs number of corrections", ylim=(0, None))
ax2.legend(loc="lower right")
fig.tight_layout()
print(f"10-90 % edge width: truth {width_10_90(edge(x)):.2f} voxels; " +
      ", ".join(f"{n} resampling{'s' if n > 1 else ''} {width_10_90(resample_n(n)):.2f}" for n in ns))
```

Measured from 10 % to 90 % of its height, the sampled true edge is 1.65 voxels wide. One
resampling widens it to 2.22 voxels and five resamplings to 3.31: each correction applied
separately smears the edge a little further (left panel, and the blue line on the right).
Composing the corrections into one map keeps the blur at the level of a single resampling
however many corrections are applied (orange line; it wobbles because the blur of one
resampling depends on the fraction of a voxel it moves).

Composing is not the same as adding the shift maps together. The corrections happen one
after another in the scanner's forward model: the susceptibility field moves a point
first, and the eddy-current distortion then acts on the image as it already is. So the
eddy shift that applies to a point is the one at the place where the susceptibility shift
carried it, not the one at its true position. The composed map follows each point
through both steps.

The same comparison on the brain slice, with a susceptibility distortion and an
eddy-current shear:

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
total_shift = sdc_shift + ndimage.map_coordinates(eddy_shear, idx, order=1, mode="nearest")
composed = synth.undistort_along_pe(acquired, total_shift, order=1)

e_seq, e_comp = np.abs(sequential - img)[mask].mean(), np.abs(composed - img)[mask].mean()
print(f"mean error vs truth: two sequential resamplings {e_seq:.4f}, one composed resampling {e_comp:.4f} "
      f"({100 * (1 - e_comp / e_seq):.0f} % smaller)")
fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2))
show_image(axes[0], acquired, "acquired: distortion + eddy shear", vmin=0, vmax=0.5)
show_image(axes[1], np.abs(sequential - img) * mask, "error, two resamplings", vmin=0, vmax=0.1)
show_image(axes[2], np.abs(composed - img) * mask, "error, one composed resampling", vmin=0, vmax=0.1)
fig.tight_layout()
```

The two error maps look alike at this scale; the printed means tell them apart. For two
smooth corrections on one slice, composing makes the error 12 % smaller. The saving grows
with the number of steps (susceptibility, eddy, motion, gradient nonlinearity, and the
final resampling to the anatomical grid would be five), as the edge profile showed, and
with the roughness of the fields. The second reason to compose is that the corrections are not independent: the
eddy and motion parameters are estimated from images whose susceptibility distortion has
already been accounted for, so the tools that estimate them also apply them together.

## How the standard pipelines order these

The major pipelines follow the same order and differ mainly in which program does each
step. All of them write the rotated b-vectors and a quality report.

| Step | FSL {cite:p}`andersson2016,andersson2016b` | MRtrix {cite:p}`tournier2019` | QSIPrep {cite:p}`cieslak2021` | HCP pipelines {cite:p}`glasser2013` |
|---|---|---|---|---|
| Denoising (Ch 8) | — | `dwidenoise` | yes | — |
| Unringing (Ch 9) | — | `mrdegibbs` | yes | — |
| Susceptibility field (Ch 10) | topup | topup (wrapped) | topup, or TORTOISE DRBUDDI | topup |
| Eddy currents, motion, outliers (Ch 11–12) | eddy | eddy (wrapped) | eddy, or TORTOISE DIFFPREP | eddy |
| Gradient nonlinearity (Ch 13) | — | — | in the composed transform, with a coefficient file | gradwarp |
| One composed resampling | eddy applies the topup field with its own | as FSL | yes | eddy as FSL; gradwarp applied after it |
| Bias field | — | `dwibiascorrect` | yes | — |

## Quality control

A pipeline's report is the first thing to read, before any map. The measures that decide
whether a dataset is usable:

| Measure | What it flags | Typical action | Chapter |
|---|---|---|---|
| Framewise displacement per volume | between-volume motion | exclude subjects above a threshold (often 1–2 mm mean) | [12](./12-motion-and-dropout.md) |
| Outlier slices per volume | dropout, spikes | exclude subjects with more than a few percent of slices replaced | [12](./12-motion-and-dropout.md), this chapter |
| Residual distortion at the frontal lobe | wrong metadata or failed topup | check `PhaseEncodingDirection` and `TotalReadoutTime` | [10](./10-susceptibility-distortion.md) |
| b=0 SNR in white matter | noise level | denoise; question high-b results when the white-matter SNR of the highest-b images falls below about 3, where the noise floor dominates | [8](./08-noise.md) |
| Ghost level in the background | readout calibration | rescan or exclude the volume | this chapter |
| Model residuals across the brain | anything not modeled | inspect the residual map | — |

*Framewise displacement* (FD) is one number per volume that summarizes how far the head
moved since the previous volume, adding the three translations and the three rotations
(converted to millimeters of movement at the surface of a 50 mm sphere).

### Reading a report: one simulated subject

The panel below is a mock report for one subject with 60 diffusion volumes. The motion and
outlier traces are generated for the example; the ghost level is measured, volume by
volume, on the toy brain with the same measure as the ghost figure above.

```{code-cell} python
:tags: [hide-input]
rng = np.random.default_rng(4)
n_vol, n_slc = 60, 40
# motion: a slow drift plus small jitter, and one sudden movement at volume 38
trans = np.cumsum(rng.normal(0, 0.05, (n_vol, 3)), axis=0)
rot = np.cumsum(rng.normal(0, 0.0006, (n_vol, 3)), axis=0)       # radians
trans[38:] += np.array([1.2, -0.6, 0.4]); rot[38:] += np.array([0.012, 0.0, -0.006])
fd = np.r_[0, np.abs(np.diff(trans, axis=0)).sum(1) + 50 * np.abs(np.diff(rot, axis=0)).sum(1)]

# outlier slices: rare random ones, plus dropout in the moving volume (multiband pairs, Ch 12)
outl = rng.random((n_slc, n_vol)) < 0.004
outl[[4, 5, 6, 24, 25, 26], 38] = True
outl[[10, 30], 39] = True

# ghost level: small readout phase errors, one volume with a calibration glitch
ph = np.abs(rng.normal(0.03, 0.01, n_vol)); ph[51] = 0.3
ghost = np.array([ghost_level(np.abs(synth.nyquist_ghost(img, p))) for p in ph])

fig, axes = plt.subplots(3, 1, figsize=(9.5, 7.0), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.3, 1]))
axes[0].plot(np.arange(n_vol), fd, "-o", ms=3)
axes[0].axhline(1.0, color=PALETTE[7], ls="--", lw=1); axes[0].text(0.5, 1.05, "1 mm", color=PALETTE[7], fontsize=8)
axes[0].set(ylabel="FD (mm)", title="framewise displacement")
axes[1].imshow(outl, aspect="auto", cmap="Greys", interpolation="nearest", extent=(-0.5, n_vol - 0.5, n_slc - 0.5, -0.5))
axes[1].set(ylabel="slice", title="outlier slices (black = replaced)"); axes[1].grid(False)
axes[2].plot(np.arange(n_vol), 100 * ghost, "-o", ms=3, color=PALETTE[1])
axes[2].axhline(5.0, color=PALETTE[7], ls="--", lw=1); axes[2].text(0.5, 5.4, "5 %", color=PALETTE[7], fontsize=8)
axes[2].set(ylabel="ghost (%)", xlabel="volume", title="ghost level")
fig.tight_layout()

print(f"mean FD {fd.mean():.2f} mm; volumes with FD above 1 mm: {[int(v) for v in np.nonzero(fd > 1)[0]]}")
print(f"outlier slices replaced: {outl.sum()} of {outl.size} ({100 * outl.mean():.1f} %); "
      f"in volume 38: {outl[:, 38].sum()} of {n_slc}")
print(f"ghost level: median {100 * np.median(ghost):.1f} %, volume {np.argmax(ghost)} {100 * ghost.max():.1f} %")
```

The walk-through reads the three traces in turn.

- **Motion.** One spike stands out at volume 38, where the head moved suddenly by more
  than 1 mm; the rest of the trace is small drift. The mean FD over the run is 0.23 mm, well
  below the subject-level thresholds in the table. The subject is kept; eddy has already
  realigned volume 38 to the others.
- **Outlier slices.** Replacements cluster in volume 38 and the one after it, the same
  volumes as the motion spike: motion during the diffusion encoding causes signal dropout
  ([Chapter 12](./12-motion-and-dropout.md)). In total 0.8 % of slices were replaced, below "a few percent", so the
  replacement is trusted and the subject is kept.
- **Ghost level.** About 1.6 % (the median) in every volume except volume 51, at 15.1 %. A
  readout calibration failed for that one volume. The ghost overlaps the brain, and outlier
  replacement does not catch it: outlier detection looks for slices that lost signal, and
  a ghost adds signal instead. Volume 51 is excluded.

Decision: keep the subject, drop volume 51, and record mean FD and the percentage of
replaced slices as covariates for the group analysis.

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
