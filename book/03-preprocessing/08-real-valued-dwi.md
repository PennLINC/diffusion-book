---
title: "8b. Real-valued diffusion MRI"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** the synthetic series of [Chapter 8](./08-noise.md) with a simulated object phase ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
- **Simulated live in this page:** one slice of the simulated brain under the acquired 76-volume HBCD protocol (six b=0 volumes, then the shells interleaved; not the b-sorted 75-volume teaching scheme of [Chapter 5](../02-diffusion-encoding/05-diffusion-encoding.md)), with the simulator's object phase and eddy-current phase ramp, phase-corrected and scored against its noise-free run.
- **`noise-sweep`** (pending): four noise levels with one coil, plus an 8-coil run reconstructed with GRAPPA, the parallel-imaging method of [Chapter 3](../01-mri-physics/03-reconstruction.md) that fills in skipped k-space lines from the coil data ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-noise-sweep)).
- **`truth`** (pending): the 27 analytic ground-truth maps and the true fiber orientations ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-truth), [Appendix E](../appendices/e-truth-map-catalogue.md)).

Live-tier figures simulate one slice of the simulated brain in the page through TRXScan's Python package ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md#live-tier)); they run in seconds at build time.
Pipeline-tier datasets are simulated offline by TRXScan ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)) and are marked *pending* until their release; the figures that need them say so where they will appear.
:::

## Learning goals

After this chapter you can:

- explain why a phase-corrected real-valued image has Gaussian noise with no floor
- estimate and remove the background phase of a diffusion volume and take its real part
- state what the approach requires, what it costs, and when it fails

```{code-cell} python
:tags: [hide-cell]
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

from dwibook import kspace, phantoms, schemes, synth
from dwibook.plotting import PALETTE, animate, complex_noise_cloud, set_style, show_image

set_style()
```

## The idea

The noise floor of [Chapter 8](./08-noise.md) exists because the magnitude of a complex number is never
negative: noise that would have pushed a weak signal below zero is folded back up. The
complex image has no such floor, but its signal does not lie along one axis: every voxel
carries a phase from the field offset, the coils, the eddy currents, and motion during the
encoding ([Chapter 3](../01-mri-physics/03-reconstruction.md)), and that phase varies across the image and between volumes.
A phase map is drawn with a cyclic color scale, because an angle of +180° is the same as
−180°: the scale is dark at 0°, bluish for negative and reddish for positive angles, and
light at ±180°, where its two ends meet. The first panel of the four-panel figure below is a
typical example, a smooth gradient of color across the brain.

If the phase is smooth, it can be estimated and removed. Rotating each voxel's complex value
by the negative of its estimated phase puts the signal on the real axis, and the real part
of the result is then a real-valued image in which the signal is signed and the noise is
Gaussian with zero mean {cite:p}`eichner2015`. Weak signals average to their true value
instead of to a floor; high-b images can be averaged and fitted without bias; and nothing
about the acquisition changes, only that the phase is saved.

The picture of [Chapter 3](../01-mri-physics/03-reconstruction.md) shows why. The animation measures one weak voxel (SNR 1.5)
3000 times. Its true value sits at a phase of 125°, so the cloud of measurements lies in
the upper left of the complex plane, and its real part is mostly negative. The cloud is then
rotated by −125° onto the real axis. Below the plane are two histograms: the real part of
the cloud as drawn (blue), and the magnitude (yellow), which rotation does not change.
Watch the blue mean arrive on the dashed true value once the rotation is complete, with
part of the histogram below zero, while the yellow mean stays above it.

```{code-cell} python
:tags: [hide-input]
PHI, SNR_DEMO = np.deg2rad(125), 1.5
turns = np.concatenate([np.zeros(6), np.linspace(0, 1, 24), np.ones(10)])
fig, (ax_plane, ax_hist) = plt.subplots(2, 1, figsize=(5.4, 7.4), gridspec_kw={"height_ratios": [1.4, 1]})

stats_rot = {}  # filled by the last frame, which is fully rotated

def rotate_frame(i):
    ax_plane.clear(); ax_hist.clear()
    stats_rot.update(complex_noise_cloud(ax_plane, ax_hist, SNR_DEMO, sigma=1.0, phase=PHI, seed=3, show=("real", "magnitude"),
                        rotated=turns[i], extent=(-4.2, 5.2, -3.2, 4.6), hist_ymax=0.62))
    ax_plane.set_title(f"measured phase 125°, rotated by −{125 * turns[i]:.0f}°")
    ax_hist.set_xlabel("value (units of σ)")
    fig.tight_layout()

animate(fig, rotate_frame, range(len(turns)), fps=8, width=460, dpi=70,
        alt="a cloud of noisy measurements of one weak voxel sits in the upper left of the complex plane, around a true value at a phase of 125 degrees, and rotates clockwise until it is centered on the positive real axis. Below it, the histogram of the real part moves from mostly negative values to a bell curve centered on the true value of 1.5 noise units, with some values below zero, while the histogram of the magnitude stays where it is, with its mean above the true value")

print(f"true value {SNR_DEMO:.2f} σ; mean real part after rotation {stats_rot['mean_real']:.2f} σ; "
      f"mean magnitude {stats_rot['mean_magnitude']:.2f} σ")
```

The rotation works only if the phase of every voxel is known. The measured phase cannot be
used directly: at low SNR it is dominated by the noise, and rotating each measurement by its
own noisy phase puts every one of them on the positive real axis, which is the magnitude
again. What is needed is the phase of the true signal, and the method rests on that phase
being smooth across the image, so that it can be estimated from the neighborhood of each
voxel with the noise averaged out. That is the filter of the next section.

## Estimating the background phase

The phase to remove is the slowly varying one. A low-pass filter of the complex image (a
Gaussian blur of the real and imaginary parts separately, then the angle of the result)
estimates it while averaging out the noise; the width of the filter is the one assumption,
and it must be wide enough to suppress noise and narrow enough to follow the true phase.
The four panels below show one b = 3000 volume: the true phase that the simulation put in;
the phase as measured, the same gradient buried in speckle wherever the signal is weak; the
magnitude image; and the real part after rotating each voxel by a phase estimated with a
2-voxel filter. The real part is drawn with a diverging color scale, so that negative
values, which a magnitude cannot have, show as blue.

```{code-cell} python
:tags: [hide-input]
vol = phantoms.brain_volume()
sl = slice(20, 36)
tissue = {k: vol[k][:, :, sl] for k in ("wm", "gm", "csf")}
mask = vol["mask"][:, :, sl]
bvals, bvecs = schemes.multi_shell({1000: 12, 2000: 12, 3000: 12}, n_b0=3)
clean = synth.synthetic_dwi(tissue, bvals, bvecs)
wm = tissue["wm"] > 0.9
K = 9
sigma = clean[wm][:, bvals == 0].mean() / 20.0

# a smooth, volume-dependent phase of the kind real data carry, plus complex noise
yy, xx, zz = np.indices(mask.shape, dtype=float)
rng = np.random.default_rng(1)
phase = np.stack([2.5 * rng.choice([-1, 1]) * (xx / xx.max() - 0.5) + 1.5 * rng.choice([-1, 1]) * (yy / yy.max() - 0.5) + 0.8 * np.sin(2 * np.pi * (rng.random() + zz / 40)) for _ in range(len(bvals))], axis=-1)
complex_noisy = synth.add_complex_noise(clean * np.exp(1j * phase), sigma, seed=0)

def phase_correct(series, width_vox=2.0):
    smooth = ndimage.gaussian_filter(series.real, (width_vox, width_vox, width_vox, 0)) + 1j * ndimage.gaussian_filter(series.imag, (width_vox, width_vox, width_vox, 0))
    return (series * np.exp(-1j * np.angle(smooth))).real

real_valued = phase_correct(complex_noisy)
magnitude = np.abs(complex_noisy)
v3 = np.flatnonzero(np.isclose(bvals, 3000))[0]

fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
axes[0].imshow(np.where(mask[:, :, K], phase[:, :, K, v3], np.nan), cmap="twilight", vmin=-np.pi, vmax=np.pi); axes[0].set_axis_off(); axes[0].set_title("true background phase, b = 3000")
axes[1].imshow(np.where(mask[:, :, K], np.angle(complex_noisy[:, :, K, v3]), np.nan), cmap="twilight", vmin=-np.pi, vmax=np.pi); axes[1].set_axis_off(); axes[1].set_title("measured phase (noisy)")
show_image(axes[2], magnitude[:, :, K, v3], "magnitude", vmin=0, vmax=0.12)
show_image(axes[3], real_valued[:, :, K, v3], "phase-corrected real part", kind="diff", vmin=-0.12, vmax=0.12)
fig.tight_layout()
fig.colorbar(axes[3].images[0], ax=axes[3], shrink=0.75)
```

The real-valued image has negative (blue) voxels in the background and in the darkest
tissue, which is the point: noise is allowed to go both ways. In the magnitude image the
same background is a uniform dark gray, never black: that is the floor.

The filter width is a trade-off, and the next figure shows both ways to get it wrong. Each
column uses a different width. The top row is the error of the estimated phase (estimated
minus true) in the b = 3000 volume; the bottom row is the error of the real part, averaged
over all 39 volumes so that the random noise mostly cancels and any bias remains. With a filter
narrower than a voxel (left), the estimate follows the noise: the phase error is speckle,
each measurement is rotated partly by its own noise, and the real part drifts back toward
the magnitude, too bright (red) everywhere. With a filter much wider than the features of
the phase (right), the estimate is smooth but wrong in two ways that add up. It averages
across the edge of the brain into the empty background, so the estimate at each voxel is
centered on a point shifted toward the middle of the brain, and along the in-plane phase
ramp that shift turns into an error that grows toward the edges, positive on one side and
negative on the other. And it averages through the slices, where the simulated phase
follows a gentle curve; the average over a curve misses the curve's value at the displayed
slice, which offsets the whole slice. In this volume the two
errors add on the left and largely cancel on the right. Wherever the phase is wrong, the
signal is rotated partly onto the imaginary axis, and the real part comes out too dark
(blue), most of all at the edge of the brain. A width
of about two voxels (middle) sits between the two. The printed numbers put the same result
in white matter at b = 3000: a bias of about +4 % with the narrow filter, −4 % with the wide
one, and almost none at two voxels.

```{code-cell} python
:tags: [hide-input]
widths = [0.5, 2.0, 8.0]
b3 = np.isclose(bvals, 3000)
fig, axes = plt.subplots(2, 3, figsize=(9.5, 6.2))
for j, w in enumerate(widths):
    smooth = ndimage.gaussian_filter(complex_noisy.real, (w, w, w, 0)) + 1j * ndimage.gaussian_filter(complex_noisy.imag, (w, w, w, 0))
    phase_err = np.angle(np.exp(1j * (np.angle(smooth) - phase)))
    rv = (complex_noisy * np.exp(-1j * np.angle(smooth))).real
    resid = (rv - clean).mean(axis=-1)
    show_image(axes[0, j], np.where(mask[:, :, K], phase_err[:, :, K, v3], 0), f"filter width {w:g} voxels\nphase error", kind="diff", vmin=-1, vmax=1)
    show_image(axes[1, j], np.where(mask[:, :, K], resid[:, :, K], 0), "real part − true\n(mean of all volumes)", kind="diff", vmin=-0.01, vmax=0.01)
    rms = np.sqrt((phase_err[..., b3][mask] ** 2).mean())
    bias_w = 100 * (rv - clean)[..., b3][wm].mean() / clean[..., b3][wm].mean()
    print(f"filter width {w:>3g} voxels: RMS phase error {rms:.2f} rad; WM bias of the real part at b = 3000 {bias_w:+.1f} %")
fig.tight_layout()
fig.colorbar(axes[0, 2].images[0], ax=axes[0, :], shrink=0.8, label="radians")
fig.colorbar(axes[1, 2].images[0], ax=axes[1, :], shrink=0.8, label="signal units")
```

## Measure it: the distribution and the bias

```{code-cell} python
:tags: [hide-input]
resid_m = (magnitude - clean)[..., v3][wm]
resid_r = (real_valued - clean)[..., v3][wm]
x = np.linspace(-4 * sigma, 4 * sigma, 300)
fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.hist(resid_m, bins=50, density=True, color=PALETTE[3], alpha=0.5, label=f"magnitude: mean {resid_m.mean() / sigma:+.2f} σ")
ax.hist(resid_r, bins=50, density=True, color=PALETTE[0], alpha=0.5, label=f"real-valued: mean {resid_r.mean() / sigma:+.2f} σ")
ax.plot(x, np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi)), color="0.3", lw=1.5, label="Gaussian, mean 0")
ax.set(title="residual in white matter at b = 3000 (SNR about 3)", xlabel="noisy − true"); ax.legend(fontsize=8)
fig.tight_layout()

print("mean signal error in white matter, relative to the true signal at that b:")
for label, s in [("magnitude", magnitude), ("real-valued", real_valued)]:
    print(f"  {label:>12}", end="")
    for b in [0, 1000, 2000, 3000]:
        v = np.isclose(bvals, b)
        print(f"   b={b}: {100 * (s[..., v] - clean[..., v])[wm].mean() / clean[..., v][wm].mean():+5.1f} %", end="")
    print()
```

The real-valued residual is Gaussian around zero at b = 3000, where the magnitude residual
has a positive mean. Fits to real-valued data need one adjustment. The quickest way to fit a
tensor takes the logarithm of each signal value, which turns the exponential decay into a
straight line ([Chapter 15](../04-modeling/15-signal-representations.md)); but a weak voxel can now come out negative, and a
negative number has no logarithm. Real-valued data are therefore fitted to the signal
itself, with a nonlinear fit, which treats a negative value like any other.

## When it fails

- **Rough phase.** Near air-tissue interfaces, and in volumes with strong eddy-current or
  motion phase, the true phase varies faster than the filter allows. The estimate is then
  wrong, part of the signal is rotated onto the imaginary axis and lost, and the real part
  is biased downward. The figure after this list shows the effect; a per-volume check of
  the imaginary part after correction shows where it happens.
- **Partial Fourier.** The reconstruction already assumed a smooth phase ([Chapter 3](../01-mri-physics/03-reconstruction.md)); the
  two assumptions compound, and the phase estimate should be made after the partial-Fourier
  reconstruction, from the complex image it produced.
- **Multi-coil combination.** The phase must survive the combination: a sensitivity-weighted
  combination keeps it, a root-sum-of-squares discards it ([Chapter 3](../01-mri-physics/03-reconstruction.md)).

To see the first failure, the next figure adds a sharp phase feature to every volume: a
3-radian bump a few voxels wide at the front of the brain, where the air in the sinuses
produces the steepest field changes. The phase is then estimated with the same 2-voxel
filter. Look at the front of the brain in each panel. The left panel is the true phase with
the bump. The middle panel is the real part minus the true signal, averaged over the three
b = 0 volumes: a dark (blue) hole where the filter could not follow the bump and signal was
rotated away. The right panel is the imaginary part after correction, which should be pure
noise around zero; the lost signal appears there instead, so this is the map to check in
real data.

```{code-cell} python
:tags: [hide-input]
cy, cx = 7, 26   # the front of the brain, mid-line, in the display slice
bump = 3.0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 1.5**2) - (zz - K) ** 2 / (2 * 3.0**2))
rough_phase = phase + bump[..., None]
rough = synth.add_complex_noise(clean * np.exp(1j * rough_phase), sigma, seed=0)
smooth = ndimage.gaussian_filter(rough.real, (2, 2, 2, 0)) + 1j * ndimage.gaussian_filter(rough.imag, (2, 2, 2, 0))
rotated = rough * np.exp(-1j * np.angle(smooth))
b0 = bvals == 0
fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
axes[0].imshow(np.where(mask[:, :, K], np.angle(np.exp(1j * rough_phase[:, :, K, 0])), np.nan), cmap="twilight", vmin=-np.pi, vmax=np.pi)
axes[0].set_axis_off(); axes[0].set_title("true phase with a sharp feature")
show_image(axes[1], np.where(mask[:, :, K], (rotated.real - clean)[:, :, K][..., b0].mean(-1), 0), "real part − true (b = 0 mean)", kind="diff", vmin=-0.15, vmax=0.15)
show_image(axes[2], np.where(mask[:, :, K], rotated.imag[:, :, K][..., b0].mean(-1), 0), "imaginary part after correction", kind="diff", vmin=-0.15, vmax=0.15)
fig.tight_layout()
fig.colorbar(axes[2].images[0], ax=axes[1:], shrink=0.8)
near = (np.hypot(yy - cy, xx - cx) <= 3) & (np.abs(zz - K) <= 1) & mask
far = mask & ~(np.hypot(yy - cy, xx - cx) <= 8)
rel = lambda region: 100 * (rotated.real - clean)[..., b0][region].mean() / clean[..., b0][region].mean()
print(f"real part at b = 0, relative to the true signal: within 3 voxels of the feature {rel(near):+.0f} %; elsewhere {rel(far):+.1f} %")
```

Real-valued conversion and complex-domain denoising ([Chapter 8](./08-noise.md)) solve different halves of
the problem and are used together. Complex denoising removes most of the random
fluctuation; the real part removes the floor, which denoising alone makes small but cannot
eliminate, because a magnitude is still taken at the end. The phase matters for the
denoising too: the complex demonstration of [Chapter 8](./08-noise.md) had no phase at all, but in real
data the phase changes from volume to volume, and denoising treats that change as structure
to keep, so it removes less noise. The phase is therefore dealt with first: a smooth
estimate of each volume's phase is removed, the complex series is denoised, and the real
part is taken, with the phase estimate refined from the denoised series if needed
{cite:p}`corderogrande2019`.

## Measure it: one slice, simulated live

What the simulator adds to the toy example is a phase that behaves like a scanner's: an
object phase modeled on real HBCD scans, plus an eddy-current phase ramp that changes with
the diffusion direction, so that no single phase map serves every volume. The same
per-volume correction as above, applied to one simulated slice under the acquired 76-volume
HBCD protocol (in acquisition order, with the shells interleaved), is scored against the
noise-free run of the same slice.

```{code-cell} python
:tags: [hide-input]
import trxscan as ts
from dwibook import phantom as ph

g = ph.gtab()
art = ts.Artifacts(eddy_phase=1.7e-5, seed=1)                      # the phase ramp a 3 T HBCD scan shows
noisy = ph.run(g, ph.PROTO, art.replace(noise=3.0), kspace=False)   # k-space noise for an SNR of about 3 in WM at b = 3000
clean = ph.run(g, ph.PROTO, art, kspace=False)
cplx = noisy.complex                                                # (nx, ny, 1, n_vol) complex64
mag_sim = np.abs(cplx)
real_sim = phase_correct(cplx, width_vox=2.0)
ref = clean.magnitude.get_fdata()
obj = ph.grid()
k = ph.slice_index()
wm_sim = obj.image("wm").get_fdata()[:, :, k] > 0.9
b = np.where(g.b0s_mask, 0.0, g.bvals)   # the protocol's b0 volumes carry a small nominal b
v3 = np.flatnonzero(np.isclose(b, 3000, atol=50))[0]
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
axes[0].imshow(np.rot90(np.where(obj.image("mask").get_fdata()[:, :, k] > 0, noisy.phase.get_fdata()[:, :, 0, v3], np.nan)), cmap="twilight", vmin=-np.pi, vmax=np.pi); axes[0].set_axis_off(); axes[0].set_title("simulated phase, b = 3000")
show_image(axes[1], ph.axial(mag_sim, v3), "magnitude")
show_image(axes[2], ph.axial(real_sim, v3), "phase-corrected real part")
show_image(axes[3], ph.axial(real_sim, v3) - ph.axial(ref, v3), "real part minus noise-free", kind="diff")
fig.tight_layout()
print("mean signal error in white matter, relative to the noise-free signal at that b:")
for label, s in [("magnitude", mag_sim), ("real-valued", real_sim)]:
    print(f"  {label:>12}", end="")
    for bb in [0, 1000, 2000, 3000]:
        v = np.isclose(b, bb, atol=50)
        err = (s[:, :, 0, :][..., v] - ref[:, :, 0, :][..., v])[wm_sim]
        print(f"   b={bb}: {100 * err.mean() / ref[:, :, 0, :][..., v][wm_sim].mean():+5.1f} %", end="")
    print()
```

The simulated phase is smooth inside the head, so the two-voxel filter recovers it and the
real-valued error at b = 3000 sits near zero where the magnitude is biased upward. The
remaining differences are the Rician-free noise itself and the voxels at the brain edge,
where the filter averages phase across the boundary; the eddy-current ramp, which varies
from volume to volume, is handled because the phase is estimated per volume.

## What this implies for acquisition

- **Save the phase** ([Chapter 3](../01-mri-physics/03-reconstruction.md)); nothing else at the scanner is required. The
  request to the scanner operator is to reconstruct and export phase images alongside the
  magnitude images for the diffusion series (on Siemens scanners, the reconstruction
  setting "Magnitude et phase"; other vendors have an equivalent). Check afterward that the
  DICOM-to-NIfTI conversion kept them: dcm2niix writes the phase as a separate file with a
  `_ph` suffix.
- **Prefer a combination that keeps the phase** if the coil combination is configurable:
  an adaptive or sensitivity-weighted combination rather than sum of squares.
- **Keep partial Fourier moderate**, so that the phase assumption both reconstructions rely
  on holds.

## Further reading

Real-valued diffusion MRI {cite:p}`eichner2015` and complex-domain denoising
{cite:p}`corderogrande2019`.
