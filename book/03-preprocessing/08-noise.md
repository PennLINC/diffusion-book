---
title: "8. Thermal noise"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** a synthetic diffusion series on a 16-slice block of the packaged 3 mm volume, with a fiber orientation assigned to every white matter voxel ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
- **`noise-sweep`** (pending): four noise levels with one coil, plus an 8-coil GRAPPA run ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-noise-sweep)).
- **`truth`** (pending): the 27 analytic ground-truth maps and the true fiber orientations ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-truth), [Appendix E](../appendices/e-truth-map-catalogue.md)).

Pipeline-tier datasets are simulated offline by TRXScan ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)) and are marked *pending* until their release; the figures that need them say so where they will appear.
:::

## Learning goals

After this chapter you can:

- measure the SNR of a diffusion dataset and say where in the brain and at which b-values
  it is lowest
- explain how the noise floor biases high-b signal and the diffusion measures derived from it
- apply MP-PCA denoising to magnitude and to complex data and measure what each removes
- state which acquisition choices reduce noise at the source

```{code-cell} python
:tags: [hide-cell]
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # one BLAS thread: MP-PCA is many small SVDs
import numpy as np
import matplotlib.pyplot as plt
from dipy.denoise.localpca import mppca

from dwibook import kspace, phantoms, schemes, signal, synth
from dwibook.plotting import INK, PALETTE, animate, complex_noise_cloud, set_style, show_image

set_style()
```

## The physics

Thermal noise from the subject and the receive electronics enters every k-space sample as
independent Gaussian noise. In the complex image it is still Gaussian with zero mean. The
magnitude operation changes that. [Chapter 3](../01-mri-physics/03-reconstruction.md) showed why with one picture: repeated
measurements of a voxel form a cloud of points around the true value in the complex plane,
and the magnitude is each point's distance from the origin. The animation below slides the
true value toward the origin, from SNR 5 to SNR 0 (SNR is the true signal divided by the
noise standard deviation σ). On the left are the cloud and, under it, the histogram of the
magnitudes; on the right, the mean magnitude against the true signal. Watch the yellow mean
separate from the dashed true value once the cloud reaches the origin, and the point on the
right leave the identity line and level off at the noise floor.

```{code-cell} python
:tags: [hide-input]
snrs = np.concatenate([np.full(4, 5.0), np.linspace(5, 0, 26), np.full(6, 0.0)])
a = np.linspace(0, 5.5, 200)
fig = plt.figure(figsize=(9.5, 5.4))
gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1], height_ratios=[1.35, 0.8])
ax_plane, ax_hist, ax_curve = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])

def sweep_frame(i):
    for ax in (ax_plane, ax_hist, ax_curve):
        ax.clear()
    s = complex_noise_cloud(ax_plane, ax_hist, snrs[i], sigma=1.0, seed=3, extent=(-4, 9, -3.6, 3.6), hist_ymax=0.72)
    ax_plane.set_title(f"true signal: SNR {snrs[i]:.1f}")
    ax_hist.set_xlabel("magnitude (units of σ)")
    ax_curve.plot(a, a, color="0.5", lw=1, ls="--", label="identity (no bias)")
    ax_curve.plot(a, kspace.rician_mean(a, 1.0), color=PALETTE[3], label="mean magnitude")
    ax_curve.axhline(np.sqrt(np.pi / 2), color=INK["secondary"], lw=1, ls=":", label="noise floor, 1.25 σ")
    ax_curve.plot([snrs[i]], [s["mean_magnitude"]], "o", ms=8, color=PALETTE[3], mec=INK["primary"])
    ax_curve.set(xlim=(-0.2, 5.5), ylim=(0, 5.8), xlabel="true signal (units of σ)", ylabel="mean magnitude (units of σ)",
                 title="mean magnitude vs. true signal")
    ax_curve.set_aspect("equal", adjustable="box")
    ax_curve.legend(loc="upper left")
    fig.tight_layout()

animate(fig, sweep_frame, range(len(snrs)), fps=6, width=760, dpi=70,
        alt="left: a cloud of noisy measurements in the complex plane slides from a true value of 5 noise units toward the origin, with a histogram of their magnitudes below; while the cloud is far from the origin the mean magnitude sits on the true value, and as the cloud reaches and surrounds the origin the mean stays near 1.25 noise units while the true value drops to zero. Right: the mean magnitude plotted against the true signal follows the identity line at high signal and levels off at the noise floor of 1.25 noise units at low signal")
```

```{code-cell} python
:tags: [hide-input]
for s in [0, 1, 3, 5]:
    print(f"true signal {s} σ: mean magnitude {kspace.rician_mean(s, 1.0):.2f} σ")
```

The distance from the origin is never negative, so noise that would push a weak signal
below zero is folded back to a positive value, and the average ends up too high. In the
language of statistics the magnitude follows a **Rician** distribution for a single coil
(the **Rayleigh** distribution, with mean $\sigma\sqrt{\pi/2} \approx 1.25\,\sigma$, where
there is no signal) and a **non-central chi** distribution after a root-sum-of-squares
combination of several coils, whose floor is higher still.

In diffusion MRI the weak-signal case is not an edge case. At b = 3000 gray matter retains
about 15 % of its b=0 signal and white matter along the fibers less than 1 % ([Chapter 5](../02-diffusion-encoding/05-diffusion-encoding.md)).
Those volumes sit at SNR 1–4, where the floor is a substantial fraction of the measurement.
The figure follows one white matter voxel with SNR 20 at b = 0, the level used in the rest
of this chapter, measured along and across its fibers. The vertical axis is logarithmic, so
a single exponential decay is a straight line. Compare each solid line (the true signal)
with the dotted line of the same color (the mean magnitude that would be measured), and
note where the dotted line meets the dashed noise floor.

```{code-cell} python
:tags: [hide-input]
b_axis = np.linspace(0, 3000, 121)
fig, ax = plt.subplots(figsize=(6.5, 3.6))
for cos, label, color in [(1.0, "along the fibers", PALETTE[0]), (0.0, "across the fibers", PALETTE[1])]:
    true = 20.0 * signal.white_matter(b_axis, cos)
    ax.plot(b_axis, true, color=color, label=f"{label}: true signal")
    ax.plot(b_axis, kspace.rician_mean(true, 1.0), color=color, ls=":", lw=2.4, label=f"{label}: mean magnitude")
ax.axhline(np.sqrt(np.pi / 2), color=INK["secondary"], lw=1, ls="--", label="noise floor, 1.25 σ")
ax.set(yscale="log", ylim=(0.08, 30), xlim=(0, 3000), xlabel="b (s/mm²)", ylabel="signal (units of σ)",
       title="white matter, SNR 20 at b = 0")
ax.set_yticks([0.1, 0.3, 1, 3, 10, 30], ["0.1", "0.3", "1", "3", "10", "30"])
ax.legend(loc="lower left", fontsize=7)
fig.tight_layout()
for b in [1000, 2000, 3000]:
    t = 20.0 * signal.white_matter(np.array([b]), 1.0)[0]
    print(f"along the fibers, b = {b}: true signal {t:.2f} σ, mean magnitude {kspace.rician_mean(t, 1.0):.2f} σ")
```

Along the fibers the true signal falls in a straight line on this scale, but the measured
mean bends away from it and flattens onto the floor. At b = 2000 the true signal is
0.67 σ and the mean magnitude 1.39 σ; at b = 3000 they are 0.12 σ and 1.26 σ, ten times too
high, and the measurement no longer depends on the tissue at all. Across the fibers the
signal stays well above the floor and is measured almost correctly. The consequences for
fitted quantities follow directly:

- **Signal decay with b appears too shallow**, because the measured high-b values are too
  high. Fitted diffusivities come out too low, most of all along the fibers.
- **Anisotropy is wrong.** In white matter the floor raises the along-fiber signal the most,
  which shrinks the difference between directions that the tensor fit reads as anisotropy,
  so FA fitted from high-b volumes comes out too low.
  In nearly isotropic tissue such as gray matter, the random spread of the noise does the
  opposite: the fit always ranks the three diffusivities from largest to smallest, so noise
  pulls them apart and produces anisotropy that is not there.
- **Kurtosis and multi-compartment fits**, which read the curvature of the decay above
  b = 1500, are affected most, because the floor adds curvature of its own: the bend of
  the dotted line above.

## The artifact-free reference

The toy demonstrations use a synthetic series built from the simulated brain's tissue fractions on
a 16-slice block of the 3 mm volume, with fibers oriented along the local white matter
boundary (see `dwibook.synth`). The scheme is a reduced multi-shell: 3 b=0, 12 directions
each at b = 1000, 2000, and 3000. The noise-free series is the reference against which every
number below is measured.

```{code-cell} python
:tags: [hide-input]
vol = phantoms.brain_volume()
sl = slice(20, 36)
tissue = {k: vol[k][:, :, sl] for k in ("wm", "gm", "csf")}
mask = vol["mask"][:, :, sl]
bvals, bvecs = schemes.multi_shell({1000: 12, 2000: 12, 3000: 12}, n_b0=3)
clean = synth.synthetic_dwi(tissue, bvals, bvecs)
wm = tissue["wm"] > 0.9
gm = tissue["gm"] > 0.9
K = 9  # display slice (through the ventricles)
b0_wm = clean[wm][:, bvals == 0].mean()
print(f"series shape {clean.shape}; white matter b=0 signal {b0_wm:.3f}")
```

## See it: noise added

The noise level is set so that white matter has SNR 20 at b = 0, a typical value for a
2 mm acquisition on a 3 T scanner.

```{code-cell} python
:tags: [hide-input]
SNR0 = 20.0
sigma = b0_wm / SNR0
noisy = synth.add_noise(clean, sigma, seed=0)

shell = lambda b: np.flatnonzero(np.isclose(bvals, b))[0]
fig, axes = plt.subplots(2, 4, figsize=(11, 5.5))
for j, b in enumerate([0, 1000, 2000, 3000]):
    v = shell(b)
    show_image(axes[0, j], clean[:, :, K, v], f"b = {b}, noise-free", vmin=0, vmax=0.25)
    show_image(axes[1, j], noisy[:, :, K, v], f"b = {b}, SNR {SNR0:.0f} at b = 0", vmin=0, vmax=0.25)
fig.tight_layout()
for b in [0, 1000, 2000, 3000]:
    v = shell(b)
    print(f"b = {b:>4}: WM SNR {clean[wm][:, v].mean() / sigma:5.1f}   GM SNR {clean[gm][:, v].mean() / sigma:5.1f}")
```

## See it: the distribution of the noise

The images show the effect; the distributions show the cause. The same noise realization is
examined twice: as the real and imaginary components of the complex image, and as the
magnitude. The residual is the noisy value minus the noise-free value, so a distribution
centered on zero means no bias.

```{code-cell} python
:tags: [hide-input]
complex_noisy = synth.add_complex_noise(clean, sigma, seed=0)
v3 = shell(3000)
wm_hi = wm  # white matter at b = 3000: SNR about 3
background = ~vol["mask"][:, :, sl]
a_true = clean[..., v3][wm_hi].mean()

fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
x = np.linspace(-4 * sigma, 4 * sigma, 300)
gauss = np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
resid_c = (complex_noisy - clean)[..., v3][wm_hi]
axes[0].hist(resid_c.real, bins=50, density=True, color=PALETTE[0], alpha=0.5, label="real part")
axes[0].hist(resid_c.imag, bins=50, density=True, color=PALETTE[1], alpha=0.5, label="imaginary part")
axes[0].plot(x, gauss, color="0.3", lw=1.5, label="Gaussian, mean 0")
axes[0].set(title="complex data, WM at b = 3000: residual", xlabel="noisy − true"); axes[0].legend(fontsize=7)

resid_m3 = (noisy - clean)[..., v3][wm_hi]
resid_m0 = (noisy - clean)[..., shell(0)][wm_hi]
axes[1].hist(resid_m0, bins=50, density=True, color=PALETTE[2], alpha=0.5, label=f"b = 0 (SNR {SNR0:.0f}): mean {resid_m0.mean() / sigma:+.2f} σ")
axes[1].hist(resid_m3, bins=50, density=True, color=PALETTE[3], alpha=0.5, label=f"b = 3000 (SNR {a_true / sigma:.1f}): mean {resid_m3.mean() / sigma:+.2f} σ")
axes[1].plot(x, gauss, color="0.3", lw=1.5, label="Gaussian, mean 0")
axes[1].set(title="magnitude data, WM: residual", xlabel="noisy − true"); axes[1].legend(fontsize=7)

m = np.linspace(0, 5 * sigma, 300)
axes[2].hist(noisy[..., v3][background], bins=50, density=True, color=PALETTE[4], alpha=0.5, label="measured")
axes[2].plot(m, kspace.noncentral_chi_pdf(m, 0.0, sigma, 1), color="0.3", lw=1.5, label=f"Rayleigh, mean {np.sqrt(np.pi / 2):.2f} σ")
axes[2].set(title="magnitude data, background (no signal)", xlabel="magnitude"); axes[2].legend(fontsize=7)
fig.tight_layout()
```

In the complex data the residual is Gaussian with zero mean at every signal level. In the
magnitude data it is Gaussian with zero mean only where the signal is strong (b = 0); where
the signal is comparable to the noise (b = 3000) it is skewed and its mean is positive, and
where there is no signal it is the Rayleigh distribution with a mean of 1.25 σ. The
positive mean is the bias that the rest of this chapter measures and removes.

## Measure it: the bias

The bias is the mean difference between the noisy and the noise-free series, which for pure
noise would be zero. Inside white matter at b = 3000 it is not:

```{code-cell} python
:tags: [hide-input]
def bias_table(series, label):
    print(f"{label:>28}", end="")
    for b in [0, 1000, 2000, 3000]:
        v = np.isclose(bvals, b)
        err = (series[..., v] - clean[..., v])[wm]
        print(f"   b={b}: {100 * err.mean() / clean[..., v][wm].mean():+5.1f} %", end="")
    print()

print("mean signal error in white matter, relative to the true signal at that b")
bias_table(noisy, "magnitude, noisy")
```

The floor also reaches the tensor fit. Fitting only the b ≤ 1000 volumes, where the SNR is
still high, keeps the tensor errors modest; the high-b volumes are where kurtosis and
compartment models would read the bias as tissue.

## Correction step by step: MP-PCA

Marchenko-Pastur PCA denoising {cite:p}`veraart2016` uses the redundancy of a diffusion
series. In a small neighborhood of voxels, the signals across all volumes are well
described by a few principal components; noise fills the remaining components with a known
spectrum (the Marchenko-Pastur distribution for random matrices). Components whose variance
matches that spectrum are removed. The method needs no model of the tissue, estimates the
noise level itself, and is the standard first step of current pipelines. Applied to
magnitude data it removes the random part of the noise but not the floor, because the floor
is a bias, not a fluctuation.

```{code-cell} python
:tags: [hide-input]
den_mag, sigma_est = mppca(noisy, patch_radius=2, return_sigma=True)
den_mag = np.clip(den_mag, 0, None)
print(f"true noise SD {sigma:.4f}; MP-PCA estimate inside the brain {np.median(sigma_est[mask]):.4f}")
bias_table(noisy, "magnitude, noisy")
bias_table(den_mag, "magnitude, MP-PCA")
```

Denoising in the **complex domain** avoids the floor: the noise in the real and imaginary
channels is Gaussian with zero mean, so removing its random part leaves an unbiased signal,
and the magnitude is taken afterward, from a series with far less noise
{cite:p}`corderogrande2019`. The same MP-PCA can be applied by treating the real and
imaginary parts as additional volumes. This requires that the phase was saved ([Chapter 3](../01-mri-physics/03-reconstruction.md)).
The next page describes the other use of the phase: correcting it so that the data can be
kept as real values, in which case no floor arises in the first place.

```{code-cell} python
:tags: [hide-input]
complex_noisy = synth.add_complex_noise(clean, sigma, seed=0)
stacked = np.concatenate([complex_noisy.real, complex_noisy.imag], axis=-1)
den_stacked = mppca(stacked, patch_radius=2)
den_complex = np.abs(den_stacked[..., : len(bvals)] + 1j * den_stacked[..., len(bvals) :])
bias_table(den_complex, "complex, MP-PCA")

fig, axes = plt.subplots(1, 4, figsize=(11, 3))
v = shell(3000)
show_image(axes[0], clean[:, :, K, v], "b = 3000, noise-free", vmin=0, vmax=0.12)
show_image(axes[1], noisy[:, :, K, v], "noisy", vmin=0, vmax=0.12)
show_image(axes[2], den_mag[:, :, K, v], "MP-PCA on magnitude", vmin=0, vmax=0.12)
show_image(axes[3], den_complex[:, :, K, v], "MP-PCA on complex data", vmin=0, vmax=0.12)
fig.tight_layout()
```

The magnitude-domain result is smoother but still too bright where the signal is weak; the
complex-domain result is close to the reference. A third option, when only magnitudes are
available, is to estimate the floor from the noise level and subtract it from the squared
signal (the method of moments, $\sqrt{M^2 - 2\sigma^2}$); it removes the mean bias but not
the fluctuation and fails where the signal is below the floor.

## Residual error versus truth

The tensor fit summarizes the effect on derived measures. All fits use the b ≤ 1000 volumes
and are compared with the fit of the noise-free series.

```{code-cell} python
:tags: [hide-input]
low = bvals <= 1000
ref = synth.dti_maps(clean[..., low], bvals[low], bvecs[low], mask=mask)
rows = []
for label, series in [("noisy", noisy), ("MP-PCA magnitude", den_mag), ("MP-PCA complex", den_complex)]:
    m = synth.dti_maps(series[..., low], bvals[low], bvecs[low], mask=mask)
    rows.append((label, m))
    fa_err = (m["fa"] - ref["fa"])[wm]
    md_err = (m["md"] - ref["md"])[wm] * 1e3
    fa_gm = (m["fa"] - ref["fa"])[gm]
    print(f"{label:>18}: WM FA error {fa_err.mean():+.3f} ± {fa_err.std():.3f}   WM MD error {md_err.mean():+.3f} ± {md_err.std():.3f} (x10^-3)   GM FA error {fa_gm.mean():+.3f}")

fig, axes = plt.subplots(1, 4, figsize=(11, 3))
show_image(axes[0], ref["fa"][:, :, K], "FA, noise-free", kind="scalar", vmin=0, vmax=0.9)
for ax, (label, m) in zip(axes[1:], rows):
    show_image(ax, m["fa"][:, :, K], f"FA, {label}", kind="scalar", vmin=0, vmax=0.9)
fig.tight_layout()
```

Gray matter is where noise-induced anisotropy shows: its true FA is near zero, and every
fit of noisy data overestimates it. Denoising reduces the spread of the errors in both
tissues; only the complex-domain version also removes the bias.

## Measure it: the simulated datasets

:::{admonition} Simulated dataset pending
:class: note
This section will load the `noise-sweep` dataset (four noise levels, plus an 8-coil GRAPPA
run) and the `truth` maps, repeat the measurements above on the simulated acquisition, and
compare MP-PCA's noise estimate with the noise map TRXScan wrote.
:::

## What acquisition choices reduce it

- **Shorter TE** is the largest lever on SNR ([Chapter 7](../02-diffusion-encoding/07-acquisition-parameters.md)); gradient strength, partial
  Fourier, and in-plane acceleration all shorten it.
- **Larger voxels** raise SNR in proportion to their volume, at the cost of partial volume.
- **Fewer coils do not help**; the floor rises with coil count but so does SNR. Save the
  phase instead, so that denoising can work in the complex domain.
- **Plan the highest shell around its SNR**, not the b=0 SNR. If b = 3000 lands below
  SNR 3 in the tissue of interest, add averages or lower the shell.
- **Denoise before anything else.** Every later step (unringing, registration, fitting)
  works better on denoised data, and denoising after interpolation is less effective
  because interpolation correlates the noise.

## Further reading

The Rician distribution {cite:p}`gudbjartsson1995`, MP-PCA denoising
{cite:p}`veraart2016`, and complex-domain denoising {cite:p}`corderogrande2019`.
