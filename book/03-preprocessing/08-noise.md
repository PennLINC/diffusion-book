---
title: "8. Thermal noise"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** a synthetic diffusion series on a 16-slice block of the packaged 3 mm volume, with a fiber orientation assigned to every white matter voxel ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
- **`noise-sweep`** (pending): four noise levels with one coil, plus an 8-coil run reconstructed with GRAPPA, the parallel-imaging method of [Chapter 3](../01-mri-physics/03-reconstruction.md) that fills in skipped k-space lines from the coil data ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-noise-sweep)).
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
  pulls them apart and produces anisotropy that is not there {cite:p}`jones2004squashing`.
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
centered on zero means no bias. The left panel is the complex data in white matter at
b = 3000; the middle panel is the magnitude of the same voxels at b = 0 and at b = 3000;
the right panel is the magnitude outside the head, where there is no signal at all. In each
panel, compare the histogram with the gray curve: in the first two panels it is unbiased
Gaussian noise, in the third the Rayleigh distribution of pure noise.

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
noise would be zero. It is computed below for every white matter voxel and each shell, as a
percentage of the true signal at that b-value:

```{code-cell} python
:tags: [hide-input]
B_SHELLS = [0, 1000, 2000, 3000]

def wm_bias(series):
    """Mean error in white matter per shell, in percent of the true signal at that b."""
    out = []
    for b in B_SHELLS:
        v = np.isclose(bvals, b)
        out.append(100 * (series[..., v] - clean[..., v])[wm].mean() / clean[..., v][wm].mean())
    return np.array(out)

bias = {"magnitude, noisy": wm_bias(noisy)}
print("magnitude, noisy: " + ",  ".join(f"b = {b}: {e:+.1f} %" for b, e in zip(B_SHELLS, bias["magnitude, noisy"])))
```

The bias grows as the signal falls: under 1 % at b = 0 and b = 1000, 2.4 % at b = 2000, and
6.7 % at b = 3000.
These are averages over all 12 directions of a shell; along the fibers, where the signal
is weakest, it is much larger, as the plot of signal against b showed.

## Correction step by step: MP-PCA

Marchenko-Pastur PCA denoising {cite:p}`veraart2016` uses the redundancy of a diffusion
series. Take a small block of neighboring voxels, here 5 × 5 × 5 = 125 of them, and look
at their 39 signal values each. The tissue in the block is made of a few kinds of
compartments, so those 125 signal curves are close to mixtures of a few shared shapes.
Principal component analysis (PCA) finds those shapes and ranks them by how much of the
variation between voxels each one explains. Noise, being independent in every voxel and
volume, cannot be explained by a few shapes; it spreads its variance over all 39
components. Random-matrix theory predicts exactly how: for pure noise of standard deviation
σ, the component variances fall inside a known range (the **Marchenko-Pastur** range),
which depends only on σ and on the block size.

The figure shows one such block in white matter. Each bar is one component's variance, in
units of σ², sorted from largest to smallest, on a logarithmic scale. The gray band is the
range that pure noise would produce. Look for the few bars that stand far above the band:
those carry the tissue. All the rest sit inside the band, and the open circles, the same
analysis of the noise-free block, show that their true content is nearly zero.

```{code-cell} python
:tags: [hide-input]
def patch_spectrum(series, center, r=2):
    i, j, k = center
    X = series[i - r : i + r + 1, j - r : j + r + 1, k - r : k + r + 1].reshape(-1, series.shape[-1])
    X = X - X.mean(axis=0)
    return np.sort(np.linalg.eigvalsh(X.T @ X / X.shape[0]))[::-1], X.shape[0]

center = (13, 33, K)
ev_noisy, n_vox = patch_spectrum(noisy, center)
ev_clean, _ = patch_spectrum(clean, center)
gamma = len(bvals) / n_vox
mp_lo, mp_hi = (1 - np.sqrt(gamma)) ** 2, (1 + np.sqrt(gamma)) ** 2   # in units of σ²
n_signal = int((ev_noisy / sigma**2 > mp_hi).sum())

fig, ax = plt.subplots(figsize=(7, 3.4))
idx = np.arange(1, len(ev_noisy) + 1)
ax.axhspan(mp_lo, mp_hi, color="0.85", zorder=0, label="range for pure noise (Marchenko-Pastur)")
ax.bar(idx[:n_signal], ev_noisy[:n_signal] / sigma**2, color=PALETTE[0], width=0.7, label="noisy block: kept (signal)")
ax.bar(idx[n_signal:], ev_noisy[n_signal:] / sigma**2, color=PALETTE[3], width=0.7, label="noisy block: discarded (noise)")
ax.plot(idx, np.maximum(ev_clean / sigma**2, 1e-3), "o", mfc="none", mec=INK["primary"], ms=5, label="noise-free block")
ax.set(yscale="log", ylim=(1e-2, 5e3), xlim=(0.3, len(idx) + 0.7), xlabel="component (sorted by variance)",
       ylabel="variance (units of σ²)", title=f"PCA of one {n_vox}-voxel block in white matter, {len(bvals)} volumes")
ax.legend(loc="upper right", fontsize=7)
fig.tight_layout()
print(f"{n_signal} of {len(ev_noisy)} components lie above the noise range (upper edge {mp_hi:.2f} σ²); "
      f"the other {len(ev_noisy) - n_signal} are discarded")
```

MP-PCA keeps the components above the band (blue), discards the rest (yellow), and
rebuilds each voxel from what is left. Because the height of the band depends on σ, the
same fit also estimates the noise level. The method needs no model of the tissue and is the
standard first step of current pipelines.

```{code-cell} python
:tags: [hide-input]
den_mag, sigma_est = mppca(noisy, patch_radius=2, return_sigma=True)
den_mag = np.clip(den_mag, 0, None)
bias["magnitude, MP-PCA"] = wm_bias(den_mag)
print(f"true noise SD {sigma:.4f}; MP-PCA estimate inside the brain {np.median(sigma_est[mask]):.4f}")
```

Denoising in the **complex domain** avoids the floor: the noise in the real and imaginary
channels is Gaussian with zero mean, so removing its random part leaves an unbiased signal,
and the magnitude is taken afterward, from a series with far less noise
{cite:p}`corderogrande2019`. The same MP-PCA can be applied by treating the real and
imaginary parts as additional volumes. This requires that the phase was saved ([Chapter 3](../01-mri-physics/03-reconstruction.md)).

```{code-cell} python
:tags: [hide-input]
complex_noisy = synth.add_complex_noise(clean, sigma, seed=0)
stacked = np.concatenate([complex_noisy.real, complex_noisy.imag], axis=-1)
den_stacked = mppca(stacked, patch_radius=2)
den_complex = np.abs(den_stacked[..., : len(bvals)] + 1j * den_stacked[..., len(bvals) :])
bias["complex, MP-PCA"] = wm_bias(den_complex)

fig, axes = plt.subplots(1, 4, figsize=(11, 3))
v = shell(3000)
show_image(axes[0], clean[:, :, K, v], "b = 3000, noise-free", vmin=0, vmax=0.12)
show_image(axes[1], noisy[:, :, K, v], "noisy", vmin=0, vmax=0.12)
show_image(axes[2], den_mag[:, :, K, v], "MP-PCA on magnitude", vmin=0, vmax=0.12)
show_image(axes[3], den_complex[:, :, K, v], "MP-PCA on complex data", vmin=0, vmax=0.12)
fig.tight_layout()
```

Both denoised images are smoother than the noisy one; the magnitude-domain result is still
too bright where the signal is weak, most plainly outside the head, which should be black
and is gray, while the complex-domain result is close to the reference. The next figure separates the two effects.
On the left are the residuals in white matter at b = 3000, as in the histograms earlier in
the chapter; on the right, the mean error (the bias) for each method and shell.

```{code-cell} python
:tags: [hide-input]
fig, (ax_h, ax_b) = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1, 1.1]})
colors = {"magnitude, noisy": PALETTE[3], "magnitude, MP-PCA": PALETTE[1], "complex, MP-PCA": PALETTE[0]}
series_by = {"magnitude, noisy": noisy, "magnitude, MP-PCA": den_mag, "complex, MP-PCA": den_complex}
bins = np.linspace(-3, 4, 50)
for label, s in series_by.items():
    r = (s - clean)[..., v3][wm] / sigma
    ax_h.hist(r, bins=bins, density=True, histtype="step", lw=2, color=colors[label], label=f"{label}: mean {r.mean():+.2f} σ, SD {r.std():.2f} σ")
ax_h.axvline(0, color=INK["secondary"], lw=1, ls="--")
ax_h.set(title="WM residual at b = 3000", xlabel="noisy − true (units of σ)", ylabel="density", ylim=(0, 1.75))
ax_h.legend(fontsize=7, loc="upper right")
width = 0.26
for n, (label, e) in enumerate(bias.items()):
    ax_b.bar(np.arange(len(B_SHELLS)) + (n - 1) * width, e, width, color=colors[label], label=label)
ax_b.axhline(0, color=INK["secondary"], lw=1)
ax_b.set_xticks(range(len(B_SHELLS)), [f"b = {b}" for b in B_SHELLS])
ax_b.set(title="bias in white matter", ylabel="mean error (% of true signal)")
ax_b.legend(fontsize=7, loc="upper left")
fig.tight_layout()
for label, e in bias.items():
    print(f"{label:>18}: bias at b = 3000 {e[-1]:+.1f} %")
```

Each histogram has a width and a center. The width is the **fluctuation**, the part of the
noise that differs from one voxel and one volume to the next. The center is the **bias**,
the part that pushes every weak measurement the same way. MP-PCA on magnitude data narrows
the histogram but leaves its center where it was, and its bias bars are nearly as tall as
those of the noisy data. The reason follows from how the method works: the floor raises
each weak value by an amount set by its true signal, so similar voxels in a block are
raised alike, and a change shared by the whole block is exactly the kind of structure PCA
keeps as signal. The floor is a bias, not a fluctuation,
so a method that removes fluctuations cannot remove it. MP-PCA on complex data narrows the
histogram just as much and also leaves it centered on zero, because before the magnitude is
taken there is no floor to keep.

A third option, when only magnitudes are available, is to subtract the floor
arithmetically. On average, the squared magnitude equals the squared true signal plus
2σ², so subtracting 2σ² from the squared measurement and taking the square root removes
most of the floor (the **method of moments**, $\sqrt{M^2 - 2\sigma^2}$, with $M$
the measured magnitude and $\sigma$ the noise SD). Applied voxel by voxel it trades one
problem for another: wherever noise has pushed $M$ below $\sqrt{2}\,\sigma$ there is no
square root to take, and those voxels must be set to zero, so weak regions come out speckled
with zeros and the fluctuation is not reduced at all. It works best on values that have
already been averaged or denoised.

The next page describes the other use of the phase: correcting it so that the data can be
kept as real values, in which case no floor arises in the first place. The two uses are
complementary. Complex denoising removes most of the fluctuation, and because a magnitude
is still taken at the end, a small floor remains where the denoised signal is weakest; the
real part removes the floor but not the fluctuation. Pipelines that have the phase do both:
denoise the complex data, then take the phase-corrected real part.

## Residual error versus truth

The tensor fit summarizes the effect on derived measures. All fits use only the b ≤ 1000
volumes, where the SNR is still high (9.5 in white matter at b = 1000), which is how
tensors are usually fitted; the high-b volumes, where the floor matters most, are the ones
kurtosis and compartment models read ([Chapter 15](../04-modeling/15-signal-representations.md)). Each fit is compared with
the fit of the noise-free series. The maps show the FA error, the fitted FA minus the
noise-free FA: red where FA is too high, blue where it is too low.

```{code-cell} python
:tags: [hide-input]
low = bvals <= 1000
ref = synth.dti_maps(clean[..., low], bvals[low], bvecs[low], mask=mask)
rows = []
print(f"{'':>18}   {'WM FA error':>15}   {'WM MD error (µm²/ms)':>20}   {'GM FA error':>11}")
print(f"{'':>18}   {'mean':>7} {'SD':>7}   {'mean':>10} {'SD':>9}   {'mean':>11}")
for label, series in [("noisy", noisy), ("MP-PCA magnitude", den_mag), ("MP-PCA complex", den_complex)]:
    m = synth.dti_maps(series[..., low], bvals[low], bvecs[low], mask=mask)
    rows.append((label, m))
    fa_err = (m["fa"] - ref["fa"])[wm]
    md_err = (m["md"] - ref["md"])[wm] * 1e3
    fa_gm = (m["fa"] - ref["fa"])[gm]
    print(f"{label:>18}   {fa_err.mean():+7.3f} {fa_err.std():7.3f}   {md_err.mean():+10.3f} {md_err.std():9.3f}   {fa_gm.mean():+11.3f}")

fig, axes = plt.subplots(1, 4, figsize=(11, 3))
show_image(axes[0], ref["fa"][:, :, K], "FA, noise-free", kind="scalar", vmin=0, vmax=0.9)
for ax, (label, m) in zip(axes[1:], rows):
    show_image(ax, np.where(mask[:, :, K], m["fa"][:, :, K] - ref["fa"][:, :, K], 0), f"FA error, {label}", kind="diff", vmin=-0.2, vmax=0.2)
fig.tight_layout()
fig.colorbar(axes[3].images[0], ax=axes[1:], shrink=0.8, label="FA − noise-free FA")
```

The number to take from the table is the last column. Gray matter is where noise-induced
anisotropy shows: its true FA is near zero, and the fit of the noisy data overestimates it
by 0.10 on average, which is why red dominates the speckle of the second panel. Denoising
cuts that to about 0.03 and narrows the spread of the white matter FA error from 0.036 to
0.024; the magnitude and complex versions do about equally well here, because at
b ≤ 1000 the signal is well above the floor. The difference between them is in the
high-b shells, in the bias figure above.

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
- **Fewer coils do not help.** With a root-sum-of-squares combination every coil adds its
  own noise to the sum, so the floor rises with the number of coils (roughly as its square
  root, [Chapter 3](../01-mri-physics/03-reconstruction.md)), but the extra coils also raise the SNR, and fewer coils
  would lower the signal along with the floor. Save the phase instead, so that denoising can work in the complex domain.
- **Plan the highest shell around its SNR**, not the b=0 SNR. If b = 3000 lands below
  SNR 3 in the tissue of interest, add averages or lower the shell.
- **Denoise before anything else.** Every later step (unringing, registration, fitting)
  works better on denoised data, and denoising after interpolation is less effective
  because interpolation correlates the noise.

## Further reading

The Rician distribution {cite:p}`gudbjartsson1995`, MP-PCA denoising
{cite:p}`veraart2016`, and complex-domain denoising {cite:p}`corderogrande2019`.
