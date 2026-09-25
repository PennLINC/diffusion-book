---
title: Thermal noise
subtitle: Chapter 8
kernelspec:
  name: python3
  display_name: Python 3
---

## Learning goals

After this chapter you can:

- measure the SNR of a diffusion dataset and say where in the brain and at which b-values
  it is lowest
- explain how the noise floor biases high-b signal and the diffusion measures derived from it
- apply MP-PCA denoising to magnitude and to complex data and measure what each removes
- state which acquisition choices reduce noise at the source

**Datasets used:** `noise-sweep`, `truth` (pending); the toy tier uses a synthetic series
**Simulation tier:** toy + phantom

```{code-cell} python
:tags: [hide-cell]
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # one BLAS thread: MP-PCA is many small SVDs
import numpy as np
import matplotlib.pyplot as plt
from dipy.denoise.localpca import mppca

from dwibook import kspace, phantoms, schemes, synth
from dwibook.plotting import PALETTE, set_style, show_image

set_style()
```

## The physics

Thermal noise from the subject and the receive electronics enters every k-space sample as
independent Gaussian noise. In the complex image it is still Gaussian with zero mean. The
magnitude operation changes that (Chapter 3): where the signal is weak relative to the noise,
the magnitude is biased upward toward a floor, and the distribution is Rician for a single
coil and non-central chi after a root-sum-of-squares combination.

In diffusion MRI the weak-signal case is not an edge case. At b = 3000 gray matter retains
about 15 % of its b=0 signal and white matter along the fibers less than 1 % (Chapter 5).
Those volumes sit at SNR 1–4, where the floor is a substantial fraction of the measurement.
The consequences for fitted quantities follow directly:

- **Signal decay with b appears too shallow**, because the measured high-b values are too
  high. Fitted diffusivities come out too low.
- **Anisotropy appears too high**, because the direction with the lowest true signal (along
  the fibers) is raised most by the floor, and the extra spread between directions enters
  the tensor fit as anisotropy. In isotropic regions, noise produces anisotropy that is not
  there.
- **Kurtosis and multi-compartment fits**, which read the curvature of the decay above
  b = 1500, are affected most, because the floor adds curvature of its own.

## The phantom dataset

The simulated dataset for this chapter is `noise-sweep` (four noise levels and an 8-coil
GRAPPA run), scored against `truth`. The simulator settings that produce it are listed
under its name in [Appendix A](#app-a-datasets), and its files in Appendix B.

## The artifact-free reference

The toy demonstrations use a synthetic series built from the phantom's tissue fractions on
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
imaginary parts as additional volumes. This requires that the phase was saved (Chapter 3).
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

## Measure it: the phantom

:::{admonition} Phantom figure pending
:class: note
This section will load the `noise-sweep` dataset (four noise levels, plus an 8-coil GRAPPA
run) and the `truth` maps, repeat the measurements above on the simulated acquisition, and
compare MP-PCA's noise estimate with the noise map TRXScan wrote.
:::

## What acquisition choices reduce it

- **Shorter TE** is the largest lever on SNR (Chapter 7); gradient strength, partial
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
