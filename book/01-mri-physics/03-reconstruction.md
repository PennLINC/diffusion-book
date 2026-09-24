---
title: Image reconstruction from k-space
subtitle: Chapter 3
kernelspec:
  name: python3
  display_name: Python 3
---

## Learning goals

After this chapter you can:

- describe what a reconstructed diffusion image contains beyond its magnitude, and what is
  lost when only the magnitude is saved
- explain how multiple receive coils are combined and how they are used to shorten the
  readout (parallel imaging), and what that costs in noise
- explain how partial-Fourier data are completed, and when that fails
- recognize what compressed sensing requires
- describe the noise in a magnitude image, and state how large the resulting bias is at the
  signal levels typical of high b-value diffusion data

**Datasets used:** `slab-kspace` (pending)
**Simulation tier:** toy + phantom

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt

from dwibook import kspace, phantoms
from dwibook.plotting import PALETTE, set_style, show_image, show_kspace

set_style()
N = 128
obj = phantoms.brain_image()  # synthetic b=0 slice, adult preset, TE 88 ms
mask = phantoms.brain_slice()["mask"]
img = obj * np.exp(1j * phantoms.brain_phase())  # the same slice with a smooth phase
```

## Magnitude and phase

For a fully sampled Cartesian acquisition the reconstruction is an inverse Fourier
transform, and its result is a complex number in every voxel, because every k-space sample is
complex to begin with: it is the pair of quadrature-demodulated receiver channels described
in Chapter 1. The magnitude is the image that is normally viewed and analyzed. The phase is
measured relative to the receiver's reference, so a constant offset is arbitrary; its spatial
and temporal variations are not. The phase is usually discarded. In diffusion MRI the
phase carries information that is used later in this book: the static field offset
(Chapter 10), the eddy-current phase that changes with each diffusion direction
(Chapter 11), and the phase acquired through motion during the diffusion encoding
(Chapter 12). Saving only the magnitude removes all of it.

```{code-cell} python
:tags: [hide-input]
rec = kspace.ifft2c(kspace.fft2c(img))
fig, axes = plt.subplots(1, 3, figsize=(9, 3))
show_kspace(axes[0], kspace.fft2c(img), "acquired k-space (log scale)")
show_image(axes[1], np.abs(rec), "magnitude")
axes[2].imshow(np.where(mask, np.angle(rec), np.nan), cmap="twilight", vmin=-np.pi, vmax=np.pi); axes[2].set_axis_off(); axes[2].set_title("phase (radians)")
fig.tight_layout()
```

## Multi-coil combination

Modern scanners receive with an array of many small coils. Each coil records the same object
weighted by its own sensitivity, a smooth field that is strongest near the coil. Each coil's
voltage is demodulated separately, and differences in cable length and electronics give each
channel its own phase offset, so the sensitivities are complex. The
reconstruction has to combine the coil images into one, and the two standard methods differ
in what they require:

- **Root sum of squares** requires no information about the coils. It is a magnitude
  combination, it inherits a smooth intensity variation across the image, and it raises the
  noise floor in proportion to the number of coils (see the noise section).
- **Sensitivity-weighted combination** (Roemer) requires the coil sensitivities, which are
  measured in a calibration scan or estimated from the center of k-space. It gives the best
  SNR and preserves the phase.

The coil model used here is the one TRXScan uses: coils on a ring around the head, each with a
Gaussian falloff, plus a smooth phase per coil.

```{code-cell} python
:tags: [hide-input]
NC = 8
sens = kspace.ring_coil_sensitivities(NC, N, N)
coil_ksp = kspace.fft2c(sens * img)
coil_img = kspace.ifft2c(coil_ksp)

fig, axes = plt.subplots(2, 4, figsize=(10, 5))
for c in range(4):
    show_image(axes[0, c], coil_img[c], f"coil {c + 1} of {NC}")
show_image(axes[1, 0], np.abs(sens[0]), "sensitivity of coil 1")
show_image(axes[1, 1], kspace.sos_combine(coil_img), "root sum of squares")
show_image(axes[1, 2], kspace.roemer_combine(coil_img, sens), "sensitivity-weighted")
axes[1, 3].imshow(np.where(mask, np.angle(kspace.roemer_combine(coil_img, sens)), np.nan), cmap="twilight"); axes[1, 3].set_axis_off(); axes[1, 3].set_title("phase, sensitivity-weighted")
fig.tight_layout()
```

## Parallel imaging

Because each coil views the object from a different position, the set of coil images
contains spatial information that can substitute for some of the phase-encode lines.
Acquiring only every $R$-th line shortens the EPI readout by a factor of $R$ (Chapter 2). The
raw result is an image folded over on itself $R$ times, in every coil. Two families of
reconstruction unfold it:

- **SENSE** works in the image domain. At each pixel of the folded image, the $R$ true pixels
  that overlap there are unknowns, and the coil sensitivities at those positions provide the
  equations to solve for them.
- **GRAPPA** works in k-space and does not need sensitivity maps. A block of fully sampled
  central lines (the autocalibration signal, ACS) is used to learn how each missing line can
  be predicted from neighboring acquired lines across all coils; the learned weights are then
  applied throughout k-space. TRXScan simulates GRAPPA with 24 ACS lines, so the phantom data
  in later chapters went through this reconstruction.

```{code-cell} python
:tags: [hide-input]
R, ACS = 2, 24
mask_r2 = kspace.regular_undersampling_mask(N, N, R, acs_lines=ACS)
under = np.where(mask_r2, coil_ksp, 0)
aliased = kspace.ifft2c(np.where(kspace.regular_undersampling_mask(N, N, R), coil_ksp, 0)) * R

sense = kspace.sense_reconstruct(aliased, sens, R)
grappa_ksp = kspace.grappa_reconstruct(under, mask_r2, R, acs=(N // 2 - ACS // 2, N // 2 + ACS // 2))
grappa = kspace.roemer_combine(kspace.ifft2c(grappa_ksp), sens)
ref = kspace.roemer_combine(coil_img, sens)

fig, axes = plt.subplots(1, 4, figsize=(11, 3))
show_image(axes[0], kspace.sos_combine(aliased), f"R = {R}: folded image")
show_image(axes[1], sense, "SENSE")
show_image(axes[2], grappa, f"GRAPPA ({ACS} ACS lines)")
show_image(axes[3], np.abs(grappa) - np.abs(ref), "GRAPPA minus reference", kind="diff", vmin=-0.05, vmax=0.05)
fig.tight_layout()
```

Without noise, both methods recover the image almost exactly. The cost of parallel imaging is
noise. Fewer samples raise the noise by $\sqrt{R}$, and the unfolding amplifies it further by
a spatially varying factor (the g-factor) that depends on how different the coil
sensitivities are at the positions that fold together. The amplification can be measured
directly by adding noise and comparing the reconstructions:

```{code-cell} python
:tags: [hide-input]
sigma = 0.01
noisy_full = kspace.add_complex_noise(coil_ksp, sigma, seed=1)
noisy_under = np.where(mask_r2, kspace.add_complex_noise(coil_ksp, sigma, seed=1), 0)
rec_full = kspace.roemer_combine(kspace.ifft2c(noisy_full), sens)
rec_r2 = kspace.roemer_combine(kspace.ifft2c(kspace.grappa_reconstruct(noisy_under, mask_r2, R, acs=(N // 2 - ACS // 2, N // 2 + ACS // 2))), sens)
noise_full = np.std((rec_full - ref)[mask])
noise_r2 = np.std((rec_r2 - ref)[mask])
print(f"noise SD inside the brain: R = 1: {noise_full:.4f}; R = 2 with GRAPPA: {noise_r2:.4f}; "
      f"ratio {noise_r2 / noise_full:.2f} (sqrt(R) alone would give {np.sqrt(R):.2f})")
```

In practice, $R = 2$ is common in diffusion protocols because the shorter readout reduces
distortion and TE; $R = 3$ and above are used with caution because the noise penalty grows
quickly with the coil geometry of a head array.

## Partial Fourier reconstruction

A partial-Fourier acquisition (Chapter 2) skips the lines whose mirror images were acquired.
The reconstruction supplies them, and the methods differ in what they assume about the
image phase:

- **Zero filling** assumes nothing. The missing lines are set to zero, which blurs the image
  along the phase-encode direction.
- **Homodyne** assumes the phase is smooth. It estimates the phase from the fully sampled
  center, uses the acquired lines to stand in for their missing mirrors, and removes the
  estimated phase.
- **POCS** makes the same assumption iteratively, alternating between agreement with the
  acquired data and agreement with the estimated phase.

```{code-cell} python
:tags: [hide-input]
pf_mask = kspace.partial_fourier_mask(N, N, 0.625)
full = kspace.fft2c(img)
recs = {"zero filling": kspace.zero_fill(full, pf_mask), "homodyne": kspace.homodyne(full, pf_mask), "POCS": kspace.pocs(full, pf_mask, 20)}

fig, axes = plt.subplots(2, 3, figsize=(9, 6))
for ax_top, ax_bot, (name, r) in zip(axes[0], axes[1], recs.items()):
    err = np.abs(np.abs(r) - obj)
    show_image(ax_top, r, name)
    show_image(ax_bot, err, f"error, mean {err[mask].mean():.4f}", vmin=0, vmax=0.1)
fig.tight_layout()
```

The two phase-aware methods recover the sharpness that zero filling loses. Both fail in the
same situation: where the phase changes faster than the fully sampled center can resolve,
the estimate is wrong and the reconstruction shows ringing and signal loss at that location.
In diffusion data this occurs near air–tissue interfaces and wherever eddy currents or
motion imprint sharp phase changes. This is one reason partial Fourier factors are kept
moderate (6/8 or 7/8) in diffusion protocols.

```{code-cell} python
:tags: [hide-input]
yy, xx = np.mgrid[0:N, 0:N] / N - 0.5
fast_phase = np.exp(1j * (phantoms.brain_phase() + 14 * np.sin(6 * np.pi * yy) * (np.abs(xx) < 0.12)))
full_fast = kspace.fft2c(obj * fast_phase)
fig, axes = plt.subplots(1, 3, figsize=(9, 3))
axes[0].imshow(np.where(mask, np.angle(obj * fast_phase), np.nan), cmap="twilight"); axes[0].set_axis_off(); axes[0].set_title("a rapidly varying phase stripe")
show_image(axes[1], kspace.homodyne(full_fast, pf_mask), "homodyne reconstruction")
show_image(axes[2], np.abs(np.abs(kspace.homodyne(full_fast, pf_mask)) - obj), "error", vmin=0, vmax=0.3)
fig.tight_layout()
```

## Compressed sensing

Regular under-sampling produces coherent aliasing: neat copies of the object, which parallel
imaging removes using coil information. Random under-sampling produces incoherent aliasing
that resembles noise, and a different kind of information removes it: most medical images
are compressible, meaning they can be represented by a small number of coefficients in a
suitable transform (wavelets are the usual choice). A reconstruction that looks for the
image with the fewest such coefficients among all images consistent with the sampled
k-space can recover the object from far fewer samples than the Nyquist rule requires. The
three requirements, incoherent sampling, a sparsifying transform, and an iterative
nonlinear solver, define compressed sensing {cite:p}`lustig2007`.

```{code-cell} python
:tags: [hide-input]
cs_mask = kspace.random_undersampling_mask(N, N, accel=3, acs_lines=12, seed=3)
reg_mask = kspace.regular_undersampling_mask(N, N, 3)
ksp_obj = kspace.fft2c(obj)
zf_random = kspace.zero_fill(ksp_obj, cs_mask)
zf_regular = kspace.zero_fill(ksp_obj, reg_mask) * 3
cs = kspace.cs_reconstruct(ksp_obj, cs_mask, lam=0.01, iters=100)

fig, axes = plt.subplots(1, 4, figsize=(11, 3))
axes[0].imshow(cs_mask, cmap="gray", aspect="auto"); axes[0].set_axis_off(); axes[0].set_title(f"random lines, {cs_mask[:, 0].mean():.0%} sampled")
show_image(axes[1], zf_regular, "regular R = 3: coherent aliasing")
show_image(axes[2], zf_random, "random: incoherent aliasing")
show_image(axes[3], cs, "compressed-sensing reconstruction")
fig.tight_layout()
print(f"mean error inside the brain: zero-filled random {np.abs(np.abs(zf_random) - obj)[mask].mean():.4f}, "
      f"compressed sensing {np.abs(np.abs(cs) - obj)[mask].mean():.4f}")
```

The reconstruction above uses a simple wavelet and a short solver so that every step is
visible; the residual blockiness comes from that choice, and production methods use better
transforms. In diffusion MRI, compressed sensing in k-space is used mainly for multi-shot
and 3-D readouts, since single-shot EPI already collects all of k-space in one pass. The more
common use is in q-space: sampling diffusion directions and b-values sparsely and
reconstructing with a sparsity prior, which is CS-DSI (Chapter 6).

## Noise in magnitude images

Thermal noise enters the data as independent Gaussian noise in k-space, and the Fourier
transform preserves that: the complex image also has Gaussian noise with zero mean. Taking
the magnitude changes the statistics {cite:p}`gudbjartsson1995`:

- With a single coil the magnitude follows a Rician distribution. Where there is no signal,
  the magnitude is not zero but has a positive mean (the noise floor). Where the signal is
  strong, the noise is approximately Gaussian again.
- A root-sum-of-squares combination of $L$ coils follows a non-central chi distribution
  {cite:p}`constantinides1997`, and its noise floor rises with the number of coils.
- After GRAPPA the noise varies across the image and is correlated between coils, so neither
  distribution holds exactly.

```{code-cell} python
:tags: [hide-input]
sigma = 0.02
noisy = kspace.ifft2c(kspace.add_complex_noise(kspace.fft2c(obj), sigma, seed=0))
tissue = phantoms.brain_slice()
wm = tissue["wm"] > 0.95
background = (tissue["wm"] + tissue["gm"] + tissue["csf"]) == 0
a_wm = obj[wm].mean()
m = np.linspace(0, 0.35, 400)

coil_noisy = kspace.ifft2c(kspace.add_complex_noise(kspace.fft2c(sens * obj), sigma, seed=0))
sos = kspace.sos_combine(coil_noisy)

fig, axes = plt.subplots(1, 3, figsize=(10, 3))
axes[0].hist(np.abs(noisy[background]), bins=60, density=True, color=PALETTE[0], alpha=0.5)
axes[0].plot(m, kspace.noncentral_chi_pdf(m, 0.0, sigma, 1), color=PALETTE[0], label="background (no signal)")
axes[0].hist(np.abs(noisy[wm]), bins=60, density=True, color=PALETTE[1], alpha=0.5)
axes[0].plot(m, kspace.noncentral_chi_pdf(m, a_wm, sigma, 1), color=PALETTE[1], label="white matter")
axes[0].set(title="single coil: magnitude distribution", xlabel="magnitude", xlim=(0, 0.35)); axes[0].legend()

axes[1].hist(sos[background], bins=60, density=True, color=PALETTE[2], alpha=0.5)
axes[1].plot(m, kspace.noncentral_chi_pdf(m, 0.0, sigma, NC), color=PALETTE[2], label=f"{NC}-coil sum of squares")
axes[1].set(title="multi-coil: background distribution", xlabel="magnitude", xlim=(0, 0.35)); axes[1].legend()

snr = np.linspace(0, 6, 200)
axes[2].plot(snr, kspace.rician_mean(snr * sigma, sigma) / sigma, label="measured magnitude")
axes[2].plot(snr, snr, color="0.5", lw=1, ls="--", label="true signal")
axes[2].set(title="bias of the magnitude", xlabel="true SNR", ylabel="value / noise SD"); axes[2].legend()
fig.tight_layout()
```

The right panel gives the practical numbers. The magnitude overestimates the true signal by
about 30 % at SNR 1, 5 % at SNR 3, and 2 % at SNR 5:

```{code-cell} python
:tags: [hide-input]
for s in [0.5, 1, 2, 3, 5, 10]:
    print(f"SNR {s:>4}: measured / true = {kspace.rician_mean(s * sigma, sigma) / (s * sigma):.3f}")
```

Diffusion-weighted images at high b-value routinely have SNR between 1 and 3. Their
magnitudes are therefore biased upward, the signal appears to decay less with b than it
does, and fitted diffusivities come out too low. Chapter 8 measures this on the phantom and
shows what denoising in the complex domain and noise-aware fitting do about it.

## See it and measure it: the TRXScan slab

:::{admonition} Phantom figure pending
:class: note
This section will load the `slab-kspace` dataset (8 coils, GRAPPA R = 2, partial Fourier
6/8, exported raw k-space), apply the GRAPPA, partial-Fourier, and coil-combination steps
above to the simulator's own k-space, and confirm that they reproduce TRXScan's reconstructed
magnitude and phase. It waits on the k-space export flag in the simulator (implementation
plan §4, item T2).
:::

## Complex data: what phase makes possible

Every step in this chapter before the magnitude operation is linear, and the noise stays
Gaussian with zero mean. Keeping the complex image, which BIDS supports as `part-mag` and
`part-phase` pairs and which TRXScan writes by default, preserves that property:

- **Denoising in the complex domain** operates on Gaussian noise without a floor, so the
  low-SNR high-b volumes that matter most for microstructure can be denoised without bias
  (Chapter 8).
- **Averaging** repeated acquisitions in the complex domain reduces noise toward zero;
  averaging magnitudes converges to the noise floor instead.
- **The phase is diagnostic.** Eddy-current and motion-related phase can be inspected per
  volume before any correction is attempted (Chapters 11 and 12).

The costs are twice the storage, a phase image that wraps and is not interpretable without
a reference, and a preprocessing pipeline that accepts complex input. Chapter 19 collects
what complex data provide across the whole analysis chain.

## What this implies for acquisition

- **Save the phase.** It costs storage and nothing else at the scanner, and it cannot be
  recovered afterward.
- **Parallel imaging is a noise decision.** $R = 2$ shortens the readout and TE, but the
  noise increases by more than $\sqrt{2}$ and becomes spatially structured.
- **Partial Fourier depends on smooth phase.** Moderate factors are safe; aggressive ones
  fail near sinuses and under strong eddy currents.
- **More coils raise the noise floor** of magnitude images even as they raise SNR, and the
  floor is what biases high-b measurements.

## Further reading

Phased arrays and their combination {cite:p}`roemer1990`, SENSE {cite:p}`pruessmann1999`,
GRAPPA {cite:p}`griswold2002`, homodyne reconstruction {cite:p}`noll1991`, compressed
sensing in MRI {cite:p}`lustig2007`, and the noise statistics of magnitude images
{cite:p}`gudbjartsson1995,constantinides1997`.
