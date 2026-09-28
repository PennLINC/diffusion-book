---
title: "8b. Real-valued diffusion MRI"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** the synthetic series of [Chapter 8](./08-noise.md) with a simulated object phase ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
- **`noise-sweep`** (pending): four noise levels with one coil, plus an 8-coil GRAPPA run ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-noise-sweep)).
- **`truth`** (pending): the 27 analytic ground-truth maps and the true fiber orientations ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-truth), [Appendix E](../appendices/e-truth-map-catalogue.md)).

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
from dwibook.plotting import PALETTE, set_style, show_image

set_style()
```

## The idea

The noise floor of [Chapter 8](./08-noise.md) exists because the magnitude of a complex number is never
negative: noise that would have pushed a weak signal below zero is folded back up. The
complex image has no such floor, but its signal does not lie along one axis: every voxel
carries a phase from the field offset, the coils, the eddy currents, and motion during the
encoding ([Chapter 3](../01-mri-physics/03-reconstruction.md)), and that phase varies across the image and between volumes.

If the phase is smooth, it can be estimated and removed. Rotating each voxel's complex value
by the negative of its estimated phase puts the signal on the real axis, and the real part
of the result is then a real-valued image in which the signal is signed and the noise is
Gaussian with zero mean {cite:p}`eichner2015`. Weak signals average to their true value
instead of to a floor; high-b images can be averaged and fitted without bias; and nothing
about the acquisition changes, only that the phase is saved.

## Estimating the background phase

The phase to remove is the slowly varying one. A low-pass filter of the complex image (a
Gaussian blur of the real and imaginary parts separately, then the angle of the result)
estimates it while averaging out the noise; the width of the filter is the one assumption,
and it must be wide enough to suppress noise and narrow enough to follow the true phase.

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
show_image(axes[2], magnitude[:, :, K, v3], "magnitude", vmin=-0.05, vmax=0.12)
show_image(axes[3], real_valued[:, :, K, v3], "phase-corrected real part", vmin=-0.05, vmax=0.12)
fig.tight_layout()
```

The real-valued image has negative voxels in the background and in the darkest tissue,
which is the point: noise is allowed to go both ways.

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
has a positive mean. Fits to real-valued data need one adjustment: values can be negative,
so a fit to the logarithm of the signal (the linear tensor fit of [Chapter 15](../04-modeling/15-signal-representations.md)) is not
possible, and nonlinear fits to the signal itself are used instead.

## When it fails

- **Rough phase.** Near air-tissue interfaces, and in volumes with strong eddy-current or
  motion phase, the true phase varies faster than the filter allows. The estimate is then
  wrong, part of the signal is rotated onto the imaginary axis and lost, and the real part
  is biased downward. A per-volume check of the imaginary residual shows where this happens.
- **Partial Fourier.** The reconstruction already assumed a smooth phase ([Chapter 3](../01-mri-physics/03-reconstruction.md)); the
  two assumptions compound, and the phase estimate should be made after the partial-Fourier
  reconstruction, from the complex image it produced.
- **Multi-coil combination.** The phase must survive the combination: a sensitivity-weighted
  combination keeps it, a root-sum-of-squares discards it ([Chapter 3](../01-mri-physics/03-reconstruction.md)).

Real-valued conversion and complex-domain denoising ([Chapter 8](./08-noise.md)) address the same problem
from two sides and are often combined: denoise the complex data, then phase-correct and take
the real part.

## Measure it: the simulated datasets

:::{admonition} Simulated dataset pending
:class: note
This section will apply the phase correction to the `noise-sweep` dataset, whose phase
images carry the simulator's object phase and eddy-current phase ramp, and compare the
real-valued fits with the `truth` maps where the magnitude fits were biased.
:::

## What this implies for acquisition

- **Save the phase** ([Chapter 3](../01-mri-physics/03-reconstruction.md)); nothing else at the scanner is required.
- **Prefer a combination that keeps the phase** if the coil combination is configurable.
- **Keep partial Fourier moderate**, so that the phase assumption both reconstructions rely
  on holds.

## Further reading

Real-valued diffusion MRI {cite:p}`eichner2015` and complex-domain denoising
{cite:p}`corderogrande2019`.
