---
title: Spatial encoding and k-space
subtitle: Chapter 2
kernelspec:
  name: python3
  display_name: Python 3
---

## Learning goals

After this chapter you can:

- explain what k-space is and why the scanner records the Fourier transform of the image
  rather than the image itself
- predict what happens to an image when k-space is sampled too coarsely or not far enough
- describe the single-shot EPI readout, compute its timing, and state how partial Fourier
  and in-plane acceleration change it
- recognize Gibbs ringing and know where it comes from
- explain why diffusion MRI uses single-shot EPI despite its drawbacks

**Datasets used:** `slab-kspace` (pending)
**Simulation tier:** toy + phantom

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt

from dwibook import kspace, phantoms
from dwibook.plotting import PALETTE, set_style, show_image, show_kspace

set_style()
img = phantoms.brain_image()  # synthetic b=0 slice of the phantom, adult preset, TE 88 ms
mask = phantoms.brain_slice()["mask"]
```

## Gradients as encoding

A gradient coil adds a field that varies linearly with position, so spins at different
positions precess at different frequencies. Switching a gradient on for a short time gives
every spin a phase that depends on its position along the gradient direction. The receiver
records the sum of all spins in the slice as one complex number (the quadrature-demodulated
signal of Chapter 1), and with a position-dependent phase applied that sum is one sample of
the Fourier transform of the image:

$$s(\mathbf{k}) = \int \rho(\mathbf{r})\, e^{-2\pi i\, \mathbf{k}\cdot\mathbf{r}}\, d\mathbf{r}.$$

The coordinate $\mathbf{k}$ is set by the gradient history: the longer and stronger the
gradients that have been applied, the farther from the center of k-space the current sample
lies. A pulse sequence is a plan for moving through k-space and recording samples along the
way; reconstruction is an inverse Fourier transform of those samples. Slice selection uses
the same principle during excitation: a gradient along the slice direction makes the RF pulse
resonant only within a slab.

This chapter assumes the gradient fields are exactly linear. They are not, and the
consequences for image geometry and for the diffusion encoding are the subject of Chapter 13.

## The Fourier relationship

The images in this chapter are a synthetic b=0 slice of the book's phantom: the tissue
fractions of one axial slice, weighted by proton density and T2 decay at the HBCD echo time
(Chapter 1). Its k-space is computed directly.

```{code-cell} python
ksp = kspace.fft2c(img)

fig, axes = plt.subplots(1, 3, figsize=(9, 3))
show_image(axes[0], img, "image (synthetic b=0)")
show_kspace(axes[1], ksp, "k-space magnitude (log scale)")
axes[2].imshow(np.angle(ksp), cmap="twilight"); axes[2].set_axis_off(); axes[2].set_title("k-space phase")
fig.tight_layout()
```

Most of the energy is at the center of k-space. The center encodes contrast and coarse
shape; the periphery encodes edges and fine detail. Reconstructing from only one or the other
shows the division:

```{code-cell} python
ny, nx = ksp.shape
c = ny // 2
low = np.zeros_like(ksp); low[c - 8 : c + 8, c - 8 : c + 8] = ksp[c - 8 : c + 8, c - 8 : c + 8]
high = ksp - low

fig, axes = plt.subplots(1, 2, figsize=(6, 3))
show_image(axes[0], kspace.ifft2c(low), "central 16 × 16 samples only")
show_image(axes[1], kspace.ifft2c(high), "everything except the center")
fig.tight_layout()
```

## FOV, resolution, and aliasing

Two numbers describe how k-space is sampled: the spacing between samples and the distance to
the outermost sample. The spacing sets the field of view (finer spacing, larger FOV), and the
outermost sample sets the voxel size (farther out, smaller voxels). Both limits have a
characteristic failure:

- Stopping too close to the center gives large voxels and blurred edges, with ringing
  (below).
- Sampling too coarsely for the size of the object makes the image wrap around on itself.
  This is aliasing, and it is the effect that parallel imaging (Chapter 3) deliberately
  induces and then removes.

```{code-cell} python
lowres = np.zeros_like(ksp); lowres[c - 16 : c + 16, c - 16 : c + 16] = ksp[c - 16 : c + 16, c - 16 : c + 16]
every_other = np.where(kspace.regular_undersampling_mask(ny, nx, 2), ksp, 0)

fig, axes = plt.subplots(1, 3, figsize=(9, 3))
show_image(axes[0], img, "full sampling")
show_image(axes[1], kspace.ifft2c(lowres), "outer k-space dropped: larger voxels")
show_image(axes[2], kspace.ifft2c(every_other), "every other line dropped: FOV halved")
fig.tight_layout()
```

## The EPI readout

A conventional sequence records one line of k-space per excitation and needs one excitation
per line. Echo-planar imaging (EPI) records the whole plane after a single excitation: the
readout gradient alternates direction, sweeping back and forth along $k_x$, while short
phase-encode blips step $k_y$ one line at a time. The path through k-space is a zigzag, and
the time between successive lines is the echo spacing.

```{code-cell} python
tr = kspace.epi_trajectory(16, 16, echo_spacing_ms=0.6)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [1, 1.4]})
for i, (ky, d) in enumerate(zip(tr.lines, tr.directions)):
    xs = np.arange(16) if d > 0 else np.arange(16)[::-1]
    ax1.plot(xs - 8, np.full(16, ky - 8), "-", color=plt.cm.viridis(i / 15), lw=1.5)
    if i < 15:
        ax1.annotate("", xy=(xs[-1] - 8, ky - 7), xytext=(xs[-1] - 8, ky - 8), arrowprops=dict(arrowstyle="->", color="0.5", lw=1))
ax1.set(xlabel="$k_x$", ylabel="$k_y$", title="EPI path through k-space (color = time)", aspect="equal")
ax1.grid(False)

n = 4
t = np.linspace(0, n * tr.echo_spacing_ms, 400)
gx = np.sign(np.sin(np.pi * t / tr.echo_spacing_ms))
blip = ((t % tr.echo_spacing_ms) > 0.92 * tr.echo_spacing_ms).astype(float)
ax2.plot(t, gx, label="readout gradient")
ax2.plot(t, 0.9 * blip - 2.2, label="phase-encode blips")
ax2.plot(t, np.cumsum(gx) / 60 - 4.2, label="$k_x$ position", color=PALETTE[6])
ax2.set(xlabel="time (ms)", yticks=[], title="gradient waveforms for the first four lines")
ax2.legend(loc="upper right")
fig.tight_layout()
```

The timing of the readout determines several artifacts. With $N_y$ lines and an echo spacing
of 0.5–1 ms, the readout lasts 40–90 ms. TRXScan uses the HBCD protocol's 91.7 ms total
readout time for any matrix size. For a 128-line matrix:

```{code-cell} python
esp = 91.7 / 128
for label, kw in [("full", {}), ("partial Fourier 6/8", {"partial_fourier": 0.75}),
                  ("R = 2", {"accel": 2}), ("6/8 and R = 2", {"partial_fourier": 0.75, "accel": 2})]:
    tr = kspace.epi_trajectory(128, 128, esp, **kw)
    print(f"{label:>20}: {tr.lines.size:3d} lines, readout {tr.readout_ms:5.1f} ms, "
          f"k-space center reached after {tr.time_to_center_ms:5.1f} ms")
```

Three consequences of the long readout, each treated in its own chapter:

- The phase-encode direction is sampled slowly, about one line per millisecond, so a small
  frequency offset displaces signal a long way along that axis. An off-resonance of 100 Hz
  moves signal by about ten voxels. This is susceptibility distortion (Chapter 10), and it is
  why the phase-encode direction and readout time must be recorded in the image metadata.
- Signal decays with T2* during the readout, so lines acquired late are weaker. The result is
  blurring along the phase-encode direction.
- Odd and even lines are read in opposite directions. A timing mismatch between them
  produces a faint copy of the image shifted by half the field of view, the Nyquist ghost
  (Chapter 14).

## Partial Fourier

For an object with no phase, k-space is symmetric about its center, so half of the lines are
redundant. Partial Fourier acquisitions skip a fraction of the lines on one side, typically
acquiring 5/8 to 7/8 of them. The timings printed above show the two benefits: the readout is
shorter, and the center of k-space is reached sooner, which shortens the minimum echo time.
The cost is that real images do have phase (from field inhomogeneity, coil phase, eddy
currents, and motion), so the symmetry is only approximate and the reconstruction has to
estimate the phase from the acquired part (Chapter 3). In-plane acceleration is the other way
to shorten the readout: keep every $R$-th line and recover the missing ones using multiple
receive coils.

## Truncation and Gibbs ringing

Every acquisition stops at some outermost k-space sample. A Fourier representation cut off
at a finite frequency overshoots at every sharp edge by about 9 % of the step, and the
overshoot does not shrink with more samples; it only moves closer to the edge.

```{code-cell} python
x = np.linspace(-1, 1, 2001)
step = (np.abs(x) < 0.5).astype(float)
fig, ax = plt.subplots(figsize=(7.5, 3))
for n_harm, color in zip([8, 16, 64], PALETTE[:3]):
    approx = 0.5 + sum((2 / (np.pi * k)) * np.sin(np.pi * k / 2) * np.cos(np.pi * k * x) for k in range(1, n_harm + 1))
    ax.plot(x, approx, color=color, label=f"{n_harm} frequencies", lw=1.5)
ax.plot(x, step, color="0.3", lw=1, ls="--", label="edge")
ax.set(xlabel="position", ylabel="intensity", xlim=(0.2, 0.8), title="An edge reconstructed from a limited number of frequencies")
ax.legend()
fig.tight_layout()
```

In an image, the overshoot appears as ripples parallel to every sharp boundary. In the brain
the sharpest boundaries are between CSF and tissue, so the ventricle walls and the cortical
surface ring most. The example below reconstructs the slice from a 64 × 64 k-space, a
resolution of 4 mm, on the 2 mm grid:

```{code-cell} python
crop = np.zeros_like(ksp); crop[c - 32 : c + 32, c - 32 : c + 32] = ksp[c - 32 : c + 32, c - 32 : c + 32]
ringing = np.abs(kspace.ifft2c(crop))
row = 58  # through the frontal horns of the lateral ventricles
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3), gridspec_kw={"width_ratios": [1, 1.6]})
show_image(ax1, ringing, "64 × 64 samples reconstructed on 128 × 128")
ax1.axhline(row, color=PALETTE[1], lw=1)
ax2.plot(img[row], color="0.3", lw=1, ls="--", label="object")
ax2.plot(ringing[row], color=PALETTE[1], lw=1.5, label="truncated")
ax2.set(xlabel="x (voxels)", ylabel="intensity", xlim=(20, 108), title="profile along the marked row")
ax2.legend()
fig.tight_layout()
```

The ringing is small in absolute terms but it sits exactly where CSF meets tissue, and it
changes with b-value because the CSF signal changes with b-value. Its effect on diffusion
metrics at tissue borders, and the correction for it, are covered in Chapter 9.

## Why diffusion MRI uses single-shot EPI

Splitting the lines of k-space over several excitations (multi-shot EPI) shortens each
readout, reducing distortion and blur and allowing higher resolution. Diffusion encoding
makes this difficult. The diffusion gradients are strong enough that small bulk movements of
the head during the encoding, including pulsation, give each shot a different, unknown phase.
In a single-shot acquisition that phase is common to every line and has no effect on the
magnitude image. In a multi-shot acquisition the shots disagree, and the disagreement appears
as ghosts:

```{code-cell} python
shot = np.arange(ny) % 2  # two interleaved shots
phase_error = np.exp(1j * np.deg2rad(60))  # the second shot acquired with a 60° bulk phase
ksp_multishot = np.where(shot[:, None] == 1, ksp * phase_error, ksp)

fig, axes = plt.subplots(1, 2, figsize=(6, 3))
show_image(axes[0], kspace.ifft2c(ksp), "single shot")
show_image(axes[1], kspace.ifft2c(ksp_multishot), "two shots with a 60° phase difference")
fig.tight_layout()
```

Multi-shot diffusion imaging is possible with navigator echoes or with reconstructions that
estimate the per-shot phase (Chapter 23), but the standard acquisition remains single-shot
EPI. Its long readout is the origin of most of the artifacts corrected in Part III.

## Measure it: a TRXScan slice and its k-space

:::{admonition} Phantom figure pending
:class: note
This section will load the `slab-kspace` dataset, show one acquired slice with the raw
k-space TRXScan recorded for it, and derive the EPI timing from the BIDS JSON sidecar
(`EffectiveEchoSpacing`, `TotalReadoutTime`). It waits on the k-space export flag in the
simulator (implementation plan §4, item T2).
:::

## What this implies for acquisition

- **Voxel size and FOV are k-space decisions.** Smaller voxels require more lines, a longer
  readout, and therefore more distortion and blur.
- **Readout length drives the main EPI artifacts.** Partial Fourier and in-plane
  acceleration both shorten it and both reduce the minimum TE. Chapter 7 discusses their costs.
- **Ringing is a property of every acquisition**, not a malfunction. It is worst at CSF
  boundaries and can be reduced after the fact (Chapter 9).
- **Single-shot EPI is a compromise** accepted so that diffusion encoding is robust to
  motion. The metadata that describe the readout, `PhaseEncodingDirection` and
  `TotalReadoutTime`, are required by the corrections in Part III and should be checked
  before any processing.

## Further reading

The k-space description of MRI {cite:p}`ljunggren1983,twieg1983`, echo-planar imaging
{cite:p}`mansfield1977`, and the textbooks {cite:t}`haacke1999` and {cite:t}`nishimura2010`.
