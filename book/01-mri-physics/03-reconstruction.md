---
title: "3. Image reconstruction from k-space"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** the k-space of a synthetic b=0 slice, with simulated coil sensitivities, undersampling, and partial Fourier ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
- **`slab-kspace`** (pending): a five-slice slab with its raw k-space exported: 8 coils, GRAPPA 2, partial Fourier 6/8 ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-slab-kspace)).

Pipeline-tier datasets are simulated offline by TRXScan ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)) and are marked *pending* until their release; the figures that need them say so where they will appear.
:::

## Learning goals

After this chapter you can:

- describe what a reconstructed diffusion image contains beyond its magnitude, and what is
  lost when only the magnitude is saved
- explain how multiple receive coils are combined and how they are used to shorten the
  readout (parallel imaging), and what that costs in noise
- explain how partial-Fourier data are completed, and when that fails
- describe the noise in a magnitude image, and state how large the resulting bias is at the
  signal levels typical of high b-value diffusion data

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

from dwibook import kspace, phantoms
from dwibook.plotting import INK, PALETTE, complex_noise_cloud, set_style, show_image, show_kspace

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
in [Chapter 1](./01-spins-and-signal.md). The magnitude is the image that is normally viewed and analyzed. The phase is
measured relative to the receiver's reference, so a constant offset is arbitrary; its spatial
and temporal variations are not. The phase is usually discarded. In diffusion MRI the
phase carries information that is used later in this book: the static field offset
([Chapter 10](../03-preprocessing/10-susceptibility-distortion.md)), the eddy-current phase that changes with each diffusion direction
([Chapter 11](../03-preprocessing/11-eddy-currents.md)), and the phase acquired through motion during the diffusion encoding
([Chapter 12](../03-preprocessing/12-motion-and-dropout.md)). Saving only the magnitude removes all of it.

```{code-cell} python
:tags: [hide-input]
rec = kspace.ifft2c(kspace.fft2c(img))
fig, axes = plt.subplots(1, 3, figsize=(9, 3))
show_kspace(axes[0], kspace.fft2c(img), "acquired k-space (log scale)")
show_image(axes[1], np.abs(rec), "magnitude")
axes[2].imshow(np.where(mask, np.angle(rec), np.nan), cmap="twilight", vmin=-np.pi, vmax=np.pi); axes[2].set_axis_off(); axes[2].set_title("phase (radians)")
fig.tight_layout()
```

What to look at: the magnitude shows the anatomy, and the phase (shown inside the brain
only; the color wheel runs once around from −π to π) shows almost none of it. It is a
smooth wash of color across the head, the kind of phase a slightly uneven magnetic field
leaves. Smooth, slowly varying phase is the normal case, and several steps below rely on it.

## Multi-coil combination

Modern scanners receive with an array of many small coils. Each coil records the same object
weighted by its own sensitivity, a smooth field that is strongest near the coil. Each coil's
voltage is demodulated separately, and differences in cable length and electronics give each
channel its own phase offset, so the sensitivities are complex. The
reconstruction has to combine the coil images into one, and the two standard methods differ
in what they require:

- **Root sum of squares** requires no information about the coils. It is a magnitude
  combination, it inherits a smooth intensity variation across the image, and it raises the
  noise floor as the number of coils grows, roughly with its square root (see the noise
  section).
- **Sensitivity-weighted combination** (Roemer) requires the coil sensitivities, which are
  measured in a calibration scan or estimated from the center of k-space. It gives the best
  SNR and preserves the phase.

The coil model has the same form as TRXScan's: coils on a ring around the head, each with a
Gaussian falloff, plus a smooth phase per coil. TRXScan's default falloff is broad, so that
each of its coils sees the whole head almost evenly. The coils here fall off as the elements
of a real head array do: each one sees the cortex beneath it brightly, and the far side of
the head about seventeen times more weakly. Each coil image is shown on its own intensity
scale.

```{code-cell} python
:tags: [hide-input]
NC = 8
sens = kspace.ring_coil_sensitivities(NC, N, N, width=0.3, floor=0.02)
coil_ksp = kspace.fft2c(sens * img)
coil_img = kspace.ifft2c(coil_ksp)

fig, axes = plt.subplots(3, 4, figsize=(10, 7.5))
for c in range(NC):
    show_image(axes[c // 4, c % 4], coil_img[c], f"coil {c + 1} of {NC}")
show_image(axes[2, 0], np.abs(sens[0]), "sensitivity of coil 1")
show_image(axes[2, 1], kspace.sos_combine(coil_img), "root sum of squares")
show_image(axes[2, 2], kspace.roemer_combine(coil_img, sens), "sensitivity-weighted")
axes[2, 3].imshow(np.where(mask, np.angle(kspace.roemer_combine(coil_img, sens)), np.nan), cmap="twilight"); axes[2, 3].set_axis_off(); axes[2, 3].set_title("phase, sensitivity-weighted")
fig.tight_layout()

rim = mask & ~ndimage.binary_erosion(mask, iterations=4)          # outer 4 voxels of the brain
core = mask & (np.hypot(*(np.mgrid[:N, :N] - N / 2)) < 12)         # the middle of the brain
sos_weight = np.sqrt((np.abs(sens) ** 2).sum(0))                 # what root sum of squares multiplies the image by
print(f"root sum of squares weights the edge of the brain {sos_weight[rim].mean() / sos_weight[core].mean():.1f} times "
      f"more than its middle; the sensitivity-weighted image weights them equally")
```

What to look at: each coil image is bright on its own side and dark on the far side. The two
combinations look alike at a glance, but the root-sum-of-squares image carries the coils'
combined sensitivity as a brightness that rises toward the edge of the head: the cortex
looks brighter than deep gray matter of the same kind, which the sensitivity-weighted image
does not show. In your own data a bright outer rim that is not anatomy is this weighting.
It is harmless for diffusion measures, which are ratios of images with the same weighting
(as long as the head does not move within the coils), but it matters for anything that compares raw intensities across the brain.

## Parallel imaging

Because each coil views the object from a different position, the set of coil images
contains spatial information that can substitute for some of the phase-encode lines.
Acquiring only every $R$-th line shortens the EPI readout by a factor of $R$ ([Chapter 2](./02-spatial-encoding-kspace.md)). The
raw result is an image folded over on itself $R$ times, in every coil. Two families of
reconstruction unfold it:

- **SENSE** works in the image domain. At each pixel of the folded image, the $R$ true pixels
  that overlap there are unknowns, and the coil sensitivities at those positions provide the
  equations to solve for them.
- **GRAPPA** works in k-space and does not need sensitivity maps. A block of fully sampled
  central lines (the autocalibration signal, ACS) is used to learn how each missing line can
  be predicted from neighboring acquired lines across all coils; the learned weights are then
  applied throughout k-space. TRXScan simulates GRAPPA with 24 ACS lines, so the simulated brain data
  in later chapters went through this reconstruction. The simulator acquires the ACS lines
  inside each volume's EPI train, as the figures here do; scanners usually acquire them
  once, in a separate short calibration scan before the diffusion series, so that the
  diffusion readouts stay as short as possible.

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

Without noise, both methods recover the image almost exactly: the folded image on the left
has become the unfolded brain, and the difference panel is nearly empty.

The next figure takes the SENSE step apart for one position. Skipping every second line
halves the field of view, so two points of the object half a field of view apart (the
circles, 64 voxels apart) land on the same point of the folded image, and one measured
value holds the sum of both. A single coil gives one equation in two unknowns, which cannot
be solved. A second coil gives a second equation with different weights, because it sees the
two points with different sensitivities: the coil at the back of the head sees the lower
point strongly and the upper point weakly, the coil at the front the reverse. Two equations
in two unknowns can be solved. The folded panels show one half of the field of view, the
whole of what an $R = 2$ acquisition records.

```{code-cell} python
:tags: [hide-input]
p1, p2, col = 34, 34 + N // 2, 60           # two points half a field of view apart, same column
back, front = 2, 6                          # coil 3 sits behind the head (bottom of the image), coil 7 in front
sw = sens[[back, front]][:, [p1, p2], col]  # the weights: each coil's sensitivity at the two points
meas = aliased[[back, front], p1, col]      # what each coil measured at the folded point
rho = np.linalg.solve(sw, meas)
print(f"true values at the two points: {abs(img[p1, col]):.3f} (upper), {abs(img[p2, col]):.3f} (lower)")
print(f"solved from coils 3 and 7:     {abs(rho[0]):.3f} (upper), {abs(rho[1]):.3f} (lower)")

fig, axes = plt.subplots(1, 4, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1, 1, 1, 1.15]})
show_image(axes[0], np.abs(ref), "full FOV: two points 64 rows apart")
for p, color in [(p1, PALETTE[1]), (p2, PALETTE[2])]:
    axes[0].add_patch(plt.Circle((col, p), 4, fill=False, color=color, lw=2))
axes[0].annotate("", xy=(118, p2), xytext=(118, p1), arrowprops=dict(arrowstyle="<->", color="white", lw=1))
for ax, c in zip(axes[1:3], (back, front)):
    show_image(ax, np.abs(aliased[c, : N // 2]), f"coil {c + 1}, folded (R = 2)", vmin=0, vmax=np.abs(aliased[c]).max())
    ax.add_patch(plt.Circle((col, p1), 4, fill=False, color=PALETTE[3], lw=2))
axes[3].set_axis_off()
eq = (f"one folded point, two coils\n(m: measured, ρ: true value):\n\n"
      f"coil 3:  m₃ = {abs(sw[0, 0]):.2f}·ρ(upper) + {abs(sw[0, 1]):.2f}·ρ(lower)\n"
      f"coil 7:  m₇ = {abs(sw[1, 0]):.2f}·ρ(upper) + {abs(sw[1, 1]):.2f}·ρ(lower)\n\n"
      f"(weights = coil sensitivities at the two\n points, magnitudes shown; the true\n weights are complex)\n\n"
      f"solve: ρ(upper) = {abs(rho[0]):.2f},  ρ(lower) = {abs(rho[1]):.2f}")
axes[3].text(0.0, 0.5, eq, transform=axes[3].transAxes, va="center", ha="left", fontsize=9, family="monospace")
fig.tight_layout()
```

GRAPPA reaches the same result without ever forming the folded image or the sensitivity
maps. It works line by line in k-space: each missing sample, in every coil, is predicted as
a weighted sum of acquired samples nearby, from every coil (the kernel in the right panel
below). The weights are learned where the answer is known, in the fully sampled band at
the center of k-space (the ACS lines in the left panel), and then the same weights are slid
over the rest of k-space to fill every missing line.

```{code-cell} python
:tags: [hide-input]
fig, (ax_m, ax_k) = plt.subplots(1, 2, figsize=(10.5, 4), gridspec_kw={"width_ratios": [1, 1.45]})
ax_m.imshow(mask_r2[:, :1].repeat(24, 1), cmap="Greys", vmin=0, vmax=1.6, aspect="auto", interpolation="nearest")
ax_m.set(xticks=[], ylabel="$k_y$ line", title="lines acquired (dark) at R = 2")
lo, hi = N // 2 - ACS // 2, N // 2 + ACS // 2
ax_m.annotate("", xy=(26, lo - 0.5), xytext=(26, hi - 0.5), annotation_clip=False,
              arrowprops=dict(arrowstyle="-", color=PALETTE[1], lw=3))
ax_m.text(29, N / 2, f"{ACS} ACS lines:\nevery line acquired,\nweights learned here", color=PALETTE[1],
          va="center", fontsize=8.5, clip_on=False)
ax_m.set_xlim(-0.5, 23.5)

rows, cols = np.arange(9), np.arange(7)
for r in rows:
    acq = r % 2 == 0
    ax_k.plot(cols, np.full(cols.size, r), "o", ms=9, color=INK["primary"] if acq else "white",
              mec=INK["primary"], mew=1)
tr, tc = 5, 3                                   # the missing sample to predict
src = [(r, c) for r in (tr - 3, tr - 1, tr + 1, tr + 3) for c in (tc - 1, tc, tc + 1)]
for r, c in src:
    ax_k.plot(c, r, "o", ms=9, color=PALETTE[0])
    ax_k.annotate("", xy=(tc, tr), xytext=(c, r), arrowprops=dict(arrowstyle="-", color=PALETTE[0], lw=0.6, alpha=0.6))
ax_k.plot(tc, tr, "*", ms=20, color=PALETTE[1], mec="white")
ax_k.set(xlim=(-0.8, 12.2), ylim=(8.8, -0.8), xticks=[], yticks=[], title="the GRAPPA kernel, in one coil's k-space")
for s in ax_k.spines.values():
    s.set_visible(False)
ax_k.grid(False)
ax_k.text(6.9, 1.2, "● acquired\n○ skipped", fontsize=9, va="center")
ax_k.text(6.9, 4.1, "blue: 12 acquired\nsamples (4 lines × 3)\nin each of the 8 coils", color=PALETTE[0], fontsize=9, va="center")
ax_k.text(6.9, 7.2, "star: the missing sample,\npredicted as a weighted\nsum of all 96", color=PALETTE[1], fontsize=9, va="center")
ax_k.set_xlabel("$k_x$ →", loc="left")
ax_k.set_ylabel("$k_y$ ↓", loc="top")
fig.tight_layout()
```

The two methods differ in what they take as known (sensitivity maps for SENSE, a band of
calibration lines for GRAPPA) but make the same bet: the coils see the object differently
enough to tell the folded copies apart. The cost of parallel imaging is
noise. Fewer samples raise the noise by the square root of the reduction in samples: $\sqrt{R}$,
or slightly less when the ACS lines are acquired in the same scan, as they are here. The
unfolding amplifies it further by a spatially varying factor (the g-factor) that depends on
how different the coil sensitivities are at the positions that fold together; coils with
broad, overlapping sensitivities give a large g-factor. The amplification can be measured
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
      f"ratio {noise_r2 / noise_full:.2f}")
print(f"from the fewer samples alone ({mask_r2[:, 0].sum()} of {N} lines): {np.sqrt(N / mask_r2[:, 0].sum()):.2f}; "
      f"the rest, {noise_r2 / noise_full / np.sqrt(N / mask_r2[:, 0].sum()):.2f}, is the average g-factor")
```

An average hides where the amplification happens. Repeating the measurement with many
different noise draws gives the noise level of every voxel separately. Each map below is
that noise level after GRAPPA, divided by the noise level of the fully sampled scan and by
the square root of the reduction in samples, so what remains is the g-factor: 1 where the
unfolding costs nothing beyond the fewer samples, higher where it amplifies the noise. The
right map repeats the measurement at $R = 3$, where the three folded copies are a third of
the field of view apart.

```{code-cell} python
:tags: [hide-input]
N_REP = 40
acs_band = (N // 2 - ACS // 2, N // 2 + ACS // 2)
inner = ndimage.binary_erosion(mask, iterations=3)
var_full = sum(np.abs(kspace.roemer_combine(kspace.ifft2c(kspace.add_complex_noise(coil_ksp, sigma, seed=100 + rep)), sens) - ref) ** 2
               for rep in range(N_REP))
sd_full = ndimage.uniform_filter(np.sqrt(var_full / N_REP), 5)    # 5 × 5 smoothing tames the scatter of 40 draws
gmaps = {}
for r_acc in (2, 3):
    m_acc = kspace.regular_undersampling_mask(N, N, r_acc, acs_lines=ACS)
    var = np.zeros((N, N))
    for rep in range(N_REP):
        nz = np.where(m_acc, kspace.add_complex_noise(coil_ksp, sigma, seed=100 + rep), 0)
        var += np.abs(kspace.roemer_combine(kspace.ifft2c(kspace.grappa_reconstruct(nz, m_acc, r_acc, acs=acs_band)), sens) - ref) ** 2
    gmaps[r_acc] = ndimage.uniform_filter(np.sqrt(var / N_REP), 5) / sd_full / np.sqrt(N / m_acc[:, 0].sum())
    g_in = gmaps[r_acc][inner]
    print(f"R = {r_acc} ({m_acc[:, 0].sum()} of {N} lines): g-factor inside the brain, median {np.median(g_in):.2f}, "
          f"95th percentile {np.percentile(g_in, 95):.2f}, maximum {g_in.max():.2f}")
rows_mid = inner & (np.abs(np.mgrid[:N, :N][0] - N // 2) < 8)
print(f"R = 3, the 16 rows through the middle of the head: median g-factor {np.median(gmaps[3][rows_mid]):.2f}")

fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1, 1.2]})
show_image(axes[0], np.abs(ref), "the slice")
for ax, r_acc in zip(axes[1:], (2, 3)):
    im = ax.imshow(np.where(inner, gmaps[r_acc], np.nan), cmap="magma", vmin=0.8, vmax=2.4)
    ax.set_axis_off(); ax.set_title(f"g-factor, R = {r_acc} with GRAPPA")
fig.colorbar(im, ax=axes[2], fraction=0.046, label="g-factor")
fig.tight_layout()
```

At $R = 2$ the map is close to 1 almost everywhere (median 1.06, nowhere above 1.22): two
points half a field of view apart are always seen very differently by this ring of coils,
as the front and back coils of the SENSE example showed. At $R = 3$ the amplification
concentrates in a horizontal band across the middle of the head, at right angles to the
phase-encode direction, with a median of 1.79 there and up to 2.48. There the coils are all
far away, their sensitivities are weak and alike, and the three folded copies are hard to
tell apart. That is where an accelerated diffusion image is noisiest, and an average over
the brain (median 1.33) understates it.

In practice, $R = 2$ is common in diffusion protocols because the shorter readout reduces
distortion and TE; $R = 3$ and above are used with caution because the noise penalty grows
quickly with the coil geometry of a head array.

## Partial Fourier reconstruction

For an object with no phase, k-space is symmetric about its center, so half of the lines are
redundant. A partial-Fourier acquisition ([Chapter 2](./02-spatial-encoding-kspace.md)) skips a fraction of the lines on one side,
typically acquiring 5/8 to 7/8 of them, to shorten the readout and the echo time. The figure
shows the full k-space of the slice, the same k-space with the first quarter and the first
three eighths of the phase-encode lines skipped, and the image reconstructed from each when
the missing lines are simply left at zero:

```{code-cell} python
:tags: [hide-input]
ksp_obj = kspace.fft2c(obj)
pcol = 60  # a column through the lateral ventricles
fig = plt.figure(figsize=(10.5, 9.8))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.7])
axp = fig.add_subplot(gs[2, :])
axp.plot(obj[:, pcol], color="0.3", lw=1, ls="--", label="object")
for col, (label, frac, color) in enumerate([("full", 1.0, None), ("partial Fourier 6/8", 0.75, PALETTE[0]),
                                             ("partial Fourier 5/8", 0.625, PALETTE[1])]):
    pf_mask = kspace.partial_fourier_mask(N, N, frac)
    acquired = np.where(pf_mask, ksp_obj, 0)
    zf = np.abs(kspace.ifft2c(acquired))
    show_kspace(fig.add_subplot(gs[0, col]), acquired, f"k-space: {label} ({pf_mask[:, 0].sum()} of {N} lines)")
    ax_im = fig.add_subplot(gs[1, col])
    show_image(ax_im, zf, f"image: {label}, zero-filled", vmin=0, vmax=1)
    for y0, y1 in [(0, 10), (N - 11, N - 1)]:            # mark the column above and below the head
        ax_im.plot([pcol, pcol], [y0, y1], color=PALETTE[3], lw=2)
    if color:
        axp.plot(zf[:, pcol], color=color, lw=1.5, label=f"{label}, zero-filled")
axp.set(xlim=(30, 95), xlabel="row (voxels, along phase-encode)", ylabel="intensity",
        title="profile down the marked column, through the ventricles")
axp.legend(loc="upper right", fontsize=8)
fig.tight_layout()
```

Left at zero, the missing lines cost resolution along the phase-encode axis: the image blurs
from top to bottom, more so the more lines are skipped. The profile at the bottom shows it
along one column: the sharp steps at the ventricle walls become gentler slopes, and the
narrow bright peaks of CSF lose height. The symmetry can recover them, but
real images do have phase (from field inhomogeneity, coil phase, eddy currents, and motion),
so the symmetry is only approximate and the reconstruction has to estimate the phase from
the acquired part. The methods differ in what they assume about that phase:

- **Zero filling** assumes nothing and accepts the blur.
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
wm_signal = obj[phantoms.brain_slice()["wm"] > 0.95].mean()   # mean white matter signal, the yardstick for the errors

fig, axes = plt.subplots(2, 3, figsize=(9, 6))
for ax_top, ax_bot, (name, r) in zip(axes[0], axes[1], recs.items()):
    err = np.abs(np.abs(r) - obj)
    show_image(ax_top, np.abs(r), f"{name}, 5/8")
    show_image(ax_bot, err, f"error: mean {err[mask].mean() / wm_signal * 100:.1f} % of WM", vmin=0, vmax=0.1)
    print(f"{name:>12}: mean error inside the brain {err[mask].mean() / wm_signal * 100:.1f} % of the white matter signal")
fig.tight_layout()
```

The error panels share one scale, from zero (black) to half the white matter signal
(white), and the printed errors are relative to the white matter signal. Zero filling
leaves the blur of the previous figure, strongest at every edge; homodyne removes most of
it but leaves faint vertical streaks; POCS comes closest to the object.

The two phase-aware methods recover the sharpness that zero filling loses. Both fail in the
same situation: where the phase changes faster than the fully sampled center can resolve,
the estimate is wrong and the reconstruction shows ringing and signal loss at that location.
In diffusion data this occurs near air–tissue interfaces and wherever eddy currents or
motion imprint sharp phase changes. This is one reason partial Fourier factors are kept
moderate (6/8 or 7/8) in diffusion protocols.

The figure below forces the failure. A vertical band down the middle of the slice is given a
phase that swings back and forth many times from top to bottom, much faster than the
smooth phase elsewhere. It is an exaggerated stand-in for the steep phase that the
air-filled sinuses and ear canals put on nearby tissue.

```{code-cell} python
:tags: [hide-input]
yy, xx = np.mgrid[0:N, 0:N] / N - 0.5
band = np.abs(xx) < 0.12
fast_phase = np.exp(1j * (phantoms.brain_phase() + 14 * np.sin(6 * np.pi * yy) * band))
full_fast = kspace.fft2c(obj * fast_phase)
hd_fast = np.abs(kspace.homodyne(full_fast, pf_mask))
err_fast = np.abs(hd_fast - obj)
print(f"homodyne error, mean inside the band {err_fast[mask & band].mean() / wm_signal * 100:.0f} % of the white matter signal; "
      f"elsewhere in the brain {err_fast[mask & ~band].mean() / wm_signal * 100:.1f} %")
fig, axes = plt.subplots(1, 3, figsize=(9, 3))
axes[0].imshow(np.where(mask, np.angle(obj * fast_phase), np.nan), cmap="twilight"); axes[0].set_axis_off(); axes[0].set_title("a rapidly varying phase band")
show_image(axes[1], hd_fast, "homodyne reconstruction, 5/8", vmin=0, vmax=1)
show_image(axes[2], err_fast, "error", vmin=0, vmax=0.3)
fig.tight_layout()
```

Outside the band the reconstruction is as good as before. Inside it the smooth-phase
assumption is wrong, and the band is filled with horizontal streaks and dark gaps: signal
has been lost or put back in the wrong place along phase-encode. In real data the same thing
happens, less regularly, in the frontal lobe above the sinuses and in the temporal lobes
next to the ear canals.

## Noise in magnitude images

Thermal noise enters the data as independent Gaussian noise in k-space, and the Fourier
transform preserves that: the complex image also has Gaussian noise with zero mean. A
voxel's measured value is therefore a complex number, the true value plus a random complex
offset. Imagine measuring the same voxel many times. Each measurement is a point in the
complex plane (real part horizontal, imaginary part vertical), and together they form a
round cloud centered on the true value, with a radius set by the noise standard deviation
$\sigma$. The magnitude of a measurement is its distance from the origin.

The figure measures three voxels 3000 times each, with the same noise and a true signal of
6, 1, and 0 times $\sigma$ (the signal-to-noise ratio, SNR). The top row is the complex
plane: gray dots are the measurements, the black × is the true value, the black + is the
origin (with no signal the two coincide), and the yellow line is the distance from the origin to one measurement, swung down
onto the real axis by the dashed arc to show the magnitude it gives. The bottom row is the
histogram of all the magnitudes, on the same horizontal scale as the plane above it, with
the true value (dashed) and the mean magnitude (yellow) marked. Watch the gap between the
two lines as the cloud moves toward the origin.

```{code-cell} python
:tags: [hide-input]
fig, axes = plt.subplots(2, 3, figsize=(11, 5.0), gridspec_kw={"height_ratios": [1.1, 1]})
stats = {}
for col, (snr, label) in enumerate([(6, "strong signal"), (1, "weak signal"), (0, "no signal")]):
    stats[snr] = complex_noise_cloud(axes[0, col], axes[1, col], snr, sigma=1.0, seed=3, extent=(-4, 10, -4, 4))
    axes[0, col].set_title(f"{label}: SNR {snr}")
    axes[1, col].set_xlabel("magnitude (units of σ)")
fig.tight_layout()
for snr in [6, 1, 0]:
    s = stats[snr]
    print(f"SNR {snr}: true value {s['true']:.2f} σ, mean magnitude {s['mean_magnitude']:.2f} σ")
```

For the strong signal the cloud is far from the origin. Moving a measurement up or down
barely changes its distance from the origin, moving it left or right changes that distance
by the same amount, so the magnitudes scatter symmetrically around the true value and their
mean is correct. For a weak signal the cloud reaches the origin. A measurement pushed past
the origin by the noise does not give a negative magnitude; it gives a positive one on the
other side, because a distance cannot be negative. Every such measurement adds to the
average instead of cancelling the ones that landed high, so the mean magnitude is too
large: 1.57 σ instead of 1 σ here. With no signal at all the cloud surrounds the origin,
and the magnitudes still average 1.26 σ in this sample (1.25 σ in the long run), although
the true value is zero. That offset is the **noise floor**,
and the upward error it causes is the **magnitude bias**.

Statisticians have names for these distributions {cite:p}`gudbjartsson1995`:

- With a single coil the magnitude follows a **Rician** distribution. With no signal it
  reduces to the **Rayleigh** distribution, whose mean is $\sigma\sqrt{\pi/2} \approx 1.25\,\sigma$:
  the noise floor. Where the signal is strong, the noise is approximately Gaussian again.
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
about 55 % at SNR 1, 6 % at SNR 3, and 2 % at SNR 5:

```{code-cell} python
:tags: [hide-input]
for s in [0.5, 1, 2, 3, 5, 10]:
    print(f"SNR {s:>4}: measured / true = {kspace.rician_mean(s * sigma, sigma) / (s * sigma):.3f}")
```

Diffusion-weighted images at high b-value routinely have SNR between 1 and 3. Their
magnitudes are therefore biased upward, the signal appears to decay less with b than it
does, and fitted diffusivities come out too low. The first section of [Chapter 8](../03-preprocessing/08-noise.md) plots exactly
this, the true and the measured signal of one white matter voxel against b, measures the
effect on the simulated datasets, and shows what denoising in the complex domain and
noise-aware fitting do about it.

## See it and measure it: the TRXScan slab

:::{admonition} Simulated dataset pending
:class: note
This section will load the `slab-kspace` dataset (five slices, 8 coils, GRAPPA R = 2,
partial Fourier 6/8, with the raw k-space exported), apply the GRAPPA, partial-Fourier, and
coil-combination steps above to the simulator's own k-space, and compare the result with
TRXScan's reconstructed magnitude and phase. It will appear when the dataset is released
([Appendix A](../appendices/a-trxscan-cookbook.md#ds-slab-kspace)). A single slice of such
k-space, simulated live, is shown at the end of [Chapter 2](./02-spatial-encoding-kspace.md).
:::

## Complex data: what phase makes possible

The Fourier transform, GRAPPA, and a sensitivity-weighted coil combination are all linear:
each output value is a weighted sum of the measured samples, so the noise in the complex
image stays Gaussian with zero mean. Not every step in this chapter is. A root-sum-of-squares
combination is itself a magnitude, and the phase-aware partial-Fourier reconstructions
(homodyne and POCS) estimate the phase from the data and use it, so they are not linear
either. Keeping the complex image, which BIDS {cite:p}`gorgolewski2016` supports as `part-mag` and `part-phase` pairs
and which TRXScan writes by default, keeps the zero-mean noise available to what comes next,
with one condition. The diffusion encoding leaves each volume a phase of its own, from
motion during the encoding, that changes from volume to volume and from repeat to repeat
(the same random phase that ruled out multi-shot EPI in [Chapter 2](./02-spatial-encoding-kspace.md)).
Complex values from different volumes cannot be combined until that phase has been
estimated and removed, which is what [Chapter 8b](../03-preprocessing/08-real-valued-dwi.md)
does. After that:

- **Denoising in the complex domain** operates on Gaussian noise without a floor, so the
  low-SNR high-b volumes that matter most for microstructure can be denoised without bias
  ([Chapter 8](../03-preprocessing/08-noise.md)).
- **Averaging** repeated acquisitions in the complex domain reduces noise toward zero;
  averaging magnitudes converges to the noise floor instead. For a voxel at SNR 1 averaged
  over 16 repeats whose phases have been aligned (printed below), the complex average is
  almost unbiased, while the average of the magnitudes stays as biased as a single
  measurement. Averaged without that alignment, repeats with different phases partly
  cancel, and the complex average is too low instead.
- **The phase is diagnostic.** Eddy-current and motion-related phase can be inspected per
  volume before any correction is attempted (Chapters [11](../03-preprocessing/11-eddy-currents.md) and [12](../03-preprocessing/12-motion-and-dropout.md)).

```{code-cell} python
:tags: [hide-input]
n_avg = 16
print(f"true signal 1 σ, {n_avg} repeats: average of the magnitudes {kspace.rician_mean(1.0, 1.0):.2f} σ; "
      f"magnitude of the complex average {kspace.rician_mean(1.0, 1.0 / np.sqrt(n_avg)):.2f} σ")
```

The costs are twice the storage, a phase image that wraps and is not interpretable without
a reference, and a preprocessing pipeline that accepts complex input. [Chapter 19](../04-modeling/19-what-your-data-allow.md) collects
what complex data provide across the whole analysis chain.

## What this implies for acquisition

- **Save the phase.** It costs storage and nothing else at the scanner, and it cannot be
  recovered afterward.
- **Parallel imaging is a noise decision.** $R = 2$ shortens the readout and TE, but the
  noise increases by about $\sqrt{2}$ times the g-factor and becomes spatially structured.
- **Partial Fourier depends on smooth phase.** Moderate factors are safe; aggressive ones
  fail near sinuses and under strong eddy currents.
- **More coils raise the noise floor** of magnitude images even as they raise SNR, and the
  floor is what biases high-b measurements.

## Further reading

Phased arrays and their combination {cite:p}`roemer1990`, SENSE {cite:p}`pruessmann1999`,
GRAPPA {cite:p}`griswold2002`, homodyne reconstruction {cite:p}`noll1991`, and the noise
statistics of magnitude images {cite:p}`gudbjartsson1995,constantinides1997`.
