---
title: "2. Spatial encoding and k-space"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** the k-space of a synthetic b=0 slice built from the packaged tissue maps ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
- **Simulated live in this page:** one slice of the simulated brain acquired by TRXScan with eight coils, GRAPPA 2 and partial Fourier, with the per-coil k-space it sampled and the timing of every line.
- **`slab-kspace`** (pending): a five-slice slab with its raw k-space exported: 8 coils, GRAPPA 2, partial Fourier 6/8 ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-slab-kspace)).

Live-tier figures simulate one slice of the simulated brain in the page through TRXScan's Python package ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md#live-tier)); they run in seconds at build time.
Pipeline-tier datasets are simulated offline by TRXScan ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)) and are marked *pending* until their release; the figures that need them say so where they will appear.
:::

## Learning goals

After this chapter you can:

- explain what k-space is and why the scanner records the Fourier transform of the image
  rather than the image itself
- predict what happens to an image when k-space is sampled too coarsely or not far enough
- describe the single-shot EPI readout, compute its timing, and state how partial Fourier
  and in-plane acceleration change it
- recognize Gibbs ringing and know where it comes from
- explain why diffusion MRI uses single-shot EPI despite its drawbacks

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt

from dwibook import kspace, phantoms
from dwibook.plotting import INK, PALETTE, animate, set_style, show_image, show_kspace

set_style()
img = phantoms.brain_image()  # synthetic b=0 slice of the simulated brain, adult preset, TE 88 ms
mask = phantoms.brain_slice()["mask"]
```

## Gradients as encoding

Without a gradient, every spin in the slice precesses at the same frequency, and the
receiver has no way to tell where any of the signal came from. A gradient coil adds a small
field that grows steadily with position along one direction, so it makes the precession
frequency depend on position: spins on one side of the object turn slightly faster than
spins on the other side. After the gradient has been on for a while, each spin's phase (the
angle it has turned through, relative to a spin at the center) is proportional to its
position, so the phase winds across the object like a corkscrew. The longer the gradient
stays on, the more turns the corkscrew makes over the same distance.

The receiver still cannot tell spins apart. It records the sum of the signal over the whole
object as one complex number (the quadrature-demodulated signal of [Chapter 1](./01-spins-and-signal.md)). Adding up
arrows that point in different directions is the same as multiplying the object by a
striped pattern with the corkscrew's pitch and adding the result, so that one number says
how much of the object varies on that scale. It is one k-space sample. The label $k$ is the
number of turns the corkscrew makes per millimeter, in cycles/mm; while the gradient is on,
$k$ grows steadily with time, so the receiver records a new sample, farther from $k = 0$,
at every moment.

The animation follows one row of the slice while a readout gradient is on. The row is
coarsened to 8 mm voxels so that each voxel's arrow stays visible.

- **Top:** each voxel is an arrow whose length is its signal and whose direction is its
  phase; the gray curve is the corkscrew seen from the side, with its pitch printed above.
- **Middle:** the object multiplied by a cosine stripe and by a sine stripe of the same
  pitch. The sum of each is written at the right; the two sums are the two parts of the
  complex number the receiver records.
- **Bottom:** the two sums plotted against $k$ as the recording goes on. The dot is the
  current sample, and the open circles mark where the scanner keeps a sample, once every
  $1/\text{FOV}$.

What to look at: at the start every arrow points the same way and the sum is the total
signal of the row. As the pitch tightens the arrows fan out and cancel, and the sums fall
toward zero within the first few samples; what remains farther out measures the fine
structure of the row, its edges and narrow CSF spaces. In the last frame neighboring voxels
point in opposite directions, one turn every two voxels: that is the finest pattern the grid
can hold, and the edge of k-space. A real readout first applies the gradient with the
opposite sign, winding the corkscrew the other way out to the far edge, and then sweeps back
through $k = 0$ to this edge, so it records the whole line; the animation shows the second
half. For an object with no phase of its own, the first half carries no new information:
its cosine sums repeat those of the second half and its sine sums are the same with the
sign flipped.

```{code-cell} python
:tags: [hide-input]
prof = img[60:68].mean(0).reshape(32, 4).mean(1)  # one row of the slice, coarsened to 32 voxels of 8 mm
DX = 8.0                                            # mm per coarse voxel
x1 = (np.arange(prof.size) - prof.size // 2) * DX   # voxel centers (mm)
xf = np.linspace(x1[0] - DX / 2, x1[-1] + DX / 2, 1200)
KMAX = 1 / (2 * DX)                                 # edge of k-space: one turn every two voxels
N_WIND, N_HOLD = 70, 12
ks = np.linspace(0, KMAX, N_WIND)
total = prof.sum()
cos_sum = np.array([(prof * np.cos(2 * np.pi * k * x1)).sum() for k in ks]) / total
sin_sum = np.array([(prof * np.sin(2 * np.pi * k * x1)).sum() for k in ks]) / total
k_grid = np.arange(prof.size // 2 + 1) / (prof.size * DX)  # sample spacing 1/FOV
g_cos = np.array([(prof * np.cos(2 * np.pi * k * x1)).sum() for k in k_grid]) / total
g_sin = np.array([(prof * np.sin(2 * np.pi * k * x1)).sum() for k in k_grid]) / total

fig = plt.figure(figsize=(9, 7.2))
gs = fig.add_gridspec(4, 1, height_ratios=[1.25, 0.8, 0.8, 1.35], hspace=0.8)
ax_sp, ax_c, ax_s, ax_k = (fig.add_subplot(gs[i]) for i in range(4))
for ax in (ax_sp, ax_c, ax_s):
    ax.set(xlim=(xf[0], xf[-1]), yticks=[])
    ax.grid(False)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelbottom=False)
ax_s.tick_params(labelbottom=True)
ax_s.set_xlabel("position along the row (mm)")

helix, = ax_sp.plot(xf, np.zeros_like(xf), color="0.55", lw=1)
L = prof / prof.max()                               # arrow length: the voxel's signal
quiv = ax_sp.quiver(x1, np.zeros_like(x1), L, np.zeros_like(x1), color=PALETTE[0], angles="uv",
                    scale_units="inches", scale=3.2, pivot="middle", width=0.005, headwidth=3)
ax_sp.set(ylim=(-1.25, 1.25))
ax_sp.set_title("spins along one row: arrow length = signal, direction = phase", loc="left")
pitch = ax_sp.text(1.0, 1.08, "", transform=ax_sp.transAxes, ha="right", fontsize=9, color=INK["secondary"])

bars, stripes, labels = [], [], []
for ax, color, name in [(ax_c, PALETTE[0], "cosine"), (ax_s, PALETTE[1], "sine")]:
    stripes.append(ax.plot(xf, np.ones_like(xf), color="0.6", lw=1)[0])
    bars.append(ax.bar(x1, L, width=0.8 * DX, color=color))
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set(ylim=(-1.25, 1.25))
    ax.set_title(f"row × {name} stripe of the same pitch (gray: the stripe)", loc="left")
    labels.append(ax.text(1.0, 1.08, "", transform=ax.transAxes, ha="right", fontsize=9, color=color))

line_c, = ax_k.plot([], [], color=PALETTE[0], lw=1.5, label="cosine sum")
line_s, = ax_k.plot([], [], color=PALETTE[1], lw=1.5, label="sine sum")
samp_c, = ax_k.plot([], [], "o", mfc="none", color=PALETTE[0], ms=5)
samp_s, = ax_k.plot([], [], "o", mfc="none", color=PALETTE[1], ms=5)
dot_c, = ax_k.plot([], [], "o", color=PALETTE[0], ms=8)
dot_s, = ax_k.plot([], [], "o", color=PALETTE[1], ms=8)
ax_k.axhline(0, color="0.6", lw=0.8)
ax_k.set(xlim=(-0.002, KMAX * 1.03), ylim=(-0.45, 1.08), xlabel="k (cycles/mm): grows with time while the gradient is on",
         ylabel="sum (÷ total)", title="k-space along this row, recorded one sample at a time")
ax_k.legend(loc="upper right")
fig.subplots_adjust(left=0.07, right=0.97, top=0.95, bottom=0.07)

def frame(i):
    j = min(i, N_WIND - 1)
    k = ks[j]
    ph = -2 * np.pi * k * x1                       # phase wound by the gradient, proportional to position
    quiv.set_UVC(L * np.cos(ph), L * np.sin(ph))
    helix.set_ydata(np.sin(-2 * np.pi * k * xf))
    pitch.set_text("no winding yet" if k == 0 else f"one turn every {1 / k:.0f} mm")
    for fn, bs, st, lab, sums in zip((np.cos, np.sin), bars, stripes, labels, (cos_sum, sin_sum)):
        st.set_ydata(fn(2 * np.pi * k * xf))
        for b, v in zip(bs, L * fn(2 * np.pi * k * x1)):
            b.set_height(v)
        lab.set_text(f"sum = {sums[j]:+.2f}")
    line_c.set_data(ks[: j + 1], cos_sum[: j + 1])
    line_s.set_data(ks[: j + 1], sin_sum[: j + 1])
    passed = k_grid <= k + 1e-12
    samp_c.set_data(k_grid[passed], g_cos[passed])
    samp_s.set_data(k_grid[passed], g_sin[passed])
    dot_c.set_data([k], [cos_sum[j]])
    dot_s.set_data([k], [sin_sum[j]])

animate(fig, frame, range(N_WIND + N_HOLD), fps=9, width=760, dpi=64,
        alt="one row of the brain slice as 32 arrows, one per voxel; as a readout gradient stays on, the arrows' directions wind into a corkscrew whose turns get closer together, ending with neighboring arrows pointing in opposite directions; below, the row is multiplied by cosine and sine stripes of the same pitch, and the two sums are plotted against k: they start at the full signal and fall toward zero, then wobble near zero as the dot moves out to the edge of k-space")
```

The same thing in symbols: the recorded signal at position $\mathbf{k}$ in k-space is the
image $\rho(\mathbf{r})$ multiplied by a stripe pattern and summed over the object,

$$s(\mathbf{k}) = \int \rho(\mathbf{r})\, e^{-2\pi i\, \mathbf{k}\cdot\mathbf{r}}\, d\mathbf{r},$$

where $e^{-2\pi i\,\mathbf{k}\cdot\mathbf{r}} = \cos(2\pi\,\mathbf{k}\cdot\mathbf{r}) - i \sin(2\pi\,\mathbf{k}\cdot\mathbf{r})$
holds the cosine and sine stripes of the animation. This is the Fourier transform of the
image. In two dimensions the gradients along $x$ and $y$ combine, and the stripes can point in
any direction: $k_x$ and $k_y$ set how many stripes cross the image horizontally and
vertically. Each panel below is one stripe pattern (the cosine part), labeled by its
position in k-space in cycles per field of view, with the image multiplied by it underneath;
the sum of that product is the real part of one k-space sample.

```{code-cell} python
:tags: [hide-input]
yy, xx = np.mgrid[:128, :128] - 64
ksp0 = kspace.fft2c(img)
fig, axes = plt.subplots(2, 4, figsize=(11, 5.6))
for col, (kx, ky) in enumerate([(0, 0), (3, 0), (0, 6), (5, 5)]):
    stripe = np.cos(2 * np.pi * (kx * xx + ky * yy) / 128)
    show_image(axes[0, col], stripe, f"$(k_x, k_y)$ = ({kx}, {ky})", kind="diff", vmin=-1, vmax=1)
    sample = ksp0[64 + ky, 64 + kx].real / ksp0[64, 64].real
    show_image(axes[1, col], img * stripe, f"image × stripe: sum = {sample:+.2f}", kind="diff", vmin=-1, vmax=1)
fig.text(0.01, 0.73, "stripe", rotation=90, va="center", fontsize=9)
fig.text(0.01, 0.28, "image × stripe", rotation=90, va="center", fontsize=9)
fig.tight_layout(rect=(0.02, 0, 1, 1))
```

What to look at: the (0, 0) pattern is flat, so its sum is the total signal of the slice, the
largest value in k-space (the sums are shown relative to it). The other patterns multiply
some tissue by red (+) and some by blue (−), so the sum is small; it is not zero because the
slice is not uniform, and its size says how strongly the image contains that stripe.

The coordinate $\mathbf{k}$ is set by the gradient history: the longer and stronger the
gradients that have been applied, the farther from the center of k-space the current sample
lies. A pulse sequence is a plan for moving through k-space and recording samples along the
way; reconstruction is an inverse Fourier transform of those samples. Slice selection uses
the same principle during excitation: a gradient along the slice direction makes the RF pulse
resonant only within a slab.

This chapter assumes the gradient fields are exactly linear. They are not, and the
consequences for image geometry and for the diffusion encoding are the subject of [Chapter 13](../03-preprocessing/13-gradient-nonlinearity.md).

## The Fourier relationship

The images in this chapter are a synthetic b=0 slice of the simulated brain: the tissue
fractions of one axial slice, weighted by proton density and T2 decay at the HBCD echo time
([Chapter 1](./01-spins-and-signal.md)). Its k-space is computed directly. The axes of a k-space image are
spatial frequencies, in cycles per millimeter: $k_x$ runs horizontally and $k_y$, the
phase-encode axis, vertically. A sample at the center describes the mean of the image; a
sample at the edge describes a pattern that repeats every two voxels, the finest the grid
can hold. For 2 mm voxels the outermost sample is at 0.25 cycles/mm.

```{code-cell} python
:tags: [hide-input]
ksp = kspace.fft2c(img)
VOXEL = 2.0  # mm

fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
show_image(axes[0], img, "image (synthetic b=0, 2 mm voxels)")
show_kspace(axes[1], ksp, "k-space magnitude (log scale)", voxel_mm=VOXEL)
show_kspace(axes[2], ksp, "k-space phase", voxel_mm=VOXEL, phase=True)
fig.tight_layout()
```

The later k-space panels in this chapter share these axes and omit the labels. Most of the
energy is at the center of k-space. The center encodes contrast and coarse
shape; the periphery encodes edges and fine detail. Reconstructing from only one or the other
shows the division. In each pair below, the left panel is the part of k-space that was
kept (the rest set to zero) and the right panel is the image reconstructed from it:

```{code-cell} python
:tags: [hide-input]
ny, nx = ksp.shape
c = ny // 2
low = np.zeros_like(ksp); low[c - 8 : c + 8, c - 8 : c + 8] = ksp[c - 8 : c + 8, c - 8 : c + 8]
high = ksp - low

fig, axes = plt.subplots(1, 4, figsize=(12, 3))
show_kspace(axes[0], low, "kept: the central 16 × 16 samples")
show_image(axes[1], kspace.ifft2c(low), "reconstructed from them")
show_kspace(axes[2], high, "kept: everything except the center")
show_image(axes[3], kspace.ifft2c(high), "reconstructed from that")
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
  This is aliasing, and it is the effect that parallel imaging ([Chapter 3](./03-reconstruction.md)) deliberately
  induces and then removes.

```{code-cell} python
:tags: [hide-input]
lowres = np.zeros_like(ksp); lowres[c - 16 : c + 16, c - 16 : c + 16] = ksp[c - 16 : c + 16, c - 16 : c + 16]
every_other = np.where(kspace.regular_undersampling_mask(ny, nx, 2), ksp, 0)

fig, axes = plt.subplots(2, 3, figsize=(9, 6))
show_kspace(axes[0, 0], ksp, "k-space: full sampling")
show_kspace(axes[0, 1], lowres, "k-space: central 32 × 32 kept")
show_kspace(axes[0, 2], every_other, "k-space: every other line kept")
show_image(axes[1, 0], img, "image: full sampling")
show_image(axes[1, 1], kspace.ifft2c(lowres), "image: larger voxels")
show_image(axes[1, 2], kspace.ifft2c(every_other), "image: FOV halved, wrapped")
fig.tight_layout()
```

## The EPI readout

A conventional sequence records one line of k-space per excitation and needs one excitation
per line. Echo-planar imaging (EPI) records the whole plane after a single excitation: the
readout gradient alternates direction, sweeping back and forth along $k_x$, while short
phase-encode blips step $k_y$ one line at a time. The path through k-space is a zigzag, and
the time between successive lines is the echo spacing.

```{code-cell} python
:tags: [hide-input]
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
:tags: [hide-input]
esp = 91.7 / 128
for label, kw in [("full", {}), ("partial Fourier 6/8", {"partial_fourier": 0.75}),
                  ("R = 2", {"accel": 2}), ("6/8 and R = 2", {"partial_fourier": 0.75, "accel": 2})]:
    tr = kspace.epi_trajectory(128, 128, esp, **kw)
    print(f"{label:>20}: {tr.lines.size:3d} lines, readout {tr.readout_ms:5.1f} ms, "
          f"k-space center reached after {tr.time_to_center_ms:5.1f} ms")
```

Three consequences of the long readout, each treated in its own chapter:

- The phase-encode direction is sampled slowly, about one line per 0.7 ms, so a small
  frequency offset displaces signal a long way along that axis. An off-resonance (a spin
  precessing slightly faster or slower than the scanner assumes) of 100 Hz moves signal by
  about nine voxels. This is susceptibility distortion ([Chapter 10](../03-preprocessing/10-susceptibility-distortion.md)), and it is
  why the phase-encode direction and readout time must be recorded in the image metadata.
- Signal decays with T2* during the readout, so lines acquired late are weaker. The result is
  blurring along the phase-encode direction.
- Odd and even lines are read in opposite directions. A timing mismatch between them
  produces a faint copy of the image shifted by half the field of view, the Nyquist ghost
  ([Chapter 14](../03-preprocessing/14-assembled-pipeline.md)).

The figure below produces each of the three from the synthetic slice. It acquires the
slice's k-space line by line, top to bottom, one line every echo spacing, and lets each line
see the object as it is at that moment in the readout. The top row is the same acquisition
without the fault; the phase-encode direction is vertical in every panel.

```{code-cell} python
:tags: [hide-input]
t_line = (np.arange(ny) - ny // 2) * esp  # ms from the k-space center, for each ky line (acquired top to bottom)
rr, cc = np.mgrid[:ny, :nx]

def epi_acquire(obj, field_hz=0.0, t2star_ms=np.inf, odd_shift=0.0):
    """k-space of obj acquired line by line; each line sees the phase and decay of its own moment."""
    k = np.array([kspace.fft2c(obj * np.exp(2j * np.pi * field_hz * t * 1e-3))[j] for j, t in enumerate(t_line)])
    k = k * np.exp(-t_line / t2star_ms)[:, None]              # T2* decay, relative to the center line
    if odd_shift:                                             # odd lines read backwards, off by a fraction of a sample
        hyb = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(k, axes=1), axis=1), axes=1)
        hyb[1::2] *= np.exp(2j * np.pi * odd_shift * (np.arange(nx) - nx // 2) / nx)
        k = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(hyb, axes=1), axis=1), axes=1)
    return np.abs(kspace.ifft2c(k))

def pe_shift(a, b, up=50):
    """Shift of b relative to a along phase-encode (voxels), by upsampled cross-correlation of row sums."""
    pa, pb = a.sum(1), b.sum(1)
    F = np.fft.fft(pb) * np.conj(np.fft.fft(pa))
    Fp = np.zeros(ny * up, complex); Fp[: ny // 2] = F[: ny // 2]; Fp[-ny // 2 :] = F[-ny // 2 :]
    s = np.argmax(np.fft.ifft(Fp).real) / up
    return s - ny if s > ny / 2 else s

uniform = epi_acquire(img, field_hz=100.0)
print(f"uniform 100 Hz offset: image moves {abs(pe_shift(img, uniform)):.1f} voxels along phase-encode "
      f"(100 Hz × {ny} lines × {esp:.3f} ms = {100 * ny * esp * 1e-3:.1f})")

field = 100 * np.exp(-((rr - 34) / 9.0) ** 2 / 2 - ((cc - 64) / 14.0) ** 2 / 2)  # frontal offset, up to 100 Hz
distorted = epi_acquire(img, field_hz=field)
blurred = epi_acquire(img, t2star_ms=45.0)
ghosted = epi_acquire(img, odd_shift=0.1)

def psf_pe(w, up=64):
    """Image of a single point along phase-encode (finely interpolated, scaled to its peak) for line weights w."""
    buf = np.zeros(ny * up, complex); buf[: ny // 2] = w[ny // 2 :]; buf[-ny // 2 :] = w[: ny // 2]
    psf = np.abs(np.fft.fftshift(np.fft.ifft(buf)))
    return (np.arange(ny * up) - ny * up // 2) / up, psf / psf.max()

def fwhm(w):
    """Width at half maximum (voxels) of that point image."""
    xs_p, p = psf_pe(w)
    return (p > 0.5).sum() * (xs_p[1] - xs_p[0])

w = np.exp(-t_line / 45.0)
print(f"T2* = 45 ms: lines weighted from {w[0]:.2f} (first) to {w[-1]:.2f} (last) of the center line; "
      f"point-spread width along phase-encode {fwhm(w):.2f} voxels, against {fwhm(np.ones(ny)):.2f} without decay "
      f"and {fwhm(np.exp(-t_line / 20.0)):.2f} with T2* = 20 ms")
ghost_zone = np.roll(mask, ny // 2, axis=0) & ~mask
print(f"Nyquist ghost: mean {ghosted[ghost_zone].mean() / img[mask].mean() * 100:.1f} % of the brain's mean signal, "
      f"{ny // 2} voxels (half the field of view) from the brain")

cr0, cr1, cc0, cc1 = 12, 72, 24, 104  # (a) zoom window on the front of the brain (rows, columns)
br0, br1, bc0, bc1 = 44, 76, 44, 84   # (b) zoom window around the lateral ventricles
fig = plt.figure(figsize=(11, 10.4))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.75], width_ratios=[1.33, 1.25, 1])
axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)])
edges = img[cr0:cr1, cc0:cc1]
show_image(axes[0, 0], edges, "no off-resonance (front of the brain)", vmin=0, vmax=0.8)
axes[0, 0].contour(field[cr0:cr1, cc0:cc1], levels=[50], colors=[PALETTE[3]], linewidths=1.2, linestyles="--")
show_image(axes[1, 0], distorted[cr0:cr1, cc0:cc1], "(a) off-resonance at the front, up to 100 Hz", vmin=0, vmax=0.8)
for ax in axes[:, 0]:
    ax.contour(edges, levels=[0.55], colors=[PALETTE[1]], linewidths=0.7)
show_image(axes[0, 1], img[br0:br1, bc0:bc1], "no decay (zoom on the ventricles)", vmin=0, vmax=0.8)
show_image(axes[1, 1], blurred[br0:br1, bc0:bc1], "(b) T2* = 45 ms during the readout", vmin=0, vmax=0.8)
show_image(axes[0, 2], img, "no line mismatch (window 0–8 % of max)", vmin=0, vmax=0.08 * img.max())
show_image(axes[1, 2], ghosted, "(c) odd/even mismatch: Nyquist ghost", vmin=0, vmax=0.08 * img.max())
axp = fig.add_subplot(gs[2, :])
for t2s, color, ls in [(np.inf, "0.3", "--"), (45.0, PALETTE[1], "-"), (20.0, PALETTE[6], "-")]:
    xs_p, p = psf_pe(np.exp(-t_line / t2s))
    axp.plot(xs_p, p, color=color, ls=ls, lw=1.5, label="no decay" if np.isinf(t2s) else f"T2* = {t2s:.0f} ms")
axp.axhline(0.5, color="0.7", lw=0.8)
axp.set(xlim=(-6, 6), xlabel="distance along phase-encode (voxels)", ylabel="intensity (÷ peak)",
        title="(b) closer: one bright point imaged with each decay, along phase-encode")
axp.legend(loc="upper right")
fig.tight_layout(rect=(0.035, 0, 1, 1), h_pad=2)
for ax in axes[:, 0]:
    ax.annotate("", xy=(-0.06, 0.15), xytext=(-0.06, 0.85), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="<->", color=INK["secondary"], lw=1))
    ax.text(-0.1, 0.5, "phase-encode", transform=ax.transAxes, rotation=90, va="center", ha="center",
            fontsize=8, color=INK["secondary"])
```

What to look at in each column:

- **(a)** The thin orange lines trace the bright edges (CSF and the brain surface) of the
  undistorted image, in both panels; the dashed yellow line in the top panel encloses the
  region where the offset exceeds 50 Hz. Below, the frontal tissue has slid down, along
  phase-encode, away from its orange outline, by up to about nine voxels. Because the offset
  changes with position, neighboring rows slide by different amounts: tissue is stretched
  and darkened on the side of the offset region where the shift grows, and squeezed together
  and brightened, piled up against the front of the ventricles, where it shrinks again. Away
  from the offset the image and the outlines still agree.
- **(b)** The two ventricle crops look almost the same: at 2 mm voxels and a T2* of 45 ms the
  blur is small. The plot below makes it visible. It is the image of a single bright point,
  along phase-encode: with decay the peak widens from 1.2 to 1.4 voxels at half height and
  gains broad wings, and with the shorter T2* of 20 ms found near air-filled sinuses it
  widens to about 2.6 voxels. Every edge that crosses the phase-encode direction is softened
  by this profile; edges along it are not.
- **(c)** With the window set to show only the faintest signal (everything above 8 % of the
  maximum is white), a dim copy of the brain appears above and below it, shifted by half the
  field of view along phase-encode and wrapped around the edges of the image. It is brighter
  toward the left and right sides than in the middle, because a timing mismatch produces a
  phase error that grows with distance from the center of the readout. The clean
  acquisition above has nothing outside the brain.

## Partial Fourier

For an object with no phase, k-space is symmetric about its center, so half of the lines are
redundant. Partial Fourier acquisitions skip a fraction of the lines on one side, typically
acquiring 5/8 to 7/8 of them. The timings printed above show the two benefits: the readout is
shorter, and the center of k-space is reached sooner, which shortens the minimum echo time.
The price is that the skipped lines have to be supplied by the reconstruction, which relies
on the symmetry, and real images do have phase (from field inhomogeneity, coil phase, eddy
currents, and motion), so the symmetry is only approximate. [Chapter 3](./03-reconstruction.md) shows what the
skipped lines cost in resolution and how the reconstruction recovers them. In-plane
acceleration is the other way to shorten the readout: keep every $R$-th line and recover the
missing ones using multiple receive coils.

## Truncation and Gibbs ringing

Every acquisition stops at some outermost k-space sample. A Fourier representation cut off
at a finite frequency overshoots at every sharp edge by about 9 % of the step, and the
overshoot does not shrink with more samples; it only moves closer to the edge.

```{code-cell} python
:tags: [hide-input]
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
resolution of 4 mm, on the 2 mm grid. The box marks the region around the lateral
ventricles that the zoomed panels show; the difference panel, truncated image minus object,
isolates the ripples from the anatomy:

```{code-cell} python
:tags: [hide-input]
crop = np.zeros_like(ksp); crop[c - 32 : c + 32, c - 32 : c + 32] = ksp[c - 32 : c + 32, c - 32 : c + 32]
ringing = np.abs(kspace.ifft2c(crop))
row = 58  # through the frontal horns of the lateral ventricles
r0, r1, c0, c1 = 40, 80, 40, 88  # zoom window (rows, columns)

fig, axes = plt.subplots(2, 2, figsize=(10, 9))
show_image(axes[0, 0], ringing, "64 × 64 samples reconstructed on 128 × 128", vmin=0, vmax=1)
axes[0, 0].add_patch(plt.Rectangle((c0 - 0.5, r0 - 0.5), c1 - c0, r1 - r0, fill=False, color=PALETTE[1], lw=1))
axes[0, 0].axhline(row, color=PALETTE[1], lw=1)
show_image(axes[0, 1], ringing[r0:r1, c0:c1], "zoom on the ventricles", vmin=0, vmax=0.7, interpolation="nearest")
show_image(axes[1, 0], (ringing - img)[r0:r1, c0:c1], "zoom: truncated minus object", kind="diff", vmin=-0.15, vmax=0.15, interpolation="nearest")
axes[1, 1].plot(img[row], color="0.3", lw=1, ls="--", label="object")
axes[1, 1].plot(ringing[row], color=PALETTE[1], lw=1.5, label="truncated")
axes[1, 1].set(xlabel="x (voxels)", ylabel="intensity", xlim=(20, 108), title="profile along the marked row")
axes[1, 1].legend()
fig.tight_layout()
```

In the difference panel every CSF boundary is lined by alternating bands of over- and
undershoot, one voxel apart, that fade with distance from the edge. The ringing is small in absolute terms but it sits exactly where CSF meets tissue, and it
changes with b-value because the CSF signal changes with b-value. Its effect on diffusion
metrics at tissue borders, and the correction for it, are covered in [Chapter 9](../03-preprocessing/09-gibbs-ringing.md).

## Why diffusion MRI uses single-shot EPI

A conventional sequence could acquire the lines of k-space over several excitations, with a
short readout each time and therefore little distortion and blur. Diffusion encoding rules
this out for routine use. The diffusion gradients are strong enough that small bulk
movements of the head during the encoding, including pulsation, give each excitation a
different, unknown phase. In a single-shot acquisition that phase is common to every line and
has no effect on the magnitude image; if the lines came from different excitations, the
phases would disagree and the image would carry ghosts. The standard acquisition is therefore
single-shot EPI, and its long readout is the origin of most of the artifacts corrected in
Part III. The multi-shot and non-EPI readouts that work around the phase problem are
uncommon in practice and are described in [Chapter 23](../05-advanced/23-frontiers.md).

## Measure it: a TRXScan slice and its k-space

Everything above used a synthetic object and our own toy Fourier transform. Here the
simulator acquires one slice of the simulated brain the way a scanner would: eight receive coils,
GRAPPA 2, 6/8 partial Fourier, the HBCD readout, and it hands back the k-space it actually
sampled for every coil, before any reconstruction.

```{code-cell} python
:tags: [hide-input]
import trxscan as ts
from dwibook import phantom as ph

sim = ph.run(ph.gtab(3), ph.PROTO, ts.Artifacts(noise=2e-4, ghost=0.015, seed=1), kspace=True)
k = sim.kspace
print(f"acquired k-space {k.acquired.shape} (volume, slice, coil, ky, kx); "
      f"{int(k.mask[:, 0].sum())} of {k.shape[0]} phase-encode lines sampled "
      f"(partial Fourier {sim.protocol.partial_fourier:g}, GRAPPA {sim.protocol.accel})")
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
show_kspace(axes[0], k.acquired[0, 0, 0], "coil 0, as acquired (b = 0)")
show_kspace(axes[1], k.reconstructed[0, 0, 0], "coil 0, after GRAPPA and windowing")
show_image(axes[2], np.rot90(k.combined[0, 0]), "combined complex image, magnitude")
show_image(axes[3], ph.axial(sim.phase, 0), "phase (radians)", kind="phase")
fig.tight_layout()
```

The un-acquired lines are exactly zero in the raw k-space: the block at one edge is the
partial Fourier band and the alternate rows are the GRAPPA undersampling, which the reconstruction
fills in from the eight coils' complementary spatial sensitivities before the inverse
transform. The same object, once through the real readout, also tells us the timing of every
line, which is what the BIDS sidecar summarizes:

```{code-cell} python
:tags: [hide-input]
r = sim.readout
print(f"{r.ny} phase-encode lines at {r.t_line_ms:.3f} ms each; first to last acquired line "
      f"{r.total_readout_ms:.1f} ms (sidecar TotalReadoutTime {sim.sidecar['TotalReadoutTime'] * 1e3:.1f} ms); "
      f"k-space centre reached {r.time_to_center_ms:.1f} ms into the train; TE {r.t_echo_ms:.0f} ms")
fig, ax = plt.subplots(figsize=(7, 2.8))
ax.plot(r.t_read_ms[r.ky_order], r.ky_order - r.ny // 2, ".-", color=PALETTE[0], ms=3, lw=0.8)
acq = r.acquired_lines
ax.plot(r.t_read_ms[acq], acq - r.ny // 2, "o", color=PALETTE[1], ms=3, label="acquired lines")
ax.axhline(0, color="0.6", lw=0.8)
ax.set(xlabel="time since the readout began (ms)", ylabel="$k_y$ line", title="the EPI train the simulator used, line by line")
ax.legend(loc="lower right")
fig.tight_layout()
```

The train walks from the top of k-space down, one line per echo spacing. Partial Fourier
skips the first quarter of the lines, so the centre (where the echo forms and the contrast is
decided) is reached that much sooner and the readout ends earlier; GRAPPA skips every second
line outside the central calibration band. The simulator keeps the echo time you asked for,
so the shorter path to the centre is a TE you may lower, not one it lowers for you. Chapter 3 reconstructs this k-space in Python and checks the result against
what the simulator produced.

## What this implies for acquisition

- **Voxel size and FOV are k-space decisions.** Smaller voxels require more lines, a longer
  readout, and therefore more distortion and blur.
- **Readout length drives the main EPI artifacts.** Partial Fourier and in-plane
  acceleration both shorten it and both reduce the minimum TE, but only in-plane
  acceleration reduces distortion, because only it changes the spacing of the lines. [Chapter 7](../02-diffusion-encoding/07-acquisition-parameters.md) discusses their costs.
- **Ringing is a property of every acquisition**, not a malfunction. It is worst at CSF
  boundaries and can be reduced after the fact ([Chapter 9](../03-preprocessing/09-gibbs-ringing.md)).
- **Single-shot EPI is a compromise** accepted so that diffusion encoding is robust to
  motion. The metadata that describe the readout, `PhaseEncodingDirection` and
  `TotalReadoutTime`, are required by the corrections in Part III and should be checked
  before any processing.

## Further reading

The k-space description of MRI {cite:p}`ljunggren1983,twieg1983`, echo-planar imaging
{cite:p}`mansfield1977`, and the textbooks {cite:t}`haacke1999` and {cite:t}`nishimura2010`.
