---
title: "7. Acquisition parameter choices"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** a synthetic b=0 slice and single-voxel signal models built from the packaged tissue maps and presets ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
- **`voxel-sweep`** (pending): 1.5, 2.0, 2.5, and 3.0 mm voxels at fixed scheme and noise ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-voxel-sweep)).
- **`te-sweep`** (pending): four echo times at fixed b ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-te-sweep)).
- **`gnl`** (pending): gradient nonlinearity on two gradient systems, a severity sweep, and warp-only and encoding-only runs, with the true displacement field and gradient deviation ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-gnl)).
- **`ref-schemes`** (pending): the simulated brain under the 30-direction, 64-direction, HBCD, and DSI schemes, compared at equal total scan time (the per-volume noise is scaled with the number of volumes), plus a CS-DSI subset of the DSI run, which takes about a quarter of its time ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-ref-schemes)).

Pipeline-tier datasets are simulated offline by TRXScan ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)) and are marked *pending* until their release; the figures that need them say so where they will appear.
:::

## Learning goals

After this chapter you can:

- state what each of the main acquisition parameters controls, what it costs, and which
  artifact or analysis it affects: TE, TR, voxel size, b-values, number of directions and
  b=0 volumes, partial Fourier, multiband, in-plane acceleration, phase-encode direction,
  and gradient hardware
- estimate SNR, scan time, and distortion from a proposed protocol before it is run
- design a diffusion protocol under a scan-time budget and justify each choice

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt

from dwibook import kspace, phantoms, presets, schemes, signal, synth
from dwibook.plotting import INK, PALETTE, TISSUE_COLORS, set_style, show_image

set_style()
```

Every parameter below trades signal, time, and artifact against each other. The
relationships are simple enough to compute on paper, and the point of this chapter is to
do so before the protocol is run rather than after the data are found wanting.

## Echo time

TE is set by the diffusion encoding ([Chapter 5](./05-diffusion-encoding.md)) and the readout ([Chapter 2](../01-mri-physics/02-spatial-encoding-kspace.md)), and the
scanner reports the minimum it can reach for a given b-value. Signal falls as
$e^{-\mathrm{TE}/T_2}$ ([Chapter 1](../01-mri-physics/01-spins-and-signal.md)). Each 10 ms of TE costs about 14 % of adult white matter
signal and 12 % of gray matter signal; CSF, whose T2 is about two seconds, barely changes.
Shortening TE is the single most effective way to raise SNR, and the ways to do it are
stronger gradients, *partial Fourier* (skipping part of one side of k-space, whose content
the reconstruction infers from the other side), and *in-plane acceleration* (skipping k-space
lines, every other one at an acceleration factor R = 2, and filling the gaps from the receive coils' sensitivity patterns,
[Chapter 3](../01-mri-physics/03-reconstruction.md)). Both shorten the time the readout needs to reach the center of k-space, where
the echo is formed.

The figure plots the b=0 signal of each tissue against TE, relative to its value at
TE = 60 ms, with markers at 60 ms (a short TE, reachable with strong gradients and
acceleration) and at the 88 ms of the HBCD protocol. Look at the gap between the two
markers on the white and gray matter curves: that is the signal the longer TE gives up.

```{code-cell} python
:tags: [hide-input]
te = np.linspace(55, 130, 200)
te_marks = (60.0, presets.TE_HBCD_MS)
t2 = presets.T2_MS["adult"]
fig, ax = plt.subplots(figsize=(6.5, 3.4))
for k in ("WM", "GM", "CSF"):
    ax.plot(te, np.exp(-(te - 60) / t2[k]), color=TISSUE_COLORS[k], label=k)
    ax.plot(te_marks, [np.exp(-(t - 60) / t2[k]) for t in te_marks], "o", color=TISSUE_COLORS[k], ms=7)
for t, lab in zip(te_marks, ("TE 60 ms", f"HBCD TE {presets.TE_HBCD_MS:.0f} ms")):
    ax.axvline(t, color=INK["secondary"], lw=0.8, ls=":")
    ax.text(t + 1, 0.42, lab, fontsize=8, color=INK["secondary"])
ax.set(xlabel="TE (ms)", ylabel="b=0 signal relative to TE = 60 ms", ylim=(0.4, 1.12),
       title="Signal lost to T2 decay as TE grows (adult preset)")
ax.legend(loc="center right")
fig.tight_layout()
for k in ("WM", "GM", "CSF"):
    print(f"{k:>3}: {1 - np.exp(-10 / t2[k]):.0%} lost per 10 ms of TE; "
          f"at TE {presets.TE_HBCD_MS:.0f} ms, {np.exp(-(presets.TE_HBCD_MS - 60) / t2[k]):.2f} of the signal at TE 60 ms")
```

## Repetition time

TR sets three things: how much longitudinal magnetization has recovered before the next
excitation, how many slices fit in one TR, and the scan time. The steady-state signal is
$1 - e^{-\mathrm{TR}/T_1}$. White matter recovers within about 3 s; CSF, with a T1 of about
4 s, does not, and at a short TR its b=0 signal is suppressed: in the b=0 image the
ventricles look darker relative to the tissue than they would at a long TR, and the
fraction of CSF that a voxel's signal implies comes out too small. The left panel below
shows the recovery; at TR = 3 s white matter reaches 97 % of its full signal and CSF 53 %.

The minimum TR is the time to acquire all slices once: the number of slices times the time
per slice, divided by the *multiband* factor. The time per slice runs from the excitation
to the end of the readout: TE, plus the part of the readout that remains after the
k-space center, plus some overhead for fat suppression, excitation, and spoiling. For an
HBCD-like TE near 90 ms and an unshortened readout it is roughly 120–150 ms; the figure
uses 140 ms. Multiband (also called simultaneous multi-slice) excites several slices at
once and reads them in one readout; the receive coils' different views of each slice let
the reconstruction pull them apart. With multiband 3, three slices share each readout, so
a volume takes a third of the time. It has two costs. Separating the slices amplifies the
noise where the coils see the slices alike, by a g-factor like that of in-plane
acceleration (below). And the shorter TR it allows leaves less time for T1 recovery, so
pushing TR down lowers the signal of every volume, most of all in CSF and gray matter.

```{code-cell} python
:tags: [hide-input]
tr = np.linspace(0.5, 8, 200)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2))
for name, t1 in presets.T1_MS.items():
    ax1.plot(tr, 1 - np.exp(-tr * 1000 / t1), color=TISSUE_COLORS[name], label=name)
ax1.set(xlabel="TR (s)", ylabel="fraction of full signal", title="T1 recovery per TR"); ax1.legend()
t_slice = 0.14
for mb, color in zip([1, 2, 3, 4], PALETTE):
    n_slices = np.arange(30, 100)
    ax2.plot(n_slices, n_slices * t_slice / mb, color=color, label=f"multiband {mb}")
ax2.set(xlabel="number of 2 mm slices", ylabel="minimum TR (s)", title="TR needed to cover the slices"); ax2.legend()
fig.tight_layout()
print("recovered fraction at TR = 3 s: " + ", ".join(f"{name} {1 - np.exp(-3000 / t1):.0%}" for name, t1 in presets.T1_MS.items()))
print(f"70 slices at {t_slice * 1000:.0f} ms each: TR {70 * t_slice:.1f} s without multiband, {70 * t_slice / 3:.1f} s with multiband 3")
```

Seventy slices of 2 mm without multiband need a TR near 10 s; with multiband 3 it is about
3.3 s. That factor goes directly into the scan time ([Chapter 6](./06-qspace-sampling.md)) or, at fixed scan time, into
the number of volumes.

## Voxel size

Signal is proportional to voxel volume, because a bigger voxel holds more water. A 2 mm
isotropic voxel holds 8 mm³ and a 1.5 mm voxel 3.4 mm³, so moving from 2 mm to 1.5 mm gives
up more than half the SNR (a factor of 2.4). The trade is partial volume. A large voxel at
the cortex or the ventricle wall contains a mixture of tissues, and its diffusion measures
are the mixture's.

The figure makes the trade visible with an exaggerated step. The synthetic slice is shown at
its native 2 mm in-plane and block-averaged to 4 mm in-plane (same slice thickness), which
multiplies the voxel volume, and so the SNR, by 4; the noise is scaled accordingly. Each
title gives the fraction of brain voxels that are tissue mixtures, with no class above 80 %.
Look at the left image's graininess and the right image's blurred cortex and ventricle
edges.

```{code-cell} python
:tags: [hide-input]
tissue = phantoms.brain_slice()
img = phantoms.brain_image()

def block(a, f):
    n = (a.shape[0] // f) * f
    return a[:n, :n].reshape(n // f, f, n // f, f).mean(axis=(1, 3))

def mixed_fraction(f):
    fr = np.stack([block(tissue[k], f) for k in ("wm", "gm", "csf")])
    inside = fr.sum(0) > 0.5
    return np.mean(fr.max(0)[inside] < 0.8)

sigma = 0.03
fig, axes = plt.subplots(1, 2, figsize=(7, 3.4))
for ax, f in zip(axes, [1, 2]):
    im = block(img, f)
    noisy = kspace.ifft2c(kspace.add_complex_noise(kspace.fft2c(im), sigma / f**2, seed=0))  # in-plane only: SNR ∝ area
    show_image(ax, noisy, f"{2 * f} mm in-plane, mixed voxels {mixed_fraction(f):.0%}", vmin=0, vmax=1)
fig.tight_layout()
```

Resolution also determines what tractography can resolve: fibers that cross within a voxel
are averaged, and the models of [Chapter 16](../04-modeling/16-fiber-orientation.md) recover them only if the voxel is small relative
to the bundle geometry. Most current protocols use 1.5–2 mm isotropic voxels as the
compromise; the HBCD protocol of the simulated brain uses 1.7 mm.

:::{admonition} Simulated dataset pending
:class: note
The `voxel-sweep` dataset (1.5, 2.0, 2.5, and 3.0 mm) will show the same slice of the
simulated acquisition at each resolution, with matched noise, and the effect on FA at the
ventricle wall.
:::

## b-values

[Chapter 5](./05-diffusion-encoding.md) gave the signal per tissue as a function of b. The choice of b-values is the
choice of what to measure: b ≈ 1000 for tensor metrics, b = 2000–3000 for fiber orientation
and multi-compartment models, higher only with strong gradients. The SNR of each shell
follows from the b=0 SNR and the tissue decay: multiply the b=0 SNR by the fraction of
signal the tissue keeps at that b. The figure does this for a b=0 SNR of 30, for white
matter with the gradient across and along the fibers and for gray matter. The shaded band
is SNR below 3, where a magnitude image's noise floor ([Chapter 3](../01-mri-physics/03-reconstruction.md)) is comparable to the signal
and the measured value no longer follows the tissue. Look at where each curve enters the
band.

```{code-cell} python
:tags: [hide-input]
snr0 = 30.0
b = np.linspace(0, 3500, 300)
snr_curves = {
    "WM, gradient across fibers": (snr0 * signal.white_matter(b, 0.0), dict(color=TISSUE_COLORS["WM"], ls="-")),
    "WM, gradient along fibers": (snr0 * signal.white_matter(b, 1.0), dict(color=TISSUE_COLORS["WM"], ls="--")),
    "GM": (snr0 * signal.gray_matter(b), dict(color=TISSUE_COLORS["GM"], ls="-")),
}
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.axhspan(0, 3, color=INK["grid"], zorder=0)
ax.text(3450, 1.5, "noise-floor zone (SNR < 3)", ha="right", va="center", fontsize=8, color=INK["secondary"])
for name, (s, sty) in snr_curves.items():
    ax.plot(b, s, label=name, **sty)
for bs in (500, 1000, 2000, 3000):
    ax.axvline(bs, color=INK["secondary"], lw=0.6, ls=":")
ax.set(xlabel="b (s/mm²)", ylabel="SNR", ylim=(0, snr0 * 1.05), xlim=(0, 3500),
       title=f"SNR by b-value for SNR {snr0:.0f} at b = 0 (dotted: HBCD shells)")
ax.legend(loc="upper right")
fig.tight_layout()
print(f"SNR at b = 3000 for SNR {snr0:.0f} at b = 0: WM across {snr0 * signal.white_matter(3000, 0.0):.1f}, "
      f"WM along {snr0 * signal.white_matter(3000, 1.0):.1f}, GM {snr0 * signal.gray_matter(3000):.1f}")
```

Across the fibers white matter keeps a usable SNR at every shell. Gray matter reaches about
SNR 4 at b = 3000, just above the band, where the magnitude bias of [Chapter 3](../01-mri-physics/03-reconstruction.md) is a few
percent and denoising ([Chapter 8](../03-preprocessing/08-noise.md)) becomes necessary rather than optional. Along the fibers,
white matter falls into the band well before b = 3000.

## Number of directions and b=0 volumes

[Chapter 6](./06-qspace-sampling.md) measured the effect of direction count on tensor precision: precision improves
with the square root of the number of volumes. The same applies to b=0 volumes, which
normalize every diffusion-weighted volume; one b=0 volume per 10–15 diffusion-weighted
volumes is the usual ratio, and they should be spread through the acquisition rather than
grouped at the start.

## Partial Fourier, multiband, and in-plane acceleration

Three ways to shorten the acquisition, with different costs (Chapters [2](../01-mri-physics/02-spatial-encoding-kspace.md) and [3](../01-mri-physics/03-reconstruction.md)):

| Option | What it shortens | Cost |
|---|---|---|
| Partial Fourier (6/8, 7/8 of the lines) | EPI readout and TE, but not distortion | some blurring along phase-encode; errors where the image phase varies quickly |
| In-plane acceleration (R = 2) | EPI readout, TE, and distortion, by R | more noise: at least √R, times the g-factor, and uneven across the image |
| Multiband (2–4) | TR by the factor | g-factor noise; slice leakage; less T1 recovery at the shorter TR; dropout affects several slices at once ([Chapter 12](../03-preprocessing/12-motion-and-dropout.md)) |

The costs in plain words:

- **Partial Fourier** fills in the skipped lines using the symmetry of k-space: for a real
  image, one half of k-space mirrors the other. Real images carry a phase ([Chapter 3](../01-mri-physics/03-reconstruction.md)), and the
  fill-in works only where that phase is smooth, which it usually is. Where it changes
  quickly ("rough" phase: near air–tissue boundaries, or when the head moves during the
  diffusion encoding) the filled-in lines are wrong and the image is blurred or distorted
  in intensity there.
- **In-plane acceleration** collects R times fewer lines, so the image averages less signal
  and its noise rises by at least √R (the square root of the acceleration factor, about
  1.4 at R = 2). The reconstruction then separates the folded copies of the image using the
  coils' sensitivity patterns, and where the coils look alike (usually the middle of the
  head) that separation amplifies the noise further. The factor by which it does is the
  *g-factor*, a map that is near 1 at the edges and larger in the center.
- **Multiband** separates the simultaneously excited slices by the same coil-sensitivity
  trick, with the same g-factor noise penalty where the coils see the slices alike (though
  without the √R loss, since no lines are skipped). The separation is imperfect, and a
  faint copy of one slice can appear in another: *slice leakage*. And the time it saves
  is only free if TR stays long enough for T1 recovery: at a TR of 3 s white matter
  recovers 97 % of its signal but CSF only 53 % (the figure in the TR section).

The figure shows the readout timelines for a 128-line EPI. Each tick is one acquired
k-space line, and the diamond marks the center of k-space, where the echo forms. Look at two
things: where the diamond sits (it sets how short TE can be) and how long each bar is (the
duration of the echo train). The time between lines, the *echo spacing*, is an
illustration: it is chosen as 91.7 ms / 128 so that the unaccelerated readout lasts as long
as HBCD's total readout time. HBCD's own readout is not like the top bar: its 91.7 ms is
already an effective value that includes its in-plane acceleration (next section).

```{code-cell} python
:tags: [hide-input]
esp = presets.READOUT_HBCD_MS / 128                    # illustrative echo spacing
print(f"echo spacing {esp:.3f} ms per acquired line")
epi_cases = [("full", {}), ("6/8 partial Fourier", {"partial_fourier": 0.75}),
             ("R = 2", {"accel": 2}), ("6/8 + R = 2", {"partial_fourier": 0.75, "accel": 2})]
fig, ax = plt.subplots(figsize=(7.5, 2.9))
for i, (label, kw) in enumerate(epi_cases):
    traj = kspace.epi_trajectory(128, 128, esp, **kw)
    y = len(epi_cases) - 1 - i
    ax.barh(y, traj.readout_ms, height=0.5, color=PALETTE[i], alpha=0.25)
    ax.vlines(traj.times_ms, y - 0.25, y + 0.25, color=PALETTE[i], lw=0.5)
    ax.plot(traj.time_to_center_ms, y, "D", color=INK["primary"], ms=8, mfc="white", mew=1.5)
    ax.text(traj.readout_ms + 1.5, y, f"{traj.readout_ms:.0f} ms, center at {traj.time_to_center_ms:.0f} ms",
            va="center", fontsize=8, color=INK["secondary"])
    print(f"{label:>20}: {traj.lines.size:3d} lines, readout {traj.readout_ms:5.1f} ms, k-space center at {traj.time_to_center_ms:5.1f} ms")
ax.set_yticks(range(len(epi_cases)), [lab for lab, _ in epi_cases][::-1])
ax.set(xlabel="time from the first line (ms)", xlim=(0, 125), title="EPI readout: one tick per k-space line, ◇ = k-space center")
ax.grid(axis="y", visible=False)
fig.tight_layout()
```

Partial Fourier alone and R = 2 alone each reach the k-space center in half the time, 23 ms
instead of 46 ms; together they cut the echo train from 91 ms to 34 ms and reach the center in
a quarter of the time. Only one of them reduces susceptibility distortion. The
displacement depends on how long the readout takes to step from one line of the *full*
k-space grid to the next, the *effective* echo spacing. In-plane acceleration skips every
other line, so each step covers twice the distance in the same time: the effective echo
spacing, and with it the distortion, halves. Partial Fourier leaves out lines at one edge
of k-space but keeps the spacing of the rest, so the distortion is the same as with full
sampling; it buys a shorter TE, not a smaller displacement. The next section is the same
trade viewed from the artifact side.

## Phase-encode direction and readout time

An *off-resonance* is a spin precessing slightly faster or slower than the scanner assumes,
because the magnetic field it sits in is slightly off. The scanner reads position along the
phase-encode axis from precession phase, so a spin with extra phase is placed in the wrong
row: signal is displaced along the phase-encode axis by the frequency offset times the
total readout time ([Chapter 2](../01-mri-physics/02-spatial-encoding-kspace.md)), in voxels. The main cause is air next to tissue, which bends the
field; near the frontal sinuses and the ear canals the offset reaches 100–200 Hz. The
total readout time that sets the displacement is the *effective* one, defined on the full
reconstructed matrix: BIDS records it as `TotalReadoutTime` = effective echo spacing ×
(number of phase-encode lines in the reconstructed image − 1), where the effective echo
spacing is the time between acquired lines divided by the acceleration factor. It is
shortened by in-plane acceleration but not by partial Fourier, and it is not the duration
of the echo train. HBCD's 91.7 ms is this effective value, with HBCD's in-plane
acceleration already included.

The figure applies a smooth field map with a frontal hot spot (left, bottom) to the
synthetic b=0 slice (left, top), with the front of the head at the top and phase encoding
running top to bottom. The four panels on the right are the images the scanner would
produce at two total readout times, 30 ms (a short readout, for example from a coarser
matrix or stronger acceleration) and 90 ms (about the HBCD readout), and with the two
phase-encode polarities, anterior→posterior (AP) and
posterior→anterior (PA). Look at the frontal lobe: at 90 ms it is pushed far out of place,
in opposite directions for AP and PA; at 30 ms the same field does a third of the damage.

```{code-cell} python
:tags: [hide-input]
tissue_d = phantoms.brain_slice()
img_d = phantoms.brain_image()
field_hz = synth.synthetic_fieldmap(tissue_d["mask"], tissue_d["voxel_mm"], amplitude_hz=150.0)
readouts_ms = (30.0, 90.0)
fig = plt.figure(figsize=(9.5, 5.4))
gs = fig.add_gridspec(2, 3, width_ratios=(1, 1, 1))
ax = fig.add_subplot(gs[0, 0]); show_image(ax, img_d, "undistorted b=0", vmin=0, vmax=img_d.max())
ax = fig.add_subplot(gs[1, 0])
im = ax.imshow(np.where(tissue_d["mask"], field_hz, np.nan), cmap="RdBu_r", vmin=-150, vmax=150)
ax.set_axis_off(); ax.set_title("field offset (Hz)")
fig.colorbar(im, ax=ax, shrink=0.8)
for r, pol in enumerate((+1, -1)):
    for c, ro in enumerate(readouts_ms):
        shift_vox = pol * field_hz * ro / 1000.0          # displacement = offset × readout time, sign set by polarity
        ax = fig.add_subplot(gs[r, c + 1])
        show_image(ax, synth.displace_along_pe(img_d, shift_vox), f"{'AP' if pol > 0 else 'PA'}, readout {ro:.0f} ms",
                   vmin=0, vmax=img_d.max())
        ax.contour(tissue_d["mask"], levels=[0.5], colors=[PALETTE[3]], linewidths=0.7)
fig.tight_layout()
peak = field_hz[tissue_d["mask"]].max()
print(f"largest offset inside the brain: {peak:.0f} Hz -> {peak * 30 / 1000:.1f} voxels at 30 ms, {peak * 90 / 1000:.1f} voxels at 90 ms")
print(f"100 Hz at the HBCD readout of {presets.READOUT_HBCD_MS:.0f} ms: {100 * presets.READOUT_HBCD_MS / 1000:.1f} voxels")
```

The orange outline is the true brain edge. At the HBCD readout of 92 ms a 100 Hz offset
moves signal nine voxels. Two acquisition choices limit the damage: shorten the readout
(above), and acquire a second set of volumes with the opposite phase-encode polarity so that
the distortion can be estimated and removed ([Chapter 10](../03-preprocessing/10-susceptibility-distortion.md)). The polarity choice itself decides
whether frontal tissue is stretched or compressed; neither is better, but the choice must be
recorded correctly in the metadata (`PhaseEncodingDirection`, `TotalReadoutTime`) or the
correction will be applied backward.

## Gradient hardware

Maximum gradient amplitude sets the minimum TE at a given b ([Chapter 5](./05-diffusion-encoding.md)): at b = 3000, going
from 40 to 80 mT/m shortens the TE by about 25 ms, which is 30 % of the white matter signal. Stronger gradients also
bring larger deviations from linearity in the field they produce, which warp the image and
alter the effective b-value and direction voxel by voxel, increasingly with distance from
the isocenter. [Chapter 13](../03-preprocessing/13-gradient-nonlinearity.md) covers the correction; the acquisition-side decisions are to
position the head near the isocenter and to obtain the scanner's gradient coefficient
file, without which the correction cannot be applied.

:::{admonition} Simulated dataset pending
:class: note
The `gnl` dataset ([Chapter 13](../03-preprocessing/13-gradient-nonlinearity.md)) includes the simulated brain on an 80 mT/m whole-body and a
300 mT/m system at the same b-value; the comparison will be shown here.
:::

## Complex export and multi-echo options

Saving the phase costs nothing at acquisition and enables complex-domain denoising and
several diagnostics ([Chapter 3](../01-mri-physics/03-reconstruction.md)). Most scanners can export it as a second image series.
Multi-echo and multi-TE diffusion acquisitions ([Chapter 20](../05-advanced/20-echo-time.md)) add relaxation
information at the cost of TR or TE; they are choices for specific analyses rather than
defaults.

## Worked example: a protocol in ten minutes

The choices below assemble a multi-shell protocol for a 3 T scanner with 80 mT/m gradients
and a 32-channel head coil, within ten minutes of diffusion scanning.

```{code-cell} python
:tags: [hide-input]
voxel_mm, n_slices, mb, n_pe = 2.0, 72, 3, 120
esp_ms = presets.READOUT_HBCD_MS / n_pe                 # echo spacing: time between acquired lines (an assumed value)
overhead_ms = 10.0                                      # excitation, fat suppression, spoiling (an assumed value)
bvals, _ = schemes.hbcd()
n_ap = len(bvals)
traj_we = kspace.epi_trajectory(n_pe, n_pe, esp_ms, partial_fourier=0.75, accel=2)
te_ms = signal.min_te(3000, 80, t_post_ms=traj_we.time_to_center_ms)["te"]
after_center_ms = traj_we.readout_ms - traj_we.time_to_center_ms
t_slice_s = (te_ms + after_center_ms + overhead_ms) / 1000   # excitation to the end of the readout, plus overhead
tr_s = n_slices * t_slice_s / mb
trt_ms = esp_ms / 2 * (n_pe - 1)                        # effective readout time: R = 2 halves it, partial Fourier does not
snr0 = 30
print(f"6/8 partial Fourier, R = 2, echo spacing {esp_ms:.2f} ms: echo train {traj_we.readout_ms:.1f} ms, "
      f"k-space center at {traj_we.time_to_center_ms:.1f} ms")
print(f"b_max 3000 on 80 mT/m  ->  TE about {te_ms:.0f} ms")
print(f"time per slice: TE {te_ms:.0f} + rest of readout {after_center_ms:.0f} + overhead {overhead_ms:.0f} = {t_slice_s * 1000:.0f} ms")
print(f"voxel {voxel_mm} mm, {n_slices} slices, multiband {mb}  ->  TR {tr_s:.2f} s "
      f"(without multiband {n_slices * t_slice_s:.1f} s)")
print("recovered fraction at this TR: " + ", ".join(f"{name} {1 - np.exp(-tr_s * 1000 / t1):.0%}" for name, t1 in presets.T1_MS.items()))
print(f"effective total readout time {trt_ms:.1f} ms  ->  100 Hz moves signal {100 * trt_ms / 1000:.1f} voxels")
print(f"expected SNR at b = 3000 in WM across fibers, for SNR {snr0} at b = 0: {snr0 * signal.white_matter(3000, 0.0):.1f}")

# scan-time budget, one stacked bar per variant
shell_counts = {int(round(bv / 500) * 500): n for bv, n in schemes.shells_of(bvals).items()}
segments = [(f"b = {bv}" if bv else "b = 0", n) for bv, n in sorted(shell_counts.items())]
variants = {
    "minimal: scheme AP\n+ 6 b=0 PA": segments + [("reverse polarity", 6)],
    "HBCD: scheme AP\n+ full copy PA": segments + [("reverse polarity", n_ap)],
}
seg_colors = dict(zip([s for s, _ in segments] + ["reverse polarity"], PALETTE[:len(segments)] + [INK["secondary"]]))
fig, ax = plt.subplots(figsize=(8.5, 2.8))
for i, (name, segs) in enumerate(variants.items()):
    left = 0.0
    for lab, n in segs:
        w = schemes.scan_time_s(n, tr_s) / 60
        ax.barh(i, w, left=left, color=seg_colors[lab], label=lab if i == 0 else None, height=0.55)
        left += w
    ax.text(left + 0.1, i, f"{left:.1f} min", va="center", fontsize=8)
    print(f"{name.replace(chr(10), ' ')}: {sum(n for _, n in segs)} volumes, {left:.1f} min")
ax.axvline(10, color=INK["primary"], ls="--", lw=1); ax.text(10.1, 0.5, "10 min\nbudget", fontsize=8, va="center")
ax.set_yticks(range(len(variants)), list(variants)); ax.invert_yaxis()
ax.set(xlabel="diffusion scan time (min)", xlim=(0, 11.5), title=f"Where the time goes (TR {tr_s:.2f} s)")
ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
ax.grid(axis="y", visible=False)
fig.tight_layout()
```

The timing is computed from the protocol itself. The example assumes an echo spacing of
0.76 ms between acquired lines of a 120-line matrix (the illustrative spacing of the
figures above) and 10 ms of overhead per slice. With 6/8 partial Fourier and R = 2 the echo
train lasts 33.6 ms and reaches the k-space center after 11.5 ms, which gives a TE of about
67 ms at b = 3000. Each slice then takes TE, plus the 22 ms of readout left after the
center, plus the overhead: 99 ms. Seventy-two slices at multiband 3 need a TR of 2.37 s
(7.1 s without multiband). At that TR white matter recovers 94 % of its full signal, gray
matter 83 %, and CSF 45 %: the price of the short TR, paid in every volume. The effective
total readout time is the echo spacing divided by R, times 119: 45.5 ms, half of what the
same echo spacing would give without acceleration.

The bars show where the time goes: each colored block is the volumes of one shell, gray is
the reverse-polarity volumes. The minimal version, with six reverse-polarity b=0 volumes for
distortion correction, takes under five minutes. The HBCD protocol instead acquires the full
scheme in both polarities. That doubles the number of volumes, not the number of
directions, since the copy repeats the same directions, and it lets every diffusion-weighted
volume be corrected from both polarities ([Chapter 10](../03-preprocessing/10-susceptibility-distortion.md)).
It still fits the ten-minute budget with time to spare for a fieldmap. The table collects each decision, why it was made, what it costs, and the chapter
that tests it.

| Parameter | Choice | Why | Cost | Tested in |
|---|---|---|---|---|
| Voxel size | 2.0 mm isotropic | enough SNR for b = 3000; partial volume acceptable for tracts | cortex and ventricle walls are mixtures | this chapter, voxel size |
| Slices, multiband | 72 slices, multiband 3 | TR 2.37 s instead of 7.1 s | g-factor noise; slice leakage; CSF recovers only 45 % per TR; dropout hits three slices at once | [Chapter 12](../03-preprocessing/12-motion-and-dropout.md) |
| Readout | 6/8 partial Fourier, R = 2 | TE about 67 ms at b = 3000; effective total readout time 45.5 ms | noise × √2 × g-factor; blurring along phase-encode | [Chapter 1](../01-mri-physics/01-spins-and-signal.md), [Chapter 3](../01-mri-physics/03-reconstruction.md) |
| b-values | 0, 500, 1000, 2000, 3000 (HBCD) | kurtosis, NODDI, and MAP-MRI supported; the tensor and CSD only marginal with these direction counts (Table 6.1) | WM across fibers at SNR 18.7 at b = 3000; along fibers in the noise floor | [Chapter 8](../03-preprocessing/08-noise.md) |
| Phase encoding | AP, with a full PA copy | distortion can be estimated and removed; 100 Hz moves signal 4.5 voxels | doubles the scan time | [Chapter 10](../03-preprocessing/10-susceptibility-distortion.md) |
| Phase export | on | complex-domain denoising and diagnostics | storage only | [Chapter 3](../01-mri-physics/03-reconstruction.md), [Chapter 8](../03-preprocessing/08-noise.md) |

## What this implies for acquisition

- **Compute before scanning.** TR from slices and multiband, TE from b-value and gradient
  amplitude, scan time from volumes and TR, SNR at the highest shell from the b=0 SNR and
  tissue decay, distortion from readout time. All are one-line calculations.
- **Shorten the readout** with moderate partial Fourier and R = 2. Both reduce TE; only
  R = 2 reduces distortion, and its noise cost is acceptable.
- **Use multiband** to shorten TR and spend the saved time on volumes, keeping in mind
  its g-factor noise and that a very short TR costs signal through incomplete T1 recovery.
- **Acquire reverse-polarity volumes** and record the phase-encode metadata correctly.
- **Save the phase.**

## Further reading

Protocol design for diffusion studies is discussed in {cite:t}`jones2010` and
{cite:p}`jones2013`; the HBCD acquisition is documented in the study's protocol papers.
