---
title: Acquisition parameter choices
subtitle: Chapter 7
kernelspec:
  name: python3
  display_name: Python 3
---

## Learning goals

After this chapter you can:

- state what each of the main acquisition parameters controls, what it costs, and which
  artifact or analysis it affects: TE, TR, voxel size, b-values, number of directions and
  b=0 volumes, partial Fourier, multiband, in-plane acceleration, phase-encode direction,
  and gradient hardware
- estimate SNR, scan time, and distortion from a proposed protocol before it is run
- design a diffusion protocol under a scan-time budget and justify each choice

**Datasets used:** `voxel-sweep`, `te-sweep`, `ref-schemes` (all pending)
**Simulation tier:** toy + phantom

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt

from dwibook import kspace, phantoms, presets, schemes, signal
from dwibook.plotting import PALETTE, TISSUE_COLORS, set_style, show_image

set_style()
```

Every parameter below trades signal, time, and artifact against each other. The
relationships are simple enough to compute on paper, and the point of this chapter is to
do so before the protocol is run rather than after the data are found wanting.

## Echo time

TE is set by the diffusion encoding (Chapter 5) and the readout (Chapter 2), and the
scanner reports the minimum it can reach for a given b-value. Signal falls as
$e^{-\mathrm{TE}/T_2}$ (Chapter 1). Each 10 ms of TE costs about 14 % of adult white matter
signal and 12 % of gray matter signal; CSF is unaffected. Shortening TE is the single most
effective way to raise SNR, and the ways to do it are stronger gradients, partial Fourier,
and in-plane acceleration.

```{code-cell} python
:tags: [hide-input]
te = np.array([60, 70, 80, 88, 100, 120])
print("b=0 signal relative to TE = 60 ms (adult preset)")
print(f"{'TE':>5}" + "".join(f"{t:>8}" for t in ("WM", "GM", "CSF")))
for t in te:
    print(f"{t:>5}" + "".join(f"{np.exp(-(t - 60) / presets.T2_MS['adult'][k]):8.2f}" for k in ("WM", "GM", "CSF")))
```

## Repetition time

TR sets three things: how much longitudinal magnetization has recovered before the next
excitation, how many slices fit in one TR, and the scan time. The steady-state signal is
$1 - e^{-\mathrm{TR}/T_1}$. White matter recovers within about 3 s; CSF, with a T1 of about
4 s, does not, and at a short TR its b=0 signal is suppressed. TRXScan represents this
saturation as a per-compartment scale on the b=0 signal (the `--tissue-s0` option).

The minimum TR is the time to acquire all slices once: the number of slices times the time
per slice (about TE plus half the readout plus some overhead, roughly 120–150 ms), divided
by the multiband factor.

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
```

Seventy slices of 2 mm without multiband need a TR near 10 s; with multiband 3 it is about
3.3 s. That factor goes directly into the scan time (Chapter 6) or, at fixed scan time, into
the number of volumes.

## Voxel size

Signal is proportional to voxel volume: a 1.5 mm isotropic voxel has 3.4 mm³, a 2.5 mm voxel
15.6 mm³, a factor of 4.6 in SNR. The trade is partial volume. A 2.5 mm voxel at the cortex
or the ventricle wall contains a mixture of tissues, and its diffusion measures are the
mixture's. The synthetic slice below is shown at its native 2 mm and block-averaged to 4 mm,
with noise scaled by the volume ratio, and the fraction of brain voxels that contain a
mixture of tissue classes is counted at each size.

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
print("fraction of brain voxels that are tissue mixtures (no class above 80 %):")
for f in [1, 2]:
    print(f"  {2 * f} mm: {mixed_fraction(f):.0%}")
```

Resolution also determines what tractography can resolve: fibers that cross within a voxel
are averaged, and the models of Chapter 16 recover them only if the voxel is small relative
to the bundle geometry. Most current protocols use 1.5–2 mm isotropic voxels as the
compromise; the HBCD protocol of the phantom uses 1.7 mm.

:::{admonition} Phantom figure pending
:class: note
The `voxel-sweep` dataset (1.5, 2.0, 2.5, and 3.0 mm) will show the same slice of the
simulated acquisition at each resolution, with matched noise, and the effect on FA at the
ventricle wall.
:::

## b-values

Chapter 5 gave the signal per tissue as a function of b. The choice of b-values is the
choice of what to measure: b ≈ 1000 for tensor metrics, b = 2000–3000 for fiber orientation
and multi-compartment models, higher only with strong gradients. The SNR of each shell
follows from the b=0 SNR and the tissue decay:

```{code-cell} python
:tags: [hide-input]
snr0 = 30.0
print(f"SNR of white matter (across fibers) and gray matter by shell, for SNR {snr0:.0f} at b = 0")
for b in [0, 500, 1000, 2000, 3000]:
    print(f"  b = {b:>4}: WM {snr0 * signal.white_matter(b, 0.0):5.1f}   GM {snr0 * signal.gray_matter(b):5.1f}")
```

At b = 3000 gray matter is at SNR 4, where the magnitude bias of Chapter 3 is a few percent
and denoising (Chapter 8) becomes necessary rather than optional; along the fibers, white
matter is lower still.

## Number of directions and b=0 volumes

Chapter 6 measured the effect of direction count on tensor precision: precision improves
with the square root of the number of volumes. The same applies to b=0 volumes, which
normalize every diffusion-weighted volume; one b=0 volume per 10–15 diffusion-weighted
volumes is the usual ratio, and they should be spread through the acquisition rather than
grouped at the start.

## Partial Fourier, multiband, and in-plane acceleration

Three ways to shorten the acquisition, with different costs (Chapters 2 and 3):

| Option | What it shortens | Cost |
|---|---|---|
| Partial Fourier (6/8, 7/8) | EPI readout and TE | blurring along phase-encode; sensitive to rough phase |
| In-plane acceleration (R = 2) | EPI readout and TE by R | noise up by more than √R; spatially varying |
| Multiband (2–4) | TR by the factor | slice leakage; dropout affects several slices at once (Chapter 12) |

```{code-cell} python
:tags: [hide-input]
esp = presets.READOUT_HBCD_MS / 128
print("EPI timing for a 128-line matrix, HBCD echo spacing")
for label, kw in [("full", {}), ("6/8", {"partial_fourier": 0.75}), ("R = 2", {"accel": 2}), ("6/8 + R = 2", {"partial_fourier": 0.75, "accel": 2})]:
    tr = kspace.epi_trajectory(128, 128, esp, **kw)
    print(f"  {label:>12}: readout {tr.readout_ms:5.1f} ms, center at {tr.time_to_center_ms:5.1f} ms")
```

Partial Fourier and in-plane acceleration together halve the readout and reach the k-space
center in a quarter of the time. The readout length is what sets susceptibility distortion,
so the next section is the same trade viewed from the artifact side.

## Phase-encode direction and readout time

Off-resonance displaces signal along the phase-encode axis by the frequency offset times
the total readout time (Chapter 2). Near the frontal sinuses and the ear canals the offset
reaches 100–200 Hz.

```{code-cell} python
:tags: [hide-input]
print("displacement (voxels) = frequency offset × total readout time")
print(f"{'offset':>8}" + "".join(f"{rt:>10.0f} ms" for rt in (30, 60, 90)))
for hz in [25, 50, 100, 200]:
    print(f"{hz:>6} Hz" + "".join(f"{hz * rt / 1000:13.1f}" for rt in (30, 60, 90)))
```

At the HBCD readout of 92 ms a 100 Hz offset moves signal nine voxels. Two acquisition
choices limit the damage: shorten the readout (above), and acquire a second set of volumes
with the opposite phase-encode polarity so that the distortion can be estimated and
removed (Chapter 10). The polarity choice itself, anterior–posterior or posterior–anterior,
decides whether frontal tissue is stretched or compressed; neither is better, but the choice
must be recorded correctly in the metadata (`PhaseEncodingDirection`, `TotalReadoutTime`)
or the correction will be applied backward.

## Gradient hardware

Maximum gradient amplitude sets the minimum TE at a given b (Chapter 5): about 15 ms and
20 % of white matter signal between 40 and 80 mT/m at b = 3000. Stronger gradients also
bring larger deviations from linearity in the field they produce, which warp the image and
alter the effective b-value and direction voxel by voxel, increasingly with distance from
the isocenter. Chapter 13 covers the correction; the acquisition-side decisions are to
position the head near the isocenter and to obtain the scanner's gradient coefficient
file, without which the correction cannot be applied.

:::{admonition} Phantom figure pending
:class: note
The `gnl` dataset (Chapter 13) includes the phantom simulated on an 80 mT/m whole-body and a
300 mT/m system at the same b-value; the comparison will be shown here.
:::

## Complex export and multi-echo options

Saving the phase costs nothing at acquisition and enables complex-domain denoising and
several diagnostics (Chapter 3). Most scanners can export it as a second image series.
Multi-echo and multi-TE diffusion acquisitions (Chapters 20 and 21) add relaxation
information at the cost of TR or TE; they are choices for specific analyses rather than
defaults.

## Worked example: a protocol in ten minutes

The choices below assemble a multi-shell protocol for a 3 T scanner with 80 mT/m gradients
and a 32-channel head coil, within ten minutes of diffusion scanning.

```{code-cell} python
:tags: [hide-input]
voxel_mm, n_slices, mb, t_slice_s = 2.0, 72, 3, 0.14
tr_s = n_slices * t_slice_s / mb
bvals, _ = schemes.hbcd()
n_ap = len(bvals)
te_ms = signal.min_te(3000, 80, t_post_ms=kspace.epi_trajectory(120, 120, presets.READOUT_HBCD_MS / 120, partial_fourier=0.75, accel=2).time_to_center_ms)["te"]
print(f"voxel {voxel_mm} mm, {n_slices} slices, multiband {mb}  ->  TR {tr_s:.2f} s")
print(f"6/8 partial Fourier, R = 2, b_max 3000 on 80 mT/m  ->  TE about {te_ms:.0f} ms")
print(f"{n_ap} volumes (HBCD shells) + 6 reverse-polarity b=0  ->  {schemes.scan_time_s(n_ap + 6, tr_s) / 60:.1f} min")
print(f"{n_ap} volumes AP + {n_ap} volumes PA (full reverse-polarity copy) ->  {schemes.scan_time_s(2 * n_ap, tr_s) / 60:.1f} min")
snr0 = 30
print(f"expected SNR at b = 3000 in WM across fibers, for SNR {snr0} at b = 0: {snr0 * signal.white_matter(3000, 0.0):.1f}")
```

The minimal version, with six reverse-polarity b=0 volumes for distortion correction, takes
under five minutes. The HBCD protocol instead acquires the full scheme in both polarities,
which doubles the directions and averages the distortion correction over all volumes; it
fits the ten-minute budget with time to spare for a fieldmap. Each line of the summary is a
decision that a later chapter tests: the TE against Chapter 1, the readout against
Chapter 10, multiband 3 against Chapter 12, and the SNR at b = 3000 against Chapter 8.

## What this implies for acquisition

- **Compute before scanning.** TR from slices and multiband, TE from b-value and gradient
  amplitude, scan time from volumes and TR, SNR at the highest shell from the b=0 SNR and
  tissue decay, distortion from readout time. All are one-line calculations.
- **Shorten the readout** with moderate partial Fourier and R = 2; both reduce TE and
  distortion, and the noise cost is acceptable at R = 2.
- **Use multiband** to bring TR to 3–4 s and spend the saved time on volumes.
- **Acquire reverse-polarity volumes** and record the phase-encode metadata correctly.
- **Save the phase.**

## Further reading

Protocol design for diffusion studies is discussed in {cite:t}`jones2010` and
{cite:p}`jones2013`; the HBCD acquisition is documented in the study's protocol papers.
