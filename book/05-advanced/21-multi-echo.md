---
title: "21. Multi-echo diffusion MRI"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** a synthetic b=0 slice built from the packaged tissue maps, read out with several echoes in the page ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
:::

## Learning goals

After this chapter you can:

- describe a multi-echo EPI readout and say how the echoes differ in weighting, SNR, and
  distortion
- combine echoes and estimate T2* per voxel from a diffusion acquisition
- state what multi-echo readouts add to a diffusion protocol and what they cost

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

from dwibook import kspace, phantoms, presets, synth
from dwibook.plotting import INK, PALETTE, TISSUE_COLORS, set_style, show_image

set_style()
```

## Multi-TE versus multi-echo

[Chapter 20](./20-multi-te.md) and this chapter both collect images at several echo times, but in different
ways, and the difference decides what the extra images measure. The timing diagram below
draws both. Time runs left to right. The black bars are the RF pulses (the 90° excitation
and the 180° refocusing pulse), the orange blocks are the diffusion gradient pulses, the
blue boxes are the EPI readouts that collect each image ([Chapter 2](../01-mri-physics/02-spatial-encoding-kspace.md)), and the dotted lines mark
the spin echo, the moment at which the 180° pulse has brought the spins back into phase.

```{code-cell} python
:tags: [hide-input]
esp = presets.READOUT_HBCD_MS / 128
tr = kspace.epi_trajectory(128, 128, esp, partial_fourier=0.75, accel=2)
RO = tr.readout_ms
te1 = 70.0
echo_times = [te1 + k * (RO + 2.0) for k in range(3)]
t2star = {"WM": 45.0, "GM": 40.0, "CSF": 300.0}  # T2* at 3 T, approximate
T180, T_POST = 6.0, 23.0  # ms: refocusing pulse window; end of 2nd gradient pulse to readout center
DELTA = (te1 - T180 - 2 * T_POST) / 2  # gradient pulse duration that fits the shortest echo time


def draw_scan(ax, y, te, readouts):
    """one spin-echo diffusion scan on the line at height y; readouts = [(center, label), ...]"""
    ax.hlines(y, 0, 185, color=INK["grid"], lw=1)
    ax.add_patch(Rectangle((0, y), 2.0, 0.55, color=INK["primary"]))
    ax.text(1, y + 0.62, "90°", ha="center", fontsize=7)
    ax.add_patch(Rectangle((te / 2 - 1.5, y), 3.0, 0.55, color=INK["primary"]))
    ax.text(te / 2, y + 0.62, "180°", ha="center", fontsize=7)
    for start in (te / 2 - T180 / 2 - DELTA, te / 2 + T180 / 2):
        ax.add_patch(Rectangle((start, y), DELTA, 0.35, color=PALETTE[1], alpha=0.85))
    for center, label in readouts:
        ax.add_patch(Rectangle((center - RO / 2, y), RO, 0.45, facecolor=PALETTE[0], alpha=0.3, edgecolor=PALETTE[0]))
        ax.text(center, y + 0.22, label, ha="center", va="center", fontsize=7)
    ax.plot([te, te], [y - 0.12, y + 0.75], ls=":", color=INK["secondary"], lw=1)


fig, (axa, axb, axc) = plt.subplots(3, 1, figsize=(9, 6.2), sharex=True, gridspec_kw={"height_ratios": [2.3, 1, 1.6]})
multi_te = [70.0, 100.0, 130.0]
for row, te in enumerate(multi_te):
    y = 2.0 * (len(multi_te) - 1 - row)
    draw_scan(axa, y, te, [(te, f"TE {te:.0f} ms")])
    axa.text(-3, y + 0.2, f"scan {row + 1}", ha="right", va="center", fontsize=8)
axa.set(ylim=(-0.4, 5.0), yticks=[], title="multi-TE (Chapter 20): one scan per echo time; the 180° pulse moves to TE / 2")
draw_scan(axb, 0.0, te1, [(te, f"echo {k + 1}, TE {te:.0f} ms") for k, te in enumerate(echo_times)])
axb.text(-3, 0.2, "one scan", ha="right", va="center", fontsize=8)
axb.set(ylim=(-0.3, 1.0), yticks=[], title="multi-echo (this chapter): one 180° pulse, one spin echo, three readouts")
for ax in (axa, axb):
    ax.grid(False)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
axa.legend(handles=[Patch(color=INK["primary"], label="RF pulse"), Patch(color=PALETTE[1], label="diffusion gradient"),
                    Patch(facecolor=PALETTE[0], alpha=0.3, edgecolor=PALETTE[0], label="EPI readout")],
           loc="upper right", fontsize=7, ncol=3)

# white matter signal (b = 0) through the multi-echo scan: T2 decay times reversible dephasing (T2')
t2_wm = presets.T2_MS["adult"]["WM"]
t2_prime = 1 / (1 / t2star["WM"] - 1 / t2_wm)
t = np.linspace(0, 185, 800)
dephase_time = np.where(t < te1 / 2, t, np.abs(t - te1))  # the 180° pulse reverses the dephasing
envelope = np.exp(-t / t2_wm) * np.exp(-dephase_time / t2_prime)
axc.plot(t, np.exp(-t / t2_wm), ls="--", color=INK["secondary"], lw=1, label=f"T2 decay alone (T2 {t2_wm:.0f} ms)")
axc.plot(t, np.exp(-dephase_time / t2_prime), ls=":", color=PALETTE[3], lw=1.5, label="effect of static field differences alone (1 = phases agree)")
axc.plot(t, envelope, color=TISSUE_COLORS["WM"], lw=1.8, label=f"white matter signal (T2* {t2star['WM']:.0f} ms)")
for k, te in enumerate(echo_times):
    axc.axvspan(te - RO / 2, te + RO / 2, color=PALETTE[0], alpha=0.1, lw=0)
    s_te = np.exp(-te / t2_wm) * np.exp(-abs(te - te1) / t2_prime)
    axc.plot(te, s_te, "o", color=TISSUE_COLORS["WM"])
axc.axvline(te1 / 2, color=INK["primary"], lw=0.8)
axc.text(te1 / 2 + 1, 0.9, "180°", fontsize=7)
axc.set(xlabel="time after excitation (ms)", ylabel="signal (b = 0)", ylim=(0, 1.02), xlim=(-25, 185),
        title="white matter signal during the multi-echo scan")
axc.legend(fontsize=7, loc="upper right")
fig.tight_layout()
print(f"readout {RO:.0f} ms; echo times " + ", ".join(f"{te:.0f}" for te in echo_times) + " ms")
```

In the top panel, the multi-TE acquisition of [Chapter 20](./20-multi-te.md), each echo time is a separate scan.
The 180° pulse sits at half the echo time, and the spin echo and its readout move with it,
so every image is collected at its own spin echo and is weighted by T2 alone. In the middle
panel, the multi-echo acquisition of this chapter, there is one excitation, one 180° pulse,
and one spin echo, and the EPI readout is repeated twice more after the first. The
diffusion gradients are played once, before the first readout, so all three echoes carry
the same diffusion weighting.

The bottom panel shows what the later echoes measure. The spin-echo animation of
[Chapter 1](../01-mri-physics/01-spins-and-signal.md) showed the spin phases fanning out after excitation, being flipped by the 180°
pulse, lining up again at the echo time, and then drifting apart again. The dotted curve
is that story on its own: the loss caused by small static differences in the magnetic
field across the voxel grows after excitation, is undone by the 180° pulse so that the
phases agree again exactly at the spin echo, and grows again afterwards. The solid curve is
the white matter signal, the dotted effect multiplied by the true T2 decay (dashed). It
falls at the T2* rate after excitation (T2* is the decay time that includes the static
dephasing), meets the dashed T2 curve at the spin echo, and falls at the T2* rate again
after it.
Echo 1 sits on the spin echo and is T2-weighted; echoes 2 and 3 carry additional T2* decay
over the time since the spin echo, and their SNR is correspondingly lower.

```{code-cell} python
:tags: [hide-input]
fig, ax = plt.subplots(figsize=(5, 3.2))
for name, t2s in t2star.items():
    ax.plot(echo_times, [np.exp(-(te - te1) / t2s) for te in echo_times], "o-", color=TISSUE_COLORS[name], label=f"{name} (T2* {t2s:.0f} ms)")
ax.set(xlabel="echo time (ms)", ylabel="signal relative to echo 1", title="T2* decay between echoes", xticks=echo_times)
ax.legend(fontsize=8)
fig.tight_layout()
print("signal at echo 3 relative to echo 1: " + ", ".join(f"{name} {np.exp(-(echo_times[2] - te1) / t2s):.2f}" for name, t2s in t2star.items()))
```

With the readout shortened by partial Fourier and in-plane acceleration to about 34 ms,
the three echoes are centered at 70, 106, and 141 ms. By the third echo white matter keeps
only about a fifth of its signal, while CSF, with its long T2*, keeps about four fifths.

## Echo combination and T2* mapping

Three uses follow from having several echoes of every volume:

- **Combination.** A weighted sum of the echoes, with weights proportional to each echo's
  expected SNR, has a higher SNR than any single echo. The gain is modest, because the
  later echoes are weak, but it comes at no cost in scan time.
- **T2\* per volume.** Two or more echoes give the T2* decay in every voxel of every
  diffusion-weighted volume, so relaxation and diffusion are measured together. This is a
  gradient-echo relaxometry measurement, sensitive to iron and myelin, obtained inside the
  diffusion scan.
- **Recovery of readout-time corruption.** Signal corrupted in one echo by an event during
  that readout, such as an RF spike, is intact in the other echoes, so the volume can be
  repaired from its own echoes rather than from the model prediction of [Chapter 12](../03-preprocessing/12-motion-and-dropout.md). This
  does not extend to the usual motion dropout: motion during the diffusion encoding
  dephases the signal before the first readout begins, so every echo loses it together.

The T2* estimate from two echoes is one line of arithmetic. Between echoes ΔTE apart the
signal is multiplied by exp(−ΔTE / T2*), so T2* = ΔTE / ln(S₁ / S₂), where S₁ and S₂ are the
first and second echoes. The synthetic slice below applies it voxel by voxel: tissue T2*
values are assigned, noise is added to each echo, and T2* is estimated from the ratio.

```{code-cell} python
:tags: [hide-input]
t = phantoms.brain_slice()
mask = t["mask"]
b0 = phantoms.brain_image(te1)  # the b = 0 image at the spin echo (echo 1)
t2s_map = (t["wm"] * t2star["WM"] + t["gm"] * t2star["GM"] + t["csf"] * t2star["CSF"]) / np.maximum(t["wm"] + t["gm"] + t["csf"], 1e-6)
echoes = []
for k, te in enumerate(echo_times):
    decay = np.where(mask, np.exp(-(te - te1) / np.maximum(t2s_map, 1)), 0)
    echoes.append(synth.add_noise(b0 * decay, 0.006, seed=k))
dte = echo_times[1] - echo_times[0]
ratio_wm = np.exp(-dte / t2star["WM"])
print(f"worked example, white matter: echoes {dte:.0f} ms apart, S2 / S1 = {ratio_wm:.2f}, T2* = {dte:.0f} / ln(1 / {ratio_wm:.2f}) = {dte / np.log(1 / ratio_wm):.0f} ms")
ratio = np.clip(echoes[1] / np.maximum(echoes[0], 1e-6), 1e-3, 0.999)
t2s_est = np.where(mask, -dte / np.log(ratio), np.nan)
weights = np.array([np.exp(-(te - te1) / 45.0) for te in echo_times])
combined = sum(w * e for w, e in zip(weights, echoes)) / np.sqrt(np.sum(weights**2))

fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
show_image(axes[0], echoes[0], f"echo 1, TE {echo_times[0]:.0f} ms", vmin=0, vmax=0.5)
show_image(axes[1], echoes[2], f"echo 3, TE {echo_times[2]:.0f} ms", vmin=0, vmax=0.5)
show_image(axes[2], combined, "SNR-weighted combination", vmin=0, vmax=0.5)
show_image(axes[3], t2s_est, "T2* from echoes 1 and 2 (ms)", kind="scalar", vmin=20, vmax=120)
fig.tight_layout()
wm = t["wm"] > 0.9
csf_vox = t["csf"] > 0.9
noise_single = np.std((echoes[0] - b0)[wm])
signal_comb_true = b0 * sum(w * np.exp(-(te - te1) / 45.0) for w, te in zip(weights, echo_times)) / np.sqrt(np.sum(weights**2))
noise_comb = np.std((combined - signal_comb_true)[wm])
snr_single = b0[wm].mean() / noise_single
snr_comb = signal_comb_true[wm].mean() / noise_comb
gain = snr_comb / snr_single
print(f"white matter: T2* estimate {np.nanmedian(t2s_est[wm]):.0f} ms (assigned 45); SNR echo 1 {snr_single:.0f}, "
      f"SNR-weighted combination of three echoes {snr_comb:.0f} (gain {gain:.2f}, worth {gain**2:.2f} times as many averages)")
q25, q50, q75 = np.nanpercentile(t2s_est[csf_vox], [25, 50, 75])
print(f"CSF: signal falls by {1 - np.exp(-dte / t2star['CSF']):.0%} between echoes 1 and 2; "
      f"T2* estimate median {q50:.0f} ms (assigned 300), middle half {q25:.0f}-{q75:.0f} ms")
```

The first three panels share one gray scale. Echo 3 is dark everywhere except in CSF, which
has kept most of its signal; the combination looks like echo 1 with slightly less noise.
The SNR gain printed above is about a tenth, the same improvement as acquiring about 1.2
times as many volumes, and it costs no extra scan time. In white matter the second echo is
0.45 of the first, which gives the assigned 45 ms back. The gray scale is set for tissue,
so the ventricles, much brighter than tissue at every echo, are clipped to white. The T2*
map is clipped too: CSF sits at the top of its color scale because its assigned T2* of
300 ms lies beyond the 120 ms range shown. The estimate itself is close to the assigned
value (printed above) even though CSF loses only about a tenth of its signal between
echoes 1 and 2, because CSF is bright enough that this small change stands clear of the
noise; a dim tissue with a long T2* would not.

## Distortion and SNR per echo

Every echo is read with the same EPI train, so the susceptibility displacement ([Chapter 10](../03-preprocessing/10-susceptibility-distortion.md))
is the same in all of them and one correction applies to all. What differs is the signal
near air-tissue interfaces. There the field changes steeply across the thickness of a
slice, the spins at the top and bottom of the slice precess at different rates, and their
phases spread apart. The spread grows with the time since the spin echo, so the later
echoes lose more signal above the sinuses and in the temporal poles than the first. The toy
below places a field that varies through the slice, strongest at one spot at the front of
the brain, and shows each echo brightened by the expected T2* decay of white matter, so
that white matter has the same brightness in all three and the extra loss stands out
(gray matter and the noise still change from echo to echo):

```{code-cell} python
:tags: [hide-input]
rows, cols = np.indices(mask.shape)
r_front = np.argwhere(mask)[:, 0].min() + 4  # a few voxels inside the anterior edge (top of the image)
c_mid = int(np.median(np.argwhere(mask)[:, 1]))
SLICE_MM = t["voxel_mm"]
G_PEAK = 5.0  # Hz per mm through the slice, at the center of the spot
g_through = G_PEAK * np.exp(-((rows - r_front) ** 2 + (cols - c_mid) ** 2) / (2 * 8.0**2))


def through_slice_loss(time_ms):
    """magnitude of the slice-averaged phase factor: a linear phase across the slice averages to a sinc"""
    return np.abs(np.sinc(g_through * SLICE_MM * time_ms * 1e-3))


fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
for k, (ax, te) in enumerate(zip(axes, echo_times)):
    shown = echoes[k] * through_slice_loss(te - te1) / np.exp(-(te - te1) / t2star["WM"])
    show_image(ax, shown, f"echo {k + 1}: {te - te1:.0f} ms after the spin echo", vmin=0, vmax=0.5)
    ax.add_patch(plt.Circle((c_mid, r_front), 16, fill=False, color=PALETTE[3], lw=1, ls="--"))
show_image(axes[3], np.where(mask, through_slice_loss(echo_times[2] - te1), np.nan), "fraction of signal kept, echo 3", kind="scalar", vmin=0, vmax=1)
fig.tight_layout()
print("signal kept at the center of the spot: " + ", ".join(f"echo {k + 1} {float(through_slice_loss(te - te1)[r_front, c_mid]):.2f}" for k, te in enumerate(echo_times)))
```

Look inside the dashed circle. Echo 1 is centered on the spin echo, which refocuses the
static field, so in this toy it loses nothing there. Echo 2 has lost about a fifth of the
signal at the center of the spot and echo 3 nearly two thirds, as printed above; the
right-hand panel maps the fraction echo 3 keeps, which is one everywhere outside the spot.
(In a real acquisition the first
echo also loses some signal, because its readout extends about 17 ms either side of the
spin echo.) A combination weighted by SNR handles this well, because it already takes most
of its signal from the first echo; a T2* map computed near such a spot, on the other hand,
reports the extra loss as a short T2*.

## Measure it: the simulated datasets

:::{admonition} Simulated dataset pending
:class: note
The simulator's readout model is single-echo. A multi-echo readout, a planned
simulator extension, would add per-echo k-space with T2* decay and dephasing between echoes; until it
exists, this chapter is toy-only.
:::

## What this implies for acquisition

- **Multi-echo readouts cost TR, not directions**: each extra echo adds a readout duration
  per slice, which lengthens the minimum TR ([Chapter 7](../02-diffusion-encoding/07-acquisition-parameters.md)).
- **Shorten the readout first** (partial Fourier, in-plane acceleration) so that the extra
  echoes arrive before the signal is gone.
- **The gain is relaxometry and robustness**, not a large SNR increase; choose it when T2*
  per volume or repair of readout-time corruption is wanted.
- **Do not confuse it with multi-TE**: varying the spin-echo time ([Chapter 20](./20-multi-te.md)) measures T2
  per compartment; extra gradient echoes measure T2* per voxel.

## Further reading

Integrated diffusion-relaxometry acquisitions {cite:p}`hutter2018` and the EPI readout
timing of [Chapter 2](../01-mri-physics/02-spatial-encoding-kspace.md).
