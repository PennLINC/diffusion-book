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

from dwibook import kspace, phantoms, presets, synth
from dwibook.plotting import PALETTE, TISSUE_COLORS, set_style, show_image

set_style()
```

## Multi-echo readouts

A single-shot EPI readout collects one image after the spin echo ([Chapter 2](../01-mri-physics/02-spatial-encoding-kspace.md)). A multi-echo
readout repeats the EPI train two or three times after the same excitation, so that each
volume comes with several images acquired at successive echo times, separated by the
readout duration. The diffusion weighting is the same for all echoes, because the diffusion
gradients are played once, before the first readout. What differs is the transverse decay
between echoes: the later echoes are weighted by T2* decay accumulated since the spin echo,
and their SNR is correspondingly lower.

This is a different acquisition from the multi-TE one of [Chapter 20](./20-multi-te.md). There, the spin-echo
time itself is varied between scans, and the compartments separate by T2. Here, the
additional echoes are gradient echoes after one spin echo, and the decay between them is
governed by T2*, which includes the field inhomogeneity that a spin echo refocuses only at
its own echo time.

```{code-cell} python
:tags: [hide-input]
esp = presets.READOUT_HBCD_MS / 128
tr = kspace.epi_trajectory(128, 128, esp, partial_fourier=0.75, accel=2)
te1 = 70.0
echo_times = [te1 + k * (tr.readout_ms + 2.0) for k in range(3)]
t2star = {"WM": 45.0, "GM": 40.0, "CSF": 300.0}  # T2* at 3 T, approximate

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [1.5, 1]})
t = np.linspace(0, 180, 1500)
for k, te in enumerate(echo_times):
    on = (t > te - tr.readout_ms / 2) & (t < te + tr.readout_ms / 2)
    ax1.plot(t, np.where(on, np.sign(np.sin(2 * np.pi * (t - te) / (2 * esp))), np.nan) * 0.8 + 1, color=PALETTE[k], lw=1, label=f"echo {k + 1}: TE {te:.0f} ms")
ax1.axvline(te1, color="0.6", lw=1, ls="--"); ax1.text(te1, 2.0, "spin echo", ha="center", fontsize=8, color="0.4")
ax1.set(xlabel="time (ms)", yticks=[], ylim=(-0.2, 3.2), title=f"three EPI readouts of {tr.readout_ms:.0f} ms after one excitation")
ax1.legend(loc="upper right", ncol=3)
for name, t2s in t2star.items():
    ax2.plot(echo_times, [np.exp(-(te - te1) / t2s) for te in echo_times], "o-", color=TISSUE_COLORS[name], label=name)
ax2.set(xlabel="echo time (ms)", ylabel="signal relative to echo 1", title="T2* decay between echoes")
ax2.legend()
fig.tight_layout()
```

With the readout shortened by partial Fourier and in-plane acceleration to about 34 ms,
three echoes fit within 150 ms. White matter has lost about half of its signal by the
third echo; CSF almost none.

## Echo combination and T2* mapping

Three uses follow from having several echoes of every volume:

- **Combination.** A weighted sum of the echoes, with weights proportional to each echo's
  expected SNR, has a higher SNR than any single echo. The gain is modest, because the
  later echoes are weak, but it comes at no cost in scan time.
- **T2\* per volume.** Two or more echoes give the T2* decay in every voxel of every
  diffusion-weighted volume, so relaxation and diffusion are measured together. This is a
  gradient-echo relaxometry measurement, sensitive to iron and myelin, obtained inside the
  diffusion scan.
- **Recovery of signal dropout.** Signal lost in one echo to a motion event or an RF spike
  during that readout is present in the other echoes, so the volume can be repaired from
  its own echoes rather than from the model prediction of [Chapter 12](../03-preprocessing/12-motion-and-dropout.md).

The synthetic slice demonstrates the T2* estimate from two echoes: tissue T2* values are
assigned, noise is added to each echo, and T2* is estimated voxel by voxel from the ratio.

```{code-cell} python
:tags: [hide-input]
t = phantoms.brain_slice()
mask = t["mask"]
b0 = phantoms.brain_image()
t2s_map = (t["wm"] * t2star["WM"] + t["gm"] * t2star["GM"] + t["csf"] * t2star["CSF"]) / np.maximum(t["wm"] + t["gm"] + t["csf"], 1e-6)
echoes = []
for k, te in enumerate(echo_times):
    decay = np.where(mask, np.exp(-(te - te1) / np.maximum(t2s_map, 1)), 0)
    echoes.append(synth.add_noise(b0 * decay, 0.006, seed=k))
dte = echo_times[1] - echo_times[0]
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
noise_single = np.std((echoes[0] - b0)[wm])
signal_comb_true = b0 * sum(w * np.exp(-(te - te1) / 45.0) for w, te in zip(weights, echo_times)) / np.sqrt(np.sum(weights**2))
noise_comb = np.std((combined - signal_comb_true)[wm])
snr_single = b0[wm].mean() / noise_single
snr_comb = signal_comb_true[wm].mean() / noise_comb
print(f"white matter: T2* estimate {np.nanmedian(t2s_est[wm]):.0f} ms (assigned 45); SNR echo 1 {snr_single:.0f}, SNR-weighted combination of three echoes {snr_comb:.0f} (gain {snr_comb / snr_single:.2f})")
```

## Distortion and SNR per echo

Every echo is read with the same EPI train, so the susceptibility displacement ([Chapter 10](../03-preprocessing/10-susceptibility-distortion.md))
is the same in all of them and one correction applies to all. What differs is the signal
near air-tissue interfaces: the through-slice dephasing that causes signal loss in EPI grows
with echo time, so the later echoes lose more signal above the sinuses and in the temporal
poles than the first. A combination weighted by SNR handles this automatically, taking most
of its signal from the first echo where the later ones are dark.

## Measure it: the simulated datasets

:::{admonition} Simulated dataset pending
:class: note
The simulator's readout model is single-echo. A multi-echo readout (implementation plan
item T5) would add per-echo k-space with T2* decay and dephasing between echoes; until it
exists, this chapter is toy-only.
:::

## What this implies for acquisition

- **Multi-echo readouts cost TR, not directions**: each extra echo adds a readout duration
  per slice, which lengthens the minimum TR ([Chapter 7](../02-diffusion-encoding/07-acquisition-parameters.md)).
- **Shorten the readout first** (partial Fourier, in-plane acceleration) so that the extra
  echoes arrive before the signal is gone.
- **The gain is relaxometry and robustness**, not a large SNR increase; choose it when T2*
  per volume or dropout recovery is wanted.
- **Do not confuse it with multi-TE**: varying the spin-echo time ([Chapter 20](./20-multi-te.md)) measures T2
  per compartment; extra gradient echoes measure T2* per voxel.

## Further reading

Integrated diffusion-relaxometry acquisitions {cite:p}`hutter2018` and the EPI readout
timing of [Chapter 2](../01-mri-physics/02-spatial-encoding-kspace.md).
