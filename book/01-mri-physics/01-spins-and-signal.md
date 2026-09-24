---
title: Spins, precession, and the MR signal
subtitle: Chapter 1
kernelspec:
  name: python3
  display_name: Python 3
---

## Learning goals

After this chapter you can:

- state where the MR signal comes from and what T1, T2, and T2* describe
- explain, from a simulation, why a spin echo recovers signal that a simple readout loses, and
  why diffusion MRI uses one
- compute the b=0 signal of white matter, gray matter, and CSF at a given echo time using the
  tissue parameters of the book's phantom
- state the practical consequences of echo time and tissue T2 for diffusion data

**Datasets used:** none (toy tier); the phantom comparison of the three presets is planned
**Simulation tier:** toy

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt

from dwibook import phantoms, presets
from dwibook.plotting import TISSUE_COLORS, set_style, show_image

set_style()
```

## Magnetization and precession

Hydrogen nuclei (protons) have a magnetic moment. In the scanner's static field $B_0$ the
moments precess about the field direction at the Larmor frequency,

$$f_0 = \frac{\gamma}{2\pi} B_0, \qquad \frac{\gamma}{2\pi} = 42.6\ \mathrm{MHz/T},$$

which is about 128 MHz at 3 T. The moments are almost completely randomized by thermal
motion; the net alignment along the field is a few parts per million. That small net
magnetization is the quantity MRI measures.

```{code-cell} python
:tags: [hide-input]
for name, b0 in presets.B0_T.items():
    print(f"{name:>4}: Larmor frequency {presets.GAMMA_BAR_MHZ_PER_T * b0:6.1f} MHz")
```

The precession frequency depends on the local field. Any deviation of the field from $B_0$,
whether from an imperfection of the magnet, from the susceptibility of tissue and air, or
from a gradient applied on purpose, changes the frequency in proportion. This dependence is
used deliberately to encode position (Chapter 2) and appears as an artifact when the field is
inhomogeneous near air-filled sinuses (Chapter 10).

## Excitation and relaxation

A radio-frequency pulse at the Larmor frequency tips the magnetization away from the field
direction by a chosen flip angle. After a 90° pulse the magnetization lies in the transverse
plane, precesses, and induces a voltage in the receive coil. That voltage is the MR signal.

Two processes then return the magnetization to equilibrium:

- **T1 (longitudinal relaxation)** rebuilds the component along the field. It sets how much
  signal is available again for the next excitation, and therefore constrains the repetition
  time TR.
- **T2 (transverse relaxation)** is the loss of transverse magnetization as neighboring
  spins dephase one another. It sets how much signal remains at the echo time TE.
- **T2\*** is the faster decay seen in practice: T2 plus the additional dephasing caused by
  static field inhomogeneity. The inhomogeneity part is reversible, and the spin echo below
  reverses it.

The tissue values used throughout this book are those of the phantom. The T2 values are
TRXScan's `adult` compartment preset. TRXScan does not model T1, so the T1 values are 3 T
literature figures and appear only in this chapter's simulations.

```{code-cell} python
:tags: [hide-input]
tissues = presets.tissues("adult")
print(f"{'tissue':>6} {'T1 (ms)':>8} {'T2 (ms)':>8}")
for t in tissues.values():
    print(f"{t.name:>6} {t.t1_ms:8.0f} {t.t2_ms:8.0f}")
```

```{code-cell} python
:tags: [hide-input]
t = np.linspace(0, 3000, 600)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
for name, tis in tissues.items():
    mxy, mz = phantoms.bloch_free_precession(t, t1=tis.t1_ms, t2=tis.t2_ms)
    ax1.plot(t, mz, color=TISSUE_COLORS[name], label=name)
    ax2.plot(t[t <= 400], np.abs(mxy[t <= 400]), color=TISSUE_COLORS[name], label=name)
ax1.set(xlabel="time after a 90° pulse (ms)", ylabel="longitudinal magnetization", title="T1 recovery")
ax2.set(xlabel="time after a 90° pulse (ms)", ylabel="transverse magnetization", title="T2 decay")
ax2.axvline(presets.TE_HBCD_MS, color="0.6", lw=1, ls="--")
ax2.text(presets.TE_HBCD_MS + 5, 0.9, "TE = 88 ms", fontsize=8, color="0.4")
ax1.legend()
fig.tight_layout()
```

Two practical points follow from the right panel. At a typical diffusion echo time of 88 ms,
white and gray matter have lost about three quarters of their signal while CSF, with a T2 of
about two seconds, has lost almost none. And the white and gray matter curves nearly
coincide, so T2 alone provides little gray–white contrast at this TE. This is why b=0
diffusion images look flat compared with a T1-weighted anatomical image, and why CSF is the
brightest tissue in them.

## Detection: why the signal is complex

The voltage induced in the receive coil is a single real-valued oscillation at the Larmor
frequency, around 128 MHz at 3 T. Its amplitude follows the transverse magnetization and its
phase follows the precession. The receiver does not record this oscillation directly. It
mixes the coil voltage with two reference signals from the scanner's oscillator, one a cosine
at the Larmor frequency and one a sine (90° apart), and low-pass filters each product. This
is quadrature demodulation. The result is two slowly varying real channels, called in-phase
(I) and quadrature (Q), which are stored as the real and imaginary parts of one complex
number. The complex value is the transverse magnetization as seen in a frame rotating with
the reference oscillator: its magnitude is the amount of transverse magnetization and its
angle is the precession phase relative to the reference.

Two consequences follow. First, every sample the scanner records, and therefore every k-space
value and every reconstructed voxel, is complex; the magnitude image is a derived quantity
(Chapter 3). Second, which channel is called real and which imaginary is a convention, and a
constant phase offset depends on the reference oscillator, cable lengths, and receiver
electronics. The absolute phase of an image therefore has no physical meaning. Differences in
phase, between voxels, between acquisitions, or between echoes, are what carry information.

The demonstration below uses a carrier of 2 kHz instead of 128 MHz so that the oscillation is
visible, and an off-resonance of 50 Hz so that the demodulated signal has a phase that changes
with time.

```{code-cell} python
:tags: [hide-input]
fs, carrier, offset = 40_000.0, 2_000.0, 50.0  # Hz; sampling rate, carrier, off-resonance
T2 = 20.0  # ms
t = np.arange(0, 0.06, 1 / fs)  # s
coil_voltage = np.exp(-t * 1000 / T2) * np.cos(2 * np.pi * (carrier + offset) * t)  # real

# quadrature demodulation: multiply by the two references, then low-pass (moving average)
kernel = np.ones(80)
lowpass = lambda x: np.convolve(x, kernel, mode="same") / np.convolve(np.ones_like(x), kernel, mode="same")
I = lowpass(2 * coil_voltage * np.cos(2 * np.pi * carrier * t))
Q = lowpass(-2 * coil_voltage * np.sin(2 * np.pi * carrier * t))
signal = I + 1j * Q

fig, axes = plt.subplots(1, 4, figsize=(12, 3))
axes[0].plot(t[:400] * 1000, coil_voltage[:400], lw=0.8, color="0.3")
axes[0].set(title="coil voltage (real, at the carrier)", xlabel="time (ms)")
axes[1].plot(t * 1000, I, label="I (real)")
axes[1].plot(t * 1000, Q, label="Q (imaginary)")
axes[1].set(title="demodulated channels", xlabel="time (ms)"); axes[1].legend()
axes[2].plot(t * 1000, np.abs(signal))
axes[2].plot(t * 1000, np.exp(-t * 1000 / T2), color="0.5", lw=1, ls="--")
axes[2].set(title="magnitude of I + iQ (dashed: T2 decay)", xlabel="time (ms)", ylim=(0, 1.05))
axes[3].plot(t * 1000, np.unwrap(np.angle(signal)) / (2 * np.pi), color="0.3")
axes[3].set(title="phase of I + iQ (cycles)", xlabel="time (ms)")
fig.tight_layout()
```

The magnitude decays with T2 regardless of the off-resonance; the phase advances linearly
at the off-resonance frequency. That separation is what the rest of the book relies on: the
magnitude carries the tissue signal, and the phase carries the frequency offset, which
becomes a position (Chapter 2), a field map (Chapter 10), or an eddy-current signature
(Chapter 11).

## Spin echo versus gradient echo

After the 90° pulse, spins in a voxel precess at slightly different frequencies because the
field is not perfectly uniform. Their contributions drift out of phase, and the summed signal,
the free induction decay (FID), falls off with T2*. A gradient echo refocuses only the
dephasing that a gradient itself introduced, so it remains T2*-weighted.

A 180° pulse applied at time TE/2 reverses the accumulated phase of every spin. Each spin
continues to precess at its own rate, so at time TE the phases realign and a spin echo forms.
The dephasing caused by the static field is undone; only the T2 loss remains. The echo
amplitude depends on T2, not T2*.

Diffusion MRI is built on the spin echo for two reasons:

1. The diffusion-encoding gradients occupy tens of milliseconds, so the echo time is long
   (60–100 ms). A T2*-weighted readout would retain very little signal at such times; a spin
   echo retains the T2-weighted fraction.
2. The refocusing pulse reverses deterministic dephasing but not random dephasing. The
   diffusion gradients (Chapter 5) impart a phase proportional to each spin's displacement.
   Spins that moved randomly during the encoding do not refocus, and the resulting signal
   loss is the diffusion contrast.

### See it: a Bloch simulator

The simulation below follows a few thousand spins with different off-resonance frequencies.
With T2 = 80 ms and a field inhomogeneity that gives T2* = 16 ms, a 90° pulse alone produces an
FID that is gone within 40 ms. Adding a 180° pulse at 25 ms produces an echo at 50 ms whose
amplitude matches the T2 curve.

```{code-cell} python
:tags: [hide-input]
T1, T2, T2P = 1000.0, 80.0, 20.0
T2STAR = 1 / (1 / T2 + 1 / T2P)
TE = 50.0
t = np.linspace(0, 120, 2401)
offsets = phantoms.offresonance_ensemble(4000, t2prime_ms=T2P, seed=0)

fid, _ = phantoms.bloch_sequence(t, [(0.0, 90.0, 0.0)], T1, T2, offsets)
echo, _ = phantoms.bloch_sequence(t, [(0.0, 90.0, 0.0), (TE / 2, 180.0, 90.0)], T1, T2, offsets)

fig, ax = plt.subplots(figsize=(7.5, 3.2))
ax.plot(t, np.abs(fid), label="FID (90° only)")
ax.plot(t, np.abs(echo), label="spin echo (90°, then 180° at TE/2)")
ax.plot(t, np.exp(-t / T2), color="0.5", lw=1, ls="--", label="T2 decay")
ax.plot(t, np.exp(-t / T2STAR), color="0.5", lw=1, ls=":", label="T2* decay")
ax.axvline(TE / 2, color="0.8", lw=1)
ax.axvline(TE, color="0.8", lw=1)
ax.text(TE / 2 + 1, 0.95, "180°", fontsize=8, color="0.4")
ax.text(TE + 1, 0.95, "TE", fontsize=8, color="0.4")
ax.set(xlabel="time (ms)", ylabel="signal (fraction of maximum)", ylim=(0, 1.02))
ax.legend()
fig.tight_layout()
```

## Measure it: contrast at the three presets

TRXScan provides three tissue presets: `adult` (3 T literature T2 values) and `neonatal` and
`infant` (the longer T2 values of unmyelinated tissue). With a long TR, the b=0 signal of each
tissue is its proton density times the T2 decay at TE. At the HBCD echo time:

```{code-cell} python
:tags: [hide-input]
TE = presets.TE_HBCD_MS
print(f"b=0 signal at TE = {TE:.0f} ms, relative to proton density")
print(f"{'preset':>9} {'WM':>6} {'GM':>6} {'CSF':>6} {'GM/WM':>7} {'CSF/WM':>7}")
for preset, t2 in presets.T2_MS.items():
    s = {k: np.exp(-TE / v) for k, v in t2.items()}
    print(f"{preset:>9} {s['WM']:6.2f} {s['GM']:6.2f} {s['CSF']:6.2f} {s['GM']/s['WM']:7.2f} {s['CSF']/s['WM']:7.2f}")
```

The same relation applied voxel by voxel to the phantom's tissue fractions gives the synthetic
b=0 images used in the next two chapters. The adult image is dark in tissue and bright in CSF;
the neonatal image retains more tissue signal at the same TE.

```{code-cell} python
:tags: [hide-input]
fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
for ax, preset in zip(axes, ["adult", "neonatal", "infant"]):
    show_image(ax, phantoms.brain_image(te_ms=TE, preset=preset), f"{preset} preset, TE {TE:.0f} ms", vmin=0, vmax=1)
fig.tight_layout()
```

```{code-cell} python
:tags: [hide-input]
te = np.linspace(0, 200, 201)
fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
for ax, preset in zip(axes, ["adult", "neonatal"]):
    for name, t2 in presets.T2_MS[preset].items():
        ax.plot(te, np.exp(-te / t2), color=TISSUE_COLORS[name], label=name)
    ax.axvline(presets.TE_HBCD_MS, color="0.6", lw=1, ls="--")
    ax.set(title=f"{preset} preset", xlabel="TE (ms)")
axes[0].set_ylabel("b=0 signal / proton density")
axes[0].legend()
fig.tight_layout()
```

Both presets give weak gray–white contrast at 88 ms. In the adult, both tissues have short
T2 and are strongly attenuated; in the neonate, both have long T2 and are mildly attenuated.
In both, CSF is far brighter than tissue. This matters for diffusion fitting because voxels
at the edge of the ventricles and sulci contain a mixture of tissue and CSF, and the CSF
contribution dominates the b=0 signal of the mixture (Chapter 17).

:::{admonition} Phantom figure pending
:class: note
Simulated b=0 volumes of the full phantom under all three presets (dataset `presets`) will be
added when the offline pipeline produces them.
:::

## What this implies for acquisition

- **Echo time costs signal.** Each additional 10 ms of TE removes about 14 % of adult white
  matter signal. The diffusion encoding sets the minimum TE (Chapter 5); stronger gradients
  shorten it.
- **CSF dominates long-TE b=0 images.** Partial-volume CSF does not decay while tissue does.
  Fits at tissue–CSF boundaries need a free-water term (Chapter 17).
- **TR and T1.** A short TR saturates long-T1 tissue, most of all CSF. TRXScan represents this
  with a per-compartment b=0 scale rather than a T1 model; Chapter 7 uses it.
- **Age changes the numbers.** Unmyelinated tissue has long T2, so the same protocol yields
  more tissue signal and different contrast in neonates. The presets exist for this.
- **Field strength** raises the available magnetization and SNR, but shortens T2* and
  lengthens T1, which shifts the practical choices of TE and TR.

## Further reading

The original descriptions of nuclear induction {cite:p}`bloch1946` and the spin echo
{cite:p}`hahn1950`; textbook treatments in {cite:t}`haacke1999` and {cite:t}`nishimura2010`.
