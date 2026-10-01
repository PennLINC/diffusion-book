---
title: "1. Spins, precession, and the MR signal"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** Bloch-equation simulations of single spins, and a synthetic b=0 slice built from the packaged tissue maps ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
- **`presets`** (pending): b=0 volumes of the simulated brain under the adult, neonatal, and infant presets ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-presets)).

Pipeline-tier datasets are simulated offline by TRXScan ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)) and are marked *pending* until their release; the figures that need them say so where they will appear.
:::

## Learning goals

After this chapter you can:

- state where the MR signal comes from and what T1, T2, and T2* describe
- explain, from a simulation, why a spin echo recovers signal that a simple readout loses, and
  why diffusion MRI uses one
- compute the b=0 signal (the signal with no diffusion weighting) of white matter, gray
  matter, and CSF at a given echo time using the tissue parameters of the simulated brain
- state the practical consequences of echo time and tissue T2 for diffusion data

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt

from dwibook import phantoms, presets
from dwibook.plotting import INK, PALETTE, TISSUE_COLORS, animate, set_style, show_image

set_style()
```

## Magnetization and precession

Hydrogen nuclei (protons) behave like tiny bar magnets. In the scanner's static field $B_0$
they do not simply line up with the field. Like a spinning top tilted in gravity, each one
swings around the field direction, a motion called precession. The rate of that swing, the
Larmor frequency, is proportional to the field:

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
used deliberately to encode position ([Chapter 2](./02-spatial-encoding-kspace.md)) and appears as an artifact when the field is
inhomogeneous near air-filled sinuses ([Chapter 10](../03-preprocessing/10-susceptibility-distortion.md)).

## Excitation and relaxation

A radio-frequency pulse at the Larmor frequency tips the magnetization away from the field
direction by a chosen flip angle. After a 90° pulse the magnetization lies in the transverse
plane, precesses, and induces a voltage in the receive coil. That voltage is the MR signal.

The animation follows the net magnetization of a voxel, drawn as one arrow, through one such
cycle. Two things are changed so that it fits in a few seconds: the precession is slowed down
enormously (the real arrow turns 128 million times per second at 3 T), and the relaxation is
compressed, with T1 only three times T2 rather than ten or more times as in white and gray
matter (in CSF, whose T2 is very long, T1 is only about twice T2). The
orange line is the arrow's shadow on the transverse plane. That component is what the
receive coil detects, so it is the signal.

```{code-cell} python
:tags: [hide-input]
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3-D projection)

# display time runs in frames; precession is slowed and relaxation shortened so one cycle fits in a few seconds
N_REST, N_TIP, N_FREE = 8, 16, 80
PREC = 2 * np.pi / 10          # radians of precession per frame
T1_DISP, T2_DISP = 60.0, 20.0  # relaxation times in frames
path = [(0.0, 0.0, 1.0)] * N_REST
for i in range(1, N_TIP + 1):  # the RF pulse tips the vector while it precesses
    tip, phi = np.pi / 2 * i / N_TIP, PREC * i
    path.append((np.sin(tip) * np.cos(phi), np.sin(tip) * np.sin(phi), np.cos(tip)))
for i in range(1, N_FREE + 1):  # free precession: transverse part decays (T2), longitudinal part regrows (T1)
    phi, mxy, mz = PREC * (N_TIP + i), np.exp(-i / T2_DISP), 1 - np.exp(-i / T1_DISP)
    path.append((mxy * np.cos(phi), mxy * np.sin(phi), mz))
path = np.array(path)

fig = plt.figure(figsize=(8.4, 3.8))
ax3 = fig.add_subplot(1, 2, 1, projection="3d")
ax2 = fig.add_subplot(1, 2, 2)
ax3.set_axis_off()
ax3.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-0.05, 1.1))
ax3.view_init(elev=22, azim=-60)
ring = np.linspace(0, 2 * np.pi, 100)
ax3.plot(np.cos(ring), np.sin(ring), 0, color="0.85", lw=0.8)
ax3.plot([0, 0], [0, 0], [0, 1.1], color="0.6", lw=0.8, ls="--")
ax3.text(0, 0, 1.15, "B₀", color="0.4", fontsize=9)
ax3.text(1.05, 0, 0, "transverse\nplane", color="0.5", fontsize=7)
trail, = ax3.plot([], [], [], color=PALETTE[0], lw=0.6, alpha=0.5)
vec, = ax3.plot([], [], [], color=PALETTE[0], lw=3)
shadow, = ax3.plot([], [], [], color=PALETTE[1], lw=2)
stage = fig.text(0.26, 0.93, "", ha="center", va="top", fontsize=9)
frames_t = np.arange(len(path))
l_xy, = ax2.plot([], [], color=PALETTE[1], label="transverse (the signal the coil receives)")
l_z, = ax2.plot([], [], color=PALETTE[0], label="longitudinal (along B₀)")
ax2.set(xlim=(0, len(path)), ylim=(0, 1.05), xticks=[], xlabel="time", ylabel="magnetization")
ax2.legend(loc="center right", fontsize=7)
fig.tight_layout(rect=(0, 0, 1, 0.88))

def frame(i):
    x, y, z = path[i]
    vec.set_data_3d([0, x], [0, y], [0, z])
    shadow.set_data_3d([0, x], [0, y], [0, 0])
    trail.set_data_3d(path[: i + 1, 0], path[: i + 1, 1], path[: i + 1, 2])
    l_xy.set_data(frames_t[: i + 1], np.hypot(path[: i + 1, 0], path[: i + 1, 1]))
    l_z.set_data(frames_t[: i + 1], path[: i + 1, 2])
    stage.set_text("at rest: aligned with B₀" if i < N_REST else
                   "RF pulse: tipped into the transverse plane" if i < N_REST + N_TIP else
                   "precessing; transverse part decays (T2),\nlongitudinal part regrows (T1)")

animate(fig, frame, range(len(path)), fps=12, width=680, dpi=80,
        alt="an arrow representing the net magnetization points along the main field, is tipped into the transverse plane by an RF pulse while it precesses, then spirals back up: its transverse component shrinks with T2 while its longitudinal component regrows with T1; a plot alongside traces both components over time")
```

Two processes return the magnetization to equilibrium, and both are visible above:

- **T1 (longitudinal relaxation)** rebuilds the component along the field. It sets how much
  signal is available again for the next excitation, and therefore constrains the repetition
  time TR.
- **T2 (transverse relaxation)** is the loss of transverse magnetization as neighboring
  spins dephase one another. It sets how much signal remains at the echo time TE.

In practice the transverse signal decays faster than T2 alone predicts, because the field
is never perfectly uniform across a voxel and spins in slightly different fields drift out
of phase. This faster decay is **T2\***. The extra decay caused by the field differences is
sometimes given its own time constant, T2′, so that $1/T_2^* = 1/T_2 + 1/T_2'$: the two
losses add as rates. The part added by the static field is reversible, and the spin echo
below reverses it.

The tissue values used throughout this book are those of the simulated brain. The T2 values are
TRXScan's `adult` compartment preset. TRXScan does not model T1, so the T1 values are 3 T
literature figures (white and gray matter from {cite:t}`wansapura1999`; the 4 s for CSF is
approximate) and appear only in this chapter's simulations.

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
print("recovered along the field 3 s after a 90° pulse: " + ", ".join(f"{n} {1 - np.exp(-3000 / tis.t1_ms):.0%}" for n, tis in tissues.items()))
```

The left panel shows T1 recovery, which matters for how often the scanner can excite
again. Three seconds after a pulse, white matter has almost fully recovered while CSF, with
the longest T1, is only about halfway back. If the next excitation comes that soon, CSF
starts with less magnetization and gives less signal; this is the repetition-time trade
of [Chapter 7](../02-diffusion-encoding/07-acquisition-parameters.md). The rest of this chapter assumes a repetition time long enough for full recovery.

Two practical points follow from the right panel. At a typical diffusion echo time of 88 ms,
white and gray matter have lost about three quarters of their signal while CSF, with a T2 of
about two seconds, has lost almost none. And the white and gray matter curves nearly
coincide, so T2 alone provides little gray–white contrast at this TE. This is why b=0
images, the images in a diffusion series taken with the diffusion gradients off, look flat
compared with a T1-weighted anatomical image, and why CSF is the brightest tissue in them.

## Detection: why the signal is complex

The precessing magnetization induces a voltage in the receive coil, and that voltage
oscillates at the Larmor frequency, around 128 MHz at 3 T. The scanner does not keep this
fast oscillation. Instead it records the transverse magnetization as an arrow seen from a
viewpoint that itself turns at the Larmor frequency, like a camera riding on a merry-go-round.
From there, magnetization that precesses at exactly the Larmor frequency stands still, and
magnetization in a slightly stronger or weaker field turns slowly one way or the other. The
spin-echo animation later in this chapter uses the same turning viewpoint.

Each recorded sample is this arrow, stored as one complex number: two values, the arrow's
horizontal and vertical components, called the real and imaginary parts. The arrow's length,
the **magnitude**, is the amount of transverse magnetization. Its angle, the **phase**, is how
far it has turned relative to the scanner's reference.

The figure shows both views. So that the oscillation is visible, the carrier (the fast
oscillation at the Larmor frequency) here is 2 kHz instead of 128 MHz, and the magnetization is given a 50 Hz off-resonance (a field slightly
different from nominal), so that the arrow turns slowly in the turning frame.

```{code-cell} python
:tags: [hide-input]
fs, carrier, offset = 40_000.0, 2_000.0, 50.0  # Hz; sampling rate, carrier, off-resonance
T2 = 20.0  # ms
# the filter averages over 1 ms, so the oscillation is simulated from 1 ms before t = 0 and the
# extra stretch is cropped after filtering; otherwise the first samples would be averaged over
# half a window and come out distorted
t_all = np.arange(-0.001, 0.06, 1 / fs)  # s
coil_all = np.exp(-t_all * 1000 / T2) * np.cos(2 * np.pi * (carrier + offset) * t_all)  # real

# quadrature demodulation: multiply by the two references, then low-pass (a moving average, applied twice)
kernel = np.ones(20) / 20  # 0.5 ms: one carrier period
lowpass = lambda x: np.convolve(x, kernel, mode="same")
keep = t_all >= 0
t, coil_voltage = t_all[keep], coil_all[keep]
I = lowpass(lowpass(2 * coil_all * np.cos(2 * np.pi * carrier * t_all)))[keep]
Q = lowpass(lowpass(-2 * coil_all * np.sin(2 * np.pi * carrier * t_all)))[keep]
signal = I + 1j * Q

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1.5, 1]})
ax0.plot(t[:600] * 1000, coil_voltage[:600], lw=0.8, color="0.3")
ax0.plot(t[:600] * 1000, np.exp(-t[:600] * 1000 / T2), color="0.6", lw=1, ls="--", label="T2 decay")
ax0.set(title="what the coil sees: a fast oscillation", xlabel="time (ms)", ylabel="coil voltage")
ax0.legend(loc="upper right", fontsize=8)
keep = t <= 0.04
ax1.plot(signal.real[keep], signal.imag[keep], color="0.8", lw=1)
for k, t_ms in enumerate([0, 5, 10, 15, 20]):
    z = signal[int(round(t_ms / 1000 * fs))]
    c = PALETTE[(0, 1, 2, 3, 6)[k]]
    ax1.annotate("", xy=(z.real, z.imag), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=2))
    ax1.text(z.real * 1.12 + 0.02, z.imag * 1.12 + 0.02, f"{t_ms} ms", fontsize=8, color=c, ha="center", va="center")
ax1.axhline(0, color="0.85", lw=0.8, zorder=0)
ax1.axvline(0, color="0.85", lw=0.8, zorder=0)
ax1.set(xlim=(-0.8, 1.25), ylim=(-0.7, 0.98), aspect="equal", xlabel="real part", ylabel="imaginary part",
        title="what is recorded: an arrow in the turning frame")
fig.tight_layout()
print(f"phase turned per 5 ms at {offset:.0f} Hz: {360 * offset * 0.005:.0f} degrees; "
      f"magnitude at 5 ms: {np.abs(signal[int(0.005 * fs)]):.2f} (T2 decay predicts {np.exp(-5 / T2):.2f})")
```

On the left is the coil voltage: it swings up and down many times per millisecond, and
only its envelope (dashed) carries the decay. On the right is what the scanner stores: the
arrow drawn every 5 ms. Each arrow is a quarter turn further round than the one before,
because the 50 Hz offset turns the phase by 90° every 5 ms, and each is shorter, because
the transverse magnetization decays with T2. The two effects separate cleanly: the length
follows T2 whatever the offset, and the angle advances at a rate set by the offset alone.
That separation is what the rest of the book relies on: the magnitude carries the tissue
signal, and the phase carries the frequency offset, which becomes a position ([Chapter 2](./02-spatial-encoding-kspace.md)), a
field map ([Chapter 10](../03-preprocessing/10-susceptibility-distortion.md)), or an eddy-current signature ([Chapter 11](../03-preprocessing/11-eddy-currents.md)).

Two consequences follow. First, every sample the scanner records, and therefore every k-space
value and every reconstructed voxel, is complex; the magnitude image is a derived quantity
([Chapter 3](./03-reconstruction.md)). Second, the phase is measured against the scanner's own reference, and it
includes a constant offset from the reference oscillator, cable lengths, and receiver
electronics. The absolute phase of an image therefore has no physical meaning. Differences in
phase, between voxels, between acquisitions, or between echoes, are what carry information.

:::{dropdown} How the receiver produces the complex signal (quadrature demodulation)
The coil voltage is a single real-valued oscillation at the Larmor frequency. The receiver
mixes (multiplies) it with two reference signals from the scanner's oscillator, one a cosine
at the Larmor frequency and one a sine, 90° apart, and low-pass filters each product to
remove the fast components. This is quadrature demodulation. The result is two slowly
varying real channels, called in-phase (I) and quadrature (Q), which are stored as the real
and imaginary parts of one complex number. Their combination is the transverse
magnetization as seen in a frame rotating with the reference oscillator, the turning
viewpoint above. Which channel is called real and which imaginary is a convention. The
figure above was computed exactly this way, with a moving average as the low-pass filter.
:::

## Spin echo versus gradient echo

After the 90° pulse, spins in a voxel precess at slightly different frequencies because the
field is not perfectly uniform. Their contributions drift out of phase, and the summed signal,
the free induction decay (FID), falls off with T2*. One way to form an echo is a
**gradient echo**: a gradient, a deliberate variation of the field across the scanner
([Chapter 2](./02-spatial-encoding-kspace.md)), is switched on in one direction and then in the other, so that the dephasing it
caused is undone. It undoes only that dephasing, not the dephasing from the field's own
imperfections, so a gradient echo remains T2*-weighted. The alternative is a **spin echo**.

A 180° pulse applied at time TE/2 reverses the accumulated phase of every spin. Each spin
continues to precess at its own rate, so at time TE the phases realign and a spin echo forms.
The dephasing caused by the static field is undone; only the T2 loss remains. The echo
amplitude depends on T2, not T2*.

The animation shows the spins of one voxel as arrows in the transverse plane, viewed from
above in a frame that turns at the average precession rate, so that a spin at exactly the
average frequency stands still. Spins in a slightly stronger field run ahead, spins in a
weaker field fall behind, and the fan spreads; the black arrow, their sum, is the signal,
and it shrinks. The 180° pulse flips the fan over: the spins that were furthest ahead are
now furthest behind. Each keeps its own speed, so, like runners who turn around at a whistle
and run back at their own pace, they arrive together at the echo time.

In the right panel each spin also takes small random steps in phase, as it does when its
molecule moves while the diffusion gradients of [Chapter 5](../02-diffusion-encoding/05-diffusion-encoding.md) are on. Those steps are
different for every spin and are not undone by the 180° pulse, so the echo comes back
smaller. That missing signal is what diffusion MRI measures. T2 decay is left out of the
animation so that only the phases are visible.

```{code-cell} python
:tags: [hide-input]
from scipy.special import erfinv

N_ENS, N_SHOW, N_FRAMES, TE_F = 600, 12, 100, 60
rng = np.random.default_rng(1)
omega = 0.07 * np.sqrt(2) * erfinv(np.linspace(-0.99, 0.99, N_ENS))  # each spin's off-resonance, radians per frame
show = np.linspace(0, N_ENS - 1, N_SHOW).astype(int)
kicks = rng.normal(scale=0.13, size=(N_FRAMES, N_ENS))  # random phase steps from molecular motion

def phase_history(moving):
    ph, out = np.zeros(N_ENS), []
    for i in range(N_FRAMES):
        if i == TE_F // 2:
            ph = -ph  # the 180° pulse reverses every spin's phase
        out.append(ph.copy())
        ph = ph + omega + (kicks[i] if moving else 0.0)
    return np.array(out)

cases = {"field differences only": phase_history(False), "field differences + random motion": phase_history(True)}
sums = {name: np.abs(np.exp(1j * ph).mean(axis=1)) for name, ph in cases.items()}
colors = plt.cm.twilight(np.linspace(0.1, 0.9, N_SHOW))

fig = plt.figure(figsize=(8, 5.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1])
fans, totals = [], []
for col, name in enumerate(cases):
    ax = fig.add_subplot(gs[0, col])
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="0.85"))
    ax.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), aspect="equal", title=name)
    ax.set_axis_off()
    fans.append([ax.plot([], [], color=c, lw=1.4)[0] for c in colors])
    totals.append(ax.annotate("", xy=(0, 1), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=INK["primary"], lw=2.2)))
axs = fig.add_subplot(gs[1, :])
lines = [axs.plot([], [], color=PALETTE[k], label=name)[0] for k, name in enumerate(cases)]
for x, label in [(TE_F // 2, "180°"), (TE_F, "TE: echo")]:
    axs.axvline(x, color="0.75", lw=1)
    axs.text(x + 1, 1.0, label, fontsize=8, color="0.4", va="top")
axs.set(xlim=(0, N_FRAMES), ylim=(0, 1.05), xticks=[], xlabel="time", ylabel="summed signal")
axs.legend(loc="lower left", fontsize=8)
stage = fig.suptitle("", fontsize=10)
fig.tight_layout()

def frame(i):
    for fan, total, ph in zip(fans, totals, cases.values()):
        for line, p in zip(fan, ph[i, show]):
            line.set_data([0, -np.sin(p)], [0, np.cos(p)])
        s = np.exp(1j * ph[i]).mean()
        total.xy = (-s.imag, s.real)
    for line, name in zip(lines, cases):
        line.set_data(np.arange(i + 1), sums[name][: i + 1])
    stage.set_text("the spins drift apart and the sum shrinks" if i < TE_F // 2 else
                   "the 180° pulse reverses the phases: the fastest spins are now furthest behind" if i < TE_F - 6 else
                   "spin echo: the phases line up again" if i <= TE_F + 3 else "and drift apart again")

animate(fig, frame, range(N_FRAMES), fps=12, width=620, dpi=65,
        alt="arrows for twelve spins start together, fan out because each precesses at a slightly different rate, are mirrored by the 180-degree pulse, and fan back into line at the echo time; in a second panel the spins also take random phase steps from molecular motion, and at the echo they only partly realign, so the echo is smaller")
```

Diffusion MRI is built on the spin echo for two reasons:

1. The diffusion-encoding gradients occupy tens of milliseconds, so the echo time is long
   (60–100 ms). A T2*-weighted readout would retain very little signal at such times; a spin
   echo retains the T2-weighted fraction.
2. The refocusing pulse reverses deterministic dephasing but not random dephasing. The
   diffusion gradients ([Chapter 5](../02-diffusion-encoding/05-diffusion-encoding.md)) impart a phase proportional to each spin's displacement.
   Spins that moved randomly during the encoding do not refocus, and the resulting signal
   loss is the diffusion contrast.

### See it: a Bloch simulator

The simulation below follows a few thousand spins with different off-resonance frequencies.
With T2 = 80 ms and a field inhomogeneity with T2′ = 20 ms, which gives T2* = 16 ms, a 90° pulse alone produces an
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

TRXScan provides three tissue presets: `adult` (T2 values for adult tissue at 3 T) and `neonatal` and
`infant` (the longer T2 values of unmyelinated tissue {cite:p}`leppert2009`). With a long TR, the b=0 signal of each
tissue is its **proton density**, the amount of MR-visible hydrogen it holds relative to pure
water, times the T2 decay at TE. The table leaves proton density out and shows only the
decay: the fraction of each tissue's signal that survives to the HBCD echo time of 88 ms
{cite:p}`dean2024`.

```{code-cell} python
:tags: [hide-input]
TE = presets.TE_HBCD_MS
print(f"b=0 signal at TE = {TE:.0f} ms, relative to proton density")
print(f"{'preset':>9} {'WM':>6} {'GM':>6} {'CSF':>6} {'GM/WM':>7} {'CSF/WM':>7}")
for preset, t2 in presets.T2_MS.items():
    s = {k: np.exp(-TE / v) for k, v in t2.items()}
    print(f"{preset:>9} {s['WM']:6.2f} {s['GM']:6.2f} {s['CSF']:6.2f} {s['GM']/s['WM']:7.2f} {s['CSF']/s['WM']:7.2f}")
```

Read the last two columns. In every preset gray matter is barely brighter than white matter
(a ratio near 1), so the echo time produces little gray–white contrast. CSF is several times
brighter than white matter in the adult but less than twice as bright in the neonate,
because neonatal tissue keeps much more of its own signal. The infant preset has the same
T2 values as the neonatal one, so its row is identical.

The same relation applied voxel by voxel to the simulated brain's tissue fractions, now
including proton density, gives the synthetic b=0 images used in the next two chapters. All
three are shown on the same gray scale. Look at the tissue around the ventricles: dark in
the adult, mid-gray in the neonate, while the ventricles are bright in both.

```{code-cell} python
:tags: [hide-input]
fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
for ax, preset in zip(axes, ["adult", "neonatal", "infant"]):
    show_image(ax, phantoms.brain_image(te_ms=TE, preset=preset), f"{preset} preset, TE {TE:.0f} ms", vmin=0, vmax=1)
fig.tight_layout()
```

The decay curves show where these numbers come from and how they would change with the
echo time. The dashed line is the HBCD echo time; each curve's height there is one entry of
the table.

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

In the adult, white and gray matter both have short T2 and fall steeply together; in the
neonate both have long T2 and fall slowly together. Either way the two tissue curves stay
close, and CSF stays near the top.

### Worked example: a voxel at the edge of a ventricle

The brightness of CSF matters most in voxels that are only partly CSF, at the edges of the
ventricles and in the sulci. Take an adult voxel that is 80 % white matter and 20 % CSF by
volume, imaged at TE = 88 ms. Each tissue contributes its volume fraction times its proton
density times its T2 decay:

```{code-cell} python
:tags: [hide-input]
frac = {"WM": 0.8, "CSF": 0.2}
contrib = {k: frac[k] * phantoms.PROTON_DENSITY[k] * np.exp(-TE / presets.T2_MS["adult"][k]) for k in frac}
total = sum(contrib.values())
for k, c in contrib.items():
    print(f"{k:>4}: {frac[k]:.0%} of the volume x proton density {phantoms.PROTON_DENSITY[k]:.2f} "
          f"x T2 decay {np.exp(-TE / presets.T2_MS['adult'][k]):.2f} = {c:.3f}  ({c / total:.0%} of the voxel's b=0 signal)")
```

CSF fills a fifth of the voxel but supplies more than half of its b=0 signal. Any model
fitted to this voxel sees mostly free water, not white matter, which is why partial-volume
CSF has to be handled explicitly ([Chapter 17](../04-modeling/17-microstructure-models.md)).

:::{admonition} Simulated dataset pending
:class: note
Simulated b=0 volumes of the full simulated brain under all three presets (dataset `presets`) will be
added when the offline pipeline produces them.
:::

## What this implies for acquisition

- **Echo time costs signal.** Each additional 10 ms of TE removes about 14 % of adult white
  matter signal. The diffusion encoding sets the minimum TE ([Chapter 5](../02-diffusion-encoding/05-diffusion-encoding.md)); stronger gradients
  shorten it.
- **CSF dominates long-TE b=0 images.** Partial-volume CSF does not decay while tissue does,
  as the worked example shows. Fits at tissue–CSF boundaries need a free-water term, an extra
  model component for freely diffusing water ([Chapter 17](../04-modeling/17-microstructure-models.md)).
- **TR and T1.** A short TR leaves long-T1 tissue, most of all CSF, only partly recovered
  (saturated), so it gives less signal. The book's simulator does not model T1 directly; it
  imitates this by scaling down each tissue's b=0 signal, as [Chapter 7](../02-diffusion-encoding/07-acquisition-parameters.md) shows.
- **Age changes the numbers.** Unmyelinated tissue has long T2 {cite:p}`leppert2009`, so the same protocol yields
  more tissue signal and different contrast in neonates. The presets exist for this.
- **Field strength** raises the available magnetization and SNR, but shortens T2* and
  lengthens T1, which shifts the practical choices of TE and TR.

## Further reading

The original descriptions of nuclear induction {cite:p}`bloch1946` and the spin echo
{cite:p}`hahn1950`; textbook treatments in {cite:t}`haacke1999` and {cite:t}`nishimura2010`.
