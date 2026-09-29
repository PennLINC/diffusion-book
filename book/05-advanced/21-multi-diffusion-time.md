---
title: "21. Multi-diffusion-time DWI"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** random walks in restricted geometries and single-voxel signals; no pipeline dataset can show diffusion-time dependence because every simulated compartment is Gaussian ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
:::

## Learning goals

After this chapter you can:

- explain why the apparent diffusion coefficient depends on the diffusion time in
  restricted and hindered tissue, and show it with a simulation
- contrast pulsed-gradient and oscillating-gradient encoding
- state why axon diameter mapping requires strong gradients
- state why the simulated datasets cannot show any of this

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt

from dwibook import phantoms, presets, signal
from dwibook.plotting import INK, PALETTE, animate, set_style

set_style()
```

## Time-dependent diffusion

[Chapter 4](../02-diffusion-encoding/04-diffusion-in-tissue.md) showed the mean squared displacement of molecules in free, hindered, and
restricted geometries: linear in time for free water, linear with a smaller slope for
hindered water, and leveling off for restricted water. The apparent diffusion coefficient
that a measurement reports is the mean squared displacement divided by the diffusion time,
so in anything but free water it depends on how long the molecules were allowed to move.
At short diffusion times few molecules have met a wall and the ADC is close to the free
value; at long times it settles toward the hindered value, or toward zero across a closed
compartment. How soon it settles depends on how far apart the barriers are: between
obstacles a micrometer apart it is over within a fraction of a millisecond, before any
measurable diffusion time, while across a cell body ten micrometers wide it takes tens of
milliseconds, as the figure below shows. The rate at which it approaches its long-time value carries information about
the size and arrangement of the barriers {cite:p}`novikov2014,fieremans2016`.

Standard diffusion protocols fix the diffusion time by the pulse timing, at 30–50 ms, and
never see the dependence. Acquisitions that vary it do.


## See it: ADC versus diffusion time

The random walks of [Chapter 4](../02-diffusion-encoding/04-diffusion-in-tissue.md) provide the measurement directly. The diffusion encoding of
[Chapter 5](../02-diffusion-encoding/05-diffusion-encoding.md) gives each molecule a phase proportional to how far it moved along the gradient
direction during the diffusion time Δ. The constant of proportionality is 2π*q*, where *q*,
in cycles per millimeter, is set by the gradient strength and the pulse duration. The
signal is how well those phases still agree: the average, over all molecules, of an arrow
pointing at each molecule's phase. For every walk the displacement of every molecule after
a time Δ is known, so the signal, and from it the ADC, can be computed at any Δ. Here the
b-value is held at 500 s/mm² and *q* is adjusted for each Δ.

```{code-cell} python
:tags: [hide-input]
D_FREE, STEP = 3.0, 0.5  # um^2/ms, um (a fine step so that the first milliseconds are resolved)
DT = STEP**2 / (4 * D_FREE)
N_STEPS = int(100 / DT)  # 100 ms
N_WALK = 6000
walks = {
    "free": phantoms.random_walk_2d(N_WALK, N_STEPS, STEP, seed=0),
    "hindered (obstacles)": phantoms.random_walk_2d(N_WALK, N_STEPS, STEP, seed=0, radius=1.4, geometry="obstacles", spacing=4.0),
    "restricted, 12 µm cell": phantoms.random_walk_2d(N_WALK, N_STEPS, STEP, seed=0, radius=6.0, geometry="disc"),
    "restricted, 2 µm axon (across)": phantoms.random_walk_2d(N_WALK, N_STEPS, STEP, seed=0, radius=1.0, geometry="channel"),
}
deltas_ms = np.array([1, 2, 5, 10, 20, 40, 70, 100])
B_TARGET = 500.0  # s/mm^2, held fixed: q is adjusted for each diffusion time (keeps the free signal well above the Monte Carlo floor)


def adc_at(pos, d):
    """ADC (um^2/ms) along x after diffusion time d (ms), from the phases of all walkers"""
    k = int(round(d / DT))
    dx = (pos[k] - pos[0])[:, 0] * 1e-3  # mm, along x (across the channel)
    q = np.sqrt(B_TARGET / (d * 1e-3)) / (2 * np.pi)  # 1/mm, from b = (2 pi q)^2 Delta
    e = np.abs(np.mean(np.exp(2j * np.pi * q * dx)))
    return -np.log(max(e, 1e-6)) / B_TARGET * 1e3


fig, ax = plt.subplots(figsize=(7, 3.4))
adc = {}
for (name, pos), color in zip(walks.items(), PALETTE):
    adc[name] = [adc_at(pos, d) for d in deltas_ms]
    ax.plot(deltas_ms, adc[name], "o-", color=color, label=name)
ax.axhline(D_FREE, color="0.5", lw=1, ls="--")
ax.set(xlabel="diffusion time Δ (ms)", ylabel="apparent diffusion coefficient (µm²/ms)", xscale="log", title="ADC measured from the random walks of Chapter 4")
ax.legend(fontsize=8)
fig.tight_layout()
print("ADC (µm²/ms) at Δ = 1, 10, 40, 100 ms: " + "; ".join(
    f"{name} " + ", ".join(f"{a:.2f}" for d, a in zip(deltas_ms, vals) if d in (1, 10, 40, 100)) for name, vals in adc.items()))
```

Read the four curves one at a time.

- **Free water** (blue) gives the same ADC at every diffusion time, within the scatter of
  the Monte Carlo estimate, and sits on the dashed line of the free diffusivity.
- **The hindered walk** (orange) sits at its plateau across the whole range. With obstacles
  about a micrometer apart, molecules meet them within a fraction of a millisecond, and the
  slowing from the detours around them (the tortuosity) has set in before the earliest
  diffusion time shown. Time dependence in hindered tissue therefore comes from structure
  larger than the spacing of the barriers, such as the variation of axon packing along a
  bundle, and is a subtle effect.
- **The 2 µm axon** (yellow), measured across its width, is already filled by 1 ms: every
  molecule has crossed it, the displacement cannot grow any further, and from then on the
  ADC (displacement squared divided by time) simply falls toward zero as 1/Δ.
- **The 12 µm cell body** (green) is still at about two thirds of the free value at 1 ms
  (printed above) and falls steeply over the next tens of milliseconds as its molecules
  reach the walls. By the 40 ms of a standard protocol its molecules have filled it too,
  and its ADC is small and falling as 1/Δ.

The two restricted walks are not subtle, but a standard protocol at 40 ms sees both after
their molecules have filled the available space: it measures how small the displacement
is, not how quickly it stopped growing. Measuring the approach itself requires diffusion
times of a few milliseconds or structures of many micrometers. The animation
below follows the cell body. On the left, a few hundred of its molecules spread from their
starting points; on the right, the marker moves along that walk's ADC curve.

```{code-cell} python
:tags: [hide-input]
cell = walks["restricted, 12 µm cell"]
R_CELL = 6.0
anim_deltas = np.geomspace(0.25, 100, 28)
cell_adc = np.array([adc_at(cell, d) for d in anim_deltas])
show = np.arange(300)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.6), gridspec_kw={"width_ratios": [1, 1.4]})
ax1.add_patch(plt.Circle((0, 0), R_CELL, fill=False, color=INK["primary"], lw=1.2))
ax1.scatter(cell[0, show, 0], cell[0, show, 1], s=4, color=INK["grid"], zorder=1)
dots = ax1.scatter(cell[0, show, 0], cell[0, show, 1], s=6, color=PALETTE[2], zorder=2)
trails = [ax1.plot([], [], lw=0.7, color=PALETTE[c])[0] for c in (0, 1, 4)]
ax1.set(aspect="equal", xlim=(-7, 7), ylim=(-7, 7), xlabel="µm", ylabel="µm")
ax1.grid(False)
ax2.plot(anim_deltas, cell_adc, color=PALETTE[2], label="12 µm cell")
ax2.axhline(D_FREE, color="0.5", lw=1, ls="--", label="free water")
marker, = ax2.plot([], [], "o", color=PALETTE[2], ms=8)
ax2.set(xscale="log", xlabel="diffusion time Δ (ms)", ylabel="ADC (µm²/ms)", ylim=(0, 3.4))
ax2.legend(fontsize=8, loc="lower left")
ax1.set_title("Δ = 0.0 ms (gray: starting points)")
ax2.set_title("ADC at this diffusion time")
fig.tight_layout()


def frame(i):
    d = anim_deltas[i]
    k = int(round(d / DT))
    dots.set_offsets(cell[k, show])
    for w, line in enumerate(trails):
        k0 = max(0, k - int(5 / DT))  # the last 5 ms of each traced path
        line.set_data(cell[k0 : k + 1, w, 0], cell[k0 : k + 1, w, 1])
    marker.set_data([d], [cell_adc[i]])
    ax1.set_title(f"Δ = {d:.1f} ms (gray: starting points)")
    ax2.set_title("ADC at this diffusion time")


animate(fig, frame, list(range(len(anim_deltas))) + [len(anim_deltas) - 1] * 5, fps=4, width=700, dpi=70,
        alt="Left: 300 molecules inside a 12 micrometer circle spread from their starting points until they fill the circle, with the last 5 milliseconds of three paths traced; right: a marker moves along the cell's ADC curve, which starts close to the free value of 3 square micrometers per millisecond at a fraction of a millisecond and falls steeply over tens of milliseconds")
```

At the shortest times the molecules have moved much less than the cell radius, few have
touched the wall, and the ADC is close to the free value. By a few tens of milliseconds
the cloud fills the circle: the molecules' displacements are now limited by the cell's
size rather than by time, and the ADC, displacement squared divided by time, keeps
falling as time grows.

## PGSE versus OGSE

The pulsed-gradient spin echo (PGSE) of [Chapter 5](../02-diffusion-encoding/05-diffusion-encoding.md) cannot reach short diffusion times at a
useful b-value, because b grows with the square of the pulse duration times the diffusion
time, roughly the cube of the timing. The oscillating-gradient spin echo (OGSE) replaces
each pulse by a gradient that oscillates at a chosen frequency {cite:p}`does2003`. The
figure compares the two within the same timing: two 20 ms gradient windows on either side of
the 180° pulse, at 80 mT/m.

```{code-cell} python
:tags: [hide-input]
G_MAX, WIN, GAP, F_OSC = 80.0, 20.0, 6.0, 100.0  # mT/m, ms, ms (180° pulse), Hz
t = np.linspace(0, 2 * WIN + GAP, 4601)
dt_s = (t[1] - t[0]) * 1e-3
first, second = t < WIN, t >= WIN + GAP


def apodized_cosine(tw):
    """Cosine lobes over a whole number of periods, with the first and last quarter periods replaced
    by half-sine lobes at twice the frequency: the waveform starts and ends at zero, and the phase
    it winds up averages to zero over every period"""
    quarter = 1e3 / (4 * F_OSC)  # ms
    g = np.cos(2 * np.pi * F_OSC * tw * 1e-3)
    ends = (tw < quarter) | (tw > WIN - quarter)
    return G_MAX * np.where(ends, np.sin(4 * np.pi * F_OSC * np.minimum(tw, WIN - tw) * 1e-3), g)


waves = {
    "PGSE": np.where(first | second, G_MAX, 0.0),
    f"OGSE, {F_OSC:.0f} Hz": np.where(first, apodized_cosine(t), 0.0) + np.where(second, apodized_cosine(t - WIN - GAP), 0.0),
}
fig, axes = plt.subplots(2, 2, figsize=(10, 4.6), sharex=True)
b_num = {}
for row, ((name, g), color) in enumerate(zip(waves.items(), PALETTE)):
    g_eff = np.where(second, -g, g)  # the 180° pulse reverses the phase gathered before it
    q_t = signal.GAMMA * np.cumsum(g_eff * 1e-3) * dt_s / (2 * np.pi) * 1e-3  # 1/mm
    b_num[name] = np.sum((2 * np.pi * q_t) ** 2) * dt_s  # s/mm^2
    axes[row, 0].plot(t, g, color=color)
    axes[row, 0].axvspan(WIN, WIN + GAP, color=INK["grid"], lw=0)
    axes[row, 0].set(ylabel="gradient (mT/m)", title=f"{name}: gradient as played")
    axes[row, 1].plot(t, q_t, color=color)
    axes[row, 1].axvspan(WIN, WIN + GAP, color=INK["grid"], lw=0)
    axes[row, 1].set(ylabel="q (1/mm)", title=f"{name}: phase wound per unit displacement")
for ax in axes[1]:
    ax.set_xlabel("time (ms); gray: 180° pulse")
fig.tight_layout()
b_pgse = signal.b_value(G_MAX, WIN, WIN + GAP)
b_ogse = b_num[f"OGSE, {F_OSC:.0f} Hz"]
print(f"b at {G_MAX:.0f} mT/m in two {WIN:.0f} ms windows: PGSE {b_num['PGSE']:.0f} s/mm² (b_value formula: {b_pgse:.0f}), "
      f"OGSE at {F_OSC:.0f} Hz {b_ogse:.0f} s/mm², a ratio of 1/{b_num['PGSE'] / b_ogse:.0f}")
print(f"PGSE at {G_MAX:.0f} mT/m with δ = Δ = 5 ms: b = {signal.b_value(G_MAX, 5, 5):.0f} s/mm²; "
      f"with δ = Δ = 40 ms: b = {signal.b_value(G_MAX, 40, 40):.0f} s/mm²")
```

The left column shows the gradients as the scanner plays them; the right column shows *q*,
the phase each molecule gains per micrometer it moves, as it builds up during the encoding.
In PGSE, *q* is wound up during the first pulse, held while the molecules move, and unwound
by the second: a molecule's phase reports where it was at the start compared with where it
is 26 ms later. The OGSE waveform here is a cosine (its first and last quarter-periods are
softened into half-sine lobes so that the gradient starts and ends at zero, the usual
"apodized cosine"). Its *q* swings up and back down through zero every 10 ms and averages
to zero, so no part of the encoding compares positions far apart in time: the phase reports
only displacements over a few milliseconds, and the effective diffusion time is about a
quarter of the oscillation period, 2.5 ms at 100 Hz, set by the frequency rather than by
the pulse separation. (A sine-modulated waveform would not do this: its *q* stays on one
side of zero during each window, and much of its b-value would come from a slow, PGSE-like
comparison across the 180° pulse.)

The price is b-value. In the same 46 ms, at the same amplitude, the oscillating waveform
reaches about 1/160 of the PGSE b-value, as printed above. Shortening PGSE instead
fails faster still: with the pulses shortened from 40 to 5 ms (δ = Δ), a factor of 8, b
falls by 8³ = 512, from about 20 000 to about 40 s/mm². OGSE acquisitions therefore work at
low b-values, typically a few hundred s/mm², and need strong gradients to reach even those.

## Axon diameter and gradient strength

The long-time limit of the signal across an impermeable cylinder depends only on its
radius, provided the gradient pulses are short enough that molecules barely move during
them (δ ≪ *R*²/*D*, well under a millisecond for a 1 µm axon). The figure shows that
short-pulse signal against *q* for four radii, with dotted lines at the *q* that three
gradient systems reach with 10 ms pulses:

```{code-cell} python
:tags: [hide-input]
q_mm = np.linspace(0, 140, 300)  # 1/mm
fig, ax = plt.subplots(figsize=(7, 3.4))
for r_um, color in zip([0.5, 1.0, 2.0, 4.0], PALETTE):
    ax.plot(q_mm, signal.cylinder_sgp(q_mm, r_um), color=color, label=f"radius {r_um} µm")
for name, g in presets.GMAX_MT_PER_M.items():
    q_reach = signal.q_value(g, 10.0)  # 10 ms pulses
    ax.axvline(q_reach, color="0.6", lw=1, ls=":")
    ax.text(q_reach, 0.3, f"{name}\n{g:.0f} mT/m", fontsize=7, ha="center", color="0.4")
ax.set(xlabel="q (1/mm)", ylabel="signal across the cylinder", title="restricted signal versus q (short-pulse limit), and the q reachable with 10 ms pulses")
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout()
print("q reached with 10 ms pulses, and the displacement scale 1/q: " + "; ".join(
    f"{g:.0f} mT/m q = {signal.q_value(g, 10.0):.0f}/mm, 1/q = {1e3 / signal.q_value(g, 10.0):.0f} µm" for g in presets.GMAX_MT_PER_M.values()))

# With pulses much longer than R^2/D, molecules cross the axon many times during each pulse and the
# phase averages out (motional narrowing); Neuman's long-pulse limit for a cylinder
D_AX = presets.ADULT_DIFFUSIVITY["WM_intra"] * 1e-6  # m^2/s
g_con, delta_s = presets.GMAX_MT_PER_M["Connectom 300"] * 1e-3, 10e-3


def neuman_cylinder(r_um):
    r = r_um * 1e-6
    return np.exp(-7 / 96 * (signal.GAMMA * g_con) ** 2 * r**4 / D_AX * (2 * delta_s - 99 / 112 * r**2 / D_AX))


q_con = signal.q_value(presets.GMAX_MT_PER_M["Connectom 300"], 10.0)
print("at 300 mT/m with 10 ms pulses, signal across the cylinder, short-pulse curve versus long-pulse (Neuman) value: " + "; ".join(
    f"radius {r} µm {float(signal.cylinder_sgp(q_con, r)):.2f} vs {neuman_cylinder(r):.2f}" for r in (1.0, 2.0)))
```

A useful way to read *q* is through its inverse. A molecule that moves 1/*q* along the
gradient gains one full turn of phase, so 1/*q* is the displacement scale the measurement
resolves: displacements much smaller than 1/*q* barely change the phase and barely change
the signal. At the *q* of a 40 or 80 mT/m system, 1/*q* is tens of micrometers (printed
above), far larger than an axon, and cylinders of 0.5, 1, and 2 µm radius are
indistinguishable: their signal has barely decayed. Only the 300 mT/m system reaches a
1/*q* under ten micrometers, where the 1 and 2 µm curves separate, and even there the
smallest axons remain out of reach. At long times a molecule's start and end points are
independent and each is anywhere in the cross-section, so the displacement distribution
across the cylinder is the autocorrelation of the cross-section (its overlap with a shifted
copy of itself), and the signal, its Fourier transform ([Chapter 4](../02-diffusion-encoding/04-diffusion-in-tissue.md)), is the squared magnitude of
the cross-section's Fourier transform. Telling radii apart means measuring at a *q* high
enough to see the shape of that transform. For the axons of the human brain, mostly below
2 µm in diameter, that is beyond what ordinary gradients reach.

The figure is also optimistic. Pulses of 10 ms are far longer than the time a molecule takes
to cross an axon, so during each pulse it crosses many times and its phase averages over
the cross-section; the attenuation is then much weaker than the short-pulse curve
{cite:p}`neuman1974`. At 300 mT/m, the 2 µm radius curve predicts a signal of 0.51 at the
marker, but the long-pulse value is 0.92, and for a 1 µm radius 0.99 instead of 0.85
(printed above). Shortening the pulses would restore the curve but lower *q* by the same
factor, so the conclusion only strengthens.

This is the reason axon diameter mapping {cite:p}`assaf2008` is a strong-gradient technique,
why its estimates are weighted toward the largest axons in a voxel, and why claims of
diameter measurement on standard hardware should be read with the plot above in mind.

## Exchange

Water crosses membranes. Picture two rooms joined by a leaky door: over a short visit, the
people in each room stay in their room and each room can be counted on its own; over a
long one, enough people drift through the door that the two crowds mix. Compartments
behave the same way. Over diffusion times longer than the exchange time between
compartments (tens to hundreds of milliseconds in gray matter, longer across myelinated
axons), molecules move between them, the compartments blend, and their signals stop being
separable, which is a further reason compartment models depend on the diffusion time.
Exchange is measured by acquisitions that vary the time between two encodings
(filter-exchange imaging); it is absent from the simulated brain.

## Why the simulated datasets cannot show this

The simulated brain's compartments are Gaussian: a stick, a tensor, and balls, with diffusivities
that do not depend on time. Every diffusion-time-dependent effect in this chapter arises
from barriers with a size, which the simulated brain does not have. A restricted compartment,
a planned simulator extension, would add a cylinder with a radius and a time-dependent
signal; until it exists, the toy random walks are the only simulation in this book that
shows time dependence, and the simulated brain's ground truth for microstructure should be read as
long-time-limit quantities.

## What this implies for acquisition

- **Record the diffusion time** (Δ and δ) of every protocol; measured diffusivities in
  tissue depend on it.
- **Do not compare ADC or compartment fractions across protocols with different diffusion
  times** without accounting for the dependence.
- **Short diffusion times need OGSE and strong gradients**; axon diameter needs the
  strongest gradients available and still reaches only the largest axons.
- **The time dependence itself is a measurement** with several acquisitions at different
  Δ; it separates barrier geometry that a single Δ cannot.

## Further reading

Time-dependent diffusion in white matter {cite:p}`fieremans2016`, its structural
interpretation {cite:p}`novikov2014`, oscillating gradients {cite:p}`does2003`, and axon
diameter mapping {cite:p}`assaf2008`.
