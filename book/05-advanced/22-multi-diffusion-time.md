---
title: Multi-diffusion-time DWI
subtitle: Chapter 22
kernelspec:
  name: python3
  display_name: Python 3
---

## Learning goals

After this chapter you can:

- explain why the apparent diffusion coefficient depends on the diffusion time in
  restricted and hindered tissue, and show it with a simulation
- contrast pulsed-gradient and oscillating-gradient encoding
- state why axon diameter mapping requires strong gradients
- state why the phantom cannot show any of this

**Datasets used:** none (toy tier); no phantom demonstration is possible with Gaussian compartments
**Simulation tier:** toy

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt

from dwibook import phantoms, presets, signal
from dwibook.plotting import PALETTE, set_style

set_style()
```

## Time-dependent diffusion

Chapter 4 showed the mean squared displacement of molecules in free, hindered, and
restricted geometries: linear in time for free water, linear with a smaller slope for
hindered water, and leveling off for restricted water. The apparent diffusion coefficient
that a measurement reports is the mean squared displacement divided by the diffusion time,
so in anything but free water it depends on how long the molecules were allowed to move.
At short diffusion times few molecules have met a wall and the ADC is close to the free
value; at long times it settles toward the hindered value, or toward zero across a closed
compartment. The rate at which it approaches its long-time value carries information about
the size and arrangement of the barriers {cite:p}`novikov2014,fieremans2016`.

Standard diffusion protocols fix the diffusion time by the pulse timing, at 30–50 ms, and
never see the dependence. Acquisitions that vary it do.

## See it: ADC versus diffusion time

The random walks of Chapter 4 provide the measurement directly: for each geometry, the
displacement of every molecule after a time Δ is known, so the signal at a given q is the
average of the encoding phase (Chapter 5), and the ADC follows from it.

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
fig, ax = plt.subplots(figsize=(7, 3.4))
for (name, pos), color in zip(walks.items(), PALETTE):
    adc = []
    for d in deltas_ms:
        k = int(round(d / DT))
        dx = (pos[k] - pos[0])[:, 0] * 1e-3  # mm, along x (across the channel)
        q = np.sqrt(B_TARGET / (d * 1e-3)) / (2 * np.pi)  # 1/mm, from b = (2 pi q)^2 Delta
        e = np.abs(np.mean(np.exp(2j * np.pi * q * dx)))
        adc.append(-np.log(max(e, 1e-6)) / B_TARGET * 1e3)  # um^2/ms
    ax.plot(deltas_ms, adc, "o-", color=color, label=name)
ax.axhline(D_FREE, color="0.5", lw=1, ls="--")
ax.set(xlabel="diffusion time Δ (ms)", ylabel="apparent diffusion coefficient (µm²/ms)", xscale="log", title="ADC measured from the random walks of Chapter 4")
ax.legend(fontsize=8)
fig.tight_layout()
```

Free water gives the same ADC at every diffusion time, within the scatter of the Monte
Carlo estimate. The hindered walk sits at its plateau across the whole range: with obstacles
about a micrometer apart, molecules meet them within a fraction of a millisecond, and the
tortuosity has set in before the earliest diffusion time shown. Time dependence in hindered
tissue therefore comes from structure larger than the spacing of the barriers, such as the
variation of axon packing along a bundle, and is a subtle effect. The two restricted walks
are not subtle. The 2 µm axon has reached its long-time value by 1 ms: every molecule has
crossed it and no further displacement is possible. The 12 µm cell body starts near the
free value and falls over tens of milliseconds as its molecules reach the walls; at the
40 ms of a standard protocol it is close to its long-time value. This is why standard
protocols see time-independent ADCs, and why measuring the approach requires diffusion
times of a few milliseconds or structures of many micrometers.

## PGSE versus OGSE

The pulsed-gradient spin echo cannot reach short diffusion times: the pulse duration itself
is tens of milliseconds at ordinary gradient strength, because the b-value depends on the
duration cubed (Chapter 5). Oscillating-gradient spin echo (OGSE) replaces each pulse by a
gradient waveform that oscillates at a chosen frequency; the effective diffusion time is
then set by the oscillation period rather than by the pulse separation, and frequencies of
50–200 Hz probe diffusion times of a few milliseconds {cite:p}`does2003`. The price is
b-value: an oscillating gradient of the same amplitude and duration produces a much smaller
b, so OGSE acquisitions work at low b-values and need strong gradients to reach a useful
weighting.

## Axon diameter and gradient strength

The long-time limit of the signal across an impermeable cylinder depends only on its
radius: the displacement distribution is the cylinder's cross-section, and the signal is
its Fourier transform (Chapter 4). Measuring the radius therefore means measuring the
signal decay at high q, where cylinders of different radii differ. How high depends on the
radius, and for the axons of the human brain, mostly below 2 µm in diameter, it is beyond
what ordinary gradients reach:

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
ax.set(xlabel="q (1/mm)", ylabel="signal across the cylinder", title="restricted signal versus q, and the q reachable with 10 ms pulses")
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout()
```

At the q that a 40 or 80 mT/m system reaches, cylinders of 0.5, 1, and 2 µm radius are
indistinguishable: their signal has barely decayed. Only the 300 mT/m system reaches a q
where the 1 and 2 µm curves separate, and even there the smallest axons remain out of reach.
This is the reason axon diameter mapping {cite:p}`assaf2008` is a strong-gradient technique,
why its estimates are weighted toward the largest axons in a voxel, and why claims of
diameter measurement on standard hardware should be read with the plot above in mind.

## Exchange

Water crosses membranes. Over diffusion times longer than the exchange time between
compartments (tens to hundreds of milliseconds in gray matter, longer across myelinated
axons), the compartments blend and their signals stop being separable, which is a further
reason compartment models depend on the diffusion time. Exchange is measured by
acquisitions that vary the time between two encodings (filter-exchange imaging); it is
absent from the phantom.

## Why the phantom cannot show this

The phantom's compartments are Gaussian: a stick, a tensor, and balls, with diffusivities
that do not depend on time. Every diffusion-time-dependent effect in this chapter arises
from barriers with a size, which the phantom does not have. A restricted compartment
(implementation plan item T6) would add a cylinder with a radius and a time-dependent
signal; until it exists, the toy random walks are the only simulation in this book that
shows time dependence, and the phantom's ground truth for microstructure should be read as
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
