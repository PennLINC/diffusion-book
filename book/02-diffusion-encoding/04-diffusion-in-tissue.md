---
title: Diffusion in tissue
subtitle: Chapter 4
kernelspec:
  name: python3
  display_name: Python 3
---

## Learning goals

After this chapter you can:

- state how far water molecules move during a diffusion measurement and why that distance
  makes the measurement sensitive to cell-scale structure
- distinguish free, hindered, and restricted diffusion, and say which compartments of brain
  tissue show each
- explain what the diffusion time changes
- describe the displacement distribution that the diffusion MRI signal measures

**Datasets used:** none (toy tier)
**Simulation tier:** toy

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt

from dwibook import phantoms, presets
from dwibook.plotting import PALETTE, TISSUE_COLORS, set_style

set_style()
```

## Brownian motion and the Einstein relation

Water molecules at body temperature move continuously and change direction on collision
every fraction of a picosecond. Over the time scales of MRI (milliseconds), the result is a
random walk. Its statistics are simple: the mean displacement is zero, and the mean squared
displacement grows in proportion to time,

$$\langle x^2 \rangle = 2 D t \quad \text{(per axis)},$$

where $D$ is the diffusion coefficient. Free water at 37 °C has $D \approx 3 \times 10^{-3}$
mm²/s, which is 3 µm² per millisecond. In the 50 ms that a typical diffusion measurement
spans, the root-mean-square displacement along one axis is about 17 µm:

```{code-cell} python
:tags: [hide-input]
D = presets.ADULT_DIFFUSIVITY["CSF"] * 1e3  # mm^2/s -> um^2/ms (1 mm^2 = 1e6 um^2, 1 s = 1e3 ms)
for t in [1, 10, 50, 100]:
    print(f"{t:>4} ms: rms displacement per axis {np.sqrt(2 * D * t):5.1f} um")
```

Cell bodies are 10–20 µm across, axons 1 µm or less. Over the measurement time, a molecule
moves far enough to encounter membranes, myelin, and organelles, and its displacement is
reduced by them. This is the basis of diffusion MRI: the measurement reports displacements
on the scale of micrometers, from voxels that are millimeters across, and the reduction in
displacement relative to free water reflects the microstructure of the voxel.

## Free, hindered, and restricted diffusion

Three cases cover what tissue does to the random walk:

- **Free:** no barriers. The mean squared displacement grows linearly forever. CSF in the
  ventricles is the example in the brain.
- **Hindered:** barriers slow the walk but do not enclose it. Displacement still grows
  linearly with time, but with a smaller effective coefficient; the ratio to the free value
  is called the tortuosity factor. The extracellular space between axons and cells is the
  example.
- **Restricted:** the walk is confined inside a closed compartment. Displacement grows
  at first and then stops growing once the molecule has reached the walls. Water inside an
  axon, measured across the axon, is the example.

The simulation below runs the same random walk in four geometries. The step size and time
step are chosen so that the free walk has $D = 3$ µm²/ms.

```{code-cell} python
:tags: [hide-input]
D_FREE = 3.0        # um^2/ms
STEP = 1.0          # um per step
DT = STEP**2 / (4 * D_FREE)  # ms per step, from <r^2> = 4 D t in two dimensions
N_STEPS = 600       # 50 ms
walks = {
    "free (CSF)": phantoms.random_walk_2d(2000, N_STEPS, STEP, seed=0),
    "hindered (extracellular)": phantoms.random_walk_2d(2000, N_STEPS, STEP, seed=0, radius=1.6, geometry="obstacles", spacing=4.0),
    "restricted, one axis (axon)": phantoms.random_walk_2d(2000, N_STEPS, STEP, seed=0, radius=1.0, geometry="channel"),
    "restricted (cell body)": phantoms.random_walk_2d(2000, N_STEPS, STEP, seed=0, radius=5.0, geometry="disc"),
}
t_ms = np.arange(N_STEPS + 1) * DT

fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
for ax, (name, pos) in zip(axes, walks.items()):
    for w in range(12):
        ax.plot(pos[:, w, 0], pos[:, w, 1], lw=0.6, alpha=0.8)
    ax.set(title=name, aspect="equal", xlim=(-20, 20), ylim=(-20, 20), xlabel="µm")
    ax.grid(False)
fig.tight_layout()
```

## Measure it: displacement versus time

```{code-cell} python
:tags: [hide-input]
def msd_axis(pos, axis):
    return np.mean((pos[:, :, axis] - pos[0, :, axis]) ** 2, axis=1)

fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.plot(t_ms, msd_axis(walks["free (CSF)"], 0), color=PALETTE[0], label="free (CSF)")
ax.plot(t_ms, msd_axis(walks["hindered (extracellular)"], 0), color=PALETTE[1], label="hindered (extracellular)")
ax.plot(t_ms, msd_axis(walks["restricted, one axis (axon)"], 1), color=PALETTE[2], label="axon, along the axis")
ax.plot(t_ms, msd_axis(walks["restricted, one axis (axon)"], 0), color=PALETTE[2], ls=":", label="axon, across the axis")
ax.plot(t_ms, msd_axis(walks["restricted (cell body)"], 0), color=PALETTE[3], label="restricted (cell body)")
ax.plot(t_ms, 2 * D_FREE * t_ms, color="0.5", lw=1, ls="--", label="2 D t (free water)")
ax.set(xlabel="diffusion time (ms)", ylabel="mean squared displacement along one axis (µm²)")
ax.legend(fontsize=8)
fig.tight_layout()
```

The free walk follows the straight line. The hindered walk is also a straight line, with a
smaller slope: the obstacles reduce the effective diffusion coefficient but do not bound the
displacement. The cell-body walk bends over and levels off at a value set by the size of the
compartment. The axon-like channel does both at once: along its axis the displacement grows
freely, across it the displacement stops growing within a few milliseconds. This dependence on time is the key property: a measurement made at a short
diffusion time cannot distinguish restricted from free water, because the molecules have not
yet reached the walls; a measurement at a long diffusion time can. In standard clinical
diffusion MRI the diffusion time is fixed by the sequence, typically 30–50 ms, and the
measured diffusion coefficient is an apparent one (ADC) that depends on that time.
Chapter 22 covers acquisitions that vary it deliberately.

## Compartments in brain tissue

A white matter voxel contains water in several environments, and the diffusion signal is
the sum of their contributions:

- **Intra-axonal water** is restricted across the axon (diameter about 1 µm, well below
  the displacement scale) and nearly free along it. This is the source of anisotropy in
  white matter.
- **Extracellular water** is hindered by the packed axons, more so across them than along
  them, so it is also anisotropic.
- **Gray matter** contains cell bodies, dendrites, and extracellular space with no
  dominant orientation; its diffusion is reduced relative to free water and close to
  isotropic.
- **CSF** is free water, with the highest diffusion coefficient in the brain.
- **Myelin water** has a very short T2 and contributes little at diffusion echo times.

The phantom in this book represents these as TRXScan does: white matter as an intra-axonal
component with diffusion only along the fiber plus an extracellular component with reduced
diffusion across it, gray matter as an isotropic component plus a slowly diffusing cell-body
component, and CSF as free water. The numbers are those of the `adult` preset:

```{code-cell} python
:tags: [hide-input]
d = presets.ADULT_DIFFUSIVITY
print(f"WM intra-axonal, along the fiber : {d['WM_intra'] * 1e3:.2f} x 10^-3 mm^2/s (zero across)")
print(f"WM extracellular, along / across : {d['WM_extra'][0] * 1e3:.2f} / {d['WM_extra'][1] * 1e3:.2f}")
print(f"GM                               : {d['GM'] * 1e3:.2f}")
print(f"CSF                              : {d['CSF'] * 1e3:.2f}")
```

The restricted-channel walk above shows what anisotropy looks like in terms of displacement:
molecules travel freely along the channel and hardly at all across it.

```{code-cell} python
:tags: [hide-input]
pos = walks["restricted, one axis (axon)"]
disp = pos[-1] - pos[0]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.2))
ax1.scatter(disp[:, 0], disp[:, 1], s=3, alpha=0.4)
ax1.set(title="displacements after 50 ms", xlabel="across the channel (µm)", ylabel="along the channel (µm)", xlim=(-25, 25), ylim=(-25, 25), aspect="equal")
ax2.hist(disp[:, 1], bins=40, density=True, alpha=0.5, label="along")
ax2.hist(disp[:, 0], bins=40, density=True, alpha=0.5, label="across")
ax2.set(title="displacement distributions", xlabel="displacement (µm)")
ax2.legend()
fig.tight_layout()
```

## The displacement distribution and the signal

The histogram above is a displacement distribution, also called the diffusion propagator:
the probability that a molecule has moved by a given amount in a given time. For free
diffusion it is a Gaussian whose width grows with time; for restricted diffusion it is
narrower and, at long times, bounded by the compartment size.

The diffusion MRI signal is a measurement of this distribution. As Chapter 5 shows, the
encoding gradients make the signal equal to the Fourier transform of the displacement
distribution, evaluated at a spatial frequency set by the gradient strength and duration.
Sampling the signal at one strength (one b-value) measures a single number, the apparent
diffusion coefficient in that direction; sampling at many strengths and directions measures
the distribution itself. That is the choice between the sampling schemes of Chapter 6.

## What this implies for acquisition

- **The diffusion time is a parameter of the measurement**, set by the gradient timing, and
  values measured at different diffusion times are not directly comparable in restricted or
  hindered tissue.
- **Measurements are sensitive to structures smaller than the voxel** because displacements
  are on the micrometer scale. They do not resolve those structures; they report averages
  over the voxel.
- **Anisotropy comes from oriented barriers**, chiefly axons. A voxel with fibers in more
  than one orientation, which is most of white matter, has a more complex displacement
  distribution than a single ellipsoid, which is the reason for the models in Part IV.
- **CSF contamination** raises the apparent diffusion of any voxel it touches, because free
  water has the largest displacements of all.

## Further reading

Einstein's derivation of the mean squared displacement {cite:p}`einstein1905`, the first
diffusion images of the brain {cite:p}`lebihan1986`, and the review of what diffusion MRI
can and cannot measure about microstructure {cite:p}`alexander2019`.
