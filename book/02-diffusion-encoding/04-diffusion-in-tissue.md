---
title: "4. Diffusion in tissue"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** random walks in free, hindered, and restricted geometries, and single-voxel compartment signals ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
:::

## Learning goals

After this chapter you can:

- state how far water molecules move during a diffusion measurement and why that distance
  makes the measurement sensitive to cell-scale structure
- distinguish free, hindered, and restricted diffusion, and say which compartments of brain
  tissue show each
- explain what the diffusion time changes
- describe the displacement distribution that the diffusion MRI signal measures

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt

from dwibook import phantoms, presets
from dwibook.plotting import INK, PALETTE, TISSUE_COLORS, animate, set_style

set_style()
```

## Brownian motion and the Einstein relation

Water molecules at body temperature move continuously and change direction on collision
every fraction of a picosecond. Over the time scales of MRI (milliseconds), the result is a
random walk: each molecule's path is unpredictable, but a large group of them behaves in a
simple, predictable way.

The animation follows 2000 molecules of free water that all start at the same point, with no
barriers, for 50 ms, about the time a typical diffusion measurement spans. Individually they
wander in all directions; together they form a cloud that stays centered where it started
and spreads. The dashed circle marks the typical (root-mean-square) distance from the start.
It grows as the square root of time: the cloud spreads quickly at first and then more
slowly. On the right, the positions along one axis form a bell-shaped (Gaussian)
distribution that widens in step.

```{code-cell} python
:tags: [hide-input]
D_FREE = 3.0        # um^2/ms
STEP = 1.0          # um per step
DT = STEP**2 / (4 * D_FREE)  # ms per step, from <r^2> = 4 D t in two dimensions
N_STEPS = 600       # 50 ms
t_ms = np.arange(N_STEPS + 1) * DT
free = phantoms.random_walk_2d(2000, N_STEPS, STEP, seed=0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.8), gridspec_kw={"width_ratios": [1, 1.2]})
cloud = ax1.scatter(free[0, :, 0], free[0, :, 1], s=2, alpha=0.4, color=PALETTE[0])
paths = [ax1.plot([], [], lw=0.8, color=PALETTE[c])[0] for c in (1, 3, 5)]
ring = plt.Circle((0, 0), 0, fill=False, color=INK["primary"], lw=1, ls="--")
ax1.add_patch(ring)
ax1.set(aspect="equal", xlim=(-40, 40), ylim=(-40, 40), xlabel="µm", ylabel="µm")
ax1.grid(False)
x_axis = np.linspace(-45, 45, 300)
bins = np.linspace(-45, 45, 46)

def frame(i):
    t = t_ms[i]
    cloud.set_offsets(free[i])
    for w, line in enumerate(paths):
        line.set_data(free[: i + 1, w, 0], free[: i + 1, w, 1])
    ring.set_radius(np.sqrt(4 * D_FREE * t))
    ax1.set_title(f"t = {t:4.1f} ms")
    ax2.cla()
    ax2.hist(free[i, :, 0], bins=bins, density=True, color=PALETTE[0], alpha=0.5)
    if t > 0:
        ax2.plot(x_axis, np.exp(-x_axis**2 / (4 * D_FREE * t)) / np.sqrt(4 * np.pi * D_FREE * t), color=INK["primary"], lw=1.2, label="Gaussian, variance 2 D t")
        ax2.legend(loc="upper right")
    ax2.set(xlim=(-45, 45), ylim=(0, 0.12), xlabel="position along one axis (µm)", ylabel="fraction of molecules per µm", title="distribution along one axis")

animate(fig, frame, list(range(0, 60, 3)) + list(range(60, N_STEPS + 1, 15)), fps=8, width=640, dpi=80,
        alt="2000 random walkers start at one point and spread into a widening cloud; a dashed circle of radius square root of 4 D t grows with them, and a histogram of their positions along one axis widens as a Gaussian")
```

The same thing in symbols: the mean squared displacement grows in proportion to time,

$$\langle x^2 \rangle = 2 D t \quad \text{(per axis)},$$

where $x$ is a molecule's displacement along one axis, $\langle \cdot \rangle$ means the
average over all the molecules, and $D$ is the diffusion coefficient. The average
displacement itself, $\langle x \rangle$, is zero, because every direction is equally
likely; it is the spread that grows. The histogram on the right has variance $2Dt$. The
circle in the animation adds the two axes of the plane, so its radius is $\sqrt{4Dt}$.
Free water at 37 °C has $D \approx 3 \times 10^{-3}$ mm²/s, which is 3 µm² per millisecond.
The root-mean-square displacement along one axis, $\sqrt{2Dt}$, is then:

```{code-cell} python
:tags: [hide-input]
D = presets.ADULT_DIFFUSIVITY["CSF"] * 1e3  # mm^2/s -> um^2/ms (1 mm^2 = 1e6 um^2, 1 s = 1e3 ms)
for t in [1, 10, 50, 100]:
    print(f"{t:>4} ms: rms displacement per axis {np.sqrt(2 * D * t):5.1f} um")
```

In 50 ms, a typical molecule of free water has moved about 17 µm along any one axis.

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
  linearly with time, but with a smaller effective coefficient, because every path has to
  detour around the obstacles. The extracellular space between axons and cells is the
  example.
- **Restricted:** the walk is confined inside a closed compartment. Displacement grows
  at first and then stops growing once the molecule has reached the walls. Water inside an
  axon, measured across the axon, is the example.

The simulation below runs the same random walk, with the same $D = 3$ µm²/ms, in four
geometries whose barriers are drawn in gray: an open plane; a lattice of impermeable discs
standing in for packed cells; a channel 2 µm wide, which is an axon seen side-on, cut along
its length (a thick one, so the walls are easy to see); and a disc 10 µm across, a cell body. The animation traces twelve molecules in
each over 50 ms. The free walkers wander off in every direction; the hindered ones get less far,
because every path detours around the obstacles; the walkers in the channel run freely
along it and bounce between its walls; the walkers in the cell body are trapped.

```{code-cell} python
:tags: [hide-input]
from matplotlib.patches import Circle, Rectangle

walks = {
    "free (CSF)": free,
    "hindered (extracellular)": phantoms.random_walk_2d(2000, N_STEPS, STEP, seed=0, radius=1.6, geometry="obstacles", spacing=4.0),
    "restricted, one axis (axon)": phantoms.random_walk_2d(2000, N_STEPS, STEP, seed=0, radius=1.0, geometry="channel"),
    "restricted (cell body)": phantoms.random_walk_2d(2000, N_STEPS, STEP, seed=0, radius=5.0, geometry="disc"),
}
barrier = {"color": "0.8", "zorder": 0}

fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
lines = []
for ax, (name, pos) in zip(axes, walks.items()):
    if name.startswith("hindered"):
        for cx in np.arange(-20, 21, 4.0):
            for cy in np.arange(-20, 21, 4.0):
                ax.add_patch(Circle((cx, cy), 1.6, **barrier))
    elif name.startswith("restricted, one axis"):
        ax.add_patch(Rectangle((-20, -20), 19, 40, **barrier))
        ax.add_patch(Rectangle((1, -20), 19, 40, **barrier))
    elif name.startswith("restricted (cell"):
        ax.add_patch(Rectangle((-20, -20), 40, 40, **barrier))
        ax.add_patch(Circle((0, 0), 5.0, color="white", zorder=0))
    lines.append([ax.plot([], [], lw=0.7, alpha=0.9)[0] for _ in range(12)])
    ax.set(title=name, aspect="equal", xlim=(-20, 20), ylim=(-20, 20), xlabel="µm")
    ax.grid(False)
fig.tight_layout()
stamp = fig.text(0.5, 0.01, "", ha="center", fontsize=9)

def frame(i):
    for ax_lines, pos in zip(lines, walks.values()):
        for w, line in enumerate(ax_lines):
            line.set_data(pos[: i + 1, w, 0], pos[: i + 1, w, 1])
    stamp.set_text(f"t = {t_ms[i]:4.1f} ms")

animate(fig, frame, range(0, N_STEPS + 1, 12), fps=10, width=900, dpi=70,
        alt="twelve random walks traced over 50 ms in four geometries: free walks spread widely, hindered walks spread by detours around a lattice of obstacles, walks in a narrow channel spread only along it, and walks in a small disc stay inside it")
```

## Measure it: displacement versus time

The animation shows the difference by eye; a plot makes it measurable. At every moment,
take each molecule's displacement from its starting point along one axis, square it, and
average over all 2000 molecules: that is $\langle x^2 \rangle$, the mean squared
displacement. The plot below shows how it grows with time in each geometry. The channel
gets two curves, one along its length and one across it; the dashed gray line is $2Dt$ for
free water.

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

The free walk follows the dashed line. The hindered walk is also a straight line, with a
smaller slope: the obstacles slow the spread but never stop it. The cell-body walk bends
over and levels off at a value set by the size of the compartment: once the molecules have
reached the walls, waiting longer does not take them any farther. The axon-like channel does
both at once: along its length the displacement grows steadily, almost as fast as in free
water, while across it the displacement stops growing within a few milliseconds.

The time over which displacements are allowed to build up is called the **diffusion time**,
and the plot shows why it matters. At a short diffusion time every curve is still close to
the free line, because the molecules have not yet reached the walls: a measurement that
short cannot tell restricted water from free water. At a long diffusion time the curves
have separated, and the measurement can.

A scanner does not follow individual molecules; it reports a single diffusion coefficient
per direction, computed as if the diffusion were free. Reading the Einstein relation
backwards, that coefficient is the mean squared displacement divided by $2t$. For free
water it is $D$ at every time. For anything else it changes with the diffusion time, so it
is called the **apparent diffusion coefficient (ADC)**. The plot below shows it for the
same walks, with the 30–50 ms diffusion times of standard clinical scans shaded.

```{code-cell} python
:tags: [hide-input]
curves = {
    "free (CSF)": msd_axis(walks["free (CSF)"], 0),
    "hindered (extracellular)": msd_axis(walks["hindered (extracellular)"], 0),
    "axon, along the axis": msd_axis(walks["restricted, one axis (axon)"], 1),
    "axon, across the axis": msd_axis(walks["restricted, one axis (axon)"], 0),
    "restricted (cell body)": msd_axis(walks["restricted (cell body)"], 0),
}
styles = [(PALETTE[0], "-"), (PALETTE[1], "-"), (PALETTE[2], "-"), (PALETTE[2], ":"), (PALETTE[3], "-")]
fig, ax = plt.subplots(figsize=(8.5, 3.6))
ax.axvspan(30, 50, color="0.92", zorder=0)
ax.text(40, 3.55, "clinical diffusion times", ha="center", va="top", fontsize=8, color=INK["secondary"])
for (name, msd), (color, ls) in zip(curves.items(), styles):
    ax.plot(t_ms[1:], msd[1:] / (2 * t_ms[1:]), color=color, ls=ls, label=name)
ax.set(xlabel="diffusion time (ms)", ylabel="ADC = ⟨x²⟩ / 2t  (µm²/ms)", ylim=(0, 3.6), xlim=(0, t_ms[-1]))
ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
fig.tight_layout()
i40 = np.argmin(np.abs(t_ms - 40))
print("ADC at 40 ms (µm²/ms, the same number in 10^-3 mm²/s):")
for name, msd in curves.items():
    print(f"  {name:<26} {msd[i40] / (2 * t_ms[i40]):6.3f}")
```

The free curve stays flat near 3, as it should (the wobble is the randomness of 2000
molecules). The hindered curve drops within the first few milliseconds and then stays flat
at a lower value (1.24 at 40 ms, less than half the free value), so the obstacles lower
the ADC by about the same factor at any clinical diffusion time. The restricted curves keep falling as the diffusion time grows, because
their displacement has stopped growing while $2t$ has not; at 40 ms, water trapped across
the channel has an ADC hundreds of times smaller than free water. Values measured at
different diffusion times are therefore not interchangeable, even in the same tissue.
[Chapter 21](../05-advanced/21-multi-diffusion-time.md) covers acquisitions that vary the
diffusion time deliberately.

## Compartments in brain tissue

One word first. Diffusion is **isotropic** when molecules spread equally far in every
direction, so the cloud of displacements is round, and **anisotropic** when they spread
farther in some directions than others, so the cloud is stretched into an ellipse. The
channel above is the extreme case: long along its length, almost nothing across.

A white matter voxel, about 2 mm across, holds on the order of a million axons running
roughly in one direction, and its water sits in several environments at once. The sketch
below shows a small piece of one, not to scale, with an ellipse drawn in each environment
whose shape shows how far a typical molecule gets in each direction during a measurement.

```{code-cell} python
:tags: [hide-input]
from matplotlib.patches import Ellipse, FancyBboxPatch

fig, ax = plt.subplots(figsize=(7, 5))
ax.add_patch(Rectangle((0, 0), 100, 56, color="#f3efe6", zorder=0))          # extracellular space
ax.add_patch(Rectangle((72, 0), 28, 56, color="#dcebf7", zorder=1))          # CSF at the voxel edge
for y in [6, 17, 28, 39, 50]:
    ax.add_patch(FancyBboxPatch((2, y - 3.2), 66, 6.4, boxstyle="round,pad=0,rounding_size=3.2", color="0.6", zorder=2))      # myelin
    ax.add_patch(FancyBboxPatch((2.9, y - 2.3), 64.2, 4.6, boxstyle="round,pad=0,rounding_size=2.3", color="#f7cdb4", zorder=3))  # axon interior
for xy, w, h, color in [((35, 28), 30, 3.4, PALETTE[1]), ((35, 11.5), 30, 9, PALETTE[0]), ((86, 28), 22, 22, PALETTE[2])]:
    ax.add_patch(Ellipse(xy, w, h, fill=False, lw=2.2, color=color, zorder=5))
ax.annotate("intra-axonal water:\nrestricted across,\nnearly free along", (24, 29.8), (14, 60), fontsize=8, color=PALETTE[1], ha="center",
            arrowprops=dict(arrowstyle="-", color=PALETTE[1], lw=0.8), annotation_clip=False)
ax.annotate("extracellular water: hindered,\nmore across than along", (35, 7), (35, -12), fontsize=8, color=PALETTE[0], ha="center", va="top",
            arrowprops=dict(arrowstyle="-", color=PALETTE[0], lw=0.8), annotation_clip=False)
ax.text(86, 60, "CSF: free,\nsame in every\ndirection", fontsize=8, color=PALETTE[2], ha="center", va="bottom")
ax.annotate("myelin sheath\n(its water is invisible\nat diffusion echo times)", (56, 53.2), (50, 60), fontsize=8, color=INK["secondary"], ha="center",
            arrowprops=dict(arrowstyle="-", color="0.55", lw=0.8), annotation_clip=False)
ax.annotate("", xy=(68, -26), xytext=(2, -26), arrowprops=dict(arrowstyle="<->", color=INK["secondary"]), annotation_clip=False)
ax.text(35, -30, "fiber direction", fontsize=8, color=INK["secondary"], ha="center", va="top")
ax.set(xlim=(-2, 102), ylim=(-36, 80), aspect="equal")
ax.set_axis_off()
fig.tight_layout()
```

In the sketch, the orange tubes are axons, each wrapped in a gray myelin sheath; the beige
space between them is extracellular; the blue strip at the edge is CSF, which a voxel at
the border of a ventricle or sulcus partly contains. Each ellipse has its long axis along
the direction of easiest movement. The diffusion signal of the voxel is the sum
of the contributions of these environments:

- **Intra-axonal water** is restricted across the axon (diameter about 1 µm, well below
  the displacement scale) and nearly free along it: the thinnest, most stretched ellipse.
  This is the main source of anisotropy in white matter.
- **Extracellular water** is hindered by the packed axons, more so across them than along
  them, so its ellipse is stretched too, but less.
- **CSF** is free water, with the highest diffusion coefficient in the brain: the largest
  ellipse, and a circle.
- **Myelin water**, trapped between the wrapped layers of the myelin sheath, has a T2 of
  roughly 10–20 ms, so its signal has decayed away by the echo times of diffusion scans
  ([Chapter 1](../01-mri-physics/01-spins-and-signal.md)) and contributes almost nothing.

**Gray matter** is not in the sketch: it contains cell bodies, dendrites, and
extracellular space with no dominant orientation, so its diffusion is reduced relative to
free water and close to isotropic.

The simulated brain in this book represents these as TRXScan does: white matter as an
intra-axonal component with diffusion only along the fiber plus an extracellular component
with reduced diffusion across it, gray matter as an isotropic component plus a slowly
diffusing cell-body component, and CSF as free water. The chart shows each diffusion
coefficient of the `adult` preset as a fraction of free water's:

```{code-cell} python
:tags: [hide-input]
d, f = presets.ADULT_DIFFUSIVITY, presets.ADULT_FRACTIONS
bars = [
    (f"WM intra-axonal ({f['WM_intra']:.0%} of WM), along", d["WM_intra"], TISSUE_COLORS["WM"]),
    ("WM intra-axonal, across", 0.0, TISSUE_COLORS["WM"]),
    (f"WM extracellular ({1 - f['WM_intra']:.0%}), along", d["WM_extra"][0], TISSUE_COLORS["WM"]),
    ("WM extracellular, across", d["WM_extra"][1], TISSUE_COLORS["WM"]),
    (f"GM main component ({1 - f['GM_restricted']:.0%} of GM)", d["GM"], TISSUE_COLORS["GM"]),
    (f"GM cell bodies ({f['GM_restricted']:.0%})", f["d_soma"], TISSUE_COLORS["GM"]),
    ("CSF", d["CSF"], TISSUE_COLORS["CSF"]),
]
fig, ax = plt.subplots(figsize=(7.5, 3.4))
y = np.arange(len(bars))[::-1]
ax.barh(y, [v / d["CSF"] for _, v, _ in bars], color=[c for *_, c in bars], height=0.6)
for yi, (_, v, _) in zip(y, bars):
    ax.text(v / d["CSF"] + 0.015, yi, f"{v / d['CSF']:.0%}  ({v * 1e3:.2f} × 10⁻³ mm²/s)", va="center", fontsize=8)
ax.set(yticks=y, yticklabels=[n for n, *_ in bars], xlim=(0, 1.45), xlabel="diffusion coefficient as a fraction of free water")
ax.grid(axis="y", visible=False)
fig.tight_layout()
```

Water inside axons moves along them at a bit over half the speed of free water, and not at
all across them. The extracellular water between axons moves across them at a fifth of the
free rate. Gray matter's main component sits in between, and its cell-body component is
the slowest pool in the model.

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

On the left, each dot is one molecule's net movement over the 50 ms; together they form a
tall, thin streak, the anisotropic cloud in its purest form. On the right, the same
movements are counted along each axis. Along the channel they form a broad bell curve
tens of micrometers wide. Across it they form a narrow peak that can never be wider than
the channel itself (a molecule cannot get more than 2 µm from where it started), however
long the measurement runs.

## The displacement distribution and the signal

The histogram above is a displacement distribution, also called the diffusion propagator:
the probability that a molecule has moved by a given amount in a given time. For free
diffusion it is a Gaussian whose width grows with time; for restricted diffusion it is
narrower and, at long times, bounded by the compartment size.

The diffusion MRI signal is a measurement of this distribution. [Chapter 5](./05-diffusion-encoding.md)
shows how: the encoding gradients give each molecule a phase that depends on how far it
moved, and the signal is what survives when all those phases are added up. The figure
below previews the result using the two distributions above, the wide one along the
channel and the narrow one across it. On the right, the signal is computed from them for
increasing gradient strength. Where molecules moved far (the wide distribution), even a
weak gradient scatters their phases and the signal falls quickly. Where they barely moved
(the narrow one), the gradient must be much stronger before the signal drops at all. (The
small ripples near zero come from using only 2000 molecules.)

```{code-cell} python
:tags: [hide-input]
q = np.linspace(0, 0.3, 121)  # cycles per um: gradient strength x pulse duration, in the units of the displacements
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.2))
for x, color, name in [(disp[:, 1], PALETTE[0], "along the channel (wide)"), (disp[:, 0], PALETTE[1], "across the channel (narrow)")]:
    counts, edges = np.histogram(x, bins=np.linspace(-45, 45, 91))
    ax1.stairs(counts / counts.max(), edges, fill=True, color=color, alpha=0.55, label=name)
    ax2.plot(q, np.abs(np.exp(2j * np.pi * np.outer(q, x)).mean(axis=1)), color=color, lw=2, label=name)
ax1.set(xlabel="displacement (µm)", ylabel="molecules (each scaled to its peak)", yticks=[], ylim=(0, 1.35), title="displacement distribution")
ax1.legend(fontsize=8)
ax2.set(xlabel="gradient strength →", ylabel="signal (fraction of no-gradient)", xticks=[], ylim=(0, 1.05), title="signal it produces")
fig.tight_layout()
```

In the language of mathematics, the signal is the Fourier transform of the displacement
distribution (exactly so only when the gradient pulses are brief compared with the time
between them, a condition [Chapter 5](./05-diffusion-encoding.md) returns to), the same operation that links an image to its k-space in
[Chapter 2](../01-mri-physics/02-spatial-encoding-kspace.md), and the gradient strength
picks which spatial frequency is read. Sampling the signal at one gradient strength
measures a single number, the apparent diffusion coefficient in that direction; sampling
at many strengths and directions measures the distribution itself. That is the choice
between the sampling schemes of [Chapter 6](./06-qspace-sampling.md). Chapter 5 also
combines gradient strength, pulse duration, and timing into the single number that
scanners report, the b-value.

## What this implies for acquisition

- **The diffusion time is a parameter of the measurement**, set by the gradient timing, and
  values measured at different diffusion times are not directly comparable in restricted or
  hindered tissue.
- **Measurements are sensitive to structures smaller than the voxel** because displacements
  are on the micrometer scale. They do not resolve those structures; they report averages
  over the voxel.
- **Anisotropy comes from oriented barriers**, chiefly axons. A voxel with fibers in more
  than one orientation, which is most of white matter, has a displacement cloud more
  complex than the single stretched ellipse that one fiber direction gives, which is the
  reason for the models in Part IV.
- **CSF contamination** raises the apparent diffusion of any voxel it touches, because free
  water has the largest displacements of all.

## Further reading

Einstein's derivation of the mean squared displacement {cite:p}`einstein1905`, the first
diffusion images of the brain {cite:p}`lebihan1986`, and the review of what diffusion MRI
can and cannot measure about microstructure {cite:p}`alexander2019`.
