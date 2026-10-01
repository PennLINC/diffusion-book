---
title: "6. q-space sampling schemes"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** single-voxel signals under each sampling scheme, generated in the page ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
- **`ref-clean`** (pending): the artifact-free, noise-free reference series with its truth maps and true fiber orientations ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-ref-clean)).
- **`ref-schemes`** (pending): the simulated brain under the 30-direction, 64-direction, HBCD, and DSI schemes, compared at equal total scan time (the per-volume noise is scaled with the number of volumes), plus a CS-DSI subset of the DSI run, which takes about a quarter of its time ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-ref-schemes)).

Pipeline-tier datasets are simulated offline by TRXScan ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)) and are marked *pending* until their release; the figures that need them say so where they will appear.
:::

## Learning goals

After this chapter you can:

- describe the four families of sampling scheme (single-shell, multi-shell, DSI, CS-DSI)
  and generate each
- state what each scheme makes possible and what it cannot support
- explain why the number and distribution of directions matter, with a measured example
- estimate the scan time of a scheme
- use Table 6.1 to match a scheme to an analysis

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel

from dwibook import presets, schemes, signal
from dwibook.plotting import INK, PALETTE, TISSUE_COLORS, set_style

set_style()
```

## What is being sampled

Each diffusion-weighted volume applies one gradient direction at one b-value. The set of
all (direction, b-value) pairs in an acquisition is its sampling scheme. It is convenient to
picture each measurement as a point in a three-dimensional space whose direction is the
gradient direction and whose distance from the origin is $q$, the tightness of the phase
winding set by the gradient strength and pulse duration, $q = \gamma G \delta / 2\pi$
([Chapter 5](./05-diffusion-encoding.md)). This is q-space. The
b-value grows with the square of the winding, so with the pulse timing held fixed the
distance from the origin grows with $\sqrt{b}$, not with $b$: the b = 4000 shell lies only
twice as far out as the b = 1000 shell. [Chapter 4](./04-diffusion-in-tissue.md) noted that the signal is the Fourier
transform of the displacement distribution evaluated at one such spatial frequency (exactly
so when the pulses are brief compared with their separation, [Chapter 5](./05-diffusion-encoding.md)), so the
diffusion signal is a function defined on q-space: every point has a value, and a scheme
is a choice of where to read it. Every model in Part IV needs a particular kind of
coverage.

The figure builds this up. On the left, a set of gradient directions, each drawn as a point
on a sphere and its opposite (the signal at $-q$ equals the signal at $q$, so every
measurement counts for both). In the middle, the same idea at two b-values: each b-value
is a sphere, a *shell*, of radius $\sqrt{b}$. On the right, the outer shell's points are
colored by the signal of one white matter voxel with a single fiber running vertically
(the simulated brain's model, [Chapter 5](./05-diffusion-encoding.md)). Look at where the bright points are: around the
equator, where the gradient runs across the fiber and the water inside the axons cannot
move. The points near the poles, along the fiber, are dark.

```{code-cell} python
:tags: [hide-input]
fiber_z = np.array([0.0, 0.0, 1.0])
_, dirs = schemes.single_shell(1000, 60, n_b0=0)
both = np.concatenate([dirs, -dirs])               # each measurement and its antipode
shells_demo = {1000: PALETTE[0], 3000: PALETTE[1]}
lim = np.sqrt(3000) * 1.05

fig = plt.figure(figsize=(11, 4.3))
ax = fig.add_subplot(1, 3, 1, projection="3d")
ax.scatter(*both.T, s=12, color=INK["secondary"], depthshade=False)
ax.set_title("60 gradient directions\nand their opposites")
ax.set(xlim=(-1.05, 1.05), ylim=(-1.05, 1.05), zlim=(-1.05, 1.05))

ax = fig.add_subplot(1, 3, 2, projection="3d")
for b, col in shells_demo.items():
    ax.scatter(*(np.sqrt(b) * both).T, s=12, color=col, depthshade=False, label=f"b = {b}: radius √b = {np.sqrt(b):.0f}")
ax.legend(loc="upper left", fontsize=7)
ax.set_title("the same directions on two shells\nof radius √b")
ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim))

ax = fig.add_subplot(1, 3, 3, projection="3d")
s_shell = signal.white_matter(3000, both @ fiber_z)
sc = ax.scatter(*(np.sqrt(3000) * both).T, s=22, c=s_shell, cmap="magma", vmin=0, vmax=s_shell.max(), depthshade=False)
ax.plot([0, 0], [0, 0], [-lim, lim], color=INK["secondary"], lw=1.5, ls="--")
ax.set_title("b = 3000 shell colored by the signal\nof one fiber (dashed line)")
ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim))
fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.02, label="S / S₀")
for ax in fig.axes[:3]:
    ax.set_box_aspect((1, 1, 1)); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.view_init(elev=12)
fig.tight_layout()
print(f"b = 3000 shell: S/S0 = {s_shell.min():.3f} for the direction closest to the fiber, "
      f"{s_shell.max():.3f} for the direction closest to perpendicular")
```

Four families cover current practice. The scheme generators used below are in
`dwibook.schemes`; the directions of single-shell and multi-shell schemes are spread over
the sphere by simulated electrostatic repulsion, the standard method {cite:p}`jones1999`.

## Single-shell

All diffusion-weighted volumes share one b-value. This is the scheme of clinical diffusion
tensor imaging (DTI), typically b = 1000 s/mm² with 6 to 64 directions plus b=0 volumes, and
of high angular resolution diffusion imaging (HARDI), typically b = 2000–3000 with 60 or more
directions.

The model these schemes were designed for is the *diffusion tensor*, the simplest
description of direction-dependent diffusion. It pictures the diffusivity in every
direction as an ellipsoid, long along the direction in which water moves most easily and
short across it ([Chapter 15](../04-modeling/15-signal-representations.md) fits it). An ellipsoid in three dimensions takes six numbers to
pin down: three for the lengths of its axes and three for the angles that orient it. Each
diffusion-weighted direction, divided by the b=0 image, gives one equation in those six
unknowns. That count sets the first rule below.

- **Six directions** is the mathematical minimum for a tensor fit: six equations for six
  unknowns. Some quick clinical protocols still use it, but research protocols avoid it:
  every measurement then influences the result with no redundancy, so noise in any one
  volume passes straight into the tensor, and a single corrupted volume cannot be dropped.
- **Around 30 directions** gives a tensor fit whose precision no longer depends on how the
  fibers are oriented relative to the directions {cite:p}`jones2004`.
- **60–90 directions at b ≥ 2000** resolves crossing fibers with the orientation models of
  [Chapter 16](../04-modeling/16-fiber-orientation.md). Most white matter voxels contain more than one fiber orientation
  {cite:p}`jeurissen2013`, which is the reason HARDI exists.

A single shell provides the *angular profile* of the signal at one b-value: how the signal
changes as the gradient direction turns, the pattern of bright and dark points in the
right-hand panel above. Why does HARDI use a higher b-value than DTI? The figure below
takes one plane through q-space and plots the signal against the gradient's angle in that
plane, for a single fiber and for two equal fibers crossing at 90° (their axes are the
dashed lines). Each curve is divided by its own maximum, so only its shape is compared, not
how much signal is left.

```{code-cell} python
:tags: [hide-input]
phi = np.linspace(0, 2 * np.pi, 721)          # gradient angle in the plane of the fibers
profiles = {
    "one fiber": lambda b: signal.white_matter(b, np.cos(phi)),
    "two fibers crossing at 90°": lambda b: 0.5 * signal.white_matter(b, np.cos(phi)) + 0.5 * signal.white_matter(b, np.sin(phi)),
}
b_styles = {1000: dict(color=PALETTE[0], ls="-"), 3000: dict(color=PALETTE[1], ls="-")}
fig, axes = plt.subplots(1, 2, figsize=(8, 4.2), subplot_kw=dict(projection="polar"))
for ax, (name, prof) in zip(axes, profiles.items()):
    axes_deg = [0] if name == "one fiber" else [0, 90]
    for a in axes_deg:
        ax.plot([np.deg2rad(a), np.deg2rad(a + 180)], [1.08, 1.08], color=INK["secondary"], lw=1, ls="--")
    for b, sty in b_styles.items():
        s = prof(b)
        ax.plot(phi, s / s.max(), lw=2, label=f"b = {b}", **sty)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0.5, 1.0]); ax.set_yticklabels(["0.5", "1"], fontsize=7)
    ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_title(name, pad=14)
axes[1].legend(loc="upper left", bbox_to_anchor=(0.95, 1.12), fontsize=8)
fig.tight_layout()

print("angular contrast = largest / smallest signal over all gradient angles")
for name, prof in profiles.items():
    for b in b_styles:
        s = prof(b)
        print(f"  {name:>27}, b = {b}: {s.max() / s.min():6.2f}   (smallest point {s.min() / s.max():.2f} of the largest)")
```

Look first at the single fiber. The signal is largest when the gradient runs across the
fiber and smallest along it, so the profile is a figure-eight lying across the fiber. Now
look at the crossing. Each gradient direction that runs along one fiber runs across the
other, so the crossing profile has four lobes, pointing along the two fiber axes, with dips
between them at 45°. At b = 1000 those dips are shallow: the smallest point is 0.77 of the
largest, the angular contrast is only 1.30, and the profile is nearly round. The two
fibers blur into one another. At b = 3000 the dips fall to 0.18 of the largest, the
contrast is 5.51, and the four lobes are distinct. The orientation models of
[Chapter 16](../04-modeling/16-fiber-orientation.md) work by separating exactly these lobes, which is why HARDI uses b ≥ 2000. The price is signal: at b = 3000 the
crossing retains less than a third of $S_0$ at its brightest ([Chapter 5](./05-diffusion-encoding.md)).

A single shell does not provide the shape of the decay with b, the next section's subject,
so any model that needs that shape (kurtosis, multi-compartment models) cannot be fit
reliably from it.

## Multi-shell

Several b-values, each with its own direction set. The HBCD scheme of the reference
protocol is one example (b = 500, 1000, 2000, 3000) {cite:p}`dean2024`; the Human Connectome Project's
1000/2000/3000 with 90 directions each is another {cite:p}`sotiropoulos2013`. The shells sample the decay with b as
well as the angular profile.

Why does the decay need more than one shell? [Chapter 5](./05-diffusion-encoding.md) showed that tissue signal plotted
against b on a log scale is not a straight line: it bends, because tissue is a mixture of
water pools that decay at different rates. A single shell gives one point on that curve
for each direction, besides b = 0, and a straight line through the b = 0 point (a single
exponential) fits any one point exactly. The bend is invisible. Several shells put several
points on the curve, and the bend shows. In the figure, the dashed line is what one shell at
b = 1000 must assume; the filled points are what the four HBCD shells measure.

```{code-cell} python
:tags: [hide-input]
b_fine = np.linspace(0, 3200, 200)
hbcd_shells = np.array([0, 500, 1000, 2000, 3000])
curves = {
    "white matter, gradient across the fiber": (lambda b: signal.white_matter(b, 0.0), TISSUE_COLORS["WM"]),
    "gray matter": (signal.gray_matter, TISSUE_COLORS["GM"]),
}
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, (name, (f, col)) in zip(axes, curves.items()):
    adc = -np.log(f(1000.0)) / 1000.0                  # all that one shell at b = 1000 can give
    ax.semilogy(b_fine, np.exp(-b_fine * adc), color=INK["secondary"], ls="--", lw=1.5,
                label="straight line through the b = 1000 point")
    ax.semilogy(hbcd_shells, f(hbcd_shells.astype(float)), "o-", color=col, lw=1, ms=7,
                label="four shells (HBCD) and b = 0")
    ax.semilogy([1000], [f(1000.0)], "o", mfc="white", mec=INK["primary"], mew=1.5, ms=11,
                label="one shell (b = 1000)", zorder=0)
    ax.set(xlabel="b (s/mm²)", title=name, ylim=(0.08, 1.1))
    ax.set_yticks([0.1, 0.2, 0.5, 1.0], labels=["0.1", "0.2", "0.5", "1"]); ax.minorticks_off()
    ax.legend(fontsize=7, loc="lower left")
    print(f"{name}: at b = 3000 the straight line predicts S/S0 = {np.exp(-3000 * adc):.3f}; the tissue gives {f(3000.0):.3f}")
axes[0].set_ylabel("S / S₀ (log scale)")
fig.tight_layout()
```

In both tissues the measured points bend away from the dashed line and stay above it:
at b = 3000 the white matter signal across the fiber is 0.624 where the straight line
predicts 0.506, and gray matter is 0.144 where the line predicts 0.118. Only a scheme with
more than one non-zero shell can see that difference. The models that describe how the
decay bends, and what the bend says about the water pools, are the subject of Chapters
[15](../04-modeling/15-signal-representations.md) to [17](../04-modeling/17-microstructure-models.md); the ones Table 6.1 refers to are diffusion kurtosis, NODDI, multi-tissue
spherical deconvolution, and MAP-MRI, and all of them need multiple shells.

Two design details matter:

- **Directions should be spread across shells as well as within them**, so that the
  combined set covers the sphere uniformly {cite:p}`caruyer2013`.
- **Shells should be interleaved in acquisition order** and b=0 volumes spread throughout,
  so that motion or scanner drift affects all shells equally ([Chapter 5](./05-diffusion-encoding.md)). The acquired HBCD
  protocol (76 volumes, the one the live tier simulates) does this: six b=0 volumes, then
  all four shells interleaved. The bundled 75-volume copy this book uses for its figures is
  sorted by b instead, in blocks of increasing b with a b=0 about every eight volumes,
  because that makes the shells easy to see; acquired in that order, motion late in the
  scan would fall mostly on the b = 3000 shell.

## DSI

Diffusion spectrum imaging samples q-space on a Cartesian grid, typically all lattice
points within a sphere: 257 points for a radius of 4 grid units, 515 for a radius of 5
{cite:p}`wedeen2005`. The idea is the one behind image reconstruction in [Chapter 2](../01-mri-physics/02-spatial-encoding-kspace.md). There,
each k-space sample is one stripe-pattern component of the image, and a full grid of
samples is turned back into the image by an inverse Fourier transform. Here, each q-space
sample is one component of the *displacement distribution* (also called the propagator):
the probability that a water molecule moved a given distance in a given direction during
the diffusion time. A full grid of q-space samples is turned back into that distribution
the same way, with no model of the tissue. The grid extends to high b-values (often
4000–8000 s/mm²) and needs a strong gradient system. The cost is the number of volumes:
257 volumes at a TR of 4 s is 17 minutes.

The figure shows the idea in two dimensions, for one plane through q-space and a single
fiber running at 30° (dashed line). On the left, the grid of samples, each colored by the
signal of the fiber voxel: bright across the fiber, dark along it. On the right, the
inverse Fourier transform of that grid. Look at the direction of the elongation: the
signal pattern is stretched *across* the fiber, and the displacement distribution is
stretched *along* it, because water moves farther along the axons than across them.

```{code-cell} python
:tags: [hide-input]
R_GRID, B_EDGE, ANGLE = 5, 5000.0, 30.0      # grid radius (points), b at the grid edge, fiber angle
k = np.arange(-R_GRID, R_GRID + 1)
qy, qx = np.meshgrid(k, k, indexing="ij")
r = np.hypot(qx, qy)
inside = r <= R_GRID
b_grid = B_EDGE * (r / R_GRID) ** 2                    # b grows with the square of the distance
ux = np.divide(qx, r, out=np.zeros_like(r), where=r > 0)
uy = np.divide(qy, r, out=np.zeros_like(r), where=r > 0)
fib = np.array([np.cos(np.deg2rad(ANGLE)), np.sin(np.deg2rad(ANGLE))])
s_grid = signal.white_matter(b_grid, ux * fib[0] + uy * fib[1])

n_pad, c = 64, 32                                      # zero-pad and taper the edge before the transform
taper = np.where(inside, 0.5 * (1 + np.cos(np.pi * r / (R_GRID + 1))), 0)
padded = np.zeros((n_pad, n_pad))
padded[c - R_GRID:c + R_GRID + 1, c - R_GRID:c + R_GRID + 1] = s_grid * taper
prop = np.clip(np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(padded)))), 0, None)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.8))
sc = ax1.scatter(qx[inside], qy[inside], c=s_grid[inside], s=70, cmap="magma", vmin=0, vmax=1, edgecolors=INK["grid"])
ax1.plot([-R_GRID * fib[0], R_GRID * fib[0]], [-R_GRID * fib[1], R_GRID * fib[1]], color=INK["secondary"], ls="--", lw=1)
ax1.set(aspect="equal", xlabel="$q_x$ (grid steps)", ylabel="$q_y$ (grid steps)",
        title=f"q-space grid, b = 0 to {B_EDGE:.0f}\ncolored by signal")
fig.colorbar(sc, ax=ax1, shrink=0.8, label="S / S₀")
half = 12
ax2.imshow(prop[c - half:c + half + 1, c - half:c + half + 1], cmap="magma", extent=(-half, half, half, -half), origin="upper")
ax2.plot([-half * fib[0], half * fib[0]], [-half * fib[1], half * fib[1]], color="white", ls="--", lw=1)
ax2.set(xlabel="displacement x (relative units)", ylabel="displacement y", title="inverse Fourier transform:\ndisplacement distribution")
ax2.invert_yaxis(); ax2.grid(False)
fig.tight_layout()
```

## CS-DSI

Compressed-sensing DSI acquires a random subset of the grid points, typically a quarter to
a third of them, and fills in the rest during reconstruction {cite:p}`menzel2011`. The
fill-in rests on an assumption, called a *sparsity prior*: the displacement distribution
is smooth and simple enough that a few numbers in a suitable basis describe it, the way a
few frequencies describe a chord. The reconstruction therefore looks for the simplest
distribution that agrees with the samples that were measured. This works only under three
conditions: the subset must be irregular (random gaps leave noise-like errors that the
search can remove, where regular gaps leave copies that it cannot tell from real
structure), the distribution must really be compressible in the chosen basis, and the
reconstruction must be run as an iterative search rather than a single Fourier transform.
The result is DSI-like information in a multi-shell-like scan time.

## See it: the schemes

The four panels below draw four schemes in q-space at a common scale, each point at radius
$\sqrt{b}$ and colored by its b-value. Look at the geometry: a single-shell scheme is one
sphere of points (larger for the higher b-value of HARDI), a multi-shell scheme is several
concentric spheres, and DSI fills a grid inside a ball rather than lying on spheres at all.

```{code-cell} python
:tags: [hide-input]
b_hbcd, v_hbcd = schemes.hbcd()
b_dsi, v_dsi = schemes.dsi_grid(radius=4, b_max=4000)
cs_idx = schemes.cs_subset(b_dsi, v_dsi, 64, seed=1)
examples = {
    "single-shell, 6 dirs, b = 1000": schemes.single_shell(1000, 6, n_b0=1),
    "single-shell, 30 dirs, b = 1000 (DTI)": schemes.single_shell(1000, 30, n_b0=3),
    "single-shell, 64 dirs, b = 2000 (HARDI)": schemes.single_shell(2000, 64, n_b0=4),
    "multi-shell (HBCD, 75 volumes)": (b_hbcd, v_hbcd),
    "DSI grid (257 points)": (b_dsi, v_dsi),
    "CS-DSI (64 of 257 points)": (b_dsi[cs_idx], v_dsi[cs_idx]),
}
b_max = max(b.max() for b, _ in examples.values())  # shared limits and color scale
shown = ["single-shell, 30 dirs, b = 1000 (DTI)", "single-shell, 64 dirs, b = 2000 (HARDI)",
         "multi-shell (HBCD, 75 volumes)", "DSI grid (257 points)"]
fig = plt.figure(figsize=(9, 8.5))
for i, name in enumerate(shown):
    ax = fig.add_subplot(2, 2, i + 1, projection="3d")
    schemes.plot_scheme(*examples[name], ax=ax, title=name, b_max=b_max, colorbar=False)
fig.tight_layout(rect=(0, 0, 0.9, 1))
sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, b_max))
fig.colorbar(sm, cax=fig.add_axes((0.91, 0.25, 0.015, 0.5)), label="b (s/mm²)")
```

The six-direction scheme is the 30-direction sphere with fewer points, and CS-DSI is a
random subset of the DSI grid; a slice through the grid shows its Cartesian structure and
what the CS subset keeps:

```{code-cell} python
:tags: [hide-input]
q_dsi = np.sqrt(b_dsi)[:, None] * v_dsi
plane = np.abs(q_dsi[:, 2]) < 1e-6
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5), sharex=True, sharey=True)
ax1.scatter(q_dsi[plane, 0], q_dsi[plane, 1], s=14)
ax1.set(title="DSI grid, plane through the origin", aspect="equal",
        xlabel="$q_x$, drawn as $\\sqrt{b}$ (√(s/mm²))", ylabel="$q_y$, drawn as $\\sqrt{b}$")
kept = np.zeros(len(b_dsi), bool); kept[cs_idx] = True
ax2.scatter(q_dsi[plane & ~kept, 0], q_dsi[plane & ~kept, 1], s=14, color="0.85", label="not acquired")
ax2.scatter(q_dsi[plane & kept, 0], q_dsi[plane & kept, 1], s=14, label="acquired")
ax2.set(title="CS-DSI subset, same plane", aspect="equal", xlabel="$q_x$, drawn as $\\sqrt{b}$ (√(s/mm²))"); ax2.legend(fontsize=7)
fig.tight_layout()
```

## Free-form and multidimensional sampling

The families above vary direction and b-value. Other acquisitions add further dimensions:
several diffusion times ([Chapter 21](../05-advanced/21-multi-diffusion-time.md)), several echo times ([Chapter 20](../05-advanced/20-echo-time.md)), or the shape of the
encoding (b-tensor encoding, [Chapter 22](../05-advanced/22-frontiers.md)). Each adds sensitivity to a tissue property that
direction and b-value alone cannot separate.

## Measure it: direction count and tensor precision

How many directions does a tensor fit need? The simulation below builds the signal of one
white matter voxel from the simulated brain's model ([Chapter 5](./05-diffusion-encoding.md)) for a fiber along a fixed axis, adds
noise at SNR 20 on the b=0 image, fits the tensor with dipy, and repeats 400 times for each
scheme. The fitted quantity is the fractional anisotropy (FA), a number from 0 to 1 that says
how direction-dependent the diffusion is: 0 means water moves equally in all directions,
1 means it moves along only one line ([Chapter 15](../04-modeling/15-signal-representations.md) defines it). The spread of the fitted
FA is the precision of the scheme.

```{code-cell} python
:tags: [hide-input]
rng = np.random.default_rng(0)
fiber = np.array([0.0, 0.0, 1.0])
SNR, N_REP = 20.0, 400
results = {}
for n_dirs in [6, 12, 30, 64]:
    bvals, bvecs = schemes.single_shell(1000, n_dirs, n_b0=max(1, n_dirs // 10))
    gtab = gradient_table(bvals, bvecs=bvecs)
    clean = signal.white_matter(bvals, bvecs @ fiber)
    noisy = np.abs(clean[None, :] + (rng.normal(size=(N_REP, len(bvals))) + 1j * rng.normal(size=(N_REP, len(bvals)))) / SNR)
    fit = TensorModel(gtab).fit(noisy)
    results[n_dirs] = fit.fa
fa_true = TensorModel(gtab).fit(signal.white_matter(bvals, bvecs @ fiber)[None, :]).fa[0]

fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.boxplot([results[n] for n in results], tick_labels=[f"{n} dirs\n({n + max(1, n // 10)} vols)" for n in results], widths=0.5)
ax.axhline(fa_true, color="0.5", lw=1, ls="--")
ax.set(ylabel="fitted FA", title=f"FA of one WM voxel, SNR {SNR:.0f} at b = 0, {N_REP} noise realizations")
fig.tight_layout()
for n, fa in results.items():
    print(f"{n:>2} directions: FA {fa.mean():.3f} ± {fa.std():.3f}   (noise-free value {fa_true:.3f})")

# gray matter: no true anisotropy at all
b_gm, v_gm = schemes.single_shell(1000, 30, n_b0=3)
clean_gm = signal.gray_matter(b_gm)
noisy_gm = np.abs(clean_gm[None, :] + (rng.normal(size=(N_REP, len(b_gm))) + 1j * rng.normal(size=(N_REP, len(b_gm)))) / SNR)
fa_gm = TensorModel(gradient_table(b_gm, bvecs=v_gm)).fit(noisy_gm).fa
print(f"gray matter, 30 directions: FA {fa_gm.mean():.3f} ± {fa_gm.std():.3f}   (noise-free value 0)")
```

Precision improves roughly with the square root of the number of measurements, as it does
for any average. With only six directions the mean also shifts upward (0.859 against the
noise-free 0.850); from twelve directions on it sits on the true value. The upward shift
comes from noise adding apparent anisotropy, and it is small here only because this
voxel's FA is already high. The reason is that the fit reports the tensor's three diffusivities
(the lengths of the ellipsoid's axes) sorted from largest to smallest. Noise pushes each of
them up or down at random, and the sorting always puts whichever one noise pushed up in
first place and whichever it pushed down in last place, so the three come out more
different from each other than they really are. FA measures exactly that difference. The
effect is clearest in gray matter, where water moves equally in all directions and the
true FA is 0: the last line above shows the fit reporting FA of about 0.1 from noise alone
{cite:p}`pierpaoli1996`.
(At high b the noise floor has the opposite effect on white matter, lowering its FA {cite:p}`jones2004squashing`;
[Chapter 8](../03-preprocessing/08-noise.md) shows both.)

Precision depends on how the directions are spread as well as on how many there are. A fit
is *well-conditioned* when small errors in the measurements produce only small errors in
the result. If the directions cluster, some orientations of the ellipsoid are barely
measured, and the noise in those is magnified; how badly then depends on which way the
fiber happens to point. The figure repeats the simulation for three direction sets while
turning the fiber from the z axis (0°) to the x axis (90°): 6 directions spread over the
sphere, 30 spread over the sphere, and 30 clustered within 45° of the z axis. The small
spheres show each set, with each direction and its opposite. Look at which lines are flat.

```{code-cell} python
:tags: [hide-input]
def cone_directions(n, half_angle_deg, seed):
    """n random directions within half_angle_deg of the z axis."""
    g = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        v = g.standard_normal(3); v /= np.linalg.norm(v)
        v = v if v[2] >= 0 else -v
        if np.degrees(np.arccos(v[2])) <= half_angle_deg:
            out.append(v)
    return np.array(out)

_, v6 = schemes.single_shell(1000, 6, n_b0=0)
_, v30 = schemes.single_shell(1000, 30, n_b0=0)
dir_sets = {"6 spread": (v6, 1), "30 spread": (v30, 3), "30 clustered (45° cone)": (cone_directions(30, 45, seed=1), 3)}
set_colors = {"6 spread": PALETTE[1], "30 spread": PALETTE[0], "30 clustered (45° cone)": PALETTE[3]}
angles = np.arange(0, 91, 10)
N_ANG = 300
fa_sd = {}
for name, (v, n_b0) in dir_sets.items():
    bvals = np.r_[np.zeros(n_b0), np.full(len(v), 1000.0)]
    bvecs = np.r_[np.tile([1.0, 0.0, 0.0], (n_b0, 1)), v]
    model = TensorModel(gradient_table(bvals, bvecs=bvecs))
    sds = []
    for a in angles:
        f = np.array([np.sin(np.deg2rad(a)), 0.0, np.cos(np.deg2rad(a))])
        clean = signal.white_matter(bvals, bvecs @ f)
        noisy = np.abs(clean[None, :] + (rng.normal(size=(N_ANG, len(bvals))) + 1j * rng.normal(size=(N_ANG, len(bvals)))) / SNR)
        sds.append(model.fit(noisy).fa.std())
    fa_sd[name] = np.array(sds)

fig = plt.figure(figsize=(9, 5.2))
gs = fig.add_gridspec(2, 3, height_ratios=(1, 1.6))
for i, (name, (v, _)) in enumerate(dir_sets.items()):
    ax = fig.add_subplot(gs[0, i], projection="3d")
    both = np.concatenate([v, -v])
    ax.scatter(*both.T, s=10, color=set_colors[name], depthshade=False)
    ax.plot([0, 0], [0, 0], [-1.2, 1.2], color=INK["secondary"], lw=1)
    ax.text(0, 0, 1.35, "z", color=INK["secondary"], fontsize=8, ha="center")
    ax.set_title(name, fontsize=9, color=set_colors[name])
    ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1)); ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.view_init(elev=15, azim=-60)
ax = fig.add_subplot(gs[1, :])
for name, sd in fa_sd.items():
    ax.plot(angles, sd, "o-", color=set_colors[name], label=name)
ax.set(xlabel="fiber angle from the z axis (degrees)", ylabel="spread of fitted FA (SD)", xticks=angles, ylim=(0, None),
       title=f"FA precision vs fiber orientation, SNR {SNR:.0f} at b = 0, {N_ANG} noise realizations per point")
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
for name, sd in fa_sd.items():
    print(f"{name:>24}: FA SD from {sd.min():.3f} to {sd.max():.3f} across fiber angles "
          f"(fiber along z: {sd[0]:.3f}, fiber along x: {sd[-1]:.3f})")
```

The 30 spread directions give the same precision whatever the fiber's orientation, which
is what "around 30 directions" in the list above means {cite:p}`jones2004`. The six spread
directions are noisier, and their precision depends on how the fiber happens to lie
relative to the six (an SD between 0.049 and 0.081 here). The clustered set is the
cautionary case: it has as many volumes as the spread set of 30, yet when the fiber runs
through the cluster its FA is more than four times noisier (SD 0.120 against 0.027), because the directions all measure
nearly the same thing and nothing constrains the ellipsoid across the cluster.

For the tensor, or for a fiber orientation distribution, half the sphere is enough, since
each direction also measures its opposite; directions clustered in one region are what
cost precision. The direction generator used in this book (`schemes.electrostatic_directions`)
produces such half-sphere sets. Eddy-current correction is a different matter. The
distortion an eddy current causes flips sign when the gradient does, while the diffusion
signal does not, so a scheme whose directions are spread over the whole sphere (most
directions then have a near-opposite partner) lets the correction of
[Chapter 11](../03-preprocessing/11-eddy-currents.md) tell eddy-current distortion from
real anatomy. Acquisition schemes are therefore best designed on the whole sphere, as FSL
recommends for eddy {cite:p}`andersson2016`; a half-sphere set can be spread over the whole
sphere by flipping about half of its directions to their opposites, chosen so the flipped
ones are spread evenly, which changes nothing for the models.

:::{admonition} In practice: the b-value and b-vector files
:class: tip
A scheme travels with the image as two small text files in the format of FSL {cite:p}`jenkinson2012`. The `.bval` file
holds one b-value per volume (s/mm², 0 for the b=0 volumes). The `.bvec` file holds three
rows, x, y, and z, with one column per volume: the unit gradient direction of that volume
(zeros for b=0). The converter that makes the NIfTI image from the scanner's DICOM files
(usually `dcm2niix` {cite:p}`li2016`) writes both.

- **The directions are in the image's frame, not the scanner's.** FSL's b-vectors are
  given along the image's voxel axes (first, second, third array dimension), and when the
  image's affine has a positive determinant (the voxel order FSL calls neurological) the
  x component is also negated. So a b-vector file belongs to one particular image: if a
  tool reorders, flips, or reslices the voxel axes, the b-vectors must be changed to match.
  MRtrix's own format (`-grad`) instead uses scanner coordinates and converts on reading
  and writing.
- **Typical errors** are one axis with the wrong sign (after a reorientation or a format
  conversion), two axes swapped (after the image is transposed), the file transposed (one
  row per volume), and b-values or b-vectors left out of step with the volumes after some
  are removed or series are concatenated.
- **These errors hide.** Applying the same flip or swap to every direction leaves the
  tensor's three diffusivities, and so MD and FA, unchanged; only the fitted directions are
  wrong, and with them tractography.
- **Check every new protocol.** MRtrix's `dwigradcheck` tries every flip and axis swap and
  keeps the one whose fiber directions join up into the longest streamlines
  {cite:p}`jeurissen2014grad,tournier2019`. A quick manual check is to fit the tensor and display the
  principal direction, colored red for left-right, green for anterior-posterior, blue for
  inferior-superior ([Chapter 15](../04-modeling/15-signal-representations.md)): the corpus
  callosum at the midline should be red, the corticospinal tract blue, the cingulum green,
  and the direction lines should follow the tracts rather than cross them.
- **Motion rotates the directions.** When motion correction rotates a volume to match the
  others, its b-vector must rotate with it; eddy writes the rotated set as
  `*.eddy_rotated_bvecs` ([Chapter 12](../03-preprocessing/12-motion-and-dropout.md)).
:::

## Scan time

One volume is acquired per TR (the repetition time, the time taken to acquire every slice of
one volume once), so scan time is the number of volumes times TR. What sets TR is the
subject of [Chapter 7](./07-acquisition-parameters.md): mainly the number of slices, and *multiband*, which excites and reads
several slices at once {cite:p}`feinberg2010,setsompop2012`. With 2 mm slices and a multiband factor of 3 (three slices at a
time), a whole-brain TR is about 3.5 s, the value used below:

```{code-cell} python
:tags: [hide-input]
TR = 3.5
for name, (b, v) in examples.items():
    print(f"{name:>40}: {len(b):3d} volumes, {schemes.scan_time_s(len(b), TR) / 60:4.1f} min")
```

Reverse phase-encode acquisitions for distortion correction ([Chapter 10](../03-preprocessing/10-susceptibility-distortion.md)) add either a few
b=0 volumes or a full second copy of the scheme, which doubles the number of volumes (not
the number of directions: the copy repeats the same directions with the opposite
phase-encode polarity).

## Table 6.1: scheme to model

Most "no" entries in the table come from the two figures above. A single shell at
b = 1000 has too little angular contrast to separate crossing fibers, so fiber-orientation
methods are marginal there; any single shell gives one point per direction on the decay
curve, so every model of how the decay bends (kurtosis, multi-tissue CSD, NODDI, MAP-MRI)
is ruled out.

The other entries count directions per shell. Fiber-orientation methods such as
constrained spherical deconvolution (CSD) fit one shell at a time, so what matters is the
best single high-b shell, not the total over all shells: the rule used here asks for 45
directions on one shell at b of about 2000 or more, the number of unknowns in the order-8 fit that CSD
software uses by default ([Chapter 16](../04-modeling/16-fiber-orientation.md)). With
fewer, CSD still runs but its fiber orientation distributions come out broader. Likewise
the tensor is fit to the low shells (b ≤ 1000), where its assumptions hold, and the
"around 30 directions" rule applies to those.

The verdicts in the table are those of `schemes.analysis_matrix`, which
applies these counting rules to a scheme's b-values; the same function builds the table
of [Chapter 19](../04-modeling/19-what-your-data-allow.md). Its output for the five schemes:

```{code-cell} python
:tags: [hide-input]
table_schemes = {
    "30 dirs": examples["single-shell, 30 dirs, b = 1000 (DTI)"][0],
    "64 dirs": examples["single-shell, 64 dirs, b = 2000 (HARDI)"][0],
    "HBCD": b_hbcd,
    "DSI-257": b_dsi,
    "CS-DSI-64": b_dsi[cs_idx],
}
verdicts = {name: schemes.analysis_matrix(b) for name, b in table_schemes.items()}
analyses = [row[0] for row in verdicts["HBCD"]][:-1]  # the last row, complex denoising, is not about the scheme
print(f"{'':>36}" + "".join(f"{name:>11}" for name in table_schemes))
for i, analysis in enumerate(analyses):
    print(f"{analysis:>36}" + "".join(f"{verdicts[name][i][1]:>11}" for name in table_schemes))
print()
for name in ("HBCD", "CS-DSI-64"):
    for analysis, verdict, reason in verdicts[name][1:5]:
        print(f"{name}, {analysis}: {verdict} ({reason})")
```

The table spells these out. The rows are grouped by what the analysis needs from the data,
and each analysis carries a few words on what it measures; Part IV introduces each one
properly. The two tractography rows are not in the function; each follows the model it
traces (the tensor, or single-shell CSD).

| Analysis | 30 dirs, b = 1000 | 64 dirs, b = 2000 | multi-shell (HBCD) | DSI-257 | CS-DSI-64 |
|---|---|---|---|---|---|
| ***Needs one shell*** | | | | | |
| ADC / mean diffusivity (average rate of diffusion) | yes | yes (b-dependent value) | yes | yes | yes |
| DTI: FA, principal direction (the tensor ellipsoid) | yes | marginal: high b breaks the Gaussian assumption | marginal: 18 directions at b ≤ 1000 (6 + 12) | yes, inner shells | marginal as acquired (9 inner points)\* |
| Deterministic tractography, tensor-based (tracing along the ellipsoid's long axis) | yes | marginal | marginal | yes | marginal\* |
| ***Needs high angular contrast (b ≥ 2000, many directions)*** | | | | | |
| Constrained spherical deconvolution, single-tissue (fiber directions within a voxel) | marginal: low angular contrast at b = 1000 | yes | marginal: best shell (b = 3000) has 29 directions | yes | marginal as acquired\* |
| Probabilistic tractography on fODFs (tracing along the fiber directions found by deconvolution) | marginal | yes | marginal | yes | marginal\* |
| ***Needs two or more non-zero shells*** | | | | | |
| Diffusion kurtosis (how much the decay bends) | no: needs ≥ 2 non-zero shells | no | yes | yes | yes |
| Multi-tissue CSD (fiber directions, with gray matter and CSF separated out) | no: needs multiple shells | no | marginal: 29 directions on the best shell | marginal: grid points must be binned into shells | no as acquired\* |
| NODDI, spherical mean, free-water models (sizes of the water pools) | no | no | yes | yes | yes |
| MAP-MRI / propagator (the displacement distribution, from a fitted basis) | no | no | yes, ≥ 3 shells preferred | yes | yes |
| ***Needs a q-space grid*** | | | | | |
| DSI / model-free propagator (the displacement distribution, by Fourier transform) | no | no | no | yes | no as acquired\* |

\* The function judges only the points that were measured and knows nothing of
compressed-sensing reconstruction. CS-DSI is acquired in order to be reconstructed onto
the full grid; once that fill-in succeeds, the starred cells behave like the DSI column,
with the quality of the reconstruction as the extra uncertainty.

"Marginal" means the fit runs but its assumptions are strained or its precision is poor;
Part IV shows each case on the simulated datasets. [Chapter 19](../04-modeling/19-what-your-data-allow.md) extends this table with acquisition
parameters beyond the scheme.

## Measure it: the simulated datasets under five schemes

:::{admonition} Simulated dataset pending
:class: note
This section will load the `ref-schemes` dataset, the simulated brain under the 30-direction,
64-direction, HBCD, and DSI schemes at equal total scan time (plus the CS-DSI subset), and show the same slice at
matched b-values for each. Part IV fits every model of Table 6.1 to these data.
:::

## What this implies for acquisition

- **Decide the analysis first, then the scheme.** Table 6.1 is read from the left column.
- **Thirty well-spread directions at b = 1000** is the floor for a tensor study; more
  directions buy precision in proportion to the square root of their number.
- **Any model of non-Gaussian diffusion needs at least two non-zero shells**, one of them
  at b ≥ 2000.
- **Full DSI is a strong-gradient, long-scan acquisition**; CS-DSI recovers most of it from a
  quarter to a third of the volumes.
- **Interleave shells and b=0 volumes** and spread directions across shells.
- **Scan time is volumes times TR**; [Chapter 7](./07-acquisition-parameters.md) shows what sets TR.

## Further reading

Direction schemes and how many are needed {cite:p}`jones1999,jones2004`, multi-shell design
{cite:p}`caruyer2013`, DSI {cite:p}`wedeen2005`, compressed-sensing DSI {cite:p}`menzel2011`,
and the prevalence of crossing fibers {cite:p}`jeurissen2013`.
