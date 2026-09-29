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
- **`ref-schemes`** (pending): the simulated brain under the 30-direction, 64-direction, HBCD, DSI, and CS-DSI schemes at matched scan time ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-ref-schemes)).

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
winding set by the gradient strength and pulse duration ([Chapter 5](./05-diffusion-encoding.md)). This is q-space. The
b-value grows with the square of the winding, so with the pulse timing held fixed the
distance from the origin grows with $\sqrt{b}$, not with $b$: the b = 4000 shell lies only
twice as far out as the b = 1000 shell. [Chapter 4](./04-diffusion-in-tissue.md) noted that the signal is the Fourier
transform of the displacement distribution evaluated at one such spatial frequency, so the
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

- **Six directions** is the mathematical minimum for a tensor fit. It is never used in
  practice because every measurement then influences the result with no redundancy.
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
protocol is one example (b = 500, 1000, 2000, 3000); the Human Connectome Project's
1000/2000/3000 with 90 directions each is another. The shells sample the decay with b as
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
  so that motion or scanner drift affects all shells equally ([Chapter 5](./05-diffusion-encoding.md)).

## DSI

Diffusion spectrum imaging samples q-space on a Cartesian grid, typically all lattice
points within a sphere: 257 points for a radius of 4 grid units, 515 for a radius of 5
{cite:p}`wedeen2005`. Because the signal is the Fourier transform of the displacement
distribution, a grid of samples allows that distribution to be reconstructed directly
by an inverse Fourier transform, without a model. The grid extends to high b-values (often
4000–8000 s/mm²) and needs a strong gradient system. The cost is the number of volumes:
257 volumes at a TR of 4 s is 17 minutes.

## CS-DSI

Compressed-sensing DSI acquires a random subset of the grid points, typically a quarter to
a third of them, and reconstructs the displacement distribution with the sparsity prior
introduced in [Chapter 3](../01-mri-physics/03-reconstruction.md), applied in q-space {cite:p}`menzel2011`. The same three
requirements apply: the subset must be irregular, the distribution must be compressible in
some basis, and the reconstruction is iterative. The result is DSI-like information in a
multi-shell-like scan time.

## Free-form and multidimensional sampling

The families above vary direction and b-value. Other acquisitions add further dimensions:
several diffusion times ([Chapter 22](../05-advanced/22-multi-diffusion-time.md)), several echo times ([Chapter 20](../05-advanced/20-multi-te.md)), or the shape of the
encoding (b-tensor encoding, [Chapter 23](../05-advanced/23-frontiers.md)). Each adds sensitivity to a tissue property that
direction and b-value alone cannot separate.

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
ax1.set(title="DSI grid, plane through the origin", aspect="equal", xlabel="$q_x$", ylabel="$q_y$")
kept = np.zeros(len(b_dsi), bool); kept[cs_idx] = True
ax2.scatter(q_dsi[plane & ~kept, 0], q_dsi[plane & ~kept, 1], s=14, color="0.85", label="not acquired")
ax2.scatter(q_dsi[plane & kept, 0], q_dsi[plane & kept, 1], s=14, label="acquired")
ax2.set(title="CS-DSI subset, same plane", aspect="equal", xlabel="$q_x$"); ax2.legend(fontsize=7)
fig.tight_layout()
```

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
```

Precision improves roughly with the square root of the number of measurements, as it does
for any average. The mean also shifts: with few directions FA is biased upward, because
noise adds apparent anisotropy. Note that a well-spread 6-direction set is already
well-conditioned, so the gain from more directions comes from averaging, not from a better
conditioned fit; a badly spread set of any size is worse than either.

```{code-cell} python
:tags: [hide-input]
b6, v6 = schemes.single_shell(1000, 6, n_b0=1)
b30, v30 = schemes.single_shell(1000, 30, n_b0=1)
cone = np.column_stack([0.3 * rng.standard_normal(30), 0.3 * rng.standard_normal(30), np.ones(30)])
cone /= np.linalg.norm(cone, axis=1, keepdims=True)
print(f"condition number of the tensor fit: 6 spread directions {signal.condition_number(b6, v6):.2f}, "
      f"30 spread {signal.condition_number(b30, v30):.2f}, 30 clustered in a cone {signal.condition_number(np.full(30, 1000.0), cone):.1f}")
```

## Scan time

One volume is acquired per TR, so scan time is the number of volumes times TR. With a
multiband factor of 3 and 2 mm slices, a whole-brain TR is about 3.5 s:

```{code-cell} python
:tags: [hide-input]
TR = 3.5
for name, (b, v) in examples.items():
    print(f"{name:>40}: {len(b):3d} volumes, {schemes.scan_time_s(len(b), TR) / 60:4.1f} min")
```

Reverse phase-encode acquisitions for distortion correction ([Chapter 10](../03-preprocessing/10-susceptibility-distortion.md)) add either a few
b=0 volumes or a full second copy of the scheme.

## Table 6.1: scheme to model

Most "no" entries in the table come from the two figures above. A single shell at
b = 1000 has too little angular contrast to separate crossing fibers, so fiber-orientation
methods are marginal there; any single shell gives one point per direction on the decay
curve, so every model of how the decay bends (kurtosis, multi-tissue CSD, NODDI, MAP-MRI)
is ruled out.

| Analysis | 30 dirs, b = 1000 | 64 dirs, b = 2000 | multi-shell (HBCD) | DSI-257 | CS-DSI-64 |
|---|---|---|---|---|---|
| ADC / mean diffusivity | yes | yes (b-dependent value) | yes | yes | yes |
| DTI (FA, principal direction) | yes | marginal: high b breaks the Gaussian assumption | yes, using the b ≤ 1000 shells | yes, inner shells | yes, inner points |
| Diffusion kurtosis | no: needs ≥ 2 non-zero shells | no | yes | yes | marginal |
| Constrained spherical deconvolution (single-tissue) | marginal: low angular contrast at b = 1000 | yes | yes, highest shell | yes | yes |
| Multi-tissue CSD | no: needs multiple shells | no | yes | yes | yes |
| NODDI, spherical mean, free-water models | no | no | yes | yes | marginal |
| MAP-MRI / propagator | no | no | yes, ≥ 3 shells preferred | yes | yes |
| DSI / model-free propagator | no | no | no | yes | yes |
| Deterministic tractography (tensor) | yes | yes | yes | yes | yes |
| Probabilistic tractography (fODF) | marginal | yes | yes | yes | yes |

"Marginal" means the fit runs but its assumptions are strained or its precision is poor;
Part IV shows each case on the simulated datasets. [Chapter 19](../04-modeling/19-what-your-data-allow.md) extends this table with acquisition
parameters beyond the scheme.

## Measure it: the simulated datasets under five schemes

:::{admonition} Simulated dataset pending
:class: note
This section will load the `ref-schemes` dataset, the simulated brain under the 30-direction,
64-direction, HBCD, DSI, and CS-DSI schemes at matched scan time, and show the same slice at
matched b-values for each. Part IV fits every model of Table 6.1 to these data.
:::

## What this implies for acquisition

- **Decide the analysis first, then the scheme.** Table 6.1 is read from the left column.
- **Thirty well-spread directions at b = 1000** is the floor for a tensor study; more
  directions buy precision in proportion to the square root of their number.
- **Any model of non-Gaussian diffusion needs at least two non-zero shells**, one of them
  at b ≥ 2000.
- **Full DSI is a strong-gradient, long-scan acquisition**; CS-DSI recovers most of it at a
  third of the volumes.
- **Interleave shells and b=0 volumes** and spread directions across shells.
- **Scan time is volumes times TR**; [Chapter 7](./07-acquisition-parameters.md) shows what sets TR.

## Further reading

Direction schemes and how many are needed {cite:p}`jones1999,jones2004`, multi-shell design
{cite:p}`caruyer2013`, DSI {cite:p}`wedeen2005`, compressed-sensing DSI {cite:p}`menzel2011`,
and the prevalence of crossing fibers {cite:p}`jeurissen2013`.
