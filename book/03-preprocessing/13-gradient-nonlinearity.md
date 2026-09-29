---
title: "13. Gradient nonlinearity"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** a synthetic series on the packaged 2 mm slice with a radial gradient warp ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
- **`gnl`** (pending): gradient nonlinearity on two gradient systems, a severity sweep, and warp-only and encoding-only runs, with the true displacement field and gradient deviation ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-gnl)).
- **`truth`** (pending): the 27 analytic ground-truth maps and the true fiber orientations ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-truth), [Appendix E](../appendices/e-truth-map-catalogue.md)).

Pipeline-tier datasets are simulated offline by TRXScan ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)) and are marked *pending* until their release; the figures that need them say so where they will appear.
:::

:::{admonition} Why this chapter comes last among the artifacts
:class: note
Gradient nonlinearity is a fixed property of the scanner, but its corrections are the last
ones a pipeline applies ([Chapter 14](./14-assembled-pipeline.md)): the geometric part joins the corrections of
Chapters [10](./10-susceptibility-distortion.md) through [12](./12-motion-and-dropout.md) in one resampling, and the encoding part is handed to the
model fits of Part IV.
:::

## Learning goals

After this chapter you can:

- explain why a gradient coil's field is not linear and name the two consequences: a
  spatial warp of the image and a per-voxel error in the diffusion encoding
- predict how both grow with distance from the isocenter
- apply the geometric correction (gradwarp) and the encoding correction (a per-voxel
  b-matrix) and show that each removes a different error
- state which acquisition choices reduce the problem

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt

from dwibook import phantoms, presets, schemes, signal, synth
from dwibook.plotting import INK, PALETTE, animate, set_style, show_image

set_style()
```

## The physics

Spatial encoding assumes that each gradient coil produces a field that grows in a straight
line with position ([Chapter 2](../01-mri-physics/02-spatial-encoding-kspace.md)): twice as far from the center, twice the field. The scanner
reads a spin's position off the field it felt. A real coil makes a straight ramp only near
the **isocenter**, the center of the magnet, where the head should be. Farther out the field
bends away from the ramp, by a few percent at 10 cm on a whole-body system and more on
head-only or high-performance gradient systems, where strength is bought at the expense of
linearity.

```{code-cell} python
:tags: [hide-input]
A_DEMO, R_MM, G_NOM = 0.03, 100.0, 40.0   # 3 % at 100 mm (the chapter's toy coil); a 40 mT/m gradient
x_mm = np.linspace(-150, 150, 301)
ideal = G_NOM * x_mm / 1000                               # mT
real = G_NOM * x_mm * (1 + A_DEMO * x_mm**2 / R_MM**2) / 1000

disp = x_mm * A_DEMO * x_mm**2 / R_MM**2                  # displayed minus true position (mm)
b_err = 100 * ((1 + 3 * A_DEMO * x_mm**2 / R_MM**2) ** 2 - 1)   # b error for a gradient along this axis (%)

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
for ax in axes:
    ax.axvspan(-9, 9, color=INK["grid"], alpha=0.7, lw=0)
    ax.axvline(0, color=INK["secondary"], lw=0.8)
    ax.set(xlabel="position along the gradient (cm)", xlim=(-15, 15))
ax = axes[0]
ax.plot(x_mm / 10, ideal, color=INK["secondary"], ls="--", label="ideal: a straight ramp")
ax.plot(x_mm / 10, real, color=PALETTE[1], label="real coil (this chapter's model)")
ax.text(0.4, -5.2, "isocenter", fontsize=8, color=INK["secondary"])
ax.text(-8.6, -6.4, "a centered head", fontsize=8, color=INK["secondary"])
ax.set(ylabel="gradient field (mT)", title="the field")
ax.legend(loc="upper left")
axes[1].plot(x_mm / 10, disp, color=PALETTE[1])
axes[1].plot(10, 100 * A_DEMO, "o", color=PALETTE[1])
axes[1].annotate(f"10 cm: {100 * A_DEMO:.0f} mm", (10, 100 * A_DEMO), xytext=(2.5, 7), fontsize=8,
                 arrowprops=dict(arrowstyle="->", color=INK["secondary"], lw=0.8))
axes[1].set(ylabel="displayed − true position (mm)", title="the image error (grows as distance³)")
axes[2].plot(x_mm / 10, b_err, color=PALETTE[0])
axes[2].set(ylabel="b error (%)", title="the b-value error (grows as distance²)")
fig.tight_layout()
print(f"field above the ramp: {100 * A_DEMO * (90 / R_MM) ** 2:.1f} % at 9 cm (edge of a centered head), "
      f"{100 * A_DEMO * (150 / R_MM) ** 2:.1f} % at 15 cm")
print(f"at 9 cm: displayed {A_DEMO * 90**3 / R_MM**2:.1f} mm too far out, b {b_err[np.isclose(x_mm, 90)][0]:.0f} % too high; "
      f"at 15 cm: {A_DEMO * 150**3 / R_MM**2:.1f} mm, b {b_err[np.isclose(x_mm, 150)][0]:.0f} % too high")
```

In the left panel the orange curve is hard to tell from the straight ramp inside the gray
band, where a centered head sits; the two part only beyond it. The other two panels
magnify the difference. The middle one is the image error: how far from its true position
a spin is displayed, 3 mm at 10 cm. The right one is the error in the diffusion
weighting, explained below. Both are near zero at the isocenter and climb steeply toward
the edges: at the edge of a centered head (9 cm) a spin is displayed 2.2 mm too far out
and its b-value is 15 % too high.

The shape of the curve is fixed by the coil's wiring. The vendor describes it with a set of
**spherical-harmonic coefficients**: a list of numbers that describes the field's shape in
three dimensions, delivered as the scanner's *coefficient file*. The coil in this figure
bends above the ramp; many real coils bend below it along some axes and above it along
others, and the consequences below then change sign.

Two things follow from one bent field. The figure shows both on a square grid centered on
the isocenter.

```{code-cell} python
:tags: [hide-input]
def phi(y, x):
    """Apparent position for true position (y, x) in mm under the toy coil."""
    f = 1 + A_DEMO * (x**2 + y**2) / R_MM**2
    return y * f, x * f

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 4.4))
lines = np.arange(-120, 121, 20.0)
t_line = np.linspace(-120, 120, 200)
for c in lines:
    ax1.plot(t_line, np.full_like(t_line, c), color=INK["grid"], lw=1.2)
    ax1.plot(np.full_like(t_line, c), t_line, color=INK["grid"], lw=1.2)
    wy, wx = phi(np.full_like(t_line, c), t_line); ax1.plot(wx, wy, color=PALETTE[0], lw=1)
    wy, wx = phi(t_line, np.full_like(t_line, c)); ax1.plot(wx, wy, color=PALETTE[0], lw=1)
ax1.plot(0, 0, "+", color=PALETTE[1], ms=12, mew=2)
ax1.set(xlim=(-170, 170), ylim=(-170, 170), aspect="equal", xlabel="x (mm)", ylabel="y (mm)",
        title="1. image error: where each point is displayed\n(gray: true grid, blue: as imaged)")

pos = np.arange(-120, 121, 60.0)
PY, PX = np.meshgrid(pos, pos, indexing="ij")
r2 = (PX**2 + PY**2) / R_MM**2
# effective gradient for a nominal gradient along x: the x-column of the Jacobian of phi
gx = 1 + A_DEMO * r2 + 2 * A_DEMO * PX**2 / R_MM**2
gy = 2 * A_DEMO * PX * PY / R_MM**2
L = 36.0
ax2.quiver(PX - L / 2, PY, np.full_like(PX, L), 0 * PY, angles="xy", scale_units="xy", scale=1,
           color="#b8b6b0", width=0.022, headwidth=2.2, headlength=2.5, headaxislength=2.3)
q = ax2.quiver(PX - L / 2, PY, gx * L, gy * L, np.hypot(gx, gy) ** 2, angles="xy", scale_units="xy", scale=1,
               cmap="viridis", clim=(1.0, 1.4), width=0.008)
ax2.plot(0, 0, "+", color=PALETTE[1], ms=12, mew=2)
fig.colorbar(q, ax=ax2, shrink=0.75, label="effective b / nominal b")
ax2.set(xlim=(-170, 170), ylim=(-170, 170), aspect="equal", xlabel="x (mm)",
        title="2. encoding error: the gradient each point feels\n(gray: nominal, color: actual)")
fig.tight_layout()
print(f"at 120 mm along x: displayed {phi(0.0, 120.0)[1] - 120:.1f} mm too far out; "
      f"gradient {100 * (1 + 3 * A_DEMO * 1.44 - 1):.0f} % too strong, b {100 * ((1 + 3 * A_DEMO * 1.44) ** 2 - 1):.0f} % too high")
```

1. **The image is warped.** Each spin is displayed where the field says it is, not where it
   is. In the left panel the grid near the orange cross (the isocenter) is untouched, while
   the corners are pushed outward by a centimeter or more. Where the image is stretched,
   the same signal is spread over more voxels, so it also looks darker.
2. **The diffusion encoding is wrong.** A diffusion gradient is the same coil switched on
   for a few tens of milliseconds, so it is bent too. The right panel shows the gradient
   that each position actually feels when the scanner asks for a left-right gradient of one
   fixed strength (gray): away from the isocenter it is longer, and off the axes it also
   tilts a little. A stronger gradient means a larger b-value (b grows with the square of
   the gradient strength, [Chapter 5](../02-diffusion-encoding/05-diffusion-encoding.md)), and a tilted one means the diffusion was measured
   along a slightly different direction than the gradient table says.

The two errors come from the same field but behave differently. The warp moves voxels;
the encoding error changes what was measured inside each voxel, so it survives any
unwarping of the image {cite:p}`bammer2003`.

The practical consequence is that neither error is easy to see. The warp is the same in
every volume of the diffusion series, so the volumes line up with one another and nothing
within the series looks wrong. It shows up as a head that is slightly the wrong shape, and
as a mismatch with any image that was corrected for it, which is usually the anatomical
image of the same session (see the warning box below). The encoding error leaves no mark on
the images at all; it only biases the numbers fitted from them.

:::{dropdown} The same thing in symbols
A spin at true position $r$ is displayed at $\phi(r)$, the position a perfectly linear
coil would assign to the field it felt. Its intensity is divided by $\det J(r)$, the local
volume change, where $J(r) = \partial\phi/\partial r$ is the gradient-deviation tensor. A
nominal diffusion gradient $g$ is felt as $J(r)\,g$ (written $J^\top g$ in some papers,
depending on the convention for $J$), so the effective b-value is $b\,\lVert J g\rVert^2$
for a unit $g$ and the effective direction is $J g / \lVert J g\rVert$. In the toy model of
this chapter, $\phi(r) = r\,(1 + a\,r^2/R^2)$ with $a$ = 0.03 and $R$ = 100 mm.
:::

## The artifact-free reference

The toy coil is the one drawn above: a point 10 cm from the isocenter is displayed 3 %
(3 mm) too far out, and because the displacement grows with the cube of the distance, a
point 15 cm out is displayed 10 mm too far. In a scanner the isocenter should sit near the
middle of the head, where these errors are smallest. Here the head is deliberately placed
off center: the isocenter sits just behind the back of the 2 mm slice, as if the head had
been positioned about 9 cm too far forward, so that the frontal lobe lies 18 cm from the
isocenter and the effects are large enough to see. Both effects are computed from that one
field: the images are warped, and each voxel's signal is computed with its own effective
b-value and direction.

```{code-cell} python
:tags: [hide-input]
t = phantoms.brain_slice()
mask = t["mask"]
vox = t["voxel_mm"]
A = 0.03  # 3 % apparent displacement at 100 mm from the isocenter (whole-body class)
center = (110.0, 63.5)  # isocenter, in (row, column) voxels: posterior of the brain
bvals, bvecs = schemes.multi_shell({1000: 12, 2000: 12}, n_b0=2)
orientation = synth.orientation_field(t["wm"])
reference = synth.synthetic_dwi(t, bvals, bvecs, orientation=orientation)

_, jac = synth.gnl_warp(np.zeros(mask.shape), A, vox, center)  # per-voxel gradient deviation (2 x 2, in-plane)
yy, xx = np.indices(mask.shape, dtype=float)
dist_mm = np.hypot(yy - center[0], xx - center[1]) * vox

# encoding error: each voxel sees g_eff = J g (in-plane part), so b and direction change
def encoded_series():
    out = np.zeros(mask.shape + (len(bvals),))
    for v in range(len(bvals)):
        g_in = jac @ bvecs[v, :2]                       # (ny, nx, 2)
        g_eff = np.concatenate([g_in, np.broadcast_to(bvecs[v, 2], mask.shape)[..., None]], axis=-1)
        norm = np.linalg.norm(g_eff, axis=-1)
        b_eff = bvals[v] * norm**2
        cos = np.sum(orientation * g_eff, axis=-1) / np.maximum(norm, 1e-12)
        s0 = {k: phantoms.PROTON_DENSITY[k] * np.exp(-presets.TE_HBCD_MS / presets.T2_MS["adult"][k]) for k in ("WM", "GM", "CSF")}
        out[..., v] = (t["wm"] * s0["WM"] * signal.white_matter(b_eff, cos) + t["gm"] * s0["GM"] * signal.gray_matter(b_eff) + t["csf"] * s0["CSF"] * signal.csf(b_eff))
    return out

encoded = encoded_series()                       # encoding error only
acquired, _ = synth.gnl_warp(encoded, A, vox, center)  # plus the spatial warp
b_scale = np.linalg.norm(jac @ np.array([0.0, 1.0]), axis=-1) ** 2
disp_mm = A_DEMO * dist_mm**3 / R_MM**2          # how far each point is displayed from where it is
rows = np.nonzero(mask.any(axis=1))[0]
print(f"isocenter {(center[0] - (rows.min() + rows.max()) / 2) * vox:.0f} mm behind the middle of the brain; "
      f"frontal edge {dist_mm[mask].max():.0f} mm from it, displayed {disp_mm[mask].max():.0f} mm too far out")
print(f"effective b / nominal b for a left-right gradient: {b_scale[mask].min():.2f} to {b_scale[mask].max():.2f} across the brain")
```

## See it: the warp and the encoding deviation

The warp is easiest to see by flipping between the true image and the acquired one. The
grid lines are drawn at fixed true positions in the first frame and where the scanner
displays those same positions in the second.

```{code-cell} python
:tags: [hide-input]
R_vox = R_MM / vox
def warp_pts(py, px):
    dy, dx = py - center[0], px - center[1]
    f = 1 + A * (dy**2 + dx**2) / R_vox**2
    return center[0] + dy * f, center[1] + dx * f

fig, ax = plt.subplots(figsize=(3.6, 4.0))
im = ax.imshow(reference[..., 0], cmap="gray", vmin=0, vmax=0.5)
ax.contour(mask, levels=[0.5], colors=[PALETTE[1]], linewidths=0.8)
ax.plot(center[1], center[0], "+", color=PALETTE[3], ms=10, mew=2)
tt = np.linspace(0, mask.shape[0] - 1, 120)
grid = [(np.full_like(tt, c), tt) for c in range(4, 124, 12)] + [(tt, np.full_like(tt, c)) for c in range(4, 124, 12)]
glines = [ax.plot(px, py, color=PALETTE[0], lw=0.6, alpha=0.8)[0] for py, px in grid]
ax.set(xlim=(-0.5, 127.5), ylim=(127.5, -0.5)); ax.set_axis_off()
label = ax.set_title("")
fig.tight_layout()

def frame(k):
    im.set_data([reference, acquired][k][..., 0])
    for ln, (py, px) in zip(glines, grid):
        wy, wx = (py, px) if k == 0 else warp_pts(py, px)
        ln.set_data(wx, wy)
    label.set_text(["true geometry", "acquired (b = 0)"][k])
    return [im, label, *glines]

animate(fig, frame, [0, 1], fps=1, width=340, dpi=80,
        alt="the b = 0 slice flips between its true geometry and the acquired image; in the acquired frame the brain spills past the orange true outline at the front, the grid lines spread apart toward the front, and the frontal lobe looks darker")
```

In the acquired frame the front of the brain spills past the orange outline (the true
edge), and the grid lines fan out toward the front. The yellow cross is the isocenter,
just behind the brain; near it nothing moves.

```{code-cell} python
:tags: [hide-input]
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
show_image(axes[0], reference[..., 0], "true geometry", vmin=0, vmax=0.5)
axes[0].contour(mask, levels=[0.5], colors=[PALETTE[1]], linewidths=0.8)
ax = axes[1]; ax.imshow(acquired[..., 0], cmap="gray", vmin=0, vmax=0.5); ax.contour(mask, levels=[0.5], colors=[PALETTE[1]], linewidths=0.8); ax.set_axis_off(); ax.set_title("acquired: stretched away from the isocenter")
im = axes[2].imshow(np.where(mask, disp_mm, np.nan), cmap="viridis"); axes[2].set_axis_off(); axes[2].set_title("displacement (mm)"); fig.colorbar(im, ax=axes[2], shrink=0.7)
im = axes[3].imshow(np.where(mask, b_scale, np.nan), cmap="RdBu_r", vmin=0.8, vmax=1.2); axes[3].set_axis_off(); axes[3].set_title("effective b / nominal b (L-R gradient)"); fig.colorbar(im, ax=axes[3], shrink=0.7)
for ax in axes:
    ax.plot(center[1], center[0], "+", color=PALETTE[3], ms=8, mew=2)
fig.tight_layout()
```

The displacement map shows how far each point moves: almost nothing near the isocenter
(yellow cross), most at the frontal edge. The effective b-value is too high in the same
places, by up to 21 % for a left-right gradient. The two go together for one reason: this
coil's field climbs faster than the ramp away from the isocenter, so the local gradient is
steeper than nominal. A steeper gradient both spreads positions apart (the image is
stretched) and encodes more diffusion weighting (b is too high). For a coil that bends
below the ramp both signs flip: the image is compressed and b is too low.

## Correction step by step: gradwarp

The geometric correction, usually called *gradwarp*, puts every voxel back where it
belongs. From the coefficient file it computes how far each point was displaced, then
resamples the image from the displayed positions back to the true ones.

Moving the voxels is not quite enough, because the warp also changed their brightness: a
region that was stretched spread its signal over more voxels and looks darker, and a
compressed one looks brighter. The second half of the correction multiplies by the
*Jacobian*, a correction for how much each voxel was squeezed or stretched, which restores
its brightness.

In practice the displacement is not applied on its own. It is composed with the
susceptibility, eddy-current, and motion corrections so that the data are resampled once
([Chapter 14](./14-assembled-pipeline.md)).

| Tool | What it provides |
|---|---|
| `gradunwarp` (HCP pipelines {cite:p}`glasser2013`) | the displacement field from the coefficient file, and the resampling |
| TORTOISE `CreateNonlinearityDisplacementMap` | the same displacement field |
| QSIPrep | applies the field inside its combined transform when given the coefficient file |

:::{admonition} The anatomical images are usually already corrected
:class: warning
The T1-weighted and T2-weighted images suffer the same warp, but most scanners correct them
automatically before export; the diffusion images usually are not corrected. An
uncorrected diffusion series therefore does not line up with the anatomical image of the
same session, by an amount that grows away from the isocenter. A misregistration that is
worst at the edges of the head, and that no rigid or affine registration can remove, is
the sign.
:::

:::{admonition} In practice: check whether the scanner already corrected the diffusion data
:class: tip
Some scanners apply gradwarp to diffusion images too, before export. Look before running
it, because correcting twice warps the image the other way.

- **Siemens** records its distortion correction in the `ImageType` field of the DICOM header
  (and the BIDS JSON): `DIS2D` for the two-dimensional version, `DIS3D` for the
  three-dimensional one, `ND` when none was applied.
- **GE** calls it gradwarp and records it in the protocol and private header fields; check
  the protocol and, if in doubt, ask the site.
- **The two-dimensional versions correct only within each slice**, not along the slice
  axis, so a 2D-corrected series still carries the through-plane part of the warp.
- **The encoding error remains either way.** No scanner-side image correction changes the
  b-values and directions each voxel received; that needs the per-voxel table below.

Given a coefficient file (`--gradient-file`), QSIPrep reads `ImageType` for each run and
applies only what is missing: the full 3D correction for uncorrected data, the
through-plane part for `DIS2D` data, and no spatial correction for `DIS3D` data, while
still writing the gradient-deviation map. When the tags are missing or cannot be trusted,
`--force gradwarp3D` or `--force gradwarp1D` (through-plane only) overrides them.
:::

```{code-cell} python
:tags: [hide-input]
unwarped = synth.gnl_unwarp(acquired, A, vox, center)
errs = {"acquired": acquired[..., 0] - reference[..., 0],
        "after gradwarp": unwarped[..., 0] - reference[..., 0],
        "encoding error alone": encoded[..., 0] - reference[..., 0]}
print("mean absolute error of the b=0 image inside the brain: "
      + ", ".join(f"{k} {np.abs(v)[mask].mean():.4f}" for k, v in errs.items()))
fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.3))
for ax, (k, v) in zip(axes, errs.items()):
    im = ax.imshow(v, cmap="RdBu_r", vmin=-0.3, vmax=0.3); ax.set_axis_off(); ax.set_title(k)
    ax.contour(mask, levels=[0.5], colors=[INK["secondary"]], linewidths=0.5)
fig.colorbar(im, ax=axes, shrink=0.7, label="image − truth")
```

Before correction the b=0 error (image minus truth) sits at tissue edges, strongest toward
the front, where boundaries moved farthest. After gradwarp only faint traces of
interpolation remain. The encoding error alone leaves the b=0 image untouched, because b = 0
has no encoding to get wrong.

## Correction step by step: per-voxel b-matrix

The encoding error cannot be fixed by changing the images, because it is not in their
geometry; it is in the gradient table. The fix is to give every voxel its own gradient
table. The pipeline writes a *gradient-deviation image*, which stores for every voxel how
the gradients there differed from nominal (the right panel of the grid figure, as numbers).
The model fit then converts the nominal table into that voxel's effective b-values and
directions and fits each voxel with its own table.

The deviation must be evaluated on the true grid, after gradwarp, because it is a property
of where the tissue really is.

:::{dropdown} Where the per-voxel correction lives in the tools
| Tool | Role |
|---|---|
| TORTOISE `CreateGradientNonlinearityBMatrix` | writes the per-voxel b-matrices |
| QSIPrep | writes a `graddev` gradient-deviation image when given the coefficient file |
| FSL `dtifit --gradnonlin` | tensor fit with the deviation image |
| FSL `bedpostx -g` | fiber-orientation fit with the deviation image |
| any other model | a loop over voxels with each voxel's table, as this chapter's fits do |
:::

The deviation image describes the gradients at each point of the scanner with the head in
the position it was scanned in, so it assumes the head stayed there. The deviation changes
slowly with position, so the millimeter movements of a typical scan matter little, but a
head that moved a long way between volumes breaks the assumption. And if the data are later
resampled or rotated, for example into AC-PC alignment or a template, the deviation image
must be resampled and its tensors rotated with them, just as the b-vectors must be
([Chapter 12](./12-motion-and-dropout.md)).

```{code-cell} python
:tags: [hide-input]
def fit_per_voxel(series, use_graddev):
    """Log-linear tensor fit, with the nominal table or with each voxel's effective table."""
    md = np.full(mask.shape, np.nan); fa = np.full(mask.shape, np.nan)
    dw = bvals > 0
    logS = np.log(np.clip(series, 1e-6, None))
    for (i, j) in zip(*np.nonzero(mask)):
        if use_graddev:
            g_in = jac[i, j] @ bvecs[:, :2].T          # (2, n)
            g = np.column_stack([g_in.T, bvecs[:, 2]])
            n = np.linalg.norm(g, axis=1)
            b = bvals * n**2; g = g / np.maximum(n[:, None], 1e-12)
        else:
            b, g = bvals, bvecs
        X = np.column_stack([np.ones(len(b)), signal.dti_design_matrix(b, g)])
        beta = np.linalg.lstsq(X, logS[i, j], rcond=None)[0]
        D = np.array([[beta[1], beta[4], beta[5]], [beta[4], beta[2], beta[6]], [beta[5], beta[6], beta[3]]])
        ev = np.clip(np.linalg.eigvalsh(D), 0, None)
        md[i, j] = ev.mean()
        fa[i, j] = np.sqrt(1.5 * np.sum((ev - ev.mean()) ** 2) / max(np.sum(ev**2), 1e-20))
    return md, fa

md_ref, fa_ref = fit_per_voxel(reference, False)
md_nom, fa_nom = fit_per_voxel(encoded, False)    # encoding error, nominal gradient table
md_cor, fa_cor = fit_per_voxel(encoded, True)     # encoding error, per-voxel b-matrix
```

## Residual error versus truth

The encoding error is isolated first: the fits below use the series with the encoding
deviation but without the spatial warp, so that no interpolation is involved.

```{code-cell} python
:tags: [hide-input]
fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.6))
for ax, md, title in zip(axes, [md_nom, md_cor], ["nominal gradient table", "per-voxel gradient table"]):
    im = ax.imshow(np.where(mask, 100 * (md - md_ref) / md_ref, np.nan), cmap="RdBu_r", vmin=-35, vmax=35)
    ax.plot(center[1], center[0], "+", color=PALETTE[3], ms=8, mew=2)
    ax.set_axis_off(); ax.set_title(f"MD error (%)\n{title}")
fig.colorbar(im, ax=axes, shrink=0.8, label="(fitted − true) / true, %")
print(f"MD error with the nominal table: {np.percentile((100 * (md_nom - md_ref) / md_ref)[mask], 1):.1f} % to "
      f"{np.percentile((100 * (md_nom - md_ref) / md_ref)[mask], 99):.1f} % (1st to 99th percentile in the brain)")
```

With the nominal table, MD is too high everywhere away from the isocenter, by up to about
a third at the front. The signal decayed under a larger b than the table says, and the fit,
which assumes the nominal b, can explain the extra decay only by faster diffusion. With
each voxel's own table the map is flat.

The profile below averages the same error over white matter in rings of equal distance
from the isocenter.

```{code-cell} python
:tags: [hide-input]
wm = t["wm"] > 0.9
bins = np.arange(20, 130, 10)
def radial(err):
    return [np.nanmean(np.abs(err)[wm & (dist_mm >= lo) & (dist_mm < lo + 10)]) for lo in bins]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.4))
ax1.plot(bins + 5, radial(100 * (md_nom - md_ref) / md_ref), label="nominal b-matrix")
ax1.plot(bins + 5, radial(100 * (md_cor - md_ref) / md_ref), label="per-voxel b-matrix")
ax1.set(xlabel="distance from isocenter (mm)", ylabel="|MD error| in WM (%)", title="mean diffusivity"); ax1.legend()
ax2.plot(bins + 5, radial(fa_nom - fa_ref), label="nominal b-matrix")
ax2.plot(bins + 5, radial(fa_cor - fa_ref), label="per-voxel b-matrix")
ax2.set(xlabel="distance from isocenter (mm)", ylabel="|FA error| in WM", title="fractional anisotropy"); ax2.legend()
fig.tight_layout()
far = wm & (dist_mm > 80)
rel = lambda md: 100 * np.nanmean(np.abs(md - md_ref)[far] / md_ref[far])
print(f"white matter more than 80 mm from the isocenter, encoding error only: MD error {rel(md_nom):.1f} % with the nominal table, {rel(md_cor):.1f} % with the per-voxel table")
md_full_nom, _ = fit_per_voxel(unwarped, False)
md_full_cor, _ = fit_per_voxel(unwarped, True)
print(f"same region, full pipeline (warp, gradwarp, then fit): {rel(md_full_nom):.1f} % with the nominal table, {rel(md_full_cor):.1f} % with the per-voxel table")
```

The error with the nominal gradient table grows with distance from the isocenter, as the
effective b-value departs from the nominal one. In the white matter more than 80 mm from
the isocenter it averages 16.6 %, far larger than the group differences most studies
report. The per-voxel table removes it, down to 1.0 %.

The second line of output runs the whole chain: warp, gradwarp, then the fit. There a
residual of 9.9 % remains even with the per-voxel table, and it has nothing to do with the
encoding. It is interpolation error, concentrated at the edges of white matter, and the toy
pays it twice: once when it simulates the acquisition by resampling the true images onto
the warped grid, and once when gradwarp resamples them back. A scanner forms the warped
image directly, so a real pipeline pays only the second, the same cost every resampling
carries ([Chapter 14](./14-assembled-pipeline.md)); the 9.9 % therefore overstates what
gradwarp itself costs. In this toy the warp is large (up to 18 mm), so the cost is large
too.

Most default pipelines apply neither correction unless they are given the coefficient
file.

## Where it sits in the pipeline

Gradwarp is a resampling and belongs with the other geometric corrections (susceptibility,
eddy, motion), composed into one transform. The gradient-deviation image is computed on
the final grid and passed to the model fit; it is the last thing the preprocessing produces
and the first thing the reconstruction consumes. Applying only gradwarp leaves the
encoding error; applying only the b-matrix correction leaves the geometry wrong, which
matters wherever the diffusion data meet an anatomical image or an atlas.

## Measure it: the simulated datasets

:::{admonition} Simulated dataset pending
:class: note
This section will load the `gnl` dataset (whole-body and Connectom presets, a severity
sweep, and warp-only and encoding-only runs) with the truth displacement field and
gradient-deviation image TRXScan wrote, and score `gradunwarp`, TORTOISE, and the
per-voxel fits against them.
:::

## What acquisition choices reduce it

- **Position the head at the isocenter.** The displacement grows with the cube of the
  distance and the b-value error with its square, so a few centimeters matter (worked
  example below).
- **Obtain the coefficient file** for the scanner before the study. Without it neither
  correction can be applied, and it is not part of the standard export.
- **Know the gradient system.** Head-insert and high-performance gradients are less linear
  than whole-body gradients; the gain in TE ([Chapter 5](../02-diffusion-encoding/05-diffusion-encoding.md)) comes with a larger correction.
- **Do not compare uncorrected diffusivities across scanners** or across head positions.

```{code-cell} python
:tags: [hide-input]
front = (rows.min(), (np.nonzero(mask[rows.min()])[0].mean()))   # frontal pole (row, column)
mid_row = (rows.min() + rows.max()) / 2
for label_, off_mm in [("head centered", 0.0), ("head 3 cm off center", 30.0)]:
    c = (mid_row + off_mm / vox, front[1])                     # isocenter moved backward = head moved forward
    _, jac_c = synth.gnl_warp(np.zeros(mask.shape), A, vox, c)
    J = jac_c[int(front[0]), int(round(front[1]))]
    d = np.hypot(front[0] - c[0], 0) * vox
    b_ap = np.linalg.norm(J @ np.array([1.0, 0.0])) ** 2         # anterior-posterior gradient at the frontal pole
    print(f"{label_:21s}: frontal pole {d:3.0f} mm from the isocenter, displaced {A * d**3 / R_MM**2:.1f} mm, "
          f"b error {100 * (b_ap - 1):.0f} % (front-back gradient)")
```

Moving the head 3 cm off center takes the frontal pole from 87 to 117 mm from the
isocenter, a factor of 1.34. Its displacement grows by about 1.34³ ≈ 2.4 times, from 2.0
to 4.8 mm, and its b-value error by about 1.34² ≈ 1.8 times, from 14 % to 26 %, for a
gradient pointing front to back.

## Further reading

The encoding error and its correction {cite:p}`bammer2003`, the HCP implementation
{cite:p}`glasser2013`, and the design note for the simulator's implementation in the
TRXScan repository.
