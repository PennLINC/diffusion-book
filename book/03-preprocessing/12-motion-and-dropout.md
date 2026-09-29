---
title: "12. Head motion, multiband, and slice dropout"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** a single-shell synthetic series on the packaged 3 mm volume, moved and re-encoded volume by volume ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
- **Simulated live in this page:** one slice of the simulated brain re-simulated at the head poses measured in the real subject, as a multiband-3 acquisition with dropout events.
- **`motion-mb`** (pending): a measured head-motion trace re-simulated volume by volume, and multiband 3 with 10 % dropout events ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-motion-mb)).
- **`truth`** (pending): the 27 analytic ground-truth maps and the true fiber orientations ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-truth), [Appendix E](../appendices/e-truth-map-catalogue.md)).

Live-tier figures simulate one slice of the simulated brain in the page through TRXScan's Python package ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md#live-tier)); they run in seconds at build time.
Pipeline-tier datasets are simulated offline by TRXScan ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)) and are marked *pending* until their release; the figures that need them say so where they will appear.
:::

## Learning goals

After this chapter you can:

- separate motion between volumes from motion during a volume and say what each does to
  the data
- correct between-volume motion by registration, rotate the b-vectors accordingly, and
  measure the residual
- recognize slice dropout, explain why multiband spreads it over several slices, detect it
  from the model residuals, and replace it
- report motion with the quality measures pipelines use

```{code-cell} python
:tags: [hide-cell]
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # one BLAS thread: registration is many small operations
import numpy as np
import matplotlib.pyplot as plt
from dipy.align.imaffine import AffineRegistration, MutualInformationMetric
from dipy.align.transforms import RigidTransform3D
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel

from dwibook import phantoms, schemes, synth
from dwibook.plotting import INK, PALETTE, animate, set_style, show_image

set_style()
```

## The physics

A diffusion acquisition lasts minutes and consists of dozens of volumes, each acquired
slice by slice within a few seconds. Head motion affects it on two time scales:

- **Between volumes.** The head is in a different position for different volumes. Each
  volume is internally consistent but misaligned with the others, so a voxel does not
  contain the same tissue throughout the series. The fix is registration, with one
  addition specific to diffusion: a rotation of the head is a rotation of the gradient
  direction relative to the tissue, so the b-vectors must be rotated by the same amount
  {cite:p}`leemans2009`.
- **During the diffusion encoding of one slice.** The encoding gradients make the signal
  phase proportional to displacement ([Chapter 5](../02-diffusion-encoding/05-diffusion-encoding.md)). A small movement during the 40 ms
  between the pulses gives all spins in a slice a large, spatially varying phase, and the
  signal of that slice is partly or completely lost. This is **slice dropout**: a dark
  slice, or a dark band of slices, in one volume. It grows with b (with the pulse timing
  fixed, the phase per unit of movement grows as the square root of b), so it is most
  frequent in the highest shell, and it is the dominant motion artifact in children and
  clinical populations.

A third case sits between the two: the head moves while a volume is being acquired, but
not during the brief encoding of any one slice. Each slice is then intact, but the slices
of one volume were taken with the head in different positions, so the volume is not a
rigid copy of the head. Registering the whole volume as one rigid body cannot undo this.
FSL eddy can model it by letting the head pose change within the volume
(`--mporder`, *slice-to-volume* correction {cite:p}`andersson2017`); for that it must know
when each slice (or multiband group) was acquired, given as a slice-timing file
(`--slspec`) or read from the BIDS JSON sidecar (`--json`).

### Why a rotated head needs rotated b-vectors

The scanner applies each diffusion gradient in its own fixed frame. If the head has turned,
the same gradient meets the fibers at a different angle, and the signal of that volume is
the signal for that different angle. Registration turns the image back, but it does not
change which direction was actually measured relative to the tissue:

```{code-cell} python
:tags: [hide-input]
def rot2(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

theta = np.linspace(0, 2 * np.pi, 200)
head = np.array([0.75 * np.cos(theta), np.sin(theta)])                      # top-down view, anterior up
nose = np.array([[-0.15, 0.0, 0.15], [0.97, 1.18, 0.97]])
fiber = np.array([[-0.45, 0.45], [-0.35, 0.35]])                             # a fiber bundle at about 38 degrees
g = np.array([1.0, 0.0])                                                     # the gradient, fixed in the scanner
TURN = 10.0

def draw_head(ax, deg, title):
    R = rot2(deg)
    for shape, kw in [(head, dict(color=INK["secondary"], lw=1.5)), (nose, dict(color=INK["secondary"], lw=1.5)),
                      (fiber, dict(color=PALETTE[0], lw=5, solid_capstyle="round"))]:
        xy = R @ shape
        ax.plot(xy[0], xy[1], **kw)
    ax.set(xlim=(-1.5, 1.6), ylim=(-1.55, 1.45), aspect="equal", title=title)
    ax.set_axis_off()

def arrow(ax, vec, **kw):
    ax.annotate("", xy=(1.25 * vec[0], -1.3 + 1.25 * vec[1]), xytext=(0, -1.3), arrowprops=dict(arrowstyle="-|>", lw=2, **kw))

fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.6))
draw_head(axes[0], 0, "reference position")
arrow(axes[0], g, color=PALETTE[1])
axes[0].text(1.3, -1.3, "g", color=PALETTE[1], va="center")
draw_head(axes[1], TURN, f"head turned {TURN:.0f}°,\nsame gradient from the scanner")
arrow(axes[1], g, color=PALETTE[1])
axes[1].text(1.3, -1.3, "g", color=PALETTE[1], va="center")
draw_head(axes[2], 0, "after registration: image turned back")
arrow(axes[2], g, color=INK["secondary"], linestyle="--")
arrow(axes[2], rot2(-TURN) @ g, color=PALETTE[1])
axes[2].text(1.3, -1.3, "b-vector as\nin the table", color=INK["secondary"], va="bottom", fontsize=8)
axes[2].text(1.3, -1.55, "rotated b-vector", color=PALETTE[1], va="center", fontsize=8)
fig.tight_layout()
```

In the middle panel the head (and its fiber, in blue) has turned 10° while the scanner's
gradient (orange arrow) has not, so the angle between gradient and fiber is 10° different
from what it would have been. After registration (right) the image is back in the reference
position, and in that frame the gradient the tissue actually experienced points 10° the
other way (solid arrow), not along the direction written in the b-vector table (dashed). A
fit that uses the table's direction assigns this volume's signal to the wrong angle. The fix
is to rotate each volume's b-vector by the rotation the registration found
{cite:p}`leemans2009`.

### Why movement during the encoding erases signal

The diffusion pulses wind each spin's phase in proportion to its position along the gradient
and then unwind it ([Chapter 5](../02-diffusion-encoding/05-diffusion-encoding.md)). A spin
that has moved between the pulses keeps a leftover phase proportional to how far it moved,
and the pulses are strong: at b = 1000 (with the 20 ms pulses 40 ms apart of Chapter 5), a
displacement of about 36 µm, under 2 % of a 2 mm voxel, is enough for a full turn. What
matters is whether the spins within one voxel moved by *different* amounts. The animation
follows the spins across one 2 mm voxel. If the whole head shifts, every spin gets the same
leftover phase: the arrows turn together, and the signal, the length of their sum, is
unchanged (the image only records its size). If the head turns, spins on one side of the
voxel move farther along the gradient than spins on the other side, their phases fan out,
and the sum shrinks.

```{code-cell} python
:tags: [hide-input]
DELTA, SMALL_DELTA, VOX = 0.040, 0.020, 2.0                 # s, s, mm
def k_of(b):                                                 # phase per mm of displacement, rad/mm
    return np.sqrt(b / (DELTA - SMALL_DELTA / 3))
def kept(b, turn_deg):                                       # |mean phasor| across one voxel for a turn of the head
    x = np.linspace(-VOX / 2, VOX / 2, 401)
    return np.abs(np.mean(np.exp(1j * k_of(b) * np.radians(turn_deg) * x)))

x_spins = np.linspace(-VOX / 2, VOX / 2, 13)
turns = np.linspace(0, 1.0, 101)
frames = [("still", 0.0)] * 2 + [("shift", d) for d in (0.1, 0.3, 0.5) for _ in range(2)] + \
         [("turn", a) for a in np.linspace(0, 1.0, 11)] + [("turn", 1.0)] * 2

fig, (ax, axc) = plt.subplots(1, 2, figsize=(8.4, 3.4), gridspec_kw=dict(width_ratios=[1.35, 1]))
for b, colr in [(1000, PALETTE[0]), (3000, PALETTE[1])]:
    axc.plot(turns, [kept(b, a) for a in turns], color=colr, label=f"b = {b}")
dot, = axc.plot([], [], "o", color=INK["primary"], ms=7)
axc.set(xlabel="head turn between the pulses (degrees)", ylabel="signal kept in the voxel", ylim=(0, 1.05))
axc.legend(loc="upper right")

def frame(i):
    kind, amount = frames[i]
    phase = k_of(1000) * (amount if kind == "shift" else np.radians(amount) * x_spins) * np.ones_like(x_spins)
    frac = kept(1000, amount) if kind == "turn" else 1.0
    ax.clear()
    ax.add_patch(plt.Rectangle((-1.1, -0.35), 2.2, 0.7, fill=False, ec=INK["grid"], lw=1))
    for xs, ph in zip(x_spins, phase):
        ax.annotate("", xy=(xs + 0.13 * np.cos(ph), 0.25 * np.sin(ph)), xytext=(xs, 0), arrowprops=dict(arrowstyle="-|>", color=PALETTE[0], lw=1.4))
    mean = np.mean(np.exp(1j * phase)) if kind == "shift" else frac * np.exp(1j * 0)
    ax.plot([-1.0, 1.0], [-0.75, -0.75], color=INK["grid"], lw=8, solid_capstyle="butt")
    ax.plot([-1.0, -1.0 + 2.0 * abs(mean)], [-0.75, -0.75], color=PALETTE[1], lw=8, solid_capstyle="butt")
    ax.text(0, -1.0, f"sum of the arrows (b = 1000): {100 * abs(mean):.0f} % of full signal", ha="center", fontsize=9)
    ax.set(xlim=(-1.25, 1.25), ylim=(-1.15, 0.5), aspect="equal")
    ax.set_axis_off()
    ax.set_title({"still": "head still: spins realign",
                  "shift": f"whole head shifts {amount:.1f} mm: spins turn together",
                  "turn": f"head turns {amount:.1f}°: spins fan out"}[kind], fontsize=9)
    dot.set_data([amount if kind == "turn" else 0.0], [frac])
    fig.tight_layout()

animate(fig, frame, range(len(frames)), fps=2, width=700,
        alt="a row of spin arrows across one voxel after the second diffusion pulse; with the head still or shifted as a whole the arrows stay parallel and the summed signal bar stays full; as the head turns by up to 1 degree the arrows fan out and the bar shrinks, following a curve of signal kept versus turn angle that falls faster at b = 3000")
print(f"displacement for one full turn of phase at b = 1000: {1000 * 2 * np.pi / k_of(1000):.0f} µm")
print("signal kept in a 2 mm voxel for a turn of the head between the pulses:")
for a in (0.25, 0.5, 1.0):
    print(f"  {a:.2f} degrees: b = 1000 {100 * kept(1000, a):.0f} %, b = 3000 {100 * kept(3000, a):.0f} %")
```

A shift of the whole head, even by half a millimeter, turns every arrow by the same amount
and the bar stays full. A turn of a fraction of a degree fans the arrows across the voxel,
and the bar shrinks; the curve on the right shows how fast. A turn of half a degree leaves
the voxel 66 % of its signal at b = 1000 but 19 % at b = 3000, and a full degree leaves 4 %
at b = 1000 (the small rebound of the b = 3000 curve is the arrows wrapping past a full
turn; the signal stays low). A turn of a degree in the few tens of milliseconds between the pulses is a quick
jerk (a cough, a swallow, a startle), and it happens in a scan of dozens of volumes. Because
the turn affects every voxel of the slice being encoded, the whole slice goes dark: this is
**slice dropout**. (Pulsation of the brain with the heartbeat moves neighboring tissue by
different amounts in the same way, which is why dropout-like signal loss also appears near
the brainstem without any head motion.)

**Multiband** acquisition excites several slices at once and separates them using the
coils ([Chapter 7](../02-diffusion-encoding/07-acquisition-parameters.md)). Each excitation,
or shot, carries its own diffusion encoding, so a movement during one shot affects all
slices of that group, spread across the brain at regular intervals, rather than one slice:

```{code-cell} python
:tags: [hide-input]
N_DEMO, HIT = 15, 7
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))
for ax, mb in zip(axes, (1, 3)):
    group = [HIT % (N_DEMO // mb) + j * (N_DEMO // mb) for j in range(mb)]
    ax.add_patch(plt.matplotlib.patches.Ellipse((0, N_DEMO / 2 - 0.5), 1.9, N_DEMO + 2, fill=False, ec=INK["secondary"], lw=1.2))
    for s in range(N_DEMO):
        hit = s in group
        ax.plot([-0.8, 0.8], [s, s], color=PALETTE[1] if hit else INK["grid"], lw=6 if hit else 4, solid_capstyle="butt")
    ax.set(xlim=(-1.3, 1.3), ylim=(-3.2, N_DEMO + 1), title=f"{'single-band' if mb == 1 else f'multiband {mb}'}: {N_DEMO // mb} shots of {mb} slice{'s' if mb > 1 else ''}")
    ax.set_axis_off()
    ax.text(0, -2.8, f"one movement darkens slice{'s' if mb > 1 else ''} {', '.join(map(str, group))}", ha="center", fontsize=8)
fig.tight_layout()
```

The head is seen from the side, with its slices as horizontal bars. Without multiband, a
movement during one shot darkens one slice (orange). With multiband 3 the same movement
darkens three slices a third of the brain apart, because they were encoded together.

## The artifact-free reference

The toy demonstrations use the 3 mm volume with a single-shell scheme (12 directions at
b = 1000 plus 2 b=0), 14 volumes in all. Three volumes are acquired with the head rotated
and translated; the series is computed with the gradients seen from the rotated head, then
the images are moved.

```{code-cell} python
:tags: [hide-input]
vol = phantoms.brain_volume()
mask = vol["mask"]
K = phantoms.VOLUME_VENTRICLE_SLICE
bvals, bvecs = schemes.single_shell(1000, 12, n_b0=2)
still = [((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))] * len(bvals)
poses = list(still)
poses[5] = ((4.0, 0.0, 0.0), (0.0, 0.5, 0.0))    # 4 degrees about the row axis, 1.5 mm shift
poses[8] = ((0.0, -3.0, 2.0), (0.7, 0.0, 0.0))   # 3 and 2 degrees about two axes, 2 mm shift
poses[11] = ((0.0, 0.0, 6.0), (0.0, 0.0, 0.5))   # 6 degrees in-plane
reference, _ = synth.moving_series(vol, bvals, bvecs, still)
moved, bvecs_head = synth.moving_series(vol, bvals, bvecs, poses)
print(f"series {moved.shape}; volumes 5, 8, 11 moved")
```

## See it: between-volume motion

The animation cycles through the diffusion-weighted volumes of the series as a viewer
would scroll through them; the outline is the true brain edge. Three volumes jump.

```{code-cell} python
:tags: [hide-input]
fig, ax = plt.subplots(figsize=(3.6, 4.0))
im = ax.imshow(moved[:, :, K, 2], cmap="gray", vmin=0, vmax=0.2)
ax.contour(mask[:, :, K], levels=[0.5], colors=[PALETTE[1]], linewidths=0.8)
ax.set_axis_off()
label = ax.set_title("")
moved_volumes = {v for v, (rot, tr) in enumerate(poses) if any(rot) or any(tr)}

def frame(v):
    im.set_data(moved[:, :, K, v])
    label.set_text(f"volume {v}" + (" (head moved)" if v in moved_volumes else ""))
    return im, label

animate(fig, frame, range(2, len(bvals)), fps=2, width=320,
        alt="axial slice of successive diffusion-weighted volumes; three volumes are displaced")
```

## Correction step by step: registration and b-vector rotation

A pipeline registers every volume rigidly to a reference (six parameters: three rotations,
three translations), because it cannot know in advance which volumes moved. For
diffusion-weighted volumes, FSL eddy registers to a predicted image rather than to the b=0
([Chapter 11](./11-eddy-currents.md)). The toy version takes two shortcuts: it registers
only the three volumes it knows were moved, and it registers them to the first b=0 volume
directly, using mutual information, a similarity measure that tolerates the difference in
contrast. The rotation recovered by the registration is then applied to the b-vector of
that volume.

```{code-cell} python
:tags: [hide-input]
affreg = AffineRegistration(metric=MutualInformationMetric(nbins=32, sampling_proportion=None),
                            level_iters=[100, 30], sigmas=[1.0, 0.0], factors=[2, 1], verbosity=0)
registered = moved.copy()
bvecs_rot = bvecs.copy()
static = moved[..., 0]
for v in [5, 8, 11]:
    mapping = affreg.optimize(static, moved[..., v], RigidTransform3D(), None)
    registered[..., v] = mapping.transform(moved[..., v])
    R_est = mapping.affine[:3, :3]  # maps the moved volume's grid onto the static grid
    bvecs_rot[v] = R_est @ bvecs[v]
    err_deg = np.degrees(np.arccos(np.clip(abs(bvecs_rot[v] @ bvecs_head[v]), 0, 1)))
    print(f"volume {v:>2}: rotation recovered to within {err_deg:.1f}° of the true gradient direction in the head's frame")
```

## Residual error versus truth

Three fits show what each step buys: the moved series with the nominal b-vectors
(uncorrected), the registered series with the nominal b-vectors (images aligned, gradients
not rotated), and the registered series with the rotated b-vectors.

```{code-cell} python
:tags: [hide-input]
ref_fit = synth.dti_maps(reference, bvals, bvecs, mask=mask)
cases = [("uncorrected", moved, bvecs), ("registered, b-vectors not rotated", registered, bvecs), ("registered, b-vectors rotated", registered, bvecs_rot)]
wm = vol["wm"] > 0.9
fig, axes = plt.subplots(2, 4, figsize=(12, 6.2), layout="constrained")
show_image(axes[0, 0], ref_fit["fa"][:, :, K], "FA, reference", kind="scalar", vmin=0, vmax=0.9)
show_image(axes[1, 0], wm[:, :, K], "white matter (where\ndirection error is scored)", vmin=0, vmax=1)
for j, (label, s, bv) in enumerate(cases, start=1):
    m = synth.dti_maps(s, bvals, bv, mask=mask)
    fa_err = np.abs(m["fa"] - ref_fit["fa"])[wm].mean()
    ang = np.degrees(np.arccos(np.clip(np.abs(np.sum(m["evecs"][..., 0] * ref_fit["evecs"][..., 0], axis=-1)), 0, 1)))
    show_image(axes[0, j], (m["fa"] - ref_fit["fa"])[:, :, K] * mask[:, :, K], f"FA error,\n{label}", kind="diff", vmin=-0.3, vmax=0.3)
    im_ang = axes[1, j].imshow(np.where(wm[:, :, K], ang[:, :, K], np.nan), cmap="magma", vmin=0, vmax=5)
    axes[1, j].set_facecolor("black"); axes[1, j].set_xticks([]); axes[1, j].set_yticks([]); axes[1, j].grid(False)
    axes[1, j].set_title("principal-direction error")
    print(f"{label:>34}: WM FA error {fa_err:.3f}, principal-direction error median {np.median(ang[wm]):.1f}°, "
          f"voxels off by more than 3°: {100 * np.mean(ang[wm] > 3):.0f} %")
fig.colorbar(im_ang, ax=axes[1, 1:], shrink=0.8, label="degrees")
```

The top row is the FA error, the bottom row the angle between each white-matter voxel's
fitted principal direction and the true one. Registration removes the large FA errors at
the edges and most of the direction error (bright voxels in the second column; the median
falls from 3.4° to 1.7°). Rotating the b-vectors changes FA very little and darkens the
direction map only slightly: the median falls to 1.4° and the share of white-matter voxels
off by more than 3° from 15 % to 12 %. The gain is modest here because only 3 of 12
diffusion-weighted volumes moved, by a few degrees each. It grows with the size of the
rotations and the number of rotated volumes, and the direction error is what tractography
follows, so pipelines always rotate.

:::{admonition} In practice: keeping the b-vectors right
:class: tip
- **Use the rotated table the pipeline wrote.** FSL eddy leaves the input `bvecs` file
  untouched and writes the rotated directions to a separate file,
  `<output>.eddy_rotated_bvecs`. Fitting the corrected data with the original file silently
  discards the rotation shown above.
- **Know which frame the table is in.** FSL `bvecs` are given along the image's voxel axes,
  not the scanner's axes, and the sign of the x component depends on the handedness of the
  image's affine (FSL flips it for images stored in the "neurological" orientation). Other
  tools store the table in the scanner frame (MRtrix) or read it as given (DIPY), so a table
  copied between tools or converters without conversion can end up with one axis flipped.
- **Resampling again means rotating again.** Any later resampling of the data, for example
  to AC-PC alignment or to a template, must rotate the b-vectors by the rotational part of
  that transform. A nonlinear warp rotates each region differently, so there the model
  (tensor or fiber orientations) must be reoriented voxel by voxel; one table no longer fits
  all voxels.
- **Check the table.** Fit a tensor and look at the principal-direction color map: the
  corpus callosum should run left-right and the corticospinal tract up-down. MRtrix's
  `dwigradcheck` tests the axis flips and permutations automatically.
:::

## Slice dropout

A within-volume event is simulated in a single volume: the head moves during one multiband
excitation, and the three slices of that shot lose 60 % of their signal.

```{code-cell} python
:tags: [hide-input]
MB = 3
n_slices = mask.shape[2]
shot = 12
dropped = [shot + k * (n_slices // MB) for k in range(MB)]
dwi_drop = synth.dropout(reference, volume=4, slices=dropped, attenuation=0.4)

fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.4))
show_image(axes[0], dwi_drop[:, :, dropped[1], 4], f"volume 4, slice {dropped[1]}: dropped", vmin=0, vmax=0.2)
show_image(axes[1], dwi_drop[:, :, dropped[1] + 1, 4], f"volume 4, slice {dropped[1] + 1}: normal", vmin=0, vmax=0.2)
axes[2].imshow(dwi_drop[:, 26, :, 4].T, cmap="gray", vmin=0, vmax=0.2, origin="lower", aspect="auto"); axes[2].set_axis_off()
axes[2].set_title(f"side (sagittal) view: slices\n{', '.join(map(str, dropped))} dark")
fig.tight_layout()
```

The dropped slice (left) shows nearly the same anatomy as its neighbor (middle) at 40 % of the
brightness. Seen from the side (right), the event is a set of dark horizontal lines, one per
slice of the shot, evenly spaced through the brain (the top one, slice 46, crosses only a thin
cap of brain): the multiband pattern from the schematic
above.

### Detection and replacement

Dropout is detected from the model: fit the tensor to all volumes, predict every volume
from the fit, and compare each slice of each volume with its prediction. A dropped slice
has a residual far outside the distribution of that slice across volumes. The affected
slice is then replaced by the prediction, and the fit is repeated without it
{cite:p}`andersson2016b`. FSL eddy's outlier replacement (`--repol`) is built on this
principle, but the version below is a toy. Eddy predicts each volume with its Gaussian
process from the other volumes, not with a tensor; it standardizes each slice's difference
from its prediction by the spread of those differences across all the slices of that
shell, which are many, so one outlier barely moves the scale; and it flags a slice whose
signal falls more than a set number of standard deviations below its prediction
(`--ol_nstd`, default 4). The toy uses a tensor fit and, because it has only 12 values per
slice, the robust scale described next.

"Far outside the distribution" needs a yardstick that the outlier cannot bend. The usual
z-score, the distance from the mean in units of the standard deviation, fails here, because
the dropped slice pulls the mean toward itself and inflates the standard deviation. With 12
volumes it can never reach a large z, however extreme it is. The robust alternative uses the
**median** in place of the mean and the **median absolute deviation** (MAD, the median of
each value's distance from the median) in place of the standard deviation; one outlier
barely moves either. A slice's residual in 12 volumes, 11 of them near zero and one at −0.6:

```{code-cell} python
:tags: [hide-input]
example = np.array([0.01, -0.02, 0.0, 0.02, -0.01, 0.015, -0.005, 0.01, -0.015, 0.005, 0.0, -0.6])
z_usual = (example[-1] - example.mean()) / example.std()
med_ex = np.median(example)
mad_ex = 1.4826 * np.median(np.abs(example - med_ex))  # 1.4826 makes the MAD match the SD for normal data
print(f"mean {example.mean():+.3f}, SD {example.std():.3f}  ->  usual z of the outlier {z_usual:.1f}")
print(f"median {med_ex:+.3f}, scaled MAD {mad_ex:.3f}  ->  robust z of the outlier {(example[-1] - med_ex) / mad_ex:.0f}")
```

With the mean and standard deviation the dropped value scores about −3, which a threshold
of 4 would miss; the outlier has hidden itself by inflating the yardstick. With the median
and MAD it scores in the dozens and stands out unmistakably. The detector below computes
this robust z for every slice in every diffusion-weighted volume and flags values below −4.

```{code-cell} python
:tags: [hide-input]
gtab = gradient_table(bvals, bvecs=bvecs)
fit = TensorModel(gtab, fit_method="WLS", return_S0_hat=True).fit(dwi_drop, mask=mask)
pred = fit.predict(gtab, S0=fit.S0_hat)
dw = np.flatnonzero(bvals > 0)  # only the diffusion-weighted volumes are candidates for dropout
resid = np.zeros((n_slices, len(dw)))
for k in range(n_slices):
    m = mask[:, :, k]
    if m.sum() > 20:
        # relative residual: (measured - predicted) / predicted, averaged over the slice
        resid[k] = np.mean(((dwi_drop[:, :, k, :][..., dw] - pred[:, :, k, :][..., dw]) / np.maximum(pred[:, :, k, :][..., dw], 1e-3))[m], axis=0)
# a robust z-score: median and median absolute deviation, so that the outlier itself does not
# inflate the scale it is judged against (with 12 volumes, one outlier would otherwise
# dominate the standard deviation and hide itself)
med = np.median(resid, axis=1, keepdims=True)
mad = 1.4826 * np.median(np.abs(resid - med), axis=1, keepdims=True) + 1e-6
z = (resid - med) / mad
flagged = [(int(k), int(dw[j])) for k, j in np.argwhere(z < -4)]
print("slices with residual more than 4 robust SD below their own median (slice, volume):", flagged)

fig, ax = plt.subplots(figsize=(7, 3.2))
im = ax.imshow(z, cmap="RdBu_r", vmin=-8, vmax=8, aspect="auto", origin="lower", extent=(dw[0] - 0.5, dw[-1] + 0.5, -0.5, n_slices - 0.5))
ax.set(xlabel="volume", ylabel="slice", title="standardized residual of each slice in each diffusion-weighted volume")
fig.colorbar(im, ax=ax, shrink=0.8, label="robust z")
ax.scatter([v for _, v in flagged], [k for k, _ in flagged], s=90, facecolors="none", edgecolors=INK["primary"], lw=1.5)
fig.tight_layout()
is_flagged = np.zeros(z.shape, bool)
for k, v in flagged:
    is_flagged[k, list(dw).index(v)] = True
others = z[~is_flagged]
print(f"robust z of the flagged slices: {', '.join(f'{val:.0f}' for val in z[is_flagged])}; "
      f"most extreme value anywhere else: {others[np.abs(others).argmax()]:.1f}, in slice {np.argwhere(np.where(is_flagged, 0, np.abs(z)) == np.abs(others).max())[0][0]}")
```

Each cell is one slice of one volume. The three circled cells, all in volume 4, are the
only ones below −4: the three slices of the one multiband shot, found from the residuals
alone, without being told where the event happened. The margin is narrower than in the
worked example (about −5, while a few cells elsewhere reach ±4). The dropped slice pulls the
tensor fit of its own slice toward itself, so the other volumes of that slice also miss
their predictions and its MAD grows. The same pull, of an outlier on the prediction it is
judged against, is why FSL eddy repeats the detection after replacing what it found,
predicting again each time with the outliers replaced.

The flagged slices are then replaced by their predictions. Here is slice 29 of volume 4 at
each stage:

```{code-cell} python
:tags: [hide-input]
repaired = dwi_drop.copy()
for k, v in flagged:
    repaired[:, :, k, v] = pred[:, :, k, v]
first_pass = repaired.copy()
# second pass: refit with the replaced slices, predict again, replace again
fit2 = TensorModel(gtab, fit_method="WLS", return_S0_hat=True).fit(repaired, mask=mask)
pred2 = fit2.predict(gtab, S0=fit2.S0_hat)
for k, v in flagged:
    repaired[:, :, k, v] = pred2[:, :, k, v]
k = dropped[1]
panels = [(dwi_drop, "dropped slice (acquired)"), (first_pass, "replaced, first pass"), (repaired, "replaced, after refitting"), (reference, "reference (no dropout)")]
fig, axes = plt.subplots(1, 4, figsize=(11, 3.2), layout="constrained")
for ax, (s, title) in zip(axes, panels):
    show_image(ax, s[:, :, k, 4], title, vmin=0, vmax=0.2)
wm_k = wm[:, :, k]
for s, title in panels[:3]:
    fa_k = synth.dti_maps(s, bvals, bvecs, mask=mask)["fa"][:, :, k]
    print(f"{title:>26}: signal {s[:, :, k, 4][wm_k].mean() / reference[:, :, k, 4][wm_k].mean():.0%} of the reference in WM, "
          f"FA error in WM {np.abs(fa_k - ref_fit['fa'][:, :, k])[wm_k].mean():.3f}")
```

The first replacement is brighter than the dropped slice but still darker than the
reference, because the tensor it was predicted from was fitted with the dark slice included
and was pulled down by it. Refitting with the replaced slice and predicting again closes
part of the remaining gap: the white-matter signal rises from 68 % to 82 % of the reference
and the FA error in the slice falls from 0.034 to 0.020 (0.056 with the dropout left in).
Further rounds continue the approach; eddy iterates in the same way, with its own
predictor. The replacement is never
identical to the reference: it is a prediction from the other volumes, smoother than a real
measurement. Replacement uses the other volumes' information to fill the gap; it cannot
recover the
measurement, and a dataset with many dropped slices in the same shell is effectively a
dataset with fewer directions. Pipelines report the number of replaced slices per volume
as a quality measure, and studies set a threshold above which a scan is excluded.

## Reporting motion

Pipelines summarize motion with two numbers per volume. The first is a displacement since
the previous volume. The version used below, and in much of the fMRI literature, is Power's
**framewise displacement** (FD) {cite:p}`power2012`: the sum of the absolute changes in the
three translations, in millimeters, and the three rotations converted to millimeters as
the distance a point on a 50 mm sphere, roughly the head's surface, travels. The second is
the count of outlier slices from the detection above. A worked conversion, and the FD of the
toy series:

```{code-cell} python
:tags: [hide-input]
VOXEL_MM, RADIUS_MM = 3.0, 50.0
print(f"a 1° rotation moves a point 50 mm from the center by {RADIUS_MM * np.radians(1.0):.2f} mm")
rot_deg = np.array([p[0] for p in poses]); trans_mm = VOXEL_MM * np.array([p[1] for p in poses])
fd_toy = np.r_[0.0, np.abs(np.diff(trans_mm, axis=0)).sum(axis=1) + RADIUS_MM * np.radians(np.abs(np.diff(rot_deg, axis=0))).sum(axis=1)]
fig, ax = plt.subplots(figsize=(6.5, 2.6))
ax.bar(np.arange(len(bvals)), fd_toy, color=PALETTE[0])
ax.axhline(0.5, color=INK["secondary"], ls="--", lw=1)
ax.text(len(bvals) - 0.5, 0.55, "0.5 mm", ha="right", va="bottom", fontsize=8, color=INK["secondary"])
ax.set(xlabel="volume", ylabel="framewise displacement (mm)")
ax.grid(axis="x", visible=False)
fig.tight_layout()
print("FD above zero: " + ", ".join(f"volume {v} {fd_toy[v]:.1f} mm" for v in np.flatnonzero(fd_toy > 0)))
```

Each moved volume produces two bars: one for moving away from the reference position and
one for moving back in the next volume. A threshold such as the dashed 0.5 mm line is one
way to count high-motion volumes; the live section below uses the same measure on a real
subject's motion.

Not every tool reports this FD. FSL eddy, and its quality report `eddy_quad`, give instead
the root-mean-square displacement of the voxels in the brain mask, relative to the
first volume (*absolute*) and to the previous volume (*relative*). The two measures weigh
rotations differently and are not interchangeable, so a threshold set on one does not
transfer to the other, and thresholds also differ between studies, populations, and the
measure being analyzed. Report which measure and which threshold were used.

Both numbers should be inspected before any group analysis, because
motion correlates with age and with clinical status, and residual motion effects bias FA
downward and MD upward in a way that can masquerade as a group difference.

## Measure it: one slice, simulated live

The simulator can move the head the way a real one moved. The simulated brain was built from
a real subject, and it comes with the head motion that qsiprep estimated for that subject,
one rigid pose per volume; for each volume the fiber paths and tissue maps are moved by that pose and the signal is re-simulated, so the
fibre-to-gradient angles change as they do in a moving head, not just the image position.
Multiband dropout events are added on top, and the simulator records which shots dropped.

```{code-cell} python
:tags: [hide-input]
import trxscan as ts
from dwibook import phantom as ph

trace = ph.phantom().motion["AP"]
g = ph.gtab(16)
fig, ax = plt.subplots(1, 2, figsize=(9, 2.8))
trace.plot(n_vol=g.bvals.size, ax=ax)
fig.tight_layout()
still = ph.run(g, ph.PROTO.replace(mb=3), ts.Artifacts(noise=2e-4, seed=1))
moved = ph.run(g, ph.PROTO.replace(mb=3), ts.Artifacts(noise=2e-4, motion=trace, dropout=0.25, seed=1))
print(f"{len(moved.dropout)} dropout events in {g.bvals.size} volumes: " + ", ".join(f"volume {d.volume} shot {d.shot} x{d.attenuation:.2f}" for d in moved.dropout))
```

The two plots are the measured head pose of each volume, translations and rotations, over
the part of the scan simulated here: flat stretches are a still head, and steps are
movements between volumes. The printed line lists the dropout events the simulator added;
"x0.40", for example, means that shot kept 40 % of its signal.

```{code-cell} python
:tags: [hide-input]
fd = trace.framewise_displacement(g.bvals.size)
worst = int(np.argmax(fd))
fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2))
show_image(axes[0], ph.axial(still.magnitude, worst), f"volume {worst}, still")
show_image(axes[1], ph.axial(moved.magnitude, worst), f"volume {worst}, moved (FD {fd[worst]:.1f} mm)")
show_image(axes[2], ph.axial(moved.magnitude, worst) - ph.axial(still.magnitude, worst), "difference", kind="diff")
fig.tight_layout()
```

The difference image is not a pure shift: where the head rotated, the white matter signal
changes because the fibres now make a different angle with the diffusion gradient. Motion
correction can register the images back, but the b-vectors of a rotated volume must be
rotated with it, which is why every pipeline reports the rotation it applied. The dropped
shots are the ground truth an outlier detector is scored against; the offline `motion-mb`
dataset runs the full 76-volume trace and FSL eddy on it.

## What acquisition choices reduce it

- **Shorter scans move less.** Multiband and short TR reduce the time per volume and the
  total time.
- **Interleaved shells and spread b=0 volumes.** Interleaving means ordering the volumes
  so that the shells alternate through the scan (b = 1000, 2000, 1000, 2000, …) rather
  than acquiring one shell after the other. A burst of movement then costs a few directions
  of every shell rather than most of one shell.
- **Multiband is a trade-off.** A higher multiband factor shortens the scan, so there is less
  time to move, but each dropout event darkens more slices, because more slices share each
  shot.
- **Padding, instruction, and, for children, mock scanning** reduce motion more than any
  sequence parameter.
- **Acquire enough directions that replacing some slices leaves a usable scheme.**

## Further reading

Rotating the b-matrix {cite:p}`leemans2009`, integrated motion and eddy correction
{cite:p}`andersson2016`, outlier detection and replacement {cite:p}`andersson2016b`,
slice-to-volume motion correction {cite:p}`andersson2017`, and framewise displacement
{cite:p}`power2012`.
