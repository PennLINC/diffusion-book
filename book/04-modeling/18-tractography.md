---
title: "18. Tractography"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** a synthetic series on the packaged 3 mm volume, tracked against its known fiber orientations ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
- **`ref-clean`** (pending): the artifact-free, noise-free reference series with its truth maps and true fiber orientations ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-ref-clean)).
- **`ref-schemes`** (pending): the simulated brain under the 30-direction, 64-direction, HBCD, and DSI schemes, compared at equal total scan time (the per-volume noise is scaled with the number of volumes), plus a CS-DSI subset of the DSI run, which takes about a quarter of its time ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-ref-schemes)).
- **`truth`** (pending): the 27 analytic ground-truth maps and the true fiber orientations ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-truth), [Appendix E](../appendices/e-truth-map-catalogue.md)).

Pipeline-tier datasets are simulated offline by TRXScan ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)) and are marked *pending* until their release; the figures that need them say so where they will appear.
:::

## Learning goals

After this chapter you can:

- explain how streamlines are generated from local fiber orientations, and what the
  step size, curvature limit, stopping criterion, and seeding decide
- run deterministic and probabilistic tracking with anatomical constraints
- evaluate a tractogram against a known orientation field and count the streamlines that
  end in the wrong tissue
- state how the acquisition bounds what tractography can recover

```{code-cell} python
:tags: [hide-cell]
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import warnings
warnings.filterwarnings("ignore")  # dipy's fit notices are not part of the lesson
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from dipy.direction import peaks_from_model, ProbabilisticDirectionGetter
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response_ssst
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ActStoppingCriterion, StreamlineStatus
from dipy.tracking.streamline import Streamlines
from dipy.tracking.utils import seeds_from_mask
from scipy import ndimage

from dwibook import phantoms, schemes, synth
from dwibook.plotting import INK, PALETTE, TISSUE_COLORS, animate, set_style, show_image

set_style()
sphere = get_sphere(name="repulsion724")
```

## From orientations to streamlines

Tractography turns the per-voxel fiber orientations of [Chapter 16](./16-fiber-orientation.md) into curves. Starting
from a seed point, a streamline is grown by stepping a fixed distance along the local
orientation, re-evaluating the orientation at the new position, and repeating until a
stopping rule applies {cite:p}`mori1999,conturo1999,basser2000`. Each streamline is grown in both directions
from its seed, and the two halves are joined.

The animation below runs this recipe on a small two-dimensional toy. A curved bundle arches
over a fluid-filled ventricle; beside it, a straight bundle turns sharply left at its top.
Each white matter voxel holds a short stick showing its fiber orientation, and the tracker
sees nothing but those sticks and the tissue map. Seeds are white circles; every step is
half a voxel. Watch the four panels grow together: (a) a single deterministic streamline,
(b) 25 probabilistic streamlines from the same seed, (c) a streamline that stops at the
sharp corner, and (d) probabilistic streamlines from seeds scattered through the arch,
kept (black) or rejected (red) by where they end.

```{code-cell} python
:tags: [hide-input]
# A 2-D toy: an arched bundle over a ventricle, and a straight bundle with a sharp corner.
WM, GM, CSF = 0, 1, 2
H, W = 20, 30
yy, xx = np.mgrid[0:H, 0:W] + 0.5                        # voxel centres (row = y, column = x)
tissue = np.full((H, W), GM)                             # gray matter everywhere else
r = np.hypot(xx - 10, yy - 1)
arch = (r >= 4.5) & (r < 8.5) & (yy > 1)
tissue[arch] = WM
tissue[(r < 4.5) & (yy > 1)] = CSF                       # the ventricle under the arch
leg = (xx > 22) & (xx < 26) & (yy > 1) & (yy < 18)       # straight bundle ...
bar = (xx > 18) & (xx < 26) & (yy > 14) & (yy < 18)      # ... turning sharply left at the top
tissue[leg | bar] = WM
field = np.zeros((H, W, 2))                              # unit orientation (x, y) per WM voxel
field[arch] = np.stack([-(yy - 1), xx - 10], -1)[arch] / r[arch, None]   # tangent to the arch
field[leg] = (0, 1)
field[bar] = (1, 0)

STEP = 0.5  # voxels

def voxel(p):
    return int(np.clip(p[1], 0, H - 1e-6)), int(np.clip(p[0], 0, W - 1e-6))

def grow(seed, heading, limit_deg=45, spread_deg=0, rng=None, n_max=80):
    """Grow one end of a streamline; returns the points and how it stopped."""
    pts, d = [np.array(seed, float)], np.array(heading, float)
    for _ in range(n_max):
        v = field[voxel(pts[-1])]
        v = v if v @ d >= 0 else -v                      # orientations have no sign
        turn = np.degrees(np.arccos(np.clip(v @ d, -1, 1)))
        if turn > limit_deg:
            return pts, "curvature"
        if spread_deg:                                   # probabilistic: draw around v, inside the limit
            for _ in range(50):
                a = np.radians(np.clip(rng.normal(0, spread_deg), -2 * spread_deg, 2 * spread_deg))
                u = np.array([v[0] * np.cos(a) - v[1] * np.sin(a), v[0] * np.sin(a) + v[1] * np.cos(a)])
                if u @ d >= np.cos(np.radians(limit_deg)):
                    break
            v = u
        d = v
        pts.append(pts[-1] + STEP * d)
        t = tissue[voxel(pts[-1])]
        if t == GM:
            return pts, "gm"
        if t == CSF:
            return pts, "csf"
    return pts, "length"

def streamline(seed, **kw):
    v = field[voxel(seed)]
    return [grow(seed, v, **kw), grow(seed, -v, **kw)]  # both directions from the seed

rng = np.random.default_rng(3)
act_seeds = np.stack([xx[arch], yy[arch]], -1)[rng.choice(arch.sum(), 24, replace=False)] + rng.uniform(-0.4, 0.4, (24, 2))
arch_seed = (10, 7.5)
panels = {
    "a  deterministic: follow the local direction": [streamline(arch_seed)],
    "b  probabilistic: draw around it, many times": [streamline(arch_seed, spread_deg=15, rng=rng) for _ in range(25)],
    "c  curvature limit (45° per step)": [streamline((24, 7)), streamline((20, 16))],
    "d  anatomical constraints (ACT)": [streamline(p, spread_deg=15, rng=rng) for p in act_seeds],
}

tint = np.array([mpl.colors.to_rgb(TISSUE_COLORS[k]) for k in ("WM", "GM", "CSF")])
tint = 1 - 0.3 * (1 - tint)                              # pale tissue colours behind the sticks
BAD = PALETTE[7]
sticks = [[(x - 0.35 * f[0], y - 0.35 * f[1]), (x + 0.35 * f[0], y + 0.35 * f[1])]
          for x, y, f in zip(xx[tissue == WM], yy[tissue == WM], field[tissue == WM])]

fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.8))
artists = []
for ax, (title, lines) in zip(axes.flat, panels.items()):
    ax.imshow(tint[tissue], origin="lower", extent=(0, W, 0, H), interpolation="nearest")
    ax.add_collection(mpl.collections.LineCollection(sticks, colors=INK["secondary"], lw=0.8))
    ax.set(xticks=[], yticks=[], xlim=(0, W), ylim=(0, H))
    ax.set_title(title, loc="left", fontsize=10)
    ax.grid(False)
    seeds_xy = np.array([sl[0][0][0] for sl in lines])
    ax.plot(seeds_xy[:, 0], seeds_xy[:, 1], ls="", marker="o", ms=3.5, mfc="white", mec=INK["primary"], mew=0.8, zorder=5)
    if title.startswith("c"):
        ax.annotate("needs a 90° turn:\nstopped", (23.8, 14.2), xytext=(16, 12), fontsize=8, ha="center",
                    arrowprops=dict(arrowstyle="->", color=INK["secondary"], lw=0.8))
    for sl in lines:
        rejected = any(how in ("csf", "curvature") for _, how in sl)
        ln, = ax.plot([], [], lw=1.2 if len(lines) == 1 or "c " in title else 0.8, color=INK["primary"])
        ends = [ax.plot([], [], ls="", marker="o", ms=4, color=INK["primary"])[0] for _ in sl]
        artists.append((ln, ends, sl, rejected))
handles = [mpl.patches.Patch(color=tint[k], label=n) for k, n in [(WM, "white matter (sticks: fiber orientation)"), (GM, "gray matter"), (CSF, "CSF")]]
handles += [mpl.lines.Line2D([], [], ls="", marker="o", ms=4, mfc="white", mec=INK["primary"], label="seed"),
            mpl.lines.Line2D([], [], color=INK["primary"], marker="o", ms=4, label="accepted: ends in gray matter"),
            mpl.lines.Line2D([], [], color=BAD, marker="X", ms=6, label="rejected: entered CSF or turned too sharply")]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False)
fig.tight_layout(rect=(0, 0.07, 1, 1))

n_steps = max(len(p) for *_, sl, _ in artists for p, _ in sl)

def frame(i):
    for ln, ends, sl, rejected in artists:
        (pb, hb), (pf, hf) = sl[1], sl[0]
        n = min(i + 1, len(pb)), min(i + 1, len(pf))
        pts = np.array(pb[: n[0]][::-1] + pf[1: n[1]])
        done = n[0] == len(pb) and n[1] == len(pf)
        ln.set_data(pts[:, 0], pts[:, 1])
        ln.set(color=BAD if done and rejected else INK["primary"], zorder=4 if done and rejected else 2)
        for mk, (p, how), k in zip(ends, sl[::-1], (n[0], n[1])):
            if k == len(p):                                # this end has stopped: show how
                bad = how in ("csf", "curvature")
                mk.set_data([p[-1][0]], [p[-1][1]])
                mk.set(marker="X" if bad else "o", color=BAD if bad else INK["primary"], ms=7 if bad else 4)
            else:
                mk.set_data([], [])

animate(fig, frame, list(range(n_steps)) + [n_steps - 1] * 12, fps=8, width=760, dpi=70,
        alt="four panels of a toy white matter map with orientation sticks, streamlines growing half a voxel per step from white seed circles: "
            "(a) one deterministic streamline follows the arched bundle to gray matter at both ends; "
            "(b) 25 probabilistic streamlines from the same seed fan out across the width of the arch; "
            "(c) a streamline in the straight bundle stops at a 90-degree corner, marked with a red X, while one in the horizontal arm runs to gray matter; "
            "(d) probabilistic streamlines seeded through the arch end in gray matter (black) or are rejected where they enter the ventricle (red X)")
```

The decisions that shape a tractogram, and where the animation shows them:

- **The local model.** The tracker only sees the sticks. Here there is one per voxel; a
  tensor always gives one, and at a crossing it gives a single averaged direction that
  belongs to neither bundle. A fiber ODF ([Chapter 16](./16-fiber-orientation.md)) can
  give several, and the tracker picks among them.
- **Deterministic or probabilistic** (panels a and b). Deterministic tracking takes the
  peak nearest the current heading, so a seed always gives the same streamline.
  Probabilistic tracking draws each step's direction from the ODF around that peak (here a
  spread of 15°), so repeated streamlines from one seed fan out across the bundle and
  explore the uncertainty in the orientation.
- **Step size and curvature limit** (panel c). The limit is a maximum turn per step, here
  45° per half-voxel step. The straight bundle turns 90° from one voxel to the next, so the
  streamline stops at the corner; the limit exists to keep streamlines from doubling back
  or hopping onto a neighboring bundle, and it cannot tell that from a real sharp bend.
  Because the limit is per step, it depends on the step size: 30° per half-voxel step, as
  the probabilistic tracker below uses, allows up to about 60° of turn per voxel travelled.
- **Stopping criteria** (panel d). Classic tracking stops when FA falls below a threshold.
  Anatomically constrained tractography (ACT) {cite:p}`smith2012` instead uses a tissue
  segmentation: streamlines continue in white matter, terminate when they enter gray
  matter, and are rejected if they enter CSF or leave the brain, so that every accepted
  streamline connects two gray matter regions. A streamline that stops inside white matter,
  like the one in panel c, is rejected too.
- **Seeding.** From every white matter voxel, from the gray-white interface, or from a
  region of interest. Panel b seeds one point 25 times, panel d one point in each of 24
  voxels; the seeding density sets the number of streamlines and biases the density toward
  the seeded region.
- **Filtering.** The density of streamlines does not measure the density of fibers.
  SIFT {cite:p}`smith2013` removes streamlines and SIFT2 {cite:p}`smith2015` weights them so that their density
  matches the fiber density the ODFs imply; this is how the simulated brain's tractogram was
  built.

## The synthetic volume

The toy tier tracks on the 3 mm tissue volume with a single-shell scheme at b = 2000 (30
directions plus three b=0) and noise at SNR 25. The synthetic fibers run tangent to the
white matter boundary in every voxel and lie in the axial plane, so the answer key is an
orientation field with one fiber population per voxel, and a correct streamline follows
that field.

```{code-cell} python
:tags: [hide-input]
vol = phantoms.brain_volume()
mask = vol["mask"]
wm_frac, gm_frac, csf_frac = vol["wm"], vol["gm"], vol["csf"]
bvals, bvecs = schemes.single_shell(2000, 30, n_b0=3)
clean = synth.synthetic_dwi(vol, bvals, bvecs)
truth = synth.orientation_field(wm_frac)
sigma = clean[wm_frac > 0.9][:, bvals == 0].mean() / 25
noisy = synth.add_noise(clean, sigma, seed=0)
gtab = gradient_table(bvals, bvecs=bvecs)
affine = np.eye(4)  # voxel coordinates throughout
response, _ = auto_response_ssst(gtab, noisy, roi_radii=8, fa_thr=0.7)
csd = ConstrainedSphericalDeconvModel(gtab, response, sh_order_max=6)  # 28 coefficients from 30 directions
csd_fit = csd.fit(noisy, mask=mask)
pk = peaks_from_model(csd, noisy, sphere, relative_peak_threshold=0.5, min_separation_angle=25, mask=mask, npeaks=3)
```

## Tracking

Both trackers use ACT with the tissue fractions: white matter is the tracking domain, gray
matter the termination map, CSF the exclusion map. Seeds are placed in every voxel with
more than half white matter, one per voxel. Both step half a voxel; the deterministic
tracker follows the CSD peaks with dipy's default limit of 60° per step, and the
probabilistic tracker draws from the CSD fiber ODF within 30° per step.

```{code-cell} python
:tags: [hide-input]
class RecordingTracking(LocalTracking):
    """dipy's LocalTracking, also remembering how each end of each streamline stopped."""
    def _tracker(self, seed, first_step, streamline):
        steps, status = super()._tracker(seed, first_step, streamline)
        self.ends.append(status)
        return steps, status

def track(getter, **kw):
    """Every streamline (kept or not), with the stopping status of its two ends."""
    tracker = RecordingTracking(getter, stopping, seeds, affine, step_size=0.5, return_all=True, random_seed=0, **kw)
    lines, ends = [], []
    tracker.ends = []
    for s in tracker:
        lines.append(s)
        ends.append(tuple(tracker.ends))
        tracker.ends = []
    return lines, ends

def accepted(lines, ends):
    """ACT's verdict, as dipy applies it with return_all=False: both ends in gray matter."""
    ok = (StreamlineStatus.ENDPOINT, StreamlineStatus.OUTSIDEIMAGE)
    return np.array([len(e) == 2 and all(x in ok for x in e) and 2 <= len(s) <= 500 for s, e in zip(lines, ends)])

stopping = ActStoppingCriterion(gm_frac, csf_frac)
seeds = seeds_from_mask(wm_frac > 0.5, affine, density=1)
prob_getter = ProbabilisticDirectionGetter.from_shcoeff(csd_fit.shm_coeff, max_angle=30.0, sphere=sphere)
all_det, det_ends = track(pk, max_cross=1)
all_prob, prob_ends = track(prob_getter)
det_ok, prob_ok = accepted(all_det, det_ends), accepted(all_prob, prob_ends)
det = Streamlines([s for s, k in zip(all_det, det_ok) if k])
prob = Streamlines([s for s, k in zip(all_prob, prob_ok) if k])
print(f"seeds {len(seeds)}; accepted streamlines: deterministic {len(det)}, probabilistic {len(prob)}")
```

Only the streamlines ACT accepts, those that end in gray matter at both ends without
crossing CSF, are kept; the rest are rejected as anatomically implausible.

Drawing all of them at once, projected through the whole volume, gives a tangle in which
nothing can be followed. The figure below instead shows a thin axial slab, 3 voxels (9 mm)
thick through the widest part of the white matter, with only the parts of each accepted
streamline that lie inside it, over the white matter fraction of the middle slice (white
matter light, everything else dark; anterior at the top). Each short segment is colored by
its own direction, with the usual convention: red for left-right, green for
anterior-posterior, blue for superior-inferior (there is almost no blue here, because the
synthetic fibers lie in the axial plane). Expect the probabilistic streamlines to spread
more and reach more of the white matter, since each step can depart from the peak.

```{code-cell} python
:tags: [hide-input]
Z0 = int(np.argmax((wm_frac > 0.5).sum(axis=(0, 1))))  # the axial slice with the most white matter

def slab_segments(streamlines, z0=Z0, half=1.0):
    """The segments of every streamline inside an axial slab, each colored by its direction."""
    segs, cols = [], []
    for s in streamlines:
        inside = np.abs(s[:, 2] - z0) <= half
        keep = inside[1:] & inside[:-1]
        if not keep.any():
            continue
        a, b = s[:-1][keep], s[1:][keep]
        d = np.abs(b - a); d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        segs.append(np.stack([a[:, [1, 0]], b[:, [1, 0]]], 1))
        cols.append(d[:, [1, 0, 2]])  # RGB = (left-right, anterior-posterior, superior-inferior)
    return np.concatenate(segs), np.clip(np.concatenate(cols), 0, 1)

rows = np.flatnonzero(wm_frac[:, :, Z0].max(axis=1) > 0.5)
cols_ = np.flatnonzero(wm_frac[:, :, Z0].max(axis=0) > 0.5)
fig, axes = plt.subplots(1, 2, figsize=(9, 5.4))
for ax, (name, sl) in zip(axes, [("deterministic", det), ("probabilistic", prob)]):
    segs, cols = slab_segments(sl)
    ax.imshow(wm_frac[:, :, Z0], cmap="gray", vmin=0, vmax=2.5)
    ax.add_collection(mpl.collections.LineCollection(segs, colors=cols, lw=0.5, alpha=0.8))
    ax.set(title=name, xticks=[], yticks=[], xlim=(cols_[0] - 2, cols_[-1] + 2), ylim=(rows[-1] + 2, rows[0] - 2))
    ax.grid(False)
fig.tight_layout()

wm_vox = wm_frac > 0.5
for name, sl in [("deterministic", det), ("probabilistic", prob)]:
    visited = np.zeros(wm_frac.shape, bool)
    for s in sl:
        v = np.round(s).astype(int)
        visited[v[:, 0], v[:, 1], v[:, 2]] = True
    print(f"{name:>14}: accepted streamlines pass through {(visited & wm_vox).sum() / wm_vox.sum():.0%} of white matter voxels")
```

In the deterministic panel the streamlines crowd into a few smooth, tightly packed paths:
long anterior-posterior (green) bundles along the edges of the thick white matter beside
the ventricles, turning left-right (red) at their front and back ends, with the middle of
those thick regions left almost empty. Streamlines seeded in neighboring voxels merge onto
the same paths because they follow the same peaks. The probabilistic streamlines are
rougher and spread across the thick white matter instead of crowding at its edges. Over
the whole volume, the accepted deterministic streamlines pass through 76% of the white
matter voxels and the probabilistic ones through 90%.

## Measure it: agreement with the orientation field

Two scores. The angle between each streamline segment and the true orientation at that
position measures how faithfully the tracker followed the fibers; the fraction of
streamlines ACT rejects, and why, measures their anatomical plausibility.

```{code-cell} python
:tags: [hide-input]
def segment_angle_error(streamlines, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    errs = []
    for k in rng.choice(len(streamlines), size=min(n, len(streamlines)), replace=False):
        s = streamlines[k]
        if len(s) < 3:
            continue
        d = s[1:] - s[:-1]
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        v = np.clip(np.round(s[:-1]).astype(int), 0, np.array(mask.shape) - 1)
        tr = truth[v[:, 0], v[:, 1], v[:, 2]]
        cos = np.abs(np.sum(d * tr, axis=1))
        errs.append(np.degrees(np.arccos(np.clip(cos, 0, 1))))
    return np.concatenate(errs)

fig, ax = plt.subplots(figsize=(7, 3.2))
bins = np.arange(0, 91, 2)
for (name, s, s_all, ends, ok), col in zip([("deterministic", det, all_det, det_ends, det_ok),
                                            ("probabilistic", prob, all_prob, prob_ends, prob_ok)], (PALETTE[6], PALETTE[3])):
    e = segment_angle_error(s)
    csf = np.array([StreamlineStatus.INVALIDPOINT in x for x in ends])
    in_wm = np.array([StreamlineStatus.TRACKPOINT in x for x in ends]) & ~csf
    print(f"{name:>14}: segment angle error median {np.median(e):4.1f}°, 90th percentile {np.percentile(e, 90):4.1f}°; "
          f"accepted by ACT {ok.sum()} of {len(s_all)} ({ok.mean():.0%}); rejected: an end entered CSF {csf.sum()}, "
          f"an end stopped inside white matter {in_wm.sum()}, other {(~ok & ~csf & ~in_wm).sum()}")
    ax.hist(e, bins=bins, density=True, histtype="step", lw=1.6, color=col, label=f"{name} (median {np.median(e):.1f}°)")
    ax.axvline(np.median(e), color=col, lw=0.8, ls="--")
ax.set(xlabel="angle between streamline step and true fiber orientation (°)", ylabel="fraction of steps per degree", xlim=(0, 90))
ax.legend()
fig.tight_layout()
```

The histogram shows the angle error of every step. The deterministic tracker's steps stay
within a few degrees of the true orientation (median 3.8°, 90th percentile 11.8°). The
probabilistic tracker's are spread much more widely (median 18.2°, 90th percentile 33.4°):
each step is a draw from the width of the ODF lobe, so the streamline zigzags around the
fiber direction even where, on average, it follows the bundle.

ACT accepts 60% of the deterministic streamlines and 65% of the probabilistic ones. The
breakdown of the rejections shows why the probabilistic tracker comes out ahead here:
slightly more of its streamlines enter CSF (2925 against 2816), but far fewer stop inside
white matter (3268 against 4412). A streamline stops inside white matter when the tracker
finds no acceptable direction, because no peak lies within its turning limit (panel c of
the animation) or the peak is too weak to count. The remaining few hundred ("other") never
got past their seed or ran past the length cap.

The synthetic volume has one fiber population per voxel and no crossings, so these numbers
say nothing about how either tracker handles crossings. That is where probabilistic
tracking matters most in real data: its random draws let some streamlines follow a fanning
branch or the other lobe of a crossing, which a deterministic tracker never does. The next
section shows, in a toy, what passing through a crossing can get wrong.

The rejection rate is a useful summary of a dataset: it rises with noise, with poor
orientation estimates, and with misregistration between the diffusion data and the tissue
segmentation ([Chapter 10](../03-preprocessing/10-susceptibility-distortion.md)).

## How the acquisition bounds tractography

Tractography compounds every earlier choice. Angular precision of the peaks ([Chapter 16](./16-fiber-orientation.md))
sets how far a streamline drifts per step; voxel size sets which crossings are resolved at
all and how well the segmentation aligns; distortion and motion residuals (Part III)
misplace the fibers relative to the anatomy that ACT uses to accept them. Better data do
not remove every error, though. In the tractography challenge of {cite:t}`maierhein2017`,
run on simulated data with a known answer, most submissions found most of the true
bundles, but they also produced more invalid bundles than valid ones: plausible-looking
streamline bundles that do not exist. The authors traced this to an ambiguity inherent in
inferring long-range connections from local orientations, which no method or acquisition
removes on its own.

One way such false bundles arise needs no noise at all. In the figure below, two bundles
meet at a shallow angle. In anatomy (a) they cross: the purple fibers run from upper left
to lower right. In anatomy (b) they kiss: they touch in the middle and turn back, so the
purple fibers run from upper left to upper right. Every voxel holds the same fiber
orientations in both anatomies, one in each arm and two in the central diamond (the
sticks in panel c), so the diffusion signal cannot tell them apart. Seeded on the left
arms and taking the peak nearest its heading, a tracker goes straight through the middle
(panel c). That is right for (a) and wrong for (b), where it reports an upper-left to
lower-right connection that does not exist, and nothing about those streamlines looks
wrong: they are smooth, and they stay in white matter.

```{code-cell} python
:tags: [hide-input]
# Two anatomies with the same voxel orientations: bundles that cross, and bundles that kiss.
KW, KH, HALF_WIDTH = 28, 18, 1.6                          # grid (voxels) and half-width of each bundle
theta = np.radians(33)
cx, cy = KW / 2, KH / 2
a_dir = np.array([np.cos(theta), -np.sin(theta)])       # bundle A: top-left to bottom-right
b_dir = np.array([np.cos(theta), np.sin(theta)])        # bundle B: bottom-left to top-right

def band(direction, x, y):
    normal = np.array([-direction[1], direction[0]])
    return np.abs((x - cx) * normal[0] + (y - cy) * normal[1]) < HALF_WIDTH

ky, kx = np.mgrid[0:KH, 0:KW] + 0.5
in_a, in_b = band(a_dir, kx, ky), band(b_dir, kx, ky)   # the voxel content is the same in both anatomies
tt = np.linspace(-KW, KW, 400)
offsets = np.linspace(-HALF_WIDTH, HALF_WIDTH, 5)[1:-1]

def fiber(direction, o):
    normal = np.array([-direction[1], direction[0]])
    return np.array([cx, cy]) + tt[:, None] * direction + o * normal

def peak_track(seed, heading, limit_deg=45, n_max=200):
    """Deterministic tracking on the voxel peaks: take the peak closest to the current heading."""
    pts, d = [np.array(seed, float)], np.array(heading, float)
    for _ in range(n_max):
        i, j = int(pts[-1][1]), int(pts[-1][0])
        if not (0 <= i < KH and 0 <= j < KW) or not (in_a[i, j] or in_b[i, j]):
            break
        peaks = [p for p, inside in ((a_dir, in_a[i, j]), (b_dir, in_b[i, j])) if inside]
        peaks = [p if p @ d >= 0 else -p for p in peaks]
        best = max(peaks, key=lambda p: p @ d)
        if best @ d < np.cos(np.radians(limit_deg)):
            break
        d = best
        pts.append(pts[-1] + 0.5 * d)
    return np.array(pts)

COL_A, COL_B = PALETTE[6], PALETTE[3]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
for ax in axes:
    ax.set(xlim=(0, KW), ylim=(0, KH), xticks=[], yticks=[], aspect="equal")
    ax.grid(False)
    ax.imshow(np.where((in_a | in_b)[..., None], (0.88, 0.92, 0.98), (1, 1, 1)), origin="lower", extent=(0, KW, 0, KH))
for o in offsets:                                        # crossing: each fiber runs straight through
    for direction, col in ((a_dir, COL_A), (b_dir, COL_B)):
        f = fiber(direction, o)
        axes[0].plot(f[:, 0], f[:, 1], color=col, lw=1.6)
    for direction, col in ((a_dir, COL_A), (b_dir, COL_B)):   # kissing: each fiber bounces back at the midline
        f = fiber(direction, o)
        left = f[f[:, 0] <= cx]
        mirrored = np.c_[2 * cx - left[::-1, 0], left[::-1, 1]]
        f = np.r_[left, mirrored]
        axes[1].plot(f[:, 0], f[:, 1], color=col, lw=1.6)
sticks = []
for x, y, ia, ib in zip(kx.ravel(), ky.ravel(), in_a.ravel(), in_b.ravel()):
    for p, inside in ((a_dir, ia), (b_dir, ib)):
        if inside:
            sticks.append([(x - 0.4 * p[0], y - 0.4 * p[1]), (x + 0.4 * p[0], y + 0.4 * p[1])])
axes[2].add_collection(mpl.collections.LineCollection(sticks, colors=INK["secondary"], lw=0.7))
for o in offsets:
    for direction, col in ((a_dir, COL_A), (b_dir, COL_B)):
        start = fiber(direction, o)[np.argmin(np.abs(tt + 11))]  # a seed on the left arm
        s = peak_track(start, direction)
        axes[2].plot(s[:, 0], s[:, 1], color=col, lw=1.6)
        axes[2].plot(*start, marker="o", ms=3.5, mfc="white", mec=INK["primary"], mew=0.8)
axes[0].set_title("a  anatomy 1: the bundles cross", loc="left", fontsize=10)
axes[1].set_title("b  anatomy 2: the bundles kiss", loc="left", fontsize=10)
axes[2].set_title("c  the voxels (same for a and b), tracked", loc="left", fontsize=10)
fig.tight_layout()
```

The evaluation on the simulated datasets, whose tractogram generated the data, is the
direct test of how often this happens.

## From streamlines to a connectome

A **structural connectome** is a table with one row and one column per gray matter region
(from a parcellation of the anatomical image) and, in each cell, a number for the
connection between those two regions, built by assigning each streamline's two ends to
regions. Every step of that construction adds a choice that changes the answer {cite:p}`yeh2021`:

- **Edge weights.** The raw cell value is a streamline count, and streamline counts are
  not fiber counts: they depend on the seeding, the step size, the curvature limit, and
  the number of streamlines generated. Common alternatives divide by region size or by
  streamline length, weight by SIFT2 so that counts track the fiber density the ODFs
  imply, or store the mean FA along the streamlines instead. Each answers a different
  question, and they are not interchangeable across studies.
- **Length and seeding biases.** Long connections are harder to track, since every step
  is another chance to stop or turn off, so they are under-represented. Seeding from white
  matter favors long bundles, which pass more seeds; seeding from the gray-white interface
  favors large regions. Large regions also collect more streamline ends simply by being
  large.
- **Thresholding.** Every tractogram contains false connections (the kissing figure above,
  and the invalid bundles of the challenge), so connectomes are usually thresholded to keep
  only strong or consistent edges. A strict threshold removes false positives and true
  weak connections with them; a lenient one keeps both. There is no setting that avoids
  both errors, and the network measures computed afterward (hubs, path lengths,
  modularity) change with the threshold chosen.

Report the parcellation, the tracking parameters, the number of streamlines, the edge
weight, and the threshold, and compare connectomes only when all of them match.

## Measure it: the simulated datasets

:::{admonition} Simulated dataset pending
:class: note
This section will track on the `ref-schemes` dataset and evaluate the result against the
tractogram that generated it: bundle overlap and overreach, and the fraction of valid
connections, for the 30-direction, 64-direction, and multi-shell schemes, with renders from
the pipeline's TRXViz output. It will also compare SIFT2-weighted streamline density with
the simulated brain's true fiber density.
:::

## What this implies for acquisition

- **Tractography needs orientation estimates, so it needs the acquisition of [Chapter 16](./16-fiber-orientation.md)**:
  b ≥ 2000 and 45 or more directions, multi-shell preferred.
- **Resolution decides which crossings exist in the data at all.**
- **The tissue segmentation is part of the tractography input**; distortion correction and
  registration to the anatomical image must be right for ACT to work.
- **Streamline counts are not fiber counts**; use SIFT2 or an equivalent and report the
  filtering, and for connectomes report the edge weight and the threshold.

## Further reading

The first streamline tractography {cite:p}`mori1999,conturo1999` and an early in vivo
tensor implementation {cite:p}`basser2000`, anatomical constraints {cite:p}`smith2012`,
SIFT {cite:p}`smith2013` and SIFT2 {cite:p}`smith2015`, the tractography challenge
{cite:p}`maierhein2017`, and the review by {cite:t}`jeurissen2019`.
