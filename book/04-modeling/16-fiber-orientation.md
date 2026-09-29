---
title: "16. Fiber orientation estimation"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** a synthetic series with a fiber crossing built from the packaged tissue maps ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
- **`ref-clean`** (pending): the artifact-free, noise-free reference series with its truth maps and true fiber orientations ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-ref-clean)).
- **`ref-schemes`** (pending): the simulated brain under the 30-direction, 64-direction, HBCD, DSI, and CS-DSI schemes at matched scan time ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-ref-schemes)).
- **`truth`** (pending): the 27 analytic ground-truth maps and the true fiber orientations ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-truth), [Appendix E](../appendices/e-truth-map-catalogue.md)).

Pipeline-tier datasets are simulated offline by TRXScan ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)) and are marked *pending* until their release; the figures that need them say so where they will appear.
:::

## Learning goals

After this chapter you can:

- explain why the tensor fails where fibers cross and what an orientation distribution
  function adds
- fit q-ball, spherical deconvolution (single- and multi-tissue), and DSI reconstructions
  and extract peaks
- score peak directions and peak counts against a known answer
- state what each method needs from the sampling scheme

```{code-cell} python
:tags: [hide-cell]
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import warnings
warnings.filterwarnings("ignore")  # dipy's fit notices are not part of the lesson
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.optimize import nnls
from dipy.core.gradients import gradient_table
from dipy.core.sphere import Sphere
from dipy.data import get_sphere
from dipy.direction import peak_directions, peaks_from_model
from dipy.reconst.csdeconv import AxSymShResponse, ConstrainedSphericalDeconvModel, response_from_mask_ssst
from dipy.reconst.dsi import DiffusionSpectrumModel
from dipy.reconst.dti import TensorModel
from dipy.reconst.mcsd import MultiShellDeconvModel, multi_shell_fiber_response, response_from_mask_msmt
from dipy.reconst.shm import CsaOdfModel, real_sh_descoteaux_from_index

from dwibook import phantoms, schemes, signal, synth
from dwibook.plotting import INK, PALETTE, animate, set_style, show_image

set_style()
sphere = get_sphere(name="repulsion724")
```

## The crossing-fiber problem

The tensor has one principal direction per voxel. Most white matter voxels at 2 mm contain
fibers in more than one orientation {cite:p}`jeurissen2013`: a bundle passing through
another, or a bundle fanning into the cortex. In such a voxel the tensor's long axis points
somewhere between the true orientations, its anisotropy is reduced (two crossing bundles
look like a flat disk), and tractography that follows the tensor stops or turns onto the
wrong bundle. Every method in this chapter exists to return more than one direction per
voxel.

The figure below simulates three single voxels with the book's white matter signal model
(an intra-axonal stick plus an extra-axonal tensor, [Chapter 4](../02-diffusion-encoding/04-diffusion-in-tissue.md)): one fiber population, and two
equal populations crossing at 60° and at 90°, all lying in the plane of the page. Each voxel
is measured noise-free with 30 directions at b = 1000 and 60 at b = 3000. The middle row is
the tensor fitted to the b = 1000 shell, drawn as its cross-section in the page plane (each
axis proportional to the diffusivity along it). The bottom row is the result of
constrained spherical deconvolution (introduced below) on the b = 3000 shell, drawn as a
polar plot: the distance from the center in each direction is the estimated amount of
fiber pointing that way.

```{code-cell} python
:tags: [hide-input]
# polar glyphs: the ODF evaluated on a ring of 360 directions in the plane of the page
RING = np.deg2rad(np.arange(361))
ring_xy = np.c_[np.cos(RING[:-1]), np.sin(RING[:-1])]
ring_sphere = Sphere(xyz=np.c_[ring_xy, np.zeros(360)])

def fiber_response(b, sh_order_max):
    """The single-fiber signal of signal.white_matter at one b-value (fiber along z), as the
    zonal spherical-harmonic coefficients that dipy's CSD takes as its response function."""
    ls = np.arange(0, sh_order_max + 1, 2)
    B = real_sh_descoteaux_from_index(np.zeros_like(ls), ls, sphere.theta[:, None], sphere.phi[:, None])
    return AxSymShResponse(1.0, np.linalg.lstsq(B, signal.white_matter(b, sphere.z), rcond=None)[0])

def in_plane(*angles_deg):
    a = np.deg2rad(angles_deg)
    return np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], axis=-1)

def voxel_signal(bvals_, bvecs_, dirs):
    """Equal-fraction mixture of white matter fiber populations along ``dirs`` (S0 = 1)."""
    return np.mean([signal.white_matter(bvals_, bvecs_ @ d) for d in dirs], axis=0)

def glyph(ax, amp, dirs=(), color=PALETTE[0], title=None):
    """Polar glyph of ``amp`` (one value per ring direction), scaled so its maximum is 1;
    ``dirs`` are the true fiber directions (2-D, in plot coordinates), drawn dashed."""
    r = np.clip(amp, 0, None) / amp.max()
    x, y = np.append(r * ring_xy[:, 0], r[0]), np.append(r * ring_xy[:, 1], 0)
    ax.fill(x, y, color=color, alpha=0.3, lw=0)
    ax.plot(x, y, color=color, lw=1.5)
    for d in dirs:
        ax.plot([-d[0], d[0]], [-d[1], d[1]], color=INK["primary"], lw=0.9, ls="--")
    ax.set(xlim=(-1.15, 1.15), ylim=(-1.15, 1.15), aspect="equal")
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=9)

bv_x, bg_x = schemes.multi_shell({1000: 30, 3000: 60}, n_b0=2)
lo, hi = bv_x <= 1000, (bv_x == 0) | (bv_x == 3000)
g_lo, g_hi = gradient_table(bv_x[lo], bvecs=bg_x[lo]), gradient_table(bv_x[hi], bvecs=bg_x[hi])
csd_hi = ConstrainedSphericalDeconvModel(g_hi, fiber_response(3000, 8), sh_order_max=8)
cases = {"one fiber": [0], "two fibers at 60°": [-30, 30], "two fibers at 90°": [-45, 45]}
fig, axes = plt.subplots(3, 3, figsize=(7.5, 7.8))
for col, (name, angs) in enumerate(cases.items()):
    dirs = in_plane(*angs)
    s = voxel_signal(bv_x, bg_x, dirs)
    ax = axes[0, col]
    for d in dirs:
        ax.plot([-d[0], d[0]], [-d[1], d[1]], color=PALETTE[1], lw=5, solid_capstyle="round")
    ax.set(xlim=(-1.15, 1.15), ylim=(-1.15, 1.15), aspect="equal", title=f"true fibers: {name}")
    ax.set_axis_off()
    ten = TensorModel(g_lo).fit(s[lo])
    lam, vec = np.linalg.eigh(ten.quadratic_form[:2, :2])  # the tensor's cross-section in the page plane
    t_ = np.linspace(0, 2 * np.pi, 200)
    ell = vec @ (lam[:, None] / ten.evals.max() * np.array([np.cos(t_), np.sin(t_)]))
    ax = axes[1, col]
    ax.fill(ell[0], ell[1], color=PALETTE[0], alpha=0.3, lw=0)
    ax.plot(ell[0], ell[1], color=PALETTE[0], lw=1.5)
    for d in dirs:
        ax.plot([-d[0], d[0]], [-d[1], d[1]], color=INK["primary"], lw=0.9, ls="--")
    ax.set(xlim=(-1.15, 1.15), ylim=(-1.15, 1.15), aspect="equal", title=f"tensor, FA = {ten.fa:.2f}")
    ax.set_axis_off()
    glyph(axes[2, col], csd_hi.fit(s[hi]).odf(ring_sphere), dirs[:, :2], color=PALETTE[2], title="CSD fiber ODF")
    print(f"{name:>18}: tensor FA {ten.fa:.2f}, eigenvalues {', '.join(f'{v * 1e3:.2f}' for v in ten.evals)} x 10^-3 mm²/s")
fig.tight_layout()
```

What to look at: the tensor's shape (middle row) against the true fibers (dashed lines). For
one fiber the tensor is a long ellipse along the fiber. For the 60° crossing it is a
shorter ellipse along the bisector of the two fibers, pointing along neither, and its FA has
dropped. For the 90° crossing it is a circle in the page plane (the two largest
eigenvalues are equal): the tensor is a flat disk, its principal direction is arbitrary, and
it says nothing about either fiber. Its FA is still 0.48, because the disk is thin in the
direction out of the page, so a moderate FA does not tell a crossing from a loosely
organized single bundle. The deconvolution in the bottom row has one lobe per fiber
in all three voxels.

The methods estimate an **orientation distribution function** (ODF): a function on the
sphere that gives, for every direction, how much of the voxel's diffusion or fiber content
points that way. Its peaks are the fiber directions. Two definitions are in use. The
**diffusion ODF** is the angular profile of the water displacements, which peaks along fibers
but broadly; q-ball imaging and DSI estimate it. The **fiber ODF** is the distribution of
fiber orientations itself. It is obtained by **deconvolution**, undoing a known blur: the
signal of a single coherent bundle (the **response function**) is removed from the measured
signal; spherical deconvolution estimates it, with sharper peaks and better separation of
crossings.

The idea is easiest to see in one dimension, on a circle of orientations (0° to 180°, where
0° and 180° are the same orientation) instead of a sphere. In the sketch below each fiber
population contributes the same broad bump centered on its orientation, so the measured
profile is the sum of one bump per fiber: mathematically, the fiber spikes *convolved* with
the bump. Deconvolution asks which spikes, blurred by the known bump, reproduce the measured
profile. Here it is solved by non-negative least squares on noise-free profiles.

```{code-cell} python
:tags: [hide-input]
phi = np.arange(180.0)                                     # orientation, degrees (period 180)
def bump(center, width=25.0):
    d = (phi - center + 90) % 180 - 90                     # circular distance in degrees
    return np.exp(-0.5 * (d / width) ** 2)
K = np.stack([bump(c) for c in phi], axis=1)               # column c: the bump of a fiber at c
fig, axes = plt.subplots(2, 4, figsize=(12, 5), sharex=True)
for row, (a1, a2) in enumerate([(45, 135), (70, 110)]):
    spikes = np.zeros(180); spikes[[a1, a2]] = 0.5
    measured = K @ spikes                                  # the circular convolution
    recovered, _ = nnls(K, measured)
    axes[row, 0].plot(phi, bump(90), color=PALETTE[0])
    axes[row, 1].vlines([a1, a2], 0, 0.5, color=PALETTE[1], lw=3)
    axes[row, 2].plot(phi, measured, color=PALETTE[0])
    axes[row, 3].vlines(phi[recovered > 1e-3], 0, recovered[recovered > 1e-3], color=PALETTE[2], lw=3)
    for a in (a1, a2):  # the true orientations, on the measured profile
        axes[row, 2].axvline(a, color=INK["secondary"], lw=0.8, ls="--")
    axes[row, 0].set_ylabel(f"crossing {a2 - a1}°", fontsize=10)
    for col, ax in enumerate(axes[row]):
        ax.set(xlim=(0, 180), xticks=[0, 45, 90, 135, 180], ylim=(0, None) if col == 2 else (0, 1.08))
for ax, title in zip(axes[0], ["A. response: the profile\nof one fiber", "B. true fibers (spikes)",
                              "C. measured profile\n= B blurred by A", "D. deconvolution of C by A"]):
    ax.set_title(title, fontsize=10)
for ax in axes[1]:
    ax.set_xlabel("orientation (degrees)")
fig.tight_layout()
```

What to look at: panel C, where the dashed lines mark the true orientations. With the fibers
90° apart (top) the measured profile has two humps at the right places; with them 40° apart (bottom) the two bumps overlap into a single
hump centered between the fibers, and a peak search on it returns one direction. Panel D
recovers both spikes in either case, because the shape of the blur (panel A) is known. In
the language of this chapter: the **diffusion ODF** is the blurry measured shape (C), the
**fiber ODF** is the sharp underlying one (B, and its estimate D), and the **response
function** is the blur of a single fiber (A). On the sphere the same operation is spherical
deconvolution. With real data the deconvolution cannot be exact, since noise and a finite
number of directions limit how sharp the answer can be, which is why the constraint that
fiber amounts cannot be negative matters in practice.

## The synthetic crossing

The toy tier adds a second fiber population to the 2 mm slice: inside a band of rows
through the middle of the brain, half of the white matter signal comes from a second fiber
population. The first population runs along the white matter boundary, as everywhere else
in the slice; the second lies in the slice plane at 60° to the first, so that every white matter voxel
of the band holds a 60° crossing, an angle at which the methods start to differ (a 90°
crossing is resolved by all of them). The two orientation fields are the answer key. The scheme has three shells with 30
directions each (b = 1000, 2000, 3000) plus four b=0, and noise is added at SNR 25.

```{code-cell} python
:tags: [hide-input]
t = phantoms.brain_slice()
mask = t["mask"]
wm = t["wm"] > 0.9
bvals, bvecs = schemes.multi_shell({1000: 30, 2000: 30, 3000: 30}, n_b0=4)
region = synth.crossing_region(mask.shape)
CROSSING_DEG = 60
clean, truth1, truth2 = synth.synthetic_dwi_crossing(t, bvals, bvecs, region, second=CROSSING_DEG, fraction=0.5)
sigma = clean[wm][:, bvals == 0].mean() / 25
noisy = synth.add_noise(clean, sigma, seed=0)
# dipy's reconstruction models expect a 3-D image: give the slice a singleton third axis
data = noisy[:, :, None, :]
mask3 = mask[:, :, None]
gtab = gradient_table(bvals, bvecs=bvecs)
crossing = region & wm
single = ~region & wm
print(f"white matter voxels with one fiber population: {single.sum()}, with two: {crossing.sum()}")
```

## The methods

| Method | What it estimates | What it needs | Intuition |
|---|---|---|---|
| **Q-ball imaging** {cite:p}`tuch2004`; constant-solid-angle (CSA) variant {cite:p}`aganj2010` | diffusion ODF | one shell, b ≥ 2000, 60 or more directions | reads the blurry profile straight off one shell; peaks are broad |
| **Diffusion spectrum imaging** (DSI) {cite:p}`wedeen2005` | diffusion ODF, from the full displacement distribution | a Cartesian q-space grid ([Chapter 6](../02-diffusion-encoding/06-qspace-sampling.md)), hundreds of volumes to b ≈ 4000 | measures the displacements in every direction and assumes no tissue model, at the cost of a long scan |
| **Constrained spherical deconvolution** (CSD) {cite:p}`tournier2007` | fiber ODF | one shell, b ≥ 2000, 45 or more directions; a response function from single-fiber voxels | removes the known blur of one fiber; forbidding negative fiber amounts suppresses spurious peaks |
| **Multi-shell multi-tissue CSD** {cite:p}`jeurissen2014` | fiber ODF, plus gray matter and CSF amounts | two or more shells; white matter, gray matter, and CSF masks for the three responses | the tissues decay differently across shells, so gray matter and CSF are not mistaken for fiber |

Each model returns an ODF per voxel; the same peak extraction is applied to all: local
maxima of the ODF on a sphere of 724 directions, separated by at least 25°, above a
fraction of the largest peak.

```{code-cell} python
:tags: [hide-input]
results, recon = {}, {}  # peaks per method; the model and data, kept for the single-voxel glyphs below

def peaks(name, model, data_, gtab_, rel=0.25, m=None):
    recon[name] = (model, data_)
    results[name] = peaks_from_model(model, data_, sphere, relative_peak_threshold=rel, min_separation_angle=25, mask=mask3 if m is None else m, npeaks=3)

# q-ball (CSA) on the b = 2000 shell
keep = (bvals == 0) | (bvals == 2000)
g2 = gradient_table(bvals[keep], bvecs=bvecs[keep])
peaks("q-ball (CSA), b = 2000", CsaOdfModel(g2, sh_order_max=6), data[..., keep], g2, rel=0.5)
# single-shell CSD on the b = 3000 shell; the response function comes from single-fiber
# white matter voxels (in practice selected by high FA, here known from the answer key)
single_fiber = (wm & ~region)[:, :, None]
keep = (bvals == 0) | (bvals == 3000)
g3 = gradient_table(bvals[keep], bvecs=bvecs[keep])
response, ratio = response_from_mask_ssst(g3, data[..., keep], single_fiber)
peaks("CSD, b = 3000", ConstrainedSphericalDeconvModel(g3, response, sh_order_max=6), data[..., keep], g3)
# single-shell CSD on the b = 1000 shell (for comparison)
keep = (bvals == 0) | (bvals == 1000)
g1 = gradient_table(bvals[keep], bvecs=bvecs[keep])
response1, _ = response_from_mask_ssst(g1, data[..., keep], single_fiber)
peaks("CSD, b = 1000", ConstrainedSphericalDeconvModel(g1, response1, sh_order_max=6), data[..., keep], g1)
# multi-shell multi-tissue CSD on all shells, responses from the tissue fractions
resp_wm, resp_gm, resp_csf = response_from_mask_msmt(gtab, data, (t["wm"] > 0.9)[:, :, None] & ~region[:, :, None], (t["gm"] > 0.9)[:, :, None], (t["csf"] > 0.9)[:, :, None])
msmt_response = multi_shell_fiber_response(sh_order_max=6, bvals=np.unique(np.round(bvals / 50) * 50), wm_rf=resp_wm, gm_rf=resp_gm, csf_rf=resp_csf)
# the multi-tissue fit solves a convex problem per voxel and is the slowest method here; white matter only
peaks("multi-tissue CSD, 3 shells", MultiShellDeconvModel(gtab, msmt_response, sh_order_max=6), data, gtab, m=wm[:, :, None])
```

DSI needs its own acquisition, the 257-point grid of [Chapter 6](../02-diffusion-encoding/06-qspace-sampling.md). The toy tier simulates it
on the same slice (257 volumes, same noise level) and reconstructs it with dipy's DSI model.

```{code-cell} python
:tags: [hide-input]
b_dsi, v_dsi = schemes.dsi_grid(radius=4, b_max=4000, n_b0=1)
clean_dsi, _, _ = synth.synthetic_dwi_crossing(t, b_dsi, v_dsi, region, second=CROSSING_DEG, fraction=0.5)
noisy_dsi = synth.add_noise(clean_dsi, sigma, seed=1)[:, :, None, :]
g_dsi = gradient_table(b_dsi, bvecs=v_dsi, big_delta=0.030, small_delta=0.010)
# the DSI reconstruction is the slowest of the five; white matter voxels only
peaks("DSI, 257 points", DiffusionSpectrumModel(g_dsi, qgrid_size=17, filter_width=18), noisy_dsi, g_dsi, rel=0.5, m=wm[:, :, None])
```

Before scoring the whole band, look at one voxel of it. The figure draws the ODF each method
estimates in a single white matter voxel of the crossing band (the noisy data, in the page
plane, oriented as in the brain images), with the tensor fitted to the b = 1000 shell added
for reference. The dashed lines are the two true fiber directions, 60° apart; each title
gives the number of peaks the extraction found in this voxel.

```{code-cell} python
:tags: [hide-input]
cand = np.argwhere(crossing)
vi, vj = cand[np.argmin(np.hypot(cand[:, 0] - 64, cand[:, 1] - 64))]  # the band voxel nearest the slice center
ring_img = Sphere(xyz=np.c_[-ring_xy[:, 1], ring_xy[:, 0], np.zeros(360)])  # ring in (row, col), drawn as on screen
true_xy = [(d[1], -d[0]) for d in (truth1[vi, vj], truth2[vi, vj])]
keep = bvals <= 1000
odfs = {"tensor, b = 1000": TensorModel(gradient_table(bvals[keep], bvecs=bvecs[keep])).fit(data[vi, vj, 0, keep]).odf(ring_img)}
for name, (model, data_) in recon.items():
    odfs[name] = model.fit(data_[vi, vj, 0]).odf(ring_img)
fig, axes = plt.subplots(2, 3, figsize=(8.4, 6.0))
for k, (ax, (name, odf)) in enumerate(zip(axes.flat, odfs.items())):
    n = np.count_nonzero(results[name].peak_values[vi, vj, 0] > 0) if name in results else 1
    glyph(ax, odf, true_xy, color=PALETTE[0] if k in (0, 1, 5) else PALETTE[2],
          title=f"{name}\n{n} peak{'s' if n > 1 else ''} found")
    print(f"{name:>28}: {n} peak(s); ODF minimum / maximum on the ring = {max(odf.min(), 0) / odf.max():.2f}")
fig.tight_layout()
```

What to look at: the width of the lobes around the dashed lines. The tensor's ODF (drawn the
same way) is a single lobe between the two fibers, along neither. The diffusion ODFs of
q-ball and DSI (blue) have two lobes, but broad ones that run into each other across the
space between the fibers, and the DSI shape sits on a rounded base. The fiber ODFs of the
three deconvolution fits (green) are empty between the fibers, and at b = 3000 the lobes are
narrow and lie close to the dashed lines; at b = 1000 they are misplaced. The ratio printed
after each method, the ODF's smallest value over its largest on this ring, measures that
base: 0 for a shape that is empty between its lobes, larger for a rounder one.

## Measure it: peaks against the answer key

Two scores per method: in single-fiber white matter, the angle between the strongest peak
and the true orientation and how often a second (false) peak appears; in the crossing
region, how often both fibers are found and the angle of the better-matched peak to each.

```{code-cell} python
:tags: [hide-input]
def angle(a, b):
    cos = np.abs(np.sum(a * b, axis=-1)) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)
    return np.degrees(np.arccos(np.clip(cos, 0, 1)))

def score(pk):
    dirs = pk.peak_dirs[:, :, 0]            # (ny, nx, 3 peaks, 3)
    vals = pk.peak_values[:, :, 0]
    n_peaks = np.count_nonzero(vals > 0, axis=-1)
    err_single = angle(dirs[..., 0, :], truth1)[single]
    false_peaks = (n_peaks[single] >= 2).mean()
    both = (n_peaks[crossing] >= 2).mean()
    # in the crossing: best match of any detected peak to each true fiber
    e1 = np.min(np.where(vals[crossing] > 0, angle(dirs[crossing], truth1[crossing][:, None, :]), 90), axis=-1)
    e2 = np.min(np.where(vals[crossing] > 0, angle(dirs[crossing], truth2[crossing][:, None, :]), 90), axis=-1)
    return err_single.mean(), false_peaks, both, 0.5 * (e1.mean() + e2.mean())

print(f"{'method':>28}   single-fiber WM: angle error, false 2nd peak   crossing: both found, angle error")
for name, pk in results.items():
    e, fp, both, ec = score(pk)
    print(f"{name:>28}:            {e:5.1f}°          {fp:4.0%}                  {both:4.0%}        {ec:5.1f}°")
```

```{code-cell} python
:tags: [hide-input]
count_colors = plt.get_cmap("viridis")(np.linspace(0, 1, 4))
fig, axes = plt.subplots(1, 5, figsize=(14, 3.6))
for ax, (name, pk) in zip(axes, results.items()):
    n_peaks = np.count_nonzero(pk.peak_values[:, :, 0] > 0, axis=-1) * mask
    ax.imshow(n_peaks, cmap=ListedColormap(count_colors), vmin=-0.5, vmax=3.5); ax.set_axis_off(); ax.set_title(f"peaks per voxel\n{name}", fontsize=9)
    ax.contour(region, levels=[0.5], colors=["white"], linewidths=0.6)
fig.legend(handles=[Patch(color=c, label=f"{k} peak{'s' if k != 1 else ''}") for k, c in enumerate(count_colors)],
           loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.08, 1, 1))
```

The crossing band lies between the two white lines. The multi-tissue and DSI fits were run
in white matter only, so the rest of their brain shows 0. What to look at: white matter
inside the band should be green (two peaks) and white matter outside it blue (one peak).
Blue voxels inside the band are missed crossings. The single-shell fits (the first three
panels) also show gray matter and CSF in yellow: where there is no fiber, the ODF is nearly
round, and its small noisy bumps pass the peak threshold as spurious peaks. Single-shell CSD at b = 3000 finds both fibers in every
voxel of the band, to within a few degrees. DSI finds them nearly always, with broader peaks
and a larger angular error. The multi-tissue fit and q-ball find most of them; both are
limited here by the 30 directions per shell, which cap the harmonic order at 6, and the
multi-tissue fit also spends part of that angular resolution on separating three tissues.
(Q-ball and the deconvolution methods build the ODF as a sum of smooth standard shapes on the
sphere, the **spherical harmonics**; the maximum **order** sets how finely the shapes may
vary with direction. A higher order allows sharper shapes but has more coefficients to
estimate, so it needs more directions; the figure after this paragraph shows the effect.)
CSD at b = 1000 misses a third of the crossings, and the peaks it reports are off by
15°, because the angular contrast of the signal is low at that b-value ([Chapter 5](../02-diffusion-encoding/05-diffusion-encoding.md)): the
shell is fine for a tensor and inadequate for deconvolution. False peaks in single-fiber
white matter are the other failure mode; at this SNR none of the methods produced them, but
they appear at lower SNR and in gray matter and CSF partial volume, which is what the
multi-tissue model is for.

The same deconvolution at three harmonic orders, on one noise-free voxel with two fibers
crossing at 45° (60 directions at b = 3000):

```{code-cell} python
:tags: [hide-input]
bv_o, bg_o = schemes.single_shell(3000, 60, n_b0=1)
g_o = gradient_table(bv_o, bvecs=bg_o)
dirs = in_plane(-22.5, 22.5)
s = voxel_signal(bv_o, bg_o, dirs)
fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.4))
for ax, order in zip(axes, (4, 6, 8)):
    fit = ConstrainedSphericalDeconvModel(g_o, fiber_response(3000, order), sh_order_max=order).fit(s)
    pk_dirs = peak_directions(fit.odf(sphere), sphere, relative_peak_threshold=0.25, min_separation_angle=25)[0]
    n = len(pk_dirs)
    sep = f", {np.degrees(np.arccos(abs(pk_dirs[0] @ pk_dirs[1]))):.0f}° apart" if n == 2 else ""
    glyph(ax, fit.odf(ring_sphere), dirs[:, :2], color=PALETTE[2], title=f"order {order}: {n} peak{'s' if n > 1 else ''}{sep}")
    print(f"order {order}: {(order + 1) * (order + 2) // 2} coefficients to estimate; {n} peak(s) found{sep}")
fig.tight_layout()
```

What to look at: the lobes against the dashed true directions. At order 4 the two fibers
are a single lobe along their bisector, because a sum of order-4 shapes cannot vary quickly
enough with direction to have two peaks 45° apart. At order 6 the lobe splits, but its two
peaks are pulled toward each other. At order 8 the lobes are narrower and the peaks sit on
the true directions. An order-8 fit has 45 coefficients to estimate, more than the 30
measurements a 30-direction shell provides, which is why the fits above stop at order 6.

```{code-cell} python
:tags: [hide-input]
pk = results["CSD, b = 3000"]
r0, r1, c0, c1 = 66, 90, 72, 92  # a patch of white matter across the lower edge of the band
fig, ax = plt.subplots(figsize=(5.2, 6))
ax.imshow(phantoms.brain_image()[r0:r1, c0:c1], cmap="gray", vmin=0, vmax=0.5, extent=(c0 - 0.5, c1 - 0.5, r1 - 0.5, r0 - 0.5))
for i in range(r0, r1):
    for j in range(c0, c1):
        if not wm[i, j]:
            continue
        for k in range(3):
            if pk.peak_values[i, j, 0, k] <= 0:
                continue
            d = pk.peak_dirs[i, j, 0, k]  # (row, col, slice) components
            ax.plot([j - 0.45 * d[1], j + 0.45 * d[1]], [i - 0.45 * d[0], i + 0.45 * d[0]], color=PALETTE[1] if k == 0 else PALETTE[2], lw=1.8)
edge = np.flatnonzero(region[:, 0]).max() + 0.5
ax.axhline(edge, color="white", ls="--", lw=1)
ax.text(c1 - 1, edge - 0.7, "crossing band", color="white", ha="right", fontsize=9)
ax.text(c1 - 1, edge + 1.3, "one fiber population", color="white", ha="right", fontsize=9)
ax.set(title="CSD peaks at b = 3000\n(orange: first peak, aqua: second)", xticks=[], yticks=[])
fig.tight_layout()
```

The patch above straddles the lower edge of the crossing band (dashed line). What to look at:
each white matter voxel carries one line per detected peak. Below the edge each voxel has a
single orange line that follows the white matter; above it each voxel has a second, aqua
line at about 60° to the first. The lines keep a consistent pattern from voxel to voxel,
which is what tractography ([Chapter 18](18-tractography.md)) needs in order to follow either bundle through
the crossing.

## How close can two fibers be?

A common rule of thumb says that crossings under about 45° merge into one peak. The
animation tests it in one voxel. The crossing angle sweeps from 20° to 90°; the voxel is
measured with a single shell of 60 directions at b = 2000 (a typical clinical research
shell), noise-free, and fitted with q-ball (CSA) and CSD, both at harmonic order 8. The
peak extraction is the one used above (relative threshold 0.5 for q-ball, 0.25 for CSD,
minimum separation 25°).

```{code-cell} python
:tags: [hide-input]
bv_s, bg_s = schemes.single_shell(2000, 60, n_b0=1)
g_s = gradient_table(bv_s, bvecs=bg_s)
sweep_models = {"q-ball (CSA)": (CsaOdfModel(g_s, sh_order_max=8), 0.5, PALETTE[0]),
                "CSD": (ConstrainedSphericalDeconvModel(g_s, fiber_response(2000, 8), sh_order_max=8), 0.25, PALETTE[2])}

def sweep_fit(angle_deg, s_=None):
    """Glyph on the ring, peak count, and angle between the two strongest peaks, per method."""
    s_ = voxel_signal(bv_s, bg_s, in_plane(-angle_deg / 2, angle_deg / 2)) if s_ is None else s_
    out = {}
    for name, (model, rel, _) in sweep_models.items():
        fit = model.fit(s_)
        pk_dirs = peak_directions(fit.odf(sphere), sphere, relative_peak_threshold=rel, min_separation_angle=25)[0]
        sep = np.degrees(np.arccos(abs(pk_dirs[0] @ pk_dirs[1]))) if len(pk_dirs) > 1 else np.nan
        out[name] = (fit.odf(ring_sphere), len(pk_dirs), sep)
    return out

fine = {a: sweep_fit(a) for a in range(20, 91)}  # 1° steps, for the threshold
for name in sweep_models:
    two = [a for a in fine if fine[a][name][1] == 2]
    print(f"{name:>13}: two peaks from a crossing angle of {min(two)}° up (noise-free, b = 2000); "
          f"peaks found {fine[50][name][2]:.0f}° apart for a 50° crossing, {fine[60][name][2]:.0f}° apart for 60°")
rng = np.random.default_rng(0)
print("\nwith noise at SNR 30 (100 noisy copies of the voxel per angle), fraction with exactly two peaks:")
for a in (40, 50, 60):
    clean_s = voxel_signal(bv_s, bg_s, in_plane(-a / 2, a / 2))
    noisy_s = np.abs(clean_s + rng.normal(0, 1 / 30, (100, clean_s.size)) + 1j * rng.normal(0, 1 / 30, (100, clean_s.size)))
    frac = {name: np.mean([len(peak_directions(o, sphere, relative_peak_threshold=rel, min_separation_angle=25)[0]) == 2
                           for o in model.fit(noisy_s).odf(sphere)]) for name, (model, rel, _) in sweep_models.items()}
    print(f"  {a}° crossing: " + ", ".join(f"{name} {f:.0%}" for name, f in frac.items()))
```

```{code-cell} python
:tags: [hide-input]
frames = list(range(20, 91, 2)) + [90] * 4 + list(range(88, 20, -2))
fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.1))
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.84, wspace=0.08)

def update(a):
    for ax, (name, (_, _, color)) in zip(axes, sweep_models.items()):
        ax.clear()
        odf, n, _ = fine[a][name]
        glyph(ax, odf, in_plane(-a / 2, a / 2)[:, :2], color=color, title=f"{name}, crossing {a}°\n{n} peak{'s' if n > 1 else ''} found")
        ax.title.set_fontsize(11)
animate(fig, update, frames, fps=6, width=560, dpi=75,
        alt="two polar ODF glyphs side by side, q-ball on the left and CSD on the right, for a voxel whose two fibers (dashed lines) cross at an angle that sweeps from 20 to 90 degrees and back; at small angles both glyphs are a single lobe along the bisector; the CSD glyph splits into two narrow lobes on the dashed lines at a slightly smaller angle than the broad q-ball glyph does")
```

What to look at: the moment each glyph splits, and the title's peak count changing from 1 to
2. Below the split angle each method reports one fiber along the bisector, the tensor's
error in a sharper form. In this noise-free voxel CSD first finds two peaks at a 44°
crossing and q-ball at 48°, so the rule of thumb holds for CSD and is slightly optimistic for
q-ball. The larger difference is in the shape: the q-ball lobes are broad and run into each
other, so just after the split its peaks are pulled toward each other (40° apart for a 50°
crossing, against 45° for CSD), while the CSD lobes are narrow and lie near the dashed lines.
With noise the difference grows: at SNR 30 a 50° crossing is found in every noisy copy by
CSD and in 77% of them by q-ball, and a 40° crossing is missed by both. Higher b-values, more
directions, and higher harmonic orders move the split angle down; noise moves it up.

## What each method needs

| Method | Shells | Directions | Notes |
|---|---|---|---|
| Q-ball (CSA) | one, b ≥ 2000 | ≥ 60 | broad peaks; crossings under about 50° merge (48° noise-free at b = 2000, above) |
| CSD, single tissue | one, b ≥ 2000 | ≥ 45 | needs single-fiber voxels for the response; false peaks in GM/CSF partial volume; crossings under about 45° merge (44° noise-free at b = 2000) |
| Multi-tissue CSD | ≥ 2 (3 preferred) | ≥ 45 on the top shell | needs tissue masks; suppresses partial-volume peaks |
| DSI | Cartesian grid, b to 4000+ | 200–500 | model-free; long scan; strong gradients |
| CS-DSI | random subset of the grid | ~60–100 | needs the compressed-sensing reconstruction (not in dipy) |

CS-DSI is compressed-sensing DSI ([Chapter 6](../02-diffusion-encoding/06-qspace-sampling.md)): it acquires a random subset of the DSI grid
and fills in the missing points with a reconstruction that assumes the displacement
distribution can be described by a few coefficients (a sparsity prior), trading scan time
for that assumption.

## Measure it: the simulated datasets

:::{admonition} Simulated dataset pending
:class: note
This section will run the same reconstructions on the `ref-schemes` dataset and score them
against the truth peaks TRXScan writes with `--truth-peaks` (up to three orientations per
voxel with their mass fractions), and against the `gfa` and `qa` truth maps, on the real
crossing geometry of the simulated brain's tractogram.
:::

## What this implies for acquisition

- **Crossings need b ≥ 2000 and 45 or more directions.** A b = 1000 tensor protocol
  cannot resolve them, whatever model is fitted.
- **Multi-shell adds tissue separation** and is the default choice for tractography
  studies; the low shell is not wasted, it removes partial-volume false peaks.
- **Angular precision improves with SNR and direction count**; false peaks decrease with
  regularization and with multi-tissue modeling.
- **DSI buys model independence at a large scan-time cost**; CS-DSI is the practical
  variant.

## Further reading

Q-ball {cite:p}`tuch2004,aganj2010`, DSI {cite:p}`wedeen2005`, CSD {cite:p}`tournier2007`,
multi-tissue CSD {cite:p}`jeurissen2014`, and the prevalence of crossings
{cite:p}`jeurissen2013`.
