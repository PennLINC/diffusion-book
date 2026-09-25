---
title: Tractography
subtitle: Chapter 18
kernelspec:
  name: python3
  display_name: Python 3
---

## Learning goals

After this chapter you can:

- explain how streamlines are generated from local fiber orientations, and what the
  step size, curvature limit, stopping criterion, and seeding decide
- run deterministic and probabilistic tracking with anatomical constraints
- evaluate a tractogram against a known orientation field and count the streamlines that
  end in the wrong tissue
- state how the acquisition bounds what tractography can recover

**Datasets used:** `ref-schemes`, `truth` (pending); the toy tier uses the 3 mm volume
**Simulation tier:** toy + phantom

```{code-cell} python
:tags: [hide-cell]
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import warnings
warnings.filterwarnings("ignore")  # dipy's fit notices are not part of the lesson
import numpy as np
import matplotlib.pyplot as plt
from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from dipy.direction import peaks_from_model, ProbabilisticDirectionGetter
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response_ssst
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ActStoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.tracking.utils import seeds_from_mask
from scipy import ndimage

from dwibook import phantoms, schemes, synth
from dwibook.plotting import PALETTE, set_style, show_image

set_style()
sphere = get_sphere(name="repulsion724")
```

## From orientations to streamlines

Tractography turns the per-voxel fiber orientations of Chapter 16 into curves. Starting
from a seed point, a streamline is grown by stepping a fixed distance along the local
orientation, re-evaluating the orientation at the new position, and repeating until a
stopping rule applies {cite:p}`basser2000`. The decisions that shape the result:

- **The local model.** The tensor gives one direction per voxel and stops or turns at
  crossings; a fiber ODF offers several.
- **Deterministic or probabilistic.** Deterministic tracking follows the peak nearest the
  current heading; probabilistic tracking draws a direction from the ODF at each step, so
  that repeated seeds explore the uncertainty and the crossings.
- **Step size and curvature limit.** A fraction of a voxel per step, and a maximum turning
  angle per step (typically 30–45° per voxel of travel) that prevents doubling back.
- **Stopping criteria.** Classic tracking stops when FA falls below a threshold.
  Anatomically constrained tractography (ACT) {cite:p}`smith2012` instead uses a tissue
  segmentation: streamlines continue in white matter, terminate when they enter gray
  matter, and are rejected if they enter CSF or leave the brain, so that every accepted
  streamline connects two gray matter regions.
- **Seeding.** From every white matter voxel, from the gray-white interface, or from a
  region of interest; seeding density sets the number of streamlines and biases the
  density toward the seeded region.
- **Filtering.** The density of streamlines does not measure the density of fibers.
  SIFT and SIFT2 {cite:p}`smith2015` weight or remove streamlines so that their density
  matches the fiber density the ODFs imply; this is how the phantom's tractogram was
  built.

## The synthetic volume

The toy tier tracks on the 3 mm tissue volume with a single-shell scheme at b = 2000 (30
directions plus three b=0) and noise at SNR 25. The synthetic fibers run tangent to the
white matter boundary in every voxel, so the answer key is an orientation field, and a
correct streamline follows that field.

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
print(f"volume {mask.shape}, {len(bvals)} volumes; CSD fitted")
```

## Tracking

Both trackers use ACT with the tissue fractions: white matter is the tracking domain, gray
matter the termination map, CSF the exclusion map. Seeds are placed in every voxel with
more than half white matter, one per voxel.

```{code-cell} python
:tags: [hide-input]
stopping = ActStoppingCriterion(gm_frac, csf_frac)
seeds = seeds_from_mask(wm_frac > 0.5, affine, density=1)
det = Streamlines(LocalTracking(pk, stopping, seeds, affine, step_size=0.5, max_cross=1, return_all=False))
prob_getter = ProbabilisticDirectionGetter.from_shcoeff(csd_fit.shm_coeff, max_angle=30.0, sphere=sphere)
prob = Streamlines(LocalTracking(prob_getter, stopping, seeds, affine, step_size=0.5, return_all=False))
print(f"seeds {len(seeds)}; accepted streamlines: deterministic {len(det)}, probabilistic {len(prob)}")
```

`return_all=False` keeps only the streamlines ACT accepts, those that end in gray matter at
both ends without crossing CSF; the rest are rejected as anatomically implausible.

```{code-cell} python
:tags: [hide-input]
def draw(ax, streamlines, title, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(streamlines), size=min(n, len(streamlines)), replace=False)
    ax.imshow(np.max(wm_frac, axis=2), cmap="gray", vmin=0, vmax=2)  # white matter silhouette, anterior at the top
    for k in idx:
        s = streamlines[k]
        d = np.abs(np.gradient(s, axis=0)); d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
        color = np.clip(d.mean(0)[[1, 0, 2]], 0, 1)  # RGB = (left-right, anterior-posterior, superior-inferior)
        ax.plot(s[:, 1], s[:, 0], color=color, lw=0.4, alpha=0.7)
    ax.set(title=title, xticks=[], yticks=[])
    ax.set_aspect("equal")

fig, axes = plt.subplots(1, 2, figsize=(9, 5))
draw(axes[0], det, f"deterministic ({len(det)} streamlines, 1500 shown)")
draw(axes[1], prob, f"probabilistic ({len(prob)} streamlines, 1500 shown)")
fig.tight_layout()
```

The projection is axial (all slices superimposed), colored by direction: green for
anterior-posterior, red for left-right, blue for superior-inferior. The synthetic fibers
follow the white matter boundaries, so the streamlines run along the gyri and around the
ventricles.

## Measure it: agreement with the orientation field

Two scores. The angle between each streamline segment and the true orientation at that
position, averaged over all segments, measures how faithfully the tracker followed the
fibers; the fraction of streamlines rejected by ACT (before filtering) or ending in CSF
measures the anatomical plausibility. The probabilistic tracker is run again with
`return_all=True` to count its rejections.

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

all_prob = Streamlines(LocalTracking(prob_getter, stopping, seeds, affine, step_size=0.5, return_all=True))
all_det = Streamlines(LocalTracking(pk, stopping, seeds, affine, step_size=0.5, max_cross=1, return_all=True))
for name, s, s_all in [("deterministic", det, all_det), ("probabilistic", prob, all_prob)]:
    e = segment_angle_error(s)
    print(f"{name:>14}: segment angle error median {np.median(e):4.1f}°, 90th percentile {np.percentile(e, 90):4.1f}°; "
          f"accepted by ACT {len(s)} of {len(s_all)} ({len(s) / len(s_all):.0%})")
```

The deterministic tracker follows the field more tightly; the probabilistic one explores
more, which is what allows it to pass through crossings and fanning regions in real data,
at the price of more streamlines that wander into the wrong tissue and are rejected. The
rejection rate is a useful summary of a dataset: it rises with noise, with poor
orientation estimates, and with misregistration between the diffusion data and the tissue
segmentation (Chapter 10).

## How the acquisition bounds tractography

Tractography compounds every earlier choice. Angular precision of the peaks (Chapter 16)
sets how far a streamline drifts per step; voxel size sets which crossings are resolved at
all and how well the segmentation aligns; distortion and motion residuals (Part III)
misplace the fibers relative to the anatomy that ACT uses to accept them. The tractography
challenge of {cite:t}`maierhein2017` showed that even with careful methods, tractograms
contain many plausible-looking streamlines that do not exist, and that the acquisition
sets the floor on that rate. The evaluation on the phantom, whose tractogram generated the
data, is the direct test.

## Measure it: the phantom

:::{admonition} Phantom figure pending
:class: note
This section will track on the `ref-schemes` dataset and evaluate the result against the
tractogram that generated it: bundle overlap and overreach, and the fraction of valid
connections, for the 30-direction, 64-direction, and multi-shell schemes, with renders from
the pipeline's TRXViz output. It will also compare SIFT2-weighted streamline density with
the phantom's true fiber density.
:::

## What this implies for acquisition

- **Tractography needs orientation estimates, so it needs the acquisition of Chapter 16**:
  b ≥ 2000 and 45 or more directions, multi-shell preferred.
- **Resolution decides which crossings exist in the data at all.**
- **The tissue segmentation is part of the tractography input**; distortion correction and
  registration to the anatomical image must be right for ACT to work.
- **Streamline counts are not fiber counts**; use SIFT2 or an equivalent and report the
  filtering.

## Further reading

The first streamline tractography {cite:p}`basser2000`, anatomical constraints
{cite:p}`smith2012`, SIFT2 {cite:p}`smith2015`, the tractography challenge
{cite:p}`maierhein2017`, and the review by {cite:t}`jeurissen2019`.
