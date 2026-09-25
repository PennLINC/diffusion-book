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
from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from dipy.direction import peaks_from_model
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, response_from_mask_ssst
from dipy.reconst.dsi import DiffusionSpectrumModel
from dipy.reconst.mcsd import MultiShellDeconvModel, multi_shell_fiber_response, response_from_mask_msmt
from dipy.reconst.shm import CsaOdfModel

from dwibook import phantoms, schemes, synth
from dwibook.plotting import PALETTE, set_style, show_image

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

The methods estimate an **orientation distribution function** (ODF), a function on the
sphere whose peaks are the fiber directions. Two definitions are in use. The **diffusion
ODF** is the angular profile of the displacement distribution, which peaks along fibers
but broadly; q-ball imaging and DSI estimate it. The **fiber ODF** is the distribution of
fiber orientations itself, obtained by deconvolving the signal of a single coherent bundle
(the response function) out of the measured signal; spherical deconvolution estimates it,
with sharper peaks and better separation of crossings.

## The synthetic crossing

The toy tier adds a second fiber population to the 2 mm slice: inside a band of rows
through the middle of the brain, half of the white matter signal comes from fibers at 60°
to the boundary-tangent fibers, within the slice plane, so that every white matter voxel
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

- **Q-ball imaging** {cite:p}`tuch2004` estimates the diffusion ODF from a single shell by
  a spherical transform of the signal; the constant-solid-angle variant
  {cite:p}`aganj2010` corrects the normalization. It needs 60 or more directions at
  b ≥ 2000 to resolve crossings, and its peaks are broad.
- **Diffusion spectrum imaging** {cite:p}`wedeen2005` reconstructs the displacement
  distribution by an inverse Fourier transform of the Cartesian q-space grid ([Chapter 6](../02-diffusion-encoding/06-qspace-sampling.md))
  and projects it to a diffusion ODF. Model-free, at the cost of the grid acquisition.
- **Constrained spherical deconvolution** (CSD) {cite:p}`tournier2007` estimates the
  fiber ODF from one shell by deconvolving a response function measured in
  single-fiber voxels, with a non-negativity constraint that suppresses spurious peaks. It
  works best at b ≥ 2000 with 45 or more directions.
- **Multi-shell multi-tissue CSD** {cite:p}`jeurissen2014` uses one response per tissue
  type (white matter, gray matter, CSF) and the decay across shells to separate them, so
  that gray matter and CSF partial volume do not produce false fiber peaks. It needs at
  least two shells and tissue masks or segmentations for the responses.

Each model returns an ODF per voxel; the same peak extraction is applied to all: local
maxima of the ODF on a sphere of 724 directions, separated by at least 25°, above a
fraction of the largest peak.

```{code-cell} python
:tags: [hide-input]
def peaks(model, data_, gtab_, rel=0.25, m=None):
    return peaks_from_model(model, data_, sphere, relative_peak_threshold=rel, min_separation_angle=25, mask=mask3 if m is None else m, npeaks=3)

results = {}
# q-ball (CSA) on the b = 2000 shell
keep = (bvals == 0) | (bvals == 2000)
g2 = gradient_table(bvals[keep], bvecs=bvecs[keep])
results["q-ball (CSA), b = 2000"] = peaks(CsaOdfModel(g2, sh_order_max=6), data[..., keep], g2, rel=0.5)
# single-shell CSD on the b = 3000 shell; the response function comes from single-fiber
# white matter voxels (in practice selected by high FA, here known from the answer key)
single_fiber = (wm & ~region)[:, :, None]
keep = (bvals == 0) | (bvals == 3000)
g3 = gradient_table(bvals[keep], bvecs=bvecs[keep])
response, ratio = response_from_mask_ssst(g3, data[..., keep], single_fiber)
results["CSD, b = 3000"] = peaks(ConstrainedSphericalDeconvModel(g3, response, sh_order_max=6), data[..., keep], g3)
# single-shell CSD on the b = 1000 shell (for comparison)
keep = (bvals == 0) | (bvals == 1000)
g1 = gradient_table(bvals[keep], bvecs=bvecs[keep])
response1, _ = response_from_mask_ssst(g1, data[..., keep], single_fiber)
results["CSD, b = 1000"] = peaks(ConstrainedSphericalDeconvModel(g1, response1, sh_order_max=6), data[..., keep], g1)
# multi-shell multi-tissue CSD on all shells, responses from the tissue fractions
resp_wm, resp_gm, resp_csf = response_from_mask_msmt(gtab, data, (t["wm"] > 0.9)[:, :, None] & ~region[:, :, None], (t["gm"] > 0.9)[:, :, None], (t["csf"] > 0.9)[:, :, None])
msmt_response = multi_shell_fiber_response(sh_order_max=6, bvals=np.unique(np.round(bvals / 50) * 50), wm_rf=resp_wm, gm_rf=resp_gm, csf_rf=resp_csf)
# the multi-tissue fit solves a convex problem per voxel and is the slowest method here; white matter only
results["multi-tissue CSD, 3 shells"] = peaks(MultiShellDeconvModel(gtab, msmt_response, sh_order_max=6), data, gtab, m=wm[:, :, None])
print("peak extraction done for", list(results))
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
results["DSI, 257 points"] = peaks(DiffusionSpectrumModel(g_dsi, qgrid_size=17, filter_width=18), noisy_dsi, g_dsi, rel=0.5, m=wm[:, :, None])
```

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
fig, axes = plt.subplots(1, 5, figsize=(14, 3.2))
for ax, (name, pk) in zip(axes, results.items()):
    n_peaks = np.count_nonzero(pk.peak_values[:, :, 0] > 0, axis=-1) * mask
    ax.imshow(n_peaks, cmap="viridis", vmin=0, vmax=3); ax.set_axis_off(); ax.set_title(f"peaks per voxel\n{name}", fontsize=9)
    ax.contour(region, levels=[0.5], colors=["white"], linewidths=0.6)
fig.tight_layout()
```

The crossing band is outlined. Single-shell CSD at b = 3000 finds both fibers in every
voxel of the band, to within a few degrees. DSI finds them nearly always, with broader peaks
and a larger angular error. The multi-tissue fit and q-ball find most of them; both are
limited here by the 30 directions per shell, which cap the harmonic order at 6, and the
multi-tissue fit also spends part of that angular resolution on separating three tissues.
CSD at b = 1000 misses a third of the crossings, and the peaks it reports are off by
15°, because the angular contrast of the signal is low at that b-value ([Chapter 5](../02-diffusion-encoding/05-diffusion-encoding.md)): the
shell is fine for a tensor and inadequate for deconvolution. False peaks in single-fiber
white matter are the other failure mode; at this SNR none of the methods produced them, but
they appear at lower SNR and in gray matter and CSF partial volume, which is what the
multi-tissue model is for.

```{code-cell} python
:tags: [hide-input]
pk = results["CSD, b = 3000"]
r0, r1, c0, c1 = 48, 80, 40, 88
fig, ax = plt.subplots(figsize=(7, 5))
ax.imshow(phantoms.brain_image()[r0:r1, c0:c1], cmap="gray", vmin=0, vmax=0.5, extent=(c0 - 0.5, c1 - 0.5, r1 - 0.5, r0 - 0.5))
for i in range(r0, r1):
    for j in range(c0, c1):
        if not wm[i, j]:
            continue
        for k in range(3):
            if pk.peak_values[i, j, 0, k] <= 0:
                continue
            d = pk.peak_dirs[i, j, 0, k]  # (row, col, slice) components
            ax.plot([j - 0.45 * d[1], j + 0.45 * d[1]], [i - 0.45 * d[0], i + 0.45 * d[0]], color=PALETTE[1] if k == 0 else PALETTE[2], lw=1)
ax.set(title="CSD peaks at b = 3000 inside the crossing band (orange: first peak, aqua: second)", xticks=[], yticks=[])
fig.tight_layout()
```

## What each method needs

| Method | Shells | Directions | Notes |
|---|---|---|---|
| Q-ball (CSA) | one, b ≥ 2000 | ≥ 60 | broad peaks; crossings under 45° merge |
| CSD, single tissue | one, b ≥ 2000 | ≥ 45 | needs single-fiber voxels for the response; false peaks in GM/CSF partial volume |
| Multi-tissue CSD | ≥ 2 (3 preferred) | ≥ 45 on the top shell | needs tissue masks; suppresses partial-volume peaks |
| DSI | Cartesian grid, b to 4000+ | 200–500 | model-free; long scan; strong gradients |
| CS-DSI | random subset of the grid | ~60–100 | needs the compressed-sensing reconstruction (not in dipy) |

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
