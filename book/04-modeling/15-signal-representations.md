---
title: Signal representations
subtitle: Chapter 15
kernelspec:
  name: python3
  display_name: Python 3
---

## Learning goals

After this chapter you can:

- fit the diffusion tensor, the kurtosis tensor, and a propagator representation (MAP-MRI)
  and state the assumptions of each
- state the minimum sampling each representation needs and show what happens when the
  data fall short of it
- choose a tensor fitting method and say when a robust one is needed
- read the fitted maps against a known answer

**Datasets used:** `ref-schemes`, `truth` (pending); the toy tier uses a synthetic multi-shell series
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
from dipy.reconst.dti import TensorModel
from dipy.reconst.dki import DiffusionKurtosisModel
from dipy.reconst.mapmri import MapmriModel

from dwibook import phantoms, presets, schemes, synth
from dwibook.plotting import PALETTE, set_style, show_image

set_style()
```

## Representations versus models

The chapters of Part IV fit two kinds of description to the same data. A **signal
representation** describes the measured signal as a function of b-value and direction with
no claim about the tissue: the diffusion tensor, the kurtosis tensor, and the propagator
bases are representations. Their parameters (FA, MD, kurtosis, return-to-origin probability)
are summaries of the signal that change with tissue but do not name a tissue property. A
**biophysical model** (Chapter 17) assigns the signal to compartments with named
properties, such as an intra-axonal fraction, at the cost of assumptions that may not hold.
Representations are the safer choice when the question is whether something differs;
models are needed when the question is what differs.

## The synthetic series

The toy tier uses the 2 mm slice with a three-shell scheme (20, 20, and 30 directions at
b = 1000, 2000, 3000, plus four b=0), noise at SNR 25 in white matter at b = 0, and the
noise-free fit of the same scheme as the reference. The synthetic white matter is the
phantom's two-compartment model, so it has curvature in its decay (Chapter 5) and the
kurtosis and propagator representations have something to measure.

```{code-cell} python
:tags: [hide-input]
t = phantoms.brain_slice()
mask = t["mask"]
wm, gm = t["wm"] > 0.9, t["gm"] > 0.9
bvals, bvecs = schemes.multi_shell({1000: 20, 2000: 20, 3000: 30}, n_b0=4)
clean = synth.synthetic_dwi(t, bvals, bvecs)
sigma = clean[wm][:, bvals == 0].mean() / 25
noisy = synth.add_noise(clean, sigma, seed=0)
gtab = gradient_table(bvals, bvecs=bvecs)
sub = lambda keep: (bvals[keep], bvecs[keep])
print(f"{len(bvals)} volumes; shells {schemes.shells_of(bvals)}")
```

## The diffusion tensor

The tensor represents the signal as a single Gaussian displacement distribution: an
ellipsoid with three axes. Six parameters plus the b=0 signal describe it, so six
directions at one b-value are the mathematical minimum and about 30 the practical one
(Chapter 6). From the ellipsoid come the standard maps: **mean diffusivity** (MD, the
average of the three axes), **fractional anisotropy** (FA, how elongated the ellipsoid is,
from 0 to 1), axial and radial diffusivity, and the **principal direction**, the long axis,
which tractography follows.

The Gaussian assumption holds only at low b-value. Above about b = 1500 the curvature of
the true decay (Chapter 5) violates it, and a tensor fitted to high-b data returns
diffusivities that depend on which b-values were included:

```{code-cell} python
:tags: [hide-input]
def fit_dti(series, keep, method="WLS", **kw):
    b, v = sub(keep)
    return TensorModel(gradient_table(b, bvecs=v), fit_method=method, **kw).fit(series[..., keep], mask=mask)

ref_low = fit_dti(clean, bvals <= 1000)
rows = [("b <= 1000 (reference use)", bvals <= 1000), ("b <= 2000", bvals <= 2000), ("all shells to 3000", bvals >= 0)]
print(f"{'tensor fitted on':>28}   WM MD (x10^-3)   WM FA   GM MD (x10^-3)")
for label, keep in rows:
    f = fit_dti(noisy, keep)
    print(f"{label:>28}   {1e3 * f.md[wm].mean():.3f}            {f.fa[wm].mean():.3f}   {1e3 * f.md[gm].mean():.3f}")
```

MD falls as higher shells are added, because the fit averages a decay that is slower than
exponential at high b. The values are all "correct" for their own b-range and none is the
tissue's diffusivity; this is why tensor studies specify the b-value and why comparing MD
across protocols with different b-values is not valid. Tensor metrics are fitted to the
b ≤ 1000 shell of a multi-shell acquisition for this reason.

### Fitting methods

The tensor is fitted to the logarithm of the signal, where the model is linear. The
methods differ in how they weight the measurements and what they do with outliers:

- **Ordinary least squares** on the log signal treats all measurements equally, which
  over-weights the noisiest (lowest-signal) ones.
- **Weighted least squares** (WLS) weights each measurement by its signal, which corrects
  that; it is the default in most tools.
- **Nonlinear least squares** fits the signal itself rather than its logarithm; more
  accurate at low SNR, slower.
- **RESTORE** {cite:p}`chang2005` down-weights measurements that disagree with the fit,
  which protects the tensor from dropout slices and spikes that were not caught upstream
  (Chapter 12).

```{code-cell} python
:tags: [hide-input]
corrupted = noisy.copy()
corrupted[40:60, :, 7] *= 0.3  # a partial dropout in one b = 1000 volume, not caught upstream
corrupted[70:90, :, 15] *= 0.3
ref = ref_low
rowsm = [("WLS", {}), ("NLLS", {}), ("RESTORE", {"sigma": sigma})]
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
show_image(axes[0], ref.fa, "FA, noise-free reference", kind="scalar", vmin=0, vmax=0.9)
for ax, (method, kw) in zip(axes[1:], rowsm):
    f = fit_dti(corrupted, bvals <= 1000, method=method, **kw)
    err = np.abs(f.fa - ref.fa)[wm]
    show_image(ax, (f.fa - ref.fa) * mask, f"FA error, {method}", kind="diff", vmin=-0.3, vmax=0.3)
    print(f"{method:>8}: FA error in WM {err.mean():.3f} (rows with the dropout: {np.abs(f.fa - ref.fa)[40:60][wm[40:60]].mean():.3f})")
fig.tight_layout()
```

The uncaught dropout leaves a band of wrong FA in the WLS and NLLS fits; RESTORE
recognizes the affected measurements as outliers and fits without them. Robust fitting is
not a substitute for outlier replacement in preprocessing, which uses all volumes to decide,
but it is a useful last line of defense.

## The kurtosis tensor

Diffusion kurtosis imaging {cite:p}`jensen2005` adds a second term to the tensor: the
curvature of the log signal with b, per direction. It represents the non-Gaussian part of
the decay without naming its cause. **Mean kurtosis** (MK) is elevated wherever diffusion
is restricted or the voxel mixes compartments, so it is high in white matter and in dense
gray matter, and near zero in CSF.

Fitting the curvature requires at least two non-zero shells, and the upper shell must be
high enough for the curvature to be visible, in practice b = 2000–3000 (Chapter 5). A single
shell cannot support the fit at all:

```{code-cell} python
:tags: [hide-input]
def fit_dki(series, keep):
    b, v = sub(keep)
    return DiffusionKurtosisModel(gradient_table(b, bvecs=v), fit_method="WLS").fit(series[..., keep], mask=mask)

try:
    fit_dki(noisy, bvals <= 1000)
except Exception as e:
    print("DKI on a single shell:", type(e).__name__, "-", str(e)[:90])

ref_dki = fit_dki(clean, bvals >= 0)
mk_ref = ref_dki.mk(min_kurtosis=0, max_kurtosis=3)
fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2))
show_image(axes[0], mk_ref * mask, "MK, noise-free, three shells", kind="scalar", vmin=0, vmax=1.5)
for ax, (label, keep) in zip(axes[1:], [("two shells (1000, 2000)", bvals <= 2000), ("three shells", bvals >= 0)]):
    f = fit_dki(noisy, keep)
    mk = f.mk(min_kurtosis=0, max_kurtosis=3)
    show_image(ax, mk * mask, f"MK, noisy, {label}", kind="scalar", vmin=0, vmax=1.5)
    print(f"{label:>24}: MK error in WM {np.abs(mk - mk_ref)[wm].mean():.3f}, in GM {np.abs(mk - mk_ref)[gm].mean():.3f}")
fig.tight_layout()
```

Kurtosis is a second-order quantity and inherits twice the noise sensitivity of the tensor;
it is the representation most improved by denoising (Chapter 8) and most damaged by the
Rician floor, which adds curvature of its own at high b.

## Propagator representations: MAP-MRI

Mean apparent propagator MRI {cite:p}`ozarslan2013` represents the full displacement
distribution (Chapter 4) in a basis of functions, from which several scalar summaries are
computed: the **return-to-origin probability** (RTOP, high where displacement is small,
i.e. restricted), its axial and planar variants, the **mean squared displacement**, and
**non-Gaussianity**. Because it represents the whole distribution it needs the whole
q-space: at least two shells, preferably three or more, with directions spread across them.
A Laplacian regularization keeps the fit stable at the sampling densities of typical
protocols {cite:p}`fick2016`. The diffusion time enters the scaling of the displacement
maps, so the pulse timing must be known to report them in physical units.

```{code-cell} python
:tags: [hide-input]
gtab_map = lambda keep: gradient_table(bvals[keep], bvecs=bvecs[keep], big_delta=0.030, small_delta=0.010)
def fit_map(series, keep):
    return MapmriModel(gtab_map(keep), radial_order=4, laplacian_regularization=True, laplacian_weighting=0.2, positivity_constraint=False, bval_threshold=1500).fit(series[..., keep], mask=mask)

ref_map = fit_map(clean, bvals >= 0)
fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2))
show_image(axes[0], np.cbrt(ref_map.rtop()) * mask, "RTOP^(1/3), noise-free (1/mm)", kind="scalar", vmin=0, vmax=120)
show_image(axes[1], ref_map.msd() * mask * 1e3, "mean squared displacement (x10^-3 mm²)", kind="scalar", vmin=0, vmax=0.2)
show_image(axes[2], ref_map.ng() * mask, "non-Gaussianity", kind="scalar", vmin=0, vmax=0.6)
fig.tight_layout()
for label, keep in [("two shells", bvals <= 2000), ("three shells", bvals >= 0)]:
    f = fit_map(noisy, keep)
    print(f"{label:>12}: RTOP error in WM {100 * np.abs(np.cbrt(f.rtop()) - np.cbrt(ref_map.rtop()))[wm].mean() / np.cbrt(ref_map.rtop())[wm].mean():.1f} %, "
          f"non-Gaussianity error in WM {np.abs(f.ng() - ref_map.ng())[wm].mean():.3f}")
```

## Beyond the pulse pair: QTI

Every representation above reads the signal from a single-direction encoding. Encoding
with several gradient directions inside one measurement (b-tensor encoding) provides a
further representation, q-space trajectory imaging, whose parameters separate microscopic
anisotropy from orientation dispersion, a distinction the tensor cannot make. The phantom's
truth maps include these quantities, but the simulator does not yet produce b-tensor
acquisitions, so this book states the idea (Chapter 23) without fitting it.

## Measure it: the phantom

:::{admonition} Phantom figure pending
:class: note
This section will fit the tensor, kurtosis, and MAP-MRI representations to the
`ref-schemes` dataset (30-direction, 64-direction, HBCD, DSI, and CS-DSI schemes) and score
each against the analytic `truth` maps TRXScan writes for the same phantom: FA, MD, AD, RD,
MK, AK, RK, RTOP, RTAP, RTPP, MSD, and non-Gaussianity.
:::

## What this implies for acquisition

- **A tensor needs 30 directions at b ≈ 1000**, and its values are specific to that b.
- **Kurtosis needs two shells with the upper one at b ≥ 2000**, and benefits more from
  denoising than any other representation.
- **Propagator representations need three or more shells** with directions spread across
  them, and the pulse timing recorded.
- **Fit the tensor to the low shell of a multi-shell scheme**, not to all of it.
- **Use a robust fit** if outlier replacement was not run.

## Further reading

The diffusion tensor {cite:p}`basser1994`, kurtosis {cite:p}`jensen2005`, MAP-MRI
{cite:p}`ozarslan2013,fick2016`, and the distinction between representations and models
{cite:p}`novikov2018`.
