---
title: "17. Biophysical microstructure models"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** a synthetic multi-shell series built from the packaged tissue maps, with known compartment fractions ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
- **`ref-clean`** (pending): the artifact-free, noise-free reference series with its truth maps and true fiber orientations ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-ref-clean)).
- **`ref-schemes`** (pending): the simulated brain under the 30-direction, 64-direction, HBCD, DSI, and CS-DSI schemes at matched scan time ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-ref-schemes)).
- **`truth`** (pending): the 27 analytic ground-truth maps and the true fiber orientations ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-truth), [Appendix E](../appendices/e-truth-map-catalogue.md)).

Pipeline-tier datasets are simulated offline by TRXScan ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)) and are marked *pending* until their release; the figures that need them say so where they will appear.
:::

## Learning goals

After this chapter you can:

- describe the compartment picture behind ball-and-stick, NODDI, the spherical mean
  technique, free-water elimination, and IVIM, and what each parameter claims to measure
- fit a free-water model and a spherical-mean model and compare their parameters with a
  known answer
- explain the degeneracy of multi-compartment fits and show what breaks it
- separate model mismatch from sampling and noise effects

```{code-cell} python
:tags: [hide-cell]
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import warnings
warnings.filterwarnings("ignore")  # dipy's solver notices are not part of the lesson
import numpy as np
import matplotlib.pyplot as plt
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel
from dipy.reconst.fwdti import FreeWaterTensorModel
from scipy.special import erf

from dwibook import phantoms, presets, schemes, signal, synth
from dwibook.plotting import PALETTE, TISSUE_COLORS, set_style, show_image

set_style()
```

## Compartments

A biophysical model writes the voxel signal as a sum of compartments, each with a shape
(a stick, a cylinder, a ball, a zeppelin), a diffusivity, and a volume fraction, and fits
those quantities to the data. The parameters have names that sound like histology
(intra-axonal fraction, neurite density, free-water fraction), and that is both the appeal
and the risk: the names are only as true as the assumptions, and the assumptions are
simplifications of tissue. The standard model of white matter {cite:p}`novikov2019`
collects the common ingredients: impermeable sticks for axons, a Gaussian extra-axonal
compartment, a free-water ball, and an orientation distribution over the sticks.

The models in use differ in which of these they keep and which they fix:

- **Ball-and-stick** {cite:p}`behrens2003`: one or more sticks plus a ball, all with one
  diffusivity. It is the model behind FSL's bedpostx and probabilistic tractography, where
  the point is the stick directions and their uncertainty rather than the fractions.
- **NODDI** {cite:p}`zhang2012`: sticks with a Watson orientation distribution (the
  orientation dispersion index, ODI), a coupled extra-axonal compartment, and a free-water
  ball, with the diffusivities fixed. Reports the neurite density (ICVF), ODI, and the
  free-water fraction. Fixing the diffusivities is what makes it fit from two shells, and
  what makes its values depend on those fixed numbers.
- **Spherical mean technique** {cite:p}`kaden2016`: averages the signal over directions on
  each shell, which removes the orientation distribution entirely, and fits the
  compartments to the per-shell means. Two shells suffice, crossings do not matter, and
  the diffusivities can be fitted rather than fixed.
- **Free-water elimination** {cite:p}`pasternak2009`: a tensor plus a free-water ball with
  the diffusivity of CSF. It corrects the tensor at tissue-CSF boundaries and yields a
  free-water map. With one shell the fit is ill-posed and needs a spatial prior; with two
  or more shells it is determined.
- **IVIM** {cite:p}`lebihan1986`: a tissue compartment plus a fast pseudo-diffusion
  compartment representing blood flow in capillaries, visible only at b < 200. Needs
  several low b-values and is used for perfusion, not microstructure.

## The synthetic series and its answer key

The synthetic white matter is itself a two-compartment model: an intra-axonal stick with
fraction 0.55 and diffusivity 1.7 × 10⁻³ mm²/s, and an extra-axonal tensor with axial 1.7 and
radial 0.6 × 10⁻³. Gray matter is two balls, CSF a free ball. The tissue fractions are
known per voxel. This is the simulated brain's own model ([Chapter 4](../02-diffusion-encoding/04-diffusion-in-tissue.md)), so the answer key is exact
and the fits below measure model mismatch as much as noise.

```{code-cell} python
:tags: [hide-input]
t = phantoms.brain_slice()
mask = t["mask"]
wm, gm, csf = t["wm"] > 0.95, t["gm"] > 0.9, t["csf"] > 0.9
bvals, bvecs = schemes.multi_shell({1000: 20, 2000: 20, 3000: 30}, n_b0=4)
clean = synth.synthetic_dwi(t, bvals, bvecs)
sigma = clean[wm][:, bvals == 0].mean() / 25
noisy = synth.add_noise(clean, sigma, seed=0)
gtab = gradient_table(bvals, bvecs=bvecs)
print(f"true intra-axonal fraction {presets.ADULT_FRACTIONS['WM_intra']}, axial diffusivity {presets.ADULT_DIFFUSIVITY['WM_intra'] * 1e3:.1f} x 10^-3 mm^2/s")
```

## Free-water elimination

The free-water model separates the CSF contribution from the tissue tensor. Its two
outputs are a free-water fraction, which should follow the CSF fraction of the voxel, and a
tissue tensor whose FA no longer collapses at the ventricle wall. dipy's implementation
fits it to two or more shells; the fit uses the b ≤ 2000 volumes here.

```{code-cell} python
:tags: [hide-input]
keep = bvals <= 2000
g2 = gradient_table(bvals[keep], bvecs=bvecs[keep])
fw = FreeWaterTensorModel(g2).fit(noisy[..., keep], mask=mask)
plain = TensorModel(gradient_table(bvals[bvals <= 1000], bvecs=bvecs[bvals <= 1000]), fit_method="WLS").fit(noisy[..., bvals <= 1000], mask=mask)
# the tissue-only reference: the same voxels with their CSF removed
tissue_only = synth.synthetic_dwi({"wm": t["wm"], "gm": t["gm"], "csf": np.zeros_like(t["csf"])}, bvals, bvecs)
ref = TensorModel(gradient_table(bvals[bvals <= 1000], bvecs=bvecs[bvals <= 1000]), fit_method="WLS").fit(tissue_only[..., bvals <= 1000], mask=mask)
border = (t["csf"] > 0.2) & (t["csf"] < 0.8) & (t["wm"] > 0.2)

fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
show_image(axes[0], t["csf"] * mask, "true CSF fraction", kind="scalar", vmin=0, vmax=1)
show_image(axes[1], fw.f * mask, "free-water fraction, fitted", kind="scalar", vmin=0, vmax=1)
show_image(axes[2], plain.fa * mask, "FA, plain tensor", kind="scalar", vmin=0, vmax=0.9)
show_image(axes[3], fw.fa * mask, "FA, free-water-corrected tensor", kind="scalar", vmin=0, vmax=0.9)
fig.tight_layout()
print(f"at tissue-CSF borders ({border.sum()} voxels): free-water fraction {fw.f[border].mean():.2f} vs true CSF fraction {t['csf'][border].mean():.2f}")
print(f"  FA there: plain tensor {plain.fa[border].mean():.2f}, free-water-corrected {fw.fa[border].mean():.2f}, tissue-only reference {ref.fa[border].mean():.2f}")
```

The free-water map follows the CSF fraction in shape but overestimates it, and the
corrected FA at the border rises toward, and past, the value of the tissue alone. Both are
the same model mismatch: the synthetic extra-axonal water diffuses at 1.7 × 10⁻³ mm²/s along
the fibers, fast enough that the model assigns part of it to the free-water ball, which
inflates the fraction and, by removing signal from the tissue compartment, sharpens its
anisotropy. On real tissue the same trade-off exists wherever a fast tissue component is
present, which is why free-water fractions are compared within a study rather than read as
a CSF percentage.

## The spherical mean technique

Averaging the signal over all directions of a shell removes every effect of fiber
orientation, including crossings and dispersion, and leaves a function of b alone. A
two-compartment model of that function has two parameters, an intra-axonal fraction and a
diffusivity, which can be fitted from two or three shells by a grid search:

```{code-cell} python
:tags: [hide-input]
shells, means = synth.spherical_mean(noisy, bvals)
fig, ax = plt.subplots(figsize=(6, 3.2))
for name, m, color in [("WM", wm, TISSUE_COLORS["WM"]), ("GM", gm, TISSUE_COLORS["GM"]), ("CSF", csf, TISSUE_COLORS["CSF"])]:
    ax.semilogy(shells, means[m].mean(0) / means[m].mean(0)[0], "o-", color=color, label=name)
ax.set(xlabel="b (s/mm²)", ylabel="spherical mean / b=0", title="per-shell spherical means")
ax.legend()
fig.tight_layout()

smt = synth.smt_fit(shells, means)
print(f"spherical-mean fit in pure white matter: intra-axonal fraction {np.median(smt['f'][wm]):.2f} (true 0.55), "
      f"axial diffusivity {np.median(smt['d_par'][wm]) * 1e3:.2f} (true 1.70) x 10^-3 mm^2/s")
```

The fraction comes out close to the truth and the diffusivity exact; the residual offset in
the fraction is model mismatch, because the fit ties the extra-axonal radial diffusivity to
the fraction (a tortuosity assumption) and the synthetic tissue does not obey that rule.

## Degeneracy: what the sampling determines

A multi-compartment fit has a landscape of solutions, and with too little data that
landscape has a valley rather than a minimum: many combinations of fraction and
diffusivity explain the measurements equally well {cite:p}`jelescu2016`. The cost of the
spherical-mean model for one white matter voxel, as a function of its two parameters, shows
this directly:

```{code-cell} python
:tags: [hide-input]
i, j = np.argwhere(wm)[len(np.argwhere(wm)) // 2]
f_grid = np.linspace(0.05, 0.95, 91)
d_grid = np.linspace(0.8e-3, 2.4e-3, 81)
F, D = np.meshgrid(f_grid, d_grid, indexing="ij")

def stick_mean(bd):
    bd = np.maximum(bd, 1e-9)
    return np.sqrt(np.pi / (4 * bd)) * erf(np.sqrt(bd))

D_PERP = presets.ADULT_DIFFUSIVITY["WM_extra"][1]  # the model here is the synthetic tissue's own, with the radial diffusivity known

def cost(shell_set):
    c = np.zeros_like(F)
    for b in shell_set:
        meas = means[i, j, list(shells).index(b)] / means[i, j, 0]
        model = F * stick_mean(b * D) + (1 - F) * np.exp(-b * D_PERP) * stick_mean(b * (D - D_PERP))
        c += (model - meas) ** 2
    return c

fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
for ax, shell_set in zip(axes, [[1000], [1000, 2000], [1000, 2000, 3000]]):
    c = cost(shell_set)
    im = ax.imshow(np.log10(c + 1e-9), origin="lower", aspect="auto", cmap="viridis", extent=(d_grid[0] * 1e3, d_grid[-1] * 1e3, f_grid[0], f_grid[-1]), vmin=-8, vmax=-2)
    ax.plot(1.7, 0.55, "o", color="white", ms=6)
    ax.grid(False)
    ax.set(xlabel="axial diffusivity (x10^-3 mm²/s)", ylabel="intra-axonal fraction", title=f"shells {shell_set}")
cbar = fig.colorbar(im, ax=axes, shrink=0.8, label="log10 cost")
```

With one shell the cost is low along a whole curve: any fraction can be traded for a
diffusivity with no change in the fit, so the measurements do not distinguish between
those combinations. The second shell narrows the
valley; the third, at b = 3000, closes it around the true values (white dot). The model in
this figure is the synthetic tissue's own, with the radial diffusivity known, so the only
ambiguity is the one the sampling leaves; a model with more unknowns has a larger valley. This is the practical meaning of Table 6.1's requirement of two or more
shells for compartment models, and of the higher shell being high: the curvature of the
decay is the only information that separates the parameters. The other ways to break the
degeneracy add a different kind of information rather than more of the same: several
diffusion times ([Chapter 22](../05-advanced/22-multi-diffusion-time.md)), several echo times ([Chapter 20](../05-advanced/20-multi-te.md)), or b-tensor encoding
([Chapter 23](../05-advanced/23-frontiers.md)).

## NODDI and ball-and-stick on the simulated datasets

Neither is fitted here. NODDI's implementation (AMICO, or the original MATLAB toolbox) is
an optional dependency of this book, and ball-and-stick's reference implementation is
FSL's bedpostx, which the offline pipeline runs inside the QSIPrep image. Both will be
scored on the simulated datasets, where the truth includes NODDI-style ICVF, ODI, and ISOVF maps.

## Limits of Gaussian compartments

The simulated brain's compartments are Gaussian: a stick, a tensor, and balls. Real tissue has
finite axon diameters, exchange, and time-dependent diffusion ([Chapter 4](../02-diffusion-encoding/04-diffusion-in-tissue.md)), none of which is
in the simulated brain. A model that assumes restriction (a cylinder with a diameter) fits the
simulated brain differently than it fits tissue, and its parameters on simulated data should not be
read as validation of what they mean in vivo. The simulated brain is useful for a narrower
question: given a model, how much of its error comes from the sampling scheme and the noise
as opposed to the model itself. The fits above separate the two by comparing against a
known answer at several schemes.

## Measure it: the simulated datasets

:::{admonition} Simulated dataset pending
:class: note
This section will fit the free-water, spherical-mean, NODDI (AMICO), and ball-and-stick
(bedpostx) models to the `ref-schemes` dataset and score them against the `truth` maps
`icvf`, `odi`, `isovf`, and the simulated brain's known compartment fractions, scheme by scheme.
:::

## What this implies for acquisition

- **Two shells are the minimum for any compartment model**, and the upper shell should be
  at b ≥ 2000; three shells make the fit well determined.
- **Fixed diffusivities (NODDI) make a model fit from less data** and make its values
  depend on those fixed numbers; report them.
- **Free-water correction needs multi-shell data** to be well posed.
- **IVIM needs its own low-b shells** (b = 0 to 200 in several steps); no standard
  diffusion protocol contains them.
- **A model's assumptions are acquisition decisions**: check that the protocol supports the
  model before scanning, not after.

## Further reading

The standard model and its estimation {cite:p}`novikov2019`, degeneracy
{cite:p}`jelescu2016`, ball-and-stick {cite:p}`behrens2003`, NODDI {cite:p}`zhang2012`,
the spherical mean technique {cite:p}`kaden2016`, and free-water elimination
{cite:p}`pasternak2009`.
