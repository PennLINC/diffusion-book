---
title: "20. Multi-TE diffusion MRI"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** single-voxel signals with the compartment T2 values of the presets ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
- **`te-sweep`** (pending): four echo times at fixed b ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-te-sweep)).
- **`truth`** (pending): the 27 analytic ground-truth maps and the true fiber orientations ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-truth), [Appendix E](../appendices/e-truth-map-catalogue.md)).

Pipeline-tier datasets are simulated offline by TRXScan ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)) and are marked *pending* until their release; the figures that need them say so where they will appear.
:::

## Learning goals

After this chapter you can:

- explain why diffusion measures depend on the echo time when tissue compartments have
  different T2 values
- fit a joint diffusion-relaxation model to data acquired at several echo times and
  recover compartment T2 values
- state what a multi-TE acquisition costs and how to sample the (b, TE) plane

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from dwibook import phantoms, presets, schemes, signal, synth
from dwibook.plotting import PALETTE, TISSUE_COLORS, set_style, show_image

set_style()
```

## Compartmental T2

The tissue compartments of [Chapter 17](../04-modeling/17-microstructure-models.md) differ not only in how water diffuses in them but in
how fast their signal decays with echo time. Intra-axonal water has a longer T2 than
extracellular water in white matter (values of roughly 80–90 ms versus 50–60 ms at 3 T have
been reported), and CSF has a T2 of seconds. A diffusion acquisition at a single echo time
therefore weights the compartments by their T2 decay at that TE, and every fraction a
diffusion model reports is a signal fraction at that TE, not a volume fraction. The
consequence is visible when the same model is fitted to data acquired at different echo
times: the apparent intra-axonal fraction rises with TE, because the compartment with the
longer T2 retains more of the signal {cite:p}`veraart2018`.

The simulated brain assigns one T2 per tissue (68 ms for the white matter fiber compartment, 76 ms
for gray matter, 2000 ms for CSF), so in the simulator the TE dependence appears between
tissues but not between the two white matter compartments. The toy demonstration below
gives the intra-axonal stick a T2 of 90 ms and the extra-axonal tensor a T2 of 60 ms to show
the within-tissue effect; both values are literature figures, not the simulated brain's.

## See it: the apparent fraction depends on TE

A single white matter voxel is simulated at four echo times with the spherical-mean fit of
[Chapter 17](../04-modeling/17-microstructure-models.md), which knows nothing about T2:

```{code-cell} python
:tags: [hide-input]
bvals, bvecs = schemes.multi_shell({1000: 30, 2000: 30, 3000: 30}, n_b0=4)
fiber = np.array([0.0, 0.0, 1.0])
cos = bvecs @ fiber
te_list = [60.0, 80.0, 100.0, 130.0]
print(f"{'TE (ms)':>8}   apparent intra-axonal fraction (spherical-mean fit)   true volume fraction")
for te in te_list:
    s = signal.multi_te_white_matter(bvals, te, cos)
    shells, means = synth.spherical_mean(s[None, :], bvals)
    fit = synth.smt_fit(shells, means)
    print(f"{te:>8.0f}   {fit['f'][0]:.2f}                                                  {presets.ADULT_FRACTIONS['WM_intra']:.2f}")
```

The fitted fraction climbs by roughly a tenth across the range of echo times used in
practice. It sits above the volume fraction even at the shortest TE, for two reasons that
add: at any TE the compartment with the longer T2 has already gained signal share, and the
spherical-mean fit's tortuosity assumption does not match the synthetic tissue exactly
([Chapter 17](../04-modeling/17-microstructure-models.md)). The trend with TE is the relaxation effect alone. Two studies with different echo times therefore report different "neurite
densities" for the same tissue, and a study that changes TE between scanners or protocol
versions introduces an apparent change in microstructure.

## Diffusion-relaxation correlation

Acquiring several echo times turns the confound into a measurement. The signal of a
compartment is the product of its T2 decay and its diffusion attenuation, so on a grid of
(b, TE) values the compartments separate along both axes at once: a compartment with long T2
and restricted diffusion occupies a different corner of the plane than one with short T2
and hindered diffusion. Fitting a model to the whole plane recovers the compartment
fractions at TE = 0 (which are volume fractions, up to proton density) and the compartment
T2 values, which are new tissue parameters in their own right.

```{code-cell} python
:tags: [hide-input]
b_grid = np.array([0, 1000, 2000, 3000], float)
te_grid = np.array([60, 80, 100, 130], float)
B, TE = np.meshgrid(b_grid, te_grid, indexing="ij")
# spherical means of the two-compartment WM voxel at every (b, TE)
def sm_signal(f, t2i, t2e, b, te):
    from scipy.special import erf
    d_i = presets.ADULT_DIFFUSIVITY["WM_intra"]
    d_par, d_perp, _ = presets.ADULT_DIFFUSIVITY["WM_extra"]
    stick_mean = lambda bd: np.sqrt(np.pi / (4 * np.maximum(bd, 1e-9))) * erf(np.sqrt(np.maximum(bd, 1e-9)))
    intra = f * np.exp(-te / t2i) * np.where(b > 0, stick_mean(b * d_i), 1.0)
    extra = (1 - f) * np.exp(-te / t2e) * np.exp(-b * d_perp) * np.where(b > 0, stick_mean(b * (d_par - d_perp)), 1.0)
    return intra + extra

truth = (0.55, 90.0, 60.0)
rng = np.random.default_rng(0)
data = sm_signal(*truth, B, TE) + rng.normal(scale=0.004, size=B.shape)

fig, ax = plt.subplots(figsize=(6, 3.4))
for k, te in enumerate(te_grid):
    ax.semilogy(b_grid, data[:, k], "o-", color=plt.cm.viridis(k / 3), label=f"TE {te:.0f} ms")
ax.set(xlabel="b (s/mm²)", ylabel="spherical mean signal", title="one white matter voxel on the (b, TE) grid")
ax.legend()
fig.tight_layout()

def residual(p):
    f, t2i, t2e = p
    return (sm_signal(f, t2i, t2e, B, TE) - data).ravel()

fit = least_squares(residual, x0=[0.5, 70.0, 70.0], bounds=([0, 20, 20], [1, 300, 300]))
print(f"joint fit: intra-axonal fraction {fit.x[0]:.2f} (true 0.55), T2 intra {fit.x[1]:.0f} ms (true 90), T2 extra {fit.x[2]:.0f} ms (true 60)")
```

The joint fit returns the volume fraction and the two compartment T2 values from a
16-point grid, where any single echo time returned a TE-dependent fraction and no T2
information at all. Accelerated acquisitions that interleave echo times and diffusion
weightings in one scan (ZEBRA) make such grids practical in clinical times
{cite:p}`hutter2018`.

## Sampling the (b, TE) plane

A full grid is rarely necessary. What the fit needs is enough spread along both axes to
separate the compartments: at least three echo times spanning a range comparable to the
compartment T2 values (60–130 ms), and the usual two or three shells. The cost is scan time
proportional to the number of echo times, and SNR: every volume at a long TE has lost more
signal, so the longest echo time sets the noise floor for the whole fit. Sampling
strategies place more diffusion directions at short TE, where the signal is strong, and
fewer at long TE, where the relaxation information lives.

## Measure it: the simulated datasets

:::{admonition} Simulated dataset pending
:class: note
This section will load the `te-sweep` dataset (the HBCD scheme at four echo times, once the
simulator accepts a per-run TE) and fit the joint model across tissues: the recovered T2
values are compared with the preset T2 of each compartment, and the TE dependence of the
fitted fractions with the truth maps.
:::

## What this implies for acquisition

- **Report the echo time with every diffusion metric**, and keep it fixed within a study.
- **Fractions from single-TE data are signal fractions**; comparing them across protocols
  with different TE compares different quantities.
- **A multi-TE acquisition needs three or more echo times** spanning 60–130 ms and a
  multi-shell scheme at each; budget the SNR at the longest TE.
- **The simulator's per-compartment T2** is what makes this measurable on the simulated datasets; the
  `te-sweep` dataset is the test.

## Further reading

TE-dependent diffusion imaging {cite:p}`veraart2018` and integrated diffusion-relaxometry
{cite:p}`hutter2018`.
