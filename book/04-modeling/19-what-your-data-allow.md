---
title: "19. What your data allow"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** sampling schemes evaluated from their b-values and direction counts, without images ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
- **`ref-schemes`** (pending): the simulated brain under the 30-direction, 64-direction, HBCD, DSI, and CS-DSI schemes at matched scan time ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-ref-schemes)).
- **`truth`** (pending): the 27 analytic ground-truth maps and the true fiber orientations ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-truth), [Appendix E](../appendices/e-truth-map-catalogue.md)).

Pipeline-tier datasets are simulated offline by TRXScan ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)) and are marked *pending* until their release; the figures that need them say so where they will appear.
:::

## Learning goals

After this chapter you can:

- read a scheme's b-values and say which analyses it supports, which are marginal, and
  which are not possible
- state what complex data add across the analysis chain and what they cost
- work through a dataset you did not design and decide what to do with it

```{code-cell} python
:tags: [hide-cell]
import numpy as np

from dwibook import schemes
```

## The decision matrix

Every requirement in Parts II through IV reduces to a few counts: how many non-zero shells,
how many directions on each, how high the top shell goes, how many b=0 volumes, whether the
phase was saved. The helper below applies those rules to a b-value table. It is the
book's Table 6.1 as a function, and it can be run on any `.bval` file.

```{code-cell} python
:tags: [hide-input]
def report(name, bvals, complex_data=False):
    print(f"\n{name}: {len(bvals)} volumes, shells {schemes.shells_of(bvals)}")
    for analysis, verdict, reason in schemes.analysis_matrix(bvals, complex_data=complex_data):
        print(f"  {analysis:<36} {verdict:<9} {reason}")

report("30 directions at b = 1000 (clinical DTI)", schemes.single_shell(1000, 30, n_b0=3)[0])
report("64 directions at b = 2000 (HARDI)", schemes.single_shell(2000, 64, n_b0=4)[0])
report("HBCD multi-shell, phase saved", schemes.hbcd()[0], complex_data=True)
report("DSI grid, 257 points", schemes.dsi_grid(radius=4)[0])
```

The verdicts are rules of thumb, not guarantees. A "yes" means the fit is determined and
the sampling is in the range where the method was developed; "marginal" means the fit
runs but its assumptions are strained or its precision is poor, and the chapter for that
method shows what that looks like; "no" means the fit cannot be performed or its result
has no meaning. The table below collects the rules with the chapter that demonstrates each.

| Analysis | Needs | Chapter |
|---|---|---|
| ADC, MD | ≥ 3 directions, 1 b=0 | 15 |
| DTI: FA, principal direction | ≥ 30 directions at b ≈ 1000 (6 minimum) | 6, 15 |
| Diffusion kurtosis | ≥ 2 non-zero shells, top shell ≥ 2000 | 15 |
| MAP-MRI, propagator | ≥ 3 shells, directions spread across them, pulse timing recorded | 15 |
| Q-ball, single-shell CSD | 1 shell at b ≥ 2000, ≥ 45–60 directions | 16 |
| Multi-tissue CSD | ≥ 2 shells, ≥ 45 directions on the top shell, tissue masks | 16 |
| DSI | Cartesian grid, 200+ points, strong gradients | 6, 16 |
| Free water, NODDI, spherical mean | ≥ 2 shells, top shell ≥ 2000 | 17 |
| IVIM | several shells at b < 200 | 17 |
| Deterministic tractography | any DTI or fODF scheme | 18 |
| Probabilistic tractography with ACT | fODF scheme plus a registered tissue segmentation | 18 |
| Complex-domain denoising | phase saved | 3, 8 |
| Distortion correction (topup) | reverse-polarity volumes and correct metadata | 10 |
| Eddy correction with prediction | full, well-spread direction set | 11 |
| Gradient nonlinearity correction | coefficient file | 13 |

## Complex data revisited

Three chapters used the phase, and the case for saving it is now complete:

- **Denoising without bias** ([Chapter 8](../03-preprocessing/08-noise.md)): the largest gain, and the one that reaches
  every model fitted at high b, since the Rician floor biases exactly the volumes those
  models depend on.
- **Diagnostics** (Chapters [11](../03-preprocessing/11-eddy-currents.md) and [12](../03-preprocessing/12-motion-and-dropout.md)): the eddy-current and motion phase are visible per
  volume before any correction.
- **Averaging** repeated acquisitions without the floor.

The costs are storage (twice the data), a phase image that needs a reference and unwraps
poorly in noise, and pipeline support: not every tool accepts complex input, and the
denoising must be run before the magnitude is taken, which fixes its position in the
pipeline. None of these costs applies at the scanner.

## Worked retrospective cases

Datasets are more often inherited than designed. Three common cases:

**Single shell, b = 1000, 32 directions, one b=0, magnitude only.** DTI is fully
supported and the study should stay there: FA, MD, the principal direction, deterministic
tractography, tract-based statistics. Crossing-fiber models will run (CSD at b = 1000 is
"marginal") but resolve few crossings; kurtosis and compartment models are not possible.
With no reverse-polarity volumes, distortion correction falls to a fieldmap if one exists,
otherwise to registration to the anatomical image. The one b=0 volume limits outlier
detection and eddy's prediction; report motion carefully.

**Two shells, b = 1000 and 2500, 30 directions each, reverse-polarity b=0s.** Kurtosis,
free-water, NODDI, and the spherical mean technique are supported; multi-tissue CSD is
marginal on 30 directions at the top shell and should be run with a lower harmonic order;
MAP-MRI is marginal with two shells. Fit the tensor to the b = 1000 shell only. Distortion
correction with topup is available.

**Full HBCD-style multi-shell with phase, 1.7 mm, both polarities.** Everything in the
table except DSI and IVIM. Denoise in the complex domain first; correct with topup and
eddy using both polarities; fit multi-tissue CSD, kurtosis, MAP-MRI, and the compartment
models; track probabilistically with ACT. The remaining limits are the ones no processing
removes: the fixed diffusion time, the Gaussian assumptions of the models, and the
resolution.

## Measure it: the simulated datasets

:::{admonition} Simulated dataset pending
:class: note
This section will show every cell of the matrix on the `ref-schemes` dataset: the same
simulated brain, five schemes, each model fitted where the rules allow, scored against the truth
maps, so that "marginal" is a number rather than a word.
:::

## What this implies for acquisition

- **Decide the analyses, then read the matrix from the left**; the cheapest scheme that
  says "yes" to all of them is the protocol.
- **Multi-shell with the top shell at b ≥ 2000, 45 or more directions there, reverse
  polarity, and the phase saved** supports every analysis in the table except DSI and
  IVIM, at a scan time under ten minutes ([Chapter 7](../02-diffusion-encoding/07-acquisition-parameters.md)).
- **Record the metadata**: phase-encode direction, readout time, diffusion timing, and
  the gradient coefficient file.

## Further reading

The reviews of {cite:t}`alexander2019` on microstructure and {cite:t}`jeurissen2019` on
tractography, and the cautions of {cite:t}`jones2013`.
