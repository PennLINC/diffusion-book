---
title: "Appendix E: Ground-truth map catalogue"
subtitle: The 27 maps, their definitions, and where each is used
kernelspec:
  name: python3
  display_name: Python 3
---

`trxscan-microstructure` evaluates closed-form expressions for 27 microstructure scalars
from the same per-voxel mixture of stick, tensor, and ball compartments that the simulator
draws its signal from, so the maps are the exact answer for a noise-free, artifact-free
acquisition of the simulated brain. The expressions are ports of dipy's implementations and are
validated against dipy at 10⁻⁸, so a dipy fit of the simulated data can be compared with
them directly. Because the compartments are Gaussian, the maps are long-diffusion-time
quantities ([Chapter 22](../05-advanced/22-multi-diffusion-time.md)), and the NODDI-style fractions are the simulated brain's own compartment
fractions rather than a fit.

```{code-cell} python
:tags: [hide-cell]
from dwibook import truth
```

| Family | Map | Definition | Used in |
|---|---|---|---|
| DTI | `fa` | fractional anisotropy of the voxel's mean tensor | 8–15 |
| DTI | `md` | mean diffusivity (mm²/s) | 8–15 |
| DTI | `rd` | radial diffusivity | 15 |
| DTI | `ad` | axial diffusivity | 15 |
| DKI | `mk` | mean kurtosis | 15 |
| DKI | `ak` | axial kurtosis | 15 |
| DKI | `rk` | radial kurtosis | 15 |
| DKI | `mkt` | mean kurtosis tensor | 15 |
| DKI | `kfa` | kurtosis fractional anisotropy | 15 |
| QTI | `micro_fa` | microscopic FA: anisotropy of the compartments regardless of their arrangement | 15, 23 |
| QTI | `coherence` | orientation coherence of the compartments | 23 |
| QTI | `k_bulk` | isotropic (bulk) kurtosis: variance of compartment mean diffusivities | 15, 23 |
| QTI | `k_shear` | anisotropic (shear) kurtosis | 15, 23 |
| MAP-MRI | `rtop` | return-to-origin probability (1/mm³) | 15 |
| MAP-MRI | `rtap` | return-to-axis probability | 15 |
| MAP-MRI | `rtpp` | return-to-plane probability | 15 |
| MAP-MRI | `msd` | mean squared displacement (mm²) at the given diffusion time | 15 |
| MAP-MRI | `qiv` | q-space inverse variance | 15 |
| MAP-MRI | `ng` | non-Gaussianity | 15 |
| MAP-MRI | `ngpar` | non-Gaussianity parallel to the principal direction | 15 |
| MAP-MRI | `ngperp` | non-Gaussianity perpendicular to it | 15 |
| MAP-MRI | `pa` | propagator anisotropy | 15 |
| ODF | `gfa` | generalized FA of the diffusion ODF | 16 |
| ODF | `qa` | quantitative anisotropy of the ODF peaks | 16 |
| NODDI-style | `icvf` | intra-cellular (intra-axonal) volume fraction of the tissue | 17 |
| NODDI-style | `odi` | orientation dispersion index of the fiber mixture | 17 |
| NODDI-style | `isovf` | isotropic (CSF) volume fraction | 17 |

The MAP-MRI maps come out in physical units when the pipeline passes the diffusion timing
(`--big-delta 0.030 --small-delta 0.010`, [Appendix A](./a-trxscan-cookbook.md)); without it they are in dipy's
normalized units and only their contrast is meaningful.

## Truth peaks

Separately from the scalar maps, `trxscan --truth-peaks` writes up to three fiber
orientations per acquisition voxel, each a unit vector scaled by its share of the voxel's
fiber mass, as a nine-volume image. [Chapter 16](../04-modeling/16-fiber-orientation.md) scores estimated peaks against them.

## Loading the maps

The names are available programmatically and the loader reads them from any dataset that
carries a truth run:

```{code-cell} python
:tags: [hide-input]
for family, names in truth.TRUTH_MAPS.items():
    print(f"{family:<12} {', '.join(names)}")
print(f"\n{len(truth.ALL_TRUTH_MAPS)} maps; load with truth.load_truth(dataset, prefix) and compare with truth.summarize_error(fit, truth_map, mask)")
```
