---
title: Gradient nonlinearity
subtitle: Chapter 13
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Stub
:class: note
This chapter is a placeholder from the Phase 1 skeleton. Its learning goals and section
structure are final; the content is not written yet.
:::

## Learning goals

After this chapter you can:

- explain why a gradient coil's field is not linear and write the two consequences: a spatial warp of the image and a per-voxel error in the diffusion encoding
- predict how both grow with distance from the isocentre and differ between whole-body and high-performance (Connectom-class) gradient systems
- apply the geometric correction (gradwarp) with `gradunwarp` or TORTOISE, including the intensity Jacobian, and place it correctly relative to susceptibility-distortion correction
- apply the encoding correction with a per-voxel gradient-deviation tensor (`graddev`) in tensor, ODF and microstructure fits
- score both corrections against the simulator's ground-truth displacement field and gradient-deviation image

**Datasets used:** `gnl`, `truth`
**Simulation tier:** phantom

## The physics

A gradient coil for axis $a$ produces a field whose spatial dependence is a solid-harmonic
expansion. The $l = 1$ term is the nominal linear gradient; the higher-order terms are the
nonlinearity, and they grow as $(r/R_0)^l$ with distance $r$ from the isocentre. Writing the
apparent position as $\phi(r) = r + d(r)$, two things happen to a spin at true position $r$:

1. **Spatial encoding error.** The scanner places it at $\phi(r)$, so the image is warped
   (compressed toward the isocentre) and its intensity is scaled by $1/|\det \nabla\phi|$.
2. **Diffusion encoding error.** The gradient it experiences is $G_\mathrm{eff}(r) = J(r)^\top g$
   with $J_{ij} = \partial\phi_i/\partial x_j$, so both the b-vector direction and the b-value
   deviate voxel by voxel. FA and MD are biased far from the isocentre even after the image has
   been unwarped.

*To be written: the expansion, the size of the effect for the whole-body and Connectom presets,
and why head-only and high-performance gradient systems are worse.*

## The TRXScan flags

`--gnl whole-body-80 | connectom-300 | <coefficients.grad>` adds both effects from one synthetic
coefficient set; `--gnl-scale` is the severity knob, `--isocenter` places the scanner origin,
and `--gnl-no-warp` / `--gnl-no-encoding` switch off one effect at a time so each correction
can be scored on its own. The run writes the truth alongside the DWI: the coefficient file
(`_desc-gnl_coeff.grad`, the Siemens grammar that TORTOISE and qsiprep read), the displacement
field (`_desc-gnl_disp.nii.gz`, mm) and the gradient-deviation image (`_desc-gnl_graddev.nii.gz`,
nine volumes in the HCP layout).

## The artifact-free reference

*To be written.*

## Correction step by step: gradwarp

*To be written: unwarping with `gradunwarp` (HCP) and with TORTOISE's
`CreateNonlinearityDisplacementMap` as qsiprep runs it; the Jacobian intensity modulation;
resampling once by composing the gradwarp and susceptibility fields; frame conventions and
the isocentre.*

## Correction step by step: per-voxel b-matrix (graddev)

*To be written: computing $J$ per voxel (`CreateGradientNonlinearityBMatrix`, the qsiprep
`graddev` output); using it in fits (FSL `dtifit --gradnonlin` and `bedpostx -g`, a per-voxel
gradient table in dipy, `odx graddev` for ODFs); why it must be evaluated in the final,
resampled frame.*

## Where it sits in the pipeline

*To be written: order relative to denoising, unringing, susceptibility and eddy correction;
what qsiprep does with a coefficient file; what happens if only one of the two corrections is
applied.*

## Residual error versus truth

*To be written: displacement error against `_desc-gnl_disp`, $J$ against `_desc-gnl_graddev`,
FA/MD bias as a function of distance from the isocentre before and after each correction, and
angular error of the peaks against the truth peaks.*

## What this implies for acquisition

*To be written: position the head near the isocentre; know the gradient system and obtain its
coefficient file; head-insert and high-performance gradients trade linearity for strength.*

## Further reading

*To be written.*

```{code-cell} python
:tags: [remove-cell]
# Build-time smoke test: every chapter executes at least one cell.
import dwibook
assert dwibook.__version__
```
