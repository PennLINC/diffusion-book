---
title: "0.3 Notation and units"
kernelspec:
  name: python3
  display_name: Python 3
---

Terms and symbols as used throughout the book, with units and the chapter that introduces
each. Vectors are bold; a hat marks a unit vector. [Appendix D](../appendices/d-glossary.md) is the full glossary.

## Terms

Acquisition terms that recur in the book, each in one plain sentence. The chapter column
says where the term is explained properly.

| Term | In plain words | Chapter |
|---|---|---|
| proton density | how much MR-visible hydrogen a tissue holds, relative to pure water; it scales the signal before any decay | 1 |
| spin echo, gradient echo | two ways of forming the signal peak that is recorded; a spin echo undoes the dephasing from field differences, a gradient echo does not | 1 |
| b=0 image | an image taken with no diffusion weighting; its contrast comes mainly from proton density and T2 | 1, 5 |
| k-space | the grid of spatial-frequency samples the scanner actually records; an image is computed from it | 2 |
| FOV (field of view) | the width of the region the image covers | 2 |
| readout axis, phase-encode axis | the two in-plane image axes: along the readout, samples are taken within one fast sweep; along phase-encode, line by line, much more slowly, so that this axis collects the distortions | 2 |
| EPI (echo-planar imaging) | the fast readout that records a whole slice's k-space after one excitation, used for nearly all diffusion MRI | 2 |
| partial Fourier | skipping part of one side of k-space to shorten the echo time, at some cost in sharpness | 2 |
| parallel imaging: GRAPPA, SENSE | skipping k-space lines to shorten the readout and filling them in from the differences between receive coils | 3 |
| ACS (autocalibration signal) | a small fully sampled block at the center of k-space from which GRAPPA learns how to fill in the skipped lines | 3 |
| g-factor | the extra, position-dependent noise amplification that parallel imaging adds | 3 |
| multiband | exciting and recording several slices at once, to shorten the time per volume | 7 |

## Physics and encoding

| Symbol | Meaning | Unit | Chapter |
|---|---|---|---|
| $B_0$ | static magnetic field | T | 1 |
| $\gamma$, $\gamma/2\pi$ | gyromagnetic ratio of the proton; 42.58 MHz/T | rad/(s T), MHz/T | 1 |
| $f_0$, $\omega_0$ | Larmor frequency | Hz, rad/s | 1 |
| $\Delta f$, $\Delta\omega$ | off-resonance (field offset) | Hz, rad/s | 1, 10 |
| $M_0$, $M_z$, $M_{xy}$ | equilibrium, longitudinal, and transverse magnetization | arbitrary | 1 |
| $T_1$, $T_2$, $T_2^*$, $T_2'$ | relaxation times; $1/T_2^* = 1/T_2 + 1/T_2'$ | ms | 1 |
| TE, TR | echo time, repetition time | ms, s | 1, 7 |
| $\mathbf{G}$, $G$ | gradient vector and amplitude | mT/m | 2, 5 |
| $\mathbf{k}$, $k_x$, $k_y$ | spatial frequency; k-space coordinates | 1/mm | 2 |
| $\Delta k$, $k_{\mathrm{max}}$ | k-space sampling interval and extent; FOV $= 1/\Delta k$, voxel $= 1/(2k_{\mathrm{max}})$ | 1/mm | 2 |
| $N_x$, $N_y$ | matrix size along readout and phase-encode axes | – | 2 |
| $\Delta t_\mathrm{esp}$ | echo spacing of the EPI train | ms | 2 |
| TotalReadoutTime | effective readout time recorded in the JSON sidecar: the effective echo spacing $\Delta t_\mathrm{esp}/R$ times ($N_y - 1$), with $N_y$ the reconstructed phase-encode matrix; shortened by $R$ but not by partial Fourier, so it differs from the duration of the EPI train | s | 2, 10 |
| $R$ | in-plane (parallel imaging) acceleration factor | – | 3 |
| $\sigma$ | noise standard deviation of one channel | signal units | 3, 8 |
| $L$ | number of receive coils | – | 3 |
| $\delta$, $\Delta$ | diffusion pulse duration and separation | ms | 5 |
| $b$ | diffusion weighting, $\gamma^2 G^2 \delta^2 (\Delta - \delta/3)$ | s/mm² | 5 |
| $\mathbf{q}$, $q$ | q-vector, $\gamma G \delta / 2\pi$ | 1/mm | 4, 5 |
| $\hat{\mathbf{g}}$ | gradient direction (b-vector) | unit | 5 |
| $\theta$ | angle between gradient and fiber | degrees | 5 |
| $S$, $S_0$ | signal with and without diffusion weighting | arbitrary | 5 |
| $P(\mathbf{r}, t)$ | displacement distribution (propagator) | 1/mm³ | 4 |

## Tissue and models

| Symbol | Meaning | Unit | Chapter |
|---|---|---|---|
| $D$ | diffusion coefficient; free water 3 × 10⁻³ mm²/s = 3 µm²/ms | mm²/s | 4 |
| ADC | apparent diffusion coefficient (measured, depends on b, direction, time) | mm²/s | 4, 21 |
| $D_\parallel$, $D_\perp$ | diffusivity along and across a fiber | mm²/s | 5 |
| $f$ | volume or signal fraction of a compartment (intra-axonal unless stated) | – | 17 |
| $\mathbf{D}$ | diffusion tensor | mm²/s | 15 |
| $\lambda_1 \ge \lambda_2 \ge \lambda_3$ | tensor eigenvalues | mm²/s | 15 |
| MD, FA, AD, RD | mean diffusivity, fractional anisotropy, axial and radial diffusivity | mm²/s, – | 15 |
| MK, AK, RK | mean, axial, radial kurtosis | – | 15 |
| RTOP, MSD, NG | return-to-origin probability, mean squared displacement, non-Gaussianity | 1/mm³, mm², – | 15 |
| ODF, fODF | diffusion and fiber orientation distribution functions | – | 16 |
| $J$ | gradient-deviation tensor from gradient nonlinearity | – | 13 |
| $\phi(\mathbf{r})$ | apparent position under gradient nonlinearity, $\mathbf{r} + \mathbf{d}(\mathbf{r})$ | mm | 13 |
| SNR | signal-to-noise ratio, signal / $\sigma$ | – | 8 |

## Data and files

| Term | Meaning |
|---|---|
| `part-mag`, `part-phase` | BIDS magnitude and phase images of a complex series |
| `.bval`, `.bvec` | FSL-layout gradient table: one row of b-values, three rows of unit-vector components |
| `PhaseEncodingDirection`, `TotalReadoutTime` | sidecar fields that distortion correction reads |
| AP, PA | anterior-posterior and posterior-anterior phase-encode polarities |
| `part-mag_dwi.nii.gz` | the default TRXScan output name pattern ([Appendix A](../appendices/a-trxscan-cookbook.md)) |

## Conventions

- Image arrays in the toy tier are `(rows, columns)` or `(rows, columns, slices)` with rows
  running anterior to posterior; the phase-encode axis is the row axis. A series appends the
  volume axis last.
- b-vectors in the toy tier are expressed in the same `(row, column, slice)` frame as the
  arrays.
- Unless stated otherwise, tissue parameters are the simulated brain's `adult` preset ([Chapter 1](../01-mri-physics/01-spins-and-signal.md)):
  T2 of 68 ms (white matter), 76 ms (gray matter), 2000 ms (CSF); white matter
  intra-axonal fraction 0.55 with diffusivity 1.7 × 10⁻³ mm²/s, extra-axonal 1.7 and
  0.6 × 10⁻³, gray matter two components (80 % at 0.85 × 10⁻³ and a 20 % cell-body
  component at 0.3 × 10⁻³), CSF 3.0 × 10⁻³.

```{code-cell} python
:tags: [remove-cell]
import dwibook
assert dwibook.__version__
```
