---
title: Notation and units
kernelspec:
  name: python3
  display_name: Python 3
---

Symbols as used throughout the book, with units and the chapter that introduces each.
Vectors are bold; a hat marks a unit vector.

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
| $\Delta k$, $k_\max$ | k-space sampling interval and extent; FOV $= 1/\Delta k$, voxel $= 1/(2k_\max)$ | 1/mm | 2 |
| $N_x$, $N_y$ | matrix size along readout and phase-encode axes | – | 2 |
| $\Delta t_\mathrm{esp}$ | echo spacing of the EPI train | ms | 2 |
| TotalReadoutTime | duration of the EPI train, as recorded in the JSON sidecar | s | 2, 10 |
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
| ADC | apparent diffusion coefficient (measured, depends on b, direction, time) | mm²/s | 4, 22 |
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
| `part-mag_dwi.nii.gz` | the default TRXScan output name pattern (Appendix A) |

## Conventions

- Image arrays in the toy tier are `(rows, columns)` or `(rows, columns, slices)` with rows
  running anterior to posterior; the phase-encode axis is the row axis. A series appends the
  volume axis last.
- b-vectors in the toy tier are expressed in the same `(row, column, slice)` frame as the
  arrays.
- Unless stated otherwise, tissue parameters are the phantom's `adult` preset (Chapter 1):
  T2 of 68 ms (white matter), 76 ms (gray matter), 2000 ms (CSF); white matter
  intra-axonal fraction 0.55 with diffusivity 1.7 × 10⁻³ mm²/s, extra-axonal 1.7 and
  0.6 × 10⁻³, gray matter 0.85 × 10⁻³, CSF 3.0 × 10⁻³.

```{code-cell} python
:tags: [remove-cell]
import dwibook
assert dwibook.__version__
```
