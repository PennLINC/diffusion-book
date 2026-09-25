---
title: "Appendix D: Glossary"
kernelspec:
  name: python3
  display_name: Python 3
---

Terms as they are used in this book, with the chapter that introduces each.

| Term | Meaning | Chapter |
|---|---|---|
| ACT | Anatomically constrained tractography: streamlines are accepted or rejected by the tissue they start and end in. | 18 |
| ADC | Apparent diffusion coefficient: the diffusivity a measurement reports, which depends on b-value, direction, and diffusion time. | 4, 22 |
| Aliasing | Wrap-around of the image when k-space is sampled too coarsely; deliberately induced and undone by parallel imaging. | 2, 3 |
| b-value | The strength of diffusion weighting, in s/mm², set by gradient amplitude, pulse duration, and pulse separation. | 5 |
| b-tensor encoding | Diffusion weighting along several axes within one measurement (linear, planar, spherical), which separates microscopic anisotropy from fiber arrangement. | 23 |
| b-vector | The unit direction of the diffusion weighting for one volume; must be rotated with any rotation applied to the image. | 5, 12 |
| Blip-up/blip-down | A pair of acquisitions with opposite phase-encode polarity, distorted in opposite directions, from which the field is estimated. | 10 |
| CSD | Constrained spherical deconvolution: estimates the fiber ODF by deconvolving a single-fiber response from the signal. | 16 |
| CS-DSI | Compressed-sensing DSI: a random subset of the DSI grid reconstructed with a sparsity prior. | 6 |
| Compartment | A pool of water with its own diffusion behavior (intra-axonal, extracellular, CSF), summed in the voxel signal. | 4, 17 |
| Degeneracy | The situation in which several parameter combinations of a model fit the data equally well. | 17 |
| Diffusion time | The interval over which displacement is measured, approximately the separation of the two encoding pulses. | 4, 5 |
| DKI | Diffusion kurtosis imaging: a representation of the non-Gaussian part of the signal decay. | 15 |
| Dropout | Loss of signal in one slice of one volume from motion during the diffusion encoding. | 12 |
| DSI | Diffusion spectrum imaging: Cartesian sampling of q-space and Fourier reconstruction of the displacement distribution. | 6, 16 |
| DTI | Diffusion tensor imaging: a single Gaussian ellipsoid per voxel; the source of FA, MD, and the principal direction. | 15 |
| Echo spacing | Time between successive lines of an EPI readout. | 2 |
| Eddy currents | Fields induced by the switching diffusion gradients that persist into the readout and distort each volume differently. | 11 |
| EPI | Echo-planar imaging: a readout that collects all of k-space after one excitation. | 2 |
| FA | Fractional anisotropy: how elongated the diffusion tensor is, from 0 to 1. | 15 |
| Fieldmap | A map of the static field offset in Hz, from which the EPI displacement is computed. | 10 |
| Free water | The CSF-like compartment with unrestricted diffusion; a nuisance in tissue fits and a model output. | 17 |
| Gibbs ringing | Ripples next to sharp edges caused by the finite extent of k-space. | 2, 9 |
| Gradient nonlinearity | Departure of a gradient coil's field from linearity, which warps the image and alters the local b-matrix. | 13 |
| graddev | The per-voxel gradient-deviation tensor used to correct the b-matrix for gradient nonlinearity. | 13 |
| GRAPPA | Parallel imaging in k-space: missing lines are predicted from neighboring lines across coils using weights fitted on calibration lines. | 3 |
| HARDI | High angular resolution diffusion imaging: many directions at one high b-value. | 6 |
| Isocenter | The center of the magnet, where the gradients are most linear. | 13 |
| k-space | The Fourier domain in which the scanner records the signal. | 2 |
| MAP-MRI | Mean apparent propagator MRI: a basis representation of the displacement distribution. | 15 |
| MD | Mean diffusivity: the average of the three tensor eigenvalues. | 15 |
| MP-PCA | Marchenko-Pastur PCA denoising: removes noise components identified by their random-matrix spectrum. | 8 |
| MSMT-CSD | Multi-shell multi-tissue CSD: one response per tissue, separated by the decay across shells. | 16 |
| Multiband | Simultaneous excitation of several slices, separated by the coils; shortens TR. | 7, 12 |
| NODDI | A compartment model reporting neurite density, orientation dispersion, and free-water fraction, with fixed diffusivities. | 17 |
| Noise floor | The positive mean of a magnitude image where there is no signal; biases low-SNR measurements. | 3, 8 |
| ODF | Orientation distribution function: a function on the sphere whose peaks are fiber directions (diffusion ODF or fiber ODF). | 16 |
| OGSE | Oscillating-gradient spin echo: reaches short diffusion times. | 22 |
| Partial Fourier | Skipping part of k-space on one side and reconstructing it from the other by symmetry. | 2, 3 |
| PGSE | Pulsed-gradient spin echo: the standard diffusion encoding, two gradient pulses around a 180° pulse. | 5 |
| Phase-encode direction | The image axis along which EPI is sampled slowly and along which off-resonance displaces signal. | 2, 10 |
| Propagator | The displacement distribution of water molecules over the diffusion time. | 4 |
| q-space | The space of diffusion encodings (direction and magnitude); a scheme is a set of points in it. | 4, 6 |
| Quadrature demodulation | Mixing the coil voltage with two references 90° apart; the origin of the complex signal. | 1 |
| Readout time | Duration of the EPI train; sets the susceptibility displacement. | 2, 10 |
| RESTORE | A tensor fitting method that down-weights outlier measurements. | 15 |
| Rician | The distribution of a magnitude image's noise for a single coil. | 3 |
| SENSE | Parallel imaging in the image domain: aliased pixels are unfolded using coil sensitivities. | 3 |
| Shell | A set of directions at one b-value. | 6 |
| SIFT2 | Streamline weighting that makes streamline density consistent with the fiber ODFs. | 18 |
| Simulated datasets | The toy-tier series built in the pages and the pipeline-tier series produced offline by TRXScan ([Appendix A](./a-trxscan-cookbook.md#app-a-datasets)). | 0.2 |
| SNR | Signal-to-noise ratio, here the signal divided by the noise standard deviation of one channel. | 8 |
| Spherical mean | The average of the signal over all directions of a shell; removes orientation effects. | 17 |
| Spin echo | Refocusing of static dephasing by a 180° pulse; the basis of diffusion sequences. | 1 |
| Susceptibility distortion | Displacement along the phase-encode axis caused by field offsets near air-tissue interfaces. | 10 |
| T1, T2, T2* | Longitudinal relaxation time; transverse relaxation time; transverse decay including static inhomogeneity. | 1 |
| TE, TR | Echo time; repetition time. | 1, 7 |
| Tractography | Generation of streamlines from local fiber orientations. | 18 |
| Truth maps | The 27 analytic microstructure maps TRXScan writes for the simulated brain ([Appendix E](./e-truth-map-catalogue.md)). | 0.2 |
| Unringing | Removal of Gibbs ringing by sub-voxel shifts. | 9 |

```{code-cell} python
:tags: [remove-cell]
import dwibook
assert dwibook.__version__
```
