---
title: "Appendix D: Glossary"
kernelspec:
  name: python3
  display_name: Python 3
---

Terms as they are used in this book, with the chapter that introduces each. Every entry
starts with a plain-language sentence; the sentences after it give the technical meaning.

| Term | Meaning | Chapter |
|---|---|---|
| ACT | A rule that keeps tractography anatomically sensible. Anatomically constrained tractography: streamlines are accepted or rejected according to the tissues they start in, pass through, and end in. | 18 |
| ADC | The diffusivity a measurement reports, which need not be the tissue's true one. Apparent diffusion coefficient: estimated from the signal decay, it depends on b-value, gradient direction, and diffusion time. | 4, 21 |
| Aliasing | Parts of the image wrapping around onto the opposite side. It occurs when k-space is sampled too coarsely for the field of view; parallel imaging induces it deliberately and then undoes it with the coils. | 2, 3 |
| AMICO | Software that fits microstructure models such as NODDI quickly. It rewrites the fit as a linear problem over a dictionary of precomputed signals. | 17 |
| Anisotropy | Diffusion that is faster in some directions than in others, as along an axon compared with across it. At the voxel scale it is measured by FA; see also microscopic anisotropy. | 4 |
| b=0 image | An image acquired with the diffusion gradients off, the reference against which diffusion-weighted images are compared. Its contrast is set by T2 and proton density; several are usually acquired per series. | 1, 5 |
| b-tensor encoding | Diffusion weighting in several directions within one measurement. A gradient waveform that changes direction during the encoding gives linear, planar, or spherical weighting, which separates microscopic anisotropy from fiber arrangement. | 22 |
| b-value | How strongly an image is sensitized to diffusion. In s/mm², it grows with the square of the gradient amplitude and of the pulse duration, and with the diffusion time. | 5 |
| b-vector | The direction in which an image is sensitized to diffusion. The unit gradient direction of one volume; it must be rotated with any rotation applied to the image. | 5, 12 |
| Ball | A model compartment in which water diffuses equally in all directions, like CSF. An isotropic Gaussian compartment with a single diffusivity. | 0.2, 17 |
| BIDS | A standard way of naming and organizing neuroimaging files. The Brain Imaging Data Structure: fixed file names and folders, with JSON "sidecar" files that hold the acquisition parameters. | 2 |
| Blip-up/blip-down | Two acquisitions distorted in opposite directions, from which the distortion can be measured. The phase-encode polarity is reversed between them, and the field is estimated from their difference. | 10 |
| bval, bvec files | The two text files that travel with a diffusion series and record how each volume was weighted. The .bval file lists one b-value per volume; the .bvec file lists one gradient direction per volume, as three rows (x, y, z). | 0.3, 19 |
| Compartment | A pool of water with its own diffusion behavior. Intra-axonal, extracellular, and CSF water are the usual compartments; their signals add up in the voxel. | 4, 17 |
| CSD | A method that finds the fiber directions in a voxel by removing the blur that a single fiber's signal adds. Constrained spherical deconvolution: deconvolves a single-fiber response function from the signal to estimate the fiber ODF. | 16 |
| CS-DSI | A faster DSI that measures only a random subset of its points. Compressed-sensing DSI: the missing grid points are recovered with a sparsity prior. | 6 |
| Degeneracy | Different answers fitting the data equally well. Several parameter combinations of a model reproduce the measured signal within the noise, so the fit cannot choose among them. | 17 |
| Diffusion tensor | The simplest description of diffusion in three dimensions: an ellipsoid whose shape and orientation give the diffusivity in every direction. A 3 × 3 symmetric matrix with six unknowns, fitted per voxel in DTI. | 6, 15 |
| Diffusion time | How long molecules are allowed to move before their displacement is measured. Approximately the separation Δ of the two encoding pulses; for OGSE, a fraction of the oscillation period. | 4, 5 |
| Diffusivity | How fast water spreads by random motion. The diffusion coefficient D, in mm²/s or µm²/ms; free water at body temperature has about 3 × 10⁻³ mm²/s. | 4 |
| DIPY | A Python library for diffusion MRI analysis, used in this book's code for tensor fitting and other methods. | 6 |
| DKI | An extension of the tensor that also measures how far the signal decay departs from a simple exponential. Diffusion kurtosis imaging: the kurtosis quantifies the non-Gaussian part of the displacement distribution, seen as curvature of log signal against b. | 15 |
| Dropout | A slice of one volume that is suddenly dark. Signal lost because motion during the diffusion encoding dephased that slice. | 12 |
| DSI | Measuring on a full three-dimensional grid of diffusion encodings, from which the distribution of displacements is computed directly. Diffusion spectrum imaging: Cartesian sampling of q-space and Fourier reconstruction of the propagator. | 6, 16 |
| DTI | Describing each voxel's diffusion with one ellipsoid. Diffusion tensor imaging: a single Gaussian per voxel; the source of FA, MD, and the principal direction. | 15 |
| Echo spacing | The time from one line of k-space to the next in an EPI readout. With the number of lines, it sets the readout time. | 2 |
| Eddy currents | Electric currents that the switching gradients induce in the scanner's metal, whose fields linger and distort the image. They persist into the readout and distort each diffusion-weighted volume differently. | 11 |
| EPI | A fast readout that collects a whole image after one excitation. Echo-planar imaging zigzags through all of k-space in tens of milliseconds, at the cost of sensitivity to field offsets. | 2 |
| Exchange | Water moving between compartments, for instance across a cell membrane. Over diffusion times longer than the exchange time, compartments blend and stop being separable; the simulated brain has none. | 17, 21 |
| FA | How elongated a voxel's diffusion ellipsoid is. Fractional anisotropy, from 0 (the same in every direction) to 1 (along one axis only); it falls both when tissue is less anisotropic and when fibers cross or spread. | 15 |
| Fiber ODF | A function on the sphere that says how much fiber runs in each direction within a voxel. Estimated by CSD; sharper than the diffusion ODF and the input to most tractography. | 16 |
| Fieldmap | A map of how far the magnetic field departs from its nominal value at each point. In Hz; the EPI displacement is computed from it. | 10 |
| Free water | Water that diffuses without obstacles, like CSF. A compartment with the free diffusivity; a nuisance in tissue fits and, as the free-water fraction, a model output. | 17 |
| FSL (topup, eddy) | A widely used neuroimaging software suite. Its topup tool estimates the susceptibility field from blip-up/blip-down pairs, and its eddy tool corrects eddy currents and motion. | 10, 11 |
| Gaussian diffusion | Diffusion whose displacements follow a bell curve, as for free water or a single tensor. The signal then decays as one exponential in b; barriers make diffusion non-Gaussian. | 4, 15 |
| Gibbs ringing | Ripples beside sharp edges in the image. Caused by the finite extent of k-space: the high spatial frequencies that were not measured. | 2, 9 |
| Gradient echo | An echo formed by reversing a gradient rather than by a 180° pulse. It undoes only what the gradient did, so static field differences keep dephasing the spins and the signal decays with T2*; an EPI readout is a train of gradient echoes. | 1 |
| Gradient nonlinearity | The gradient coil's field not being perfectly linear, most of all far from the center of the scanner. It warps the image and alters the local b-value and direction (the b-matrix). | 13 |
| graddev | A per-voxel correction for gradient nonlinearity. The gradient-deviation tensor used to correct the b-matrix. | 13 |
| GRAPPA | Parallel imaging that fills in skipped lines of k-space. Missing lines are predicted from neighboring lines across coils, with weights fitted on fully sampled calibration lines. | 3 |
| HARDI | Many diffusion directions at one strong b-value, for resolving fiber directions. High angular resolution diffusion imaging. | 6 |
| Hindered diffusion | Diffusion slowed by obstacles that molecules can go around, as between axons. Displacement keeps growing with time, more slowly than for free water; the ADC is lower than free and changes little with diffusion time. | 4 |
| Isocenter | The center of the magnet, where the gradients are most linear. | 13 |
| k-space | The form in which the scanner records an image: as amounts of each spatial frequency, each a stripe pattern across the image. Each sample is one Fourier coefficient; the image is the inverse Fourier transform of k-space. | 2 |
| Larmor frequency | The rate at which spins precess, proportional to the magnetic field. 42.58 MHz per tesla for hydrogen, about 128 MHz at 3 T; gradients make it depend on position. | 1 |
| Magnetization | The net magnetism of the water in a voxel, drawn as one arrow; the quantity MRI measures. At rest it points along the main field; an RF pulse tips it, and its transverse part generates the signal. | 1 |
| MAP-MRI | A flexible mathematical description of the displacement distribution. Mean apparent propagator MRI: a basis representation from which return-to-origin probability, mean squared displacement, and non-Gaussianity are computed. | 15 |
| MD | The average diffusivity over all directions. Mean diffusivity: the mean of the three tensor eigenvalues. | 15 |
| Mean squared displacement | The average of the squared distance molecules move in a given time. It grows as 2Dt per axis for free diffusion, more slowly when hindered, and levels off when restricted. | 4 |
| Microscopic anisotropy | How elongated the individual water compartments are, whether or not they are aligned with each other. A voxel of randomly oriented axons has an FA near zero but high microscopic anisotropy; b-tensor encoding measures it. | 15, 22 |
| MP-PCA | A denoising method that separates signal from noise by treating the image series as a matrix. Marchenko-Pastur PCA: principal components whose eigenvalues fall within the spectrum expected of pure noise are removed. | 8 |
| MRtrix3 | Diffusion MRI software used for denoising (dwidenoise), unringing (mrdegibbs), CSD, and tractography. | 9, 14 |
| MSMT-CSD | CSD that accounts separately for white matter, gray matter, and CSF. Multi-shell multi-tissue CSD: one response function per tissue, told apart by their different decay across shells. | 16 |
| Multi-echo | A diffusion acquisition with several echo times per excitation: several images are read out after one excitation, each at a later echo time. The later echoes are gradient echoes after a single spin echo, so they add T2* weighting; compare multi-TE. | 20 |
| Multi-shell | A diffusion scheme with directions at two or more b-values. Needed by any model of non-Gaussian diffusion or of several compartments. | 6 |
| Multi-shot EPI | EPI that splits the lines of k-space over several excitations. Each readout is shorter, reducing distortion and blur, but each shot has its own phase, which must be measured (navigator) or estimated to avoid ghosts. | 2, 22 |
| Multi-TE | A diffusion acquisition at several echo times, one echo time per excitation; the echo time changes between volumes or between scans, and each image has its own spin echo. It separates compartments by their T2; compare multi-echo. | 20 |
| Multiband | Exciting and reading several slices at once. Simultaneous multi-slice: the coils separate the slices; it shortens TR. | 7, 12 |
| Navigator | A short extra readout that measures something about each shot, typically its phase, so the reconstruction can correct it. In multi-shot diffusion imaging, a low-resolution image of the center of k-space acquired after each shot. | 22 |
| Neurite density | The share of a voxel's signal that comes from inside axons and dendrites (neurites). A compartment-model output (NODDI's ICVF); a signal fraction at the scan's echo time, not a volume fraction. | 17, 20 |
| NODDI | A widely used compartment model of tissue: neurites, the space around them, and free water. It reports neurite density, orientation dispersion, and free-water fraction, with the diffusivities fixed. | 17 |
| Noise floor | The positive signal that magnitude images show even where there is no tissue. The mean of magnitude noise; it biases low-SNR measurements upward. | 3, 8 |
| ODF | A function on the sphere whose peaks point along fiber directions. Orientation distribution function: a diffusion ODF (from the signal) or a fiber ODF (after deconvolution). | 16 |
| Off-resonance | Spins precessing at a slightly different frequency from the one the scanner expects. Caused by field offsets such as susceptibility; in EPI it displaces signal along the phase-encode axis. | 1 |
| OGSE | Diffusion encoding with oscillating gradients, to measure displacements over only a few milliseconds. Oscillating-gradient spin echo, usually with cosine (or apodized-cosine) modulation, so that each oscillation winds the phase up and back down around zero; the effective diffusion time is then about a quarter of the oscillation period rather than the pulse separation, and the b-values reached are low. | 21 |
| Orientation dispersion | How spread out the directions of the fibers in a voxel are. Zero for perfectly parallel fibers; it lowers FA without any change in the fibers themselves (NODDI's ODI). | 15, 17 |
| Partial Fourier | Skipping part of k-space on one side and filling it in from the other by symmetry. It shortens the echo time; it costs some SNR and sharpness. | 2, 3 |
| Partial volume | A voxel containing more than one tissue, so that its signal is a mixture. Larger voxels have more of it, especially at boundaries such as the cortex and the ventricle walls. | 7 |
| PGSE | The standard diffusion encoding: two gradient pulses on either side of a 180° pulse. Pulsed-gradient spin echo. | 5, 21 |
| Phase-encode direction | The image axis that EPI samples slowly, along which field offsets displace signal. | 2, 10 |
| Propagator | The distribution of how far molecules move over the diffusion time. The probability of each displacement; the diffusion signal is its Fourier transform. | 4 |
| Proton density | How much MRI-visible hydrogen, mostly in water, a tissue contains. It scales the signal before any relaxation; CSF has the most. | 1 |
| q | How finely a diffusion measurement resolves displacement. q = γGδ / 2π, in cycles per mm: a molecule that moves 1/q along the gradient gains one full turn of phase. | 6, 21 |
| q-space | The space of all possible diffusion encodings, direction and strength. A sampling scheme is a set of points in it; the distance from the origin grows as √b. | 6 |
| QSIPrep | A pipeline that assembles preprocessing steps from several packages into one reproducible workflow. | 12, 14 |
| Quadrature demodulation | How the scanner turns the coil's oscillating voltage into one complex number per sample. The voltage is mixed with two references 90° apart, and the results become the real and imaginary parts. | 1 |
| Readout time | How long the EPI readout lasts. Two quantities go by this name: the duration of the EPI train (which sets the echo time and the T2* blur), and the total readout time that sets the susceptibility displacement, defined by BIDS as the effective echo spacing times (phase-encode matrix − 1), which shrinks with parallel imaging and is unchanged by partial Fourier. | 2, 10 |
| Response function | The signal a voxel of one perfectly aligned fiber population would give. Estimated from single-fiber voxels and deconvolved by CSD; one per tissue in MSMT-CSD. | 16 |
| RESTORE | A tensor fit that ignores outlier measurements. It iteratively down-weights measurements that disagree with the fit, such as slices with dropout. | 15 |
| Restricted diffusion | Diffusion inside a closed space, such as across an axon or within a cell. Displacement stops growing once molecules reach the walls, and the ADC then falls with diffusion time. | 4 |
| RF | Radio frequency: the band at which spins precess, and at which the scanner transmits pulses and receives the signal. An RF pulse at the Larmor frequency tips the magnetization. | 1 |
| Rician | The shape of the noise in a magnitude image, which is not centered on zero where the signal is weak. The magnitude of complex Gaussian noise from a single coil; the source of the noise floor. | 3 |
| SENSE | Parallel imaging that unfolds a deliberately aliased image using how strongly each coil sees each point. Unfolding in the image domain with the coil sensitivities. | 3 |
| Shell | A set of diffusion directions at one b-value. | 6 |
| SIFT2 | Weighting streamlines so that their density agrees with the fiber ODFs. It makes streamline counts more quantitative. | 18 |
| Signal fraction, volume fraction | What a model reports versus what is in the tissue. A compartment's signal fraction is its share of the measured signal, weighted by its T2 decay at the echo time; it equals the volume fraction only at TE = 0, with equal proton density, and with equal T1 recovery between excitations (or a TR long enough for full recovery). | 20 |
| Simulated datasets | The data this book uses in place of real scans. The toy-tier series built in the pages and the pipeline-tier series produced offline by TRXScan ([Appendix A](./a-trxscan-cookbook.md#app-a-datasets)). | 0.2 |
| SNR | How strong the signal is relative to the noise. The signal divided by the noise standard deviation of one channel. | 8 |
| Spherical harmonics | Smooth patterns on a sphere, used like a Fourier series to describe how something varies with direction. The basis in which ODFs and signals on a shell are represented; the maximum order sets the angular detail. | 16 |
| Spherical mean | The average signal over all directions of a shell. It removes the effect of fiber orientation. | 17 |
| Spin echo | Signal that returns when a 180° pulse reverses the spreading of the spins' phases. It refocuses dephasing from static field differences, recovering the signal that T2* decay would lose; the basis of diffusion sequences. | 1 |
| Stick | A model compartment in which water moves only along one direction, standing for the inside of an axon. Zero diffusivity across, one diffusivity along; the intra-axonal compartment of the simulated brain. | 0.2, 17 |
| Streamline | One curve traced by tractography through the fiber directions. A path, not an axon; a set of streamlines is a tractogram. | 18 |
| Susceptibility distortion | Warping of EPI images near air-tissue boundaries. Displacement along the phase-encode axis caused by the field offsets there. | 10 |
| T1 | How quickly the magnetization grows back along the main field after a pulse. The longitudinal relaxation time; it sets how much signal is available when TR is short. | 1 |
| T2 | How quickly the signal decays because the spins' phases drift apart through their interactions. The transverse relaxation time; the loss a spin echo cannot recover. | 1 |
| T2* | How quickly the signal decays in practice, when static field differences across the voxel add to T2. Transverse decay including static inhomogeneity; shorter than T2; a spin echo recovers the extra part at its echo time. | 1, 20 |
| TE, TR | The echo time, from excitation to the center of the readout; the repetition time, between successive excitations of the same slice. | 1, 7 |
| TORTOISE | Diffusion preprocessing software with its own corrections for distortion, motion, and gradient nonlinearity. | 10 |
| Tortuosity | How much obstacles lengthen the path water must take. In models, a rule that ties the extra-axonal radial diffusivity to the intra-axonal fraction: the more densely axons are packed, the slower diffusion across them. | 17 |
| Tractogram | The set of streamlines from one tractography run. | 16, 18 |
| Tractography | Tracing curves through the voxel-wise fiber directions to reconstruct pathways. It generates streamlines from local fiber orientations. | 18 |
| Truth maps | The known answers the simulator writes alongside its data. The 27 analytic microstructure maps TRXScan writes for the simulated brain ([Appendix E](./e-truth-map-catalogue.md)). | 0.2 |
| Unringing | Removing Gibbs ringing. The image is resampled at sub-voxel shifts chosen to minimize the ripples. | 9 |
| Voxel | The three-dimensional pixel of an MRI image, typically 1–2.5 mm on a side for diffusion. Its signal is the sum of everything inside it. | 1 |
| Zeppelin | A model compartment in which water diffuses faster along one axis than across it, standing for the space around axons. A cylindrically symmetric tensor. | 17 |

```{code-cell} python
:tags: [remove-cell]
import dwibook
assert dwibook.__version__
```
