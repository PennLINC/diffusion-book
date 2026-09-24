# Diffusion MRI: an executable book — outline

Status: stage 1 (outline + plan), 2026-09-24. Companion: [implementation-plan.md](implementation-plan.md).

## Premise

Every figure in the book is produced by code the reader can run. Realistic brain-level data
come from **TRXScan** (a headless Fiberfox-style simulator: tractogram + tissue maps + gradient
scheme + fieldmap → BIDS complex 4D DWI with acquisition artifacts) and its companion
`trxscan-microstructure`, which writes the **analytic ground truth** for the same phantom. Because
the simulator and the answer key are derived from one per-voxel mixture, every preprocessing and
modeling chapter can end with a quantitative "fit vs. truth" comparison rather than a qualitative
picture.

Two kinds of simulation feed the book, and each chapter says which it uses:

| Tier | What | Where it runs | Used for |
|---|---|---|---|
| **Toy** | small numpy/scipy simulations written in the notebook (Bloch equations, Shepp–Logan k-space, random walks, single-voxel signal models) | at book build time, seconds | physics intuition, anything TRXScan cannot express (diffusion time, TR, pulse sequences) |
| **Phantom** | TRXScan runs on a real-anatomy phantom (tissue maps + 1M-streamline ACT+SIFT2 tractogram) | offline pipeline, minutes–hours; outputs versioned and fetched at build time | realistic artifacts, preprocessing walkthroughs, model fits vs. truth, tractography |

Phantom runs are never executed at build time. Notebooks load pre-simulated outputs (see the
plan for the data pipeline), so the book builds in minutes on CI.

## Conventions used throughout

- Notation table (b, q, Δ, δ, G, γ, TE, TR, D, S₀, ADC, FA, …) in the front matter; every chapter links to it.
- Every chapter has: *Learning goals* → *Physics* → *See it* (figure code) → *Measure it* (a number against truth where possible) → *What this implies for acquisition* → *Further reading*.
- All phantom figures use one phantom, `sub-0001a`, so readers learn one brain. A second subject (NIBS `sub-60501`, with a measured fieldmap and real motion traces) appears only in the distortion, eddy, and motion chapters, where a real field or trace matters.
- Simulated "raw" data are always shown as the scanner would give them: BIDS `part-mag`/`part-phase` NIfTI + `.bval`/`.bvec` + JSON sidecar.

---

## Part 0 — Front matter

### 0.1 How to read and run this book
Executable cells, the data download, the environment, how to reproduce the offline simulations.

### 0.2 The phantom
What `sub-0001a` is (tissue probability maps, tractogram with SIFT2 weights, fieldmap), what TRXScan does to it, what the ground-truth maps are and what they are *not* (noise-free, artifact-free, Gaussian compartments). Renders: tissue maps, tractogram (TRXViz headless render), the 27 truth maps as a gallery.

### 0.3 Notation and units

---

## Part I — MRI physics

### Chapter 1 — Spins, precession, and the MR signal
- Nuclear magnetization, Larmor precession, excitation, T1/T2/T2* relaxation, the Bloch equations, spin echo vs. gradient echo, why diffusion MRI uses a spin echo.
- *Toy:* a Bloch simulator (numpy ODE) producing FID, spin-echo, and T2* decay curves; a T1/T2 contrast table at the TRXScan compartment presets (`neonatal`, `adult`, `infant`).
- *Phantom:* b=0 images at the three presets to show why low-b GM/WM contrast differs.

### Chapter 2 — Spatial encoding and k-space
- Gradients as frequency/phase encoding; the Fourier relationship; FOV, matrix, resolution, Nyquist; the EPI trajectory; readout time, echo spacing, bandwidth; partial Fourier; multi-shot vs. single-shot; why EPI dominates diffusion. The whole chapter assumes perfectly linear gradients; what happens when they are not is Ch. 13.
- *Toy:* Shepp–Logan (and a simple crossing-fiber geometric phantom) → k-space → image; figures varying matrix size, FOV, partial Fourier fraction; the EPI trajectory drawn on k-space with timing annotations; truncation → Gibbs ringing.
- *Phantom:* a single TRXScan slice with the k-space that produced it (needs the k-space export flag, plan §4) and its EPI readout timing derived from the JSON sidecar.

### Chapter 3 — Image reconstruction from k-space
- Inverse FFT; complex images, magnitude and phase; multi-coil acquisition and combination (sum-of-squares, Roemer/adaptive); parallel imaging (SENSE concept, GRAPPA in detail since TRXScan implements it), g-factor; partial Fourier reconstruction (zero-filling, homodyne, POCS); k-space apodization; compressed sensing in k-space (random undersampling + sparsity prior); noise: Gaussian in k-space → Rician in magnitude → non-central χ after multi-coil/GRAPPA.
- *Toy:* full reconstruction pipeline on Shepp–Logan with synthetic coil sensitivities; GRAPPA kernel fitting from ACS lines; CS reconstruction with a wavelet/TV prior (sigpy or a hand-written FISTA); noise histograms.
- *Phantom:* TRXScan runs with `--coils 8 --accel 2`, PF 6/8, and noise; reconstruct the exported k-space in Python and confirm agreement with TRXScan's own reconstruction; show the phase image and what it carries (object phase, eddy phase ramp, background).
- **Complex data implications** (first appearance; revisited in Ch. 8 and 19): what is possible only if phase is kept (complex denoising, Rician-bias avoidance, phase-based motion/eddy diagnostics).

---

## Part II — Diffusion physics and encoding

### Chapter 4 — Diffusion in tissue
- Brownian motion, Einstein relation, mean squared displacement; free, hindered, restricted diffusion; diffusion time and the "what does a water molecule see" argument; intra/extra-axonal compartments; GM somas; CSF; exchange (mention).
- *Toy:* 2-D random-walk Monte Carlo in free space, between parallel plates, and inside a cylinder; MSD vs. time curves; the propagator; how the simulated propagator's Fourier transform is the q-space signal.

### Chapter 5 — Diffusion encoding
- Stejskal–Tanner PGSE; the b-value and q-vector; Δ, δ, G; the signal equation S = S₀ exp(−b·D) and its tensor/propagator generalizations; the b-value budget: how Gmax and slew rate set the minimum TE for a given b (with the 80 mT/m "whole-body" vs. 300 mT/m "Connectom" presets TRXScan already knows); twice-refocused spin echo; eddy currents born from the diffusion gradients; b-tensor encoding (LTE/PTE/STE) as the generalization.
- *Toy:* sequence-diagram plots (RF, Gx/Gy/Gz, ADC) for SE-EPI with PGSE; TE_min(b, Gmax, slew) curves; single-voxel signal-vs-b for stick/ball/tensor; cross-term between diffusion and imaging gradients.
- *Phantom:* the HBCD scheme's b-values/directions and the per-volume phase ramp TRXScan imprints with `--eddy-phase` (motivating Ch. 11).

### Chapter 6 — q-space sampling schemes
For each scheme: what is sampled, how directions are generated, what the scheme makes possible, what it cannot support, and its scan-time cost. The four required families are single-shell, multi-shell, DSI, and CS-DSI.
1. **Single-shell**: from the minimal DTI-oriented sets (6-direction minimum, 12/30; condition number; b≈1000 rationale; why b0s matter and how many) to HARDI (b=1000–3000, 60–90 directions; what ODF methods need); electrostatic-repulsion direction generation.
2. **Multi-shell**: HBCD-like (b=500/1000/2000/3000), HCP (1000/2000/3000), shell interleaving and ordering for motion robustness; b0 spacing.
3. **DSI**: Cartesian q-space grid (e.g., 257/515 points, radial grid size), the q-space Fourier relation (signal ↔ propagator), scan time, the FOV/resolution trade-off in q-space.
4. **CS-DSI**: sparse random subsets of the DSI grid with a compressed-sensing reconstruction of the propagator (sparsity in a wavelet/dictionary basis), how many samples are enough, and how reconstruction quality degrades with fewer q-samples. The unrelated k-space CS acceleration of EPI is cross-referenced to Ch. 3 and Ch. 7.
5. Free-form / multi-dimensional: multi-Δ, multi-TE, b-tensor (pointer to Part V).
- *Toy:* 3-D sphere plots of every scheme (dipy `disperse_charges`, HemiSphere), q-space grid and CS-subset plots, angular coverage/condition-number plots, scan-time estimates from TR × volumes.
- *Phantom:* the same phantom simulated under 5 schemes (DTI-30, HARDI-64, multi-shell HBCD, DSI-257, CS-DSI-64 as a subset of the DSI run) at matched scan time — used as the reference dataset by all of Part IV.
- **Table 6.1: scheme → model matrix** (which of DTI, DKI, MAP-MRI, CSD, MSMT-CSD, QBI, DSI, NODDI, SMT, free-water, standard model, IVIM each scheme supports and why). This table is reproduced and refined in Ch. 19.

### Chapter 7 — Acquisition parameter choices
Each parameter: what it controls physically, what it costs, what artifact/analysis it affects, and a simulated sweep.
- **TE**: T2 weighting per compartment, SNR loss, the b/TE coupling. *Phantom:* TE sweep (needs `--te`, plan §4) at fixed b; CSF/GM/WM b0 signal vs. TE.
- **TR**: T1 saturation, scan time, slice count; *Toy:* steady-state signal vs. TR per tissue; *Phantom:* `--tissue-s0` as the saturation proxy.
- **Voxel size / matrix / slice thickness**: SNR ∝ voxel volume, partial volume, resolution vs. crossing-fiber resolvability. *Phantom:* 1.5 / 2.0 / 2.5 / 3.0 mm runs (via `prepare_acquisition_grid.py --voxel`).
- **b-values**: contrast vs. SNR, which models need which b; *Toy:* SNR-efficiency curves; *Phantom:* signal decay curves per tissue.
- **Number of directions and b0s**; **shell design** (link to Ch. 6).
- **Partial Fourier** (6/8, 7/8, off): SNR/TE gain vs. blurring and phase-error sensitivity.
- **Multiband / SMS** (`--mb`), **in-plane acceleration** (`--accel`, `--coils`, ACS lines): scan time vs. g-factor noise and slice leakage/dropout.
- **Phase-encode direction, echo spacing, readout time, reverse-PE pairs** (`--reverse-pe`): distortion magnitude, what topup needs.
- **Gradient hardware**: Gmax/slew (what b is reachable at what TE), and gradient nonlinearity as the price of strong or head-only gradients: the spatial warp and encoding error grow with distance from the isocentre, so head positioning and the availability of the vendor coefficient file are acquisition decisions. *Phantom:* `--gnl whole-body-80` vs. `--gnl connectom-300` at matched b; the correction itself is Ch. 13.
- **Complex vs. magnitude export**; **multi-echo/multi-TE options** (pointer to Part V).
- Closing worked example: designing an HBCD-like protocol under a 10-minute budget, with the trade-offs made explicit.

---

## Part III — Artifacts and preprocessing

Each chapter follows one template: **(a)** the physics of the artifact, **(b)** the TRXScan flag(s) that produce it and what they model, **(c)** the artifact-free reference from the same phantom, **(d)** the correction method(s) step by step, **(e)** residual error vs. truth (image-space error maps, and downstream effect on FA/MD/peaks), **(f)** what acquisition choices reduce the problem at the source.

### Chapter 8 — Noise and denoising
Gaussian → Rician → non-central χ; SNR definition and measurement; noise floor and its bias on high-b signal and on DTI metrics; MP-PCA (dipy `mppca`/`localpca`), patch2self, NLMeans; **complex-domain denoising** (why phase helps; magnitude vs. complex MP-PCA); noise maps (`--noise-map` writes the truth σ map); Rician bias correction. *Phantom:* `--noise` sweep, `--coils 8 --accel 2` for nc-χ, complex vs. magnitude denoising vs. truth.

### Chapter 9 — Gibbs ringing
Truncation artifact; why it is worse at low b and near CSF; effect on MD/FA at tissue borders; unringing (Kellner sub-voxel shifts, dipy `gibbs_removal`), including the partial-Fourier variant; apodization as the acquisition-side alternative. *Phantom:* `--oversample 2` (ringing intrinsic to the acquisition) vs. `--oversample 1` (none), and `KspaceWindow` variants (needs a CLI flag, plan §4).

### Chapter 10 — Susceptibility distortion
B0 inhomogeneity, off-resonance → PE-axis displacement ∝ fieldmap × readout time; pile-up and stretching; signal loss; blip-up/blip-down pairs; fieldmap-based (fugue), reverse-PE (topup, DRBUDDI), fieldmap-less (SyN) corrections; readout-time and PE-direction metadata. *Phantom:* AP/PA pair with the atlas fieldmap (`sub-0001a`) and with a measured fieldmap (`sub-60501`); synthetic GRE fieldmap (`--gre-out`, `phasediff` vs. `phase1/phase2`); correction with FSL topup (run offline inside the QSIPrep Docker image) and a didactic Python unwarp from the known field; error vs. undistorted reference.

### Chapter 11 — Eddy currents
Diffusion-gradient-induced fields; direction- and b-dependent shear/scale/translation along PE; the phase ramp on complex data; correction by registration to b0, Gaussian-process prediction (FSL eddy), or replaying a measured trace; b-vector rotation. *Phantom:* `--eddy` / `--eddy-quad` (model) and `--eddy-trace` (real per-volume field replayed from `sub-60501`'s confounds), `--eddy-phase` for the phase view.

### Chapter 12 — Head motion, multiband, and slice dropout
Rigid motion between and within volumes; multiband shot structure; signal dropout from motion during diffusion encoding; motion–eddy coupling; registration-based correction; outlier detection and replacement (eddy `--repol`, SHORELine); b-vector rotation after registration; motion QC metrics (FD, dropout counts). *Phantom:* `--motion` with the real trace from `sub-60501` (faithful per-volume re-simulation), `--mb 3 --dropout-rate 0.1` writing the dropped-shot truth TSV; score outlier detection against it.

### Chapter 13 — Gradient nonlinearity
The gradient coil's field is a solid-harmonic expansion whose $l \geq 3$ terms grow with distance from the isocentre; one field $\phi(r) = r + d(r)$ produces **two** artifacts: a spatial warp of the image (compression toward the isocentre plus an intensity Jacobian) and a per-voxel deviation of the diffusion encoding ($G_\mathrm{eff} = J^\top g$, so b-vectors *and* b-values differ voxel by voxel, biasing FA/MD and peak directions far from the isocentre). Whole-body vs. head-insert/Connectom-class gradients; why the encoding error survives an image unwarp. **Correction step by step:** (a) *gradwarp* — geometric unwarping with the vendor coefficient file via HCP `gradunwarp` or TORTOISE's `CreateNonlinearityDisplacementMap` (the path qsiprep runs), Jacobian intensity modulation, composing with the susceptibility field so the data are resampled once, frame and isocentre conventions; (b) *graddev* — the per-voxel gradient-deviation tensor (`CreateGradientNonlinearityBMatrix`, qsiprep's `graddev` output, HCP nine-volume layout) applied as a per-voxel b-matrix in fits (FSL `dtifit --gradnonlin`, `bedpostx -g`, a per-voxel gradient table in dipy, `odx graddev` for ODFs), and why it must be evaluated in the final resampled frame; (c) where both sit in the pipeline. *Phantom:* `gnl` dataset — `--gnl whole-body-80` and `--gnl connectom-300`, a `--gnl-scale` sweep, and the `--gnl-no-warp` / `--gnl-no-encoding` runs that isolate each effect; TRXScan writes the truth coefficient file, displacement field and graddev image, so gradwarp is scored against `_desc-gnl_disp`, the estimated $J$ against `_desc-gnl_graddev`, and FA/MD/peak error vs. distance from the isocentre before and after each correction.

### Chapter 14 — Remaining artifacts and the assembled pipeline
Nyquist ghosting (`ghost_offset`), k-space spikes (`n_spikes`, needs a CLI flag), receive-field bias, partial volume/CSF and free water, slice-timing/interleave effects. Then the **assembled pipeline**: ordering (denoise → unring → gradwarp + distortion + eddy + motion with a single resampling → graddev in the final frame → bias), why each order choice matters, how qsiprep/MRtrix/FSL pipelines order these, and QC. *Phantom:* a "kitchen sink" run with every artifact on (including `--gnl`), corrected end to end by qsiprep with the coefficient file, with per-step error-vs-truth plots.

---

## Part IV — Reconstruction and modeling

Every model section states: *the signal model*, *its assumptions*, *the minimum q-space sampling it needs and why*, *what breaks when the assumption fails*, and *the fit vs. TRXScan truth* on the Chapter 6 reference datasets.

### Chapter 15 — Signal representations
- **DTI**: tensor model, fit methods (LLS, WLS, NLLS, RESTORE), eigen-decomposition, FA/MD/AD/RD, color FA, the minimum-6-direction argument, why b≤1000 is "Gaussian enough"; failure in crossings and at high b; fitting with a per-voxel b-matrix from the gradient-deviation image (Ch. 13) instead of one gradient table. Fit vs. truth `fa/md/rd/ad`.
- **DKI**: kurtosis tensor, needs ≥2 non-zero shells and b up to ~2000–3000; MK/AK/RK; kurtosis-based WMTI (mention). Fit vs. truth `mk/ak/rk/kfa`.
- **MAP-MRI / SHORE**: propagator bases, needs multi-shell (ideally ≥3 shells, DSI-like coverage); RTOP/RTAP/RTPP, MSD, QIV, NG. Fit vs. truth `rtop/rtap/rtpp/msd/qiv/ng`.
- **QTI / b-tensor** (µFA, k_bulk/k_shear) as the case where the *sampling type* (not just b) unlocks a parameter: truth exists (`micro_fa`, `k_bulk`, `k_shear`) but the acquisition cannot be simulated by TRXScan today (plan §4).
- Effect of sampling on each: fits on DTI-30 vs. HARDI-64 vs. multi-shell vs. DSI, and a "what happens if you fit DKI to single-shell data" demonstration.

### Chapter 16 — Fiber orientation estimation
- dODF vs. fODF; QBI/CSA; DSI (q-space Fourier → propagator → ODF); CSD and MSMT-CSD (response functions, single-shell vs. multi-shell, tissue separation); peak extraction; crossing-fiber resolution vs. b, directions, and SNR. Fit vs. `--truth-peaks` (angular error, number of peaks, GFA/QA vs. truth `gfa/qa`).
- Sampling requirements: single-shell b≥2000 for CSD; multi-shell for MSMT; DSI grid for DSI; CS-DSI as the sparse variant (link to Ch. 6.4), with ODFs from the CS-reconstructed propagator compared against the full-grid DSI ODFs and the truth peaks.

### Chapter 17 — Biophysical microstructure models
- Compartment models and the standard model; **ball-and-stick**, **NODDI** (ICVF/ODI/ISOVF; needs ≥2 shells), **SMT / spherical mean** (rotational invariance; needs multi-shell), **free-water DTI** (needs ≥2 shells or a prior), **IVIM** (low-b shells), the **standard model degeneracy** and what breaks it (multi-Δ, b-tensor, multi-TE, high b). Fits vs. truth `icvf/odi/isovf` and the phantom's known compartment fractions.
- Honest limits: the phantom's compartments are Gaussian (stick/tensor/ball), so models that assume restriction fit the phantom differently than real tissue; the book uses this to separate "model mismatch" from "sampling/noise" effects.

### Chapter 18 — Tractography
- Local models feeding tracking; deterministic vs. probabilistic; step size, curvature, stopping criteria; seeding strategies; anatomically constrained tractography (5TT), PFT; SIFT/SIFT2 weighting; bundle segmentation (brief); connectomes (brief).
- *Phantom:* track on the reconstructed fODFs; evaluate against the ground-truth tractogram that generated the data (bundle overlap/overreach, valid/invalid connections, Tractometer-style scores); repeat on DTI-30 vs. HARDI vs. multi-shell to show how the acquisition bounds tractography quality. Renders through TRXViz's headless CLI.

### Chapter 19 — What your data allow
- The decision matrix, refined: acquisition (scheme × b-max × directions × complex export × multi-TE/Δ/echo) → analyses that are valid, marginal, or impossible, with the chapter that demonstrates each cell.
- Complex reconstruction revisited: what phase enables (complex denoising, Rician-free high-b, phase-based QC) and what it costs (storage, pipeline support).
- Retrospective questions: "I have single-shell b=1000, 32 directions — what can I do?" style worked cases.

---

## Part V — Advanced acquisitions

### Chapter 20 — Multi-TE diffusion MRI
Compartmental T2 differences; TE-dependence of diffusion metrics; diffusion–relaxation correlation (TEdDI, MTE-NODDI); sampling in the (b, TE) plane. *Phantom:* the same scheme at several TE (needs `--te`; plan §4) using TRXScan's per-compartment T2, fit a joint T2–diffusion model, compare with the preset T2s. *Toy:* 2-compartment (b, TE) signal surfaces.

### Chapter 21 — Multi-echo diffusion MRI
Multiple EPI readouts per excitation; per-volume T2* mapping; echo combination (weighted, complex); distortion and SNR differing per echo; use for dropout recovery and relaxometry. *Toy:* multi-echo readout timing and per-echo distortion. *Phantom:* only if the multi-echo readout extension lands in mrsim-acq (plan §4, stretch).

### Chapter 22 — Multi-diffusion-time DWI
Time-dependent diffusion in restricted/hindered geometries; PGSE vs. OGSE; exchange; axon diameter sensitivity and its gradient-strength dependence; sampling in (b, Δ). *Toy:* restricted-cylinder and sphere signals (Callaghan / GPD approximations), Monte Carlo from Ch. 4 at several Δ. *Phantom:* not possible with TRXScan's Gaussian compartments; stated explicitly, with what a restricted-compartment extension would require.

### Chapter 23 — Frontiers (survey)
b-tensor encoding and µFA; diffusion relaxometry beyond TE; high-gradient systems; spiral/multi-shot readouts; deep-learning reconstruction and denoising; simulation as validation (closing the loop the book has been using).

---

## Appendices
- **A. TRXScan cookbook**: every flag used in the book, the commands that generated each dataset, and how to regenerate.
- **B. Data manifest**: every file the notebooks load, with checksums and the generating command.
- **C. Software environment** and versions.
- **D. Glossary.**
- **E. Ground-truth map catalogue**: the 27 truth maps, their definitions, and which chapter uses each.

## Figure/dataset dependency summary

| Dataset id | Phantom | Scheme | Key flags | Chapters |
|---|---|---|---|---|
| `ref-clean` | sub-0001a, 2.5 mm | HBCD 4-shell | no noise, `--oversample 1` | 0, 6, 15–18 (truth baseline) |
| `ref-schemes` | sub-0001a, 2.5 mm | DTI-30, HARDI-64, HBCD, DSI-257 (+ CS-DSI-64 subset) | modest noise | 6, 15–19 |
| `slab-kspace` | sub-0001a, 5 axial slices | HBCD subset (12 vols) | k-space export, `--coils 8 --accel 2`, PF 6/8 | 2, 3 |
| `noise-sweep` | sub-0001a | HBCD | `--noise` × 4 levels, `--coils`/`--accel` | 8 |
| `gibbs` | sub-0001a | HBCD | `--oversample 2` vs `1`, window variants | 9 |
| `sdc-pair` | sub-0001a + sub-60501 | HBCD | `--reverse-pe`, `--gre-out` | 10 |
| `eddy` | sub-60501 | HBCD | `--eddy`, `--eddy-quad`, `--eddy-trace`, `--eddy-phase` | 11 |
| `motion-mb` | sub-60501 | HBCD | `--motion`, `--mb 3 --dropout-rate 0.1` | 12 |
| `gnl` | sub-0001a | HBCD | `--gnl whole-body-80` / `connectom-300`, `--gnl-scale` sweep, `--gnl-no-warp`, `--gnl-no-encoding`; writes truth coeff/disp/graddev | 7, 13, 15 |
| `kitchen-sink` | sub-0001a | HBCD | everything on, incl. `--gnl` | 14 |
| `voxel-sweep` | sub-0001a | HBCD | 1.5/2.0/2.5/3.0 mm | 7 |
| `te-sweep` | sub-0001a | HBCD | `--te` × 4 (new flag) | 7, 20 |
| `truth` | sub-0001a | — | `trxscan-microstructure`, `--truth-peaks` | all Part IV |
