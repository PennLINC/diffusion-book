---
title: "Appendix A: TRXScan cookbook"
subtitle: Every flag and command behind the book's datasets
kernelspec:
  name: python3
  display_name: Python 3
---

This appendix is generated from the pipeline configuration
(`pipelines/config/datasets.yaml`), the one file that describes what the offline pipeline
runs. The command lines below are the expansion of that file into `trxscan` and
`trxscan-microstructure` invocations, produced by the same rules the pipeline driver uses,
so the cookbook cannot drift from the pipeline. Paths are relative to the pipeline's working
directory; the tissue maps under `work/` come from TRXScan's `prepare_acquisition_grid.py`
([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)), and the schemes under `schemes/` from `dwibook.schemes`.

```{code-cell} python
:tags: [hide-cell]
from dwibook import cookbook

cfg = cookbook.load_config()
```

## The simulator in one paragraph

TRXScan takes a tractogram, tissue volume-fraction maps, a gradient scheme, and a fieldmap,
and writes a BIDS complex 4-D diffusion series (`part-mag`, `part-phase`, `.bval`, `.bvec`,
JSON sidecars). The signal stage rasterizes the streamlines into a per-voxel orientation
mixture and evaluates stick, tensor, and ball compartments per gradient; the acquisition
stage simulates each slice's k-space with EPI distortion from the fieldmap, T2 and T2*
decay during the readout, eddy currents, Nyquist ghosting, partial Fourier, Gibbs ringing
from a finer object grid, multi-coil reception, GRAPPA, and k-space noise; motion and
multiband dropout act per volume or per shot. The companion `trxscan-microstructure`
writes 27 analytic ground-truth maps ([Appendix E](./e-truth-map-catalogue.md)) from the same mixture. Its physics is a
port of MITK Fiberfox {cite:p}`neher2014`.

## Flags used in this book

```{code-cell} python
:tags: [hide-input]
used = set(cookbook.flags_used(cfg))
print(f"{'flag':<28} what it controls")
for flag, what in cookbook.FLAG_GLOSSARY.items():
    if any(f in used for f in flag.split(" / ")):
        print(f"{flag:<28} {what}")
```

Flags marked *planned* correspond to simulator changes listed in the implementation plan;
the datasets that need them are generated once those changes land.

## Shared settings

```{code-cell} python
:tags: [hide-input]
d = cfg["defaults"]
print(f"acquisition voxel {d['voxel_mm']} mm; oversampling {d['oversample']}; {d['subsample']} streamlines sampled by SIFT2 weight with seed {d['seed']}; preset {d['params']}; default scheme {d['scheme']}")
print(f"tools: TRXScan branch {cfg['tools']['trxscan_branch']}; container image {cfg['tools']['qsiprep_image']}")
```

(app-a-datasets)=
## The datasets

Each chapter lists the datasets it uses in the box at its top and links here. For each dataset this section gives what is simulated and why, the chapters that use it, and the exact command lines, rendered from the pipeline configuration. The line *pipeline description* under each name is the one-line description the configuration file carries.

(ds-ref-clean)=
### ref-clean

The baseline. The simulated brain sub-0001a under the HBCD scheme at 2.5 mm with no noise, no oversampling (so no ringing beyond the acquisition matrix's own truncation), and every artifact flag off, together with the truth maps and true fiber peaks for the same streamline subset. Every artifact-free reference at the pipeline tier is this dataset. The truth run uses the same subsample and seed as the simulation so that the answer key matches the data voxel for voxel.

*Used in:* Chapters [6](../02-diffusion-encoding/06-qspace-sampling.md), [15](../04-modeling/15-signal-representations.md), [16](../04-modeling/16-fiber-orientation.md), [17](../04-modeling/17-microstructure-models.md), [18](../04-modeling/18-tractography.md).

```{code-cell} python
:tags: [hide-input]
cookbook.print_commands(cfg, "ref-clean")
```

(ds-ref-schemes)=
### ref-schemes

The same anatomy under four schemes at matched scan time and modest noise: 30 directions at b = 1000, 64 directions at b = 2000, the four-shell HBCD scheme, and the 257-point DSI grid with b up to 4000. A 64-point subset of the DSI run stands in for CS-DSI, so the five schemes of Chapter 6 come from four simulations. Part IV fits every model to these series and scores the fits against `truth`.

*Used in:* Chapters [6](../02-diffusion-encoding/06-qspace-sampling.md), [7](../02-diffusion-encoding/07-acquisition-parameters.md), [15](../04-modeling/15-signal-representations.md), [16](../04-modeling/16-fiber-orientation.md), [17](../04-modeling/17-microstructure-models.md), [18](../04-modeling/18-tractography.md), [19](../04-modeling/19-what-your-data-allow.md).

```{code-cell} python
:tags: [hide-input]
cookbook.print_commands(cfg, "ref-schemes")
```

(ds-presets)=
### presets

The simulated brain sub-0001a under the HBCD scheme with each of the three tissue presets: adult, neonatal, and infant. Nothing else changes between the runs, so the differences in contrast between their b=0 volumes come from the proton density, T1, and T2 values of the presets alone. Chapter 1 shows the b=0 images side by side with the synthetic slice it builds in the page.

*Used in:* Chapter [1](../01-mri-physics/01-spins-and-signal.md).

```{code-cell} python
:tags: [hide-input]
cookbook.print_commands(cfg, "presets")
```

(ds-slab-kspace)=
### slab-kspace

Five axial slices through the ventricles (slices 28 to 33) and the first twelve volumes of the HBCD scheme, simulated with eight receive coils, GRAPPA with acceleration 2, and partial Fourier 6/8, with the raw multi-coil k-space written next to the images. It is the only dataset with k-space, and it waits on the acquisition flags and the k-space export (simulator items T1 and T2). Chapters 2 and 3 use it to show a real EPI trajectory and to reproduce the simulator's reconstruction step by step.

*Used in:* Chapters [2](../01-mri-physics/02-spatial-encoding-kspace.md), [3](../01-mri-physics/03-reconstruction.md).

```{code-cell} python
:tags: [hide-input]
cookbook.print_commands(cfg, "slab-kspace")
```

(ds-noise-sweep)=
### noise-sweep

The simulated brain sub-0001a under the HBCD scheme at four k-space noise levels with a single coil, from nearly noise-free to strongly noisy, plus one run with eight coils and GRAPPA 2, whose magnitude noise follows a non-central chi distribution rather than a Rician one. Chapter 8 measures the noise floor and the denoisers on these series; Chapter 8b applies phase correction to their complex images.

*Used in:* Chapters [8](../03-preprocessing/08-noise.md), [8b](../03-preprocessing/08-real-valued-dwi.md).

```{code-cell} python
:tags: [hide-input]
cookbook.print_commands(cfg, "noise-sweep")
```

(ds-gibbs)=
### gibbs

Three runs that differ only in how the object is rasterized and windowed. With `--oversample 1` the tissue maps are simulated on the acquisition grid, so the only ringing is the acquisition's own truncation. With `--oversample 2` they are simulated on a grid twice as fine, and the k-space of the sharper object is truncated to the acquisition matrix, which rings at every edge as a real scan does. The Hann-windowed variant apodizes that k-space at acquisition (simulator item T1). Chapter 9 compares unringing after the fact with apodization.

*Used in:* Chapter [9](../03-preprocessing/09-gibbs-ringing.md).

```{code-cell} python
:tags: [hide-input]
cookbook.print_commands(cfg, "gibbs")
```

(ds-sdc-pair)=
### sdc-pair

Blip-up/blip-down pairs for two source anatomies: sub-0001a with a population-atlas field, and sub-60501 with the field measured in that subject. Each anatomy is simulated with the phase-encode direction anterior-posterior and then reversed (`--reverse-pe`), and the AP run also writes a synthetic gradient-echo fieldmap at 2.5 mm (`--gre-out`). The pipeline precomputes FSL topup on each pair. Chapter 10 corrects the pairs with topup and with the fieldmap and scores each against the undistorted reference.

*Used in:* Chapter [10](../03-preprocessing/10-susceptibility-distortion.md).

```{code-cell} python
:tags: [hide-input]
cookbook.print_commands(cfg, "sdc-pair")
```

(ds-eddy)=
### eddy

The simulated brain sub-60501 under the HBCD scheme with three eddy-current models: a linear-plus-quadratic field proportional to the diffusion gradient (`--eddy`, `--eddy-quad`), a per-volume field replayed from the parameters FSL eddy estimated in the real subject (`--eddy-trace`), and the phase ramp the eddy field adds to the complex image (`--eddy-phase`). The pipeline precomputes FSL eddy on each run. Chapter 11 scores the estimated fields against the simulated ones.

*Used in:* Chapter [11](../03-preprocessing/11-eddy-currents.md).

```{code-cell} python
:tags: [hide-input]
cookbook.print_commands(cfg, "eddy")
```

(ds-motion-mb)=
### motion-mb

Two runs from sub-60501. In the motion run the head pose measured in the real subject is replayed volume by volume (`--motion`): the tissue maps and streamlines are moved before each volume is simulated, so the angles between fibers and gradients change as they do in a moving head. In the dropout run a multiband-3 acquisition loses 10 % of its shots (`--mb 3 --dropout-rate 0.1`), and the affected slices are recorded as ground truth. The pipeline precomputes FSL eddy with outlier replacement. Chapter 12 scores the motion estimates and the outlier detection.

*Used in:* Chapter [12](../03-preprocessing/12-motion-and-dropout.md).

```{code-cell} python
:tags: [hide-input]
cookbook.print_commands(cfg, "motion-mb")
```

(ds-gnl)=
### gnl

Gradient nonlinearity on sub-0001a, with the isocenter placed 20 mm anterior and 30 mm inferior of the volume center so that the far slices see a large field deviation. Five runs: an 80 mT/m whole-body gradient system, the same at twice the severity, the same with only the spatial warp or only the encoding deviation switched on, and a 300 mT/m Connectom-class system. TRXScan writes the true coefficient file, displacement field, and gradient-deviation image next to each series, and the pipeline precomputes the corrections of gradunwarp and TORTOISE. Chapter 13 scores each correction against the true fields; Chapter 7 uses the two gradient systems to show what a stronger gradient buys.

*Used in:* Chapters [7](../02-diffusion-encoding/07-acquisition-parameters.md), [13](../03-preprocessing/13-gradient-nonlinearity.md).

```{code-cell} python
:tags: [hide-input]
cookbook.print_commands(cfg, "gnl")
```

(ds-kitchen-sink)=
### kitchen-sink

Every artifact at once, on sub-0001a: oversampling 2 (ringing), noise, eight coils with GRAPPA 2, multiband 3 with 5 % dropout, modeled eddy currents with their phase ramp, whole-body gradient nonlinearity, an AP/PA pair, and a synthetic gradient-echo fieldmap. The pipeline runs QSIPrep on it end to end with the coefficient file. Chapter 14 reads the QSIPrep report and scores the output against `truth`.

*Used in:* Chapter [14](../03-preprocessing/14-assembled-pipeline.md).

```{code-cell} python
:tags: [hide-input]
cookbook.print_commands(cfg, "kitchen-sink")
```

(ds-voxel-sweep)=
### voxel-sweep

The simulated brain sub-0001a under the HBCD scheme at 1.5, 2.0, 2.5, and 3.0 mm isotropic voxels with the same noise level, so that the change in signal-to-noise ratio and in partial-volume mixing with voxel size can be seen on the same slice. Chapter 7 uses it for the resolution trade-off.

*Used in:* Chapter [7](../02-diffusion-encoding/07-acquisition-parameters.md).

```{code-cell} python
:tags: [hide-input]
cookbook.print_commands(cfg, "voxel-sweep")
```

(ds-te-sweep)=
### te-sweep

The simulated brain sub-0001a under the HBCD scheme at echo times of 70, 88, 110, and 140 ms (`--te`, simulator item T1). Because each compartment relaxes with its own T2, the ratio of white matter to CSF signal and the diffusion contrast change with TE. Chapter 20 fits the joint diffusion-relaxation model across these runs.

*Used in:* Chapters [7](../02-diffusion-encoding/07-acquisition-parameters.md), [20](../05-advanced/20-multi-te.md).

```{code-cell} python
:tags: [hide-input]
cookbook.print_commands(cfg, "te-sweep")
```

(ds-truth)=
### truth

The 27 analytic microstructure maps and the true fiber peaks for sub-0001a, evaluated by `trxscan-microstructure` from the same per-voxel mixture the simulator draws its signal from, with the same streamline subsample and seed as the other datasets ([Appendix E](./e-truth-map-catalogue.md)). No acquisition is simulated. Every *Measure it* section of Parts III and IV scores its fit against these maps.

*Used in:* Chapters [8](../03-preprocessing/08-noise.md), [8b](../03-preprocessing/08-real-valued-dwi.md), [9](../03-preprocessing/09-gibbs-ringing.md), [10](../03-preprocessing/10-susceptibility-distortion.md), [11](../03-preprocessing/11-eddy-currents.md), [12](../03-preprocessing/12-motion-and-dropout.md), [13](../03-preprocessing/13-gradient-nonlinearity.md), [14](../03-preprocessing/14-assembled-pipeline.md), [15](../04-modeling/15-signal-representations.md), [16](../04-modeling/16-fiber-orientation.md), [17](../04-modeling/17-microstructure-models.md), [18](../04-modeling/18-tractography.md), [19](../04-modeling/19-what-your-data-allow.md), [20](../05-advanced/20-multi-te.md).

```{code-cell} python
:tags: [hide-input]
cookbook.print_commands(cfg, "truth")
```

## Regenerating a dataset

Every dataset directory carries a `provenance.json` with the exact command lines, the
TRXScan commit, and the container image tag. To regenerate one:

```bash
cd pipelines
micromamba run -n dwibook snakemake -c 8 <dataset-id>
```

The Snakefile prepares the acquisition grids for the source anatomy and voxel size the dataset
needs, runs the commands above, runs the precomputations the chapter needs inside the
QSIPrep container, validates the outputs, and writes the provenance file.
