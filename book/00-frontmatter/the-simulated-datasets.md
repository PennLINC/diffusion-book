---
title: "0.2 The simulated datasets"
subtitle: One brain, simulated end to end, with its answer key
kernelspec:
  name: python3
  display_name: Python 3
---

Every simulated dataset in this book comes from one brain, so that a reader who learns its
anatomy in [Chapter 1](../01-mri-physics/01-spins-and-signal.md) recognizes it in [Chapter 18](../04-modeling/18-tractography.md). This page describes what the
simulated brain is made of, what the simulator does to it, which datasets are made from it,
what the ground truth is, and what the ground truth is not.

:::{tip} Skim this page on a first read
This page is a reference for the whole book, and it uses terms that the chapters introduce
later. On a first read, look at the diagram below and the tissue maps, and read the first
paragraph of each section. Come back when a chapter points here.
:::

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from dwibook import phantoms, presets, schemes
from dwibook.plotting import INK, PALETTE, TISSUE_COLORS, set_style, show_image

set_style()
```

## The simulator in one picture

The simulator, TRXScan, starts from a description of one real person's brain and produces
the files a scanner would have produced if that person had been scanned. Because the
simulator also knows the brain it started from, it can write out the correct answer for
every quantity the book later estimates from the data. Read the diagram from left to right:
the gray boxes are the inputs, the blue boxes are the two stages of the simulation, the
green box is the data a reader would receive from a real scanner, and the orange box is
the answer key that no real scanner can provide.

```{code-cell} python
:tags: [hide-input]
def box(ax, x0, y0, x1, y1, title, body, color):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=color + "22", ec=color, lw=1.4))
    ax.text((x0 + x1) / 2, y1 - 0.14, title, ha="center", va="top", fontsize=9, weight="bold", color=INK["primary"])
    ax.text((x0 + x1) / 2, y1 - 0.45, body, ha="center", va="top", fontsize=7.8, color=INK["secondary"], linespacing=1.3)

def arrow(ax, xy_from, xy_to, dashed=False):
    ax.annotate("", xy=xy_to, xytext=xy_from, arrowprops=dict(arrowstyle="-|>", color=INK["secondary"], lw=1.3,
                                                             ls="--" if dashed else "-", shrinkA=0, shrinkB=0))

fig, ax = plt.subplots(figsize=(11.5, 4.9))
ax.set(xlim=(0, 11.5), ylim=(0, 4.9))
ax.set_axis_off()
subject, stage, out, truth = "#8a8a8a", PALETTE[0], PALETTE[2], PALETTE[1]
box(ax, 0.1, 3.95, 2.3, 4.75, "field map", "field offset (Hz)\nin every voxel", subject)
box(ax, 0.1, 2.85, 2.3, 3.65, "tissue fractions", "share of WM, GM, CSF\nin every voxel", subject)
box(ax, 0.1, 1.75, 2.3, 2.55, "fiber streamlines", "the white matter\nfiber paths", subject)
ax.text(1.2, 1.55, "inputs: one real\nsubject's anatomy", ha="center", va="top", fontsize=8, style="italic", color=INK["secondary"])
box(ax, 3.0, 1.85, 5.4, 3.55, "1. signal stage", "the diffusion signal of each\nvoxel's water, for every\ngradient direction and b-value,\nkept separate per tissue", stage)
box(ax, 6.1, 1.85, 8.5, 3.55, "2. acquisition stage", "what a scanner records:\nk-space, T2 decay, distortion,\neddy currents, ghosts, coils,\nnoise, head motion", stage)
box(ax, 9.2, 1.85, 11.4, 3.55, "output data", "BIDS diffusion series:\nmagnitude + phase images,\ngradient table, sidecars", out)
box(ax, 3.0, 0.1, 8.0, 1.2, "truth maps (the answer key)", "FA, MD, fiber orientations, ... computed from the same\nvoxel mixture, with no scanner in between", truth)
arrow(ax, (2.3, 3.25), (3.0, 3.25))
arrow(ax, (2.3, 2.15), (3.0, 2.15))
ax.plot([2.3, 7.3], [4.35, 4.35], color=INK["secondary"], lw=1.3)
arrow(ax, (7.3, 4.35), (7.3, 3.55))
arrow(ax, (5.4, 2.7), (6.1, 2.7))
arrow(ax, (8.5, 2.7), (9.2, 2.7))
arrow(ax, (4.2, 1.85), (4.2, 1.2))
ax.plot([10.3, 10.3], [1.85, 0.65], color=INK["secondary"], lw=1.3, ls="--")
arrow(ax, (10.3, 0.65), (8.0, 0.65), dashed=True)
ax.text(9.15, 0.55, "fit a model to the data,\nscore it against the truth", ha="center", va="top", fontsize=7.8, color=INK["secondary"])
fig.tight_layout()
```

The field map enters only the acquisition stage, because it changes where the scanner puts
the signal, not what signal the tissue gives off ([Chapter 10](../03-preprocessing/10-susceptibility-distortion.md)). The truth maps branch off
before the acquisition stage: they describe the tissue, not the images.

## Tissue maps and tractogram

The simulated brain is built from one adult subject's anatomy, processed to three inputs
that a simulator can use:

- **Tissue fractions.** Three maps at 1 mm resolution give the fraction of white matter,
  gray matter, and CSF in every voxel, from an automatic segmentation of the subject's
  anatomical (T1-weighted) image. Voxels at tissue boundaries hold mixtures, as real
  voxels do.
- **A tractogram.** A set of one million streamlines: curves that trace the white matter
  fiber bundles, estimated from the subject's own diffusion data by the methods of
  [Chapter 18](../04-modeling/18-tractography.md). In the simulator the streamlines *are* the white matter fibers: a voxel's
  fibers point along the streamline segments that pass through it.
- **A field map.** How far the magnetic field in each voxel departs from the scanner's
  nominal field, expressed as the shift in precession frequency in Hz ([Chapter 1](../01-mri-physics/01-spins-and-signal.md)).

:::{dropdown} Details for readers who know tractography
The tractogram was generated by anatomically constrained tractography (ACT) and weighted by
SIFT2 ([Chapter 18](../04-modeling/18-tractography.md)), so that the streamline density is consistent with the fiber density the
data imply. Each voxel's fiber orientation mixture is the set of streamline segments
passing through it, weighted by their SIFT2 weights. The field map comes from a population
atlas of susceptibility-induced field for this brain and, for a second source anatomy used
in Chapters [10](../03-preprocessing/10-susceptibility-distortion.md) through [12](../03-preprocessing/12-motion-and-dropout.md), from that subject's own measurement.
:::

The three tissue maps ship with this book in reduced form, one axial slice at 2 mm and
1 mm and a 3 mm volume, as the object of every toy-tier figure. Brighter means more of that
tissue in the voxel:

```{code-cell} python
:tags: [hide-input]
t = phantoms.brain_slice()
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
for ax, key, name in zip(axes, ("wm", "gm", "csf"), ("white matter", "gray matter", "CSF")):
    show_image(ax, t[key], f"{name} fraction", kind="scalar", vmin=0, vmax=1)
show_image(axes[3], phantoms.brain_image(), f"synthetic b=0 image, TE {presets.TE_HBCD_MS:.0f} ms", vmin=0, vmax=1)
fig.tight_layout()
```

The slice passes through the lateral ventricles and the deep gray matter; anterior is at
the top. The synthetic b=0 image on the right (an image taken with no diffusion weighting)
is what [Chapter 1](../01-mri-physics/01-spins-and-signal.md) builds from the fractions, each tissue weighted by its proton density and
T2 decay, and it is the object that Part I encodes and reconstructs. The 3 mm volume shows
the same brain in three directions:

```{code-cell} python
:tags: [hide-input]
vol = phantoms.brain_volume()
fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.4))
show_image(axes[0], vol["wm"][:, :, phantoms.VOLUME_VENTRICLE_SLICE], "axial slice of the 3 mm volume (WM)", kind="scalar", vmin=0, vmax=1)
show_image(axes[1], vol["wm"][:, 26, :].T[::-1], "sagittal (anterior to the left)", kind="scalar", vmin=0, vmax=1)
show_image(axes[2], vol["wm"][31, :, :].T[::-1], "coronal", kind="scalar", vmin=0, vmax=1)
fig.tight_layout()
print(f"volume {vol['wm'].shape} at {vol['voxel_mm']:g} mm; {int(vol['mask'].sum())} brain voxels")
```

The simulated brain itself is available to every page through the `trxscan` package. The
full subject is a 280 MB download (the NIBS subject `sub-60501`, whose measured fieldmap and
head-motion traces Chapters [10](../03-preprocessing/10-susceptibility-distortion.md) through
[12](../03-preprocessing/12-motion-and-dropout.md) use), and a 20 MB slab of it, six axial
slices through the ventricles with a 34,000-streamline subset, is what the live-tier cells
use. The figure below shows its tissue maps as the simulator sees them, and its field map.

```{code-cell} python
:tags: [hide-input]
from dwibook import phantom as ph

obj = ph.grid()
k = ph.slice_index()
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
for ax, name, title in zip(axes, ("wm", "gm", "csf"), ("white matter", "gray matter", "CSF")):
    show_image(ax, np.rot90(obj.image(name).get_fdata()[:, :, k]), f"{title}, acquisition grid", kind="scalar", vmin=0, vmax=1)
show_image(axes[3], np.rot90(obj.image("sim_fmap").get_fdata()[:, :, k]), "measured field (Hz), simulation grid", kind="diff", vmin=-80, vmax=80)
fig.tight_layout()
print(f"{ph.phantom().n_streamlines} streamlines; acquisition grid {obj.dims} at {obj.voxel_mm[0]:g} mm, simulation grid {obj.sim_dims}")
```

The first three panels are the same tissue fractions as before, now on the grid of the
book's simulated acquisitions. The fourth is the field map: each voxel's frequency offset
in Hz, red for a field slightly above nominal and blue for one slightly below, white for
none. Offsets of tens of Hz look negligible next to 128 MHz, but [Chapter 10](../03-preprocessing/10-susceptibility-distortion.md) shows that they
move the image by several voxels along one axis.

:::{admonition} Why several voxel sizes appear
:class: note
The reference protocol the simulator imitates uses 1.7 mm voxels. The book's own simulated
acquisitions (the live and pipeline tiers) use 2.5 mm voxels, which keeps each simulation
small enough to run in seconds to minutes. Internally the simulator computes the object on
a grid twice as fine as the acquisition, because a real object has detail finer than a
voxel, and that detail causes the ringing of [Chapter 9](../03-preprocessing/09-gibbs-ringing.md). The toy tier uses the packaged
2 mm slice (1 mm where fine detail matters) and the 3 mm volume, which are small enough to
ship with the book.
:::

## What TRXScan simulates

TRXScan is a headless diffusion-MRI simulator {cite:p}`neher2014` that turns the three
inputs into a diffusion series as a scanner would record it. It works in the two stages of
the diagram. The **signal stage** decides how much signal each voxel's water gives for
each diffusion measurement, from the tissue it contains and the direction of its fibers.
The **acquisition stage** decides what a scanner would actually record from that signal,
with the imperfections of a real scanner that Part III corrects.

:::{dropdown} The signal stage in detail (Chapters 4, 5, and 17 explain the terms)
Each voxel's fiber orientations, from the streamlines, are combined with its tissue
fractions into a mixture of compartments: an intra-axonal stick and an extra-axonal tensor
per fiber orientation for white matter, two balls for gray matter, and a free ball for CSF,
with the diffusivities and T2 values of a chosen preset (adult, neonatal, or infant). For
every gradient direction and b-value of the scheme, the compartment signals are evaluated
and kept separate, so that each can relax with its own T2.
:::

:::{dropdown} The acquisition stage in detail (Chapters 2, 3, and 8 through 13 explain the terms)
Each slice's k-space is simulated from the compartment images with the effects of a
single-shot spin-echo EPI acquisition: the field map's displacement, T2 and T2* decay along
the readout, eddy currents proportional to the diffusion gradient, Nyquist ghosting,
partial Fourier, Gibbs ringing (the object is simulated on a finer grid than the
acquisition), multiple receive coils, GRAPPA, and k-space noise. Head motion moves the
tractogram and tissue maps before each volume is re-simulated, so the fiber-gradient
angles change as they do in a moving head; multiband dropout attenuates the slices of one
shot.
:::

The output is a BIDS complex diffusion series (BIDS is the standard folder and file layout
for neuroimaging data): magnitude and phase images, the gradient table, and JSON sidecars
with the phase-encode direction, readout time, and echo time that the corrections of
Part III read. [Appendix A](../appendices/a-trxscan-cookbook.md) lists the flags that switch each effect on and the commands behind
every dataset.

The default protocol is the HBCD study's multi-shell scheme:

```{code-cell} python
:tags: [hide-input]
bvals, _ = schemes.hbcd()
print(f"{len(bvals)} volumes per phase-encode polarity: " + ", ".join(f"{n} at b = {b:.0f}" for b, n in schemes.shells_of(bvals).items()))
print(f"echo time {presets.TE_HBCD_MS:.0f} ms, total readout time {presets.READOUT_HBCD_MS:.1f} ms, 1.7 mm isotropic voxels (the book's datasets use 2.5 mm)")
```

## The datasets

Three kinds of simulated data appear in the book, and every chapter opens with a box that
lists which it uses.

**Toy tier.** Small simulations written in the page and run when the book is built: a
spin's Bloch equations, the k-space of one slice, random walks, single-voxel signal
models, and synthetic diffusion series built from the packaged tissue maps above with a
fiber orientation assigned to every white matter voxel (tangent to the local white matter
boundary, with a second population added where a chapter needs a crossing). The true
orientations, fractions, and diffusivities are known exactly, and the reference for each
artifact chapter is the same series without the artifact. These run in seconds and are the
answer key for most of the book's measurements. The packaged files are listed in
[Appendix B](../appendices/b-data-manifest.md#app-b-package-data).

(live-tier)=
**Live tier.** One slice of the simulated brain, acquired in the page by TRXScan through
its Python package (`pip install trxscan`): the acquisition stage is exactly per slice and
the signal stage rasterizes only the streamlines that cross it, so a slice under the full
76-volume protocol takes a few seconds, including the per-coil k-space and the readout
timing. The slab the cells use is downloaded once (20 MB); `DWIBOOK_PHANTOM=sub-60501` runs
the same cells on the full subject. Everything a live cell shows is bit-identical to the
same slice of a pipeline-tier run, which is what makes the two tiers one simulator rather
than two.

**Pipeline tier.** Full TRXScan simulations of the brain, made offline by a Snakemake
pipeline, versioned, and downloaded by the pages that use them. Each is a directory of
BIDS files with a provenance record; [Appendix A](../appendices/a-trxscan-cookbook.md#app-a-datasets) describes every dataset and gives
the commands that produce it. Until a dataset is released, the figures that need it are
marked *Simulated dataset pending*.

| Dataset | What varies | Chapters |
|---|---|---|
| [`ref-clean`](../appendices/a-trxscan-cookbook.md#ds-ref-clean) | nothing: the artifact-free, noise-free reference with its truth maps | 6, 15–18 |
| [`ref-schemes`](../appendices/a-trxscan-cookbook.md#ds-ref-schemes) | the sampling scheme: 30-direction, 64-direction, HBCD, DSI, CS-DSI | 6, 7, 15–19 |
| [`presets`](../appendices/a-trxscan-cookbook.md#ds-presets) | the tissue preset: adult, neonatal, infant | 1 |
| [`slab-kspace`](../appendices/a-trxscan-cookbook.md#ds-slab-kspace) | a five-slice slab with raw k-space, 8 coils, GRAPPA 2, partial Fourier | 2, 3 |
| [`noise-sweep`](../appendices/a-trxscan-cookbook.md#ds-noise-sweep) | the noise level, and single-coil versus 8-coil GRAPPA | 8, 8b |
| [`gibbs`](../appendices/a-trxscan-cookbook.md#ds-gibbs) | ringing on or off, and apodization at acquisition | 9 |
| [`sdc-pair`](../appendices/a-trxscan-cookbook.md#ds-sdc-pair) | the phase-encode polarity, an atlas versus a measured field | 10 |
| [`eddy`](../appendices/a-trxscan-cookbook.md#ds-eddy) | the eddy-current model: geometric, replayed, phase | 11 |
| [`motion-mb`](../appendices/a-trxscan-cookbook.md#ds-motion-mb) | a replayed head-motion trace; multiband with dropout events | 12 |
| [`gnl`](../appendices/a-trxscan-cookbook.md#ds-gnl) | the gradient system and the severity of its nonlinearity | 7, 13 |
| [`kitchen-sink`](../appendices/a-trxscan-cookbook.md#ds-kitchen-sink) | every artifact at once, corrected by QSIPrep | 14 |
| [`voxel-sweep`](../appendices/a-trxscan-cookbook.md#ds-voxel-sweep) | the voxel size: 1.5 to 3.0 mm | 7 |
| [`te-sweep`](../appendices/a-trxscan-cookbook.md#ds-te-sweep) | the echo time: 70 to 140 ms | 7, 20 |
| [`truth`](../appendices/a-trxscan-cookbook.md#ds-truth) | no acquisition: the 27 analytic truth maps and true fiber peaks | 8–20 |

## The ground-truth maps

Because the simulator knows each voxel's tissue mixture exactly, it can compute the correct
value of the quantities that the models of Part IV estimate from the data, such as how
anisotropic the diffusion is or which way the fibers point. A model fitted to the simulated
data can then be scored on the quantity it claims to measure, which is what every
*Measure it* section does.

:::{dropdown} What the truth maps contain
The companion tool `trxscan-microstructure` evaluates closed-form expressions for 27
microstructure scalars from the same per-voxel mixture that the simulator draws its signal
from: the tensor measures (FA, MD, AD, RD), kurtosis measures, propagator measures, ODF
anisotropies, the b-tensor quantities, and NODDI-style fractions ([Appendix E](../appendices/e-truth-map-catalogue.md)). A separate
option writes the true fiber orientations per voxel.
:::

## What the ground truth is not

The simulated brain is a model, and its answer key is exact for that model, not for real
tissue. A method that recovers the truth here has shown that it works on data like these;
it has not shown that the tissue in a real brain is built the way the simulator assumes.

:::{dropdown} The four limits, and where the book returns to each
- **Gaussian compartments.** The stick, tensor, and balls have diffusivities that do not
  depend on diffusion time. There are no axon diameters, no cell sizes, no exchange, and no
  time-dependent diffusion ([Chapter 22](../05-advanced/22-multi-diffusion-time.md)). The truth maps are long-diffusion-time quantities.
- **One T2 per tissue.** The white matter fiber compartment has a single T2, so the
  TE dependence of [Chapter 20](../05-advanced/20-multi-te.md) arises between tissues, not between the intra- and
  extra-axonal compartments.
- **Model-consistent truth.** The NODDI-style fractions are the simulation's own
  compartment fractions. A model that assumes a different tissue structure will disagree
  with them for reasons that are the model's, not the data's; the simulated datasets
  separate sampling and noise effects from model mismatch, and do not validate any model's
  meaning in vivo.
- **A single-shot spin-echo EPI scanner.** Multi-echo readouts, b-tensor encoding, and
  several diffusion times are not simulated, so Part V is toy-tier only.
:::

Within those limits the simulated datasets do what no real dataset can: they let every
step of the chain from acquisition to model be scored against a known answer, under the
artifacts a real scanner produces, at the parameters of a specific protocol.

## Further reading

The Fiberfox simulator that TRXScan reimplements {cite:p}`neher2014`, dipy, whose
implementations the truth maps are validated against {cite:p}`garyfallidis2014`, and the
tractography challenge that established simulated data with a known answer as the standard
for evaluating tractography {cite:p}`maierhein2017`.
