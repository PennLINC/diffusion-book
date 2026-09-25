---
title: "0.1 How to read and run this book"
subtitle: Executable cells, data, environment, reproduction
kernelspec:
  name: python3
  display_name: Python 3
---

## Who this book is for

This book is for people who are new to MRI and need to work with diffusion data: to plan
an acquisition, to judge whether a dataset supports an analysis, to run a preprocessing
pipeline and read its quality report, or to interpret a map someone else produced. It
covers the physics only as far as the physics explains what appears in the data and what
can be done about it. Every artifact is shown on simulated data whose correct answer is
known, every correction is scored against that answer, and every model is fitted where its
assumptions hold and where they do not.

The chapters are meant to be read in order the first time. Part I covers how an image is
made and reconstructed, Part II how diffusion is encoded and sampled, Part III what goes
wrong and how it is fixed, Part IV what is fitted to the corrected data, and Part V what
lies beyond a standard acquisition. [Chapter 19](../04-modeling/19-what-your-data-allow.md) collects the requirements of everything
before it into one decision table and is the page to return to.

## Executable cells

Every figure and every number in the text is produced by code in the page. The code is
collapsed by default; each code cell has a *Source* toggle that shows it. Reading the book
does not require reading the code, but the code is short and deliberately literal, and it
is the exact procedure behind each figure. Cells that only set up imports are hidden
entirely.

Each chapter opens with a box listing the simulated datasets it uses, with links to their
descriptions in [Appendix A](../appendices/a-trxscan-cookbook.md#app-a-datasets). It then follows one structure: *learning goals*, the physics, a *See it* section
that produces the figures, a *Measure it* section that reports a number against the known
answer, *What this implies for acquisition*, and further reading. In Part III the middle
sections follow the artifact template: the physics of the artifact, the simulator flags
that produce it, the artifact-free reference, the correction step by step, the residual
against truth, and the acquisition choices that reduce it.

## Two kinds of simulation

The figures come from two sources, and every chapter says which:

- **Toy tier.** Small simulations written in the page and run when the book is built: a
  spin's Bloch equations, the k-space of one brain slice, random walks, single-voxel
  signal models, and synthetic diffusion series built from the simulated brain's tissue maps with
  a known fiber orientation in every voxel. They run in seconds and are the answer key for
  most of the book's measurements.
- **Pipeline tier.** Full simulations of the same brain ([Chapter 0.2](./the-simulated-datasets.md)) by TRXScan, a
  diffusion-MRI simulator that models the acquisition from the diffusion signal through
  k-space to the reconstructed complex image, with the artifacts of a real scanner. These
  runs take minutes to hours, so they are made offline by a pipeline, versioned, and
  downloaded by the pages that use them. Sections that depend on them are marked *Simulated dataset pending* until the corresponding dataset has been released.

The toy tier is fully reproducible from the repository alone. The pipeline tier is
reproducible from the repository plus the simulator and the simulation inputs, which
[Appendix A](../appendices/a-trxscan-cookbook.md) documents command by command.

## Running the book yourself

The repository holds the pages, the helper package `dwibook`, and the pipeline. To execute
any chapter locally:

```bash
git clone https://github.com/PennLINC/diffusion-book
cd diffusion-book
micromamba create -n dwibook -f environment.yml
micromamba run -n dwibook pip install -e .
OMP_NUM_THREADS=1 micromamba run -n dwibook myst start --execute
```

The last command serves the book at a local address and re-executes a page whenever it is
edited. The chapters can also be opened as notebooks in JupyterLab, since each page is a
MyST Markdown notebook. [Appendix C](../appendices/c-software-environment.md) lists the versions of every package the published
build used.

The pipeline-tier datasets are fetched on first use by the pages that need them and cached; to
build against a local pipeline output instead, set `DWIBOOK_DATA` to its directory
([Appendix B](../appendices/b-data-manifest.md)).

## Reproducing the simulations

[Appendix A](../appendices/a-trxscan-cookbook.md) lists the command line behind every dataset, generated from the pipeline's own
configuration, and each downloaded dataset carries a `provenance.json` with the commands,
the simulator version, and the container image that produced it. The pipeline itself is a
Snakemake workflow in the repository's `pipelines` directory; it requires the TRXScan
binaries and the simulation inputs, which are distributed separately from the book.

## Conventions

- Spelling is American; the tissue colors are fixed throughout (white matter blue, gray
  matter orange, CSF aqua); difference maps use a diverging scale centered on zero.
- Images of the simulated brain are axial slices with anterior at the top and, in the toy tier,
  radiological orientation (image left is the subject's right). The phase-encode axis of
  every simulated acquisition is anterior-posterior, as in the reference protocol.
- Units: b in s/mm², diffusivities in mm²/s (with 10⁻³ mm²/s = 1 µm²/ms), times in ms
  unless stated, field offsets in Hz, displacements in voxels or mm as labeled. The
  notation page (0.3) lists every symbol.
- The reference protocol is the multi-shell scheme of the HBCD study (b = 500, 1000,
  2000, 3000; 75 volumes per polarity; echo time 88 ms; 1.7 mm voxels), which is what the
  simulator reproduces by default and what the toy tier approximates at 2 mm.
