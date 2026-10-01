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

```{code-cell} python
:tags: [hide-cell]
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from dwibook.plotting import INK, PALETTE, set_style

set_style()
```

The chapters are meant to be read in order the first time, because each step of the chain
below depends on the one before it. Part I covers how an image is made and reconstructed,
Part II how diffusion is encoded and sampled, Part III what goes wrong and how it is fixed,
Part IV what is fitted to the corrected data, and Part V what lies beyond a standard
acquisition.

```{code-cell} python
:tags: [hide-input]
steps = [  # (part, label, chapters)
    ("I", "protons give\na signal", "Ch 1"),
    ("I", "gradients encode\nposition (k-space)", "Ch 2"),
    ("I", "k-space becomes\nan image", "Ch 3"),
    ("II", "water diffuses\nin tissue", "Ch 4"),
    ("II", "diffusion weighting\nand its sampling", "Ch 5–7"),
    ("III", "artifacts, and\nhow to correct them", "Ch 8–14"),
    ("IV", "models turn the\nimages into maps", "Ch 15–19"),
    ("V", "beyond a standard\nacquisition", "Ch 20–22"),
]
parts = {"I": "how an image is made", "II": "how diffusion is measured", "III": "what goes wrong",
         "IV": "what is fitted", "V": "further"}
colors = dict(zip(parts, PALETTE[:5]))
W, GAP = 1.45, 0.28
fig, ax = plt.subplots(figsize=(13.2, 2.6))
ax.set(xlim=(-0.05, len(steps) * (W + GAP) - GAP + 0.05), ylim=(0, 2.25))
ax.set_axis_off()
xs = [i * (W + GAP) for i in range(len(steps))]
for i, (x, (part, label, ch)) in enumerate(zip(xs, steps)):
    c = colors[part]
    ax.add_patch(FancyBboxPatch((x, 0.35), W, 1.05, boxstyle="round,pad=0.02,rounding_size=0.1", fc=c + "22", ec=c, lw=1.4))
    ax.text(x + W / 2, 1.02, label, ha="center", va="center", fontsize=8.6, color=INK["primary"], linespacing=1.25)
    ax.text(x + W / 2, 0.52, ch, ha="center", va="center", fontsize=8, color=INK["secondary"])
    if i:
        ax.annotate("", xy=(x, 0.875), xytext=(x - GAP, 0.875),
                    arrowprops=dict(arrowstyle="-|>", color=INK["secondary"], lw=1.2, shrinkA=0, shrinkB=0))
for part, desc in parts.items():
    idx = [i for i, s in enumerate(steps) if s[0] == part]
    x0, x1 = xs[idx[0]], xs[idx[-1]] + W
    ax.plot([x0, x1], [1.62, 1.62], color=colors[part], lw=2.5, solid_capstyle="butt")
    ax.text((x0 + x1) / 2, 1.72, f"Part {part}\n{desc}", ha="center", va="bottom", fontsize=8.6, color=INK["primary"], linespacing=1.2)
fig.tight_layout()
```

Each box is one step from the scanner to a finished map, and the arrows give the order in
which the book takes them. [Chapter 19](../04-modeling/19-what-your-data-allow.md) collects the requirements of everything before it into
one decision table and is the page to return to.

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

## Three kinds of simulation

The figures come from three sources, and every chapter says which.
[Chapter 0.2](./the-simulated-datasets.md) describes each in full.

- **Toy tier.** Small simulations written in the page, which run in seconds when the book
  is built. Those that need a brain use the tissue maps of one real subject, `sub-0001a`,
  packaged with the book.
- **Live tier.** One slice of a second subject's simulated brain, `sub-60501`, acquired in
  the page by the TRXScan simulator in a few seconds
  ([Chapter 0.2](./the-simulated-datasets.md#live-tier)).
- **Pipeline tier.** Full-brain simulations by TRXScan, mostly of `sub-0001a` and, where a
  chapter needs a field map, eddy currents, or motion measured in a real person, of `sub-60501`, made
  offline and downloaded by the pages that use them. Sections that depend on one are
  marked *Simulated dataset pending* until it has been released.

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
configuration, and each downloaded dataset is a BIDS dataset {cite:p}`gorgolewski2016` that carries, in its
`derivatives/trxscan` directory, a `provenance.json` with the commands, the simulator
version, and the container image that produced it. The pipeline itself is a
Snakemake workflow {cite:p}`molder2021` in the repository's `pipelines` directory; it requires the TRXScan
binaries and the simulation inputs, which are distributed separately from the book.

## Conventions

This section is reference material. It uses terms such as phase-encode axis and b-value
that Chapters 2 through 5 define, and it will make sense by the end of
[Chapter 5](../02-diffusion-encoding/05-diffusion-encoding.md); come back to it then.

- Spelling is American; the tissue colors are fixed throughout (white matter blue, gray
  matter orange, CSF aqua); difference maps use a diverging scale centered on zero.
- Images of the simulated brain are axial slices with anterior at the top and, in the toy tier,
  radiological orientation (image left is the subject's right). The phase-encode axis of
  every simulated acquisition is anterior-posterior, as in the reference protocol.
- Units: b in s/mm², diffusivities in mm²/s (with 10⁻³ mm²/s = 1 µm²/ms), times in ms
  unless stated, field offsets in Hz, displacements in voxels or mm as labeled. The
  notation page (0.3) lists every symbol.
- The reference protocol is the multi-shell scheme of the HBCD study (b = 500, 1000,
  2000, 3000; echo time 88 ms; effective total readout time 91.7 ms; 1.7 mm voxels)
  {cite:p}`dean2024`, which
  is what the simulator reproduces by default and what the toy tier approximates at 2 mm.
  It appears in two versions with the same shells. The toy tier and most pipeline-tier
  datasets use a bundled copy of 75 volumes per polarity, sorted by b for teaching (shells
  in blocks of increasing b, a b=0 volume about every eight). The live tier uses the
  protocol as HBCD acquires it: 76 volumes that start with six b=0 volumes and interleave
  the shells throughout. The sorted order is not how HBCD acquires its data.
