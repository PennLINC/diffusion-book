---
title: "19. What your data allow"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** sampling schemes evaluated from their b-values and direction counts, without images ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
- **`ref-schemes`** (pending): the simulated brain under the 30-direction, 64-direction, HBCD, and DSI schemes, compared at equal total scan time (the per-volume noise is scaled with the number of volumes), plus a CS-DSI subset of the DSI run, which takes about a quarter of its time ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-ref-schemes)).
- **`truth`** (pending): the 27 analytic ground-truth maps and the true fiber orientations ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-truth), [Appendix E](../appendices/e-truth-map-catalogue.md)).

Pipeline-tier datasets are simulated offline by TRXScan ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)) and are marked *pending* until their release; the figures that need them say so where they will appear.
:::

## Learning goals

After this chapter you can:

- read a scheme's b-values and say which analyses it supports, which are marginal, and
  which are not possible
- state what complex data add across the analysis chain and what they cost
- work through a dataset you did not design and decide what to do with it

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch, Rectangle

from dwibook import schemes
from dwibook.plotting import INK, PALETTE, set_style

set_style()
```

## The decision matrix

Every requirement in Parts II through IV reduces to a few counts: how many non-zero shells,
how many directions on each, how high the top shell goes, how many b=0 volumes, whether the
phase was saved. The function `schemes.analysis_matrix` applies those rules to a b-value
table; it is the book's [Table 6.1](../02-diffusion-encoding/06-qspace-sampling.md) as a
function, and it can be run on any `.bval` file. The figure below applies it to four
reference schemes and to two of the inherited datasets worked through later in this
chapter.

```{code-cell} python
:tags: [hide-input]
# Each column: (header, b-values, phase saved?). Verdicts come only from analysis_matrix.
COLUMNS = [
    ("Clinical DTI\n30 dirs, b 1000", schemes.single_shell(1000, 30, n_b0=3)[0], False),
    ("HARDI\n64 dirs, b 2000", schemes.single_shell(2000, 64, n_b0=4)[0], False),
    ("HBCD, phase\n4 shells (case 3)", schemes.hbcd()[0], True),
    ("DSI grid\n257 points", schemes.dsi_grid(radius=4)[0], False),
    ("Case 1\n32 dirs, b 1000", schemes.single_shell(1000, 32, n_b0=1)[0], False),
    ("Case 2\nb 1000 + 2500", schemes.multi_shell({1000: 30, 2500: 30}, n_b0=2)[0], False),
]
N_REFERENCE = 4  # columns before the gap are reference schemes; after it, inherited cases

# Display labels only: (name, what it needs · where the book shows it).
ROW_LABELS = {
    "mean diffusivity / ADC": ("Mean diffusivity (ADC)", "≥ 3 directions and 1 b=0 · Ch 15"),
    "DTI (FA, direction)": ("Tensor (DTI): FA, fiber direction", "≥ 30 dirs at b ≤ 1200 (6 minimum) · Ch 6, 15"),
    "diffusion kurtosis": ("Diffusion kurtosis", "≥ 2 shells, top ≥ 2000, ≥ 30 dirs · Ch 15"),
    "single-shell CSD": ("Fiber ODF, single-shell CSD", "≥ 45 dirs on one shell at b ≈ 2000 or more · Ch 16"),
    "multi-tissue CSD": ("Fiber ODF, multi-tissue CSD", "≥ 2 shells, one as above · Ch 16"),
    "NODDI / spherical mean / free water": ("NODDI, spherical mean, free water", "≥ 2 shells, top ≥ 2000 · Ch 17"),
    "MAP-MRI / propagator": ("MAP-MRI (propagator)", "≥ 3 shells, ≥ 60 dirs · Ch 15"),
    "DSI (model-free propagator)": ("DSI (model-free propagator)", "Cartesian grid, 200+ points · Ch 6, 16"),
    "complex-domain denoising": ("Complex-domain denoising", "phase saved · Ch 3, 8, 8b"),
}

matrix = [schemes.analysis_matrix(b, complex_data=c) for _, b, c in COLUMNS]
analyses = [row[0] for row in matrix[0]]


def tint(color, amount):
    """Mix a color with white (amount = share of the original color)."""
    return tuple(1 - amount * (1 - v) for v in to_rgb(color))


# Lightness falls from "no" to "yes", so the three verdicts separate in grayscale too.
STYLE = {
    "yes": (PALETTE[5], "white"),
    "marginal": (PALETTE[3], INK["primary"]),
    "no": (tint(PALETTE[7], 0.22), INK["secondary"]),
}

# Number the distinct reasons behind "marginal" cells for the notes under the figure.
notes = {}
for j, rows in enumerate(matrix):
    for analysis, verdict, reason in rows:
        if verdict == "marginal":
            notes.setdefault((analysis, reason), []).append(COLUMNS[j][0].split("\n")[0])
note_id = {key: k + 1 for k, key in enumerate(notes)}
SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")

# Geometry in inches: the axes spans the whole figure with 1 data unit = 1 inch.
label_w, cell_w, cell_h, gap, head_h, foot_h = 2.85, 0.80, 0.40, 0.18, 0.78, 0.40
n_rows, n_cols = len(analyses), len(COLUMNS)
W = label_w + n_cols * cell_w + gap + 0.05
H = head_h + n_rows * cell_h + foot_h
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W), ax.set_ylim(H, 0), ax.set_axis_off()


def col_x(j):
    return label_w + j * cell_w + (gap if j >= N_REFERENCE else 0)


for i, analysis in enumerate(analyses):
    y = head_h + i * cell_h
    name, needs = ROW_LABELS.get(analysis, (analysis, ""))
    ax.text(label_w - 0.08, y + 0.15, name, ha="right", va="center", fontsize=8.5, color=INK["primary"])
    ax.text(label_w - 0.08, y + 0.30, needs, ha="right", va="center", fontsize=6.8, color=INK["secondary"])
    for j, rows in enumerate(matrix):
        a, verdict, reason = rows[i]
        assert a == analysis
        face, ink = STYLE[verdict]
        x = col_x(j)
        ax.add_patch(Rectangle((x + 0.02, y + 0.02), cell_w - 0.04, cell_h - 0.04, facecolor=face, edgecolor="none"))
        text = verdict
        if verdict == "marginal":
            text += str(note_id[(analysis, reason)]).translate(SUPERSCRIPT)
        ax.text(x + cell_w / 2, y + cell_h / 2, text, ha="center", va="center", fontsize=7.5, color=ink,
                fontweight="bold" if verdict == "yes" else "normal")

for j, (header, _, _) in enumerate(COLUMNS):
    ax.text(col_x(j) + cell_w / 2, head_h - 0.06, header, ha="center", va="bottom", fontsize=6.8,
            color=INK["primary"], linespacing=1.2)
for lo, hi, title in [(0, N_REFERENCE, "Reference schemes"), (N_REFERENCE, n_cols, "Inherited datasets")]:
    x0, x1 = col_x(lo) + 0.04, col_x(hi - 1) + cell_w - 0.04
    ax.plot([x0, x1], [0.26, 0.26], color=INK["secondary"], lw=0.8)
    ax.text((x0 + x1) / 2, 0.22, title, ha="center", va="bottom", fontsize=8, color=INK["secondary"])

legend = [Patch(facecolor=STYLE[v][0], label=v) for v in ("yes", "marginal", "no")]
ax.legend(handles=legend, loc="lower right", bbox_to_anchor=(W - 0.05, H - 0.02), bbox_transform=ax.transData,
          ncol=3, fontsize=8, handlelength=1.4, columnspacing=1.2, frameon=False)
plt.show()

for (analysis, reason), cols in notes.items():
    k = str(note_id[(analysis, reason)]).translate(SUPERSCRIPT)
    reason = reason.replace(">=", "≥").replace("<=", "≤")
    print(f"{k} {ROW_LABELS.get(analysis, (analysis,))[0]} — {', '.join(cols)}: {reason}")
```

Read each row across to see which schemes support that analysis, or each column down to
see everything one dataset supports; green is "yes", amber "marginal", pale red "no", and
the numbered notes under the figure give the reason for each marginal cell. The two
single-shell columns show the trade most clearly: b = 1000 gives a good tensor but only a
marginal fiber ODF, b = 2000 the reverse. Only multi-shell columns turn the kurtosis and
compartment-model rows green. The fiber-ODF rows need something else: one shell at
b ≈ 2000 or more with 45 or more directions. Single-shell CSD uses only one shell, so
directions on other shells do not help it. Multi-tissue CSD fits all shells together, and
the rule asking it for one well-sampled high shell is a conservative one:
[Chapter 16](./16-fiber-orientation.md) shows the multi-tissue fit resolving 60° crossings from three
30-direction shells once its harmonic order is high enough. The HBCD scheme has
four shells but only 18 and 29 directions on its two high ones, so both fiber-ODF rows are
marginal for it, as is the tensor, with 18 directions at b ≤ 1200. The DSI grid spreads
its points over 14 radii rather than a few shells, so multi-tissue CSD on it needs the
points binned into shells first.

The verdicts are rules of thumb, not guarantees. A "yes" means the fit is determined and
the sampling is in the range where the method was developed; "marginal" means the fit
runs but its assumptions are strained or its precision is poor, and the chapter for that
method shows what that looks like; "no" means the fit cannot be performed or its result
has no meaning.

The same verdicts as text, with the reason for every cell, are in the collapsed output
below.

```{code-cell} python
:tags: [hide-input, hide-output]
def report(name, bvals, complex_data=False):
    print(f"\n{name}: {len(bvals)} volumes, shells {schemes.shells_of(bvals)}")
    for analysis, verdict, reason in schemes.analysis_matrix(bvals, complex_data=complex_data):
        print(f"  {analysis:<36} {verdict:<9} {reason}")

for header, bvals, complex_data in COLUMNS:
    report(header.replace("\n", ": "), bvals, complex_data=complex_data)
```

To run the same check on your own dataset, point it at the FSL-format `.bval` and `.bvec`
files that come with the images:

```python
bvals, bvecs = schemes.read_fsl("sub-01_dwi")  # reads sub-01_dwi.bval and sub-01_dwi.bvec
for row in schemes.analysis_matrix(bvals, complex_data=False): print(*row, sep="  |  ")
```

Some requirements cannot be read from the b-values at all, because they depend on extra
volumes, extra files, or metadata. They are collected below for reference.

:::{dropdown} Requirements the b-value table cannot show
| Analysis | Needs | Chapter |
|---|---|---|
| IVIM (perfusion signal) | several shells at b < 200 | 17 |
| Deterministic tractography | any scheme with a "yes" for the tensor or a fiber ODF | 18 |
| Probabilistic tractography with ACT (anatomically constrained tractography) | a fiber ODF plus a registered tissue segmentation | 18 |
| Distortion correction with topup (FSL's field-map estimator) | reverse-polarity volumes and correct phase-encoding metadata | 10 |
| Eddy-current and motion correction with eddy (FSL's correction tool), using its prediction | a full, well-spread direction set | 11 |
| Gradient nonlinearity correction | the scanner's gradient coefficient file | 13 |
:::

## Complex data revisited

Three chapters used the phase, and the case for saving it is now complete:

- **Denoising without bias** ([Chapter 8](../03-preprocessing/08-noise.md) and
  [Chapter 8b](../03-preprocessing/08-real-valued-dwi.md)): the largest gain, and the one
  that reaches every model fitted at high b. Taking the magnitude turns noise into a
  positive floor (the pictures in [Chapter 3](../01-mri-physics/03-reconstruction.md),
  section "Noise in magnitude images"), and that floor raises exactly the weak, high-b
  signals those models depend on. Chapter 8b shows the floor disappearing when the phase
  is used.
- **Diagnostics** (Chapters [11](../03-preprocessing/11-eddy-currents.md) and [12](../03-preprocessing/12-motion-and-dropout.md)): the eddy-current and motion phase are visible per
  volume before any correction.
- **Averaging** repeated acquisitions without the floor.

The costs are all downstream. Storage doubles. The phase cannot be used as recorded: it
contains a smooth background that must be estimated and removed first, and in noisy
voxels it is close to random. And not every tool accepts complex input; the denoising
must run before the magnitude is taken, which fixes its place at the start of the
pipeline. None of these costs is paid at the scanner: saving the phase adds no scan time,
only a reconstruction setting.

## Worked retrospective cases

Datasets are more often inherited than designed. The three cases below are the "Case 1",
"Case 2", and "HBCD, phase" columns of the figure above; each paragraph says what to do
with the dataset.

**Case 1: single shell, b = 1000, 32 directions, one b=0, magnitude only.** Stay with the
tensor: FA, mean diffusivity, the principal direction, deterministic tractography,
tract-based statistics. Single-shell CSD runs but resolves crossings poorly at b = 1000
(in [Chapter 16](./16-fiber-orientation.md) it missed a third of them).
Without reverse-polarity volumes, correct distortion with a fieldmap if one exists,
otherwise by registration to the anatomical image. The single b=0 volume weakens outlier
detection and eddy's prediction, so report motion carefully.

:::{dropdown} Tract-based spatial statistics: what to check
Tract-based spatial statistics (TBSS, FSL's standard group analysis of FA) {cite:p}`smith2006` registers every
subject's FA map to a template, thins the group's mean FA to a one-voxel-wide "skeleton"
along the center of the main tracts, and then gives each skeleton voxel the highest FA
found nearby in each subject, perpendicular to the skeleton. That projection is meant to
absorb small misregistrations, and it is where the pitfalls are:

- **Projection hides misregistration instead of fixing it.** Where registration is off by
  more than the search distance, or where two tracts lie close together, the maximum can
  come from the wrong tract, or from the edge of the right one. Check the registered FA
  maps, not only the skeleton.
- **CSF partial volume.** Near the ventricles and in atrophy, CSF in the voxel lowers FA
  and raises MD ([Chapter 17](./17-microstructure-models.md), free-water elimination), so a
  group difference in brain size or ventricle size can appear as an FA difference.
  Free-water-corrected maps, or brain volume as a covariate, separate the two.
- **Crossing fibers.** On the skeleton as elsewhere, FA falls where a second bundle
  crosses ([Chapter 16](./16-fiber-orientation.md)). A group difference in a crossing
  region can come from the crossing bundle, and a higher FA there can mean the crossing
  bundle was lost, not that the tract improved.
:::

**Case 2: two shells, b = 1000 and 2500, 30 directions each, reverse-polarity b=0s.** Fit
kurtosis, free water, NODDI, and the spherical mean technique. Fit the tensor to the
b = 1000 shell only. Multi-tissue CSD is marginal, with 30 directions on the top shell:
run it (a constrained fit does not need its harmonic order lowered to match the direction
count, [Chapter 16](./16-fiber-orientation.md)) but check its peaks in known crossing
regions. Treat MAP-MRI results from two shells with caution.
Correct distortion with topup.

**Case 3: HBCD-style multi-shell with phase, 1.7 mm, both polarities.** The strengths are
the many shells and the phase: kurtosis, the compartment models, MAP-MRI, and
complex-domain denoising all get a "yes". Three rows are marginal. The tensor has only 18
directions at b ≤ 1200, so its maps are noisier than from a 30-direction clinical scan.
Neither fiber-ODF method has a shell with 45 directions (18 at b = 2000, 29 at b = 3000),
so single-shell CSD on either shell gives broad, low-contrast ODFs. Denoise in the complex
domain first; correct with topup and eddy using both polarities; fit kurtosis, MAP-MRI,
and the compartment models. For orientations, prefer multi-tissue CSD, which fits all four
shells together, over single-shell CSD, check its peaks in known crossing regions before
trusting them, and track probabilistically with ACT. The remaining limits are the ones no
processing removes: the fixed diffusion time, the Gaussian assumptions of the models, and
the resolution. (The column uses the book's bundled HBCD scheme, 75 volumes sorted by b
for readability. The protocol HBCD acquires {cite:p}`dean2024` has 76 volumes, with the shells interleaved
after six b=0 volumes and the same directions per shell, so every verdict is the same.)

## Measure it: the simulated datasets

:::{admonition} Simulated dataset pending
:class: note
This section will show every cell of the matrix on the `ref-schemes` dataset: the same
simulated brain, five schemes, each model fitted where the rules allow, scored against the truth
maps, so that "marginal" is a number rather than a word.
:::

## What this implies for acquisition

- **Decide the analyses, then read the matrix along their rows**; the cheapest scheme
  that says "yes" to all of them is the protocol.
- **Three shells: b ≈ 1000 with 30 or more directions, a shell at b ≈ 2000 or more with
  45 or more directions, and a third shell, with reverse polarity and the phase saved.**
  This supports every analysis in the matrix except DSI (and IVIM, which needs several
  shells below b = 200). The third shell is what turns MAP-MRI from "marginal" to "yes";
  with two shells every other row is already green.
- **Record the metadata**: phase-encode direction, readout time, diffusion timing, and
  the gradient coefficient file.

The cell below checks two such protocols against the matrix and times them at the TR of
[Chapter 7](../02-diffusion-encoding/07-acquisition-parameters.md)'s worked example.

```{code-cell} python
:tags: [hide-input]
tr_s = 72 * 0.14 / 3  # Chapter 7's worked example: 72 slices, 0.14 s each, multiband 3
for name, shells_, n_b0 in [("two shells", {1000: 30, 2000: 45}, 6), ("three shells", {1000: 30, 2000: 30, 3000: 45}, 6)]:
    b_rec = schemes.multi_shell(shells_, n_b0=n_b0)[0]
    verdicts = {a: v for a, v, _ in schemes.analysis_matrix(b_rec, complex_data=True)}
    minutes = schemes.scan_time_s(len(b_rec) + 6, tr_s) / 60  # plus six reverse-polarity b=0 volumes
    not_yes = [f"{a} ({v})" for a, v in verdicts.items() if v != "yes"]
    print(f"{name}, {len(b_rec)} volumes + 6 reverse b=0: {minutes:.1f} min at TR {tr_s:.2f} s; not 'yes': {', '.join(not_yes)}")
```

The three-shell protocol (30, 30, and 45 directions at b = 1000, 2000, and 3000, plus six
b=0) takes 6.6 minutes including six reverse-polarity b=0 volumes, and only DSI is not
"yes". The two-shell protocol (30 directions at b = 1000, 45 at b = 2000) takes 4.9
minutes, and MAP-MRI drops to marginal.

## Further reading

The reviews of {cite:t}`alexander2019` on microstructure and {cite:t}`jeurissen2019` on
tractography, and the cautions of {cite:t}`jones2013`.
