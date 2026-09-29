---
title: "17. Biophysical microstructure models"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** a synthetic multi-shell series built from the packaged tissue maps, with known compartment fractions ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
- **`ref-clean`** (pending): the artifact-free, noise-free reference series with its truth maps and true fiber orientations ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-ref-clean)).
- **`ref-schemes`** (pending): the simulated brain under the 30-direction, 64-direction, HBCD, DSI, and CS-DSI schemes at matched scan time ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-ref-schemes)).
- **`truth`** (pending): the 27 analytic ground-truth maps and the true fiber orientations ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-truth), [Appendix E](../appendices/e-truth-map-catalogue.md)).

Pipeline-tier datasets are simulated offline by TRXScan ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)) and are marked *pending* until their release; the figures that need them say so where they will appear.
:::

## Learning goals

After this chapter you can:

- describe the compartment picture behind ball-and-stick, NODDI, the spherical mean
  technique, free-water elimination, and IVIM, and what each parameter claims to measure
- fit a free-water model and a spherical-mean model and compare their parameters with a
  known answer
- explain the degeneracy of multi-compartment fits and show what breaks it
- separate model mismatch from sampling and noise effects

```{code-cell} python
:tags: [hide-cell]
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import warnings
warnings.filterwarnings("ignore")  # dipy's solver notices are not part of the lesson
import numpy as np
import matplotlib.pyplot as plt
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel
from dipy.reconst.fwdti import FreeWaterTensorModel
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch
from scipy.optimize import brentq
from scipy.special import erf, jnp_zeros

from dwibook import phantoms, presets, schemes, signal, synth
from dwibook.plotting import INK, PALETTE, TISSUE_COLORS, set_style, show_image

set_style()
```

## Compartments

A biophysical model writes the voxel signal as a sum of compartments, each with a shape,
a diffusivity, and a volume fraction, and fits those quantities to the data. Four shapes
recur. The figure draws each one twice. On top is how far a water molecule in it typically
moves in 40 ms, a common diffusion time (one standard deviation along each axis, to the
10 µm scale bar). Below is how its signal falls with b when the gradient points along the
axon (solid) or across it (dashed), computed from the diffusivities of the book's white
matter and CSF.

```{code-cell} python
:tags: [hide-input]
D_AX = presets.ADULT_DIFFUSIVITY["WM_intra"]      # along an axon, 1.7 x 10^-3 mm^2/s
D_RAD = presets.ADULT_DIFFUSIVITY["WM_extra"][1]  # across, between axons, 0.6 x 10^-3
D_FREE = presets.ADULT_DIFFUSIVITY["CSF"]         # free water at body temperature, 3.0 x 10^-3
R_UM, G_MAX = 3.0, 80.0                           # a wide axon (6 um across), an 80 mT/m scanner

def cylinder_across(b):
    """Signal across an impermeable cylinder of radius R_UM, for the pulse timing that reaches
    each b on a G_MAX system (Gaussian phase approximation, van Gelderen et al. 1994)."""
    alpha = jnp_zeros(1, 20) / (R_UM * 1e-6)      # 1/m
    d, g, out = D_AX * 1e-6, G_MAX * 1e-3, []     # SI units
    for bi in np.atleast_1d(b):
        if bi == 0:
            out.append(1.0)
            continue
        timing = signal.min_te(bi, G_MAX)
        dl, DL, a2 = timing["delta"] * 1e-3, timing["big_delta"] * 1e-3, alpha ** 2
        num = (2 * d * a2 * dl - 2 + 2 * np.exp(-d * a2 * dl) + 2 * np.exp(-d * a2 * DL)
               - np.exp(-d * a2 * (DL - dl)) - np.exp(-d * a2 * (DL + dl)))
        den = d ** 2 * alpha ** 6 * ((R_UM * 1e-6) ** 2 * a2 - 1)
        out.append(np.exp(-2 * signal.GAMMA ** 2 * g ** 2 * np.sum(num / den)))
    return np.array(out)

b_curve = np.linspace(0, 3000, 61)
T_MS = 40.0
rms_um = lambda dd: np.sqrt(2 * dd * T_MS * 1e-3) * 1e3  # one-axis rms displacement, um
comps = [
    ("stick", "axon interior:\nmoves only along the axon", rms_um(D_AX), 0.0,
     signal.stick(b_curve, D_AX, 1.0), signal.stick(b_curve, D_AX, 0.0)),
    ("cylinder", "axon with a width:\nacross, stopped by the wall", rms_um(D_AX), R_UM,
     signal.stick(b_curve, D_AX, 1.0), cylinder_across(b_curve)),
    ("ball", "free water:\nthe same in every direction", rms_um(D_FREE), rms_um(D_FREE),
     signal.ball(b_curve, D_FREE), None),
    ("zeppelin", "water between axons:\nfaster along than across", rms_um(D_AX), rms_um(D_RAD),
     signal.zeppelin(b_curve, D_AX, D_RAD, 1.0), signal.zeppelin(b_curve, D_AX, D_RAD, 0.0)),
]
fig, axes = plt.subplots(2, 4, figsize=(11, 5.4))
for k, (name, desc, a_par, a_perp, s_par, s_perp) in enumerate(comps):
    ax, color = axes[0, k], PALETTE[k]
    ax.set_xlim(-22, 22); ax.set_ylim(-16, 16); ax.set_aspect("equal"); ax.axis("off")
    if name == "stick":
        ax.plot([-20, 20], [0, 0], color=INK["secondary"], lw=1, ls=":")
        ax.plot([-a_par, a_par], [0, 0], color=color, lw=4, solid_capstyle="round")
    elif name == "cylinder":
        ax.add_patch(FancyBboxPatch((-20, -R_UM), 40, 2 * R_UM, boxstyle="round,pad=0,rounding_size=2.5",
                                    fill=False, ec=INK["secondary"], lw=1.2))
        ax.add_patch(Ellipse((0, 0), 2 * a_par, 2 * R_UM * 0.85, color=color, alpha=0.6))
    elif name == "ball":
        ax.add_patch(Circle((0, 0), a_par, color=color, alpha=0.6))
    else:
        ax.add_patch(Ellipse((0, 0), 2 * a_par, 2 * a_perp, color=color, alpha=0.6))
    if name != "ball":
        ax.annotate("", xy=(20, -13), xytext=(8, -13), arrowprops=dict(arrowstyle="->", color=INK["secondary"]))
        ax.text(14, -11.5, "axon axis", ha="center", fontsize=7, color=INK["secondary"])
    ax.set_title(f"{name}\n{desc}", fontsize=9)
    ax2 = axes[1, k]
    if s_perp is None:
        ax2.plot(b_curve, s_par, color=color, label="any direction")
    else:
        ax2.plot(b_curve, s_par, color=color, label="gradient along the axis")
        ax2.plot(b_curve, s_perp, color=color, ls="--", label="gradient across")
    ax2.set(ylim=(0, 1.05), xlabel="b (s/mm²)")
    ax2.legend(fontsize=7, loc="lower left")
axes[1, 0].set_ylabel("S / S₀")
axes[0, 0].plot([-20, -10], [13, 13], color=INK["primary"], lw=1)
axes[0, 0].text(-15, 14, "10 µm", ha="center", fontsize=7)
fig.tight_layout()
print(f"typical displacement in {T_MS:.0f} ms: along an axon {rms_um(D_AX):.1f} µm, free water {rms_um(D_FREE):.1f} µm, "
      f"across the space between axons {rms_um(D_RAD):.1f} µm")
print(f"cylinder {2 * R_UM:.0f} µm across, signal with the gradient across it: {cylinder_across([1000])[0]:.2f} at b = 1000, "
      f"{cylinder_across([3000])[0]:.2f} at b = 3000 (a stick: 1.00)")
```

- The **stick** is the inside of an axon treated as a line: water moves along it (11.7 µm
  in 40 ms) and not at all across it, so with the gradient across the axon the signal does
  not fall (dashed line at 1).
- The **cylinder** is the same axon with a width. Water moving across it is stopped by the
  wall after a few micrometers, so it loses a little signal: for an axon 6 µm across, wide
  for the human brain, the signal is still 0.95 at b = 3000 on an 80 mT/m scanner. That is
  why most models use the stick: at ordinary gradient strengths the width is nearly
  invisible ([Chapter 22](../05-advanced/22-multi-diffusion-time.md)).
- The **ball** is water free to move equally in every direction, as in CSF. It moves
  farthest (15.5 µm) and its signal falls fastest, to almost nothing by b = 2000.
- The **zeppelin** is the water between axons: hindered but not trapped, it moves farther
  along the axons than across them (11.7 against 6.9 µm) and loses signal in every
  direction. It is the tensor's ellipsoid ([Chapter 15](./15-signal-representations.md)) with two equal short axes.

The parameters of a model built from these shapes have names that sound like histology
(intra-axonal fraction, neurite density, free-water fraction), and that is both the appeal
and the risk: the names are only as true as the assumptions, and the assumptions are
simplifications of tissue. The standard model of white matter {cite:p}`novikov2019`
collects the common ingredients: sticks for axons, a zeppelin for the water between them,
a free-water ball, and an orientation distribution over the sticks.

The models in use differ in which of the four shapes above they keep and which numbers
they fix rather than fit (diffusivities in 10⁻³ mm²/s):

| Model | Compartments kept | What is fixed | Minimum shells | Outputs | Typical use |
|---|---|---|---|---|---|
| **Ball-and-stick** {cite:p}`behrens2003` | one or more sticks + a ball | one diffusivity shared by all compartments | one | stick directions, their fractions and uncertainty | fiber directions for probabilistic tractography (FSL's bedpostx) |
| **NODDI** {cite:p}`zhang2012` | sticks spread by a Watson distribution + a zeppelin tied to them + a ball | the diffusivities (1.7 along axons, 3.0 for free water); the zeppelin's radial diffusivity by a tortuosity rule | two, top ≥ 2000 | neurite density (ICVF), orientation dispersion (ODI), free-water fraction (ISOVF) | group studies of white and gray matter |
| **Spherical mean technique** {cite:p}`kaden2016` | stick + zeppelin, averaged over directions | the zeppelin's radial diffusivity, by a tortuosity rule | two | intra-axonal fraction, axial diffusivity | microstructure where fibers cross or fan out |
| **Free-water elimination** {cite:p}`pasternak2009` | a full tensor + a ball | the ball's diffusivity (3.0) | two (one with a spatial prior) | free-water fraction, corrected FA and MD | tissue next to CSF, edema, atrophy |
| **IVIM** {cite:p}`lebihan1986` | a tissue ball + a fast "blood" ball | usually nothing | several b-values between 0 and about 200, plus a higher one | perfusion fraction, pseudo-diffusion coefficient | perfusion, not microstructure |

A **tortuosity rule** ties the zeppelin's radial diffusivity to the stick fraction: the
more of the voxel the axons fill, the longer the detours water between them must take to
move across them, so the slower its radial diffusion. It removes one unknown at the price
of one assumption. Fixing diffusivities is what lets NODDI fit from two shells, and it is
also what makes its values depend on those fixed numbers. The spherical mean technique averages the signal
over all directions of each shell, which removes the orientation distribution entirely, so
crossings do not matter and the diffusivity can be fitted. Free-water elimination with one
shell is ill-posed and needs a spatial prior; with two or more shells it is determined.
Two of the rows add something the four shapes do not show, drawn below: NODDI's
**orientation dispersion**, and IVIM's blood compartment.

```{code-cell} python
:tags: [hide-input]
def watson_sample(kappa, n, rng):
    """Unit vectors from a Watson distribution about the x axis (rejection sampling)."""
    out = []
    while len(out) < n:
        v = rng.normal(size=(4 * n, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        keep = rng.random(4 * n) < np.exp(kappa * (v[:, 0] ** 2 - 1))
        out.extend(v[keep])
    return np.array(out[:n])

rng = np.random.default_rng(3)
fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
for ax, odi in zip(axes, [0.1, 0.5]):
    kappa = 1 / np.tan(np.pi * odi / 2)           # ODI = (2 / pi) arctan(1 / kappa)
    dirs = watson_sample(kappa, 160, rng)
    centers = rng.uniform([-1, -0.6], [1, 0.6], size=(160, 2))
    for (cx, cy), v in zip(centers, dirs):
        dx, dy = 0.14 * v[0], 0.14 * v[1]          # the stick seen from the side
        ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], color=TISSUE_COLORS["WM"], lw=1.4, solid_capstyle="round")
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"ODI = {odi}: {'nearly parallel' if odi < 0.3 else 'widely fanned'} (kappa = {kappa:.1f})")
fig.tight_layout()
```

NODDI's sticks do not all point one way: their directions spread around a mean according
to a Watson distribution, summarized by the **orientation dispersion index** (ODI), 0 for
perfectly parallel sticks and 1 for directions spread uniformly. The figure draws 160
sticks, seen from the side, at ODI 0.1, typical of a tight bundle such as the corpus
callosum, and at 0.5, closer to the fanning fibers of cortex. The same dispersion lowers
FA, which is why FA alone cannot separate fewer axons from more spread ones.

```{code-cell} python
:tags: [hide-input]
F_BLOOD, D_BLOOD = 0.08, 10e-3                    # blood fraction (exaggerated), pseudo-diffusivity
b_ivim = np.linspace(0, 1000, 201)
tissue_part = (1 - F_BLOOD) * signal.gray_matter(b_ivim)
blood_part = F_BLOOD * signal.ball(b_ivim, D_BLOOD)
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.axvspan(0, 200, color=PALETTE[7], alpha=0.08, lw=0)
ax.plot(b_ivim, tissue_part + blood_part, color=TISSUE_COLORS["GM"], label="voxel signal")
ax.plot(b_ivim, tissue_part, color=INK["secondary"], ls="--", lw=1.2, label="tissue alone")
ax.annotate("blood: mostly gone by b ≈ 200", xy=(40, (tissue_part + blood_part)[8]), xytext=(260, 0.97),
            fontsize=8, color=PALETTE[7], arrowprops=dict(arrowstyle="->", color=PALETTE[7]))
ax.set(xlabel="b (s/mm²)", ylabel="S / S₀", ylim=(0.35, 1.02), title="IVIM: gray matter with a blood compartment")
ax.legend(loc="lower left")
fig.tight_layout()
k200 = np.searchsorted(b_ivim, 200)
print(f"blood fraction {F_BLOOD:.2f}; blood signal left at b = 200: {blood_part[k200]:.3f} "
      f"({blood_part[k200] / F_BLOOD:.0%} of its b = 0 value)")
```

IVIM (intravoxel incoherent motion) treats blood moving through randomly oriented
capillaries as a very fast ball: over the diffusion time the blood travels far and in
changing directions, so its signal behaves like diffusion with a large pseudo-diffusivity
(10 × 10⁻³ mm²/s here, three times that of free water). The figure uses a blood fraction of
8 %, more than brain tissue has, so the drop is easy to see. By b = 200 the blood signal is
down to 0.011 of the voxel's b = 0 signal (14 % of its own starting value), and above that
the curve runs just above, and nearly parallel to, the tissue-alone line. Only
b-values below about 200 see the blood at all, which is why IVIM needs several of them and
why a standard diffusion protocol, with a b = 0 and then nothing below 1000, cannot fit
it.

## The synthetic series and its answer key

The synthetic white matter is itself a two-compartment model: an intra-axonal stick with
fraction 0.55 and diffusivity 1.7 × 10⁻³ mm²/s, and an extra-axonal tensor with axial 1.7 and
radial 0.6 × 10⁻³. Gray matter is two balls, CSF a free ball. The tissue fractions are
known per voxel. This is the simulated brain's own model ([Chapter 4](../02-diffusion-encoding/04-diffusion-in-tissue.md)), so the answer key is exact
and the fits below measure model mismatch as much as noise.

```{code-cell} python
:tags: [hide-input]
t = phantoms.brain_slice()
mask = t["mask"]
wm, gm, csf = t["wm"] > 0.95, t["gm"] > 0.9, t["csf"] > 0.9
bvals, bvecs = schemes.multi_shell({1000: 20, 2000: 20, 3000: 30}, n_b0=4)
clean = synth.synthetic_dwi(t, bvals, bvecs)
sigma = clean[wm][:, bvals == 0].mean() / 25
noisy = synth.add_noise(clean, sigma, seed=0)
gtab = gradient_table(bvals, bvecs=bvecs)
print(f"true intra-axonal fraction {presets.ADULT_FRACTIONS['WM_intra']}, axial diffusivity {presets.ADULT_DIFFUSIVITY['WM_intra'] * 1e3:.1f} x 10^-3 mm^2/s")
```

## Free-water elimination

A voxel on the wall of a ventricle holds both tissue and CSF, and a tensor fitted to it
describes the mixture, not the tissue. Take a voxel that is 70 % white matter and 30 % CSF
by volume, with its fibers along one axis, and fit a tensor to its noise-free b = 1000
signal from the book's signal model:

```{code-cell} python
:tags: [hide-input]
pv_bvals, pv_bvecs = schemes.multi_shell({1000: 30, 2000: 30}, n_b0=4)
along_x = np.array([[[1.0, 0.0, 0.0]]])
def one_voxel(f_csf):
    tissue = {"wm": np.array([[1 - f_csf]]), "gm": np.zeros((1, 1)), "csf": np.array([[f_csf]])}
    return synth.synthetic_dwi(tissue, pv_bvals, pv_bvecs, orientation=along_x)
b1k = pv_bvals <= 1000
pure = synth.dti_maps(one_voxel(0.0)[..., b1k], pv_bvals[b1k], pv_bvecs[b1k])
mix = synth.dti_maps(one_voxel(0.3)[..., b1k], pv_bvals[b1k], pv_bvecs[b1k])
s_mix = one_voxel(0.3)[0, 0]
csf_b0_share = 0.3 * synth.synthetic_dwi({"wm": np.zeros((1, 1)), "gm": np.zeros((1, 1)), "csf": np.ones((1, 1))},
                                         pv_bvals[:1], pv_bvecs[:1], orientation=along_x)[0, 0, 0] / s_mix[0]
fw_one = FreeWaterTensorModel(gradient_table(pv_bvals, bvecs=pv_bvecs)).fit(one_voxel(0.3))
print(f"pure white matter:       tensor FA {pure['fa'].item():.2f}, MD {pure['md'].item() * 1e3:.2f} x 10^-3 mm^2/s")
print(f"70 % WM + 30 % CSF:      tensor FA {mix['fa'].item():.2f}, MD {mix['md'].item() * 1e3:.2f} x 10^-3 mm^2/s")
print(f"CSF's share of that voxel's b = 0 signal: {csf_b0_share:.0%}")
print(f"free-water model on the same voxel (b = 1000 and 2000): free-water fraction {fw_one.f.item():.2f}, FA {fw_one.fa.item():.2f}")
```

The FA falls from 0.85 to 0.40 although the white matter in the voxel is unchanged. CSF is
30 % of the volume but 68 % of the b = 0 signal, because CSF holds more water and keeps its
signal much longer after excitation (its T2 is long, [Chapter 1](../01-mri-physics/01-spins-and-signal.md)); at b = 1000 that bright,
isotropic water swamps the direction dependence of the tissue. The free-water model fits a
tissue tensor plus a ball with the diffusivity of free water, and for this voxel it
recovers most of the tissue's FA (0.90). Its free-water fraction, 0.70, is close to CSF's
share of the signal, not to its share of the volume: the model splits the *signal*, and
the brightness of CSF is part of what it measures.

The same model on the whole slice: its two outputs are a free-water fraction, which should
follow the CSF of each voxel, and a tissue tensor whose FA no longer collapses at the
ventricle wall. dipy's implementation fits it to two or more shells; the fit uses the
b ≤ 2000 volumes here.

```{code-cell} python
:tags: [hide-input]
keep = bvals <= 2000
g2 = gradient_table(bvals[keep], bvecs=bvecs[keep])
fw = FreeWaterTensorModel(g2).fit(noisy[..., keep], mask=mask)
plain = TensorModel(gradient_table(bvals[bvals <= 1000], bvecs=bvecs[bvals <= 1000]), fit_method="WLS").fit(noisy[..., bvals <= 1000], mask=mask)
# the tissue-only reference: the same voxels with their CSF removed
tissue_only = synth.synthetic_dwi({"wm": t["wm"], "gm": t["gm"], "csf": np.zeros_like(t["csf"])}, bvals, bvecs)
ref = TensorModel(gradient_table(bvals[bvals <= 1000], bvecs=bvecs[bvals <= 1000]), fit_method="WLS").fit(tissue_only[..., bvals <= 1000], mask=mask)
border = (t["csf"] > 0.2) & (t["csf"] < 0.8) & (t["wm"] > 0.2)

fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
show_image(axes[0], t["csf"] * mask, "true CSF fraction", kind="scalar", vmin=0, vmax=1)
show_image(axes[1], fw.f * mask, "free-water fraction, fitted", kind="scalar", vmin=0, vmax=1)
show_image(axes[2], plain.fa * mask, "FA, plain tensor", kind="scalar", vmin=0, vmax=0.9)
show_image(axes[3], fw.fa * mask, "FA, free-water-corrected tensor", kind="scalar", vmin=0, vmax=0.9)
fig.tight_layout()
print(f"at tissue-CSF borders ({border.sum()} voxels): free-water fraction {fw.f[border].mean():.2f} vs true CSF fraction {t['csf'][border].mean():.2f}")
print(f"  FA there: plain tensor {plain.fa[border].mean():.2f}, free-water-corrected {fw.fa[border].mean():.2f}, tissue-only reference {ref.fa[border].mean():.2f}")
```

The border is one or two voxels wide, too thin to judge by eye at this size. The next
figure zooms in on a small CSF-filled gap between two stretches of white matter (left) and
plots every quantity along the white line (right). Watch the two border voxels, shaded
gray.

```{code-cell} python
:tags: [hide-input]
row, c0, c1 = 77, 38, 57                          # a line: white matter, CSF, white matter
cols = np.arange(c0, c1)
te_s0 = synth.synthetic_dwi({"wm": t["wm"], "gm": t["gm"], "csf": t["csf"]}, bvals[:1], bvecs[:1])[..., 0]
csf_only = synth.synthetic_dwi({"wm": np.zeros_like(t["csf"]), "gm": np.zeros_like(t["csf"]), "csf": t["csf"]}, bvals[:1], bvecs[:1])[..., 0]
csf_share = csf_only / np.maximum(te_s0, 1e-9)
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 2.2]})
r0, r1, q0, q1 = 64, 91, 30, 70
show_image(ax0, (t["csf"] * mask)[r0:r1, q0:q1], "true CSF fraction (zoom)", kind="scalar", vmin=0, vmax=1)
ax0.plot([c0 - q0, c1 - 1 - q0], [row - r0, row - r0], color="white", lw=1.5)
x = cols - c0
lines = [
    ("true CSF fraction (volume)", t["csf"][row, cols], TISSUE_COLORS["CSF"], "-", "o"),
    ("CSF share of the b = 0 signal", csf_share[row, cols], TISSUE_COLORS["CSF"], ":", None),
    ("free-water fraction, fitted", fw.f[row, cols], TISSUE_COLORS["CSF"], "--", "s"),
    ("FA, plain tensor", plain.fa[row, cols], PALETTE[7], "-", "o"),
    ("FA, free-water-corrected", fw.fa[row, cols], PALETTE[6], "-", "s"),
    ("FA, tissue-only reference", ref.fa[row, cols], INK["secondary"], "--", None),
]
for xb in np.flatnonzero((t["csf"][row, cols] > 0.15) & (t["csf"][row, cols] < 0.85)):
    ax1.axvspan(xb - 0.5, xb + 0.5, color=INK["grid"], lw=0)
for label, y, color, ls, marker in lines:
    ax1.plot(x, y, color=color, ls=ls, marker=marker, ms=3.5, lw=1.6, label=label)
ax1.set(xlabel="position along the line (voxels)", ylabel="fraction or FA", ylim=(-0.02, 1.05))
ax1.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
fig.tight_layout()
for xb in np.flatnonzero((t["csf"][row, cols] > 0.15) & (t["csf"][row, cols] < 0.85)):
    c = cols[xb]
    print(f"border voxel {xb:2d}: CSF volume {t['csf'][row, c]:.2f}, CSF signal share {csf_share[row, c]:.2f}, fitted free water {fw.f[row, c]:.2f}; "
          f"FA plain {plain.fa[row, c]:.2f}, corrected {fw.fa[row, c]:.2f}, tissue-only {ref.fa[row, c]:.2f}")
gap = cols[t["csf"][row, cols] >= 0.85]
print(f"corrected FA inside the CSF gap: {', '.join(f'{v:.2f}' for v in fw.fa[row, gap])}")
print(f"pure white matter (all {wm.sum()} voxels): fitted free-water fraction {np.median(fw.f[wm]):.2f} (median; true 0)")
```

In the profile, the plain FA (red) drops at each border voxel to about a third of its
white matter value (0.32 and 0.33), while the corrected FA (purple) stays with the
tissue-only reference (dashed gray), a little above it (0.89 against 0.85, 0.82 against
0.74). The fitted free-water fraction (green squares) follows the dotted line, CSF's share
of the signal, and not the solid line, its share of the volume: 0.77 where the volume
fraction is 0.38. Inside the CSF gap the corrected FA means nothing, since almost no
tissue is left to describe; one voxel there reads 1.0. Free-water-corrected FA maps are
read in tissue, not in CSF.

Averaged over all 73 border voxels the picture is the same. The free-water fraction (0.77)
is far above the CSF volume fraction (0.42) because it measures signal, and CSF is bright;
it depends on the echo time and is not a CSF percentage. The corrected FA (0.73) recovers
from the plain tensor's 0.23 and slightly overshoots the tissue alone (0.66). That
overshoot, and the free-water fraction of 0.08 the fit finds in pure white matter, which
contains none, are model mismatch: the synthetic water between axons diffuses at
1.7 × 10⁻³ mm²/s along the fibers, fast enough that the model assigns part of it to the
free-water ball, and removing that signal from the tissue compartment sharpens its
anisotropy. On real tissue the same trade-off exists wherever a fast tissue component is
present, which is one more reason free-water fractions are compared within a study rather
than read as a CSF percentage.

## The spherical mean technique

Averaging the signal over all directions of a shell removes every effect of fiber
orientation, including crossings and dispersion, and leaves a function of b alone. A
two-compartment model of that function has two parameters, an intra-axonal fraction and a
diffusivity, which can be fitted from two or three shells by a **grid search**: compute
the model's prediction for every combination of the two parameters on a fine grid, and keep
the combination closest to the measurements. The figure plots the spherical means of the
three tissues on a log scale, where a single free compartment decays along a straight
line.

```{code-cell} python
:tags: [hide-input]
shells, means = synth.spherical_mean(noisy, bvals)
fig, ax = plt.subplots(figsize=(6, 3.2))
for name, m, color in [("WM", wm, TISSUE_COLORS["WM"]), ("GM", gm, TISSUE_COLORS["GM"]), ("CSF", csf, TISSUE_COLORS["CSF"])]:
    ax.semilogy(shells, means[m].mean(0) / means[m].mean(0)[0], "o-", color=color, label=name)
ax.set(xlabel="b (s/mm²)", ylabel="spherical mean / b=0", title="per-shell spherical means")
ax.legend()
fig.tight_layout()

smt = synth.smt_fit(shells, means)
print(f"spherical-mean fit in pure white matter: intra-axonal fraction {np.median(smt['f'][wm]):.2f} (true 0.55), "
      f"axial diffusivity {np.median(smt['d_par'][wm]) * 1e3:.2f} (true 1.70) x 10^-3 mm^2/s")
print(f"radial diffusivity the tortuosity rule implies: {np.median(smt['d_par'][wm]) * (1 - np.median(smt['f'][wm])) * 1e3:.2f} "
      f"(true {D_RAD * 1e3:.2f}) x 10^-3 mm^2/s")
```

White matter bends: the direction-averaged stick loses signal slowly at high b (water
along the axons is lost only to gradients that point along them), so its curve flattens,
and that bend is what the two-compartment fit reads. Gray matter, two balls, bends less.
CSF falls steeply and would continue straight down, but by b = 2000 its signal has reached
the noise floor ([Chapter 8](../03-preprocessing/08-noise.md)) and flattens at about 0.01; that flattening is noise, not
tissue.

The fitted fraction (0.60) is close to the truth (0.55) and the diffusivity (1.75) is within
one grid step (0.05) of it. The residual offset in the fraction is model mismatch: the fit
ties the extra-axonal radial diffusivity to the fraction by the tortuosity rule (radial =
axial × (1 − fraction), 0.70 × 10⁻³ mm²/s for the fitted values), and the synthetic tissue,
with a radial diffusivity of 0.6 × 10⁻³, does not obey that rule.

## Degeneracy: what the sampling determines

A multi-compartment fit has a landscape of solutions, and with too little data that
landscape has a valley rather than a minimum: many combinations of fraction and
diffusivity explain the measurements equally well {cite:p}`jelescu2016`. The cost of the
spherical-mean model for one white matter voxel, as a function of its two parameters, shows
this directly:

```{code-cell} python
:tags: [hide-input]
i, j = np.argwhere(wm)[len(np.argwhere(wm)) // 2]
f_grid = np.linspace(0.05, 0.95, 91)
d_grid = np.linspace(0.8e-3, 2.4e-3, 81)
F, D = np.meshgrid(f_grid, d_grid, indexing="ij")

def stick_mean(bd):
    bd = np.maximum(bd, 1e-9)
    return np.sqrt(np.pi / (4 * bd)) * erf(np.sqrt(bd))

D_PERP = presets.ADULT_DIFFUSIVITY["WM_extra"][1]  # the model here is the synthetic tissue's own, with the radial diffusivity known

def cost(shell_set):
    c = np.zeros_like(F)
    for b in shell_set:
        meas = means[i, j, list(shells).index(b)] / means[i, j, 0]
        model = F * stick_mean(b * D) + (1 - F) * np.exp(-b * D_PERP) * stick_mean(b * (D - D_PERP))
        c += (model - meas) ** 2
    return c

fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
for ax, shell_set in zip(axes, [[1000], [1000, 2000], [1000, 2000, 3000]]):
    c = cost(shell_set)
    im = ax.imshow(np.log10(c + 1e-9), origin="lower", aspect="auto", cmap="viridis", extent=(d_grid[0] * 1e3, d_grid[-1] * 1e3, f_grid[0], f_grid[-1]), vmin=-8, vmax=-2)
    ax.plot(1.7, 0.55, "o", color="white", ms=6)
    ax.grid(False)
    ax.set(xlabel="axial diffusivity (x10^-3 mm²/s)", ylabel="intra-axonal fraction", title=f"shells {shell_set}")
cbar = fig.colorbar(im, ax=axes, shrink=0.8, label="log10 cost")
```

Each panel is a map of how badly each (diffusivity, fraction) pair fits, dark for a good
fit; the white dot is the truth. With one shell the cost is low along a whole curve: any
fraction can be traded for a diffusivity with no change in the fit, so the measurements do
not distinguish between those combinations. The second shell narrows the valley; the
third, at b = 3000, closes it around the true values.

Why the extra shell helps is easiest to see on the decay curves themselves. The next figure
takes the true pair and a second pair from the same one-shell valley, with a much smaller
fraction and a correspondingly different diffusivity, and draws the spherical mean each one
predicts; the dots are this voxel's measurements.

```{code-cell} python
:tags: [hide-input]
def sm_model(b, f, d):
    b = np.asarray(b, float)
    return f * stick_mean(b * d) + (1 - f) * np.exp(-b * D_PERP) * stick_mean(b * (d - D_PERP))

F_TRUE, DD_TRUE, F_ALT = 0.55, D_AX, 0.30
target = sm_model(1000, F_TRUE, DD_TRUE)
DD_ALT = brentq(lambda d: sm_model(1000, F_ALT, d) - target, 0.7e-3, 4e-3)
b_line = np.linspace(0, 3000, 121)
fig, ax = plt.subplots(figsize=(6.5, 3.4))
ax.plot(b_line, sm_model(b_line, F_TRUE, DD_TRUE), color=PALETTE[0], label=f"fraction {F_TRUE:.2f}, diffusivity {DD_TRUE * 1e3:.2f} (true)")
ax.plot(b_line, sm_model(b_line, F_ALT, DD_ALT), color=PALETTE[1], ls="--", label=f"fraction {F_ALT:.2f}, diffusivity {DD_ALT * 1e3:.2f}")
ax.plot(shells, means[i, j] / means[i, j, 0], "o", color=INK["primary"], ms=5, label="this voxel, measured")
for bb in (1000, 3000):
    ax.axvline(bb, color=INK["secondary"], lw=0.8, ls=":")
ax.set(xlabel="b (s/mm²)", ylabel="spherical mean / b=0", ylim=(0, 1.02), title="two fits from the one-shell valley")
ax.legend(fontsize=7)
fig.tight_layout()
for bb in (1000, 2000, 3000):
    print(f"b = {bb}: true pair {sm_model(bb, F_TRUE, DD_TRUE):.3f}, alternative pair {sm_model(bb, F_ALT, DD_ALT):.3f}, "
          f"measured {means[i, j, list(shells).index(bb)] / means[i, j, 0]:.3f}")
```

At b = 1000 the two curves pass through the same point, so a one-shell protocol cannot
choose between them. They leave b = 1000 with different slopes and bends and separate by
b = 3000 (0.251 against 0.209), a difference about four times the largest gap between a
measured point and the true curve (0.010, at b = 3000), which is what lets the three-shell
fit choose the true pair.

The model in both figures is the synthetic tissue's own, with the radial diffusivity
known, so the only ambiguity is the one the sampling leaves; a model with more unknowns has
a larger valley. This is the practical meaning of Table 6.1's requirement of two or more
shells for compartment models, and of the higher shell being high: the curvature of the
decay is the only information in a single-diffusion-time acquisition that separates the
parameters.

The other ways to break the degeneracy add a different kind of information rather than
more of the same: several diffusion times ([Chapter 22](../05-advanced/22-multi-diffusion-time.md)), several echo times ([Chapter 20](../05-advanced/20-multi-te.md)), or
b-tensor encoding ([Chapter 23](../05-advanced/23-frontiers.md)).

## NODDI and ball-and-stick on the simulated datasets

Neither is fitted here. NODDI's implementation (AMICO, or the original MATLAB toolbox) is
an optional dependency of this book, and ball-and-stick's reference implementation is
FSL's bedpostx, which the offline pipeline runs inside the QSIPrep image. Both will be
scored on the simulated datasets, where the truth includes NODDI-style ICVF, ODI, and ISOVF maps.

## Limits of Gaussian compartments

The simulated brain's compartments are Gaussian: a stick, a tensor, and balls. Real tissue has
finite axon diameters (the cylinder above), **exchange** (water crossing between
compartments, for example through the axon membrane, during the measurement, so a molecule
does not stay in one compartment), and **time-dependent diffusion** (an apparent
diffusivity that changes with how long the water is watched, because a longer diffusion
time gives it more chances to meet membranes and other obstacles; [Chapter 4](../02-diffusion-encoding/04-diffusion-in-tissue.md), [Chapter 22](../05-advanced/22-multi-diffusion-time.md)).
None of these is in the simulated brain. A model that assumes restriction (a cylinder with a diameter) fits the
simulated brain differently than it fits tissue, and its parameters on simulated data should not be
read as validation of what they mean in vivo. The simulated brain is useful for a narrower
question: given a model, how much of its error comes from the sampling scheme and the noise
as opposed to the model itself. The fits above separate the two by comparing against a
known answer at several schemes.

## Measure it: the simulated datasets

:::{admonition} Simulated dataset pending
:class: note
This section will fit the free-water, spherical-mean, NODDI (AMICO), and ball-and-stick
(bedpostx) models to the `ref-schemes` dataset and score them against the `truth` maps
`icvf`, `odi`, `isovf`, and the simulated brain's known compartment fractions, scheme by scheme.
:::

## What this implies for acquisition

- **Two shells are the minimum for any compartment model**, and the upper shell should be
  at b ≥ 2000; three shells make the fit well determined.
- **Fixed diffusivities (NODDI) make a model fit from less data** and make its values
  depend on those fixed numbers; report them.
- **Free-water correction needs multi-shell data** to be well posed.
- **IVIM needs its own low-b shells** (b = 0 to 200 in several steps); no standard
  diffusion protocol contains them.
- **A model's assumptions are acquisition decisions**: check that the protocol supports the
  model before scanning, not after.

## Further reading

The standard model and its estimation {cite:p}`novikov2019`, degeneracy
{cite:p}`jelescu2016`, ball-and-stick {cite:p}`behrens2003`, NODDI {cite:p}`zhang2012`,
the spherical mean technique {cite:p}`kaden2016`, and free-water elimination
{cite:p}`pasternak2009`.
