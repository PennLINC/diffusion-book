---
title: "15. Signal representations"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** a synthetic multi-shell series built from the packaged tissue maps ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
- **`ref-clean`** (pending): the artifact-free, noise-free reference series with its truth maps and true fiber orientations ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-ref-clean)).
- **`ref-schemes`** (pending): the simulated brain under the 30-direction, 64-direction, HBCD, DSI, and CS-DSI schemes at matched scan time ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-ref-schemes)).
- **`truth`** (pending): the 27 analytic ground-truth maps and the true fiber orientations ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-truth), [Appendix E](../appendices/e-truth-map-catalogue.md)).

Pipeline-tier datasets are simulated offline by TRXScan ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)) and are marked *pending* until their release; the figures that need them say so where they will appear.
:::

## Learning goals

After this chapter you can:

- fit the diffusion tensor, the kurtosis tensor, and a propagator representation (MAP-MRI)
  and state the assumptions of each
- state the minimum sampling each representation needs and show what happens when the
  data fall short of it
- choose a tensor fitting method and say when a robust one is needed
- read the fitted maps against a known answer

```{code-cell} python
:tags: [hide-cell]
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import warnings
warnings.filterwarnings("ignore")  # dipy's fit notices are not part of the lesson
import numpy as np
import matplotlib.pyplot as plt
from dipy.core.gradients import gradient_table
from dipy.core.sphere import Sphere
from dipy.reconst.dti import TensorModel
from dipy.reconst.dki import DiffusionKurtosisModel
from dipy.reconst.mapmri import MapmriModel

from dwibook import phantoms, presets, schemes, signal, synth
from dwibook.plotting import INK, PALETTE, TISSUE_COLORS, set_style, show_image

set_style()
```

## Representations versus models

The chapters of Part IV fit two kinds of description to the same data. A **signal
representation** describes the measured signal as a function of b-value and direction with
no claim about the tissue: the diffusion tensor, the kurtosis tensor, and the propagator
bases are representations. Their parameters (FA, MD, kurtosis, return-to-origin probability)
are summaries of the signal that change with tissue but do not name a tissue property. A
**biophysical model** ([Chapter 17](./17-microstructure-models.md)) assigns the signal to compartments with named
properties, such as an intra-axonal fraction, at the cost of assumptions that may not hold.
Representations are the safer choice when the question is whether something differs;
models are needed when the question is what differs. For example, a lower FA in a patient
group says only that the signal depends less on direction there, which fewer axons, more
crossing fibers, more dispersed fibers, or more free water could each produce; a lower
intra-axonal fraction claims specifically that there is less water inside axons, and is
right only if the model's compartments match the tissue.

## The synthetic series

The toy tier uses the 2 mm slice with a three-shell scheme (20, 20, and 30 directions at
b = 1000, 2000, 3000, plus four b=0), noise at SNR 25 in white matter at b = 0 (the signal-to-noise
ratio: the white matter b = 0 signal is 25 times the standard deviation of the noise), and the
noise-free fit of the same scheme as the reference. The synthetic white matter is the
simulated brain's two-compartment model, so it has curvature in its decay ([Chapter 5](../02-diffusion-encoding/05-diffusion-encoding.md)) and the
kurtosis and propagator representations have something to measure.

```{code-cell} python
:tags: [hide-input]
t = phantoms.brain_slice()
mask = t["mask"]
wm, gm = t["wm"] > 0.9, t["gm"] > 0.9
bvals, bvecs = schemes.multi_shell({1000: 20, 2000: 20, 3000: 30}, n_b0=4)
clean = synth.synthetic_dwi(t, bvals, bvecs)
sigma = clean[wm][:, bvals == 0].mean() / 25
noisy = synth.add_noise(clean, sigma, seed=0)
gtab = gradient_table(bvals, bvecs=bvecs)
sub = lambda keep: (bvals[keep], bvecs[keep])
print(f"{len(bvals)} volumes; shells {schemes.shells_of(bvals)}")
```

## The diffusion tensor

The tensor represents the signal as a single Gaussian displacement distribution: an
ellipsoid with three axes. Six parameters plus the b=0 signal describe it, so six
directions at one b-value are the mathematical minimum and about 30 the practical one
([Chapter 6](../02-diffusion-encoding/06-qspace-sampling.md)). From the ellipsoid come the standard maps: **mean diffusivity** (MD, the
average of the three axes), **fractional anisotropy** (FA, how elongated the ellipsoid is,
from 0 to 1), axial and radial diffusivity, and the **principal direction**, the long axis,
which tractography follows. **Axial diffusivity** (AD) is the length of the long axis and
**radial diffusivity** (RD) the average of the two short ones.

The figure makes these concrete. The top row fits a tensor to one voxel of each pure tissue,
built from the book's own compartment signals at b = 1000, and draws its ellipsoid, with
axes proportional to how far water typically moves in each direction; all three are drawn
at the same scale. Read each axis as a distance: MD is the average of the three axis
lengths (squared), and FA is how unequal they are. The overall size of the shape is not MD;
a cigar and a sphere with the same MD look different in size. CSF is a
large sphere (fast and the same in every direction), gray matter a smaller sphere (slower,
still the same in every direction), and white matter a cigar along its fibers. The bottom
row shows the same quantities as maps of the noise-free slice, fitted to b ≤ 1000: look for
CSF bright in the MD map, white matter bright in the FA map, and, in the color map, the hue
of each white matter bundle telling which way it runs.

```{code-cell} python
:tags: [hide-input]
def fit_dti(series, keep, method="WLS", **kw):
    b, v = sub(keep)
    return TensorModel(gradient_table(b, bvecs=v), fit_method=method, **kw).fit(series[..., keep], mask=mask)

ref_low = fit_dti(clean, bvals <= 1000)

# One voxel of each pure tissue from the compartment signals, fibers along the x axis.
low = bvals <= 1000
fiber, across = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
voxels = {"CSF": signal.csf(bvals), "GM": signal.gray_matter(bvals),
          "WM": signal.white_matter(bvals, bvecs @ fiber)}
glyph_fits = {k: TensorModel(gradient_table(bvals[low], bvecs=bvecs[low])).fit(s[low]) for k, s in voxels.items()}

u, v = np.mgrid[0:2 * np.pi:48j, 0:np.pi:24j]
unit = np.stack([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)]).reshape(3, -1)
scale = np.sqrt(glyph_fits["CSF"].evals.max())  # common scale: axes ~ sqrt(eigenvalue), CSF radius = 1
names = {"CSF": "CSF", "GM": "gray matter", "WM": "white matter"}
fig = plt.figure(figsize=(9.5, 6.8))
for i, (k, f) in enumerate(glyph_fits.items()):
    ax = fig.add_subplot(2, 3, i + 1, projection="3d")
    pts = f.evecs @ (np.sqrt(np.clip(f.evals, 0, None))[:, None] * unit) / scale
    ax.plot_surface(*pts.reshape(3, *u.shape), color=TISSUE_COLORS[k], linewidth=0, shade=True, alpha=0.9)
    ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1)); ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
    ax.set_title(f"{names[k]}\nMD {1e3 * f.md:.2f} ×10⁻³ mm²/s, FA {f.fa:.2f}", fontsize=9, y=0.93)
    print(f"{names[k]:>12}: MD {1e3 * f.md:.2f} x10^-3 mm²/s, FA {f.fa:.2f}, "
          f"AD {1e3 * f.ad:.2f}, RD {1e3 * f.rd:.2f} x10^-3 mm²/s")

evec = ref_low.evecs[..., :, 0]  # principal direction, (row, column, slice) components
dec = np.clip(ref_low.fa, 0, 1)[..., None] * np.abs(evec[..., [1, 0, 2]]) * mask[..., None]
panels = [(ref_low.md * 1e3 * mask, "MD (×10⁻³ mm²/s)", dict(vmin=0, vmax=3)),
          (ref_low.fa * mask, "FA", dict(vmin=0, vmax=1))]
for i, (img, title, kw) in enumerate(panels):
    ax = fig.add_subplot(2, 3, 4 + i)
    show_image(ax, img, title, kind="scalar", **kw)
    fig.colorbar(ax.images[0], ax=ax, shrink=0.75, pad=0.02)
ax = fig.add_subplot(2, 3, 6)
ax.imshow(np.clip(dec / np.percentile(dec[mask], 99), 0, 1)); ax.set_axis_off()
ax.set_title("FA colored by principal direction\nred: left-right, green: up-down", fontsize=9)
fig.tight_layout()
```

The printed numbers are the ones on the glyphs. The two spheres have FA 0 because diffusion
in them is the same in every direction. The white matter cigar has FA 0.85 but an MD close
to gray matter's, because it is fast along the fibers (AD 1.72 ×10⁻³ mm²/s) and slow across
them (RD 0.23 ×10⁻³ mm²/s), and MD averages the three directions: one long axis and two
short ones average to about the same as gray matter's three medium ones, even though the
cigar looks smaller. Shape and average are separate facts. The synthetic fibers
all lie in the plane of the slice, so the color map has no blue (through-plane) component.

The Gaussian assumption holds only at low b-value. Above about b = 1500 the curvature of
the true decay ([Chapter 5](../02-diffusion-encoding/05-diffusion-encoding.md), and the
one-shell-versus-four-shells figure of [Chapter 6](../02-diffusion-encoding/06-qspace-sampling.md))
violates it, and a tensor fitted to high-b data returns diffusivities that depend on which
b-values were included. The figure plots the white matter log signal against b, averaged
over directions and voxels, and three straight tensor fits, each using the shells up to a
different b. A tensor is a straight line on this plot, and its slope is the MD. Look at how
the line tilts less as higher shells are included: the points bend upward, and a straight
line drawn through more of the bend has to be flatter. The right panel plots the three
slopes against the highest b in each fit.

```{code-cell} python
:tags: [hide-input]
ranges = {"b ≤ 1000": bvals <= 1000, "b ≤ 2000": bvals <= 2000, "b ≤ 3000": bvals >= 0}
fits_by_range = {k: fit_dti(noisy, keep, return_S0_hat=True) for k, keep in ranges.items()}

shells = np.array([0.0, 1000.0, 2000.0, 3000.0])
s0 = noisy[..., bvals == 0].mean(-1)
log_sig = [0.0] + [np.mean(np.log(noisy[wm][:, bvals == b] / s0[wm][:, None])) for b in shells[1:]]
bb = np.linspace(0, 3200, 161)
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4.0), gridspec_kw=dict(width_ratios=[1.7, 1]))
ax2.plot([1000, 2000, 3000], [1e3 * f.md[wm].mean() for f in fits_by_range.values()], "-", color=INK["secondary"], lw=1)
for (label, f), col, bmax in zip(fits_by_range.items(), PALETTE[:3], [1000, 2000, 3000]):
    ax2.plot(bmax, 1e3 * f.md[wm].mean(), "o", color=col, ms=9)
    md, icpt = f.md[wm].mean(), np.mean(np.log(f.S0_hat[wm] / s0[wm]))  # slope, and the fit's own intercept
    ax.plot(bb[bb <= bmax], icpt - bb[bb <= bmax] * md, color=col, lw=2.2,
            label=f"tensor fitted to {label}: slope = MD = {1e3 * md:.3f} ×10⁻³ mm²/s")
    ax.plot(bb[bb >= bmax], icpt - bb[bb >= bmax] * md, color=col, lw=1.2, ls=":")
ax.plot(shells, log_sig, "o", color=INK["primary"], ms=7, zorder=3,
        label="white matter, measured (mean over directions)")
ax.set(xlabel="b (s/mm²)", ylabel="log signal, ln(S / S₀)", xlim=(0, 3200))
ax.legend(loc="lower left", fontsize=7.5)
ax2.set(xlabel="highest b in the tensor fit (s/mm²)", ylabel="white matter MD (×10⁻³ mm²/s)",
        xticks=[1000, 2000, 3000], xlim=(700, 3300), ylim=(0.6, 0.76), title="the slope, i.e. MD, flattens")
fig.tight_layout()
```

```{code-cell} python
:tags: [hide-input]
labels = {"b ≤ 1000": "b <= 1000 (reference use)", "b ≤ 2000": "b <= 2000", "b ≤ 3000": "all shells to 3000"}
print(f"{'tensor fitted on':>28}   WM MD (x10^-3)   WM FA   GM MD (x10^-3)")
for key, f in fits_by_range.items():
    print(f"{labels[key]:>28}   {1e3 * f.md[wm].mean():.3f}            {f.fa[wm].mean():.3f}   {1e3 * f.md[gm].mean():.3f}")
```

White matter MD falls from 0.725 to 0.699 to 0.667 ×10⁻³ mm²/s as the b = 2000 and
b = 3000 shells are added, and gray matter MD falls the same way (0.757 to 0.684), because
the fit averages a decay that is slower than exponential at high b. The values are all
"correct" for their own b-range and none is the tissue's diffusivity; this is why tensor
studies specify the b-value and why comparing MD across protocols with different b-values
is not valid. Tensor metrics are fitted to the b ≤ 1000 shell of a multi-shell acquisition
for this reason.

### Fitting methods

A tensor is fitted by finding the ellipsoid whose predicted signals are closest to the
measured ones; the methods differ in which measurements they trust most.

- **Weighted least squares** (WLS) is the default in most tools. It trusts bright
  measurements more than dim ones, because the dim ones are the noisiest.
- **RESTORE** {cite:p}`chang2005` is WLS that also sets aside measurements that disagree
  badly with the rest, which protects the tensor from dropout slices and spikes that were
  not caught upstream ([Chapter 12](../03-preprocessing/12-motion-and-dropout.md)).

Two other methods appear in software: ordinary least squares, which trusts every
measurement equally and is less accurate than WLS, and nonlinear least squares, which is
slightly more accurate than WLS at low SNR but slower.

The figure plants a dropout that preprocessing missed: in one b = 1000 volume, twenty rows
of the slice keep only 30 % of their signal (left panel). The other panels show how far the
FA fitted by each method is from the noise-free reference; red and blue are errors, white
is none. Look at the horizontal band at the height of the dropout.

```{code-cell} python
:tags: [hide-input]
corrupted = noisy.copy()
corrupted[40:60, :, 7] *= 0.3  # a partial dropout in one b = 1000 volume, not caught upstream
ref = ref_low
rowsm = [("WLS", {}), ("NLLS", {}), ("RESTORE", {"sigma": sigma})]
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
show_image(axes[0], corrupted[..., 7] * mask, "b = 1000 volume, dropout band", kind="magnitude")
for ax, (method, kw) in zip(axes[1:], rowsm):
    f = fit_dti(corrupted, bvals <= 1000, method=method, **kw)
    err = np.abs(f.fa - ref.fa)[wm]
    show_image(ax, (f.fa - ref.fa) * mask, f"FA error, {method}", kind="diff", vmin=-0.3, vmax=0.3)
    print(f"{method:>8}: FA error in WM {err.mean():.3f} (rows with the dropout: {np.abs(f.fa - ref.fa)[40:60][wm[40:60]].mean():.3f})")
fig.tight_layout()
```

The uncaught dropout leaves a band of wrong FA in the WLS and NLLS fits; RESTORE
recognizes the affected measurements as outliers and fits without them. In the dropout rows
the white matter FA error is 0.040 with WLS and 0.023 with RESTORE; over all white matter,
0.023 and 0.019. Robust fitting is
not a substitute for outlier replacement in preprocessing, which uses all volumes to decide,
but it is a useful last line of defense.

## The kurtosis tensor

The previous figure showed the white matter log signal bending away from a straight line.
Diffusion kurtosis imaging {cite:p}`jensen2005` measures that bend. It adds one term to the
tensor, so that in each direction the log signal is allowed to curve with b instead of
falling in a straight line; the **kurtosis** is how strongly it curves. A straight line
means Gaussian diffusion and zero kurtosis. The figure shows the three pure tissues, each
with the straight tensor fit (to b ≤ 1000) and the curved kurtosis fit (to all shells),
for the gradient across the fibers in white matter. Look at the gap between the two lines
at high b: wide in white matter, smaller in gray matter, and absent in CSF, whose signal is
a straight line.

```{code-cell} python
:tags: [hide-input]
tissue_curves = {"WM": lambda b: signal.white_matter(b, 0.0), "GM": signal.gray_matter, "CSF": signal.csf}
titles = {"WM": "white matter, across the fibers", "GM": "gray matter", "CSF": "CSF"}
bb = np.linspace(0, 3200, 161)
fig, axes = plt.subplots(1, 3, figsize=(10, 3.7))
for ax, (k, curve) in zip(axes, tissue_curves.items()):
    dki = DiffusionKurtosisModel(gtab, fit_method="WLS").fit(voxels[k])
    d_k = across @ dki.quadratic_form @ across                  # apparent diffusivity across
    kap = dki.akc(Sphere(xyz=across[None]))[0]                  # apparent kurtosis across
    kap = 0.0 if abs(kap) < 0.005 else kap
    d_t = across @ glyph_fits[k].quadratic_form @ across        # tensor, b <= 1000
    ax.semilogy(bb, np.exp(-bb * d_t), color=INK["secondary"], ls="--", lw=1.5, label="straight (tensor, b ≤ 1000)")
    ax.semilogy(bb, np.exp(-bb * d_k + bb**2 * d_k**2 * kap / 6), color=TISSUE_COLORS[k], lw=2,
                label="curved (kurtosis, all shells)")
    ax.semilogy(shells, curve(shells), "o", color=TISSUE_COLORS[k], mec=INK["primary"], ms=7, label="signal at the shells")
    ax.set(xlabel="b (s/mm²)", title=f"{titles[k]}\nkurtosis {kap:.2f}")
    print(f"{titles[k]:>32}: apparent kurtosis {kap:.2f}; at b = 3000 the straight line gives "
          f"{np.exp(-3000 * d_t):.3g}, the signal is {curve(3000.0):.3g}")
    if k == "WM":
        top, bottom = np.exp(-2800 * d_k + 2800**2 * d_k**2 * kap / 6), np.exp(-2800 * d_t)
        ax.annotate("", xy=(2800, bottom), xytext=(2800, top), arrowprops=dict(arrowstyle="<->", color=INK["primary"]))
        ax.text(2650, bottom * 0.84, "kurtosis =\nhow much\nthis bends", ha="center", va="top", fontsize=8)
for ax, lo, ticks in [(axes[0], 0.25, [0.3, 0.5, 1.0]), (axes[1], 0.08, [0.1, 0.2, 0.5, 1.0]),
                      (axes[2], 5e-5, [1e-4, 1e-3, 1e-2, 0.1, 1.0])]:
    ax.set_ylim(lo, 1.15); ax.set_yticks(ticks, labels=[f"{v:g}" for v in ticks]); ax.minorticks_off()
axes[0].set_ylabel("S / S₀ (log scale)")
axes[1].legend(loc="lower left", fontsize=7)
fig.tight_layout()
```

The kurtosis fit puts CSF's curve on top of its straight line (kurtosis 0), gray matter
bends a little (0.36), and white matter across the fibers bends most (3.07): at b = 3000
its signal is 0.624 where the tensor predicts 0.503. The bend
represents the non-Gaussian part of the decay without naming its cause. **Mean kurtosis**
(MK), the kurtosis averaged over directions, is elevated wherever diffusion is restricted
or the voxel mixes compartments, so it is high in white matter and in dense gray matter,
and near zero in CSF.

Fitting the curvature requires at least two non-zero shells, and the upper shell must be
high enough for the curvature to be visible, in practice b = 2000–3000 ([Chapter 5](../02-diffusion-encoding/05-diffusion-encoding.md)). A single
shell cannot support the fit at all: it gives one point per direction on the curve above,
besides b = 0, and a bend needs at least two.

```{code-cell} python
:tags: [hide-input]
def fit_dki(series, keep):
    b, v = sub(keep)
    return DiffusionKurtosisModel(gradient_table(b, bvecs=v), fit_method="WLS").fit(series[..., keep], mask=mask)

try:
    fit_dki(noisy, bvals <= 1000)
except ValueError:
    print("DKI on one shell (b = 1000 plus b = 0): refused; the fit needs at least three distinct b-values.")

ref_dki = fit_dki(clean, bvals >= 0)
mk_ref = ref_dki.mk(min_kurtosis=0, max_kurtosis=3)
fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2))
show_image(axes[0], mk_ref * mask, "MK, noise-free, three shells", kind="scalar", vmin=0, vmax=1.5)
for ax, (label, keep) in zip(axes[1:], [("two shells (1000, 2000)", bvals <= 2000), ("three shells", bvals >= 0)]):
    f = fit_dki(noisy, keep)
    mk = f.mk(min_kurtosis=0, max_kurtosis=3)
    show_image(ax, mk * mask, f"MK, noisy, {label}", kind="scalar", vmin=0, vmax=1.5)
    print(f"{label:>24}: MK error in WM {np.abs(mk - mk_ref)[wm].mean():.3f}, in GM {np.abs(mk - mk_ref)[gm].mean():.3f}")
fig.tight_layout()
```

Kurtosis is estimated from the bend of the decay curve, a small difference between how
fast the signal falls at low and at high b. Noise that barely moves the average slope (the
diffusivity) moves that difference much more, so MK is noisier than MD from the same data;
it is the representation most improved by denoising ([Chapter 8](../03-preprocessing/08-noise.md)) and most damaged by the
Rician floor, which adds curvature of its own at high b.

## Propagator representations: MAP-MRI

The tensor and the kurtosis tensor describe the signal. A propagator representation goes
one step further and describes the **displacement distribution** ([Chapter 4](../02-diffusion-encoding/04-diffusion-in-tissue.md)): for the
water in a voxel, how likely each displacement is during the measurement. The figure draws
this distribution along one line through the voxel for two tissues, using the simulated
brain's compartments and the pulse timing used below. In CSF, water moves freely and the
distribution is a wide bell curve (a Gaussian). In white matter, measured across the
fibers, it is the sum of two parts: water inside the axons, which cannot move farther than
the axon is wide (drawn here with a width of 1 µm, since the simulated brain's axons have
zero width), and water between the axons, which moves slowly across them. Three numbers
summarize each curve; look at how each is read off the drawing.

```{code-cell} python
:tags: [hide-input]
t_d = 0.030 - 0.010 / 3  # effective diffusion time (s) of the pulse timing used below
x = np.linspace(-0.045, 0.045, 3601)  # displacement (mm)
dx = x[1] - x[0]
gauss = lambda var: np.exp(-x**2 / (2 * var)) / np.sqrt(2 * np.pi * var)
f_in, d_perp = presets.ADULT_FRACTIONS["WM_intra"], presets.ADULT_DIFFUSIVITY["WM_extra"][1]
profiles = {
    "CSF": (gauss(2 * presets.ADULT_DIFFUSIVITY["CSF"] * t_d), TISSUE_COLORS["CSF"]),
    "white matter, across the fibers": (f_in * gauss(0.001**2) + (1 - f_in) * gauss(2 * d_perp * t_d), TISSUE_COLORS["WM"]),
}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, (name, (p, col)) in zip(axes, profiles.items()):
    p0, rms = np.interp(0, x, p), np.sqrt(np.sum(x**2 * p) * dx)
    # best-fitting Gaussian: the one whose shape is closest to p (largest normalized overlap)
    cands = [gauss(s**2) for s in np.linspace(0.0005, 0.03, 600)]
    cos = [np.sum(p * g) / np.sqrt(np.sum(p**2) * np.sum(g**2)) for g in cands]
    g = cands[int(np.argmax(cos))]
    best = g * np.sum(p * g) / np.sum(g * g)  # least-squares projection of p onto that Gaussian
    ng = np.sqrt(max(0.0, 1 - max(cos)**2))   # = |p - best| / |p|, as MAP-MRI defines it
    xu = x * 1e3
    ax.fill_between(xu, p, best, color=PALETTE[3], alpha=0.35, lw=0, label="gap to the best-fitting Gaussian = non-Gaussianity")
    ax.plot(xu, best, color=INK["secondary"], ls="--", lw=1.3, label="best-fitting Gaussian")
    ax.plot(xu, p, color=col, lw=2.2, label="displacement distribution")
    ax.plot([0], [p0], "o", color=INK["primary"], ms=6, zorder=4)
    ax.annotate(f"height at zero = return to origin\n({p0:.0f} per mm)", xy=(0, p0), xytext=(6, p0 + 50),
                fontsize=8, va="center", arrowprops=dict(arrowstyle="-", color=INK["secondary"], lw=0.8))
    yb = np.interp(rms, x, p)
    ax.annotate("", xy=(-rms * 1e3, yb), xytext=(rms * 1e3, yb), arrowprops=dict(arrowstyle="<->", color=INK["primary"]))
    for sx in (-rms * 1e3, rms * 1e3):
        ax.plot([sx, sx], [0, yb], color=INK["secondary"], lw=0.7, ls=":")
    ax.text(rms * 1e3 + 2, yb + 8, f"width = √MSD\n= {rms * 1e3:.1f} µm", fontsize=8, va="bottom")
    ax.set(xlabel="displacement during the measurement (µm)", title=f"{name}\nnon-Gaussianity {ng:.2f}",
           xlim=(-40, 40), ylim=(-8, 330))
    print(f"{name:>32}: height at zero {p0:.0f} /mm, RMS displacement {rms * 1e3:.1f} µm, non-Gaussianity {ng:.2f}")
axes[0].set_ylabel("probability density (per mm)")
axes[0].legend(loc="upper left", fontsize=7)
fig.tight_layout()
```

The **return-to-origin probability** (RTOP) is the height of the curve at zero: the
likelihood that a molecule ends the measurement where it started. It is high where water is
held in place, as inside axons, and low where it moves freely. The **mean squared
displacement** (MSD) is the width: how far water typically travels. **Non-Gaussianity**
(NG) is the gap between the curve and the bell curve closest to it, on a scale from 0 (a
perfect Gaussian, as in CSF) to 1. In the figure, white matter is 251 per mm high at zero
against 32 for CSF, 3.9 µm wide against 12.6 µm, and has a non-Gaussianity of 0.19 against
0. RTOP asks the return-to-origin question for motion in
every direction at once; its two variants ask it for motion across the fiber only
(return-to-axis, RTAP) and along it only (return-to-plane, RTPP).

Mean apparent propagator MRI {cite:p}`ozarslan2013` estimates this distribution in three
dimensions from the measured signal, by representing it in a basis of functions, and
computes these summaries from it. Because it represents the whole distribution it needs the
whole q-space: at least two shells, preferably three or more, with directions spread across
them. A Laplacian regularization, a penalty on fits that wiggle more than the data justify,
keeps the fit stable at the sampling densities of typical protocols {cite:p}`fick2016`. The
displacements grow with the diffusion time, so the pulse timing must be known to report
these maps in physical units.

```{code-cell} python
:tags: [hide-input]
gtab_map = lambda keep: gradient_table(bvals[keep], bvecs=bvecs[keep], big_delta=0.030, small_delta=0.010)
def fit_map(series, keep):
    return MapmriModel(gtab_map(keep), radial_order=4, laplacian_regularization=True, laplacian_weighting=0.2, positivity_constraint=False, bval_threshold=1500).fit(series[..., keep], mask=mask)

ref_map = fit_map(clean, bvals >= 0)
fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2))
show_image(axes[0], np.cbrt(ref_map.rtop()) * mask, "RTOP^(1/3), noise-free (1/mm)", kind="scalar", vmin=0, vmax=120)
show_image(axes[1], ref_map.msd() * mask * 1e3, "mean squared displacement (x10^-3 mm²)", kind="scalar", vmin=0, vmax=0.2)
show_image(axes[2], ref_map.ng() * mask, "non-Gaussianity", kind="scalar", vmin=0, vmax=0.6)
fig.tight_layout()
csf_m = t["csf"] > 0.9
c_ref = np.cbrt(ref_map.rtop())
print("noise-free RTOP^(1/3): " + ", ".join(f"{n} {c_ref[m].mean():.0f} /mm (1/RTOP^(1/3) = {1e3 / c_ref[m].mean():.0f} µm)"
                                           for n, m in [("WM", wm), ("GM", gm), ("CSF", csf_m)]))
print(f"noise-free WM non-Gaussianity {ref_map.ng()[wm].mean():.3f}")
for label, keep in [("two shells", bvals <= 2000), ("three shells", bvals >= 0)]:
    f = fit_map(noisy, keep)
    print(f"{label:>12}: RTOP error in WM {100 * np.abs(np.cbrt(f.rtop()) - np.cbrt(ref_map.rtop()))[wm].mean() / np.cbrt(ref_map.rtop())[wm].mean():.1f} %, "
          f"non-Gaussianity error in WM {np.abs(f.ng() - ref_map.ng())[wm].mean():.3f} (mean WM value {f.ng()[wm].mean():.3f})")
f = fit_map(clean, bvals <= 2000)
print(f"two shells, noise-free: non-Gaussianity error in WM {np.abs(f.ng() - ref_map.ng())[wm].mean():.3f}")
```

Look first at where RTOP is high. The left map is brightest in white matter, where water
inside axons is held in place, intermediate in gray matter, and darkest in CSF, where water
moves freely; the same order as the heights at zero in the previous figure. RTOP is a
probability per unit volume, so its units are 1/mm³; taking the cube root turns it into an
inverse length, and one over that number is roughly the size of the region water explores
during the measurement: about 11 µm in white matter, 14 µm in gray matter, and 29 µm in
CSF. The mean squared displacement map is the reverse picture, largest in CSF (it saturates
the color scale there). Non-Gaussianity is highest in white matter, as in the 1-D figure;
the nonzero values in CSF are an artifact of the fit, since free water is exactly Gaussian.

The printed errors say what a third shell buys. RTOP error in white matter falls from
2.7 % with two shells to 1.6 % with three, so a propagator protocol should reach
b = 3000. Non-Gaussianity behaves differently: its two-shell error (0.021) is systematic,
present even without noise (0.023 noise-free), because b ≤ 2000 hardly sees the non-Gaussian
part of the decay; with the third shell the error doubles to 0.040, and the white matter
mean rises from 0.193 to 0.224, because at SNR 25 the Rician noise floor at b = 3000 adds
curvature of its own that the fit reads as non-Gaussianity. The highest shell carries both
the information and the noise, so a protocol that reports non-Gaussianity needs enough SNR
at its top shell, or denoising ([Chapter 8](../03-preprocessing/08-noise.md)).

## Beyond the pulse pair: QTI

Every representation above reads the signal from a single-direction encoding. Encoding
with several gradient directions inside one measurement (b-tensor encoding: the gradient
turns during the encoding, so one measurement weighs diffusion along a plane or along all
directions at once) provides a further representation, q-space trajectory imaging, whose
parameters separate microscopic anisotropy (elongated compartments, whatever their
orientation) from orientation dispersion (how spread those orientations are), a
distinction the tensor cannot make: to the tensor, many randomly oriented sticks and a
ball of free water both look isotropic. The simulated brain's
truth maps include these quantities, but the simulator does not yet produce b-tensor
acquisitions, so this book states the idea ([Chapter 23](../05-advanced/23-frontiers.md)) without fitting it.

## Measure it: the simulated datasets

:::{admonition} Simulated dataset pending
:class: note
This section will fit the tensor, kurtosis, and MAP-MRI representations to the
`ref-schemes` dataset (30-direction, 64-direction, HBCD, DSI, and CS-DSI schemes) and score
each against the analytic `truth` maps TRXScan writes for the same simulated brain: FA, MD, AD, RD,
MK, AK, RK, RTOP, RTAP, RTPP, MSD, and non-Gaussianity.
:::

## What this implies for acquisition

- **A tensor needs 30 directions at b ≈ 1000**, and its values are specific to that b.
- **Kurtosis needs two shells with the upper one at b ≥ 2000**, and benefits more from
  denoising than any other representation.
- **Propagator representations need three or more shells** with directions spread across
  them, and the pulse timing recorded.
- **Fit the tensor to the low shell of a multi-shell scheme**, not to all of it.
- **Use a robust fit** if outlier replacement was not run.

## Further reading

The diffusion tensor {cite:p}`basser1994`, kurtosis {cite:p}`jensen2005`, MAP-MRI
{cite:p}`ozarslan2013,fick2016`, and the distinction between representations and models
{cite:p}`novikov2018`.
