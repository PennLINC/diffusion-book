---
title: "9. Gibbs ringing"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** a synthetic series on the packaged 1 mm slice, acquired at the k-space band of a 2 mm matrix ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
- **Simulated live in this page:** one slice of the simulated brain acquired with the object oversampled (ringing intrinsic), on the acquisition grid (no ringing), and with a Hann window.
- **`gibbs`** (pending): ringing intrinsic to the acquisition versus none, plus a Hann-apodized variant ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-gibbs)).
- **`truth`** (pending): the 27 analytic ground-truth maps and the true fiber orientations ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-truth), [Appendix E](../appendices/e-truth-map-catalogue.md)).

Live-tier figures simulate one slice of the simulated brain in the page through TRXScan's Python package ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md#live-tier)); they run in seconds at build time.
Pipeline-tier datasets are simulated offline by TRXScan ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)) and are marked *pending* until their release; the figures that need them say so where they will appear.
:::

## Learning goals

After this chapter you can:

- explain why every acquisition rings at sharp edges and why diffusion data ring worst at
  the ventricles
- show how the ringing changes with b-value and what that does to fitted diffusivities
- apply sub-voxel-shift unringing and measure what it removes
- compare correcting the ringing after the fact with apodizing at acquisition

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt
from dipy.denoise.gibbs import gibbs_removal
from scipy import ndimage
from scipy.special import sici

from dwibook import kspace, phantoms, presets, schemes, synth
from dwibook.plotting import INK, PALETTE, TISSUE_COLORS, animate, set_style, show_image

set_style()
```

## The physics

An acquisition samples k-space out to a finite frequency, the edge of k-space $k_{max}$, and
the image is reconstructed from that limited set ([Chapter 2](../01-mri-physics/02-spatial-encoding-kspace.md)).
A sharp edge in the object needs all frequencies to be represented; with the high ones
missing, the reconstruction rings. The simplest place to watch this is a single step edge in
one dimension, dark on the left and bright on the right. The animation keeps k-space out to a
growing $k_{max}$: the finer the acquisition, the smaller its voxels, since one voxel is
$1/(2 k_{max})$ wide. The curve is what the truncated k-space describes between the samples;
the dots are the voxel centers, the only values the image actually stores. Here the edge
happens to fall exactly on a voxel center.

```{code-cell} python
:tags: [hide-input]
def ringing_curve(x, k_max, edge=0.0):
    """A unit step at `edge` with only the frequencies |k| <= k_max kept (x in mm, k in cycles/mm)."""
    return 0.5 + sici(2 * np.pi * k_max * (np.asarray(x, dtype=float) - edge))[0] / np.pi

def voxel_centers(dx, shift=0.0, half_width=10.0):
    """Voxel centers every dx, offset by `shift` voxels from the edge, within +-half_width."""
    n = np.arange(-np.ceil(half_width / dx) - 1, np.ceil(half_width / dx) + 2)
    c = (n + shift) * dx
    return c[np.abs(c) <= half_width]

x = np.linspace(-10, 10, 4001)
voxel_sizes = np.geomspace(5.0, 0.5, 54)  # mm; k_max = 1 / (2 * voxel)
N_HOLD = 14
fig, ax = plt.subplots(figsize=(8, 3.6))
fig.subplots_adjust(left=0.09, right=0.97, bottom=0.15, top=0.9)

def frame(i):
    ax.clear()
    dx = voxel_sizes[min(i, len(voxel_sizes) - 1)]
    k_max = 1 / (2 * dx)
    curve = ringing_curve(x, k_max)
    ax.plot(x, (x >= 0).astype(float), color="0.3", lw=1, ls="--", label="the object: a step")
    ax.plot(x, curve, color=PALETTE[1], lw=1.5, label="reconstruction from |k| ≤ k_max")
    c = voxel_centers(dx)
    ax.plot(c, ringing_curve(c, k_max), "o", color=PALETTE[0], ms=5, label="voxel centers (what the image stores)")
    peak = 1 / (2 * k_max)  # the first maximum sits one voxel from the edge
    top = ringing_curve(peak, k_max)
    ax.annotate("", xy=(peak, top), xytext=(peak, 1.0), arrowprops=dict(arrowstyle="<->", color=INK["primary"], lw=1, shrinkA=0, shrinkB=0))
    ax.text(peak + 0.25, 1.13, f"overshoot {100 * (top - 1):.1f} % of the step", fontsize=9, color=INK["primary"], va="bottom")
    ax.set(xlim=(-10, 10), ylim=(-0.2, 1.3), xlabel="position (mm)", ylabel="intensity",
           title=f"k-space kept to {k_max:.2f} cycles/mm: voxel = {dx:.2f} mm")
    ax.legend(loc="upper left", fontsize=8, frameon=False)

animate(fig, frame, range(len(voxel_sizes) + N_HOLD), fps=10, width=720, dpi=72,
        alt="a step edge reconstructed from k-space truncated at a growing maximum frequency: the reconstructed curve overshoots the step by about 9 percent right next to the edge and ripples away from it; as more frequencies are kept the ripples crowd closer to the edge but the overshoot keeps the same height, and dots at the voxel centers, one voxel apart, land alternately on the peaks and troughs of the ripples")
```

```{code-cell} python
:tags: [hide-input]
for dx in (5.0, 2.0, 0.5):
    k_max = 1 / (2 * dx)
    xv = np.linspace(1e-6, 4 * dx, 40001)
    curve = ringing_curve(xv, k_max)
    i_peak = np.argmax(curve)
    i_dip = i_peak + np.argmin(curve[i_peak:])
    print(f"voxel {dx:.1f} mm: overshoot {100 * (curve[i_peak] - 1):.1f} % of the step, "
          f"{xv[i_peak] / dx:.2f} voxel from the edge; the undershoot that follows is {(xv[i_dip] - xv[i_peak]) / dx:.2f} voxel further out")
```

Two things stay the same in every frame and in every row of the printout. The overshoot is
about 9 % of the step, no matter how many frequencies are kept; adding frequencies only
squeezes the ripples closer to the edge. And the ripples are tied to the voxel: an overshoot
and an undershoot alternate every voxel, so in the image the artifact looks like fine
stripes, one voxel wide, parallel to every sharp boundary.

In the brain the sharpest boundary is between CSF and tissue. In diffusion data this
boundary has a property that makes the ringing worse than in an anatomical image: its
contrast changes sign with b-value. The figure follows one CSF/tissue edge (the caudate
nucleus beside a lateral ventricle) through a b=0 volume, a b=1000 volume, and the apparent
diffusivity computed from the two.

```{code-cell} python
:tags: [hide-input]
TE = presets.TE_HBCD_MS
D = {"CSF": presets.ADULT_DIFFUSIVITY["CSF"], "tissue": presets.ADULT_DIFFUSIVITY["GM"]}  # mm^2/s
S0 = {"CSF": phantoms.PROTON_DENSITY["CSF"] * np.exp(-TE / presets.T2_MS["adult"]["CSF"]),
      "tissue": phantoms.PROTON_DENSITY["GM"] * np.exp(-TE / presets.T2_MS["adult"]["GM"])}
B = 1000.0

def edge_profile(xv, b, shift=0.0):
    """CSF for x < 0, tissue for x > 0 at b-value b, ringing at the acquisition's resolution (x in voxels).
    `shift` moves the voxel centers that many voxels relative to the edge."""
    s_csf, s_tis = (S0[t] * np.exp(-b * D[t]) for t in ("CSF", "tissue"))
    return s_csf + (s_tis - s_csf) * ringing_curve(xv, 0.5, edge=-shift)

xv = np.linspace(-6.5, 6.5, 2601)
cv = np.arange(-6, 7)  # voxel centers; the edge falls on the center of voxel 0
fig, axes = plt.subplots(3, 1, figsize=(8, 7.5), sharex=True)
for ax, b, label in [(axes[0], 0.0, "b = 0: CSF bright"), (axes[1], B, "b = 1000: CSF dark")]:
    true = np.where(xv < 0, S0["CSF"] * np.exp(-b * D["CSF"]), S0["tissue"] * np.exp(-b * D["tissue"]))
    ax.plot(xv, true, color="0.3", lw=1, ls="--", label="object")
    ax.plot(xv, edge_profile(xv, b), color=PALETTE[1], lw=1.5, label="acquired")
    ax.plot(cv, edge_profile(cv, b), "o", color=PALETTE[0], ms=5, label="voxel centers")
    ax.set(ylabel="signal", title=label)
    ax.legend(loc="lower right" if b else "upper right", fontsize=8, frameon=False)
axes[0].set_ylim(0, 1.1)
axes[1].set_ylim(0, 0.16)
adc_curve = np.log(edge_profile(xv, 0) / edge_profile(xv, B)) / B * 1e3
adc_vox = np.log(edge_profile(cv, 0) / edge_profile(cv, B)) / B * 1e3
axes[2].plot(xv, np.where(xv < 0, D["CSF"], D["tissue"]) * 1e3, color="0.3", lw=1, ls="--", label="true diffusivity")
axes[2].plot(xv, adc_curve, color=PALETTE[1], lw=1.5, label="apparent: ln(S(b=0) / S(b=1000)) / b")
axes[2].plot(cv, adc_vox, "o", color=PALETTE[0], ms=5)
for n in range(1, 6):
    axes[2].annotate("too low" if adc_vox[6 + n] < D["tissue"] * 1e3 else "too high", (n, adc_vox[6 + n]),
                     xytext=(0, -14 if adc_vox[6 + n] < D["tissue"] * 1e3 else 7), textcoords="offset points",
                     ha="center", fontsize=7.5, color=INK["secondary"])
axes[2].set(ylabel="diffusivity (x10⁻³ mm²/s)", xlabel="position (voxels; CSF left, tissue right)", ylim=(0, 3.6),
            title="apparent diffusivity from the two volumes")
axes[2].legend(loc="upper right", fontsize=8, frameon=False)
for ax in axes:
    ax.axvspan(-6.5, 0, color=TISSUE_COLORS["CSF"], alpha=0.07, lw=0)
fig.tight_layout()

print(f"true tissue diffusivity {D['tissue'] * 1e3:.2f} x10^-3 mm^2/s; apparent, voxels 1-5 into the tissue: "
      + ", ".join(f"{v:.2f}" for v in adc_vox[7:12]))
print(f"b=0 ripple in the first tissue voxel: {100 * (edge_profile(1.0, 0) / S0['tissue'] - 1):+.0f} % of the tissue signal; "
      f"b=1000: {100 * (edge_profile(1.0, B) / (S0['tissue'] * np.exp(-B * D['tissue'])) - 1):+.0f} %")
```

At b = 0 (top) CSF is much brighter than tissue, so the first tissue voxel sits in an
undershoot; at b = 1000 (middle) CSF has decayed below the tissue and the same voxel sits in
an overshoot. The ripples flip sign between the volumes, and the diffusivity computed from
their ratio (bottom) alternates: too low in the first tissue voxel, too high in the next,
and so on outward. The error is large because the ripple is 9 % of the CSF/tissue step, and
at b = 0 that step is several times the tissue signal itself. In real data this is a stripe
pattern of over- and underestimated diffusivity along every CSF boundary, and it propagates
into FA and into every model fitted downstream. In your own MD or FA maps, look for
alternating light and dark bands, each one voxel wide, hugging the ventricles.

## The artifact-free reference

The toy version follows the design of the `gibbs` dataset. The synthetic series is built on
the packaged 1 mm slice, and the acquisition keeps only the k-space band of a 2 mm matrix. The reference
is the same series block-averaged to 2 mm, which is what a ring-free 2 mm image of the object
would be.

```{code-cell} python
:tags: [hide-input]
fine_t = phantoms.brain_slice(1.0)
bvals, bvecs = schemes.multi_shell({1000: 6, 2000: 6}, n_b0=2)
fine = synth.synthetic_dwi(fine_t, bvals, bvecs)  # 256 x 256 x 14 at 1 mm

def block2(a):
    return a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2, *a.shape[2:]).mean(axis=(1, 3))

def acquire_2mm(series, window=None):
    k = kspace.fft2c(np.moveaxis(series, -1, 0))[:, 64:192, 64:192]  # keep the central 128 x 128 band
    if window is not None:
        k = k * window
    return np.moveaxis(np.abs(kspace.ifft2c(k)), 0, -1) * (128 / 256)

reference = block2(fine)
acquired = acquire_2mm(fine)
tissue2 = {k: block2(fine_t[k]) for k in ("wm", "gm", "csf")}
mask = (tissue2["wm"] + tissue2["gm"] + tissue2["csf"]) > 0.5
csf = tissue2["csf"] > 0.5
rim = ndimage.binary_dilation(csf, iterations=2) & ~csf & mask  # the two voxels of tissue next to CSF
print(f"series {acquired.shape}; rim voxels next to CSF: {rim.sum()}")
```

## See it: the ringing and its sign change

```{code-cell} python
:tags: [hide-input]
row = 58  # through the frontal horns of the lateral ventricles
v0, v1 = 0, 2  # a b=0 and a b=1000 volume
fig = plt.figure(figsize=(10, 8.6))
gs = fig.add_gridspec(3, 3, width_ratios=[1, 1, 1.6], height_ratios=[1, 1, 0.8])
axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)])
ax_res = fig.add_subplot(gs[2, :])
for r, (v, label, vmax, color) in enumerate([(v0, "b = 0", 0.5, PALETTE[1]), (v1, "b = 1000", 0.2, PALETTE[0])]):
    show_image(axes[r, 0], reference[..., v], f"{label}, reference", vmin=0, vmax=vmax)
    show_image(axes[r, 1], acquired[..., v], f"{label}, acquired at 2 mm", vmin=0, vmax=vmax)
    axes[r, 1].axhline(row, color=PALETTE[1], lw=1)
    axes[r, 2].plot(reference[row, 40:90, v], color="0.3", lw=1, ls="--", label="reference")
    axes[r, 2].plot(acquired[row, 40:90, v], color=PALETTE[1], lw=1.5, label="acquired")
    axes[r, 2].set(title=f"{label}: profile through the ventricles", xlabel="column (voxels from 40)")
    axes[r, 2].legend(fontsize=8)
    ref_line = reference[row, 40:90, v]
    ax_res.plot((acquired[row, 40:90, v] - ref_line) / (ref_line.max() - ref_line.min()), color=color, lw=1.5, marker="o", ms=3, label=label)
ax_res.axhline(0, color="0.6", lw=0.8)
ax_res.set(title="acquired minus reference along the same row, as a fraction of each profile's range",
           xlabel="column (voxels from 40)", ylabel="residual")
ax_res.legend(fontsize=8)
fig.tight_layout()
```

At b = 0 the ripples next to the ventricle dip below the tissue value on the tissue side;
at b = 1000 the same voxels rise above it. The bottom panel puts the two residuals on one
axis: at each ventricle wall (columns 0-5, 23-29, and 43-47) the two residuals swing in
opposite directions, the sign flip from the 1-D figure above seen in the image. The fitted diffusivity of
those voxels reads the ratio of the two, and is wrong in opposite directions in alternating
voxels.

## Correction step by step: sub-voxel shifts

The unringing method of {cite:t}`kellner2016` starts from the continuous ringing curve of
the first animation. That curve is fixed by the acquisition: the k-space samples determine it
completely, everywhere between the voxel centers as well as at them. What the image stores
is the curve sampled at the voxel centers, and how much ringing the samples show depends on
where those centers fall relative to the edge. The animation takes the b=0 edge from above
and slides the sampling dots along the fixed curve by fractions of a voxel. For each shift it
reports the *total variation* of the voxels near the edge: the sum of the jumps between
neighboring voxels, minus the one jump from CSF to tissue that any image of the edge must
make. What is left measures how much neighboring voxels zig-zag.

```{code-cell} python
:tags: [hide-input]
xv = np.linspace(-7.5, 7.5, 3001)
curve0 = edge_profile(xv, 0)
true0 = np.where(xv < 0, S0["CSF"], S0["tissue"])

def samples(shift):
    c = np.arange(-7, 7) + shift
    return c, edge_profile(c, 0)  # edge_profile's shift is 0 here: the dots move, the curve stays

def zigzag(shift):
    """Total variation of the 14 voxels around the edge, minus the one jump any image of the step must make."""
    c, s = samples(shift)
    return np.abs(np.diff(s)).sum() - (S0["CSF"] - S0["tissue"])

shifts_all = np.linspace(0, 1, 201)
tv_all = np.array([zigzag(s) for s in shifts_all])
best = shifts_all[np.argmin(tv_all)]
sweep = list(np.linspace(0, 1, 51)) + [best] * 16

fig, (ax, ax_tv) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [2.2, 1]})
fig.subplots_adjust(left=0.07, right=0.98, bottom=0.15, top=0.9, wspace=0.25)

def frame(i):
    s = sweep[i]
    ax.clear(); ax_tv.clear()
    ax.plot(xv, true0, color="0.3", lw=1, ls="--", label="object")
    ax.plot(xv, curve0, color=PALETTE[1], lw=1.5, label="ringing curve (fixed by the acquisition)")
    c, v = samples(s)
    ax.plot(c, v, "o-", color=PALETTE[0], ms=5, lw=0.8, label="voxels sampled with this shift")
    ax.set(xlim=(-7.5, 7.5), ylim=(0.1, 1.1), xlabel="position (voxels; CSF left, tissue right)", ylabel="signal at b = 0",
           title=f"sampling grid shifted by {s % 1:.2f} voxel")
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    done = shifts_all <= s + 1e-9
    ax_tv.plot(shifts_all[done], tv_all[done], color=PALETTE[0], lw=1.5)
    ax_tv.plot([s], [zigzag(s)], "o", color=PALETTE[0], ms=6)
    ax_tv.set(xlim=(0, 1), ylim=(0, tv_all.max() * 1.15), xlabel="shift (voxels)", ylabel="total variation",
              title=f"zig-zag = {zigzag(s):.3f}")
    if i >= 51:
        ax_tv.plot(shifts_all, tv_all, color=PALETTE[0], lw=1.5)
        ax_tv.annotate("least zig-zag", (best, zigzag(best)), xytext=(best, tv_all.max() * 0.55), ha="center",
                       fontsize=9, arrowprops=dict(arrowstyle="->", color=INK["primary"], lw=1))

animate(fig, frame, range(len(sweep)), fps=10, width=760, dpi=70,
        alt="left: the fixed ringing curve of a CSF-tissue edge at b = 0 with voxel samples as dots; the dots slide along the curve as the sampling grid shifts by fractions of a voxel. With no shift the dots land on the peaks and troughs of the ripples and zig-zag strongly; with a half-voxel shift they land near the points where the curve crosses the true values and lie almost flat. Right: the total variation of the samples traced against the shift, largest at zero shift and smallest at half a voxel, where the animation stops")
```

```{code-cell} python
:tags: [hide-input]
def worst_ripple(shift):
    c, v = samples(shift)
    far = np.abs(c) >= 1
    return np.max(np.abs(v[far] - np.where(c[far] < 0, S0["CSF"], S0["tissue"]))) / (S0["CSF"] - S0["tissue"])

print(f"total variation with no shift {zigzag(0):.3f}; smallest {zigzag(best):.3f}, at a shift of {best:.2f} voxel")
print(f"largest ripple in a voxel off the edge: {100 * worst_ripple(0):.1f} % of the step with no shift, "
      f"{100 * worst_ripple(best):.1f} % at the best shift")
```

With no shift the voxel centers land on the peaks and troughs of the ringing curve, and the
samples zig-zag as much as they can. Half a voxel over, the same curve is sampled close to
where it crosses the true values, and the zig-zag almost disappears. The curve did not change;
only where it was sampled. That is the whole method in plain words: because the ringing is
fixed by the acquisition, re-sampling each voxel at the shift where its neighbors zig-zag
least removes most of the ringing without blurring the edge. In practice the image is
recomputed at a handful of sub-voxel shifts (a shift in the image is a phase ramp, a
multiplication by a linearly changing phase, in k-space), the total variation is measured in
a small window on either side of each voxel, and each voxel takes its value from the shift
that minimizes it. It is applied per 2-D slice, volume by volume. Because it is a local
operation on the magnitude image, it belongs immediately after denoising and before anything
that resamples the data.

The next cell applies dipy's implementation to the synthetic series and compares it with the
acquisition-side alternative, *apodization*: multiplying k-space by a window that falls
smoothly to zero at its edge instead of stopping abruptly, so the high frequencies are faded
out rather than cut off. The window used here is the *Hann window*, one period of a raised
cosine that is 1 at the center of k-space and 0 at the edge. Each version is scored on the
tissue rim next to CSF two ways: the average absolute error against the ring-free reference,
and a *ripple amplitude*, the average size of the second difference $s_{i-1} - 2 s_i + s_{i+1}$
between each voxel and its two neighbors along one image axis. The second difference measures
how far a voxel sticks out from the straight line through its neighbors: it is near zero for a
smooth profile and large for a one-voxel zig-zag, which makes it a direct measure of the
ringing.

```{code-cell} python
:tags: [hide-input]
unrung = gibbs_removal(acquired.copy(), slice_axis=2, n_points=3, inplace=False)
hann = np.outer(np.hanning(128), np.hanning(128))
apodized = acquire_2mm(fine, window=hann)

def rim_error(series):
    return np.abs(series - reference)[rim].mean()

def ripple(series):
    """Voxel-to-voxel oscillation along the phase-encode axis in the rim: the ringing itself."""
    second_diff = series[:-2] - 2 * series[1:-1] + series[2:]
    return np.abs(second_diff)[rim[1:-1]].mean()

print("in the rim next to CSF, all volumes:   error vs reference   ripple amplitude")
for label, s in [("reference", reference), ("acquired", acquired), ("unringed", unrung), ("Hann apodized", apodized)]:
    print(f"  {label:>14}:                     {rim_error(s):.4f}               {ripple(s):.4f}")

fig, axes = plt.subplots(1, 4, figsize=(11, 3))
show_image(axes[0], reference[..., v0], "reference", vmin=0, vmax=0.5)
show_image(axes[1], acquired[..., v0], "acquired", vmin=0, vmax=0.5)
show_image(axes[2], unrung[..., v0], "unringed", vmin=0, vmax=0.5)
show_image(axes[3], apodized[..., v0], "Hann apodized", vmin=0, vmax=0.5)
fig.tight_layout()
```

Read the table column by column. The ripple amplitude is not zero even in the reference
(0.0419), because real anatomy bends too; the ringing is what the acquired series adds on top
of that (0.0524). Unringing brings it down to 0.0363, slightly below the reference, and
apodization to 0.0166, far below it, because the Hann window smooths away real detail along
with the ripples. The error column tells the other half: unringing lowers the error against
the reference from 0.0161 to 0.0145, while apodization raises it to 0.0182, because the blur it
introduces is an error too. The images show the same thing: the unringed slice keeps the
sharp ventricle walls, the apodized one is visibly softer.

Apodization removes the ripples because the truncation is no longer abrupt, and it costs
resolution: the edges are blurred. Unringing keeps the resolution.
Most scanners apply a mild filter by default; check the reconstruction settings, because
unringing data that were already apodized does nothing useful.

## Residual error versus truth

```{code-cell} python
:tags: [hide-input]
ref_fit = synth.dti_maps(reference, bvals, bvecs, mask=mask)
rows = []
for label, s in [("acquired", acquired), ("unringed", unrung), ("Hann apodized", apodized)]:
    m = synth.dti_maps(s, bvals, bvecs, mask=mask)
    rows.append((label, m))
    md_err = (m["md"] - ref_fit["md"])[rim] * 1e3
    fa_err = (m["fa"] - ref_fit["fa"])[rim]
    print(f"{label:>14}: rim MD error {md_err.mean():+.3f} ± {md_err.std():.3f} (x10^-3 mm^2/s)   rim FA error {fa_err.mean():+.3f} ± {fa_err.std():.3f}")

fig, axes = plt.subplots(1, 4, figsize=(11, 3))
show_image(axes[0], ref_fit["md"] * 1e3, "MD, reference", kind="scalar", vmin=0.5, vmax=3.0)
for ax, (label, m) in zip(axes[1:], rows):
    show_image(ax, (m["md"] - ref_fit["md"]) * 1e3 * mask, f"MD error, {label}", kind="diff", vmin=-0.5, vmax=0.5)
fig.tight_layout()
```

The error map of the acquired data shows the striped over- and underestimation along the
ventricles and the cortex. Unringing removes the stripes, which the ripple measure above
confirms; the error against the block-averaged reference falls less, because part of that
error is not ringing at all but the difference between a band-limited image and a
box-averaged one, which no unringing method addresses. What remains of the ringing sits at
corners and where two boundaries are within a few voxels of each other. Apodization removes
the stripes too, but replaces them with a blur that biases every boundary voxel toward its
neighbor, which the MD error shows as a systematic offset rather than a spread.

## Measure it: one slice, simulated live

On the simulated acquisition the ringing is intrinsic: the object is simulated on a grid
twice as fine as the acquisition matrix and only the acquired band of its k-space is kept, so
every tissue boundary rings the way a real one does. Simulating the same slice with the
object placed directly on the acquisition matrix (no oversampling) gives the ringing-free
reference, and a Hann window at reconstruction gives the apodized alternative.

```{code-cell} python
:tags: [hide-input]
import trxscan as ts
from dwibook import phantom as ph

g = ph.gtab(1)  # the b = 0 volume is where CSF is brightest and the ringing largest
clean = ts.Artifacts(seed=1)
ringing = ph.run(g, ph.PROTO, clean)
noring = ph.run(g, ph.PROTO.replace(oversample=1), clean)
hann = ph.run(g, ph.PROTO, clean.replace(window="hann"))
a, b, c = (ph.axial(s.magnitude, 0) for s in (ringing, noring, hann))
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
show_image(axes[0], a, "acquired (oversampled object)")
show_image(axes[1], b, "object on the acquisition grid")
show_image(axes[2], (a - b), "difference: the ringing", kind="diff", vmin=-0.1 * a.max(), vmax=0.1 * a.max())
show_image(axes[3], c, "Hann-apodized reconstruction")
fig.tight_layout()
```

```{code-cell} python
:tags: [hide-input]
csf = b > 0.7 * np.percentile(b[b > 0], 99)  # the brightest tissue at b = 0 is CSF
rim = ndimage.binary_dilation(csf, iterations=2) & ~csf & (b > 0.05 * b.max())  # the tissue ring around it
for label, img in [("acquired", a), ("Hann apodized", c)]:
    rel = (img - b)[rim] / b[rim].mean()
    print(f"{label:>14}: deviation from the ringing-free image in the {int(rim.sum())} voxels around CSF: "
          f"{rel.mean():+.3f} ± {rel.std():.3f} of their mean intensity")
```

The difference image is the ringing alone: stripes parallel to every CSF boundary, largest
at the ventricles. Apodization removes the stripes and replaces them with a blur, exactly as
on the synthetic series above; the numbers are smaller than in the toy case because the
simulated brain's 2.5 mm voxels already average over the sharpest edges.

## What acquisition choices reduce it

- **Ringing scales with voxel size relative to the anatomy**: the ripple is one voxel wide,
  so larger voxels ring over a larger distance. Higher resolution helps.
- **Partial Fourier changes the ringing** along the phase-encode axis, because the
  reconstruction fills in the skipped part of k-space from the acquired part
  (see "Partial Fourier reconstruction" in [Chapter 3](../01-mri-physics/03-reconstruction.md));
  the unringing method has a partial-Fourier-aware variant.
- **Do not apodize at the scanner** if the data will be unrung; do check whether the
  scanner already did.
- **Order matters:** denoise, then unring, then everything that resamples.

## Further reading

Sub-voxel-shift unringing {cite:p}`kellner2016` and its implementation in dipy and MRtrix
{cite:p}`garyfallidis2014,tournier2019`.
