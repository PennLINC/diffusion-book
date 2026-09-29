---
title: "22. Frontiers"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** single-voxel signals generated in the page ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
:::

## Learning goals

After this chapter you can:

- describe what b-tensor encoding, diffusion-relaxation correlation, high-gradient
  systems, non-EPI readouts, and learned reconstruction each add to the acquisition or
  analysis chain
- explain, with a simulation, why b-tensor encoding separates microscopic anisotropy
  from orientation dispersion
- state the role of simulation in validating the methods of this book

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Rectangle

from dwibook import kspace, phantoms, presets, schemes, signal
from dwibook.plotting import INK, PALETTE, set_style, show_image

set_style()
```

## b-tensor encoding

Every acquisition in this book applies its diffusion weighting along one direction per
measurement. The pulse pair of [Chapter 5](../02-diffusion-encoding/05-diffusion-encoding.md) can be replaced by a gradient waveform that
changes direction during the encoding, so that a single measurement weights diffusion
along several axes at once. The weighting is then described by a tensor rather than a
vector: linear encoding (one axis) is the conventional case, planar encoding weights two
axes, and spherical encoding weights all three equally {cite:p}`westin2016`.

What this buys is a measure of *microscopic anisotropy*: how elongated the individual
water compartments are (each axon or cell on its own), whether or not those compartments
are lined up with each other. The tensor FA of [Chapter 15](../04-modeling/15-signal-representations.md) cannot provide it, because FA
describes the voxel as a whole. The cartoon shows the three encoding shapes and two voxels
that FA cannot tell apart:

```{code-cell} python
:tags: [hide-input]
fig, axes = plt.subplots(1, 5, figsize=(12.5, 3.4), gridspec_kw={"width_ratios": [1, 1, 1, 1.25, 1.25]})
for ax in axes:
    ax.set(aspect="equal", xlim=(-1.25, 1.25), ylim=(-1.25, 1.25), xticks=[], yticks=[])
    ax.grid(False)
    for side in ax.spines.values():
        side.set_visible(False)
enc_color = PALETTE[0]
axes[0].add_patch(FancyArrowPatch((-1, -0.55), (1, 0.55), arrowstyle="<->", mutation_scale=14, color=enc_color, lw=2))
axes[0].set_title("linear: one axis\n(this book so far)", fontsize=9)
axes[1].add_patch(Ellipse((0, 0), 2.0, 0.8, angle=20, facecolor=enc_color, alpha=0.35, edgecolor=enc_color, lw=1.5))
for ang in np.deg2rad([20, 110]):
    axes[1].add_patch(FancyArrowPatch((-np.cos(ang) * (1.0 if ang < 1 else 0.4), -np.sin(ang) * (1.0 if ang < 1 else 0.4)),
                                      (np.cos(ang) * (1.0 if ang < 1 else 0.4), np.sin(ang) * (1.0 if ang < 1 else 0.4)),
                                      arrowstyle="<->", mutation_scale=10, color=enc_color, lw=1.2))
axes[1].set_title("planar: two axes", fontsize=9)
axes[2].add_patch(Circle((0, 0), 0.95, facecolor=enc_color, alpha=0.35, edgecolor=enc_color, lw=1.5))
axes[2].add_patch(Ellipse((0, 0), 1.9, 0.5, fill=False, edgecolor=enc_color, lw=0.8, ls="--"))
axes[2].set_title("spherical: all three axes\nequally", fontsize=9)
rng = np.random.default_rng(3)
centers = [(x, y) for x in np.linspace(-0.8, 0.8, 5) for y in np.linspace(-0.8, 0.8, 5)]
for ax, elongated in ((axes[3], True), (axes[4], False)):
    ax.add_patch(Rectangle((-1.1, -1.1), 2.2, 2.2, fill=False, edgecolor=INK["secondary"], lw=1))
    for (x, y) in centers:
        x, y = x + rng.uniform(-0.05, 0.05), y + rng.uniform(-0.05, 0.05)
        if elongated:
            ax.add_patch(Ellipse((x, y), 0.36, 0.1, angle=rng.uniform(0, 180), color=PALETTE[1]))
        else:
            ax.add_patch(Circle((x, y), 0.1, color=PALETTE[2]))
axes[3].set_title("elongated cells, random directions\nFA ≈ 0, microscopic anisotropy high", fontsize=9)
axes[4].set_title("round cells\nFA ≈ 0, no microscopic anisotropy", fontsize=9)
fig.text(0.3, 0.02, "encoding shapes (b-tensors)", ha="center", fontsize=9, color=INK["secondary"])
fig.text(0.78, 0.02, "two voxels", ha="center", fontsize=9, color=INK["secondary"])
fig.subplots_adjust(left=0.01, right=0.99, top=0.8, bottom=0.08, wspace=0.12)
```

In both voxels on the right, the diffusion averaged over the whole voxel is the same in
every direction, so a tensor fit gives an FA near zero for both. Only the voxel of
elongated cells has microscopic anisotropy. Linear encoding along any one direction sees,
in that voxel, a mixture of cells that are fast and slow along it; spherical encoding sees
every cell at once in all directions, and so sees only each cell's average diffusivity.
The difference between the two is what b-tensor encoding measures.

Spherical encoding has a property that no linear acquisition has: its signal depends only
on the trace of each compartment's diffusion tensor, not on its shape or orientation.
Combined with linear encoding at the same b-value, it separates two things that
conventional data confound. A voxel of coherent fibers and a voxel of dispersed or crossing
fibers with the same microscopic anisotropy give the same spherical-encoding signal;
their linear-encoding signals, averaged over directions, differ from the spherical one by
an amount that depends only on the microscopic anisotropy {cite:p}`lasic2014`. The figure
below tests this on three white matter voxels built from the same stick-plus-tensor
tissue, arranged in three ways.

```{code-cell} python
:tags: [hide-input]
b = np.linspace(0, 3000, 100)
dirs = schemes.electrostatic_directions(60)
d_par, d_perp, _ = presets.ADULT_DIFFUSIVITY["WM_extra"]
d_intra = presets.ADULT_DIFFUSIVITY["WM_intra"]
f = presets.ADULT_FRACTIONS["WM_intra"]
# a WM voxel = the simulated brain's stick + tensor. Spherical encoding sees each compartment's
# mean diffusivity (trace / 3), whatever its shape or orientation, so the signal is a sum over compartments
spherical = f * np.exp(-b * d_intra / 3) + (1 - f) * np.exp(-b * (d_par + 2 * d_perp) / 3)


def voxel_signal(fibers, bval):
    """linear-encoding signal of the voxel along each of the 60 directions, equal fiber fractions"""
    return np.mean([signal.white_matter(bval, dirs @ fib) for fib in fibers], axis=0)


def powder_linear(fibers):
    """direction-averaged linear-encoding signal of a voxel with the given fiber orientations, equal fractions"""
    out = np.zeros_like(b)
    for fib in fibers:
        cos = dirs @ fib
        out += np.mean(signal.white_matter(b[:, None], cos[None, :]), axis=1) / len(fibers)
    return out


arrangements = {
    "coherent": [np.array([0, 0, 1.0])],
    "90° crossing": [np.array([0, 0, 1.0]), np.array([0, 1.0, 0])],
    "fully dispersed": list(schemes.electrostatic_directions(200)),  # fibers spread evenly over the sphere
}
coherent, crossing, dispersed = (powder_linear(fibers) for fibers in arrangements.values())


def tensor_fa(fibers, bval=1000.0):
    """FA of a log-linear tensor fit to the voxel's linear-encoding signal on one shell"""
    s = voxel_signal(fibers, bval)
    gx, gy, gz = dirs.T
    design = np.column_stack([gx**2, gy**2, gz**2, 2 * gx * gy, 2 * gx * gz, 2 * gy * gz])
    dxx, dyy, dzz, dxy, dxz, dyz = np.linalg.lstsq(design, -np.log(s) / bval, rcond=None)[0]
    ev = np.linalg.eigvalsh([[dxx, dxy, dxz], [dxy, dyy, dyz], [dxz, dyz, dzz]])
    return np.sqrt(1.5 * np.sum((ev - ev.mean()) ** 2) / np.sum(ev**2))


styles = {"coherent": (PALETTE[0], "-", "o"), "90° crossing": (PALETTE[1], "--", "s"), "fully dispersed": (PALETTE[2], ":", "^")}
B_SHOW = 2000.0
angle = np.rad2deg(np.arccos(np.abs(dirs[:, 2])))  # angle between each gradient direction and the z axis
fig, (ax0, ax) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 1.25]})
for name, fibers in arrangements.items():
    color, _, mk = styles[name]
    s = voxel_signal(fibers, B_SHOW)
    ax0.plot(angle, s, mk, color=color, ms=4, alpha=0.8, label=f"{name}: mean {s.mean():.3f}")
ax0.set(xlabel="angle between gradient and z axis (°)", ylabel="signal / S₀", title=f"each of the 60 directions, linear encoding, b = {B_SHOW:.0f}")
ax0.legend(fontsize=8)
ax.semilogy(b, spherical, color="0.3", lw=2, label="spherical encoding: any fiber arrangement")
for name, curve in zip(arrangements, (coherent, crossing, dispersed)):
    color, ls, _ = styles[name]
    ax.semilogy(b, curve, color=color, ls=ls, label=f"linear, direction-averaged: {name}")
ax.set(xlabel="b (s/mm²)", ylabel="signal / S₀", ylim=(0.05, 1.05), title="the same voxels, averaged over directions")
ax.legend(fontsize=8)
fig.tight_layout()
print("tensor FA at b = 1000: " + ", ".join(f"{name} {tensor_fa(fibers):.2f}" for name, fibers in arrangements.items()))
i3 = np.argmin(abs(b - 3000))
print(f"at b = 3000: direction-averaged linear {coherent[i3]:.3f} (all three arrangements), spherical {spherical[i3]:.3f}")
```

The left panel shows what a conventional acquisition sees. The coherent voxel's signal
depends strongly on the direction of the gradient, low along the fibers and high across
them; the crossing voxel's signal varies less, and the fully dispersed voxel gives nearly
the same signal in every direction. A tensor fit to these three voxels gives the very
different FA values printed above, falling from coherent to dispersed.

Yet the mean over directions, given in the legend, is the same for all three. The right
panel shows that mean at every b-value: the three direction-averaged linear curves
coincide, because averaging removes the arrangement, and all three lie above the spherical
curve by the same margin, because that margin is set by the microscopic anisotropy alone.
b-tensor encoding therefore reports the same microscopic anisotropy for all three voxels,
while FA reports three different values for the same tissue.

The simulated brain's truth maps include the b-tensor quantities (microscopic FA and the
isotropic and anisotropic kurtosis, [Appendix E](../appendices/e-truth-map-catalogue.md)); simulating the b-tensor acquisition itself
is a planned simulator extension.

## Diffusion relaxometry

Diffusion time ([Chapter 21](./21-multi-diffusion-time.md)) and echo time ([Chapter 20](./20-echo-time.md)) are further dimensions being added to the
encoding, and acquisitions that vary several at once, *multidimensional diffusion MRI*, are
the current frontier of the field. [Chapter 20](./20-echo-time.md) varied the echo time and added
echoes. The general form is a joint acquisition over b-value, direction, echo time,
inversion time, and diffusion time, from which compartments are separated by every
property at once. Such data support model-free analyses, *correlation spectra* of
diffusivity against T2 or T1, that need no assumption about the number of compartments.
The schematic shows the idea for the white matter voxel of [Chapter 20](./20-echo-time.md) (55 % intra-axonal,
45 % extra-axonal, with the T2 values used there), with a tenth of the signal from CSF
added so that a third, well-separated peak appears:

```{code-cell} python
:tags: [hide-input]
peaks = {  # (mean diffusivity in µm²/ms, T2 in ms, signal fraction); widths below are illustrative
    "intra-axonal": (d_intra / 3 * 1e3, 90.0, 0.9 * f, PALETTE[0]),  # Chapter 20's voxel (f = 0.55), 90 % of the signal
    "extra-axonal": ((d_par + 2 * d_perp) / 3 * 1e3, 60.0, 0.9 * (1 - f), PALETTE[1]),
    "CSF": (presets.ADULT_DIFFUSIVITY["CSF"] * 1e3, 2000.0, 0.10, PALETTE[2]),
}
logd = np.linspace(np.log10(0.2), np.log10(5), 200)
logt = np.linspace(np.log10(30), np.log10(4000), 200)
LD, LT = np.meshgrid(logd, logt)
W = 0.09  # decades
spec = sum(w * np.exp(-((LD - np.log10(dd)) ** 2 + (LT - np.log10(tt)) ** 2) / (2 * W**2)) for dd, tt, w, _ in peaks.values())
fig = plt.figure(figsize=(7.5, 5))
gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0.05, wspace=0.05)
ax = fig.add_subplot(gs[1, 0])
ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
ax.contourf(10**LD, 10**LT, spec, levels=12, cmap="Greys")
for name, (dd, tt, w, color) in peaks.items():
    ax.annotate(name, (dd, tt), xytext=(8, 8), textcoords="offset points", color=color, fontsize=9)
ax.set(xscale="log", yscale="log", xlabel="mean diffusivity (µm²/ms)", ylabel="T2 (ms)")
ax.set_xticks([0.2, 0.5, 1, 2, 5], labels=["0.2", "0.5", "1", "2", "5"])
ax.set_yticks([30, 100, 300, 1000, 3000], labels=["30", "100", "300", "1000", "3000"])
ax.minorticks_off()
ax.grid(False)
ax_top.plot(10**logd, spec.sum(axis=0), color=INK["primary"])
ax_top.set(yticks=[], title="schematic D-T2 correlation spectrum of one white matter voxel")
ax_top.tick_params(which="both", labelbottom=False)
ax_right.plot(spec.sum(axis=1), 10**logt, color=INK["primary"])
ax_right.set(xticks=[])
ax_right.tick_params(which="both", labelleft=False)
for name, (dd, tt, w, color) in peaks.items():
    ax_top.axvline(dd, color=color, lw=0.8, ls=":")
    ax_right.axhline(tt, color=color, lw=0.8, ls=":")
```

Each compartment is a peak on a map of diffusivity (horizontal) against T2 (vertical). The
curves along the top and right edges are what a diffusion-only or a relaxation-only
measurement sees: the same spectrum collapsed onto one axis. On the T2 edge the
intra-axonal and extra-axonal peaks run together into one hump, and on the diffusivity edge
they are only partly resolved, because the two compartments are close in both properties.
On the map they are two distinct peaks: one compartment is slower *and* longer-lived than
the other, so their separation along the diagonal combines the differences on both axes.
The peak positions are the book's
toy values; the widths are illustrative. Such acquisitions cost scan time in proportion to
the number of dimensions sampled, and their design is an open problem.

## High-gradient systems

[Chapter 5](../02-diffusion-encoding/05-diffusion-encoding.md) gave the echo-time cost of a b-value and [Chapter 21](./21-multi-diffusion-time.md) the q needed for axon
diameters; both are set by gradient amplitude. Systems at 200–300 mT/m (the Connectom class)
and head-only gradient inserts beyond that reach b = 10 000 at echo times that whole-body
systems need for much lower b-values:

```{code-cell} python
:tags: [hide-input]
b_targets = [3000.0, 10000.0]
t2_wm = presets.T2_MS["adult"]["WM"]
fig, ax = plt.subplots(figsize=(6.5, 3.4))
width = 0.36
rows = []
for j, bt in enumerate(b_targets):
    tes = [signal.min_te(bt, g)["te"] for g in presets.GMAX_MT_PER_M.values()]
    x = np.arange(len(tes)) + (j - 0.5) * width
    bars = ax.bar(x, tes, width, color=PALETTE[j], label=f"b = {bt:.0f} s/mm²")
    for xi, te in zip(x, tes):
        ax.text(xi, te + 3, f"{te:.0f}", ha="center", fontsize=8)
    rows.append(tes)
ax.set(xticks=np.arange(len(presets.GMAX_MT_PER_M)), xticklabels=list(presets.GMAX_MT_PER_M), ylabel="minimum TE (ms)",
       title="shortest echo time that reaches each b-value")
ax.legend(fontsize=8)
fig.tight_layout()
for (name, g), te3, te10 in zip(presets.GMAX_MT_PER_M.items(), *rows):
    print(f"{name} ({g:.0f} mT/m): min TE {te3:.0f} ms at b = 3000 (white matter keeps {np.exp(-te3 / t2_wm):.2f} of its signal to T2), "
          f"{te10:.0f} ms at b = 10000 ({np.exp(-te10 / t2_wm):.2f})")
```

The 300 mT/m system reaches b = 10 000 at a shorter echo time than a clinical 40 mT/m system
needs for b = 3000, and so keeps more of the white matter signal (printed above, for the
simplified timing of [Chapter 5](../02-diffusion-encoding/05-diffusion-encoding.md)). Such systems also reach the q where small axons begin to
be distinguishable. Their costs: their nonlinearity is larger ([Chapter 13](../03-preprocessing/13-gradient-nonlinearity.md)); they reach the
limit set by *peripheral nerve stimulation* sooner (rapidly switched gradients induce
electric currents in the body strong enough to make nerves fire, felt as tingling or
twitching, which limits how fast the gradients may be switched); and they exist at a
handful of sites. The methods developed on them (diameter mapping, high-b compartment
models) set expectations that standard hardware cannot meet, which is a recurring source
of over-interpretation.

## Beyond single-shot EPI

[Chapter 2](../01-mri-physics/02-spatial-encoding-kspace.md) explained why diffusion uses single-shot EPI and what it costs. Splitting the
lines of k-space over several excitations (multi-shot EPI) shortens each readout, which
reduces distortion and blur and allows higher resolution, but the diffusion gradients give
each shot a different, unknown bulk phase from small movements of the head during the
encoding. In a single-shot acquisition that phase is common to every line and drops out of
the magnitude image. In a multi-shot acquisition the shots disagree, and the disagreement
appears as ghosts. The simulation below acquires the odd lines of k-space in one shot and
the even lines in another, with a 60° phase difference between them:

```{code-cell} python
:tags: [hide-input]
img = phantoms.brain_image()  # synthetic b=0 slice, 2 mm
ksp = kspace.fft2c(img)
shot = np.arange(ksp.shape[0]) % 2  # two interleaved shots
PHASE_DEG = 60.0
phase_error = np.exp(1j * np.deg2rad(PHASE_DEG))  # the second shot acquired with a 60° bulk phase
ksp_multishot = np.where(shot[:, None] == 1, ksp * phase_error, ksp)
recon = kspace.ifft2c(ksp_multishot)

fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
show_image(axes[0], kspace.ifft2c(ksp), "single shot")
show_image(axes[1], recon, f"two shots with a {PHASE_DEG:.0f}° phase difference")
fig.tight_layout()
print(f"brain kept at {abs(1 + phase_error) / 2:.2f} of its intensity; ghost, shifted by half the image, at {abs(1 - phase_error) / 2:.2f}")
```

Compare the two panels. In the right one the brain is slightly dimmer, and a second copy of
it, the ghost, appears shifted by half the image along the phase-encode axis (top to
bottom), so that it wraps around and overlaps the top and bottom of the real brain. With
two shots a phase difference φ leaves the brain at cos(φ/2) of its intensity and puts
sin(φ/2) into the ghost, the two numbers printed above: at 60°, a disagreement that small
head movements during the diffusion encoding readily produce, the ghost is half as bright
as the brain.

Multi-shot diffusion imaging therefore needs the per-shot phase. It comes either from a
*navigator*, a short extra readout after each shot that measures that shot's phase, or from
a reconstruction that estimates it from the data themselves. The simulation above used one
number per shot, but real shot phase varies across the image: a rigid movement during the
encoding gives a phase that ramps linearly across the image, and brain pulsation, which
moves different parts of the brain differently, gives a nonlinear pattern. That is why a
navigator is a low-resolution 2-D image rather than a single number, and why the correction
is applied voxel by voxel.
The main alternatives to single-shot EPI trade its robustness for resolution or reduced
distortion:

| Approach | What changes | Gain | Cost |
|---|---|---|---|
| Multi-shot EPI with navigators | k-space split over several shots, each followed by a navigator | shorter readout: less distortion and blur, higher resolution | longer scan; extra navigator time; relies on the navigator phase matching the shot |
| Self-navigated multi-shot EPI | per-shot phase estimated from the image data | as above, without navigator time | heavier reconstruction; needs enough coils to unfold each shot |
| Readout-segmented EPI (RESOLVE) | k-space split along the readout axis into segments, each read in one shot; a 2-D navigator after each | much shorter echo spacing: less distortion and blur; the most widely available multi-shot option on clinical scanners | scan time grows with the number of segments; segments reacquired when the navigator shows too much motion |
| Spiral readouts | k-space sampled along spirals from the center | short echo time, efficient sampling | off-resonance blurs instead of shifting; needs a good field map |
| Reduced field of view | only a small region is excited along the phase-encode axis | fewer lines, short readout; used in the spinal cord and optic nerve | covers only that region |
| Simultaneous multi-slice ([Chapter 7](../02-diffusion-encoding/07-acquisition-parameters.md)) | several slices excited and read together | shorter TR, more directions per minute | noise amplification when slices are unfolded; now standard |
| 3-D segmented readouts | a thick slab encoded in 3-D over several excitations | no slice-profile penalty, thin sections | the same shot-to-shot phase problem, in three dimensions |

Multi-shot and reduced-field-of-view methods are established for the spinal cord and optic
nerve and increasingly used for sub-millimeter brain imaging.

## Learned reconstruction and denoising

Neural networks trained on paired data now reconstruct images from undersampled k-space
(extending compressed sensing, [Chapter 6](../02-diffusion-encoding/06-qspace-sampling.md)), denoise diffusion series (extending MP-PCA,
[Chapter 8](../03-preprocessing/08-noise.md)), and predict full-quality microstructure maps from short protocols. The gains
are real and the caveats are the same as for every learned method: the output is only as
general as the training data, and errors are plausible-looking rather than noisy, which
makes them harder to detect than the artifacts of Part III. Simulation with a known ground
truth is the natural test, which is the last topic.

## Simulation as validation

Every chapter of this book measured a method against a known answer, and the reason that
was possible is that the data were simulated. Real data have no ground truth; validation
against histology is possible for a few quantities in a few samples, and validation against
other MRI methods only tests agreement.

A simulator that models the acquisition from the diffusion signal through k-space to the
reconstructed image, and that writes the analytic answer for the same simulated brain,
allows a method to be scored on the quantity it claims to measure, under the artifacts it
will meet, at the acquisition parameters of a specific protocol. The toy tier of this book
did this with synthetic tissue on a single slice or a small volume; the pipeline tier, when
the pipeline delivers it, will do it with a full tractogram, a realistic acquisition, and
twenty-seven truth maps.

The limits of the simulator are the limits of the validation, and this book has stated
them where they apply: Gaussian compartments, no exchange, no diameters, a single diffusion
time, one echo per readout. Each is a direction the simulator can be extended, and each
extension makes a further chapter of this book testable.

## What this implies for acquisition

- **b-tensor encoding** is available on some scanners as a research sequence; it adds
  microscopic anisotropy to what a protocol can measure at modest scan-time cost.
- **Multidimensional protocols** need design; sample the dimensions that the question
  needs and no others.
- **Do not transfer expectations across gradient systems**: a method validated at
  300 mT/m may not be measurable at 80.
- **Ask how a learned method was validated** before using its output as a measurement.

## Further reading

b-tensor encoding {cite:p}`westin2016,lasic2014`, the theory of what diffusion MRI can and
cannot resolve {cite:p}`novikov2019`, integrated diffusion-relaxometry {cite:p}`hutter2018`,
and the simulator this book is built on {cite:p}`neher2014`.
