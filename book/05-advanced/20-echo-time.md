---
title: "20. Echo time in diffusion MRI"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** single-voxel signals with the compartment T2 values of the presets, and a synthetic b=0 slice built from the packaged tissue maps, read out with several echoes ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
- **`te-sweep`** (pending): the HBCD multi-shell scheme at four echo times ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-te-sweep)).
- **`truth`** (pending): the 27 analytic ground-truth maps and the true fiber orientations ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-truth), [Appendix E](../appendices/e-truth-map-catalogue.md)).

Pipeline-tier datasets are simulated offline by TRXScan ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)) and are marked *pending* until their release; the figures that need them say so where they will appear.
:::

## Learning goals

After this chapter you can:

- state what the echo time changes in a diffusion measurement: the overall signal, the
  balance between tissues in a mixed voxel, and the balance between compartments within a
  tissue
- explain why a diffusion model fitted at one echo time reports signal fractions, not volume
  fractions
- tell a multi-TE acquisition (one echo time per excitation) from a multi-echo acquisition
  (several echo times per excitation), and say which relaxation time each measures
- fit a joint diffusion-relaxation model to multi-TE data and recover compartment T2 values
- combine the echoes of a multi-echo acquisition and estimate T2* in every voxel
- state what each approach costs and choose between them

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from scipy.optimize import least_squares
from scipy.special import erf

from dwibook import kspace, phantoms, presets, schemes, signal, synth
from dwibook.plotting import INK, PALETTE, TISSUE_COLORS, animate, set_style, show_image

set_style()
```

## What the echo time does

Every diffusion image is acquired at some echo time, and the echo time is not a free choice:
the diffusion gradients have to be played between the excitation and the readout, so the
shortest possible TE grows with the b-value and shrinks with the gradient strength
([Chapter 5](../02-diffusion-encoding/05-diffusion-encoding.md), [Chapter 7](../02-diffusion-encoding/07-acquisition-parameters.md)). Whatever TE a protocol ends up with, it changes the
measurement in three ways:

- **How much signal there is.** Every tissue loses signal to T2 decay ([Chapter 1](../01-mri-physics/01-spins-and-signal.md)), about
  14 % of adult white matter signal per 10 ms at 3 T, so a longer TE lowers the SNR at every
  b-value.
- **Which tissue dominates a mixed voxel.** CSF, with a T2 of seconds, loses almost nothing
  while tissue decays, so a voxel partly filled with CSF is weighted toward it more heavily
  the longer the TE ([Chapter 1](../01-mri-physics/01-spins-and-signal.md) works through an example).
- **Which compartment dominates within a tissue.** The same argument applies inside white
  matter, where the water inside and outside the axons relaxes at different rates. This
  effect is the least visible of the three and matters most for microstructure models; the
  next section shows it.

## Signal fractions depend on the echo time

The tissue compartments of [Chapter 17](../04-modeling/17-microstructure-models.md) differ not only in how water diffuses in them but in
how fast their signal decays with echo time. Intra-axonal water has a longer T2 than
extracellular water in white matter (values of roughly 80–90 ms versus 50–60 ms at 3 T have
been reported), and CSF has a T2 of seconds.

The animation below follows one white matter voxel, without diffusion weighting, as the
echo time grows. The voxel is 55 % intra-axonal and 45 % extra-axonal by volume. On the
left, each compartment's signal decays with its own T2; on the right, the bar marked
"volume" never changes, while the bar marked "signal" shows each compartment's share of
what the scanner receives at the current TE.

```{code-cell} python
:tags: [hide-input]
T2_INTRA, T2_EXTRA = 90.0, 60.0  # ms: literature values for the toy voxel (see the note below)
F_VOL = presets.ADULT_FRACTIONS["WM_intra"]


def compartment_signals(te):
    """b = 0 signal of each compartment at echo time te, proton density 1"""
    return F_VOL * np.exp(-te / T2_INTRA), (1 - F_VOL) * np.exp(-te / T2_EXTRA)


def intra_share(te):
    a, e = compartment_signals(te)
    return a / (a + e)


te_axis = np.linspace(0, 160, 321)
a_curve, e_curve = compartment_signals(te_axis)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
ax1.plot(te_axis, a_curve, color=PALETTE[0], label=f"intra-axonal (T2 {T2_INTRA:.0f} ms)")
ax1.plot(te_axis, e_curve, color=PALETTE[1], label=f"extra-axonal (T2 {T2_EXTRA:.0f} ms)")
ax1.axvspan(60, 130, color=INK["grid"], alpha=0.6, lw=0)
ax1.text(95, 0.57, "typical diffusion TE", ha="center", fontsize=8, color=INK["secondary"])
marker = ax1.axvline(0, color=INK["primary"], lw=1, ls="--")
dots = ax1.scatter([0, 0], [F_VOL, 1 - F_VOL], color=[PALETTE[0], PALETTE[1]], zorder=3)
ax1.set(xlabel="echo time TE (ms)", ylabel="signal at b = 0", xlim=(0, 160), ylim=(0, 0.62))
ax1.legend(loc="center right", fontsize=8)
ax2.bar(0, F_VOL, color=PALETTE[0], width=0.6)
ax2.bar(0, 1 - F_VOL, bottom=F_VOL, color=PALETTE[1], width=0.6)
ax2.text(0, F_VOL / 2, f"{F_VOL:.0%}", ha="center", va="center", color="white", fontsize=9)
sig_a = ax2.bar(1, F_VOL, color=PALETTE[0], width=0.6)[0]
sig_e = ax2.bar(1, 1 - F_VOL, bottom=F_VOL, color=PALETTE[1], width=0.6)[0]
sig_txt = ax2.text(1, F_VOL / 2, "", ha="center", va="center", color="white", fontsize=9)
ax2.set(xticks=[0, 1], xticklabels=["volume", "signal"], ylim=(0, 1), ylabel="share of the voxel")
ax2.grid(False)


def frame(te):
    marker.set_xdata([te, te])
    a, e = compartment_signals(te)
    dots.set_offsets([[te, a], [te, e]])
    s = a / (a + e)
    sig_a.set_height(s)
    sig_e.set_y(s)
    sig_e.set_height(1 - s)
    sig_txt.set_position((1, s / 2))
    sig_txt.set_text(f"{s:.0%}")
    ax1.set_title(f"TE = {te:.0f} ms")
    ax2.set_title("intra-axonal share (blue)")


te_frames = list(np.arange(0, 151, 5.0)) + [150.0] * 6
animate(fig, frame, te_frames, fps=6, width=720, dpi=70,
        alt="Left: two decaying curves, intra-axonal signal with T2 90 ms and extra-axonal signal with T2 60 ms, with a dashed marker sweeping from TE 0 to 150 ms. Right: a fixed bar of volume shares, 55 percent intra-axonal, next to a bar of signal shares whose intra-axonal part grows from 55 percent at TE 0 to 74 percent at TE 150 ms")
print("intra-axonal share of the b = 0 signal: " + ", ".join(f"TE {te:.0f} ms {intra_share(te):.2f}" for te in [0, 60, 90, 130])
      + f" (volume share {F_VOL:.2f})")
```

At TE = 0 the two bars agree. As the echo time grows, the extra-axonal signal dies faster
than the intra-axonal signal, so the intra-axonal share of the received signal grows, even
though nothing about the tissue has changed. Diffusion protocols cannot use a short TE,
because the diffusion gradients take tens of milliseconds to play ([Chapter 5](../02-diffusion-encoding/05-diffusion-encoding.md)); over the
60–130 ms they use, the intra-axonal share is 0.63–0.72 for a volume share of 0.55.

A diffusion acquisition at a single echo time therefore weights the compartments by their
T2 decay at that TE, and every fraction a diffusion model reports is a *signal fraction* at
that TE, not a *volume fraction*. The consequence is visible when the same model is fitted
to data acquired at different echo times: the apparent intra-axonal fraction rises with TE,
because the compartment with the longer T2 retains more of the signal {cite:p}`veraart2018`.

The same white matter voxel, now with diffusion weighting, is simulated at a range of echo
times and fitted with the spherical-mean model of [Chapter 17](../04-modeling/17-microstructure-models.md), which knows nothing about T2:

```{code-cell} python
:tags: [hide-input]
bvals, bvecs = schemes.multi_shell({1000: 30, 2000: 30, 3000: 30}, n_b0=4)
fiber = np.array([0.0, 0.0, 1.0])
cos = bvecs @ fiber
te_list = np.arange(50.0, 151.0, 10.0)
fitted = []
for te in te_list:
    s = signal.multi_te_white_matter(bvals, te, cos, T2_INTRA, T2_EXTRA)
    shells, means = synth.spherical_mean(s[None, :], bvals)
    fitted.append(synth.smt_fit(shells, means, f_grid=np.linspace(0, 1, 401))["f"][0])
fitted = np.array(fitted)

fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.plot(te_list, fitted, "o-", color=PALETTE[0], label="fitted intra-axonal fraction")
ax.plot(te_axis, intra_share(te_axis), ":", color=PALETTE[0], label="intra-axonal share of the b = 0 signal")
ax.axhline(F_VOL, color=INK["secondary"], ls="--", lw=1, label=f"true volume fraction ({F_VOL:.2f})")
ax.set(xlabel="echo time TE (ms)", ylabel="intra-axonal fraction", xlim=(40, 160), ylim=(0.5, 0.85),
       title="what a diffusion-only fit reports at each echo time")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
i60, i130 = int(np.argmin(abs(te_list - 60))), int(np.argmin(abs(te_list - 130)))
print(f"fitted intra-axonal fraction: TE 60 ms {fitted[i60]:.2f}, TE 130 ms {fitted[i130]:.2f} "
      f"(change {fitted[i130] - fitted[i60]:+.2f}); true volume fraction {F_VOL:.2f}")
offset = fitted - intra_share(te_list)
print(f"fitted fraction minus the b = 0 signal share (the model mismatch): TE 60 ms {offset[i60]:+.2f}, TE 130 ms {offset[i130]:+.2f}")
```

Follow the solid line from left to right: the fitted fraction climbs steadily with echo
time, by the amount printed above between 60 and 130 ms, parallel to the dotted signal
share. It sits above the dashed volume fraction at every TE, for two reasons that add. The
first is the relaxation effect of the animation: at any realistic TE the longer-T2
compartment has already gained signal share. The second is model mismatch: the
spherical-mean fit makes a *tortuosity* assumption, that the more densely axons are packed,
the more slowly water moves sideways between them, and so ties the extra-axonal radial
diffusivity to the fraction; the synthetic tissue does not follow that rule ([Chapter 17](../04-modeling/17-microstructure-models.md)).
The mismatch is nearly the same at every TE (second printed line: it shrinks slightly as the
extra-axonal signal it depends on fades), so the *trend* with TE is mostly the relaxation
effect.

Two studies with different echo times therefore report different "neurite densities" for
the same tissue, and a study that changes TE between scanners or protocol versions
introduces an apparent change in microstructure.

:::{note}
The simulated brain assigns one T2 per tissue (68 ms for the white matter fiber compartment,
76 ms for gray matter, 2000 ms for CSF), so in the simulator the TE dependence appears
between tissues but not between the two white matter compartments. The voxel in this
chapter gives the intra-axonal stick a T2 of 90 ms and the extra-axonal tensor a T2 of
60 ms to show the within-tissue effect; both values are literature figures, not the
simulated brain's.
:::

## What several echo times buy

Acquiring more than one echo time turns the echo time from a confound into a measurement.
With images at several echo times it becomes possible to:

- **separate compartments by relaxation as well as by diffusion**, and report their fractions
  extrapolated to TE = 0, where no relaxation has happened yet and signal fractions become
  volume fractions (as long as the compartments hold water at the same density);
- **measure a relaxation time** in every voxel, or in every compartment, alongside the
  diffusion measures, as a tissue property in its own right;
- **combine images to gain SNR**, when the extra echo times fit into the existing
  repetition time and so add no scan time (the last section says when they do).

There are two ways to collect the extra echo times, and they do not measure the same
relaxation time. A **multi-TE** acquisition gets one echo time from each excitation and
changes the echo time between volumes or between scans. A **multi-echo** acquisition reads
out several images after each excitation, one at each of several echo times. The timing
diagram below draws both. Time runs left to right. The black bars are the RF pulses (the 90°
excitation and the 180° refocusing pulse), the orange blocks are the diffusion gradient
pulses, the blue boxes are the EPI readouts that collect each image ([Chapter 2](../01-mri-physics/02-spatial-encoding-kspace.md)), and the
dotted lines mark the spin echo, the moment at which the 180° pulse has brought the spins
back into phase.

```{code-cell} python
:tags: [hide-input]
esp = presets.READOUT_HBCD_MS / 128  # echo spacing (ms) of the train drawn here: HBCD's effective total readout time spread over 128 lines
tr = kspace.epi_trajectory(128, 128, esp, partial_fourier=0.75, accel=2)
RO = tr.readout_ms
te1 = 70.0
echo_times = [te1 + k * (RO + 2.0) for k in range(3)]
t2star = {"WM": 45.0, "GM": 55.0, "CSF": 300.0}  # T2* at 3 T, approximate (cortical gray matter)
T180, T_POST = 6.0, 23.0  # ms: refocusing pulse window; end of 2nd gradient pulse to readout center
DELTA = (te1 - T180 - 2 * T_POST) / 2  # gradient pulse duration that fits the shortest echo time


def draw_scan(ax, y, te, readouts):
    """one spin-echo diffusion scan on the line at height y; readouts = [(center, label), ...]"""
    ax.hlines(y, 0, 185, color=INK["grid"], lw=1)
    ax.add_patch(Rectangle((0, y), 2.0, 0.55, color=INK["primary"]))
    ax.text(1, y + 0.62, "90°", ha="center", fontsize=7)
    ax.add_patch(Rectangle((te / 2 - 1.5, y), 3.0, 0.55, color=INK["primary"]))
    ax.text(te / 2, y + 0.62, "180°", ha="center", fontsize=7)
    for start in (te / 2 - T180 / 2 - DELTA, te / 2 + T180 / 2):
        ax.add_patch(Rectangle((start, y), DELTA, 0.35, color=PALETTE[1], alpha=0.85))
    for center, label in readouts:
        ax.add_patch(Rectangle((center - RO / 2, y), RO, 0.45, facecolor=PALETTE[0], alpha=0.3, edgecolor=PALETTE[0]))
        ax.text(center, y + 0.22, label, ha="center", va="center", fontsize=7)
    ax.plot([te, te], [y - 0.12, y + 0.75], ls=":", color=INK["secondary"], lw=1)


fig, (axa, axb, axc) = plt.subplots(3, 1, figsize=(9, 6.2), sharex=True, gridspec_kw={"height_ratios": [2.3, 1, 1.6]})
multi_te = [70.0, 100.0, 130.0]
for row, te in enumerate(multi_te):
    y = 2.0 * (len(multi_te) - 1 - row)
    draw_scan(axa, y, te, [(te, f"TE {te:.0f} ms")])
    axa.text(-3, y + 0.2, f"scan {row + 1}", ha="right", va="center", fontsize=8)
axa.set(ylim=(-0.4, 5.0), yticks=[], title="multi-TE: one echo time per excitation; the 180° pulse moves to TE / 2")
draw_scan(axb, 0.0, te1, [(te, f"echo {k + 1}, TE {te:.0f} ms") for k, te in enumerate(echo_times)])
axb.text(-3, 0.2, "one scan", ha="right", va="center", fontsize=8)
axb.set(ylim=(-0.3, 1.0), yticks=[], title="multi-echo: several echo times per excitation; one 180° pulse, three readouts")
for ax in (axa, axb):
    ax.grid(False)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
axa.legend(handles=[Patch(color=INK["primary"], label="RF pulse"), Patch(color=PALETTE[1], label="diffusion gradient"),
                    Patch(facecolor=PALETTE[0], alpha=0.3, edgecolor=PALETTE[0], label="EPI readout")],
           loc="upper right", fontsize=7, ncol=3)

# white matter signal (b = 0) through the multi-echo scan: T2 decay times reversible dephasing (T2')
t2_wm = presets.T2_MS["adult"]["WM"]
t2_prime = 1 / (1 / t2star["WM"] - 1 / t2_wm)
t = np.linspace(0, 185, 800)
dephase_time = np.where(t < te1 / 2, t, np.abs(t - te1))  # the 180° pulse reverses the dephasing
envelope = np.exp(-t / t2_wm) * np.exp(-dephase_time / t2_prime)
axc.plot(t, np.exp(-t / t2_wm), ls="--", color=INK["secondary"], lw=1, label=f"T2 decay alone (T2 {t2_wm:.0f} ms)")
axc.plot(t, np.exp(-dephase_time / t2_prime), ls=":", color=PALETTE[3], lw=1.5, label="effect of static field differences alone (1 = phases agree)")
axc.plot(t, envelope, color=TISSUE_COLORS["WM"], lw=1.8, label=f"white matter signal (T2* {t2star['WM']:.0f} ms)")
for k, te in enumerate(echo_times):
    axc.axvspan(te - RO / 2, te + RO / 2, color=PALETTE[0], alpha=0.1, lw=0)
    s_te = np.exp(-te / t2_wm) * np.exp(-abs(te - te1) / t2_prime)
    axc.plot(te, s_te, "o", color=TISSUE_COLORS["WM"])
axc.axvline(te1 / 2, color=INK["primary"], lw=0.8)
axc.text(te1 / 2 + 1, 0.9, "180°", fontsize=7)
axc.set(xlabel="time after excitation (ms)", ylabel="signal (b = 0)", ylim=(0, 1.02), xlim=(-25, 185),
        title="white matter signal during the multi-echo scan")
axc.legend(fontsize=7, loc="upper right")
fig.tight_layout()
print(f"EPI train duration {RO:.0f} ms ({len(tr.lines)} lines, echo spacing {esp:.2f} ms); effective total readout time {esp / 2 * 127:.0f} ms; echo times " + ", ".join(f"{te:.0f}" for te in echo_times) + " ms")
```

In the top panel, the multi-TE acquisition, each echo time is a separate scan.
The 180° pulse sits at half the echo time, and the spin echo and its readout move with it,
so every image is collected at its own spin echo and is weighted by T2 alone. In the middle
panel, the multi-echo acquisition, there is one excitation, one 180° pulse,
and one spin echo, and the EPI readout is repeated twice more after the first. The
diffusion gradients are played once, before the first readout, so all three echoes carry
the same diffusion weighting.

The bottom panel shows what the later echoes measure. The spin-echo animation of
[Chapter 1](../01-mri-physics/01-spins-and-signal.md) showed the spin phases fanning out after excitation, being flipped by the 180°
pulse, lining up again at the echo time, and then drifting apart again. The dotted curve
is that story on its own: the loss caused by small static differences in the magnetic
field across the voxel grows after excitation, is undone by the 180° pulse so that the
phases agree again exactly at the spin echo, and grows again afterwards. The solid curve is
the white matter signal, the dotted effect multiplied by the true T2 decay (dashed). It
falls at the T2* rate after excitation (T2* is the decay time that includes the static
dephasing), meets the dashed T2 curve at the spin echo, and falls at the T2* rate again
after it.
Echo 1 sits on the spin echo and is T2-weighted; echoes 2 and 3 carry additional T2* decay
over the time since the spin echo, and their SNR is correspondingly lower.

The two acquisitions therefore answer different questions. A multi-TE series measures **T2**,
and because diffusion weighting is applied at every echo time, T2 can be separated by
compartment. A multi-echo series measures **T2\*** in every voxel, which adds the dephasing by
static field differences to T2; it is a gradient-echo measure, sensitive to iron and myelin.
The next two sections take each in turn.

(sec-multi-te)=
## Multi-TE: one echo time per excitation

In a multi-TE acquisition every excitation yields one image at one echo time, and the echo
time changes from volume to volume. The echo times can be separate scans, as in the timing
diagram, or different volumes of a single scan.

### Diffusion-relaxation correlation

Acquiring several echo times turns the confound into a measurement. The signal of a
compartment is the product of its T2 decay and its diffusion attenuation, so on a grid of
(b, TE) values the compartments separate along both axes at once. The maps below show the
direction-averaged signal of each compartment of the voxel over that plane, and their sum,
which is what the scanner measures:

```{code-cell} python
:tags: [hide-input]
def sm_parts(f, t2i, t2e, b, te):
    """direction-averaged (spherical mean) signal of each compartment at every (b, TE)"""
    d_i = presets.ADULT_DIFFUSIVITY["WM_intra"]
    d_par, d_perp, _ = presets.ADULT_DIFFUSIVITY["WM_extra"]
    stick_mean = lambda bd: np.sqrt(np.pi / (4 * np.maximum(bd, 1e-9))) * erf(np.sqrt(np.maximum(bd, 1e-9)))
    intra = f * np.exp(-te / t2i) * np.where(b > 0, stick_mean(b * d_i), 1.0)
    extra = (1 - f) * np.exp(-te / t2e) * np.exp(-b * d_perp) * np.where(b > 0, stick_mean(b * (d_par - d_perp)), 1.0)
    return intra, extra


def sm_signal(f, t2i, t2e, b, te):
    intra, extra = sm_parts(f, t2i, t2e, b, te)
    return intra + extra


b_fine, te_fine = np.linspace(0, 3000, 121), np.linspace(40, 160, 121)
Bf, TEf = np.meshgrid(b_fine, te_fine)  # rows: TE, columns: b
intra_map, extra_map = sm_parts(F_VOL, T2_INTRA, T2_EXTRA, Bf, TEf)
total_map = intra_map + extra_map
extent = [b_fine[0], b_fine[-1], te_fine[0], te_fine[-1]]
fig, axes = plt.subplots(1, 4, figsize=(13, 3.5), layout="constrained")
vmax = total_map.max()
for ax, img, title in zip(axes[:3], [intra_map, extra_map, total_map], ["intra-axonal signal", "extra-axonal signal", "sum: what is measured"]):
    im = ax.imshow(img, origin="lower", extent=extent, aspect="auto", cmap="magma", vmin=0, vmax=vmax)
    cs = ax.contour(Bf, TEf, img, levels=[0.02, 0.05, 0.1, 0.2], colors="white", linewidths=0.7)
    ax.clabel(cs, fontsize=7, fmt="%.2f")
    ax.set(xlabel="b (s/mm²)", title=title)
    ax.grid(False)
axes[0].set_ylabel("echo time TE (ms)")
fig.colorbar(im, ax=axes[2], label="signal (1 = whole voxel at TE 0, b 0)")
im = axes[3].imshow(intra_map / total_map, origin="lower", extent=extent, aspect="auto", cmap="viridis", vmin=0.5, vmax=1)
cs = axes[3].contour(Bf, TEf, intra_map / total_map, levels=[0.7, 0.8, 0.9], colors="white", linewidths=0.7)
axes[3].clabel(cs, fontsize=7, fmt="%.1f")
axes[3].set(xlabel="b (s/mm²)", title="intra-axonal share of the sum")
axes[3].grid(False)
fig.colorbar(im, ax=axes[3], label="share")
print("intra-axonal share of the measured signal: "
      + ", ".join(f"b {b:.0f}, TE {te:.0f} ms {sm_parts(F_VOL, T2_INTRA, T2_EXTRA, b, te)[0] / sm_signal(F_VOL, T2_INTRA, T2_EXTRA, b, te):.2f}"
                  for b, te in [(0, 60), (3000, 60), (0, 130), (3000, 130)]))
```

Compare where the two compartments keep their signal. The intra-axonal map fades slowly in
both directions and still holds a few percent of signal in the upper-right corner (high b,
long TE). The extra-axonal map is concentrated in the lower-left corner: it is lost both to
diffusion weighting, because water moves easily across the fibers outside the axons, and to
echo time, because of its shorter T2. The right-hand panel shows the consequence: the
intra-axonal share of the measured signal rises from the lower-left corner to the
upper-right one, as printed above. Moving along the b axis and moving along the TE axis
both enrich the signal in the intra-axonal compartment, but by different amounts, and a
model fitted to the whole plane uses that difference to tell the compartments apart.

### Fitting the (b, TE) plane

Such a fit recovers the compartment fractions at TE = 0, where no relaxation has happened
yet. These are the compartments' shares of the water signal, which equal volume fractions
if the compartments hold water at the same density (the *proton density*, the
concentration of hydrogen nuclei that give the MRI signal) and recover equally between
excitations. It also recovers the compartment T2 values, which are new tissue parameters in
their own right. Below, the voxel is sampled on a 4 × 4 grid of b and TE with a little
noise, and the three parameters are fitted:

```{code-cell} python
:tags: [hide-input]
b_grid = np.array([0, 1000, 2000, 3000], float)
te_grid = np.array([60, 80, 100, 130], float)
B, TE = np.meshgrid(b_grid, te_grid, indexing="ij")
truth = (F_VOL, T2_INTRA, T2_EXTRA)
rng = np.random.default_rng(0)
data = sm_signal(*truth, B, TE) + rng.normal(scale=0.004, size=B.shape)

fig, ax = plt.subplots(figsize=(6, 3.4))
for k, te in enumerate(te_grid):
    color = plt.cm.viridis(k / 3)
    ax.semilogy(b_fine, sm_signal(*truth, b_fine, te) / sm_signal(*truth, 0.0, te), color=color, label=f"TE {te:.0f} ms")
    ax.semilogy(b_grid, data[:, k] / data[0, k], "o", color=color, ms=5)
ax.set(xlabel="b (s/mm²)", ylabel="signal / signal at b = 0", title="one white matter voxel on the (b, TE) grid")
ax.legend()
fig.tight_layout()
print("noise-free signal at b = 3000 relative to b = 0: " + ", ".join(f"TE {te:.0f} ms {sm_signal(*truth, 3000.0, te) / sm_signal(*truth, 0.0, te):.3f}" for te in te_grid))
print("b = 0 signal relative to TE 60 ms: " + ", ".join(f"TE {te:.0f} ms {sm_signal(*truth, 0.0, te) / sm_signal(*truth, 0.0, 60.0):.2f}" for te in te_grid))


def residual(p):
    f, t2i, t2e = p
    return (sm_signal(f, t2i, t2e, B, TE) - data).ravel()


fit = least_squares(residual, x0=[0.5, 70.0, 70.0], bounds=([0, 20, 20], [1, 300, 300]))
print(f"joint fit: intra-axonal fraction {fit.x[0]:.2f} (true {F_VOL:.2f}), T2 intra {fit.x[1]:.0f} ms (true {T2_INTRA:.0f}), T2 extra {fit.x[2]:.0f} ms (true {T2_EXTRA:.0f})")

# the same fit repeated with 200 noise draws, to show how much one fit can vary
sweep = []
for seed in range(200):
    noisy = sm_signal(*truth, B, TE) + np.random.default_rng(seed).normal(scale=0.004, size=B.shape)
    sweep.append(least_squares(lambda p: (sm_signal(*p, B, TE) - noisy).ravel(), x0=[0.5, 70.0, 70.0], bounds=([0, 20, 20], [1, 300, 300])).x)
lo, med, hi = np.percentile(np.array(sweep), [5, 50, 95], axis=0)
print(f"over 200 noise draws (5th, 50th, 95th percentile): fraction {lo[0]:.2f}, {med[0]:.2f}, {hi[0]:.2f}; "
      f"T2 intra {lo[1]:.0f}, {med[1]:.0f}, {hi[1]:.0f} ms; T2 extra {lo[2]:.0f}, {med[2]:.0f}, {hi[2]:.0f} ms")
```

Each curve (the noise-free signal) and its dots (the noisy grid) are divided by their own
b = 0 value, so only the shape of the diffusion decay is compared. The curves at longer TE
lie slightly higher at high b: the signal that survives a long echo time is richer in
intra-axonal water, which loses less signal to diffusion weighting, so the decay flattens a
little as TE grows. The change in shape is small, from 0.28 to 0.30 of the b = 0 signal at
b = 3000 between TE 60 and 130 ms (first printed line), and comparable to the noise on the
dots; most of the relaxation information is in
how the b = 0 signal itself falls with TE (second printed line), which the normalization
hides. A diffusion-only fit at one TE sees one of these curves and cannot tell whether its
shape comes from diffusion or relaxation. The joint fit uses the unnormalized signal at all
16 points and returns a fraction and two T2 values, where any single echo time returned a
TE-dependent fraction and no T2 information at all. One noise draw is not the whole story,
so the fit is repeated with 200 draws (last printed line). The fits center on the true
values, with no bias from the echo time, but they scatter: the fraction falls between 0.51
and 0.61 in nine fits out of ten, and the intra-axonal T2 between 82 and 96 ms. The
scatter is the price of the noise: the two compartment T2 values are only 30 ms apart,
and separating two decays that similar from a few points is a hard fit.

### Sampling the (b, TE) plane

A full grid is rarely necessary. What the fit needs is enough spread along both axes to
separate the compartments: at least three echo times spanning a range comparable to the
compartment T2 values (60–130 ms), and the usual two or three shells. The cost is scan time
proportional to the number of echo times, and SNR: every volume at a long TE has lost more
signal, so the volumes at the longest echo time have the lowest SNR, and they limit the
precision of the whole fit. For this voxel the loss is:

```{code-cell} python
:tags: [hide-input]
rel = sm_signal(*truth, 0.0, 130.0) / sm_signal(*truth, 0.0, 60.0)
print(f"white matter b = 0 signal (and SNR) at TE 130 ms relative to TE 60 ms: {rel:.2f}; "
      f"averages needed at TE 130 ms to match the SNR at TE 60 ms: {1 / rel**2:.1f}")
```

Because noise does not depend on TE, the SNR falls in proportion to the signal, and
averaging recovers it only as the square root of the number of averages. The long-TE
volumes are needed for what they add, the decay with TE, and a joint fit like the one above
uses only the direction-averaged signal of each shell, which a modest number of
directions already gives. Many designs therefore acquire the full set of directions at the
shortest TE, where the signal is strong and fiber orientation is estimated, and a smaller
set at each longer TE, spending the time saved on averages where the signal is weakest.

(sec-multi-echo)=
## Multi-echo: several echo times per excitation

In a multi-echo acquisition the EPI readout is repeated after each excitation, so every
diffusion-weighted volume comes with several images at increasing echo times. All of them
share one diffusion weighting and one spin echo; what changes from echo to echo is the time
since the spin echo, and with it the T2\* decay.

```{code-cell} python
:tags: [hide-input]
fig, ax = plt.subplots(figsize=(5, 3.2))
for name, t2s in t2star.items():
    ax.plot(echo_times, [np.exp(-(te - te1) / t2s) for te in echo_times], "o-", color=TISSUE_COLORS[name], label=f"{name} (T2* {t2s:.0f} ms)")
ax.set(xlabel="echo time (ms)", ylabel="signal relative to echo 1", title="T2* decay between echoes", xticks=echo_times)
ax.legend(fontsize=8)
fig.tight_layout()
print("signal at echo 3 relative to echo 1: " + ", ".join(f"{name} {np.exp(-(echo_times[2] - te1) / t2s):.2f}" for name, t2s in t2star.items()))
```

With partial Fourier and twofold in-plane acceleration, each EPI train acquires 48 lines
and lasts about 34 ms. That is the duration of the train, which sets how far apart the
echoes can be; it is not the total readout time of [Chapter 10](../03-preprocessing/10-susceptibility-distortion.md), which sets the
susceptibility displacement and here is about 45 ms (printed with the timing diagram).
The three echoes are centered at 70, 106, and 141 ms. By the third echo white matter keeps
only about a fifth of its signal, while CSF, with its long T2*, keeps about four fifths.

### Echo combination and T2* mapping

Three uses follow from having several echoes of every diffusion-weighted volume:

- **Combination.** A weighted sum of the echoes, with weights proportional to each echo's
  expected SNR, has a higher SNR than any single echo. The gain is modest, because the
  later echoes are weak, but it adds no scan time as long as the extra readouts fit into the
  repetition time the protocol already has.
- **T2\* per volume.** Two or more echoes give the T2* decay in every voxel of every
  diffusion-weighted volume, so relaxation and diffusion are measured together. This is a
  gradient-echo relaxometry measurement, sensitive to iron and myelin, obtained inside the
  diffusion scan.
- **Recovery of readout-time corruption.** Signal corrupted in one echo by an event during
  that readout, such as an RF spike, is intact in the other echoes, so the volume can be
  repaired from its own echoes rather than from the model prediction of [Chapter 12](../03-preprocessing/12-motion-and-dropout.md). This
  does not extend to the usual motion dropout: motion during the diffusion encoding
  dephases the signal before the first readout begins, so every echo loses it together.

The T2* estimate from two echoes is one line of arithmetic. Between echoes ΔTE apart the
signal is multiplied by exp(−ΔTE / T2*), so T2* = ΔTE / ln(S₁ / S₂), where S₁ and S₂ are the
first and second echoes. The synthetic slice below applies it voxel by voxel: tissue T2*
values are assigned, noise is added to each echo, and T2* is estimated from the ratio.

```{code-cell} python
:tags: [hide-input]
t = phantoms.brain_slice()
mask = t["mask"]
b0 = phantoms.brain_image(te1)  # the b = 0 image at the spin echo (echo 1)
t2s_map = (t["wm"] * t2star["WM"] + t["gm"] * t2star["GM"] + t["csf"] * t2star["CSF"]) / np.maximum(t["wm"] + t["gm"] + t["csf"], 1e-6)
echoes = []
for k, te in enumerate(echo_times):
    decay = np.where(mask, np.exp(-(te - te1) / np.maximum(t2s_map, 1)), 0)
    echoes.append(synth.add_noise(b0 * decay, 0.006, seed=k))
dte = echo_times[1] - echo_times[0]
ratio_wm = np.exp(-dte / t2star["WM"])
print(f"worked example, white matter: echoes {dte:.0f} ms apart, S2 / S1 = {ratio_wm:.2f}, T2* = {dte:.0f} / ln(1 / {ratio_wm:.2f}) = {dte / np.log(1 / ratio_wm):.0f} ms")
ratio = np.clip(echoes[1] / np.maximum(echoes[0], 1e-6), 1e-3, 0.999)
t2s_est = np.where(mask, -dte / np.log(ratio), np.nan)
weights = np.array([np.exp(-(te - te1) / 45.0) for te in echo_times])
combined = sum(w * e for w, e in zip(weights, echoes)) / np.sqrt(np.sum(weights**2))

fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
show_image(axes[0], echoes[0], f"echo 1, TE {echo_times[0]:.0f} ms", vmin=0, vmax=0.5)
show_image(axes[1], echoes[2], f"echo 3, TE {echo_times[2]:.0f} ms", vmin=0, vmax=0.5)
show_image(axes[2], combined, "SNR-weighted combination", vmin=0, vmax=0.5)
show_image(axes[3], t2s_est, "T2* from echoes 1 and 2 (ms)", kind="scalar", vmin=20, vmax=120)
fig.tight_layout()
wm = t["wm"] > 0.9
csf_vox = t["csf"] > 0.9
noise_single = np.std((echoes[0] - b0)[wm])
signal_comb_true = b0 * sum(w * np.exp(-(te - te1) / 45.0) for w, te in zip(weights, echo_times)) / np.sqrt(np.sum(weights**2))
noise_comb = np.std((combined - signal_comb_true)[wm])
snr_single = b0[wm].mean() / noise_single
snr_comb = signal_comb_true[wm].mean() / noise_comb
gain = snr_comb / snr_single
print(f"white matter: T2* estimate {np.nanmedian(t2s_est[wm]):.0f} ms (assigned 45); SNR echo 1 {snr_single:.0f}, "
      f"SNR-weighted combination of three echoes {snr_comb:.0f} (gain {gain:.2f}, worth {gain**2:.2f} times as many averages)")
q25, q50, q75 = np.nanpercentile(t2s_est[csf_vox], [25, 50, 75])
print(f"CSF: signal falls by {1 - np.exp(-dte / t2star['CSF']):.0%} between echoes 1 and 2; "
      f"T2* estimate median {q50:.0f} ms (assigned 300), middle half {q25:.0f}-{q75:.0f} ms")
```

The first three panels share one gray scale. Echo 3 is dark everywhere except in CSF, which
has kept most of its signal; the combination looks like echo 1 with slightly less noise.
The SNR gain printed above is about a tenth, the same improvement as acquiring about 1.2
times as many volumes, and it adds no scan time if the TR has room for the extra readouts. In white matter the second echo is
0.45 of the first, which gives the assigned 45 ms back. The gray scale is set for tissue,
so the ventricles, much brighter than tissue at every echo, are clipped to white. The T2*
map is clipped too: CSF sits at the top of its color scale because its assigned T2* of
300 ms lies beyond the 120 ms range shown. The estimate itself is close to the assigned
value (printed above) even though CSF loses only about a tenth of its signal between
echoes 1 and 2, because CSF is bright enough that this small change stands clear of the
noise; a dim tissue with a long T2* would not.

### Distortion and signal loss per echo

Every echo is read with the same EPI train, so the susceptibility displacement ([Chapter 10](../03-preprocessing/10-susceptibility-distortion.md))
is the same in all of them and one correction applies to all. What differs is the signal
near air-tissue interfaces. There the field changes steeply across the thickness of a
slice, the spins at the top and bottom of the slice precess at different rates, and their
phases spread apart. The spread grows with the time since the spin echo, so the later
echoes lose more signal above the sinuses and in the temporal poles than the first. The toy
below places a field that varies through the slice, strongest at one spot at the front of
the brain, and shows each echo brightened by the expected T2* decay of white matter, so
that white matter has the same brightness in all three and the extra loss stands out
(gray matter and the noise still change from echo to echo):

```{code-cell} python
:tags: [hide-input]
rows, cols = np.indices(mask.shape)
r_front = np.argwhere(mask)[:, 0].min() + 4  # a few voxels inside the anterior edge (top of the image)
c_mid = int(np.median(np.argwhere(mask)[:, 1]))
SLICE_MM = t["voxel_mm"]
G_PEAK = 5.0  # Hz per mm through the slice, at the center of the spot
g_through = G_PEAK * np.exp(-((rows - r_front) ** 2 + (cols - c_mid) ** 2) / (2 * 8.0**2))


def through_slice_loss(time_ms):
    """magnitude of the slice-averaged phase factor: a linear phase across the slice averages to a sinc"""
    return np.abs(np.sinc(g_through * SLICE_MM * time_ms * 1e-3))


fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
for k, (ax, te) in enumerate(zip(axes, echo_times)):
    shown = echoes[k] * through_slice_loss(te - te1) / np.exp(-(te - te1) / t2star["WM"])
    show_image(ax, shown, f"echo {k + 1}: {te - te1:.0f} ms after the spin echo", vmin=0, vmax=0.5)
    ax.add_patch(plt.Circle((c_mid, r_front), 16, fill=False, color=PALETTE[3], lw=1, ls="--"))
show_image(axes[3], np.where(mask, through_slice_loss(echo_times[2] - te1), np.nan), "fraction of signal kept, echo 3", kind="scalar", vmin=0, vmax=1)
fig.tight_layout()
print("signal kept at the center of the spot: " + ", ".join(f"echo {k + 1} {float(through_slice_loss(te - te1)[r_front, c_mid]):.2f}" for k, te in enumerate(echo_times)))
```

Look inside the dashed circle. Echo 1 is centered on the spin echo, which refocuses the
static field, so in this toy it loses nothing there. Echo 2 has lost about a fifth of the
signal at the center of the spot and echo 3 nearly two thirds, as printed above; the
right-hand panel maps the fraction echo 3 keeps, which is one everywhere outside the spot.
(In a real acquisition the first
echo also loses some signal, because its 34 ms train extends on both sides of the spin
echo.) A combination weighted by SNR handles this well, because it already takes most
of its signal from the first echo; a T2* map computed near such a spot, on the other hand,
reports the extra loss as a short T2*.

## Combining the two

The two approaches can be combined in one acquisition. ZEBRA {cite:p}`hutter2018` reads
several gradient echoes after each excitation, as a multi-echo acquisition does, and changes
the inversion time from slice to slice, so that diffusion, T2\* and T1 weighting are sampled
together in one scan of clinical length. Varying the spin-echo time between volumes as well
adds T2. A model of such data has to carry both relaxation times, because every echo away
from the spin echo is weighted by T2 over the whole echo time and by the static dephasing
over the time since the spin echo, as the bottom panel of the timing diagram shows.

## Measure it: the simulated datasets

:::{admonition} Simulated dataset pending
:class: note
The `te-sweep` dataset (the HBCD scheme at four echo times, once the simulator accepts a
per-run TE) will be fitted with the joint multi-TE model across tissues: the recovered T2
values are compared with the preset T2 of each tissue, and the TE dependence of the
fitted fractions with the truth maps. Because the simulator gives both white matter
compartments one T2, the dataset will show the between-tissue effect, not the within-tissue
one of this chapter's toy voxel.

The simulator's readout model is single-echo. A multi-echo readout, a planned simulator
extension, would add per-echo k-space with T2\* decay and dephasing between echoes; until it
exists, the multi-echo sections of this chapter are toy-only.
:::

## What this implies for acquisition

- **Report the echo time with every diffusion metric**, and keep it fixed within a study.
- **Fractions from single-TE data are signal fractions**; comparing them across protocols
  with different TE compares different quantities.
- **Choose between the two by what you need to measure.** Multi-TE measures T2 and can
  separate it by compartment; multi-echo measures T2\* per voxel, which includes static
  dephasing and is sensitive to iron and myelin.
- **A multi-TE acquisition needs three or more echo times** spanning 60–130 ms and a
  multi-shell scheme at each; it costs scan time in proportion to the number of echo times,
  and the SNR at the longest TE limits the fit.
- **A multi-echo readout costs TR, and only through TR costs volumes**: each extra echo adds a
  readout duration per slice, which lengthens the minimum TR ([Chapter 7](../02-diffusion-encoding/07-acquisition-parameters.md)). If the protocol's TR
  is already longer than that minimum (chosen, for instance, for full T1 recovery), the
  extra echoes are free; if the TR is set by the readouts, it grows and fewer volumes fit
  into each minute. Shorten the readout first (partial Fourier, in-plane acceleration) so
  that the extra echoes arrive before the signal is gone.
  The gain is relaxometry and robustness to readout-time corruption, not a large SNR
  increase.

## Further reading

TE-dependent diffusion imaging {cite:p}`veraart2018`, integrated diffusion-relaxometry
acquisitions {cite:p}`hutter2018`, and the EPI readout timing of [Chapter 2](../01-mri-physics/02-spatial-encoding-kspace.md).
