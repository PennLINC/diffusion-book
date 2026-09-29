---
title: "20. Multi-TE diffusion MRI"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** single-voxel signals with the compartment T2 values of the presets ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
- **`te-sweep`** (pending): four echo times at fixed b ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-te-sweep)).
- **`truth`** (pending): the 27 analytic ground-truth maps and the true fiber orientations ([Appendix A](../appendices/a-trxscan-cookbook.md#ds-truth), [Appendix E](../appendices/e-truth-map-catalogue.md)).

Pipeline-tier datasets are simulated offline by TRXScan ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)) and are marked *pending* until their release; the figures that need them say so where they will appear.
:::

## Learning goals

After this chapter you can:

- explain why diffusion measures depend on the echo time when tissue compartments have
  different T2 values
- fit a joint diffusion-relaxation model to data acquired at several echo times and
  recover compartment T2 values
- state what a multi-TE acquisition costs and how to sample the (b, TE) plane

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.special import erf

from dwibook import phantoms, presets, schemes, signal, synth
from dwibook.plotting import INK, PALETTE, TISSUE_COLORS, animate, set_style, show_image

set_style()
```

## Compartmental T2

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

## See it: the apparent fraction depends on TE

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
```

Follow the solid line from left to right: the fitted fraction climbs steadily with echo
time, by the amount printed above between 60 and 130 ms, parallel to the dotted signal
share. It sits above the dashed volume fraction at every TE, for two reasons that add. The
first is the relaxation effect of the animation: at any realistic TE the longer-T2
compartment has already gained signal share. The second is model mismatch: the
spherical-mean fit makes a *tortuosity* assumption, that the more densely axons are packed,
the more slowly water moves sideways between them, and so ties the extra-axonal radial
diffusivity to the fraction; the synthetic tissue does not follow that rule ([Chapter 17](../04-modeling/17-microstructure-models.md)).
The mismatch is the same at every TE, so the *trend* with TE is the relaxation effect alone.

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

## Diffusion-relaxation correlation

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
16 points and returns a fraction and two T2 values close to the true ones, where any single
echo time returned a TE-dependent fraction and no T2 information at all. The remaining
error is noise: the two compartment T2 values are only 30 ms apart, and separating two
decays that similar from a few points is a hard fit.

Grids of this kind need not cost one scan per point. ZEBRA {cite:p}`hutter2018` reads
several gradient echoes after each excitation ([Chapter 21](./21-multi-echo.md)) and varies the inversion time from
slice to slice, so that diffusion weighting, T2* weighting, and T1 weighting are sampled
together in one scan of clinical length.

## Sampling the (b, TE) plane

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
averaging recovers it only as the square root of the number of averages. Sampling
strategies therefore place more diffusion directions at short TE, where the signal is
strong, and fewer at long TE, where the relaxation information lives.

## Measure it: the simulated datasets

:::{admonition} Simulated dataset pending
:class: note
This section will load the `te-sweep` dataset (the HBCD scheme at four echo times, once the
simulator accepts a per-run TE) and fit the joint model across tissues: the recovered T2
values are compared with the preset T2 of each compartment, and the TE dependence of the
fitted fractions with the truth maps. Because the simulator gives both white matter
compartments one T2, the dataset will show the between-tissue effect, not the within-tissue
one of this chapter's toy voxel.
:::

## What this implies for acquisition

- **Report the echo time with every diffusion metric**, and keep it fixed within a study.
- **Fractions from single-TE data are signal fractions**; comparing them across protocols
  with different TE compares different quantities.
- **A multi-TE acquisition needs three or more echo times** spanning 60–130 ms and a
  multi-shell scheme at each; budget the SNR at the longest TE.

## Further reading

TE-dependent diffusion imaging {cite:p}`veraart2018` and integrated diffusion-relaxometry
{cite:p}`hutter2018`.
