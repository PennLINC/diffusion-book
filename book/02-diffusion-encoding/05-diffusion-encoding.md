---
title: "5. Diffusion encoding"
kernelspec:
  name: python3
  display_name: Python 3
---

:::{admonition} Simulated datasets in this chapter
:class: note
- **Built in this page:** single-voxel compartment signals under the HBCD scheme, which ships with the book ([Appendix B](../appendices/b-data-manifest.md#app-b-package-data)).
:::

## Learning goals

After this chapter you can:

- describe how a pair of gradient pulses makes the signal sensitive to displacement, and
  why the result depends on the b-value
- state what the b-value is made of (gradient strength, pulse duration, pulse separation)
  and why reaching a high b-value costs echo time
- compare the signal that white matter, gray matter, and CSF retain at the b-values used in
  practice
- name the two side effects of strong diffusion gradients that later chapters correct

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt

from dwibook import phantoms, presets, schemes, signal, synth
from dwibook.plotting import PALETTE, TISSUE_COLORS, animate, set_style, show_image

set_style()
```

## The pulsed-gradient spin echo

Diffusion encoding adds two identical gradient pulses to the spin-echo sequence of
[Chapter 1](../01-mri-physics/01-spins-and-signal.md), one on each side of the 180° pulse {cite:p}`stejskal1965`. The first pulse gives
every spin a phase proportional to its position along the gradient. The 180° pulse reverses
that phase. The second pulse adds the same position-dependent phase again. A spin that did
not move between the pulses ends with zero net phase. A spin that moved ends with a phase
proportional to its displacement.

Within a voxel, the spins have moved by different random amounts, so their phases are
spread out and their sum is reduced. The signal loss is the measurement. Molecules that
moved farther, because diffusion is faster or the compartment is larger, spread the phase
more and reduce the signal more.

```{code-cell} python
:tags: [hide-input]
fig, ax = plt.subplots(figsize=(9, 3.2))
t = np.linspace(0, 100, 2000)
TE, delta, Delta = 88.0, 20.0, 40.0
t1 = TE / 2 - (Delta - delta) / 2 - delta  # pulses placed symmetrically about the 180°
t2 = t1 + Delta                            # second pulse starts Delta after the first
ro_half = 12.0                             # half the readout duration in this sketch

rf = np.zeros_like(t); rf[(t > 0) & (t < 2)] = 1.0; rf[(t > TE / 2 - 1) & (t < TE / 2 + 1)] = 2.0
diff = np.zeros_like(t); diff[(t > t1) & (t < t1 + delta)] = 1.0; diff[(t > t2) & (t < t2 + delta)] = 1.0
epi = np.where(np.abs(t - TE) < ro_half, np.sign(np.sin(2 * np.pi * (t - TE) / 1.4)), 0.0)  # readout after the second pulse, its k-space center at TE

ax.fill_between(t, 6, 6 + rf, color="0.3"); ax.text(-1, 6.4, "RF", ha="right", fontsize=9)
ax.fill_between(t, 3.5, 3.5 + 1.6 * diff, color=PALETTE[0], alpha=0.8); ax.text(-1, 4.1, "diffusion\ngradient", ha="right", fontsize=9)
ax.plot(t, 1.2 + 0.8 * epi, color=PALETTE[1], lw=1); ax.text(-1, 1.2, "readout\n(EPI)", ha="right", fontsize=9)
for x, label in [(1, "90°"), (TE / 2, "180°"), (TE, "TE")]:
    ax.axvline(x, color="0.8", lw=1, zorder=0); ax.text(x, 8.4, label, ha="center", fontsize=9, color="0.4")
ax.annotate("", xy=(t1 + delta, 5.5), xytext=(t1, 5.5), arrowprops=dict(arrowstyle="<->", color="0.4"))
ax.text(t1 + delta / 2, 5.65, "δ", ha="center", fontsize=9)
ax.annotate("", xy=(t2, 3.2), xytext=(t1, 3.2), arrowprops=dict(arrowstyle="<->", color="0.4"))
ax.text((t1 + t2) / 2, 2.6, "Δ", ha="center", fontsize=9)
ax.set(xlim=(-12, 105), ylim=(0, 9), yticks=[], xlabel="time (ms)", title="pulsed-gradient spin echo with an EPI readout")
ax.grid(False)
fig.tight_layout()
```

The blue blocks are the two diffusion-gradient pulses, each lasting $\delta$, with their
starts separated by $\Delta$, placed symmetrically on either side of the 180° pulse. The
image is read out after the second pulse. The echo time is the moment the readout passes
through the center of k-space ([Chapter 2](../01-mri-physics/02-spatial-encoding-kspace.md));
in this sketch that is the middle of the readout, though partial Fourier, later in this
chapter, moves the center earlier in the readout.

The animation runs this sequence on three rows of molecules lined up along the gradient
direction, each drawn as an arrow pointing in the direction of its phase. The first gradient
pulse winds the phases into a helix, because each molecule's phase advances in proportion to
its position, and the sum over the row, on the right, drops to zero. That loss is temporary.
The 180° pulse reverses every phase, and the second pulse adds the same winding again,
which exactly unwinds a molecule that stayed where it was: the top row ends aligned, with
its full signal. A molecule that wandered between the pulses is unwound by the wrong amount,
in proportion to how far it moved, and keeps a leftover phase. The faster the diffusion,
the farther the molecules moved, the more their leftover phases scatter, and the shorter
the sum.

```{code-cell} python
:tags: [hide-input]
N_MOL, N_SHOW, N_FRAMES = 800, 25, 100
G1, FLIP, G2, READ = (10, 30), 45, (60, 80), 90   # frames: first pulse, 180°, second pulse, readout
K = 2 * np.pi * 2 / (G1[1] - G1[0])               # phase per unit position per frame of gradient: 2 turns per unit
rng = np.random.default_rng(0)
x0 = np.linspace(-1, 1, N_MOL)                    # starting positions along the gradient axis
rows = {"no motion": 0.0, "slow diffusion": 0.006, "fast diffusion": 0.016}  # random step per frame
show = np.linspace(0, N_MOL - 1, N_SHOW).astype(int)

def history(step):
    x, ph, xs, phs = x0.copy(), np.zeros(N_MOL), [], []
    for i in range(N_FRAMES):
        if i == FLIP:
            ph = -ph                              # the 180° pulse reverses every phase
        if G1[0] <= i < G1[1] or G2[0] <= i < G2[1]:
            ph = ph + K * x                       # a gradient adds phase in proportion to position
        xs.append(x.copy())
        phs.append(ph.copy())
        x = x + rng.normal(scale=step, size=N_MOL)
    return np.array(xs), np.array(phs)

hist = {name: history(step) for name, step in rows.items()}

fig = plt.figure(figsize=(9, 5.4))
gs = fig.add_gridspec(4, 2, height_ratios=[0.8, 1, 1, 1], width_ratios=[5, 1])
axt = fig.add_subplot(gs[0, 0])
t = np.arange(N_FRAMES)
on = ((t >= G1[0]) & (t < G1[1])) | ((t >= G2[0]) & (t < G2[1]))
axt.fill_between(t, 0, on.astype(float), color=PALETTE[0], alpha=0.7, step="post")
for x, label in [(0, "90°"), (FLIP, "180°"), (READ, "readout")]:
    axt.axvline(x, color="0.6", lw=1)
    axt.text(x + 0.8, 1.15, label, fontsize=8, color="0.4")
for g in (G1, G2):
    axt.text((g[0] + g[1]) / 2, 0.35, "gradient", ha="center", fontsize=8, color="white")
cursor = axt.axvline(0, color=PALETTE[1], lw=2)
axt.set(xlim=(0, N_FRAMES), ylim=(0, 1.5), yticks=[], xticks=[])
axt.grid(False)
quivers, sums = [], []
for r, name in enumerate(rows):
    ax = fig.add_subplot(gs[r + 1, 0])
    xs, phs = hist[name]
    quivers.append(ax.quiver(xs[0, show], np.zeros(N_SHOW), np.cos(phs[0, show]), np.sin(phs[0, show]),
                             color=PALETTE[r], pivot="middle", scale=22, width=0.006, headwidth=3))
    ax.set(xlim=(-1.15, 1.15), ylim=(-1, 1), yticks=[], xticks=[])
    ax.set_ylabel(name, fontsize=8, rotation=0, ha="right", va="center")
    ax.grid(False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    axs = fig.add_subplot(gs[r + 1, 1])
    axs.add_patch(plt.Circle((0, 0), 1, fill=False, color="0.85"))
    arrow = axs.annotate("", xy=(1, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=PALETTE[r], lw=2))
    label = axs.text(1.35, 0, "", va="center", fontsize=8)
    axs.set(xlim=(-1.2, 2.4), ylim=(-1.2, 1.2), aspect="equal")
    axs.set_axis_off()
    if r == 0:
        axs.set_title("sum of all molecules", fontsize=8)
    sums.append((arrow, label))
fig.text(0.42, 0.015, "position along the gradient →", ha="center", fontsize=8, color="0.3")
stage = fig.suptitle("", fontsize=10)
fig.tight_layout(rect=(0, 0.03, 1, 0.95))

def frame(i):
    cursor.set_xdata([i, i])
    for q, (arrow, label), (xs, phs) in zip(quivers, sums, hist.values()):
        q.set_offsets(np.column_stack([xs[i, show], np.zeros(N_SHOW)]))
        q.set_UVC(np.cos(phs[i, show]), np.sin(phs[i, show]))
        s = np.exp(1j * phs[i]).mean()
        arrow.xy = (s.real, s.imag)
        label.set_text(f"{abs(s):.2f}")
    stage.set_text("first gradient: the phase winds with position" if G1[0] <= i < G1[1] else
                   "180° pulse: every phase is reversed" if FLIP - 3 <= i <= FLIP + 3 else
                   "second gradient: the same winding is added again" if G2[0] <= i < G2[1] else
                   "readout: molecules that moved keep a leftover phase" if i >= G2[1] else
                   "the molecules wander between the pulses" if i >= G1[1] else "before the encoding: all in phase")

animate(fig, frame, range(N_FRAMES), fps=12, width=760, dpi=75,
        alt="three rows of molecules along the gradient axis, drawn as arrows showing their phase: the first gradient winds the phases into a helix, the 180-degree pulse reverses them, and the second gradient unwinds them; molecules that did not move end aligned with a full-length sum, slowly diffusing molecules end slightly scattered with a shorter sum, and fast-diffusing molecules end scattered with a much shorter sum")
```

## The b-value

Three settings determine how sensitive the measurement is: the gradient amplitude $G$, the
duration of each pulse $\delta$, and the separation between the pulses $\Delta$. In the
animation's terms, $G$ and $\delta$ together set how tightly the first pulse winds the phase,
and so how much leftover phase a given displacement produces; $\Delta$ sets how long the
molecules have to wander. The three are combined into one number, the b-value, which grows
with the square of the winding and with the wandering time. Doubling the gradient strength
makes $b$ four times larger; doubling the separation roughly doubles it. The formula is


$$b = \gamma^2 G^2 \delta^2 \left(\Delta - \tfrac{\delta}{3}\right),$$

in units of s/mm², where $\gamma$ is the constant of [Chapter 1](../01-mri-physics/01-spins-and-signal.md) that converts field into
precession frequency. For water diffusing freely with coefficient $D$, the signal falls
exponentially with $b$:

$$S(b) = S_0\, e^{-b D}.$$

$S_0$ is the signal without diffusion weighting (the b=0 image of [Chapter 1](../01-mri-physics/01-spins-and-signal.md)). With $D$ in
mm²/s and $b$ in s/mm², the product $bD$ is dimensionless; at $b = 1000$ s/mm² free water
($D = 3 \times 10^{-3}$) retains $e^{-3} \approx 5$ % of its signal, and the water between
white matter fibers, diffusing across them ($D \approx 0.6 \times 10^{-3}$), retains about 55 %.
White matter as a whole keeps more than that across the fibers, about 80 %, because the
water inside the axons cannot move across them at all; a single $D$ does not describe it,
as the next figure shows.

The diffusion time of [Chapter 4](./04-diffusion-in-tissue.md) is set mostly by the
separation $\Delta$. Because molecules keep moving while each pulse is on, the time that
counts is the *effective diffusion time* $\Delta - \delta/3$, the same bracket that appears
in the formula for $b$. The winding also has a name of its own: the *q-value*,
$q = \gamma G \delta / 2\pi$, counts how many turns of phase the pulse pair leaves per unit
of displacement, so that $b = (2\pi q)^2 (\Delta - \delta/3)$. [Chapter 4](./04-diffusion-in-tissue.md)
described the signal as the Fourier transform of the displacement distribution, read at
spatial frequency $q$. That description is exact only when the pulses are brief compared
with their separation ($\delta \ll \Delta$, the *narrow-pulse* condition), so that each
molecule has a well-defined position during each pulse. With the 20 ms pulses and 40 ms
separation of a clinical scanner (below) the molecules move during the pulses, and the
signal instead reports, roughly, the displacement between each molecule's average position
during the first pulse and during the second {cite:p}`mitra1995`. For freely diffusing water nothing is lost,
because the formula for $b$ already accounts for the pulse length; for water in small
compartments, and for methods that rebuild the displacement distribution from q-space
([Chapter 6](./06-qspace-sampling.md)), long pulses make restricted displacements look
smaller than they are.

Gradient strength is measured in millitesla per meter (mT/m), and it is the scanner's
hardware limit. The three lines below are example timings on three classes of system: a
clinical scanner (40 mT/m), a research whole-body scanner (80 mT/m), and the Connectom
scanner built for diffusion (300 mT/m).

```{code-cell} python
:tags: [hide-input]
for system, (g, d, D) in {"clinical": (40, 20, 40), "whole-body": (80, 14, 34), "Connectom": (300, 6, 26)}.items():
    print(f"{system:>10}: G = {g:>3} mT/m, delta = {d:>2} ms, Delta = {D:>2} ms  ->  b = {signal.b_value(g, d, D):5.0f} s/mm^2")
```

Stronger gradients buy b-value with shorter timing. The 80 mT/m system reaches about 1.7
times the b-value of the clinical one with pulses 6 ms shorter, and the 300 mT/m system
reaches b ≈ 5600 with pulses of only 6 ms. Shorter pulses mean a shorter sequence, and the
last section of this chapter shows why that matters.

## Seeing the signal loss

The simulation below applies the phase argument of the first section to 20 000 molecules
whose displacements are drawn from the free-diffusion distribution of [Chapter 4](./04-diffusion-in-tissue.md),
with $D = 1 \times 10^{-3}$ mm²/s, typical of brain tissue. Each molecule's leftover phase
is proportional to its displacement, and the signal is the length of the average of all
the phase arrows. The top panel shows that this length matches the exponential formula.
The bottom row shows why, at three b-values: each dot is one molecule's leftover phase,
placed on a circle (300 molecules shown), and the arrow is the average of all 20 000. At
b = 250 the dots bunch near the starting direction (pointing right) and the arrow is long.
At b = 1000 most sit within a quarter turn either side of the start, with some straying
farther, and the arrow is shorter. At b = 3000 they wrap all the way around, pull in
opposite directions, and the arrow is short.

```{code-cell} python
:tags: [hide-input]
rng = np.random.default_rng(0)
D = 1.0e-3  # mm^2/s
Delta_s = 0.040
bvals = np.array([0, 250, 500, 1000, 2000, 3000], float)
b_phasor = [250, 1000, 3000]
measured, phases = [], {}
for b in bvals:
    q = np.sqrt(b / Delta_s) / (2 * np.pi)          # 1/mm, narrow-pulse relation b = (2 pi q)^2 Delta
    dx = rng.normal(scale=np.sqrt(2 * D * Delta_s), size=20_000)  # mm
    phases[b] = 2 * np.pi * q * dx                   # leftover phase of each molecule, rad
    measured.append(np.abs(np.mean(np.exp(1j * phases[b]))))

fig = plt.figure(figsize=(8, 6))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])
ax = fig.add_subplot(gs[0, :])
b_fine = np.linspace(0, 3000, 200)
ax.plot(b_fine, np.exp(-b_fine * D), color="0.5", ls="--", label="exp(−b D)")
ax.plot(bvals, measured, "o", color=PALETTE[0], label="20 000 simulated molecules")
for i, b in enumerate(b_phasor):
    m = measured[list(bvals).index(b)]
    ax.plot(b, m, "o", ms=11, mfc="none", color=PALETTE[1])
    ax.annotate(f"b = {b}", (b, m), (b + 90, m + 0.12), fontsize=8, color=PALETTE[1])
ax.set(xlabel="b (s/mm²)", ylabel="S / S₀", ylim=(0, 1.1), title=f"D = {D * 1e3:.1f} × 10⁻³ mm²/s")
ax.legend()
for i, b in enumerate(b_phasor):
    axp = fig.add_subplot(gs[1, i])
    ph = phases[b][:300]
    axp.add_patch(plt.Circle((0, 0), 1, fill=False, color="0.8"))
    axp.scatter(np.cos(ph), np.sin(ph), s=8, alpha=0.35, color=PALETTE[0])
    s = np.exp(1j * phases[b]).mean()
    axp.annotate("", xy=(s.real, s.imag), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=PALETTE[1], lw=2.2))
    axp.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), aspect="equal", title=f"b = {b}: sum length {abs(s):.2f}")
    axp.set_axis_off()
fig.tight_layout()
```

## Signal versus b for brain tissue

The exponential holds for a single freely diffusing pool. Tissue is a mixture of pools,
and its signal is the sum of their decays. Using the simulated brain's compartments ([Chapter 4](./04-diffusion-in-tissue.md)):

```{code-cell} python
:tags: [hide-input]
b = np.linspace(0, 4000, 200)
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.semilogy(b, signal.white_matter(b, 0.0), color=TISSUE_COLORS["WM"], label="WM, gradient across the fibers")
ax.semilogy(b, signal.white_matter(b, 1.0), color=TISSUE_COLORS["WM"], ls="--", label="WM, gradient along the fibers")
ax.semilogy(b, signal.gray_matter(b), color=TISSUE_COLORS["GM"], label="GM")
ax.semilogy(b, signal.csf(b), color=TISSUE_COLORS["CSF"], label="CSF")
for shell in [500, 1000, 2000, 3000]:
    ax.axvline(shell, color="0.6", lw=1, ls=":", zorder=0)
ax.set(xlabel="b (s/mm²)", ylabel="S / S₀ (log scale)", ylim=(1e-3, 1.1), title="dotted lines: the HBCD shells")
ax.legend(fontsize=8)
fig.tight_layout()
```

Several practical facts are visible here:

- **CSF is gone by b = 1000.** Its signal is below 5 %, so at higher b-values the CSF
  contribution is only noise. This is why high-b images look like white matter maps and
  why free-water contamination affects mainly the low-b data.
- **White matter is strongly anisotropic.** Along the fibers the signal at b = 3000 is
  below 1 %; across them it is about 60 %, because in this model the intra-axonal water
  does not move across the axon at all and its share of the signal never decays. The ratio
  between directions is the information tractography and tensor fitting use.
- **Mixtures do not give straight lines on a log scale.** A single compartment would. The
  across-fiber white matter curve flattens because one of its compartments does not decay;
  the gray matter curve bends slightly because of its slowly diffusing component. This
  curvature is what diffusion kurtosis and multi-compartment models ([Chapter 15](../04-modeling/15-signal-representations.md) and
  [Chapter 17](../04-modeling/17-microstructure-models.md)) measure, and it is only visible above about b = 1500, which is why those
  models need high b-values.
- **Signal at high b is small for most tissue.** At b = 3000 gray matter retains about
  15 % of $S_0$ and white matter along the fibers less than 1 %. If the b=0 image has
  SNR 30, those measurements have SNR 4 and below 1, the range where the noise floor of
  [Chapter 3](../01-mri-physics/03-reconstruction.md) biases the values. Only across-fiber white matter stays well above the floor.

## See it: the brain at three b-values

The same curves, applied voxel by voxel to the simulated brain's tissue maps, give the images
below: one slice at b = 0, 1000, and 3000, with the gradient along two different axes. Every
white matter voxel has a fiber direction, which in this synthetic slice runs parallel to the
nearest white matter boundary, a simple stand-in for real tract directions.

```{code-cell} python
:tags: [hide-input]
tissue = phantoms.brain_slice()
brain = tissue["mask"]
gradients = {"gradient left-right": (0.0, 1.0, 0.0), "gradient anterior-posterior": (1.0, 0.0, 0.0)}  # (row, column, slice) = (A-P, L-R, slice)
b_show = [0, 1000, 3000]
series = synth.synthetic_dwi(tissue, np.repeat(b_show, 2), np.tile(list(gradients.values()), (len(b_show), 1)))

fig, axes = plt.subplots(2, 3, figsize=(9, 6.4))
for col, b_val in enumerate(b_show):
    pair = series[..., 2 * col : 2 * col + 2]
    vmax = np.percentile(pair[brain], 99.5)  # one brightness scale per column, shared by both gradients
    for row, name in enumerate(gradients):
        title = "b = 0 (no gradient)" if b_val == 0 else f"b = {b_val}, {name}"
        show_image(axes[row, col], pair[..., row], title, vmin=0, vmax=vmax)
fig.tight_layout()
pure = {k.upper(): tissue[k] > 0.9 for k in ("wm", "gm", "csf")}  # voxels of nearly one tissue
for col, b_val in enumerate(b_show[1:], start=1):
    kept = {k: series[..., 2 * col][m].mean() / series[..., 0][m].mean() for k, m in pure.items()}
    print(f"b = {b_val}, gradient left-right: " + ", ".join(f"{k} keeps {v:.0%}" for k, v in kept.items()) + " of its b=0 signal")
```

Each column has its own brightness scale; at b = 3000 the image holds a small fraction of
the b=0 signal (printed above). Three changes happen at once:

- **The contrast inverts.** At b = 0 the ventricles are the brightest structure in the brain,
  because CSF has the longest T2 ([Chapter 1](../01-mri-physics/01-spins-and-signal.md)). By b = 1000 they are black.
- **The brightest voxels at high b are white matter whose fibers cross the gradient.** The
  water inside those axons barely moves across them and keeps its signal. Averaged over all
  of white matter, whatever the fiber direction, the signal falls about as fast as gray
  matter's (printed above); it is the direction dependence, not the average, that sets
  white matter apart.
- **The picture depends on the gradient direction.** Compare the two rows: a patch of white
  matter that is bright in one row is dark in the other, because its fibers cross one
  gradient and run along the other. No single diffusion-weighted image describes the tissue;
  the pattern across directions does, which is why a diffusion series contains many
  directions ([Chapter 6](./06-qspace-sampling.md)).

## The cost of a high b-value: echo time

The b-value grows with the square of the gradient amplitude and roughly the cube of the
pulse duration. On a given scanner the amplitude is fixed, so higher b-values require
longer pulses, which push the echo out and cost signal through T2 decay ([Chapter 1](../01-mri-physics/01-spins-and-signal.md)). The
function below computes the minimum echo time for a simplified sequence in which the
two pulses sit directly against the 180° pulse and the EPI readout needs 23 ms to reach
the center of k-space. The echo time is defined by that moment, because the center of
k-space holds the image's overall brightness and contrast ([Chapter 2](../01-mri-physics/02-spatial-encoding-kspace.md)).
The 23 ms is that of the HBCD protocol {cite:p}`dean2024`, which uses 6/8 partial Fourier: it skips a quarter
of the k-space lines, all from the side read first, so the readout reaches the center
sooner. The right panel shows how much white matter signal is left at each echo time, as a
fraction of what it would have with no T2 decay at all.

```{code-cell} python
:tags: [hide-input]
b = np.linspace(200, 5000, 100)
t2_wm = presets.T2_MS["adult"]["WM"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
for (name, g), color in zip(presets.GMAX_MT_PER_M.items(), PALETTE):
    te = np.array([signal.min_te(bb, g)["te"] for bb in b])
    ax1.plot(b, te, color=color, label=f"{name} mT/m")
    ax2.plot(b, np.exp(-te / t2_wm), color=color, label=f"{name} mT/m")
ax1.set(xlabel="b (s/mm²)", ylabel="minimum TE (ms)", title="echo time needed to reach b")
ax2.set(xlabel="b (s/mm²)", ylabel="WM signal at that TE (T2 = 68 ms)", title="signal remaining before diffusion weighting")
ax1.legend(); ax2.legend()
fig.tight_layout()
for name, g in presets.GMAX_MT_PER_M.items():
    r = signal.min_te(3000, g)
    print(f"{name:>14}: b = 3000 needs delta = {r['delta']:4.1f} ms, TE >= {r['te']:5.1f} ms, "
          f"WM b=0 signal {np.exp(-r['te'] / t2_wm):.2f} of its no-decay value")
```

The difference between a 40 mT/m clinical system and an 80 mT/m research system is about
25 ms of echo time at b = 3000 (115 versus 90 ms), so the clinical system keeps about 30 %
less white matter signal (0.19 versus 0.27) before any diffusion weighting. A 300 mT/m
system gains a similar amount of echo time again. This is the reason gradient strength is the headline
specification of a diffusion scanner, and why high-b protocols on 80 mT/m hardware use
echo times near 90 ms, the value the HBCD protocol and the simulated brain use.

## Two side effects of strong gradients

Rapidly switched, strong gradients have consequences beyond diffusion weighting. Both are
treated in Part III; they are named here because they originate in the encoding.

- **Eddy currents.** The changing gradient field induces currents in the scanner's
  conductive structures, which produce a slowly decaying field of their own. That field is
  still present during the EPI readout and distorts each diffusion-weighted image in a way
  that depends on the gradient direction and strength ([Chapter 11](../03-preprocessing/11-eddy-currents.md)). A twice-refocused spin
  echo, with two 180° pulses and four gradient lobes, cancels much of it at the cost of a
  longer echo time {cite:p}`reese2003`.
- **Motion sensitivity.** The phase imparted by the pulses is proportional to displacement,
  so bulk motion of the head or pulsation of the brain during the 40 ms between the pulses
  produces a large, spatially varying phase. This is the reason single-shot readouts are
  used ([Chapter 2](../01-mri-physics/02-spatial-encoding-kspace.md)) and a cause of signal dropout in individual slices ([Chapter 12](../03-preprocessing/12-motion-and-dropout.md)).

The pulse pair described here weights each image along one direction. Newer acquisitions
shape the gradients to weight several directions within one image (b-tensor encoding);
[Chapter 22](../05-advanced/22-frontiers.md) describes what that adds.

## The reference scheme

The book's reference scheme is the HBCD scheme bundled with the simulation inputs, which is
also TRXScan's default. It has 75 volumes per phase-encode direction, grouped into shells
(sets of volumes that share a b-value but differ in gradient direction). The plot shows
every volume in the order it is stored in the bundled file, at the height of its b-value;
the b=0 volumes are the black diamonds along the bottom.

```{code-cell} python
:tags: [hide-input]
bvals, bvecs = schemes.hbcd()
shells = schemes.shells_of(bvals)
shell_of = np.round(bvals / 50) * 50
fig, ax = plt.subplots(figsize=(9, 2.8))
idx = np.arange(len(bvals))
for b_shell, color in zip(shells, ["k"] + PALETTE):
    sel = shell_of == b_shell
    label = f"b = 0 ({shells[b_shell]} volumes)" if b_shell == 0 else f"b = {b_shell:.0f} ({shells[b_shell]})"
    ax.vlines(idx[sel], 0, bvals[sel], color=color, lw=1, alpha=0.35)
    ax.plot(idx[sel], bvals[sel], "D" if b_shell == 0 else "o", color=color, ms=5, label=label)
ax.set(xlabel="volume, in the order of the bundled file (sorted by b; not HBCD's acquisition order)", ylabel="b (s/mm²)", xlim=(-1, len(bvals)), ylim=(-150, 3300))
ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.28), frameon=False)
fig.tight_layout()
```

This bundled copy is sorted for teaching: the shells come in blocks of increasing b-value,
with a b=0 volume about every eight volumes, so each shell is easy to pick out. It is
**not** the order in which HBCD acquires its data. The acquired HBCD protocol, which the
book's live tier simulates ([Chapter 0.2](../00-frontmatter/the-simulated-datasets.md)),
has 76 volumes: it opens with six b=0 volumes and then interleaves all four shells through
the rest of the series. The difference matters. If the shells were acquired in blocks and
the subject moved late in the scan, the damage would fall mostly on the b = 3000 shell;
interleaved, it is shared among all shells. Likewise, b=0 volumes spread through a
series, as in the bundled copy, give fresh reference images for tracking slow signal drift
and head motion across the whole scan. [Chapter 6](./06-qspace-sampling.md) covers the design of schemes like this one.

## What this implies for acquisition

- **b-value is the primary contrast setting.** Around 1000 s/mm² for tensor imaging and
  clinical ADC; 2000–3000 for fiber-orientation and multi-compartment models; above that only
  on strong gradient systems.
- **Every increase in b costs echo time and therefore signal**, through longer gradient
  pulses. Ask for the gradient amplitude of the scanner before designing a high-b protocol.
- **High-b images are low-SNR images.** Budget the number of averages or directions with
  the expected SNR at the highest shell in mind, not the SNR of the b=0 image.
- **Spread b=0 volumes through the acquisition**, and consider interleaving the shells so
  that motion late in the scan does not concentrate in one shell.
- **The diffusion time is a consequence of the timing** ($\Delta$, typically 30–50 ms) and
  is rarely reported; record it, because measured diffusivities depend on it.

## Further reading

The pulsed-gradient spin echo {cite:p}`stejskal1965`, the first applications to the brain
{cite:p}`lebihan1986`, the twice-refocused variant {cite:p}`reese2003`, and the overview in
{cite:t}`jones2010`.
