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

from dwibook import presets, schemes, signal
from dwibook.plotting import PALETTE, TISSUE_COLORS, set_style

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
t1 = (TE / 2) - 3 - delta  # first pulse ends 3 ms before the 180°
t2 = t1 + Delta            # second pulse starts Delta after the first

rf = np.zeros_like(t); rf[(t > 0) & (t < 2)] = 1.0; rf[(t > TE / 2 - 1) & (t < TE / 2 + 1)] = 2.0
diff = np.zeros_like(t); diff[(t > t1) & (t < t1 + delta)] = 1.0; diff[(t > t2) & (t < t2 + delta)] = 1.0
epi = np.where((t > t2 + delta + 2) & (t < 100), np.sign(np.sin(2 * np.pi * (t - TE) / 1.4)), 0.0)  # readout after the second pulse; TE at its k-space center

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

## The b-value

Three settings determine how sensitive the measurement is: the gradient amplitude $G$, the
duration of each pulse $\delta$, and the separation between the pulses $\Delta$. They are
combined into one number, the b-value,

$$b = \gamma^2 G^2 \delta^2 \left(\Delta - \tfrac{\delta}{3}\right),$$

in units of s/mm². For water diffusing freely with coefficient $D$, the signal falls
exponentially with $b$:

$$S(b) = S_0\, e^{-b D}.$$

$S_0$ is the signal without diffusion weighting (the b=0 image of [Chapter 1](../01-mri-physics/01-spins-and-signal.md)). With $D$ in
mm²/s and $b$ in s/mm², the product $bD$ is dimensionless; at $b = 1000$ s/mm² free water
($D = 3 \times 10^{-3}$) retains $e^{-3} \approx 5$ % of its signal and white matter across the
fibers ($D \approx 0.6 \times 10^{-3}$) retains about 55 %. The separation $\Delta$ is
approximately the diffusion time of [Chapter 4](./04-diffusion-in-tissue.md).

```{code-cell} python
:tags: [hide-input]
for g, d, D in [(40, 20, 40), (80, 14, 34), (300, 6, 26)]:
    print(f"G = {g:>3} mT/m, delta = {d:>2} ms, Delta = {D:>2} ms  ->  b = {signal.b_value(g, d, D):6.0f} s/mm^2")
```

## Seeing the signal loss

The simulation below applies the phase argument of the first section to 20 000 molecules
whose displacements are drawn from the free-diffusion distribution of [Chapter 4](./04-diffusion-in-tissue.md). The
magnitude of the summed signal matches the exponential formula.

```{code-cell} python
:tags: [hide-input]
rng = np.random.default_rng(0)
D = 1.0e-3  # mm^2/s
Delta_s = 0.040
bvals = np.array([0, 250, 500, 1000, 2000, 3000], float)
measured = []
for b in bvals:
    q = np.sqrt(b / Delta_s) / (2 * np.pi)          # 1/mm, narrow-pulse relation b = (2 pi q)^2 Delta
    dx = rng.normal(scale=np.sqrt(2 * D * Delta_s), size=20_000)  # mm
    measured.append(np.abs(np.mean(np.exp(2j * np.pi * q * dx))))
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.plot(bvals, np.exp(-bvals * D), color="0.5", ls="--", label="exp(−b D)")
ax.plot(bvals, measured, "o", label="20 000 simulated molecules")
ax.set(xlabel="b (s/mm²)", ylabel="S / S₀", title=f"D = {D * 1e3:.1f} × 10⁻³ mm²/s")
ax.legend()
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

## The cost of a high b-value: echo time

The b-value grows with the square of the gradient amplitude and roughly the cube of the
pulse duration. On a given scanner the amplitude is fixed, so higher b-values require
longer pulses, which push the echo out and cost signal through T2 decay ([Chapter 1](../01-mri-physics/01-spins-and-signal.md)). The
function below computes the minimum echo time for a simplified sequence in which the
two pulses sit directly against the 180° pulse and the EPI readout needs 23 ms to reach
the center of k-space (a 6/8 partial Fourier readout of the HBCD protocol, [Chapter 2](../01-mri-physics/02-spatial-encoding-kspace.md)).

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
          f"WM b=0 signal {np.exp(-r['te'] / t2_wm):.2f} of proton density")
```

The difference between a 40 mT/m clinical system and an 80 mT/m research system is about
25 ms of echo time at b = 3000, which is 30 % of the white matter signal. A 300 mT/m system
gains a similar amount again. This is the reason gradient strength is the headline
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

The pulse pair described here applies weighting along one direction per image. Newer
acquisitions vary the gradient direction within one encoding to probe several directions
at once (b-tensor encoding), which separates microscopic anisotropy from orientation
dispersion; [Chapter 15](../04-modeling/15-signal-representations.md) and [Chapter 23](../05-advanced/23-frontiers.md) describe what that adds.

## The reference scheme

TRXScan's default protocol is the HBCD scheme bundled with the simulation inputs. It has 75 volumes
per phase-encode direction:

```{code-cell} python
:tags: [hide-input]
bvals, bvecs = schemes.hbcd()
for b, n in schemes.shells_of(bvals).items():
    print(f"b = {b:5.0f}: {n:2d} volumes")
print(f"acquisition order of the first 20 volumes: {bvals[:20].astype(int).tolist()}")
```

The b=0 volumes are spread through the acquisition rather than collected at the start, and
the shells are interleaved. Both choices make the scheme robust to motion and drift:
if the subject moves halfway through, every shell is affected equally rather than one shell
being lost. [Chapter 6](./06-qspace-sampling.md) covers the design of schemes like this one.

## What this implies for acquisition

- **b-value is the primary contrast setting.** Around 1000 s/mm² for tensor imaging and
  clinical ADC; 2000–3000 for fiber-orientation and multi-compartment models; above that only
  on strong gradient systems.
- **Every increase in b costs echo time and therefore signal**, through longer gradient
  pulses. Ask for the gradient amplitude of the scanner before designing a high-b protocol.
- **High-b images are low-SNR images.** Budget the number of averages or directions with
  the expected SNR at the highest shell in mind, not the SNR of the b=0 image.
- **Interleave shells and spread b=0 volumes** through the acquisition.
- **The diffusion time is a consequence of the timing** ($\Delta$, typically 30–50 ms) and
  is rarely reported; record it, because measured diffusivities depend on it.

## Further reading

The pulsed-gradient spin echo {cite:p}`stejskal1965`, the first applications to the brain
{cite:p}`lebihan1986`, the twice-refocused variant {cite:p}`reese2003`, and the overview in
{cite:t}`jones2010`.
