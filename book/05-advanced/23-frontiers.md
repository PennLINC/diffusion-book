---
title: Frontiers
subtitle: Chapter 23
kernelspec:
  name: python3
  display_name: Python 3
---

## Learning goals

After this chapter you can:

- describe what b-tensor encoding, diffusion-relaxation correlation, high-gradient
  systems, non-EPI readouts, and learned reconstruction each add to the acquisition or
  analysis chain
- explain, with a simulation, why b-tensor encoding separates microscopic anisotropy
  from orientation dispersion
- state the role of simulation in validating the methods of this book

**Datasets used:** none (toy tier)
**Simulation tier:** toy

```{code-cell} python
:tags: [hide-cell]
import numpy as np
import matplotlib.pyplot as plt

from dwibook import presets, schemes, signal
from dwibook.plotting import PALETTE, set_style

set_style()
```

## b-tensor encoding

Every acquisition in this book applies its diffusion weighting along one direction per
measurement. The pulse pair of Chapter 5 can be replaced by a gradient waveform that
changes direction during the encoding, so that a single measurement weights diffusion
along several axes at once. The weighting is then described by a tensor rather than a
vector: linear encoding (one axis) is the conventional case, planar encoding weights two
axes, and spherical encoding weights all three equally {cite:p}`westin2016`.

Spherical encoding has a property that no linear acquisition has: its signal depends only
on the trace of each compartment's diffusion tensor, not on its shape or orientation.
Combined with linear encoding at the same b-value, it separates two things that
conventional data confound. A voxel of coherent fibers and a voxel of dispersed or crossing
fibers with the same microscopic anisotropy give the same spherical-encoding signal;
their linear-encoding signals, averaged over directions, differ from the spherical one by
an amount that depends only on the microscopic anisotropy {cite:p}`lasic2014`. The tensor's
FA (Chapter 15) falls in the dispersed voxel; the microscopic anisotropy does not.

```{code-cell} python
:tags: [hide-input]
b = np.linspace(0, 3000, 100)
dirs = schemes.electrostatic_directions(60)
d_par, d_perp, _ = presets.ADULT_DIFFUSIVITY["WM_extra"]
d_intra = presets.ADULT_DIFFUSIVITY["WM_intra"]
f = presets.ADULT_FRACTIONS["WM_intra"]
# a WM voxel = the phantom's stick + tensor; its trace does not depend on how the fibers are arranged
trace_mean = f * d_intra / 3 + (1 - f) * (d_par + 2 * d_perp) / 3
spherical = np.exp(-b * trace_mean)

def powder_linear(fibers):
    """direction-averaged linear-encoding signal of a voxel with the given fiber orientations, equal fractions"""
    out = np.zeros_like(b)
    for fib in fibers:
        cos = dirs @ fib
        out += np.mean(signal.white_matter(b[:, None], cos[None, :]), axis=1) / len(fibers)
    return out

coherent = powder_linear([np.array([0, 0, 1.0])])
crossing = powder_linear([np.array([0, 0, 1.0]), np.array([0, 1.0, 0])])
rng = np.random.default_rng(0)
dispersed = powder_linear([v / np.linalg.norm(v) for v in rng.normal(size=(40, 3))])

fig, ax = plt.subplots(figsize=(7, 3.4))
ax.semilogy(b, spherical, color="0.3", lw=2, label="spherical encoding: any fiber arrangement")
ax.semilogy(b, coherent, color=PALETTE[0], label="linear, direction-averaged: coherent fibers")
ax.semilogy(b, crossing, color=PALETTE[1], ls="--", label="linear, direction-averaged: 90° crossing")
ax.semilogy(b, dispersed, color=PALETTE[2], ls=":", label="linear, direction-averaged: fully dispersed")
ax.set(xlabel="b (s/mm²)", ylabel="signal / S₀", ylim=(0.05, 1.05), title="the same white matter voxel, three fiber arrangements")
ax.legend(fontsize=8)
fig.tight_layout()
```

The three direction-averaged linear curves coincide, because the direction average removes
the arrangement, and all three lie above the spherical curve by the same margin, because
the margin is set by the microscopic anisotropy alone. A tensor fit to any of the three
voxels would give FA of 0.85, 0.4, and near zero; b-tensor encoding gives the same
microscopic anisotropy for all three. Diffusion time (Chapter 22) and echo time (Chapter 20)
are the other dimensions being added to the encoding, and acquisitions that vary several at
once, multidimensional diffusion MRI, are the current frontier of the field. The phantom's
truth includes the b-tensor quantities (microscopic FA, isotropic and anisotropic kurtosis);
simulating the acquisition (implementation plan item T7) would close that loop.

## Diffusion relaxometry

Chapters 20 and 21 varied the echo time and added echoes. The general form is a joint
acquisition over b-value, direction, echo time, inversion time, and diffusion time, from
which compartments are separated by every property at once. Such data support
model-free analyses, correlation spectra of diffusivity against T2 or T1, that need no
assumption about the number of compartments. They cost scan time in proportion to the
number of dimensions sampled, and their design is an open problem.

## High-gradient systems

Chapter 5 gave the echo-time cost of a b-value and Chapter 22 the q needed for axon
diameters; both are set by gradient amplitude. Systems at 200–300 mT/m (the Connectom class)
and head-only gradient inserts beyond that reach b = 10 000 at echo times that whole-body
systems need for b = 3000, and reach the q where small axons begin to be distinguishable.
Their nonlinearity is larger (Chapter 13), their peripheral nerve stimulation limits are
reached sooner, and they exist in a handful of sites. The methods developed on them
(diameter mapping, high-b compartment models) set expectations that standard hardware cannot
meet, which is a recurring source of over-interpretation.

## Beyond single-shot EPI

Chapter 2 explained why diffusion uses single-shot EPI and what it costs. Multi-shot EPI
with navigator or self-navigated phase correction, spiral readouts, and reduced-FOV
acquisitions each trade the robustness of single-shot for resolution or reduced distortion;
they are established for the spinal cord and optic nerve and increasingly used for
sub-millimeter brain imaging. Simultaneous multi-slice (Chapter 7) is now standard.
Three-dimensional readouts, which sample k-space in segments over several excitations,
remove the slice profile penalty at the cost of the same phase-consistency problem in
three dimensions.

## Learned reconstruction and denoising

Neural networks trained on paired data now reconstruct images from undersampled k-space
(extending compressed sensing, Chapter 3), denoise diffusion series (extending MP-PCA,
Chapter 8), and predict full-quality microstructure maps from short protocols. The gains
are real and the caveats are the same as for every learned method: the output is only as
general as the training data, and errors are plausible-looking rather than noisy, which
makes them harder to detect than the artifacts of Part III. Simulation with a known ground
truth is the natural test, which is the last topic.

## Simulation as validation

Every chapter of this book measured a method against a known answer, and the reason that
was possible is that the data were simulated. Real data have no ground truth; validation
against histology is possible for a few quantities in a few samples, and validation against
other MRI methods only tests agreement. A simulator that models the acquisition from the
diffusion signal through k-space to the reconstructed image, and that writes the analytic
answer for the same phantom, allows a method to be scored on the quantity it claims to
measure, under the artifacts it will meet, at the acquisition parameters of a specific
protocol. The toy tier of this book did this with synthetic tissue on a single slice or a
small volume; the phantom tier, when the pipeline delivers it, will do it with a full
tractogram, a realistic acquisition, and twenty-seven truth maps. The limits of the
simulator are the limits of the validation, and this book has stated them where they
apply: Gaussian compartments, no exchange, no diameters, a single diffusion time, one echo
per readout. Each is a direction the simulator can be extended, and each extension makes a
further chapter of this book testable.

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
