---
title: Diffusion MRI, executed
subtitle: An executable book on diffusion-weighted MRI, from spins to tractography
kernelspec:
  name: python3
  display_name: Python 3
---

Every figure in this book is produced by code you can run, and every model fit or artifact
correction is scored against a known answer. The answer exists because the brain in these
pages is simulated: a tractogram and tissue maps pass through **TRXScan**, a headless
diffusion-MRI simulator that models the scanner from the diffusion signal through k-space
to the reconstructed complex image, and the same phantom yields analytic ground-truth maps.

:::{admonition} Status
:class: warning
This is a skeleton. Chapters exist as stubs with their learning goals and structure; content
lands chapter by chapter. See the [implementation plan](https://github.com/PennLINC/diffusion-book/blob/main/docs/implementation-plan.md).
:::

## What is in the book

| Part | Chapters | What you will be able to do |
|---|---|---|
| Front matter | 0.1–0.3 | run the book, know the phantom, read the notation |
| I. MRI physics | 1–3 | simulate spins, encode an image in k-space, reconstruct it with multiple coils, parallel imaging and partial Fourier |
| II. Diffusion physics and encoding | 4–7 | derive the diffusion signal, design a q-space scheme (single-shell, multi-shell, DSI, CS-DSI), choose TE/TR/voxel/b |
| III. Artifacts and preprocessing | 8–13 | recognise noise, Gibbs ringing, distortion, eddy currents, motion and dropout in simulated data, correct them, and measure what remains |
| IV. Reconstruction and modeling | 14–18 | fit DTI/DKI/MAP-MRI, estimate fiber orientations, fit microstructure models, run tractography, and decide what a given acquisition allows |
| V. Advanced acquisitions | 19–22 | reason about multi-TE, multi-echo and multi-diffusion-time DWI, and where the field is heading |

## Two kinds of simulation

Small **toy** simulations (a Bloch equation, a Shepp–Logan k-space, a random walk) are written
in the notebook and run when the book builds. Brain-level **phantom** data come from TRXScan
runs made offline and versioned; notebooks download them. The phantom chapter explains both.

```{code-cell} python
:tags: [hide-input]
import dwibook
from dwibook import data

print(f"dwibook {dwibook.__version__}; registered datasets: {data.registered_datasets() or 'none yet'}")
```
