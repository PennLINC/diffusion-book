"""Helper package for the diffusion MRI executable book.

Modules
-------
data      : fetch and load the pre-simulated datasets (pooch registry, local override)
phantoms  : toy phantoms and simulators used at build time (Shepp-Logan, Bloch, random walks)
kspace    : centered FFTs, sampling masks, EPI helpers for the k-space chapters
schemes   : gradient-scheme generators and sphere / q-space plots
truth     : the TRXScan ground-truth maps and comparison metrics
plotting  : house style for slice mosaics, error maps, fit-vs-truth panels
"""

__version__ = "0.1.0.dev0"
