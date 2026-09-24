"""Tissue parameters used across the book.

The T2 values and diffusivities are TRXScan's compartment presets (``CompartmentParams`` in
``src/compartments.rs``), so numbers the reader computes here match what the simulator does to
the phantom. TRXScan does not model T1 (TR saturation is a signal scale, not a relaxation), so
the T1 values are 3 T literature figures and are used only in the toy Bloch simulations.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tissue:
    name: str
    t1_ms: float
    t2_ms: float
    md: float  # mean diffusivity, mm^2/s (isotropic compartments) or the extra-axonal MD for WM


#: Compartment T2 (ms) per TRXScan preset; keys are the ``--params`` names.
T2_MS: dict[str, dict[str, float]] = {
    "neonatal": {"WM": 180.0, "GM": 220.0, "CSF": 2500.0},
    "adult": {"WM": 68.0, "GM": 76.0, "CSF": 2000.0},
    "infant": {"WM": 180.0, "GM": 220.0, "CSF": 2500.0},
}

#: 3 T literature T1 (ms). Not part of TRXScan; used by the toy Bloch simulations only.
T1_MS: dict[str, float] = {"WM": 830.0, "GM": 1330.0, "CSF": 4000.0}

#: Diffusivities (mm^2/s) of the ``adult`` preset: WM intra-axonal stick and extra-axonal
#: tensor eigenvalues, GM ball, CSF ball.
ADULT_DIFFUSIVITY = {
    "WM_intra": 0.0017,
    "WM_extra": (0.0017, 0.0006, 0.0006),
    "GM": 0.00085,
    "CSF": 0.003,
}

#: Compartment fractions of the ``adult`` preset: intra-axonal share of the WM fiber
#: compartment, restricted (soma) share of GM, and the soma ball diffusivity (mm^2/s).
ADULT_FRACTIONS = {"WM_intra": 0.55, "GM_restricted": 0.20, "d_soma": 0.0003}

#: Echo time (ms) of the HBCD-like protocol TRXScan simulates by default.
TE_HBCD_MS = 88.0

#: Total EPI readout duration (ms) of the HBCD-like protocol, pinned by TRXScan for any matrix.
READOUT_HBCD_MS = 91.7

#: Maximum gradient amplitude (mT/m) of representative gradient systems.
GMAX_MT_PER_M = {"clinical 40": 40.0, "whole-body 80": 80.0, "Connectom 300": 300.0}

#: Gyromagnetic ratio of 1H, MHz/T (gamma / 2 pi).
GAMMA_BAR_MHZ_PER_T = 42.577

#: Field strengths the book refers to, tesla.
B0_T = {"1.5T": 1.5, "3T": 3.0, "7T": 7.0}


def tissues(preset: str = "adult") -> dict[str, Tissue]:
    """WM/GM/CSF as :class:`Tissue` records for a TRXScan preset (T1 from the literature)."""
    md = {"WM": sum(ADULT_DIFFUSIVITY["WM_extra"]) / 3, "GM": ADULT_DIFFUSIVITY["GM"], "CSF": ADULT_DIFFUSIVITY["CSF"]}
    return {n: Tissue(n, T1_MS[n], T2_MS[preset][n], md[n]) for n in ("WM", "GM", "CSF")}
