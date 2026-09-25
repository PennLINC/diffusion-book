"""Render the simulator commands behind every dataset (Appendix A).

The pipeline configuration ``pipelines/config/datasets.yaml`` is the single description of
what the offline pipeline runs. This module expands each dataset entry into the ``trxscan``
and ``trxscan-microstructure`` command lines it implies, so that the book's cookbook is
generated from the same file the pipeline executes rather than maintained by hand. The
expansion mirrors the pipeline driver's rules: defaults, per-dataset overrides, named
variants, parameter sweeps, per-scheme runs, reverse-polarity pairs, and truth maps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

#: Flags used by the book's datasets, with the artifact or feature each controls.
FLAG_GLOSSARY: dict[str, str] = {
    "--oversample": "simulate the object on a finer grid than the acquisition, so Gibbs ringing is intrinsic (Chapter 9)",
    "--noise": "complex Gaussian noise variance in k-space; Rician magnitude, phase noise (Chapter 8)",
    "--noise-map": "spatially varying noise level; writes the true sigma map (Chapter 8)",
    "--coils": "number of ring-arranged receive coils, combined with Roemer weights (Chapter 3)",
    "--accel": "GRAPPA in-plane acceleration factor (Chapter 3)",
    "--reverse-pe": "flip the phase-encode polarity: the second image of a blip-up/blip-down pair (Chapter 10)",
    "--gre-out": "also write a synthetic dual-echo GRE fieldmap from the same field (Chapter 10)",
    "--gre-res": "resolution of that fieldmap, with intravoxel dephasing (Chapter 10)",
    "--eddy": "linear eddy-current field proportional to the diffusion gradient (Chapter 11)",
    "--eddy-quad": "quadratic eddy-current term (Chapter 11)",
    "--eddy-trace": "replay a measured per-volume eddy field from a confounds TSV (Chapter 11)",
    "--eddy-phase": "eddy-current phase ramp in the complex output (Chapter 11)",
    "--motion": "per-volume head poses from a confounds TSV; faithful re-simulation (Chapter 12)",
    "--mb": "multiband factor (Chapter 12)",
    "--dropout-rate": "probability per diffusion-weighted volume of a within-volume dropout event; writes the truth TSV (Chapter 12)",
    "--gnl": "gradient nonlinearity preset or coefficient file; writes truth coefficient, displacement, and graddev (Chapter 13)",
    "--gnl-scale": "severity of the nonlinear terms (Chapter 13)",
    "--gnl-no-warp": "encoding deviation only, no spatial warp (Chapter 13)",
    "--gnl-no-encoding": "spatial warp only, no encoding deviation (Chapter 13)",
    "--isocenter": "scanner isocenter in world mm (Chapter 13)",
    "--te": "echo time in ms; planned flag, implementation plan item T1 (Chapters 7, 20)",
    "--window": "k-space apodization at reconstruction; planned flag, item T1 (Chapter 9)",
    "--export-kspace": "write raw per-coil k-space; planned flag, item T2 (Chapters 2, 3)",
    "--subsample": "keep N streamlines sampled by SIFT2 weight; identical subset in the truth run",
    "--seed": "noise, dropout, and subsample realization",
    "--params": "compartment preset: adult, neonatal, infant (Chapter 1)",
    "--weights": "SIFT2 streamline weights from the TRX data-per-streamline array",
    "--truth-peaks": "write up to three ground-truth fiber orientations per voxel (Chapter 16)",
    "--big-delta / --small-delta": "diffusion timing for MAP-MRI truth maps in physical units (Chapter 15)",
}


def default_config_path() -> Path:
    """The repository's ``pipelines/config/datasets.yaml`` (the package sits next to it)."""
    return Path(__file__).resolve().parents[1] / "pipelines" / "config" / "datasets.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    import yaml

    p = Path(path) if path else default_config_path()
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _phantoms_of(ds: dict[str, Any]) -> list[str]:
    p = ds["phantom"]
    return [p] if isinstance(p, str) else list(p)


def _base_flags(cfg: dict[str, Any], ds: dict[str, Any], phantom: str, voxel: float, oversample: int) -> list[str]:
    d = cfg["defaults"]
    grid = f"work/{phantom}/{voxel:g}mm"
    flags = [
        f"--wm {grid}/wm.nii.gz", f"--gm {grid}/gm.nii.gz", f"--csf {grid}/csf.nii.gz", f"--mask {grid}/mask.nii.gz",
        f"--streamlines {cfg['phantoms'][phantom]['tract'].split('/')[-1]}",
        f"--weights {cfg['phantoms'][phantom]['weights']}",
        f"--subsample {ds.get('subsample', d['subsample'])}", f"--seed {ds.get('seed', d['seed'])}",
        f"--params {ds.get('params', d['params'])}",
    ]
    if oversample > 1:
        flags += [f"--oversample {oversample}"] + [f"--sim-{k} {grid}/sim/{k}.nii.gz" for k in ("wm", "gm", "csf", "mask", "fmap")]
    else:
        flags += ["--oversample 1", f"--fmap {grid}/fmap.nii.gz"]
    return flags


def _scheme_flags(scheme: str) -> list[str]:
    return [f"--bval schemes/{scheme}.bval", f"--bvec schemes/{scheme}.bvec"]


def _fill(flags: list[str], phantom_cfg: dict[str, Any], out: str) -> list[str]:
    return [f.replace("{out}", out).replace("{phantom.motion_ap}", phantom_cfg.get("motion_ap", "<motion_ap.tsv>")) for f in flags]


def render_commands(cfg: dict[str, Any], dataset_id: str) -> list[str]:
    """The command lines for one dataset id, in the order the pipeline runs them."""
    d = cfg["defaults"]
    ds = cfg["datasets"][dataset_id]
    cmds: list[str] = []
    for phantom in _phantoms_of(ds):
        pcfg = cfg["phantoms"][phantom]
        if ds.get("truth_only"):
            grid = f"work/{phantom}/{ds.get('voxel_mm', d['voxel_mm']):g}mm"
            cmds.append(" ".join([
                "trxscan-microstructure", f"--wm {grid}/wm.nii.gz", f"--gm {grid}/gm.nii.gz", f"--csf {grid}/csf.nii.gz", f"--mask {grid}/mask.nii.gz",
                f"--streamlines {pcfg['tract'].split('/')[-1]}", f"--weights {pcfg['weights']}",
                f"--subsample {d['subsample']}", f"--seed {d['seed']}", f"--params {d['params']}",
                "--big-delta 0.030 --small-delta 0.010", f"--out data/{dataset_id}/{phantom}",
            ]))
            continue
        # runs: the cross product of schemes x variants x sweep values (each optional)
        schemes = ds.get("schemes", [ds.get("scheme", d["scheme"])])
        variants = ds.get("variants", [{"name": None, "extra_flags": []}])
        sweep = ds.get("sweep")
        sweep_values = sweep["values"] if sweep else [None]
        for scheme in schemes:
            for var in variants:
                for val in sweep_values:
                    voxel = ds.get("voxel_mm", d["voxel_mm"])
                    if sweep and sweep.get("key") == "voxel_mm":
                        voxel = float(val)
                    oversample = var.get("oversample", ds.get("oversample", d["oversample"]))
                    name_parts = [dataset_id, phantom] + ([scheme] if len(schemes) > 1 else []) + ([var["name"]] if var.get("name") else []) + ([f"{sweep.get('flag', sweep.get('key')).lstrip('-')}-{val}"] if sweep else [])
                    out = "data/" + dataset_id + "/" + "_".join(name_parts)
                    flags = _base_flags(cfg, ds, phantom, voxel, oversample) + _scheme_flags(scheme)
                    flags += _fill(list(ds.get("extra_flags", d["extra_flags"])), pcfg, out)
                    flags += _fill(list(var.get("extra_flags", [])), pcfg, out)
                    if sweep and "flag" in sweep:
                        flags += [f"{sweep['flag']} {val}"]
                    if ds.get("truth_peaks"):
                        flags += ["--truth-peaks"]
                    cmds.append(" ".join(["trxscan"] + flags + [f"--out {out}"]))
                    if ds.get("reverse_pe_pair"):
                        cmds.append(" ".join(["trxscan"] + flags + ["--reverse-pe", f"--out {out}_dir-PA"]))
        for extra in ds.get("extra_runs", []):
            voxel = ds.get("voxel_mm", d["voxel_mm"])
            out = f"data/{dataset_id}/{dataset_id}_{phantom}_{extra['name']}"
            flags = _base_flags(cfg, ds, phantom, voxel, ds.get("oversample", d["oversample"])) + _scheme_flags(ds.get("scheme", d["scheme"]))
            flags += _fill(list(extra.get("extra_flags", [])), pcfg, out)
            cmds.append(" ".join(["trxscan"] + flags + [f"--out {out}"]))
        if ds.get("truth"):
            cmds.append(f"trxscan-microstructure ... --subsample {d['subsample']} --seed {d['seed']} --out data/{dataset_id}/{phantom}_truth  # identical subset to the trxscan runs above")
    return cmds


def flags_used(cfg: dict[str, Any]) -> list[str]:
    """Every flag that appears in any dataset's command lines, sorted."""
    seen: set[str] = set()
    for ds_id in cfg["datasets"]:
        for cmd in render_commands(cfg, ds_id):
            for tok in cmd.split():
                if tok.startswith("--"):
                    seen.add(tok)
    return sorted(seen)
