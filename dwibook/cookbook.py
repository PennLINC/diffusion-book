"""Render the simulator commands and file layout behind every dataset (Appendix A).

The pipeline configuration ``pipelines/config/datasets.yaml`` is the single description of
what the offline pipeline runs. This module expands each dataset entry into the ``trxscan``
and ``trxscan-microstructure`` command lines it implies and into the BIDS layout the
dataset directory has once the pipeline has run, so that the book's cookbook and manifest
are generated from the same file the pipeline executes rather than maintained by hand. The
expansion mirrors the pipeline driver's rules: defaults, per-dataset overrides, named
variants, parameter sweeps, per-scheme runs, reverse-polarity pairs, and truth maps.

Layout. Every dataset is a BIDS raw dataset: ``sub-<anatomy>/dwi/`` holds one complex
diffusion series per run, named by the run's ``acq-`` label (the scheme, variant, or sweep
point) and its phase-encode ``dir-``; ``sub-<anatomy>/fmap/`` holds the synthetic
gradient-echo fieldmap when a run writes one. TRXScan writes its ground-truth outputs next
to the series, and the driver moves them into the ``derivatives/trxscan`` dataset; the
precomputed corrections form one derivative dataset per tool. The ``truth`` dataset holds
only truth maps and is itself a derivative-type dataset.
"""

from __future__ import annotations

import re
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

#: The derivative dataset that holds everything TRXScan writes beyond the raw series.
TRUTH_PIPELINE = "derivatives/trxscan"


def default_config_path() -> Path:
    """The repository's ``pipelines/config/datasets.yaml`` (the package sits next to it)."""
    return Path(__file__).resolve().parents[1] / "pipelines" / "config" / "datasets.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    import yaml

    p = Path(path) if path else default_config_path()
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ----------------------------------------------------------------------------- runs

def bids_label(text: Any) -> str:
    """A BIDS entity label: the alphanumeric characters of ``text``."""
    return re.sub(r"[^A-Za-z0-9]", "", str(text))


def _phantoms_of(ds: dict[str, Any]) -> list[str]:
    p = ds["phantom"]
    return [p] if isinstance(p, str) else list(p)


def _drop_flags(flags: list[str], names: tuple[str, ...]) -> list[str]:
    """Remove the named flags and their values from a flag list."""
    kept: list[str] = []
    skip = False
    for tok in flags:
        if skip:
            skip = False
            continue
        if tok in names:
            skip = True
            continue
        kept.append(tok)
    return kept


def _fill(flags: list[str], phantom_cfg: dict[str, Any], run: dict[str, Any]) -> list[str]:
    return [f.replace("{gre}", _gre_prefix(run)).replace("{phantom.motion_ap}", phantom_cfg.get("motion_ap", "<motion_ap.tsv>"))
            for f in flags]


def _entities(run: dict[str, Any], with_dir: bool = True) -> str:
    ents = run["sub"]
    if run.get("acq"):
        ents += f"_acq-{run['acq']}"
    if with_dir and run.get("dir"):
        ents += f"_dir-{run['dir']}"
    return ents


def _out_prefix(run: dict[str, Any]) -> str:
    return f"data/{run['dataset']}/{run['sub']}/dwi/{_entities(run)}"


def _gre_prefix(run: dict[str, Any]) -> str:
    return f"data/{run['dataset']}/{run['sub']}/fmap/{_entities(run, with_dir=False)}"


def _runs(cfg: dict[str, Any], dataset_id: str) -> list[dict[str, Any]]:
    """The simulation runs of one dataset: one dict per ``trxscan`` invocation."""
    d = cfg["defaults"]
    ds = cfg["datasets"][dataset_id]
    if ds.get("truth_only"):
        return []
    runs: list[dict[str, Any]] = []
    schemes = ds.get("schemes", [ds.get("scheme", d["scheme"])])
    variants = ds.get("variants", [{}])
    sweep = ds.get("sweep")
    sweep_values = sweep["values"] if sweep else [None]
    for sub in _phantoms_of(ds):
        for scheme in schemes:
            for var in variants:
                for i, val in enumerate(sweep_values):
                    voxel = ds.get("voxel_mm", d["voxel_mm"])
                    if sweep and sweep.get("key") == "voxel_mm":
                        voxel = float(val)
                    parts = ([scheme] if len(schemes) > 1 else []) + ([var["name"]] if var.get("name") else [])
                    if sweep:
                        auto = (sweep.get("flag") or sweep.get("key")).lstrip("-") + str(val)
                        parts.append(sweep["labels"][i] if sweep.get("labels") else auto)
                    run = {
                        "dataset": dataset_id, "sub": sub, "acq": bids_label("".join(parts)) or None,
                        "dir": var.get("dir", "AP"), "scheme": scheme, "voxel": voxel,
                        "oversample": var.get("oversample", ds.get("oversample", d["oversample"])),
                        "params": var.get("params", ds.get("params", d["params"])),
                        "extra": list(ds.get("extra_flags", d["extra_flags"])) + list(var.get("extra_flags", []))
                                 + ([f"{sweep['flag']} {val}"] if sweep and "flag" in sweep else []),
                        "truth_peaks": bool(ds.get("truth_peaks")),
                    }
                    runs.append(run)
                    if ds.get("reverse_pe_pair"):
                        # one fieldmap per pair: the reversed run does not write a second GRE
                        runs.append({**run, "dir": "PA", "extra": _drop_flags(run["extra"], ("--gre-out", "--gre-res")) + ["--reverse-pe"]})
        for extra in ds.get("extra_runs", []):
            runs.append({
                "dataset": dataset_id, "sub": sub, "acq": bids_label(extra["name"]), "dir": "AP",
                "scheme": ds.get("scheme", d["scheme"]), "voxel": ds.get("voxel_mm", d["voxel_mm"]),
                "oversample": ds.get("oversample", d["oversample"]), "params": ds.get("params", d["params"]),
                "extra": list(extra.get("extra_flags", [])), "truth_peaks": False,
            })
    return runs


def _base_flags(cfg: dict[str, Any], ds: dict[str, Any], run: dict[str, Any]) -> list[str]:
    d = cfg["defaults"]
    grid = f"work/{run['sub']}/{run['voxel']:g}mm"
    pcfg = cfg["phantoms"][run["sub"]]
    flags = [
        f"--wm {grid}/wm.nii.gz", f"--gm {grid}/gm.nii.gz", f"--csf {grid}/csf.nii.gz", f"--mask {grid}/mask.nii.gz",
        f"--streamlines {pcfg['tract'].split('/')[-1]}", f"--weights {pcfg['weights']}",
        f"--subsample {ds.get('subsample', d['subsample'])}", f"--seed {ds.get('seed', d['seed'])}",
        f"--params {run['params']}",
    ]
    if run["oversample"] > 1:
        flags += [f"--oversample {run['oversample']}"] + [f"--sim-{k} {grid}/sim/{k}.nii.gz" for k in ("wm", "gm", "csf", "mask", "fmap")]
    else:
        flags += ["--oversample 1", f"--fmap {grid}/fmap.nii.gz"]
    return flags


def _truth_command(cfg: dict[str, Any], dataset_id: str, sub: str) -> str:
    d = cfg["defaults"]
    ds = cfg["datasets"][dataset_id]
    pcfg = cfg["phantoms"][sub]
    grid = f"work/{sub}/{ds.get('voxel_mm', d['voxel_mm']):g}mm"
    return " ".join([
        "trxscan-microstructure", f"--wm {grid}/wm.nii.gz", f"--gm {grid}/gm.nii.gz", f"--csf {grid}/csf.nii.gz", f"--mask {grid}/mask.nii.gz",
        f"--streamlines {pcfg['tract'].split('/')[-1]}", f"--weights {pcfg['weights']}",
        f"--subsample {ds.get('subsample', d['subsample'])}", f"--seed {ds.get('seed', d['seed'])}", f"--params {ds.get('params', d['params'])}",
        "--big-delta 0.030 --small-delta 0.010", f"--out work/{dataset_id}/{sub}_truth",
    ])


def render_commands(cfg: dict[str, Any], dataset_id: str) -> list[str]:
    """The command lines for one dataset id, in the order the pipeline runs them.

    ``trxscan`` writes each series in place under ``sub-<anatomy>/dwi/``; the truth-map runs
    write to ``work/`` and the driver renames their outputs into the dataset's derivatives.
    """
    ds = cfg["datasets"][dataset_id]
    cmds: list[str] = []
    for run in _runs(cfg, dataset_id):
        flags = _base_flags(cfg, ds, run) + [f"--bval schemes/{run['scheme']}.bval", f"--bvec schemes/{run['scheme']}.bvec"]
        flags += _fill(run["extra"], cfg["phantoms"][run["sub"]], run)
        if run["truth_peaks"]:
            flags.append("--truth-peaks")
        cmds.append(" ".join(["trxscan"] + flags + [f"--out {_out_prefix(run)}"]))
    if ds.get("truth") or ds.get("truth_only"):
        cmds += [_truth_command(cfg, dataset_id, sub) for sub in _phantoms_of(ds)]
    return cmds


def print_commands(cfg: dict[str, Any], dataset_id: str) -> None:
    """Print one dataset's configured description, prerequisites, and command lines."""
    ds = cfg["datasets"][dataset_id]
    print(f"pipeline description: {ds['description'].strip()}")
    if ds.get("requires"):
        print(f"waits on simulator items {', '.join(ds['requires'])}")
    for cmd in render_commands(cfg, dataset_id):
        print(cmd)


def flags_used(cfg: dict[str, Any]) -> list[str]:
    """Every flag that appears in any dataset's command lines, sorted."""
    seen: set[str] = set()
    for ds_id in cfg["datasets"]:
        for cmd in render_commands(cfg, ds_id):
            for tok in cmd.split():
                if tok.startswith("--"):
                    seen.add(tok)
    return sorted(seen)


# ----------------------------------------------------------------------------- file layout
# The TRXScan file names come from its writers (``io::write_complex_dwi`` and the truth
# outputs in ``bin/trxscan.rs``); the driver renames the non-raw ones into BIDS derivative
# names. The precomputed names are the contract the pipeline's precompute scripts must honor,
# and the chapters load files by these names. Files that wait on a planned simulator change
# are tagged with the implementation-plan item that adds them.

_DWI_SUFFIXES = ("_part-mag_dwi.nii.gz", "_part-mag_dwi.json", "_part-phase_dwi.nii.gz",
                 "_part-phase_dwi.json", "_dwi.bval", "_dwi.bvec")
_DATASET_FILES = ("dataset_description.json", "README", "participants.tsv")


def _truth_map_files(root: str, sub: str) -> list[tuple[str, str | None]]:
    from dwibook.truth import ALL_TRUTH_MAPS, truth_filename  # local import: truth.py loads nibabel

    return [(f"{root}/{sub}/dwi/{truth_filename(sub, n)}", None) for n in ALL_TRUTH_MAPS]


def _run_files(run: dict[str, Any]) -> list[tuple[str, str | None]]:
    """(path, planned-item) pairs for one simulation run: raw series plus its ground truth."""
    ds, sub = run["dataset"], run["sub"]
    ents, no_dir = _entities(run), _entities(run, with_dir=False)
    flags = " ".join(run["extra"]).split()
    raw = f"data/{ds}/{sub}"
    truth = f"data/{ds}/{TRUTH_PIPELINE}/{sub}"
    files: list[tuple[str, str | None]] = [(f"{raw}/dwi/{ents}{s}", None) for s in _DWI_SUFFIXES]
    if "--gre-out" in flags:
        files += [(f"{raw}/fmap/{no_dir}_{k}.{ext}", None) for k in ("magnitude1", "magnitude2", "phasediff") for ext in ("nii.gz", "json")]
    if "--noise" in flags:
        files.append((f"{truth}/dwi/{ents}_desc-noise_dwimap.nii.gz", None))
    if "--mb" in flags:
        files.append((f"{truth}/dwi/{ents}_desc-dropout_dwi.tsv", None))
    if run["truth_peaks"]:
        files += [(f"{truth}/dwi/{ents}_model-truth_param-peaks_dwimap.{ext}", None) for ext in ("nii.gz", "json")]
    if "--gnl" in flags:
        files.append((f"{truth}/dwi/{ents}_desc-gnl_coeff.grad", None))
        files += [(f"{truth}/dwi/{ents}_desc-graddev_dwimap.{ext}", None) for ext in ("nii.gz", "json")]
    if "--export-kspace" in flags:
        files += [(f"{truth}/dwi/{ents}_desc-kspace_dwi.nii.gz", "T2"), (f"{truth}/dwi/{ents}_desc-kspacemask_dwi.nii.gz", "T2")]
    return files


def _derivative_files(tool: str, dataset_id: str, runs: list[dict[str, Any]], subs: list[str]) -> list[tuple[str, str | None]]:
    root = f"data/{dataset_id}/derivatives/{tool}"
    files: list[tuple[str, str | None]] = [(f"{root}/dataset_description.json", None)]
    if tool == "qsiprep":
        for sub in subs:
            files.append((f"{root}/{sub}.html", None))
            files += [(f"{root}/{sub}/dwi/{sub}_space-ACPC_{s}", None) for s in (
                "desc-preproc_dwi.nii.gz", "desc-preproc_dwi.bval", "desc-preproc_dwi.bvec", "desc-preproc_dwi.json",
                "desc-brain_mask.nii.gz", "dwiref.nii.gz", "desc-image_qc.tsv", "desc-confounds_timeseries.tsv")]
    elif tool == "topup":
        for pair in sorted({_entities(r, with_dir=False) for r in runs}):
            sub = pair.split("_")[0]
            files += [(f"{root}/{sub}/fmap/{pair}_desc-topup_fieldmap.{ext}", None) for ext in ("nii.gz", "json")]
            files += [(f"{root}/{sub}/dwi/{pair}_desc-topup_{s}", None) for s in ("fieldcoef.nii.gz", "movpar.txt", "dwi.nii.gz")]
    elif tool == "eddy":
        for r in runs:
            ents = _entities(r)
            files += [(f"{root}/{r['sub']}/dwi/{ents}_desc-eddy_{s}", None) for s in (
                "dwi.nii.gz", "dwi.bvec", "parameters.txt", "movementrms.txt", "outliermap.txt")]
    elif tool in ("gradunwarp", "tortoise-gnl"):
        for r in runs:
            ents, no_dir = _entities(r), _entities(r, with_dir=False)
            files.append((f"{root}/{r['sub']}/xfm/{no_dir}_from-apparent_to-true_mode-image_xfm.nii.gz", None))
            if tool == "gradunwarp":
                files.append((f"{root}/{r['sub']}/dwi/{ents}_desc-unwarped_dwi.nii.gz", None))
            else:
                files.append((f"{root}/{r['sub']}/dwi/{ents}_desc-graddev_dwimap.nii.gz", None))
    else:
        raise ValueError(f"unknown precompute tool {tool!r}")
    return files


def expected_files(cfg: dict[str, Any], dataset_id: str) -> list[tuple[str, str | None]]:
    """Every file a dataset directory holds once the pipeline has run, as ``(path, planned)``.

    Paths are relative to the pipeline's data root (``data/<dataset>/...``); ``planned`` is the
    implementation-plan item a file waits on, or None.
    """
    ds = cfg["datasets"][dataset_id]
    subs = _phantoms_of(ds)
    root = f"data/{dataset_id}"
    if ds.get("truth_only"):
        files: list[tuple[str, str | None]] = [(f"{root}/dataset_description.json", None), (f"{root}/provenance.json", None)]
        for sub in subs:
            files += _truth_map_files(root, sub)
        return files
    files = [(f"{root}/{f}", None) for f in _DATASET_FILES]
    files += [(f"{root}/{TRUTH_PIPELINE}/{f}", None) for f in ("dataset_description.json", "provenance.json")]
    runs = _runs(cfg, dataset_id)
    for run in runs:
        files += _run_files(run)
    # The gradient-nonlinearity displacement does not depend on the phase-encode direction, so
    # a blip-up/blip-down pair shares one pair of transforms.
    for no_dir in sorted({_entities(r, with_dir=False) for r in runs if "--gnl" in r["extra"]}):
        sub = no_dir.split("_")[0]
        files += [(f"{root}/{TRUTH_PIPELINE}/{sub}/xfm/{no_dir}_from-{a}_to-{b}_mode-image_xfm.{ext}", None)
                  for a, b in (("true", "apparent"), ("apparent", "true")) for ext in ("nii.gz", "json")]
    if ds.get("truth"):
        for sub in subs:
            files += _truth_map_files(f"{root}/{TRUTH_PIPELINE}", sub)
    for tool in ds.get("precompute", []):
        files += _derivative_files(tool, dataset_id, runs, subs)
    return files


def print_tree(cfg: dict[str, Any], dataset_id: str) -> None:
    """Print the expected files of one dataset as a directory tree."""
    tree: dict[str, Any] = {}
    for path, planned in expected_files(cfg, dataset_id):
        node = tree
        parts = path.split("/")
        for part in parts[:-1]:
            node = node.setdefault(part + "/", {})
        node[parts[-1]] = planned

    def walk(node: dict[str, Any], indent: str) -> None:
        names = sorted(node, key=lambda n: (not n.endswith("/"), n))
        for i, name in enumerate(names):
            last = i == len(names) - 1
            child = node[name]
            tag = "" if child is None or isinstance(child, dict) else f"   (planned: item {child})"
            print(f"{indent}{'└── ' if last else '├── '}{name}{tag}")
            if isinstance(child, dict):
                walk(child, indent + ("    " if last else "│   "))

    walk(tree, "")
