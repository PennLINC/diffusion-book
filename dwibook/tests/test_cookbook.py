import re

from dwibook import cookbook


def test_config_loads_and_lists_all_datasets():
    cfg = cookbook.load_config()
    assert set(cfg["datasets"]) >= {"ref-clean", "ref-schemes", "noise-sweep", "gibbs", "sdc-pair", "eddy", "motion-mb", "gnl", "kitchen-sink", "voxel-sweep", "te-sweep", "truth", "presets"}


def test_render_commands_expands_sweeps_variants_and_pairs():
    cfg = cookbook.load_config()
    assert len(cookbook.render_commands(cfg, "ref-clean")) == 2  # simulation + truth
    assert len([c for c in cookbook.render_commands(cfg, "ref-schemes") if c.startswith("trxscan ")]) == 4  # one per scheme
    noise = cookbook.render_commands(cfg, "noise-sweep")
    assert sum("--noise 5e-5" in c for c in noise) == 1 and sum("--coils 8" in c for c in noise) == 1
    voxel = cookbook.render_commands(cfg, "voxel-sweep")
    assert any("work/sub-0001a/1.5mm/wm.nii.gz" in c for c in voxel) and len(voxel) == 4
    ks = cookbook.render_commands(cfg, "kitchen-sink")
    assert len(ks) == 2 and "--reverse-pe" in ks[1] and "--oversample 2" in ks[0]
    assert "--gre-out" in ks[0] and "--gre-out" not in ks[1]  # one fieldmap per pair
    gnl = cookbook.render_commands(cfg, "gnl")
    assert sum("--gnl-no-warp" in c for c in gnl) == 1 and all("--isocenter 0,-20,-30" in c for c in gnl)
    presets = cookbook.render_commands(cfg, "presets")
    assert len(presets) == 3 and sum("--params neonatal" in c for c in presets) == 1 and all(c.count("--params") == 1 for c in presets)
    truth = cookbook.render_commands(cfg, "truth")
    assert len(truth) == 1 and truth[0].startswith("trxscan-microstructure")


def test_every_used_flag_is_documented():
    cfg = cookbook.load_config()
    documented = set()
    for k in cookbook.FLAG_GLOSSARY:
        documented.update(k.split(" / "))
    undocumented = [f for f in cookbook.flags_used(cfg) if f not in documented and not f.startswith(("--wm", "--gm", "--csf", "--mask", "--streamlines", "--bval", "--bvec", "--out", "--fmap", "--sim-"))]
    assert undocumented == [], undocumented


BIDS_DWI = re.compile(r"^data/[a-z-]+/sub-[A-Za-z0-9]+/dwi/sub-[A-Za-z0-9]+(_acq-[A-Za-z0-9]+)?_dir-(AP|PA)(_part-(mag|phase))?_dwi\.(nii\.gz|json|bval|bvec)$")


def test_expected_files_are_bids_and_unique():
    cfg = cookbook.load_config()
    for ds_id in cfg["datasets"]:
        files = cookbook.expected_files(cfg, ds_id)
        paths = [p for p, _ in files]
        assert len(set(paths)) == len(paths), ds_id  # no two runs write the same file
        assert f"data/{ds_id}/dataset_description.json" in paths
        if cfg["datasets"][ds_id].get("truth_only"):
            continue  # a derivative-type dataset: truth maps only, no raw series
        raw = [p for p in paths if "/derivatives/" not in p and p.split("/")[2].startswith("sub-")]
        for p in raw:
            assert BIDS_DWI.match(p) or "/fmap/" in p, p
    gnl = dict(cookbook.expected_files(cfg, "gnl"))
    assert "data/gnl/sub-0001a/dwi/sub-0001a_acq-wb80x2_dir-AP_part-mag_dwi.nii.gz" in gnl
    assert "data/gnl/derivatives/trxscan/sub-0001a/dwi/sub-0001a_acq-wb80_dir-AP_desc-graddev_dwimap.nii.gz" in gnl
    assert "data/gnl/derivatives/trxscan/sub-0001a/xfm/sub-0001a_acq-wb80_from-true_to-apparent_mode-image_xfm.nii.gz" in gnl
    assert "data/gnl/derivatives/gradunwarp/sub-0001a/xfm/sub-0001a_acq-wb80_from-apparent_to-true_mode-image_xfm.nii.gz" in gnl
    sdc = dict(cookbook.expected_files(cfg, "sdc-pair"))
    assert "data/sdc-pair/sub-0001a/dwi/sub-0001a_dir-PA_part-phase_dwi.nii.gz" in sdc
    assert "data/sdc-pair/sub-60501/fmap/sub-60501_phasediff.nii.gz" in sdc
    assert "data/sdc-pair/derivatives/topup/sub-60501/fmap/sub-60501_desc-topup_fieldmap.nii.gz" in sdc
    truth = [p for p, _ in cookbook.expected_files(cfg, "truth") if p.endswith(".nii.gz")]
    assert len(truth) == 27 and "data/truth/sub-0001a/dwi/sub-0001a_model-truth_param-kbulk_dwimap.nii.gz" in truth
    assert "data/ref-clean/derivatives/trxscan/sub-0001a/dwi/sub-0001a_model-truth_param-fa_dwimap.nii.gz" in dict(cookbook.expected_files(cfg, "ref-clean"))
    assert "data/ref-clean/derivatives/trxscan/sub-0001a/dwi/sub-0001a_dir-AP_model-truth_param-peaks_dwimap.nii.gz" in dict(cookbook.expected_files(cfg, "ref-clean"))
    slab = dict(cookbook.expected_files(cfg, "slab-kspace"))
    assert slab["data/slab-kspace/derivatives/trxscan/sub-0001a/dwi/sub-0001a_dir-AP_desc-kspace_dwi.nii.gz"] == "T2"
    ks = dict(cookbook.expected_files(cfg, "kitchen-sink"))
    assert "data/kitchen-sink/sub-0001a/dwi/sub-0001a_dir-PA_part-mag_dwi.nii.gz" in ks
    assert "data/kitchen-sink/sub-0001a/fmap/sub-0001a_magnitude1.nii.gz" in ks
    assert "data/kitchen-sink/derivatives/qsiprep/sub-0001a/dwi/sub-0001a_space-ACPC_desc-preproc_dwi.nii.gz" in ks
    noise = dict(cookbook.expected_files(cfg, "noise-sweep"))
    assert "data/noise-sweep/sub-0001a/dwi/sub-0001a_acq-coils8r2_dir-AP_dwi.bval" in noise
    assert "data/noise-sweep/derivatives/trxscan/sub-0001a/dwi/sub-0001a_acq-noise1_dir-AP_desc-noise_dwimap.nii.gz" in noise


def test_commands_name_outputs_by_bids_entities():
    cfg = cookbook.load_config()
    cmds = cookbook.render_commands(cfg, "ref-schemes")
    assert any(c.endswith("--out data/ref-schemes/sub-0001a/dwi/sub-0001a_acq-dsi257_dir-AP") for c in cmds)
    sdc = cookbook.render_commands(cfg, "sdc-pair")
    assert any("--gre-out data/sdc-pair/sub-0001a/fmap/sub-0001a " in c and c.endswith("sub-0001a_dir-AP") for c in sdc)
    assert cookbook.render_commands(cfg, "truth")[0].endswith("--out work/truth/sub-0001a_truth")
    assert cookbook.bids_label("wb80-warp-only") == "wb80warponly"
