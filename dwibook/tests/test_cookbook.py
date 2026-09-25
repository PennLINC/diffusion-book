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
