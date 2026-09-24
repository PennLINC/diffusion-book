import numpy as np
import nibabel as nib
import pytest

from dwibook import data


def test_unknown_dataset_raises_helpful_error(monkeypatch):
    monkeypatch.delenv("DWIBOOK_DATA", raising=False)
    with pytest.raises(KeyError, match="not in the registry"):
        data.load_dataset("does-not-exist")


def test_local_override_wins(tmp_path, monkeypatch):
    ds_dir = tmp_path / "toy"
    ds_dir.mkdir()
    img = nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4))
    nib.save(img, ds_dir / "x.nii.gz")
    np.savetxt(ds_dir / "s.bval", np.array([[0, 1000, 1000]]), fmt="%d")
    np.savetxt(ds_dir / "s.bvec", np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]), fmt="%.3f")
    monkeypatch.setenv("DWIBOOK_DATA", str(tmp_path))
    ds = data.load_dataset("toy")
    assert ds.path == ds_dir
    assert ds.volume("x.nii.gz").shape == (4, 4, 4)
    b, v = ds.bvals_bvecs("s")
    assert b.tolist() == [0, 1000, 1000] and v.shape == (3, 3)
    with pytest.raises(FileNotFoundError):
        ds.file("missing.nii.gz")


def test_registry_parses():
    assert isinstance(data.registered_datasets(), list)
