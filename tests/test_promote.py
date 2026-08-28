from pathlib import Path

from training.promote import current_checkpoint_path, promote, rollback


def test_current_checkpoint_path_none_when_no_pointer(tmp_path: Path):
    assert current_checkpoint_path(str(tmp_path)) is None


def test_promote_writes_pointer_and_history(tmp_path: Path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "adapter_model.bin").write_text("weights")

    served = promote(str(tmp_path), "2026-01-05", str(candidate))

    assert current_checkpoint_path(str(tmp_path)) == served
    assert (Path(served) / "adapter_model.bin").exists()

    history = (tmp_path / "serving" / "promotion_history.jsonl").read_text().splitlines()
    assert len(history) == 1


def test_rollback_to_previous_cycle(tmp_path: Path):
    c1 = tmp_path / "c1"
    c1.mkdir()
    (c1 / "w.bin").write_text("v1")
    c2 = tmp_path / "c2"
    c2.mkdir()
    (c2 / "w.bin").write_text("v2")

    promote(str(tmp_path), "cycle-1", str(c1))
    promote(str(tmp_path), "cycle-2", str(c2))

    rolled_back_to = rollback(str(tmp_path))
    assert rolled_back_to == current_checkpoint_path(str(tmp_path))
    assert "cycle-1" in rolled_back_to


def test_rollback_with_single_promotion_goes_to_base(tmp_path: Path):
    c1 = tmp_path / "c1"
    c1.mkdir()
    (c1 / "w.bin").write_text("v1")
    promote(str(tmp_path), "cycle-1", str(c1))

    rollback(str(tmp_path))
    assert current_checkpoint_path(str(tmp_path)) is None
