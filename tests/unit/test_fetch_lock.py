"""The fetch lock that the compute queue waits on (replaces a PowerShell script that the
default execution policy blocked)."""

import os

import pytest

from energy.data import fetch as F


def test_lock_exists_only_while_fetching(sample_cfg, tmp_path):
    cfg = sample_cfg.model_copy(update={"root": tmp_path})
    assert F.active_fetches(cfg) == []
    with F.fetch_lock(cfg) as path:
        assert path.exists()
        assert F.active_fetches(cfg) == [os.getpid()]
    assert not path.exists()
    assert F.active_fetches(cfg) == []


def test_lock_is_removed_after_an_error(sample_cfg, tmp_path):
    cfg = sample_cfg.model_copy(update={"root": tmp_path})
    with pytest.raises(RuntimeError), F.fetch_lock(cfg):
        raise RuntimeError("download failed")
    assert F.active_fetches(cfg) == []


def test_stale_lock_of_a_dead_process_is_ignored_and_cleaned(sample_cfg, tmp_path):
    cfg = sample_cfg.model_copy(update={"root": tmp_path})
    stale = F.lock_dir(cfg) / "fetch-999999.lock"
    stale.write_text("999999")
    (F.lock_dir(cfg) / "fetch-garbage.lock").write_text("not a pid")
    assert F.active_fetches(cfg) == []
    assert not stale.exists()
    F.wait_for_fetches(cfg, poll_s=0.01)  # returns immediately
