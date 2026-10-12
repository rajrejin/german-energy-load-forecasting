"""Artifact store implementations behind one interface; Azurite test runs when it is reachable."""

import socket

import pytest

from energy.store.base import KeyNotFoundError, make_store
from energy.store.local import LocalStore


def azurite_up() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 10000), timeout=0.5):
            return True
    except OSError:
        return False


@pytest.fixture(params=["local", "azurite"])
def store(request, tmp_path):
    if request.param == "local":
        return LocalStore(tmp_path / "store")
    if not azurite_up():
        pytest.skip("Azurite not running on 127.0.0.1:10000 (docker run -p 10000:10000 azurite)")
    s = make_store("azurite", container="energy-test")
    for k in s.list_keys("t/"):
        s.delete(k)
    return s


def test_put_get_exists_delete(store):
    store.put_bytes("t/a.bin", b"hello")
    assert store.exists("t/a.bin")
    assert store.get_bytes("t/a.bin") == b"hello"
    store.put_json("t/b.json", {"x": 1})
    assert store.get_json("t/b.json") == {"x": 1}
    assert store.list_keys("t/") == ["t/a.bin", "t/b.json"]
    store.delete("t/a.bin")
    store.delete("t/a.bin")  # deleting a missing key is not an error
    assert not store.exists("t/a.bin")
    with pytest.raises(KeyNotFoundError):
        store.get_bytes("t/a.bin")


def test_overwrite_and_directory_roundtrip(store, tmp_path):
    store.put_bytes("t/o.bin", b"1")
    store.put_bytes("t/o.bin", b"22")
    assert store.get_bytes("t/o.bin") == b"22"
    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    (src / "m.json").write_text("{}")
    (src / "sub" / "w.bin").write_bytes(b"abc")
    keys = store.upload_dir(src, "t/model")
    assert sorted(keys) == ["t/model/m.json", "t/model/sub/w.bin"]
    out = tmp_path / "dst"
    store.download_dir("t/model", out)
    assert (out / "sub" / "w.bin").read_bytes() == b"abc"


def test_local_store_rejects_path_escape(tmp_path):
    s = LocalStore(tmp_path / "root")
    with pytest.raises(ValueError):
        s.put_bytes("../evil.txt", b"x")


def test_make_store_rejects_unknown_kind():
    with pytest.raises(ValueError):
        make_store("ftp")
