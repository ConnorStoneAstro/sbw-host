import os

import pytest

from sbw_host import tap


def test_explicit_and_env_token(monkeypatch):
    assert tap.load_token(" abc ") == "abc"
    monkeypatch.setenv("RSP_TOKEN", "fromenv")
    assert tap.load_token() == "fromenv"
    monkeypatch.setenv("RSP_IMAGE_TOKEN", "imgenv")
    assert tap.load_image_token() == "imgenv"


def test_combined_token_file_preferred(monkeypatch, tmp_path):
    monkeypatch.delenv("RSP_TOKEN", raising=False)
    monkeypatch.delenv("RSP_IMAGE_TOKEN", raising=False)
    monkeypatch.setattr(tap, "_SEARCH_DIRS", (tmp_path,))
    (tmp_path / ".rsp_token").write_text("taponly\n")
    assert tap.load_token() == "taponly"
    assert tap.load_image_token() == "taponly"       # falls back to the TAP token
    (tmp_path / ".rsp_image_token").write_text("imgonly\n")
    assert tap.load_image_token() == "imgonly"
    (tmp_path / ".rsp_image_tap_token").write_text("both\n")
    assert tap.load_token() == "both" and tap.load_image_token() == "both"


def test_missing_token_message(monkeypatch, tmp_path):
    monkeypatch.delenv("RSP_TOKEN", raising=False)
    monkeypatch.setattr(tap, "_SEARCH_DIRS", (tmp_path,))
    with pytest.raises(RuntimeError, match="rsp_image_tap_token"):
        tap.load_token()
