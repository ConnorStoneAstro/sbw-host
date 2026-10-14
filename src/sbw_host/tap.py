"""Connection helpers for the Rubin Science Platform TAP and image services.

Tokens
------
Create a token at https://data.lsst.cloud (Account -> Security tokens) with the
scopes ``read:tap`` (catalogue queries) and ``read:image`` (cutouts for the
diagnostic figures).  One token with both scopes is simplest.  The token is
found, in order, from:

1. the ``token=`` argument,
2. the ``RSP_TOKEN`` environment variable,
3. a file named ``.rsp_image_tap_token`` or ``.rsp_token`` in the current working
   directory, next to this package, or in the home directory
   (also ``~/.rsp-tap.token``).

If you keep separate tokens, ``.rsp_image_token`` (or ``RSP_IMAGE_TOKEN``) is used
for the image services and ``.rsp_token`` for TAP.
"""

from __future__ import annotations

import os
from pathlib import Path

import pyvo
import requests

RSP_BASE = "https://data.lsst.cloud"
DEFAULT_TAP_URL = f"{RSP_BASE}/api/tap"

_PKG_DIR = Path(__file__).resolve().parent
_SEARCH_DIRS = (Path.cwd(), _PKG_DIR, _PKG_DIR.parent, Path.home())
TOKEN_NAMES = (".rsp_image_tap_token", ".rsp_token", ".rsp-tap.token")
IMAGE_TOKEN_NAMES = (".rsp_image_tap_token", ".rsp_image_token", ".rsp-image.token")


def _read_first(names) -> str | None:
    for d in _SEARCH_DIRS:
        for name in names:
            path = d / name
            try:
                if path.is_file():
                    text = path.read_text().strip()
                    if text:
                        return text
            except OSError:
                continue
    return None


def load_token(token: str | None = None) -> str:
    """Return an RSP token with ``read:tap`` scope (see module docstring for the search order)."""
    if token:
        return token.strip()
    env = os.environ.get("RSP_TOKEN")
    if env:
        return env.strip()
    found = _read_first(TOKEN_NAMES)
    if found:
        return found
    raise RuntimeError(
        "No RSP token found. Pass token=..., set RSP_TOKEN, or save the token in a file named "
        ".rsp_image_tap_token (or .rsp_token) in the working directory, next to the sbw_host "
        "package, or in your home directory. Create one at https://data.lsst.cloud "
        "(Account -> Security tokens, scopes read:tap and read:image)."
    )


def load_image_token(token: str | None = None) -> str:
    """Return an RSP token with ``read:image`` scope, falling back to the TAP token."""
    if token:
        return token.strip()
    env = os.environ.get("RSP_IMAGE_TOKEN")
    if env:
        return env.strip()
    return _read_first(IMAGE_TOKEN_NAMES) or load_token()


def token_scopes(token: str | None = None) -> list[str]:
    """Ask the RSP which scopes a token carries (useful for troubleshooting)."""
    r = requests.get(f"{RSP_BASE}/auth/api/v1/token-info",
                     headers={"Authorization": f"Bearer {load_token(token)}"}, timeout=30)
    r.raise_for_status()
    return list(r.json().get("scopes", []))


def make_session(token: str | None = None) -> requests.Session:
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {load_token(token)}"
    return session


def discover_tap_url(dataset: str = "dp2") -> str:
    """Look up the TAP URL via the RSP discovery service, with a static fallback."""
    try:
        r = requests.get(f"{RSP_BASE}/repertoire/discovery", timeout=10)
        r.raise_for_status()
        return r.json()["datasets"][dataset]["services"]["tap"]["url"]
    except Exception:
        return DEFAULT_TAP_URL


def get_tap_service(token: str | None = None, dataset: str = "dp2",
                    session: requests.Session | None = None) -> pyvo.dal.TAPService:
    """Build an authenticated pyvo TAPService for the given dataset."""
    session = session or make_session(token)
    return pyvo.dal.TAPService(discover_tap_url(dataset), session=session)
