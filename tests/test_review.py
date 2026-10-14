import os

import numpy as np
from astropy.table import Table

from hostassoc import review as R


def make_wide():
    n = 10
    t = Table()
    t["transient_key"] = [f"k{i}" for i in range(n)]
    t["transient_ra"] = np.full(n, 53.1); t["transient_dec"] = np.full(n, -28.1)
    t["n_candidates"] = np.full(n, 5)
    t["host_1_objectId"] = np.arange(100, 100 + n)
    t["host_2_objectId"] = np.arange(200, 200 + n)
    t["dlr_host_objectId"] = np.arange(100, 100 + n)
    t["dlr_host_d_dlr"] = np.full(n, 2.0)
    # rows 0-3: DLR picks host_2 (disagreement); row 4: DLR none; row 5: SBW none
    t["dlr_host_objectId"][:4] = t["host_2_objectId"][:4]
    t["dlr_host_d_dlr"][4] = 7.0
    t["host_1_objectId"][5] = 0
    return t


def test_host_columns_and_selection():
    t = make_wide()
    assert R.host_columns(t) == [1, 2]
    sel = R.select_disagreements(t, n=20, seed=1)
    assert sel.meta["n_disagreements_total"] == 4 and set(sel["transient_key"]) == {"k0", "k1", "k2", "k3"}
    sel2 = R.select_disagreements(t, n=2, seed=1)
    assert len(sel2) == 2 and set(sel2["transient_key"]) <= {"k0", "k1", "k2", "k3"}
    sel3 = R.select_disagreements(t, n=20, include_dlr_none=True)
    assert sel3.meta["n_disagreements_total"] == 5 and "k4" in set(sel3["transient_key"])
    assert "k5" not in set(sel3["transient_key"])     # no SBW host -> never reviewed


def test_token_argument_accepts_string_or_file(tmp_path, monkeypatch):
    monkeypatch.delenv("RSP_TOKEN", raising=False)
    R._set_token("abc")
    assert os.environ["RSP_TOKEN"] == "abc"
    f = tmp_path / "tok"; f.write_text("filetoken\n")
    R._set_token(str(f))
    assert os.environ["RSP_TOKEN"] == "filetoken"
