"""Regression tests for Stage 11 R11-A: dibenz[b,d]ofuran/thiophene/selenophene
retained names with verified atom_locants.

Scope:
- Adds curated retained-name entries for the sulfur and selenium analogues
  of dibenzofuran (NAMING_ERROR for both before R11-A).
- Repairs an existing latent locant bug in the dibenzofuran entry: it
  previously had no ``atom_locants`` map, so the engine emitted symmetric
  but wrong locants for chloro-substituted forms (e.g. 3-chlorodibenzofuran
  rendered as 4-chlorodibenzofuran, which round-trips to the *other*
  C2v-related canonical SMILES).

The three scaffolds share a single C2v skeletal layout and thus the same
atom_locants map, with the heteroatom slot at canonical-RDKit idx 6.
Locants verified by OPSIN chloro-probing positions 1-4 on each.
"""
from __future__ import annotations

import os
import shutil
import tempfile

import pytest
from rdkit import Chem

from iupac_namer.engine import name_smiles


def _canon(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m is not None else None


def _opsin_roundtrip(name: str) -> str | None:
    if not name:
        return None
    try:
        from py2opsin import py2opsin
    except ImportError:  # pragma: no cover
        pytest.skip("py2opsin not available")
    cwd = os.getcwd()
    td = tempfile.mkdtemp(prefix="r11a_test_")
    try:
        os.chdir(td)
        rt = py2opsin(name)
        return rt or None
    finally:
        os.chdir(cwd)
        try:
            shutil.rmtree(td)
        except Exception:
            pass


# ---- parent-form pinning -------------------------------------------


_PARENT_CASES = [
    ("c1ccc2c(c1)oc1ccccc12",    "dibenzofuran"),
    ("c1ccc2c(c1)sc1ccccc12",    "dibenzothiophene"),
    ("c1ccc2c(c1)[se]c1ccccc12", "dibenzoselenophene"),
]


@pytest.mark.parametrize("smi,expected", _PARENT_CASES)
def test_parent_emits_curated_name(smi: str, expected: str) -> None:
    name = name_smiles(smi)
    assert name == expected, (
        f"for {smi!r} expected {expected!r} but got {name!r}"
    )


@pytest.mark.parametrize("smi,expected", _PARENT_CASES)
def test_parent_round_trips_via_opsin(smi: str, expected: str) -> None:
    name = name_smiles(smi)
    assert name is not None and "NAMING ERROR" not in name
    rt = _opsin_roundtrip(name)
    assert rt is not None
    assert _canon(rt) == _canon(smi)


# ---- chloro-locant verification ------------------------------------


_CHLORO_CASES = [
    # dibenzofuran (latent locant bug — pre-R11-A engine emitted wrong locants)
    ("Clc1cccc2oc3ccccc3c12",     "1-chlorodibenzofuran"),
    ("Clc1ccc2oc3ccccc3c2c1",     "2-chlorodibenzofuran"),
    ("Clc1ccc2c(c1)oc1ccccc12",   "3-chlorodibenzofuran"),
    ("Clc1cccc2c1oc1ccccc12",     "4-chlorodibenzofuran"),
    # dibenzothiophene (NAMING_ERROR pre-R11-A)
    ("Clc1cccc2sc3ccccc3c12",     "1-chlorodibenzothiophene"),
    ("Clc1ccc2sc3ccccc3c2c1",     "2-chlorodibenzothiophene"),
    ("Clc1ccc2c(c1)sc1ccccc12",   "3-chlorodibenzothiophene"),
    ("Clc1cccc2c1sc1ccccc12",     "4-chlorodibenzothiophene"),
    # dibenzoselenophene (NAMING_ERROR pre-R11-A)
    ("Clc1cccc2[se]c3ccccc3c12",   "1-chlorodibenzoselenophene"),
    ("Clc1ccc2[se]c3ccccc3c2c1",   "2-chlorodibenzoselenophene"),
    ("Clc1ccc2c(c1)[se]c1ccccc12", "3-chlorodibenzoselenophene"),
    ("Clc1cccc2c1[se]c1ccccc12",   "4-chlorodibenzoselenophene"),
]


@pytest.mark.parametrize("smi,expected", _CHLORO_CASES)
def test_chloro_locant_round_trips(smi: str, expected: str) -> None:
    name = name_smiles(smi)
    assert name == expected, (
        f"for {smi!r} expected {expected!r} but got {name!r}"
    )
    rt = _opsin_roundtrip(name)
    assert rt is not None
    assert _canon(rt) == _canon(smi), (
        f"locant drift: in={smi!r} name={name!r} rt={rt!r}"
    )


# ---- regression sentinels ------------------------------------------


def test_no_naming_error_for_dibenzothiophene() -> None:
    """Specific guard: pre-R11-A `c1ccc2c(c1)sc1ccccc12` returned
    NAMING_ERROR (no curated entry for the S analog)."""
    name = name_smiles("c1ccc2c(c1)sc1ccccc12")
    assert name == "dibenzothiophene"


def test_no_locant_drift_for_dibenzofuran() -> None:
    """Specific guard: pre-R11-A 3-chlorodibenzofuran rendered as
    '4-chlorodibenzofuran' which round-trips to a *different* canonical
    structure (locant drift between equivalent C2v positions).  Pin the
    correct emission."""
    name = name_smiles("Clc1ccc2c(c1)oc1ccccc12")
    assert name == "3-chlorodibenzofuran", (
        f"expected '3-chlorodibenzofuran' but got {name!r}; "
        f"if this drifts back, the atom_locants map regressed"
    )
