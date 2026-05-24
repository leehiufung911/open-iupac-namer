"""Regression tests for Stage 10 R10-A-5: naphthyridine retained names.

Adds curated retained-name entries for the four naphthyridine isomers
that were previously missing from the engine (1,5- and 1,8- already
worked):

- 1,6-naphthyridine
- 1,7-naphthyridine
- 2,6-naphthyridine (C2-symmetric)
- 2,7-naphthyridine (C2-symmetric)

Before R10-A-5 the engine returned ``[NAMING ERROR: No valid naming
plan found ...]`` for these scaffolds.
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
    td = tempfile.mkdtemp(prefix="r10a5_test_")
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


_PARENT_CASES = [
    ("c1cnc2ccncc2c1", "1,6-naphthyridine"),
    ("c1cnc2cnccc2c1", "1,7-naphthyridine"),
    ("c1cc2cnccc2cn1", "2,6-naphthyridine"),
    ("c1cc2ccncc2cn1", "2,7-naphthyridine"),
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
    assert rt is not None, f"OPSIN rejected emitted name {name!r}"
    assert _canon(rt) == _canon(smi), (
        f"round-trip drift: in={smi!r} -> name={name!r} -> rt={rt!r}"
    )


# Chloro-substituted forms verified by OPSIN chloro-probing.
_CHLORO_LOCANT_CASES = [
    # 1,6-naphthyridine: L2 -> idx1, L3 -> idx0, L4 -> idx9,
    #   L5 -> idx7, L7 -> idx5, L8 -> idx4
    ("Clc1ccc2cnccc2n1", "2-chloro-1,6-naphthyridine"),
    ("Clc1cnc2ccncc2c1", "3-chloro-1,6-naphthyridine"),
    ("Clc1ccnc2ccncc12", "4-chloro-1,6-naphthyridine"),
    ("Clc1nccc2ncccc12", "5-chloro-1,6-naphthyridine"),
    # 1,7-naphthyridine
    ("Clc1ccc2ccncc2n1", "2-chloro-1,7-naphthyridine"),
    ("Clc1cnc2cnccc2c1", "3-chloro-1,7-naphthyridine"),
    ("Clc1ccnc2cnccc12", "4-chloro-1,7-naphthyridine"),
    # 2,6-naphthyridine (C2-symmetric, locants pin the chosen orbit)
    ("Clc1nccc2cnccc12", "1-chloro-2,6-naphthyridine"),
    ("Clc1cncc2ccncc12", "4-chloro-2,6-naphthyridine"),
    # 2,7-naphthyridine (C2-symmetric)
    ("Clc1nccc2ccncc12", "1-chloro-2,7-naphthyridine"),
]


@pytest.mark.parametrize("smi,expected", _CHLORO_LOCANT_CASES)
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


def test_15_naphthyridine_still_works() -> None:
    """Sentinel: 1,5-naphthyridine (already covered) must still resolve."""
    name = name_smiles("c1ccc2ncccc2n1")
    assert name == "1,5-naphthyridine"


def test_quinoline_still_works() -> None:
    """Sentinel: 6,6 benzo-pyridine path must still resolve."""
    name = name_smiles("c1ccc2ncccc2c1")
    assert name == "quinoline"


def test_no_naming_error_for_naphthyridine() -> None:
    name = name_smiles("c1cnc2ccncc2c1")
    assert name is not None
    assert "NAMING ERROR" not in name
