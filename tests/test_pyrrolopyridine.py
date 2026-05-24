"""Regression tests for Stage 10 R10-A-2: pyrrolo[?,?-?]pyridine retained names.

Adds curated retained-name entries for the four pyrrolopyridine
(azaindole) regio-isomers (5,6-fused, 1 pyrrole-NH + 1 pyridine-N):

- 1H-pyrrolo[2,3-b]pyridine (= 7-azaindole)
- 1H-pyrrolo[3,2-b]pyridine (= 4-azaindole)
- 1H-pyrrolo[2,3-c]pyridine
- 1H-pyrrolo[3,2-c]pyridine (= 5-azaindole)

Before R10-A-2 the engine returned ``[NAMING ERROR: No valid naming
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
    td = tempfile.mkdtemp(prefix="r10a2_test_")
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
    ("c1cnc2[nH]ccc2c1", "1H-pyrrolo[2,3-b]pyridine"),
    ("c1cnc2cc[nH]c2c1", "1H-pyrrolo[3,2-b]pyridine"),
    ("c1cc2cc[nH]c2cn1", "1H-pyrrolo[2,3-c]pyridine"),
    ("c1cc2[nH]ccc2cn1", "1H-pyrrolo[3,2-c]pyridine"),
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
    # 1H-pyrrolo[2,3-b]pyridine: L4 -> idx8, L5 -> idx0, L6 -> idx1
    ("Clc1ccnc2[nH]ccc12", "4-chloro-1H-pyrrolo[2,3-b]pyridine"),
    ("Clc1cnc2[nH]ccc2c1", "5-chloro-1H-pyrrolo[2,3-b]pyridine"),
    ("Clc1ccc2cc[nH]c2n1", "6-chloro-1H-pyrrolo[2,3-b]pyridine"),
    # 1H-pyrrolo[3,2-b]pyridine: L5 -> idx1, L6 -> idx0, L7 -> idx8
    ("Clc1ccc2[nH]ccc2n1", "5-chloro-1H-pyrrolo[3,2-b]pyridine"),
    ("Clc1cnc2cc[nH]c2c1", "6-chloro-1H-pyrrolo[3,2-b]pyridine"),
    ("Clc1ccnc2cc[nH]c12", "7-chloro-1H-pyrrolo[3,2-b]pyridine"),
    # 1H-pyrrolo[2,3-c]pyridine: L4 -> idx1, L5 -> idx0, L7 -> idx7
    ("Clc1cncc2[nH]ccc12", "4-chloro-1H-pyrrolo[2,3-c]pyridine"),
    ("Clc1nccc2cc[nH]c12", "7-chloro-1H-pyrrolo[2,3-c]pyridine"),
    # 1H-pyrrolo[3,2-c]pyridine: L4 -> idx7, L6 -> idx0, L7 -> idx1
    ("Clc1nccc2[nH]ccc12", "4-chloro-1H-pyrrolo[3,2-c]pyridine"),
    ("Clc1cc2[nH]ccc2cn1", "6-chloro-1H-pyrrolo[3,2-c]pyridine"),
    ("Clc1cncc2cc[nH]c12", "7-chloro-1H-pyrrolo[3,2-c]pyridine"),
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


def test_indole_still_works() -> None:
    """Sentinel: parent indole (1-N) must still resolve."""
    name = name_smiles("c1ccc2[nH]ccc2c1")
    assert name == "1H-indole"


def test_no_naming_error_for_pyrrolopyridine() -> None:
    name = name_smiles("c1cnc2[nH]ccc2c1")
    assert name is not None
    assert "NAMING ERROR" not in name
