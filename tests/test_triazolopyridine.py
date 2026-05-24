"""Regression tests for Stage 10 R10-A-4: [1,2,3]triazolo[4,5-b]pyridine.

Adds curated retained-name entries for the two tautomers:

- 1H-[1,2,3]triazolo[4,5-b]pyridine
- 3H-[1,2,3]triazolo[4,5-b]pyridine

5,6-fused with 3 N's in the 5-ring + 1 pyridine N.  Same skeletal
topology as imidazo[4,5-b]pyridine; the additional N just shifts the
ring-NH position.

Before R10-A-4 the engine returned ``[NAMING ERROR: No valid naming
plan found ...]`` for both tautomers.
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
    td = tempfile.mkdtemp(prefix="r10a4_test_")
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
    ("c1cnc2nn[nH]c2c1", "1H-[1,2,3]triazolo[4,5-b]pyridine"),
    ("c1cnc2[nH]nnc2c1", "3H-[1,2,3]triazolo[4,5-b]pyridine"),
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


# Chloro-substituted forms.  Engine elides hyphen before bracket.
_CHLORO_LOCANT_CASES = [
    # 1H form: L5 -> idx1, L6 -> idx0, L7 -> idx8
    ("Clc1ccc2[nH]nnc2n1", "5-chloro-1H-[1,2,3]triazolo[4,5-b]pyridine"),
    ("Clc1cnc2nn[nH]c2c1", "6-chloro-1H-[1,2,3]triazolo[4,5-b]pyridine"),
    ("Clc1ccnc2nn[nH]c12", "7-chloro-1H-[1,2,3]triazolo[4,5-b]pyridine"),
    # 3H form
    ("Clc1ccc2nn[nH]c2n1", "5-chloro-3H-[1,2,3]triazolo[4,5-b]pyridine"),
    ("Clc1cnc2[nH]nnc2c1", "6-chloro-3H-[1,2,3]triazolo[4,5-b]pyridine"),
    ("Clc1ccnc2[nH]nnc12", "7-chloro-3H-[1,2,3]triazolo[4,5-b]pyridine"),
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


def test_imidazo45b_pyridine_still_works() -> None:
    """Sentinel: same-topology imidazo[4,5-b]pyridine (R10-A-1) intact."""
    name = name_smiles("c1cnc2nc[nH]c2c1")
    assert name == "1H-imidazo[4,5-b]pyridine"


def test_no_naming_error_for_triazolopyridine() -> None:
    name = name_smiles("c1cnc2nn[nH]c2c1")
    assert name is not None
    assert "NAMING ERROR" not in name
