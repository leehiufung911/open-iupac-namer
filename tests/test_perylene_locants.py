"""Regression tests for Stage 13 R13-A-2: perylene atom_locants fix.

Pre-fix the engine emitted "chloroperylene" with no locant for
2-chloro and 3-chloro forms — atom_locants returned nothing for
those positions and assembly dropped the locant entirely.

Perylene is a 6,6,6,6,6 peri-fused PAH (two naphthalenes joined at
their peri positions, sharing a central 6-ring) with D2h symmetry.
20 atoms in canonical 'c1cc2cccc3c4cccc5cccc(c(c1)c23)c54' = 12
peripheral CH (locants 1,2,3,4,5,6,7,8,9,10,11,12) + 6 outer ring-
junction (3a, 6a, 6b, 9a, 12a, 12b) + 2 inner peri atoms (12c, 12d).

Stage 13 R13-A-2: ``stage2_fusion_base: False`` keeps the
architectural ≤3-ring Stage 2B invariant intact (5-ring scaffold).
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
    td = tempfile.mkdtemp(prefix="r13a2_test_")
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


def test_parent_perylene() -> None:
    name = name_smiles("c1cc2cccc3c4cccc5cccc(c(c1)c23)c54")
    assert name == "perylene"


# ---- chloro-locant verification ------------------------------------


# By D2h symmetry there are 3 distinct substitutable positions:
# 1 (= 6 = 7 = 12), 2 (= 5 = 8 = 11), 3 (= 4 = 9 = 10).  OPSIN
# canonicalises each substituted perylene to the lowest-locant orbit.
_MONO_CHLORO_CASES = [
    ("Clc1ccc2cccc3c4cccc5cccc(c1c23)c54", "1-chloroperylene"),
    ("Clc1cc2cccc3c4cccc5cccc(c(c1)c23)c54", "2-chloroperylene"),
    ("Clc1ccc2c3cccc4cccc(c5cccc1c52)c43", "3-chloroperylene"),
]


@pytest.mark.parametrize("smi,expected", _MONO_CHLORO_CASES)
def test_mono_chloro_perylene(smi: str, expected: str) -> None:
    name = name_smiles(smi)
    assert name == expected, (
        f"for {smi!r} expected {expected!r} but got {name!r}"
    )
    rt = _opsin_roundtrip(name)
    assert rt is not None
    assert _canon(rt) == _canon(smi)


# Multi-substituted forms — verify locant *combinations* don't drift.
_MULTI_SUB_CASES = [
    ("Clc1ccc2cccc3c4c(Cl)ccc5cccc(c1c23)c54", "1,7-dichloroperylene"),
    ("Clc1ccc2c3cccc4c(Cl)ccc(c5cccc1c52)c43", "3,9-dichloroperylene"),
]


@pytest.mark.parametrize("smi,expected", _MULTI_SUB_CASES)
def test_multi_sub_perylene(smi: str, expected: str) -> None:
    name = name_smiles(smi)
    rt = _opsin_roundtrip(name)
    assert rt is not None
    assert _canon(rt) == _canon(smi), (
        f"OPSIN round-trip mismatch: input {smi!r} -> name {name!r} -> "
        f"opsin {rt!r} -> canon {_canon(rt)!r} vs target {_canon(smi)!r}"
    )


# ---- regression sentinel -------------------------------------------


def test_no_missing_locant_on_2chloro() -> None:
    """Pre-R13-A-2, 2-chloroperylene rendered as plain 'chloroperylene'
    — assembly dropped the locant because atom_locants returned nothing."""
    name = name_smiles("Clc1cc2cccc3c4cccc5cccc(c(c1)c23)c54")
    assert name == "2-chloroperylene"
    assert "-" in name, "locant must be present (not 'chloroperylene')"


def test_no_missing_locant_on_3chloro() -> None:
    """Pre-R13-A-2, 3-chloroperylene also dropped the locant."""
    name = name_smiles("Clc1ccc2c3cccc4cccc(c5cccc1c52)c43")
    assert name == "3-chloroperylene"
