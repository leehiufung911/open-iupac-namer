"""Regression tests for Stage 13 R13-A-3: benz[a]anthracene curated entry.

Pre-fix the scaffold ``c1ccc2cc3c(ccc4ccccc43)cc2c1`` (benz[a]anthracene,
the angular 6,6,6,6 tetracyclic PAH) returned ``[NAMING ERROR: No
valid naming plan found ...]`` — there was no curated retained-name
entry, and the fused-ring path couldn't synthesise the angular
tetracyclic name from primitives.

Benz[a]anthracene = anthracene with an extra benzene fused at the
[a] bond.  No symmetry; every periphery atom is distinct.  18 atoms
in the canonical = 12 peripheral CH (locants 1..12) + 6 ring-
junction (4a, 6a, 7a, 11a, 12a, 12b).

Stage 13 R13-A-3: ``stage2_fusion_base: False`` keeps the
architectural ≤3-ring Stage 2B invariant intact (4-ring scaffold).
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
    td = tempfile.mkdtemp(prefix="r13a3_test_")
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


def test_parent_benzaanthracene() -> None:
    name = name_smiles("c1ccc2cc3c(ccc4ccccc43)cc2c1")
    assert name == "benz[a]anthracene"


# ---- chloro-locant verification ------------------------------------


# Benz[a]anthracene has no symmetry — every periphery atom is
# distinct.  Probe the four corner positions plus a central one.
_MONO_CHLORO_CASES = [
    ("Clc1cccc2ccc3cc4ccccc4cc3c12",  "1-chlorobenz[a]anthracene"),
    ("Clc1ccc2ccc3cc4ccccc4cc3c2c1",  "2-chlorobenz[a]anthracene"),
    ("Clc1ccc2c(ccc3cc4ccccc4cc32)c1", "3-chlorobenz[a]anthracene"),
    ("Clc1cc2cc3ccccc3cc2c2ccccc12",  "5-chlorobenz[a]anthracene"),
    ("Clc1c2ccccc2cc2c1ccc1ccccc12",  "7-chlorobenz[a]anthracene"),
    ("Clc1c2ccccc2cc2ccc3ccccc3c12",  "12-chlorobenz[a]anthracene"),
]


@pytest.mark.parametrize("smi,expected", _MONO_CHLORO_CASES)
def test_mono_chloro_benzaanthracene(smi: str, expected: str) -> None:
    name = name_smiles(smi)
    assert name == expected, (
        f"for {smi!r} expected {expected!r} but got {name!r}"
    )
    rt = _opsin_roundtrip(name)
    assert rt is not None
    assert _canon(rt) == _canon(smi)


# Multi-substituted forms — verify locant *combinations* don't drift.
# 7,12-dimethyl is the well-known carcinogen DMBA.
_MULTI_SUB_CASES = [
    ("Cc1c2ccccc2c(C)c2c1ccc1ccccc12", "7,12-dimethylbenz[a]anthracene"),
    ("Clc1c(Cl)c2cc3ccccc3cc2c2ccccc12", "5,6-dichlorobenz[a]anthracene"),
]


@pytest.mark.parametrize("smi,expected", _MULTI_SUB_CASES)
def test_multi_sub_benzaanthracene(smi: str, expected: str) -> None:
    name = name_smiles(smi)
    rt = _opsin_roundtrip(name)
    assert rt is not None
    assert _canon(rt) == _canon(smi), (
        f"OPSIN round-trip mismatch: input {smi!r} -> name {name!r} -> "
        f"opsin {rt!r} -> canon {_canon(rt)!r} vs target {_canon(smi)!r}"
    )


# ---- regression sentinel -------------------------------------------


def test_no_naming_error_on_parent() -> None:
    """Pre-R13-A-3 the parent returned ``[NAMING ERROR: ...]``."""
    name = name_smiles("c1ccc2cc3c(ccc4ccccc43)cc2c1")
    assert "NAMING ERROR" not in name
    assert name == "benz[a]anthracene"


def test_no_naming_error_on_substituted() -> None:
    """Pre-R13-A-3 a substituted form would also fail."""
    name = name_smiles("Cc1c2ccccc2c(C)c2c1ccc1ccccc12")
    assert "NAMING ERROR" not in name
    assert name == "7,12-dimethylbenz[a]anthracene"
