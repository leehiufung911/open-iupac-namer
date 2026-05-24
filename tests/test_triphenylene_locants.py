"""Regression tests for Stage 13 R13-A-1: triphenylene atom_locants fix.

Triphenylene's curated entry previously had no ``atom_locants`` map,
so the engine emitted "1-chlorotriphenylene" for *every* chloro-
substituted form (the latent symmetric-locant collapse class — same
shape as the dibenzofuran R11-A and anthracene R11-B/R12-A bugs).

Triphenylene has D3h symmetry: the 18 atoms in the canonical RDKit
form ``c1ccc2c(c1)c1ccccc1c1ccccc21`` divide into 4 orbits — peripheral
α-CH (L=1,4,5,8,9,12 ↔ idx {2,5,7,10,13,16}), peripheral β-CH
(L=2,3,6,7,10,11 ↔ idx {0,1,8,9,14,15}), outer junction (L=4a, 4b, 8a,
8b, 12a, 12b ↔ idx {4,6,11,12,17,3}).  The atom_locants map picks one
canonical orientation consistent with the OPSIN chloro-probing.

Stage 13 R13-A-1: ``stage2_fusion_base: False`` keeps the architectural
≤3-ring Stage 2B invariant intact (4-ring scaffold).
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
    td = tempfile.mkdtemp(prefix="r13a1_test_")
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


def test_parent_triphenylene() -> None:
    name = name_smiles("c1ccc2c(c1)c1ccccc1c1ccccc21")
    assert name == "triphenylene"


# ---- chloro-locant verification ------------------------------------


# By D3h symmetry there are exactly 2 distinct substitutable
# positions: 1 (= 4 = 5 = 8 = 9 = 12; α-position) and 2 (= 3 = 6 = 7 =
# 10 = 11; β-position).  OPSIN canonicalises every substituted
# triphenylene to the lowest-locant orbit member.
_MONO_CHLORO_CASES = [
    ("Clc1cccc2c3ccccc3c3ccccc3c12", "1-chlorotriphenylene"),
    ("Clc1ccc2c3ccccc3c3ccccc3c2c1", "2-chlorotriphenylene"),
]


@pytest.mark.parametrize("smi,expected", _MONO_CHLORO_CASES)
def test_mono_chloro_triphenylene(smi: str, expected: str) -> None:
    name = name_smiles(smi)
    assert name == expected, (
        f"for {smi!r} expected {expected!r} but got {name!r}"
    )
    rt = _opsin_roundtrip(name)
    assert rt is not None
    assert _canon(rt) == _canon(smi)


# Multi-substituted forms — verify locant *combinations* don't drift.
_MULTI_SUB_CASES = [
    ("Clc1ccc2c3ccc(Cl)cc3c3ccccc3c2c1", "2,7-dichlorotriphenylene"),
    ("Clc1cccc2c3cccc(Cl)c3c3ccccc3c12", "1,8-dichlorotriphenylene"),
]


@pytest.mark.parametrize("smi,expected", _MULTI_SUB_CASES)
def test_multi_sub_triphenylene(smi: str, expected: str) -> None:
    name = name_smiles(smi)
    rt = _opsin_roundtrip(name)
    assert rt is not None
    assert _canon(rt) == _canon(smi), (
        f"OPSIN round-trip mismatch: input {smi!r} -> name {name!r} -> "
        f"opsin {rt!r} -> canon {_canon(rt)!r} vs target {_canon(smi)!r}"
    )


# ---- regression sentinel -------------------------------------------


def test_no_locant_collapse_on_2chloro() -> None:
    """Pre-R13-A-1, every chloro-triphenylene rendered as
    "1-chlorotriphenylene" — α/β orbits all collapsed.  Now 2-chloro
    must come out as 2-, not 1-."""
    name = name_smiles("Clc1ccc2c3ccccc3c3ccccc3c2c1")
    assert name == "2-chlorotriphenylene"
