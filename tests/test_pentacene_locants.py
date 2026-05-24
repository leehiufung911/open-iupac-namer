"""Regression tests for Stage 14 R14-A-2: pentacene atom_locants addition.

Pentacene has D2h symmetry: its 22 atoms divide into 4 peripheral orbits and
4 junction-pair orbits under D2h:
  * orbit {1,4,8,11}  (end-ring corners) — lowest locant → 1
  * orbit {2,3,9,10}  (end-ring edges)   — lowest locant → 2
  * orbit {5,7,12,14} (inner-ring peri)  — lowest locant → 5
  * orbit {6,13}      (center positions) — lowest locant → 6

RDKit canonical: 'c1ccc2cc3cc4cc5ccccc5cc4cc3cc2c1'
Full locant sequence: 1,2,3,4,4a,5,5a,6,6a,7,7a,8,9,10,11,11a,12,12a,13,13a,14,14a

Before R14-A-2, all mono-chloro pentacene derivatives collapsed to
'1-chloropentacene' (latent symmetric-locant collapse, same class as R14-A-1
for pyrene and Stage 13 R13-A-1 for triphenylene).
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
    td = tempfile.mkdtemp(prefix="r14a2_test_")
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


def test_parent_pentacene() -> None:
    """Parent pentacene via RDKit canonical SMILES (the curated entry key)."""
    name = name_smiles("c1ccc2cc3cc4cc5ccccc5cc4cc3cc2c1")
    assert name == "pentacene"


# ---- mono-chloro locant verification (4 distinct positions) --------

# D2h symmetry gives 4 distinguishable mono-Cl positions:
#   1-chloropentacene: orbit {1,4,8,11}
#   2-chloropentacene: orbit {2,3,9,10}
#   5-chloropentacene: orbit {5,7,12,14}
#   6-chloropentacene: orbit {6,13}

_MONO_CHLORO_CASES = [
    # orbit {1,4,8,11} — canonical lowest locant L=1
    ("Clc1cccc2cc3cc4cc5ccccc5cc4cc3cc12", "1-chloropentacene"),
    # orbit {2,3,9,10} — canonical lowest locant L=2
    ("Clc1ccc2cc3cc4cc5ccccc5cc4cc3cc2c1", "2-chloropentacene"),
    # orbit {5,7,12,14} — canonical lowest locant L=5
    ("Clc1c2ccccc2cc2cc3cc4ccccc4cc3cc12", "5-chloropentacene"),
    # orbit {6,13} — canonical lowest locant L=6
    ("Clc1c2cc3ccccc3cc2cc2cc3ccccc3cc12", "6-chloropentacene"),
]


@pytest.mark.parametrize("smi,expected", _MONO_CHLORO_CASES)
def test_mono_chloro_pentacene(smi: str, expected: str) -> None:
    name = name_smiles(smi)
    assert name == expected, (
        f"for {smi!r} expected {expected!r} but got {name!r}"
    )
    rt = _opsin_roundtrip(name)
    assert rt is not None
    assert _canon(rt) == _canon(smi)


# ---- multi-substituted forms (verify locant combinations) ----------

_MULTI_SUB_CASES = [
    # 1,2-diCl (adjacent positions in end ring, cross-orbit)
    ("Clc1ccc2cc3cc4cc5ccccc5cc4cc3cc2c1Cl", "1,2-dichloropentacene"),
    # 5,6-diCl (adjacent positions crossing orbit boundary, inner ring area)
    ("Clc1c2ccccc2cc2cc3cc4ccccc4cc3c(Cl)c12", "5,6-dichloropentacene"),
    # 6,13-diCl (both members of center orbit)
    ("Clc1c2cc3ccccc3cc2c(Cl)c2cc3ccccc3cc12", "6,13-dichloropentacene"),
]


@pytest.mark.parametrize("smi,expected", _MULTI_SUB_CASES)
def test_multi_sub_pentacene(smi: str, expected: str) -> None:
    name = name_smiles(smi)
    rt = _opsin_roundtrip(name)
    assert rt is not None
    assert _canon(rt) == _canon(smi), (
        f"OPSIN round-trip mismatch: input {smi!r} -> name {name!r} -> "
        f"opsin {rt!r} -> canon {_canon(rt)!r} vs target {_canon(smi)!r}"
    )


# ---- regression sentinels -------------------------------------------


def test_no_locant_collapse_2chloro() -> None:
    """Pre-R14-A-2, every chloro-pentacene rendered as '1-chloropentacene'.
    After fix, orbit {2,3,9,10} must come out as '2-chloropentacene'."""
    name = name_smiles("Clc1ccc2cc3cc4cc5ccccc5cc4cc3cc2c1")
    assert name == "2-chloropentacene", (
        f"Locant collapse regression: expected '2-chloropentacene', got {name!r}"
    )


def test_no_locant_collapse_5chloro() -> None:
    """Pre-R14-A-2, orbit {5,7,12,14} collapsed to '1-chloropentacene'.
    After fix, it must come out as '5-chloropentacene'."""
    name = name_smiles("Clc1c2ccccc2cc2cc3cc4ccccc4cc3cc12")
    assert name == "5-chloropentacene", (
        f"Locant collapse regression: expected '5-chloropentacene', got {name!r}"
    )


def test_no_locant_collapse_6chloro() -> None:
    """Pre-R14-A-2, center orbit {6,13} collapsed to '1-chloropentacene'.
    After fix, it must come out as '6-chloropentacene'."""
    name = name_smiles("Clc1c2cc3ccccc3cc2cc2cc3ccccc3cc12")
    assert name == "6-chloropentacene", (
        f"Locant collapse regression: expected '6-chloropentacene', got {name!r}"
    )
