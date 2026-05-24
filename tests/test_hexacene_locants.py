"""Regression tests for Stage 14 R14-A-3: hexacene atom_locants addition.

Hexacene has D2h symmetry: its 26 atoms divide into 4 peripheral orbits and
5 junction orbits under D2h:
  * orbit {1,4,9,12}   (end-ring corners)  — lowest locant → 1
  * orbit {2,3,10,11}  (end-ring edges)    — lowest locant → 2
  * orbit {5,8,13,16}  (2nd-ring peri)     — lowest locant → 5
  * orbit {6,7,14,15}  (3rd-ring peri)     — lowest locant → 6

RDKit canonical: 'c1ccc2cc3cc4cc5cc6ccccc6cc5cc4cc3cc2c1'
Full locant sequence: 1,2,3,4,4a,5,5a,6,6a,7,7a,8,8a,9,10,11,12,13,14,15,16,16a,...

Before R14-A-3, all mono-chloro hexacene derivatives collapsed to
'1-chlorohexacene' (latent symmetric-locant collapse, same class as R14-A-2
for pentacene).
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
    td = tempfile.mkdtemp(prefix="r14a3_test_")
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


def test_parent_hexacene() -> None:
    """Parent hexacene via RDKit canonical SMILES (the curated entry key)."""
    name = name_smiles("c1ccc2cc3cc4cc5cc6ccccc6cc5cc4cc3cc2c1")
    assert name == "hexacene"


# ---- mono-chloro locant verification (4 distinct positions) --------

# D2h symmetry gives 4 distinguishable mono-Cl positions:
#   1-chlorohexacene: orbit {1,4,9,12}
#   2-chlorohexacene: orbit {2,3,10,11}
#   5-chlorohexacene: orbit {5,8,13,16}
#   6-chlorohexacene: orbit {6,7,14,15}

_MONO_CHLORO_CASES = [
    # orbit {1,4,9,12} — canonical lowest locant L=1
    ("Clc1cccc2cc3cc4cc5cc6ccccc6cc5cc4cc3cc12", "1-chlorohexacene"),
    # orbit {2,3,10,11} — canonical lowest locant L=2
    ("Clc1ccc2cc3cc4cc5cc6ccccc6cc5cc4cc3cc2c1", "2-chlorohexacene"),
    # orbit {5,8,13,16} — canonical lowest locant L=5
    ("Clc1c2ccccc2cc2cc3cc4cc5ccccc5cc4cc3cc12", "5-chlorohexacene"),
    # orbit {6,7,14,15} — canonical lowest locant L=6
    ("Clc1c2cc3ccccc3cc2cc2cc3cc4ccccc4cc3cc12", "6-chlorohexacene"),
]


@pytest.mark.parametrize("smi,expected", _MONO_CHLORO_CASES)
def test_mono_chloro_hexacene(smi: str, expected: str) -> None:
    name = name_smiles(smi)
    assert name == expected, (
        f"for {smi!r} expected {expected!r} but got {name!r}"
    )
    rt = _opsin_roundtrip(name)
    assert rt is not None
    assert _canon(rt) == _canon(smi)


# ---- multi-substituted forms (verify locant combinations) ----------

_MULTI_SUB_CASES = [
    # 1,4-diCl (within orbit {1,4,9,12}, same end ring)
    ("Clc1ccc(Cl)c2cc3cc4cc5cc6ccccc6cc5cc4cc3cc12", "1,4-dichlorohexacene"),
    # 5,6-diCl (adjacent positions crossing orbit boundary)
    ("Clc1c2ccccc2cc2cc3cc4cc5ccccc5cc4cc3c(Cl)c12", "5,6-dichlorohexacene"),
]


@pytest.mark.parametrize("smi,expected", _MULTI_SUB_CASES)
def test_multi_sub_hexacene(smi: str, expected: str) -> None:
    name = name_smiles(smi)
    rt = _opsin_roundtrip(name)
    assert rt is not None
    assert _canon(rt) == _canon(smi), (
        f"OPSIN round-trip mismatch: input {smi!r} -> name {name!r} -> "
        f"opsin {rt!r} -> canon {_canon(rt)!r} vs target {_canon(smi)!r}"
    )


# ---- regression sentinels -------------------------------------------


def test_no_locant_collapse_2chloro() -> None:
    """Pre-R14-A-3, every chloro-hexacene rendered as '1-chlorohexacene'.
    After fix, orbit {2,3,10,11} must come out as '2-chlorohexacene'."""
    name = name_smiles("Clc1ccc2cc3cc4cc5cc6ccccc6cc5cc4cc3cc2c1")
    assert name == "2-chlorohexacene", (
        f"Locant collapse regression: expected '2-chlorohexacene', got {name!r}"
    )


def test_no_locant_collapse_5chloro() -> None:
    """Pre-R14-A-3, orbit {5,8,13,16} collapsed to '1-chlorohexacene'.
    After fix, it must come out as '5-chlorohexacene'."""
    name = name_smiles("Clc1c2ccccc2cc2cc3cc4cc5ccccc5cc4cc3cc12")
    assert name == "5-chlorohexacene", (
        f"Locant collapse regression: expected '5-chlorohexacene', got {name!r}"
    )


def test_no_locant_collapse_6chloro() -> None:
    """Pre-R14-A-3, orbit {6,7,14,15} collapsed to '1-chlorohexacene'.
    After fix, it must come out as '6-chlorohexacene'."""
    name = name_smiles("Clc1c2cc3ccccc3cc2cc2cc3cc4ccccc4cc3cc12")
    assert name == "6-chlorohexacene", (
        f"Locant collapse regression: expected '6-chlorohexacene', got {name!r}"
    )
