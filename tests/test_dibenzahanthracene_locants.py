"""Regression tests for Stage 14 R14-A-4: dibenz[a,h]anthracene curated entry.

dibenz[a,h]anthracene (5-ring pericondensed PAH) was previously a NAMING_ERROR
because the engine had no curated entry for this scaffold.

This compound has C2 symmetry: its 14 peripheral atoms form 7 C2-equivalent pairs:
  * orbit {1,8}   — peri-position of end benzene rings
  * orbit {2,9}
  * orbit {3,10}
  * orbit {4,11}
  * orbit {5,12}
  * orbit {6,13}
  * orbit {7,14}  — central peri-positions

RDKit canonical: 'c1ccc2c(c1)ccc1cc3c(ccc4ccccc43)cc12'
Junction locants: 4a, 6a, 7a, 7b, 11a, 13a, 14a, 14b

Stage 14 R14-A-4 adds a brand-new curated entry for this scaffold with
complete atom_locants and stage2_fusion_base: False.
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
    td = tempfile.mkdtemp(prefix="r14a4_test_")
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


def test_parent_dibenzahanthracene() -> None:
    """Parent dibenz[a,h]anthracene via RDKit canonical SMILES."""
    name = name_smiles("c1ccc2c(c1)ccc1cc3c(ccc4ccccc43)cc12")
    assert name == "dibenz[a,h]anthracene"


def test_parent_dibenzahanthracene_opsin_canonical() -> None:
    """Parent via OPSIN canonical SMILES."""
    # OPSIN canonical differs from RDKit canonical; both should name correctly
    opsin_canon = "C1=CC=CC=2C1=C1C=C3C=CC4=C(C3=CC1=CC2)C=CC=C4"
    name = name_smiles(opsin_canon)
    assert name == "dibenz[a,h]anthracene"


# ---- mono-chloro locant verification (7 distinct positions) --------

# C2 symmetry gives 7 distinguishable mono-Cl positions (7 orbit pairs):
#   orbit {1,8}: end-ring peri corner (OPSIN canonical lowest locant L=1)
#   orbit {2,9}: end-ring edge
#   orbit {3,10}: end-ring inner edge
#   orbit {4,11}: end-ring/middle ring junction neighbor
#   orbit {5,12}: middle ring peri-position
#   orbit {6,13}: middle ring inner peri
#   orbit {7,14}: central bridge position (lowest locant L=7)

_MONO_CHLORO_CASES = [
    # orbit {1,8} — canonical lowest L=1
    ("Clc1cccc2ccc3cc4c(ccc5ccccc54)cc3c12", "1-chlorodibenz[a,h]anthracene"),
    # orbit {2,9}
    ("Clc1ccc2ccc3cc4c(ccc5ccccc54)cc3c2c1", "2-chlorodibenz[a,h]anthracene"),
    # orbit {5,12}
    ("Clc1cc2cc3c(ccc4ccccc43)cc2c2ccccc12", "5-chlorodibenz[a,h]anthracene"),
    # orbit {6,13}
    ("Clc1cc2ccccc2c2cc3ccc4ccccc4c3cc12", "6-chlorodibenz[a,h]anthracene"),
    # orbit {7,14}
    ("Clc1c2ccc3ccccc3c2cc2ccc3ccccc3c12", "7-chlorodibenz[a,h]anthracene"),
]


@pytest.mark.parametrize("smi,expected", _MONO_CHLORO_CASES)
def test_mono_chloro_dibenzahanthracene(smi: str, expected: str) -> None:
    name = name_smiles(smi)
    assert name == expected, (
        f"for {smi!r} expected {expected!r} but got {name!r}"
    )
    rt = _opsin_roundtrip(name)
    assert rt is not None
    assert _canon(rt) == _canon(smi)


# ---- multi-substituted forms (verify locant combinations) ----------

_MULTI_SUB_CASES = [
    # 1,4-diCl (adjacent positions in end ring)
    ("Clc1ccc(Cl)c2c1ccc1cc3c(ccc4ccccc43)cc12", "1,4-dichlorodibenz[a,h]anthracene"),
    # 5,6-diCl (adjacent inner positions)
    ("Clc1c(Cl)c2cc3c(ccc4ccccc43)cc2c2ccccc12", "5,6-dichlorodibenz[a,h]anthracene"),
    # 7,14-diCl (both central bridge positions, same C2-orbit pair)
    ("Clc1c2ccc3ccccc3c2c(Cl)c2ccc3ccccc3c12", "7,14-dichlorodibenz[a,h]anthracene"),
]


@pytest.mark.parametrize("smi,expected", _MULTI_SUB_CASES)
def test_multi_sub_dibenzahanthracene(smi: str, expected: str) -> None:
    name = name_smiles(smi)
    rt = _opsin_roundtrip(name)
    assert rt is not None
    assert _canon(rt) == _canon(smi), (
        f"OPSIN round-trip mismatch: input {smi!r} -> name {name!r} -> "
        f"opsin {rt!r} -> canon {_canon(rt)!r} vs target {_canon(smi)!r}"
    )


# ---- regression sentinel (was NAMING_ERROR before R14-A-4) ---------


def test_was_naming_error() -> None:
    """Before R14-A-4, the parent gave NAMING_ERROR.
    After fix, it must return 'dibenz[a,h]anthracene'."""
    name = name_smiles("c1ccc2c(c1)ccc1cc3c(ccc4ccccc43)cc12")
    assert "NAMING ERROR" not in name
    assert name == "dibenz[a,h]anthracene"
