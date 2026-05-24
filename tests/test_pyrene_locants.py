"""Regression tests for Stage 14 R14-A-1: pyrene atom_locants addition.

Pyrene's entry in the OPSIN-extracted rings table (rings_from_opsin.json)
had no ``atom_locants`` map, so the engine emitted "1-chloropyrene" for
*every* chloro-substituted form (the latent symmetric-locant collapse class
— same shape as the Stage 13 R13-A-1 triphenylene fix).

Pyrene has D2h symmetry: the 16 atoms in the canonical RDKit form
``c1cc2ccc3cccc4ccc(c1)c2c34`` divide into 3 peripheral orbits under D2h:
  * α-positions {1,3,6,8} ↔ RDKit idx {13,1,6,8}: all give OPSIN canonical
    "Clc1ccc2ccc3cccc4ccc1c2c34"; IUPAC lowest locant → 1-chloropyrene
  * β-positions {2,7} ↔ RDKit idx {0,7}: IUPAC → 2-chloropyrene
  * γ-positions {4,5,9,10} ↔ RDKit idx {3,4,10,11}: IUPAC → 4-chloropyrene
Plus 6 ring-junction atoms (L=3a, 5a, 8a, 10a, 10b, 10c).

Before this fix, ALL chloro positions collapsed to "1-chloropyrene" because
the retained-lookup had no atom_locants for the SMILES key, so the engine
could not distinguish the 3 symmetry-distinct positions.

Stage 14 R14-A-1: ``stage2_fusion_base: False`` keeps the architectural
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
    td = tempfile.mkdtemp(prefix="r14a1_test_")
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


def test_parent_pyrene_opsin_canonical() -> None:
    """Parent via OPSIN canonical SMILES (from rings_from_opsin.json)."""
    name = name_smiles("c1ccc2ccc3cccc4ccc1c2c34")
    assert name == "pyrene"


def test_parent_pyrene_rdkit_canonical() -> None:
    """Parent via RDKit canonical SMILES (the curated entry key)."""
    name = name_smiles("c1cc2ccc3cccc4ccc(c1)c2c34")
    assert name == "pyrene"


# ---- mono-chloro locant verification (3 distinct positions) --------

# By D2h symmetry, pyrene has exactly 3 distinguishable mono-Cl positions:
#   1-chloropyrene: α-orbit {1,3,6,8} — OPSIN canonical Clc1ccc2ccc3cccc4ccc1c2c34
#   2-chloropyrene: β-orbit {2,7} — OPSIN canonical Clc1cc2ccc3cccc4ccc(c1)c2c34
#   4-chloropyrene: γ-orbit {4,5,9,10} — OPSIN canonical Clc1cc2cccc3ccc4cccc1c4c32

_MONO_CHLORO_CASES = [
    # α-position (orbit {1,3,6,8}) — canonical lowest locant L=1
    ("Clc1ccc2ccc3cccc4ccc1c2c34", "1-chloropyrene"),
    # β-position (orbit {2,7}) — canonical lowest locant L=2
    ("Clc1cc2ccc3cccc4ccc(c1)c2c34", "2-chloropyrene"),
    # γ-position (orbit {4,5,9,10}) — canonical lowest locant L=4
    ("Clc1cc2cccc3ccc4cccc1c4c32", "4-chloropyrene"),
]


@pytest.mark.parametrize("smi,expected", _MONO_CHLORO_CASES)
def test_mono_chloro_pyrene(smi: str, expected: str) -> None:
    name = name_smiles(smi)
    assert name == expected, (
        f"for {smi!r} expected {expected!r} but got {name!r}"
    )
    rt = _opsin_roundtrip(name)
    assert rt is not None
    assert _canon(rt) == _canon(smi)


# ---- multi-substituted forms (verify locant combinations) ----------

# For dichloro compounds crossing symmetry orbits, the lowest-locant set is
# unambiguous.  D2h means {1,2}, {1,4}, {2,7}, {4,5} are each uniquely named.

_MULTI_SUB_CASES = [
    # 1,2-diCl: adjacent α,β pair (same OPSIN canonical as 2,3-diCl by D2h)
    ("Clc1cc2ccc3cccc4ccc(c1Cl)c2c34", "1,2-dichloropyrene"),
    # 1,4-diCl: α,γ cross-orbit (OPSIN canonical Clc1ccc2c(Cl)cc3cccc4ccc1c2c43)
    ("Clc1ccc2c(Cl)cc3cccc4ccc1c2c43", "1,4-dichloropyrene"),
    # 2,7-diCl: both β-orbit members (the unique 2,7 dimer)
    ("Clc1cc2ccc3cc(Cl)cc4ccc(c1)c2c34", "2,7-dichloropyrene"),
    # 4,5-diCl: adjacent pair within γ-orbit
    ("Clc1c(Cl)c2cccc3ccc4cccc1c4c32", "4,5-dichloropyrene"),
]


@pytest.mark.parametrize("smi,expected", _MULTI_SUB_CASES)
def test_multi_sub_pyrene(smi: str, expected: str) -> None:
    name = name_smiles(smi)
    rt = _opsin_roundtrip(name)
    assert rt is not None
    assert _canon(rt) == _canon(smi), (
        f"OPSIN round-trip mismatch: input {smi!r} -> name {name!r} -> "
        f"opsin {rt!r} -> canon {_canon(rt)!r} vs target {_canon(smi)!r}"
    )


# ---- regression sentinel -------------------------------------------


def test_no_locant_collapse_2chloro() -> None:
    """Pre-R14-A-1, every chloro-pyrene rendered as '1-chloropyrene'.
    After fix, the β-position (orbit {2,7}) must come out as '2-chloropyrene',
    not '1-chloropyrene'."""
    name = name_smiles("Clc1cc2ccc3cccc4ccc(c1)c2c34")
    assert name == "2-chloropyrene", (
        f"Locant collapse regression: expected '2-chloropyrene', got {name!r}"
    )


def test_no_locant_collapse_4chloro() -> None:
    """Pre-R14-A-1, γ-position (orbit {4,5,9,10}) collapsed to '1-chloropyrene'.
    After fix, it must come out as '4-chloropyrene'."""
    name = name_smiles("Clc1cc2cccc3ccc4cccc1c4c32")
    assert name == "4-chloropyrene", (
        f"Locant collapse regression: expected '4-chloropyrene', got {name!r}"
    )
