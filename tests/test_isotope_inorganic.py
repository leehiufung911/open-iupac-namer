"""Tests for Stage 17 R17-A: bare-element isotope curated lookup.

Pre-fix: ``[2H]O`` (HOD), ``[2H]O[2H]`` (D2O), ``[3H]O`` (HTO),
``[15NH3]`` (15N-ammonia) and similar bare-element isotopologues fell
through the regular plan-search path and emitted NAMING_ERROR.  The
engine's _INORGANIC_CURATED_SMILES table only matched the unisotoped
canonical SMILES (``"O"`` / ``"N"``), so isotope-bearing variants
weren't recognised.

R17-A adds explicit isotopologue entries to ``_INORGANIC_CURATED_SMILES``
keyed by canonical SMILES.  Each entry's name follows OPSIN's accepted
``(MASS_ELEM_COUNT)PARENT`` syntax: ``(2H1)water`` (with explicit count
``1``, since OPSIN rejects ``(2H)water``), ``(2H2)water`` for D2O, etc.

Closing this category of audit failures is bounded by OPSIN's parser:
``[2H][2H]`` (D2) cannot be named because OPSIN rejects all
``(2H...)hydrogen`` / ``(2H...)dihydrogen`` / ``(2H)hydride`` forms.
"""
from __future__ import annotations

import os
import shutil
import tempfile

from rdkit import Chem
from py2opsin import py2opsin

from iupac_namer.engine import name_smiles


def _opsin_rt(name: str) -> str | None:
    if not name:
        return None
    td = tempfile.mkdtemp(prefix="r17a_")
    cwd = os.getcwd()
    try:
        os.chdir(td)
        return py2opsin(name)
    except Exception:
        return None
    finally:
        os.chdir(cwd)
        try:
            shutil.rmtree(td)
        except Exception:
            pass


def _canon(s: str | None) -> str | None:
    m = Chem.MolFromSmiles(s) if s else None
    return Chem.MolToSmiles(m) if m else None


def test_hod_roundtrips() -> None:
    """``[2H]O`` (HOD) → ``(2H1)water``."""
    smi = "[2H]O"
    name = name_smiles(smi)
    assert name == "(2H1)water", f"got {name!r}"
    assert _canon(_opsin_rt(name)) == _canon(smi)


def test_d2o_roundtrips() -> None:
    """``[2H]O[2H]`` (D2O / heavy water) → ``(2H2)water``."""
    smi = "[2H]O[2H]"
    name = name_smiles(smi)
    assert name == "(2H2)water"
    assert _canon(_opsin_rt(name)) == _canon(smi)


def test_hto_roundtrips() -> None:
    """``[3H]O`` (HTO) → ``(3H1)water``."""
    smi = "[3H]O"
    name = name_smiles(smi)
    assert name == "(3H1)water"
    assert _canon(_opsin_rt(name)) == _canon(smi)


def test_t2o_roundtrips() -> None:
    smi = "[3H]O[3H]"
    name = name_smiles(smi)
    assert name == "(3H2)water"
    assert _canon(_opsin_rt(name)) == _canon(smi)


def test_dto_roundtrips() -> None:
    smi = "[2H]O[3H]"
    name = name_smiles(smi)
    assert name == "(2H1,3H1)water"
    assert _canon(_opsin_rt(name)) == _canon(smi)


def test_15n_ammonia_roundtrips() -> None:
    """``[15NH3]`` → ``(15N)ammonia``."""
    smi = "[15NH3]"
    name = name_smiles(smi)
    assert name == "(15N)ammonia"
    assert _canon(_opsin_rt(name)) == _canon(smi)


def test_unisotoped_water_unaffected() -> None:
    """Control: regular ``O`` still emits ``water``."""
    assert name_smiles("O") == "water"


def test_unisotoped_ammonia_unaffected() -> None:
    """Control: regular ``N`` still emits ``ammonia``."""
    assert name_smiles("N") == "ammonia"
