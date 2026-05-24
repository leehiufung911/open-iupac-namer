"""Regression tests for Stage 10 R10-A-3: bridgehead-N bicyclic retained names.

Adds curated retained-name entries for four 5,6-fused bicyclic
heterocycles where the shared ring-junction atom is itself an N
(bridgehead N).  The bridgehead N is labeled L4 in IUPAC numbering,
mirroring indolizine's atom_locants pattern.

- imidazo[1,2-a]pyridine    (zolpidem core)
- imidazo[1,5-a]pyridine
- [1,2,4]triazolo[4,3-a]pyridine
- tetrazolo[1,5-a]pyridine

Before R10-A-3 the engine returned ``[NAMING ERROR: No valid naming
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
    td = tempfile.mkdtemp(prefix="r10a3_test_")
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
    ("c1ccn2ccnc2c1", "imidazo[1,2-a]pyridine"),
    ("c1ccn2cncc2c1", "imidazo[1,5-a]pyridine"),
    ("c1ccn2cnnc2c1", "[1,2,4]triazolo[4,3-a]pyridine"),
    ("c1ccn2nnnc2c1", "tetrazolo[1,5-a]pyridine"),
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


# Chloro-substituted forms verified by OPSIN chloro-probing.  All four
# scaffolds share the same 9-atom topology, so the locant maps are
# identical; the chloro probes confirm the substitutable positions
# (2,3 in the 5-ring + 5,6,7,8 in the 6-ring).
_CHLORO_LOCANT_CASES = [
    # imidazo[1,2-a]pyridine: L2 -> idx5, L3 -> idx4, L5 -> idx2,
    #   L6 -> idx1, L7 -> idx0, L8 -> idx8
    ("Clc1cn2ccccc2n1",   "2-chloroimidazo[1,2-a]pyridine"),
    ("Clc1cnc2ccccn12",   "3-chloroimidazo[1,2-a]pyridine"),
    ("Clc1cccc2nccn12",   "5-chloroimidazo[1,2-a]pyridine"),
    ("Clc1ccc2nccn2c1",   "6-chloroimidazo[1,2-a]pyridine"),
    ("Clc1ccn2ccnc2c1",   "7-chloroimidazo[1,2-a]pyridine"),
    ("Clc1cccn2ccnc12",   "8-chloroimidazo[1,2-a]pyridine"),
    # imidazo[1,5-a]pyridine: L1 -> idx6, L3 -> idx4
    ("Clc1ncn2ccccc12",   "1-chloroimidazo[1,5-a]pyridine"),
    ("Clc1ncc2ccccn12",   "3-chloroimidazo[1,5-a]pyridine"),
    ("Clc1ccc2cncn2c1",   "6-chloroimidazo[1,5-a]pyridine"),
    # [1,2,4]triazolo[4,3-a]pyridine: L3 -> idx4.  Engine elides the
    # hyphen before the bracketed fusion descriptor (no '-' between
    # "chloro" and "[" — both forms parse identically via OPSIN).
    ("Clc1nnc2ccccn12",   "3-chloro[1,2,4]triazolo[4,3-a]pyridine"),
    ("Clc1cccc2nncn12",   "5-chloro[1,2,4]triazolo[4,3-a]pyridine"),
    # tetrazolo[1,5-a]pyridine: only the 6-ring carbons substitutable
    ("Clc1cccc2nnnn12",   "5-chlorotetrazolo[1,5-a]pyridine"),
    ("Clc1ccc2nnnn2c1",   "6-chlorotetrazolo[1,5-a]pyridine"),
    ("Clc1ccn2nnnc2c1",   "7-chlorotetrazolo[1,5-a]pyridine"),
    ("Clc1cccn2nnnc12",   "8-chlorotetrazolo[1,5-a]pyridine"),
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


def test_indolizine_still_works() -> None:
    """Sentinel: parent indolizine (the all-C analogue) must still resolve."""
    name = name_smiles("c1ccn2cccc2c1")
    assert name == "indolizine"


def test_no_naming_error_for_imidazopyridine_zolpidem_core() -> None:
    name = name_smiles("c1ccn2ccnc2c1")
    assert name is not None
    assert "NAMING ERROR" not in name
