"""Regression tests for Stage 10 R10-A-1: imidazo[4,5-?]pyridine retained names.

Adds curated retained-name entries for the four imidazopyridine tautomers
(5,6-fused with 3 N in the imidazo ring + 1 pyridine N):

- 1H-imidazo[4,5-b]pyridine
- 3H-imidazo[4,5-b]pyridine
- 1H-imidazo[4,5-c]pyridine
- 3H-imidazo[4,5-c]pyridine

Before R10-A-1 the engine returned ``[NAMING ERROR: No valid naming plan
found ...]`` for these scaffolds.  The fix is purely a curated
retained-name lookup extension (``data_loader._RING_CURATED_SMILES``);
no perception or assembly changes were needed.
"""
from __future__ import annotations

import os
import shutil
import tempfile

import pytest
from rdkit import Chem

from iupac_namer.engine import name_smiles


# ---- helpers --------------------------------------------------------


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
    td = tempfile.mkdtemp(prefix="r10a1_test_")
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


_PARENT_CASES = [
    ("c1cnc2nc[nH]c2c1", "1H-imidazo[4,5-b]pyridine"),
    ("c1cnc2[nH]cnc2c1", "3H-imidazo[4,5-b]pyridine"),
    ("c1cc2[nH]cnc2cn1", "1H-imidazo[4,5-c]pyridine"),
    ("c1cc2nc[nH]c2cn1", "3H-imidazo[4,5-c]pyridine"),
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


# ---- substituent-locant verification --------------------------------


# Chloro-substituted forms verified via OPSIN chloro-probing.  Locants
# pin the curated atom_locants map.
_CHLORO_LOCANT_CASES = [
    # 1H-imidazo[4,5-b]pyridine: L5 -> idx1, L6 -> idx0, L7 -> idx8
    ("Clc1ccc2[nH]cnc2n1", "5-chloro-1H-imidazo[4,5-b]pyridine"),
    ("Clc1cnc2nc[nH]c2c1", "6-chloro-1H-imidazo[4,5-b]pyridine"),
    ("Clc1ccnc2nc[nH]c12", "7-chloro-1H-imidazo[4,5-b]pyridine"),
    # 3H-imidazo[4,5-b]pyridine: same map
    ("Clc1ccc2nc[nH]c2n1", "5-chloro-3H-imidazo[4,5-b]pyridine"),
    ("Clc1cnc2[nH]cnc2c1", "6-chloro-3H-imidazo[4,5-b]pyridine"),
    # 1H-imidazo[4,5-c]pyridine: L4 -> idx7, L6 -> idx0, L7 -> idx1
    ("Clc1nccc2[nH]cnc12", "4-chloro-1H-imidazo[4,5-c]pyridine"),
    ("Clc1cc2[nH]cnc2cn1", "6-chloro-1H-imidazo[4,5-c]pyridine"),
    # 3H-imidazo[4,5-c]pyridine
    ("Clc1nccc2nc[nH]c12", "4-chloro-3H-imidazo[4,5-c]pyridine"),
    ("Clc1cc2nc[nH]c2cn1", "6-chloro-3H-imidazo[4,5-c]pyridine"),
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


# ---- regression sentinels ------------------------------------------


def test_benzimidazole_still_works() -> None:
    """Sentinel: 5,6-fused benzo-N path must still resolve."""
    name = name_smiles("c1ccc2[nH]cnc2c1")
    assert name == "1H-benzimidazole"


def test_benzotriazole_still_works() -> None:
    """Sentinel: R9-B benzotriazole entries must not regress."""
    name = name_smiles("c1ccc2[nH]nnc2c1")
    assert name == "1H-benzotriazole"


def test_no_naming_error_for_imidazopyridine() -> None:
    name = name_smiles("c1cnc2nc[nH]c2c1")
    assert name is not None
    assert "NAMING ERROR" not in name
