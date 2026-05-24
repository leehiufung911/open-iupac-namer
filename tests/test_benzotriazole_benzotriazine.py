"""Regression tests for Stage 9 R9-B: benzo-fused N-rich aromatic rings.

Adds curated retained-name entries for benzo-fused 5,6 and 6,6 aromatic
heterocycles whose N-count exceeds 2 (so they don't already match the
benzimidazole / cinnoline / quinoxaline patterns that Stage 5/6
introduced):

- 1H-benzotriazole       (5,6 fusion, 3 N in the 5-ring)
- 2H-benzotriazole       (tautomer, NH on N2)
- 1,2,3-benzotriazine    (6,6 fusion, 3 N in the heterocyclic ring)
- 1,2,3,4-benzotetrazine (6,6 fusion, 4 N in the heterocyclic ring)

Before Stage 9 R9-B the engine returned ``[NAMING ERROR: No valid
naming plan found ...]`` for all four.  The fix is purely a curated
retained-name lookup extension (``data_loader._RING_CURATED_SMILES``);
no perception or assembly changes were needed because the existing
benzimidazole/cinnoline machinery already handles the locant/atom_locants
plumbing.

These tests pin both the parent SMILES → name and chloro-substituted
locant assignments via OPSIN round-trip.
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
    """Run OPSIN on `name` from a per-call temp CWD to dodge the
    py2opsin temp-file race when tests run in parallel."""
    if not name:
        return None
    try:
        from py2opsin import py2opsin
    except ImportError:  # pragma: no cover
        pytest.skip("py2opsin not available")
    cwd = os.getcwd()
    td = tempfile.mkdtemp(prefix="r9b_test_")
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


# (input_smiles, expected_name) — input is the canonical SMILES, expected
# is the IUPAC-acceptable name we expect the engine to emit.
_PARENT_CASES = [
    ("c1ccc2[nH]nnc2c1", "1H-benzotriazole"),
    ("c1ccc2n[nH]nc2c1", "2H-benzotriazole"),
    ("c1ccc2nnncc2c1",   "1,2,3-benzotriazine"),
    ("c1ccc2nnnnc2c1",   "1,2,3,4-benzotetrazine"),
]


@pytest.mark.parametrize("smi,expected", _PARENT_CASES)
def test_parent_emits_curated_name(smi: str, expected: str) -> None:
    """Engine emits the curated retained name for the parent skeleton."""
    name = name_smiles(smi)
    assert name == expected, (
        f"for {smi!r} expected {expected!r} but got {name!r}"
    )


@pytest.mark.parametrize("smi,expected", _PARENT_CASES)
def test_parent_round_trips_via_opsin(smi: str, expected: str) -> None:
    """Engine name → OPSIN round-trip canonicalises back to the input."""
    name = name_smiles(smi)
    assert name is not None and "NAMING ERROR" not in name
    rt = _opsin_roundtrip(name)
    assert rt is not None, f"OPSIN rejected emitted name {name!r}"
    assert _canon(rt) == _canon(smi), (
        f"round-trip drift: in={smi!r} -> name={name!r} -> rt={rt!r}"
    )


# ---- substituent-locant verification --------------------------------


# Tuples of (input_smiles, expected_name) for chloro-substituted forms.
# These pin that the curated atom_locants map produces the correct
# locant when a chloro group is attached at a specific ring position.
_CHLORO_LOCANT_CASES = [
    # 1H-benzotriazole chloro positions on the benzo ring
    ("Clc1ccc2[nH]nnc2c1", "5-chloro-1H-benzotriazole"),
    ("Clc1cccc2[nH]nnc12", "4-chloro-1H-benzotriazole"),
    # 1,2,3-benzotriazine chloro position
    ("Clc1cccc2nnncc12",   "5-chloro-1,2,3-benzotriazine"),
]


@pytest.mark.parametrize("smi,expected", _CHLORO_LOCANT_CASES)
def test_chloro_locant_round_trips(smi: str, expected: str) -> None:
    """Chloro-substituted forms render with the correct locant and
    survive OPSIN round-trip (canonical structure preserved)."""
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
    """Sentinel: adding 3-N benzo-fused names must not regress the
    existing 2-N benzimidazole curated entry."""
    name = name_smiles("c1ccc2[nH]cnc2c1")
    assert name == "1H-benzimidazole"


def test_indole_still_works() -> None:
    """Sentinel: parent indole (1-N) must still resolve."""
    name = name_smiles("c1ccc2[nH]ccc2c1")
    assert name == "1H-indole"


def test_quinoxaline_still_works() -> None:
    """Sentinel: 6,6 benzo-diazine path must still resolve."""
    name = name_smiles("c1ccc2nccnc2c1")
    assert name == "quinoxaline"


def test_no_naming_error_for_benzotriazole() -> None:
    """Specific guard against the pre-fix failure mode: the engine
    returned ``[NAMING ERROR: No valid naming plan found ...]`` for
    ``c1ccc2[nH]nnc2c1`` before R9-B."""
    name = name_smiles("c1ccc2[nH]nnc2c1")
    assert name is not None
    assert "NAMING ERROR" not in name
    assert "naming plan" not in name.lower()
