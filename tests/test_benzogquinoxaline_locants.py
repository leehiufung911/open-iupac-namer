"""Regression tests for Stage 13 R13-A-4: benzo[g]quinoxaline curated entry.

Pre-fix the scaffold ``c1ccc2cc3nccnc3cc2c1`` (benzo[g]quinoxaline,
quinoxaline with an extra benzene fused at the [g] = 6-7 bond; 3-ring
anthracene-like topology with a central pyrazine) returned ``[NAMING
ERROR: No valid naming plan found ...]`` — there was no curated
entry, and the engine couldn't synthesise the benzo-fused N-
heterocyclic name from primitives.

The 14 atoms split as 8 substitutable CH (locants 2, 3, 5, 6, 7, 8,
9, 10) + 2 N (locants 1, 4) + 4 ring-junction (4a, 5a, 9a, 10a).
C2v symmetry through the N1-N4 axis collapses {2,3}, {5,10}, {6,9},
{7,8} pairs.

Stage 13 R13-A-4: ``stage2_fusion_base: False`` keeps the
architectural ≤3-ring Stage 2B invariant intact.
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
    td = tempfile.mkdtemp(prefix="r13a4_test_")
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


def test_parent_benzogquinoxaline() -> None:
    name = name_smiles("c1ccc2cc3nccnc3cc2c1")
    assert name == "benzo[g]quinoxaline"


# ---- chloro-locant verification ------------------------------------


# By C2v symmetry there are 4 distinct substitutable CH positions:
# 2 (= 3), 5 (= 10), 6 (= 9), 7 (= 8).  OPSIN canonicalises every
# substituted form to the lowest-locant orbit member.
_MONO_CHLORO_CASES = [
    ("Clc1cnc2cc3ccccc3cc2n1", "2-chlorobenzo[g]quinoxaline"),
    ("Clc1c2ccccc2cc2nccnc12", "5-chlorobenzo[g]quinoxaline"),
    ("Clc1cccc2cc3nccnc3cc12", "6-chlorobenzo[g]quinoxaline"),
    ("Clc1ccc2cc3nccnc3cc2c1", "7-chlorobenzo[g]quinoxaline"),
]


@pytest.mark.parametrize("smi,expected", _MONO_CHLORO_CASES)
def test_mono_chloro_benzogquinoxaline(smi: str, expected: str) -> None:
    name = name_smiles(smi)
    assert name == expected, (
        f"for {smi!r} expected {expected!r} but got {name!r}"
    )
    rt = _opsin_roundtrip(name)
    assert rt is not None
    assert _canon(rt) == _canon(smi)


# Multi-substituted forms — verify locant *combinations* don't drift.
_MULTI_SUB_CASES = [
    ("Clc1nc2cc3ccccc3cc2nc1Cl", "2,3-dichlorobenzo[g]quinoxaline"),
    ("Clc1c2ccccc2c(Cl)c2nccnc12", "5,10-dichlorobenzo[g]quinoxaline"),
]


@pytest.mark.parametrize("smi,expected", _MULTI_SUB_CASES)
def test_multi_sub_benzogquinoxaline(smi: str, expected: str) -> None:
    name = name_smiles(smi)
    rt = _opsin_roundtrip(name)
    assert rt is not None
    assert _canon(rt) == _canon(smi), (
        f"OPSIN round-trip mismatch: input {smi!r} -> name {name!r} -> "
        f"opsin {rt!r} -> canon {_canon(rt)!r} vs target {_canon(smi)!r}"
    )


# ---- regression sentinel -------------------------------------------


def test_no_naming_error_on_parent() -> None:
    """Pre-R13-A-4 the parent returned ``[NAMING ERROR: ...]``."""
    name = name_smiles("c1ccc2cc3nccnc3cc2c1")
    assert "NAMING ERROR" not in name
    assert name == "benzo[g]quinoxaline"


def test_no_naming_error_on_substituted() -> None:
    """Pre-R13-A-4 a substituted form would also fail."""
    name = name_smiles("Clc1nc2cc3ccccc3cc2nc1Cl")
    assert "NAMING ERROR" not in name
    assert name == "2,3-dichlorobenzo[g]quinoxaline"
