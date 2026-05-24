"""Regression tests for Stage 11 R11-B / Stage 12 R12-A-2: anthracene atom_locants fix.

The anthracene curated retained-name entry previously had no
``atom_locants`` map, so the engine emitted "1-chloroanthracene" for
every chloro-substituted form regardless of position — a symmetric-
locant collapse identical in shape to the dibenzofuran R11-A bug.

Anthracene has D2h symmetry: the 14 atoms in the canonical RDKit form
``c1ccc2cc3ccccc3cc2c1`` divide into 4 orbits, mapped to IUPAC locants
{1, 2, 9, 4a} (and their D2h images {4=1, 5=1, 8=1; 3=2, 6=2, 7=2;
10=9; 8a=4a, 9a=4a, 10a=4a}).  Without the explicit atom_locants map,
the engine couldn't pick the canonical lowest-locant choice consistent
with OPSIN's canonicalisation.

Locants verified via OPSIN chloro-probing of L=1, 2, 9.

Stage 12 R12-A-2: anthracene's curated entry also carries
``stage2_fusion_base: False`` to keep the architectural ≤3-ring Stage 2
invariant intact — see
tests/test_fused_ring_hetero.py::test_stage2_excludes_four_plus_ring_systems.
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
    td = tempfile.mkdtemp(prefix="r11b_test_")
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


def test_parent_anthracene() -> None:
    name = name_smiles("c1ccc2cc3ccccc3cc2c1")
    assert name == "anthracene"


# ---- chloro-locant verification ------------------------------------


# Single-chloro positions: by D2h symmetry there are 3 distinct
# substitutable positions — 1 (= 4 = 5 = 8), 2 (= 3 = 6 = 7), 9 (= 10).
_MONO_CHLORO_CASES = [
    ("Clc1cccc2cc3ccccc3cc12",   "1-chloroanthracene"),
    ("Clc1ccc2cc3ccccc3cc2c1",   "2-chloroanthracene"),
    ("Clc1c2ccccc2cc2ccccc12",   "9-chloroanthracene"),
]


@pytest.mark.parametrize("smi,expected", _MONO_CHLORO_CASES)
def test_mono_chloro_anthracene(smi: str, expected: str) -> None:
    name = name_smiles(smi)
    assert name == expected, (
        f"for {smi!r} expected {expected!r} but got {name!r}"
    )
    rt = _opsin_roundtrip(name)
    assert rt is not None
    assert _canon(rt) == _canon(smi)


# Multi-substituted forms — verify locant *combinations* don't drift.
_MULTI_SUB_CASES = [
    ("Clc1cccc2cc3c(Cl)cccc3cc12",   "1,5-dichloroanthracene"),
    ("Clc1c2ccccc2c(Cl)c2ccccc12",   "9,10-dichloroanthracene"),
    ("Cc1ccc2cc3cc(C)ccc3cc2c1",     "2,6-dimethylanthracene"),
]


@pytest.mark.parametrize("smi,expected", _MULTI_SUB_CASES)
def test_multi_sub_anthracene(smi: str, expected: str) -> None:
    name = name_smiles(smi)
    assert name == expected, (
        f"for {smi!r} expected {expected!r} but got {name!r}"
    )


# ---- regression sentinel -------------------------------------------


def test_no_locant_collapse_on_2chloro() -> None:
    """Specific guard: pre-R11-B 2-chloroanthracene rendered as
    '1-chloroanthracene' — D2h-symmetric collapse to lowest equivalent
    position.  Because anthracene's positions 1 and 2 are NOT equivalent
    under D2h, this would scramble locants on multi-substituted forms.
    """
    name = name_smiles("Clc1ccc2cc3ccccc3cc2c1")
    assert name == "2-chloroanthracene"


def test_no_locant_collapse_on_9chloro() -> None:
    """Pre-R11-B, 9-chloroanthracene rendered as '1-chloroanthracene'
    — the central-ring position 9 was confused with outer position 1."""
    name = name_smiles("Clc1c2ccccc2cc2ccccc12")
    assert name == "9-chloroanthracene"
