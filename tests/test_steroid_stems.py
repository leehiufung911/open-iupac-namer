"""Regression tests for Stage 6 R2-A steroid biochemical stem emission.

Background: ``data/opsin_extracted/retained_names_from_opsin.json`` stores
17 biochemical tetracycle retained stems (``androst``, ``pregn``,
``cholest``, ``estr``, ``gon``, ``campest``, ``ergost``, ``stigmast``,
``poriferast``, ``gorgost``, ``spirost``, ``furost``, ``prost``,
``thrombox``, and their pipe-aliased variants) that the engine
previously emitted as bare stems.  OPSIN rejects those bare stems — it
requires the full ``-ane`` hydrocarbon suffix per IUPAC P-101.

The Stage 6 R2-A fix is localised in
``iupac_namer/natural_products/steroid.py``: a canonical-SMILES-to-stem
map built from the XML reference SMILES, rewriting the retained-plan
match name to ``<stem>ane`` (and ``5α-<stem>ane`` / ``5β-<stem>ane`` for
the two ring-junction diastereomers we have XML reference SMILES for).

These tests pin:
    * engine output for each of the 15 unique stems,
    * α/β descriptor emission when the input differs from the XML's
      reference by only a C5 stereo flip.
"""

from __future__ import annotations

import pytest
from rdkit import Chem

from iupac_namer.engine import name_smiles
from iupac_namer.natural_products.steroid import (
    STEROID_STEMS,
    try_steroid_stem_name,
)


# ---------------------------------------------------------------------------
# Direct engine emission for each unique steroid stem
# ---------------------------------------------------------------------------

STEROID_STEM_CASES: list[tuple[str, str]] = [
    # (input_smiles, expected_engine_output)
    ("C[C@@]12CCC[C@H]1[C@@H]1CCC3CCCC[C@]3(C)[C@H]1CC2", "androstane"),
    ("C[C@@]12CCC[C@H]1[C@@H]1CCC3CCCC[C@@H]3[C@H]1CC2", "estrane"),
    ("C1C[C@H]2CC[C@H]3[C@@H](CCC4CCCC[C@H]34)[C@@H]2C1", "gonane"),
    ("CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CCC4CCCC[C@]4(C)[C@H]3CC[C@]12C",
     "cholestane"),
    ("CC[C@H]1CC[C@H]2[C@@H]3CCC4CCCC[C@]4(C)[C@H]3CC[C@]12C",
     "pregnane"),
    ("CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CCC4CCCC[C@]4(C)[C@H]3CC[C@]12C",
     "cholane"),
    ("CC(C)[C@H](C)CC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CCC4CCCC[C@]4(C)[C@H]3CC[C@]12C",
     "campestane"),
    ("CC(C)[C@@H](C)CC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CCC4CCCC[C@]4(C)[C@H]3CC[C@]12C",
     "ergostane"),
    ("CC[C@H](CC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CCC4CCCC[C@]4(C)[C@H]3CC[C@]12C)C(C)C",
     "stigmastane"),
    ("CC[C@@H](CC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CCC4CCCC[C@]4(C)[C@H]3CC[C@]12C)C(C)C",
     "poriferastane"),
    ("CC(C)[C@@H](C)[C@@]1(C)C[C@@H]1[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CCC4CCCC[C@]4(C)[C@H]3CC[C@]12C",
     "gorgostane"),
    ("C[C@H]1[C@H]2[C@H](C[C@H]3[C@@H]4CCC5CCCC[C@]5(C)[C@H]4CC[C@]23C)O[C@]11CCC(C)CO1",
     "spirostane"),
    ("CC(C)CCC1O[C@H]2C[C@H]3[C@@H]4CCC5CCCC[C@]5(C)[C@H]4CC[C@]3(C)[C@H]2[C@@H]1C",
     "furostane"),
    ("CCCCCCC[C@H]1CCC[C@@H]1CCCCCCCC", "prostane"),
    ("CCCCCCC[C@H]1CCCO[C@@H]1CCCCCCCC", "thromboxane"),
]


@pytest.mark.parametrize("smi,expected_name", STEROID_STEM_CASES)
def test_engine_emits_steroid_stem(smi: str, expected_name: str) -> None:
    """Engine emits the proper ``<stem>ane`` name for each steroid stem."""
    assert name_smiles(smi) == expected_name


# ---------------------------------------------------------------------------
# 5α / 5β descriptor emission
# ---------------------------------------------------------------------------

STEROID_ALPHA_BETA_CASES: list[tuple[str, str]] = [
    # (input_smiles, expected_engine_output)
    # 5α-androstane: explicit [C@H] at C5
    ("C[C@@]12CCC[C@H]1[C@@H]1CC[C@H]3CCCC[C@]3(C)[C@H]1CC2",
     "5\u03b1-androstane"),
    # 5β-androstane: explicit [C@@H] at C5
    ("C[C@@]12CCC[C@H]1[C@@H]1CC[C@@H]3CCCC[C@]3(C)[C@H]1CC2",
     "5\u03b2-androstane"),
]


@pytest.mark.parametrize("smi,expected_name", STEROID_ALPHA_BETA_CASES)
def test_engine_emits_5_alpha_beta(smi: str, expected_name: str) -> None:
    """Engine emits the 5α / 5β descriptor for steroid ring-junction isomers."""
    assert name_smiles(smi) == expected_name


# ---------------------------------------------------------------------------
# Steroid-stem module unit tests
# ---------------------------------------------------------------------------

def test_steroid_stems_constant_contents() -> None:
    """STEROID_STEMS includes all 15 base names and the 5α/5β variants
    for stems that define them."""
    assert "androstane" in STEROID_STEMS
    assert "pregnane" in STEROID_STEMS
    assert "cholestane" in STEROID_STEMS
    assert "estrane" in STEROID_STEMS
    assert "gonane" in STEROID_STEMS
    assert "campestane" in STEROID_STEMS
    assert "ergostane" in STEROID_STEMS
    assert "stigmastane" in STEROID_STEMS
    assert "poriferastane" in STEROID_STEMS
    assert "gorgostane" in STEROID_STEMS
    assert "spirostane" in STEROID_STEMS
    assert "furostane" in STEROID_STEMS
    assert "prostane" in STEROID_STEMS
    assert "thromboxane" in STEROID_STEMS
    assert "5\u03b1-androstane" in STEROID_STEMS
    assert "5\u03b2-androstane" in STEROID_STEMS


def test_try_steroid_stem_name_recognises_canonical_match() -> None:
    """try_steroid_stem_name returns the proper stem for canonical-matching
    input SMILES, independent of the caller-supplied opsin_name parameter."""
    smi = "CC[C@H]1CC[C@H]2[C@@H]3CCC4CCCC[C@]4(C)[C@H]3CC[C@]12C"  # pregn
    mol = Chem.MolFromSmiles(smi)
    assert try_steroid_stem_name("pregn", mol) == "pregnane"
    assert try_steroid_stem_name("", mol) == "pregnane"      # no name needed
    assert try_steroid_stem_name("garbage", mol) == "pregnane"


def test_try_steroid_stem_name_returns_none_for_non_steroid() -> None:
    """Non-steroid molecules return None."""
    mol = Chem.MolFromSmiles("CCO")  # ethanol
    assert try_steroid_stem_name("androst", mol) is None
    assert try_steroid_stem_name("", mol) is None


def test_try_steroid_stem_name_handles_none_mol() -> None:
    """None input returns None without raising."""
    assert try_steroid_stem_name("androst", None) is None
