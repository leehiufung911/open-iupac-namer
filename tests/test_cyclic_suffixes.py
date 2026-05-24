"""Regression tests for Stage 6 R2-E cyclic-suffix classifier.

Covers ``iupac_namer.perception.fg.cyclic_suffixes`` and the engine
dispatch hook that consumes it.  The module is a read-only classifier
at R2-E — its sole behavioural contract is

    classify_cyclic_suffix(mol)  ↦ a structural summary dataclass
    detect(mol, ...)             ↦ None for every input (no emission)

so these tests pin three things:

1. The classifier assigns the correct ``motif`` / ``ring_size`` /
   ``tautomer_form`` / ``is_fused`` labels for a spread of imide,
   lactam, and lactone reference structures across ring sizes 4-7.
2. The engine dispatch hook is strictly non-regressing: existing
   keto-form names (``succinimide``, ``phthalimide``, ``caprolactam``,
   ``butyrolactone``, …) round-trip through the engine untouched.
3. The classifier returns ``None`` for non-ring-FG molecules, so a
   future emission layer cannot accidentally fire on ethane, ethanol,
   or acetic acid.
"""

from __future__ import annotations

import pytest
from rdkit import Chem

from iupac_namer.engine import name_smiles
from iupac_namer.perception.fg.cyclic_suffixes import (
    CyclicSuffixClassification,
    all_motif_names,
    classify_cyclic_suffix,
    detect,
)
from iupac_namer.types import OutputForm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mol(smi: str):
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None, f"bad test SMILES {smi!r}"
    return mol


# ---------------------------------------------------------------------------
# Vocabulary sanity
# ---------------------------------------------------------------------------

def test_all_motif_names_include_the_big_four() -> None:
    names = set(all_motif_names())
    for expected in (
        "dicarboximide",
        "imide",
        "lactam",
        "lactone",
        "carbolactone",
        "beta-lactam",
        "gamma-butyrolactone",
    ):
        assert expected in names, f"missing motif {expected!r}"


# ---------------------------------------------------------------------------
# Imide classification
# ---------------------------------------------------------------------------

def test_succinimide_keto_classifies_as_5_ring_imide() -> None:
    # Classic succinimide — 5-ring C(=O)-N(H)-C(=O) with two CH2 bridges.
    cls = classify_cyclic_suffix(_mol("O=C1CCC(=O)N1"))
    assert isinstance(cls, CyclicSuffixClassification)
    assert cls.motif == "imide"
    assert cls.ring_size == 5
    assert cls.tautomer_form == "keto"
    assert cls.is_fused is False
    # Two carbonyls detected, both inside the ring.
    assert len(cls.carbonyl_atoms) == 2


def test_glutarimide_keto_classifies_as_6_ring_imide() -> None:
    cls = classify_cyclic_suffix(_mol("O=C1CCCC(=O)N1"))
    assert cls is not None
    assert cls.motif == "imide"
    assert cls.ring_size == 6
    assert cls.tautomer_form == "keto"
    assert cls.is_fused is False


def test_phthalimide_classifies_as_fused_5_ring_imide() -> None:
    cls = classify_cyclic_suffix(_mol("O=C1NC(=O)c2ccccc12"))
    assert cls is not None
    assert cls.motif == "imide"
    assert cls.ring_size == 5
    assert cls.is_fused is True


def test_cyclohexane_dicarboximide_fused_imide() -> None:
    # Keto form: O=C1NC(=O)C2CCCCC12 — 5-ring imide fused to cyclohexane.
    cls = classify_cyclic_suffix(_mol("O=C1NC(=O)C2CCCCC12"))
    assert cls is not None
    assert cls.motif == "imide"
    assert cls.ring_size == 5
    assert cls.is_fused is True


def test_cyclohexane_dicarboximide_iminol_is_tautomer_labelled() -> None:
    # Iminol tautomer: O=C1N=C(O)C2CCCCC12 — one C=O, other is N=C-OH.
    cls = classify_cyclic_suffix(_mol("O=C1N=C(O)C2CCCCC12"))
    assert cls is not None
    assert cls.motif == "imide"
    assert cls.ring_size == 5
    assert cls.tautomer_form == "iminol"
    assert cls.is_fused is True


# ---------------------------------------------------------------------------
# Lactam classification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "smiles,ring_size,retained",
    [
        # beta-lactam: 4-ring C=O-N
        ("O=C1CCN1", 4, "beta-lactam"),
        # gamma-lactam: 5-ring (2-pyrrolidinone)
        ("O=C1CCCN1", 5, "gamma-lactam"),
        # delta-lactam: 6-ring
        ("O=C1CCCCN1", 6, "delta-lactam"),
        # epsilon-lactam (caprolactam): 7-ring
        ("O=C1CCCCCN1", 7, "epsilon-lactam"),
    ],
)
def test_lactam_retained_name_hint_by_ring_size(
    smiles: str, ring_size: int, retained: str
) -> None:
    cls = classify_cyclic_suffix(_mol(smiles))
    assert cls is not None
    assert cls.motif == "lactam"
    assert cls.ring_size == ring_size
    assert cls.tautomer_form == "keto"
    assert cls.is_fused is False
    assert cls.retained_name_hint == retained


# ---------------------------------------------------------------------------
# Lactone classification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "smiles,ring_size,retained",
    [
        ("O=C1CCO1",     4, "beta-propiolactone"),
        ("O=C1CCCO1",    5, "gamma-butyrolactone"),
        ("O=C1CCCCO1",   6, "delta-valerolactone"),
        ("O=C1CCCCCO1",  7, "epsilon-caprolactone"),
    ],
)
def test_lactone_retained_name_hint_by_ring_size(
    smiles: str, ring_size: int, retained: str
) -> None:
    cls = classify_cyclic_suffix(_mol(smiles))
    assert cls is not None
    assert cls.motif == "lactone"
    assert cls.ring_size == ring_size
    assert cls.tautomer_form == "keto"
    assert cls.is_fused is False
    assert cls.retained_name_hint == retained


# ---------------------------------------------------------------------------
# Classifier returns None for irrelevant inputs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "smiles",
    [
        "CCO",           # ethanol
        "CC(=O)O",       # acetic acid — exocyclic C=O, not a ring FG
        "CC(=O)N",       # acetamide  — acyclic amide
        "C1CCCCC1",      # cyclohexane
        "c1ccccc1",      # benzene
        "CC(=O)OC",      # methyl acetate — acyclic ester
        "C1CCNC1",       # pyrrolidine — no C=O
    ],
)
def test_classifier_is_none_for_non_ring_fg(smiles: str) -> None:
    assert classify_cyclic_suffix(_mol(smiles)) is None


# ---------------------------------------------------------------------------
# Engine dispatch is non-regressing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "smiles,expected_name",
    [
        # P-66.6.3 ring-carbonyl SUFFIX rule (updated): a ring carbon C=O
        # that is the senior characteristic group is expressed as the
        # multiplied ``-one``/``-dione`` suffix, NOT an ``oxo``/``dioxo``
        # prefix.  Previously the engine emitted the non-preferred keto
        # prefix forms (``2-oxopyrrolidine``, ``1,3-dioxoisoindoline``, …);
        # the ring-carbonyl FG synthesis now promotes them to the preferred
        # suffix.  All forms below round-trip through OPSIN.
        #
        # succinimide stays a retained whole-molecule PIN (higher tier,
        # P-66.2 via retained_names_expanded.json) and is unaffected.
        ("O=C1CCC(=O)N1",        "succinimide"),
        ("O=C1CCCC(=O)N1",       "piperidine-2,6-dione"),
        ("O=C1NC(=O)c2ccccc12",  "isoindoline-1,3-dione"),
        ("O=C1NC(=O)C2CCCCC12",  "octahydro-1H-isoindole-1,3-dione"),
        # Lactams — preferred ``-one`` ring suffix (P-66.6.3).
        ("O=C1CCCN1",            "pyrrolidin-2-one"),
        ("O=C1CCCCN1",           "piperidin-2-one"),
        ("O=C1CCCCCN1",          "azepan-2-one"),
        # Lactones — preferred ``-one`` ring suffix (P-66.6.3).
        ("O=C1CCCO1",            "oxolan-2-one"),
        ("O=C1CCCCO1",           "oxan-2-one"),
        ("O=C1CCCCCO1",          "oxepan-2-one"),
    ],
)
def test_engine_dispatch_emits_preferred_one_dione_suffix(
    smiles: str, expected_name: str
) -> None:
    assert name_smiles(smiles) == expected_name


# ---------------------------------------------------------------------------
# detect() stub contract
# ---------------------------------------------------------------------------

def test_detect_stub_returns_none_for_succinimide() -> None:
    # The R2-E dispatch is read-only; detect always returns None so the
    # hook never pre-empts the existing plan search.
    mol = _mol("O=C1CCC(=O)N1")
    result = detect(
        mol,
        output_form=OutputForm.STANDALONE,
        free_valence=None,
        decision_ctx=None,
        strategy=None,
        session=None,
        depth=0,
    )
    assert result is None


def test_detect_stub_ignores_substituent_form() -> None:
    # Non-STANDALONE output forms short-circuit before classification.
    mol = _mol("O=C1CCC(=O)N1")
    result = detect(
        mol,
        output_form=OutputForm.SUBSTITUENT,
        free_valence=None,
        decision_ctx=None,
        strategy=None,
        session=None,
        depth=0,
    )
    assert result is None
