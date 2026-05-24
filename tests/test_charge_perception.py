"""Regression tests for Stage 6 R2-B charge perception.

Pins the behaviour of ``iupac_namer.perception.charge_perception`` and
the engine dispatch hook that consumes it.  Every probe in this suite
came from one of the audit CSVs (``eval/opsin_audit_fg_raw.csv`` /
``eval/opsin_audit_hw_charge_raw.csv``); each one is checked end-to-end:

1. the engine emits the surface name we expect;
2. the emitted name round-trips through OPSIN (``py2opsin``) to the
   same canonical SMILES as the input.

The OPSIN round-trip is the critical guarantee — even if our exact
surface form drifts (e.g. ``acetamidinium`` vs ``ethanamidinium``,
which OPSIN treats as synonyms), the test catches us as long as the
SMILES round-trip holds.
"""

from __future__ import annotations

import pytest
from rdkit import Chem

from iupac_namer.engine import name_smiles
from iupac_namer.perception.charge_perception import (
    ChargeClassification,
    classify_charges,
    detect,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canon(smiles: str) -> str | None:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def _roundtrip(name: str) -> str | None:
    """Run OPSIN over ``name`` and return the canonical SMILES it emits."""
    from py2opsin import py2opsin

    smi = py2opsin(name)
    return _canon(smi) if smi else None


def _opsin_matches(input_smiles: str, our_name: str) -> bool:
    """True when OPSIN(our_name) and the canonical input agree."""
    return _roundtrip(our_name) == _canon(input_smiles)


# ---------------------------------------------------------------------------
# Classifier sanity
# ---------------------------------------------------------------------------


def test_classify_methylium_cation() -> None:
    cls = classify_charges(Chem.MolFromSmiles("[CH3+]"))
    assert len(cls) == 1
    assert cls[0].suffix_hint == "ylium"
    assert cls[0].is_cation
    assert cls[0].site_atom_indices == (0,)


def test_classify_methanide_anion() -> None:
    cls = classify_charges(Chem.MolFromSmiles("[CH3-]"))
    assert len(cls) == 1
    assert cls[0].suffix_hint == "ide"
    assert cls[0].is_anion


def test_classify_diazonium_motif() -> None:
    cls = classify_charges(Chem.MolFromSmiles("CC[N+]#N"))
    assert len(cls) == 1
    assert cls[0].suffix_hint == "diazonium"
    assert cls[0].charge_sign == "+"
    # Both N atoms belong to the motif.
    assert len(cls[0].site_atom_indices) == 2


def test_classify_amidinium_motif() -> None:
    cls = classify_charges(Chem.MolFromSmiles("CC(=[NH2+])N"))
    assert len(cls) == 1
    assert cls[0].suffix_hint == "amidinium"
    # Central C + both N atoms.
    assert len(cls[0].site_atom_indices) == 3


def test_classify_acylium_motif() -> None:
    cls = classify_charges(Chem.MolFromSmiles("[C+](C)=O"))
    assert len(cls) == 1
    assert cls[0].suffix_hint == "acylium"
    # C+ + =O atoms.
    assert len(cls[0].site_atom_indices) == 2


def test_classify_borohydride_uses_surface_shortcut() -> None:
    cls = classify_charges(Chem.MolFromSmiles("[BH4-]"))
    assert len(cls) == 1
    assert cls[0].surface_name == "boranuide"


# ---------------------------------------------------------------------------
# Negative classification — must not claim
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("smi", [
    "CC",
    "CCO",
    "[NH4+]",          # azanium (named upstream)
    "[Na+]",           # monatomic salt
    "[Cl-]",           # monatomic salt
    "c1cc[n+](C)cc1",  # ring N+ - takes the substitutive ring-cation path
    "[C+]1=CC=CC=C1",  # phenylium - retained ring lookup
    "[O+]1=CC=CC=C1",  # pyrylium - retained ring lookup
])
def test_classify_does_not_claim_existing_paths(smi: str) -> None:
    """The classifier must remain silent for charge motifs that the rest
    of the engine already names.  Otherwise the engine would double-
    dispatch and the regression hits would be enormous."""
    cls = classify_charges(Chem.MolFromSmiles(smi))
    assert cls == ()


def test_classify_returns_empty_for_neutral_input() -> None:
    assert classify_charges(Chem.MolFromSmiles("c1ccccc1")) == ()


def test_classify_returns_empty_for_none() -> None:
    assert classify_charges(None) == ()


# ---------------------------------------------------------------------------
# End-to-end engine probes (audit-derived)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("smi,expected_name", [
    # ylium / ide on simple chains and rings (FG/HW-charge audits)
    ("[CH3+]",        "methylium"),
    ("[CH3-]",        "methanide"),
    ("[CH2+]C",       "ethan-1-ylium"),
    ("[CH2-]C",       "ethan-1-ide"),
    ("[CH2+]CCCC",    "pentan-1-ylium"),
    ("[CH2-]CCCC",    "pentan-1-ide"),
    ("[CH+]1CCCCC1",  "cyclohexan-1-ylium"),
    # diazonium (FG audit, Top-3 Gap 7)
    ("CC[N+]#N",                                "ethane-1-diazonium"),
    ("CCCCC[N+]#N",                             "pentane-1-diazonium"),
    ("C1(CCCCC1)[N+]#N",                        "cyclohexane-1-diazonium"),
    ("C1(=CC=CC2=CC=CC=C12)[N+]#N",             "naphthalene-1-diazonium"),
    # acylium / carbonylium (FG audit, Top-3 Gap 13)
    ("[C+](C)=O",                  "acetylium"),
    ("[C+](CCCC)=O",               "pentanoylium"),
    ("C1(CCCCC1)[C+]=O",           "cyclohexanecarbonylium"),
    # amidinium (FG audit, Top-3 Gap 4)
    ("CC(=[NH2+])N",                            "acetamidinium"),
    ("C1(CCCCC1)C(=[NH2+])N",                   "cyclohexan-1-amidinium"),
    # boranuide (HW-charge audit)
    ("[BH4-]", "boranuide"),
    # carbocyclic ring cations — closed-shell -ylium (HW-charge audit)
    # tropylium / cyclopropenylium / cyclooctatetraenylium have NO retained
    # ylium name, so the systematic cyclo<stem>-<ene>-1-ylium PIN applies.
    ("c1ccc[cH+]cc1",      "cyclohepta-2,4,6-trien-1-ylium"),
    ("C1=C[CH+]1",         "cycloprop-2-en-1-ylium"),
    ("[C+]1=CC=CC=CC=C1",  "cycloocta-1,3,5,7-tetraen-1-ylium"),
    # retained ring-cation names stay senior to the systematic form.
    ("[C+]1=CC=CC=C1",     "phenylium"),
    ("[CH+]1C=CC=C1",      "cyclopentadienylium"),
])
def test_engine_emits_exact_surface_name(smi: str, expected_name: str) -> None:
    """Pin the exact surface name the engine emits for each probe."""
    assert name_smiles(smi) == expected_name


@pytest.mark.parametrize("smi", [
    # All probes round-trip through OPSIN to the same canonical SMILES.
    "[CH3+]",
    "[CH3-]",
    "[CH2+]C",
    "[CH2-]C",
    "[CH2+]CCCC",
    "[CH2-]CCCC",
    "[CH+]1CCCCC1",
    "CC[N+]#N",
    "CCCCC[N+]#N",
    "C1(CCCCC1)[N+]#N",
    "C1(=CC=CC2=CC=CC=C12)[N+]#N",
    "[C+](C)=O",
    "[C+](CCCC)=O",
    "C1(CCCCC1)[C+]=O",
    "CC(=[NH2+])N",
    "C1(CCCCC1)C(=[NH2+])N",
    "[BH4-]",
    # carbocyclic ring cations (systematic + retained).
    "c1ccc[cH+]cc1",
    "C1=C[CH+]1",
    "[C+]1=CC=CC=CC=C1",
    "[C+]1=CC=CC=C1",
    "[CH+]1C=CC=C1",
])
def test_engine_output_round_trips_through_opsin(smi: str) -> None:
    """Stronger guarantee: OPSIN canonicalises our name back to the
    input's canonical SMILES.  This is the gate that makes the
    audit-CSV row turn from WRONG to COVERED."""
    name = name_smiles(smi)
    assert _roundtrip(name) == _canon(smi), (
        f"Round-trip failed for {smi!r}: "
        f"name={name!r} opsin={_roundtrip(name)!r} expected={_canon(smi)!r}"
    )


# ---------------------------------------------------------------------------
# Non-regression tests — molecules that should NOT change behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("smi,expected_substring", [
    # Existing path: ring-N+ -> -ium suffix via ring_cation_locants.
    ("c1cc[n+](C)cc1",   "pyridin"),
    # Existing path: retained phenylium via ring lookup.
    ("[C+]1=CC=CC=C1",   "phenylium"),
    # Existing path: retained pyrylium via ring lookup.
    ("[O+]1=CC=CC=C1",   "pyrylium"),
    # Existing path: monatomic ammonium.
    ("[NH4+]",           "azanium"),
    # Existing path: neutral carboxylic acid.
    ("CC(=O)O",          "acetic acid"),
])
def test_charge_perception_does_not_disturb_existing_names(
    smi: str, expected_substring: str
) -> None:
    name = name_smiles(smi)
    assert expected_substring in name


# ---------------------------------------------------------------------------
# Frozen dataclass invariant
# ---------------------------------------------------------------------------


def test_charge_classification_is_frozen() -> None:
    cls = ChargeClassification(
        site_atom_indices=(0,),
        charge_sign="+",
        suffix_hint="ylium",
    )
    with pytest.raises((AttributeError, Exception)):
        cls.suffix_hint = "ide"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Engine hook — substituent / acyl forms must defer
# ---------------------------------------------------------------------------


def test_detect_returns_none_for_substituent_form() -> None:
    from iupac_namer.types import OutputForm
    from iupac_namer.engine import name as iname  # noqa: F401

    mol = Chem.MolFromSmiles("[CH3+]")
    # We don't need a real session/strategy — substituent gating
    # short-circuits before any of those are accessed.
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
