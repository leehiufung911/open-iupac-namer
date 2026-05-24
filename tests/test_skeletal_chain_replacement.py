"""Skeletal replacement ("a") nomenclature for acyclic chains (P-15.4.3 / P-51.4.1.1).

When an unbranched acyclic chain carries four or more skeletal heterounits
together with at least one carbon atom (and no principal characteristic group),
IUPAC mandates skeletal replacement ("a") nomenclature as the PIN.  These tests
pin the engine output for the common cleanly-round-tripping cases and assert the
threshold / scope guards (fewer than four heterounits => substitutive, not
replacement).

All expected names were confirmed by OPSIN round-trip
(name -> SMILES -> canonical SMILES recovers the input structure).
"""
from __future__ import annotations

import pytest
from rdkit import Chem

from iupac_namer.engine import name_smiles
from iupac_namer.perception.skeletal_chain import compute_name


# ---------------------------------------------------------------------------
# Positive cases: four or more heterounits -> replacement PIN.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "smiles, expected",
    [
        # All-chalcogen heterogeneous run O-Te-Se-S (the canonical task case).
        ("CCO[Te][Se]SCC", "3-oxa-6-thia-5-selena-4-telluraoctane"),
        # Polyethylene-glycol style: four ether oxygens.
        ("COCCOCCOCCOC", "2,5,8,11-tetraoxadodecane"),
        # Densely substituted: O on alternate positions.
        ("COCOCOCOC", "2,4,6,8-tetraoxanonane"),
        ("CCOCCOCCOCCOCC", "3,6,9,12-tetraoxatetradecane"),
        ("COCCOCCOCCOCCOC", "2,5,8,11,14-pentaoxapentadecane"),
        ("COCCOCCOCCOCCOCCOC", "2,5,8,11,14,17-hexaoxaoctadecane"),
        # Mixed chalcogens with a disulfane unit (adjacent same-element run).
        ("COCSSCCOCC[Se]C", "2,8-dioxa-4,5-dithia-11-selenadodecane"),
        ("CCO[Se]SOCCC", "3,6-dioxa-5-thia-4-selenanonane"),
        ("COCCSCCSCCOC", "2,11-dioxa-5,8-dithiadodecane"),
        # Si-terminated chains (P-51.4.1.4 allows P/Si/... termini).
        ("[SiH3]OCS[SiH3]", "2-oxa-4-thia-1,5-disilapentane"),
        ("C[SiH2]PC[SiH2]C[SiH2]C", "3-phospha-2,5,7-trisilaoctane"),
        ("C[SiH2]C[SiH2]C[SiH2]CSCC", "8-thia-2,4,6-trisiladecane"),
        ("COC[SiH2]C[SiH2]C[SiH2]C", "2-oxa-4,6,8-trisilanonane"),
        ("CC[SiH2]C[SiH2]C[SiH2]C[SiH2]C", "2,4,6,8-tetrasiladecane"),
        # Nitrogen + oxygen.  Heteroatom locant set is {2,4,6,8} either way;
        # the seniority tie-break (Appendix 1: O senior to N) gives O the low
        # locants -> "2,4,6-trioxa-8-azanonane" is the PIN.  (OPSIN also parses
        # the equivalent "2-aza-4,6,8-trioxanonane" for the same structure.)
        ("CNCOCOCOC", "2,4,6-trioxa-8-azanonane"),
    ],
)
def test_skeletal_chain_pin(smiles: str, expected: str) -> None:
    assert name_smiles(smiles) == expected


# ---------------------------------------------------------------------------
# Numbering: lowest-locant set, seniority tie-break (P-15.4.3.2.1).
# ---------------------------------------------------------------------------

def test_numbering_low_locant_set() -> None:
    """8-thia-2,4,6-trisiladecane: 2,4,6,8 lower than 3,5,7,9."""
    assert (
        name_smiles("C[SiH2]C[SiH2]C[SiH2]CSCC")
        == "8-thia-2,4,6-trisiladecane"
    )


def test_numbering_seniority_tiebreak_o_over_si() -> None:
    """Locant sets equal -> O gets the low locant over Si (Appendix 1)."""
    assert (
        name_smiles("COC[SiH2]C[SiH2]C[SiH2]C")
        == "2-oxa-4,6,8-trisilanonane"
    )


# ---------------------------------------------------------------------------
# Negative / scope guards: must NOT trigger replacement.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "smiles",
    [
        "COCCCOCCCOC",   # only 3 ether oxygens -> 3 heterounits (< 4)
        "CCCCOCCCC",     # 1 heterounit
        "COCCOCCOC",     # 3 heterounits
        "CCSCC",         # 1 heterounit
    ],
)
def test_below_threshold_declines(smiles: str) -> None:
    """Fewer than four heterounits: skeletal module declines (returns None)."""
    mol = Chem.MolFromSmiles(smiles)
    assert compute_name(mol) is None


def test_all_carbon_chain_declines() -> None:
    """No heteroatoms -> not a replacement chain."""
    mol = Chem.MolFromSmiles("CCCCCCCCCC")
    assert compute_name(mol) is None


def test_ring_declines() -> None:
    """Cyclic systems are out of scope for the acyclic chain module."""
    mol = Chem.MolFromSmiles("C1COCOCO1")
    assert compute_name(mol) is None


def test_branched_declines() -> None:
    """A branch off the heterochain is out of scope (deferred)."""
    # 2,4,6,8-tetraoxanonane skeleton with an extra methyl branch on a carbon.
    mol = Chem.MolFromSmiles("COCOC(C)OCOC")
    assert compute_name(mol) is None


def test_terminal_chalcogen_declines() -> None:
    """Terminal O/S/Se/Te/N is not a skeletal terminus (P-51.4.1.4).

    HO-CH2-O-CH2-O-CH2-O-CH2-OH has terminal hydroxyl oxygens, which are a
    characteristic group, not skeletal ether O; the module must decline rather
    than emit the (chemically wrong) "1,3,5,7,9-pentaoxanonane".
    """
    mol = Chem.MolFromSmiles("OCOCOCOCO")
    assert compute_name(mol) is None


def test_pcg_double_bond_oxo_declines() -> None:
    """A chain ketone (C=O) carries an unsaturated bond / PCG -> declined."""
    # 3,6-dioxa-5-thia-4-selenanonan-7-one skeleton.
    mol = Chem.MolFromSmiles("CCO[Se]SOC(CC)=O")
    assert compute_name(mol) is None


# ---------------------------------------------------------------------------
# Round-trip safety: the engine output must recover the input via OPSIN.
# (Exercised through the audit round-trip harness for one representative.)
# ---------------------------------------------------------------------------

def test_round_trip_representative() -> None:
    from tests.audit._audit_helpers import assert_round_trip

    name = assert_round_trip("CCO[Te][Se]SCC")
    assert name == "3-oxa-6-thia-5-selena-4-telluraoctane"


# ---------------------------------------------------------------------------
# Carbon-free chains (P-21.2.2 homogeneous / P-21.2.3.1 alternating a(ba)n).
#
# Replacement ("a") nomenclature (P-21.2.3.2) requires >=1 carbon; an
# all-heteroatom chain is instead a homogeneous parent hydride (one element,
# e.g. trisulfane) or, when two heteroatoms strictly alternate and terminate at
# the same (less-senior) element, an a(ba)n heterogeneous parent hydride (e.g.
# dithioxane).  Nitrogen-containing heterogeneous chains use amine names
# (P-21.2.3.1 note) and are deliberately NOT named by this rule.
#
# All expected names confirmed by OPSIN round-trip.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "smiles, expected",
    [
        # Homogeneous polychalcogen parent hydrides (P-21.2.2), length >= 3.
        ("SSS", "trisulfane"),
        ("SSSS", "tetrasulfane"),
        ("SSSSS", "pentasulfane"),
        ("OOO", "trioxidane"),
        ("OOOO", "tetraoxidane"),
        ("[SeH][Se][SeH]", "triselane"),
        ("[SeH][Se][Se][SeH]", "tetraselane"),
        ("[TeH][Te][TeH]", "tritellane"),
    ],
)
def test_homogeneous_chalcogen_chain_pin(smiles: str, expected: str) -> None:
    assert name_smiles(smiles) == expected


@pytest.mark.parametrize(
    "smiles, expected",
    [
        # a(ba) — 3-atom alternating parent hydrides (P-21.2.3.1).  Terminal
        # element is the LESS-senior one; middle is the more-senior one.
        ("SOS", "dithioxane"),            # HS-O-SH  (S terminal, O middle)
        ("[SiH3]S[SiH3]", "disilathiane"),
        ("[SiH3]O[SiH3]", "disiloxane"),
        ("[SiH3][Se][SiH3]", "disilaselenane"),
        ("[SiH3][Te][SiH3]", "disilatellurane"),
        ("P[Se]P", "diphosphaselenane"),
        ("PSP", "diphosphathiane"),
        ("POP", "diphosphoxane"),
        ("[SeH]O[SeH]", "diselenoxane"),
        ("[SeH]S[SeH]", "diselenathiane"),
        ("[TeH]O[TeH]", "ditelluroxane"),
        ("[TeH][Se][TeH]", "ditelluraselenane"),
        ("[GeH3]O[GeH3]", "digermoxane"),
        ("[GeH3]S[GeH3]", "digermathiane"),
        ("[SnH3]O[SnH3]", "distannoxane"),
        ("[AsH2]O[AsH2]", "diarsoxane"),
        ("[SbH2]O[SbH2]", "distiboxane"),
        # a(ba)n — 5-atom alternating parent hydrides.
        ("SOSOS", "trithioxane"),
        ("[SiH3]O[SiH2]O[SiH3]", "trisiloxane"),
        ("PSPSP", "triphosphathiane"),
        ("P[Se]P[Se]P", "triphosphaselenane"),
        ("[SnH3]O[SnH2]O[SnH3]", "tristannoxane"),
    ],
)
def test_alternating_heterochain_pin(smiles: str, expected: str) -> None:
    assert name_smiles(smiles) == expected


@pytest.mark.parametrize(
    "smiles",
    [
        # Nitrogen present in a heterogeneous chain -> amine names apply
        # (P-21.2.3.1 note); the a(ba)n rule must NOT claim it.
        ["Si", "N", "Si"],
        ["N", "O", "N"],
        ["P", "N", "P"],
        # Senior element at the termini (e.g. O-S-O) is not a valid a(ba)n
        # parent hydride and (being symmetric) cannot be reoriented -> decline.
        ["O", "S", "O"],
        ["O", "Se", "O"],
        # Three distinct kinds -> not an alternating two-element chain.
        ["S", "O", "Se"],
        # Homogeneous (one element) -> handled by the homogeneous dispatcher,
        # not the alternating rule.
        ["S", "S", "S"],
        # Even length cannot be a(ba)n (needs odd 2k-1).
        ["S", "O", "S", "O"],
    ],
)
def test_alternating_heterochain_declines(smiles: list[str]) -> None:
    from iupac_namer.perception.skeletal_chain import (
        _compute_alternating_heterochain,
    )

    assert _compute_alternating_heterochain(smiles) is None


def test_carbon_free_alternating_declines_in_compute_name() -> None:
    """compute_name declines a homogeneous carbon-free chain (None), leaving it
    to the engine's homogeneous-heteroatom-chain dispatcher (P-21.2.2)."""
    mol = Chem.MolFromSmiles("SSS")
    assert compute_name(mol) is None
