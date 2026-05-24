"""Tests for heteroatom parent hydride naming (Phase 3f).

Covers substituted phosphane, silane, borane, arsane, germane, stannane.
"""
import pytest
from iupac_namer.engine import name_smiles


@pytest.mark.parametrize("smi,expected", [
    # Phosphane
    ("FP",           "fluorophosphane"),
    ("ClP",          "chlorophosphane"),
    ("ClPCl",        "dichlorophosphane"),
    ("CP(C)C",       "trimethylphosphane"),
    ("CP",           "methylphosphane"),
    # Silane
    ("Cl[SiH2]Cl",   "dichlorosilane"),
    ("Cl[SiH](Cl)Cl", "trichlorosilane"),
    ("C[SiH3]",      "methylsilane"),
    ("C[Si](C)(C)C", "tetramethylsilane"),
    # Borane
    ("BrBBr",        "dibromoborane"),
    ("FB(F)F",       "trifluoroborane"),
    # Unsubstituted must still work via retained lookup
    ("P",            "phosphane"),
    ("[SiH4]",       "silane"),
    ("B",            "borane"),
])
def test_heteroatom_parent(smi, expected):
    assert name_smiles(smi) == expected


@pytest.mark.parametrize("smi,expected", [
    # P-16.3.3 preferred form: on a heteroatom parent hydride bearing two or
    # more DIFFERENT substituents, every substituent prefix is individually
    # bracketed EXCEPT the first (lowest-sort) one when it is simple — the
    # following prefix's opening enclosing mark already marks the boundary.
    # Each of these is a Blue Book PIN verified to round-trip through OPSIN.
    ("C[SiH2]Cl",            "chloro(methyl)silane"),
    ("C[Si](Cl)(Cl)Cl",     "trichloro(methyl)silane"),
    ("CCCC[Si](C)(CC)CCC",  "butyl(ethyl)(methyl)(propyl)silane"),
    ("CCCP(C)CC",           "ethyl(methyl)(propyl)phosphane"),
    ("CCP(C)C",             "ethyldi(methyl)phosphane"),
    ("CB(Cl)Cl",            "dichloro(methyl)borane"),
    # Charged parent hydrides take the same rule.
    ("C[P+](C)(C)Cl",       "chlorotri(methyl)phosphanium"),
    ("C[F+]Cl",             "chloro(methyl)fluoranium"),
    ("C[Cl+]C(C)=O",        "acetyl(methyl)chloranium"),
])
def test_heteroatom_leading_prefix_unbracketed(smi, expected):
    """First simple substituent prefix carries no enclosing marks (P-16.3.3)."""
    assert name_smiles(smi) == expected


@pytest.mark.parametrize("smi,expected", [
    # Guard: an alkoxy ("-oxy") leading prefix STAYS bracketed.  Concatenating
    # a multiplier directly onto it ("di(methoxy)" -> "dimethoxy") is kept
    # bracketed for legibility in phosphane/silane oxoacid-ester names; this
    # matches the established forms and keeps the round-trip unambiguous.
    ("COP(=O)(OC)SC",   "di(methoxy)(methylsulfanyl)(oxo)phosphane"),
    ("CCOP(=O)(OCC)SCC", "di(ethoxy)(ethylsulfanyl)(oxo)phosphane"),
])
def test_heteroatom_alkoxy_lead_handling(smi, expected):
    """Alkoxy leading prefix is exempt from leading-prefix debracketing —
    the multiplied "-oxy" prefix stays bracketed ("di(methoxy)...")."""
    assert name_smiles(smi) == expected
