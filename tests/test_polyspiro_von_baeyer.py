"""
tests/test_polyspiro_von_baeyer.py

Tests for von-Baeyer polyspiro nomenclature (P-24.2) and the flat
multi-component polyspiro chain form (P-24.6).

Covers:
  * all-monocyclic polyspiro (>= 3 rings) named in von-Baeyer form
    dispiro/trispiro/...[a.b.c.d...]alkane  (P-24.2.2 / P-24.2.3)
  * heteroatom polyspiro via skeletal a-replacement  (P-24.2.4)
  * charged ring-nitrogen polyspiro (aza...ium)
  * endocyclic unsaturation (-ene)  (P-31.1.5)
  * substituted polyspiro
  * flat dispiro[A-x,y'-B-z',w''-C] for fused/retained components (P-24.6)
  * monospiro and binary heterospiro must remain on their existing paths

Every expected name is OPSIN round-trip verified.  These cases previously
failed (NAMING_ERROR for charged/poly, OPSIN-unparseable nested forms).
"""

from __future__ import annotations

import pytest
from rdkit import Chem

from iupac_namer.engine import name_smiles

try:
    from py2opsin import py2opsin
    HAVE_OPSIN = True
except ImportError:
    HAVE_OPSIN = False


def _constitutional_match(smi_in: str, smi_out: str) -> bool:
    m1 = Chem.MolFromSmiles(smi_in)
    m2 = Chem.MolFromSmiles(smi_out)
    if m1 is None or m2 is None:
        return False
    Chem.RemoveStereochemistry(m1)
    Chem.RemoveStereochemistry(m2)
    return Chem.MolToSmiles(m1) == Chem.MolToSmiles(m2)


def _roundtrip(smi: str, name: str) -> bool:
    parsed = py2opsin([name], output_format="SMILES")
    if not parsed or not parsed[0]:
        return False
    return _constitutional_match(smi, parsed[0])


# ---------------------------------------------------------------------------
# All-monocyclic von-Baeyer polyspiro (P-24.2.2 / P-24.2.3)
# ---------------------------------------------------------------------------

class TestVonBaeyerPolyspiroAllCarbon:

    @pytest.mark.parametrize("smi, expected", [
        # dispiro[5.2.5.2]hexadecane — three cyclohexanes via two spiro atoms
        ("C1CCC2(CC1)CCC1(CC2)CCCCC1",
         "dispiro[5.2.5^{9}.2^{6}]hexadecane"),
        # trispiro[2.0.2^4.1.2^8.1^3]undecane (branched, the BB worked example)
        ("C1CC12CC1(CC1)C1(CC1)C2",
         "trispiro[2.0.2^{4}.1.2^{8}.1^{3}]undecane"),
        # pentaspiro of cyclopropanes
        ("C1CC12C1(CC1)C1(CC1)C1(CC1)C21CC1",
         "pentaspiro[2.0.2^{4}.0.2^{7}.0.2^{10}.0.2^{13}.0^{3}]pentadecane"),
        # dispiro with a larger central ring
        ("C1CCC2(CC1)CCCC1(CCCCC1)CC2",
         "dispiro[5.2.5^{9}.3^{6}]heptadecane"),
        # all-small dispiro
        ("C1CC2(CC1)CC1(CC2)CC1",
         "dispiro[2.1.4^{5}.2^{3}]undecane"),
    ])
    def test_named(self, smi, expected):
        name = name_smiles(smi)
        assert name == expected, f"{smi}: got {name!r}, want {expected!r}"
        if HAVE_OPSIN:
            assert _roundtrip(smi, name), f"round-trip failed: {name}"

    @pytest.mark.parametrize("smi", [
        "C1CCC2(C1)CCC1(CCC2)CCCCC1",
        "C1CCC2(CC1)CCC3(CC2)CCC2(CC3)CCCCC2",   # trispiro
        "C1CC2(C1)C1(CC1)CC2",
        "C1CCCCC12CCC1(C2)CCC2(C1)CCCCC2",       # bigger trispiro
        "C1CC12CCC1(CC1)CC1(CC1)CC2",
    ])
    def test_roundtrip_only(self, smi):
        name = name_smiles(smi)
        assert "spiro[" in name and "ERROR" not in name, f"got {name!r}"
        if HAVE_OPSIN:
            assert _roundtrip(smi, name), f"round-trip failed: {name}"


# ---------------------------------------------------------------------------
# Heteroatom polyspiro via a-replacement (P-24.2.4)
# ---------------------------------------------------------------------------

class TestVonBaeyerPolyspiroHetero:

    @pytest.mark.parametrize("smi, frag", [
        ("O1CC2(OC1)CC1(CCCO1)CC2", "trioxa"),       # three ring O
        ("C1CCC2(CC1)CC1(CCC2)OCCO1", "dioxa"),       # two ring O
        ("O1CCC2(O1)CCC1(CC2)OCCO1", "tetraoxa"),     # four ring O
        ("N1CCC2(CC1)CCC1(CC2)CCNCC1", "diaza"),      # two ring N (neutral)
        ("O1CCCC12CCC1(CC2)CCCCC1", "oxa"),           # one ring O
        ("S1CCCC12CCC1(CC2)CCCC1", "thia"),           # one ring S
    ])
    def test_replacement(self, smi, frag):
        name = name_smiles(smi)
        assert frag in name and "spiro[" in name, f"{smi}: got {name!r}"
        if HAVE_OPSIN:
            assert _roundtrip(smi, name), f"round-trip failed: {name}"

    def test_charged_ring_nitrogen_dispiro(self):
        """Two cationic ring N+ across a dispiro system (the
        6,8-diazoniadispiro[5.1.6.2]hexadecane structure).  Engine emits the
        aza...diium replacement+charge form which round-trips."""
        smi = "C1CCC[N+]2(CC1)CC[N+]1(CCCCC1)C2"
        name = name_smiles(smi)
        assert "diaza" in name and "dispiro" in name, f"got {name!r}"
        if HAVE_OPSIN:
            assert _roundtrip(smi, name), f"round-trip failed: {name}"


# ---------------------------------------------------------------------------
# Endocyclic unsaturation (P-31.1.5)
# ---------------------------------------------------------------------------

class TestPolyspiroUnsaturation:

    @pytest.mark.parametrize("smi", [
        "C1=CCC2(C1)CCC1(CC2)CCCCC1",
        "C1=CC2(CC1)CCC1(CC2)CCCCC1",
        "C1=CC12CC1(CC1)C1(CC1)CC2",
    ])
    def test_ene(self, smi):
        name = name_smiles(smi)
        assert "ene" in name and "spiro[" in name, f"{smi}: got {name!r}"
        if HAVE_OPSIN:
            assert _roundtrip(smi, name), f"round-trip failed: {name}"


# ---------------------------------------------------------------------------
# Substituted polyspiro (numbering exposed to the engine must be correct)
# ---------------------------------------------------------------------------

class TestPolyspiroSubstituted:

    @pytest.mark.parametrize("smi, frag", [
        ("CC1CC12CCCCC2", "methyl"),                       # monospiro baseline
        ("C1CCC2(CC1)CCC1(CC2)CC(C)CCC1", "methyl"),       # methyl on dispiro
        ("C1CCC2(CC1)OCC1(CO2)CCCCC1", "oxa"),             # dioxa dispiro
    ])
    def test_substituent(self, smi, frag):
        name = name_smiles(smi)
        assert frag in name and "ERROR" not in name, f"{smi}: got {name!r}"
        if HAVE_OPSIN:
            assert _roundtrip(smi, name), f"round-trip failed: {name}"


# ---------------------------------------------------------------------------
# Flat multi-component polyspiro of fused/retained components (P-24.6)
# ---------------------------------------------------------------------------

class TestMultiComponentPolyspiro:

    def test_fluorene_cyclohexane_indene(self):
        """dispiro[fluorene-9,1'-cyclohexane-4',1''-indene]: a chain of two
        fused components joined through a central cyclohexane.  Must produce a
        FLAT dispiro[A-x,y'-B-z',w''-C] form (not the OPSIN-unparseable nested
        spiro[A-spiro[B-C]] form), and cite fused components by retained name.
        """
        smi = "C1=CC2(CCC3(CC2)c2ccccc2-c2ccccc23)c2ccccc21"
        name = name_smiles(smi)
        assert name.startswith("dispiro["), f"expected flat dispiro, got {name!r}"
        assert name.count("spiro") == 1, f"nested spiro leaked: {name!r}"
        assert "fluorene" in name, f"fused component not cited by name: {name!r}"
        if HAVE_OPSIN:
            assert _roundtrip(smi, name), f"round-trip failed: {name}"

    def test_two_fused_components_via_central_cyclohexane(self):
        """Two indene-derived fused components joined through a central
        cyclohexane.  Exercises the flat dispiro chain with two distinct fused
        partners on either side of a monocyclic link."""
        smi = "C1=Cc2ccccc2C13CCC1(CC3)C3=Cc2ccccc2C13"
        name = name_smiles(smi)
        assert name.startswith("dispiro["), f"expected flat dispiro, got {name!r}"
        assert name.count("spiro") == 1, f"nested spiro leaked: {name!r}"
        assert "cyclohexane" in name, f"central ring not cited: {name!r}"
        if HAVE_OPSIN:
            assert _roundtrip(smi, name), f"round-trip failed: {name}"


# ---------------------------------------------------------------------------
# Regression: monospiro and binary heterospiro stay on their own paths
# ---------------------------------------------------------------------------

class TestSpiroRegression:

    @pytest.mark.parametrize("smi, expected", [
        ("C1CCC2(CC1)CCCCC2", "spiro[5.5]undecane"),
        ("C1CC11CCCC1", "spiro[2.4]heptane"),
        ("O1CCCC12CCCCC2", "1-oxaspiro[4.5]decane"),
        ("O1CC2(CCCCC2)OC1", "1,3-dioxaspiro[4.5]decane"),
    ])
    def test_monospiro_unchanged(self, smi, expected):
        name = name_smiles(smi)
        assert name == expected, f"{smi}: got {name!r}, want {expected!r}"
        if HAVE_OPSIN:
            assert _roundtrip(smi, name), f"round-trip failed: {name}"

    @pytest.mark.parametrize("smi, frag", [
        # spiro of one fused + one monocyclic (P-24.5) must NOT become VB.
        ("C12(C=CC3=CC=CC=C13)CCCCC2", "spiro[cyclohexane-1,1'-"),
        ("C1=CC=CC=2OC3=CC=CC=C3C3(C12)CCNCC3", "spiro[piperidine-4,9'-xanthene]"),
        # decalin spiro dioxolane (single spiro atom, fused partner)
        ("C1CCC2CC3(CCC2C1)OCCO3", "spiro["),
    ])
    def test_p245_component_form_unchanged(self, smi, frag):
        name = name_smiles(smi)
        assert frag in name, f"{smi}: got {name!r}, want substring {frag!r}"
        if HAVE_OPSIN:
            assert _roundtrip(smi, name), f"round-trip failed: {name}"
