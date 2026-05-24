"""Ring-unsaturation-degree correctness for systematic (non-benzenoid) rings.

The Hantzsch-Widman / replacement-nomenclature paths previously miscomputed the
number of endocyclic double bonds relative to the mancude (maximally
non-cumulative unsaturated) reference state, producing names that round-trip to
a DIFFERENT structure:

  * carrier-free fully-mancude HW rings (1,4-dioxine, 1,4-dithiine,
    1,4-oxathiine) were emitted SATURATED ("1,4-dioxane" etc.)
  * partially-saturated HW rings (4,5,6,7-tetrahydro-1,4-thiazepine,
    2,3-dihydro-1H-phosphole) were emitted FULLY mancude (adding unsaturation)
  * fully-aromatic heteromacrocycles (aza[14]annulene) were emitted SATURATED
    ("1-azacyclotetradecane")

This module pins the corrected behaviour AND verifies every emitted name
round-trips through OPSIN to the input structure (the only definition of
"correct" here — a name that parses back to a different molecule is wrong even
if it is a legal IUPAC string).

See P-31.1.4 (ring unsaturation / hydro prefixes), P-25.7.1.3 (indicated-H),
P-22.1.4 / P-31.1.3 (annulenes / replacement nomenclature).
"""
from __future__ import annotations

import os

os.environ.setdefault(
    "JAVA_HOME",
    os.environ.get("JAVA_HOME", ""),
)
os.environ["PATH"] = (
    os.environ["JAVA_HOME"] + "/bin" + os.pathsep + os.environ.get("PATH", "")
)

import pytest
from rdkit import Chem

from iupac_namer.engine import name_smiles


def _canonical(smiles: str) -> str | None:
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(m) if m is not None else None


def _opsin_round_trip(name: str) -> str | None:
    try:
        from py2opsin import py2opsin
    except ImportError:  # pragma: no cover
        pytest.skip("py2opsin not installed")
    out = py2opsin(name)
    if not out:
        return None
    return _canonical(out)


# ---------------------------------------------------------------------------
# 1. Carrier-free fully-mancude HW rings → unsaturated stem (not saturated)
# ---------------------------------------------------------------------------

CARRIER_FREE_MANCUDE = [
    ("C1=COC=CO1", "1,4-dioxine"),    # was wrongly 1,4-dioxane
    ("C1=CSC=CS1", "1,4-dithiine"),   # was wrongly 1,4-dithiane
    ("C1=COC=CS1", "1,4-oxathiine"),  # was wrongly 1,4-oxathiane
]


@pytest.mark.parametrize("smi,expected", CARRIER_FREE_MANCUDE)
def test_carrier_free_mancude_name(smi, expected):
    assert name_smiles(smi) == expected


# ---------------------------------------------------------------------------
# 2. Partially-saturated HW rings → mancude parent + hydro prefix
# ---------------------------------------------------------------------------

PARTIAL_HYDRO = [
    ("C1=CSCCCN1", "4,5,6,7-tetrahydro-1,4-thiazepine"),  # was 1,4-thiazepine
    ("C1=CPCC1", "2,3-dihydro-1H-phosphole"),             # was phosphole
    ("C1=C[AsH]CC1", "2,3-dihydro-1H-arsole"),
    ("O1CCOC=C1", "2,3-dihydro-1,4-dioxine"),
]


@pytest.mark.parametrize("smi,expected", PARTIAL_HYDRO)
def test_partial_hydro_name(smi, expected):
    assert name_smiles(smi) == expected


# ---------------------------------------------------------------------------
# 3. Fully-aromatic heteromacrocycles → mancude polyene (not saturated)
# ---------------------------------------------------------------------------

MANCUDE_MACROCYCLE = [
    # aza[14]annulene — was wrongly "1-azacyclotetradecane".  Either Kekulé
    # locant set round-trips; we pin the heptaene parent form.
    ("c1ccccccncccccc1",
     "1-azacyclotetradeca-2,4,6,8,10,12,14-heptaene"),
    ("c1cccccncnccccc1",
     "1,3-diazacyclotetradeca-1,3,5,7,9,11,13-heptaene"),
]


@pytest.mark.parametrize("smi,expected", MANCUDE_MACROCYCLE)
def test_mancude_macrocycle_name(smi, expected):
    assert name_smiles(smi) == expected


# ---------------------------------------------------------------------------
# 3b. Fully-saturated all-heteroatom (carbon-free) rings that RDKit aromatises
#     by lone-pair donation but which carry ZERO Kekulé double bonds → must use
#     the SATURATED HW stem, NOT the mancude azole/phosphole (P-31.1.4).
#
#     Odd-membered all-NH / all-PH / all-chalcogen rings are flagged aromatic by
#     RDKit yet are structurally the saturated ring.  Naming them with the
#     unsaturated stem (pentaazole / pentaphosphole) round-trips through OPSIN
#     to a DIFFERENT structure (two real double bonds + one indicated-H carrier).
# ---------------------------------------------------------------------------

ALL_HETEROATOM_SATURATED = [
    ("N1NNNN1", "1,2,3,4,5-pentaazolidine"),    # was wrongly 1,2,3,4,5-pentaazole
    ("P1PPPP1", "1,2,3,4,5-pentaphospholane"),  # was wrongly 1,2,3,4,5-pentaphosphole
    ("S1SSSS1", "1,2,3,4,5-pentathiolane"),     # was wrongly 1,2,3,4,5-pentathiole
    ("O1OOOO1", "1,2,3,4,5-pentaoxolane"),      # was wrongly 1,2,3,4,5-pentaoxole
    ("N1NN1", "1,2,3-triaziridine"),            # was wrongly 1,2,3-triazirene
    ("P1PP1", "1,2,3-triphosphirane"),          # was wrongly 1,2,3-triphosphirene
    ("S1SS1", "1,2,3-trithiirane"),             # was wrongly 1,2,3-trithiirene
    ("O1OO1", "1,2,3-trioxirane"),              # was wrongly 1,2,3-trioxirene
]


@pytest.mark.parametrize("smi,expected", ALL_HETEROATOM_SATURATED)
def test_all_heteroatom_saturated_name(smi, expected):
    assert name_smiles(smi) == expected


# The genuine MANCUDE all-heteroatom ring (explicit Kekulé double bonds) MUST
# still use the unsaturated stem — the structural-saturation override only fires
# when there are ZERO Kekulé endocyclic double bonds.
ALL_HETEROATOM_MANCUDE = [
    ("N1=NN=NN1", "1,2,3,4,5-pentaazole"),  # 2 real DBs + indicated-H → mancude
]


@pytest.mark.parametrize("smi,expected", ALL_HETEROATOM_MANCUDE)
def test_all_heteroatom_mancude_name(smi, expected):
    assert name_smiles(smi) == expected


# ---------------------------------------------------------------------------
# 4. Round-trip: EVERY corrected name must parse back to the input structure
# ---------------------------------------------------------------------------

ALL_CORRECTED = (
    CARRIER_FREE_MANCUDE + PARTIAL_HYDRO + MANCUDE_MACROCYCLE
    + ALL_HETEROATOM_SATURATED + ALL_HETEROATOM_MANCUDE
)


@pytest.mark.parametrize("smi,_expected", ALL_CORRECTED)
def test_corrected_names_round_trip(smi, _expected):
    got = name_smiles(smi)
    assert not got.startswith("[NAMING ERROR"), f"Namer failed for {smi}: {got}"
    assert _opsin_round_trip(got) == _canonical(smi), (
        f"Round-trip mismatch for {smi}: name={got!r}"
    )


# ---------------------------------------------------------------------------
# 5. Guards — the fix MUST NOT regress these
# ---------------------------------------------------------------------------

class TestGuards:
    """Behaviours the unsaturation fix must leave unchanged."""

    # Single-O / single-N 6-rings keep the retained pyran/pyridine path: their
    # HW unsaturated stems "oxine" / "azine" are OPSIN-rejected.
    def test_2H_pyran_retained(self):
        assert name_smiles("O1CC=CC=C1") == "2H-pyran"

    def test_4H_pyran_retained(self):
        assert name_smiles("C1=COC=CC1") == "4H-pyran"

    def test_thiopyran_retained(self):
        name = name_smiles("S1CC=CC=C1")
        assert "thiopyran" in name.lower() or "thiine" in name.lower()

    # Heavy-element 6-rings limited to 2 DBs by valence stay bare (no spurious
    # hydro prefix and no spurious indicated-H).
    def test_iodinine_bare(self):
        assert name_smiles("[IH]1CC=CC=C1") == "iodinine"

    def test_mercurinine_bare(self):
        assert name_smiles("[Hg]1CC=CC=C1") == "mercurinine"

    # Fully-saturated HW / retained rings stay saturated.
    def test_morpholine(self):
        assert name_smiles("O1CCNCC1") == "morpholine"

    def test_oxane(self):
        name = name_smiles("O1CCCCC1")
        assert "oxane" in name

    # Indicated-H mancude tautomers unchanged.
    def test_2H_1_3_oxazine(self):
        assert name_smiles("O1CN=CC=C1") == "2H-1,3-oxazine"

    def test_1H_azepine(self):
        assert name_smiles("N1C=CC=CC=C1") == "1H-azepine"

    # Genuinely aromatic HW rings unchanged.
    def test_phosphinine(self):
        assert name_smiles("[P]1=CC=CC=C1") == "phosphinine"

    def test_arsinine(self):
        assert name_smiles("[As]1=CC=CC=C1") == "arsinine"

    # Saturated macrocycles (crown ethers / azacyclotetradecane) stay -ane.
    def test_saturated_aza_macrocycle(self):
        assert name_smiles("C1CCCCCCNCCCCCC1") == "1-azacyclotetradecane"

    def test_crown_ether_saturated(self):
        assert name_smiles("C1COCCOCCOCCO1") == "1,4,7,10-tetraoxacyclododecane"

    # Carbon-bearing aromatics kekulise to >0 double bonds, so the
    # structural-saturation override (which fires only at 0 Kekulé DBs) leaves
    # them untouched: pyrrole / thiophene / furan / imidazole / pyrazole keep
    # their aromatic (retained) names.
    def test_pyrrole_unchanged(self):
        assert name_smiles("c1cc[nH]c1") == "1H-pyrrole"

    def test_thiophene_unchanged(self):
        assert name_smiles("c1ccsc1") == "thiophene"

    def test_furan_unchanged(self):
        assert name_smiles("c1ccoc1") == "furan"

    def test_imidazole_unchanged(self):
        assert name_smiles("c1cnc[nH]1") == "1H-imidazole"

    def test_pyrazole_unchanged(self):
        assert name_smiles("c1cc[nH]n1") == "1H-pyrazole"
