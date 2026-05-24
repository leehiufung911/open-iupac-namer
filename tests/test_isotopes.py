"""
tests/test_isotopes.py

Stage 6 R1-D — isotope label perception + emission tests.

Covers the 10-probe matrix from ``docs/opsin_audit_misc.md`` §3:

* Boughton D / D4 equivalents (emitted here in IUPAC bracketed style).
* Locanted 13C on ethanol / propan-1-ol.
* Unlocanted 13C on methane.
* 14C on methane.
* 15N on indole.
* Combined stereo + isotope on ethan-1-ol.

Also verifies:

* The ``collect_isotope_labels`` helper correctly groups atoms by
  (locant, element, mass_number).
* ``render_isotope_labels`` produces the expected IUPAC bracket strings.
* Non-isotope-labeled SMILES round-trip to identical names as the
  pre-R1-D baseline (guards the 1177/1181 eval score).
"""

from __future__ import annotations

import pytest
from rdkit import Chem

from iupac_namer.engine import name_smiles
from iupac_namer.isotope import (
    collect_isotope_labels,
    render_isotope_label,
    render_isotope_labels,
)
from iupac_namer.types import IsotopeLabel, Locant


# ---------------------------------------------------------------------------
# Regression guard — non-isotope inputs must not acquire isotope prefixes.
# ---------------------------------------------------------------------------

REGRESSION_PROBES = [
    ("CCO",          "ethanol"),
    ("CC(=O)C",      "propan-2-one"),
    ("c1ccccc1",     "benzene"),
    ("C1CCCCC1",     "cyclohexane"),
    ("CCCCCCCC",     "octane"),
    ("c1ccc2ccccc2c1", "naphthalene"),
]


@pytest.mark.parametrize("smiles, expected", REGRESSION_PROBES)
def test_non_isotope_unchanged(smiles: str, expected: str) -> None:
    """Baseline: unlabeled molecules produce the pre-R1-D name unchanged."""
    assert name_smiles(smiles) == expected


# ---------------------------------------------------------------------------
# IUPAC-bracketed emission
# ---------------------------------------------------------------------------


def test_deuterium_on_methanol_single() -> None:
    """Single D on methanol C1 → '(1-2H)-methanol'."""
    name = name_smiles("[2H]CO")
    assert "2H" in name
    assert "methanol" in name


def test_perdeuteromethane() -> None:
    """CD4 → '(1-2H4)-methane' (locant 1, count subscript 4)."""
    name = name_smiles("[2H]C([2H])([2H])[2H]")
    assert "2H4" in name
    assert "methane" in name


def test_methanol_d4_includes_hydroxyl() -> None:
    """CD3-OD → 3 D on C1 plus the hydroxyl D cited as O-2H (P-82.2.3.1.1).

    The O-bound deuterium is addressed by the italic element-symbol locant
    rather than collapsed onto the anchor carbon's numeric locant, so the
    name round-trips to the correct CD3-OD structure (not CD4-OH).
    """
    name = name_smiles("[2H]C([2H])([2H])O[2H]")
    assert "2H3" in name and "O-2H" in name
    assert "methanol" in name


def test_carbon_13_on_methane() -> None:
    """13C on methane → '(1-13C)-methane'."""
    name = name_smiles("[13CH4]")
    assert "13C" in name
    assert "methane" in name


def test_carbon_14_on_methane() -> None:
    """14C on methane → '(1-14C)-methane'."""
    name = name_smiles("[14CH4]")
    assert "14C" in name
    assert "methane" in name


def test_carbon_13_locanted_on_ethanol_c1() -> None:
    """13C at the hydroxyl-bearing carbon of ethanol → '(1-13C)-ethanol'."""
    name = name_smiles("O[13CH2]C")
    assert "1-13C" in name
    assert "ethanol" in name


def test_carbon_13_locanted_on_ethanol_c2() -> None:
    """13C at the methyl carbon of ethanol → '(2-13C)-ethanol'."""
    name = name_smiles("OC[13CH3]")
    assert "2-13C" in name
    assert "ethanol" in name


def test_nitrogen_15_on_indole() -> None:
    """15N on indole N1 → '(1-15N)-1H-indole'."""
    name = name_smiles("c1ccc2c(c1)cc[15nH]2")
    assert "15N" in name
    assert "indole" in name


def test_stereo_plus_isotope() -> None:
    """Combined R/S and deuterium: stereo then isotope then stem."""
    name = name_smiles("[2H][C@H](C)O")
    assert "(1R)" in name
    assert "2H" in name
    assert "ethanol" in name
    # Stereo precedes isotope bracket.
    assert name.index("1R") < name.index("2H")


# ---------------------------------------------------------------------------
# collect_isotope_labels helper
# ---------------------------------------------------------------------------


def _loc(label: str) -> Locant:
    """Build a numeric Locant from a digit string."""
    return Locant(
        label=label, is_numeric=True, _numeric_value=int(label), suffix=""
    )


def test_collect_single_deuterium_on_methane() -> None:
    mol = Chem.MolFromSmiles("[2H]C")
    labels = collect_isotope_labels(mol, {1: _loc("1")})
    assert labels == (IsotopeLabel(locant=_loc("1"), element="H",
                                    mass_number=2, count=1),)


def test_collect_thirteen_c_unmapped_drops() -> None:
    """A 13C atom that's not in atom_to_locant produces no label."""
    mol = Chem.MolFromSmiles("[13CH3]CO")
    labels = collect_isotope_labels(mol, {1: _loc("1")})
    assert labels == ()


def test_collect_groups_by_locant_element_mass() -> None:
    """Three D's on the same C all collapse into one label with count=3."""
    mol = Chem.MolFromSmiles("[2H]C([2H])([2H])CO")
    labels = collect_isotope_labels(mol, {1: _loc("2"), 4: _loc("1")})
    assert len(labels) == 1
    lbl = labels[0]
    assert lbl.count == 3
    assert lbl.mass_number == 2
    assert lbl.element == "H"
    assert lbl.locant == _loc("2")


# ---------------------------------------------------------------------------
# render_isotope_labels
# ---------------------------------------------------------------------------


def test_render_unlocanted_single_label() -> None:
    lbl = IsotopeLabel(locant=None, element="H", mass_number=2, count=1)
    assert render_isotope_label(lbl) == "2H"
    assert render_isotope_labels((lbl,)) == "(2H)"


def test_render_unlocanted_multi_count() -> None:
    lbl = IsotopeLabel(locant=None, element="H", mass_number=2, count=4)
    assert render_isotope_label(lbl) == "2H4"
    assert render_isotope_labels((lbl,)) == "(2H4)"


def test_render_locanted() -> None:
    lbl = IsotopeLabel(locant=_loc("1"), element="C", mass_number=13, count=1)
    assert render_isotope_label(lbl) == "1-13C"
    assert render_isotope_labels((lbl,)) == "(1-13C)"


def test_render_multiple_labels_joined() -> None:
    l1 = IsotopeLabel(locant=_loc("1"), element="H", mass_number=2, count=1)
    l2 = IsotopeLabel(locant=_loc("2"), element="C", mass_number=13, count=1)
    out = render_isotope_labels((l1, l2))
    assert out == "(1-2H,2-13C)"


def test_render_empty_is_empty_string() -> None:
    assert render_isotope_labels(()) == ""
