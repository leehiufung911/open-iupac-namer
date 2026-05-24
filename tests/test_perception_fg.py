"""
tests/test_perception_fg.py

Unit tests for iupac_namer.perception.fg_detection.FGDetection.

Each test constructs a molecule from SMILES and exercises FGDetection
directly, without going through the full Perception facade.
"""

from __future__ import annotations

import pytest
from rdkit import Chem

from iupac_namer.perception.atoms import AtomAnalysis
from iupac_namer.perception.fg_detection import FGDetection
from iupac_namer.perception.rings import RingAnalysis
from iupac_namer.types import AmbiguityPoint, DetectedFG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def mol_from_smiles(smiles: str) -> object:
    """Return a sanitised RDKit mol. Fail fast if SMILES is invalid."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles!r}"
    return mol


def make_fgd(smiles: str) -> FGDetection:
    """Construct FGDetection for a SMILES string."""
    mol = mol_from_smiles(smiles)
    aa = AtomAnalysis(mol)
    ra = RingAnalysis(mol, aa)
    return FGDetection(mol, aa, ra)


# ---------------------------------------------------------------------------
# Basic single-FG detection
# ---------------------------------------------------------------------------


class TestSingleFGDetection:
    def test_ethanol_detects_alcohol(self):
        """CCO → 1 alcohol."""
        fgd = make_fgd("CCO")
        alcohols = fgd.fgs_by_type("alcohol")
        assert len(alcohols) == 1, f"Expected 1 alcohol, got {len(alcohols)}"

    def test_acetone_detects_ketone(self):
        """CC(=O)C → 1 ketone."""
        fgd = make_fgd("CC(=O)C")
        ketones = fgd.fgs_by_type("ketone")
        assert len(ketones) == 1, f"Expected 1 ketone, got {len(ketones)}"

    def test_propanal_detects_aldehyde(self):
        """CCC=O → 1 aldehyde."""
        fgd = make_fgd("CCC=O")
        aldehydes = fgd.fgs_by_type("aldehyde")
        assert len(aldehydes) == 1, f"Expected 1 aldehyde, got {len(aldehydes)}"

    def test_acetonitrile_detects_nitrile(self):
        """CC#N → 1 nitrile."""
        fgd = make_fgd("CC#N")
        nitriles = fgd.fgs_by_type("nitrile")
        assert len(nitriles) == 1, f"Expected 1 nitrile, got {len(nitriles)}"


# ---------------------------------------------------------------------------
# Subsumption deconfliction
# ---------------------------------------------------------------------------


class TestSubsumptionDeconfliction:
    def test_acetic_acid_no_separate_ketone_or_alcohol(self):
        """CC(=O)O → carboxylic_acid only; no separate ketone or alcohol."""
        fgd = make_fgd("CC(=O)O")
        all_types = {fg.type for fg in fgd.detected_fgs}
        assert "carboxylic_acid" in all_types, "Expected carboxylic_acid"
        assert "ketone" not in all_types, "ketone should be subsumed by carboxylic_acid"
        assert "alcohol" not in all_types, "alcohol should be subsumed by carboxylic_acid"

    def test_acetamide_no_separate_ketone_or_amine(self):
        """CC(=O)N → amide only; no separate ketone or amine."""
        fgd = make_fgd("CC(=O)N")
        all_types = {fg.type for fg in fgd.detected_fgs}
        assert "amide" in all_types, "Expected amide"
        assert "ketone" not in all_types, "ketone should be subsumed by amide"
        assert "amine" not in all_types, "amine should be subsumed by amide"

    def test_acetic_acid_count(self):
        """CC(=O)O → exactly 1 detected FG (the carboxylic acid)."""
        fgd = make_fgd("CC(=O)O")
        acids = fgd.fgs_by_type("carboxylic_acid")
        assert len(acids) == 1

    def test_acetamide_count(self):
        """CC(=O)N → exactly 1 detected FG (the amide)."""
        fgd = make_fgd("CC(=O)N")
        amides = fgd.fgs_by_type("amide")
        assert len(amides) == 1


# ---------------------------------------------------------------------------
# Non-overlapping multi-FG detection
# ---------------------------------------------------------------------------


class TestMultiFGDetection:
    def test_aminoethanol_detects_amine_and_alcohol(self):
        """NCCO → 1 amine + 1 alcohol (no overlap)."""
        fgd = make_fgd("NCCO")
        amines = fgd.fgs_by_type("amine")
        alcohols = fgd.fgs_by_type("alcohol")
        assert len(amines) == 1, f"Expected 1 amine, got {len(amines)}"
        assert len(alcohols) == 1, f"Expected 1 alcohol, got {len(alcohols)}"

    def test_glycine_detects_amine_and_acid(self):
        """NCC(=O)O → 1 amine + 1 carboxylic_acid."""
        fgd = make_fgd("NCC(=O)O")
        amines = fgd.fgs_by_type("amine")
        acids = fgd.fgs_by_type("carboxylic_acid")
        assert len(amines) == 1, f"Expected 1 amine, got {len(amines)}"
        assert len(acids) == 1, f"Expected 1 carboxylic_acid, got {len(acids)}"

    def test_no_overlap_in_aminoethanol(self):
        """Amine and alcohol atoms must not overlap in NCCO."""
        fgd = make_fgd("NCCO")
        amines = fgd.fgs_by_type("amine")
        alcohols = fgd.fgs_by_type("alcohol")
        if amines and alcohols:
            overlap = amines[0].atoms & alcohols[0].atoms
            assert not overlap, f"Unexpected atom overlap: {overlap}"


# ---------------------------------------------------------------------------
# Prefix-only group detection
# ---------------------------------------------------------------------------


class TestPrefixOnlyGroups:
    def test_chlorobenzene_detects_chloro(self):
        """Clc1ccccc1 → 1 chloro (prefix-only)."""
        fgd = make_fgd("Clc1ccccc1")
        chloros = fgd.fgs_by_type("chloro")
        assert len(chloros) == 1, f"Expected 1 chloro, got {len(chloros)}"

    def test_chloro_not_suffix_eligible(self):
        """chloro is prefix-only, so suffix_eligible must be False."""
        fgd = make_fgd("Clc1ccccc1")
        chloros = fgd.fgs_by_type("chloro")
        assert len(chloros) == 1
        assert chloros[0].suffix_eligible is False


# ---------------------------------------------------------------------------
# N-oxide additive group detection
# ---------------------------------------------------------------------------


class TestAdditiveGroups:
    def test_pyridine_n_oxide_detected(self):
        """[O-][n+]1ccccc1 → N-oxide additive group."""
        fgd = make_fgd("[O-][n+]1ccccc1")
        additives = fgd.additive_groups
        n_oxides = [a for a in additives if a.get("type") == "oxide" and a.get("center_element") == "N"]
        assert len(n_oxides) == 1, f"Expected 1 N-oxide, got {len(n_oxides)}: {additives}"

    def test_additive_groups_is_list(self):
        """additive_groups should always be a list."""
        fgd = make_fgd("CCO")
        assert isinstance(fgd.additive_groups, list)

    def test_simple_molecule_no_additive_groups(self):
        """Simple alcohol has no additive groups."""
        fgd = make_fgd("CCO")
        assert fgd.additive_groups == []


# ---------------------------------------------------------------------------
# suffix_eligible_fgs
# ---------------------------------------------------------------------------


class TestSuffixEligibleFGs:
    def test_ethanol_suffix_eligible(self):
        """CCO: the alcohol FG is suffix-eligible."""
        fgd = make_fgd("CCO")
        eligible = fgd.suffix_eligible_fgs()
        assert len(eligible) >= 1
        types = {fg.type for fg in eligible}
        assert "alcohol" in types

    def test_chlorobenzene_no_suffix_eligible(self):
        """Clc1ccccc1: chloro is prefix-only; no suffix-eligible FG."""
        fgd = make_fgd("Clc1ccccc1")
        eligible = fgd.suffix_eligible_fgs()
        assert all(fg.suffix_eligible for fg in eligible)
        # chloro should NOT appear
        types = {fg.type for fg in eligible}
        assert "chloro" not in types

    def test_suffix_eligible_returns_tuple(self):
        """suffix_eligible_fgs returns a tuple."""
        fgd = make_fgd("CCO")
        assert isinstance(fgd.suffix_eligible_fgs(), tuple)


# ---------------------------------------------------------------------------
# fgs_by_type
# ---------------------------------------------------------------------------


class TestFGsByType:
    def test_acetic_acid_by_type(self):
        """fgs_by_type('carboxylic_acid') returns only carboxylic acids."""
        fgd = make_fgd("CC(=O)O")
        result = fgd.fgs_by_type("carboxylic_acid")
        assert all(fg.type == "carboxylic_acid" for fg in result)

    def test_missing_type_returns_empty(self):
        """fgs_by_type for absent type returns empty tuple."""
        fgd = make_fgd("CCCC")
        result = fgd.fgs_by_type("carboxylic_acid")
        assert result == ()

    def test_fgs_by_type_returns_tuple(self):
        fgd = make_fgd("CCO")
        assert isinstance(fgd.fgs_by_type("alcohol"), tuple)


# ---------------------------------------------------------------------------
# fg_at_atom
# ---------------------------------------------------------------------------


class TestFGAtAtom:
    def test_fg_at_oxygen_in_ethanol(self):
        """In CCO, the O atom should belong to the alcohol FG."""
        mol = mol_from_smiles("CCO")
        aa = AtomAnalysis(mol)
        ra = RingAnalysis(mol, aa)
        fgd = FGDetection(mol, aa, ra)

        # Find which atom index is oxygen
        o_idx = next(
            atom.GetIdx()
            for atom in mol.GetAtoms()
            if atom.GetSymbol() == "O"
        )
        fg = fgd.fg_at_atom(o_idx)
        assert fg is not None, "Expected an FG at the oxygen atom"
        assert fg.type == "alcohol"

    def test_fg_at_carbon_chain_returns_none(self):
        """A plain carbon chain has no FG; fg_at_atom returns None."""
        mol = mol_from_smiles("CCCC")
        aa = AtomAnalysis(mol)
        ra = RingAnalysis(mol, aa)
        fgd = FGDetection(mol, aa, ra)

        result = fgd.fg_at_atom(0)
        assert result is None


# ---------------------------------------------------------------------------
# detected_fgs and ambiguity_points are tuples
# ---------------------------------------------------------------------------


class TestReturnTypes:
    def test_detected_fgs_is_tuple(self):
        fgd = make_fgd("CCO")
        assert isinstance(fgd.detected_fgs, tuple)

    def test_ambiguity_points_is_tuple(self):
        fgd = make_fgd("CCO")
        assert isinstance(fgd.ambiguity_points, tuple)

    def test_detected_fgs_are_detected_fg_instances(self):
        fgd = make_fgd("CCO")
        for fg in fgd.detected_fgs:
            assert isinstance(fg, DetectedFG)

    def test_ambiguity_points_are_ambiguity_point_instances(self):
        fgd = make_fgd("CCO")
        for ap in fgd.ambiguity_points:
            assert isinstance(ap, AmbiguityPoint)


# ---------------------------------------------------------------------------
# Facade integration
# ---------------------------------------------------------------------------


class TestFacadeIntegration:
    def test_perception_fgs_property(self):
        """Perception.fgs should return a FGDetection instance."""
        from iupac_namer.perception import Perception

        mol = mol_from_smiles("CCO")
        perc = Perception(mol)
        fgd = perc.fgs
        assert isinstance(fgd, FGDetection)

    def test_perception_fgs_cached(self):
        """Calling Perception.fgs twice returns the same object."""
        from iupac_namer.perception import Perception

        mol = mol_from_smiles("CC(=O)O")
        perc = Perception(mol)
        fgd1 = perc.fgs
        fgd2 = perc.fgs
        assert fgd1 is fgd2

    def test_glycine_via_facade(self):
        """NCC(=O)O via facade: 1 amine + 1 carboxylic_acid."""
        from iupac_namer.perception import Perception

        mol = mol_from_smiles("NCC(=O)O")
        perc = Perception(mol)
        fgd = perc.fgs
        assert len(fgd.fgs_by_type("amine")) == 1
        assert len(fgd.fgs_by_type("carboxylic_acid")) == 1
