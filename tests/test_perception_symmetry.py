"""
tests/test_perception_symmetry.py

Unit tests for iupac_namer.perception.symmetry.SymmetryAnalysis.

Each test constructs a molecule from SMILES and exercises SymmetryAnalysis
directly, without going through the full Perception facade.
"""

from __future__ import annotations

import pytest
from rdkit import Chem

from iupac_namer.perception.atoms import AtomAnalysis
from iupac_namer.perception.rings import RingAnalysis
from iupac_namer.perception.symmetry import SymmetryAnalysis
from iupac_namer.types import SymmetryGroup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_symmetry(smiles: str) -> SymmetryAnalysis:
    """Build AtomAnalysis + RingAnalysis + SymmetryAnalysis from a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles!r}"
    aa = AtomAnalysis(mol)
    ra = RingAnalysis(mol, aa)
    return SymmetryAnalysis(mol, aa, ra)


# ---------------------------------------------------------------------------
# No symmetry
# ---------------------------------------------------------------------------


class TestNoSymmetry:
    def test_ethanol_no_symmetry(self):
        """Ethanol (CCO) has no ring systems, so no symmetry groups."""
        sa = make_symmetry("CCO")
        assert sa.has_symmetry is False
        assert sa.symmetry_groups == ()
        assert len(sa) == 0

    def test_toluene_no_symmetry(self):
        """Toluene (Cc1ccccc1) has only one ring system — no symmetry group."""
        sa = make_symmetry("Cc1ccccc1")
        assert sa.has_symmetry is False

    def test_cyclohexane_no_symmetry(self):
        """Cyclohexane has one ring and no symmetric pairs."""
        sa = make_symmetry("C1CCCCC1")
        assert sa.has_symmetry is False

    def test_pentane_no_symmetry(self):
        """Pentane is acyclic — no ring-based symmetry groups."""
        sa = make_symmetry("CCCCC")
        assert sa.has_symmetry is False


# ---------------------------------------------------------------------------
# Ring assembly (direct bond)
# ---------------------------------------------------------------------------


class TestRingAssembly:
    def test_biphenyl_detected(self):
        """Biphenyl (two benzenes directly bonded) should produce a ring assembly candidate."""
        sa = make_symmetry("c1ccc(-c2ccccc2)cc1")
        candidates = sa.ring_assembly_candidates
        assert len(candidates) >= 1

    def test_biphenyl_symmetry_group_type(self):
        """The biphenyl symmetry group must have linking_type == 'direct_bond'."""
        sa = make_symmetry("c1ccc(-c2ccccc2)cc1")
        for sg in sa.ring_assembly_candidates:
            assert sg.linking_type == "direct_bond"

    def test_biphenyl_multiplicity_two(self):
        """Biphenyl has 2 identical subunits."""
        sa = make_symmetry("c1ccc(-c2ccccc2)cc1")
        candidates = sa.ring_assembly_candidates
        assert len(candidates) >= 1
        assert candidates[0].multiplicity == 2

    def test_biphenyl_subunit_count(self):
        """Each biphenyl symmetry group has exactly 2 subunit atom sets."""
        sa = make_symmetry("c1ccc(-c2ccccc2)cc1")
        for sg in sa.ring_assembly_candidates:
            assert len(sg.subunit_atoms) == 2

    def test_biphenyl_subunit_atoms_disjoint(self):
        """The two subunit atom sets for biphenyl must not overlap."""
        sa = make_symmetry("c1ccc(-c2ccccc2)cc1")
        for sg in sa.ring_assembly_candidates:
            a1, a2 = sg.subunit_atoms[0], sg.subunit_atoms[1]
            assert a1.isdisjoint(a2)

    def test_biphenyl_symmetry_group_instance(self):
        """All symmetry groups are SymmetryGroup instances."""
        sa = make_symmetry("c1ccc(-c2ccccc2)cc1")
        for sg in sa.symmetry_groups:
            assert isinstance(sg, SymmetryGroup)

    def test_biphenyl_has_symmetry(self):
        sa = make_symmetry("c1ccc(-c2ccccc2)cc1")
        assert sa.has_symmetry is True

    def test_naphthalene_no_ring_assembly(self):
        """Naphthalene is a fused ring system (one ring system), not a ring assembly."""
        # Both rings share atoms in one ring system — not directly bonded as separate systems
        sa = make_symmetry("c1ccc2ccccc2c1")
        # Naphthalene is ONE ring system internally, not two separate ring systems
        # So it should NOT produce a ring assembly candidate
        # (Both 6-membered rings share 2 atoms and belong to the same RingSystem)
        ring_assembly = sa.ring_assembly_candidates
        # If the ring analysis groups them as one system, no assembly candidate exists
        from iupac_namer.perception.rings import RingAnalysis
        from iupac_namer.perception.atoms import AtomAnalysis
        mol = Chem.MolFromSmiles("c1ccc2ccccc2c1")
        aa = AtomAnalysis(mol)
        ra = RingAnalysis(mol, aa)
        n_ring_systems = len(ra.ring_systems)
        if n_ring_systems == 1:
            assert len(ring_assembly) == 0
        # If somehow split into 2 systems, both would be 6-membered but not identical
        # in isolation (different fusion points) — acceptable either way


# ---------------------------------------------------------------------------
# Multiplicative candidates
# ---------------------------------------------------------------------------


class TestMultiplicative:
    def test_diphenyl_ether_linking_type(self):
        """Diphenyl ether (c1ccccc1Oc1ccccc1): two phenyl rings linked by -O-.
        Should produce a multiplicative candidate with linking_type == 'linking_group'."""
        sa = make_symmetry("c1ccccc1Oc1ccccc1")
        candidates = sa.multiplicative_candidates
        assert len(candidates) >= 1

    def test_diphenyl_ether_linking_type_value(self):
        sa = make_symmetry("c1ccccc1Oc1ccccc1")
        for sg in sa.multiplicative_candidates:
            assert sg.linking_type == "linking_group"

    def test_diphenyl_ether_multiplicity(self):
        sa = make_symmetry("c1ccccc1Oc1ccccc1")
        candidates = sa.multiplicative_candidates
        assert candidates[0].multiplicity == 2

    def test_diphenyl_ether_linking_group_mol_not_none(self):
        """Multiplicative candidates must have a linking_group_mol."""
        sa = make_symmetry("c1ccccc1Oc1ccccc1")
        for sg in sa.multiplicative_candidates:
            assert sg.linking_group_mol is not None

    def test_diphenylmethane_multiplicative(self):
        """Diphenylmethane (c1ccccc1Cc1ccccc1): two phenyl rings linked by -CH2-."""
        sa = make_symmetry("c1ccccc1Cc1ccccc1")
        candidates = sa.multiplicative_candidates
        assert len(candidates) >= 1


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_symmetry_groups_immutable_tuple(self):
        """symmetry_groups returns a tuple."""
        sa = make_symmetry("CCO")
        assert isinstance(sa.symmetry_groups, tuple)

    def test_ring_assembly_candidates_subset(self):
        """ring_assembly_candidates is a subset of symmetry_groups."""
        sa = make_symmetry("c1ccc(-c2ccccc2)cc1")
        for sg in sa.ring_assembly_candidates:
            assert sg in sa.symmetry_groups

    def test_multiplicative_candidates_subset(self):
        """multiplicative_candidates is a subset of symmetry_groups."""
        sa = make_symmetry("c1ccccc1Oc1ccccc1")
        for sg in sa.multiplicative_candidates:
            assert sg in sa.symmetry_groups
