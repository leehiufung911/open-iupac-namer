"""
tests/test_perception_rings.py

Unit tests for iupac_namer.perception.rings.RingAnalysis.

Each test constructs a molecule from SMILES and exercises RingAnalysis
directly via the Perception facade (which lazily initialises it).

Tested compounds:
 1. Cyclohexane   — monocyclic, ring_size=6, no heteroatoms, not aromatic
 2. Benzene       — monocyclic, ring_size=6, aromatic
 3. Pyridine      — monocyclic, 1 heteroatom (N), aromatic
 4. Naphthalene   — fused, 2 rings, ring_size=10, aromatic
 5. Norbornane    — bridged, bridge_sizes=(2,2,1)
 6. Spiro[4.5]decane — spiro, spiro_sizes=(4,5)
 7. Indole        — fused, heterocyclic
 8. Ethanol       — no rings at all
 9. Toluene       — one monocyclic ring system
10. Biphenyl      — 2 separate monocyclic ring systems (not fused)
"""

from __future__ import annotations

import pytest
from rdkit import Chem

from iupac_namer.perception import Perception
from iupac_namer.perception.atoms import AtomAnalysis
from iupac_namer.perception.rings import RingAnalysis
from iupac_namer.types import RingSystem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def mol_from_smiles(smiles: str) -> object:
    """Return a sanitised RDKit mol.  Fail fast if SMILES is invalid."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles!r}"
    return mol


def make_ring_analysis(smiles: str) -> RingAnalysis:
    """Convenience: build RingAnalysis for a SMILES string."""
    mol = mol_from_smiles(smiles)
    aa = AtomAnalysis(mol)
    return RingAnalysis(mol, aa)


def make_perception(smiles: str) -> Perception:
    """Convenience: build Perception facade for a SMILES string."""
    mol = mol_from_smiles(smiles)
    return Perception(mol)


# ---------------------------------------------------------------------------
# 1. Cyclohexane — monocyclic, non-aromatic, no heteroatoms
# ---------------------------------------------------------------------------


class TestCyclohexane:
    SMILES = "C1CCCCC1"

    def test_has_rings(self):
        ra = make_ring_analysis(self.SMILES)
        assert ra.has_rings is True

    def test_one_ring_system(self):
        ra = make_ring_analysis(self.SMILES)
        assert len(ra.ring_systems) == 1

    def test_type_monocyclic(self):
        ra = make_ring_analysis(self.SMILES)
        rs = ra.ring_systems[0]
        assert rs.type == "monocyclic"

    def test_ring_size(self):
        ra = make_ring_analysis(self.SMILES)
        rs = ra.ring_systems[0]
        assert rs.ring_size == 6

    def test_not_aromatic(self):
        ra = make_ring_analysis(self.SMILES)
        rs = ra.ring_systems[0]
        assert rs.aromatic is False

    def test_no_heteroatoms(self):
        ra = make_ring_analysis(self.SMILES)
        rs = ra.ring_systems[0]
        # Either None or empty tuple
        assert not rs.heteroatoms

    def test_no_bridge_or_spiro(self):
        ra = make_ring_analysis(self.SMILES)
        rs = ra.ring_systems[0]
        assert rs.bridge_sizes is None
        assert rs.spiro_sizes is None

    def test_six_ring_atoms(self):
        ra = make_ring_analysis(self.SMILES)
        rs = ra.ring_systems[0]
        assert len(rs.atom_indices) == 6

    def test_ring_system_for_atom(self):
        ra = make_ring_analysis(self.SMILES)
        # All atoms 0-5 are in the ring
        for i in range(6):
            rs = ra.ring_system_for_atom(i)
            assert rs is not None
            assert rs.type == "monocyclic"

    def test_all_ring_atoms(self):
        ra = make_ring_analysis(self.SMILES)
        assert len(ra.all_ring_atoms()) == 6


# ---------------------------------------------------------------------------
# 2. Benzene — monocyclic, aromatic
# ---------------------------------------------------------------------------


class TestBenzene:
    SMILES = "c1ccccc1"

    def test_type_monocyclic(self):
        ra = make_ring_analysis(self.SMILES)
        rs = ra.ring_systems[0]
        assert rs.type == "monocyclic"

    def test_aromatic(self):
        ra = make_ring_analysis(self.SMILES)
        rs = ra.ring_systems[0]
        assert rs.aromatic is True

    def test_ring_size_6(self):
        ra = make_ring_analysis(self.SMILES)
        rs = ra.ring_systems[0]
        assert rs.ring_size == 6

    def test_no_heteroatoms(self):
        ra = make_ring_analysis(self.SMILES)
        rs = ra.ring_systems[0]
        assert not rs.heteroatoms

    def test_via_perception_facade(self):
        p = make_perception(self.SMILES)
        assert p.rings.has_rings is True
        assert len(p.rings.ring_systems) == 1
        assert p.rings.ring_systems[0].type == "monocyclic"


# ---------------------------------------------------------------------------
# 3. Pyridine — monocyclic, 1 N heteroatom
# ---------------------------------------------------------------------------


class TestPyridine:
    SMILES = "c1ccncc1"

    def test_type_monocyclic(self):
        ra = make_ring_analysis(self.SMILES)
        assert ra.ring_systems[0].type == "monocyclic"

    def test_aromatic(self):
        ra = make_ring_analysis(self.SMILES)
        assert ra.ring_systems[0].aromatic is True

    def test_ring_size_6(self):
        ra = make_ring_analysis(self.SMILES)
        assert ra.ring_systems[0].ring_size == 6

    def test_one_heteroatom(self):
        ra = make_ring_analysis(self.SMILES)
        rs = ra.ring_systems[0]
        assert rs.heteroatoms is not None
        assert len(rs.heteroatoms) == 1
        assert rs.heteroatoms[0].element == "N"

    def test_heteroatom_position_in_range(self):
        ra = make_ring_analysis(self.SMILES)
        rs = ra.ring_systems[0]
        # position_in_ring is no longer a field; locant is None at this stage
        hp = rs.heteroatoms[0]
        assert hp.element == "N"
        assert hp.atom_idx >= 0


# ---------------------------------------------------------------------------
# 4. Naphthalene — fused, 2 rings, ring_size=10
# ---------------------------------------------------------------------------


class TestNaphthalene:
    SMILES = "c1ccc2ccccc2c1"

    def test_type_fused(self):
        ra = make_ring_analysis(self.SMILES)
        rs = ra.ring_systems[0]
        assert rs.type == "fused"

    def test_two_sssr_rings(self):
        ra = make_ring_analysis(self.SMILES)
        rs = ra.ring_systems[0]
        assert len(rs.rings) == 2

    def test_ring_size_10(self):
        ra = make_ring_analysis(self.SMILES)
        rs = ra.ring_systems[0]
        assert rs.ring_size == 10

    def test_aromatic(self):
        ra = make_ring_analysis(self.SMILES)
        rs = ra.ring_systems[0]
        assert rs.aromatic is True

    def test_fusion_info_present(self):
        ra = make_ring_analysis(self.SMILES)
        rs = ra.ring_systems[0]
        assert rs.fusion_info is not None

    def test_fusion_info_one_shared_edge(self):
        ra = make_ring_analysis(self.SMILES)
        rs = ra.ring_systems[0]
        # Naphthalene has one shared bond between its two rings
        assert len(rs.fusion_info.fusion_atoms) == 1

    def test_no_bridge_or_spiro(self):
        ra = make_ring_analysis(self.SMILES)
        rs = ra.ring_systems[0]
        assert rs.bridge_sizes is None
        assert rs.spiro_sizes is None

    def test_no_heteroatoms(self):
        ra = make_ring_analysis(self.SMILES)
        rs = ra.ring_systems[0]
        assert not rs.heteroatoms

    def test_one_ring_system(self):
        ra = make_ring_analysis(self.SMILES)
        assert len(ra.ring_systems) == 1


# ---------------------------------------------------------------------------
# 5. Norbornane — bridged, bridge_sizes=(2,2,1)
# ---------------------------------------------------------------------------


class TestNorbornane:
    # bicyclo[2.2.1]heptane
    SMILES = "C1CC2CC1CC2"

    def test_type_bridged(self):
        ra = make_ring_analysis(self.SMILES)
        rs = ra.ring_systems[0]
        assert rs.type == "bridged"

    def test_ring_size_7(self):
        ra = make_ring_analysis(self.SMILES)
        rs = ra.ring_systems[0]
        assert rs.ring_size == 7

    def test_bridge_sizes_not_none(self):
        ra = make_ring_analysis(self.SMILES)
        rs = ra.ring_systems[0]
        assert rs.bridge_sizes is not None

    def test_bridge_sizes_correct(self):
        ra = make_ring_analysis(self.SMILES)
        rs = ra.ring_systems[0]
        # Should be (2, 2, 1) for norbornane
        assert sorted(rs.bridge_sizes, reverse=True) == [2, 2, 1]

    def test_not_aromatic(self):
        ra = make_ring_analysis(self.SMILES)
        rs = ra.ring_systems[0]
        assert rs.aromatic is False

    def test_no_spiro(self):
        ra = make_ring_analysis(self.SMILES)
        rs = ra.ring_systems[0]
        assert rs.spiro_sizes is None

    def test_no_heteroatoms(self):
        ra = make_ring_analysis(self.SMILES)
        rs = ra.ring_systems[0]
        assert not rs.heteroatoms


# ---------------------------------------------------------------------------
# 6. Spiro[4.5]decane — spiro, spiro_sizes=(4,5)
# ---------------------------------------------------------------------------


class TestSpiroDecane:
    # spiro[4.5]decane: cyclopentane + cyclohexane sharing one atom
    SMILES = "C1CCCC11CCCCC1"

    def test_type_spiro(self):
        ra = make_ring_analysis(self.SMILES)
        rs = ra.ring_systems[0]
        assert rs.type == "spiro"

    def test_ring_size_10(self):
        ra = make_ring_analysis(self.SMILES)
        rs = ra.ring_systems[0]
        assert rs.ring_size == 10

    def test_spiro_sizes_not_none(self):
        ra = make_ring_analysis(self.SMILES)
        rs = ra.ring_systems[0]
        assert rs.spiro_sizes is not None

    def test_spiro_sizes_correct(self):
        ra = make_ring_analysis(self.SMILES)
        rs = ra.ring_systems[0]
        # Should be (4, 5) for spiro[4.5]decane (sizes minus 1, sorted ascending)
        assert sorted(rs.spiro_sizes) == [4, 5]

    def test_not_aromatic(self):
        ra = make_ring_analysis(self.SMILES)
        rs = ra.ring_systems[0]
        assert rs.aromatic is False

    def test_no_bridge(self):
        ra = make_ring_analysis(self.SMILES)
        rs = ra.ring_systems[0]
        assert rs.bridge_sizes is None


# ---------------------------------------------------------------------------
# 7. Indole — fused, heterocyclic (N)
# ---------------------------------------------------------------------------


class TestIndole:
    SMILES = "c1ccc2[nH]ccc2c1"

    def test_type_fused(self):
        ra = make_ring_analysis(self.SMILES)
        rs = ra.ring_systems[0]
        assert rs.type == "fused"

    def test_ring_size_9(self):
        ra = make_ring_analysis(self.SMILES)
        rs = ra.ring_systems[0]
        assert rs.ring_size == 9

    def test_has_nitrogen_heteroatom(self):
        ra = make_ring_analysis(self.SMILES)
        rs = ra.ring_systems[0]
        assert rs.heteroatoms is not None
        elements = {h.element for h in rs.heteroatoms}
        assert "N" in elements

    def test_aromatic(self):
        ra = make_ring_analysis(self.SMILES)
        rs = ra.ring_systems[0]
        assert rs.aromatic is True

    def test_two_rings(self):
        ra = make_ring_analysis(self.SMILES)
        rs = ra.ring_systems[0]
        assert len(rs.rings) == 2

    def test_fusion_info_present(self):
        ra = make_ring_analysis(self.SMILES)
        rs = ra.ring_systems[0]
        assert rs.fusion_info is not None


# ---------------------------------------------------------------------------
# 8. Ethanol — no rings
# ---------------------------------------------------------------------------


class TestEthanol:
    SMILES = "CCO"

    def test_has_rings_false(self):
        ra = make_ring_analysis(self.SMILES)
        assert ra.has_rings is False

    def test_no_ring_systems(self):
        ra = make_ring_analysis(self.SMILES)
        assert len(ra.ring_systems) == 0

    def test_ring_system_for_atom_none(self):
        ra = make_ring_analysis(self.SMILES)
        for i in range(3):
            assert ra.ring_system_for_atom(i) is None

    def test_all_ring_atoms_empty(self):
        ra = make_ring_analysis(self.SMILES)
        assert len(ra.all_ring_atoms()) == 0

    def test_via_perception_facade(self):
        p = make_perception(self.SMILES)
        assert p.rings.has_rings is False


# ---------------------------------------------------------------------------
# 9. Toluene — one ring system (monocyclic)
# ---------------------------------------------------------------------------


class TestToluene:
    SMILES = "Cc1ccccc1"

    def test_one_ring_system(self):
        ra = make_ring_analysis(self.SMILES)
        assert len(ra.ring_systems) == 1

    def test_monocyclic(self):
        ra = make_ring_analysis(self.SMILES)
        assert ra.ring_systems[0].type == "monocyclic"

    def test_ring_size_6(self):
        ra = make_ring_analysis(self.SMILES)
        assert ra.ring_systems[0].ring_size == 6

    def test_methyl_not_in_ring(self):
        ra = make_ring_analysis(self.SMILES)
        # The methyl carbon (attached externally) should NOT be in the ring atoms
        # Toluene has 7 heavy atoms; the benzene ring accounts for 6
        assert len(ra.all_ring_atoms()) == 6

    def test_ring_system_for_methyl_carbon(self):
        ra = make_ring_analysis(self.SMILES)
        mol = mol_from_smiles(self.SMILES)
        # Find the methyl carbon (not in ring)
        non_ring_carbons = [
            a.GetIdx() for a in mol.GetAtoms()
            if not a.IsInRing() and a.GetSymbol() == "C"
        ]
        assert len(non_ring_carbons) == 1
        methyl_idx = non_ring_carbons[0]
        assert ra.ring_system_for_atom(methyl_idx) is None


# ---------------------------------------------------------------------------
# 10. Biphenyl — 2 separate monocyclic ring systems
# ---------------------------------------------------------------------------


class TestBiphenyl:
    SMILES = "c1ccc(-c2ccccc2)cc1"

    def test_two_ring_systems(self):
        ra = make_ring_analysis(self.SMILES)
        assert len(ra.ring_systems) == 2

    def test_both_monocyclic(self):
        ra = make_ring_analysis(self.SMILES)
        for rs in ra.ring_systems:
            assert rs.type == "monocyclic"

    def test_both_size_6(self):
        ra = make_ring_analysis(self.SMILES)
        for rs in ra.ring_systems:
            assert rs.ring_size == 6

    def test_both_aromatic(self):
        ra = make_ring_analysis(self.SMILES)
        for rs in ra.ring_systems:
            assert rs.aromatic is True

    def test_disjoint_ring_atoms(self):
        ra = make_ring_analysis(self.SMILES)
        rs0 = ra.ring_systems[0]
        rs1 = ra.ring_systems[1]
        # The two ring systems should have no atoms in common
        assert rs0.atom_indices.isdisjoint(rs1.atom_indices)

    def test_all_ring_atoms_12(self):
        ra = make_ring_analysis(self.SMILES)
        assert len(ra.all_ring_atoms()) == 12


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------


class TestRingSystemFrozen:
    """RingSystem is a frozen dataclass — verify immutability."""

    def test_ring_system_is_frozen(self):
        ra = make_ring_analysis("c1ccccc1")
        rs = ra.ring_systems[0]
        with pytest.raises((AttributeError, TypeError)):
            rs.type = "fused"  # type: ignore[misc]


class TestDetectRingUnsaturationStub:
    """detect_ring_unsaturation is a stub that returns empty tuple."""

    def test_stub_returns_empty(self):
        ra = make_ring_analysis("c1ccccc1")
        rs = ra.ring_systems[0]
        result = ra.detect_ring_unsaturation(rs, numbering=None)
        assert result == ()


class TestPerceptionFacadeRings:
    """Verify the Perception facade lazy-initialises RingAnalysis correctly."""

    def test_perception_rings_property(self):
        p = make_perception("c1ccc2ccccc2c1")  # naphthalene
        ra = p.rings
        assert isinstance(ra, RingAnalysis)

    def test_perception_rings_cached(self):
        p = make_perception("c1ccc2ccccc2c1")
        ra1 = p.rings
        ra2 = p.rings
        assert ra1 is ra2  # same object (cached)

    def test_perception_rings_atom_dependency(self):
        """Accessing rings should also trigger atoms construction."""
        p = make_perception("c1ccccc1")
        _ = p.rings  # triggers atoms lazily
        assert p._atoms is not None
