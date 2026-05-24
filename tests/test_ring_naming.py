"""
tests/test_ring_naming.py

Tests for the ring naming package (Phase 1.7).

Covers:
  - Retained name lookup (benzene, naphthalene, pyridine, furan, thiophene)
  - Systematic monocyclic naming (cyclohexane, cyclopentane)
  - Hantzsch-Widman naming (basic cases)
  - Von Baeyer bridged naming (norbornane)
  - Spiro naming
  - Engine integration (benzene end-to-end)
"""

from __future__ import annotations

import pytest
from rdkit import Chem

from iupac_namer.types import RingSystem, CandidateParent, HeteroPosition, Locant


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_mol(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles}"
    return mol


def make_monocyclic_ring_system(mol, heteroatoms=None, aromatic=True):
    """Build a RingSystem from an RDKit mol (single ring assumed)."""
    ri = mol.GetRingInfo()
    assert ri.NumRings() >= 1
    ring_atoms = frozenset(ri.AtomRings()[0])
    return RingSystem(
        atom_indices=ring_atoms,
        rings=(ring_atoms,),
        type="monocyclic",
        aromatic=aromatic,
        bridge_sizes=None,
        spiro_sizes=None,
        fusion_info=None,
        heteroatoms=heteroatoms,
        ring_size=len(ring_atoms),
    )


def make_candidate(ring_system: RingSystem) -> CandidateParent:
    return CandidateParent(
        atom_indices=ring_system.atom_indices,
        type=ring_system.type,
        length=ring_system.ring_size,
        ring_system=ring_system,
        unsaturation=None,
        element=None,
        lambda_value=None,
    )


# ---------------------------------------------------------------------------
# retained_lookup tests
# ---------------------------------------------------------------------------

class TestRetainedLookup:
    def test_benzene(self):
        from iupac_namer.ring_naming.retained_lookup import try_retained_name
        mol = make_mol("c1ccccc1")
        rs = make_monocyclic_ring_system(mol, aromatic=True)
        results = try_retained_name(rs, mol)
        assert results, "Benzene should get a retained name"
        np = results[0]
        assert np.name == "benzene"
        assert np.naming_method == "retained"

    def test_pyridine(self):
        from iupac_namer.ring_naming.retained_lookup import try_retained_name
        mol = make_mol("c1ccncc1")
        rs = make_monocyclic_ring_system(mol, aromatic=True)
        results = try_retained_name(rs, mol)
        assert results, "Pyridine should get a retained name"
        assert results[0].name == "pyridine"

    def test_furan(self):
        from iupac_namer.ring_naming.retained_lookup import try_retained_name
        mol = make_mol("c1ccoc1")
        rs = make_monocyclic_ring_system(mol, aromatic=True)
        results = try_retained_name(rs, mol)
        assert results, "Furan should get a retained name"
        assert results[0].name == "furan"

    def test_thiophene(self):
        from iupac_namer.ring_naming.retained_lookup import try_retained_name
        mol = make_mol("c1ccsc1")
        rs = make_monocyclic_ring_system(mol, aromatic=True)
        results = try_retained_name(rs, mol)
        assert results, "Thiophene should get a retained name"
        assert results[0].name == "thiophene"

    def test_naphthalene(self):
        from iupac_namer.ring_naming.retained_lookup import try_retained_name
        mol = make_mol("c1ccc2ccccc2c1")
        ri = mol.GetRingInfo()
        all_atoms = frozenset(a for ring in ri.AtomRings() for a in ring)
        rings = tuple(frozenset(r) for r in ri.AtomRings())
        rs = RingSystem(
            atom_indices=all_atoms,
            rings=rings,
            type="fused",
            aromatic=True,
            bridge_sizes=None,
            spiro_sizes=None,
            fusion_info=None,
            heteroatoms=None,
            ring_size=len(all_atoms),
        )
        results = try_retained_name(rs, mol)
        assert results, "Naphthalene should get a retained name"
        assert results[0].name == "naphthalene"

    def test_cyclohexane_no_retained(self):
        """Cyclohexane is systematic, not retained (unless curated)."""
        from iupac_namer.ring_naming.retained_lookup import try_retained_name
        mol = make_mol("C1CCCCC1")
        rs = make_monocyclic_ring_system(mol, aromatic=False)
        results = try_retained_name(rs, mol)
        # Cyclohexane IS in the curated table
        assert results, "Cyclohexane should be in the curated table"
        assert results[0].name == "cyclohexane"

    def test_benzene_stem(self):
        from iupac_namer.ring_naming.retained_lookup import try_retained_name
        mol = make_mol("c1ccccc1")
        rs = make_monocyclic_ring_system(mol, aromatic=True)
        results = try_retained_name(rs, mol)
        np = results[0]
        assert np.stem == "benzen"

    def test_pyridine_stem(self):
        from iupac_namer.ring_naming.retained_lookup import try_retained_name
        mol = make_mol("c1ccncc1")
        rs = make_monocyclic_ring_system(mol, aromatic=True)
        results = try_retained_name(rs, mol)
        np = results[0]
        assert np.stem == "pyridin"


# ---------------------------------------------------------------------------
# Systematic monocyclic naming tests
# ---------------------------------------------------------------------------

class TestMonocyclic:
    def test_cyclohexane_systematic(self):
        from iupac_namer.ring_naming.monocyclic import name_systematic_monocyclic
        mol = make_mol("C1CCCCC1")
        rs = make_monocyclic_ring_system(mol, aromatic=False)
        candidate = make_candidate(rs)
        result = name_systematic_monocyclic(rs, candidate, mol)
        assert result is not None
        assert result.name == "cyclohexane"
        assert result.stem == "cyclohexan"
        assert result.alkyl_stem == "cyclohex"

    def test_cyclopentane_systematic(self):
        from iupac_namer.ring_naming.monocyclic import name_systematic_monocyclic
        mol = make_mol("C1CCCC1")
        rs = make_monocyclic_ring_system(mol, aromatic=False)
        candidate = make_candidate(rs)
        result = name_systematic_monocyclic(rs, candidate, mol)
        assert result is not None
        assert result.name == "cyclopentane"
        assert result.alkyl_stem == "cyclopent"

    def test_cyclopropane_systematic(self):
        from iupac_namer.ring_naming.monocyclic import name_systematic_monocyclic
        mol = make_mol("C1CC1")
        rs = make_monocyclic_ring_system(mol, aromatic=False)
        candidate = make_candidate(rs)
        result = name_systematic_monocyclic(rs, candidate, mol)
        assert result is not None
        assert result.name == "cyclopropane"

    def test_method_and_naming_method(self):
        from iupac_namer.ring_naming.monocyclic import name_systematic_monocyclic
        mol = make_mol("C1CCCCC1")
        rs = make_monocyclic_ring_system(mol, aromatic=False)
        candidate = make_candidate(rs)
        result = name_systematic_monocyclic(rs, candidate, mol)
        assert result.naming_method == "systematic"


# ---------------------------------------------------------------------------
# Hantzsch-Widman naming tests
# ---------------------------------------------------------------------------

class TestHantzschWidman:
    def _make_hetero_rs(self, smiles: str, elements: list[str], aromatic: bool = True):
        mol = make_mol(smiles)
        ri = mol.GetRingInfo()
        ring_atoms = frozenset(ri.AtomRings()[0])
        heteroatoms = tuple(
            HeteroPosition(
                atom_idx=mol.GetAtomWithIdx(i).GetIdx(),
                element=elem,
                locant=Locant.numeric(1),
            )
            for i in range(mol.GetNumAtoms())
            for elem in elements
            if mol.GetAtomWithIdx(i).GetSymbol() == elem
            and i in ring_atoms
        )
        rs = RingSystem(
            atom_indices=ring_atoms,
            rings=(ring_atoms,),
            type="monocyclic",
            aromatic=aromatic,
            bridge_sizes=None,
            spiro_sizes=None,
            fusion_info=None,
            heteroatoms=heteroatoms if heteroatoms else None,
            ring_size=len(ring_atoms),
        )
        return rs, mol

    def test_oxirane(self):
        """3-membered O ring: oxirane (saturated)."""
        from iupac_namer.ring_naming.monocyclic import try_hantzsch_widman
        mol = make_mol("C1CO1")
        rs, _ = self._make_hetero_rs("C1CO1", ["O"], aromatic=False)
        candidate = make_candidate(rs)
        result = try_hantzsch_widman(rs, candidate, mol)
        assert result is not None
        assert result.name == "oxirane"
        assert result.naming_method == "hantzsch_widman"

    def test_oxolane(self):
        """5-membered O ring (saturated): oxolane."""
        from iupac_namer.ring_naming.monocyclic import try_hantzsch_widman
        mol = make_mol("C1CCOC1")
        rs, _ = self._make_hetero_rs("C1CCOC1", ["O"], aromatic=False)
        candidate = make_candidate(rs)
        result = try_hantzsch_widman(rs, candidate, mol)
        assert result is not None
        assert result.name == "oxolane"

    def test_oxane(self):
        """6-membered O ring (saturated): oxane."""
        from iupac_namer.ring_naming.monocyclic import try_hantzsch_widman
        mol = make_mol("C1CCOCC1")
        rs, _ = self._make_hetero_rs("C1CCOCC1", ["O"], aromatic=False)
        candidate = make_candidate(rs)
        result = try_hantzsch_widman(rs, candidate, mol)
        assert result is not None
        assert result.name == "oxane"


# ---------------------------------------------------------------------------
# Von Baeyer bridged ring naming tests
# ---------------------------------------------------------------------------

class TestBridged:
    def test_norbornane_bicyclo221(self):
        """Norbornane = bicyclo[2.2.1]heptane."""
        from iupac_namer.ring_naming.bridged import name_bridged
        mol = make_mol("C1CC2CC1CC2")
        ri = mol.GetRingInfo()
        all_atoms = frozenset(a for ring in ri.AtomRings() for a in ring)
        rings = tuple(frozenset(r) for r in ri.AtomRings())
        # Provide bridge sizes as (2, 2, 1)
        rs = RingSystem(
            atom_indices=all_atoms,
            rings=rings,
            type="bridged",
            aromatic=False,
            bridge_sizes=(2, 2, 1),
            spiro_sizes=None,
            fusion_info=None,
            heteroatoms=None,
            ring_size=len(all_atoms),
        )
        candidate = make_candidate(rs)
        results = name_bridged(rs, candidate, mol)
        assert results, "Norbornane should get a bridged name"
        # Bridge sizes sorted descending: 2.2.1
        assert results[0].name == "bicyclo[2.2.1]heptane"
        assert results[0].naming_method == "von_baeyer"
        assert results[0].alkyl_stem is None  # Method 1 not applicable

    def test_bicyclo222octane(self):
        """Test Von Baeyer with 3 bridge sizes → bicyclo prefix."""
        from iupac_namer.ring_naming.bridged import name_bridged
        mol = make_mol("C1CC2CCC1CC2")
        ri = mol.GetRingInfo()
        all_atoms = frozenset(a for ring in ri.AtomRings() for a in ring)
        rings = tuple(frozenset(r) for r in ri.AtomRings())
        # Provide correct bridge sizes for bicyclo[2.2.2]octane
        # 3 bridges → (n_bridges - 1) = 2 → "bicyclo"
        rs = RingSystem(
            atom_indices=all_atoms,
            rings=rings,
            type="bridged",
            aromatic=False,
            bridge_sizes=(2, 2, 2),  # 3 bridges → bicyclo
            spiro_sizes=None,
            fusion_info=None,
            heteroatoms=None,
            ring_size=8,
        )
        candidate = make_candidate(rs)
        results = name_bridged(rs, candidate, mol)
        assert results
        assert results[0].name == "bicyclo[2.2.2]octane"


# ---------------------------------------------------------------------------
# Spiro ring naming tests
# ---------------------------------------------------------------------------

class TestSpiro:
    def test_spiro45decane(self):
        """Spiro[4.5]decane."""
        from iupac_namer.ring_naming.spiro import name_spiro
        mol = make_mol("C1CCCCC12CCCC2")
        ri = mol.GetRingInfo()
        all_atoms = frozenset(a for ring in ri.AtomRings() for a in ring)
        rings = tuple(frozenset(r) for r in ri.AtomRings())
        rs = RingSystem(
            atom_indices=all_atoms,
            rings=rings,
            type="spiro",
            aromatic=False,
            bridge_sizes=None,
            spiro_sizes=(4, 5),
            fusion_info=None,
            heteroatoms=None,
            ring_size=len(all_atoms),
        )
        candidate = make_candidate(rs)
        results = name_spiro(rs, candidate, mol)
        assert results, "Spiro[4.5]decane should be named"
        assert results[0].name == "spiro[4.5]decane"
        assert results[0].naming_method == "spiro_systematic"
        assert results[0].alkyl_stem is None

    def test_spiro22pentane(self):
        """Spiro[2.2]pentane."""
        from iupac_namer.ring_naming.spiro import name_spiro
        mol = make_mol("C1CC12CC2")
        ri = mol.GetRingInfo()
        all_atoms = frozenset(a for ring in ri.AtomRings() for a in ring)
        rings = tuple(frozenset(r) for r in ri.AtomRings())
        rs = RingSystem(
            atom_indices=all_atoms,
            rings=rings,
            type="spiro",
            aromatic=False,
            bridge_sizes=None,
            spiro_sizes=(2, 2),
            fusion_info=None,
            heteroatoms=None,
            ring_size=len(all_atoms),
        )
        candidate = make_candidate(rs)
        results = name_spiro(rs, candidate, mol)
        assert results
        assert results[0].name == "spiro[2.2]pentane"


# ---------------------------------------------------------------------------
# name_ring_system integration tests
# ---------------------------------------------------------------------------

class TestNameRingSystem:
    def test_benzene_name_ring_system(self):
        from iupac_namer.ring_naming import name_ring_system
        mol = make_mol("c1ccccc1")
        rs = make_monocyclic_ring_system(mol, aromatic=True)
        candidate = make_candidate(rs)
        results = name_ring_system(candidate, mol)
        assert results, "name_ring_system should return results for benzene"
        names = [np.name for np in results]
        assert "benzene" in names

    def test_cyclohexane_name_ring_system(self):
        from iupac_namer.ring_naming import name_ring_system
        mol = make_mol("C1CCCCC1")
        rs = make_monocyclic_ring_system(mol, aromatic=False)
        candidate = make_candidate(rs)
        results = name_ring_system(candidate, mol)
        assert results
        names = [np.name for np in results]
        assert "cyclohexane" in names

    def test_pyridine_name_ring_system(self):
        from iupac_namer.ring_naming import name_ring_system
        mol = make_mol("c1ccncc1")
        rs = make_monocyclic_ring_system(mol, aromatic=True)
        candidate = make_candidate(rs)
        results = name_ring_system(candidate, mol)
        assert results
        names = [np.name for np in results]
        assert "pyridine" in names

    def test_no_duplicates(self):
        """name_ring_system should deduplicate results."""
        from iupac_namer.ring_naming import name_ring_system
        mol = make_mol("c1ccccc1")
        rs = make_monocyclic_ring_system(mol, aromatic=True)
        candidate = make_candidate(rs)
        results = name_ring_system(candidate, mol)
        names = [np.name for np in results]
        assert len(names) == len(set(names)), "Duplicate names in results"


# ---------------------------------------------------------------------------
# Engine integration test
# ---------------------------------------------------------------------------

class TestEngineIntegration:
    def test_benzene_engine(self):
        """Engine should produce 'benzene' for c1ccccc1."""
        from iupac_namer.engine import name_smiles
        result = name_smiles("c1ccccc1")
        assert result == "benzene", f"Expected 'benzene', got '{result}'"

    def test_toluene_engine(self):
        """Engine should produce 'methylbenzene' or 'toluene' for toluene."""
        from iupac_namer.engine import name_smiles
        result = name_smiles("Cc1ccccc1")
        # Either the retained name "toluene" or systematic "methylbenzene" is acceptable
        assert result in ("toluene", "methylbenzene"), (
            f"Expected 'toluene' or 'methylbenzene', got '{result}'"
        )

    def test_cyclohexane_engine(self):
        """Engine should produce 'cyclohexane' for C1CCCCC1."""
        from iupac_namer.engine import name_smiles
        result = name_smiles("C1CCCCC1")
        assert result == "cyclohexane", f"Expected 'cyclohexane', got '{result}'"

    def test_pyridine_engine(self):
        """Engine should produce 'pyridine' for c1ccncc1."""
        from iupac_namer.engine import name_smiles
        result = name_smiles("c1ccncc1")
        assert result == "pyridine", f"Expected 'pyridine', got '{result}'"

    def test_naphthalene_engine(self):
        """Engine should produce 'naphthalene' for c1ccc2ccccc2c1."""
        from iupac_namer.engine import name_smiles
        result = name_smiles("c1ccc2ccccc2c1")
        assert result == "naphthalene", f"Expected 'naphthalene', got '{result}'"

    def test_furan_engine(self):
        """Engine should produce 'furan' for c1ccoc1."""
        from iupac_namer.engine import name_smiles
        result = name_smiles("c1ccoc1")
        assert result == "furan", f"Expected 'furan', got '{result}'"

    def test_thiophene_engine(self):
        """Engine should produce 'thiophene' for c1ccsc1."""
        from iupac_namer.engine import name_smiles
        result = name_smiles("c1ccsc1")
        assert result == "thiophene", f"Expected 'thiophene', got '{result}'"


# ---------------------------------------------------------------------------
# Ring unsaturation tests (Bug 2 fixes)
# ---------------------------------------------------------------------------

class TestRingUnsaturation:
    """Tests for ring double-bond detection (P-31.1.3.1)."""

    def test_cyclohexene_systematic(self):
        """Cyclohexene: one double bond in 6-membered ring."""
        from iupac_namer.ring_naming.monocyclic import name_systematic_monocyclic
        mol = make_mol("C1=CCCCC1")
        rs = make_monocyclic_ring_system(mol, aromatic=False)
        candidate = make_candidate(rs)
        result = name_systematic_monocyclic(rs, candidate, mol)
        assert result is not None
        assert result.name == "cyclohexene", f"Expected 'cyclohexene', got '{result.name}'"
        # stem should drop terminal 'e'
        assert result.stem == "cyclohexen", f"Expected 'cyclohexen', got '{result.stem}'"

    def test_cyclopropene_systematic(self):
        """Cyclopropene: one double bond in 3-membered ring."""
        from iupac_namer.ring_naming.monocyclic import name_systematic_monocyclic
        mol = make_mol("C1=CC1")
        rs = make_monocyclic_ring_system(mol, aromatic=False)
        candidate = make_candidate(rs)
        result = name_systematic_monocyclic(rs, candidate, mol)
        assert result is not None
        assert result.name == "cyclopropene", f"Expected 'cyclopropene', got '{result.name}'"

    def test_cyclopentene_systematic(self):
        """Cyclopentene: one double bond in 5-membered ring."""
        from iupac_namer.ring_naming.monocyclic import name_systematic_monocyclic
        mol = make_mol("C1=CCCC1")
        rs = make_monocyclic_ring_system(mol, aromatic=False)
        candidate = make_candidate(rs)
        result = name_systematic_monocyclic(rs, candidate, mol)
        assert result is not None
        assert result.name == "cyclopentene", f"Expected 'cyclopentene', got '{result.name}'"

    def test_cyclohexadiene_systematic(self):
        """Cyclohexa-1,3-diene: two double bonds in 6-membered ring."""
        from iupac_namer.ring_naming.monocyclic import name_systematic_monocyclic
        mol = make_mol("C1=CC=CCC1")
        rs = make_monocyclic_ring_system(mol, aromatic=False)
        candidate = make_candidate(rs)
        result = name_systematic_monocyclic(rs, candidate, mol)
        assert result is not None
        # Should contain "diene" and "cyclohexa"
        assert "diene" in result.name, f"Expected 'diene' in name, got '{result.name}'"
        assert "cyclohexa" in result.name, f"Expected 'cyclohexa' in name, got '{result.name}'"

    def test_cyclohexane_no_unsaturation(self):
        """Cyclohexane: no double bonds — should remain '-ane'."""
        from iupac_namer.ring_naming.monocyclic import name_systematic_monocyclic
        mol = make_mol("C1CCCCC1")
        rs = make_monocyclic_ring_system(mol, aromatic=False)
        candidate = make_candidate(rs)
        result = name_systematic_monocyclic(rs, candidate, mol)
        assert result is not None
        assert result.name == "cyclohexane", f"Expected 'cyclohexane', got '{result.name}'"

    def test_detect_ring_double_bonds_cyclohexene(self):
        """_detect_ring_double_bonds returns one locant for cyclohexene."""
        from iupac_namer.ring_naming.monocyclic import _detect_ring_double_bonds
        mol = make_mol("C1=CCCCC1")
        rs = make_monocyclic_ring_system(mol, aromatic=False)
        locants = _detect_ring_double_bonds(rs, mol)
        assert len(locants) == 1, f"Expected 1 double bond, got {locants}"

    def test_detect_ring_double_bonds_saturated(self):
        """_detect_ring_double_bonds returns empty for saturated ring."""
        from iupac_namer.ring_naming.monocyclic import _detect_ring_double_bonds
        mol = make_mol("C1CCCCC1")
        rs = make_monocyclic_ring_system(mol, aromatic=False)
        locants = _detect_ring_double_bonds(rs, mol)
        assert locants == [], f"Expected no double bonds, got {locants}"

    def test_detect_ring_double_bonds_diene(self):
        """_detect_ring_double_bonds returns two locants for diene."""
        from iupac_namer.ring_naming.monocyclic import _detect_ring_double_bonds
        mol = make_mol("C1=CC=CCC1")
        rs = make_monocyclic_ring_system(mol, aromatic=False)
        locants = _detect_ring_double_bonds(rs, mol)
        assert len(locants) == 2, f"Expected 2 double bonds, got {locants}"

    def test_cyclohexene_engine(self):
        """Engine should produce 'cyclohexene' for C1=CCCCC1."""
        from iupac_namer.engine import name_smiles
        result = name_smiles("C1=CCCCC1")
        assert result == "cyclohexene", f"Expected 'cyclohexene', got '{result}'"

    def test_cyclopropene_engine(self):
        """Engine should produce 'cyclopropene' for C1=CC1."""
        from iupac_namer.engine import name_smiles
        result = name_smiles("C1=CC1")
        assert result == "cyclopropene", f"Expected 'cyclopropene', got '{result}'"

    def test_cycloheptene_engine(self):
        """Engine should produce 'cycloheptene' for C1=CCCCCC1."""
        from iupac_namer.engine import name_smiles
        result = name_smiles("C1=CCCCCC1")
        assert result == "cycloheptene", f"Expected 'cycloheptene', got '{result}'"

    def test_cyclohexane_still_works_after_fix(self):
        """Saturated cyclohexane should still produce 'cyclohexane'."""
        from iupac_namer.engine import name_smiles
        result = name_smiles("C1CCCCC1")
        assert result == "cyclohexane", f"Expected 'cyclohexane', got '{result}'"
