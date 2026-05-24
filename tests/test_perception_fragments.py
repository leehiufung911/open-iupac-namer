"""
tests/test_perception_fragments.py

Unit tests for iupac_namer.perception.fragments.FragmentAnalysis.

Each test constructs a molecule from SMILES and exercises FragmentAnalysis
directly, without going through the full Perception facade.
"""

from __future__ import annotations

import pytest
from rdkit import Chem

from iupac_namer.perception.fragments import FragmentAnalysis
from iupac_namer.types import Fragment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def mol_from_smiles(smiles: str) -> object:
    """Return a sanitised RDKit mol.  Fail fast if SMILES is invalid."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles!r}"
    return mol


# ---------------------------------------------------------------------------
# Single-component molecules
# ---------------------------------------------------------------------------


class TestSingleComponent:
    """Molecules with one connected fragment should not be treated as salts."""

    def test_ethanol_is_not_salt(self):
        fa = FragmentAnalysis(mol_from_smiles("CCO"))
        assert not fa.is_salt

    def test_ethanol_fragment_count(self):
        fa = FragmentAnalysis(mol_from_smiles("CCO"))
        assert fa.fragment_count == 1

    def test_ethanol_fragment_tuple_len(self):
        fa = FragmentAnalysis(mol_from_smiles("CCO"))
        assert len(fa.fragments) == 1

    def test_ethanol_fragment_type(self):
        fa = FragmentAnalysis(mol_from_smiles("CCO"))
        frag = fa.fragments[0]
        assert isinstance(frag, Fragment)

    def test_ethanol_atom_indices(self):
        fa = FragmentAnalysis(mol_from_smiles("CCO"))
        frag = fa.fragments[0]
        # 3 heavy atoms -> indices 0, 1, 2
        assert frag.atom_indices == frozenset({0, 1, 2})

    def test_ethanol_charge_zero(self):
        fa = FragmentAnalysis(mol_from_smiles("CCO"))
        assert fa.fragments[0].charge == 0

    def test_benzene_not_salt(self):
        fa = FragmentAnalysis(mol_from_smiles("c1ccccc1"))
        assert not fa.is_salt
        assert fa.fragment_count == 1

    def test_large_molecule_not_salt(self):
        # Glucose
        fa = FragmentAnalysis(mol_from_smiles("OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"))
        assert not fa.is_salt
        assert fa.fragment_count == 1


# ---------------------------------------------------------------------------
# NaCl ([Na+].[Cl-]) — two-fragment salt
# ---------------------------------------------------------------------------


class TestNaClSalt:
    @pytest.fixture
    def fa(self):
        return FragmentAnalysis(mol_from_smiles("[Na+].[Cl-]"))

    def test_is_salt(self, fa):
        assert fa.is_salt

    def test_fragment_count(self, fa):
        assert fa.fragment_count == 2

    def test_total_charge_zero(self, fa):
        total = sum(f.charge for f in fa.fragments)
        assert total == 0

    def test_positive_fragment_charge(self, fa):
        pos_frags = [f for f in fa.fragments if f.charge > 0]
        assert len(pos_frags) == 1
        assert pos_frags[0].charge == 1

    def test_negative_fragment_charge(self, fa):
        neg_frags = [f for f in fa.fragments if f.charge < 0]
        assert len(neg_frags) == 1
        assert neg_frags[0].charge == -1

    def test_atom_indices_non_overlapping(self, fa):
        """Fragment atom-index sets must be disjoint."""
        idx_sets = [f.atom_indices for f in fa.fragments]
        union = frozenset().union(*idx_sets)
        total = sum(len(s) for s in idx_sets)
        # All indices are unique — union size == sum of sizes
        assert len(union) == total

    def test_each_fragment_has_one_atom(self, fa):
        for frag in fa.fragments:
            assert len(frag.atom_indices) == 1

    def test_fragment_mols_are_single_atom(self, fa):
        for frag in fa.fragments:
            assert frag.mol.GetNumAtoms() == 1  # type: ignore[attr-defined]

    def test_fragments_by_charge_positive(self, fa):
        pos = fa.fragments_by_charge(+1)
        assert len(pos) == 1

    def test_fragments_by_charge_negative(self, fa):
        neg = fa.fragments_by_charge(-1)
        assert len(neg) == 1

    def test_fragments_by_charge_zero_empty(self, fa):
        assert fa.fragments_by_charge(0) == ()


# ---------------------------------------------------------------------------
# Sodium acetate ([Na+].CC([O-])=O) — asymmetric salt
# ---------------------------------------------------------------------------


class TestSodiumAcetate:
    @pytest.fixture
    def fa(self):
        return FragmentAnalysis(mol_from_smiles("[Na+].CC([O-])=O"))

    def test_is_salt(self, fa):
        assert fa.is_salt

    def test_fragment_count(self, fa):
        assert fa.fragment_count == 2

    def test_charges_correct(self, fa):
        charges = sorted(f.charge for f in fa.fragments)
        assert charges == [-1, 1]

    def test_largest_fragment_is_acetate(self, fa):
        large = fa.largest_fragment()
        # Acetate has 4 heavy atoms (CC([O-])=O); Na+ has 1
        assert len(large.atom_indices) == 4

    def test_sodium_fragment_has_one_atom(self, fa):
        small_frags = [f for f in fa.fragments if len(f.atom_indices) == 1]
        assert len(small_frags) == 1
        small = small_frags[0]
        # Na+ has charge +1
        assert small.charge == 1

    def test_total_charge_zero(self, fa):
        assert sum(f.charge for f in fa.fragments) == 0


# ---------------------------------------------------------------------------
# Three-component mixture
# ---------------------------------------------------------------------------


class TestThreeComponents:
    """C.C.C — propane as three methane fragments ... actually 3 separate CH4."""

    @pytest.fixture
    def fa(self):
        return FragmentAnalysis(mol_from_smiles("C.C.C"))

    def test_fragment_count(self, fa):
        assert fa.fragment_count == 3

    def test_is_salt(self, fa):
        assert fa.is_salt  # Any multi-fragment mol is treated as salt-like

    def test_all_charges_zero(self, fa):
        assert all(f.charge == 0 for f in fa.fragments)

    def test_all_fragments_have_one_atom(self, fa):
        for frag in fa.fragments:
            assert len(frag.atom_indices) == 1

    def test_atom_indices_partition(self, fa):
        """Atom indices partition {0, 1, 2} with no overlap."""
        all_idxs = frozenset().union(*(f.atom_indices for f in fa.fragments))
        assert all_idxs == frozenset({0, 1, 2})


# ---------------------------------------------------------------------------
# Fragment.mol sub-mols round-trip
# ---------------------------------------------------------------------------


class TestFragmentMolValidity:
    """Verify that fragment sub-mols are valid RDKit mol objects."""

    def test_ethanol_mol_is_valid(self):
        fa = FragmentAnalysis(mol_from_smiles("CCO"))
        frag = fa.fragments[0]
        assert frag.mol is not None
        assert frag.mol.GetNumAtoms() == 3  # type: ignore[attr-defined]

    def test_salt_fragment_mols_valid(self):
        fa = FragmentAnalysis(mol_from_smiles("[Na+].[Cl-]"))
        for frag in fa.fragments:
            assert frag.mol is not None
            assert frag.mol.GetNumAtoms() >= 1  # type: ignore[attr-defined]

    def test_fragment_atom_indices_is_frozenset(self):
        fa = FragmentAnalysis(mol_from_smiles("[Na+].[Cl-]"))
        for frag in fa.fragments:
            assert isinstance(frag.atom_indices, frozenset)

    def test_fragment_is_frozen(self):
        import dataclasses
        fa = FragmentAnalysis(mol_from_smiles("CCO"))
        frag = fa.fragments[0]
        with pytest.raises(dataclasses.FrozenInstanceError):
            frag.charge = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# largest_fragment
# ---------------------------------------------------------------------------


class TestLargestFragment:
    def test_single_component_is_largest(self):
        fa = FragmentAnalysis(mol_from_smiles("CCCCCC"))
        large = fa.largest_fragment()
        assert len(large.atom_indices) == 6

    def test_largest_in_salt(self):
        # Sodium butanoate: [Na+].CCCC([O-])=O
        # Butanoate = 4C + 2O = 6 heavy atoms; Na+ has 1
        fa = FragmentAnalysis(mol_from_smiles("[Na+].CCCC([O-])=O"))
        large = fa.largest_fragment()
        assert len(large.atom_indices) == 6


# ---------------------------------------------------------------------------
# Perception facade integration
# ---------------------------------------------------------------------------


class TestPerceptionFacade:
    """Smoke test: FragmentAnalysis accessible via Perception.fragments."""

    def test_perception_fragments_property(self):
        from iupac_namer.perception import Perception

        mol = mol_from_smiles("CCO")
        p = Perception(mol)
        fa = p.fragments
        assert isinstance(fa, FragmentAnalysis)
        assert fa.fragment_count == 1

    def test_perception_fragments_cached(self):
        """Second access returns the same object (lazy init)."""
        from iupac_namer.perception import Perception

        mol = mol_from_smiles("CCO")
        p = Perception(mol)
        assert p.fragments is p.fragments

    def test_perception_salt_via_facade(self):
        from iupac_namer.perception import Perception

        mol = mol_from_smiles("[Na+].[Cl-]")
        p = Perception(mol)
        assert p.fragments.is_salt
        assert p.fragments.fragment_count == 2
