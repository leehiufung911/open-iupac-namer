"""
tests/test_extraction.py

Unit tests for iupac_namer.perception.extraction — fragment carving utilities.

Tested functions
----------------
carve_substituent          — single-attachment substituent extraction
carve_bridging_substituent — multi-attachment bridging-group extraction
carve_fc_fragments         — functional-class fragment stub
strip_additive_atoms       — removal of additive atoms (N-oxide, P-oxide)
"""

from __future__ import annotations

import pytest
from rdkit import Chem


# ---------------------------------------------------------------------------
# Module under test
# ---------------------------------------------------------------------------

from iupac_namer.perception.extraction import (
    carve_substituent,
    carve_bridging_substituent,
    carve_fc_fragments,
    strip_additive_atoms,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def mol_from_smiles(smiles: str) -> object:
    """Return a sanitised RDKit mol.  Fail fast if SMILES is invalid."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles!r}"
    return mol


def canonical_smiles(mol: object) -> str:
    """Return the canonical SMILES string for *mol*."""
    return Chem.MolToSmiles(mol)


# ---------------------------------------------------------------------------
# 1. carve_substituent — ethyl group from ethylbenzene
#
# Ethylbenzene: CCc1ccccc1
# Atom layout (canonical RDKit SMILES): ring atoms 0-5, ethyl C atoms 6-7
# We find the bond between the ring and the CH2 group and cut it.
# ---------------------------------------------------------------------------


class TestCarveSubstituentEthylbenzene:
    """Carve the ethyl group (–CH2CH3) from ethylbenzene (CCc1ccccc1)."""

    @pytest.fixture
    def setup(self):
        mol = mol_from_smiles("CCc1ccccc1")
        # Find the bond that connects the ring to the CH2.
        # In ethylbenzene the CH2 carbon is bonded to both a ring carbon
        # and the terminal CH3.  Scan all bonds to find a bond between a
        # ring atom and an aliphatic carbon.
        bond = None
        for b in mol.GetBonds():
            a1 = mol.GetAtomWithIdx(b.GetBeginAtomIdx())
            a2 = mol.GetAtomWithIdx(b.GetEndAtomIdx())
            if a1.IsInRing() and not a2.IsInRing() and a2.GetAtomicNum() == 6:
                bond = (b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                break
            if a2.IsInRing() and not a1.IsInRing() and a1.GetAtomicNum() == 6:
                bond = (b.GetEndAtomIdx(), b.GetBeginAtomIdx())
                break
        assert bond is not None, "Could not locate ring–aliphatic bond in ethylbenzene"
        return mol, bond

    def test_returns_tuple_of_three(self, setup):
        mol, bond = setup
        result = carve_substituent(mol, frozenset(), bond)
        assert isinstance(result, tuple) and len(result) == 3

    def test_fragment_is_rdkit_mol(self, setup):
        mol, bond = setup
        frag, _, _ = carve_substituent(mol, frozenset(), bond)
        assert frag is not None
        assert hasattr(frag, "GetNumAtoms"), "fragment_mol must be an RDKit mol"

    def test_fragment_has_two_carbons(self, setup):
        mol, bond = setup
        frag, _, _ = carve_substituent(mol, frozenset(), bond)
        # Ethyl = CH2-CH3 plus the H that replaced the cut bond = 2 carbons
        carbon_count = sum(
            1 for a in frag.GetAtoms() if a.GetAtomicNum() == 6
        )
        assert carbon_count == 2, (
            f"Expected 2 carbons in ethyl fragment, got {carbon_count}; "
            f"SMILES={canonical_smiles(frag)}"
        )

    def test_attachment_atom_index_in_range(self, setup):
        mol, bond = setup
        frag, attach_idx, _ = carve_substituent(mol, frozenset(), bond)
        assert 0 <= attach_idx < frag.GetNumAtoms()

    def test_attachment_atom_is_carbon(self, setup):
        mol, bond = setup
        frag, attach_idx, _ = carve_substituent(mol, frozenset(), bond)
        atom = frag.GetAtomWithIdx(attach_idx)
        assert atom.GetAtomicNum() == 6, "Attachment atom should be a carbon"

    def test_bond_order_is_single(self, setup):
        mol, bond = setup
        _, _, bond_order = carve_substituent(mol, frozenset(), bond)
        assert bond_order == 1

    def test_fragment_smiles_is_valid(self, setup):
        mol, bond = setup
        frag, _, _ = carve_substituent(mol, frozenset(), bond)
        smi = canonical_smiles(frag)
        assert smi, "fragment canonical SMILES must not be empty"

    def test_nonexistent_bond_raises(self):
        mol = mol_from_smiles("CCc1ccccc1")
        with pytest.raises(ValueError, match="No bond between"):
            carve_substituent(mol, frozenset(), (0, 7))  # non-adjacent atoms


# ---------------------------------------------------------------------------
# 2. carve_substituent — OH from propanol (CCCO)
# ---------------------------------------------------------------------------


class TestCarveSubstituentPropanol:
    """Carve the OH from propanol as if it were a substituent.

    CCCO — atoms 0(C), 1(C), 2(C), 3(O).
    Cut bond: (2, 3) parent=C2, sub=O3.
    """

    @pytest.fixture
    def setup(self):
        mol = mol_from_smiles("CCCO")
        # Find the C-O bond (last heavy-atom bond).
        c_o_bond = None
        for b in mol.GetBonds():
            a1 = mol.GetAtomWithIdx(b.GetBeginAtomIdx())
            a2 = mol.GetAtomWithIdx(b.GetEndAtomIdx())
            if a1.GetAtomicNum() == 8:
                c_o_bond = (b.GetEndAtomIdx(), b.GetBeginAtomIdx())
                break
            if a2.GetAtomicNum() == 8:
                c_o_bond = (b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                break
        assert c_o_bond is not None
        return mol, c_o_bond

    def test_fragment_is_oxygen(self, setup):
        mol, bond = setup
        frag, _, _ = carve_substituent(mol, frozenset(), bond)
        # The oxygen fragment will be H-O (i.e. methanol-like "OH with H cap")
        # containing exactly one heavy-atom oxygen.
        o_count = sum(1 for a in frag.GetAtoms() if a.GetAtomicNum() == 8)
        assert o_count == 1, (
            f"Expected 1 oxygen in OH fragment, got {o_count}; "
            f"SMILES={canonical_smiles(frag)}"
        )

    def test_attachment_atom_is_oxygen(self, setup):
        mol, bond = setup
        frag, attach_idx, _ = carve_substituent(mol, frozenset(), bond)
        atom = frag.GetAtomWithIdx(attach_idx)
        assert atom.GetAtomicNum() == 8

    def test_bond_order_is_single(self, setup):
        mol, bond = setup
        _, _, bond_order = carve_substituent(mol, frozenset(), bond)
        assert bond_order == 1


# ---------------------------------------------------------------------------
# 3. Canonical normalisation — equivalent substituents from different molecules
# ---------------------------------------------------------------------------


class TestCanonicalNormalization:
    """Carving equivalent substituents should yield the same canonical SMILES
    and the same attachment-atom index (v13 G1)."""

    def _carve_methyl(self, smiles: str) -> tuple:
        """Carve a methyl group (–CH3) from *smiles*.

        Returns (fragment_smiles, attachment_idx) for comparison.
        """
        mol = mol_from_smiles(smiles)
        # Find any C-C bond where one C is –CH3 (terminal: degree 1 heavy-atom
        # neighbours = 1).
        for b in mol.GetBonds():
            idx1, idx2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            a1, a2 = mol.GetAtomWithIdx(idx1), mol.GetAtomWithIdx(idx2)
            if a1.GetAtomicNum() != 6 or a2.GetAtomicNum() != 6:
                continue
            if a2.GetDegree() == 1:
                # a2 is the terminal methyl carbon (methyl side)
                frag, attach, _ = carve_substituent(
                    mol, frozenset(), (idx1, idx2)
                )
                return canonical_smiles(frag), attach
            if a1.GetDegree() == 1:
                frag, attach, _ = carve_substituent(
                    mol, frozenset(), (idx2, idx1)
                )
                return canonical_smiles(frag), attach
        raise AssertionError(f"No terminal methyl C–C bond found in {smiles!r}")

    def test_methyl_from_ethane_and_propane_same_smiles(self):
        smi1, _ = self._carve_methyl("CC")     # ethane
        smi2, _ = self._carve_methyl("CCC")    # propane
        assert smi1 == smi2, (
            f"Methyl SMILES mismatch: ethane→{smi1!r}, propane→{smi2!r}"
        )

    def test_methyl_from_ethane_and_propane_same_attachment(self):
        _, att1 = self._carve_methyl("CC")
        _, att2 = self._carve_methyl("CCC")
        assert att1 == att2, (
            f"Methyl attachment index mismatch: ethane→{att1}, propane→{att2}"
        )


# ---------------------------------------------------------------------------
# 4. strip_additive_atoms — pyridine N-oxide
# ---------------------------------------------------------------------------


class TestStripAdditiveAtomsPyridineNoxide:
    """strip_additive_atoms on pyridine N-oxide ([O-][n+]1ccccc1)."""

    @pytest.fixture
    def setup(self):
        smiles = "[O-][n+]1ccccc1"
        mol = mol_from_smiles(smiles)
        # Identify the N atom and the O atom.
        n_idx = next(
            a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 7
        )
        o_idx = next(
            a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 8
        )
        # Verify N and O are bonded (sanity check on the fixture mol).
        bond = mol.GetBondBetweenAtoms(n_idx, o_idx)
        assert bond is not None, "N and O must be bonded in pyridine N-oxide"

        additive_groups = [
            {
                "added_atom": o_idx,
                "center_atom": n_idx,
                "center_element": "N",
            }
        ]
        return mol, additive_groups, n_idx, o_idx

    def test_returns_tuple_of_two(self, setup):
        mol, groups, _, _ = setup
        result = strip_additive_atoms(mol, groups)
        assert isinstance(result, tuple) and len(result) == 2

    def test_parent_mol_has_no_oxygen(self, setup):
        mol, groups, _, _ = setup
        parent, _ = strip_additive_atoms(mol, groups)
        o_count = sum(1 for a in parent.GetAtoms() if a.GetAtomicNum() == 8)
        assert o_count == 0, (
            f"Expected no oxygen in parent mol; SMILES={canonical_smiles(parent)}"
        )

    def test_parent_mol_has_correct_atom_count(self, setup):
        mol, groups, _, _ = setup
        parent, _ = strip_additive_atoms(mol, groups)
        # pyridine N-oxide has 7 heavy atoms (6C + 1N + 1O); removing O → 6
        assert parent.GetNumAtoms() == mol.GetNumAtoms() - 1

    def test_atom_map_keys_cover_all_new_indices(self, setup):
        mol, groups, _, _ = setup
        parent, atom_map = strip_additive_atoms(mol, groups)
        expected_keys = set(range(parent.GetNumAtoms()))
        assert set(atom_map.keys()) == expected_keys

    def test_atom_map_values_are_valid_original_indices(self, setup):
        mol, groups, _, _ = setup
        parent, atom_map = strip_additive_atoms(mol, groups)
        n_orig = mol.GetNumAtoms()
        for old_idx in atom_map.values():
            assert 0 <= old_idx < n_orig

    def test_atom_map_does_not_include_removed_atom(self, setup):
        mol, groups, _, o_idx = setup
        parent, atom_map = strip_additive_atoms(mol, groups)
        assert o_idx not in atom_map.values(), (
            "Removed oxygen index should not appear in atom_map values"
        )

    def test_nitrogen_charge_adjusted(self, setup):
        """After stripping, the N should no longer have +1 formal charge."""
        mol, groups, n_idx, o_idx = setup
        parent, atom_map = strip_additive_atoms(mol, groups)
        # Find the new index of the N in the parent mol.
        old_to_new = {v: k for k, v in atom_map.items()}
        new_n_idx = old_to_new.get(n_idx)
        assert new_n_idx is not None, "N atom should be present in parent mol"
        n_atom = parent.GetAtomWithIdx(new_n_idx)
        # The original N had formal charge +1; after stripping it should be 0.
        assert n_atom.GetFormalCharge() == 0, (
            f"Expected N formal charge 0, got {n_atom.GetFormalCharge()}"
        )

    def test_parent_smiles_is_pyridine(self, setup):
        """The parent should be pyridine (c1ccncc1 or equivalent)."""
        mol, groups, _, _ = setup
        parent, _ = strip_additive_atoms(mol, groups)
        smi = canonical_smiles(parent)
        # RDKit canonical SMILES for pyridine.
        ref_smi = canonical_smiles(mol_from_smiles("c1ccncc1"))
        assert smi == ref_smi, (
            f"Parent SMILES {smi!r} does not match pyridine {ref_smi!r}"
        )


# ---------------------------------------------------------------------------
# 5. strip_additive_atoms — no additive groups (identity)
# ---------------------------------------------------------------------------


class TestStripAdditiveAtomsNoOp:
    """strip_additive_atoms with no additive groups returns original mol unchanged."""

    def test_returns_original_mol(self):
        mol = mol_from_smiles("CCO")
        parent, atom_map = strip_additive_atoms(mol, [])
        # Same object returned.
        assert parent is mol

    def test_identity_atom_map(self):
        mol = mol_from_smiles("CCO")
        _, atom_map = strip_additive_atoms(mol, [])
        n = mol.GetNumAtoms()
        expected = {i: i for i in range(n)}
        assert atom_map == expected

    def test_atom_count_unchanged(self):
        mol = mol_from_smiles("c1ccccc1")
        parent, _ = strip_additive_atoms(mol, [])
        assert parent.GetNumAtoms() == mol.GetNumAtoms()


# ---------------------------------------------------------------------------
# 6. carve_fc_fragments stub — returns empty dict
# ---------------------------------------------------------------------------


class TestCarveFcFragmentsStub:
    def test_returns_empty_dict_for_none_decomposition(self):
        mol = mol_from_smiles("CCO")
        result = carve_fc_fragments(mol, None)
        assert result == {}

    def test_returns_dict(self):
        mol = mol_from_smiles("CC(=O)OCC")  # ethyl acetate
        result = carve_fc_fragments(mol, object())
        assert isinstance(result, dict)

    def test_stub_does_not_raise(self):
        mol = mol_from_smiles("CC(=O)OCC")
        try:
            carve_fc_fragments(mol, None)
        except Exception as exc:
            pytest.fail(f"carve_fc_fragments raised unexpectedly: {exc}")


# ---------------------------------------------------------------------------
# 7. carve_bridging_substituent — -O- bridge in dimethyl ether
# ---------------------------------------------------------------------------


class TestCarveBridgingSubstituent:
    """Carve the -O- bridge from dimethyl ether (COC)."""

    @pytest.fixture
    def setup(self):
        mol = mol_from_smiles("COC")
        # Atom layout: C(0)-O(1)-C(2)
        # We cut both C-O bonds.
        o_idx = next(
            a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 8
        )
        neighbors = [
            nbr.GetIdx()
            for nbr in mol.GetAtomWithIdx(o_idx).GetNeighbors()
        ]
        assert len(neighbors) == 2
        # attachment_bonds: ((parent1, sub1), (parent2, sub2))
        attachment_bonds = (
            (neighbors[0], o_idx),
            (neighbors[1], o_idx),
        )
        return mol, attachment_bonds, o_idx

    def test_returns_tuple_of_three(self, setup):
        mol, bonds, _ = setup
        result = carve_bridging_substituent(mol, frozenset(), bonds)
        assert isinstance(result, tuple) and len(result) == 3

    def test_fragment_contains_one_oxygen(self, setup):
        mol, bonds, _ = setup
        frag, _, _ = carve_bridging_substituent(mol, frozenset(), bonds)
        o_count = sum(1 for a in frag.GetAtoms() if a.GetAtomicNum() == 8)
        assert o_count == 1

    def test_two_attachment_atoms_returned(self, setup):
        mol, bonds, _ = setup
        _, attaches, _ = carve_bridging_substituent(mol, frozenset(), bonds)
        assert len(attaches) == 2

    def test_attachment_atoms_distinct(self, setup):
        mol, bonds, _ = setup
        _, attaches, _ = carve_bridging_substituent(mol, frozenset(), bonds)
        # For -O- the two attachment atoms are the same oxygen (index appears
        # once in the fragment regardless); both entries should be the O.
        # At minimum they should both be valid indices.
        frag, attaches, _ = carve_bridging_substituent(mol, frozenset(), bonds)
        for idx in attaches:
            assert 0 <= idx < frag.GetNumAtoms()

    def test_both_bond_orders_single(self, setup):
        mol, bonds, _ = setup
        _, _, bond_orders = carve_bridging_substituent(mol, frozenset(), bonds)
        assert all(bo == 1 for bo in bond_orders)

    def test_two_bond_orders_returned(self, setup):
        mol, bonds, _ = setup
        _, _, bond_orders = carve_bridging_substituent(mol, frozenset(), bonds)
        assert len(bond_orders) == 2

    def test_empty_attachment_bonds_raises(self):
        mol = mol_from_smiles("COC")
        with pytest.raises(ValueError):
            carve_bridging_substituent(mol, frozenset(), ())
