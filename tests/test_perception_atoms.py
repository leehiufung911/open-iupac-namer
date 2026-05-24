"""
tests/test_perception_atoms.py

Unit tests for iupac_namer.perception.atoms.AtomAnalysis.

Each test constructs a molecule from SMILES and exercises AtomAnalysis
directly, without going through the full Perception facade.
"""

from __future__ import annotations

import pytest
from rdkit import Chem

from iupac_namer.perception.atoms import AtomAnalysis
from iupac_namer.types import AtomInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def mol_from_smiles(smiles: str) -> object:
    """Return a sanitised RDKit mol.  Fail fast if SMILES is invalid."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles!r}"
    return mol


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_ethanol_len(self):
        """CCO has 3 heavy atoms (implicit H mol)."""
        mol = mol_from_smiles("CCO")
        aa = AtomAnalysis(mol)
        # RDKit default mol has no explicit H atoms
        assert len(aa) == 3

    def test_all_atoms_property(self):
        mol = mol_from_smiles("CCO")
        aa = AtomAnalysis(mol)
        assert len(aa.all_atoms) == 3
        # Should be a tuple
        assert isinstance(aa.all_atoms, tuple)

    def test_all_atoms_are_atom_info(self):
        mol = mol_from_smiles("CC")
        aa = AtomAnalysis(mol)
        for atom in aa.all_atoms:
            assert isinstance(atom, AtomInfo)

    def test_all_atoms_frozen(self):
        """AtomInfo is a frozen dataclass."""
        import dataclasses
        mol = mol_from_smiles("C")
        aa = AtomAnalysis(mol)
        with pytest.raises(dataclasses.FrozenInstanceError):
            aa[0].element = "N"  # type: ignore[misc]

    def test_iter_all_atoms(self):
        mol = mol_from_smiles("CCO")
        aa = AtomAnalysis(mol)
        atoms = list(aa)
        assert len(atoms) == 3


# ---------------------------------------------------------------------------
# Ethanol (CCO) — basic element & property checks
# ---------------------------------------------------------------------------


class TestEthanol:
    """CCO — 3 heavy atoms: C(0), C(1), O(2) in typical RDKit ordering."""

    @pytest.fixture
    def aa(self):
        return AtomAnalysis(mol_from_smiles("CCO"))

    def test_three_atoms(self, aa):
        assert len(aa) == 3

    def test_elements(self, aa):
        elements = {a.element for a in aa}
        assert elements == {"C", "O"}

    def test_oxygen_atom(self, aa):
        # Find the oxygen
        oxygens = aa.atoms_by_element("O")
        assert len(oxygens) == 1
        o = oxygens[0]
        assert o.element == "O"
        assert o.atomic_num == 8

    def test_carbon_count(self, aa):
        carbons = aa.atoms_by_element("C")
        assert len(carbons) == 2

    def test_not_in_ring(self, aa):
        for a in aa:
            assert not a.in_ring

    def test_not_aromatic(self, aa):
        for a in aa:
            assert not a.aromatic

    def test_heavy_atom_count(self, aa):
        assert aa.heavy_atom_count() == 3

    def test_ring_atoms_empty(self, aa):
        assert aa.ring_atoms() == frozenset()

    def test_non_ring_atoms_all(self, aa):
        assert aa.non_ring_atoms() == frozenset({0, 1, 2})

    def test_getitem_idx_zero(self, aa):
        a0 = aa[0]
        assert a0.idx == 0

    def test_getitem_out_of_range(self, aa):
        with pytest.raises(IndexError):
            _ = aa[99]

    def test_bond_type_c_o(self, aa):
        # Find the O atom and its C neighbour
        o = aa.atoms_by_element("O")[0]
        assert len(o.neighbors) == 1
        c_idx = o.neighbors[0]
        bt = aa.get_bond_type(o.idx, c_idx)
        assert bt == "single"

    def test_no_bond_between_non_adjacent(self, aa):
        # In CCO, atom 0 and atom 2 are not directly bonded
        bt = aa.get_bond_type(0, 2)
        assert bt is None

    def test_has_bond_true(self, aa):
        assert aa.has_bond(0, 1)

    def test_has_bond_false(self, aa):
        assert not aa.has_bond(0, 2)


# ---------------------------------------------------------------------------
# Benzene (c1ccccc1) — ring and aromaticity
# ---------------------------------------------------------------------------


class TestBenzene:
    @pytest.fixture
    def aa(self):
        return AtomAnalysis(mol_from_smiles("c1ccccc1"))

    def test_six_atoms(self, aa):
        assert len(aa) == 6

    def test_all_carbon(self, aa):
        assert all(a.element == "C" for a in aa)

    def test_all_in_ring(self, aa):
        assert all(a.in_ring for a in aa)

    def test_all_aromatic(self, aa):
        assert all(a.aromatic for a in aa)

    def test_ring_atoms_all(self, aa):
        assert aa.ring_atoms() == frozenset(range(6))

    def test_non_ring_atoms_empty(self, aa):
        assert aa.non_ring_atoms() == frozenset()

    def test_aromatic_atoms(self, aa):
        assert aa.aromatic_atoms() == frozenset(range(6))

    def test_all_bond_types_aromatic(self, aa):
        for atom in aa:
            for _, bt in atom.bond_types:
                assert bt == "aromatic"

    def test_heavy_atom_count(self, aa):
        assert aa.heavy_atom_count() == 6

    def test_atoms_by_element_c(self, aa):
        carbons = aa.atoms_by_element("C")
        assert len(carbons) == 6

    def test_atoms_by_element_n_empty(self, aa):
        assert aa.atoms_by_element("N") == ()

    def test_degree_two_each(self, aa):
        # In benzene each C is bonded to 2 ring C's (implicit H not counted)
        assert all(a.degree == 2 for a in aa)

    def test_neighbors_two_each(self, aa):
        assert all(len(a.neighbors) == 2 for a in aa)


# ---------------------------------------------------------------------------
# Acetic acid (CC(=O)O) — double bond detection
# ---------------------------------------------------------------------------


class TestAceticAcid:
    """CC(=O)O — atoms: CH3(0), C(1), =O(2), OH(3)."""

    @pytest.fixture
    def aa(self):
        return AtomAnalysis(mol_from_smiles("CC(=O)O"))

    def test_four_heavy_atoms(self, aa):
        assert aa.heavy_atom_count() == 4

    def test_two_oxygens(self, aa):
        assert len(aa.atoms_by_element("O")) == 2

    def test_double_bond_present(self, aa):
        """The carbonyl C has a double bond to the carbonyl O."""
        bond_type_strings = set()
        for atom in aa:
            for _, bt in atom.bond_types:
                bond_type_strings.add(bt)
        assert "double" in bond_type_strings

    def test_carbonyl_oxygen_has_double_bond(self, aa):
        """The O with no hydrogen (=O) should have exactly one bond: double."""
        for o in aa.atoms_by_element("O"):
            if o.degree == 1:
                # This is the carbonyl oxygen (=O)
                assert len(o.bond_types) == 1
                _, bt = o.bond_types[0]
                assert bt == "double"
                break
        else:
            pytest.fail("No degree-1 oxygen found in acetic acid")

    def test_get_bond_type_double(self, aa):
        """get_bond_type reports double for the C=O bond."""
        # Find carbonyl C and carbonyl O
        for o in aa.atoms_by_element("O"):
            if o.degree == 1:
                c_idx = o.neighbors[0]
                bt = aa.get_bond_type(o.idx, c_idx)
                assert bt == "double"
                break
        else:
            pytest.fail("No degree-1 oxygen found in acetic acid")

    def test_not_in_ring(self, aa):
        assert all(not a.in_ring for a in aa)

    def test_not_aromatic(self, aa):
        assert all(not a.aromatic for a in aa)


# ---------------------------------------------------------------------------
# Methylenecyclohexane — mixed ring / non-ring
# ---------------------------------------------------------------------------


class TestMethylenecyclohexane:
    """C=C1CCCCC1 — exocyclic double bond on cyclohexane."""

    @pytest.fixture
    def aa(self):
        return AtomAnalysis(mol_from_smiles("C=C1CCCCC1"))

    def test_seven_heavy_atoms(self, aa):
        assert aa.heavy_atom_count() == 7

    def test_ring_and_non_ring_atoms(self, aa):
        ring = aa.ring_atoms()
        non_ring = aa.non_ring_atoms()
        assert len(ring) == 6
        assert len(non_ring) == 1

    def test_ring_non_ring_partition(self, aa):
        """ring_atoms + non_ring_atoms == all atom indices."""
        ring = aa.ring_atoms()
        non_ring = aa.non_ring_atoms()
        all_idxs = frozenset(range(len(aa)))
        assert ring | non_ring == all_idxs
        assert ring & non_ring == frozenset()

    def test_exocyclic_double_bond(self, aa):
        """The exocyclic C=C has a double bond."""
        non_ring_idx = next(iter(aa.non_ring_atoms()))
        exo_c = aa[non_ring_idx]
        assert len(exo_c.bond_types) == 1
        _, bt = exo_c.bond_types[0]
        assert bt == "double"


# ---------------------------------------------------------------------------
# Pyridine — ring with nitrogen
# ---------------------------------------------------------------------------


class TestPyridine:
    """c1ccncc1 — 6-membered aromatic ring with one N."""

    @pytest.fixture
    def aa(self):
        return AtomAnalysis(mol_from_smiles("c1ccncc1"))

    def test_six_atoms(self, aa):
        assert len(aa) == 6

    def test_one_nitrogen(self, aa):
        assert len(aa.atoms_by_element("N")) == 1

    def test_five_carbons(self, aa):
        assert len(aa.atoms_by_element("C")) == 5

    def test_all_in_ring(self, aa):
        assert all(a.in_ring for a in aa)

    def test_nitrogen_aromatic(self, aa):
        n = aa.atoms_by_element("N")[0]
        assert n.aromatic

    def test_nitrogen_atomic_num(self, aa):
        n = aa.atoms_by_element("N")[0]
        assert n.atomic_num == 7


# ---------------------------------------------------------------------------
# atoms_by_element edge cases
# ---------------------------------------------------------------------------


class TestAtomsByElement:
    def test_returns_tuple(self):
        aa = AtomAnalysis(mol_from_smiles("C"))
        result = aa.atoms_by_element("C")
        assert isinstance(result, tuple)

    def test_empty_for_missing_element(self):
        aa = AtomAnalysis(mol_from_smiles("C"))
        assert aa.atoms_by_element("N") == ()

    def test_case_sensitive(self):
        """Element symbols are case-sensitive ("C" != "c")."""
        aa = AtomAnalysis(mol_from_smiles("c1ccccc1"))
        # RDKit normalises aromatic atoms to uppercase symbol
        assert len(aa.atoms_by_element("C")) == 6
        assert aa.atoms_by_element("c") == ()


# ---------------------------------------------------------------------------
# atoms_with_charge
# ---------------------------------------------------------------------------


class TestAtomsWithCharge:
    def test_no_charge_in_ethanol(self):
        aa = AtomAnalysis(mol_from_smiles("CCO"))
        assert aa.atoms_with_charge() == ()

    def test_charged_atoms_in_acetate(self):
        """CC([O-])=O — one negatively charged oxygen."""
        aa = AtomAnalysis(mol_from_smiles("CC([O-])=O"))
        charged = aa.atoms_with_charge()
        assert len(charged) == 1
        assert charged[0].charge == -1

    def test_ammonium(self):
        """[NH4+] — one positively charged nitrogen."""
        aa = AtomAnalysis(mol_from_smiles("[NH4+]"))
        charged = aa.atoms_with_charge()
        assert len(charged) == 1
        assert charged[0].charge == 1
        assert charged[0].element == "N"


# ---------------------------------------------------------------------------
# Perception facade integration
# ---------------------------------------------------------------------------


class TestPerceptionFacade:
    """Smoke test: AtomAnalysis is accessible via Perception.atoms."""

    def test_perception_atoms_property(self):
        from iupac_namer.perception import Perception

        mol = mol_from_smiles("CCO")
        p = Perception(mol)
        aa = p.atoms
        assert isinstance(aa, AtomAnalysis)
        assert len(aa) == 3

    def test_perception_atoms_cached(self):
        """Second access returns the same object (lazy init)."""
        from iupac_namer.perception import Perception

        mol = mol_from_smiles("CCO")
        p = Perception(mol)
        assert p.atoms is p.atoms

    def test_perception_stubs_raise_not_implemented(self):
        """All former stubs are now implemented (Phase 1.3e-g). Verify no raises."""
        from iupac_namer.perception import Perception

        mol = mol_from_smiles("CCO")
        p = Perception(mol)
        # p.stereo now implemented (Phase 1.3e) — no longer raises
        _ = p.stereo
        # p.rings now implemented (Phase 1.3b) — no longer raises
        # p.fgs now implemented (Phase 1.3d) — no longer raises
        # p.symmetry now implemented (Phase 1.3f) — no longer raises
        _ = p.symmetry
        # p.chains now implemented (Phase 1.3c) — no longer raises
