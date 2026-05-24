"""
tests/test_perception_stereo.py

Unit tests for iupac_namer.perception.stereo.StereoAnalysis.

Each test constructs a molecule from SMILES and exercises StereoAnalysis
directly, without going through the full Perception facade.
"""

from __future__ import annotations

import pytest
from rdkit import Chem

from iupac_namer.perception.atoms import AtomAnalysis
from iupac_namer.perception.stereo import StereoAnalysis
from iupac_namer.types import StereoCenter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_stereo(smiles: str) -> StereoAnalysis:
    """Build AtomAnalysis + StereoAnalysis from a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles!r}"
    aa = AtomAnalysis(mol)
    return StereoAnalysis(mol, aa)


# ---------------------------------------------------------------------------
# No stereo
# ---------------------------------------------------------------------------


class TestNoStereo:
    def test_ethanol_no_stereocenters(self):
        """Ethanol (CCO) has no stereocenters."""
        sa = make_stereo("CCO")
        assert sa.has_stereo is False
        assert sa.stereocenters == ()
        assert len(sa) == 0

    def test_ethane_no_stereocenters(self):
        """Ethane (CC) has no stereocenters."""
        sa = make_stereo("CC")
        assert sa.has_stereo is False

    def test_benzene_no_stereocenters(self):
        """Benzene is aromatic and has no CIP stereocenters."""
        sa = make_stereo("c1ccccc1")
        assert sa.has_stereo is False

    def test_cyclohexane_no_stereocenters(self):
        sa = make_stereo("C1CCCCC1")
        assert sa.has_stereo is False


# ---------------------------------------------------------------------------
# Tetrahedral stereocenters
# ---------------------------------------------------------------------------


class TestTetrahedral:
    def test_r_butan2ol_one_center(self):
        """(R)-Butan-2-ol has exactly one tetrahedral stereocenter."""
        # [C@H] at the chiral carbon with (OH)(CC)(CH3) neighbours gives R
        sa = make_stereo("[C@H](O)(CC)C")
        tetrahedral = sa.tetrahedral_centers
        assert len(tetrahedral) == 1

    def test_r_butan2ol_descriptor(self):
        """The CIP descriptor for (R)-butan-2-ol should be 'R'."""
        # [C@H] at the chiral carbon with (OH)(CC)(CH3) neighbours gives R by RDKit CIP
        sa = make_stereo("[C@H](O)(CC)C")
        tetrahedral = sa.tetrahedral_centers
        assert len(tetrahedral) == 1
        assert tetrahedral[0].descriptor == "R"

    def test_s_alanine_one_center(self):
        """(S)-Alanine: N[C@@H](C)C(=O)O has one tetrahedral stereocenter."""
        sa = make_stereo("N[C@@H](C)C(=O)O")
        tetrahedral = sa.tetrahedral_centers
        assert len(tetrahedral) == 1

    def test_s_alanine_descriptor(self):
        """CIP descriptor for (S)-alanine should be 'S'."""
        sa = make_stereo("N[C@@H](C)C(=O)O")
        tetrahedral = sa.tetrahedral_centers
        # The chiral centre in N[C@@H](C)C(=O)O is S
        assert tetrahedral[0].descriptor == "S"

    def test_tetrahedral_type_field(self):
        """StereoCenter.type must be 'tetrahedral' for chiral atoms."""
        sa = make_stereo("[C@H](O)(CC)C")
        for sc in sa.tetrahedral_centers:
            assert sc.type == "tetrahedral"

    def test_all_stereocenters_are_stereocenter_instances(self):
        sa = make_stereo("[C@H](O)(CC)C")
        for sc in sa.stereocenters:
            assert isinstance(sc, StereoCenter)

    def test_two_chiral_centers(self):
        """Tartaric acid has two chiral carbons."""
        # (2R,3R)-tartaric acid
        sa = make_stereo("O=C([C@@H](O)[C@@H](O)C(=O)O)O")
        assert len(sa.tetrahedral_centers) == 2


# ---------------------------------------------------------------------------
# Double-bond stereocenters
# ---------------------------------------------------------------------------


class TestDoubleBond:
    def test_e_but2ene_detected(self):
        """(E)-but-2-ene: trans double bond should produce one E/Z entry."""
        # /C=C/ gives E (trans) for but-2-ene: C/C=C/C
        sa = make_stereo("C/C=C/C")
        db = sa.double_bond_centers
        assert len(db) >= 1

    def test_e_but2ene_descriptor_e(self):
        """Descriptor for C/C=C/C should be 'E'."""
        sa = make_stereo("C/C=C/C")
        db = sa.double_bond_centers
        assert len(db) >= 1
        # All detected double-bond centers should have a descriptor
        descriptors = {sc.descriptor for sc in db}
        assert "E" in descriptors

    def test_z_but2ene_descriptor_z(self):
        """Descriptor for C/C=C\\C should be 'Z'."""
        sa = make_stereo("C/C=C\\C")
        db = sa.double_bond_centers
        assert len(db) >= 1
        descriptors = {sc.descriptor for sc in db}
        assert "Z" in descriptors

    def test_double_bond_type_field(self):
        """StereoCenter.type must be 'double_bond' for E/Z centers."""
        sa = make_stereo("C/C=C/C")
        for sc in sa.double_bond_centers:
            assert sc.type == "double_bond"

    def test_unspecified_double_bond_no_stereo(self):
        """A plain C=C (no stereo annotation) yields no E/Z stereocenters."""
        sa = make_stereo("CC=CC")
        # Without stereo annotation, no E/Z centers
        assert len(sa.double_bond_centers) == 0


# ---------------------------------------------------------------------------
# stereo_at_atom query
# ---------------------------------------------------------------------------


class TestStereoAtAtom:
    def test_stereo_at_atom_returns_none_for_achiral(self):
        """stereo_at_atom returns None for a non-chiral atom."""
        mol = Chem.MolFromSmiles("CCO")
        aa = AtomAnalysis(mol)
        sa = StereoAnalysis(mol, aa)
        for idx in range(mol.GetNumAtoms()):
            assert sa.stereo_at_atom(idx) is None

    def test_stereo_at_atom_returns_correct_result(self):
        """stereo_at_atom returns the correct StereoCenter for a chiral atom."""
        mol = Chem.MolFromSmiles("[C@H](O)(CC)C")
        aa = AtomAnalysis(mol)
        sa = StereoAnalysis(mol, aa)

        # Find the chiral atom index
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        chiral_atoms = [
            a.GetIdx()
            for a in mol.GetAtoms()
            if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        ]
        assert len(chiral_atoms) == 1
        chiral_idx = chiral_atoms[0]

        sc = sa.stereo_at_atom(chiral_idx)
        assert sc is not None
        assert sc.atom_idx == chiral_idx
        assert sc.type == "tetrahedral"

    def test_stereo_at_atom_returns_none_for_other_atoms(self):
        """stereo_at_atom returns None for atoms that are not the chiral centre."""
        mol = Chem.MolFromSmiles("[C@H](O)(CC)C")
        aa = AtomAnalysis(mol)
        sa = StereoAnalysis(mol, aa)

        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        chiral_atoms = {
            a.GetIdx()
            for a in mol.GetAtoms()
            if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        }
        for idx in range(mol.GetNumAtoms()):
            if idx not in chiral_atoms:
                assert sa.stereo_at_atom(idx) is None


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_tetrahedral_centers_subset(self):
        """tetrahedral_centers is a subset of stereocenters."""
        sa = make_stereo("N[C@@H](C)C(=O)O")
        for sc in sa.tetrahedral_centers:
            assert sc in sa.stereocenters

    def test_double_bond_centers_subset(self):
        """double_bond_centers is a subset of stereocenters."""
        sa = make_stereo("C/C=C/C")
        for sc in sa.double_bond_centers:
            assert sc in sa.stereocenters

    def test_stereocenters_immutable(self):
        """stereocenters returns the same tuple on repeated access."""
        sa = make_stereo("[C@H](O)(CC)C")
        assert sa.stereocenters is sa.stereocenters
