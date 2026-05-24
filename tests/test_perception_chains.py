"""
tests/test_perception_chains.py

Unit tests for iupac_namer.perception.chains.ChainFinding.

Each test constructs a molecule from SMILES and exercises ChainFinding
directly (without going through the full Perception facade).

Coverage:
  - Acyclic molecules: methane, ethane, pentane, branched alkanes
  - Heteroatom-containing chains: ethanol (O not counted as chain C)
  - Unsaturated chains: propene (double bond), 1-butyne (triple bond)
  - Ring molecules: toluene (methyl chain only), cyclohexane (no chain),
    2-ethylnaphthalene (ethyl chain only)
  - find_candidate_chains return type and ordering
  - detect_chain_unsaturation locants and types
"""

from __future__ import annotations

import pytest
from rdkit import Chem

from iupac_namer.perception.atoms import AtomAnalysis
from iupac_namer.perception.chains import ChainFinding
from iupac_namer.perception.rings import RingAnalysis
from iupac_namer.types import CandidateParent, UnsaturationInfix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_chain_finding(smiles: str) -> tuple[ChainFinding, object]:
    """Construct ChainFinding from a SMILES string.

    Returns (chain_finding, mol).
    """
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles!r}"
    atoms = AtomAnalysis(mol)
    rings = RingAnalysis(mol, atoms)
    cf = ChainFinding(mol, atoms, rings)
    return cf, mol


def longest_chain_length(smiles: str) -> int:
    """Return the length of the longest candidate chain for the given SMILES."""
    cf, _ = make_chain_finding(smiles)
    candidates = cf.find_candidate_chains()
    if not candidates:
        return 0
    return candidates[0].length


# ---------------------------------------------------------------------------
# Basic chain length tests
# ---------------------------------------------------------------------------


class TestChainLength:
    def test_methane_length_1(self):
        """Methane: single carbon, chain length = 1."""
        assert longest_chain_length("C") == 1

    def test_ethane_length_2(self):
        """Ethane: two carbons, chain length = 2."""
        assert longest_chain_length("CC") == 2

    def test_pentane_length_5(self):
        """Pentane: five carbons, chain length = 5."""
        assert longest_chain_length("CCCCC") == 5

    def test_2_methylbutane_longest_4(self):
        """2-Methylbutane: longest chain is butane (4 carbons)."""
        assert longest_chain_length("CC(C)CC") == 4

    def test_2_3_dimethylbutane_longest_4(self):
        """2,3-Dimethylbutane: longest chain is 4 carbons."""
        assert longest_chain_length("CC(C)C(C)C") == 4

    def test_ethanol_chain_carbon_only(self):
        """Ethanol (CCO): chain contains only the 2 carbon atoms (length 2).

        Oxygen is a heteroatom.  Per IUPAC P-31, acyclic substitutive chains
        consist only of carbon atoms.  The chain finding algorithm must
        restrict the acyclic graph to carbon atoms only; the O of ethanol is
        excluded from the chain.  Chain length must be exactly 2.
        """
        cf, mol = make_chain_finding("CCO")
        candidates = cf.find_candidate_chains()
        assert candidates, "Expected at least one candidate chain for CCO"
        # Chain length must be exactly 2 (two C atoms only, O excluded)
        assert candidates[0].length == 2, (
            f"Expected chain length 2 (C-only), got {candidates[0].length}"
        )
        # All atoms in the chain must be carbon
        for idx in candidates[0].atom_indices:
            atom = mol.GetAtomWithIdx(idx)
            assert atom.GetSymbol() == "C", (
                f"Non-carbon atom {atom.GetSymbol()} found in chain"
            )

    def test_amine_chain_stops_at_nitrogen(self):
        """CCNCC: chain must stop at N; longest C-only chain is 2 (ethyl)."""
        cf, mol = make_chain_finding("CCNCC")
        candidates = cf.find_candidate_chains()
        assert candidates, "Expected candidate chains for CCNCC"
        # Longest carbon chain is 2 (the two ethyl groups are separate)
        assert candidates[0].length == 2, (
            f"Expected chain length 2 (C-only, stops at N), got {candidates[0].length}"
        )
        # All chain atoms must be carbon
        for idx in candidates[0].atom_indices:
            atom = mol.GetAtomWithIdx(idx)
            assert atom.GetSymbol() == "C", (
                f"Non-carbon atom {atom.GetSymbol()} in chain"
            )

    def test_tertiary_amine_chain_stops_at_nitrogen(self):
        """CCCCN(C)C: chain must stop at N; longest C-only chain is 4 (butyl)."""
        cf, mol = make_chain_finding("CCCCN(C)C")
        candidates = cf.find_candidate_chains()
        assert candidates, "Expected candidate chains for CCCCN(C)C"
        # Longest carbon chain is 4 (the butyl group)
        assert candidates[0].length == 4, (
            f"Expected chain length 4 (C-only butyl chain), got {candidates[0].length}"
        )


# ---------------------------------------------------------------------------
# Ring molecules
# ---------------------------------------------------------------------------


class TestRingMolecules:
    def test_cyclohexane_no_chains(self):
        """Cyclohexane: all atoms are ring atoms — no acyclic chains."""
        cf, _ = make_chain_finding("C1CCCCC1")
        candidates = cf.find_candidate_chains()
        assert candidates == [], (
            "Cyclohexane has no acyclic atoms, so no candidate chains"
        )

    def test_toluene_chain_length_1(self):
        """Toluene (Cc1ccccc1): acyclic portion is one methyl carbon."""
        assert longest_chain_length("Cc1ccccc1") == 1

    def test_toluene_chain_atom_not_ring(self):
        """Toluene: the chain atom must not be a ring atom."""
        cf, mol = make_chain_finding("Cc1ccccc1")
        candidates = cf.find_candidate_chains()
        assert candidates, "Expected a chain for toluene"
        # The ring has 6 atoms; chain should contain only the methyl C
        assert candidates[0].length == 1
        # The ring atoms: 6 carbons; chain must NOT overlap with ring atoms
        ring_atoms = set()
        for atom in mol.GetAtoms():
            if atom.IsInRing():
                ring_atoms.add(atom.GetIdx())
        for chain_idx in candidates[0].atom_indices:
            assert chain_idx not in ring_atoms, (
                f"Chain atom {chain_idx} is a ring atom"
            )

    def test_2_ethylnaphthalene_chain_length_2(self):
        """2-Ethylnaphthalene: acyclic chain = 2 (the ethyl group only)."""
        assert longest_chain_length("CCc1ccc2ccccc2c1") == 2


# ---------------------------------------------------------------------------
# Unsaturation detection
# ---------------------------------------------------------------------------


class TestUnsaturation:
    def test_propene_has_double_bond(self):
        """Propene (CC=C): chain of 3 atoms, 1 double bond."""
        cf, _ = make_chain_finding("CC=C")
        candidates = cf.find_candidate_chains()
        assert candidates, "Expected a candidate chain for propene"
        # At least one candidate should have a double-bond unsaturation
        any_double = any(
            c.unsaturation is not None
            and any(u.type == "en" for u in c.unsaturation)
            for c in candidates
        )
        assert any_double, "Expected a double-bond (en) unsaturation infix"

    def test_propene_chain_length_3(self):
        """Propene: chain length = 3."""
        assert longest_chain_length("CC=C") == 3

    def test_1_butyne_has_triple_bond(self):
        """1-Butyne (C#CCC): chain of 4 atoms, 1 triple bond."""
        cf, _ = make_chain_finding("C#CCC")
        candidates = cf.find_candidate_chains()
        assert candidates, "Expected a candidate chain for 1-butyne"
        any_triple = any(
            c.unsaturation is not None
            and any(u.type == "yn" for u in c.unsaturation)
            for c in candidates
        )
        assert any_triple, "Expected a triple-bond (yn) unsaturation infix"

    def test_1_butyne_chain_length_4(self):
        """1-Butyne: chain length = 4."""
        assert longest_chain_length("C#CCC") == 4

    def test_saturated_chain_no_unsaturation(self):
        """Pentane: no unsaturation."""
        cf, _ = make_chain_finding("CCCCC")
        candidates = cf.find_candidate_chains()
        assert candidates
        for c in candidates:
            assert c.unsaturation is None or c.unsaturation == (), (
                "Saturated chain should have no unsaturation"
            )


# ---------------------------------------------------------------------------
# detect_chain_unsaturation directly
# ---------------------------------------------------------------------------


class TestDetectChainUnsaturation:
    def test_single_double_bond_locant_1(self):
        """CH2=CH-CH3: double bond at position 1 (between atoms 0 and 1)."""
        cf, mol = make_chain_finding("C=CC")
        # Find chain atoms in order
        # atom 0: C, atom 1: C, atom 2: C  (canonical SMILES indexing)
        # We need to find actual atom indices — just look at the chain
        candidates = cf.find_candidate_chains()
        assert candidates
        # detect_chain_unsaturation takes an ordered list
        # The chain is 3 atoms; the double bond is between the first two
        chain_atoms = sorted(candidates[0].atom_indices)
        # Try both directions
        infixes_fwd = cf.detect_chain_unsaturation(chain_atoms)
        infixes_rev = cf.detect_chain_unsaturation(list(reversed(chain_atoms)))
        # At least one direction should find the double bond
        has_en = (
            any(u.type == "en" for u in infixes_fwd)
            or any(u.type == "en" for u in infixes_rev)
        )
        assert has_en, "Expected to find en unsaturation"

    def test_no_bonds_empty_result(self):
        """Single carbon: no bonds, no unsaturation."""
        cf, _ = make_chain_finding("C")
        infixes = cf.detect_chain_unsaturation([0])
        assert infixes == ()

    def test_double_bond_infix_type(self):
        """UnsaturationInfix type for double bond is 'en'."""
        cf, mol = make_chain_finding("C=C")
        # atom 0 and atom 1 are the two carbons
        infixes = cf.detect_chain_unsaturation([0, 1])
        assert len(infixes) == 1
        assert infixes[0].type == "en"

    def test_triple_bond_infix_type(self):
        """UnsaturationInfix type for triple bond is 'yn'."""
        cf, mol = make_chain_finding("C#C")
        infixes = cf.detect_chain_unsaturation([0, 1])
        assert len(infixes) == 1
        assert infixes[0].type == "yn"

    def test_locant_numbering_1_indexed(self):
        """Locant for the first double bond is Locant.numeric(1) when no numbering given."""
        cf, mol = make_chain_finding("C=CC")
        from iupac_namer.types import Locant
        # atom 0 =double= atom 1 -single- atom 2
        infixes = cf.detect_chain_unsaturation([0, 1, 2])
        en_infixes = [u for u in infixes if u.type == "en"]
        assert en_infixes, "Expected en infix"
        assert en_infixes[0].locants[0] == Locant.numeric(1)

    def test_diene_two_double_bonds(self):
        """1,3-Butadiene (C=CC=C): two double bonds, multiplier 'di'."""
        cf, mol = make_chain_finding("C=CC=C")
        # chain atoms in order: 0-1-2-3
        infixes = cf.detect_chain_unsaturation([0, 1, 2, 3])
        en_infixes = [u for u in infixes if u.type == "en"]
        assert en_infixes, "Expected en infix"
        assert en_infixes[0].multiplier == "di", (
            f"Expected multiplier 'di', got {en_infixes[0].multiplier!r}"
        )
        assert len(en_infixes[0].locants) == 2


# ---------------------------------------------------------------------------
# CandidateParent structure
# ---------------------------------------------------------------------------


class TestCandidateParentStructure:
    def test_returns_candidate_parent_objects(self):
        """find_candidate_chains returns CandidateParent instances."""
        cf, _ = make_chain_finding("CCCCC")
        candidates = cf.find_candidate_chains()
        for c in candidates:
            assert isinstance(c, CandidateParent)

    def test_type_is_chain(self):
        """All returned candidates have type='chain'."""
        cf, _ = make_chain_finding("CCCCC")
        candidates = cf.find_candidate_chains()
        for c in candidates:
            assert c.type == "chain"

    def test_atom_indices_is_frozenset(self):
        """atom_indices is a frozenset."""
        cf, _ = make_chain_finding("CCC")
        candidates = cf.find_candidate_chains()
        for c in candidates:
            assert isinstance(c.atom_indices, frozenset)

    def test_ring_system_is_none_for_chain(self):
        """Chains have ring_system=None."""
        cf, _ = make_chain_finding("CCCCC")
        candidates = cf.find_candidate_chains()
        for c in candidates:
            assert c.ring_system is None

    def test_longest_first(self):
        """Candidates are returned longest-first."""
        cf, _ = make_chain_finding("CC(C)CC")
        candidates = cf.find_candidate_chains()
        if len(candidates) > 1:
            lengths = [c.length for c in candidates]
            assert lengths == sorted(lengths, reverse=True), (
                "Candidates not sorted longest-first"
            )

    def test_pentane_length_field(self):
        """CandidateParent.length matches the actual atom count."""
        cf, _ = make_chain_finding("CCCCC")
        candidates = cf.find_candidate_chains()
        assert candidates
        assert candidates[0].length == len(candidates[0].atom_indices)
        assert candidates[0].length == 5


# ---------------------------------------------------------------------------
# PCG anchor filtering
# ---------------------------------------------------------------------------


class TestPCGAnchorFilter:
    def test_anchor_in_chain_passes_filter(self):
        """A chain that contains the anchor atom is returned."""
        cf, mol = make_chain_finding("CCCC")
        # Anchor = atom 0 (first carbon)
        candidates_all = cf.find_candidate_chains()
        candidates_anchored = cf.find_candidate_chains(pcg_anchors=(0,))
        assert candidates_anchored, "Chain containing anchor 0 should be returned"

    def test_unreachable_anchor_returns_all(self):
        """If no chain relates to the anchor, all chains are returned (fallback)."""
        cf, _ = make_chain_finding("CCC")
        # Anchor index 999 doesn't exist — fallback should return all chains
        candidates_all = cf.find_candidate_chains()
        candidates_filtered = cf.find_candidate_chains(pcg_anchors=(999,))
        # Both should return chains (fallback)
        assert len(candidates_filtered) == len(candidates_all)
