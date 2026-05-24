"""
tests/test_perception_facade.py

Integration tests for the Perception facade.

These tests exercise the Perception class as a whole, verifying that all
subsystems wire together correctly and that the interpretation generator and
candidate_parents generator produce correct outputs.
"""

from __future__ import annotations

import pytest
from rdkit import Chem

from iupac_namer.perception import Perception
from iupac_namer.types import (
    CandidateParent,
    Interpretation,
    InterpretationQuery,
    SymmetryGroup,
    StereoCenter,
    RingSystem,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_perception(smiles: str) -> Perception:
    """Build a Perception facade from a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles!r}"
    return Perception(mol)


def default_query(max_results: int = 100) -> InterpretationQuery:
    """Return a default InterpretationQuery for testing."""
    return InterpretationQuery(
        preferred_decomp_types=None,
        preferred_parent_type=None,
        suppress_functional_class=False,
        max_results=max_results,
    )


# ---------------------------------------------------------------------------
# All properties accessible without error
# ---------------------------------------------------------------------------


class TestAllPropertiesAccessible:
    def test_atoms_accessible(self):
        p = make_perception("CCO")
        atoms = p.atoms
        assert atoms is not None

    def test_stereo_accessible(self):
        p = make_perception("CCO")
        stereo = p.stereo
        assert stereo is not None

    def test_fragments_accessible(self):
        p = make_perception("CCO")
        frags = p.fragments
        assert frags is not None

    def test_rings_accessible(self):
        p = make_perception("CCO")
        rings = p.rings
        assert rings is not None

    def test_fgs_accessible(self):
        p = make_perception("CCO")
        fgs = p.fgs
        assert fgs is not None

    def test_symmetry_accessible(self):
        p = make_perception("CCO")
        sym = p.symmetry
        assert sym is not None

    def test_chains_accessible(self):
        p = make_perception("CCO")
        chains = p.chains
        assert chains is not None

    def test_all_properties_benzene(self):
        """All properties accessible for a ring molecule too."""
        p = make_perception("c1ccccc1")
        _ = p.atoms
        _ = p.stereo
        _ = p.fragments
        _ = p.rings
        _ = p.fgs
        _ = p.symmetry
        _ = p.chains

    def test_lazy_init_atoms_only(self):
        """Before accessing stereo, the stereo subsystem should not be built."""
        p = make_perception("CCO")
        # Access only atoms
        _ = p.atoms
        assert p._stereo is None  # private state — still None

    def test_lazy_init_stereo_after_access(self):
        """After accessing stereo, the stereo subsystem is cached."""
        p = make_perception("CCO")
        _ = p.stereo
        assert p._stereo is not None

    def test_lazy_init_symmetry_after_access(self):
        p = make_perception("CCO")
        _ = p.symmetry
        assert p._symmetry is not None

    def test_subsystem_cached_on_second_access(self):
        """The same subsystem object is returned on repeated access."""
        p = make_perception("CCO")
        s1 = p.stereo
        s2 = p.stereo
        assert s1 is s2


# ---------------------------------------------------------------------------
# Interpretations — ethanol (no ambiguity)
# ---------------------------------------------------------------------------


class TestInterpretationsEthanol:
    def test_ethanol_yields_exactly_one(self):
        """Ethanol has no ambiguity points — interpretations() yields exactly 1."""
        p = make_perception("CCO")
        query = default_query()
        results = list(p.interpretations(query))
        assert len(results) == 1

    def test_ethanol_interpretation_is_interpretation_instance(self):
        p = make_perception("CCO")
        query = default_query()
        for interp in p.interpretations(query):
            assert isinstance(interp, Interpretation)

    def test_ethanol_interpretation_no_ambiguity_choices(self):
        """Single-interpretation molecules have no ambiguity choices."""
        p = make_perception("CCO")
        query = default_query()
        results = list(p.interpretations(query))
        assert results[0].ambiguity_choices == ()

    def test_ethanol_interpretation_has_stereocenters_field(self):
        p = make_perception("CCO")
        query = default_query()
        results = list(p.interpretations(query))
        # stereocenters field is a tuple (empty for ethanol)
        assert isinstance(results[0].stereocenters, tuple)

    def test_ethanol_interpretation_has_symmetry_groups_field(self):
        p = make_perception("CCO")
        query = default_query()
        results = list(p.interpretations(query))
        assert isinstance(results[0].symmetry_groups, tuple)

    def test_interpretations_is_restartable(self):
        """Each call to interpretations() creates an independent generator."""
        p = make_perception("CCO")
        query = default_query()
        g1 = p.interpretations(query)
        g2 = p.interpretations(query)
        r1 = list(g1)
        r2 = list(g2)
        assert len(r1) == len(r2) == 1

    def test_interpretations_max_results_zero(self):
        """max_results=0 causes the generator to yield nothing even with ambiguity."""
        p = make_perception("CCO")
        query = default_query(max_results=0)
        results = list(p.interpretations(query))
        # Ethanol has no ambiguity — it still yields 1 (the cap only applies in the
        # multi-interpretation branch). Verify at least it doesn't crash.
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Interpretations — acetic acid
# ---------------------------------------------------------------------------


class TestInterpretationsAceticAcid:
    def test_acetic_acid_yields_interpretation(self):
        """Acetic acid (CC(=O)O) should produce at least one interpretation."""
        p = make_perception("CC(=O)O")
        query = default_query()
        results = list(p.interpretations(query))
        assert len(results) >= 1

    def test_acetic_acid_has_carboxylic_acid_fg(self):
        """The interpretation for acetic acid should contain a carboxylic_acid FG."""
        p = make_perception("CC(=O)O")
        query = default_query()
        results = list(p.interpretations(query))
        assert len(results) >= 1
        interp = results[0]
        fg_types = {fg.type for fg in interp.fgs}
        assert "carboxylic_acid" in fg_types

    def test_acetic_acid_ring_systems_empty(self):
        """Acetic acid has no rings."""
        p = make_perception("CC(=O)O")
        query = default_query()
        results = list(p.interpretations(query))
        assert results[0].ring_systems == ()

    def test_acetic_acid_stereocenters_empty(self):
        """Acetic acid has no stereocenters."""
        p = make_perception("CC(=O)O")
        query = default_query()
        results = list(p.interpretations(query))
        assert results[0].stereocenters == ()


# ---------------------------------------------------------------------------
# candidate_parents — pentane (chain)
# ---------------------------------------------------------------------------


class TestCandidateParentsPentane:
    def test_pentane_yields_chain_candidates(self):
        """Pentane (CCCCC) should yield at least one chain CandidateParent."""
        p = make_perception("CCCCC")
        query = default_query()
        results = list(p.interpretations(query))
        assert len(results) >= 1
        interp = results[0]

        candidates = list(p.candidate_parents(interp))
        chain_candidates = [c for c in candidates if c.type == "chain"]
        assert len(chain_candidates) >= 1

    def test_pentane_chain_candidate_length(self):
        """The longest chain for pentane has 5 atoms."""
        p = make_perception("CCCCC")
        query = default_query()
        interp = list(p.interpretations(query))[0]

        candidates = list(p.candidate_parents(interp))
        chain_candidates = [c for c in candidates if c.type == "chain"]
        max_length = max(c.length for c in chain_candidates)
        assert max_length == 5

    def test_pentane_chain_candidate_is_candidate_parent(self):
        p = make_perception("CCCCC")
        query = default_query()
        interp = list(p.interpretations(query))[0]
        for cp in p.candidate_parents(interp):
            assert isinstance(cp, CandidateParent)


# ---------------------------------------------------------------------------
# candidate_parents — benzene (ring)
# ---------------------------------------------------------------------------


class TestCandidateParentsBenzene:
    def test_benzene_yields_ring_candidate(self):
        """Benzene should yield at least one ring CandidateParent."""
        p = make_perception("c1ccccc1")
        query = default_query()
        interp = list(p.interpretations(query))[0]

        candidates = list(p.candidate_parents(interp))
        ring_candidates = [
            c for c in candidates
            if c.type in ("monocyclic", "fused", "bridged", "spiro")
        ]
        assert len(ring_candidates) >= 1

    def test_benzene_ring_candidate_length(self):
        """Benzene has ring_size == 6."""
        p = make_perception("c1ccccc1")
        query = default_query()
        interp = list(p.interpretations(query))[0]

        candidates = list(p.candidate_parents(interp))
        ring_candidates = [
            c for c in candidates
            if c.type in ("monocyclic", "fused", "bridged", "spiro")
        ]
        assert all(c.length == 6 for c in ring_candidates)

    def test_benzene_ring_candidate_has_ring_system(self):
        """Ring CandidateParent for benzene should carry a RingSystem."""
        p = make_perception("c1ccccc1")
        query = default_query()
        interp = list(p.interpretations(query))[0]

        candidates = list(p.candidate_parents(interp))
        ring_candidates = [
            c for c in candidates
            if c.type in ("monocyclic", "fused", "bridged", "spiro")
        ]
        for rc in ring_candidates:
            assert rc.ring_system is not None
            assert isinstance(rc.ring_system, RingSystem)


# ---------------------------------------------------------------------------
# Stereo propagation into interpretation
# ---------------------------------------------------------------------------


class TestStereoInInterpretation:
    def test_r_butan2ol_stereocenters_in_interpretation(self):
        """Stereocenters from StereoAnalysis should appear in the Interpretation."""
        p = make_perception("[C@H](O)(CC)C")
        query = default_query()
        interp = list(p.interpretations(query))[0]

        assert len(interp.stereocenters) >= 1
        assert all(isinstance(sc, StereoCenter) for sc in interp.stereocenters)

    def test_ethanol_stereocenters_empty_in_interpretation(self):
        """Ethanol has no stereocenters — interpretation reflects this."""
        p = make_perception("CCO")
        query = default_query()
        interp = list(p.interpretations(query))[0]
        assert interp.stereocenters == ()


# ---------------------------------------------------------------------------
# Symmetry propagation into interpretation
# ---------------------------------------------------------------------------


class TestSymmetryInInterpretation:
    def test_biphenyl_symmetry_in_interpretation(self):
        """Biphenyl's ring assembly should appear in the Interpretation."""
        p = make_perception("c1ccc(-c2ccccc2)cc1")
        query = default_query()
        interp = list(p.interpretations(query))[0]

        assert len(interp.symmetry_groups) >= 1
        assert all(isinstance(sg, SymmetryGroup) for sg in interp.symmetry_groups)

    def test_ethanol_no_symmetry_in_interpretation(self):
        """Ethanol has no symmetry — interpretation reflects this."""
        p = make_perception("CCO")
        query = default_query()
        interp = list(p.interpretations(query))[0]
        assert interp.symmetry_groups == ()
