"""
tests/test_decalin_cis_trans.py

Stage 23 R23-A: cis/trans-decalin retained-functional-class emission.

The bare retained name ``decalin`` cannot encode ring-junction
stereochemistry — both cis-decalin and trans-decalin would silently
round-trip to the same word.  OPSIN does not parse the systematic
``(4aR,8aR)-decahydronaphthalene`` form (any letter-suffix R/S on this
parent returns empty), so the IUPAC P-23.5.5 retained-functional-class
forms ``cis-decalin`` / ``trans-decalin`` are the only round-trip-safe
encoding of the cis/trans distinction.

The override fires in ``ring_naming/retained_lookup.py`` (helper
``_cis_trans_decalin_override``) when ALL of:
  * the matched curated retained name is ``decalin``,
  * the molecule consists solely of the decalin ring atoms (no
    exocyclic substituents — substituted-cis-decalin would not parse
    through OPSIN, so we deliberately gate to the bare-parent case),
  * exactly two atoms carry tetrahedral chirality, AND
  * those two atoms are the ring-junction (degree-3) carbons.

Detection rule (after RDKit canonicalization):
  * cis  → both junction chiral_tags equal (same face)
  * trans → junction chiral_tags differ (opposite faces)
"""

from __future__ import annotations

from rdkit import Chem

from iupac_namer.engine import name_smiles


# ---------------------------------------------------------------------------
# Bare cis/trans-decalin emission — primary closure for the audit-row family
# ---------------------------------------------------------------------------


class TestBareDecalinCisTransEmission:
    def test_cis_decalin_input_form_emits_cis(self):
        """OPSIN's canonical SMILES for ``cis-decalin``."""
        assert name_smiles("C1CCC[C@@H]2CCCC[C@H]12") == "cis-decalin"

    def test_trans_decalin_input_form_emits_trans(self):
        """OPSIN's canonical SMILES for ``trans-decalin``."""
        assert name_smiles("C1CCC[C@@H]2CCCC[C@@H]12") == "trans-decalin"

    def test_cis_decalin_canonical_form_emits_cis(self):
        """Same molecule, different input ordering: rdkit canonical form
        of cis-decalin (both junction H both ``@``)."""
        assert name_smiles("C1CC[C@H]2CCCC[C@H]2C1") == "cis-decalin"

    def test_trans_decalin_canonical_form_emits_trans(self):
        """rdkit canonical form of trans-decalin (junction H ``@`` and
        ``@@``)."""
        assert name_smiles("C1CC[C@H]2CCCC[C@@H]2C1") == "trans-decalin"

    def test_cis_decalin_inverted_input_emits_cis(self):
        """Both-``@@`` canonical-SMILES variant — the equality-of-tags
        test must produce the same cis answer regardless of which
        absolute orientation the input encodes."""
        # Same molecule as v1 above written with both @@ (mirror image
        # of both-@ form is again cis: cis-decalin is achiral / meso).
        assert name_smiles("C1CC[C@@H]2CCCC[C@@H]2C1") == "cis-decalin"


# ---------------------------------------------------------------------------
# Achiral controls — must NOT produce a cis/trans prefix
# ---------------------------------------------------------------------------


class TestAchiralDecalinControls:
    def test_achiral_decalin_emits_bare_decahydronaphthalene(self):
        """Without stereo at the junctions, the cis/trans override must
        not fire and the systematic PIN must be emitted.

        Per P-25.3.1.3 / P-32.4, the PIN for fully-saturated decalin is
        the systematic ``decahydronaphthalene`` (the retained ``decalin``
        is general-nomenclature only).  This is the load-bearing
        regression guard: any change that would cause achiral decalin to
        gain a spurious ``cis-`` / ``trans-`` prefix would break
        round-tripping (OPSIN would then emit stereo-bearing SMILES that
        does not match the achiral input)."""
        assert name_smiles("C1CCC2CCCCC2C1") == "decahydronaphthalene"

    def test_partial_stereo_does_not_trigger_override(self):
        """Only one junction carries chirality — RDKit treats the
        adjacent junction's stereo as undefined.  We must NOT emit
        ``cis``/``trans`` from a single-tag input because the molecule
        is then a mixture of stereoisomers, not a single configuration."""
        # Build a decalin with stereo only at one junction by writing the
        # other junction without an @ tag.  RDKit drops both tags when
        # one is unspecified for many decalin SMILES variants — so this
        # test really checks that the override stays dormant in the
        # underspecified case.  The systematic decahydronaphthalene PIN
        # is emitted (P-25.3.1.3 / P-32.4); the retained ``decalin`` is
        # general-nomenclature only.
        single = "C1CCC[C@@H]2CCCCC12"
        name = name_smiles(single)
        assert name == "decahydronaphthalene", (
            f"expected bare 'decahydronaphthalene' for partial-stereo input "
            f"(RDKit canonicalises without retaining a stereo descriptor); "
            f"got {name!r}"
        )


# ---------------------------------------------------------------------------
# OPSIN round-trip parity
# ---------------------------------------------------------------------------


class TestRoundTripParity:
    """The whole point of the override is round-trip stereo preservation
    — verify that each emitted name parses back through OPSIN to the
    same canonical SMILES as the input."""

    def _round_trip_canonical(self, smiles: str) -> tuple[str, str | None, str]:
        from py2opsin import py2opsin
        name = name_smiles(smiles)
        opsin_smiles = py2opsin(name)
        in_canon = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
        rt_canon = (
            Chem.MolToSmiles(Chem.MolFromSmiles(opsin_smiles))
            if opsin_smiles
            else None
        )
        return in_canon, rt_canon, name

    def test_cis_round_trip_canonical_equality(self):
        in_canon, rt_canon, name = self._round_trip_canonical(
            "C1CCC[C@@H]2CCCC[C@H]12"
        )
        assert rt_canon == in_canon, (
            f"cis-decalin round-trip mismatch: name={name!r}, "
            f"input_canon={in_canon!r}, rt_canon={rt_canon!r}"
        )

    def test_trans_round_trip_canonical_equality(self):
        in_canon, rt_canon, name = self._round_trip_canonical(
            "C1CCC[C@@H]2CCCC[C@@H]12"
        )
        assert rt_canon == in_canon, (
            f"trans-decalin round-trip mismatch: name={name!r}, "
            f"input_canon={in_canon!r}, rt_canon={rt_canon!r}"
        )
