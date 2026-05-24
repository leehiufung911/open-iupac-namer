"""
tests/test_stereo_retained_emission.py

Regression tests for Stage 6 R1-I: stereo emission on retained-name shortcuts
and bridged rings.

Failure modes fixed here:

1. Retained-name shortcuts like ``decalin`` cannot express ring-fusion
   cis/trans stereochemistry in the word itself.  When the input SMILES
   carries chiral tags at the bridgeheads, the engine must NOT silently
   round-trip to the bare retained stem — it must fall through to a
   systematic plan instead.

2. Bridged von-Baeyer ring parents (bicyclo[2.2.1]heptane, etc.) have
   numeric locants that round-trip cleanly through OPSIN.  The previous
   ``_collect_stereo_descriptors`` guard skipped all non-monocyclic
   rings to avoid letter-suffixed locants (``3a``, ``4b``); the Stage 6
   R1-I relaxation allows the VB class through.  Camphor-2-one, dimethyl-
   norbornene and the like now emit ``(1R,4R)`` / ``(1S,4R)``.

Round-trip correctness is validated by the authoritative eval; these
tests only assert that the stereo descriptor is present / that the bare
retained stem is not used.
"""

from __future__ import annotations

import re

import pytest

from iupac_namer.engine import name_smiles


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_STEREO_PREFIX_RE = re.compile(r"^\(\s*[^)]*[RSEZ][^)]*\)-")


def _has_descriptor_char(name: str, chars: str) -> bool:
    """True iff the leading stereodescriptor region contains any of *chars*."""
    m = _STEREO_PREFIX_RE.match(name)
    if m is None:
        return False
    region = m.group(0)
    return any(ch in region for ch in chars)


# ---------------------------------------------------------------------------
# Decalin: retained-name shortcut must not drop ring-fusion stereo
# ---------------------------------------------------------------------------


class TestDecalinRetainedStereoGuard:
    def test_achiral_decalin_still_uses_retained(self):
        """Without stereo, ``decalin`` retained name remains acceptable."""
        name = name_smiles("C1CCC2CCCCC2C1")
        assert "decalin" in name or "decahydronaphthalene" in name

    def test_cis_decalin_does_not_round_trip_to_bare_decalin(self):
        """cis-decalin: bare ``decalin`` would silently drop stereo.

        Closed by Stage 23 R23-A — the curated decalin retained match is
        upgraded to the P-23.5.5 functional-class form ``cis-decalin``
        when the input molecule consists solely of the decalin ring
        system AND both junction atoms carry equal chiral_tags (same
        face).  See ``_cis_trans_decalin_override`` in
        ``ring_naming/retained_lookup.py``.
        """
        name = name_smiles("C1CC[C@H]2CCCC[C@H]2C1")
        assert name != "decalin", (
            f"cis-decalin round-tripped to bare 'decalin' "
            f"(stereo dropped); got {name!r}"
        )
        assert name == "cis-decalin", (
            f"expected 'cis-decalin' from stage-23 R23-A override; got {name!r}"
        )

    def test_trans_decalin_does_not_round_trip_to_bare_decalin(self):
        """trans-decalin: analogous guard.  Closed by Stage 23 R23-A —
        opposite chiral_tags at the two junctions trigger the
        ``trans-decalin`` upgrade."""
        name = name_smiles("C1CC[C@H]2CCCC[C@@H]2C1")
        assert name != "decalin", (
            f"trans-decalin round-tripped to bare 'decalin' "
            f"(stereo dropped); got {name!r}"
        )
        assert name == "trans-decalin", (
            f"expected 'trans-decalin' from stage-23 R23-A override; got {name!r}"
        )


# ---------------------------------------------------------------------------
# Bridged bicyclic stereo emission
# ---------------------------------------------------------------------------


class TestBridgedRingStereoEmission:
    """Bridged von-Baeyer ring-stereo emission.

    Stage 22 R22-D closes these previously-xfailed cases via the R22-C
    OPSIN-validation pass: ``_collect_stereo_descriptors`` now admits
    bridged-parent tetrahedral R/S at plain-int locants, and the
    post-assembly validator strips descriptors when OPSIN rejects the
    candidate (tropane / morphinan derivatives — see
    ``test_bridged_tetrahedral_stereo.py`` for the dedicated coverage).
    """

    def test_camphor_2_one_retains_r_descriptors(self):
        """(1R,4R)-camphor-2-one: closed by R22-D."""
        smiles = "C[C@@]12C(C[C@@H](CC1)C2(C)C)=O"
        name = name_smiles(smiles)
        assert _has_descriptor_char(name, "RS"), (
            f"camphor-2-one emitted no R/S descriptor; got {name!r}"
        )

    def test_dimethylnorbornene_retains_r_descriptors(self):
        """7,7-dimethylbicyclo[2.2.1]hept-2-ene: closed by R22-D."""
        smiles = "CC1(C)[C@@H]2CC[C@@H]1C=C2"
        name = name_smiles(smiles)
        assert _has_descriptor_char(name, "RS"), (
            f"dimethyl-norbornene emitted no R/S descriptor; got {name!r}"
        )

    def test_achiral_bicyclo_still_names_without_stereo(self):
        """A non-stereo bicyclo[2.2.1]heptane must not spuriously gain a
        stereo prefix (regression guard for the retained-name gate)."""
        name = name_smiles("C1CC2CCC1C2")
        assert "bicyclo" in name or "norborn" in name
        assert not _STEREO_PREFIX_RE.match(name), (
            f"achiral bicycloheptane spuriously got stereo prefix: {name!r}"
        )


# ---------------------------------------------------------------------------
# Sanity: monocyclic stereo still works (no regression)
# ---------------------------------------------------------------------------


class TestMonocyclicStereoStillWorks:
    def test_r_butan2ol(self):
        name = name_smiles("C[C@H](O)CC")
        assert _has_descriptor_char(name, "RS")

    def test_e_but2ene(self):
        name = name_smiles(r"C/C=C/C")
        assert _has_descriptor_char(name, "EZ")
