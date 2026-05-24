"""
tests/test_bridged_aromatic_ene.py

Tests for the fused-aromatic / bridged hybrid VB naming path in
``iupac_namer/ring_naming/bridged.py`` (commit 9ed2f2a).

The logic under test:

1. ``_detect_vb_unsaturation`` Kekulizes a work-copy of the molecule when the
   ring system contains aromatic atoms so that aromatic bonds surface as
   explicit double bonds and can be emitted as ``-ene`` locants.

2. ``_format_vb_locant`` emits the IUPAC P-23 disambiguated form
   ``"lo(hi)"`` whenever the bond is NOT on the principal numbering edge
   (i.e. ``hi != lo + 1``).  This is required for cross-bridge / wrap-around
   aromatic bonds, otherwise OPSIN reads the locant as a principal-path bond
   and builds a molecule with the wrong valence.

All expected names are OPSIN-round-trip verified (constitutional match).
"""

from __future__ import annotations

import pytest
from rdkit import Chem

from iupac_namer.engine import name_smiles

try:
    from py2opsin import py2opsin
    HAVE_OPSIN = True
except ImportError:
    HAVE_OPSIN = False


def _constitutional_match(smi_in: str, smi_out: str) -> bool:
    m1 = Chem.MolFromSmiles(smi_in)
    m2 = Chem.MolFromSmiles(smi_out)
    if m1 is None or m2 is None:
        return False
    Chem.RemoveStereochemistry(m1)
    Chem.RemoveStereochemistry(m2)
    return Chem.MolToSmiles(m1) == Chem.MolToSmiles(m2)


def _opsin_roundtrip(smi: str, name: str) -> bool:
    assert HAVE_OPSIN, "py2opsin not available"
    parsed = py2opsin([name], output_format="SMILES")
    if not parsed or not parsed[0]:
        return False
    return _constitutional_match(smi, parsed[0])


class TestFusedAromaticBridgedVB:
    """Fused-aromatic + saturated-bridge hybrid systems named via VB."""

    def test_tetralin_with_ethano_bridge(self):
        """Benzene ortho-fused to a cyclohexane that carries an ethano bridge.

        Exercises the mixed aromatic/saturated path: the aromatic ring is
        Kekulized inside ``_detect_vb_unsaturation`` so three of its bonds
        come out as cited double bonds.
        """
        smi = "C1CC2CCC1c1ccccc12"
        name = name_smiles(smi)
        assert name == "tricyclo[6.2.2.0^{2,7}]dodeca-2,4,6-triene"
        if HAVE_OPSIN:
            assert _opsin_roundtrip(smi, name), f"round-trip failed: {name}"

    def test_benzene_fused_to_bridged_bicyclodecane(self):
        """Anthracene topology: three linearly-fused 6-rings with the central
        ring partly saturated.  Stage 12 R12-A-2 enabled the
        ``_try_derive_hydro_retained`` path on anthracene by adding atom_locants
        to its curated retained-name entry, so this scaffold now resolves to
        ``1,2,3,4,4a,9,9a,10-octahydroanthracene`` — a retained-parent +
        hydro-derivation form that beats the VB tricyclo fallback per IUPAC
        P-31.1.4 (partly-hydrogenated retained parent preferred over VB when
        both round-trip).  The name was previously
        ``tricyclo[8.4.0.0^{3,8}]tetradeca-3,5,7-triene``; both names canon-
        match the input via OPSIN, but the retained-derived form is the PIN."""
        smi = "C12CC3=CC=CC=C3CC1CCCC2"
        name = name_smiles(smi)
        assert name == "1,2,3,4,4a,9,9a,10-octahydroanthracene"
        if HAVE_OPSIN:
            assert _opsin_roundtrip(smi, name), f"round-trip failed: {name}"

    def test_cross_bridge_ene_disambiguation(self):
        """Bicyclic aromatic-bridged system where at least one aromatic
        double bond spans the cross-bridge edge.  The IUPAC P-23 form
        ``lo(hi)`` must appear in the name (e.g. ``1(14)``), otherwise OPSIN
        misreads the locant and produces a 5-valent carbon."""
        smi = "C1=CC2=CC=CC=C2C3CCCCC13"
        name = name_smiles(smi)
        # Expected: cross-bridge locant "1(14)" present
        assert "1(14)" in name, f"expected cross-bridge locant '1(14)' in {name}"
        assert name == "tricyclo[8.4.0.0^{4,9}]tetradeca-1(14),2,10,12-tetraene"
        if HAVE_OPSIN:
            assert _opsin_roundtrip(smi, name), f"round-trip failed: {name}"

    def test_cross_bridge_pentaene(self):
        """Fused-aromatic bridged system with a pentaene numbering that
        requires a ``1(13)`` cross-bridge locant."""
        smi = "C1CCC2=CC=CC3=CC=CC1=C23"
        name = name_smiles(smi)
        assert "1(13)" in name, f"expected cross-bridge locant in {name}"
        assert name == "tricyclo[7.3.1.0^{5,13}]trideca-1(13),5,7,9,11-pentaene"
        if HAVE_OPSIN:
            assert _opsin_roundtrip(smi, name), f"round-trip failed: {name}"

    def test_principal_edge_only_no_disambiguation(self):
        """Control case: a fused-aromatic bridged system where every aromatic
        bond happens to lie on the principal numbering edge.  The P-23
        ``(hi)`` suffix must NOT appear.
        """
        smi = "C1CC2C3=CC=CC=C3CC12"
        name = name_smiles(smi)
        # Principal-path enes only: 2,4,6-triene
        assert name == "tricyclo[7.2.0.0^{2,7}]undeca-2,4,6-triene"
        assert "(" not in name.split("-")[-2], (
            f"principal-path enes should not carry (hi) disambiguation: {name}"
        )
        if HAVE_OPSIN:
            assert _opsin_roundtrip(smi, name), f"round-trip failed: {name}"


@pytest.mark.skipif(not HAVE_OPSIN, reason="py2opsin not installed")
class TestOpsinVerifiedBridgedAromatic:
    """OPSIN-round-trip batch check for the fused-aromatic bridged VB path."""

    @pytest.mark.parametrize("smi,expected_name", [
        (
            "C1CC2CCC1c1ccccc12",
            "tricyclo[6.2.2.0^{2,7}]dodeca-2,4,6-triene",
        ),
        (
            # Anthracene topology — Stage 12 R12-A-2 promoted to the
            # retained-parent + hydro-derivation form via the new anthracene
            # atom_locants entry.  Was tricyclo[8.4.0.0^{3,8}]tetradeca-3,5,7-triene
            # before R12-A-2; OPSIN round-trips both forms to the same
            # canonical SMILES, the retained-derived form is the PIN.
            "C12CC3=CC=CC=C3CC1CCCC2",
            "1,2,3,4,4a,9,9a,10-octahydroanthracene",
        ),
        (
            "C1=CC2=CC=CC=C2C3CCCCC13",
            "tricyclo[8.4.0.0^{4,9}]tetradeca-1(14),2,10,12-tetraene",
        ),
        (
            "C1CCC2=CC=CC3=CC=CC1=C23",
            "tricyclo[7.3.1.0^{5,13}]trideca-1(13),5,7,9,11-pentaene",
        ),
    ])
    def test_round_trip(self, smi, expected_name):
        name = name_smiles(smi)
        assert name == expected_name, f"name mismatch: {name!r} != {expected_name!r}"
        parsed = py2opsin([name], output_format="SMILES")
        assert parsed and parsed[0], f"OPSIN could not parse: {name}"
        assert _constitutional_match(smi, parsed[0]), (
            f"constitutional mismatch: {smi} -> {name} -> {parsed[0]}"
        )
