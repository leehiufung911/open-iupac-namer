"""
tests/test_benzo_fused_bridged.py

Tests for the benzo-fused-bridged-bicyclic naming path in
``iupac_namer/ring_naming/benzo_fused_bridged.py`` (commit 3e67502).

The module detects the pattern

    benzene (aromatic 6-ring) ortho-fused to one macrocycle that is itself
    bridged by a small 1-4 atom methano/ethano/propano/butano bridge

and emits the IUPAC-preferred name

    <hydro-locs>-<count>hydro-<bh1>,<bh2>-<bridge>benzocyclo<N>ene

instead of the generic VB tricyclo[...] form (which collapses aromaticity
and fails OPSIN round-trip for the canonical form).

The path competes with the generic VB path on substituent-locant lowness,
so triggering it requires a substituent on the aromatic ring (to make the
benzofused numbering give lower locants than the VB form).

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


class TestBenzoFusedBridgedBicyclic:
    """Benzo-fused-bridged-bicyclic naming path."""

    def test_methano_benzocyclononene_with_hydroxyl(self):
        """Benzene ortho-fused to a 9-ring bridged by a 1-atom (methano)
        bridge, with a hydroxyl on the aromatic ring pushing the locant sum
        low enough that the benzofused path wins over the VB path.
        """
        smi = "OC1=CC=C2C(=C1)C1CCCCCC2C1"
        name = name_smiles(smi)
        assert "methanobenzocyclononen" in name, (
            f"expected methano-benzocyclononene form in {name!r}"
        )
        assert "heptahydro" in name, f"expected heptahydro in {name!r}"
        assert name == (
            "5,6,7,8,9,10,11-heptahydro-5,11-methanobenzocyclononen-2-ol"
        )
        if HAVE_OPSIN:
            assert _opsin_roundtrip(smi, name), f"round-trip failed: {name}"

    def test_ethano_benzocyclooctene_with_hydroxyl(self):
        """Benzene fused to an 8-ring bridged by a 2-atom (ethano) bridge.
        Exercises the ``ethano`` branch of the bridge-prefix lookup."""
        smi = "Oc1ccc2c(c1)CC1CCCC2CC1"
        name = name_smiles(smi)
        assert "ethanobenzocycloocten" in name, (
            f"expected ethano-benzocyclooctene form in {name!r}"
        )
        assert "hexahydro" in name, f"expected hexahydro in {name!r}"
        assert name == (
            "5,6,7,8,9,10-hexahydro-5,9-ethanobenzocycloocten-2-ol"
        )
        if HAVE_OPSIN:
            assert _opsin_roundtrip(smi, name), f"round-trip failed: {name}"

    def test_bridge_locants_pin_to_benzene_side(self):
        """The benzofused numbering must put the aromatic ring at locants 1-4
        (+ 4a/Na as fusion atoms), so the methano/ethano bridge locants are
        always ``5,N`` where N is the last aliphatic ring atom before the
        bridge.  Verify the pin.
        """
        smi = "OC1=CC=C2C(=C1)C1CCCCCC2C1"
        name = name_smiles(smi)
        # The bridge locants must start with "5," — pin of fusion-adjacent
        # aliphatic ring atom.
        assert "5,11-methano" in name, (
            f"expected '5,11-methano' pin in {name!r}"
        )

    def test_bridge_locants_pin_shorter_ring(self):
        smi = "Oc1ccc2c(c1)CC1CCCC2CC1"
        name = name_smiles(smi)
        assert "5,9-ethano" in name, f"expected '5,9-ethano' pin in {name!r}"

    def test_without_aromatic_substituent_falls_back_to_vb(self):
        """No substituent on the aromatic ring: the VB tricyclo[...] form
        has lower substituent locants, so the engine picks it instead of
        the benzofused form.  This is not a regression — the benzofused
        form is only preferred when it yields lower substituent locants.
        """
        smi = "C1CCCCCC2Cc3ccccc3C1C2"
        name = name_smiles(smi)
        # Should be VB tricyclo form
        assert "tricyclo" in name, (
            f"expected VB tricyclo fallback without aromatic sub, got {name!r}"
        )
        assert "methanobenzocyclo" not in name
        if HAVE_OPSIN:
            assert _opsin_roundtrip(smi, name), f"round-trip failed: {name}"


@pytest.mark.skipif(not HAVE_OPSIN, reason="py2opsin not installed")
class TestOpsinVerifiedBenzoFusedBridged:
    """OPSIN-round-trip batch check for the benzofused-bridged path."""

    @pytest.mark.parametrize("smi,expected_name", [
        (
            "OC1=CC=C2C(=C1)C1CCCCCC2C1",
            "5,6,7,8,9,10,11-heptahydro-5,11-methanobenzocyclononen-2-ol",
        ),
        (
            "Oc1ccc2c(c1)CC1CCCC2CC1",
            "5,6,7,8,9,10-hexahydro-5,9-ethanobenzocycloocten-2-ol",
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
