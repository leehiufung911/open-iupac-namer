"""
tests/test_polyspiro_articulation.py

Tests for articulation-split polyspiro naming (P-24.5).

Covers the case where perception classifies a ring system as ``spiro`` with
≥3 rings and no single atom shared by all rings.  ``name_spiro`` detects an
articulation atom, splits into two partner sub-mols, names each recursively,
and composes ``spiro[<smaller>-N,N'-<larger>]``.

All expected names are OPSIN-round-trip verified.
"""

from __future__ import annotations

import os

import pytest
from rdkit import Chem

from iupac_namer.engine import name_smiles


try:
    from py2opsin import py2opsin
    HAVE_OPSIN = True
except ImportError:
    HAVE_OPSIN = False


def _constitutional_match(smi_in: str, smi_out: str) -> bool:
    """Compare two SMILES ignoring stereochemistry."""
    m1 = Chem.MolFromSmiles(smi_in)
    m2 = Chem.MolFromSmiles(smi_out)
    if m1 is None or m2 is None:
        return False
    Chem.RemoveStereochemistry(m1)
    Chem.RemoveStereochemistry(m2)
    return Chem.MolToSmiles(m1) == Chem.MolToSmiles(m2)


def _opsin_roundtrip(smi: str, name: str) -> bool:
    """Check the generated IUPAC name re-parses to the same molecule."""
    assert HAVE_OPSIN, "py2opsin not available"
    parsed = py2opsin([name], output_format="SMILES")
    if not parsed or not parsed[0]:
        return False
    return _constitutional_match(smi, parsed[0])


class TestArticulationSplitPolyspiro:
    """Polyspiro (≥3 rings) articulation-split naming."""

    def test_decalin_spiro_dioxolane(self):
        """spiro[[1,3]dioxolane-2,2'-decalin] — simplest polyspiro."""
        smi = "C1CCC2CC3(CCC2C1)OCCO3"
        name = name_smiles(smi)
        assert "spiro" in name
        assert "dioxolane" in name
        assert "decalin" in name or "naphthalen" in name
        if HAVE_OPSIN:
            assert _opsin_roundtrip(smi, name), f"round-trip failed: {name}"

    def test_decalin_spiro_dioxolane_with_methyl_substituent(self):
        """Substituent on the decalin side of a polyspiro."""
        smi = "CC1CCC2CC3(CCC2C1)OCCO3"
        name = name_smiles(smi)
        assert "spiro" in name
        assert "methyl" in name
        if HAVE_OPSIN:
            assert _opsin_roundtrip(smi, name), f"round-trip failed: {name}"

    def test_fused_aromatic_spiro_dioxolane(self):
        """Tetrahydronaphthalene (partially aromatic fused) spiro to a
        1,3-dioxolane.  Exercises the fused-sub-mol recursion path.
        """
        smi = "O1CC2(OC1)CC3=CC=CC=C3CC2"
        name = name_smiles(smi)
        assert "spiro" in name
        if HAVE_OPSIN:
            assert _opsin_roundtrip(smi, name), f"round-trip failed: {name}"

    def test_nested_spiro_cyclohexyl_spiro_dioxolane(self):
        """Four-ring system: a spiro[4.5]decane (itself binary-spiro) joined
        by another spiro junction to a 1,3-dioxolane.  Two articulation atoms
        in the ring-graph — our scanner picks the one that yields a valid
        split.
        """
        smi = "C1CC2(CCC1)CC1(CC2)OCCO1"
        name = name_smiles(smi)
        assert "spiro" in name
        if HAVE_OPSIN:
            assert _opsin_roundtrip(smi, name), f"round-trip failed: {name}"

    def test_binary_spiro_still_uses_legacy_path(self):
        """Binary (2-ring) spiro must continue to use the a-replacement
        ``1,3-dioxaspiro[4.5]decane`` form, not the articulation-split form.
        """
        smi = "O1CC2(CCCCC2)OC1"
        name = name_smiles(smi)
        assert name == "1,3-dioxaspiro[4.5]decane"

    def test_binary_spiro_all_carbon(self):
        """Binary all-carbon spiro unaffected by the articulation-split path."""
        smi = "C1CCC2(CC1)CCCCC2"
        name = name_smiles(smi)
        assert name == "spiro[5.5]undecane"

    def test_steroid_kernel_with_acetonide(self):
        """Amcinonide-like kernel: the corticosteroid 4-ring system fused to
        a 1,3-dioxolane, itself spiro to a cyclopentane (6 rings total).

        This is the FDA-0054 topology without the enone / ester groups.
        Currently produces a systematic VB name for the polycyclic partner
        (via fused-to-bridged classification fallback); the IUPAC-preferred
        name would use ``cyclopenta[a]phenanthrene`` + ``hexadecahydro`` +
        dioxolo fusion, but that Stage 3 saturated-fused naming is not yet
        implemented.  We verify the constitutional round-trip, not the name
        form.
        """
        smi = "C[C@]12C[C@H](O)[C@]3(F)C4CCCCC4CC[C@H]3[C@@H]1C[C@H]1OC3(CCCC3)O[C@H]12"
        name = name_smiles(smi)
        # Must not fall through to NAMING ERROR
        assert "NAMING ERROR" not in name, f"failed: {name}"
        assert "spiro" in name
        if HAVE_OPSIN:
            parsed = py2opsin([name], output_format="SMILES")
            assert parsed and parsed[0], f"OPSIN could not parse: {name}"
            assert _constitutional_match(smi, parsed[0]), (
                f"constitutional mismatch: {name} -> {parsed[0]}"
            )


class TestArticulationSplitPolyspiroEdgeCases:
    """Additional edge cases for the articulation-split polyspiro path."""

    def test_triple_spiro_three_carbocyclic_partners(self):
        """Three cyclohexane rings, two spiro junctions, no atom shared by
        all three rings.

        PIN UPDATE (P-24.2.2): when every ring component is monocyclic the
        IUPAC preferred name is the von-Baeyer polyspiro form
        ``dispiro[5.2.5^{9}.2^{6}]hexadecane`` — NOT the nested P-24.5
        component form ``spiro[cyclohexane-N,N'-spiro[5.5]undecane]`` that the
        articulation-split path previously emitted (both round-trip via OPSIN,
        but only the von-Baeyer form is the PIN for all-monocyclic polyspiro).
        """
        smi = "C1CCC2(CC1)CCC1(CC2)CCCCC1"
        name = name_smiles(smi)
        assert name == "dispiro[5.2.5^{9}.2^{6}]hexadecane", (
            f"expected von-Baeyer polyspiro PIN, got {name!r}"
        )
        if HAVE_OPSIN:
            assert _opsin_roundtrip(smi, name), f"round-trip failed: {name}"

    def test_spiro_between_two_hetero_fused_partners(self):
        """Spiro junction between two oxygen-containing monocyclic rings.

        PIN UPDATE (P-24.2.4): an all-monocyclic heterocyclic polyspiro is the
        von-Baeyer skeletal-replacement form (``...trioxadispiro[...]``), not
        the nested P-24.5 component form.  The round-trip must still hold."""
        smi = "O1CC2(OC1)CC1(CCCO1)CC2"
        name = name_smiles(smi)
        assert "dispiro" in name, f"expected von-Baeyer dispiro in {name!r}"
        # All three ring oxygens expressed via a-replacement.
        assert "trioxa" in name, f"expected trioxa replacement in {name!r}"
        if HAVE_OPSIN:
            assert _opsin_roundtrip(smi, name), f"round-trip failed: {name}"

    def test_spiro_off_a_spiro_partner(self):
        """A dioxaspiro ring joined by another spiro junction to a cyclohexane.

        PIN UPDATE (P-24.2.4): all three rings are monocyclic, so the PIN is
        the von-Baeyer ``1,4-dioxadispiro[4.1.5^{7}.3^{5}]pentadecane`` form,
        not the nested ``spiro[1,4-dioxaspiro[4.5]decane-...]`` component form.
        """
        smi = "C1CCC2(CC1)CC1(CCC2)OCCO1"
        name = name_smiles(smi)
        assert "dispiro" in name, f"expected von-Baeyer dispiro in {name!r}"
        assert "dioxa" in name, f"expected dioxa replacement in {name!r}"
        if HAVE_OPSIN:
            assert _opsin_roundtrip(smi, name), f"round-trip failed: {name}"


@pytest.mark.skipif(not HAVE_OPSIN, reason="py2opsin not installed")
class TestOpsinVerified:
    """OPSIN-verified polyspiro cases — canonical target names."""

    @pytest.mark.parametrize("smi,expected_substrings", [
        ("C1CCC2CC3(CCC2C1)OCCO3", ["spiro", "dioxolane"]),  # decalin+dioxolane
        ("O1CC2(CCCCC2)OC1", ["1,3-dioxaspiro", "decane"]),  # binary heterospiro (legacy)
    ])
    def test_named_forms(self, smi, expected_substrings):
        name = name_smiles(smi)
        for sub in expected_substrings:
            assert sub in name, f"expected '{sub}' in name '{name}' for {smi}"
        # Round-trip must hold.
        parsed = py2opsin([name], output_format="SMILES")
        assert parsed and parsed[0], f"OPSIN unparseable: {name}"
        assert _constitutional_match(smi, parsed[0]), (
            f"round-trip constitutional mismatch: {smi} -> {name} -> {parsed[0]}"
        )
