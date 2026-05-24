"""
tests/test_sulfonato_fg.py

Tests for the sulfamate / sulfate-ester-anion functional-group entries
added in commit 24e2ef4.

Two new prefix-only FGs live in ``data/functional_groups.json``:

* ``sulfonatooxy``   — SMARTS ``[OX2;!H][SX4](=O)(=O)[OX1-]``
  matches R–O–SO3⁻ (heparin O-sulfonate ester anion).

* ``sulfonatoamino`` — SMARTS ``[NX3H1][SX4](=O)(=O)[OX1-]``
  matches R–NH–SO3⁻ (heparin N-sulfonate / sulfamate anion).

Both SMARTS are narrow: only the anionic ``[OX1-]`` form matches, not the
neutral sulfonate ester / sulfonamide.  The detector (in
``iupac_namer/perception/fg_detection.py``) is extended with subsumption
rules so that ``sulfonatoamino`` hides the spurious ``sulfonamide`` /
``secondary_amine`` matches.

All expected names are OPSIN-round-trip verified (constitutional match).
"""

from __future__ import annotations

import time

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


def _opsin_call(name: str) -> str | None:
    """Single OPSIN call with one retry — py2opsin uses a fixed temp-file
    name and on Windows can race with leftover Java handles, so we retry
    once after a short delay if the first call returns nothing.
    """
    assert HAVE_OPSIN, "py2opsin not available"
    for attempt in range(2):
        try:
            parsed = py2opsin([name], output_format="SMILES")
            if parsed and parsed[0]:
                return parsed[0]
        except Exception:
            pass
        time.sleep(0.5)
    return None


def _opsin_roundtrip(smi: str, name: str) -> bool:
    out = _opsin_call(name)
    if out is None:
        return False
    return _constitutional_match(smi, out)


class TestSulfonatooxy:
    """R–O–SO3⁻ → ``sulfonatooxy`` prefix."""

    def test_simple_alkyl_sulfate_anion(self):
        smi = "CCCOS(=O)(=O)[O-]"
        name = name_smiles(smi)
        assert "sulfonatooxy" in name, (
            f"expected 'sulfonatooxy' in {name!r}"
        )
        assert name == "1-(sulfonatooxy)propane"
        if HAVE_OPSIN:
            assert _opsin_roundtrip(smi, name), f"round-trip failed: {name}"

    def test_glycerol_3_sulfate_anion(self):
        """Small polyol with a terminal O-sulfate anion — a representative
        heparin-like fragment.
        """
        smi = "OCC(O)COS(=O)(=O)[O-]"
        name = name_smiles(smi)
        assert "sulfonatooxy" in name, f"expected 'sulfonatooxy' in {name!r}"
        assert name == "3-(sulfonatooxy)propane-1,2-diol"
        if HAVE_OPSIN:
            assert _opsin_roundtrip(smi, name), f"round-trip failed: {name}"


class TestSulfonatoamino:
    """R–NH–SO3⁻ → ``sulfonatoamino`` prefix."""

    def test_simple_alkyl_sulfamate_anion(self):
        smi = "CCCNS(=O)(=O)[O-]"
        name = name_smiles(smi)
        assert "sulfonatoamino" in name, (
            f"expected 'sulfonatoamino' in {name!r}"
        )
        assert name == "1-(sulfonatoamino)propane"
        if HAVE_OPSIN:
            assert _opsin_roundtrip(smi, name), f"round-trip failed: {name}"

    def test_aryl_sulfamate_anion(self):
        """Aniline N-sulfate anion — the sulfonatoamino prefix must
        subsume both ``secondary_amine`` and ``sulfonamide`` so neither
        of those alternate prefixes sneaks into the name."""
        smi = "c1ccccc1NS(=O)(=O)[O-]"
        name = name_smiles(smi)
        assert "sulfonatoamino" in name, (
            f"expected 'sulfonatoamino' in {name!r}"
        )
        assert "sulfamoyl" not in name, (
            f"sulfamoyl must be subsumed by sulfonatoamino: {name!r}"
        )
        assert "amino" not in name.replace("sulfonatoamino", ""), (
            f"free amino prefix must be subsumed: {name!r}"
        )
        if HAVE_OPSIN:
            assert _opsin_roundtrip(smi, name), f"round-trip failed: {name}"


class TestSMARTSNarrowness:
    """The new SMARTS must only match the anionic form.  Neutral
    sulfate esters and neutral sulfonamides should continue to be named
    the old way (e.g. ``hydroxysulfonyloxy``, ``sulfonamide``)."""

    def test_neutral_sulfate_ester_not_matched(self):
        """Neutral ROSO3H must NOT get the sulfonatooxy prefix."""
        smi = "CCCOS(=O)(=O)O"
        name = name_smiles(smi)
        assert "sulfonatooxy" not in name, (
            f"neutral sulfate ester should not match sulfonatooxy: {name!r}"
        )

    def test_neutral_sulfonamide_not_matched(self):
        """Neutral RNHSO2R' must NOT get the sulfonatoamino prefix —
        it should remain a sulfonamide."""
        smi = "CCCNS(=O)(=O)C"
        name = name_smiles(smi)
        assert "sulfonatoamino" not in name, (
            f"neutral sulfonamide should not match sulfonatoamino: {name!r}"
        )
        assert "sulfonamide" in name, (
            f"expected 'sulfonamide' suffix in {name!r}"
        )


@pytest.mark.skipif(not HAVE_OPSIN, reason="py2opsin not installed")
class TestOpsinVerifiedSulfonato:
    """OPSIN-round-trip batch check for sulfonato prefixes."""

    @pytest.mark.parametrize("smi,expected_name", [
        ("CCCOS(=O)(=O)[O-]",     "1-(sulfonatooxy)propane"),
        ("OCC(O)COS(=O)(=O)[O-]", "3-(sulfonatooxy)propane-1,2-diol"),
        ("CCCNS(=O)(=O)[O-]",     "1-(sulfonatoamino)propane"),
        ("c1ccccc1NS(=O)(=O)[O-]","(sulfonatoamino)benzene"),
    ])
    def test_round_trip(self, smi, expected_name):
        name = name_smiles(smi)
        assert name == expected_name, (
            f"name mismatch: {name!r} != {expected_name!r}"
        )
        out = _opsin_call(name)
        assert out, f"OPSIN could not parse: {name}"
        assert _constitutional_match(smi, out), (
            f"constitutional mismatch: {smi} -> {name} -> {out}"
        )
