"""
tests/test_spiro_lambda_valence.py

Tests for the lambda-convention (IUPAC P-14.1.1 / P-31.1.4.3) applied to the
HYPERVALENT spiro / von-Baeyer-polyspiro skeletal heteroatom (P-24).

When a skeletal heteroatom in a spiro a-replacement name carries a non-standard
valence — typically the spiro centre itself, e.g. S(IV)/S(VI), P(V)/P(VII) —
the locant must be cited inline as ``<loc>lambda<val>`` inside the
a-replacement prefix (``5lambda6-thia...``).  Without it an S(VI) spiro centre
is indistinguishable from S(IV) in ``5-thiaspiro[4.4]nona-...`` and the name
fails to round-trip.

The logic under test lives in:
  - ``iupac_namer/ring_naming/monocyclic.py`` (``compute_lambda_value_map``)
  - ``iupac_namer/ring_naming/spiro.py`` (``name_spiro`` / ``_build_spiro_heteroatom_prefix``)
  - ``iupac_namer/ring_naming/polyspiro_vb.py`` (``name_polyspiro_vb`` / ``_build_replacement_prefix``)

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


def _canon(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    Chem.RemoveStereochemistry(m)
    return Chem.MolToSmiles(m)


def _opsin_roundtrip(smi: str, name: str) -> bool:
    assert HAVE_OPSIN, "py2opsin not available"
    parsed = py2opsin([name], output_format="SMILES")
    if not parsed or not parsed[0]:
        return False
    return _canon(smi) == _canon(parsed[0])


# (smiles, expected_name) — every case OPSIN-round-trip verified.
MONOSPIRO_CASES = [
    # Unsaturated 5+5 spiro, hypervalent spiro centre.
    ("S12(C=CC=C1)C=CC=C2", "5lambda4-thiaspiro[4.4]nona-1,3,6,8-tetraene"),
    ("[SH2]12(C=CC=C1)C=CC=C2", "5lambda6-thiaspiro[4.4]nona-1,3,6,8-tetraene"),
    ("P12(C=CC=C1)C=CC=C2", "5lambda5-phosphaspiro[4.4]nona-1,3,6,8-tetraene"),
    ("[SeH2]12(C=CC=C1)C=CC=C2", "5lambda6-selenaspiro[4.4]nona-1,3,6,8-tetraene"),
    # Saturated 5+5 spiro.
    ("S1(CCCC1)2CCCC2", "5lambda4-thiaspiro[4.4]nonane"),
    ("[SH2]1(CCCC1)2CCCC2", "5lambda6-thiaspiro[4.4]nonane"),
    ("P1(CCCC1)2CCCC2", "5lambda5-phosphaspiro[4.4]nonane"),
    # Saturated 6+6 spiro.
    ("S1(CCCCC1)2CCCCC2", "6lambda4-thiaspiro[5.5]undecane"),
    ("[SH2]1(CCCCC1)2CCCCC2", "6lambda6-thiaspiro[5.5]undecane"),
    ("P1(CCCCC1)2CCCCC2", "6lambda5-phosphaspiro[5.5]undecane"),
]

# A spiro centre at standard valence must NOT receive a lambda marker.
STANDARD_VALENCE_CONTROL = [
    ("[Si]12(C=CC=C1)C=CC=C2", "5-silaspiro[4.4]nona-1,3,6,8-tetraene"),
]

# von-Baeyer polyspiro (dispiro chain) with a hypervalent spiro atom.
POLYSPIRO_CASES = [
    ("C1CC2(C1)CC1(CCC1)[SH2]2", "5lambda4-thiadispiro[3.1.3^{6}.1^{4}]decane"),
    ("C1CC2(C1)CC1(CC[SH2]1)C2", "1lambda4-thiadispiro[3.1.3^{6}.1^{4}]decane"),
    ("C1CC2(C1)CC1(CC[SH4]1)C2", "1lambda6-thiadispiro[3.1.3^{6}.1^{4}]decane"),
    ("C1CCC2(C1)CC1(CCCC1)[PH3]2",
     "6lambda5-phosphadispiro[4.1.4^{7}.1^{5}]dodecane"),
]


class TestMonospiroLambda:
    @pytest.mark.parametrize("smi,expected", MONOSPIRO_CASES)
    def test_name(self, smi, expected):
        assert name_smiles(smi) == expected

    @pytest.mark.parametrize("smi,expected", MONOSPIRO_CASES)
    def test_has_lambda(self, smi, expected):
        name = name_smiles(smi)
        assert "lambda" in name, f"expected lambda marker in {name!r}"


class TestStandardValenceControl:
    """A spiro centre at its standard valence (Si is tetravalent) must emit no
    lambda marker — guarding against over-emission."""

    @pytest.mark.parametrize("smi,expected", STANDARD_VALENCE_CONTROL)
    def test_no_lambda(self, smi, expected):
        name = name_smiles(smi)
        assert name == expected
        assert "lambda" not in name, f"unexpected lambda marker in {name!r}"


class TestPolyspiroLambda:
    @pytest.mark.parametrize("smi,expected", POLYSPIRO_CASES)
    def test_name(self, smi, expected):
        assert name_smiles(smi) == expected


@pytest.mark.skipif(not HAVE_OPSIN, reason="py2opsin not installed")
class TestOpsinRoundTrip:
    @pytest.mark.parametrize(
        "smi,expected",
        MONOSPIRO_CASES + STANDARD_VALENCE_CONTROL + POLYSPIRO_CASES,
    )
    def test_round_trip(self, smi, expected):
        name = name_smiles(smi)
        assert name == expected, f"name mismatch: {name!r} != {expected!r}"
        assert _opsin_roundtrip(smi, name), f"round-trip failed: {name}"
