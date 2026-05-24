"""
tests/test_lambda_valence.py

Tests for the lambda-valence convention for non-standard heteroatom
valences in Hantzsch-Widman rings (commit 625d507).

The logic under test lives in ``iupac_namer/ring_naming/monocyclic.py``
(``try_hantzsch_widman``) and does three things:

1. Looks up the standard valence for each ring heteroatom via
   ``_STANDARD_VALENCE``.  A heteroatom whose actual valence differs gets
   the ``<locant>lambda<valence>-`` prefix per IUPAC P-14.1.1.
2. If the non-standard valence is caused by an ENDOCYCLIC double bond
   (e.g. N=P), the parent stem switches to the maximally unsaturated form
   (e.g. ``triazaphosphinane`` → ``triazaphosphinine``).
3. If the non-standard valence is caused solely by EXOCYCLIC double
   bonds (e.g. P=O, S(=O)(=O)), the saturated parent stem is kept; the
   lambda marker is still emitted.

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


class TestLambdaValenceEndocyclic:
    """Non-standard valence caused by an endocyclic double bond → the
    parent stem switches to the maximally unsaturated (aromatic) form
    and ``lambda<n>`` is emitted.
    """

    def test_cyclotriphosphazene_hexamethyl(self):
        """1,2,3,4,4,6-hexamethyl derivative of hexahydro-1,3,5,2,4,6-
        triazatriphosphinine.  One P carries an endocyclic N=P double
        bond, so its valence is 5 and the ring uses the unsaturated stem
        ``triazatriphosphinine``.
        """
        smi = "CN1P(C)N=P(C)(C)N(C)P1C"
        name = name_smiles(smi)
        assert "4lambda5" in name, f"expected 4lambda5 in {name!r}"
        assert "triazatriphosphinine" in name, (
            f"expected aromatic stem 'triazatriphosphinine' in {name!r}"
        )
        assert name == (
            "1,2,3,4,4,6-hexamethyl-4lambda5-1,3,5,2,4,6-triazatriphosphinine"
        )
        if HAVE_OPSIN:
            assert _opsin_roundtrip(smi, name), f"round-trip failed: {name}"


class TestLambdaValenceExocyclic:
    """Non-standard valence caused solely by exocyclic double bonds →
    the saturated parent stem is kept, but ``lambda<n>`` is still
    emitted so OPSIN knows the heteroatom valence.
    """

    def test_exocyclic_P_oxo_oxazaphosphinane(self):
        """2-oxo-2-(diethylamino)-oxazaphosphinane.  The P is valence 5
        because of the exocyclic P=O (keto); the ring itself stays
        saturated (``-inane``), and ``2lambda5-`` marks the P."""
        smi = "O=P1(N(CC)CC)OCCCN1"
        name = name_smiles(smi)
        assert "2lambda5" in name, f"expected 2lambda5 in {name!r}"
        assert "oxazaphosphinane" in name, (
            f"expected saturated stem 'oxazaphosphinane' in {name!r}"
        )
        assert "2-oxo-" in name, f"expected exocyclic 2-oxo in {name!r}"
        assert name == (
            "2-(diethylamino)-2-oxo-2lambda5-1,3,2-oxazaphosphinane"
        )
        if HAVE_OPSIN:
            assert _opsin_roundtrip(smi, name), f"round-trip failed: {name}"

    def test_exocyclic_S_dioxo_thiazinane(self):
        """3-methyl-1,1-dioxo-thiazinane: S(VI) from two exocyclic S=O
        bonds.  Saturated ``-inane`` stem retained, ``1lambda6-`` marks
        the sulfur.
        """
        smi = "O=S1(=O)CCCN(C)C1"
        name = name_smiles(smi)
        assert "1lambda6" in name, f"expected 1lambda6 in {name!r}"
        assert "thiazinane" in name, (
            f"expected saturated stem 'thiazinane' in {name!r}"
        )
        assert "1,1-dioxo" in name, f"expected 1,1-dioxo in {name!r}"
        assert name == "3-methyl-1,1-dioxo-1lambda6-1,3-thiazinane"
        if HAVE_OPSIN:
            assert _opsin_roundtrip(smi, name), f"round-trip failed: {name}"


class TestLambdaLocantPinning:
    """The lambda locant must match the position of the non-standard
    heteroatom in the ring numbering.  Verify via string inspection that
    the emitted locant is consistent with the parent stem."""

    def test_lambda_locant_matches_parent_stem(self):
        """For ``1,3,2-oxazaphosphinane`` the P is atom 2, so its lambda
        marker must be ``2lambda5``.
        """
        smi = "O=P1(N(CC)CC)OCCCN1"
        name = name_smiles(smi)
        # "1,3,2-oxazaphosphinane" puts P at locant 2 → "2lambda5-"
        assert "-2lambda5-" in name, (
            f"expected '-2lambda5-' locant pin in {name!r}"
        )

    def test_lambda_locant_thiazinane(self):
        """For ``1,3-thiazinane`` the S is atom 1 → ``1lambda6-``."""
        smi = "O=S1(=O)CCCN(C)C1"
        name = name_smiles(smi)
        assert "-1lambda6-" in name, (
            f"expected '-1lambda6-' locant pin in {name!r}"
        )


@pytest.mark.skipif(not HAVE_OPSIN, reason="py2opsin not installed")
class TestOpsinVerifiedLambda:
    """OPSIN-round-trip batch check for lambda-valence ring names."""

    @pytest.mark.parametrize("smi,expected_name", [
        (
            "CN1P(C)N=P(C)(C)N(C)P1C",
            "1,2,3,4,4,6-hexamethyl-4lambda5-1,3,5,2,4,6-triazatriphosphinine",
        ),
        (
            "O=P1(N(CC)CC)OCCCN1",
            "2-(diethylamino)-2-oxo-2lambda5-1,3,2-oxazaphosphinane",
        ),
        (
            "O=S1(=O)CCCN(C)C1",
            "3-methyl-1,1-dioxo-1lambda6-1,3-thiazinane",
        ),
    ])
    def test_round_trip(self, smi, expected_name):
        name = name_smiles(smi)
        assert name == expected_name, (
            f"name mismatch: {name!r} != {expected_name!r}"
        )
        parsed = py2opsin([name], output_format="SMILES")
        assert parsed and parsed[0], f"OPSIN could not parse: {name}"
        assert _constitutional_match(smi, parsed[0]), (
            f"constitutional mismatch: {smi} -> {name} -> {parsed[0]}"
        )
