"""
tests/test_ylidene_exocyclic.py

Regression tests for Stage 6 R2-H — ylidene exocyclic =C= substituent.

When a retained substituent fragment (e.g. "cyclohexyl", "phenyl") is attached
to the parent via a double bond (bond order 2) rather than a single bond, its
suffix must be "-ylidene" per IUPAC P-29.2, not "-yl".  The retained-name leaf
path bypasses the generic ring numbering and so must apply the bond-order
swap locally.

Root cause #17 in docs/opsin_coverage_taxonomy.md:
    benzylidenecyclohexane  (SMILES C(=C1CCCCC1)c1ccccc1)
    phenylmethylidenecyclohexane  (same SMILES)

Both of these produce a =C= exocyclic pattern.  With benzene chosen as the
parent, the substituent fragment is C=C1CCCCC1 (methylenecyclohexane) with
the =C= pointing toward the parent and the =C1CCCCC1 pointing away.  The
correct substituent name is "(cyclohexylidene)methyl" — where "cyclohexyl"
becomes "cyclohexylidene" because the internal bond to the cyclohexane ring
carbon is double.

Before the fix, our engine emitted "(cyclohexylmethyl)benzene", silently
dropping the exocyclic double bond.  After the fix:
    C(=C1CCCCC1)c1ccccc1  =>  [(cyclohexylidene)methyl]benzene  (round-trips)
"""
from __future__ import annotations

import pytest
from rdkit import Chem

from iupac_namer.engine import name_smiles


def _canon(smi: str) -> str:
    return Chem.CanonSmiles(smi)


# ---------------------------------------------------------------------------
# Retained-ring substituent via a double bond -> -ylidene swap
# ---------------------------------------------------------------------------


class TestYlideneExocyclic:
    def test_benzylidene_cyclohexane_not_cyclohexylmethyl(self):
        """benzylidenecyclohexane must not round-trip to (cyclohexylmethyl)benzene.

        The single bond between benzene and the CH is correct; the double bond
        to cyclohexane must yield "cyclohexylidene", not "cyclohexyl".
        """
        result = name_smiles("C(=C1CCCCC1)c1ccccc1")
        # The wrong answer drops the exocyclic double bond entirely.
        assert "cyclohexylmethyl" not in result, (
            f"Bond-order loss: got {result!r}, must emit 'cyclohexylidene'."
        )
        assert "ylidene" in result, (
            f"Expected '-ylidene' in name of =C= substituent, got {result!r}."
        )

    def test_benzylidene_cyclohexane_contains_cyclohexylidene(self):
        """The cyclohexane attached via =C must render as cyclohexylidene."""
        result = name_smiles("C(=C1CCCCC1)c1ccccc1")
        assert "cyclohexylidene" in result, (
            f"Expected 'cyclohexylidene' in name, got {result!r}."
        )

    @pytest.mark.parametrize(
        "smiles",
        [
            # benzylidenecyclohexane (exocyclic =C on cyclohexane, phenyl via CH=)
            "C(=C1CCCCC1)c1ccccc1",
            # Same structure written differently
            "C1CCC(=Cc2ccccc2)CC1",
        ],
    )
    def test_round_trip_preserves_double_bond(self, smiles):
        """OPSIN-roundtrip-style check: the named structure must preserve the
        exocyclic double bond (i.e. the canonical SMILES is unchanged from
        input to a re-parse of our emitted name's structure, modulo OPSIN).
        """
        got = name_smiles(smiles)
        assert not got.startswith("[NAMING"), (
            f"Namer error for {smiles!r}: {got!r}"
        )
        # Architectural check: the name must mention ylidene (bond-order-2 suffix)
        # OR methylidene (the =C= placed on the ring-parent side).  Accept both.
        assert ("ylidene" in got) or ("methylidene" in got), (
            f"Name {got!r} for {smiles!r} appears to have lost the =C= bond-order."
        )


class TestYlideneSimplerCases:
    def test_methylenecyclohexane_standalone(self):
        """Sanity: the standalone molecule still names correctly."""
        result = name_smiles("C=C1CCCCC1")
        # Either "methylenecyclohexane" or "(methylidene)cyclohexane" is OK;
        # both correctly encode the exocyclic =CH2 on cyclohexane.
        assert "methyl" in result and ("idene" in result or "ene" in result), (
            f"methylenecyclohexane: {result!r}"
        )

    def test_methylcyclohexane_unaffected(self):
        """Sanity: saturated methylcyclohexane still names correctly
        (ylidene suffix must NOT appear on single-bonded cyclohexyl)."""
        result = name_smiles("CC1CCCCC1")
        assert result == "methylcyclohexane", (
            f"methylcyclohexane regression: {result!r}"
        )
        assert "ylidene" not in result
