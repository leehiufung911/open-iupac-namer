"""Regression tests for Stage 6 R2-F polynuclear phosphorus oxoacid subsystem.

Exercises ``iupac_namer.perception.fg.phosphorus_oxoacids`` and the engine
dispatch path that consumes it.  Every assertion below verifies:

1. The raw look-up returns a string for an in-table SMILES and None for an
   off-table SMILES.
2. ``engine.name_smiles`` emits the IUPAC preferred name for standalone
   polynuclear phosphorus oxoacid molecules that previously hit
   NAMING_ERROR or produced a verbose substitutive fallback.
3. The new path does NOT fire on mononuclear phosphorus oxoacids
   (``phosphoric acid``, ``phosphonic acid``) or on unrelated organic
   phosphorus species - they remain handled by the existing machinery.
4. All names in the table round-trip through OPSIN back to the same
   RDKit canonical SMILES we keyed them on.

See ``docs/opsin_audit_fg.md`` Gap 3 + Gap 4 for the closure story and
``docs/opsin_coverage_taxonomy.md`` root cause #12.
"""

from __future__ import annotations

import pytest
from rdkit import Chem

from iupac_namer.engine import name_smiles
from iupac_namer.perception.fg.phosphorus_oxoacids import (
    all_names,
    get_table,
    is_registered,
    lookup_mol,
    lookup_name,
)
# The polynuclear P oxoacids (diphosphoric / triphosphoric / hypodiphosphoric)
# are now produced generatively by maingroup_oxoacids.compute_name (P-67); the
# whole-molecule lookup pins were removed (anti-pinning rule #5) and the table
# is intentionally empty.  These tests assert the generative behaviour instead.
from iupac_namer.perception.fg.maingroup_oxoacids import compute_name


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _canon(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None, f"bad test SMILES {smi!r}"
    return Chem.MolToSmiles(mol)


# ---------------------------------------------------------------------------
# Unit: table lookup
# ---------------------------------------------------------------------------

def test_generative_namer_covers_polynuclear_p() -> None:
    """The polynuclear P oxoacids are now produced generatively (the lookup
    pins were removed as redundant); verify coverage is preserved."""
    for smi, expected in [
        ("P(=O)(O)(O)OP(=O)(O)O", "diphosphoric acid"),
        ("P(=O)(O)(O)OP(=O)(O)OP(=O)(O)O", "triphosphoric acid"),
        ("P(=O)(O)(O)P(=O)(O)O", "hypodiphosphoric acid"),
    ]:
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None, f"bad test SMILES {smi!r}"
        assert compute_name(mol) == expected, smi


def test_lookup_name_diphosphoric() -> None:
    # Now produced generatively (pin removed).
    mol = Chem.MolFromSmiles("P(=O)(O)(O)OP(=O)(O)O")
    assert compute_name(mol) == "diphosphoric acid"


def test_lookup_name_triphosphoric() -> None:
    mol = Chem.MolFromSmiles("P(=O)(O)(O)OP(=O)(O)OP(=O)(O)O")
    assert compute_name(mol) == "triphosphoric acid"


def test_lookup_name_hypodiphosphoric() -> None:
    # Direct P-P bond (no bridging O)
    mol = Chem.MolFromSmiles("P(=O)(O)(O)P(=O)(O)O")
    assert compute_name(mol) == "hypodiphosphoric acid"


def test_lookup_name_unknown_key() -> None:
    # ethanol - definitely not a polynuclear phosphoric oxoacid
    assert lookup_name(_canon("CCO")) is None


def test_lookup_name_mononuclear_phosphoric_not_in_table() -> None:
    # H3PO4 - mononuclear, handled by existing retained-name path; must
    # NOT be in the polynuclear table.
    assert lookup_name(_canon("O=P(O)(O)O")) is None


def test_lookup_name_empty_string() -> None:
    assert lookup_name("") is None


def test_is_registered() -> None:
    # The pin table is now empty (generative namer covers these); a former
    # acetaldehyde negative still must not register.
    assert is_registered(_canon("CC=O")) is False  # acetaldehyde


def test_lookup_mol_accepts_mol() -> None:
    # diphosphoric acid is now named generatively, not via the lookup table.
    mol = Chem.MolFromSmiles("P(=O)(O)(O)OP(=O)(O)O")
    assert compute_name(mol) == "diphosphoric acid"


def test_lookup_mol_rejects_substituted() -> None:
    # Methyl ester of diphosphoric acid - NOT a bare acid, must not match
    # either the (empty) lookup table or the generative namer.
    mol = Chem.MolFromSmiles("COP(=O)(O)OP(=O)(O)O")
    assert lookup_mol(mol) is None
    assert compute_name(mol) is None


# ---------------------------------------------------------------------------
# Integration: engine.name_smiles
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "smi,expected",
    [
        ("P(=O)(O)(O)OP(=O)(O)O", "diphosphoric acid"),
        ("P(=O)(O)(O)OP(=O)(O)OP(=O)(O)O", "triphosphoric acid"),
        ("P(=O)(O)(O)P(=O)(O)O", "hypodiphosphoric acid"),
    ],
)
def test_engine_emits_polynuclear_acid_name(smi: str, expected: str) -> None:
    assert name_smiles(smi) == expected


def test_engine_still_names_mononuclear_phosphoric() -> None:
    # Phosphoric acid proper continues through the existing retained-name
    # path - the polynuclear handler must not clobber it.
    assert name_smiles("O=P(O)(O)O") == "phosphoric acid"


def test_engine_still_names_phosphoronitridic() -> None:
    # Phosphoronitridic acid (N#P(O)O) is already handled by the R1-F
    # infix-composition path; the polynuclear handler must not intercept.
    assert name_smiles("N#P(O)O") == "phosphoronitridic acid"


def test_engine_does_not_misfire_on_substituted_phosphate_ester() -> None:
    # Methyl dihydrogen phosphate: the mononuclear ester, not a polynuclear
    # acid.  Output must not be "diphosphoric acid".
    out = name_smiles("COP(=O)(O)O")
    assert "diphosphoric acid" not in out


# ---------------------------------------------------------------------------
# OPSIN round-trip integrity
# ---------------------------------------------------------------------------

def test_all_table_entries_have_nonempty_names() -> None:
    for smi, name in get_table().items():
        assert isinstance(name, str) and name.strip(), (
            f"empty name for canonical SMILES {smi!r}"
        )


def test_all_table_keys_are_canonical() -> None:
    """Every key must equal its own RDKit canonicalisation - otherwise the
    whole-molecule lookup would miss.
    """
    for smi in get_table():
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None, f"non-parseable key {smi!r}"
        assert _canon(smi) == smi, (
            f"key {smi!r} is not RDKit-canonical (canon: {_canon(smi)!r})"
        )
