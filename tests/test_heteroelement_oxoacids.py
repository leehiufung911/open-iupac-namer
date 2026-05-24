"""Regression tests for Stage 6 R1-B heteroelement oxoacid subsystem.

Exercises ``iupac_namer.perception.fg.heteroelement_oxoacids`` and the
engine dispatch path that consumes it.  Every assertion below verifies:

1. The raw look-up returns a string for an in-table SMILES and None for
   an off-table SMILES.
2. ``engine.name_smiles`` emits the IUPAC preferred name for standalone
   acid / anion molecules that previously hit NAMING_ERROR.
3. The new path does NOT fire on substituted derivatives (a Sb-, Cr-, or
   Cl-bearing larger molecule still flows through the ordinary plan
   pipeline) - this protects pre-Stage-6 behaviour from regression.
"""

from __future__ import annotations

import pytest
from rdkit import Chem

from iupac_namer.engine import name_smiles
from iupac_namer.perception.fg.heteroelement_oxoacids import (
    all_names,
    get_table,
    is_registered,
    lookup_mol,
    lookup_name,
)
# The generative main-group oxoacid namer subsumes the chalcogen / pnictogen /
# halogen / boron families that were formerly hard-coded in the lookup table;
# those pins were removed (anti-pinning rule #5).  The lookup table now retains
# only the d-block / irregular acids the generative namer does not compute.
from iupac_namer.perception.fg.maingroup_oxoacids import compute_name


# ---------------------------------------------------------------------------
# Unit: table lookup
# ---------------------------------------------------------------------------

def _canon(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None, f"bad test SMILES {smi!r}"
    return Chem.MolToSmiles(mol)


def test_table_non_empty() -> None:
    """Sanity: JSON loaded and retains the d-block / irregular acids that the
    generative namer does NOT compute (the main-group families were removed as
    pins subsumed by maingroup_oxoacids.compute_name)."""
    table = get_table()
    assert len(table) >= 10, f"table has only {len(table)} entries"
    names = all_names()
    # d-block + irregular acids deliberately KEPT in the lookup table:
    assert "chromic acid" in names
    assert "permanganic acid" in names
    assert "perrhenic acid" in names
    assert "orthotelluric acid" in names


def test_generative_namer_covers_removed_families() -> None:
    """The chalcogen/pnictogen/halogen/boron acids removed from the lookup
    table are now produced generatively (P-67) — verifies the removal did not
    drop coverage."""
    for smi, expected in [
        ("[Te](O)(O)(=O)=O", "telluric acid"),
        ("O=[Se](=O)(O)O", "selenic acid"),
        ("[Sb](O)(O)(O)=O", "stiboric acid"),
        ("OB(O)O", "boric acid"),
        ("[O-][Cl+3]([O-])([O-])O", "perchloric acid"),
    ]:
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None, f"bad test SMILES {smi!r}"
        assert compute_name(mol) == expected, smi


def test_lookup_name_unknown_key() -> None:
    # ethanol - definitely not a heteroelement oxoacid
    assert lookup_name(_canon("CCO")) is None


def test_lookup_name_empty_string() -> None:
    assert lookup_name("") is None


def test_is_registered() -> None:
    assert is_registered(_canon("[Cr](=O)(=O)(O)O")) is True
    assert is_registered(_canon("CC=O")) is False  # acetaldehyde


def test_lookup_mol_accepts_mol() -> None:
    # chromic acid is a d-block acid retained in the lookup table.
    mol = Chem.MolFromSmiles("[Cr](=O)(=O)(O)O")
    assert lookup_mol(mol) == "chromic acid"


def test_lookup_mol_rejects_substituted() -> None:
    # Methyl-substituted arsenic analogue: should NOT match any key
    mol = Chem.MolFromSmiles("C[As](=O)(O)O")  # methylarsonic acid
    assert lookup_mol(mol) is None


# ---------------------------------------------------------------------------
# Unit: whole-molecule engine dispatch
# ---------------------------------------------------------------------------

# (display_name, OPSIN XML SMILES) for each representative target.
_ENGINE_TARGETS: list[tuple[str, str]] = [
    # Pnictogens (Sb)
    ("stiboric acid",      "[Sb](O)(O)(O)=O"),
    ("stibonic acid",      "[SbH](O)(O)=O"),
    ("stibinic acid",      "[SbH2](O)=O"),
    ("stibonous acid",     "[SbH](O)O"),
    ("stibonite",          "[SbH]([O-])[O-]"),
    # Chalcogens
    ("selenic acid",       "[Se](O)(O)(=O)=O"),
    ("selenous acid",      "[Se](=O)(O)O"),
    ("selenate",           "[Se](=O)(=O)([O-])[O-]"),
    ("selenite",           "[Se](=O)([O-])[O-]"),
    ("telluric acid",      "[Te](O)(O)(=O)=O"),
    ("tellurous acid",     "[Te](=O)(O)O"),
    ("orthotelluric acid", "[Te](O)(O)(O)(O)(O)O"),
    # d-block
    ("chromic acid",       "[Cr](=O)(=O)(O)O"),
    ("dichromic acid",     "[Cr](=O)(=O)(O)O[Cr](=O)(=O)O"),
    ("manganic acid",      "[Mn](=O)(=O)(O)O"),
    ("permanganic acid",   "[Mn](=O)(=O)(=O)O"),
    ("technetic acid",     "[Tc](=O)(=O)(O)O"),
    ("pertechnetic acid",  "[Tc](=O)(=O)(=O)O"),
    ("rhenic acid",        "[Re](=O)(=O)(O)O"),
    ("perrhenic acid",     "[Re](=O)(=O)(=O)O"),
    ("perruthenic acid",   "[Ru](=O)(=O)(=O)O"),
    # d-block lower-oxidation '-ous' series (P-67 / IR-8 traditional names;
    # one fewer terminal =O than the matching '-ic' acid above).  SMILES are
    # exactly OPSIN 2.8.0's output for each name.
    ("chromous acid",      "[Cr](=O)(O)O"),
    ("dichromous acid",    "[Cr](=O)(O)O[Cr](=O)O"),
    ("manganous acid",     "[Mn](=O)(O)O"),
    ("permanganous acid",  "[Mn](=O)(=O)O"),
    ("technetous acid",    "[Tc](=O)(O)O"),
    ("pertechnetous acid", "[Tc](=O)(=O)O"),
    ("rhenous acid",       "[Re](=O)(O)O"),
    ("perrhenous acid",    "[Re](=O)(=O)O"),
    ("perruthenous acid",  "[Ru](=O)(=O)O"),
    # Sulfur amido oxo-free retained form (oxo=0; outside the generative
    # amidosulfuric/amidosulfurous tier scheme).
    ("sulfinamous acid",   "S(N)O"),
    # Main-group
    ("boric acid",         "B(O)(O)O"),
    # Halogen oxyacids
    ("hypochlorous acid",  "ClO"),
    ("chlorous acid",      "Cl(=O)O"),
    ("chloric acid",       "Cl(=O)(=O)O"),
    ("perchloric acid",    "Cl(=O)(=O)(=O)O"),
    ("hypobromous acid",   "BrO"),
    ("bromic acid",        "Br(=O)(=O)O"),
    ("perbromic acid",     "Br(=O)(=O)(=O)O"),
    ("hypoiodous acid",    "IO"),
    ("iodic acid",         "I(=O)(=O)O"),
    ("periodic acid",      "I(=O)(=O)(=O)O"),
    # Anions
    ("chlorate",           "Cl(=O)(=O)[O-]"),
    ("perchlorate",        "Cl(=O)(=O)(=O)[O-]"),
    ("hypochlorite",       "Cl[O-]"),
    ("bromate",            "Br(=O)(=O)[O-]"),
    ("iodate",             "I(=O)(=O)[O-]"),
    # P / As round-trip gap fillers
    ("phosphonic acid",    "[PH](=O)(O)O"),
    ("phosphinic acid",    "[PH2](=O)O"),
    ("arsinic acid",       "[AsH2](=O)O"),
]


@pytest.mark.parametrize(("expected", "smi"), _ENGINE_TARGETS)
def test_engine_emits_expected_name(expected: str, smi: str) -> None:
    """The engine short-circuit must emit the exact IUPAC name for every target."""
    got = name_smiles(smi)
    assert got == expected, (
        f"{expected!r}: engine returned {got!r} for input {smi!r}"
    )


# ---------------------------------------------------------------------------
# Unit: regression-protection - substituted derivatives must NOT short-circuit
# ---------------------------------------------------------------------------

def test_substituted_sb_does_not_short_circuit() -> None:
    """Substituted antimony acids (e.g. methylstibonic) must fall through
    the ordinary plan pipeline - they are NOT in our lookup table and
    the canonical SMILES of a methylated compound will not match our key.
    """
    # methylstibonic acid - CAS-style, not in our table
    mol = Chem.MolFromSmiles("C[Sb](=O)(O)O")
    assert mol is not None
    smiles = Chem.MolToSmiles(mol)
    # Bare cross-check: the key is not in the table
    assert lookup_name(smiles) is None


def test_substituted_cr_does_not_short_circuit() -> None:
    """Chromate ester C-Cr(=O)(=O)-O-C: should NOT match 'chromic acid'."""
    mol = Chem.MolFromSmiles("CO[Cr](=O)(=O)OC")
    assert mol is not None
    smiles = Chem.MolToSmiles(mol)
    assert lookup_name(smiles) is None


# ---------------------------------------------------------------------------
# Unit: previously-covered acids MUST NOT regress
# ---------------------------------------------------------------------------

_PREVIOUSLY_COVERED_STANDALONES: list[tuple[str, str]] = [
    # Table entries in data_loader._INORGANIC_CURATED_SMILES that our
    # module must leave alone.
    ("phosphoric acid", "P(=O)(O)(O)O"),
    ("sulfuric acid",   "OS(=O)(=O)O"),
    ("sulfurous acid",  "OS(=O)O"),
    ("nitric acid",     "O[N+](=O)[O-]"),
    ("nitrous acid",    "ON=O"),
    ("water",           "O"),
    ("ammonia",         "N"),
]


@pytest.mark.parametrize(("expected", "smi"), _PREVIOUSLY_COVERED_STANDALONES)
def test_previously_covered_standalones_still_work(expected: str, smi: str) -> None:
    got = name_smiles(smi)
    assert got == expected, (
        f"{expected!r}: engine returned {got!r} for input {smi!r}; "
        "the heteroelement oxoacid short-circuit must not dislodge "
        "existing retained names."
    )
