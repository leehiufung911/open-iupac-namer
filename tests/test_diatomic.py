"""Regression tests for Stage 8 R8-A homonuclear-diatomic dispatch.

Pins the behaviour of ``iupac_namer.engine._name_diatomic_homonuclear``
and the ``name_smiles`` short-circuit that consumes it.  Mirrors the
elementary-atom test layout: every probe is checked end-to-end via

1. the engine emits the surface name we expect;
2. the emitted name round-trips through OPSIN (``py2opsin``) to the same
   canonical SMILES as the input.

The OPSIN round-trip is the critical guarantee — it confirms that each
diatomic name in the table is genuinely the IUPAC form OPSIN recognises,
not just a string that happens to look right.

Negative tests pin the abstain behaviour: charged / isotope / hetero /
multi-atom / wrong-bond-type forms must still flow through the regular
pipeline (so e.g. ``OO`` keeps producing "hydrogen peroxide" and
``[2H][2H]`` stays in the isotope pipeline).
"""

from __future__ import annotations

import os
import shutil
import tempfile

import pytest
from rdkit import Chem

from iupac_namer.engine import (
    _DIATOMIC_HOMONUCLEAR_NAMES,
    _name_diatomic_homonuclear,
    name_smiles,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canon(smiles: str | None) -> str | None:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def _roundtrip(name: str) -> str | None:
    """Run OPSIN over ``name`` in a per-call tempdir and return canonical SMILES.

    The per-call tempdir is needed because py2opsin writes log files into
    cwd; running multiple cases from a shared cwd risks file-handle
    collisions on Windows.
    """
    from py2opsin import py2opsin

    td = tempfile.mkdtemp(prefix="diatom_")
    cwd = os.getcwd()
    try:
        os.chdir(td)
        smi = py2opsin(name)
    finally:
        os.chdir(cwd)
        try:
            shutil.rmtree(td)
        except Exception:
            pass
    return _canon(smi) if smi else None


# ---------------------------------------------------------------------------
# Hook-level positive tests — name + OPSIN round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("smi,expected_name", [
    # Halogen X-X (single bond, P-12.7 / Group 17 dihalogens):
    ("FF",     "difluorine"),
    ("ClCl",   "dichlorine"),
    ("BrBr",   "dibromine"),
    ("II",     "diiodine"),
    # Multi-bonded simple diatomics:
    ("O=O",    "dioxygen"),
    ("N#N",    "dinitrogen"),
    # Molecular hydrogen:
    ("[H][H]", "dihydrogen"),
])
def test_diatomic_engine_emits_expected_name(
    smi: str, expected_name: str,
) -> None:
    """Engine surface name for each canonical diatomic input."""
    assert name_smiles(smi) == expected_name


@pytest.mark.parametrize("smi,expected_name", [
    ("FF",     "difluorine"),
    ("ClCl",   "dichlorine"),
    ("BrBr",   "dibromine"),
    ("II",     "diiodine"),
    ("O=O",    "dioxygen"),
    ("N#N",    "dinitrogen"),
    ("[H][H]", "dihydrogen"),
])
def test_diatomic_opsin_roundtrip(
    smi: str, expected_name: str,
) -> None:
    """The emitted name must parse back to the same canonical SMILES."""
    expected_canon = _canon(smi)
    opsin_canon = _roundtrip(expected_name)
    assert opsin_canon is not None, (
        f"OPSIN failed to parse {expected_name!r}"
    )
    assert opsin_canon == expected_canon, (
        f"Round-trip mismatch for {expected_name!r}: "
        f"expected {expected_canon!r}, got {opsin_canon!r}"
    )


# ---------------------------------------------------------------------------
# Hook-level negative tests — must not claim
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("smi", [
    # Wrong bond type — OO is hydrogen peroxide HO-OH, NOT dioxygen.
    "OO",
    # Charged species belong to charge perception / curated inorganic.
    "[Cl-]",
    "[Na+]",
    # Isotopologues belong to the isotope pipeline.
    "[2H][2H]",
    "[3H][3H]",
    "[H][2H]",
    # Heteronuclear pairs are not homonuclear diatomics.
    "BrCl",
    "ClF",
    "NO",
    # Single atoms — the elementary-atom hook handles those.
    "[Na]",
    "[Cl-]",
    "[NH3]",
    "C",
    # Multi-atom molecules.
    "CC",
    "CCO",
    "c1ccccc1",
    # Salts (multi-fragment) must abstain even if both fragments are
    # the same element.
    "[Na+].[Cl-]",
])
def test_diatomic_hook_negative(smi: str) -> None:
    """The diatomic hook must return None for every out-of-scope probe."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        pytest.skip(f"SMILES {smi!r} is not RDKit-parseable")
    assert _name_diatomic_homonuclear(mol) is None


def test_diatomic_hook_handles_none_mol() -> None:
    assert _name_diatomic_homonuclear(None) is None


# ---------------------------------------------------------------------------
# Already-covered regression guards (must not break)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("smi,expected_name", [
    # Hydrogen peroxide must still be named via the regular pipeline —
    # the diatomic hook abstains because the OO bond type is SINGLE,
    # not the DOUBLE that "dioxygen" requires.
    ("OO", "hydrogen peroxide"),
])
def test_existing_diatomic_paths_preserved(
    smi: str, expected_name: str,
) -> None:
    """Probes that already worked must keep working post-dispatch."""
    assert name_smiles(smi) == expected_name


# ---------------------------------------------------------------------------
# Table-shape sanity check
# ---------------------------------------------------------------------------


def test_diatomic_table_keys_well_formed() -> None:
    """Every key is (symbol:str, atom_count:int==2, bond_type:str)."""
    valid_bond_types = {"SINGLE", "DOUBLE", "TRIPLE"}
    for key, name in _DIATOMIC_HOMONUCLEAR_NAMES.items():
        assert isinstance(key, tuple) and len(key) == 3, key
        sym, count, bt = key
        assert isinstance(sym, str) and sym, key
        assert count == 2, key
        assert bt in valid_bond_types, key
        assert isinstance(name, str) and name.startswith("di"), (key, name)
