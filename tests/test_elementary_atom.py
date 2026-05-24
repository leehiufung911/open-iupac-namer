"""Regression tests for Stage 6 R2-C bare elementary atom dispatch.

Pins the behaviour of ``iupac_namer.engine._name_elementary_atom`` and the
``name_smiles`` short-circuit that consumes it.  Every probe in this suite
came from one of the audit CSVs (``eval/opsin_audit_misc_raw.csv`` /
``eval/opsin_audit_hw_charge_raw.csv``); each one is checked end-to-end:

1. the engine emits the surface name we expect;
2. the emitted name round-trips through OPSIN (``py2opsin``) to the same
   canonical SMILES as the input.

The OPSIN round-trip is the critical guarantee — if a parent-hydride name
("azane", "oxidane", …) drifts vs. the historical retained name
("ammonia", "water"), the SMILES round-trip will still hold and the test
will pass.

Negative tests (out-of-scope inputs) confirm that the hook stays silent for
charged species, multi-atom molecules, isotopologues, and bare/molecular
hydrogen.
"""

from __future__ import annotations

import pytest
from rdkit import Chem

from iupac_namer.engine import (
    _ELEMENTARY_ATOM_NAMES,
    _name_elementary_atom,
    name_smiles,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canon(smiles: str) -> str | None:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def _roundtrip(name: str) -> str | None:
    """Run OPSIN over ``name`` and return the canonical SMILES it emits."""
    from py2opsin import py2opsin

    smi = py2opsin(name)
    return _canon(smi) if smi else None


# ---------------------------------------------------------------------------
# Hook-level positive tests (audit-derived)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("smi,expected_name", [
    # --- Group 1 alkali metals ---
    ("[Li]", "lithium"),
    ("[Na]", "sodium"),
    ("[K]",  "potassium"),
    ("[Rb]", "rubidium"),
    ("[Cs]", "caesium"),
    ("[Fr]", "francium"),
    # --- Group 2 alkaline earths ---
    ("[Be]", "beryllium"),
    ("[Mg]", "magnesium"),
    ("[Ca]", "calcium"),
    ("[Sr]", "strontium"),
    ("[Ba]", "barium"),
    ("[Ra]", "radium"),
    # --- Group 13 (bare + parent hydrides per P-21.1.1 Table 2.1) ---
    ("[B]",  "boron"),
    ("[Ga]", "gallium"),
    ("[In]", "indium"),
    ("[Tl]", "thallium"),
    ("[AlH3]", "alumane"),
    ("[GaH3]", "gallane"),
    ("[InH3]", "indigane"),
    ("[TlH3]", "thallane"),
    # --- Group 14 (bare) ---
    ("[C]",  "carbon"),
    ("[Si]", "silicon"),
    ("[Ge]", "germanium"),
    # --- Group 15 (bare + heavy parent hydrides) ---
    ("[N]",    "nitrogen"),
    ("[P]",    "phosphorus"),
    ("[As]",   "arsenic"),
    ("[Sb]",   "antimony"),
    ("[Bi]",   "bismuth"),
    ("[SbH3]", "stibane"),
    ("[BiH3]", "bismuthane"),
    # --- Group 16 (bare + heavy parent hydrides per Table 2.1) ---
    ("[O]",    "oxygen"),
    ("[S]",    "sulfur"),
    ("[Se]",   "selenium"),
    ("[Te]",   "tellurium"),
    ("[TeH2]", "tellane"),
    ("[Po]",   "polonium"),
    ("[PoH2]", "polane"),
    # --- Group 17 (bare halogens + AtH per Table 2.1) ---
    ("[F]",  "fluorine"),
    ("[Cl]", "chlorine"),
    ("[Br]", "bromine"),
    ("[I]",  "iodine"),
    ("[At]", "astatine"),
    ("[AtH]", "astatane"),
    # --- Group 18 noble gases ---
    ("[He]", "helium"),
    ("[Ne]", "neon"),
    ("[Ar]", "argon"),
    ("[Kr]", "krypton"),
    ("[Xe]", "xenon"),
    ("[Rn]", "radon"),
    # --- Late-d-block / synthetic ---
    ("[Pd]", "palladium"),
    ("[Og]", "oganesson"),
    # --- Lambda hypervalent parent hydrides (P-14.7) ---
    ("[SH4]",  "lambda4-sulfane"),
    ("[SH6]",  "lambda6-sulfane"),
    ("[PH5]",  "lambda5-phosphane"),
    ("[AsH5]", "lambda5-arsane"),
    ("[SbH5]", "lambda5-stibane"),
    ("[IH3]",  "lambda3-iodane"),
    ("[IH5]",  "lambda5-iodane"),
    # --- Lead parent hydride ---
    ("[PbH4]", "plumbane"),
])
def test_elementary_atom_hook_positive(smi: str, expected_name: str) -> None:
    """The hook must return the expected IUPAC name for each in-scope probe."""
    mol = Chem.MolFromSmiles(smi)
    assert _name_elementary_atom(mol) == expected_name


@pytest.mark.parametrize("smi,expected_name", [
    # Mirror the parametrised list above — the public engine entry point
    # must emit the same surface name as the hook.
    ("[Li]",   "lithium"),
    ("[O]",    "oxygen"),
    ("[N]",    "nitrogen"),
    ("[B]",    "boron"),
    ("[Cl]",   "chlorine"),
    ("[Pd]",   "palladium"),
    ("[Og]",   "oganesson"),
    ("[PbH4]", "plumbane"),
    ("[SbH3]", "stibane"),
    ("[SH4]",  "lambda4-sulfane"),
    ("[IH3]",  "lambda3-iodane"),
    ("[IH5]",  "lambda5-iodane"),
    # P-21.1.1 Table 2.1 Group-13 hydrides (Phase 8 P-2 audit).
    ("[AlH3]", "alumane"),
    ("[GaH3]", "gallane"),
    ("[InH3]", "indigane"),
    ("[TlH3]", "thallane"),
    # P-21.1.1 Table 2.1 polonium / astatine.
    ("[Po]",   "polonium"),
    ("[PoH2]", "polane"),
    ("[AtH]",  "astatane"),
])
def test_engine_emits_elementary_atom_name(smi: str, expected_name: str) -> None:
    assert name_smiles(smi) == expected_name


# ---------------------------------------------------------------------------
# Hook-level negative tests — must not claim
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("smi", [
    # Charged species belong to charge perception / curated inorganic.
    "[Li+]",
    "[Na+]",
    "[K+]",
    "[NH4+]",
    "[Cl-]",
    "[OH-]",
    "[O-2]",
    # Isotopologues belong to the isotope pipeline.
    "[3H]",
    "[14C]",
    "[2H]C([2H])([2H])[2H]",   # perdeuteromethane
    "C[2H]",
    # Multi-atom molecules (chains / rings / salts).
    "CC",
    "CCO",
    "c1ccccc1",
    "CC.[Na+]",
    "[Na+].[Cl-]",
    # Bare / molecular hydrogen — out of scope for the hook.
    "[H]",
    "[H][H]",
    # Methane / water / ammonia / borane / silane already have downstream
    # plan-search names — the hook does NOT claim ``methane`` (key
    # ("C", 4) is intentionally absent from the table) so the existing
    # chain pipeline still runs.
    "C",        # methane → chain pipeline
])
def test_elementary_atom_hook_negative(smi: str) -> None:
    """The hook must return None for every out-of-scope probe."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        pytest.skip(f"SMILES {smi!r} is not RDKit-parseable")
    # ``[H]`` / ``[H][H]`` have zero heavy atoms; that's the early-out path.
    assert _name_elementary_atom(mol) is None


def test_elementary_atom_hook_handles_none_mol() -> None:
    assert _name_elementary_atom(None) is None


# ---------------------------------------------------------------------------
# Already-covered regression guards (must not break)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("smi,expected_name", [
    # Existing curated-inorganic table entries that already worked before
    # this hook landed.  These flow through the regular plan-search retained
    # path — the hook returns None for them.
    ("[Fe]", "iron"),
    ("[Al]", "aluminium"),
    ("[Sn]", "tin"),
    ("[Pb]", "lead"),
    ("[Zn]", "zinc"),
    ("[Hg]", "mercury"),
    ("[Cu]", "copper"),
    ("[Ag]", "silver"),
    ("[Au]", "gold"),
    ("[Pt]", "platinum"),
    ("[Ni]", "nickel"),
    # Existing parent-hydride entries (curated-inorganic table).
    ("B",       "borane"),
    ("N",       "ammonia"),
    ("[SiH4]",  "silane"),
    ("[GeH4]",  "germane"),
    ("[SnH4]",  "stannane"),
    ("[AsH3]",  "arsane"),
    ("O",       "water"),
    ("S",       "hydrogen sulfide"),
    ("[SeH2]",  "hydrogen selenide"),
    # Charge-perception / curated cation paths must still emit their names.
    ("[NH4+]",  "azanium"),
    ("[Li+]",   "lithium(1+)"),
    ("[Na+]",   "sodium(1+)"),
    ("[Cl-]",   "chloride"),
    ("[OH-]",   "hydroxide"),
])
def test_existing_paths_unaffected(smi: str, expected_name: str) -> None:
    assert name_smiles(smi) == expected_name


# ---------------------------------------------------------------------------
# OPSIN round-trip — the architectural guarantee
# ---------------------------------------------------------------------------


def test_all_table_entries_round_trip_through_opsin() -> None:
    """For every (element, total_H) pair in the table, OPSIN must
    reconstruct the canonical input SMILES from the name we emit.

    Batched into a single ``py2opsin`` call to avoid the well-known
    Windows ``py2opsin_temp_input.txt`` race that occurs when many
    individually-parametrised round-trip tests fire serial JVM processes
    against the same fixed temp filename — that race let cross-test
    contamination through (e.g. a metallocene test's output appearing
    as the OPSIN result for ``lithium``).  One batched call eliminates
    the race and is also significantly faster.
    """
    from py2opsin import py2opsin

    # Build per-table probes from (element, total_H).
    probes: list[tuple[str, str]] = []  # (input_smiles, expected_name)
    for (sym, h), expected_name in _ELEMENTARY_ATOM_NAMES.items():
        if h == 0:
            smi = f"[{sym}]"
        elif h == 1:
            smi = f"[{sym}H]"
        else:
            smi = f"[{sym}H{h}]"
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            pytest.skip(f"SMILES {smi!r} is not RDKit-parseable")
        # Confirm the hook returns the expected name.
        actual = _name_elementary_atom(mol)
        assert actual == expected_name, (
            f"Hook returned {actual!r} for {smi!r}, expected {expected_name!r}"
        )
        probes.append((smi, expected_name))

    names = [name for _, name in probes]
    opsin_smiles_list = py2opsin(names, output_format="SMILES")
    failures: list[str] = []
    for (smi, name), opsin_smi in zip(probes, opsin_smiles_list):
        opsin_canon = _canon(opsin_smi)
        expected_canon = _canon(smi)
        if opsin_canon != expected_canon:
            failures.append(
                f"{smi!r}: name={name!r} opsin={opsin_canon!r} "
                f"expected={expected_canon!r}"
            )
    assert not failures, (
        "OPSIN round-trip failures:\n  " + "\n  ".join(failures)
    )
