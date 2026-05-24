"""Stage 1 fused-ring naming: ortho-fused two-component systems where the
smaller partner is a 5-ring with a [1,3]-dihetero pattern (dioxolo, dithiolo,
oxathiolo, ...).

These tests verify that:
  1. The new fused-ring naming module produces the expected
     ``[1,3]<prefix>[4,5-<letter>]<base>`` form for cases where no retained
     name applies.
  2. Each generated name round-trips correctly through OPSIN to the same
     canonical SMILES as the input.
  3. Existing retained names (e.g. 1,3-benzodioxole) are still preferred
     over the new systematic form (architectural priority preserved).
"""
from __future__ import annotations

import os
import sys

# Pin OPSIN's JAVA_HOME the same way authoritative_eval.py does.
os.environ.setdefault(
    "JAVA_HOME",
    os.environ.get("JAVA_HOME", ""),
)
os.environ["PATH"] = (
    os.environ["JAVA_HOME"] + "/bin" + os.pathsep + os.environ.get("PATH", "")
)

import pytest
from rdkit import Chem

from iupac_namer.engine import name_smiles


def _canonical(smiles: str) -> str | None:
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(m) if m is not None else None


def _opsin_round_trip(name: str) -> str | None:
    """Parse ``name`` through OPSIN and return the canonical SMILES.

    Returns None on parse failure or if OPSIN is unavailable.
    """
    try:
        from py2opsin import py2opsin
    except ImportError:  # pragma: no cover - py2opsin is in test extras
        pytest.skip("py2opsin not installed")
    out = py2opsin(name)
    if not out:
        return None
    return _canonical(out)


# ---------------------------------------------------------------------------
# Stage 1 cases that previously had no name (the dispatcher fell through to
# either the VB fallback or returned a NAMING ERROR).
# ---------------------------------------------------------------------------

STAGE1_CASES = [
    # (input SMILES, expected fused-ring name)
    # Same-element pair, all-carbon aromatic base
    ("c1ccc2c(c1)SCS2", "[1,3]dithiolo[4,5-b]benzene"),
    # Mixed-element pair, all-carbon aromatic base
    ("c1ccc2c(c1)OCS2", "[1,3]oxathiolo[4,5-b]benzene"),
    # Same-element pair, single-heteroatom aromatic base
    ("c1cnc2c(c1)OCO2", "[1,3]dioxolo[4,5-b]pyridine"),
    # Same-element pair, sulfur analogue + pyridine base
    ("c1cnc2c(c1)SCS2", "[1,3]dithiolo[4,5-b]pyridine"),
]


@pytest.mark.parametrize("smi,expected_name", STAGE1_CASES)
def test_stage1_fused_ring_name(smi: str, expected_name: str) -> None:
    """Each Stage 1 input gets the expected fusion-nomenclature name."""
    got = name_smiles(smi)
    assert got == expected_name, (
        f"Expected {expected_name!r} for {smi!r}, got {got!r}"
    )


@pytest.mark.parametrize("smi,expected_name", STAGE1_CASES)
def test_stage1_round_trip(smi: str, expected_name: str) -> None:
    """The generated name must round-trip through OPSIN to the same
    canonical SMILES as the input."""
    got = name_smiles(smi)
    if got.startswith("[NAMING ERROR"):
        pytest.fail(f"Namer failed for {smi}: {got}")
    rt = _opsin_round_trip(got)
    expected_canon = _canonical(smi)
    assert rt == expected_canon, (
        f"Round-trip mismatch for {smi}: name={got!r} → {rt!r}, "
        f"expected {expected_canon!r}"
    )


def test_stage1_retained_still_preferred() -> None:
    """1,3-benzodioxole has a curated retained name — it must win over the
    new ``[1,3]dioxolo[4,5-b]benzene`` systematic form per the dispatcher's
    method-rank ordering (P-31.1.3 retained-over-systematic for ring
    parents).
    """
    got = name_smiles("c1ccc2c(c1)OCO2")
    assert got == "1,3-benzodioxole", (
        f"Retained name should win; got {got!r}"
    )


def test_stage2_naphthalene_base_supported() -> None:
    """Stage 2B: a naphthalene base (itself two fused rings) IS now supported
    as the base of a dioxolo fusion.  This case was previously excluded in
    Stage 1 — Stage 2B adds multi-ring base coverage via the retained
    naphthalene lookup on a carved 2-ring sub-system."""
    got = name_smiles("c1ccc2cc3c(cc2c1)OCO3")
    assert got == "[1,3]dioxolo[4,5-b]naphthalene", (
        f"Expected Stage 2B multi-ring base name; got {got!r}"
    )


# ---------------------------------------------------------------------------
# Stage 2 cases: extensions of Stage 1.
#
#   2A. Multi-hetero monocyclic base (pyrazine / pyrimidine / pyridazine)
#   2B. Multi-ring fused base (naphthalene / quinoline) — 3-ring system
#   2C. 6-ring smaller component (1,3-dioxino) — same prefix family but
#       a saturated extra carbon at locant 4
#
# All targets have been OPSIN-verified to round-trip to the same canonical
# SMILES as the input.
# ---------------------------------------------------------------------------

STAGE2_CASES = [
    # 2A: multi-hetero base (pyrazine, pyrimidine)
    ("c1cnc2c(n1)OCO2", "[1,3]dioxolo[4,5-b]pyrazine"),
    ("c1cnc2c(n1)SCS2", "[1,3]dithiolo[4,5-b]pyrazine"),
    ("c1cnc2c(n1)OCS2", "[1,3]oxathiolo[4,5-b]pyrazine"),
    ("c1ncc2c(n1)OCO2", "[1,3]dioxolo[4,5-d]pyrimidine"),
    ("c1ncc2c(n1)SCS2", "[1,3]dithiolo[4,5-d]pyrimidine"),
    # 2B: multi-ring base (naphthalene, quinoline)
    ("c1ccc2cc3c(cc2c1)SCS3", "[1,3]dithiolo[4,5-b]naphthalene"),
    ("c1ccc2nc3c(cc2c1)OCO3", "[1,3]dioxolo[4,5-b]quinoline"),
    # 2C: 6-ring smaller (dioxino) — benzene and pyridine bases
    ("c1ccc2c(c1)COCO2", "[1,3]dioxino[4,5-b]benzene"),
    ("c1cnc2c(c1)COCO2", "[1,3]dioxino[4,5-b]pyridine"),
]


@pytest.mark.parametrize("smi,expected_name", STAGE2_CASES)
def test_stage2_fused_ring_name(smi: str, expected_name: str) -> None:
    """Each Stage 2 input gets the expected fusion-nomenclature name."""
    got = name_smiles(smi)
    assert got == expected_name, (
        f"Expected {expected_name!r} for {smi!r}, got {got!r}"
    )


@pytest.mark.parametrize("smi,expected_name", STAGE2_CASES)
def test_stage2_round_trip(smi: str, expected_name: str) -> None:
    """The generated name must round-trip through OPSIN to the same
    canonical SMILES as the input."""
    got = name_smiles(smi)
    if got.startswith("[NAMING ERROR"):
        pytest.fail(f"Namer failed for {smi}: {got}")
    rt = _opsin_round_trip(got)
    expected_canon = _canonical(smi)
    assert rt == expected_canon, (
        f"Round-trip mismatch for {smi}: name={got!r} → {rt!r}, "
        f"expected {expected_canon!r}"
    )


def test_stage2_retained_still_preferred_for_diazine_dioxole() -> None:
    """If a curated retained name covers a Stage 2-shaped scaffold (e.g.
    4-ring fused systems with their own retained names like methylenedioxy-
    fused heterocycles), retained must still win over our systematic form.

    For the bare dioxolo-pyrazine there is no curated retained name, so
    Stage 2A IS the winning name — but emitting Stage 2 names with
    ``naming_method='systematic'`` (rank 0.9, below ``retained`` = 100)
    structurally guarantees retained wins wherever defined.  This test pins
    that the systematic Stage 2 name is what gets emitted for the bare
    dioxolo-pyrazine (no retained lookup hits)."""
    got = name_smiles("c1cnc2c(n1)OCO2")
    assert got == "[1,3]dioxolo[4,5-b]pyrazine", (
        f"Stage 2A name expected; got {got!r}"
    )


def test_stage2_excludes_four_plus_ring_systems() -> None:
    """Stage 2 only handles 2- or 3-ring fused systems.  A larger fused
    polycycle must still get *some* valid name from the existing fallback
    paths and must NOT be claimed by Stage 2."""
    # Anthracene-fused dioxole: 4-ring system (anthracene = 3 rings + dioxole)
    got = name_smiles("c1ccc2cc3cc4c(cc3cc2c1)OCO4")
    assert not got.startswith("[NAMING ERROR"), (
        f"4-ring system should still get a name via fallback paths"
    )
    # Stage 2 is restricted to ≤3 rings; the explicit fusion-name string
    # should NOT appear (anthracene-base would need Stage 3 at minimum).
    assert "[1,3]dioxolo[4,5-b]anthracene" not in got, (
        "Stage 2 should not emit a 4-ring fusion name; got " + repr(got)
    )
