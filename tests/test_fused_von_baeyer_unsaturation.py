"""Von Baeyer fallback for ortho-fused ring systems with endocyclic unsaturation.

Perception classifies a two-or-more-ring ortho-fused system as ``type="fused"``
even when no retained name and no ``[1,3]``-dihetero fusion-prefix name applies.
For such systems with endocyclic unsaturation, ``name_fused`` previously
returned nothing and the engine emitted NO plan (NAMING_ERROR):

  * benzocyclobutadiene (``C1=Cc2ccccc21``) — an all-carbon benzo-fused
    four-ring with a cyclobutadiene double bond, PIN
    ``bicyclo[4.2.0]octa-1,3,5,7-tetraene`` (P-23.2.5 + P-23.3).
  * 2,3-didehydropenam (``S1C=CN2C1CC2=O``) — a partially unsaturated fused
    β-lactam, VB parent ``4-thia-1-azabicyclo[3.2.0]hept-2-ene`` (P-31.1.3
    heteroatom replacement on the von Baeyer skeleton).

The fix adds a STRUCTURAL von Baeyer fallback in ``name_fused``: when the
fused-hetero / fusion-prefix path yields no name, the fused skeleton is
decomposed via von Baeyer and routed through ``name_bridged``, which already
builds the unsaturation suffix by Kekulising aromatic ring bonds.

Each emitted name is verified to round-trip through OPSIN to the input
structure — the only definition of "correct" here.

See P-23.2.5 (von Baeyer nomenclature), P-23.3 (unsaturation locants),
P-31.1.3 (skeletal replacement / -thia-/-aza- prefixes).
"""
from __future__ import annotations

import os

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
    try:
        from py2opsin import py2opsin
    except ImportError:  # pragma: no cover
        pytest.skip("py2opsin not installed")
    out = py2opsin(name)
    if not out:
        return None
    return _canonical(out)


# ---------------------------------------------------------------------------
# The two target cases from the build brief — exact PIN expected.
# ---------------------------------------------------------------------------

TARGET_CASES = [
    # benzocyclobutadiene
    ("C1=Cc2ccccc21", "bicyclo[4.2.0]octa-1,3,5,7-tetraene"),
    # bicyclo[4.2.0]octa-2,4,6-triene (a benzo-fused cyclobutene isomer)
    ("C1=CC2=CCC2C=C1", "bicyclo[4.2.0]octa-1(8),2,4-triene"),
]


@pytest.mark.parametrize("smi,expected", TARGET_CASES)
def test_target_von_baeyer_name(smi, expected):
    assert name_smiles(smi) == expected


def test_didehydropenam_names_and_round_trips():
    # 2,3-didehydropenam: the fused β-lactam core gains a VB name; the ring
    # ketone is the -one suffix on top of the VB parent.  Per P-31.1.4.3.4 the
    # skeletal heteroatoms (4-thia-1-aza) and the ring double bond (hept-2-ene)
    # outrank the suffix, so the carbonyl takes locant 7, not a lower one.
    smi = "O=C1C[C@H]2SC=CN12"
    name = name_smiles(smi)
    assert name == "(5R)-4-thia-1-azabicyclo[3.2.0]hept-2-en-7-one", name


# ---------------------------------------------------------------------------
# Round-trip verification: every emitted name must parse back to the input.
# ---------------------------------------------------------------------------

ROUND_TRIP_CASES = [
    "C1=Cc2ccccc21",        # benzocyclobutadiene
    "C1=CC2=CCC2C=C1",      # bicyclo[4.2.0]octa-2,4,6-triene
    "O=C1C[C@H]2SC=CN12",   # 2,3-didehydropenam
    "C1=CC2CC2=C1",         # bicyclo[3.1.0]hexadiene
    "C1CC2=CCCCC12",        # bicyclo[4.2.0]oct-1-ene
]


@pytest.mark.parametrize("smi", ROUND_TRIP_CASES)
def test_round_trip(smi):
    name = name_smiles(smi)
    assert name and "ERROR" not in name, f"no name for {smi}: {name!r}"
    rt = _opsin_round_trip(name)
    assert rt == _canonical(smi), (
        f"round-trip mismatch for {smi}: name={name!r} "
        f"opsin={rt!r} expected={_canonical(smi)!r}"
    )


# ---------------------------------------------------------------------------
# Regression: systems that already had a retained name must KEEP it (the VB
# fallback only fires when the fused-hetero path produced nothing, and retained
# names outrank von Baeyer in strategy).
# ---------------------------------------------------------------------------

RETAINED_PRESERVED = [
    ("c1ccc2c(c1)CC2", "benzocyclobutene"),
    ("C1=CC2=CC=CC2=C1", "pentalene"),
]


@pytest.mark.parametrize("smi,expected", RETAINED_PRESERVED)
def test_retained_name_preserved(smi, expected):
    assert name_smiles(smi) == expected
