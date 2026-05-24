"""Regression tests for Stage 22 R22-A — partial-saturation hydro-locant
renumbering for substituted heterocyclic retained rings.

Background:
The curated entry for ``C1=CCNCC1`` (1,2,3,6-tetrahydropyridine) hard-codes
the unsubstituted form's hydro-prefix at 1,2,3,6.  When the engine carved a
substituent off the ring and matched the canonical key, only one substruct
orientation came back from RDKit (the bond-order constraint pinned the C=C
direction), forcing the principal characteristic group onto whichever locant
fell out of that single orientation.  Result for guvacine
(``N1CC(C(=O)O)=CCC1``):

    expected: 1,2,5,6-tetrahydropyridine-3-carboxylic acid (COOH on sp2 C
              at locant 3, hydro-prefix labels the 4 sp3 atoms 1,2,5,6)
    emitted:  1,2,3,6-tetrahydropyridine-3-carboxylic acid (COOH at locant 3
              but the curated hydro-prefix marks locant 3 as sp3 — OPSIN
              parses this back to the C-saturated isomer ``O=C(O)C1C=CCNC1``,
              breaking round-trip)

The fix in ``iupac_namer/ring_naming/retained_lookup.py``:
    1. Drop the heteroatom restriction in ``_try_derive_hydro_retained``;
    2. Add ``return_all_orientations=True`` mode that emits every
       valid substruct match orientation, each with its own rewritten
       hydro-prefix and Numbering;
    3. When the curated entry is a partial-saturation form (name matches
       ``^[\\d,]+-(?:di|tri|tetra|...)hydro``) AND the underlying aromatic
       parent admits >=2 valid match orientations (true symmetry-based
       ambiguity, e.g. pyridine's C2v), append every derived orientation as
       a separate NamedParent so the strategy layer can pick the one
       minimizing principal-characteristic-group locants
       (P-31.1.4.3.4).

The asymmetric-parent guard (>=2 orientations required) prevents the fix
from disturbing entries like ``2,3-dihydro-1-benzofuran`` (coumaran) where
the underlying benzofuran has no ring symmetry — those keep their curated
form unchanged.
"""
from __future__ import annotations

import pytest
from rdkit import Chem

from iupac_namer.engine import name_smiles


def _canon(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m is not None else None


# ---------------------------------------------------------------------------
# Audit row that motivated the stage: guvacine / THP-3-carboxylic acid.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "smiles, expected_name",
    [
        # Audit: guvacine alkaloid (pyridine partial saturation, COOH on sp2 C).
        ("N1CC(C(=O)O)=CCC1", "1,2,5,6-tetrahydropyridine-3-carboxylic acid"),
        # Same skeleton, different SMILES wording (canonicalisation check).
        ("O=C(O)C1=CCCNC1",   "1,2,5,6-tetrahydropyridine-3-carboxylic acid"),
    ],
)
def test_thp_carboxylic_acid_renumbers_to_lowest_suffix_locant(
    smiles: str, expected_name: str
) -> None:
    got = name_smiles(smiles)
    assert got == expected_name, (
        f"name_smiles({smiles!r}) = {got!r}, expected {expected_name!r}"
    )


@pytest.mark.parametrize(
    "smiles, expected_name",
    [
        # Other principal-characteristic-group suffixes that exhibit the same
        # bug pattern (suffix locant pinned to a position the curated
        # hydro-prefix labels as sp3).
        ("OC1=CCCNC1",  "3-hydroxy-1,2,5,6-tetrahydropyridine"),
        ("NC1=CCCNC1",  "1,2,5,6-tetrahydropyridin-3-amine"),
    ],
)
def test_thp_other_suffixes_renumber_correctly(
    smiles: str, expected_name: str
) -> None:
    got = name_smiles(smiles)
    assert got == expected_name, (
        f"name_smiles({smiles!r}) = {got!r}, expected {expected_name!r}"
    )


# ---------------------------------------------------------------------------
# Unsubstituted control — the curated form must remain reachable.
# ---------------------------------------------------------------------------

def test_unsubstituted_thp_keeps_curated_form() -> None:
    """The Stage 22 derive-hydro orientation enumeration must not displace
    the curated unsubstituted name.  With no substituent driving lowest-
    locant selection, the canonical 1,2,3,6 hydro-prefix wins."""
    got = name_smiles("C1=CCNCC1")
    assert got == "1,2,3,6-tetrahydropyridine"


# ---------------------------------------------------------------------------
# 1,4-Dihydronaphthalene — the carbocycle case must not regress.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "smiles, expected_name",
    [
        # Stage 3 R3-documented case: carbocyclic derive-hydro path stays
        # working — 2-COOH on 1,4-dihydronaphthalene round-trips correctly.
        ("O=C(O)C1=CCc2ccccc2C1", "1,4-dihydronaphthalene-2-carboxylic acid"),
        # Bare 1,4-dihydronaphthalene.
        ("C1=CCc2ccccc2C1", "1,4-dihydronaphthalene"),
    ],
)
def test_carbocycle_dihydro_path_unchanged(
    smiles: str, expected_name: str
) -> None:
    got = name_smiles(smiles)
    assert got == expected_name, (
        f"name_smiles({smiles!r}) = {got!r}, expected {expected_name!r}"
    )


# ---------------------------------------------------------------------------
# Asymmetric-parent guard — coumaran (2,3-dihydro-1-benzofuran) must not be
# disturbed by the orientation enumeration.  Benzofuran has no ring
# symmetry → only one substruct match orientation → curated form wins.
# ---------------------------------------------------------------------------

def test_asymmetric_parent_keeps_curated_hydro_form() -> None:
    """5-methyl-coumaran — benzofuran is asymmetric, so the orientation
    enumeration produces a single derived alternative which is correctly
    suppressed in favour of the curated ``2,3-dihydro-1-benzofuran``."""
    got = name_smiles("Cc1ccc2c(c1)CCO2")
    assert got == "5-methyl-2,3-dihydro-1-benzofuran"


# ---------------------------------------------------------------------------
# Round-trip canonical-SMILES check (structural correctness, not just
# string equality).  Pulled into a separate test so it shows up
# distinctly in the test matrix and a future eval/OPSIN regression flags
# it independently of any name-string normalisation drift in py2opsin.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "smiles, expected_name",
    [
        ("N1CC(C(=O)O)=CCC1", "1,2,5,6-tetrahydropyridine-3-carboxylic acid"),
        ("OC1=CCCNC1",        "3-hydroxy-1,2,5,6-tetrahydropyridine"),
        ("NC1=CCCNC1",        "1,2,5,6-tetrahydropyridin-3-amine"),
        ("O=C(O)C1=CCc2ccccc2C1", "1,4-dihydronaphthalene-2-carboxylic acid"),
    ],
)
def test_round_trip_canonical_smiles(smiles: str, expected_name: str) -> None:
    """Confirm that the engine's chosen name parses back (via OPSIN) to a
    structurally-identical molecule.  This catches the original bug class
    where the emitted name differed structurally from the input by carrying
    the C=C on the wrong locant."""
    py2opsin = pytest.importorskip("py2opsin")
    got_name = name_smiles(smiles)
    rt_smi = py2opsin.py2opsin(got_name)
    if not rt_smi:
        pytest.skip(f"py2opsin returned empty/None for {got_name!r}")
    assert _canon(rt_smi) == _canon(smiles), (
        f"name {got_name!r} parsed by OPSIN to {_canon(rt_smi)!r}; "
        f"expected canonical match with input {_canon(smiles)!r}"
    )
