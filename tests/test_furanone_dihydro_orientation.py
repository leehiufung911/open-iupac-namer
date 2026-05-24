"""Regression tests for the furanone / thiophenone oxo-dihydro LOCANT
orientation fix (P-31.1.4.1.1 / P-31.1.4.3.4).

Root cause
----------
A monocyclic mancude 5-ring O/S heterocycle that carries BOTH a ring carbonyl
(``C=O``) and a higher-seniority principal characteristic group (e.g. ``-COOH``)
must be numbered so the senior group gets the lowest locant.  That forces the
ring carbonyl to be cited as an ``oxo`` PREFIX (not the ``-one`` suffix) and
shifts the actually-saturated ring atoms to new locants.

The engine matched the curated bare-skeleton entry ``2,3-dihydrofuran`` and
emitted ``5-oxo-2,3-dihydrofuran-2-carboxylic acid`` — but that name describes a
DIFFERENT (more saturated) structure: OPSIN reads ``2,3-dihydro`` literally as
saturation at positions 2 and 3, contradicting the C=C present between 2-3.  The
correct PIN is ``5-oxo-4,5-dihydrofuran-2-carboxylic acid`` (C=C at 2-3, the CH2
at 4 and the carbonyl carbon at 5 are the saturated/added-H positions).

Fix
---
``_try_derive_hydro_retained`` gained an ``include_oxo_carbons_as_saturation``
mode that counts a ring carbon bearing exactly one exocyclic suffix-eligible
double bond (=O/=S/=Se/=Te/=NR) as a saturation (added-H) position — lifting
that doublet off as an oxo-class prefix turns the carbon into a saturated ring
member, so it joins the dihydro count.  Section 5c of ``try_retained_name``
invokes this mode for curated 5-ring dihydro entries and offers the re-oriented
``<locs>-dihydro<parent>`` forms as extra candidates, BUT only when the dihydro
locant SET differs from the curated form's set (so the symmetric
``2,5-dihydrofuran`` case is untouched and the carbonyl stays the lowest-locant
suffix when it IS the principal group).

Tests use the project's authoritative round-trip criterion: feed our emitted
name to OPSIN and compare canonical SMILES.
"""
from __future__ import annotations

import pytest
from rdkit import Chem

from iupac_namer.engine import name_smiles

try:
    from py2opsin import py2opsin
    _HAVE_OPSIN = True
except Exception:  # pragma: no cover
    _HAVE_OPSIN = False


def _canon(smiles: str) -> str | None:
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(m) if m is not None else None


def _roundtrips(smiles: str, name: str) -> bool:
    """True iff OPSIN parses ``name`` back to the same canonical structure."""
    out = py2opsin(name)
    if not out or not out.strip():
        return False
    return _canon(out) == _canon(smiles)


# ---------------------------------------------------------------------------
# 1. Exact-name assertions for the core fix (no OPSIN needed)
# ---------------------------------------------------------------------------

class TestExactNames:
    def test_furanone_carboxylic_acid_pin(self):
        # The headline gap.  Was wrongly emitted as
        # "5-oxo-2,3-dihydrofuran-2-carboxylic acid" (a different structure).
        assert (
            name_smiles("O=C1CC=C(C(=O)O)O1")
            == "5-oxo-4,5-dihydrofuran-2-carboxylic acid"
        )

    def test_thiophenone_carboxylic_acid_pin(self):
        assert (
            name_smiles("O=C1CC=C(C(=O)O)S1")
            == "5-oxo-4,5-dihydrothiophene-2-carboxylic acid"
        )

    def test_furanone_2one_no_higher_pcg_unchanged(self):
        # When the ring carbonyl IS the principal group it stays the -one
        # suffix at the lowest locant (the curated form): must NOT change.
        assert name_smiles("O=C1CC=CO1") == "2,3-dihydrofuran-2-one"

    def test_symmetric_2_5_dihydrofuranone_lowest_oxo_locant(self):
        # Symmetric 2,5-dihydro skeleton: carbonyl must stay at locant 2,
        # NOT regress to "...-5-one" (the equal-set orientations are filtered).
        assert name_smiles("O=C1C=CCO1") == "2,5-dihydrofuran-2-one"


# ---------------------------------------------------------------------------
# 2. Round-trip set — >= 20 OPSIN-parseable oxo-dihydro heterocycles
#    varying C=C position, oxo position, ring heteroatom, and substituents.
# ---------------------------------------------------------------------------

# Each entry is an input SMILES the engine must name such that the name
# round-trips through OPSIN to the same structure.
ROUNDTRIP_SMILES = [
    # --- furanones with a higher-seniority PCG (the failing class) ---
    "O=C1CC=C(C(=O)O)O1",        # 5-oxo-4,5-dihydrofuran-2-carboxylic acid
    "O=C1C(C)C=C(C(=O)O)O1",     # 4-methyl-...
    "O=C1CC(C)=C(C(=O)O)O1",     # 3-methyl-...
    "O=C1C(Cl)C=C(C(=O)O)O1",    # 4-chloro-...
    "N#CC1=CCC(=O)O1",           # 5-oxo-4,5-dihydrofuran-2-carbonitrile
    "O=CC1=CCC(=O)O1",           # 5-oxo-4,5-dihydrofuran-2-carbaldehyde
    "NC(=O)C1=CCC(=O)O1",        # 5-oxo-4,5-dihydrofuran-2-carboxamide
    "CCOC(=O)C1=CCC(=O)O1",      # ethyl 5-oxo-4,5-dihydrofuran-2-carboxylate
    "O=C1C=C(C(=O)O)CO1",        # carboxy at 3 of the 2,5-dihydro skeleton
    "O=C1C=C(C(=O)O)OC1",        # 4-oxo regiochemistry
    # --- thiophenone analogues ---
    "O=C1CC=C(C(=O)O)S1",        # 5-oxo-4,5-dihydrothiophene-2-carboxylic acid
    "O=C1SC=C(C(=O)O)C1",        # 5-oxo-4,5-dihydrothiophene-3-carboxylic acid
    "CC1=CCC(=O)S1",             # 5-methyl-2,3-dihydrothiophen-2-one
    # --- furanones where the carbonyl IS the principal group (controls) ---
    "O=C1CC=CO1",                # 2,3-dihydrofuran-2-one
    "O=C1C=CCO1",                # 2,5-dihydrofuran-2-one
    "CC1=CC(=O)OC1",             # 4-methyl-2,5-dihydrofuran-2-one
    "CC1=CCC(=O)O1",             # 5-methyl-2,3-dihydrofuran-2-one
    "O=C1CC=C(Cl)O1",            # 5-chloro-2,3-dihydrofuran-2-one
    "O=C1CC=C(Br)O1",            # 5-bromo-2,3-dihydrofuran-2-one
    "O=C1CC=C(c2ccccc2)O1",      # 5-phenyl-2,3-dihydrofuran-2-one
    "CC1=CC(C)C(=O)O1",          # 3,5-dimethyl-2,3-dihydrofuran-2-one
    "CC(=O)C1=CCC(=O)O1",        # 5-acetyl-2,3-dihydrofuran-2-one
    # --- 6-ring pyranones (route through 2H-pyran path; must stay correct) ---
    "O=C1CCC(C(=O)O)=CO1",       # 3,4-dihydro-2H-pyran-2-one-5-carboxylic acid
    "O=C1CCC=C(C(=O)O)O1",       # 3,4-dihydro-2H-pyran-2-one-6-carboxylic acid
]


@pytest.mark.skipif(not _HAVE_OPSIN, reason="py2opsin not available")
@pytest.mark.parametrize("smiles", ROUNDTRIP_SMILES)
def test_oxo_dihydro_roundtrip(smiles):
    name = name_smiles(smiles)
    assert name, f"engine returned no name for {smiles}"
    assert _roundtrips(smiles, name), (
        f"round-trip failed: {smiles!r} -> {name!r} -> "
        f"{_canon(py2opsin(name) or '') if _HAVE_OPSIN else '?'}"
    )


def test_roundtrip_set_size():
    # Guard against accidental shrinkage of the corpus below the >=20 mandate.
    assert len(ROUNDTRIP_SMILES) >= 20
