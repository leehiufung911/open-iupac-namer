"""Regression tests for the 2,3-dihydro-1-benzofuran (coumaran) atom_locants
augmentation in ``iupac_namer/ring_naming/retained_lookup.py``.

Background (Stage 7 follow-up):
The curated entry for ``c1ccc2c(c1)CCO2`` (2,3-dihydro-1-benzofuran, aka
coumaran) shipped without ``atom_locants``.  The default sorted-atom-index
numbering misnumbered substituted forms — 4/5/6/7-methyl-coumaran round-
tripped to the wrong locants:

    * Cc1ccc2c(c1)OCC2  was emitted as 4-methyl, should be 6-methyl
    * Cc1cccc2c1OCC2    was emitted as 3-methyl, should be 7-methyl
    * Cc1cccc2c1CCO2    was emitted as 5-methyl, should be 4-methyl

The ``_CURATED_ATOM_LOCANTS_AUGMENT`` table in retained_lookup.py supplies
the missing ring-atom -> IUPAC-locant mapping, mirroring the role of
``_OPSIN_RING_ATOM_LOCANTS`` for OPSIN-data entries.  Atom-locant values
were derived from OPSIN methyl-probing on every numeric position
(2,3,4,5,6,7) with bond-generic SubstructMatch onto the canonical ring
SMILES.

Each tuple in ``_PROBES`` below pins
    (input_smiles, expected_engine_name)
and is round-tripped through OPSIN to confirm canonical-SMILES equivalence.
"""
from __future__ import annotations

import pytest
from rdkit import Chem

from iupac_namer.engine import name_smiles


# (input_smiles, expected_name)
_PROBES: list[tuple[str, str]] = [
    # Bare parent
    ("c1ccc2c(c1)CCO2", "2,3-dihydro-1-benzofuran"),
    # sp3 ring positions (2,3) — already worked pre-fix because they happen
    # to align with the default sorted-index numbering, but pinned here as
    # regression coverage so a future refactor that removes the augment
    # cannot silently break them.
    ("CC1OC2=C(C1)C=CC=C2", "2-methyl-2,3-dihydro-1-benzofuran"),
    ("CC1COC2=C1C=CC=C2", "3-methyl-2,3-dihydro-1-benzofuran"),
    # Benzene-ring positions (4,5,6,7) — these are the cases the augment
    # actually fixes.
    ("CC1=CC=CC2=C1CCO2", "4-methyl-2,3-dihydro-1-benzofuran"),
    ("CC=1C=CC2=C(CCO2)C1", "5-methyl-2,3-dihydro-1-benzofuran"),
    ("CC1=CC2=C(CCO2)C=C1", "6-methyl-2,3-dihydro-1-benzofuran"),
    ("CC1=CC=CC=2CCOC21", "7-methyl-2,3-dihydro-1-benzofuran"),
    # Stage 7 probe SMILES that previously misnumbered.  These canonicalize
    # to one of the OPSIN forms above; pinned here as the residual probes
    # the audit harness flagged.
    ("Cc1ccc2c(c1)OCC2", "6-methyl-2,3-dihydro-1-benzofuran"),
    ("Cc1cccc2c1OCC2", "7-methyl-2,3-dihydro-1-benzofuran"),
    ("Cc1cccc2c1CCO2", "4-methyl-2,3-dihydro-1-benzofuran"),
    ("Cc1ccc2c(c1)CCO2", "5-methyl-2,3-dihydro-1-benzofuran"),
]


@pytest.mark.parametrize("smiles,expected_name", _PROBES)
def test_coumaran_methyl_locants(smiles: str, expected_name: str) -> None:
    """Engine must emit the OPSIN-canonical IUPAC locant for each probe."""
    got = name_smiles(smiles)
    assert got == expected_name, (
        f"name_smiles({smiles!r}) = {got!r}, expected {expected_name!r}"
    )


@pytest.mark.parametrize("smiles,expected_name", _PROBES)
def test_coumaran_methyl_locants_roundtrip(smiles: str, expected_name: str) -> None:
    """Each probe must round-trip via OPSIN: input -> our-name -> OPSIN-SMILES
    -> same canonical structure as input.

    Skipped when py2opsin / Java are unavailable.
    """
    py2opsin = pytest.importorskip("py2opsin")
    smi_in = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    name = name_smiles(smiles)
    opsin_smi = py2opsin.py2opsin(name, output_format="SMILES")
    if not opsin_smi:
        pytest.skip(f"OPSIN/Java unavailable; got empty SMILES for {name!r}")
    smi_out = Chem.MolToSmiles(Chem.MolFromSmiles(opsin_smi))
    assert smi_out == smi_in, (
        f"round-trip mismatch: {smiles!r} -> {name!r} -> {opsin_smi!r}\n"
        f"  in  canonical: {smi_in}\n"
        f"  out canonical: {smi_out}"
    )


def test_coumaran_atom_locants_present_in_curated() -> None:
    """The augment must wire through into ``_CURATED`` so the curated lookup
    path returns atom_locants instead of ``None``.
    """
    from iupac_namer.ring_naming.retained_lookup import _CURATED

    entry = _CURATED.get("c1ccc2c(c1)CCO2")
    assert entry is not None, "curated entry missing for coumaran"
    name, sub_form, alkyl_ok, atom_locants = entry
    assert name == "2,3-dihydro-1-benzofuran"
    assert atom_locants is not None, (
        "Stage 7 atom_locants augment did not attach to coumaran curated entry"
    )
    # Spot-check a couple of pinned positions (full mapping verified by the
    # round-trip tests above).
    assert atom_locants[8] == 1   # ring O
    assert atom_locants[3] == "7a"
    assert atom_locants[4] == "3a"
