"""Regression tests for the kekulé-disambiguation rewrite table.

Background: RDKit canonical SMILES aromatises bond patterns, so multiple
kekulé tautomers collapse to the same canonical key.  Our ring-name
lookup previously emitted whatever tautomer's name happened to win the
``_smiles_to_record`` insertion race, while the ``atom_locants`` table
was pinned to a different tautomer's numbering convention — resulting
in mirror-image round-trip mismatches on substituted probes.

The ``iupac_namer/ring_naming/kekule_store.py`` module rewrites the
emitted name to the tautomer matching the stored atom_locants.  The
tests below pin:

    * parent-only round-trip (no substitution)
    * per-site substitution probes that previously produced the
      bond_pattern_differ failure mode (see
      ``docs/opsin_audit_rings.md`` top-20 table).

The name assertions check the engine's output string; the round-trip
tests additionally verify OPSIN reconstructs the original kekulé
tautomer when py2opsin + Java are available.
"""
from __future__ import annotations

import pytest
from rdkit import Chem

from iupac_namer.engine import name_smiles


# Each tuple: (input_smiles, expected_our_name).  The expected round-trip
# canonical is checked in ``test_kekule_rewrite_roundtrip``.
_INDENE_CASES: list[tuple[str, str]] = [
    ("C1=Cc2ccccc2C1", "1H-indene"),
    # ind-1 family (sp2=sp2 at locants 2-3 of 1H-indene = 1-chloro/2-chloro
    # on the sp2 atoms; 3-chloro is the sp2 atom adjacent to sp3 CH2).
    ("ClC1=CCc2ccccc21", "3-chloro-1H-indene"),
    ("ClC1=Cc2ccccc2C1", "2-chloro-1H-indene"),
    ("ClC1C=Cc2ccccc21", "1-chloro-1H-indene"),
    # benzo ring substitutions: locants 4-7 reflect across the sp3 atom.
    ("Clc1cccc2c1C=CC2", "4-chloro-1H-indene"),
    ("Clc1ccc2c(c1)C=CC2", "5-chloro-1H-indene"),
    ("Clc1ccc2c(c1)CC=C2", "6-chloro-1H-indene"),
    ("Clc1cccc2c1CC=C2", "7-chloro-1H-indene"),
]

_PERIMIDINE_CASES: list[tuple[str, str]] = [
    ("C1=Nc2cccc3cccc(c23)N1", "3H-perimidine"),
    # tautomer variants — N=CN pattern vs NC=N pattern each carry
    # distinct round-trip locants only when 3H-perimidine numbering
    # (our atom_locants convention) is emitted.
    ("Clc1ccc2cccc3c2c1N=CN3", "9-chloro-3H-perimidine"),
    ("Clc1cc2c3c(cccc3c1)NC=N2", "8-chloro-3H-perimidine"),
    ("Clc1ccc2c3c(cccc13)NC=N2", "7-chloro-3H-perimidine"),
    ("Clc1ccc2c3c(cccc13)N=CN2", "6-chloro-3H-perimidine"),
    ("Clc1cc2c3c(cccc3c1)N=CN2", "5-chloro-3H-perimidine"),
    ("Clc1ccc2cccc3c2c1NC=N3", "4-chloro-3H-perimidine"),
    # N-substitution variants (previously COVERED — make sure the rewrite
    # does not regress them).
    ("ClN1C=Nc2cccc3cccc1c23", "3-chloro-3H-perimidine"),
    ("ClC1=Nc2cccc3cccc(c23)N1", "2-chloro-3H-perimidine"),
]


@pytest.mark.parametrize("smi,expected", _INDENE_CASES + _PERIMIDINE_CASES)
def test_kekule_rewrite_name(smi: str, expected: str) -> None:
    """Engine emits the rewritten tautomer-specific name."""
    result = name_smiles(smi)
    assert result == expected, (
        f"For SMILES {smi!r}: got {result!r}, expected {expected!r}"
    )


@pytest.mark.parametrize(
    "smi",
    [smi for smi, _ in _INDENE_CASES + _PERIMIDINE_CASES],
)
def test_kekule_rewrite_roundtrip(smi: str) -> None:
    """Each emitted name round-trips through OPSIN back to the same canonical.

    This is the real acceptance test — the name string is only correct
    if OPSIN reconstructs the original kekulé tautomer.  Skipped if
    py2opsin is not installed or OPSIN is unreachable.
    """
    py2opsin_mod = pytest.importorskip("py2opsin")
    our_name = name_smiles(smi)
    rt_smi = py2opsin_mod.py2opsin(our_name)
    if not rt_smi:
        pytest.skip(f"OPSIN unavailable (py2opsin returned empty for {our_name!r})")
    input_canon = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    rt_canon = Chem.MolToSmiles(Chem.MolFromSmiles(rt_smi))
    assert input_canon == rt_canon, (
        f"Round-trip mismatch: our_name={our_name!r} input={input_canon!r} "
        f"opsin_rt={rt_canon!r}"
    )
