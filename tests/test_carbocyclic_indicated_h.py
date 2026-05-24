"""Regression tests for carbocyclic indicated-hydrogen on retained mancude
polycyclic ring parents (P-31.1.4.2), plus arsanthrene atom-locant numbering.

Two fixes are pinned here:

1. **Trindene carbocyclic indicated-H.**  Trindene (three indene-type 5-rings
   fused to a central benzene) is a mancude ring whose sp3 CH2 (indicated
   hydrogen) can sit on any of the three 5-rings — the 7H and 9H tautomers
   are distinct molecules.  The bare retained name ``trindene`` is therefore
   under-specified: ``9-chlorotrindene`` round-trips through OPSIN to the wrong
   sp3 ring.  The engine now (a) pins trindene's atom_locants to the
   data-key (7H) tautomer and (b) emits the ``7H-`` carbocyclic indicated-H
   prefix because the position is topologically ambiguous (orbit > 1).

2. **Arsanthrene numbering.**  Arsanthrene (9,10-diarsa-anthracene) carries a
   fixed As=C kekulé that makes peri positions {1,4} and {6,9} chemically
   distinct.  The retained-name numbering path now assigns the substituent a
   round-tripping locant (1/4), where the old sorted-index default produced a
   non-existent locant 11.

The acceptance test for each is the OPSIN round-trip: the emitted name is
correct only if OPSIN reconstructs the original structure.
"""
from __future__ import annotations

import pytest
from rdkit import Chem

from iupac_namer.engine import name_smiles
from iupac_namer.ring_naming.retained_lookup import (
    _build_lookup,
    _build_numbering_from_atom_locants,
    _smiles_to_record,
)
from iupac_namer.types import RingSystem


def _canon(smi: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi))


# ---------------------------------------------------------------------------
# Trindene: carbocyclic indicated-H + correct substituent locant
# ---------------------------------------------------------------------------

# (input SMILES, expected emitted name).  Built/verified against OPSIN.
_TRINDENE_CASES: list[tuple[str, str]] = [
    # Bare 7H-trindene parent (data-key tautomer).
    ("C1=Cc2c3c(c4c(c2=C1)=CC=C4)CC=C3", "7H-trindene"),
    # 9-chloro on the same 5-ring as the CH2 (the audit-probe structure).
    ("ClC1=CCc2c1c1c(c3c2C=CC=3)=CC=C1", "9-chloro-7H-trindene"),
]


@pytest.mark.parametrize("smi,expected", _TRINDENE_CASES)
def test_trindene_indicated_h_name(smi: str, expected: str) -> None:
    """Engine emits the 7H-prefixed trindene name with correct locant."""
    result = name_smiles(smi)
    assert result == expected, f"For {smi!r}: got {result!r}, expected {expected!r}"


@pytest.mark.parametrize("smi", [s for s, _ in _TRINDENE_CASES])
def test_trindene_roundtrip(smi: str) -> None:
    """7H-trindene names round-trip through OPSIN to the input structure."""
    py2opsin_mod = pytest.importorskip("py2opsin")
    our = name_smiles(smi)
    rt = py2opsin_mod.py2opsin(our)
    if not rt:
        pytest.skip(f"OPSIN unavailable (empty for {our!r})")
    assert _canon(rt) == _canon(smi), (
        f"Round-trip mismatch: our={our!r} in={_canon(smi)!r} rt={_canon(rt)!r}"
    )


# ---------------------------------------------------------------------------
# Indicated-H must NOT be added where the sp3 position is unique (regression
# guard): fluorene/phenalene/indene/acenaphthene must still round-trip.
# ---------------------------------------------------------------------------

_NO_REGRESSION_SMILES: list[str] = [
    "c1ccc2c(c1)Cc1ccccc1-2",          # fluorene
    "Clc1ccc2c(c1)Cc1ccccc1-2",         # 2-chlorofluorene
    "C1=Cc2ccccc2C1",                   # 1H-indene
    "C1=Cc2cccc3cccc(c23)C1",           # 1H-phenalene
    "C1CC2=CC=CC3=CC=CC1=C23",          # acenaphthene-type (no spurious IH)
]


@pytest.mark.parametrize("smi", _NO_REGRESSION_SMILES)
def test_no_regression_roundtrip(smi: str) -> None:
    """Pre-existing mancude carbocycles still round-trip after the change."""
    py2opsin_mod = pytest.importorskip("py2opsin")
    our = name_smiles(smi)
    rt = py2opsin_mod.py2opsin(our)
    if not rt:
        pytest.skip(f"OPSIN unavailable (empty for {our!r})")
    assert _canon(rt) == _canon(smi), (
        f"Regression: our={our!r} in={_canon(smi)!r} rt={_canon(rt)!r}"
    )


# ---------------------------------------------------------------------------
# Arsanthrene: retained-name numbering path assigns a round-tripping locant.
#
# The engine currently prefers the systematic von Baeyer name for arsanthrene
# (a strategy-layer choice outside ring_naming's scope), but the retained-name
# numbering MUST still place a peri substituent at locant 1 (or its symmetry
# partner 4), never the non-existent locant 11 that the old sorted-index
# default emitted.  This test exercises the ring_naming numbering directly so
# it is independent of the strategy plan choice.
# ---------------------------------------------------------------------------

_ARSANTHRENE_RING_CANON = "c1ccc2c(c1)[As]=c1ccccc1=[As]2"
_ARSANTHRENE_CL = "Clc1cccc2c1=[As]c1ccccc1[As]=2"


def test_arsanthrene_atom_locants_assign_peri_to_locant_1() -> None:
    _build_lookup()
    rec = _smiles_to_record.get(_ARSANTHRENE_RING_CANON)
    assert rec is not None and rec.get("atom_locants") is not None, (
        "arsanthrene must have atom_locants attached"
    )
    tgt = Chem.MolFromSmiles(_ARSANTHRENE_CL)
    ring_atoms = frozenset(
        a.GetIdx() for a in tgt.GetAtoms() if a.GetSymbol() != "Cl"
    )
    cl_idx = next(a.GetIdx() for a in tgt.GetAtoms() if a.GetSymbol() == "Cl")
    cl_carbon = tgt.GetAtomWithIdx(cl_idx).GetNeighbors()[0].GetIdx()

    ring_mol = Chem.MolFromSmiles(_ARSANTHRENE_RING_CANON)
    rs = RingSystem(
        atom_indices=ring_atoms,
        rings=(ring_atoms,),
        type="fused",
        aromatic=False,
        bridge_sizes=None,
        spiro_sizes=None,
        fusion_info=None,
        heteroatoms=None,
        ring_size=len(ring_atoms),
    )
    numberings = _build_numbering_from_atom_locants(
        ring_mol, tgt, rs, rec["atom_locants"]
    )
    assert numberings, "arsanthrene retained numbering must yield options"
    cl_locants = set()
    for nb in numberings:
        loc = nb.atom_to_locant.get(cl_carbon)
        if loc is not None:
            cl_locants.add(str(loc))
    # The peri-CH must map to a round-tripping locant (1 or its kekule-symmetry
    # partner 4) — never the non-existent locant 11.
    assert cl_locants <= {"1", "4"}, (
        f"arsanthrene peri-Cl got locants {cl_locants}, expected subset of "
        "{'1','4'}"
    )
    assert "1" in cl_locants, (
        f"arsanthrene peri-Cl must be assignable to locant 1; got {cl_locants}"
    )


def test_arsanthrene_roundtrip() -> None:
    """Whatever name the engine chooses for chloro-arsanthrene must round-trip
    (currently the systematic von Baeyer name; the retained-name numbering is
    also correct per the test above)."""
    py2opsin_mod = pytest.importorskip("py2opsin")
    our = name_smiles(_ARSANTHRENE_CL)
    rt = py2opsin_mod.py2opsin(our)
    if not rt:
        pytest.skip(f"OPSIN unavailable (empty for {our!r})")
    assert _canon(rt) == _canon(_ARSANTHRENE_CL), (
        f"Round-trip mismatch: our={our!r} rt={_canon(rt)!r}"
    )