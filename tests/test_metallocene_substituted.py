"""Stage 22 R22-E — substituted metallocene perception.

Closes the metallocene-with-substituent audit-row category surfaced in
the Stage 21 closeout (``eval/opsin_audit_rings_raw.csv`` ``tier1_cl``
rows for chloroferrocene, chlororuthenocene, and the rest of the
``-ocene`` family).  Before R22-E the engine emitted

    iron(2+) chlorocyclopentadienide cyclopentadienide

via the generic salt path; for V/Pb/Rh/Nb the standalone-metal-fragment
plan-search failure manifested as a free-valence guard rejection.  R22-E
adds an extension to the existing :mod:`iupac_namer.perception.organometallic`
module that detects 3-fragment salts of metallocene metals + two Cp
rings (each ring zero or one mono-atomic substituent) and composes the
retained ``[locant-]<prefix>METocene`` surface name.

Scope (intentionally conservative — multi-substituted single rings,
branched substituents, etc. continue to fall through to the salt path):

* Both rings unsubstituted — handled by the parent dispatcher; included
  here for symmetry.
* One ring carries one halogen / methyl / amino / hydroxy: emit
  ``<prefix>METocene`` with no locant (a single Cp position is
  unambiguous).
* Both rings carry the same single substituent: emit
  ``1,1'-di<prefix>METocene``.
* Both rings carry different single substituents (alphabetical): emit
  ``1-<lo>-1'-<hi>METocene``.

Round-trip closure is via InChI equivalence (the audit row's
``Clc1cc[cH-]c1`` and OPSIN's reverse ``Cl[c-]1cccc1`` differ in RDKit
canonical SMILES because the Cp anion delocalisation isn't applied
during canonicalisation — but their InChIs are identical, which is what
the authoritative-eval ``authoritative_match`` predicate accepts via
its InChI fallback).  See the classifier source comment for details.
"""

from __future__ import annotations

import pytest
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchi

from iupac_namer.engine import name_smiles


# ---------------------------------------------------------------------------
# Audit rows (Stage 21 closeout "H" candidate — opsin_audit_rings_raw.csv)
# ---------------------------------------------------------------------------


AUDIT_ROWS: list[tuple[str, str]] = [
    # input SMILES, expected emitted name
    ("Clc1cc[cH-]c1.[Fe+2].c1cc[cH-]c1", "chloroferrocene"),
    ("Clc1cc[cH-]c1.[Ru+2].c1cc[cH-]c1", "chlororuthenocene"),
]


@pytest.mark.parametrize("smiles,expected", AUDIT_ROWS)
def test_audit_row_emits_retained_substituted_name(
    smiles: str, expected: str,
) -> None:
    """The two audit-row inputs must emit the retained -ocene name."""
    assert name_smiles(smiles) == expected


# ---------------------------------------------------------------------------
# In-scope variants (extended coverage)
# ---------------------------------------------------------------------------


VARIANT_ROWS: list[tuple[str, str]] = [
    # methyl on each common metal centre
    ("Cc1cc[cH-]c1.[Fe+2].c1cc[cH-]c1", "methylferrocene"),
    ("Cc1cc[cH-]c1.[Ru+2].c1cc[cH-]c1", "methylruthenocene"),
    # chloro on a radical-bearing metal (V) — must go before the
    # free-valence guard
    ("Clc1cc[cH-]c1.[V+2].c1cc[cH-]c1", "chlorovanadocene"),
    # both rings substituted symmetrically — primed locants
    ("Cc1cc[cH-]c1.[Fe+2].Cc1cc[cH-]c1", "1,1'-dimethylferrocene"),
    ("Clc1cc[cH-]c1.[Fe+2].Clc1cc[cH-]c1", "1,1'-dichloroferrocene"),
    # both rings substituted asymmetrically — alphabetical primed locants
    ("Cc1cc[cH-]c1.[Pb+2].Clc1cc[cH-]c1", "1-chloro-1'-methylplumbocene"),
    # chloro on cobaltocene (audit-row sibling)
    ("Clc1cc[cH-]c1.[Co+2].c1cc[cH-]c1", "chlorocobaltocene"),
]


@pytest.mark.parametrize("smiles,expected", VARIANT_ROWS)
def test_variant_emits_retained_substituted_name(
    smiles: str, expected: str,
) -> None:
    assert name_smiles(smiles) == expected


# ---------------------------------------------------------------------------
# Round-trip via InChI (matches authoritative-eval semantics)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "smiles",
    [s for s, _ in AUDIT_ROWS + VARIANT_ROWS],
)
def test_round_trip_inchi_equivalence(smiles: str) -> None:
    """OPSIN(emitted_name) and the input SMILES have matching InChIs.

    Run as a single test per row — py2opsin's temp-file race on Windows
    is mitigated by the conftest retry wrapper.
    """
    from py2opsin import py2opsin

    name = name_smiles(smiles)
    assert name, f"no name emitted for {smiles!r}"
    rt_smi = py2opsin(name)
    assert rt_smi, f"OPSIN failed to parse {name!r}"
    rt_mol = Chem.MolFromSmiles(rt_smi)
    in_mol = Chem.MolFromSmiles(smiles)
    assert MolToInchi(rt_mol) == MolToInchi(in_mol), (
        f"InChI mismatch for {name!r}: "
        f"input={smiles!r} OPSIN(name)={rt_smi!r}"
    )


# ---------------------------------------------------------------------------
# Out-of-scope: multi-substituted single ring still falls through
# ---------------------------------------------------------------------------


def test_dichloro_on_one_ring_falls_through() -> None:
    """A single Cp ring carrying TWO substituents is out of R22-E scope
    (locant-handling on the metallocene parent for the within-ring case
    is not implemented).  The engine must defer to the existing salt
    path rather than emit a wrong R22-E name.

    The salt-path name need not be IUPAC-canonical; the load-bearing
    assertion is that R22-E does NOT mis-claim it.
    """
    out = name_smiles("Clc1ccc[c-]1Cl.[Fe+2].c1cc[cH-]c1")
    # Should not be a clean ``-ferrocene`` retained-name composition
    # (multi-sub on one ring needs locant handling we deferred).  The
    # salt-path emission falls through and the recursive Cp-anion
    # naming itself has known gaps, so we just assert the R22-E parent
    # path didn't fire.
    assert out != "dichloroferrocene"
    assert out != "1,2-dichloroferrocene"


def test_branched_substituent_falls_through() -> None:
    """A multi-atom branched substituent (e.g. -CH2CH3) is out of R22-E
    scope — the substituent classifier accepts only single-atom prefixes
    in this pass."""
    out = name_smiles("CCc1cc[cH-]c1.[Fe+2].c1cc[cH-]c1")
    assert "ethylferrocene" not in out
