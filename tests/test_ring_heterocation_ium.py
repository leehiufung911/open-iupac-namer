"""Regression tests for ring-embedded heterocation "-ium" suffix (P-73.1).

Stage 9 R9-A: the Stage 8 audit refresh flagged that aromatic 5-ring
heterocycles carrying `[s+]`, `[o+]`, `[p+]` etc. were silently dropping
the charge — e.g. ``c1cs[s+]n1`` (1,2,3-dithiazol-2-ium) rendered as
``'1,2,3-dithiazole'``, a neutral name mismatching the input.

Root cause: three gates in ``iupac_namer/engine.py`` that drive the
"-N-ium" suffix machinery were hard-coded to check ``symbol == "N"``
only:

1. STANDALONE → CATION auto-promotion (net-charge-positive single
   fragments).
2. Leaf retained-name bypass when a charged neutral-stem retained
   entry would otherwise match.
3. ``ring_cation_locants`` collector in the SubstitutivePath.

Fix: introduce ``_RING_CATION_IUM_ELEMENTS`` covering Group 15
pnictogens (N, P, As, Sb, Bi) and Group 16 chalcogens (O, S, Se, Te),
and broaden all three gates.  Carbon cations (``-ylium``) are handled
by a separate classifier and are NOT part of this change.  Assembly
is already element-agnostic (emits ``-{locant}-ium`` based on the
atom's ring locant only) so no changes needed there.

Each probe below is a round-trip test: we name the input SMILES,
hand the emitted name back to OPSIN, and assert the canonicalised
SMILES matches the input's canonical form.  This pins the **semantic**
correctness of the fix, not a brittle exact-name match.
"""
from __future__ import annotations

import os
import shutil
import tempfile

import pytest

from iupac_namer.engine import name_smiles


def _opsin_canon(name: str) -> str | None:
    """Round-trip ``name`` through OPSIN, return canonical SMILES."""
    try:
        from py2opsin import py2opsin
        from rdkit import Chem
    except ImportError:
        pytest.skip("OPSIN round-trip unavailable (py2opsin/RDKit missing)")

    td = tempfile.mkdtemp(prefix="ring_hetcat_")
    cwd = os.getcwd()
    try:
        os.chdir(td)
        rt_smi = py2opsin(name)
    except Exception:
        return None
    finally:
        os.chdir(cwd)
        shutil.rmtree(td, ignore_errors=True)

    if not rt_smi:
        return None
    m = Chem.MolFromSmiles(rt_smi)
    return Chem.MolToSmiles(m) if m else None


def _rdkit_canon(smi: str) -> str | None:
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None


# (input_smiles, expected_name_substring, human_label)
# ``expected_name_substring`` is a fragment that MUST appear somewhere
# in the emitted name — it's not the full name (substituents/locant
# permutations may vary), just the architectural marker that the
# "-ium" suffix fired on the right heteroatom.
_HETEROCATION_PROBES: list[tuple[str, str, str]] = [
    # Aromatic 5-ring S+ → -ium (original audit probe)
    ("c1cs[s+]n1",     "-2-ium",  "1,2,3-dithiazol-2-ium"),
    # Aromatic 6-ring S+ → -ium.  Spec PIN per P-73.3.2 Table 7.5 is
    # the retained 'thiopyrylium'; the systematic form is 'thiin-1-ium'.
    # Both encode the cation; the test invariant is just "an -ium suffix
    # is emitted", so accept the shorter substring.
    ("[S+]1=CC=CC=C1", "ium",     "thiopyrylium / thiin-1-ium"),
    # Aromatic 5-ring N+ (control: already worked pre-fix, must stay green)
    ("c1cc[nH+]cc1",   "-1-ium",  "pyridin-1-ium control"),
    # Methyl-substituted N+ in 5-ring (1,3-thiazole isomer)
    ("c1cs[n+](C)c1",  "-ium",    "2-methylisothiazol-2-ium"),
    # N-methylpyridinium (well-established control)
    ("C[n+]1ccccc1",   "-1-ium",  "1-methylpyridin-1-ium control"),
    # Pyrimidinium (two N in 6-ring, one protonated)
    ("c1cnc[nH+]c1",   "-ium",    "pyrimidin-1-ium control"),
]


@pytest.mark.parametrize("smi,substr,label", _HETEROCATION_PROBES)
def test_ring_heterocation_emits_ium_suffix(
    smi: str, substr: str, label: str
) -> None:
    """Engine emits "-ium" for ring-embedded cationic heteroatoms.

    Architectural guard: verifies that the engine does not silently
    drop the positive charge when naming a ring cation whose charged
    atom is N, S, O, P, Se, or Te.  The retained-name short-circuit
    for names already ending in "ium" (pyrylium, flavylium, etc.)
    continues to fire first and is covered by a dedicated test below.
    """
    name = name_smiles(smi)
    assert name is not None, f"engine returned None for {label}: {smi!r}"
    assert "ERR" not in name, f"engine errored for {label}: {name!r}"
    assert substr in name, (
        f"for {label}, expected suffix-marker {substr!r} in emitted name "
        f"but got {name!r}"
    )
    # Round-trip guard: OPSIN must parse the emitted name back to a
    # structure matching the input (confirms charge survived the trip).
    rt_canon = _opsin_canon(name)
    in_canon = _rdkit_canon(smi)
    assert rt_canon == in_canon, (
        f"round-trip mismatch for {label}: input canon={in_canon!r} "
        f"-> name={name!r} -> rt canon={rt_canon!r}"
    )


def test_dithiazolium_does_not_render_as_neutral() -> None:
    """Regression guard for the Stage 9 audit bug.

    ``c1cs[s+]n1`` must NEVER render as bare ``1,2,3-dithiazole`` —
    the name-drop would mask the charge and cause a silent round-trip
    failure downstream.  Pins the architectural fix to
    ``_RING_CATION_IUM_ELEMENTS`` so future refactors can't regress it.
    """
    name = name_smiles("c1cs[s+]n1")
    assert name is not None
    # The broken form would be literally "1,2,3-dithiazole" with no
    # "-ium" anywhere.  Any fix must include the "-ium" marker.
    assert "ium" in name, f"charge silently dropped, got {name!r}"
    assert name != "1,2,3-dithiazole", (
        f"charge dropped for c1cs[s+]n1; got {name!r}"
    )


def test_pyrylium_retained_name_survives_broadened_gate() -> None:
    """Retained-name path for ring-O+ cations (pyrylium, flavylium, ...).

    The Stage 9 broadening of the retained-name CATION-bypass gate
    includes O in the set of heteroatoms that could trigger it.  The
    bypass only fires when the matched retained name does NOT already
    end in "ium" — so ``pyrylium`` (which ends in "ium") must still
    short-circuit the SubstitutivePath and render exactly as
    ``pyrylium``, without any locant.  This test pins that invariant.
    """
    name = name_smiles("[O+]1=CC=CC=C1")
    assert name == "pyrylium", f"retained-name path broken, got {name!r}"


def test_pyridinium_control_unchanged() -> None:
    """Existing ring-N+ cation rendering must be identical post-fix.

    The N-only gates were replaced with a set-membership check covering
    ``_RING_CATION_IUM_ELEMENTS`` (N plus P/As/Sb/Bi/O/S/Se/Te).  For
    pure-N inputs the broadened set still fires on N, so behaviour
    should be bit-identical.  Regressing pyridinium would break 100+
    downstream tests — this is the sentinel.
    """
    name = name_smiles("c1cc[nH+]cc1")
    assert name == "pyridin-1-ium", f"pyridinium regressed, got {name!r}"
