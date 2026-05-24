"""Regression tests for Stage 7 multi-charge / radical-cation perception.

Pins the behaviour of the Stage 7 extensions to
``iupac_namer.perception.charge_perception`` and the engine
pre-validation hook in ``name_smiles``.  Probes are drawn from
``eval/opsin_audit_fg_raw.csv`` and ``eval/opsin_audit_hw_charge_raw.csv``
plus a handful of OPSIN-probed forms confirmed in this development
session.

Each probe is checked end-to-end:

1. the engine emits a name (no ``ValueError`` from the free-valence
   guard, no silent neutralization);
2. the emitted name round-trips through OPSIN (``py2opsin``) to the
   canonical SMILES of the input.

The OPSIN round-trip is the strong gate — even if our exact surface
form differs from the audit row (e.g. ``acetamidylium`` vs
``ethan-1-amidylium``, both of which OPSIN canonicalises to the same
SMILES), the test passes as long as the SMILES round-trip holds.

py2opsin uses a fixed temp file in CWD; concurrent test runs in the
same checkout race over it.  The ``_roundtrip`` helper isolates each
call via a per-call tempdir.
"""

from __future__ import annotations

import os
import tempfile

import pytest
from rdkit import Chem

try:
    import py2opsin as _p2o
    _HAVE_OPSIN = True
except Exception:  # pragma: no cover
    _HAVE_OPSIN = False

from iupac_namer.engine import name_smiles
from iupac_namer.perception.charge_perception import (
    classify_charges,
    detect_pre_validation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canon(smiles: str | None) -> str | None:
    if not smiles:
        return None
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(m) if m is not None else None


def _roundtrip(name: str) -> str | None:
    """Run OPSIN over ``name`` and return canonical SMILES, race-safe."""
    if not _HAVE_OPSIN:
        pytest.skip("py2opsin not available")
    import time

    cwd = os.getcwd()
    td = tempfile.mkdtemp(prefix="stage7_charge_opsin_")
    try:
        os.chdir(td)
        for attempt in range(6):
            try:
                smi = _p2o.py2opsin(name)
            except Exception:
                smi = None
                time.sleep(0.3 * (attempt + 1))
                continue
            if smi is not None:
                break
        return _canon(smi) if smi else None
    finally:
        os.chdir(cwd)
        try:
            import shutil

            shutil.rmtree(td, ignore_errors=True)
        except Exception:  # pragma: no cover
            pass


# ---------------------------------------------------------------------------
# Classifier sanity — Stage 7 motifs
# ---------------------------------------------------------------------------


def test_classify_aminylium_motif() -> None:
    cls = classify_charges(Chem.MolFromSmiles("CC[NH+]"))
    assert len(cls) == 1
    assert cls[0].suffix_hint == "aminylium"
    assert cls[0].radical_count == 2
    assert cls[0].site_charges == (1,)


def test_classify_iminylium_motif() -> None:
    cls = classify_charges(Chem.MolFromSmiles("CC=[N+]"))
    assert len(cls) == 1
    assert cls[0].suffix_hint == "iminylium"
    assert cls[0].radical_count == 2


def test_classify_amidylium_motif() -> None:
    cls = classify_charges(Chem.MolFromSmiles("CC(=O)[NH+]"))
    assert len(cls) == 1
    assert cls[0].suffix_hint == "amidylium"
    assert cls[0].radical_count == 2
    # Aminylium classifier must not overlap.
    assert "aminylium" not in {c.suffix_hint for c in cls}


def test_classify_polycarbon_diylium() -> None:
    cls = classify_charges(Chem.MolFromSmiles("[CH2+]C[CH2+]"))
    assert len(cls) == 1
    assert cls[0].suffix_hint == "diylium"
    assert cls[0].site_charges == (1, 1)
    assert cls[0].radical_count == 0


def test_classify_polycarbon_diide() -> None:
    cls = classify_charges(Chem.MolFromSmiles("[CH2-]C[CH2-]"))
    assert len(cls) == 1
    assert cls[0].suffix_hint == "diide"
    assert cls[0].site_charges == (-1, -1)


def test_classify_polycarbon_mixed() -> None:
    cls = classify_charges(Chem.MolFromSmiles("[CH2-]CC[CH2+]"))
    assert len(cls) == 1
    assert cls[0].suffix_hint == "mixed_id_ylium"


def test_classify_polycarbon_single_atom_q2() -> None:
    cls = classify_charges(Chem.MolFromSmiles("C[C+2]C"))
    assert len(cls) == 1
    assert cls[0].suffix_hint == "diylium"
    assert cls[0].site_charges == (2,)


def test_classify_polyacylium_oxal() -> None:
    cls = classify_charges(Chem.MolFromSmiles("O=[C+][C+]=O"))
    assert len(cls) == 1
    assert cls[0].suffix_hint == "polyacylium"
    # Two cation Cs + two carbonyl Os.
    assert len(cls[0].site_atom_indices) == 4


def test_classify_polyacylium_malon() -> None:
    cls = classify_charges(Chem.MolFromSmiles("O=[C+]C[C+]=O"))
    assert len(cls) == 1
    assert cls[0].suffix_hint == "polyacylium"


def test_polyacylium_does_not_steal_single_acylium() -> None:
    """``[C+](C)=O`` is a SINGLE acylium, not a polyacylium."""
    cls = classify_charges(Chem.MolFromSmiles("[C+](C)=O"))
    assert len(cls) == 1
    assert cls[0].suffix_hint == "acylium"


# ---------------------------------------------------------------------------
# Engine end-to-end — exact surface name pin
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("smi,expected_name", [
    # ---- radical-cation N motifs ----
    ("C(C)[NH+]",                            "ethan-1-aminylium"),
    ("C(CCCC)[NH+]",                         "pentan-1-aminylium"),
    ("C1(CCCCC1)[NH+]",                      "cyclohexan-1-aminylium"),
    ("C(C)=[N+]",                            "ethan-1-iminylium"),
    ("C(CCCC)=[N+]",                         "pentan-1-iminylium"),
    ("CC(CCC)=[N+]",                         "pentan-2-iminylium"),
    ("C1(CCCCC1)=[N+]",                      "cyclohexan-1-iminylium"),
    ("C(C)(=O)[NH+]",                        "acetamidylium"),
    ("C1(CCCCC1)C(=O)[NH+]",                 "cyclohexan-1-amidylium"),
    # ---- multi-charge polycations / polyanions ----
    ("[CH2+]C[CH2+]",                        "propane-1,3-diylium"),
    ("[CH2+]CC[CH2+]",                       "butane-1,4-diylium"),
    ("C[CH+]C[CH+]C",                        "pentane-2,4-diylium"),
    ("[CH+]1CC[CH+]CC1",                     "cyclohexane-1,4-diylium"),
    ("[CH2-]C[CH2-]",                        "propane-1,3-diide"),
    ("[CH2-]CC[CH2+]",                       "butan-1-id-4-ylium"),
    ("[CH+3]",                               "methanetriylium"),
    ("[CH2+2]",                              "methanediylium"),
    ("C[C+2]C",                              "propane-2,2-diylium"),
    ("[CH+2]CC",                             "propane-1,1-diylium"),
    # ---- polyacylium ----
    ("O=[C+][C+]=O",                         "oxalylium"),
    ("O=[C+]C[C+]=O",                        "malonylium"),
    ("O=[C+]CC[C+]=O",                       "succinylium"),
    ("O=[C+]CCC[C+]=O",                      "glutarylium"),
])
def test_engine_emits_exact_surface_name(smi: str, expected_name: str) -> None:
    """Pin the exact surface name the engine emits for each Stage 7 probe."""
    assert name_smiles(smi) == expected_name


# ---------------------------------------------------------------------------
# OPSIN round-trip — every probe must canonicalise back to the input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("smi", [
    # radical-cation N motifs
    "C(C)[NH+]",
    "C(CCCC)[NH+]",
    "C1(CCCCC1)[NH+]",
    "C1(=CC=CC2=CC=CC=C12)[NH+]",
    "C(C)=[N+]",
    "C(CCCC)=[N+]",
    "CC(CCC)=[N+]",
    "C1(CCCCC1)=[N+]",
    "C(C)(=O)[NH+]",
    "C1(CCCCC1)C(=O)[NH+]",
    "C1(=CC=CC2=CC=CC=C12)C(=O)[NH+]",
    # polycations / polyanions / mixed
    "[CH2+]C[CH2+]",
    "[CH2+]CC[CH2+]",
    "C[CH+]C[CH+]C",
    "[CH+]1CC[CH+]CC1",
    "[CH2-]C[CH2-]",
    "[CH2-]CC[CH2+]",
    "[CH+3]",
    "[CH2+2]",
    "C[C+2]C",
    "[CH+2]CC",
    "[CH+2]C",
    # polyacylium
    "O=[C+][C+]=O",
    "O=[C+]C[C+]=O",
    "O=[C+]CC[C+]=O",
    "O=[C+]CCC[C+]=O",
    "O=[C+]CCCC[C+]=O",
])
def test_engine_output_round_trips_through_opsin(smi: str) -> None:
    """Strong gate: OPSIN canonicalises our name back to the input's
    canonical SMILES.  This is what turns the audit-CSV row from
    NAMING_ERROR / WRONG to COVERED."""
    name = name_smiles(smi)
    rt = _roundtrip(name)
    expected = _canon(smi)
    assert rt == expected, (
        f"Round-trip failed for {smi!r}: name={name!r} opsin={rt!r} "
        f"expected={expected!r}"
    )


# ---------------------------------------------------------------------------
# Engine hook — radical-cation pre-validation entry point
# ---------------------------------------------------------------------------


def test_pre_validation_hook_returns_none_for_neutral() -> None:
    """The pre-validation hook must defer for non-radical-cation
    inputs so the standard engine dispatch keeps running."""
    mol = Chem.MolFromSmiles("c1ccccc1")
    result = detect_pre_validation(mol, strategy=None, session=None)
    assert result is None


def test_pre_validation_hook_returns_none_for_closed_shell_cation() -> None:
    """Closed-shell cations route through the post-guard ``detect``,
    not this entry point."""
    mol = Chem.MolFromSmiles("[CH3+]")
    result = detect_pre_validation(mol, strategy=None, session=None)
    assert result is None


def test_pre_validation_hook_returns_none_for_polycation() -> None:
    """Multi-charge polycations have radical_count==0 so the pre-
    validation hook defers to the post-guard dispatch."""
    mol = Chem.MolFromSmiles("[CH2+]C[CH2+]")
    result = detect_pre_validation(mol, strategy=None, session=None)
    assert result is None


# ---------------------------------------------------------------------------
# Non-regression — Stage 7 must not disturb R2-B closed-shell motifs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("smi,expected_name", [
    ("[CH3+]",          "methylium"),
    ("[CH3-]",          "methanide"),
    ("[CH2+]C",         "ethan-1-ylium"),
    ("[CH+]1CCCCC1",    "cyclohexan-1-ylium"),
    ("CC[N+]#N",        "ethane-1-diazonium"),
    ("[C+](C)=O",       "acetylium"),
    ("CC(=[NH2+])N",    "acetamidinium"),
    ("[BH4-]",          "boranuide"),
])
def test_r2b_closed_shell_motifs_still_pinned(smi, expected_name) -> None:
    assert name_smiles(smi) == expected_name


@pytest.mark.parametrize("smi,expected_substring", [
    ("c1cc[n+](C)cc1",   "pyridin"),       # ring-N+ retained path
    ("[C+]1=CC=CC=C1",   "phenylium"),     # retained ring lookup
    ("[O+]1=CC=CC=C1",   "pyrylium"),      # retained ring lookup
    ("[NH4+]",           "azanium"),       # monatomic
    ("CC(=O)O",          "acetic acid"),   # neutral baseline
])
def test_existing_paths_undisturbed(smi: str, expected_substring: str) -> None:
    name = name_smiles(smi)
    assert expected_substring in name


# ---------------------------------------------------------------------------
# Frozen dataclass invariant on the extended fields
# ---------------------------------------------------------------------------


def test_charge_classification_radical_count_is_immutable() -> None:
    from iupac_namer.perception.charge_perception import ChargeClassification

    cls = ChargeClassification(
        site_atom_indices=(0,),
        charge_sign="+",
        suffix_hint="aminylium",
        radical_count=2,
        site_charges=(1,),
    )
    with pytest.raises((AttributeError, Exception)):
        cls.radical_count = 0  # type: ignore[misc]
    with pytest.raises((AttributeError, Exception)):
        cls.site_charges = ()  # type: ignore[misc]
