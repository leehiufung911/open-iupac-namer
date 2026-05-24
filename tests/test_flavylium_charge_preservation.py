"""Regression tests for the Stage 7 follow-up flavylium charge-preservation fix.

Pins the behaviour of the ``_classify_aryl_substituted_chromenylium``
classifier in ``iupac_namer.perception.charge_perception`` and the
end-to-end ``name_smiles`` flow for the flavylium family.

Background
----------
Before this fix the engine's PCG (parent class group) selector would
treat 2-aryl-chromenylium molecules (e.g. ``flavylium`` itself,
``c1ccc(-c2ccc3ccccc3[o+]2)cc1``) as *two* ring systems linked by a
single bond and pick the neutral phenyl as parent — emitting the
silently charge-dropping name ``(chromen-2-yl)benzene``.  The new
classifier recognises the 2-aryl-chromenylium pattern *before* the
plan search runs and emits the OPSIN-compatible flavylium-family
surface name directly, with the [O+] explicitly enumerated in
``site_atom_indices``.

Each probe is checked end-to-end:

1. the engine emits the expected flavylium-family name (no silent
   neutralisation, no ``(chromen-2-yl)benzene`` decomposition);
2. the emitted name round-trips through OPSIN (``py2opsin``) to the
   canonical SMILES of the input.

The OPSIN round-trip is the strong gate — if the surface form differs
from the test's expected string but OPSIN still canonicalises it back
to the input SMILES, the round-trip assertion still passes (and that
is the architecturally correct invariant).

py2opsin uses a fixed temp file in CWD; concurrent test runs in the
same checkout race over it, so each call is isolated in a per-call
tempdir (same helper pattern as
``tests/test_polycharge_radical_cation.py``).
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
    td = tempfile.mkdtemp(prefix="stage7_flavylium_opsin_")
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
# Classifier sanity — the 2-aryl-chromenylium motif
# ---------------------------------------------------------------------------


def test_classify_bare_2_phenyl_chromenylium_emits_flavylium() -> None:
    """Bare 2-phenyl-chromenylium classifies as the flavylium retained name."""
    cls = classify_charges(Chem.MolFromSmiles("c1ccc(-c2ccc3ccccc3[o+]2)cc1"))
    assert len(cls) == 1
    assert cls[0].suffix_hint == "aryl_chromenylium"
    assert cls[0].surface_name == "flavylium"
    # Site atoms must include exactly the [o+] (and only the [o+]) so the
    # coverage gate in ``detect`` sees the charge claimed.
    assert len(cls[0].site_atom_indices) == 1


def test_classify_4prime_methyl_emits_4prime_methylflavylium() -> None:
    """``Cc1ccc(-c2ccc3ccccc3[o+]2)cc1`` -> ``4'-methylflavylium``."""
    cls = classify_charges(Chem.MolFromSmiles("Cc1ccc(-c2ccc3ccccc3[o+]2)cc1"))
    assert len(cls) == 1
    assert cls[0].suffix_hint == "aryl_chromenylium"
    assert cls[0].surface_name == "4'-methylflavylium"


def test_classify_3prime_methyl_emits_3prime_methylflavylium() -> None:
    cls = classify_charges(Chem.MolFromSmiles("Cc1cccc(-c2ccc3ccccc3[o+]2)c1"))
    assert len(cls) == 1
    assert cls[0].suffix_hint == "aryl_chromenylium"
    assert cls[0].surface_name == "3'-methylflavylium"


def test_classify_2prime_methyl_emits_2prime_methylflavylium() -> None:
    cls = classify_charges(Chem.MolFromSmiles("Cc1ccccc1-c1ccc2ccccc2[o+]1"))
    assert len(cls) == 1
    assert cls[0].suffix_hint == "aryl_chromenylium"
    assert cls[0].surface_name == "2'-methylflavylium"


def test_classify_bare_chromenylium_does_not_fire() -> None:
    """The bare chromenylium (no 2-aryl) is left to retained-ring lookup."""
    cls = classify_charges(Chem.MolFromSmiles("c1cc2ccccc2[o+]c1"))
    # The classifier yields nothing — bare chromenylium has no 2-aryl.
    assert all(c.suffix_hint != "aryl_chromenylium" for c in cls)


def test_classify_skips_when_chromenylium_core_substituted() -> None:
    """Chromenylium with a ring-substituent (e.g. 3-methyl) + 2-phenyl
    must NOT match this classifier — the engine already names it
    ``3-methyl-2-phenylchromenylium`` correctly without dropping [O+]."""
    cls = classify_charges(
        Chem.MolFromSmiles("Cc1cc2ccccc2[o+]c1-c1ccccc1"),
    )
    assert all(c.suffix_hint != "aryl_chromenylium" for c in cls)


def test_classify_skips_when_aryl_is_naphthyl() -> None:
    """A 2-naphthyl (or other fused-ring) substituent is out of scope —
    OPSIN does not unambiguously round-trip it as a flavylium name."""
    cls = classify_charges(
        Chem.MolFromSmiles("c1ccc2cc(-c3ccc4ccccc4[o+]3)ccc2c1"),
    )
    assert all(c.suffix_hint != "aryl_chromenylium" for c in cls)


def test_classify_skips_neutral_2_phenyl_chromene() -> None:
    """Without the [O+] (neutral 2H-chromene) the classifier does not fire."""
    # 2-phenyl-2H-chromene has no formal charge — classifier returns nothing.
    cls = classify_charges(Chem.MolFromSmiles("c1ccc(-C2C=Cc3ccccc3O2)cc1"))
    assert all(c.suffix_hint != "aryl_chromenylium" for c in cls)


def test_classify_skips_when_aryl_has_two_methyls() -> None:
    """A doubly-methylated aryl ring is out of scope — out-of-bounds for
    the ``2'`` / ``3'`` / ``4'`` single-locant emission this classifier
    promises."""
    cls = classify_charges(
        Chem.MolFromSmiles("Cc1cc(C)c(-c2ccc3ccccc3[o+]2)cc1"),
    )
    assert all(c.suffix_hint != "aryl_chromenylium" for c in cls)


# ---------------------------------------------------------------------------
# End-to-end: name_smiles must emit the flavylium name AND OPSIN must
# round-trip back to the canonical input SMILES (charge preserved).
# ---------------------------------------------------------------------------


_FLAVYLIUM_PROBES: list[tuple[str, str]] = [
    # (input SMILES, expected emitted name)
    ("c1ccc(-c2ccc3ccccc3[o+]2)cc1", "flavylium"),
    ("Cc1ccc(-c2ccc3ccccc3[o+]2)cc1", "4'-methylflavylium"),
    ("Cc1cccc(-c2ccc3ccccc3[o+]2)c1", "3'-methylflavylium"),
    ("Cc1ccccc1-c1ccc2ccccc2[o+]1", "2'-methylflavylium"),
]


@pytest.mark.parametrize("smiles, expected_name", _FLAVYLIUM_PROBES)
def test_engine_emits_flavylium_name(smiles: str, expected_name: str) -> None:
    assert name_smiles(smiles) == expected_name


@pytest.mark.parametrize("smiles, expected_name", _FLAVYLIUM_PROBES)
def test_engine_output_round_trips_through_opsin(
    smiles: str, expected_name: str
) -> None:
    out_name = name_smiles(smiles)
    rt = _roundtrip(out_name)
    assert rt == _canon(smiles), (
        f"OPSIN round-trip failed: input={smiles!r} (canon={_canon(smiles)!r}) "
        f"-> name={out_name!r} -> opsin={rt!r}"
    )


# ---------------------------------------------------------------------------
# Non-regression: the cases the engine already handled correctly must
# continue to work byte-identically.
# ---------------------------------------------------------------------------


def test_bare_chromenylium_still_named_via_retained_lookup() -> None:
    """The bare chromenylium ([o+] in fused 6,6, no 2-substituent) was
    already named correctly via the retained-ring lookup; confirm no
    regression."""
    assert name_smiles("c1cc2ccccc2[o+]c1") == "chromenylium"


def test_3_methyl_2_phenylchromenylium_still_named_substitutively() -> None:
    """3-methyl-2-phenylchromenylium has a substituent on the chromenylium
    core itself; this case was already named correctly by the existing
    substitutive flow (NOT via the new classifier) and must remain so."""
    out = name_smiles("Cc1cc2ccccc2[o+]c1-c1ccccc1")
    assert out == "3-methyl-2-phenylchromenylium"
    # OPSIN round-trip the existing path's output too (sanity).
    rt = _roundtrip(out)
    assert rt == _canon("Cc1cc2ccccc2[o+]c1-c1ccccc1")
