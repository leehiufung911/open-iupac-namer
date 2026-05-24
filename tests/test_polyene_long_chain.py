"""Regression tests for long-chain polyene multiplier rendering.

Stage 8 diagnostic: the audit refresh flagged lycopene (13 conjugated
double bonds) as emitting a malformed suffix ``...30-13aene``.  The
root cause was an out-of-range fallback in
``iupac_namer/perception/chains.py::_multiplier``: the local lookup
table only covered counts 2-10 (di/tri/.../deca) and fell back to
``f"{count}a"`` for larger chains — so 13 double bonds rendered as
``13aene`` instead of ``tridecaene``.

Fix: ``_multiplier`` now delegates to ``data_loader.get_multiplier``,
which has the full table (undeca, dodeca, trideca, tetradeca,
pentadeca, ...).

These tests pin a synthetic "open-chain polyene" form for each count
from 11 to 20 so any regression in the lookup path surfaces immediately.
``name_smiles`` is invoked with a plain ``C=C=C=...`` (cumulated double
bonds) skeleton; the engine's parent picks up the single longest chain
with N double bonds, so the rendered suffix is exactly the multiplier
prefix + ``ene``.
"""
from __future__ import annotations

import pytest

from iupac_namer.engine import name_smiles


# (count_of_double_bonds, expected_multiplier_prefix)
# Expected suffix on the emitted name is f"{mult}ene" (i.e. tridecaene
# for 13, tetradecaene for 14, etc.).  The chain-stem parent name
# adjusts separately — we only assert the suffix substring.
_MULTIPLIER_CASES = [
    (11, "undecaene"),
    (12, "dodecaene"),
    (13, "tridecaene"),
    (14, "tetradecaene"),
    (15, "pentadecaene"),
    (16, "hexadecaene"),
    (17, "heptadecaene"),
    (18, "octadecaene"),
    (19, "nonadecaene"),
    (20, "icosaene"),
]


@pytest.mark.parametrize("count,multiplier_suffix", _MULTIPLIER_CASES)
def test_polyene_multiplier_renders_full_word(count: int, multiplier_suffix: str) -> None:
    """A chain with N cumulated double bonds renders ``{multiplier}ene``.

    Regression guard for the chains._multiplier fallback bug — the
    failing form previously rendered as e.g. ``13aene`` for count==13.
    """
    # Construct C=C=C... with `count` double bonds — chain length count+1.
    smi = "C" + "=C" * count
    name = name_smiles(smi)
    assert name is not None, f"engine returned None for count={count} skel={smi!r}"
    assert multiplier_suffix in name, (
        f"for count={count}, expected {multiplier_suffix!r} in emitted "
        f"name but got {name!r}"
    )
    # Also guard the numeric-fallback form (e.g. "13aene") never appears.
    bad_form = f"{count}a"
    assert bad_form + "ene" not in name, (
        f"engine emitted numeric-fallback form {bad_form+'ene'!r} in {name!r}"
    )


def test_lycopene_renders_tridecaene() -> None:
    """Lycopene (11 configured + 2 unconfigured double bonds = 13 total).

    This is the original audit probe that surfaced the bug.  We don't
    assert the full name here (stereo + methyl substituent locants are
    not the point of this test) — just that the 13-ene suffix renders
    as ``tridecaene`` and not as the broken ``13aene`` fallback.
    """
    # Canonical SMILES for lycopene (C40H56) — 13 conjugated/isolated C=C
    lycopene = (
        r"CC(C)=CCC/C(C)=C/C=C/C(C)=C/C=C/C(C)=C/C=C/C=C(C)"
        r"/C=C/C=C(C)/C=C/C=C(C)/CC/C=C(C)/C"
    )
    name = name_smiles(lycopene)
    assert name is not None
    assert "tridecaene" in name, f"expected 'tridecaene' in {name!r}"
    assert "13aene" not in name, f"broken form '13aene' present in {name!r}"
