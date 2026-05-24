"""Tests for Stage 22 R22-B: recursive aryl-on-heavy-element substituent naming.

Pre-fix: ``c1cc[c]([PbH2][c]2ccccc2)cc1`` (audit row — diphenylplumbane)
emitted ``{[NAMING ERROR: No valid naming plan found for [PbH3][c]1ccccc1]}benzene``.
The carved substituent was ``[PbH3][c]1ccccc1`` (Pb with FV, aryl group
attached) and the plan search had no recipe for naming a Pb/Bi/Sb-rooted
substituent fragment because the heteroatom_center parent allowlist only
covered ``P, Si, B, As, Ge, Sn``.  The analogous Si/Sn/Ge/As cases worked
already because those elements were registered as heteroatom_center parents.

R22-B extends the heteroatom_center parent allowlist with ``Bi, Sb, Pb``
(matching the parent-hydride name table at engine.py: ``bismuthane``,
``stibane``, ``plumbane``), gated on ``ring_systems`` being present in the
interpretation.  The ring-presence gate preserves R18-A test guards:

- ``C[BiH2]`` (no ring) → carbon chain wins → ``1-(bismuthanyl)methane``
  (R18-A form), NOT ``methylbismuthane``.
- ``[BiH2][BiH2]`` (no ring) → heteroatom_chain wins → ``dibismuthane``,
  NOT ``(bismuthanyl)bismuthane``.
- ``[PbH3]c1ccccc1`` (with ring) → ring beats heteroatom_center → still
  ``(plumbyl)benzene`` (R21-A form).
- Carved ``[PbH3][c]1ccccc1`` substituent (FV at Pb, ring present) →
  Pb heteroatom_center fires → ``phenylplumbanyl`` substituent text.

Closes the diphenyl-Pb/Bi/Sb audit-row family.  Each emitted name
round-trips through OPSIN to the original canonical SMILES.
"""
from __future__ import annotations

import os
import shutil
import tempfile

from rdkit import Chem
from py2opsin import py2opsin

from iupac_namer.engine import name_smiles


def _opsin_rt(name: str) -> str | None:
    if not name:
        return None
    td = tempfile.mkdtemp(prefix="r22b_")
    cwd = os.getcwd()
    try:
        os.chdir(td)
        return py2opsin(name)
    except Exception:
        return None
    finally:
        os.chdir(cwd)
        try:
            shutil.rmtree(td)
        except Exception:
            pass


def _canon(s: str | None) -> str | None:
    m = Chem.MolFromSmiles(s) if s else None
    return Chem.MolToSmiles(m) if m else None


def _assert_roundtrips(smi: str) -> str:
    name = name_smiles(smi)
    assert name is not None and "NAMING ERROR" not in name, (
        f"naming err for {smi!r}: {name!r}"
    )
    rt = _opsin_rt(name)
    assert rt is not None, f"OPSIN rejected {name!r}"
    assert _canon(rt) == _canon(smi), (
        f"rt mismatch for {smi!r}: emit={name!r} -> rt={_canon(rt)!r}"
    )
    return name


# --- Audit row: diphenyl-Pb ---


def test_diphenylplumbane_audit_row_roundtrips() -> None:
    """The audit row that motivated R22-B."""
    name = _assert_roundtrips("c1cc[c]([PbH2][c]2ccccc2)cc1")
    # The substituent prefix should incorporate ``plumb`` (either as
    # ``plumbyl`` or ``plumbanyl``).  We don't pin the exact phrasing
    # because the heteroatom-parent path produces ``plumbanyl`` and
    # the bare prefix would be ``plumbyl``; both are OPSIN-valid.
    assert "plumb" in name, f"expected plumb-prefix in {name!r}"


def test_diphenylbismuthane_extension_roundtrips() -> None:
    name = _assert_roundtrips("c1cc[c]([BiH][c]2ccccc2)cc1")
    assert "bismuth" in name


def test_diphenylstibane_extension_roundtrips() -> None:
    name = _assert_roundtrips("c1cc[c]([SbH][c]2ccccc2)cc1")
    assert "stib" in name


# --- R21-A guards: standalone aryl-on-heavy-element ---


def test_phenylplumbane_standalone_unchanged() -> None:
    """R21-A path: standalone ``[PbH3][c]1ccccc1`` still emits
    ``(plumbyl)benzene`` (ring beats heteroatom_center)."""
    name = name_smiles("[PbH3][c]1ccccc1")
    assert name == "(plumbyl)benzene", f"R21-A regression: got {name!r}"


def test_phenylbismuthane_standalone_unchanged() -> None:
    name = name_smiles("[BiH2][c]1ccccc1")
    assert name == "(bismuthanyl)benzene", f"R21-A regression: got {name!r}"


def test_phenylstibane_standalone_unchanged() -> None:
    name = name_smiles("[SbH2][c]1ccccc1")
    assert name == "(stibanyl)benzene", f"R21-A regression: got {name!r}"


# --- R18-A guards: methyl-on-heavy-element (no ring → no R22-B activation) ---


def test_methylplumbane_r18a_form_preserved() -> None:
    """No ring system in molecule → Pb heteroatom_center NOT emitted →
    methane chain wins → ``1-(plumbyl)methane``-class name (R18-A form)."""
    name = name_smiles("C[PbH3]")
    assert "plumbyl" in name, (
        f"R22-B over-activation broke R18-A guard: {name!r}"
    )
    # Sanity: round-trip
    rt = _opsin_rt(name)
    assert _canon(rt) == _canon("C[PbH3]"), f"rt mismatch: {name!r}"


def test_methylbismuthane_r18a_form_preserved() -> None:
    name = name_smiles("C[BiH2]")
    assert "bismuthanyl" in name, (
        f"R22-B over-activation broke R18-A guard: {name!r}"
    )
    rt = _opsin_rt(name)
    assert _canon(rt) == _canon("C[BiH2]"), f"rt mismatch: {name!r}"


# --- Dimer guards (heteroatom_chain): no ring → no R22-B activation ---


def test_diplumbane_dimer_unchanged() -> None:
    """No ring system → Pb heteroatom_center NOT emitted →
    heteroatom_chain wins → ``diplumbane``."""
    name = name_smiles("[PbH3][PbH3]")
    assert name == "diplumbane", f"R22-B over-activation: got {name!r}"


def test_dibismuthane_dimer_unchanged() -> None:
    name = name_smiles("[BiH2][BiH2]")
    assert name == "dibismuthane", f"R22-B over-activation: got {name!r}"


def test_distibane_dimer_unchanged() -> None:
    name = name_smiles("[SbH2][SbH2]")
    assert name == "distibane", f"R22-B over-activation: got {name!r}"
