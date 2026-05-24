"""Tests for Stage 18 R18-A: group-13/14/15 heavy-atom substituent prefixes.

Pre-fix: Methyl/alkyl-substituted forms of group-13/14/15/16 parent
hydrides (methylbismuthane, methylplumbane, methylsilane, etc.)
emitted ``1-{[NAMING ERROR: ...]}methane`` because the
``_SINGLE_ATOM_SUBSTITUENT`` table at ``engine.py:56`` covered only
``("S",-1,1):"sulfanide"``-class organic substituents and
``("P",0,1):"phosphanyl"`` for P; the heavier group-13/14/15 elements
(B, Si, Ge, Sn, Pb, As, Sb, Bi) had no neutral-substituent entry, so
the methane parent path tried to recursively name the (e.g.) ``[BiH2]-``
fragment as a substituent and failed.

R18-A adds the missing entries:
- ``("As", 0, 1): "arsanyl"``
- ``("Sb", 0, 1): "stibanyl"``
- ``("Bi", 0, 1): "bismuthanyl"``
- ``("Si", 0, 1): "silyl"``
- ``("Ge", 0, 1): "germyl"``
- ``("Sn", 0, 1): "stannyl"``
- ``("Pb", 0, 1): "plumbyl"``
- ``("B",  0, 1): "boryl"``

Each entry is the standard substituent prefix per IUPAC P-66.5.1.2 /
P-21.1.3 retained-substituent table; OPSIN parses each form back to
the same SMILES, so methylbismuthane / phenylgermane / etc. now
round-trip cleanly.

The aryl-on-Bi case ``[BiH2][c]1ccccc1`` (audit row from
opsin_audit_substituents_raw.csv — phenylbismuthane) is still WRONG
(``(bismuthanyl)cyclohexane`` instead of ``(bismuthanyl)benzene``) —
the aryl-aromatic ring detection path doesn't currently handle the
``[c]``-no-implicit-H atom that arises from RDKit's canonicalisation
when a heavy substituent fills the C valence.  Deferred to Stage 18+
ring-aromaticity work.
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
    td = tempfile.mkdtemp(prefix="r18a_")
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
    assert name is not None and "NAMING ERROR" not in name, f"naming err: {name!r}"
    rt = _opsin_rt(name)
    assert rt is not None, f"OPSIN rejected {name!r}"
    assert _canon(rt) == _canon(smi), (
        f"rt mismatch for {smi!r}: emit={name!r} → rt={_canon(rt)!r}"
    )
    return name


def test_methylbismuthane_roundtrips() -> None:
    name = _assert_roundtrips("C[BiH2]")
    assert "bismuthanyl" in name


def test_methylplumbane_roundtrips() -> None:
    name = _assert_roundtrips("C[PbH3]")
    assert "plumbyl" in name


def test_methylstannane_roundtrips() -> None:
    name = _assert_roundtrips("C[SnH3]")


def test_methylarsane_roundtrips() -> None:
    name = _assert_roundtrips("C[AsH2]")


def test_methylstibane_roundtrips() -> None:
    name = _assert_roundtrips("C[SbH2]")
    assert "stibanyl" in name


def test_methylsilane_roundtrips() -> None:
    name = _assert_roundtrips("C[SiH3]")


def test_methylgermane_roundtrips() -> None:
    name = _assert_roundtrips("C[GeH3]")


def test_phenylgermane_roundtrips() -> None:
    name = _assert_roundtrips("[GeH3]c1ccccc1")


def test_phenylsilane_roundtrips() -> None:
    name = _assert_roundtrips("[SiH3]c1ccccc1")
