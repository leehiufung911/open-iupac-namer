"""Tests for Stage 19 R19-A: heteroatom-chain candidate-generator
extension to the group-13/14/15 elements (P, As, Sb, Bi, Si, Ge, Sn,
Pb).

Pre-fix: ``perception/__init__.py::candidate_parents`` Section 4
(heteroatom-chain candidates) only iterated over ``{N, S, O, Se, Te}``.
Bare 2-atom chains of the heavier group-13/14/15 elements (e.g.
``[BiH2][BiH2]`` dibismuthane, ``[SbH2][SbH2]`` distibane) had no
candidate parent and emitted NAMING_ERROR.

R19-A:
- Extends ``_HETEROATOM_CHAIN_ELEMENTS`` and ``_VALID_HOMO_PAIRS`` in
  perception to include P/As/Sb/Bi/Si/Ge/Sn/Pb.
- Extends ``engine.py::_HETEROATOM_CHAIN_NAMES`` (the chain-parent
  name lookup) with the corresponding dimer-hydride retained names:
  diphosphane, diarsane, distibane, dibismuthane, disilane, digermane,
  distannane, diplumbane.

Bare-element parent hydrides (``[AsH3]`` arsane, ``[SiH4]`` silane,
etc.) that already exist in ``_INORGANIC_CURATED_SMILES`` continue to
take precedence; the dimer chain is only used when no single-atom
parent is available.  As a result the As/Si/Ge/Sn dimers may emit the
``(R)R`` substituent form (``(arsanyl)arsane``) instead of the dimer
name (``diarsane``); both round-trip cleanly through OPSIN.
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
    td = tempfile.mkdtemp(prefix="r19a_")
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
    assert name is not None and "NAMING ERROR" not in name
    rt = _opsin_rt(name)
    assert rt is not None and _canon(rt) == _canon(smi), (
        f"rt mismatch: {smi!r} → name={name!r} → rt={_canon(rt)!r}"
    )
    return name


def test_dibismuthane_roundtrips() -> None:
    name = _assert_roundtrips("[BiH2][BiH2]")
    assert name == "dibismuthane"


def test_distibane_roundtrips() -> None:
    name = _assert_roundtrips("[SbH2][SbH2]")
    assert name == "distibane"


def test_diplumbane_roundtrips() -> None:
    name = _assert_roundtrips("[PbH3][PbH3]")
    assert name == "diplumbane"


def test_diarsane_roundtrips() -> None:
    """The As-As chain — emits ``(arsanyl)arsane`` (substituent form
    via the existing arsane parent-hydride) which OPSIN round-trips
    to the same SMILES as ``diarsane``.
    """
    _assert_roundtrips("[AsH2][AsH2]")


def test_distannane_roundtrips() -> None:
    _assert_roundtrips("[SnH3][SnH3]")


def test_disilane_roundtrips() -> None:
    _assert_roundtrips("[SiH3][SiH3]")


def test_digermane_roundtrips() -> None:
    _assert_roundtrips("[GeH3][GeH3]")


def test_diphosphane_roundtrips() -> None:
    _assert_roundtrips("PP")


def test_existing_chains_unaffected() -> None:
    """Controls: hydrazine / disulfane / hydrogen peroxide / diselane /
    ditellane unchanged."""
    assert name_smiles("NN") == "hydrazine"
    assert name_smiles("SS") == "disulfane"
    assert name_smiles("OO") == "hydrogen peroxide"
    assert name_smiles("[SeH][SeH]") == "diselane"
    assert name_smiles("[TeH][TeH]") == "ditellane"
