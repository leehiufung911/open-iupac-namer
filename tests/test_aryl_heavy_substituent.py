"""Tests for Stage 21 R21-A: aryl ring with heavy-atom substituent
(``[BiH2][c]1ccccc1`` — phenylbismuthane and similar).

Pre-fix: ``[BiH2][c]1ccccc1`` (audit row from
opsin_audit_substituents_raw.csv) emitted ``(bismuthanyl)cyclohexane``
— silently dropping the aromatic-ring perception when the carved ring
contained a ``[c]`` (no-implicit-H aromatic C, where the heavy
substituent had attached).

Root cause: ``ring_naming/common.py::_normalize_nh_fragment`` parsed
the carved ring fragment ``c1c[c]ccc1`` directly via ``MolFromSmiles``;
RDKit preserves the ``noImplicit`` property on the bracketed ``[c]``
atom, so the canonical SMILES of the extracted ring became
``[c]1ccccc1`` (NOT ``c1ccccc1``).  The retained-name lookup keys on
``c1ccccc1`` and missed; ``name_systematic_monocyclic`` then took the
"fully saturated" branch (no explicit double bonds in aromatic rings)
and emitted ``cyclohexane``.

Phenylsilane (``[SiH3]c1ccccc1``) doesn't trigger this — RDKit
canonicalises the Si-attached aromatic C as ``c`` (no brackets).
The bracket-vs-no-bracket asymmetry on aromatic ring atoms is an
RDKit canonicalisation idiosyncrasy that tracks with the substituent
element family.

R21-A: rewrite ``[c]`` → ``c`` (and ``[C]`` → ``C``) in the carved
ring fragment SMILES before the parse.  The rewrite gives RDKit a
chance to re-add the implicit H on the formerly-substituted aromatic
C, restoring the canonical match against the curated benzene entry.
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
    td = tempfile.mkdtemp(prefix="r21a_")
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


def test_phenylbismuthane_roundtrips() -> None:
    """The exact audit row: ``[BiH2][c]1ccccc1``."""
    smi = "[BiH2][c]1ccccc1"
    name = name_smiles(smi)
    assert name == "(bismuthanyl)benzene", f"got {name!r}"
    rt = _opsin_rt(name)
    assert _canon(rt) == _canon(smi)


def test_phenylsilane_unaffected() -> None:
    """Control: phenylsilane was already working via the standard path
    (RDKit canonicalises Si-attached aromatic C as ``c``, no brackets,
    so the [c] normalisation is a no-op for this case)."""
    name = name_smiles("[SiH3]c1ccccc1")
    assert name == "(silyl)benzene", f"got {name!r}"


def test_benzene_unaffected() -> None:
    """Control: bare benzene unchanged."""
    assert name_smiles("c1ccccc1") == "benzene"


def test_naphthalene_unaffected() -> None:
    """Control: naphthalene (fused aromatic, junctions have 0 H natively)
    unchanged — the [c] normalisation only affects atoms that lost
    a substituent during carving, not native 0-H junctions which already
    use lowercase c in canonical form.
    """
    assert name_smiles("c1ccc2ccccc2c1") == "naphthalene"


def test_cyclohexane_unaffected() -> None:
    """Control: actual cyclohexane (saturated) still emits cyclohexane."""
    assert name_smiles("C1CCCCC1") == "cyclohexane"
