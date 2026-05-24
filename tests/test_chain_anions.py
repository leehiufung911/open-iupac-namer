"""Tests for Stage 17 R17-B: heteroatom-chain anion retained names.

Pre-fix: ``[Se-][Se][SeH].[Na+]`` (audit row from
opsin_audit_fg_raw.csv — sodium triselan-1-ide) emitted
``sodium(1+) 1-{[NAMING ERROR: No valid naming plan found for [SeH-]]}diselane``
because the engine's ``perception/__init__.py::candidate_parents``
heteroatom_chain generator (Section 4) only handled NEUTRAL 2-atom
chains.  Charged endpoints (the [Se-]) fell through to a path that
tries to carve the [SeH-] as a substituent of a 2-atom diselane parent
and fails for the same reason.

R17-B adds curated retained-name entries for the small chain anions
to ``_INORGANIC_CURATED_SMILES``: ``triselan-1-ide``, ``diselan-1-ide``,
``disulfan-1-ide``, ``trisulfan-1-ide``, ``hydrazinide``.  Larger
N-atom chain anions still require the candidate-generator extension
(deferred — same shape, just generalising the chain length).
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
    td = tempfile.mkdtemp(prefix="r17b_")
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


def test_audit_sodium_triselanide_roundtrips() -> None:
    """The exact audit row: ``[Se-][Se][SeH].[Na+]`` → sodium triselan-1-ide.

    Per IUPAC P-77 the spec PIN omits the explicit "(1+)" charge marker
    on the cation when its charge is unambiguous from the element
    (alkali / alkaline-earth / aluminium); both forms round-trip.
    """
    smi = "[Se-][Se][SeH].[Na+]"
    name = name_smiles(smi)
    assert name in ("sodium triselan-1-ide", "sodium(1+) triselan-1-ide"), (
        f"got {name!r}"
    )
    assert _canon(_opsin_rt(name)) == _canon(smi)


def test_triselanide_bare_roundtrips() -> None:
    smi = "[Se-][Se][SeH]"
    name = name_smiles(smi)
    assert name == "triselan-1-ide"
    assert _canon(_opsin_rt(name)) == _canon(smi)


def test_diselanide_roundtrips() -> None:
    smi = "[Se-][SeH]"
    name = name_smiles(smi)
    assert name == "diselan-1-ide"
    assert _canon(_opsin_rt(name)) == _canon(smi)


def test_disulfanide_roundtrips() -> None:
    smi = "[S-]S"
    name = name_smiles(smi)
    assert name == "disulfan-1-ide"
    assert _canon(_opsin_rt(name)) == _canon(smi)


def test_trisulfanide_roundtrips() -> None:
    smi = "[S-]SS"
    name = name_smiles(smi)
    assert name == "trisulfan-1-ide"
    assert _canon(_opsin_rt(name)) == _canon(smi)


def test_hydrazinide_roundtrips() -> None:
    smi = "[NH-]N"
    name = name_smiles(smi)
    assert name == "hydrazinide"
    assert _canon(_opsin_rt(name)) == _canon(smi)


def test_unisotoped_chains_unaffected() -> None:
    """Controls: neutral chain hydrides remain unchanged."""
    assert name_smiles("NN") == "hydrazine"
    assert name_smiles("SS") == "disulfane"
    assert name_smiles("[SeH][SeH]") == "diselane"
