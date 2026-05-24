"""Tests for Stage 16 R16-A-1: sulfinate anion suffix emission.

Pre-fix: ``CCS(=O)[O-]`` (audit row from opsin_audit_natural_raw.csv,
3-aminopropane-1-sulfinate variant) emitted ``1-(oxidosulfinyl)ethane``
which OPSIN reverses to ``CC[SH](=O)=O`` — the protonated sulfinic-acid
tautomer, NOT the input anion.  Two architectural gaps combined:

1. ``data/functional_groups.json::sulfinic_acid`` SMARTS matched only
   the protonated form (``[SX3](=O)[OX2H1]``), so the anion form
   ``[SX3](=O)[OX1-]`` wasn't detected as a suffix-eligible FG and the
   SUFFIX_VARIANT_TABLE's ``sulfinic acid`` → ``sulfinate`` mapping never
   fired.

2. ``engine.py::name_smiles`` STANDALONE→ANION promotion gate at the
   top-level dispatch fired ONLY for zwitterions (``net_charge == 0``
   with both pos+neg atoms).  A naked single-fragment anion
   (``net_charge < 0``, no positive atoms) fell through to STANDALONE
   and the engine emitted the prefix-only form.

R16-A-1 widens both:
- Sulfinic_acid SMARTS now matches both forms ([OX2H1,OX1-]).
- The promotion gate now triggers on ``net_charge <= 0`` whenever a
  negatively-charged atom belongs to a suffix-eligible anion-variant
  FG.
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
    td = tempfile.mkdtemp(prefix="r16a1_")
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


def test_audit_3_aminopropane_1_sulfinate_roundtrips() -> None:
    """The exact audit row: ``NCCCS(=O)[O-]``."""
    smi = "NCCCS(=O)[O-]"
    name = name_smiles(smi)
    assert name == "3-aminopropane-1-sulfinate", f"got {name!r}"
    rt = _opsin_rt(name)
    assert _canon(rt) == _canon(smi)


def test_ethanesulfinate_roundtrips() -> None:
    smi = "CCS(=O)[O-]"
    name = name_smiles(smi)
    assert name == "ethanesulfinate", f"got {name!r}"
    rt = _opsin_rt(name)
    assert _canon(rt) == _canon(smi)


def test_sulfinic_acid_protonated_unaffected() -> None:
    """Control: protonated sulfinic acid still emits the acid form."""
    name = name_smiles("CCS(=O)O")
    assert name == "ethanesulfinic acid", f"got {name!r}"


def test_sulfonate_anion_unaffected() -> None:
    """Control: sulfonate (R-SO3-) was already round-tripping via the
    prefix-only ``oxidosulfonyl`` form.  After R16-A-1's promotion gate
    widens to net_charge <= 0, sulfonate may now go through the suffix
    path too — either way it should still round-trip."""
    smi = "CCS(=O)(=O)[O-]"
    name = name_smiles(smi)
    rt = _opsin_rt(name) if name else None
    assert rt is not None and _canon(rt) == _canon(smi), f"got {name!r}"


def test_carboxylate_anion_unaffected() -> None:
    """Control: ``CC(=O)[O-]`` (acetate) — was emitting
    ``1-(oxido)-1-oxoethane`` via prefix path.  Should still round-trip."""
    smi = "CC(=O)[O-]"
    name = name_smiles(smi)
    rt = _opsin_rt(name) if name else None
    assert rt is not None and _canon(rt) == _canon(smi)


def test_zwitterion_promotion_unaffected() -> None:
    """Control: zwitterion β-alanine (``[NH3+]CCC(=O)[O-]``) was already
    promoted to ANION and emits ``3-(azaniumyl)propanoate``."""
    name = name_smiles("[NH3+]CCC(=O)[O-]")
    assert name is not None and "propanoate" in name and "azaniumyl" in name, (
        f"got {name!r}"
    )


def test_sodium_acetate_salt_unaffected() -> None:
    """Control: salt ``CC(=O)[O-].[Na+]`` (sodium acetate) — multi-frag
    salt path uses its own dispatch, should be unchanged."""
    smi = "CC(=O)[O-].[Na+]"
    name = name_smiles(smi)
    rt = _opsin_rt(name) if name else None
    assert rt is not None and _canon(rt) == _canon(smi)
