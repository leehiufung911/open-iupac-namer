"""Regression tests for P-72 / P-73 polycharged & zwitterionic species.

Closes the WRONG / NAMING_ERROR gap for four structural families handled by
``iupac_namer.perception.charge_perception``:

  (A) carbanion zwitterion  — senior ``[C-]`` ``-ide`` suffix + onium cation
      expressed as a ``...ammoniumyl`` substituent prefix
      (``C[N+](C)(C)[C-](C)C`` -> ``2-(trimethylammoniumyl)propan-2-ide``);
  (B) substituted single-carbon dianion
      (``[C-2](c1ccccc1)c1ccccc1`` -> ``diphenylmethanediide``);
  (C) di-charged 2-nitrogen backbone (diazene-diium / diazane-diide)
      (``C[N+](C)=[N+](C)C`` -> ``tetramethyldiazene-1,2-diium``;
       ``CC(=O)N[N-2]`` -> ``acetyldiazane-1,1-diide``);
  (D) ring carbanion dianion on a monocyclic carbocycle
      (``[c-]1cc[c-]cc1`` -> ``benzene-1,4-diide``).

Every probe is checked end-to-end: the engine emits a name and OPSIN
round-trips that name back to the input's canonical SMILES.  The round-trip is
the load-bearing guarantee — exact surface drift is tolerated as long as the
structure is preserved (P-73/P-74 closed-shell only; the free-valence guard is
never weakened, so genuinely open-valence dianions like ``[C-2]c1ccccc1`` stay
rejected and are out of scope).
"""
from __future__ import annotations

import os
import shutil
import tempfile

import pytest
from rdkit import Chem
from py2opsin import py2opsin

from iupac_namer.engine import name_smiles
from iupac_namer.perception.charge_perception import classify_charges


def _opsin_rt(name: str) -> str | None:
    """OPSIN round-trip in an isolated temp dir (avoids the py2opsin temp-file
    race when tests run concurrently with the eval harness)."""
    if not name:
        return None
    td = tempfile.mkdtemp(prefix="pcz_")
    cwd = os.getcwd()
    try:
        os.chdir(td)
        return py2opsin(name, allow_radicals=True)
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


# ---------------------------------------------------------------------------
# Exact surface name pins (the four flagship gap cases)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("smi,expected", [
    ("C[N+](C)(C)[C-](C)C",        "2-(trimethylammoniumyl)propan-2-ide"),
    ("[C-2](c1ccccc1)c1ccccc1",    "diphenylmethanediide"),
    ("C[N+](C)=[N+](C)C",          "tetramethyldiazene-1,2-diium"),
    ("CC(=O)N[N-2]",               "acetyldiazane-1,1-diide"),
    # ylidene-substituted onium cation prefix (double-bond carving +
    # escalated bracket nesting, P-16.3.3)
    ("CC(C)=[N+](C)[C-](C)C",
     "2-[methyl(propan-2-ylidene)ammoniumyl]propan-2-ide"),
])
def test_flagship_surface_names(smi: str, expected: str) -> None:
    assert name_smiles(smi) == expected


# ---------------------------------------------------------------------------
# End-to-end OPSIN round-trip (broad structural coverage, >=20 cases)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("smi", [
    # (A) carbanion zwitterions — vary parent length and cation substituents
    "C[N+](C)(C)[C-](C)C",
    "C[N+](C)(C)[CH-]C",
    "CC[N+](CC)(CC)[C-](C)C",
    "C[N+](C)(C)[CH-]CCCC",
    "C[N+](C)(C)[CH-]CC",
    # ylidene / oxonium onium-cation prefixes (double-bond substituent carving)
    "CC(C)=[N+](C)[C-](C)C",
    "CC(C)=[O+][C-](C)C",
    # (B) substituted single-carbon dianions
    "[C-2](c1ccccc1)c1ccccc1",
    "[C-2](C)c1ccccc1",
    "[C-2](C)C",
    "[C-2](CC)CC",
    # (C) diazene-diium
    "C[N+](C)=[N+](C)C",
    "CC[N+](CC)=[N+](CC)CC",
    "C[N+](CC)=[N+](C)CC",
    # (C) diazane-diide
    "CC(=O)N[N-2]",
    "[N-2]N",
    "[N-2]NC",
    "[N-2]Nc1ccccc1",
    "[N-2]NCC",
    # (D) ring carbanion dianion (monocyclic)
    "[CH-]1CC[CH-]CC1",
    "[c-]1cc[c-]cc1",
    "[CH-]1[CH-]CCCC1",
    # all-carbon polycharge (regression — pre-existing path must still hold)
    "[CH2-]C[CH2-]",
    "[CH2+]C[CH2+]",
    "[CH2-2]",
])
def test_polycharge_round_trips_through_opsin(smi: str) -> None:
    name = name_smiles(smi)
    assert name and "NAMING ERROR" not in name, f"no name for {smi!r}: {name!r}"
    assert _canon(_opsin_rt(name)) == _canon(smi), (
        f"round-trip failed for {smi!r}: name={name!r} "
        f"opsin={_canon(_opsin_rt(name))!r} expected={_canon(smi)!r}"
    )


# ---------------------------------------------------------------------------
# Negative gating — must NOT claim (no charge dropping, no over-reach)
# ---------------------------------------------------------------------------


def test_zwitterion_requires_direct_cation_anion_bond() -> None:
    """A cation one carbon away from the carbanion is NOT a directly-bonded
    onium zwitterion; the classifier must not claim it (so it does not emit a
    wrong name)."""
    mol = Chem.MolFromSmiles("[C-](C)(C)C[N+](C)(C)C")
    cls = classify_charges(mol)
    assert all(c.suffix_hint != "carbanion_zwitterion" for c in cls)


def test_beta_alanine_zwitterion_unaffected() -> None:
    """Control: the FG-anion zwitterion β-alanine still routes through the
    engine's STANDALONE→ANION promotion (carboxylate suffix + azaniumyl
    prefix), NOT the carbanion-zwitterion classifier."""
    name = name_smiles("[NH3+]CCC(=O)[O-]")
    assert name and "propanoate" in name and "azaniumyl" in name, f"got {name!r}"


def test_quaternary_ammonium_unaffected() -> None:
    """Control: a plain quaternary ammonium cation (no anion) still names as
    the ammonium parent."""
    name = name_smiles("C[N+](C)(C)C")
    assert name == "tetramethylazanium" or "ammonium" in name, f"got {name!r}"


def test_open_valence_carbon_dianion_stays_rejected() -> None:
    """Control: ``[C-2]c1ccccc1`` carries a free valence (RDKit gives the
    bare C a radical electron); the free-valence guard must still reject it —
    we never weaken that guard for genuine open-shell species."""
    with pytest.raises(Exception):
        name_smiles("[C-2]c1ccccc1")
