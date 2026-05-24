"""Tests for the main-group oxoacid ESTER namer (P-65.6.3 / P-67.1.3.2).

Exercises ``maingroup_oxoacids.compute_oxoacid_ester_name`` and its engine
dispatch hook.  Esters of the mononuclear noncarbon oxoacids are named by
functional-class nomenclature:

    <organyl word(s)>  [<hydrogen word(s)>]  <acid-anion stem>

The generator self-gates structurally (single main-group centre; only =O,
-OH, [O-], -O-R organyl esters, direct -C substituents, centre-H; >=1 -O-R
ester group), so substituted/anhydride/halido/amido/thio/polynuclear/
charged/multi-fragment inputs fall through untouched.

Every expected name was verified to round-trip through OPSIN 2.8.0
(name -> SMILES -> same RDKit canonical) when the namer was built; see
P-67.1.3.2 worked examples for the IUPAC citations.
"""

from __future__ import annotations

import pytest
from rdkit import Chem

from iupac_namer.engine import name_smiles
from iupac_namer.perception.fg.maingroup_oxoacids import (
    compute_oxoacid_ester_name,
    compute_name,
)


def _name(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None, f"bad test SMILES {smi!r}"
    return compute_oxoacid_ester_name(mol)


# ---------------------------------------------------------------------------
# Fully esterified mononuclear acids (P-67.1.3.2)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smi,expected", [
    ("COP(=O)(OC)OC", "trimethyl phosphate"),
    ("CCOP(=O)(OCC)OCC", "triethyl phosphate"),
    ("CCCOP(=O)(OCCC)OCCC", "tripropyl phosphate"),
    ("COP(OC)OC", "trimethyl phosphite"),
    ("CCOP(OCC)OCC", "triethyl phosphite"),
    ("COS(=O)(=O)OC", "dimethyl sulfate"),
    ("CCOS(=O)(=O)OCC", "diethyl sulfate"),
    ("COS(=O)OC", "dimethyl sulfite"),
    ("CCOS(=O)OC", "ethyl methyl sulfite"),
    ("COB(OC)OC", "trimethyl borate"),
    ("CCOB(OCC)OCC", "triethyl borate"),
    ("CO[Se](=O)(=O)OC", "dimethyl selenate"),
    ("O=P(Oc1ccccc1)(Oc1ccccc1)Oc1ccccc1", "triphenyl phosphate"),
])
def test_full_esters(smi: str, expected: str) -> None:
    assert _name(smi) == expected


# ---------------------------------------------------------------------------
# Mixed esters — different organyls cited as separate words, alphanumeric
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smi,expected", [
    ("CCOS(=O)(=O)OC", "ethyl methyl sulfate"),
    ("CCOP(=O)(OC)Oc1ccccc1", "ethyl methyl phenyl phosphate"),
    ("CCOP(=O)(OC)OC", "ethyl dimethyl phosphate"),
    ("CC(C)(C)OP(=O)(OC)OC", "dimethyl (2-methylpropan-2-yl) phosphate"),
    ("C=CCOP(=O)(OCC=C)OCC=C", "tris(prop-2-en-1-yl) phosphate"),
])
def test_mixed_esters(smi: str, expected: str) -> None:
    assert _name(smi) == expected


# ---------------------------------------------------------------------------
# Phosphonate / phosphinate / phosphonite — direct-C substituent on the
# centre becomes a prefix glued to the anion stem
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smi,expected", [
    ("COP(C)(=O)OC", "dimethyl methylphosphonate"),
    ("CCOP(C)(=O)OCC", "diethyl methylphosphonate"),
    ("CCP(=O)(OC)OC", "dimethyl ethylphosphonate"),
    ("COP(=O)(OC)c1ccccc1", "dimethyl phenylphosphonate"),
    ("CO[PH](=O)OC", "dimethyl phosphonate"),           # P-H tier-2 form
    ("COP(C)(C)=O", "methyl dimethylphosphinate"),
    ("CCOP(=O)(c1ccccc1)c1ccccc1", "ethyl diphenylphosphinate"),
    ("COP(OC)c1ccccc1", "dimethyl phenylphosphonite"),   # tier-2, no oxo
])
def test_phosphonate_phosphinate(smi: str, expected: str) -> None:
    assert _name(smi) == expected


# ---------------------------------------------------------------------------
# Partial esters of polybasic acids — free -OH denoted by "hydrogen"
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smi,expected", [
    ("COP(=O)(O)O", "methyl dihydrogen phosphate"),
    ("COP(=O)(O)OC", "dimethyl hydrogen phosphate"),
    ("COS(=O)(=O)O", "methyl hydrogen sulfate"),
    ("CCCCOS(=O)(=O)O", "butyl hydrogen sulfate"),
    ("CCOP(OC)O", "ethyl methyl hydrogen phosphite"),
])
def test_partial_esters(smi: str, expected: str) -> None:
    assert _name(smi) == expected


# ---------------------------------------------------------------------------
# Self-gating: must NOT fire on non-ester / out-of-scope skeletons
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smi", [
    "OP(=O)(O)O",               # phosphoric ACID (no ester) -> handled by compute_name
    "OS(=O)(=O)O",              # sulfuric acid (no ester)
    "CCO",                      # no centre
    "COC(=O)OC",                # carbonate: carbon centre, not main-group
    "COS(C)(=O)=O",             # methyl methanesulfonate: C-S bond (sulfonate family)
    "COP(=O)([O-])OC",          # anion / salt form (net charge)
    "COP(=O)(OC)OP(=O)(OC)OC",  # polynuclear anhydride
    "COP(=O)(OC)N",             # phosphoramidate (amido N)
    "COP(=O)(Cl)OC",            # phosphorochloridate (halide)
])
def test_does_not_fire(smi: str) -> None:
    mol = Chem.MolFromSmiles(smi)
    assert compute_oxoacid_ester_name(mol) is None


def test_parent_acid_namer_still_works() -> None:
    """The bare acids must still be named by the parent generator."""
    assert compute_name(Chem.MolFromSmiles("OP(=O)(O)O")) == "phosphoric acid"
    assert compute_name(Chem.MolFromSmiles("OS(=O)(=O)O")) == "sulfuric acid"


# ---------------------------------------------------------------------------
# End-to-end engine integration (names emitted via the dispatch hook)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smi,expected", [
    ("COP(=O)(OC)OC", "trimethyl phosphate"),
    ("COS(=O)(=O)OC", "dimethyl sulfate"),
    ("CCOP(C)(=O)OCC", "diethyl methylphosphonate"),
    ("COP(=O)(O)O", "methyl dihydrogen phosphate"),
    ("CO[PH](=O)OC", "dimethyl phosphonate"),  # P-H form: runs before FV guard
])
def test_engine_integration(smi: str, expected: str) -> None:
    assert name_smiles(smi) == expected
