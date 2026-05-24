"""Regression tests for heteroaromatic-fused ring lactones (P-25.3).

Background:
Benzene-fused ring lactones already named correctly via the curated
``1,3-dihydro-2-benzofuran`` (phthalide base) entry + the exocyclic-oxo
``-one`` suffix path.  Heteroaromatic-base-fused furanones / thiophenones
(furo/thieno[3,4-x]azine ring lactones, e.g. pyridoxolactone) had no curated
base scaffold, so the engine failed with "No valid naming plan".

This adds the saturated furan/thiophene-ring base scaffolds for the azine
fusions that OPSIN names as a clean ring lactone:

    * 1,3-dihydrofuro[3,4-c]pyridine    / thieno  (lactone carbons at 1,3)
    * 5,7-dihydrofuro[3,4-d]pyrimidine  / thieno  (lactone carbons at 5,7)
    * 5,7-dihydrofuro[3,4-b]pyrazine    / thieno  (lactone carbons at 5,7)

atom_locants were derived by OPSIN chloro/methyl-probing every substitutable
ring position with a bond-generic SubstructMatch onto the canonical ring
SMILES, with ring heteroatoms and the two bridgehead carbons closed by the
canonical fused-ring perimeter walk (the unique assignment consistent with
all OPSIN anchors).  The engine names the lactone by carving the saturated
base scaffold and expressing the ring C=O as a ``-one`` suffix — the same
machinery that names phthalide as 1,3-dihydro-2-benzofuran-1-one.

Each tuple in ``_PROBES`` pins (input_smiles, expected_engine_name) and is
round-tripped through OPSIN to confirm canonical-SMILES equivalence.
"""
from __future__ import annotations

import pytest
from rdkit import Chem

from iupac_namer.engine import name_smiles


# (input_smiles, expected_name)
_PROBES: list[tuple[str, str]] = [
    # ---- furo[3,4-c]pyridine lactone family (pyridoxolactone, the eval
    #      coverage cluster) ----
    ("O=C1OCc2cnccc21", "1,3-dihydrofuro[3,4-c]pyridin-1-one"),
    ("O=C1OCc2ccncc21", "1,3-dihydrofuro[3,4-c]pyridin-3-one"),
    # 4-pyridoxolactone: 7-hydroxy-6-methyl, carbonyl at 1
    ("Cc1ncc2c(c1O)C(=O)OC2", "7-hydroxy-6-methyl-1,3-dihydrofuro[3,4-c]pyridin-1-one"),
    ("Cc1ncc2c(c1O)C(=O)OC2Cl",
     "3-chloro-7-hydroxy-6-methyl-1,3-dihydrofuro[3,4-c]pyridin-1-one"),
    ("Cc1nc(Cl)c2c(c1O)C(=O)OC2",
     "4-chloro-7-hydroxy-6-methyl-1,3-dihydrofuro[3,4-c]pyridin-1-one"),
    # 5-pyridoxolactone: carbonyl at 3
    ("Cc1ncc2c(c1O)COC2=O", "7-hydroxy-6-methyl-1,3-dihydrofuro[3,4-c]pyridin-3-one"),
    ("Cc1ncc2c(c1O)C(Cl)OC2=O",
     "1-chloro-7-hydroxy-6-methyl-1,3-dihydrofuro[3,4-c]pyridin-3-one"),
    # ---- thieno[3,4-c]pyridine analog ----
    ("O=C1SCc2cnccc21", "1,3-dihydrothieno[3,4-c]pyridin-1-one"),
    ("Cc1ncc2c(c1O)C(=O)SC2",
     "7-hydroxy-6-methyl-1,3-dihydrothieno[3,4-c]pyridin-1-one"),
    # ---- furo/thieno[3,4-d]pyrimidine lactone (carbons 5,7) ----
    ("O=C1OCc2ncncc21", "5,7-dihydrofuro[3,4-d]pyrimidin-5-one"),
    ("O=C1SCc2ncncc21", "5,7-dihydrothieno[3,4-d]pyrimidin-5-one"),
    ("O=C1OC(Cl)c2ncncc21", "7-chloro-5,7-dihydrofuro[3,4-d]pyrimidin-5-one"),
    # ---- furo/thieno[3,4-b]pyrazine lactone (carbons 5,7) ----
    ("O=C1OCc2nccnc21", "5,7-dihydrofuro[3,4-b]pyrazin-5-one"),
    ("O=C1SCc2nccnc21", "5,7-dihydrothieno[3,4-b]pyrazin-5-one"),
]


@pytest.mark.parametrize("smiles,expected_name", _PROBES)
def test_fused_lactone_name(smiles: str, expected_name: str) -> None:
    """Engine must emit the expected fused-lactone IUPAC name."""
    got = name_smiles(smiles)
    assert got == expected_name, (
        f"name_smiles({smiles!r}) = {got!r}, expected {expected_name!r}"
    )


@pytest.mark.parametrize("smiles,expected_name", _PROBES)
def test_fused_lactone_roundtrip(smiles: str, expected_name: str) -> None:
    """Each probe must round-trip via OPSIN: input -> our-name -> OPSIN-SMILES
    -> same canonical structure as input.

    Skipped when py2opsin / Java are unavailable.
    """
    py2opsin = pytest.importorskip("py2opsin")
    smi_in = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    name = name_smiles(smiles)
    opsin_smi = py2opsin.py2opsin(name, output_format="SMILES")
    if not opsin_smi:
        pytest.skip(f"OPSIN/Java unavailable; got empty SMILES for {name!r}")
    smi_out = Chem.MolToSmiles(Chem.MolFromSmiles(opsin_smi))
    assert smi_out == smi_in, (
        f"round-trip mismatch: {smiles!r} -> {name!r} -> {opsin_smi!r}\n"
        f"  in  canonical: {smi_in}\n"
        f"  out canonical: {smi_out}"
    )


def test_fused_lactone_bases_present_in_curated() -> None:
    """The six base scaffolds must wire through into ``_CURATED`` with
    atom_locants so the curated lookup returns a deterministic numbering.
    """
    from iupac_namer.ring_naming.retained_lookup import _CURATED

    expected = {
        "c1cc2c(cn1)COC2": "1,3-dihydrofuro[3,4-c]pyridine",
        "c1cc2c(cn1)CSC2": "1,3-dihydrothieno[3,4-c]pyridine",
        "c1ncc2c(n1)COC2": "5,7-dihydrofuro[3,4-d]pyrimidine",
        "c1ncc2c(n1)CSC2": "5,7-dihydrothieno[3,4-d]pyrimidine",
        "c1cnc2c(n1)COC2": "5,7-dihydrofuro[3,4-b]pyrazine",
        "c1cnc2c(n1)CSC2": "5,7-dihydrothieno[3,4-b]pyrazine",
    }
    for smiles, name in expected.items():
        entry = _CURATED.get(smiles)
        assert entry is not None, f"curated entry missing for {name} ({smiles})"
        got_name, _sub, _alkyl, atom_locants = entry
        assert got_name == name, f"{smiles}: name {got_name!r} != {name!r}"
        assert atom_locants is not None, f"{name}: atom_locants missing"
