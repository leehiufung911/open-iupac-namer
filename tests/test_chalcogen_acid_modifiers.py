"""
Tests for the modified-organic-chalcogen-oxoacid composer
(``iupac_namer/perception/fg/chalcogen_acid_modifiers.py``), P-65.3.1 / P-66.

The composer COMPUTES names for the
``-sulfin/-sulfon/-selenin/-selenon/-tellurin/-telluron`` acid family with
functional-replacement infixes (seleno, telluro, peroxo, imido, hydrazono and
combinations) that the static ``functional_groups.json`` SMARTS table does not
enumerate.  Each assertion is the exact PIN-spelled name the engine emits; the
authoritative eval harness carries the SMILES round-trip check.

The composer must DECLINE (return ``None``) for plain acids and for the exact
signatures the static SMARTS table already covers, so existing handling stays
untouched — those cases are pinned in ``test_defers_to_static_table``.
"""
from __future__ import annotations

import pytest
from rdkit import Chem

from iupac_namer.engine import name_smiles as name
from iupac_namer.perception.fg.chalcogen_acid_modifiers import compute_name


# ---------------------------------------------------------------------------
# Full-pipeline name assertions (engine produces the composed PIN)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "smiles, expected",
    [
        # --- sulfin (S IV): one oxo + one acidic position ---
        ("CCCS(O)=[Se]", "propane-1-sulfinoselenoic acid"),     # =O -> =Se
        ("CCS(O)=[Se]", "ethanesulfinoselenoic acid"),
        ("CCS(O)=[Te]", "ethanesulfinotelluroic acid"),         # =O -> =Te
        ("CCS(=O)OO", "ethanesulfinoperoxoic acid"),            # -OH -> -OOH
        ("CCS(=O)[SeH]", "ethanesulfinoselenoic Se-acid"),      # -OH -> -SeH
        ("CCS(=O)[TeH]", "ethanesulfinotelluroic Te-acid"),     # -OH -> -TeH
        ("CCS(=O)S", "ethanesulfinothioic S-acid"),             # -OH -> -SH
        # --- sulfon (S VI): two oxo + one acidic position ---
        ("CCS(=O)(O)=[Se]", "ethanesulfonoselenoic acid"),
        ("CCS(=O)(=O)OO", "ethanesulfonoperoxoic acid"),
        ("CCS(=O)(=O)[SeH]", "ethanesulfonoselenoic Se-acid"),
        ("OS(=S)(=[Se])c1ccccc1", "benzenesulfonothioselenoic acid"),
        ("N=S(O)(=S)c1ccccc1", "benzenesulfonimidothioic acid"),
        ("N=S(O)(=NN)c1ccccc1", "benzenesulfonohydrazonimidic acid"),
        ("OS(=[Se])(=[Se])c1ccccc1", "benzenesulfonodiselenoic acid"),
        ("N=S(=O)(OO)c1ccccc1", "benzenesulfonoperoxoimidic acid"),
        # --- selenin (Se IV) ---
        ("O[Se](=[Se])c1ccccc1", "benzeneseleninoselenoic acid"),
        ("O[Se](=S)c1ccccc1", "benzeneseleninothioic acid"),
        ("O=[Se](OO)c1ccccc1", "benzeneseleninoperoxoic acid"),
        ("N=[Se](O)c1ccccc1", "benzeneseleninimidic acid"),
        ("NN=[Se](O)c1ccccc1", "benzeneseleninohydrazonic acid"),
        # --- selenon (Se VI) ---
        ("N=[Se](O)(=NN)c1ccc2ccccc2c1",
         "naphthalene-2-selenonohydrazonimidic acid"),
        ("O=[Se](=O)(OO)c1ccccc1", "benzeneselenonoperoxoic acid"),
        ("O[Se](=S)(=[Se])c1ccccc1", "benzeneselenonothioselenoic acid"),
        # --- tellurin / telluron (Te) ---
        ("O=[Te](OO)c1ccccc1", "benzenetellurinoperoxoic acid"),
        ("O[Te](=S)c1ccccc1", "benzenetellurinothioic acid"),
        ("N=[Te](O)c1ccccc1", "benzenetellurinimidic acid"),
        ("O=[Te](=O)(OO)c1ccccc1", "benzenetelluronoperoxoic acid"),
        ("N=[Te](=O)(O)c1ccccc1", "benzenetelluronimidic acid"),
    ],
)
def test_composed_chalcogen_acid_names(smiles: str, expected: str) -> None:
    out = name(smiles)
    assert out == expected, f"{smiles} -> {out!r} (want {expected!r})"


def test_cyclohexane_parent() -> None:
    # ring parent: cyclohexanesulfinoselenoic acid (=O -> =Se on cyclohexyl S)
    assert name("OS(=[Se])C1CCCCC1") == "cyclohexanesulfinoselenoic acid"


# ---------------------------------------------------------------------------
# Deferral: signatures already covered by the static SMARTS table must NOT be
# claimed by the composer (it returns None so the existing FG handles them).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "smiles",
    [
        "CCS(=O)O",          # ethanesulfinic acid (plain)
        "CCS(=O)(=O)O",      # ethanesulfonic acid (plain)
        "O=[Se](O)c1ccccc1",  # benzeneseleninic acid (plain)
        "CCS(=N)O",          # ethanesulfinimidic acid (data FG)
        "CCS(=O)(=N)O",      # ethanesulfonimidic acid (data FG)
        "CCS(=N)(=N)O",      # ethanesulfonodiimidic acid (data FG)
        "CCS(O)=S",          # ethanesulfinothioic O-acid (data FG)
        "N=[Se](=O)(O)c1ccccc1",  # benzeneselenonimidic acid (data FG)
        "CCS(O)=NN",         # ethanesulfinohydrazonic acid (data FG)
    ],
)
def test_defers_to_static_table(smiles: str) -> None:
    mol = Chem.MolFromSmiles(smiles)
    assert compute_name(mol) is None, (
        f"{smiles}: composer should defer to the static SMARTS table"
    )


# ---------------------------------------------------------------------------
# Non-acids and out-of-scope shapes must be declined.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "smiles",
    [
        "CCSC",              # diethyl... thioether: no acid centre
        "CCS(=O)(=O)c1ccccc1",  # sulfone: no acidic -OH/-XH position
        "CC(=O)O",           # carboxylic acid: no chalcogen centre
        "OS(=[Se])OCC",      # selenoester (no C bonded to S): decline
    ],
)
def test_declines_non_chalcogen_acids(smiles: str) -> None:
    mol = Chem.MolFromSmiles(smiles)
    assert compute_name(mol) is None, f"{smiles}: should not be claimed"
