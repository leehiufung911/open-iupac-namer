"""Tests for retained ring name lookups added in the azole/benzo-fused expansion.

Each parametrized case verifies that a standalone heterocycle produces its
IUPAC retained name (not a systematic Hantzsch-Widman or azaazol* name).

Canonical SMILES keys were verified via:
    from rdkit import Chem; Chem.MolToSmiles(Chem.MolFromSmiles(input_smi))
"""
import pytest
from iupac_namer.engine import name_smiles


@pytest.mark.parametrize("smi,expected", [
    # 5-membered azoles
    ("c1cn[nH]c1",  "1H-pyrazole"),
    ("c1cn[nH]n1",  "1H-1,2,3-triazole"),
    ("c1nc[nH]n1",  "1H-1,2,4-triazole"),
    ("c1nnn[nH]1",  "1H-tetrazole"),
    ("c1cscn1",     "1,3-thiazole"),
    ("c1cnsc1",     "isothiazole"),
    ("c1cocn1",     "1,3-oxazole"),
    ("c1cnoc1",     "isoxazole"),
    ("c1nncs1",     "1,3,4-thiadiazole"),
    ("c1ncsn1",     "1,2,4-thiadiazole"),
    ("c1nnco1",     "1,3,4-oxadiazole"),
    ("c1ncon1",     "1,2,4-oxadiazole"),
    ("c1conn1",     "1,2,5-oxadiazole"),
    # 6-membered multi-N
    ("c1cnnnc1",    "1,2,3-triazine"),
    ("c1cnncn1",    "1,2,4-triazine"),
    ("c1ncncn1",    "1,3,5-triazine"),
    ("c1cnnnn1",    "1,2,3,4-tetrazine"),
    # Benzo-fused 5-membered
    ("c1ccc2[nH]cnc2c1",  "1H-benzimidazole"),
    ("c1ccc2ocnc2c1",     "1,3-benzoxazole"),
    ("c1ccc2scnc2c1",     "1,3-benzothiazole"),
    ("c1ccc2occc2c1",     "benzofuran"),
    ("c1ccc2sccc2c1",     "1-benzothiophene"),
    ("c1ccc2[nH]ncc2c1",  "1H-indazole"),
    ("c1ccc2c[nH]cc2c1",  "2H-isoindole"),
    ("c1ccc2cocc2c1",     "2-benzofuran"),
    # Benzo-fused 6-membered
    ("c1ccc2nccnc2c1",    "quinoxaline"),
    ("c1ccc2ncncc2c1",    "quinazoline"),
    # Cinnoline/phthalazine: InChIKey verified (c1ccc2nnccc2c1 = WCZVZNOTHYJIEI = cinnoline)
    ("c1ccc2nnccc2c1",    "cinnoline"),
    ("c1ccc2cnncc2c1",    "phthalazine"),
    # Purine and pteridine
    ("c1nc2c[nH]cnc-2n1", "9H-purine"),
    ("c1cnc2ncncc2n1",    "pteridine"),
])
def test_retained_ring(smi, expected):
    result = name_smiles(smi)
    assert result == expected, f"For SMILES {smi!r}: got {result!r}, expected {expected!r}"


# ---------------------------------------------------------------------------
# Non-substitutable retained-name fallback (P-25.3 / P-31).
#
# "xanthine" is a retained name OPSIN accepts only as a BARE scaffold; it
# refuses a substituent locant on the stem ("8-chloroxanthine" is unparseable).
# A substituted xanthine must therefore be named on the systematic
# substitutable parent — the mancude purine carrying a 2,6-dione,
# rendered by the engine as "<sub>-2,6-dioxo-...H-purine" (equivalent to and
# round-trip-identical to the "...-purine-2,6-dione" PIN).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smi", [
    "O=c1[nH]c(=O)c2[nH]c(Cl)nc2[nH]1",   # 8-chloroxanthine
    "O=c1[nH]c(=O)c2[nH]c(Br)nc2[nH]1",   # 8-bromoxanthine
    "Cc1nc2c([nH]1)c(=O)[nH]c(=O)[nH]2",  # 8-methylxanthine
    "Nc1nc2c([nH]1)c(=O)[nH]c(=O)[nH]2",  # 8-aminoxanthine
    "CCn1c(=O)[nH]c(=O)c2[nH]cnc21",      # an N-ethylxanthine
    "Cn1cnc2c1c(=O)[nH]c(=O)[nH]2",       # 7-methylxanthine
])
def test_substituted_xanthine_falls_back_to_systematic(smi):
    """A substituted xanthine must NOT emit the non-substitutable retained
    'xanthine' stem; it must round-trip via the systematic purine parent."""
    from tests.audit._audit_helpers import assert_round_trip
    name = assert_round_trip(smi)
    assert "xanthin" not in name.lower(), (
        f"substituted xanthine should fall back to the systematic purine "
        f"parent, but got non-substitutable retained name {name!r}"
    )
    assert "purin" in name.lower(), (
        f"expected the systematic purine parent for {smi!r}, got {name!r}"
    )


@pytest.mark.parametrize("smi,expected", [
    ("O=c1[nH]c(=O)c2[nH]cnc2[nH]1", "xanthine"),  # bare scaffold keeps retained name
])
def test_bare_xanthine_keeps_retained_name(smi, expected):
    """The bare xanthine scaffold (no ring substituent) must still resolve to
    its retained name — the fallback gate only fires on substituted forms."""
    result = name_smiles(smi)
    assert result == expected, (
        f"bare scaffold {smi!r} should keep retained name {expected!r}, "
        f"got {result!r}"
    )
