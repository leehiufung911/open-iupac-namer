"""Tests for alkoxycarbonyl and related acyl substituents when the ester/acid
group is demoted (not the principal characteristic group).

Per IUPAC P-66.6.3, a demoted ester -C(=O)OR attached as a branch off the
parent should be named "{R}oxycarbonyl" (e.g., methoxycarbonyl, ethoxycarbonyl).
"""
import pytest
from iupac_namer.engine import name_smiles


# ---------------------------------------------------------------------------
# Alkoxycarbonyl substituent cases
# ---------------------------------------------------------------------------

ALKOXYCARBONYL_CASES = [
    # Simple ring-attached cases: carboxylic acid is PCG, ester is demoted
    pytest.param(
        "OC(=O)c1ccc(C(=O)OCC)cc1",
        "ethoxycarbonyl",
        id="para-ethoxycarbonyl-benzoic-acid",
    ),
    pytest.param(
        "OC(=O)c1ccc(C(=O)OCC)cc1",
        "benzoic acid",  # also check parent name (retained PIN per P-66.6.3.2)
        id="para-ethoxycarbonyl-benzoic-acid-parent",
    ),
    # Chain case: ester C is part of parent chain, atoms split into ethoxy+oxo
    # OPSIN accepts "3-ethoxy-3-oxopropanoic acid" as correct
    pytest.param(
        "OC(=O)CC(=O)OCC",
        "ethoxy",  # ethoxy is part of the output even if split form
        id="chain-ethoxycarbonyl-acid-contains-oxy",
    ),
    # Methoxy chain case: OPSIN accepts "3-methoxy-3-oxopropanoic acid" as correct
    pytest.param(
        "OC(=O)CC(=O)OC",
        "methoxy",  # methoxy is part of the correct output
        id="chain-methoxy-from-methyl-ester",
    ),
]


@pytest.mark.parametrize("smi,expected_substring", ALKOXYCARBONYL_CASES)
def test_alkoxycarbonyl(smi, expected_substring):
    name = name_smiles(smi)
    assert expected_substring in name, (
        f"Expected '{expected_substring}' to appear in name,\n"
        f"  SMILES: {smi}\n"
        f"  Got:    {name!r}"
    )


def test_fully_esterified_diacid_not_demoted():
    """A fully-esterified diacid uses the FC poly-ester form (P-65.6.3.3.2.1),
    NOT an ``(alkoxycarbonyl)`` demotion.

    ``CCCC(C(=O)OCC)C(=O)OCC`` (diethyl 2-propylmalonate) was historically
    named ``ethyl 2-(ethoxycarbonyl)pentanoate`` (one ester demoted to a
    prefix).  That demoted form is only correct for *partial* esters; when every
    acid group is esterified the PIN is ``diethyl 2-propylpropanedioate``.
    """
    name = name_smiles("CCCC(C(=O)OCC)C(=O)OCC")
    assert name == "diethyl 2-propylpropanedioate", name
    assert "ethoxycarbonyl" not in name


# ---------------------------------------------------------------------------
# Already-working cases: carbamoyl and carboxy as demoted FG prefixes
# (These use the FG prefix_form directly, not alkoxycarbonyl path)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smi,expected", [
    # When the amide C (anchor) is in the parent chain and COOH is the PCG,
    # the amide C=O is correctly decomposed into oxo + amino (not "carbamoyl").
    # IUPAC Blue Book treatment: the amide carbonyl becomes oxo, the NH2
    # becomes amino — both at the same locant.
    # OPSIN-verified: 3-amino-3-oxopropanoic acid → NC(=O)CC(=O)O ✓
    ("NC(=O)CC(=O)O", "3-amino-3-oxopropanoic acid"),
    # N-phenyl case: phenylamino + oxo
    # OPSIN-verified: 3-oxo-3-(phenylamino)propanoic acid → O=C(O)CC(=O)Nc1ccccc1 ✓
    ("O=C(O)CC(=O)Nc1ccccc1", "3-oxo-3-(phenylamino)propanoic acid"),
])
def test_demoted_amide_anchor_in_parent(smi, expected):
    """Demoted amide with anchor C in parent: decompose as oxo + amino/phenylamino."""
    name = name_smiles(smi)
    assert name == expected, (
        f"SMILES: {smi}\n"
        f"  Got:    {name!r}\n"
        f"  Expected: {expected!r}"
    )


# ---------------------------------------------------------------------------
# Non-regression: FC path for simple esters must still work
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smi,expected", [
    ("CCOC(=O)c1ccccc1", "ethyl benzoate"),
    ("COC(=O)c1ccccc1", "methyl benzoate"),
    ("CCOC(=O)CC", "ethyl propanoate"),
    ("CC(=O)Oc1ccccc1", "phenyl acetate"),
])
def test_fc_ester_not_broken(smi, expected):
    """Ensure the FC path (ester as PCG) still produces correct names."""
    name = name_smiles(smi)
    assert name == expected, (
        f"FC ester path broken:\n"
        f"  SMILES:   {smi}\n"
        f"  Expected: {expected!r}\n"
        f"  Got:      {name!r}"
    )


# ---------------------------------------------------------------------------
# Non-regression: ether prefixes still work
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smi,expected", [
    ("CCOc1ccccc1", "ethoxybenzene"),
    # P-34.1.1.4 retained PIN; methoxybenzene is the systematic equivalent.
    ("COc1ccccc1", "anisole"),
])
def test_ether_prefix_not_broken(smi, expected):
    """Ensure ether prefix detection (role='ether_prefix') still works."""
    name = name_smiles(smi)
    assert name == expected, (
        f"Ether prefix broken:\n"
        f"  SMILES:   {smi}\n"
        f"  Expected: {expected!r}\n"
        f"  Got:      {name!r}"
    )
