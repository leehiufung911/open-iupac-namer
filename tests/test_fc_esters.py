"""Tests for Phase 2d: functional-class path for intermolecular esters."""
from __future__ import annotations

import pytest
from rdkit import Chem

from iupac_namer.engine import name_smiles, name
from iupac_namer.perception import Perception
from iupac_namer.perception.extraction import carve_fc_fragments
from iupac_namer.strategy import IUPACCanonical
from iupac_namer.types import (
    InterpretationQuery,
    OutputForm,
)


# ---------------------------------------------------------------------------
# End-to-end name_smiles tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("COC(=O)c1ccccc1", "methyl benzoate"),
        ("CCOC(=O)c1ccccc1", "ethyl benzoate"),
        ("CCOC(=O)CC", "ethyl propanoate"),
        ("CC(=O)Oc1ccccc1", "phenyl acetate"),
        ("COC(=O)C", "methyl acetate"),
        ("CCOC(=O)C", "ethyl acetate"),
    ],
)
def test_simple_intermolecular_esters(smiles, expected):
    """All primary Phase 2d targets."""
    assert name_smiles(smiles) == expected


# ---------------------------------------------------------------------------
# Decomposition generation
# ---------------------------------------------------------------------------

def _first_interp(mol):
    p = Perception(mol)
    q = InterpretationQuery(
        preferred_decomp_types=None,
        preferred_parent_type=None,
        suppress_functional_class=False,
        max_results=5,
    )
    return next(iter(p.interpretations(q)))


def test_decomposition_yielded_for_ester():
    mol = Chem.MolFromSmiles("COC(=O)c1ccccc1")
    interp = _first_interp(mol)
    decomps = list(interp.decomposition_candidates(mol))
    ester_decomps = [d for d in decomps if d.subtype == "ester"]
    assert len(ester_decomps) == 1
    d = ester_decomps[0]
    assert d.type == "functional_class"
    assert d.intramolecular is False
    assert d.pieces is not None
    assert len(d.pieces) == 2


def test_decomposition_intramolecular_for_lactone():
    # butyrolactone — the ester C-O is in a ring
    mol = Chem.MolFromSmiles("O=C1CCCO1")
    interp = _first_interp(mol)
    decomps = [
        d for d in interp.decomposition_candidates(mol) if d.subtype == "ester"
    ]
    # The engine may still yield a decomposition, but it must be flagged
    # intramolecular.  Strategy then rejects it.
    for d in decomps:
        assert d.intramolecular is True


# ---------------------------------------------------------------------------
# carve_fc_fragments
# ---------------------------------------------------------------------------

def test_carve_fc_fragments_methyl_benzoate():
    mol = Chem.MolFromSmiles("COC(=O)c1ccccc1")
    interp = _first_interp(mol)
    decomp = next(
        d for d in interp.decomposition_candidates(mol) if d.subtype == "ester"
    )
    frags = carve_fc_fragments(mol, decomp)
    assert set(frags.keys()) == {"acid", "alcohol"}

    acid_mol, acid_att = frags["acid"]
    alcohol_mol, alcohol_att = frags["alcohol"]
    # Acid side has no attachment — it is named standalone.
    assert acid_att is None
    # Alcohol side has an attachment atom (the alkyl C).
    assert alcohol_att is not None

    assert Chem.MolToSmiles(acid_mol) == "O=C(O)c1ccccc1"
    # alcohol side is methane as a substituent; attachment point tracked by
    # alcohol_att index (explicit [H] stripped by RemoveHs).
    # When renamed as OutputForm.SUBSTITUENT it produces "methyl".
    assert Chem.MolToSmiles(alcohol_mol) == "C"


def test_carve_fc_fragments_phenyl_acetate():
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1")
    interp = _first_interp(mol)
    decomp = next(
        d for d in interp.decomposition_candidates(mol) if d.subtype == "ester"
    )
    frags = carve_fc_fragments(mol, decomp)
    acid_mol, _ = frags["acid"]
    alcohol_mol, _ = frags["alcohol"]
    assert Chem.MolToSmiles(acid_mol) == "CC(=O)O"
    # phenyl substituent — explicit [H] stripped by RemoveHs, attachment
    # tracked by alcohol_att index.
    assert Chem.MolToSmiles(alcohol_mol) == "c1ccccc1"


# ---------------------------------------------------------------------------
# Strategy rejection
# ---------------------------------------------------------------------------

def test_lactone_falls_back_to_substitutive():
    """A lactone (intramolecular ester) must NOT be named as "methyl …ate";
    the FC plan must be rejected and the substitutive path used instead."""
    # Butyrolactone — check that we don't get a FC-shaped name ("alkyl …ate")
    result = name_smiles("O=C1CCCO1")
    assert " " not in result.strip() or result.startswith("[")
    # Intramolecular ester should not produce the "X acid" FC form.
    assert "acid" not in result


# ---------------------------------------------------------------------------
# Symmetric diester FC path (P-65.6.3.x)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "smiles,expected",
    [
        # Diallyl oxalate (ZT-1552): both R groups are the PIN
        # ``prop-2-en-1-yl`` (allyl is general nomenclature only, per
        # P-32.1.1(1) the locant-1 must be cited on acyclic ene/yne
        # substituents).  Acid backbone is oxalic acid → "oxalate".
        ("C=CCOC(=O)C(=O)OCC=C", "bis(prop-2-en-1-yl) oxalate"),
        # Diethyl oxalate
        ("CCOC(=O)C(=O)OCC", "diethyl oxalate"),
        # Diethyl pentanedioate: 5-carbon diacid backbone.  'glutarate' is
        # retained for general nomenclature only; the PIN ester stem is the
        # systematic 'pentanedioate' (P-65.1.1.2.2; cf. Blue Book P-1
        # 'dimethyl butanedioate (PIN) / dimethyl succinate').
        ("CCOC(=O)CCCC(=O)OCC", "diethyl pentanedioate"),
        # Dimethyl butanedioate (PIN; 'dimethyl succinate' is general only)
        ("COC(=O)CCC(=O)OC", "dimethyl butanedioate"),
        # Dimethyl propanedioate (PIN; 'dimethyl malonate' is general only)
        ("COC(=O)CC(=O)OC", "dimethyl propanedioate"),
    ],
)
def test_symmetric_diester_fc(smiles, expected):
    """Symmetric diesters use the FC path to produce 'di{alkyl} {diacid-ate}'."""
    assert name_smiles(smiles) == expected


def test_symmetric_diester_decomposition_yielded():
    """A fully-esterified diacid yields an FC poly-ester decomposition.

    The general ``polyester`` path (P-65.6.3.3.2) subsumes the older
    ``symmetric_diester`` path; for diethyl oxalate it is generated in
    preference to it.  Either subtype is an acceptable functional-class ester
    decomposition for this symmetric diester.
    """
    mol = Chem.MolFromSmiles("CCOC(=O)C(=O)OCC")
    interp = _first_interp(mol)
    decomps = list(interp.decomposition_candidates(mol))
    diester_decomps = [
        d for d in decomps if d.subtype in ("polyester", "symmetric_diester")
    ]
    assert len(diester_decomps) == 1
    d = diester_decomps[0]
    assert d.type == "functional_class"
    assert d.intramolecular is False
    assert d.pieces is not None
    assert len(d.pieces) == 2


def test_asymmetric_diester_does_not_use_symmetric_path():
    """Asymmetric diesters (different R groups) must NOT use the symmetric path."""
    # methyl ethyl glutarate: methyl on one end, ethyl on the other
    smiles = "CCOC(=O)CCCC(=O)OC"
    mol = Chem.MolFromSmiles(smiles)
    interp = _first_interp(mol)
    decomps = list(interp.decomposition_candidates(mol))
    diester_decomps = [d for d in decomps if d.subtype == "symmetric_diester"]
    assert len(diester_decomps) == 0, (
        f"Asymmetric diester should not yield symmetric_diester decomposition"
    )


def test_substituted_backbone_uses_polyester_path():
    """A fully-esterified diacid with a substituted backbone is named as the
    functional-class poly-ester (P-65.6.3.3.2.1), NOT by demoting one ester to
    an ``(alkoxycarbonyl)`` substitutive prefix.

    Previously the symmetric-diester path declined substituted backbones and the
    engine fell back to ``ethyl 2-(ethoxycarbonyl)pentanoate``-style demotion.
    That demoted form is only the PIN for *partial* esters (P-65.6.3.2.3 /
    P-65.6.3.3.5).  A fully-esterified diacid takes the functional-class form:
    ``diethyl 2-propylpropanedioate``.
    """
    # Diethyl 2-propylmalonate: has a propyl branch on the backbone
    smiles = "CCCC(C(=O)OCC)C(=O)OCC"
    mol = Chem.MolFromSmiles(smiles)
    interp = _first_interp(mol)
    decomps = list(interp.decomposition_candidates(mol))
    polyester_decomps = [d for d in decomps if d.subtype == "polyester"]
    assert len(polyester_decomps) == 1, (
        "Fully-esterified diacid should yield a polyester decomposition"
    )
    # The FC poly-ester name is the PIN; no alkoxycarbonyl demotion.
    result = name_smiles(smiles)
    assert result == "diethyl 2-propylpropanedioate", result
    assert "ethoxycarbonyl" not in result
