"""
tests/test_fg_deconfliction.py

Regression tests for the FG-detection same-type deconfliction logic
(Stage 6 R1-H).

Failure mode addressed: a single chemical motif whose SMARTS pattern is
symmetric about a shared core (e.g. thiourea-style H2N-C(=X)-NH2) was
being emitted TWICE because the deconfliction PASS 2 "same-type
geminal" branch accepted any pair of same-type matches whose atom sets
differed, even when the difference was only in the peripheral arm.

The fix in ``iupac_namer/perception/fg_detection.py`` tightens the
same-type-coexistence rule:
    shared = a.atoms & b.atoms
    if shared <= {anchor}: accept both (true geminal, e.g. geminal diol)
    else:                  keep only the more-terminal match.

These tests pin down the corrected behaviour for:

- Semicarbazone / thiosemicarbazone / selenosemicarbazone family
  (FG audit Gap 10) — duplicate-emission of the -C(=X)-NH2 motif.
- Legitimate geminal polyhydroxyls — must NOT be affected.
- Retained parents (urea / thiourea / sulfamide) — must continue to
  route through the retained/functional-parent path unchanged.
- Singleton hydrazones — must not lose their FG match.
"""

from __future__ import annotations

import logging

import pytest

logging.disable(logging.WARNING)

from iupac_namer.engine import name_smiles


# ---------------------------------------------------------------------------
# Semicarbazone / thiosemicarbazone family — FG audit Gap 10
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "smiles, forbidden_substrings",
    [
        # Before the fix, these produced '...hydrazine-1,1-dicarbothioamide'
        # or '...hydrazinecarbothioamide' with a spurious extra -thiocarbamoyl
        # prefix, reflecting double-emission of the lone C(=S)-NH2 arm.
        ("CC(C)=NNC(=S)N",         ("dicarbothioamide", "thiocarbamoylhydrazinecarbothioamide")),
        ("C(c1ccccc1)=NNC(=S)N",   ("dicarbothioamide", "thiocarbamoylhydrazinecarbothioamide")),
        # Semicarbazone (O analog) — already worked, pin it as a regression.
        ("CC=NNC(=O)N",            ("dicarboxamide", "dicarboximide", "carbamoylhydrazinecarboxamide")),
        ("CC(C)=NNC(=O)N",         ("dicarboxamide", "dicarboximide", "carbamoylhydrazinecarboxamide")),
        ("C(c1ccccc1)=NNC(=O)N",   ("dicarboxamide", "dicarboximide", "carbamoylhydrazinecarboxamide")),
    ],
)
def test_semicarbazone_family_no_duplicate_fg_emission(smiles, forbidden_substrings):
    """Duplicated FG emission must not appear in the name."""
    name = name_smiles(smiles)
    assert name is not None, f"engine returned None for {smiles!r}"
    for forbidden in forbidden_substrings:
        assert forbidden not in name, (
            f"SMILES={smiles!r}: name {name!r} contains the duplicate-emission "
            f"marker {forbidden!r} — the thioamide/amide FG was matched twice "
            f"and both copies leaked into the final name."
        )


@pytest.mark.parametrize(
    "smiles, required_substrings",
    [
        # Acetone thiosemicarbazone must carry -methanethioamide (one copy).
        ("CC(C)=NNC(=S)N",       ("methanethioamide", "propan-2-ylidene")),
        ("C(c1ccccc1)=NNC(=S)N", ("methanethioamide", "phenylmethylidene")),
        # Semicarbazone analog.
        ("CC=NNC(=O)N",          ("methanamide", "ethylidene")),
    ],
)
def test_semicarbazone_family_emits_expected_skeleton(smiles, required_substrings):
    """The (thio)semicarbazone skeleton should reduce to ``methan(e)-*amide``
    with the imine-side named as an ``[(Ryl)amino]amino`` substituent."""
    name = name_smiles(smiles)
    assert name is not None, f"engine returned None for {smiles!r}"
    for required in required_substrings:
        assert required in name, (
            f"SMILES={smiles!r}: name {name!r} is missing required fragment "
            f"{required!r}"
        )


# ---------------------------------------------------------------------------
# Genuine geminal — must NOT regress
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "smiles, expected",
    [
        # Geminal diols: the two alcohol SMARTS matches share only the
        # anchor C; the deconfliction pass must accept both.
        ("C(O)O",      "methanediol"),
        ("CC(O)O",     "ethane-1,1-diol"),
        ("CCC(O)O",    "propane-1,1-diol"),
    ],
)
def test_geminal_diols_retain_both_alcohol_fgs(smiles, expected):
    """Geminal diols: the two -OH matches share only the anchor atom,
    so both FGs are legitimate and must survive deconfliction."""
    name = name_smiles(smiles)
    assert name == expected, (
        f"SMILES={smiles!r}: got {name!r}, expected {expected!r} — "
        "the geminal same-type-FG path must not regress."
    )


# ---------------------------------------------------------------------------
# Retained parents — unchanged by the FG deconfliction fix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "smiles, expected",
    [
        ("NC(=O)N",                  "urea"),
        ("NC(=S)N",                  "thiourea"),
        ("CNC(=O)NC",                "N,N'-dimethylurea"),
        ("NC(=S)Nc1ccccc1",          "N-phenylthiourea"),
        ("NS(=O)(=O)N",              "sulfamide"),
        ("CS(=O)(=O)N",              "methanesulfonamide"),
    ],
)
def test_retained_and_sulfonamide_parents_unchanged(smiles, expected):
    """Urea / thiourea / sulfamide hit the retained-name or
    functional-parent path upstream of FG SMARTS deconfliction — they
    must continue to return the retained name unchanged."""
    name = name_smiles(smiles)
    assert name == expected, (
        f"SMILES={smiles!r}: got {name!r}, expected {expected!r}"
    )


# ---------------------------------------------------------------------------
# Singleton hydrazone / thioamide / amide — must still match
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "smiles, required_substring",
    [
        # Bare hydrazone — no duplicate motif, must still parse.
        ("CC=NN",        "hydrazine"),
        ("CC(C)=NN",     "hydrazine"),
        # Bare amide/thioamide on a chain — no duplicate motif, must still
        # get the amide/thioamide suffix (via the retained-name path for
        # acetamide, or the FG suffix path otherwise).
        ("CC(=O)N",      "acetamide"),
        ("CCC(=O)N",     "propanamide"),
    ],
)
def test_singleton_fg_still_matched(smiles, required_substring):
    """The fix must not accidentally suppress singleton FG matches that
    happen to live on a carbonyl-N-like motif."""
    name = name_smiles(smiles)
    assert name is not None, f"engine returned None for {smiles!r}"
    assert required_substring in name, (
        f"SMILES={smiles!r}: name {name!r} lost the expected "
        f"{required_substring!r} fragment — the fix over-suppressed a "
        f"legitimate singleton match."
    )


# ---------------------------------------------------------------------------
# Direct unit test against the FGDetection deconfliction output
# ---------------------------------------------------------------------------

def _detect(smiles: str):
    """Helper: run FG detection and return the DetectedFG list."""
    from rdkit import Chem

    from iupac_namer.perception.atoms import AtomAnalysis
    from iupac_namer.perception.fg_detection import FGDetection
    from iupac_namer.perception.rings import RingAnalysis

    mol = Chem.MolFromSmiles(smiles)
    atoms = AtomAnalysis(mol)
    rings = RingAnalysis(mol, atoms)
    return FGDetection(mol, atoms, rings).detected_fgs


def test_thiosemicarbazone_single_thioamide_fg():
    """Acetone thiosemicarbazone: only ONE ``thioamide`` FG should survive
    deconfliction (the one that includes the terminal -NH2)."""
    fgs = [f for f in _detect("CC(C)=NNC(=S)N") if f.type == "thioamide"]
    assert len(fgs) == 1, (
        f"expected exactly 1 surviving thioamide FG, got {len(fgs)}: "
        f"{[(f.type, sorted(f.atoms)) for f in fgs]}"
    )


def test_geminal_diol_two_alcohol_fgs():
    """Propane-1,1-diol: two ``alcohol`` FG matches must survive — the
    two -OH groups are legitimately distinct."""
    fgs = [f for f in _detect("CCC(O)O") if f.type == "alcohol"]
    assert len(fgs) == 2, (
        f"expected 2 surviving alcohol FGs (geminal diol), got {len(fgs)}"
    )
