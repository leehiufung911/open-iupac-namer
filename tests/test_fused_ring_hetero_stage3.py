"""Stage 3 fused-ring naming: saturated and partly-saturated parent
rings with a [1,3]-dihetero 5- or 6-ring smaller component.

These tests verify that:

  1. A saturated or partly-saturated BASE ring (cyclohexane / piperidine /
     decalin-style / partly-aromatic benzene) fused with a [1,3]-dihetero
     smaller partner now gets a ``<hydro-prefix>[1,3]<smaller>[4,5-X]<base>``
     name, where the base portion is the canonical AROMATIC form and the
     hydro-prefix expresses saturation.

  2. Each generated name round-trips through OPSIN to the same canonical
     SMILES as the input.

  3. The Stage 3 systematic name is emitted under
     ``naming_method='fused_hetero_hydro'`` (scored above von_baeyer), so
     saturated dioxolo-benzene-class heterocycles prefer the fused name
     over a bare VB decomposition.

  4. Existing Stage 1 and Stage 2 aromatic cases are unchanged.
"""
from __future__ import annotations

import os

# Pin OPSIN's JAVA_HOME the same way authoritative_eval.py does.
os.environ.setdefault(
    "JAVA_HOME",
    os.environ.get("JAVA_HOME", ""),
)
os.environ["PATH"] = (
    os.environ["JAVA_HOME"] + "/bin" + os.pathsep + os.environ.get("PATH", "")
)

import pytest
from rdkit import Chem

from iupac_namer.engine import name_smiles


def _canonical(smiles: str) -> str | None:
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(m) if m is not None else None


def _opsin_round_trip(name: str) -> str | None:
    try:
        from py2opsin import py2opsin
    except ImportError:  # pragma: no cover
        pytest.skip("py2opsin not installed")
    out = py2opsin(name)
    if not out:
        return None
    return _canonical(out)


# ---------------------------------------------------------------------------
# Stage 3 cases: the expected name is the canonical fused-hetero form
# with a hydro- prefix.  All targets are OPSIN-verified below.
# ---------------------------------------------------------------------------

STAGE3_CASES = [
    # Fully saturated mono-ring bases
    # 5-ring smaller (dioxolo) + fully-saturated 6-ring base
    ("O1COC2C1CCCC2", "hexahydro-[1,3]dioxolo[4,5-b]benzene"),
    # 6-ring smaller (dioxino) + fully-saturated 6-ring base
    ("O1COCC2C1CCCC2", "hexahydro-[1,3]dioxino[4,5-b]benzene"),
    # 5-ring smaller + fully-saturated pyridine base (piperidine-fused)
    ("O1COC2NCCCC21", "hexahydro-[1,3]dioxolo[4,5-b]pyridine"),
    # Fully saturated 1,4-diazine base (piperazine-fused)
    ("C1CNC2OCOC2N1", "hexahydro-[1,3]dioxolo[4,5-b]pyrazine"),
    # Fully saturated 1,3-diazine base (imidazolidine-ring-fused):
    ("C1NCC2OCOC2N1", "hexahydro-[1,3]dioxolo[4,5-d]pyrimidine"),

    # Partly saturated benzene bases (hydro-locants emitted explicitly)
    # All four non-fusion atoms sp3; fusion atoms stay sp2:
    ("O1COC2=C1CCCC2", "4,5,6,7-tetrahydro-[1,3]dioxolo[4,5-b]benzene"),
    # Two adjacent sp3 atoms at 4,5:
    ("O1COC2=C1C=CCC2", "4,5-dihydro-[1,3]dioxolo[4,5-b]benzene"),
    # Two non-adjacent sp3 atoms at 4,7:
    ("O1COC2=C1CC=CC2", "4,7-dihydro-[1,3]dioxolo[4,5-b]benzene"),

    # Fully saturated multi-ring base (decalin-style):
    ("O1COC2C1CC1CCCCC1C2", "decahydro-[1,3]dioxolo[4,5-b]naphthalene"),
]


@pytest.mark.parametrize("smi,expected_name", STAGE3_CASES)
def test_stage3_fused_ring_name(smi: str, expected_name: str) -> None:
    """Each Stage 3 input gets the expected hydro-prefixed fusion name."""
    got = name_smiles(smi)
    assert got == expected_name, (
        f"Expected {expected_name!r} for {smi!r}, got {got!r}"
    )


@pytest.mark.parametrize("smi,expected_name", STAGE3_CASES)
def test_stage3_round_trip(smi: str, expected_name: str) -> None:
    """The generated name must round-trip through OPSIN to the same
    canonical SMILES as the input."""
    got = name_smiles(smi)
    if got.startswith("[NAMING ERROR"):
        pytest.fail(f"Namer failed for {smi}: {got}")
    rt = _opsin_round_trip(got)
    expected_canon = _canonical(smi)
    assert rt == expected_canon, (
        f"Round-trip mismatch for {smi}: name={got!r} → {rt!r}, "
        f"expected {expected_canon!r}"
    )


# ---------------------------------------------------------------------------
# Regression guards: Stage 1 / Stage 2 aromatic cases still resolve to the
# same names (retained where curated, systematic where not).
# ---------------------------------------------------------------------------

def test_stage3_does_not_disturb_aromatic_retained() -> None:
    """1,3-benzodioxole must still win over the hydro-prefixed form: the
    input is fully aromatic, Stage 3's hydro logic is not triggered."""
    assert name_smiles("c1ccc2c(c1)OCO2") == "1,3-benzodioxole"


def test_stage3_does_not_disturb_aromatic_systematic() -> None:
    """Aromatic dioxolo-pyrazine (no retained name) still resolves to the
    Stage 2A systematic form (no hydro- prefix)."""
    assert name_smiles("c1cnc2c(n1)OCO2") == "[1,3]dioxolo[4,5-b]pyrazine"


def test_stage3_partly_saturated_fusion_atoms_kept_sp2() -> None:
    """When only the non-fusion atoms of the base are saturated, the
    fusion atoms stay aromatic in the canonical parent — so the hydro-
    prefix lists only the non-fusion locants (4,5,6,7 for a benzene base)."""
    got = name_smiles("O1COC2=C1CCCC2")
    assert got == "4,5,6,7-tetrahydro-[1,3]dioxolo[4,5-b]benzene"
    # 3a, 7a (fusion) NOT in the hydro locants
    assert "3a" not in got and "7a" not in got


def test_stage3_fully_saturated_emits_bare_hexahydro() -> None:
    """Fully-saturated base: prefer the bare ``hexahydro-`` form (no
    explicit locants) per P-31.1.4.2.4 brevity convention."""
    got = name_smiles("O1COC2C1CCCC2")
    assert got.startswith("hexahydro-")
    assert "," not in got.split("hexahydro-", 1)[0]


def test_stage3_naming_method_is_fused_hetero_hydro() -> None:
    """Stage 3 emissions use the dedicated ``fused_hetero_hydro`` method
    rank (not plain ``systematic``) so they beat von_baeyer fallbacks."""
    from iupac_namer.perception.atoms import AtomAnalysis
    from iupac_namer.perception.rings import RingAnalysis
    from iupac_namer.ring_naming.fused import name_fused
    from iupac_namer.types import CandidateParent
    mol = Chem.MolFromSmiles("O1COC2C1CCCC2")
    aa = AtomAnalysis(mol)
    ra = RingAnalysis(mol, aa)
    rs = ra._ring_systems[0]
    cand = CandidateParent(
        atom_indices=rs.atom_indices,
        type="fused",
        length=len(rs.atom_indices),
        ring_system=rs,
        unsaturation=None,
        element=None,
        lambda_value=None,
    )
    parents = name_fused(rs, cand, mol)
    assert parents, "Stage 3 should emit a NamedParent for saturated dioxolo-benzene"
    assert parents[0].naming_method == "fused_hetero_hydro"
