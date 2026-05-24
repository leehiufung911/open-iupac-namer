"""Tests for Stage 15 R15-B + R15-C: stereo emission on complex ring parents.

Pre-fix: ``_collect_stereo_descriptors`` (engine.py) had two layers that
together silently dropped stereo descriptors on complex-ring parents:

1.  An early-return guard skipped ALL stereo emission when the parent
    was a non-monocyclic ring system (fused/bridged/spiro/retained), to
    avoid the Stage 6 R1-I regression on tetrahedral stereo at letter-
    suffixed bridgehead locants.

2.  The per-double-bond loop required BOTH bond endpoints to be in the
    parent atom set (``begin in parent_atoms and end in parent_atoms``).
    A bond crossing the parent-substituent boundary (e.g. C2 of a benzo-
    furanone parent ↔ exocyclic methylidene C of a ``=CHPh`` substituent)
    failed this check and was silently dropped.

R15-B narrows both gates for double-bond stereo:

- The early return becomes a per-descriptor flag (``skip_tetrahedral``)
  that only suppresses tetrahedral stereo on complex ring parents.
- Double-bond stereo with one endpoint in the parent and the other in a
  substituent now anchors the descriptor at the parent endpoint's locant.
- A safety gate keeps double-bond stereo on complex ring parents to
  plain-integer locants only — letter-suffixed locants like ``4a`` /
  ``12b`` (junction atoms) still skip emission.

R15-C extends the relaxation to tetrahedral stereo:

- Tetrahedral R/S on FUSED ring parents (not bridged/spiro) at plain-
  integer locants is now emitted.  Closes audit-row regressions for
  chroman C3 (``3-phenylchroman-7-ol``) and tetralin C1
  (``tetralin-1-amine``).
- Bridged von-Baeyer parents (camphor, tropane class) still skip
  tetrahedral stereo — preserves the Stage 6 R1-I -6 regression fix.
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
    td = tempfile.mkdtemp(prefix="r15b_")
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


def _assert_roundtrips(smi: str) -> str:
    """Assert that the engine names ``smi`` and OPSIN reverses it back to the
    same RDKit canonical SMILES.  Returns the emitted name for inspection.
    """
    canon_in = _canon(smi)
    name = name_smiles(smi)
    assert name is not None and "NAMING ERROR" not in name, (
        f"naming error for {smi!r}: {name!r}"
    )
    rt = _opsin_rt(name)
    assert rt is not None, f"OPSIN rejected {name!r} from {smi!r}"
    assert _canon(rt) == canon_in, (
        f"round-trip mismatch for {smi!r}: emit={name!r} → "
        f"rt={_canon(rt)!r} expected={canon_in!r}"
    )
    return name


# --- Audit-row regression cases (Stage 14 R14-B / R15-B promised cases) ---

def test_chloro_phenylmethylidene_benzofuranone_E() -> None:
    """Audit row from ``opsin_audit_rings_raw.csv``: chloro-substituted
    benzofuran-3-one with an exocyclic ``(E)``-phenylmethylidene at C2.
    Pre-fix the engine emitted the heavy-atom-correct
    ``4-chloro-2-(phenylmethylidene)-2,3-dihydro-1-benzofuran-3-one``
    but dropped the (2E)/(2Z) descriptor; OPSIN re-derived the wrong
    geometric isomer.
    """
    name = _assert_roundtrips("O=C1/C(=C\\c2ccccc2)Oc2cccc(Cl)c21")
    assert "(2E)" in name, f"expected (2E) in {name!r}"


def test_chloro_phenylmethylidene_benzofuranone_Z() -> None:
    name = _assert_roundtrips("O=C1/C(=C/c2ccccc2)Oc2cccc(Cl)c21")
    assert "(2Z)" in name, f"expected (2Z) in {name!r}"


def test_phenylmethylidene_benzofuranone_unsubstituted() -> None:
    """Audit row from ``opsin_audit_natural_raw.csv``: parent
    benzofuranone (no chloro) with the same exocyclic methylidene."""
    name = _assert_roundtrips("O=C1/C(=C/c2ccccc2)Oc2ccccc21")
    assert "(2Z)" in name or "(2E)" in name


# --- Controls: must not regress ---

def test_simple_chain_ez_unaffected() -> None:
    """Chain stereo (parent = chain) was already emitting; verify no
    regression from the gate change.
    """
    e = name_smiles("C/C=C/C")
    z = name_smiles("C/C=C\\C")
    assert e == "(2E)-but-2-ene", f"got {e!r}"
    assert z == "(2Z)-but-2-ene", f"got {z!r}"


def test_bridged_tetrahedral_emits_when_opsin_accepts() -> None:
    """Stage 22 R22-D: bridged-parent tetrahedral R/S now emits at plain-int
    locants when OPSIN accepts the candidate (camphor's
    ``bicyclo[2.2.1]heptan-2-one`` parent parses ``1R,4R`` cleanly).
    The Stage 6 R1-I tropane/morphinan unparseable cases are still handled
    safely — see ``test_bridged_tetrahedral_stereo.py`` for the OPSIN-
    validator-driven strip path.
    """
    camphor = name_smiles("C[C@@]12C(C[C@@H](CC1)C2(C)C)=O")
    assert camphor == "(1R,4R)-1,7,7-trimethylbicyclo[2.2.1]heptan-2-one", (
        f"got {camphor!r}"
    )


def test_phenylchroman_R_emits_3R() -> None:
    """Audit row from ``opsin_audit_natural_raw.csv``: chroman C3 stereo
    was dropped pre-R15-C (chroman is a fused 6+6 ring with O, so the
    pre-R15-C guard skipped tetrahedral stereo emission entirely).

    Per P-53 / P-54.4.3.2 the retained name "chroman" is general
    nomenclature only; PIN spelling is "3,4-dihydro-2H-1-benzopyran".
    The data_loader pin_eligible=False alias swaps the spelling on PIN
    emission, so the C3 stereo locant survives unchanged.
    """
    name = _assert_roundtrips("C1(=CC=CC=C1)[C@@H]1COC2=C(C1)C=CC(=C2)O")
    assert (
        "(3R)" in name
        and "3-phenyl" in name
        and "benzopyran" in name
        and "-7-ol" in name
    ), f"got {name!r}"


def test_phenylchroman_S_emits_3S() -> None:
    name = _assert_roundtrips("C1(=CC=CC=C1)[C@H]1COC2=C(C1)C=CC(=C2)O")
    assert "(3S)" in name


def test_tetralin_R_emits_1R() -> None:
    """1,2,3,4-tetrahydronaphthalene (PIN per P-25.3.1.3; tetraline is
    general nomenclature only) is a fused 6+6 carbocycle.  Its C1 amine
    emits R/S after R15-C.  The data_loader pin_eligible=False alias
    swaps the spelling on PIN emission.
    """
    name = _assert_roundtrips("[C@@H]1(N)CCCc2ccccc21")
    assert name == "(1R)-1,2,3,4-tetrahydronaphthalen-1-amine", f"got {name!r}"


def test_tetralin_S_emits_1S() -> None:
    name = _assert_roundtrips("[C@H]1(N)CCCc2ccccc21")
    assert name == "(1S)-1,2,3,4-tetrahydronaphthalen-1-amine", f"got {name!r}"


def test_double_bond_inside_parent_still_works() -> None:
    """Double-bond stereo with both endpoints in the parent atom set
    (the original supported case) must continue to work after R15-B's
    boundary-case relaxation.
    """
    # (3E)-pent-3-enoic acid — parent is the chain, both =C atoms in parent.
    # (the C=C is at C3-C4 because the suffix-bearing -COOH is C1.)
    name = name_smiles("C/C=C/CC(=O)O")
    assert name == "(3E)-pent-3-enoic acid", f"got {name!r}"
