"""Tests for Stage 14 R14-B + Stage 15 R15-A: isocyano prefix detection.

R14-B: adds the ``isocyano`` (-N#C with formal charges [N+]#[C-]) prefix-only
functional group recognised on aryl substrates via the standard prefix-only
FG path (alongside nitro/nitroso/azido).

R15-A: closes the alkyl substrate path by extending the prefix-FG exclusion
in ``perception/__init__.py`` candidate-parent generation to skip ANY N+
that is a member of any prefix-only FG (anchor or otherwise).  Pre-fix the
isocyano [N+] still leaked through an anchor-only check (its anchor is the
[C-]) and the parent search picked it as an azanium parent over the
substrate carbon, emitting ``(R)(methylidyne)azanium``.  Post-fix all
substrates (alkyl/aryl/cycloalkyl) name as ``isocyano-R``.
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
    td = tempfile.mkdtemp(prefix="isocyano_")
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


def test_phenyl_isocyanide_roundtrips() -> None:
    """``[C-]#[N+]c1ccccc1`` (audit row, phenyl isocyanide).

    Pre-fix: emitted ``[(methylidyne)azaniumyl]benzene`` (azanium parent
    + methylidyne substituent) which OPSIN couldn't reverse to the input
    canonical SMILES.  Post-fix: emitted via the new ``isocyano`` prefix
    on benzene, round-tripping cleanly.
    """
    smi = "[C-]#[N+]c1ccccc1"
    canon_in = _canon(smi)
    name = name_smiles(smi)
    assert name is not None
    assert "isocyano" in name, f"expected isocyano in {name!r}"
    rt = _opsin_rt(name)
    assert rt is not None, f"OPSIN rejected {name!r}"
    assert _canon(rt) == canon_in, f"rt mismatch: {name!r} → {_canon(rt)!r}"


def test_methyl_isocyanide_roundtrips() -> None:
    """``[C-]#[N+]C`` (methyl isocyanide).  R15-A closes the alkyl path."""
    smi = "[C-]#[N+]C"
    canon_in = _canon(smi)
    name = name_smiles(smi)
    assert name is not None
    assert "isocyano" in name and "azanium" not in name, f"got {name!r}"
    rt = _opsin_rt(name)
    assert rt is not None and _canon(rt) == canon_in, (
        f"name={name!r} rt={_canon(rt)!r} expected {canon_in!r}"
    )


def test_ethyl_isocyanide_roundtrips() -> None:
    """``CC[N+]#[C-]`` (ethyl isocyanide).  R15-A."""
    smi = "CC[N+]#[C-]"
    canon_in = _canon(smi)
    name = name_smiles(smi)
    rt = _opsin_rt(name) if name else None
    assert name is not None and "isocyano" in name and "azanium" not in name, f"got {name!r}"
    assert rt is not None and _canon(rt) == canon_in


def test_isopropyl_isocyanide_roundtrips() -> None:
    """``[N+](#[C-])C(C)C`` (isopropyl isocyanide).  R15-A."""
    smi = "[N+](#[C-])C(C)C"
    canon_in = _canon(smi)
    name = name_smiles(smi)
    rt = _opsin_rt(name) if name else None
    assert name is not None and "isocyano" in name and "azanium" not in name
    assert rt is not None and _canon(rt) == canon_in


def test_cyclopentyl_isocyanide_roundtrips() -> None:
    """``C1CCCC1[N+]#[C-]`` (cyclopentyl isocyanide).  R15-A — verifies
    the parent-search picks the cycloalkane ring, not the [N+], when the
    isocyano FG covers the [N+].
    """
    smi = "C1CCCC1[N+]#[C-]"
    canon_in = _canon(smi)
    name = name_smiles(smi)
    rt = _opsin_rt(name) if name else None
    assert name is not None and "isocyano" in name and "cyclopentane" in name
    assert rt is not None and _canon(rt) == canon_in


def test_azanium_controls_unaffected() -> None:
    """Ensure R15-A's broader N+ exclusion does NOT regress unrelated
    azanium parent-hydride emissions: methylazanium, ethylazanium,
    azanium all should keep their existing names.

    Note: tetramethylazanium was the prior name for (CH3)4N+; per
    P-62.6 the retained PIN for the fully quaternary cation NR4+ is
    "ammonium" (tetramethylammonium), so the case now expects the
    retained spec PIN form.  Partially substituted N+ (NH3R+, NH2R2+,
    NHR3+) remain on the systematic 'azanium' parent.
    """
    cases = [
        ("[NH4+]",        "azanium"),
        ("C[NH3+]",       "methylazanium"),
        ("CC[NH3+]",      "ethylazanium"),
        ("C[N+](C)(C)C",  "tetramethylammonium"),
    ]
    for smi, expected in cases:
        got = name_smiles(smi)
        assert got == expected, f"regression on {smi!r}: got {got!r}, expected {expected!r}"


def test_nitro_and_nitroso_controls_unaffected() -> None:
    """The R15-A change widened the prefix-FG exclusion from
    anchor-only to atom-set membership.  For nitro/nitroso/diazo the
    [N+] is BOTH anchor and member, so the change is a no-op for those
    FGs.  Verify their names still emit correctly.
    """
    nitro = name_smiles("C[N+](=O)[O-]")
    assert nitro is not None and "nitro" in nitro and "methane" in nitro

    nitroso = name_smiles("CN=O")
    assert nitroso is not None and "nitroso" in nitroso and "methane" in nitroso


def test_isocyano_fg_detection_marks_two_atoms() -> None:
    """Ensure the isocyano SMARTS only claims [C-] and [N+] (not the
    substrate aryl carbon).

    The atoms set will include the [#6] anchor too, by SMARTS-match
    convention (cf. nitro), but the anchor must be the [C-] terminal.
    """
    from iupac_namer.perception import Perception

    m = Chem.MolFromSmiles("[C-]#[N+]c1ccccc1")
    perc = Perception(m)
    iso = [fg for fg in perc.fgs.detected_fgs if fg.type == "isocyano"]
    assert len(iso) == 1, f"expected 1 isocyano FG, got {len(iso)}: {iso}"
    fg = iso[0]
    # anchor is the terminal [C-] (idx 0 in the canonical SMILES order
    # after RDKit re-orders for substruct match).
    assert m.GetAtomWithIdx(fg.anchor).GetSymbol() == "C"
    assert m.GetAtomWithIdx(fg.anchor).GetFormalCharge() == -1
    # FG atom set covers exactly the [C-]#[N+]+substrate-C trio.
    assert len(fg.atoms) == 3
