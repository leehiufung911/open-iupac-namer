"""tests/test_bridged_tetrahedral_stereo.py

Stage 22 R22-D regression tests.

R22-D extends the R22-C OPSIN-validation pass to cover bridged/spiro
ring parents at plain-integer locants.  The relaxation in
``_collect_stereo_descriptors`` admits those tetrahedral R/S
descriptors; ``_validate_stereo_via_opsin`` then strips them when
OPSIN cannot anchor the resulting name (Stage 6 R1-I tropane /
morphinan class — OPSIN returns an empty result for the candidate).

Coverage:

* Camphor (``bicyclo[2.2.1]heptan-2-one`` parent) — OPSIN parses
  ``(1R,4R)``, descriptors must survive.
* Norbornene-derived stereocenters — same parent class, must survive.
* Tropane-class chiral substituents
  (``8-methyl-8-azabicyclo[3.2.1]octan-3-yl ...``) — OPSIN rejects
  the candidate name when stereo is present, validator must strip.
* Achiral bridged parents — must NOT spuriously gain a stereo prefix.
"""

from __future__ import annotations

import re

from iupac_namer.engine import name_smiles


_STEREO_PREFIX_RE = re.compile(r"^\(\s*[^)]*[RSEZ][^)]*\)-")


def test_camphor_emits_1R_4R() -> None:
    """(1R,4R)-camphor: bridged plain-int locant tetrahedral R/S survives
    R22-D's OPSIN-validation pass (the candidate name parses cleanly).
    """
    name = name_smiles("C[C@@]12C(C[C@@H](CC1)C2(C)C)=O")
    assert name == "(1R,4R)-1,7,7-trimethylbicyclo[2.2.1]heptan-2-one", (
        f"got {name!r}"
    )


def test_camphor_round_trips_through_opsin() -> None:
    """The emitted name must canonicalise back to the input SMILES via
    OPSIN.  This is the load-bearing safety property R22-D buys: the
    descriptor isn't just present, it's the right one.
    """
    try:
        from py2opsin import py2opsin
    except ImportError:
        return  # OPSIN unavailable — eval will catch round-trip failures.
    from rdkit import Chem
    target = Chem.MolToSmiles(Chem.MolFromSmiles(
        "C[C@@]12C(C[C@@H](CC1)C2(C)C)=O"
    ))
    name = name_smiles("C[C@@]12C(C[C@@H](CC1)C2(C)C)=O")
    out = py2opsin(name)
    assert out, f"OPSIN rejected emitted name {name!r}"
    assert Chem.MolToSmiles(Chem.MolFromSmiles(out)) == target


def test_norbornene_chiral_emits_descriptor() -> None:
    """7,7-dimethylbicyclo[2.2.1]hept-2-ene-derived chiral form.
    Must emit a stereo descriptor (R22-D admits bridged plain-int
    tetrahedral R/S; this candidate OPSIN-parses).
    """
    name = name_smiles("CC1(C)[C@@H]2CC[C@@H]1C=C2")
    assert _STEREO_PREFIX_RE.match(name), (
        f"chiral norbornene emitted no stereo prefix; got {name!r}"
    )


def test_tropane_substituent_strips_unparseable_stereo() -> None:
    """Stage 6 R1-I regression class: a chiral
    ``8-methyl-8-azabicyclo[3.2.1]octan-3-yl`` SUBSTITUENT (atropine-
    like ester) produces a candidate name that OPSIN rejects.  The
    R22-D validator must strip the bridged tetrahedral R/S so the name
    remains parseable (matching pre-R22-D behaviour for this class).
    """
    # Atropine-shaped scaffold; the stereocenter at the C3-O of the ester
    # would emit ``(3R)`` or similar but OPSIN rejects the full name.
    name = name_smiles("CN1[C@H]2CC[C@@H]1C[C@@H](OC(=O)C(CO)c1ccccc1)C2")
    # The validator must have stripped any bridged ring R/S — the result
    # should be parseable through OPSIN even if it lacks the descriptor.
    try:
        from py2opsin import py2opsin
    except ImportError:
        return
    out = py2opsin(name)
    assert out, (
        f"R22-D validator left an OPSIN-unparseable name on a tropane "
        f"substituent: {name!r}"
    )


def test_achiral_bridged_no_spurious_stereo() -> None:
    """Achiral ``bicyclo[2.2.1]heptane`` must not gain a stereo prefix."""
    name = name_smiles("C1CC2CCC1C2")
    assert not _STEREO_PREFIX_RE.match(name), (
        f"achiral bicyclo[2.2.1]heptane spuriously got stereo prefix: {name!r}"
    )
    assert "bicyclo" in name


def test_simple_chiral_tropane_round_trip() -> None:
    """Simple ``8-methyl-8-azabicyclo[3.2.1]octane`` with a chiral SMILES
    — the parent itself can carry ``(1S,5R)`` per R22-D; the validator
    only strips when OPSIN actually rejects.  This case OPSIN parses, so
    the descriptors should survive.
    """
    name = name_smiles("CN1[C@H]2CC[C@@H]1CCC2")
    try:
        from py2opsin import py2opsin
    except ImportError:
        # Without OPSIN we still know the name is at least non-empty
        # and contains the expected stem.
        assert "azabicyclo" in name
        return
    out = py2opsin(name)
    assert out, f"R22-D validator left unparseable name {name!r}"
    # A descriptor SHOULD be present here — OPSIN accepts the candidate.
    assert _STEREO_PREFIX_RE.match(name), (
        f"simple chiral tropane lost its stereo prefix despite OPSIN "
        f"accepting the candidate; got {name!r}"
    )
