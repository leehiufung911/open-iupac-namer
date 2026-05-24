"""
tests/test_parent_cip_inheritance.py

Tests for parent-CIP inheritance on carved substituents (commits 32cb5e7 and
6bf250a).

Background
----------
When ``carve_substituent`` cuts a substituent off the parent mol, the
parent-side neighbour of the attachment atom is replaced with H.  Two
consequences for CIP:

1. At the attachment atom itself: it used to have a real neighbour (N/C/O
   etc.), now it has H.  If the atom was a stereocenter, RDKit may no
   longer perceive it as one at all (two H's → not a stereocenter).
   Commit ``32cb5e7`` fixes this by stashing ``_ParentCIPCode`` on the
   attachment atom and having ``StereoAnalysis._detect_tetrahedral`` emit
   a virtual tetrahedral stereocenter using that property.

2. At atoms INSIDE the carved substituent (2+ bonds from attachment):
   RDKit can still perceive them as stereocenters, but the CIP priority
   ranking may flip because the parent-side chain no longer contributes
   to ranking.  IUPAC P-92.1.4.3 says the free valence is treated as a
   phantom atom of the parent's identity, so the in-parent CIP descriptor
   should be reported.  Commit ``6bf250a`` extends the stash to EVERY
   parent atom that has a ``_CIPCode``, and propagates to every carved
   fragment atom via the parent-idx → fragment-idx map.

These tests exercise both cases with small peptide-like and polyol SMILES.
All expected names are OPSIN-round-trip verified with stereo preserved.
"""

from __future__ import annotations

import pytest
from rdkit import Chem

from iupac_namer.engine import name_smiles

try:
    from py2opsin import py2opsin
    HAVE_OPSIN = True
except ImportError:
    HAVE_OPSIN = False


def _stereo_match(smi_in: str, smi_out: str) -> bool:
    """Compare two SMILES PRESERVING stereochemistry."""
    m1 = Chem.MolFromSmiles(smi_in)
    m2 = Chem.MolFromSmiles(smi_out)
    if m1 is None or m2 is None:
        return False
    return Chem.MolToSmiles(m1) == Chem.MolToSmiles(m2)


def _opsin_roundtrip_stereo(smi: str, name: str) -> bool:
    assert HAVE_OPSIN, "py2opsin not available"
    parsed = py2opsin([name], output_format="SMILES")
    if not parsed or not parsed[0]:
        return False
    return _stereo_match(smi, parsed[0])


class TestParentCIPAtAttachmentAtom:
    """Commit 32cb5e7 — CIP preserved at the carved attachment atom."""

    def test_boc_alanine_methylamide(self):
        """Boc-Ala-NHMe: carving the Ala residue through the amide N-Calpha
        bond gives the alpha-carbon H's on both sides after replacement of
        the parent C=O neighbour with H.  Without the fix, RDKit drops the
        stereocenter.  The output must carry ``(2S)`` on the alanyl.
        """
        smi = "CC(C)(C)OC(=O)N[C@@H](C)C(=O)NC"
        name = name_smiles(smi)
        assert "(2S)" in name, f"expected (2S) CIP on alanyl in {name!r}"
        assert name == (
            "2-methylpropan-2-yl N-[(2S)-1-(methylamino)-1-oxopropan-2-yl]"
            "carbamate"
        )
        if HAVE_OPSIN:
            assert _opsin_roundtrip_stereo(smi, name), (
                f"stereo round-trip failed: {name}"
            )

    def test_boc_valine_amide(self):
        """Boc-Val-NH2: the valyl Calpha sits inside a carved substituent,
        and its CIP must be reported as (2S).
        """
        smi = "CC(C)[C@H](NC(=O)OC(C)(C)C)C(=O)N"
        name = name_smiles(smi)
        assert "(2S)" in name, f"expected (2S) CIP in {name!r}"
        if HAVE_OPSIN:
            assert _opsin_roundtrip_stereo(smi, name), (
                f"stereo round-trip failed: {name}"
            )

    def test_alpha_methylbenzyl_acetamide(self):
        """N-((R)-1-phenylethyl)acetamide — the stereocenter is at the
        carved attachment atom of a sec-amine substituent.
        """
        smi = "[C@@H](C)(c1ccccc1)NC(=O)C"
        name = name_smiles(smi)
        assert "(1R)" in name, f"expected (1R) CIP in {name!r}"
        if HAVE_OPSIN:
            assert _opsin_roundtrip_stereo(smi, name), (
                f"stereo round-trip failed: {name}"
            )


class TestParentCIPDeepInsideSubstituent:
    """Commit 6bf250a — CIP preserved for atoms 2+ bonds INSIDE the carved
    substituent.  This is the case that pure attachment-atom stashing
    cannot fix: RDKit perceives the center but may assign a flipped CIP
    because the parent-side branch is now an H.
    """

    def test_alanyl_alanine_amide(self):
        """(R)-Ala-(R)-Ala amide: two stereocenters, one at the parent
        principal-chain Calpha, one inside the carved alanyl substituent
        on the amide N.  Both must keep (2R) CIP.
        """
        smi = "C[C@@H](N)C(=O)N[C@H](C)C(=O)N"
        name = name_smiles(smi)
        # Two (2R) labels expected — one on the parent, one on the
        # substituent.
        assert name.count("(2R)") == 2, f"expected two (2R) in {name!r}"
        if HAVE_OPSIN:
            assert _opsin_roundtrip_stereo(smi, name), (
                f"stereo round-trip failed: {name}"
            )

    def test_threonine_amide(self):
        """Threonine amide: the beta-hydroxyl carbon is a stereocenter 2
        bonds from the amide carbonyl, i.e. 2 bonds inside the carved
        chain once the amide is parent.  Must give (2S,3R)."""
        smi = "N[C@@H]([C@@H](C)O)C(=O)N"
        name = name_smiles(smi)
        assert "(2S,3R)" in name, f"expected (2S,3R) CIP in {name!r}"
        if HAVE_OPSIN:
            assert _opsin_roundtrip_stereo(smi, name), (
                f"stereo round-trip failed: {name}"
            )

    def test_polyol_amide_internal_stereo(self):
        """2,3,4-trihydroxy-N-methylbutanamide: stereo at C2 and C3, both
        inside the butanamide chain.  Exercises the full-parent-CIP-map
        stashing (6bf250a) for atoms well inside a carved fragment after
        amide disconnection.
        """
        smi = "OC[C@@H](O)[C@H](O)C(=O)NC"
        name = name_smiles(smi)
        assert "(2S,3R)" in name, f"expected (2S,3R) in {name!r}"
        if HAVE_OPSIN:
            assert _opsin_roundtrip_stereo(smi, name), (
                f"stereo round-trip failed: {name}"
            )


@pytest.mark.skipif(not HAVE_OPSIN, reason="py2opsin not installed")
class TestOpsinVerifiedParentCIP:
    """OPSIN-round-trip batch check for parent-CIP inheritance."""

    @pytest.mark.parametrize("smi", [
        "CC(C)(C)OC(=O)N[C@@H](C)C(=O)NC",        # Boc-Ala-NHMe
        "CC(C)[C@H](NC(=O)OC(C)(C)C)C(=O)N",      # Boc-Val-NH2
        "C[C@@H](N)C(=O)N[C@H](C)C(=O)N",         # Ala-Ala amide
        "N[C@@H]([C@@H](C)O)C(=O)N",              # threonine amide
        "OC[C@@H](O)[C@H](O)C(=O)NC",             # trihydroxy-N-Me-butanamide
    ])
    def test_stereo_round_trip(self, smi):
        name = name_smiles(smi)
        parsed = py2opsin([name], output_format="SMILES")
        assert parsed and parsed[0], f"OPSIN could not parse: {name}"
        assert _stereo_match(smi, parsed[0]), (
            f"stereo mismatch: {smi} -> {name} -> {parsed[0]}"
        )
