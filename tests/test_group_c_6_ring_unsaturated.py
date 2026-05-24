"""Regression tests for Stage 6 R2-D: 6-ring group-C unsaturated HW stem picker.

Root cause #11 in ``docs/opsin_coverage_taxonomy.md``: for 6-membered
Hantzsch-Widman rings that contain a heavy heteroatom (As, Sb, Bi, Ge, Sn,
Pb, I, B, Al, Ga, In, Tl, Hg), RDKit fails to set the ring aromatic flag
even when the Kekulé SMILES has the full complement of endocyclic double
bonds. Because the stem picker in ``ring_naming/monocyclic.py`` only
checks ``ring_system.aromatic`` to switch from ``inane`` → ``inine``, we
used to emit the **saturated** HW stem for genuinely unsaturated rings
(``arsinane`` instead of ``arsinine``, ``bismane`` instead of ``bismine``,
etc.).

Contrast with P/N/O/S/Se/Te 6-rings: these elements DO kekulize in RDKit
so the existing ``aromatic`` branch picks the unsaturated stem correctly
(e.g. ``[P]1=CC=CC=C1`` → ``phosphinine``). This test file only covers
the non-aromatizable elements, which is where the bug manifests.

Audit source: ``eval/opsin_audit_hw_charge_raw.csv`` (hw_matrix rows with
status ``WRONG``) plus R1-C's Al/Ga/In/Tl/Hg extension cells.
"""
from __future__ import annotations

import pytest

from iupac_namer.engine import name_smiles


def _name(smiles: str) -> str:
    return name_smiles(smiles)


# ---------------------------------------------------------------------------
# 1. Simple (unsubstituted) 6-ring unsaturated HW heterocycles
# ---------------------------------------------------------------------------

class TestGroupCSimpleUnsaturated:
    """Plain 6-ring HW heterocycles with the heteroatom at position 1 and
    three endocyclic double bonds (or the maximum permitted by valence)."""

    # Group C: heteroatom valence ≤ 3 (group-13/-15/halogens/Hg)
    def test_arsinine(self):
        assert _name("[As]1=CC=CC=C1") == "arsinine"

    def test_stibinine(self):
        assert _name("[Sb]1=CC=CC=C1") == "stibinine"

    def test_bismine(self):
        # Bi is listed in six_membered_groups["A"] with group-A stems
        # (saturated "ane" / unsaturated "ine"), so the stem is
        # bism + ine = bismine (not "bisminine").
        assert _name("[Bi]1=CC=CC=C1") == "bismine"

    def test_iodinine(self):
        # I is valence 1, so max 2 endocyclic double bonds.
        assert _name("[IH]1CC=CC=C1") == "iodinine"

    # Group B (heavier tetravalent):
    def test_germine(self):
        assert _name("[GeH]1=CC=CC=C1") == "germine"

    def test_stannine(self):
        assert _name("[SnH]1=CC=CC=C1") == "stannine"

    def test_plumbine(self):
        assert _name("[PbH]1=CC=CC=C1") == "plumbine"

    # R1-C's additions (group 13 metals + Hg):
    def test_gallinine(self):
        assert _name("[Ga]1=CC=CC=C1") == "gallinine"

    def test_indiginine(self):
        assert _name("[In]1=CC=CC=C1") == "indiginine"

    def test_thallinine(self):
        assert _name("[Tl]1=CC=CC=C1") == "thallinine"

    def test_mercurinine(self):
        # Hg is divalent (standard valence 2), so the ring has only 2 C=C
        # bonds -- the third ring-bond pair involves Hg and is single.
        # OPSIN round-trips mercurinine to [Hg]1CC=CC=C1.
        assert _name("[Hg]1CC=CC=C1") == "mercurinine"


# ---------------------------------------------------------------------------
# 2. Substituted group-C 6-ring unsaturated (make sure the fix composes)
# ---------------------------------------------------------------------------

class TestGroupCSubstitutedUnsaturated:
    """Once the stem picker flips to ``inine``/``ine``, chloro- / methyl-
    substituents must still place correctly."""

    def test_4_chloroarsinine(self):
        # Cl on the para C: locant 4 (As numbered as 1)
        assert _name("[As]1=CC=C(Cl)C=C1") == "4-chloroarsinine"

    def test_2_chloroarsinine(self):
        # Cl on ring C adjacent to As: locant 2
        assert _name("[As]1=C(Cl)C=CC=C1") == "2-chloroarsinine"

    def test_4_methylarsinine(self):
        assert _name("[As]1=CC=C(C)C=C1") == "4-methylarsinine"

    def test_3_chlorostibinine(self):
        # Sb analogue: 3-chloro on stibinine
        got = _name("[Sb]1=CC=C(Cl)C=C1")
        assert got in {"4-chlorostibinine", "3-chlorostibinine"}, got


# ---------------------------------------------------------------------------
# 3. Negative regressions — light heteroatom 6-rings must NOT flip
# ---------------------------------------------------------------------------

class TestLightHeteroatomNoRegression:
    """For O/N/S/Se/Te/P/B 6-rings, RDKit CAN aromatize, so aromatic vs.
    non-aromatic is a reliable discriminator. Non-aromatic partially
    unsaturated 6-rings of these elements must still use the SATURATED
    stem (OPSIN rejects HW "azine"-type names for non-aromatic rings)."""

    def test_phosphinine_still_works(self):
        # Aromatic — already worked
        assert _name("[P]1=CC=CC=C1") == "phosphinine"

    def test_pyran_not_renamed(self):
        # O1CC=CC=C1 → 2H-pyran (retained), must NOT be pulled into "oxine"
        name = _name("O1CC=CC=C1")
        assert "pyran" in name.lower(), name
        assert "oxine" not in name.lower(), name

    def test_thiopyran_not_renamed(self):
        name = _name("S1CC=CC=C1")
        assert "thiopyran" in name.lower() or "thiine" in name.lower(), name

    def test_saturated_arsinane_still_saturated(self):
        # Fully saturated arsinane must stay "arsinane" (no endocyclic
        # double bond → no flip).
        assert _name("[AsH]1CCCCC1") == "arsinane"

    def test_saturated_bismane_still_saturated(self):
        assert _name("[BiH]1CCCCC1") == "bismane"

    def test_saturated_gallinane_still_saturated(self):
        assert _name("[GaH]1CCCCC1") == "gallinane"


# ---------------------------------------------------------------------------
# 4. _choose_hw_stem unit tests — direct API, no engine pipeline
# ---------------------------------------------------------------------------

class TestChooseHWStemGroupCUnsaturated:
    """Direct call to the stem-picker helper."""

    def setup_method(self):
        from iupac_namer.data_loader import get_hw_tables
        tables = get_hw_tables()
        self.stem_data_6 = tables["stems"]["6"]
        self.six_groups = tables["six_membered_groups"]

    def test_group_c_aromatic_true_picks_unsaturated(self):
        from iupac_namer.ring_naming.monocyclic import _choose_hw_stem
        # Sanity: aromatic=True always picks unsaturated, even w/o endo dbl
        assert _choose_hw_stem(
            6, self.stem_data_6, self.six_groups, ["As"],
            aromatic=True, has_endocyclic_double_bond=False,
        ) == "inine"

    def test_group_c_aromatic_false_endo_dbl_picks_unsaturated(self):
        from iupac_namer.ring_naming.monocyclic import _choose_hw_stem
        # The BUG FIX: aromatic=False but has_endocyclic_double_bond=True
        assert _choose_hw_stem(
            6, self.stem_data_6, self.six_groups, ["As"],
            aromatic=False, has_endocyclic_double_bond=True,
        ) == "inine"

    def test_group_c_no_endo_dbl_saturated(self):
        from iupac_namer.ring_naming.monocyclic import _choose_hw_stem
        # Fully saturated -> "inane"
        assert _choose_hw_stem(
            6, self.stem_data_6, self.six_groups, ["As"],
            aromatic=False, has_endocyclic_double_bond=False,
        ) == "inane"

    def test_group_b_endo_dbl_picks_unsaturated(self):
        from iupac_namer.ring_naming.monocyclic import _choose_hw_stem
        # Group B (Ge/Sn/Pb): also heavy, also not kekulizable by RDKit.
        # "ine" stem (not "inine") because the ring-size stem "ine" +
        # prefix elision yields "germine"/"stannine"/"plumbine".
        assert _choose_hw_stem(
            6, self.stem_data_6, self.six_groups, ["Ge"],
            aromatic=False, has_endocyclic_double_bond=True,
        ) == "ine"

    def test_group_a_bismuth_endo_dbl_picks_unsaturated(self):
        from iupac_namer.ring_naming.monocyclic import _choose_hw_stem
        # Bi is listed as group A (unsaturated "ine", saturated "ane").
        assert _choose_hw_stem(
            6, self.stem_data_6, self.six_groups, ["Bi"],
            aromatic=False, has_endocyclic_double_bond=True,
        ) == "ine"

    def test_light_group_B_endo_dbl_stays_saturated(self):
        from iupac_namer.ring_naming.monocyclic import _choose_hw_stem
        # N is light (RDKit aromatizes): non-aromatic partially
        # unsaturated 6-ring must keep the SATURATED stem (OPSIN
        # rejects HW "azine" for non-aromatic rings).
        assert _choose_hw_stem(
            6, self.stem_data_6, self.six_groups, ["N"],
            aromatic=False, has_endocyclic_double_bond=True,
        ) == "inane"
