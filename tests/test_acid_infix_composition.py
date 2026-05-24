"""Tests for the Stage 6 R1-F acid-infix composition dispatcher.

Covers:

* The ``InfixRule`` table loads from ``data/infix_rules.json`` and contains
  all 28 OPSIN rules.
* ``detect_acid_infix_composition`` emits the expected name for the 7
  previously-MISSING infixes and the 3 PARTIAL ones.
* Existing natively-covered infixes (thi, amid, hydrazid, chlorid, …) still
  round-trip correctly and the detector does NOT fire on them.
* Non-acid molecules (ethanol, acetonitrile, benzene) pass through unchanged.

These tests do NOT require OPSIN — they assert on the assembled name string
directly, so they run in any dev environment.
"""

from __future__ import annotations

import pytest

from iupac_namer.engine import name_smiles
from iupac_namer.perception.fg.acid_infix_composition import (
    infix_coverage_matrix,
    load_infix_rules,
)


# ---------------------------------------------------------------------------
# Rule table tests
# ---------------------------------------------------------------------------


class TestInfixRuleTable:
    """The JSON table is the single source of truth for coverage — these
    tests pin its shape so future refactors cannot silently drop a rule."""

    def test_all_28_opsin_rules_present(self):
        rules = load_infix_rules()
        aliases = {r.alias for r in rules}
        expected = {
            "amid", "azid", "bromid", "chlorid", "cyanatid", "cyanid",
            "dithioperox", "diselenoperox", "ditelluroperox", "fluorid",
            "hydrazid", "hydrazon", "imid", "iodid", "isocyanatid",
            "isocyanid", "isothiocyanatid", "isoselenocyanatid",
            "isotellurocyanatid", "nitrid", "perox", "selen", "tellur",
            "thi", "thiocyanatid", "selenocyanatid", "tellurocyanatid",
            "hydroxim",
        }
        assert aliases == expected, (
            f"Missing: {expected - aliases}, Extra: {aliases - expected}"
        )

    def test_native_coverage_baseline(self):
        """Exactly these OPSIN infixes are already covered by the native
        plan search (``functional_groups.json`` entries)."""
        matrix = infix_coverage_matrix()
        native = {alias for alias, status in matrix.items() if status == "native"}
        expected_native = {
            "amid", "bromid", "chlorid", "fluorid", "iodid",
            "hydrazid", "isothiocyanatid", "thi",
        }
        assert native == expected_native

    def test_composed_coverage(self):
        matrix = infix_coverage_matrix()
        composed = {alias for alias, status in matrix.items() if status == "composed"}
        expected_composed = {
            "nitrid", "tellur", "hydroxim", "ditelluroperox",
            "isocyanid", "isotellurocyanatid", "tellurocyanatid",
            "isoselenocyanatid", "selenocyanatid", "azid",
            "cyanatid", "cyanid", "isocyanatid", "thiocyanatid",
        }
        assert composed == expected_composed


# ---------------------------------------------------------------------------
# End-to-end naming tests
# ---------------------------------------------------------------------------


class TestClassWordInfixes:
    """Verify ``{acyl} {class-word}`` composition."""

    @pytest.mark.parametrize(
        "smiles,expected",
        [
            ("CC(=O)[N+]#[C-]",             "acetyl isocyanide"),
            ("CC(=O)N=C=[Te]",              "acetyl isotellurocyanate"),
            ("CC(=O)[Te]C#N",               "acetyl tellurocyanate"),
            ("CC(=O)N=[N+]=[N-]",           "acetyl azide"),
            ("CC(=O)[Se]C#N",               "acetyl selenocyanate"),
            ("CC(=O)N=C=[Se]",              "acetyl isoselenocyanate"),
            ("c1ccc(C(=O)[N+]#[C-])cc1",    "benzoyl isocyanide"),
            ("c1ccc(C(=O)N=C=[Te])cc1",     "benzoyl isotellurocyanate"),
            ("c1ccc(C(=O)N=[N+]=[N-])cc1",  "benzoyl azide"),
        ],
    )
    def test_class_word_composition(self, smiles, expected):
        assert name_smiles(smiles) == expected


class TestEmbeddedStemInfixes:
    """Verify ``{stem}{infix}ic acid`` composition."""

    @pytest.mark.parametrize(
        "smiles,expected",
        [
            ("N#P(O)O",         "phosphoronitridic acid"),
            ("CC(O)=[Te]",      "ethanotelluroic acid"),
            ("CCC(O)=[Te]",     "propanotelluroic acid"),
            ("CC(=NO)O",        "ethanohydroximic acid"),
        ],
    )
    def test_embedded_stem_composition(self, smiles, expected):
        assert name_smiles(smiles) == expected


class TestDoesNotRegressExistingBehavior:
    """The detector must not fire on molecules the native plan search
    already handles correctly."""

    @pytest.mark.parametrize(
        "smiles,expected",
        [
            ("CC(=O)O",     "acetic acid"),
            ("CCO",         "ethanol"),
            ("CC#N",        "acetonitrile"),
            ("c1ccccc1C#N", "benzonitrile"),
            # Retained PIN per P-66.6.1 (acetyl chloride > ethanoyl chloride).
            ("CC(=O)Cl",    "acetyl chloride"),
            ("CC(=O)N",     "acetamide"),
            ("CC(=O)NN",    "ethanohydrazide"),
            ("CC(=O)S",     "ethanethioic S-acid"),
            # P-66.1: retained acetamide PIN propagates under N-substitution
            # (acet- stem retained even when the amide nitrogen is substituted).
            ("CC(=O)NC",    "N-methylacetamide"),
        ],
    )
    def test_native_names_preserved(self, smiles, expected):
        assert name_smiles(smiles) == expected


# ---------------------------------------------------------------------------
# Acid-to-acyl transformation unit tests
# ---------------------------------------------------------------------------


class TestAcidToAcyl:
    """Unit tests for the ``_acid_to_acyl`` helper."""

    @pytest.mark.parametrize(
        "acid,acyl",
        [
            ("acetic acid",       "acetyl"),
            ("benzoic acid",      "benzoyl"),
            ("formic acid",       "formyl"),
            ("propionic acid",    "propionyl"),
            ("butyric acid",      "butyryl"),
            ("oxalic acid",       "oxalyl"),
            ("ethanoic acid",     "ethanoyl"),
            ("propanoic acid",    "propanoyl"),
            ("2-chloroacetic acid", "2-chloroacetyl"),
            ("benzenecarboxylic acid", "benzenecarbonyl"),
        ],
    )
    def test_known_acids(self, acid, acyl):
        from iupac_namer.perception.fg.acid_infix_composition import _acid_to_acyl
        assert _acid_to_acyl(acid) == acyl

    def test_non_acid_input_returns_none(self):
        from iupac_namer.perception.fg.acid_infix_composition import _acid_to_acyl
        assert _acid_to_acyl("ethanol") is None
        assert _acid_to_acyl("") is None
