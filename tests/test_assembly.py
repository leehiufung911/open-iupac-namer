"""
tests/test_assembly.py

Tests for the Assembly layer (iupac_namer/assembly.py).

Covers:
  - LeafTree and ErrorTree dispatch
  - derive_sort_name
  - merge_identical_prefixes
  - render_unsaturation
  - resolve_suffix_variant / SUFFIX_VARIANT_TABLE
  - render_suffixes
  - Substitutive assembly (simple "ethanol" case)
  - Additive assembly
  - Elision
"""
import pytest

from iupac_namer.assembly import (
    SUFFIX_VARIANT_TABLE,
    assemble,
    derive_sort_name,
    elide,
    merge_identical_prefixes,
    render_merged_prefixes,
    render_suffixes,
    render_unsaturation,
    resolve_suffix_variant,
)
from iupac_namer.types import (
    AdditiveGroup,
    AdditiveTree,
    CandidateParent,
    Choice,
    DetectedFG,
    ErrorTree,
    FreeValenceInfo,
    LeafTree,
    Locant,
    MergedPrefix,
    NamedParent,
    Numbering,
    OutputForm,
    PrefixEntry,
    SubstituentMethod,
    SubstitutiveTree,
    SuffixGroup,
    UnsaturationInfix,
)


# ---------------------------------------------------------------------------
# Helpers: minimal frozen object factories
# ---------------------------------------------------------------------------

def _empty_choices() -> tuple:
    return ()


def _leaf(text: str, output_form: OutputForm = OutputForm.STANDALONE) -> LeafTree:
    return LeafTree(
        output_form=output_form,
        free_valence=None,
        choices_made=(),
        decision_ctx=None,
        validity_warnings=None,
        text=text,
    )


def _make_candidate_parent() -> CandidateParent:
    return CandidateParent(
        atom_indices=frozenset({0, 1}),
        type="chain",
        length=2,
        ring_system=None,
        unsaturation=None,
        element=None,
        lambda_value=None,
    )


def _make_named_parent(
    stem: str,
    alkyl_stem: str | None = None,
    name: str | None = None,
    candidate_length: int | None = None,
) -> NamedParent:
    candidate = _make_candidate_parent()
    if candidate_length is not None and candidate_length != candidate.length:
        import dataclasses as _dc
        atom_indices = frozenset(range(candidate_length))
        candidate = _dc.replace(
            candidate, atom_indices=atom_indices, length=candidate_length
        )
    return NamedParent(
        candidate=candidate,
        name=name or (stem + "e"),
        stem=stem,
        alkyl_stem=alkyl_stem,
        naming_method="systematic",
        indicated_hydrogen=None,
        numbering_options=(),
    )


def _make_numbering(*pairs: tuple[int, Locant]) -> Numbering:
    assignments = tuple((idx, loc) for idx, loc in pairs)
    locant_set = tuple(sorted(loc for _, loc in assignments))
    return Numbering(_assignments=assignments, locant_set=locant_set)


def _make_detected_fg(fg_type: str = "alcohol") -> DetectedFG:
    return DetectedFG(
        type=fg_type,
        atoms=frozenset({1}),
        anchor=1,
        properties=(),
        suffix_eligible=True,
        suffix_forms=(("terminal", "-ol"),),
        prefix_form="hydroxy",
    )


def _make_suffix_group(
    base_form: str,
    locants: tuple[Locant, ...] = (),
    elides: bool = True,
) -> SuffixGroup:
    return SuffixGroup(
        fg=_make_detected_fg(),
        locants=locants,
        base_form=base_form,
        elides_terminal_e=elides,
    )


# ---------------------------------------------------------------------------
# 1. LeafTree dispatch
# ---------------------------------------------------------------------------

class TestLeafTree:
    def test_assemble_leaf_standalone(self):
        tree = _leaf("methane")
        assert assemble(tree) == "methane"

    def test_assemble_leaf_substituent(self):
        tree = _leaf("methyl", OutputForm.SUBSTITUENT)
        assert assemble(tree) == "methyl"

    def test_assemble_leaf_empty(self):
        tree = _leaf("")
        assert assemble(tree) == ""


# ---------------------------------------------------------------------------
# 2. ErrorTree dispatch
# ---------------------------------------------------------------------------

class TestErrorTree:
    def test_assemble_error_tree(self):
        tree = ErrorTree(
            output_form=OutputForm.STANDALONE,
            free_valence=None,
            choices_made=(),
            decision_ctx=None,
            validity_warnings=None,
            message="failed",
        )
        assert assemble(tree) == "[NAMING ERROR: failed]"

    def test_assemble_error_tree_long_message(self):
        msg = "unknown functional group xyz at atom 5"
        tree = ErrorTree(
            output_form=OutputForm.STANDALONE,
            free_valence=None,
            choices_made=(),
            decision_ctx=None,
            validity_warnings=None,
            message=msg,
        )
        assert assemble(tree) == f"[NAMING ERROR: {msg}]"


# ---------------------------------------------------------------------------
# 3. derive_sort_name
# ---------------------------------------------------------------------------

class TestDeriveSortName:
    def test_simple_name(self):
        assert derive_sort_name("methyl") == "methyl"

    def test_strip_locant(self):
        # "2-chloroethyl" -> strip "2-" -> "chloroethyl"
        assert derive_sort_name("2-chloroethyl") == "chloroethyl"

    def test_strip_brackets_and_locant(self):
        # "(2-chloroethyl)" -> strip brackets -> "2-chloroethyl" -> strip locant
        assert derive_sort_name("(2-chloroethyl)") == "chloroethyl"

    def test_strip_multiplier_di(self):
        # "dimethyl" -> strip "di" -> "methyl"
        assert derive_sort_name("dimethyl") == "methyl"

    def test_strip_multiplier_bis_brackets(self):
        # "bis(2-chloroethyl)" -> strip "bis" -> "(2-chloroethyl)"
        # -> strip brackets -> "2-chloroethyl" -> strip locant -> "chloroethyl"
        assert derive_sort_name("bis(2-chloroethyl)") == "chloroethyl"

    def test_strip_multiplier_tri(self):
        assert derive_sort_name("trimethyl") == "methyl"

    def test_lowercase(self):
        # Should lowercase everything
        assert derive_sort_name("Methyl") == "methyl"

    def test_no_locant_no_multiplier(self):
        assert derive_sort_name("ethyl") == "ethyl"

    def test_tris_brackets(self):
        # "tris(2-methylpropyl)" -> strip "tris" -> "(2-methylpropyl)"
        # -> strip brackets -> "2-methylpropyl" -> strip locant -> "methylpropyl"
        assert derive_sort_name("tris(2-methylpropyl)") == "methylpropyl"

    def test_square_brackets(self):
        # "[2-chloroethyl]" -> strip [] -> "2-chloroethyl" -> strip locant
        assert derive_sort_name("[2-chloroethyl]") == "chloroethyl"


# ---------------------------------------------------------------------------
# 4. merge_identical_prefixes
# ---------------------------------------------------------------------------

class TestMergeIdenticalPrefixes:
    def test_two_methyl_entries_gives_dimethyl(self):
        loc1 = Locant.numeric(2)
        loc2 = Locant.numeric(4)
        entries = [("methyl", (loc1,)), ("methyl", (loc2,))]
        result = merge_identical_prefixes(entries)
        assert len(result) == 1
        mp = result[0]
        assert mp.name == "methyl"
        assert mp.multiplier == "di"
        assert not mp.needs_brackets
        assert set(mp.locants) == {loc1, loc2}

    def test_single_ethyl_no_brackets_no_multiplier(self):
        loc = Locant.numeric(3)
        entries = [("ethyl", (loc,))]
        result = merge_identical_prefixes(entries)
        assert len(result) == 1
        mp = result[0]
        assert mp.name == "ethyl"
        assert mp.multiplier is None
        assert not mp.needs_brackets

    def test_single_compound_prefix_gets_brackets(self):
        loc = Locant.numeric(2)
        entries = [("2-chloroethyl", (loc,))]
        result = merge_identical_prefixes(entries)
        assert len(result) == 1
        mp = result[0]
        assert mp.needs_brackets is True
        assert mp.multiplier is None

    def test_two_compound_prefixes_get_bis(self):
        loc1 = Locant.numeric(1)
        loc2 = Locant.numeric(3)
        entries = [
            ("2-chloroethyl", (loc1,)),
            ("2-chloroethyl", (loc2,)),
        ]
        result = merge_identical_prefixes(entries)
        assert len(result) == 1
        mp = result[0]
        assert mp.needs_brackets is True
        assert mp.multiplier == "bis"

    def test_different_names_stay_separate(self):
        entries = [
            ("methyl", (Locant.numeric(2),)),
            ("ethyl", (Locant.numeric(4),)),
        ]
        result = merge_identical_prefixes(entries)
        assert len(result) == 2
        names = {mp.name for mp in result}
        assert names == {"methyl", "ethyl"}

    def test_sort_name_derived(self):
        entries = [("methyl", (Locant.numeric(2),))]
        result = merge_identical_prefixes(entries)
        assert result[0].sort_name == "methyl"


# ---------------------------------------------------------------------------
# 5. render_unsaturation
# ---------------------------------------------------------------------------

class TestRenderUnsaturation:
    def test_single_en_at_2(self):
        inf = UnsaturationInfix(type="en", locants=(Locant.numeric(2),), multiplier=None)
        assert render_unsaturation((inf,)) == "-2-ene"

    def test_double_en_at_2_4(self):
        inf = UnsaturationInfix(
            type="en",
            locants=(Locant.numeric(2), Locant.numeric(4)),
            multiplier="di",
        )
        assert render_unsaturation((inf,)) == "-2,4-diene"

    def test_triple_bond_at_1(self):
        inf = UnsaturationInfix(type="yn", locants=(Locant.numeric(1),), multiplier=None)
        assert render_unsaturation((inf,)) == "-1-yne"

    def test_en_and_yn_combined(self):
        # P-31.1.2.1 / P-54.1: the trailing 'e' of "-ene" is elided when
        # immediately followed by another locant-bearing unsaturation
        # ("-N-yne") so the rendered form reads "-2-en-4-yne", not the
        # earlier non-elided "-2-ene-4-yne".  Mirrors the spec PIN
        # but-1-en-3-yne for HC#C-CH=CH2.
        inf_en = UnsaturationInfix(type="en", locants=(Locant.numeric(2),), multiplier=None)
        inf_yn = UnsaturationInfix(type="yn", locants=(Locant.numeric(4),), multiplier=None)
        result = render_unsaturation((inf_en, inf_yn))
        assert result == "-2-en-4-yne"

    def test_empty(self):
        assert render_unsaturation(()) == ""

    def test_no_locant(self):
        # No-locant infix (e.g. ethene/ethyne after locant omission):
        # render without leading hyphen so elide() can handle stem+infix junction.
        inf = UnsaturationInfix(type="en", locants=(), multiplier=None)
        assert render_unsaturation((inf,)) == "ene"

    def test_no_locant_yn(self):
        # Same for triple bond (ethyne)
        inf = UnsaturationInfix(type="yn", locants=(), multiplier=None)
        assert render_unsaturation((inf,)) == "yne"


# ---------------------------------------------------------------------------
# 5b. _strip_unsaturation_locants_if_omissible
# ---------------------------------------------------------------------------

class TestStripUnsaturationLocants:
    """Tests for the 2-carbon chain locant omission rule."""

    def _infix(self, type_: str, locant: int, mult: str | None = None) -> UnsaturationInfix:
        return UnsaturationInfix(
            type=type_, locants=(Locant.numeric(locant),), multiplier=mult
        )

    def test_2carbon_single_en_locant_stripped(self):
        from iupac_namer.assembly import _strip_unsaturation_locants_if_omissible
        inf = self._infix("en", 1)
        result = _strip_unsaturation_locants_if_omissible((inf,), parent_length=2)
        assert len(result) == 1
        assert result[0].locants == ()

    def test_2carbon_single_yn_locant_stripped(self):
        from iupac_namer.assembly import _strip_unsaturation_locants_if_omissible
        inf = self._infix("yn", 1)
        result = _strip_unsaturation_locants_if_omissible((inf,), parent_length=2)
        assert len(result) == 1
        assert result[0].locants == ()

    def test_4carbon_single_en_locant_retained(self):
        from iupac_namer.assembly import _strip_unsaturation_locants_if_omissible
        inf = self._infix("en", 2)
        result = _strip_unsaturation_locants_if_omissible((inf,), parent_length=4)
        assert result[0].locants == (Locant.numeric(2),)

    def test_longer_chain_no_stripping(self):
        from iupac_namer.assembly import _strip_unsaturation_locants_if_omissible
        inf = self._infix("en", 1)
        result = _strip_unsaturation_locants_if_omissible((inf,), parent_length=3)
        assert result[0].locants == (Locant.numeric(1),)

    def test_multiple_infixes_not_stripped(self):
        from iupac_namer.assembly import _strip_unsaturation_locants_if_omissible
        inf1 = self._infix("en", 1)
        inf2 = self._infix("en", 1)
        result = _strip_unsaturation_locants_if_omissible((inf1, inf2), parent_length=2)
        assert result[0].locants == (Locant.numeric(1),)


# ---------------------------------------------------------------------------
# 5c. _saturated_chain_full_substitution_signature (P-14.3.4.4 saturation)
# ---------------------------------------------------------------------------

class TestSaturatedChainFullSubstitutionSignature:
    """The locant-multiset signature for a fully substituted saturated acid
    chain (P-14.3.4.4 complete-saturation locant omission)."""

    def test_butanoic_full_signature(self):
        # C1=acid; C2,C3 methylene (×2 each); C4 methyl (×3) → 2,2,3,3,4,4,4
        from iupac_namer.assembly import (
            _saturated_chain_full_substitution_signature as sig,
        )
        assert sig(4) == ("2", "2", "3", "3", "4", "4", "4")

    def test_propanoic_full_signature(self):
        # C1=acid; C2 methylene (×2); C3 methyl (×3) → 2,2,3,3,3
        from iupac_namer.assembly import (
            _saturated_chain_full_substitution_signature as sig,
        )
        assert sig(3) == ("2", "2", "3", "3", "3")

    def test_acetic_full_signature(self):
        # C1=acid; C2 methyl (×3) → 2,2,2
        from iupac_namer.assembly import (
            _saturated_chain_full_substitution_signature as sig,
        )
        assert sig(2) == ("2", "2", "2")

    def test_length_one_empty(self):
        # No off-C1 positions on a single-carbon parent.
        from iupac_namer.assembly import (
            _saturated_chain_full_substitution_signature as sig,
        )
        assert sig(1) == ()
        assert sig(0) == ()


# ---------------------------------------------------------------------------
# 5d. Complete-saturation locant omission, end-to-end (P-14.3.4.4)
# ---------------------------------------------------------------------------

class TestCompleteSaturationLocantOmission:
    """Perhalogenated acid chains: when a single substituent type fills every
    off-C1 position the locants are forced and omitted from the PIN."""

    def test_heptafluorobutanoic_acid(self):
        from iupac_namer.engine import name_smiles
        assert name_smiles("O=C(O)C(F)(F)C(F)(F)C(F)(F)F") == (
            "heptafluorobutanoic acid"
        )

    def test_pentafluoropropanoic_acid(self):
        from iupac_namer.engine import name_smiles
        assert name_smiles("OC(=O)C(F)(F)C(F)(F)F") == (
            "pentafluoropropanoic acid"
        )

    def test_heptachlorobutanoic_acid(self):
        from iupac_namer.engine import name_smiles
        assert name_smiles("OC(=O)C(Cl)(Cl)C(Cl)(Cl)C(Cl)(Cl)Cl") == (
            "heptachlorobutanoic acid"
        )

    def test_heptabromobutanoic_acid(self):
        from iupac_namer.engine import name_smiles
        assert name_smiles("OC(=O)C(Br)(Br)C(Br)(Br)C(Br)(Br)Br") == (
            "heptabromobutanoic acid"
        )

    # --- Negative controls: partial / mixed substitution KEEPS locants ---

    def test_partial_c2_unsubstituted_keeps_locants(self):
        # C2 is a bare CH2 → not full saturation; the 3,3,3 locants stay.
        from iupac_namer.engine import name_smiles
        assert name_smiles("OC(=O)CC(F)(F)F") == "3,3,3-trifluoropropanoic acid"

    def test_partial_c3_unsubstituted_keeps_locants(self):
        from iupac_namer.engine import name_smiles
        assert name_smiles("OC(=O)C(F)(F)CC(F)(F)F") == (
            "2,2,4,4,4-pentafluorobutanoic acid"
        )

    def test_mixed_substituents_keep_locants(self):
        # Two distinct prefix types → neither alone matches the full
        # signature, so all locants are retained.
        from iupac_namer.engine import name_smiles
        assert name_smiles("OC(=O)C(F)(F)C(Cl)(Cl)C(F)(F)F") == (
            "3,3-dichloro-2,2,4,4,4-pentafluorobutanoic acid"
        )


# ---------------------------------------------------------------------------
# 6. SUFFIX_VARIANT_TABLE / resolve_suffix_variant
# ---------------------------------------------------------------------------

class TestSuffixVariantTable:
    def test_oic_acid_standalone(self):
        assert resolve_suffix_variant("oic acid", OutputForm.STANDALONE) == "oic acid"

    def test_oic_acid_acid_stem(self):
        # -oic acid → -oate, preserving the 'o' so "propanoic acid" → "propanoate"
        # (the stem is "propan", so "propan"+"oate" = "propanoate").
        assert resolve_suffix_variant("oic acid", OutputForm.ACID_STEM) == "oate"

    def test_oic_acid_acyl(self):
        assert resolve_suffix_variant("oic acid", OutputForm.ACYL) == "oyl"

    def test_oic_acid_anion(self):
        assert resolve_suffix_variant("oic acid", OutputForm.ANION) == "oate"

    def test_ol_standalone(self):
        assert resolve_suffix_variant("ol", OutputForm.STANDALONE) == "ol"

    def test_ol_anion(self):
        assert resolve_suffix_variant("ol", OutputForm.ANION) == "olate"

    def test_carboxylic_acid_standalone(self):
        assert resolve_suffix_variant("carboxylic acid", OutputForm.STANDALONE) == "carboxylic acid"

    def test_carboxylic_acid_anion(self):
        assert resolve_suffix_variant("carboxylic acid", OutputForm.ANION) == "carboxylate"

    def test_amine_cation(self):
        assert resolve_suffix_variant("amine", OutputForm.CATION) == "aminium"

    def test_unknown_falls_back_to_base(self):
        # Unknown base_form returns base_form unchanged
        assert resolve_suffix_variant("unknown_suffix", OutputForm.STANDALONE) == "unknown_suffix"

    def test_all_table_keys_are_tuples(self):
        for key in SUFFIX_VARIANT_TABLE:
            assert isinstance(key, tuple) and len(key) == 2

    def test_all_table_values_are_strings(self):
        for val in SUFFIX_VARIANT_TABLE.values():
            assert isinstance(val, str)


# ---------------------------------------------------------------------------
# 7. Substitutive assembly — simple "ethanol" case
# ---------------------------------------------------------------------------

class TestSubstitutiveAssembly:
    def _make_ethanol_tree(self) -> SubstitutiveTree:
        """Build a minimal SubstitutiveTree representing ethanol.

        Named parent: "ethan" (stem), no alkyl_stem needed.
        Suffix: "ol" with no locant (engine omits locant 1 for terminal positions).
        No prefixes, no unsaturation, no stereo.
        """
        named_parent = _make_named_parent(stem="ethan", alkyl_stem="eth", name="ethane")
        loc1 = Locant.numeric(1)
        numbering = _make_numbering((0, loc1), (1, Locant.numeric(2)))
        # Locant 1 is omitted for terminal FGs — engine does not set it
        suffix_group = _make_suffix_group("ol", locants=(), elides=True)

        return SubstitutiveTree(
            output_form=OutputForm.STANDALONE,
            free_valence=None,
            choices_made=(),
            decision_ctx=None,
            validity_warnings=None,
            named_parent=named_parent,
            numbering=numbering,
            suffix_groups=(suffix_group,),
            unsaturation=(),
            prefixes=(),
            stereo_descriptors=None,
            indicated_hydrogen=None,
        )

    def test_ethanol_assembled(self):
        tree = self._make_ethanol_tree()
        result = assemble(tree)
        assert result == "ethanol"

    def test_ethanol_stem_is_ethan(self):
        # Verify the stem "ethan" + "-ol" after elision = "ethanol"
        # Elision: "ethan" ends in 'n' (not 'e'), so no elision needed here.
        # Actually ethan + ol = ethanol (no 'e' at end of ethan to elide)
        tree = self._make_ethanol_tree()
        result = assemble(tree)
        assert result == "ethanol"

    def test_propane_no_suffix(self):
        """Parent with no suffix should end in 'e' (terminal_vowel)."""
        named_parent = _make_named_parent(stem="propan", name="propane")
        numbering = _make_numbering(
            (0, Locant.numeric(1)),
            (1, Locant.numeric(2)),
            (2, Locant.numeric(3)),
        )
        tree = SubstitutiveTree(
            output_form=OutputForm.STANDALONE,
            free_valence=None,
            choices_made=(),
            decision_ctx=None,
            validity_warnings=None,
            named_parent=named_parent,
            numbering=numbering,
            suffix_groups=(),
            unsaturation=(),
            prefixes=(),
            stereo_descriptors=None,
            indicated_hydrogen=None,
        )
        result = assemble(tree)
        assert result == "propane"

    def test_butanoic_acid(self):
        """butan + oic acid -> butanoic acid (no elision because 'n' before 'o').

        Locant 1 is omitted for terminal FGs — engine does not set it on the suffix.
        """
        named_parent = _make_named_parent(stem="butan", name="butane")
        loc1 = Locant.numeric(1)
        numbering = _make_numbering((0, loc1))
        # No locant on terminal acid — engine omits locant 1
        suffix_group = _make_suffix_group("oic acid", locants=(), elides=False)

        tree = SubstitutiveTree(
            output_form=OutputForm.STANDALONE,
            free_valence=None,
            choices_made=(),
            decision_ctx=None,
            validity_warnings=None,
            named_parent=named_parent,
            numbering=numbering,
            suffix_groups=(suffix_group,),
            unsaturation=(),
            prefixes=(),
            stereo_descriptors=None,
            indicated_hydrogen=None,
        )
        result = assemble(tree)
        assert result == "butanoic acid"

    def test_with_methyl_prefix(self):
        """2-methylpropane: propan + 2-methyl prefix."""
        named_parent = _make_named_parent(stem="propan", name="propane")
        numbering = _make_numbering(
            (0, Locant.numeric(1)),
            (1, Locant.numeric(2)),
            (2, Locant.numeric(3)),
        )
        methyl_leaf = _leaf("methyl", OutputForm.SUBSTITUENT)
        loc2 = Locant.numeric(2)
        prefix_entry = PrefixEntry(tree=methyl_leaf, locants=(loc2,))

        tree = SubstitutiveTree(
            output_form=OutputForm.STANDALONE,
            free_valence=None,
            choices_made=(),
            decision_ctx=None,
            validity_warnings=None,
            named_parent=named_parent,
            numbering=numbering,
            suffix_groups=(),
            unsaturation=(),
            prefixes=(prefix_entry,),
            stereo_descriptors=None,
            indicated_hydrogen=None,
        )
        result = assemble(tree)
        assert result == "2-methylpropane"

    def test_with_unsaturation(self):
        """but-2-ene: stem "but" (the engine sets stem to reflect dropped -an for alkenes)
        + -2-en + terminal e = "but-2-ene".

        In IUPAC 2013 nomenclature, saturated parent butane has stem="butan",
        but but-2-ene itself has the stem set to "but" (the "-an" is dropped when
        attaching the "-en" unsaturation infix). This is handled by the engine when
        it builds NamedParent — the stem reflects the actual string root used.
        """
        # For but-2-ene the correct stem is "but" (not "butan")
        # Use candidate_length=4 so the 2-carbon omission rule does not apply.
        named_parent = _make_named_parent(stem="but", name="but-2-ene", candidate_length=4)
        numbering = _make_numbering(
            (0, Locant.numeric(1)),
            (1, Locant.numeric(2)),
            (2, Locant.numeric(3)),
            (3, Locant.numeric(4)),
        )
        infix = UnsaturationInfix(type="en", locants=(Locant.numeric(2),), multiplier=None)

        tree = SubstitutiveTree(
            output_form=OutputForm.STANDALONE,
            free_valence=None,
            choices_made=(),
            decision_ctx=None,
            validity_warnings=None,
            named_parent=named_parent,
            numbering=numbering,
            suffix_groups=(),
            unsaturation=(infix,),
            prefixes=(),
            stereo_descriptors=None,
            indicated_hydrogen=None,
        )
        result = assemble(tree)
        assert result == "but-2-ene"


# ---------------------------------------------------------------------------
# 8. Additive assembly
# ---------------------------------------------------------------------------

class TestAdditiveAssembly:
    def test_pyridine_1_oxide(self):
        """pyridine 1-oxide: numeric locant -> "1-oxide"."""
        parent_leaf = _leaf("pyridine")
        loc = Locant.numeric(1)
        ag = AdditiveGroup(type="oxide", locant=loc, multiplier=None)

        tree = AdditiveTree(
            output_form=OutputForm.STANDALONE,
            free_valence=None,
            choices_made=(),
            decision_ctx=None,
            validity_warnings=None,
            parent_tree=parent_leaf,
            additions=(ag,),
        )
        result = assemble(tree)
        assert result == "pyridine 1-oxide"

    def test_trimethylamine_oxide(self):
        """trimethylamine oxide: heteroatom (N) locant -> no locant prefix."""
        parent_leaf = _leaf("trimethylamine")
        # Non-numeric locant N -> no locant in output
        loc = Locant.hetero("N")
        ag = AdditiveGroup(type="oxide", locant=loc, multiplier=None)

        tree = AdditiveTree(
            output_form=OutputForm.STANDALONE,
            free_valence=None,
            choices_made=(),
            decision_ctx=None,
            validity_warnings=None,
            parent_tree=parent_leaf,
            additions=(ag,),
        )
        result = assemble(tree)
        assert result == "trimethylamine oxide"

    def test_dioxide(self):
        """phosphane 1,1-dioxide: numeric locant + di multiplier."""
        parent_leaf = _leaf("phosphane")
        loc = Locant.numeric(1)
        ag = AdditiveGroup(type="oxide", locant=loc, multiplier="di")

        tree = AdditiveTree(
            output_form=OutputForm.STANDALONE,
            free_valence=None,
            choices_made=(),
            decision_ctx=None,
            validity_warnings=None,
            parent_tree=parent_leaf,
            additions=(ag,),
        )
        result = assemble(tree)
        assert result == "phosphane 1-dioxide"


# ---------------------------------------------------------------------------
# 9. Elision
# ---------------------------------------------------------------------------

class TestElision:
    def test_ethan_ol_gives_ethanol(self):
        """'ethan' + 'ol' -> 'ethanol'. No 'e' at end of 'ethan', no elision needed."""
        # Actually 'ethan' does not end in 'e', so elision is not triggered.
        # The suffix '-ol' starts with 'o' (vowel), but 'n' + 'o' = 'no', no elision.
        result = elide("ethan" + "ol")
        assert result == "ethanol"

    def test_methane_ol_gives_methanol(self):
        """'methane' + 'ol' would elide the 'e': 'methan' + 'ol' = 'methanol'."""
        # When assembly joins "methane" and "ol", the 'e' before 'o' is elided.
        result = elide("methaneol")
        assert result == "methanol"

    def test_ethane_one_gives_ethanone(self):
        result = elide("ethanone")
        # 'e' before 'o' in 'one' — the 'e' at end of 'ethan' + 'one':
        # methane + one = methanone? Let's check: "ethanone" — 'e' at position 5
        # is followed by 'n', not a vowel, so no elision. Result unchanged.
        assert result == "ethanone"

    def test_propane_al_gives_propanal(self):
        """'propane' + 'al' -> 'propanal': 'e' before 'a' elided."""
        result = elide("propaneal")
        assert result == "propanal"

    def test_butane_amine_no_elision(self):
        """The assembly never joins stem+'e'+'amine' — the terminal 'e' is only
        emitted by terminal_vowel() when there is no suffix. When there IS an amine
        suffix, render_suffixes() emits "-amine" appended directly to the stem "butan".
        So "butan" + "amine" = "butanamine" without going through elide() for 'e'.

        This test verifies that elide() does NOT strip the 'e' from 'butane' when
        it's a genuine part of the word (not followed immediately by a vowel).
        'butaneamine' has 'e' before 'a' — elision IS applied: "butanamine".
        """
        result = elide("butaneamine")
        # 'e' at index 6 is followed by 'a' (vowel), and "amine" is in the
        # no_elision_patterns — so the 'e' is NOT elided. Result: "butaneamine".
        # But in practice this string never arises in normal assembly.
        assert result == "butaneamine"

    def test_no_elision_when_no_vowel_follows(self):
        """'butane' + 'nitrile' -> 'butanenitrile' (no elision before consonant)."""
        result = elide("butanenitrile")
        # 'e' at end of 'butane' is followed by 'n' (consonant), no elision
        assert result == "butanenitrile"

    def test_cyclopentane_ol(self):
        """'cyclopentane' + 'ol' -> 'cyclopentanol'."""
        result = elide("cyclopentaneol")
        assert result == "cyclopentanol"

    def test_empty_string(self):
        assert elide("") == ""

    def test_no_vowel_sequence(self):
        assert elide("methane") == "methane"


# ---------------------------------------------------------------------------
# 10. render_suffixes
# ---------------------------------------------------------------------------

class TestRenderSuffixes:
    def test_single_ol_at_1(self):
        sg = _make_suffix_group("ol", locants=(Locant.numeric(1),))
        result = render_suffixes((sg,), OutputForm.STANDALONE)
        assert result == "-1-ol"

    def test_diol_at_1_3(self):
        sg1 = _make_suffix_group("ol", locants=(Locant.numeric(1),))
        sg2 = _make_suffix_group("ol", locants=(Locant.numeric(3),))
        result = render_suffixes((sg1, sg2), OutputForm.STANDALONE)
        assert result == "-1,3-diol"

    def test_oic_acid_at_1(self):
        sg = _make_suffix_group("oic acid", locants=(Locant.numeric(1),))
        result = render_suffixes((sg,), OutputForm.STANDALONE)
        assert result == "-1-oic acid"

    def test_dioic_acid(self):
        sg1 = _make_suffix_group("oic acid", locants=(Locant.numeric(1),))
        sg2 = _make_suffix_group("oic acid", locants=(Locant.numeric(6),))
        result = render_suffixes((sg1, sg2), OutputForm.STANDALONE)
        assert result == "-1,6-dioic acid"

    def test_output_form_anion(self):
        sg = _make_suffix_group("oic acid", locants=(Locant.numeric(1),))
        result = render_suffixes((sg,), OutputForm.ANION)
        assert result == "-1-oate"

    def test_no_locant_suffix(self):
        """No locants → suffix attaches directly without hyphen.
        e.g. "ethan" + render_suffixes(("ol",no locants)) = "ethan" + "ol" = "ethanol"
        """
        sg = _make_suffix_group("ol", locants=())
        result = render_suffixes((sg,), OutputForm.STANDALONE)
        # No locants means no leading hyphen — the suffix attaches directly
        assert result == "ol"


class TestRenderedSuffixStartsWithConsonant:
    """Tests for _rendered_suffix_starts_with_consonant helper."""

    def test_triol_has_consonant(self):
        from iupac_namer.assembly import _rendered_suffix_starts_with_consonant
        assert _rendered_suffix_starts_with_consonant("-1,2,3-triol") is True

    def test_diol_has_consonant(self):
        from iupac_namer.assembly import _rendered_suffix_starts_with_consonant
        assert _rendered_suffix_starts_with_consonant("-1,2-diol") is True

    def test_dione_has_consonant(self):
        from iupac_namer.assembly import _rendered_suffix_starts_with_consonant
        assert _rendered_suffix_starts_with_consonant("-1,2-dione") is True

    def test_diamine_has_consonant(self):
        from iupac_namer.assembly import _rendered_suffix_starts_with_consonant
        assert _rendered_suffix_starts_with_consonant("-1,2-diamine") is True

    def test_ol_has_vowel(self):
        from iupac_namer.assembly import _rendered_suffix_starts_with_consonant
        assert _rendered_suffix_starts_with_consonant("-1-ol") is False

    def test_one_has_vowel(self):
        from iupac_namer.assembly import _rendered_suffix_starts_with_consonant
        assert _rendered_suffix_starts_with_consonant("-2-one") is False

    def test_al_has_vowel(self):
        from iupac_namer.assembly import _rendered_suffix_starts_with_consonant
        assert _rendered_suffix_starts_with_consonant("al") is False

    def test_no_locant_diol_has_consonant(self):
        """'diol' with no locant preamble still starts with 'd' (consonant)."""
        from iupac_namer.assembly import _rendered_suffix_starts_with_consonant
        assert _rendered_suffix_starts_with_consonant("diol") is True

    def test_empty_returns_false(self):
        from iupac_namer.assembly import _rendered_suffix_starts_with_consonant
        assert _rendered_suffix_starts_with_consonant("") is False


# ---------------------------------------------------------------------------
# 11. SaltTree dispatch
# ---------------------------------------------------------------------------

class TestSaltTree:
    def test_two_ions(self):
        from iupac_namer.types import SaltTree
        cation = _leaf("sodium")
        anion = _leaf("chloride")
        salt = SaltTree(
            output_form=OutputForm.STANDALONE,
            free_valence=None,
            choices_made=(),
            decision_ctx=None,
            validity_warnings=None,
            ion_trees=(cation, anion),
        )
        result = assemble(salt)
        assert result == "sodium chloride"


# ---------------------------------------------------------------------------
# 12. Alphabetical prefix ordering
# ---------------------------------------------------------------------------

class TestPrefixOrdering:
    def test_alphabetical_order_chloro_before_methyl(self):
        """chloro sorts before methyl alphabetically."""
        named_parent = _make_named_parent(stem="propan", name="propane")
        numbering = _make_numbering(
            (0, Locant.numeric(1)),
            (1, Locant.numeric(2)),
            (2, Locant.numeric(3)),
        )
        # methyl at 2, chloro at 1
        methyl_leaf = _leaf("methyl", OutputForm.SUBSTITUENT)
        chloro_leaf = _leaf("chloro", OutputForm.SUBSTITUENT)
        pe_methyl = PrefixEntry(tree=methyl_leaf, locants=(Locant.numeric(2),))
        pe_chloro = PrefixEntry(tree=chloro_leaf, locants=(Locant.numeric(1),))

        tree = SubstitutiveTree(
            output_form=OutputForm.STANDALONE,
            free_valence=None,
            choices_made=(),
            decision_ctx=None,
            validity_warnings=None,
            named_parent=named_parent,
            numbering=numbering,
            suffix_groups=(),
            unsaturation=(),
            prefixes=(pe_methyl, pe_chloro),  # methyl listed first, chloro second
            stereo_descriptors=None,
            indicated_hydrogen=None,
        )
        result = assemble(tree)
        # chloro (c) before methyl (m)
        assert result.index("chloro") < result.index("methyl")


# ---------------------------------------------------------------------------
# 13. Free-valence suffix rendering (Bug 1 regression)
# ---------------------------------------------------------------------------

class TestFreeValenceSuffix:
    """Regression tests for Bug 1: Method 1 (ALKYL) substituent names must not
    include a spurious hyphen between alkyl_stem and 'yl'.

    Expected: meth + yl = methyl   (not meth-yl)
    Expected: eth  + yl = ethyl    (not eth-yl)
    """

    from iupac_namer.assembly import render_free_valence_suffix  # type: ignore[attr-defined]

    def _alkyl_fv(self) -> FreeValenceInfo:
        """Monovalent ALKYL free valence (Method 1)."""
        return FreeValenceInfo(
            bond_orders=(1,),
            method=SubstituentMethod.ALKYL,
            attachment_atoms_in_fragment=(0,),
        )

    def _make_num(self, *pairs) -> Numbering:
        return _make_numbering(*pairs)

    def test_methyl_stem_plus_yl_no_hyphen(self):
        """alkyl_stem 'meth' + render_free_valence_suffix → 'yl' (no leading hyphen)."""
        from iupac_namer.assembly import render_free_valence_suffix
        fv = self._alkyl_fv()
        num = _make_numbering((0, Locant.numeric(1)))
        suffix = render_free_valence_suffix(fv, num)
        # Must be 'yl', not '-yl'
        assert suffix == "yl", f"Expected 'yl', got {suffix!r}"
        # Direct concatenation gives 'methyl'
        assert "meth" + suffix == "methyl"

    def test_alkyl_suffix_no_leading_hyphen(self):
        """Method 1 free-valence suffix must not start with '-'."""
        from iupac_namer.assembly import render_free_valence_suffix
        fv = self._alkyl_fv()
        num = _make_numbering((0, Locant.numeric(1)))
        suffix = render_free_valence_suffix(fv, num)
        assert not suffix.startswith("-"), (
            f"Method 1 suffix must not start with '-', got {suffix!r}"
        )

    def test_methyl_assembled_substituent_tree(self):
        """Full SubstitutiveTree in SUBSTITUENT form: 'meth' + ALKYL fv -> 'methyl'."""
        named_parent = _make_named_parent(stem="methan", alkyl_stem="meth", name="methane")
        numbering = _make_numbering((0, Locant.numeric(1)))
        fv = FreeValenceInfo(
            bond_orders=(1,),
            method=SubstituentMethod.ALKYL,
            attachment_atoms_in_fragment=(0,),
        )
        tree = SubstitutiveTree(
            output_form=OutputForm.SUBSTITUENT,
            free_valence=fv,
            choices_made=(),
            decision_ctx=None,
            validity_warnings=None,
            named_parent=named_parent,
            numbering=numbering,
            suffix_groups=(),
            unsaturation=(),
            prefixes=(),
            stereo_descriptors=None,
            indicated_hydrogen=None,
        )
        result = assemble(tree)
        assert result == "methyl", f"Expected 'methyl', got {result!r}"

    def test_ethyl_assembled_substituent_tree(self):
        """Full SubstitutiveTree: eth + ALKYL fv -> 'ethyl' (not 'eth-yl')."""
        named_parent = _make_named_parent(stem="ethan", alkyl_stem="eth", name="ethane")
        numbering = _make_numbering((0, Locant.numeric(1)), (1, Locant.numeric(2)))
        fv = FreeValenceInfo(
            bond_orders=(1,),
            method=SubstituentMethod.ALKYL,
            attachment_atoms_in_fragment=(0,),
        )
        tree = SubstitutiveTree(
            output_form=OutputForm.SUBSTITUENT,
            free_valence=fv,
            choices_made=(),
            decision_ctx=None,
            validity_warnings=None,
            named_parent=named_parent,
            numbering=numbering,
            suffix_groups=(),
            unsaturation=(),
            prefixes=(),
            stereo_descriptors=None,
            indicated_hydrogen=None,
        )
        result = assemble(tree)
        assert result == "ethyl", f"Expected 'ethyl', got {result!r}"

    def test_simple_substituent_prefix_no_parentheses(self):
        """Bug 2 regression: simple substituent prefix (e.g. methyl) must not be
        wrapped in parentheses in the assembled parent name."""
        named_parent = _make_named_parent(stem="butan", alkyl_stem="but", name="butane")
        numbering = _make_numbering(
            (0, Locant.numeric(1)),
            (1, Locant.numeric(2)),
            (2, Locant.numeric(3)),
            (3, Locant.numeric(4)),
        )
        # Methyl substituent at position 2 — represented as a LeafTree "methyl"
        # (as the engine would produce after its own SUBSTITUENT recursion)
        methyl_leaf = LeafTree(
            output_form=OutputForm.SUBSTITUENT,
            free_valence=None,
            choices_made=(),
            decision_ctx=None,
            validity_warnings=None,
            text="methyl",
        )
        pe = PrefixEntry(tree=methyl_leaf, locants=(Locant.numeric(2),))
        tree = SubstitutiveTree(
            output_form=OutputForm.STANDALONE,
            free_valence=None,
            choices_made=(),
            decision_ctx=None,
            validity_warnings=None,
            named_parent=named_parent,
            numbering=numbering,
            suffix_groups=(),
            unsaturation=(),
            prefixes=(pe,),
            stereo_descriptors=None,
            indicated_hydrogen=None,
        )
        result = assemble(tree)
        # Must be "2-methylbutane", not "(2-methyl)butane"
        assert result == "2-methylbutane", f"Expected '2-methylbutane', got {result!r}"
        assert "(" not in result, f"Unexpected parentheses: {result!r}"


# ---------------------------------------------------------------------------
# 14. Prefix hyphen insertion (Bug 1: missing hyphen between prefix groups)
# ---------------------------------------------------------------------------

class TestPrefixHyphenInsertion:
    """Phase 2a-3 Bug 1: When multiple locant-prefix pairs are assembled,
    a hyphen must separate each pair (P-16.3.2).

    OLD (wrong): "2-methyl5-propyl..."
    NEW (correct): "2-methyl-5-propyl..."
    """

    def _make_tree_with_two_simple_prefixes(self) -> SubstitutiveTree:
        """Build a SubstitutiveTree with methyl at 2 and propyl at 5."""
        named_parent = _make_named_parent(stem="cyclohexan", name="cyclohexane")
        numbering = _make_numbering(
            (0, Locant.numeric(1)),
            (1, Locant.numeric(2)),
            (2, Locant.numeric(3)),
            (3, Locant.numeric(4)),
            (4, Locant.numeric(5)),
            (5, Locant.numeric(6)),
        )
        methyl_leaf = LeafTree(
            output_form=OutputForm.SUBSTITUENT,
            free_valence=None,
            choices_made=(),
            decision_ctx=None,
            validity_warnings=None,
            text="methyl",
        )
        propyl_leaf = LeafTree(
            output_form=OutputForm.SUBSTITUENT,
            free_valence=None,
            choices_made=(),
            decision_ctx=None,
            validity_warnings=None,
            text="propyl",
        )
        pe_methyl = PrefixEntry(tree=methyl_leaf, locants=(Locant.numeric(2),))
        pe_propyl = PrefixEntry(tree=propyl_leaf, locants=(Locant.numeric(5),))

        return SubstitutiveTree(
            output_form=OutputForm.STANDALONE,
            free_valence=None,
            choices_made=(),
            decision_ctx=None,
            validity_warnings=None,
            named_parent=named_parent,
            numbering=numbering,
            suffix_groups=(),
            unsaturation=(),
            prefixes=(pe_methyl, pe_propyl),
            stereo_descriptors=None,
            indicated_hydrogen=None,
        )

    def test_two_prefixes_separated_by_hyphen(self):
        """Two locant-prefix pairs must be separated by a hyphen."""
        tree = self._make_tree_with_two_simple_prefixes()
        result = assemble(tree)
        # The assembled name should have a hyphen between "methyl" and "5"
        assert "methyl-5" in result, (
            f"Expected 'methyl-5' in assembled name, got {result!r}"
        )
        assert "methyl5" not in result, (
            f"Unexpected 'methyl5' (missing hyphen) in {result!r}"
        )

    def test_two_prefixes_full_name(self):
        """Full assembled name: 2-methyl-5-propylcyclohexane."""
        tree = self._make_tree_with_two_simple_prefixes()
        result = assemble(tree)
        assert result == "2-methyl-5-propylcyclohexane", (
            f"Expected '2-methyl-5-propylcyclohexane', got {result!r}"
        )

    def test_render_merged_prefixes_hyphen_between_groups(self):
        """render_merged_prefixes must insert a hyphen between locant-prefix groups."""
        from iupac_namer.assembly import render_merged_prefixes

        mp1 = MergedPrefix(
            name="methyl",
            locants=(Locant.numeric(2),),
            multiplier=None,
            sort_name="methyl",
            needs_brackets=False,
        )
        mp2 = MergedPrefix(
            name="propyl",
            locants=(Locant.numeric(5),),
            multiplier=None,
            sort_name="propyl",
            needs_brackets=False,
        )
        result = render_merged_prefixes([mp1, mp2])
        assert result == "2-methyl-5-propyl", (
            f"Expected '2-methyl-5-propyl', got {result!r}"
        )

    def test_render_merged_prefixes_three_prefixes(self):
        """Three separate simple prefixes: each pair separated by hyphen."""
        from iupac_namer.assembly import render_merged_prefixes

        mp1 = MergedPrefix(
            name="bromo",
            locants=(Locant.numeric(2),),
            multiplier=None,
            sort_name="bromo",
            needs_brackets=False,
        )
        mp2 = MergedPrefix(
            name="chloro",
            locants=(Locant.numeric(4),),
            multiplier=None,
            sort_name="chloro",
            needs_brackets=False,
        )
        mp3 = MergedPrefix(
            name="methyl",
            locants=(Locant.numeric(6),),
            multiplier=None,
            sort_name="methyl",
            needs_brackets=False,
        )
        result = render_merged_prefixes([mp1, mp2, mp3])
        assert result == "2-bromo-4-chloro-6-methyl", (
            f"Expected '2-bromo-4-chloro-6-methyl', got {result!r}"
        )

    def test_single_prefix_no_extra_hyphen(self):
        """Single prefix must not gain an extra leading or trailing hyphen."""
        from iupac_namer.assembly import render_merged_prefixes

        mp = MergedPrefix(
            name="methyl",
            locants=(Locant.numeric(2),),
            multiplier=None,
            sort_name="methyl",
            needs_brackets=False,
        )
        result = render_merged_prefixes([mp])
        assert result == "2-methyl", f"Expected '2-methyl', got {result!r}"

    def test_dimethyl_single_group_no_hyphen(self):
        """Two identical methyl groups merge into dimethyl — no extra hyphen inside."""
        from iupac_namer.assembly import render_merged_prefixes

        mp = MergedPrefix(
            name="methyl",
            locants=(Locant.numeric(2), Locant.numeric(5)),
            multiplier="di",
            sort_name="methyl",
            needs_brackets=False,
        )
        result = render_merged_prefixes([mp])
        assert result == "2,5-dimethyl", f"Expected '2,5-dimethyl', got {result!r}"

    def test_no_hyphen_before_bare_bracket_group(self):
        """P-16.3.3: an unbracketed leading simple prefix concatenates directly
        with a following locant-less bracket group — no separating hyphen.

        "chloro" + "(methyl)" -> "chloro(methyl)" (NOT "chloro-(methyl)").
        The enclosing mark itself is the boundary; a hyphen breaks the OPSIN
        round-trip.
        """
        from iupac_namer.assembly import render_merged_prefixes

        lead = MergedPrefix(
            name="chloro", locants=(), multiplier=None,
            sort_name="chloro", needs_brackets=False,
        )
        bracketed = MergedPrefix(
            name="methyl", locants=(), multiplier=None,
            sort_name="methyl", needs_brackets=True,
        )
        result = render_merged_prefixes([lead, bracketed])
        assert result == "chloro(methyl)", (
            f"Expected 'chloro(methyl)', got {result!r}"
        )

    def test_no_hyphen_multiplied_lead_before_bracket_group(self):
        """A multiplied unbracketed lead ("trichloro") still concatenates with a
        following locant-less bracket group without a hyphen."""
        from iupac_namer.assembly import render_merged_prefixes

        lead = MergedPrefix(
            name="chloro", locants=(), multiplier="tri",
            sort_name="chloro", needs_brackets=False,
        )
        bracketed = MergedPrefix(
            name="methyl", locants=(), multiplier=None,
            sort_name="methyl", needs_brackets=True,
        )
        result = render_merged_prefixes([lead, bracketed])
        assert result == "trichloro(methyl)", (
            f"Expected 'trichloro(methyl)', got {result!r}"
        )

    def test_hyphen_kept_before_bracket_group_with_locant(self):
        """Regression guard: a hyphen MUST still precede a bracket group that
        carries a locant (e.g. "2-methyl-3-(2-chloroethyl)...")."""
        from iupac_namer.assembly import render_merged_prefixes

        mp1 = MergedPrefix(
            name="methyl", locants=(Locant.numeric(2),), multiplier=None,
            sort_name="methyl", needs_brackets=False,
        )
        mp2 = MergedPrefix(
            name="2-chloroethyl", locants=(Locant.numeric(3),), multiplier=None,
            sort_name="chloroethyl", needs_brackets=True,
        )
        result = render_merged_prefixes([mp1, mp2])
        assert result == "2-methyl-3-(2-chloroethyl)", (
            f"Expected '2-methyl-3-(2-chloroethyl)', got {result!r}"
        )


# ---------------------------------------------------------------------------
# 15. Locant omission (P-14.6, Phase 2a-3 Bug 2)
# ---------------------------------------------------------------------------

class TestLocantOmission:
    """Tests for P-14.6 locant omission in suffix rendering.

    Rules implemented in _strip_locant_1_if_omissible:
    1. Terminal-always-C1 suffixes (al, oic acid, amide, nitrile):
       locant '1' omitted when there is exactly ONE group of that base_form.
    2. Single suffix at position 1 on chain of length 1 or 2: locant omitted.
    3. Longer chains / multiple groups: locant retained.
    """

    def _make_tree(
        self,
        stem: str,
        name: str,
        chain_length: int,
        base_form: str,
        locant: int | None,
        extra_suffix: tuple | None = None,
    ) -> SubstitutiveTree:
        """Build a minimal SubstitutiveTree for testing locant omission."""
        from iupac_namer.types import CandidateParent

        candidate = CandidateParent(
            atom_indices=frozenset(range(chain_length)),
            type="chain",
            length=chain_length,
            ring_system=None,
            unsaturation=None,
            element=None,
            lambda_value=None,
        )
        named_parent = NamedParent(
            candidate=candidate,
            name=name,
            stem=stem,
            alkyl_stem=None,
            naming_method="systematic",
            indicated_hydrogen=None,
            numbering_options=(),
        )
        numbering = _make_numbering(
            *[(i, Locant.numeric(i + 1)) for i in range(chain_length)]
        )
        locant_tuple = (Locant.numeric(locant),) if locant is not None else ()
        suffix_group = _make_suffix_group(base_form, locants=locant_tuple)
        suffix_groups: tuple = (suffix_group,)
        if extra_suffix is not None:
            extra_form, extra_loc = extra_suffix
            extra_locant_tuple = (Locant.numeric(extra_loc),) if extra_loc else ()
            extra_sg = _make_suffix_group(extra_form, locants=extra_locant_tuple)
            suffix_groups = (suffix_group, extra_sg)

        return SubstitutiveTree(
            output_form=OutputForm.STANDALONE,
            free_valence=None,
            choices_made=(),
            decision_ctx=None,
            validity_warnings=None,
            named_parent=named_parent,
            numbering=numbering,
            suffix_groups=suffix_groups,
            unsaturation=(),
            prefixes=(),
            stereo_descriptors=None,
            indicated_hydrogen=None,
        )

    # --- Rule 1: terminal-always-C1 suffixes ---

    def test_aldehyde_locant_1_omitted(self):
        """pentan + al at 1 → pentanal (not pentan-1-al)."""
        tree = self._make_tree("pentan", "pentane", 5, "al", locant=1)
        result = assemble(tree)
        assert result == "pentanal", f"Expected 'pentanal', got {result!r}"
        assert "-1-" not in result, f"Locant 1 should be omitted: {result!r}"

    def test_oic_acid_locant_1_omitted(self):
        """hexan + oic acid at 1 → hexanoic acid (not hexan-1-oic acid)."""
        tree = self._make_tree("hexan", "hexane", 6, "oic acid", locant=1)
        result = assemble(tree)
        assert result == "hexanoic acid", f"Expected 'hexanoic acid', got {result!r}"

    def test_amide_locant_1_omitted(self):
        """pentan + amide at 1 → pentanamide (not pentan-1-amide)."""
        tree = self._make_tree("pentan", "pentane", 5, "amide", locant=1)
        result = assemble(tree)
        assert result == "pentanamide", f"Expected 'pentanamide', got {result!r}"

    def test_nitrile_locant_1_omitted_single(self):
        """octan + nitrile at 1 → octanenitrile (not octan-1-nitrile).

        Only ONE nitrile group so Rule 1 applies (locant 1 omitted).
        The parent retains its terminal 'e' because 'nitrile' starts with a
        consonant (P-61.3.1 gives 'butanenitrile', not 'butannitrile').
        """
        tree = self._make_tree("octan", "octane", 8, "nitrile", locant=1)
        result = assemble(tree)
        # The locant 1 should be omitted: "octane" + "nitrile" → "octanenitrile"
        assert "-1-" not in result, f"Locant 1 should be omitted: {result!r}"
        assert result == "octanenitrile", f"Expected 'octanenitrile', got {result!r}"

    def test_dinitrile_locants_retained(self):
        """hepta + nitrile at 1 + nitrile at 7 → heptan-1,7-dinitrile.
        TWO nitrile groups → Rule 1 does NOT apply.
        """
        # Build with two nitrile groups
        from iupac_namer.types import CandidateParent

        candidate = CandidateParent(
            atom_indices=frozenset(range(7)),
            type="chain",
            length=7,
            ring_system=None,
            unsaturation=None,
            element=None,
            lambda_value=None,
        )
        named_parent = NamedParent(
            candidate=candidate,
            name="heptane",
            stem="heptan",
            alkyl_stem=None,
            naming_method="systematic",
            indicated_hydrogen=None,
            numbering_options=(),
        )
        numbering = _make_numbering(*[(i, Locant.numeric(i + 1)) for i in range(7)])
        sg1 = _make_suffix_group("nitrile", locants=(Locant.numeric(1),))
        sg7 = _make_suffix_group("nitrile", locants=(Locant.numeric(7),))
        tree = SubstitutiveTree(
            output_form=OutputForm.STANDALONE,
            free_valence=None,
            choices_made=(),
            decision_ctx=None,
            validity_warnings=None,
            named_parent=named_parent,
            numbering=numbering,
            suffix_groups=(sg1, sg7),
            unsaturation=(),
            prefixes=(),
            stereo_descriptors=None,
            indicated_hydrogen=None,
        )
        result = assemble(tree)
        # Both locants must be retained: "heptan-1,7-dinitrile"
        assert "1" in result, f"Locant 1 must be retained for dinitrile: {result!r}"
        assert "7" in result, f"Locant 7 must be retained for dinitrile: {result!r}"

    # --- Rule 3 (P-14.3.4.4 / P-66.6.3): chain-terminal di-suffix omission ---

    def test_dioic_acid_terminal_locants_omitted(self):
        """butane + oic acid at 1 + oic acid at 4 → butanedioic acid.

        Both acid carbons ARE chain termini by definition, so the 1,4-
        locants are forced and omitted from the PIN (P-66.6.3).
        """
        tree = self._make_tree(
            "butan", "butane", 4, "oic acid", locant=1,
            extra_suffix=("oic acid", 4),
        )
        result = assemble(tree)
        assert result == "butanedioic acid", (
            f"Expected 'butanedioic acid', got {result!r}"
        )
        assert "1,4" not in result, f"Terminal locants should be omitted: {result!r}"

    def test_dial_terminal_locants_omitted(self):
        """pentane + al at 1 + al at 5 → pentanedial (not pentane-1,5-dial)."""
        tree = self._make_tree(
            "pentan", "pentane", 5, "al", locant=1,
            extra_suffix=("al", 5),
        )
        result = assemble(tree)
        assert result == "pentanedial", f"Expected 'pentanedial', got {result!r}"
        assert "1,5" not in result, f"Terminal locants should be omitted: {result!r}"

    def test_dithial_terminal_locants_omitted(self):
        """pentane + thial at 1 + thial at 5 → pentanedithial."""
        tree = self._make_tree(
            "pentan", "pentane", 5, "thial", locant=1,
            extra_suffix=("thial", 5),
        )
        result = assemble(tree)
        assert result == "pentanedithial", (
            f"Expected 'pentanedithial', got {result!r}"
        )

    # --- Rule 2: single suffix on short chain ---

    def test_ethanol_locant_1_omitted(self):
        """ethan + ol at 1 (chain length 2) → ethanol (not ethan-1-ol)."""
        tree = self._make_tree("ethan", "ethane", 2, "ol", locant=1)
        result = assemble(tree)
        assert result == "ethanol", f"Expected 'ethanol', got {result!r}"

    def test_methanol_locant_1_omitted(self):
        """methan + ol at 1 (chain length 1) → methanol (not methan-1-ol)."""
        tree = self._make_tree("methan", "methane", 1, "ol", locant=1)
        result = assemble(tree)
        assert result == "methanol", f"Expected 'methanol', got {result!r}"

    # --- Rule 3: longer chains keep locants ---

    def test_propan_1_ol_keeps_locant(self):
        """propan + ol at 1 (chain length 3) → propan-1-ol (locant retained).
        Propan-2-ol exists, so the position is ambiguous and locant is cited.
        """
        tree = self._make_tree("propan", "propane", 3, "ol", locant=1)
        result = assemble(tree)
        assert result == "propan-1-ol", f"Expected 'propan-1-ol', got {result!r}"

    def test_non_1_locant_always_kept(self):
        """A suffix at position 2 (not 1) is NEVER omitted."""
        tree = self._make_tree("propan", "propane", 3, "ol", locant=2)
        result = assemble(tree)
        assert "2" in result, f"Locant 2 must be retained: {result!r}"
        assert result == "propan-2-ol", f"Expected 'propan-2-ol', got {result!r}"
