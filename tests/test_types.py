"""
tests/test_types.py
Basic tests for the typed dataclasses in iupac_namer/types.py.
"""
import dataclasses
import pytest

from iupac_namer.types import (
    Locant,
    OutputForm,
    Choice,
    NamingSession,
    FreeValenceInfo,
    SubstituentMethod,
    LeafTree,
    InterpretationQuery,
    Numbering,
    DetectedFG,
)


# ---------------------------------------------------------------------------
# Locant tests
# ---------------------------------------------------------------------------

class TestLocant:
    def test_numeric_str(self):
        loc = Locant.numeric(2)
        assert str(loc) == "2"

    def test_hetero_str(self):
        loc = Locant.hetero("N")
        assert str(loc) == "N"

    def test_numeric_compound_str(self):
        loc = Locant.numeric(4, "a")
        assert str(loc) == "4a"

    def test_hetero_superscript_str(self):
        loc = Locant.hetero("N", sup="2")
        assert str(loc) == "N2"

    def test_primed_str(self):
        loc = Locant.numeric(1, "'")
        assert str(loc) == "1'"

    def test_numeric_lt_hetero(self):
        """Numeric locants sort before heteroatom locants (P-14.4)."""
        num = Locant.numeric(2)
        het = Locant.hetero("N")
        assert num < het
        assert not (het < num)

    def test_numeric_ordering(self):
        loc1 = Locant.numeric(1)
        loc2 = Locant.numeric(2)
        loc3 = Locant.numeric(10)
        assert loc1 < loc2
        assert loc2 < loc3
        assert not (loc2 < loc1)

    def test_numeric_compound_ordering(self):
        """4a comes after 4 (suffix comparison)."""
        loc4 = Locant.numeric(4)
        loc4a = Locant.numeric(4, "a")
        assert loc4 < loc4a  # "" < "a"

    def test_hetero_alphabetical_ordering(self):
        loc_n = Locant.hetero("N")
        loc_o = Locant.hetero("O")
        loc_s = Locant.hetero("S")
        assert loc_n < loc_o
        assert loc_o < loc_s

    def test_equality_and_hash(self):
        a = Locant.numeric(3)
        b = Locant.numeric(3)
        assert a == b
        assert hash(a) == hash(b)
        assert a != Locant.numeric(4)

    def test_frozen_enforcement(self):
        """Assigning to a frozen field raises FrozenInstanceError."""
        loc = Locant.numeric(2)
        with pytest.raises(dataclasses.FrozenInstanceError):
            loc.label = "5"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# OutputForm tests
# ---------------------------------------------------------------------------

class TestOutputForm:
    def test_standalone_exists(self):
        form = OutputForm.STANDALONE
        assert form is OutputForm.STANDALONE

    def test_all_variants_exist(self):
        variants = [
            OutputForm.STANDALONE,
            OutputForm.SUBSTITUENT,
            OutputForm.ACID_STEM,
            OutputForm.ACYL,
            OutputForm.ANION,
            OutputForm.CATION,
            OutputForm.PARENT_HYDRIDE,
        ]
        assert len(variants) == 7

    def test_output_form_hashable(self):
        s = {OutputForm.STANDALONE, OutputForm.SUBSTITUENT}
        assert OutputForm.STANDALONE in s


# ---------------------------------------------------------------------------
# Choice tests
# ---------------------------------------------------------------------------

class TestChoice:
    def test_simple_choice(self):
        c = Choice(type="substitutive", detail="hexane as parent chain")
        assert c.type == "substitutive"
        assert c.detail == "hexane as parent chain"

    def test_choice_frozen(self):
        c = Choice(type="retained", detail="benzene")
        with pytest.raises(dataclasses.FrozenInstanceError):
            c.type = "substitutive"  # type: ignore[misc]

    def test_choice_equality(self):
        c1 = Choice(type="retained", detail="benzene")
        c2 = Choice(type="retained", detail="benzene")
        assert c1 == c2

    def test_choice_hashable(self):
        c = Choice(type="substitutive", detail="test")
        d = {c: 1}
        assert d[c] == 1


# ---------------------------------------------------------------------------
# FreeValenceInfo / suffix property
# ---------------------------------------------------------------------------

class TestFreeValenceInfo:
    def test_monovalent_single_bond_suffix(self):
        fv = FreeValenceInfo(
            bond_orders=(1,),
            method=SubstituentMethod.ALKYL,
            attachment_atoms_in_fragment=None,
        )
        assert fv.suffix == "yl"
        assert fv.is_monovalent

    def test_monovalent_double_bond_suffix(self):
        fv = FreeValenceInfo(
            bond_orders=(2,),
            method=SubstituentMethod.ALKANYL,
            attachment_atoms_in_fragment=None,
        )
        assert fv.suffix == "ylidene"

    def test_monovalent_triple_bond_suffix(self):
        fv = FreeValenceInfo(
            bond_orders=(3,),
            method=SubstituentMethod.ALKANYL,
            attachment_atoms_in_fragment=None,
        )
        assert fv.suffix == "ylidyne"

    def test_divalent_diyl_suffix(self):
        fv = FreeValenceInfo(
            bond_orders=(1, 1),
            method=SubstituentMethod.ALKANYL,
            attachment_atoms_in_fragment=None,
        )
        assert fv.suffix == "diyl"
        assert not fv.is_monovalent

    def test_fallback_suffix_for_unknown(self):
        fv = FreeValenceInfo(
            bond_orders=(1, 1, 1, 1),
            method=SubstituentMethod.ALKANYL,
            attachment_atoms_in_fragment=None,
        )
        # 4 attachment points, no entry in FREE_VALENCE_SUFFIXES -> fallback
        assert fv.suffix == "4yl"

    def test_frozen(self):
        fv = FreeValenceInfo(
            bond_orders=(1,),
            method=SubstituentMethod.ALKYL,
            attachment_atoms_in_fragment=None,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            fv.bond_orders = (2,)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# NamingSession (mutable)
# ---------------------------------------------------------------------------

class TestNamingSession:
    def test_default_construction(self):
        session = NamingSession()
        assert session.max_depth == 24
        assert session.cache == {}
        assert session._plan_seq == 0

    def test_next_seq_increments(self):
        session = NamingSession()
        assert session.next_seq() == 1
        assert session.next_seq() == 2
        assert session.next_seq() == 3

    def test_cache_store_and_lookup(self):
        session = NamingSession()
        tree = LeafTree(
            output_form=OutputForm.STANDALONE,
            free_valence=None,
            choices_made=(),
            decision_ctx=None,
            validity_warnings=None,
            text="ethanol",
        )
        session.cache_store("CCO", OutputForm.STANDALONE, (1,), tree)
        result = session.cache_lookup("CCO", OutputForm.STANDALONE, (1,))
        assert result is tree

    def test_cache_miss_returns_none(self):
        session = NamingSession()
        result = session.cache_lookup("CC", OutputForm.STANDALONE, (1,))
        assert result is None

    def test_not_frozen(self):
        """NamingSession is NOT frozen -- mutations must work."""
        session = NamingSession()
        session.max_depth = 20
        assert session.max_depth == 20


# ---------------------------------------------------------------------------
# InterpretationQuery.with_override
# ---------------------------------------------------------------------------

class TestInterpretationQuery:
    def test_with_override(self):
        q = InterpretationQuery(
            preferred_decomp_types=None,
            preferred_parent_type="chain",
            suppress_functional_class=False,
            max_results=10,
        )
        q2 = q.with_override(max_results=5, suppress_functional_class=True)
        assert q2.max_results == 5
        assert q2.suppress_functional_class is True
        # Original unchanged
        assert q.max_results == 10
        assert q.suppress_functional_class is False

    def test_frozen(self):
        q = InterpretationQuery(
            preferred_decomp_types=None,
            preferred_parent_type=None,
            suppress_functional_class=False,
            max_results=10,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            q.max_results = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# LeafTree.with_warnings
# ---------------------------------------------------------------------------

class TestLeafTreeWithWarnings:
    def test_with_warnings_adds_warning(self):
        t = LeafTree(
            output_form=OutputForm.STANDALONE,
            free_valence=None,
            choices_made=(),
            decision_ctx=None,
            validity_warnings=None,
            text="methane",
        )
        t2 = t.with_warnings("test warning")
        assert t2.validity_warnings == ("test warning",)
        # Original unchanged
        assert t.validity_warnings is None

    def test_with_warnings_appends(self):
        t = LeafTree(
            output_form=OutputForm.STANDALONE,
            free_valence=None,
            choices_made=(),
            decision_ctx=None,
            validity_warnings=("first",),
            text="methane",
        )
        t2 = t.with_warnings("second", "third")
        assert t2.validity_warnings == ("first", "second", "third")


# ---------------------------------------------------------------------------
# DetectedFG properties API
# ---------------------------------------------------------------------------

class TestDetectedFG:
    def _make_fg(self) -> DetectedFG:
        return DetectedFG(
            type="alcohol",
            atoms=frozenset({0, 1}),
            anchor=0,
            properties=(("terminal", True), ("in_ring", False)),
            suffix_eligible=True,
            suffix_forms=(("terminal", "-ol"), ("nonterminal", "-ol")),
            prefix_form="hydroxy-",
        )

    def test_get_property_found(self):
        fg = self._make_fg()
        assert fg.get_property("terminal") is True
        assert fg.get_property("in_ring") is False

    def test_get_property_missing_default(self):
        fg = self._make_fg()
        assert fg.get_property("nonexistent") is None
        assert fg.get_property("nonexistent", "fallback") == "fallback"

    def test_properties_dict(self):
        fg = self._make_fg()
        d = fg.properties_dict()
        assert d == {"terminal": True, "in_ring": False}

    def test_suffix_forms_dict(self):
        fg = self._make_fg()
        d = fg.suffix_forms_dict()
        assert d["terminal"] == "-ol"

    def test_frozen(self):
        fg = self._make_fg()
        with pytest.raises(dataclasses.FrozenInstanceError):
            fg.type = "ketone"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Numbering properties
# ---------------------------------------------------------------------------

class TestNumbering:
    def test_atom_to_locant(self):
        assignments = ((0, Locant.numeric(1)), (1, Locant.numeric(2)))
        numbering = Numbering(_assignments=assignments, locant_set=(Locant.numeric(1), Locant.numeric(2)))
        mapping = numbering.atom_to_locant
        assert mapping[0] == Locant.numeric(1)
        assert mapping[1] == Locant.numeric(2)

    def test_locant_to_atom(self):
        assignments = ((0, Locant.numeric(1)), (1, Locant.numeric(2)))
        numbering = Numbering(_assignments=assignments, locant_set=(Locant.numeric(1), Locant.numeric(2)))
        mapping = numbering.locant_to_atom
        assert mapping[Locant.numeric(1)] == 0
        assert mapping[Locant.numeric(2)] == 1

    def test_frozen(self):
        numbering = Numbering(_assignments=(), locant_set=())
        with pytest.raises(dataclasses.FrozenInstanceError):
            numbering._assignments = ((0, Locant.numeric(1)),)  # type: ignore[misc]
