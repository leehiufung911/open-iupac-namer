"""
tests/test_data_loader.py

Tests for iupac_namer/data_loader.py.
All tests are pure Python — no RDKit required.
"""

import pytest
from pathlib import Path

import iupac_namer.data_loader as dl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_function():
    """Clear cache before each test function for isolation."""
    dl.clear_cache()


# ---------------------------------------------------------------------------
# Data directory resolution
# ---------------------------------------------------------------------------

class TestDataDirectory:
    def test_data_dir_exists(self):
        assert dl._DATA_DIR.is_dir(), f"Expected data dir at {dl._DATA_DIR}"

    def test_bluebook_dir_exists(self):
        assert dl._BLUEBOOK_DIR.is_dir()

    def test_opsin_dir_exists(self):
        assert dl._OPSIN_DIR.is_dir()

    def test_data_dir_relative_to_module(self):
        module_path = Path(dl.__file__)
        expected = module_path.parent.parent / "data"
        assert dl._DATA_DIR.resolve() == expected.resolve()


# ---------------------------------------------------------------------------
# Chain stems
# ---------------------------------------------------------------------------

class TestChainStems:
    def test_loads_without_error(self):
        stems = dl.get_chain_stems()
        assert isinstance(stems, dict)
        assert len(stems) > 0

    def test_meth(self):
        assert dl.get_chain_stem(1) == "meth"

    def test_eth(self):
        assert dl.get_chain_stem(2) == "eth"

    def test_prop(self):
        assert dl.get_chain_stem(3) == "prop"

    def test_but(self):
        assert dl.get_chain_stem(4) == "but"

    def test_pent(self):
        assert dl.get_chain_stem(5) == "pent"

    def test_hex(self):
        assert dl.get_chain_stem(6) == "hex"

    def test_missing_returns_none(self):
        assert dl.get_chain_stem(9999) is None

    def test_cached_is_same_object(self):
        a = dl.get_chain_stems()
        b = dl.get_chain_stems()
        assert a is b


# ---------------------------------------------------------------------------
# Multipliers
# ---------------------------------------------------------------------------

class TestMultipliers:
    def test_simple_di(self):
        assert dl.get_multiplier(2) == "di"

    def test_simple_tri(self):
        assert dl.get_multiplier(3) == "tri"

    def test_simple_tetra(self):
        assert dl.get_multiplier(4) == "tetra"

    def test_complex_bis(self):
        assert dl.get_multiplier(2, complex=True) == "bis"

    def test_complex_tris(self):
        assert dl.get_multiplier(3, complex=True) == "tris"

    def test_complex_tetrakis(self):
        assert dl.get_multiplier(4, complex=True) == "tetrakis"

    def test_missing_returns_none(self):
        assert dl.get_multiplier(9999) is None

    def test_raw_has_both_keys(self):
        raw = dl.get_multipliers_raw()
        assert "simple" in raw
        assert "complex" in raw


# ---------------------------------------------------------------------------
# Functional groups
# ---------------------------------------------------------------------------

class TestFunctionalGroups:
    def test_loads_without_error(self):
        fg = dl.get_functional_groups()
        assert isinstance(fg, dict)

    def test_has_suffix_groups(self):
        fg = dl.get_functional_groups()
        assert "suffix_groups" in fg
        assert isinstance(fg["suffix_groups"], list)
        assert len(fg["suffix_groups"]) > 0

    def test_suffix_group_has_required_fields(self):
        fg = dl.get_functional_groups()
        first = fg["suffix_groups"][0]
        assert "name" in first
        assert "smarts" in first
        assert "seniority" in first


# ---------------------------------------------------------------------------
# HW tables
# ---------------------------------------------------------------------------

class TestHWTables:
    def test_loads_without_error(self):
        hw = dl.get_hw_tables()
        assert isinstance(hw, dict)

    def test_has_prefixes(self):
        hw = dl.get_hw_tables()
        assert "prefixes" in hw
        assert isinstance(hw["prefixes"], list)

    def test_has_stems(self):
        hw = dl.get_hw_tables()
        assert "stems" in hw


# ---------------------------------------------------------------------------
# Element seniority
# ---------------------------------------------------------------------------

class TestElementSeniority:
    def test_loads_without_error(self):
        es = dl.get_element_seniority()
        assert isinstance(es, dict)

    def test_has_parent_selection(self):
        es = dl.get_element_seniority()
        assert "parent_selection" in es
        assert isinstance(es["parent_selection"], list)

    def test_carbon_in_parent_selection(self):
        es = dl.get_element_seniority()
        assert "C" in es["parent_selection"]


# ---------------------------------------------------------------------------
# Retained names
# ---------------------------------------------------------------------------

class TestRetainedNames:
    def test_loads_without_error(self):
        rn = dl.get_retained_names()
        assert isinstance(rn, dict)

    def test_has_functional_parents(self):
        rn = dl.get_retained_names()
        assert "functional_parents" in rn

    def test_acetic_acid_present(self):
        rn = dl.get_retained_names()
        assert "acetic acid" in rn["functional_parents"]

    def test_expanded_loads_without_error(self):
        rne = dl.get_retained_names_expanded()
        assert isinstance(rne, dict)


# ---------------------------------------------------------------------------
# Retained rings
# ---------------------------------------------------------------------------

class TestRetainedRings:
    def test_loads_without_error(self):
        rr = dl.get_retained_rings()
        assert isinstance(rr, dict)

    def test_has_monocyclic_heterocycles(self):
        rr = dl.get_retained_rings()
        assert "monocyclic_heterocycles" in rr

    def test_furan_present(self):
        rr = dl.get_retained_rings()
        assert "furan" in rr["monocyclic_heterocycles"]


# ---------------------------------------------------------------------------
# Substituent names
# ---------------------------------------------------------------------------

class TestSubstituentNames:
    def test_loads_without_error(self):
        sn = dl.get_substituent_names()
        assert isinstance(sn, dict)


# ---------------------------------------------------------------------------
# Fusion components
# ---------------------------------------------------------------------------

class TestFusionComponents:
    def test_loads_without_error(self):
        fc = dl.get_fusion_components()
        assert isinstance(fc, dict)
        assert len(fc) > 0


# ---------------------------------------------------------------------------
# Blue Book tables
# ---------------------------------------------------------------------------

class TestBluebookPrefixes:
    def test_loads_without_error(self):
        bp = dl.get_bluebook_prefixes()
        assert bp is not None


class TestSeniorityOrder:
    def test_loads_without_error(self):
        so = dl.get_seniority_order()
        assert isinstance(so, list)
        assert len(so) > 0


class TestElementAPrefixes:
    def test_loads_without_error(self):
        ep = dl.get_element_a_prefixes()
        assert isinstance(ep, list)
        assert len(ep) > 0

    def test_has_expected_structure(self):
        ep = dl.get_element_a_prefixes()
        first = ep[0]
        # Each entry should have element and a_term fields
        assert "element" in first
        assert "a_term" in first

    def test_oxygen_entry(self):
        ep = dl.get_element_a_prefixes()
        oxygen = next((e for e in ep if e.get("element") == "O"), None)
        assert oxygen is not None
        assert oxygen["a_term"] == "oxa"


class TestReplacementAPrefixes:
    def test_loads_without_error(self):
        rp = dl.get_replacement_a_prefixes()
        assert isinstance(rp, list)
        assert len(rp) > 0

    def test_has_element_and_a_prefix(self):
        rp = dl.get_replacement_a_prefixes()
        first = rp[0]
        assert "element" in first
        assert "a_prefix" in first


# ---------------------------------------------------------------------------
# OPSIN-extracted tables
# ---------------------------------------------------------------------------

class TestRetainedNamesFromOpsin:
    def test_loads_without_error(self):
        rno = dl.get_retained_names_from_opsin()
        assert isinstance(rno, list)
        assert len(rno) > 0

    def test_has_name_and_smiles(self):
        rno = dl.get_retained_names_from_opsin()
        first = rno[0]
        assert "name" in first
        assert "smiles" in first


class TestRingsFromOpsin:
    def test_loads_without_error(self):
        rfo = dl.get_rings_from_opsin()
        assert isinstance(rfo, list)
        assert len(rfo) > 0

    def test_has_name_and_smiles(self):
        rfo = dl.get_rings_from_opsin()
        first = rfo[0]
        assert "name" in first
        assert "smiles" in first


# ---------------------------------------------------------------------------
# Generic load_table
# ---------------------------------------------------------------------------

class TestLoadTable:
    def test_load_chain_stems(self):
        t = dl.load_table("chain_stems")
        assert isinstance(t, dict)

    def test_load_with_json_extension(self):
        t = dl.load_table("chain_stems.json")
        assert isinstance(t, dict)

    def test_load_bluebook_table(self):
        t = dl.load_table("element_a_prefixes")
        assert isinstance(t, list)

    def test_load_opsin_table(self):
        t = dl.load_table("rings_from_opsin")
        assert isinstance(t, list)

    def test_load_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            dl.load_table("nonexistent_table_xyz")


# ---------------------------------------------------------------------------
# ACID_ADJECTIVE_TABLE
# ---------------------------------------------------------------------------

class TestAcidAdjectiveTable:
    def test_is_dict(self):
        assert isinstance(dl.ACID_ADJECTIVE_TABLE, dict)

    def test_formic_acid(self):
        assert dl.ACID_ADJECTIVE_TABLE["formic acid"] == "form"

    def test_acetic_acid(self):
        assert dl.ACID_ADJECTIVE_TABLE["acetic acid"] == "acet"

    def test_propionic_acid(self):
        assert dl.ACID_ADJECTIVE_TABLE["propionic acid"] == "propion"

    def test_butyric_acid(self):
        assert dl.ACID_ADJECTIVE_TABLE["butyric acid"] == "butyr"

    def test_benzoic_acid(self):
        assert dl.ACID_ADJECTIVE_TABLE["benzoic acid"] == "benzo"

    def test_oxalic_acid(self):
        assert dl.ACID_ADJECTIVE_TABLE["oxalic acid"] == "oxal"

    def test_succinic_acid(self):
        assert dl.ACID_ADJECTIVE_TABLE["succinic acid"] == "succin"

    def test_glutaric_acid(self):
        assert dl.ACID_ADJECTIVE_TABLE["glutaric acid"] == "glutar"

    def test_phthalic_acid(self):
        assert dl.ACID_ADJECTIVE_TABLE["phthalic acid"] == "phthal"


# ---------------------------------------------------------------------------
# SUFFIX_ELISION_TABLE & suffix_elides_terminal_e
# ---------------------------------------------------------------------------

class TestSuffixElision:
    def test_ol_elides(self):
        assert dl.suffix_elides_terminal_e("-ol") is True

    def test_one_elides(self):
        assert dl.suffix_elides_terminal_e("-one") is True

    def test_al_elides(self):
        assert dl.suffix_elides_terminal_e("-al") is True

    def test_yl_no_elision(self):
        assert dl.suffix_elides_terminal_e("-yl") is False

    def test_amine_no_elision(self):
        # Special case: -amine does NOT elide despite starting with vowel
        assert dl.suffix_elides_terminal_e("-amine") is False

    def test_nitrile_no_elision(self):
        assert dl.suffix_elides_terminal_e("-nitrile") is False

    def test_fallback_vowel_start_elides(self):
        # Unlisted suffix starting with vowel → elide
        assert dl.suffix_elides_terminal_e("-oxy") is True

    def test_fallback_consonant_no_elision(self):
        # Unlisted suffix starting with consonant → no elide
        assert dl.suffix_elides_terminal_e("-thio") is False

    def test_suffix_elision_table_is_dict(self):
        assert isinstance(dl.SUFFIX_ELISION_TABLE, dict)
        assert len(dl.SUFFIX_ELISION_TABLE) > 0


# ---------------------------------------------------------------------------
# Convenience lookup functions
# ---------------------------------------------------------------------------

class TestLookupRetainedName:
    def test_returns_none_for_unknown_smiles(self):
        result = dl.lookup_retained_name("ZZZZZZ_not_real")
        assert result is None

    def test_finds_entry_in_expanded(self):
        # retained_names_expanded.json contains SMILES-keyed entries
        # Pick a SMILES we know is in the file from earlier inspection
        rne = dl.get_retained_names_expanded()
        # Find first SMILES key in any category
        target_smiles = None
        target_record = None
        for cat_data in rne.values():
            if isinstance(cat_data, dict) and cat_data:
                target_smiles = next(iter(cat_data))
                target_record = cat_data[target_smiles]
                break
        if target_smiles is None:
            pytest.skip("No SMILES-keyed entries found in retained_names_expanded.json")
        result = dl.lookup_retained_name(target_smiles)
        assert result is not None
        assert result["smiles"] == target_smiles


class TestLookupRetainedRing:
    def test_returns_none_for_unknown_name(self):
        result = dl.lookup_retained_ring("notaringxyz123")
        assert result is None

    def test_finds_furan(self):
        result = dl.lookup_retained_ring("furan")
        assert result is not None
        assert "furan" in result.get("name", "").lower()

    def test_case_insensitive(self):
        result_lower = dl.lookup_retained_ring("furan")
        result_upper = dl.lookup_retained_ring("Furan")
        assert (result_lower is None) == (result_upper is None)
