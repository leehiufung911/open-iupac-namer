"""Regression tests for Hantzsch-Widman table extension to Al/Ga/In/Tl/Hg (Stage 6 R1-C).

The HW prefixes table (``data/hw_tables.json``) originally ended at B/bora.
Blue Book P-22.2.2.1.1 Table 2.7 prescribes the additional "a" prefixes
aluma, galla, indiga, thalla, mercura for Al, Ga, In, Tl, Hg respectively;
they all attach to the standard HW ring-size stems.

Group membership (six-membered stems): OPSIN 2.8.0 round-trips
``aluminine``/``gallinine``/``indiginine``/``thallinine``/``mercurinine``
to ``[M]1=CC=CC=C1`` style, which is the **group-C** (``inine``/``inane``)
behaviour. All five new elements are therefore in group C for
``six_membered_groups``.

Valences: Al/Ga/In/Tl = 3 (group-13 trivalent); Hg = 2 (divalent).

Not asserted here (owned by agent R2-D): 6-ring group-C unsaturated stem
selector (``inine``) -- currently emits ``inane`` when RDKit perceives the
ring as non-aromatic.
"""
from __future__ import annotations

from iupac_namer.data_loader import get_hw_tables
from iupac_namer.engine import name_smiles
from iupac_namer.ring_naming.monocyclic import _STANDARD_VALENCE


class TestHWPrefixTable:
    _EXPECTED = {
        "Al": ("aluma",   3),
        "Ga": ("galla",   3),
        "In": ("indiga",  3),
        "Tl": ("thalla",  3),
        "Hg": ("mercura", 2),
    }

    def _prefix_map(self):
        tables = get_hw_tables()
        return {e["element"]: (e["prefix"], e["bonding_number"])
                for e in tables["prefixes"]}

    def test_aluminum_prefix(self):
        assert self._prefix_map()["Al"] == self._EXPECTED["Al"]

    def test_gallium_prefix(self):
        assert self._prefix_map()["Ga"] == self._EXPECTED["Ga"]

    def test_indium_prefix(self):
        assert self._prefix_map()["In"] == self._EXPECTED["In"]

    def test_thallium_prefix(self):
        assert self._prefix_map()["Tl"] == self._EXPECTED["Tl"]

    def test_mercury_prefix(self):
        assert self._prefix_map()["Hg"] == self._EXPECTED["Hg"]


class TestSixMemberedGroup:
    def test_group_C_contains_all_five(self):
        group_c = set(get_hw_tables()["six_membered_groups"]["C"])
        for elem in ("Al", "Ga", "In", "Tl", "Hg"):
            assert elem in group_c, f"{elem} should be in 6-ring group C"

    def test_not_in_group_A_or_B(self):
        tables = get_hw_tables()
        group_a = set(tables["six_membered_groups"]["A"])
        group_b = set(tables["six_membered_groups"]["B"])
        for elem in ("Al", "Ga", "In", "Tl", "Hg"):
            assert elem not in group_a
            assert elem not in group_b


class TestStandardValence:
    def test_al_valence_3(self):
        assert _STANDARD_VALENCE["Al"] == 3

    def test_ga_valence_3(self):
        assert _STANDARD_VALENCE["Ga"] == 3

    def test_in_valence_3(self):
        assert _STANDARD_VALENCE["In"] == 3

    def test_tl_valence_3(self):
        assert _STANDARD_VALENCE["Tl"] == 3

    def test_hg_valence_2(self):
        assert _STANDARD_VALENCE["Hg"] == 2


def _name_ok(smiles):
    try:
        result = name_smiles(smiles)
    except Exception as exc:
        return False, f"EXC: {exc}"
    if not isinstance(result, str):
        return False, f"non-str: {type(result).__name__}"
    if "NAMING ERROR" in result or result.startswith("["):
        return False, result
    return True, result


class TestHWRingNoNamingError:
    """Every HW matrix cell for new elements produces SOME name (closes 80
    NAMING_ERROR rows from Stage 5 Agent 5)."""

    _CASES = [
        ("alumirane",   "[AlH]1CC1",       "alumirane"),
        ("alumirene",   "[AlH]1C=C1",      "alumirene"),
        ("alumetane",   "[AlH]1CCC1",      "alumetane"),
        ("alumolane",   "[AlH]1CCCC1",     "alumolane"),
        ("alumole",     "[AlH]1C=CC=C1",   "alumole"),
        ("aluminane",   "[AlH]1CCCCC1",    "aluminane"),
        ("gallirane",   "[GaH]1CC1",       "gallirane"),
        ("gallirene",   "[GaH]1C=C1",      "gallirene"),
        ("galletane",   "[GaH]1CCC1",      "galletane"),
        ("gallolane",   "[GaH]1CCCC1",     "gallolane"),
        ("gallole",     "[GaH]1C=CC=C1",   "gallole"),
        ("gallinane",   "[GaH]1CCCCC1",    "gallinane"),
        ("indigirane",  "[InH]1CC1",       "indigirane"),
        ("indigirene",  "[InH]1C=C1",      "indigirene"),
        ("indigetane",  "[InH]1CCC1",      "indigetane"),
        ("indigolane",  "[InH]1CCCC1",     "indigolane"),
        ("indigole",    "[InH]1C=CC=C1",   "indigole"),
        ("indiginane",  "[InH]1CCCCC1",    "indiginane"),
        ("thallirane",  "[TlH]1CC1",       "thallirane"),
        ("thallirene",  "[TlH]1C=C1",      "thallirene"),
        ("thalletane",  "[TlH]1CCC1",      "thalletane"),
        ("thallolane",  "[TlH]1CCCC1",     "thallolane"),
        ("thallole",    "[TlH]1C=CC=C1",   "thallole"),
        ("thallinane",  "[TlH]1CCCCC1",    "thallinane"),
        ("mercurirane", "[Hg]1CC1",        "mercurirane"),
        ("mercurirene", "[Hg]1C=C1",       "mercurirene"),
        ("mercuretane", "[Hg]1CCC1",       "mercuretane"),
        ("mercurolane", "[Hg]1CCCC1",      "mercurolane"),
        ("mercurole",   "[Hg]1C=CC=C1",    "mercurole"),
        ("mercurinane", "[Hg]1CCCCC1",     "mercurinane"),
    ]

    def test_all_new_hw_probes_produce_name(self):
        failures = []
        for label, smi, expected_stem in self._CASES:
            ok, name = _name_ok(smi)
            if not ok:
                failures.append(f"{label} ({smi}): {name}")
                continue
            if expected_stem not in name:
                failures.append(f"{label} ({smi}): name={name!r} missing {expected_stem!r}")
        assert not failures, "HW probes failed:\n" + "\n".join(failures)


class TestHWSaturated6RingExact:
    """Group-C saturated 6-ring uses 'inane' stem."""

    def test_aluminane(self):
        ok, name = _name_ok("[AlH]1CCCCC1")
        assert ok and name == "aluminane", (ok, name)

    def test_gallinane(self):
        ok, name = _name_ok("[GaH]1CCCCC1")
        assert ok and name == "gallinane", (ok, name)

    def test_indiginane(self):
        ok, name = _name_ok("[InH]1CCCCC1")
        assert ok and name == "indiginane", (ok, name)

    def test_thallinane(self):
        ok, name = _name_ok("[TlH]1CCCCC1")
        assert ok and name == "thallinane", (ok, name)

    def test_mercurinane(self):
        ok, name = _name_ok("[Hg]1CCCCC1")
        assert ok and name == "mercurinane", (ok, name)


class TestAluminineAromaticPerception:
    """RDKit perceives [Al]1=CC=CC=C1 as aromatic, so we correctly emit the
    unsaturated 'inine' stem for Al 6-ring. Ga/In/Tl/Hg analogues rely on
    R2-D's stem-picker fix to emit 'inine' (currently emit 'inane')."""

    def test_aluminine(self):
        ok, name = _name_ok("[Al]1=CC=CC=C1")
        assert ok and name == "aluminine", (ok, name)
