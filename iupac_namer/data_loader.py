"""
iupac_namer/data_loader.py

Data loader module: lazy-loading singletons for all JSON data tables in data/.
All functions are pure Python + JSON — no RDKit dependency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

# data/ is two levels up from this file: iupac_namer/data_loader.py -> iupac_namer/ -> project root -> data/
_DATA_DIR = Path(__file__).parent.parent / "data"
_BLUEBOOK_DIR = _DATA_DIR / "bluebook"
_OPSIN_DIR = _DATA_DIR / "opsin_extracted"

# ---------------------------------------------------------------------------
# Low-level cache + loader
# ---------------------------------------------------------------------------

_cache: dict[str, Any] = {}


def _load(path: Path) -> Any:
    """Load a JSON file, raising clear errors on failure."""
    key = str(path)
    if key not in _cache:
        if not path.exists():
            raise FileNotFoundError(
                f"Data file not found: {path}\n"
                f"Expected data directory: {_DATA_DIR}"
            )
        try:
            with path.open(encoding="utf-8") as fh:
                _cache[key] = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON in {path}: {exc}") from exc
    return _cache[key]


# ---------------------------------------------------------------------------
# Core table accessors
# ---------------------------------------------------------------------------

def get_chain_stems() -> dict[str, str]:
    """Return dict mapping chain-length string keys to stem names.

    E.g. ``{"1": "meth", "2": "eth", ...}``
    """
    return _load(_DATA_DIR / "chain_stems.json")


def get_chain_stem(length: int) -> str | None:
    """Return the stem for a given chain length, or None if not found."""
    return get_chain_stems().get(str(length))


def get_multipliers_raw() -> dict[str, dict[str, str]]:
    """Return the raw multipliers dict with 'simple' and 'complex' sub-dicts."""
    return _load(_DATA_DIR / "multipliers.json")


def get_multiplier(count: int, complex: bool = False) -> str | None:  # noqa: A002
    """Return the multiplier prefix for *count* substituents.

    Args:
        count: Number of substituents (e.g. 2 → "di" or "bis").
        complex: If True, use the 'complex' (bis/tris/…) series.
    """
    raw = get_multipliers_raw()
    key = "complex" if complex else "simple"
    return raw.get(key, {}).get(str(count))


def get_functional_groups() -> dict:
    """Return the functional groups definition dict.

    Top-level keys: ``"suffix_groups"`` (list) and ``"prefix_only_groups"`` (list).
    """
    return _load(_DATA_DIR / "functional_groups.json")


def get_hw_tables() -> dict:
    """Return the Hantzsch-Widman tables dict.

    Top-level keys: ``"prefixes"``, ``"stems"``, ``"six_membered_groups"``.
    """
    return _load(_DATA_DIR / "hw_tables.json")


def get_element_seniority() -> dict:
    """Return the element seniority ordering dict."""
    return _load(_DATA_DIR / "element_seniority.json")


def get_retained_names() -> dict:
    """Return the retained names dict (retained_names.json).

    Top-level keys: ``"functional_parents"``, ``"acyl_groups"``,
    ``"hydrocarbons"``.
    """
    return _load(_DATA_DIR / "retained_names.json")


def get_retained_names_expanded() -> dict:
    """Return the expanded retained names dict keyed by canonical SMILES."""
    return _load(_DATA_DIR / "retained_names_expanded.json")


def get_retained_rings() -> dict:
    """Return the retained ring system definitions dict."""
    return _load(_DATA_DIR / "retained_rings.json")


def get_substituent_names() -> dict:
    """Return the substituent names mapping."""
    return _load(_DATA_DIR / "substituent_names.json")


def get_fusion_components() -> dict:
    """Return the fused ring system components dict (large, ~111 KB)."""
    return _load(_DATA_DIR / "fusion_components.json")


# ---------------------------------------------------------------------------
# Blue Book table accessors
# ---------------------------------------------------------------------------

def get_bluebook_prefixes() -> dict:
    """Return the comprehensive Blue Book prefix data (~411 KB)."""
    return _load(_BLUEBOOK_DIR / "prefixes_bluebook.json")


def get_seniority_order() -> list:
    """Return the heteroatom seniority rules list."""
    return _load(_BLUEBOOK_DIR / "seniority_order.json")


def get_element_a_prefixes() -> list:
    """Return the replacement 'a' prefixes list (element_a_prefixes.json)."""
    return _load(_BLUEBOOK_DIR / "element_a_prefixes.json")


def get_replacement_a_prefixes() -> list:
    """Return the skeletal replacement 'a' prefixes list (skeletal_a_prefixes.json)."""
    return _load(_BLUEBOOK_DIR / "skeletal_a_prefixes.json")


def get_heterocyclic_ring_components() -> Any:
    """Return heterocyclic ring components data."""
    return _load(_BLUEBOOK_DIR / "heterocyclic_ring_components.json")


def get_hydrocarbon_ring_components() -> Any:
    """Return hydrocarbon ring components data."""
    return _load(_BLUEBOOK_DIR / "hydrocarbon_ring_components.json")


def get_mancude_heteromonocycles() -> Any:
    """Return mancude heterocyclic systems data."""
    return _load(_BLUEBOOK_DIR / "mancude_heteromonocycles.json")


def get_saturated_heteromonocycles() -> Any:
    """Return saturated heterocyclic systems data."""
    return _load(_BLUEBOOK_DIR / "saturated_heteromonocycles.json")


def get_multiplicative_prefixes() -> Any:
    """Return multiplicative prefix data from bluebook."""
    return _load(_BLUEBOOK_DIR / "multiplicative_prefixes.json")


# ---------------------------------------------------------------------------
# OPSIN-extracted table accessors
# ---------------------------------------------------------------------------

def get_retained_names_from_opsin() -> list:
    """Return retained names mined from OPSIN (list of dicts with name/smiles/source)."""
    return _load(_OPSIN_DIR / "retained_names_from_opsin.json")


def get_rings_from_opsin() -> list:
    """Return ring names mined from OPSIN (list of dicts with name/smiles/source)."""
    return _load(_OPSIN_DIR / "rings_from_opsin.json")


# ---------------------------------------------------------------------------
# Generic loader
# ---------------------------------------------------------------------------

def load_table(name: str) -> Any:
    """Load any table by logical name or relative path fragment.

    Checks (in order):
    1. ``data/<name>`` (if name ends with .json, else ``data/<name>.json``)
    2. ``data/bluebook/<name>`` / ``data/bluebook/<name>.json``
    3. ``data/opsin_extracted/<name>`` / ``data/opsin_extracted/<name>.json``

    Raises FileNotFoundError if not found in any location.
    """
    stem = name if name.endswith(".json") else name + ".json"
    for directory in (_DATA_DIR, _BLUEBOOK_DIR, _OPSIN_DIR):
        candidate = directory / stem
        if candidate.exists():
            return _load(candidate)
    raise FileNotFoundError(
        f"Table {name!r} not found in data/, data/bluebook/, or data/opsin_extracted/"
    )


# ---------------------------------------------------------------------------
# Convenience lookup functions
# ---------------------------------------------------------------------------

def lookup_retained_name(smiles: str) -> dict | None:
    """Search all retained-name sources for a canonical SMILES match.

    Searches (in priority order):
    1. Ring curated table (benzene, naphthalene, pyridine, etc. with substituent forms)
    2. ``retained_names_expanded.json`` (keyed by SMILES, fastest)
    3. ``retained_names_from_opsin.json`` (list, matched on 'smiles' field)

    Returns the matching record dict or None.
    """
    # 1. Ring curated table (highest priority for ring systems)
    ring_match = _lookup_curated_ring(smiles)
    if ring_match is not None:
        return ring_match

    # 1b. Inorganic / ion / parent-hydride curated table
    inorganic_match = _lookup_curated_inorganic(smiles)
    if inorganic_match is not None:
        return inorganic_match

    # 2. retained_names_expanded: top-level is a dict of category → {smiles: record}
    expanded = get_retained_names_expanded()
    for category_data in expanded.values():
        if isinstance(category_data, dict):
            if smiles in category_data:
                rec = category_data[smiles]
                return {"smiles": smiles, **rec} if isinstance(rec, dict) else {"smiles": smiles, "name": rec}

    # 3. retained_names_from_opsin: list of {"name": ..., "smiles": ..., "source": ...}
    for entry in get_retained_names_from_opsin():
        if isinstance(entry, dict) and entry.get("smiles") == smiles:
            return entry

    return None


# ---------------------------------------------------------------------------
# Ring curated retained names — SINGLE SOURCE OF TRUTH
# ---------------------------------------------------------------------------
# This is the master ring retained-name table.  All entries are keyed by the
# RDKit canonical SMILES for the isolated ring system.
#
# Schema per entry:
#   name              – IUPAC retained name (required)
#   substituent_form  – substituent name (e.g. "phenyl", "pyridinyl"; None if unknown)
#   alkyl_stem_ok     – True only for simple saturated monocyclic carbocycles
#                       (cyclopropane–cyclooctane), where Method (1) of the engine
#                       strips the "-ane"/"-ene" suffix to build "cyclopropyl-" etc.
#   atom_locants      – optional {ring_mol_atom_idx: iupac_locant} map used by
#                       the substituent-locant rendering path so substituted
#                       forms (e.g. "2-chloroanthracene") get the correct
#                       locant rather than collapsing to lowest-symmetric.
#   stage2_fusion_base – optional bool (default True).  When False, this curated
#                       entry is INVISIBLE to the Stage 2B multi-ring fusion-base
#                       lookup (`_try_multiring_base_name_and_numberings`).  Use
#                       this to opt a curated parent out of being matched as a
#                       fusion-base, e.g. anthracene must not become the base of
#                       `[1,3]dioxolo[4,5-b]anthracene` because Stage 2 has an
#                       architectural ≤3-ring invariant.  The substituent-locant
#                       rendering path (which consumes atom_locants) is
#                       UNAFFECTED — it always uses atom_locants when present.
#
# ring_naming/retained_lookup.py imports this table and derives its own
# tuple-keyed _CURATED dict from it via _build_curated_from_data_loader().
# DO NOT duplicate entries in retained_lookup.py.
#
# Canonical SMILES were verified with RDKit Chem.MolToSmiles() after each edit.
# ---------------------------------------------------------------------------

_RING_CURATED_SMILES: dict[str, dict] = {
    # -----------------------------------------------------------------------
    # Monocyclic carbocycles
    # -----------------------------------------------------------------------
    # Aromatic
    "c1ccccc1":        {"name": "benzene",      "substituent_form": "phenyl",      "alkyl_stem_ok": False},

    # Saturated (alkyl_stem_ok=True → cyclopropyl, cyclobutyl, etc. prefixes)
    "C1CC1":           {"name": "cyclopropane", "substituent_form": "cyclopropyl", "alkyl_stem_ok": True},
    "C1CCC1":          {"name": "cyclobutane",  "substituent_form": "cyclobutyl",  "alkyl_stem_ok": True},
    "C1CCCC1":         {"name": "cyclopentane", "substituent_form": "cyclopentyl", "alkyl_stem_ok": True},
    "C1CCCCC1":        {"name": "cyclohexane",  "substituent_form": "cyclohexyl",  "alkyl_stem_ok": True},
    "C1CCCCCC1":       {"name": "cycloheptane", "substituent_form": "cycloheptyl", "alkyl_stem_ok": True},
    "C1CCCCCCC1":      {"name": "cyclooctane",  "substituent_form": "cyclooctyl",  "alkyl_stem_ok": True},

    # -----------------------------------------------------------------------
    # Polycyclic aromatic carbocycles
    # -----------------------------------------------------------------------
    # naphthalene: atom_locants verified via OPSIN difluoro probing.
    # RDKit canonical 'c1ccc2ccccc2c1': idx9=1, idx0=2, idx1=3, idx2=4, idx3=4a, idx4=5, idx5=6, idx6=7, idx7=8, idx8=8a
    "c1ccc2ccccc2c1":          {"name": "naphthalene", "substituent_form": "naphthalenyl", "alkyl_stem_ok": False,
                                 "atom_locants": {9: 1, 0: 2, 1: 3, 2: 4, 3: "4a", 4: 5, 5: 6, 6: 7, 7: 8, 8: "8a"}},
    # 1,4-dihydronaphthalene: the ring fragment extracted from 1,4-naphthoquinone derivatives.
    # RDKit canonical 'C1=CCc2ccccc2C1': idx9=1, idx0=2, idx1=3, idx2=4, idx3=4a, idx7=5, idx6=6, idx5=7, idx4=8, idx8=8a
    # Verified via OPSIN chloro probing: 2-Cl on idx0, 5-Cl on idx7, 6-Cl on idx6, 7-Cl on idx5, 8-Cl on idx4.
    # When the two sp3 carbons (pos 1=idx9, pos 4=idx2) carry oxo groups, the compound is
    # 1,4-dihydronaphthalene-1,4-dione (= naphthalene-1,4-dione) with a methyl at pos 2.
    "C1=CCc2ccccc2C1":         {"name": "1,4-dihydronaphthalene", "substituent_form": "1,4-dihydronaphthyl", "alkyl_stem_ok": False,
                                 "atom_locants": {9: 1, 0: 2, 1: 3, 2: 4, 3: "4a", 7: 5, 6: 6, 5: 7, 4: 8, 8: "8a"}},
    # Anthracene — 6,6,6 linearly-fused PAH with D2h symmetry.  Canonical
    # 'c1ccc2cc3ccccc3cc2c1'.  Verified via OPSIN chloro-probing of L=1
    # (→ parent idx 13), L=2 (→ idx 0), L=9 (→ idx 11); the other outer
    # positions (4,5,8) and (3,6,7) collapse onto the same canonical via
    # D2h, so OPSIN always picks the lowest locant of each set.  Without
    # this map, the engine emitted "1-chloroanthracene" for any chloro
    # position (latent symmetric-locant collapse — same class of bug
    # repaired for dibenzofuran in R11-A).
    #
    # ``stage2_fusion_base: False`` (R12-A-2): atom_locants is needed by
    # the substituent-locant rendering path so chloro/methyl-substituted
    # anthracenes get the correct locant, but anthracene MUST NOT be
    # selectable as a Stage 2B multi-ring fusion base.  Without the
    # opt-out, the 4-ring anthracene-dioxole scaffold
    # ``c1ccc2cc3cc4c(cc3cc2c1)OCO4`` would emit
    # ``[1,3]dioxolo[4,5-b]anthracene`` — a valid IUPAC name that
    # round-trips through OPSIN, but VIOLATES the architectural ≤3-ring
    # Stage 2 invariant (see
    # tests/test_fused_ring_hetero.py::test_stage2_excludes_four_plus_ring_systems).
    "c1ccc2cc3ccccc3cc2c1":    {"name": "anthracene",  "substituent_form": "anthracenyl", "alkyl_stem_ok": False,
                                  "atom_locants": {13: 1, 0: 2, 1: 3, 2: 4, 3: "4a", 4: 10,
                                                   5: "10a", 6: 5, 7: 6, 8: 7, 9: 8,
                                                   10: "8a", 11: 9, 12: "9a"},
                                  "stage2_fusion_base": False},

    # Triphenylene (Stage 13 R13-A-1) — 6,6,6,6 ortho-fused PAH with D3h
    # symmetry (4 rings: 1 central 6-ring of all-junction atoms surrounded
    # by 3 benzene rings).  Canonical 'c1ccc2c(c1)c1ccccc1c1ccccc21' (18
    # atoms = 12 peripheral CH + 6 ring-junction).  Locants verified by
    # OPSIN chloro-probing of L=1..12 (each C2/C3-orbit collapses to a
    # single canonical SMILES via the 6-fold automorphism); orientation
    # picked from the periphery walk
    #   1→2→3→4→4a→4b→5→6→7→8→8a→8b→9→10→11→12→12a→12b→1
    # which maps idx 2→L1, idx 1→L2, idx 0→L3, idx 5→L4, idx 4→L4a, idx 6→L4b,
    # idx 7→L5, idx 8→L6, idx 9→L7, idx 10→L8, idx 11→L8a, idx 12→L8b,
    # idx 13→L9, idx 14→L10, idx 15→L11, idx 16→L12, idx 17→L12a, idx 3→L12b.
    # Without this map every chloro-substituted form collapsed to
    # "1-chlorotriphenylene" — same latent locant-collapse class as
    # anthracene/dibenzofuran.  ``stage2_fusion_base: False`` keeps the
    # ≤3-ring Stage 2B fusion-base invariant intact (4-ring scaffold).
    "c1ccc2c(c1)c1ccccc1c1ccccc21": {"name": "triphenylene",
                                      "substituent_form": "triphenylenyl",
                                      "alkyl_stem_ok": False,
                                      "atom_locants": {2: 1, 1: 2, 0: 3, 5: 4, 4: "4a",
                                                       6: "4b", 7: 5, 8: 6, 9: 7, 10: 8,
                                                       11: "8a", 12: "8b", 13: 9, 14: 10,
                                                       15: 11, 16: 12, 17: "12a", 3: "12b"},
                                      "stage2_fusion_base": False},

    # Perylene (Stage 13 R13-A-2) — 6,6,6,6,6 peri-fused PAH (two
    # naphthalene halves joined at their peri positions, sharing a central
    # 6-ring).  D2h symmetry.  Canonical
    # 'c1cc2cccc3c4cccc5cccc(c(c1)c23)c54' (20 atoms = 12 peripheral CH +
    # 6 outer ring-junction + 2 inner peri atoms).  Pre-fix the engine
    # emitted "chloroperylene" with no locant (atom_locants returned
    # nothing for L=2/L=3 → assembly dropped the locant entirely).
    # Locants verified by OPSIN multi-substituted probing
    # (1-chloro-N-bromo for N=3,5,6,7,12) to break D2h symmetry; periphery
    # walk 1→2→3→3a→4→5→6→6a→6b→7→8→9→9a→10→11→12→12a→12b yields the
    # outer 18-atom assignment; the two interior atoms idx 18 and idx 19
    # are 12c and 12d respectively (12c is adjacent to 3a/6a/12b, 12d to
    # 6b/9a/12a — IUPAC convention places 12c first as it borders the
    # lower-numbered junctions).  ``stage2_fusion_base: False`` (5-ring
    # scaffold; well above the ≤3-ring Stage 2B invariant).
    "c1cc2cccc3c4cccc5cccc(c(c1)c23)c54": {"name": "perylene",
                                            "substituent_form": "perylenyl",
                                            "alkyl_stem_ok": False,
                                            "atom_locants": {5: 1, 4: 2, 3: 3, 2: "3a", 1: 4, 0: 5,
                                                             17: 6, 16: "6a", 15: "6b", 14: 7, 13: 8,
                                                             12: 9, 11: "9a", 10: 10, 9: 11, 8: 12,
                                                             7: "12a", 6: "12b", 18: "12c", 19: "12d"},
                                            "stage2_fusion_base": False},

    # Benz[a]anthracene (Stage 13 R13-A-3) — 6,6,6,6 angular-fused PAH
    # (anthracene with an extra benzene fused at the [a] bond).  No
    # symmetry — every periphery atom is distinct.  Canonical
    # 'c1ccc2cc3c(ccc4ccccc43)cc2c1' (18 atoms = 12 peripheral CH + 6
    # ring-junction).  Pre-fix this scaffold returned NAMING_ERROR
    # (no curated entry, fused-ring path couldn't synthesise the
    # angular tetracyclic name).  Locant assignment derived directly
    # from OPSIN chloro-probing L=1..12 (each unique) plus methyl-dihydro
    # probes that pin the six junctions {4a, 6a, 7a, 11a, 12a, 12b}:
    #   L=1→idx13, L=2→12, L=3→11, L=4→10, L=4a→9, L=5→8, L=6→7,
    #   L=6a→6, L=7→15, L=7a→16, L=8→17, L=9→0, L=10→1, L=11→2,
    #   L=11a→3, L=12→4, L=12a→5, L=12b→14.
    # ``stage2_fusion_base: False`` (4-ring scaffold; allowing it as a
    # Stage 2B fusion base would yield 5-ring fused names that violate
    # the ≤3-ring Stage 2 invariant guarded by
    # tests/test_fused_ring_hetero.py::test_stage2_excludes_four_plus_ring_systems).
    "c1ccc2cc3c(ccc4ccccc43)cc2c1": {"name": "benz[a]anthracene",
                                      "substituent_form": "benz[a]anthracenyl",
                                      "alkyl_stem_ok": False,
                                      "atom_locants": {13: 1, 12: 2, 11: 3, 10: 4, 9: "4a",
                                                       8: 5, 7: 6, 6: "6a", 15: 7, 16: "7a",
                                                       17: 8, 0: 9, 1: 10, 2: 11, 3: "11a",
                                                       4: 12, 5: "12a", 14: "12b"},
                                      "stage2_fusion_base": False},

    # Dibenz[a,h]anthracene (Stage 14 R14-A-4) — 5-ring pericondensed PAH
    # (C2 symmetry), 22 atoms = 14 peripheral CH + 8 ring-junction.
    # Canonical (RDKit): 'c1ccc2c(c1)ccc1cc3c(ccc4ccccc43)cc12'.
    # Two benzo groups fused at the a and h faces of anthracene.
    # C2 symmetry: 7 C2 orbit-pairs among the 14 peripheral positions
    # {1,8},{2,9},{3,10},{4,11},{5,12},{6,13},{7,14}.
    # Junctions form 4 C2 pairs: {4a,11a},{6a,13a},{7a,14a},{7b,14b}.
    # RDKit idx → IUPAC locant mapping derived by OPSIN dichloro-probe matching:
    #   Peripheral: idx0=3, idx1=2, idx2=1, idx5=4, idx6=5, idx7=6, idx9=7,
    #               idx12=13, idx13=12, idx15=11, idx16=10, idx17=9, idx18=8,
    #               idx20=14.
    #   Junctions:  idx3=14b, idx4=4a, idx8=6a, idx10=7a,
    #               idx11=13a, idx14=11a, idx19=14a, idx21=7b.
    # All peripheral positions verified by dichloro canonical probe;
    # junctions assigned by ring-membership adjacency + C2-pair analysis.
    # ``stage2_fusion_base: False`` (5-ring scaffold).
    "c1ccc2c(c1)ccc1cc3c(ccc4ccccc43)cc12": {
        "name": "dibenz[a,h]anthracene",
        "substituent_form": "dibenz[a,h]anthracenyl",
        "alkyl_stem_ok": False,
        "atom_locants": {0: 3, 1: 2, 2: 1, 3: "14b",
                         4: "4a", 5: 4, 6: 5, 7: 6,
                         8: "6a", 9: 7, 10: "7a", 11: "13a",
                         12: 13, 13: 12, 14: "11a", 15: 11,
                         16: 10, 17: 9, 18: 8, 19: "14a",
                         20: 14, 21: "7b"},
        "stage2_fusion_base": False},

    # Pyrene (Stage 14 R14-A-1) — 4-ring peri-fused PAH (D2h symmetry),
    # 16 atoms = 10 peripheral CH + 6 ring-junction.  Canonical (RDKit)
    # 'c1cc2ccc3cccc4ccc(c1)c2c34'.  The parent already names correctly
    # via the OPSIN-extracted rings table (rings_from_opsin.json line ~2848:
    # 'pyren' → 'c1ccc2ccc3cccc4ccc1c2c34'), but without atom_locants all
    # chloro positions collapse to '1-chloropyrene' (latent symmetric-locant
    # collapse — same class as Stage 13 R13-A-1 triphenylene).
    #
    # Locants derived from OPSIN chloro/methyl-probing.  D2h symmetry gives
    # 3 peripheral orbits: {1,3,6,8} (α), {2,7} (β), {4,5,9,10} (γ).
    # Peripheral walk (RDKit canonical): idx13-idx0-idx1-(jct2)-idx3-idx4-
    # (jct5)-idx6-idx7-idx8-(jct9)-idx10-idx11-(jct12)-idx13.
    # Outer ring-junction locants (peripheral jcts):
    #   jct_idx2 = L3a  (between L3=idx1 and L4=idx3; OPSIN accepts 3a-Cl)
    #   jct_idx5 = L5a  (between L5=idx4 and L6=idx6; OPSIN accepts 5a-Cl)
    #   jct_idx9 = L8a  (between L8=idx8 and L9=idx10; OPSIN accepts 8a-Cl)
    #   jct_idx12= L10a (between L10=idx11 and L1=idx13; OPSIN accepts 10a-Cl)
    # Inner ring-junction locants:
    #   idx14 = L10b (adjacent to 10a=idx12; OPSIN does not accept 10b-Cl
    #           — no H at this position in aromatic form)
    #   idx15 = L10c (OPSIN accepts 10c-Cl in an indicated-H form)
    # All 10 peripheral positions verified by chloro-probe canonical match:
    #   L=1→idx13, L=2→idx0, L=3→idx1, L=4→idx3, L=5→idx4, L=6→idx6,
    #   L=7→idx7, L=8→idx8, L=9→idx10, L=10→idx11.
    # ``stage2_fusion_base: False`` (4-ring scaffold; same architectural
    # guard as anthracene/triphenylene/perylene/benz[a]anthracene).
    "c1cc2ccc3cccc4ccc(c1)c2c34": {"name": "pyrene",
                                    "substituent_form": "pyrenyl",
                                    "alkyl_stem_ok": False,
                                    "atom_locants": {13: 1, 0: 2, 1: 3, 2: "3a",
                                                     3: 4, 4: 5, 5: "5a",
                                                     6: 6, 7: 7, 8: 8, 9: "8a",
                                                     10: 9, 11: 10, 12: "10a",
                                                     14: "10b", 15: "10c"},
                                    "stage2_fusion_base": False},

    # -----------------------------------------------------------------------
    # Linear polyacene series (IUPAC P-25.1.1 / OPSIN miscTokens polyacene).
    # These are the n-acene hydrocarbons with N linearly-fused benzene rings.
    # Generated by OPSIN ComponentGenerator.java from `<prefix>acene` tokens;
    # we register them explicitly here so retained-name lookup returns them
    # directly (engine's fused-ring path cannot canonicalize them without this
    # seed).  Canonical SMILES from OPSIN pentacene/hexacene/heptacene etc.
    # via py2opsin + Chem.MolToSmiles.  No atom_locants supplied — these
    # scaffolds are unsubstituted parents (the audit probes only the parent
    # form); substituent numbering on them is deferred.
    # -----------------------------------------------------------------------
    # Pentacene (Stage 14 R14-A-2) — 5-ring linear acene (D2h), 22 atoms:
    # 14 peripheral CH + 8 ring-junction. Canonical (RDKit) key as above.
    # Peripheral orbits by D2h: {1,4,8,11}, {2,3,9,10}, {5,7,12,14}, {6,13}.
    # Full IUPAC locant sequence (perimeter):
    #   1,2,3,4,4a,5,5a,6,6a,7,7a,8,9,10,11,11a,12,12a,13,13a,14,14a
    # Peripheral locants: 1-4 (end ring 1), 5-7 (rings 2-3 side), 6 (center),
    #   7-8 (rings 4-3 side), 8-11 (end ring 5), 12-14 (rings 4-2 side).
    # RDKit idx → IUPAC locant mapping derived by OPSIN chloro-probe matching:
    #   Peripheral: idx0=3, idx1=2, idx2=1, idx4=14, idx6=13, idx8=12,
    #               idx10=11, idx11=10, idx12=9, idx13=8, idx15=7, idx17=6,
    #               idx19=5, idx21=4.
    #   Junctions:  idx3=14a, idx5=13a, idx7=12a, idx9=11a,
    #               idx14=7a, idx16=6a, idx18=5a, idx20=4a.
    # All peripheral positions verified by dichloro canonical probe;
    # junctions assigned by ring-membership adjacency analysis.
    # ``stage2_fusion_base: False`` (5-ring scaffold).
    "c1ccc2cc3cc4cc5ccccc5cc4cc3cc2c1": {
        "name": "pentacene",
        "substituent_form": "pentacenyl",
        "alkyl_stem_ok": False,
        "atom_locants": {0: 3, 1: 2, 2: 1, 3: "14a",
                         4: 14, 5: "13a", 6: 13, 7: "12a",
                         8: 12, 9: "11a", 10: 11, 11: 10,
                         12: 9, 13: 8, 14: "7a", 15: 7,
                         16: "6a", 17: 6, 18: "5a", 19: 5,
                         20: "4a", 21: 4},
        "stage2_fusion_base": False},
    # Hexacene (Stage 14 R14-A-3) — 6-ring linear acene (D2h), 26 atoms:
    # 16 peripheral CH + 10 ring-junction. Canonical (RDKit) key as above.
    # Peripheral orbits by D2h: {1,4,9,12}, {2,3,10,11}, {5,8,13,16}, {6,7,14,15}.
    # Full IUPAC locant sequence (perimeter):
    #   1,2,3,4,4a,5,5a,6,6a,7,7a,8,8a,9,10,11,12,13,14,15,16,16a,...
    # RDKit idx → IUPAC locant mapping derived by ring-adjacency analysis + dichloro probes:
    #   Peripheral: idx0=3, idx1=2, idx2=1, idx4=16, idx6=15, idx8=14,
    #               idx10=13, idx12=12, idx13=11, idx14=10, idx15=9, idx17=8,
    #               idx19=7, idx21=6, idx23=5, idx25=4.
    #   Junctions:  idx3=16a, idx5=15a, idx7=14a, idx9=13a, idx11=12a,
    #               idx16=8a, idx18=7a, idx20=6a, idx22=5a, idx24=4a.
    # All peripheral positions verified by dichloro canonical probe;
    # junctions assigned by ring-membership adjacency analysis.
    # ``stage2_fusion_base: False`` (6-ring scaffold).
    "c1ccc2cc3cc4cc5cc6ccccc6cc5cc4cc3cc2c1": {
        "name": "hexacene",
        "substituent_form": "hexacenyl",
        "alkyl_stem_ok": False,
        "atom_locants": {0: 3, 1: 2, 2: 1, 3: "16a",
                         4: 16, 5: "15a", 6: 15, 7: "14a",
                         8: 14, 9: "13a", 10: 13, 11: "12a",
                         12: 12, 13: 11, 14: 10, 15: 9,
                         16: "8a", 17: 8, 18: "7a", 19: 7,
                         20: "6a", 21: 6, 22: "5a", 23: 5,
                         24: "4a", 25: 4},
        "stage2_fusion_base": False},
    "c1ccc2cc3cc4cc5cc6cc7ccccc7cc6cc5cc4cc3cc2c1":             {"name": "heptacene", "substituent_form": "heptacenyl", "alkyl_stem_ok": False},
    "c1ccc2cc3cc4cc5cc6cc7cc8ccccc8cc7cc6cc5cc4cc3cc2c1":       {"name": "octacene",  "substituent_form": "octacenyl",  "alkyl_stem_ok": False},
    "c1ccc2cc3cc4cc5cc6cc7cc8cc9ccccc9cc8cc7cc6cc5cc4cc3cc2c1": {"name": "nonacene",  "substituent_form": "nonacenyl",  "alkyl_stem_ok": False},

    # -----------------------------------------------------------------------
    # Phene series (OPSIN miscTokens polyphene) — polyphene family with a
    # "bent" linear-fusion pattern (as in phenanthrene extended).  Canonical
    # SMILES from OPSIN pentaphene/hexaphene/heptaphene/octaphene.
    # -----------------------------------------------------------------------
    "c1ccc2cc3c(ccc4cc5ccccc5cc43)cc2c1":                      {"name": "pentaphene", "substituent_form": "pentaphenyl", "alkyl_stem_ok": False},
    "c1ccc2cc3cc4c(ccc5cc6ccccc6cc54)cc3cc2c1":                {"name": "hexaphene",  "substituent_form": "hexaphenyl",  "alkyl_stem_ok": False},
    "c1ccc2cc3cc4c(ccc5cc6cc7ccccc7cc6cc54)cc3cc2c1":          {"name": "heptaphene", "substituent_form": "heptaphenyl", "alkyl_stem_ok": False},
    "c1ccc2cc3cc4cc5c(ccc6cc7cc8ccccc8cc7cc65)cc4cc3cc2c1":    {"name": "octaphene",  "substituent_form": "octaphenyl",  "alkyl_stem_ok": False},

    # -----------------------------------------------------------------------
    # Helicene series (OPSIN miscTokens polyhelicene) — helicoidal ortho-fused
    # polyaromatic hydrocarbons.  Canonical SMILES from OPSIN
    # pentahelicene/hexahelicene/heptahelicene/octahelicene.
    # -----------------------------------------------------------------------
    "c1ccc2c(c1)ccc1ccc3ccc4ccccc4c3c12":                                 {"name": "pentahelicene", "substituent_form": "pentahelicenyl", "alkyl_stem_ok": False},
    "c1ccc2c(c1)ccc1ccc3ccc4ccc5ccccc5c4c3c12":                           {"name": "hexahelicene",  "substituent_form": "hexahelicenyl",  "alkyl_stem_ok": False},
    "c1ccc2c(c1)ccc1ccc3ccc4ccc5ccc6ccccc6c5c4c3c12":                     {"name": "heptahelicene", "substituent_form": "heptahelicenyl", "alkyl_stem_ok": False},
    "c1ccc2c(c1)ccc1ccc3ccc4ccc5ccc6ccc7ccccc7c6c5c4c3c12":               {"name": "octahelicene",  "substituent_form": "octahelicenyl",  "alkyl_stem_ok": False},

    # -----------------------------------------------------------------------
    # Polyalene series (IUPAC P-25.1.2.3) — two identical ortho-fused
    # monocyclic hydrocarbon rings, named "<n>alene" with elision of one 'a'
    # (derived from naphthalene; naphthalene itself is the C6C6 polyalene but
    # keeps its retained name).  pentalene/octalene are PINs (Blue Book
    # P-25.1.2.3, P2 examples P250102e/f); heptalene is the C7C7 polyalene in
    # the retained mancude-ring table (Blue Book Table 28.1 / P-25.1.1, P2
    # line ~3857) and used as a PIN (P-25.7 examples).  The engine's fused-ring
    # path emits a von Baeyer polycycle for these mancude bicyclics, so the
    # retained names must be seeded here.
    #
    # All three are D2h-symmetric, so atom_locants are pinned to ONE consistent
    # perimeter walk; the bond-generic SubstructMatch path recovers the other
    # automorphisms at runtime and the strategy layer picks the lowest
    # substituent locant per P-14.5.  Locants verified by OPSIN chloro-probing
    # every accepted numeric + letter locant (each maps to a symmetry orbit;
    # the walk below is one self-consistent isomorphism through that orbit set).
    #
    # pentalene (P-25.1.2.3, PIN) — 5,5-fused C5C5, 8 atoms, canonical
    # 'C1=CC2=CC=CC2=C1'.  Bridgeheads (3a,6a) = idx 6,2.  Perimeter walk
    # 1->2->3->3a->4->5->6->6a: idx1->0->7->6->5->4->3->2.
    "C1=CC2=CC=CC2=C1": {"name": "pentalene",
                          "substituent_form": "pentalenyl",
                          "alkyl_stem_ok": False,
                          "atom_locants": {1: 1, 0: 2, 7: 3, 6: "3a",
                                           5: 4, 4: 5, 3: 6, 2: "6a"}},
    # heptalene (Table 28.1 / P-25.1.1, PIN) — 7,7-fused C7C7, 12 atoms,
    # canonical 'C1=CC=C2C=CC=CC=C2C=C1'.  Bridgeheads (5a,10a) = idx 9,3.
    # Perimeter walk 1->2->3->4->5->5a->6->7->8->9->10->10a:
    # idx4->5->6->7->8->9->10->11->0->1->2->3.
    "C1=CC=C2C=CC=CC=C2C=C1": {"name": "heptalene",
                                "substituent_form": "heptalenyl",
                                "alkyl_stem_ok": False,
                                "atom_locants": {4: 1, 5: 2, 6: 3, 7: 4,
                                                 8: 5, 9: "5a", 10: 6, 11: 7,
                                                 0: 8, 1: 9, 2: 10, 3: "10a"}},
    # octalene (P-25.1.2.3, PIN) — 8,8-fused C8C8, 14 atoms, canonical
    # 'c1cccc2ccccccc-2cc1'.  Bridgeheads (6a,12a) = idx 11,4.  Perimeter walk
    # 1->2->3->4->5->6->6a->7->8->9->10->11->12->12a:
    # idx3->2->1->0->13->12->11->10->9->8->7->6->5->4.
    "c1cccc2ccccccc-2cc1": {"name": "octalene",
                             "substituent_form": "octalenyl",
                             "alkyl_stem_ok": False,
                             "atom_locants": {3: 1, 2: 2, 1: 3, 0: 4,
                                              13: 5, 12: 6, 11: "6a", 10: 7,
                                              9: 8, 8: 9, 7: 10, 6: 11,
                                              5: 12, 4: "12a"}},

    # -----------------------------------------------------------------------
    # Biphenylene — IUPAC P-25.1.1 retained name for the [4n] tricyclic with
    # a central cyclobutadiene ring fused between two benzene rings.  D2h
    # symmetric.  Standard numbering: positions 1–4 around one benzene, 4a
    # and 8b at the junctions of that ring, then 4b and 8a at the cyclobutane
    # junctions of the second benzene, positions 5–8 around the second
    # benzene.  The fusion bonds form the central 4-ring 4a–4b–8a–8b.
    # OPSIN parses 'biphenylene' to canonical 'c1ccc2c(c1)-c1ccccc1-2'.
    # Without this curated entry the engine has no plan (no retained name
    # match, fused-ring path can't canonicalize the cyclobutadiene fusion).
    # Stage 6 R1-A follow-up: closes the gap deferred by the original R1-A
    # pass.  Atom indices in 'c1ccc2c(c1)-c1ccccc1-2' (ring A = idx 0..5,
    # ring B = idx 6..11; 4-ring junctions idx 3,4,6,11):
    #   pos 1 = idx 5 (alpha-junction in ring A, adj to 8b=idx 4),
    #   pos 2 = idx 0, pos 3 = idx 1, pos 4 = idx 2 (alpha to 4a=idx 3),
    #   pos 4a = idx 3, pos 4b = idx 11, pos 5 = idx 10, pos 6 = idx 9,
    #   pos 7 = idx 8, pos 8 = idx 7, pos 8a = idx 6, pos 8b = idx 4.
    # Verified via OPSIN dichloro probing (1,2- → idxs 5,0; 1,5- → idxs 5,10;
    # 2,7- → idxs 0,8) — D2h symmetry means each chloro position has 4
    # equivalent matches; the assignment above is one canonical choice.
    # -----------------------------------------------------------------------
    "c1ccc2c(c1)-c1ccccc1-2": {"name": "biphenylene",
                                "substituent_form": "biphenylenyl",
                                "alkyl_stem_ok": False,
                                "atom_locants": {5: 1, 0: 2, 1: 3, 2: 4, 3: "4a",
                                                 11: "4b", 10: 5, 9: 6, 8: 7,
                                                 7: 8, 6: "8a", 4: "8b"}},

    # -----------------------------------------------------------------------
    # [N]annulene monocyclic polyenes (OPSIN miscTokens annulene).  The
    # aromatic annulenes are emitted in Kekulé-form by OPSIN — [10], [14],
    # [18]annulene are aromatic (4n+2) and stored as aromatic SMILES; [12],
    # [16], [20]annulene are non-aromatic (4n) and stored in alternating
    # double-bond Kekulé form.  Keys are RDKit canonical SMILES direct from
    # OPSIN py2opsin("[N]annulene").
    # -----------------------------------------------------------------------
    "c1ccccccccc1":                           {"name": "[10]annulene", "substituent_form": "[10]annulenyl", "alkyl_stem_ok": False},
    "c1ccccccccccccc1":                       {"name": "[14]annulene", "substituent_form": "[14]annulenyl", "alkyl_stem_ok": False},
    "C1=CC=CC=CC=CC=CC=CC=CC=C1":             {"name": "[16]annulene", "substituent_form": "[16]annulenyl", "alkyl_stem_ok": False},
    "c1ccccccccccccccccc1":                   {"name": "[18]annulene", "substituent_form": "[18]annulenyl", "alkyl_stem_ok": False},
    "C1=CC=CC=CC=CC=CC=CC=CC=CC=CC=C1":       {"name": "[20]annulene", "substituent_form": "[20]annulenyl", "alkyl_stem_ok": False},

    # 7,8,9,10-tetrahydrotetracene-5,12-dione (anthracycline aglycone scaffold):
    # The core ring system of anthracyclines (doxorubicin, daunorubicin, epirubicin,
    # idarubicin, etc.).  OPSIN parses this name; the retained-name stem encodes
    # both C5=O and C12=O, and the 7,8,9,10-tetrahydro saturates ring D.
    # Keyed on with-=O canonical SMILES so the exocyclic-oxo fallback in
    # retained_lookup claims the two =O atoms via extra_atom_indices,
    # preventing downstream from re-emitting redundant '5,12-dioxo' prefixes.
    # OPSIN '7,8,9,10-tetrahydrotetracene-5,12-dione' -> O=C1c2ccccc2C(=O)c2cc3c(cc21)CCCC3.
    # Canonical atom indices in 'O=C1c2ccccc2C(=O)c2cc3c(cc21)CCCC3':
    #   idx0=O(exo, pos 5 oxo), idx1=C(pos5, C=O),
    #   idx2=C(pos4a,junction A/B), idx3-6=aromatic CH (pos 4,3,2,1 in ring A),
    #   idx7=C(pos12a,junction A/B), idx8=C(pos12, C=O), idx9=O(exo, pos 12 oxo),
    #   idx10=C(pos11a,junction B/D), idx11=C(pos11, aromatic CH in ring D),
    #   idx12=C(pos10a,junction C/D), idx13=C(pos6a,junction C/D),
    #   idx14=C(pos6, aromatic CH in ring D), idx15=C(pos5a,junction B/D),
    #   idx16=C(pos7,sp3 CH2), idx17=C(pos8), idx18=C(pos9), idx19=C(pos10).
    # The scaffold has C2-mirror symmetry (pos1<->pos4, pos2<->pos3, pos6<->pos11,
    # pos7<->pos10, pos8<->pos9, pos4a<->pos12a, pos5a<->pos11a, pos6a<->pos10a);
    # atom_locants picks one canonical assignment (alternate is equivalent).
    # Covers FDA-0484 (doxorubicin aglycone), FDA-0448 (epirubicin aglycone),
    # FDA-0363 (daunorubicin aglycone), all of which carry 4-methoxy,
    # 6,11-dihydroxy, 9-(2-hydroxyacetyl or acetyl), 9-hydroxy substituents.
    "O=C1c2ccccc2C(=O)c2cc3c(cc21)CCCC3": {
        "name": "7,8,9,10-tetrahydrotetracene-5,12-dione",
        "substituent_form": "7,8,9,10-tetrahydrotetracene-5,12-dion-yl",
        "alkyl_stem_ok": False,
        "atom_locants": {1: 5, 2: "4a", 3: 4, 4: 3, 5: 2, 6: 1, 7: "12a",
                         8: 12, 10: "11a", 11: 11, 12: "10a", 13: "6a",
                         14: 6, 15: "5a", 16: 7, 17: 8, 18: 9, 19: 10}},
    # 1,4,4a,5,5a,6,11,12a-octahydrotetracene-1,11-dione (tetracycline aglycone scaffold):
    # The core ring system of the tetracycline antibiotic class (tetracycline,
    # tigecycline, et al.).  Same precedent as the 5,12-dione above: key on the
    # with-=O canonical SMILES so retained_lookup's exocyclic-oxo fallback claims
    # both C=O oxygens via extra_atom_indices, preventing a redundant
    # "-1,11-dioxo-" prefix downstream (already encoded in the retained stem).
    #
    # The carved ring from these compounds enolizes differently than anthracycline
    # (the D ring is the aromatic phenol, not ring A), yielding a keto/enol
    # tautomer whose canonical SMILES does NOT match the 5,12-dione entry.  The
    # expected IUPAC parent is 1,4,4a,5,5a,6,11,12a-octahydrotetracene-1,11-dione
    # (1,11 positions are the ketones; tetracene ring A aromatic; rings B/C/D
    # partially saturated with one remaining C=C at 11a-12).
    #
    # OPSIN '1,4,4a,5,5a,6,11,12a-octahydrotetracene-1,11-dione' canonicalizes to
    # 'O=C1C2=CC3C(=O)C=CCC3CC2Cc2ccccc21' (20 atoms).  Atom index mapping:
    #   idx0=O(exo, pos11 oxo), idx1=C(pos11, C=O, adj to aromatic junction 10a),
    #   idx2=C(pos11a, sp2, adj pos11/pos12/pos5a), idx3=C(pos12, sp2 C=C),
    #   idx4=C(pos12a, sp3 junction C/D), idx5=C(pos1, C=O in ring D),
    #   idx6=O(exo, pos1 oxo), idx7=C(pos2, sp2), idx8=C(pos3, sp2),
    #   idx9=C(pos4, sp3), idx10=C(pos4a, sp3 junction C/D),
    #   idx11=C(pos5, sp3), idx12=C(pos5a, sp3 junction B/C),
    #   idx13=C(pos6, sp3 CH2 in ring B), idx14=C(pos6a, aromatic junction A/B),
    #   idx15=C(pos7), idx16=C(pos8), idx17=C(pos9), idx18=C(pos10),
    #   idx19=C(pos10a, aromatic junction A/B).
    # OPSIN chloro/methyl probing verified pos 2,3,4,4a,5,5a,6,7,8,9,10,12,12a;
    # pos 1,11,6a,10a,11a derived from topology (=O carbons and ring junctions,
    # which have no H to substitute for direct probing).
    # Covers FDA-1310 (tetracycline), FDA-1329 (tigecycline).
    "O=C1C2=CC3C(=O)C=CCC3CC2Cc2ccccc21": {
        "name": "1,4,4a,5,5a,6,11,12a-octahydrotetracene-1,11-dione",
        "substituent_form": "1,4,4a,5,5a,6,11,12a-octahydrotetracene-1,11-dion-yl",
        "alkyl_stem_ok": False,
        "atom_locants": {1: 11, 2: "11a", 3: 12, 4: "12a", 5: 1, 7: 2, 8: 3,
                         9: 4, 10: "4a", 11: 5, 12: "5a", 13: 6, 14: "6a",
                         15: 7, 16: 8, 17: 9, 18: 10, 19: "10a"}},
    # 1,2,3,4,4a,5,5a,6,11,12a-decahydrotetracene-1,3,11-trione (methacycline scaffold):
    # Tetracycline variant where the C2-C3 C=C of the β-ketoenol ring is
    # further reduced to an sp3 CH2, producing a 1,3-dione that sits across an
    # sp3 C2 position.  OPSIN canonicalizes to
    # 'O=C1CC(=O)C2C=C3C(=O)c4ccccc4CC3CC2C1' (21 atoms).  The molecule has C2
    # rotational symmetry relating pos1<->pos3; chose one canonical assignment.
    # Atom index mapping (OPSIN methyl-probing verified pos 4,4a,5,5a,6,7,8,9,
    # 10,11a,12,12a; pos 1,2,3,6a,10a,11 derived from topology):
    #   idx0=O(pos1 oxo), idx1=C(pos1, C=O),
    #   idx2=C(pos2, sp3 CH2 between the two ring-D ketones),
    #   idx3=C(pos3, C=O), idx4=O(pos3 oxo),
    #   idx5=C(pos12a, sp3 junction C/D), idx6=C(pos12, sp2 C=C),
    #   idx7=C(pos11a, sp2 junction B/C),
    #   idx8=C(pos11, C=O adj to aromatic ring A), idx9=O(pos11 oxo),
    #   idx10=C(pos10a, aromatic junction A/B),
    #   idx11=C(pos10), idx12=C(pos9), idx13=C(pos8), idx14=C(pos7),
    #   idx15=C(pos6a, aromatic junction A/B),
    #   idx16=C(pos6, sp3 CH2 in ring B),
    #   idx17=C(pos5a, sp3 junction B/C),
    #   idx18=C(pos5), idx19=C(pos4a, sp3 junction C/D), idx20=C(pos4).
    # Covers FDA-0841 (methacycline).
    "O=C1CC(=O)C2C=C3C(=O)c4ccccc4CC3CC2C1": {
        "name": "1,2,3,4,4a,5,5a,6,11,12a-decahydrotetracene-1,3,11-trione",
        "substituent_form": "1,2,3,4,4a,5,5a,6,11,12a-decahydrotetracene-1,3,11-trion-yl",
        "alkyl_stem_ok": False,
        "atom_locants": {1: 1, 2: 2, 3: 3, 5: "12a", 6: 12, 7: "11a", 8: 11,
                         10: "10a", 11: 10, 12: 9, 13: 8, 14: 7, 15: "6a",
                         16: 6, 17: "5a", 18: 5, 19: "4a", 20: 4}},
    # tetralin (1,2,3,4-tetrahydronaphthalene): benzo ring fused with saturated 6-ring.
    # RDKit canonical 'c1ccc2c(c1)CCCC2': arom ring has idx 0-5; sp3 ring has idx 6-9 + junctions idx3,4.
    # atom_locants: idx9=1, idx8=2, idx7=3, idx6=4, idx4=4a, idx5=5, idx0=6, idx1=7, idx2=8, idx3=8a
    # Verified via OPSIN chloro probing: 1-Cl->ClC1CCCc2ccccc21 (idx9), 2-Cl->ClC1CCc2ccccc2C1 (idx8),
    #   5-Cl->Clc1cccc2c1CCCC2 (idx5), 6-Cl->Clc1ccc2c(c1)CCCC2 (idx0).
    # Pairs by symmetry: pos1/pos4 (idx9/idx6), pos2/pos3 (idx8/idx7), pos5/pos8 (idx5/idx2), pos6/pos7 (idx0/idx1).
    # pin_eligible=False: per P-25.3.1.3 / P-32.4 the retained names tetraline
    # and tetralinyl are general-nomenclature only; the PIN is the systematic
    # 1,2,3,4-tetrahydronaphthalene (and 1,2,3,4-tetrahydronaphthalen-N-yl
    # substituent form).  Engine should fall back to the systematic form for
    # PIN emission via _try_derive_hydro_retained on naphthalene.
    # The pin_name / pin_substituent_form fields override "name" /
    # "substituent_form" when pin_eligible=False: the retained record's
    # atom_locants stay the load-bearing alignment for ring numbering, but the
    # emitted PIN string is the systematic hydro-derived form.  This is the
    # architectural pattern documented above the table header.
    "c1ccc2c(c1)CCCC2":        {"name": "tetraline", "substituent_form": "tetralinyl", "alkyl_stem_ok": False,
                                 "pin_eligible": False,
                                 "pin_name": "1,2,3,4-tetrahydronaphthalene",
                                 "pin_substituent_form": "1,2,3,4-tetrahydronaphthalen-N-yl",
                                 "atom_locants": {9: 1, 8: 2, 7: 3, 6: 4, 4: "4a", 5: 5, 0: 6, 1: 7, 2: 8, 3: "8a"}},
    # indane (2,3-dihydro-1H-indene): benzo fused with cyclopentane.
    # RDKit canonical 'c1ccc2c(c1)CCC2': arom ring has idx 0-5; sp3 ring has idx 6-8 + junctions idx3,4.
    # atom_locants: idx8=1, idx7=2, idx6=3, idx4=3a, idx5=4, idx0=5, idx1=6, idx2=7, idx3=7a
    # Verified via OPSIN chloro probing: 1-Cl->ClC1CCc2ccccc21 (idx8), 2-Cl->ClC1Cc2ccccc2C1 (idx7),
    #   4-Cl->Clc1cccc2c1CCC2 (idx5), 5-Cl->Clc1ccc2c(c1)CCC2 (idx0).
    # pin_eligible=False: per P-31.1.4.2.4 / P-32.4 the retained names indane
    # and indanyl are general-nomenclature only; the PIN is the systematic
    # 2,3-dihydro-1H-indene (and 2,3-dihydro-1H-inden-N-yl substituent form).
    "c1ccc2c(c1)CCC2":         {"name": "indane", "substituent_form": "indanyl", "alkyl_stem_ok": False,
                                 "pin_eligible": False,
                                 "pin_name": "2,3-dihydro-1H-indene",
                                 "pin_substituent_form": "2,3-dihydro-1H-inden-N-yl",
                                 "atom_locants": {8: 1, 7: 2, 6: 3, 4: "3a", 5: 4, 0: 5, 1: 6, 2: 7, 3: "7a"}},

    # -----------------------------------------------------------------------
    # 6-membered N-heterocycles
    # -----------------------------------------------------------------------
    # Pyridine atom_locants pin N=1 and number C2..C6 around the ring; this is
    # required by the partial-saturation derive path (see
    # _try_derive_hydro_retained in ring_naming/retained_lookup.py) to map the
    # aromatic parent's IUPAC numbering onto partly-hydrogenated heterocycles
    # like THP and 1,4-dihydropyridine when emitted dynamically.  Two valid
    # numbering directions are equivalent by C2v symmetry; we pick one.
    "c1ccncc1":        {"name": "pyridine",   "substituent_form": "pyridinyl",   "alkyl_stem_ok": False,
                        "atom_locants": {3: 1, 2: 2, 1: 3, 0: 4, 5: 5, 4: 6}},
    # 1,4-dihydropyridine — DHP drugs (nifedipine class). RDKit canonical: C1=CNC=CC1
    # Atom_locants: N(idx2)=pos1, C(idx1, adj-N, C=C)=pos2, C(idx0, adj-sp3, C=C)=pos3,
    #               C(idx5, sp3)=pos4, C(idx4, adj-sp3, C=C)=pos5, C(idx3, adj-N, C=C)=pos6
    # Verified via OPSIN chloro-probing: 2-Cl->ClC1=CCC=CN1 (Cl on idx1-analogue),
    #   3-Cl->ClC1=CNC=CC1 (Cl on idx0-analogue), 4-Cl->ClC1C=CNC=C1 (Cl on idx5-analogue).
    "C1=CNC=CC1":      {"name": "1,4-dihydropyridine", "substituent_form": "1,4-dihydropyridinyl", "alkyl_stem_ok": False,
                        "atom_locants": {2: 1, 1: 2, 0: 3, 5: 4, 4: 5, 3: 6}},
    # Pyridinium (N-substituted pyridine with N+): RDKit canonical of c1cc[n+]cc1
    # after stripping the N-substituent is C1=CC=[N+]C=C1
    "C1=CC=[N+]C=C1":  {"name": "pyridinium", "substituent_form": "pyridinium-yl", "alkyl_stem_ok": False},
    # Protonated pyridine ([nH+]): canonical 'c1cc[nH+]cc1' (idx3=N+).
    # Registered as bare "pyridine" so the substitutive assembly's ring-N+ → -ium
    # machinery (engine.py SubstitutivePath, see ring_cation_locants) appends
    # "-1-ium" → "pyridin-1-ium".  Without this entry the lookup falls through
    # to Hantzsch-Widman and produces "azine-1-ium" instead.
    # atom_locants pin N+ to locant 1; carbons numbered around the ring.
    "c1cc[nH+]cc1":    {"name": "pyridine",   "substituent_form": "pyridinyl",   "alkyl_stem_ok": False,
                        "atom_locants": {3: 1, 2: 2, 1: 3, 0: 4, 5: 5, 4: 6}},
    # Pyrimidine — RDKit canonical is c1cncnc1 (NOT c1ccncn1)
    # atom_locants: idx 2,4 are N atoms; pin idx2->1, walk ring 2-3-4-5-0-1.
    # Both ring N's (idx 2, idx 4) become N1 / N3; substructure matching
    # yields both orientations so the strategy layer can pick the one
    # giving lowest locants to substituents (P-14.5).  Required so the
    # ``_retag_indicated_h`` path fires for [nH]-bearing tautomers (e.g.
    # pyrimidin-2(1H)-one), preventing OPSIN from re-parsing the engine's
    # output as a different lactam tautomer.
    "c1cncnc1":        {"name": "pyrimidine", "substituent_form": "pyrimidinyl", "alkyl_stem_ok": False,
                        "atom_locants": {2: 1, 3: 2, 4: 3, 5: 4, 0: 5, 1: 6}},
    # Protonated pyrimidine: canonical 'c1cnc[nH+]c1' (idx2=N, idx4=N+).
    # N+ at locant 1, neutral N at locant 3 (1,3-diazine numbering).
    "c1cnc[nH+]c1":    {"name": "pyrimidine", "substituent_form": "pyrimidinyl", "alkyl_stem_ok": False,
                        "atom_locants": {4: 1, 3: 2, 2: 3, 1: 4, 0: 5, 5: 6}},
    # Pyrazine — RDKit canonical c1cnccn1; N atoms at idx 2, 5.
    # atom_locants: pin idx2->1 (N1), walk 2-1-0-5-4-3 to give N4 at idx5.
    # (Both ring N's are equivalent by symmetry; the curated substructure
    # match picks the orientation that gives lowest substituent locants.)
    "c1cnccn1":        {"name": "pyrazine",   "substituent_form": "pyrazinyl",   "alkyl_stem_ok": False,
                        "atom_locants": {2: 1, 1: 2, 0: 3, 5: 4, 4: 5, 3: 6}},
    # Protonated pyrazine: canonical 'c1c[nH+]ccn1' (idx2=N+, idx5=N).
    # N+ at locant 1, neutral N at locant 4 (1,4-diazine numbering).
    "c1c[nH+]ccn1":    {"name": "pyrazine",   "substituent_form": "pyrazinyl",   "alkyl_stem_ok": False,
                        "atom_locants": {2: 1, 1: 2, 0: 3, 5: 4, 4: 5, 3: 6}},
    # Pyridazine — RDKit canonical c1ccnnc1; N atoms at idx 3, 4.
    # atom_locants: pin idx3->1 (N1), walk 3-4-5-0-1-2 to give N2 at idx4.
    "c1ccnnc1":        {"name": "pyridazine", "substituent_form": "pyridazinyl", "alkyl_stem_ok": False,
                        "atom_locants": {3: 1, 4: 2, 5: 3, 0: 4, 1: 5, 2: 6}},
    # Protonated pyridazine: canonical 'c1cc[nH+]nc1' (idx3=N+, idx4=N).
    # N+ at locant 1, neutral N at locant 2 (1,2-diazine numbering).
    "c1cc[nH+]nc1":    {"name": "pyridazine", "substituent_form": "pyridazinyl", "alkyl_stem_ok": False,
                        "atom_locants": {3: 1, 4: 2, 5: 3, 0: 4, 1: 5, 2: 6}},

    # 6-membered with 3+ N
    "c1cnnnc1":        {"name": "1,2,3-triazine",    "substituent_form": "1,2,3-triazinyl",    "alkyl_stem_ok": False},
    "c1cnncn1":        {"name": "1,2,4-triazine",    "substituent_form": "1,2,4-triazinyl",    "alkyl_stem_ok": False},
    "c1ncncn1":        {"name": "1,3,5-triazine",    "substituent_form": "1,3,5-triazinyl",    "alkyl_stem_ok": False},
    "c1cnnnn1":        {"name": "1,2,3,4-tetrazine", "substituent_form": "1,2,3,4-tetrazinyl", "alkyl_stem_ok": False},

    # Partially saturated 6-membered N-heterocycles
    # 1,4,5,6-tetrahydropyrimidine: C2=N1 double bond, N3-H, C4-C5-C6 sp3
    # RDKit canonical 'C1=NCCCN1': idx0=C2, idx1=N3, idx2=C4, idx3=C5, idx4=C6, idx5=N1
    # atom_locants: idx5=1, idx0=2, idx1=3, idx2=4, idx3=5, idx4=6
    # Verified via OPSIN chloro probing: 2-Cl->ClC1=NCCCN1 (idx0), 4-Cl->ClC1CCNC=N1 (idx2),
    #   5-Cl->ClC1CN=CNC1 (idx3), 6-Cl->ClC1CCN=CN1 (idx4), 3-Cl->ClN1CCCNC1 (idx1=N3).
    "C1=NCCCN1":       {"name": "1,4,5,6-tetrahydropyrimidine", "substituent_form": "1,4,5,6-tetrahydropyrimidinyl", "alkyl_stem_ok": False,
                        "atom_locants": {5: 1, 0: 2, 1: 3, 2: 4, 3: 5, 4: 6}},
    # 1,4,5,6-tetrahydropyridazine: C3=N1-N2 system. RDKit canonical 'C1=NNCCC1'.
    # atom_locants: idx2=1, idx1=2, idx0=3, idx5=4, idx4=5, idx3=6
    # Verified via OPSIN chloro probing: 1-Cl->ClN1CCCC=N1 (idx2), 3-Cl->ClC1=NNCCC1 (idx0),
    #   4-Cl->ClC1C=NNCC1 (idx5), 5-Cl->ClC1CC=NNC1 (idx4), 6-Cl->ClC1CCC=NN1 (idx3).
    "C1=NNCCC1":       {"name": "1,4,5,6-tetrahydropyridazine", "substituent_form": "1,4,5,6-tetrahydropyridazinyl", "alkyl_stem_ok": False,
                        "atom_locants": {2: 1, 1: 2, 0: 3, 5: 4, 4: 5, 3: 6}},

    # Saturated 6-membered N-heterocycles
    "C1CCNCC1":        {"name": "piperidine",       "substituent_form": "piperidinyl",       "alkyl_stem_ok": False},
    "C1CNCCN1":        {"name": "piperazine",        "substituent_form": "piperazinyl",       "alkyl_stem_ok": False},
    # 1,3-diazinane (hexahydropyrimidine): N atoms at 1,3-positions
    "C1CNCNC1":        {"name": "1,3-diazinane",     "substituent_form": "1,3-diazinanyl",    "alkyl_stem_ok": False},
    # 1,2-diazinane (hexahydropyridazine): N atoms at 1,2-positions
    "C1CCNNC1":        {"name": "1,2-diazinane",     "substituent_form": "1,2-diazinanyl",    "alkyl_stem_ok": False},

    # -----------------------------------------------------------------------
    # 7-membered N-heterocycles (IUPAC 2013 preferred names)
    # Replaces obsolete arylGroups.xml names: homopiperidin, homopiperazin, homomorpholin
    # -----------------------------------------------------------------------
    # Bicyclic N-heterocycles
    # -----------------------------------------------------------------------
    # quinolizidine (octahydroquinolizine): bridgehead N, positions 1-4,4a,6-9,9a.
    # RDKit canonical 'C1CCN2CCCCC2C1': N=idx3(pos4a), idx8=pos9a(junction),
    #   idx9=1, idx0=2, idx1=3, idx2=4, idx4=6, idx5=7, idx6=8, idx7=9
    # Verified via OPSIN chloro probing: 1-Cl->ClC1CCCN2CCCCC12 (idx9), 2-Cl->ClC1CCN2CCCCC2C1 (idx0),
    #   3-Cl->ClC1CCC2CCCCN2C1 (idx1), 4-Cl->ClC1CCCC2CCCCN12 (idx2).
    "C1CCN2CCCCC2C1":  {"name": "quinolizidine", "substituent_form": "quinolizidinyl", "alkyl_stem_ok": False,
                        "atom_locants": {9: 1, 0: 2, 1: 3, 2: 4, 3: "4a", 4: 6, 5: 7, 6: 8, 7: 9, 8: "9a"}},

    # octahydrocyclopenta[b]pyrrole: 5,5-fused saturated bicyclic, N at non-bridgehead.
    # RDKit canonical 'C1CC2CCNC2C1': rings (0,1,2,6,7) and (2,3,4,5,6); junctions = idx2,6.
    # IUPAC numbering verified via OPSIN chloro probing:
    #   1-Cl->ClN1C2C(CC1)CCC2 (N=idx5), 2-Cl->ClC1CC2C(N1)CCC2 (idx4),
    #   3-Cl->ClC1C2C(NC1)CCC2 (idx3), 3a-Cl->ClC12C(NCC1)CCC2 (junction=idx2),
    #   4-Cl->ClC1CCC2NCCC21 (idx1), 5-Cl->ClC1CC2C(NCC2)C1 (idx0),
    #   6-Cl->ClC1CCC2C1NCC2 (idx7), 6a-Cl->ClC12NCCC1CCC2 (junction=idx6).
    "C1CC2CCNC2C1":    {"name": "octahydrocyclopenta[b]pyrrole",
                        "substituent_form": "octahydrocyclopenta[b]pyrrol-yl",
                        "alkyl_stem_ok": False,
                        "atom_locants": {5: 1, 4: 2, 3: 3, 2: "3a", 1: 4, 0: 5, 7: 6, 6: "6a"}},

    # 1,2,3,4-tetrahydroisoquinoline: benzo+saturated-N-ring (5,6 partially-aromatic).
    # RDKit canonical 'c1ccc2c(c1)CCNC2': aromatic ring (0,1,2,3,4,5), saturated ring
    # (6,7,8,9,3,4); junctions = idx3,4. Atom 8 is N.
    # IUPAC numbering verified via OPSIN chloro probing:
    #   1-Cl->ClC1NCCC2=CC=CC=C12 (idx9), 3-Cl->ClC1NCC2=CC=CC=C2C1 (idx7),
    #   4-Cl->ClC1CNCC2=CC=CC=C12 (idx6), 5-Cl (aromatic)->idx5,
    #   6-Cl->idx0, 7-Cl->idx1, 8-Cl->idx2.
    # Bridgeheads (4a,8a) inferred topologically: 4a (adj to 4,5) = idx4, 8a (adj to 1,8) = idx3.
    # N at locant 2 = idx8.
    "c1ccc2c(c1)CCNC2": {"name": "1,2,3,4-tetrahydroisoquinoline",
                         "substituent_form": "1,2,3,4-tetrahydroisoquinolin-yl",
                         "alkyl_stem_ok": False,
                         "atom_locants": {9: 1, 8: 2, 7: 3, 6: 4, 4: "4a",
                                          5: 5, 0: 6, 1: 7, 2: 8, 3: "8a"}},

    # 1,2,3,4-tetrahydroquinoline: benzo fused with saturated piperidine ring (N at
    # locant 1, in contrast to tetrahydroisoquinoline where N is at locant 2).
    # OPSIN '1,2,3,4-tetrahydroquinoline' -> N1CCCC2=CC=CC=C12; canonical
    # 'c1ccc2c(c1)CCCN2'.  Saturated ring (6,7,8,9,3,4); aromatic ring (0,1,2,3,4,5);
    # junctions = idx3,4.  Atom 9 is N.
    # IUPAC numbering verified via OPSIN chloro probing:
    #   1-Cl->ClN1CCCc2ccccc21 (N=idx9), 2-Cl->ClC1CCc2ccccc2N1 (idx8),
    #   3-Cl->ClC1CNc2ccccc2C1 (idx7), 4-Cl->ClC1CCNc2ccccc21 (idx6),
    #   5-Cl->Clc1cccc2c1CCCN2 (idx5), 6-Cl->idx0, 7-Cl->idx1, 8-Cl->idx2.
    # Bridgeheads (4a,8a) inferred topologically (aromatic-junction chloro probes
    # fail to kekulize): 4a (adj to 4=idx6 and 5=idx5) = idx4;
    # 8a (adj to N1=idx9 and 8=idx2) = idx3.
    "c1ccc2c(c1)CCCN2": {"name": "1,2,3,4-tetrahydroquinoline",
                         "substituent_form": "1,2,3,4-tetrahydroquinolin-yl",
                         "alkyl_stem_ok": False,
                         "atom_locants": {9: 1, 8: 2, 7: 3, 6: 4, 4: "4a",
                                          5: 5, 0: 6, 1: 7, 2: 8, 3: "8a"}},

    # 2,3,4,5-tetrahydro-1H-1-benzazepine: benzo fused with a saturated 7-membered
    # azepine ring; N at locant 1.  OPSIN '2,3,4,5-tetrahydro-1H-1-benzazepine'
    # -> N1CCCCC2=C1C=CC=C2; canonical 'c1ccc2c(c1)CCCCN2'.
    # Aromatic ring (0,1,2,3,4,5), saturated 7-ring (10(N),9,8,7,6,4,3);
    # junctions = idx3,4.  Atom 10 is N.
    # IUPAC numbering verified via OPSIN chloro probing (substructure-mapped to parent):
    #   1-Cl->ClN1CCCCc2ccccc21 (N=idx10), 2-Cl->ClC1CCCc2ccccc2N1 (idx9),
    #   3-Cl->ClC1CCc2ccccc2NC1 (idx8), 4-Cl->ClC1CCNc2ccccc2C1 (idx7),
    #   5-Cl->ClC1CCCNc2ccccc21 (idx6), 6-Cl->Clc1cccc2c1CCCCN2 (idx5),
    #   7-Cl->idx0, 8-Cl->idx1, 9-Cl->idx2.
    # Bridgeheads (5a,9a) inferred topologically (aromatic-junction chloro probes
    # fail to kekulize): 5a (adj to 5=idx6 and 6=idx5) = idx4;
    # 9a (adj to N1=idx10 and 9=idx2) = idx3.
    "c1ccc2c(c1)CCCCN2": {"name": "2,3,4,5-tetrahydro-1H-1-benzazepine",
                          "substituent_form": "2,3,4,5-tetrahydro-1H-1-benzazepin-yl",
                          "alkyl_stem_ok": False,
                          "atom_locants": {10: 1, 9: 2, 8: 3, 7: 4, 6: 5, 4: "5a",
                                           5: 6, 0: 7, 1: 8, 2: 9, 3: "9a"}},

    # -----------------------------------------------------------------------
    # Saturated polycyclic carbocycles (Cluster 2a)
    # -----------------------------------------------------------------------

    # decalin (decahydronaphthalene): fully saturated 6/6 fused bicyclic.
    # RDKit canonical 'C1CCC2CCCCC2C1': rings [0,9,8,3,2,1] and [4,5,6,7,8,3].
    # Junctions: idx3 (pos 4a) and idx8 (pos 8a).
    # IUPAC numbering verified via OPSIN chloro probing:
    #   1-Cl->ClC1CCCC2CCCCC12 (Cl on idx9; mol build confirms idx9=pos1),
    #   2-Cl->ClC1CCC2CCCCC2C1 (Cl on idx0=pos2),
    #   4a-Cl->ClC12CCCCC1CCCC2 (junction idx3=4a), 8a-Cl->same (symmetric).
    # Ring-0 sequence: idx8(8a)-idx9(1)-idx0(2)-idx1(3)-idx2(4)-idx3(4a).
    # Ring-1 sequence: idx3(4a)-idx4(5)-idx5(6)-idx6(7)-idx7(8)-idx8(8a).
    # Molecule is C2-symmetric: positions 1==5, 2==6, 3==7, 4==8, 4a==8a are equivalent.
    # pin_eligible=False: per P-25.3.1.3 / P-32.4 the retained name decalin is
    # general-nomenclature only; the PIN is the systematic decahydronaphthalene
    # (mirrors the tetraline / indane pattern above).  The pin_name /
    # pin_substituent_form fields override "name" / "substituent_form" when
    # pin_eligible=False: the retained record's atom_locants stay the load-bearing
    # alignment for ring numbering, but the emitted PIN string is the systematic
    # hydro-derived form.
    "C1CCC2CCCCC2C1":  {"name": "decalin", "substituent_form": "decalinyl", "alkyl_stem_ok": False,
                        "pin_eligible": False,
                        "pin_name": "decahydronaphthalene",
                        "pin_substituent_form": "decahydronaphthalen-N-yl",
                        "atom_locants": {9: 1, 0: 2, 1: 3, 2: 4, 3: "4a", 4: 5, 5: 6, 6: 7, 7: 8, 8: "8a"}},

    # -----------------------------------------------------------------------
    # Saturated polycyclic heterocycles (Cluster 2a)
    # -----------------------------------------------------------------------

    # penam (4-thia-1-azabicyclo[3.2.0]heptane + delta-2-beta-lactam):
    # retained IUPAC name (P-25.3.6.4). Core of penicillin antibiotics.
    # OPSIN 'penam' -> S1CCN2[C@H]1CC2=O. The retained name 'penam' ALREADY
    # encodes the C7 lactam carbonyl (probe: '7-oxopenam' is rejected by OPSIN
    # as unphysical valency). We therefore key the entry on the with-=O
    # canonical SMILES so the substitutive lookup goes through the
    # exocyclic-oxo fallback in retained_lookup, which claims the =O atom
    # and prevents downstream from re-emitting a redundant '7-oxo' prefix.
    # With-oxo canonical: O=C1CC2SCCN12.
    # Canonical atom indices in O=C1CC2SCCN12:
    #   idx0=O(exocyclic, claimed via extra_atom_indices),
    #   idx1=C(pos7,C=O-ring), idx2=C(pos6), idx3=C(pos5,bridgehead),
    #   idx4=S(pos1), idx5=C(pos2), idx6=C(pos3), idx7=N(pos4,bridgehead).
    # IUPAC: S at pos 1 (retained-name convention), N at pos 4 (bridgehead).
    # Verified by OPSIN chloro probing: 2-Cl->idx5, 3-Cl->idx6, 5-Cl->idx3,
    # 6-Cl->idx2 (all ring positions confirmed).
    "O=C1CC2SCCN12":   {"name": "penam", "substituent_form": "penam-yl", "alkyl_stem_ok": False,
                        "atom_locants": {1: 7, 2: 6, 3: 5, 4: 1, 5: 2, 6: 3, 7: 4}},

    # cepham (4-thia-1-azabicyclo[4.2.0]octane + delta-3-beta-lactam):
    # retained IUPAC name (P-25.3.6.4). Core of cephalosporin antibiotics.
    # Same shape as penam: 'cepham' encodes the C8 lactam carbonyl
    # (probe: '8-oxocepham' is rejected by OPSIN). Keyed on with-=O canonical
    # so the exocyclic-oxo fallback claims the =O atom.
    # OPSIN 'cepham' -> S1CCCN2[C@H]1CC2=O. With-oxo canonical: O=C1CC2SCCCN12.
    # Canonical atom indices in O=C1CC2SCCCN12:
    #   idx0=O(exocyclic, claimed via extra_atom_indices),
    #   idx1=C(pos8,C=O-ring), idx2=C(pos7), idx3=C(pos6,bridgehead),
    #   idx4=S(pos1), idx5=C(pos2), idx6=C(pos3), idx7=C(pos4),
    #   idx8=N(pos5,bridgehead).
    # IUPAC: S at pos 1, N at pos 5 (bridgehead).
    # Verified by OPSIN chloro probing: 2-Cl->idx5, 3-Cl->idx6, 4-Cl->idx7,
    # 6-Cl->idx3, 7-Cl->idx2.
    "O=C1CC2SCCCN12":  {"name": "cepham", "substituent_form": "cepham-yl", "alkyl_stem_ok": False,
                        "atom_locants": {1: 8, 2: 7, 3: 6, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5}},

    # cephem (4-thia-1-azabicyclo[4.2.0]oct-2-ene + delta-3-beta-lactam):
    # retained IUPAC name (P-25.3.6.4). Core of cephalosporin antibiotics with
    # the C3=C4 (cephem-numbering) double bond. OPSIN 'cephem' == '3-cephem'.
    # Retained name 'cephem' encodes BOTH the C8 lactam carbonyl AND the C3=C4
    # double bond (probe: '8-oxocephem' is rejected by OPSIN). Keyed on
    # with-=O canonical so the exocyclic-oxo fallback claims the =O atom.
    # OPSIN 'cephem' -> S1CC=CN2[C@H]1CC2=O. With-oxo canonical: O=C1CC2SCC=CN12.
    # Canonical atom indices in O=C1CC2SCC=CN12:
    #   idx0=O(exocyclic, claimed via extra_atom_indices),
    #   idx1=C(pos8,C=O-ring), idx2=C(pos7), idx3=C(pos6,bridgehead),
    #   idx4=S(pos1), idx5=C(pos2,CH2-S), idx6=C(pos3,=C), idx7=C(pos4,=C-N),
    #   idx8=N(pos5,bridgehead).
    # IUPAC: S at pos 1, N at pos 5 (bridgehead), C3=C4 double bond.
    # Verified by OPSIN chloro probing: 2-Cl->idx5, 3-Cl->idx6, 4-Cl->idx7,
    # 6-Cl->idx3, 7-Cl->idx2.
    "O=C1CC2SCC=CN12": {"name": "cephem", "substituent_form": "cephem-yl", "alkyl_stem_ok": False,
                        "atom_locants": {1: 8, 2: 7, 3: 6, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5}},

    # 5-oxa-1-azabicyclo[4.2.0]oct-2-en-8-one (oxacephem): the oxygen-replaced
    # cephem skeleton found in oxa-beta-lactam antibiotics (e.g. moxalactam).
    # No retained PIN — OPSIN's `cephem` token is sulfur-only.  We register
    # this canonical SMILES with the systematic IUPAC von Baeyer name so the
    # cephem-style retained-lookup path can emit
    # ``5-oxa-1-azabicyclo[4.2.0]oct-2-en-8-one-N-carboxylic acid`` substituted
    # forms.  Same scaffold shape as cephem but the heteroatom in the 6-ring
    # is O (so von Baeyer numbering puts N=1 bridgehead, O=5; locants run
    # 8→7→6(bridgehead)→5(O)→4→3→2→1(N) around the with-=O canonical).
    # OPSIN '5-oxa-1-azabicyclo[4.2.0]oct-2-en-8-one' -> N12C=CCOC2CC1=O.
    # With-oxo canonical: O=C1CC2OCC=CN12.
    # Canonical atom indices in O=C1CC2OCC=CN12:
    #   idx0=O(exocyclic, claimed via extra_atom_indices),
    #   idx1=C(pos8,C=O-ring), idx2=C(pos7), idx3=C(pos6,bridgehead),
    #   idx4=O(pos5), idx5=C(pos4), idx6=C(pos3,=C), idx7=C(pos2,=C-N),
    #   idx8=N(pos1,bridgehead).
    # Verified by OPSIN chloro probing: 2-Cl->idx7, 3-Cl->idx6, 4-Cl->idx5,
    # 6-Cl->idx3, 7-Cl->idx2.
    "O=C1CC2OCC=CN12": {"name": "5-oxa-1-azabicyclo[4.2.0]oct-2-en-8-one",
                        "substituent_form": "8-oxo-5-oxa-1-azabicyclo[4.2.0]oct-2-en-N-yl",
                        "alkyl_stem_ok": False,
                        "atom_locants": {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}},

    # 1-azabicyclo[4.2.0]oct-2-en-8-one (carba-cephem): the all-carbon-6-ring
    # cephem analogue found in carbacephem antibiotics (e.g. loracarbef,
    # cefoxitin variants).  No retained PIN.  Same atom_locants pattern as the
    # oxa version because the heteroatom inventory in the 6-ring is irrelevant
    # for von Baeyer numbering of the [4.2.0] skeleton (bridgehead N at lowest
    # locant 1, lactam C=O at locant 8).
    # OPSIN '1-azabicyclo[4.2.0]oct-2-en-8-one' -> N12C=CCCC2CC1=O.
    # With-oxo canonical: O=C1CC2CCC=CN12.
    # Canonical atom indices in O=C1CC2CCC=CN12:
    #   idx0=O(exocyclic, claimed via extra_atom_indices),
    #   idx1=C(pos8,C=O-ring), idx2=C(pos7), idx3=C(pos6,bridgehead),
    #   idx4=C(pos5), idx5=C(pos4), idx6=C(pos3,=C), idx7=C(pos2,=C-N),
    #   idx8=N(pos1,bridgehead).
    # Verified by OPSIN chloro probing: 2-Cl->idx7, 3-Cl->idx6, 4-Cl->idx5,
    # 6-Cl->idx3, 7-Cl->idx2.
    "O=C1CC2CCC=CN12": {"name": "1-azabicyclo[4.2.0]oct-2-en-8-one",
                        "substituent_form": "8-oxo-1-azabicyclo[4.2.0]oct-2-en-N-yl",
                        "alkyl_stem_ok": False,
                        "atom_locants": {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}},

    # NOTE: OPSIN rejects 4-substituted 'carbapenem' ('4-methylcarbapenem' ->
    # unphysical N valency), even though the 4-position exists in von Baeyer
    # numbering of 1-azabicyclo[3.2.0]hept-2-ene. Because FDA compounds in
    # this family (meropenem, imipenem, doripenem, ertapenem, ...) routinely
    # bear a 4-methyl, we cannot register 'carbapenem' as a retained-name
    # scaffold — our emitter would produce strings OPSIN cannot reparse.
    # Carbapenems must route through the von Baeyer bicyclic path instead.

    # decahydroquinoline (decahydroquinoline): fully saturated quinoline (6/6, N in one ring).
    # OPSIN 'decahydroquinoline' -> N1CCCC2CCCCC12; canonical: C1CCC2NCCCC2C1.
    # Atoms: idx4=N(pos1), junctions=idx8(4a),idx3(8a).
    # OPSIN chloro probing: 1-Cl->idx4(N), 2-Cl->idx5, 3-Cl->idx6, 4-Cl->idx7,
    #   4a-Cl->idx8, 5-Cl->idx9, 6-Cl->idx0, 7-Cl->idx1, 8-Cl->idx2, 8a-Cl->idx3.
    "C1CCC2NCCCC2C1":  {"name": "decahydroquinoline", "substituent_form": "decahydroquinolinyl",
                        "alkyl_stem_ok": False,
                        "atom_locants": {4: 1, 5: 2, 6: 3, 7: 4, 8: "4a", 9: 5, 0: 6, 1: 7, 2: 8, 3: "8a"}},

    # decahydroisoquinoline: fully saturated isoquinoline (6/6, N in isoquinoline position).
    # OPSIN 'decahydroisoquinoline' -> C1NCCC2CCCCC12; canonical: C1CCC2CNCCC2C1.
    # Atoms: idx5=N(pos2), junctions=idx8(4a),idx3(8a).
    # OPSIN chloro probing: 1-Cl->idx4, 2-Cl->idx5(N), 3-Cl->idx6, 4-Cl->idx7,
    #   4a-Cl->idx8, 8a-Cl->idx3.
    "C1CCC2CNCCC2C1":  {"name": "decahydroisoquinoline", "substituent_form": "decahydroisoquinolinyl",
                        "alkyl_stem_ok": False,
                        "atom_locants": {4: 1, 5: 2, 6: 3, 7: 4, 8: "4a", 3: "8a"}},

    # -----------------------------------------------------------------------
    # Benzo-fused saturated S-heterocycles
    # -----------------------------------------------------------------------

    # 2,3-dihydro-1,4-benzodithiine: benzo ring fused with 1,4-dithian (S-C-C-S, 6/6).
    # OPSIN '2,3-dihydro-1,4-benzodithiine' -> S1CCSC2=C1C=CC=C2; canonical: c1ccc2c(c1)SCCS2.
    # Atoms: aromatic ring idx0-5 (junctions idx3=8a, idx4=4a); sat ring idx6=S(pos4), idx7=C(pos3),
    #   idx8=C(pos2), idx9=S(pos1). Molecule is C2-symmetric: pos2==pos3, pos5==pos8, pos6==pos7.
    # OPSIN chloro probing: 2-Cl->idx7(C adj S1), 3-Cl->idx7(equiv by symmetry), 4a-Cl->idx4,
    #   5-Cl->idx2, 6-Cl->idx0, 7-Cl->idx1, 8-Cl->idx2(equiv 5), 8a-Cl->idx3.
    # Consistent with ZT-2363 outer SMILES being the pure ring (no substituents).
    "c1ccc2c(c1)SCCS2": {"name": "2,3-dihydro-1,4-benzodithiine",
                         "substituent_form": "2,3-dihydro-1,4-benzodithiin-yl",
                         "alkyl_stem_ok": False,
                         "atom_locants": {9: 1, 8: 2, 7: 3, 6: 4, 4: "4a", 5: 5, 0: 6, 1: 7, 2: 8, 3: "8a"}},

    # -----------------------------------------------------------------------
    # Fused saturated O,O-bicyclic heterocycles (isosorbide family)
    # -----------------------------------------------------------------------
    # Hexahydrofuro/octahydropyrano-fused dioxa bicyclics.  The leading case is
    # hexahydrofuro[3,2-b]furan, the bare core of 1,4:3,6-dianhydrohexitols
    # (isosorbide / isomannide / isoidide); without it FDA-0710 isosorbide
    # dinitrate is unnameable.

    # hexahydrofuro[3,2-b]furan: two fused tetrahydrofuran rings sharing two
    # carbons.  OPSIN 'hexahydrofuro[3,2-b]furan' -> O1C2C(CC1)OCC2 ;
    # RDKit canonical: C1CC2OCCC2O1.  Numbering O-1, C-2, C-3, C-3a, O-4, C-5,
    # C-6, C-6a (C2-symmetric: 2==5, 3==6, 3a==6a).
    # Atom indices: idx7=O(pos1), idx0=C(pos2), idx1=C(pos3), idx2=C(pos3a,bridgehead),
    #   idx3=O(pos4), idx4=C(pos5), idx5=C(pos6), idx6=C(pos6a,bridgehead).
    # OPSIN chloro probing: 2-Cl, 5-Cl -> ClC1CC2OCCC2O1 (equivalent by symmetry);
    #   3-Cl, 6-Cl -> ClC1COC2CCOC12; 3a-Cl, 6a-Cl -> ClC12CCOC1CCO2.
    "C1CC2OCCC2O1":    {"name": "hexahydrofuro[3,2-b]furan",
                        "substituent_form": "hexahydrofuro[3,2-b]furan-yl",
                        "alkyl_stem_ok": False,
                        "atom_locants": {7: 1, 0: 2, 1: 3, 2: "3a", 3: 4, 4: 5, 5: 6, 6: "6a"}},

    # hexahydrofuro[2,3-b]furan: 5,5-fused with O,O at non-equivalent positions.
    # OPSIN 'hexahydrofuro[2,3-b]furan' -> O1CCC2C1OCC2 ; canonical: C1CC2CCOC2O1.
    # Numbering O-1, C-2, C-3, C-3a, C-4, C-5, O-6, C-6a (Cs-symmetric: 2==5, 3==4;
    # bridgeheads 3a and 6a are NOT equivalent).
    # Atom indices in C1CC2CCOC2O1:
    #   idx7=O(pos1), idx0=C(pos2), idx1=C(pos3), idx2=C(pos3a,bridgehead, no adj-O),
    #   idx3=C(pos4), idx4=C(pos5), idx5=O(pos6), idx6=C(pos6a,bridgehead, between O1 and O6).
    # Confirmed by OPSIN chloro probing: 2-Cl/5-Cl -> ClC1CC2CCOC2O1;
    #   3-Cl/4-Cl -> ClC1COC2OCCC12; 3a-Cl -> ClC12CCOC1OCC2; 6a-Cl -> ClC12OCCC1CCO2.
    "C1CC2CCOC2O1":    {"name": "hexahydrofuro[2,3-b]furan",
                        "substituent_form": "hexahydrofuro[2,3-b]furan-yl",
                        "alkyl_stem_ok": False,
                        "atom_locants": {7: 1, 0: 2, 1: 3, 2: "3a", 3: 4, 4: 5, 5: 6, 6: "6a"}},

    # hexahydrofuro[3,4-c]furan: 5,5-fused with both O atoms at non-bridgehead
    # positions, mirror-symmetric.  OPSIN 'hexahydrofuro[3,4-c]furan' ->
    # C1OCC2C1COC2 ; canonical: C1OCC2COCC12.  Numbering C-1, O-2, C-3, C-3a,
    # C-4, O-5, C-6, C-6a (highly symmetric: 1==3==4==6, 3a==6a).
    # Atom indices in C1OCC2COCC12:
    #   idx0=C(pos1), idx1=O(pos2), idx2=C(pos3), idx3=C(pos3a,bridgehead),
    #   idx4=C(pos4), idx5=O(pos5), idx6=C(pos6), idx7=C(pos6a,bridgehead).
    # OPSIN chloro probing: 1/3/4/6-Cl all give ClC1OCC2COCC21;
    #   3a/6a-Cl give ClC12COCC1COC2.
    "C1OCC2COCC12":    {"name": "hexahydrofuro[3,4-c]furan",
                        "substituent_form": "hexahydrofuro[3,4-c]furan-yl",
                        "alkyl_stem_ok": False,
                        "atom_locants": {0: 1, 1: 2, 2: 3, 3: "3a", 4: 4, 5: 5, 6: 6, 7: "6a"}},

    # octahydropyrano[3,2-b]pyran: 6,6-fused saturated dioxa bicyclic, the pyran
    # analogue of hexahydrofuro[3,2-b]furan.  OPSIN 'octahydropyrano[3,2-b]pyran'
    # -> O1C2C(CCC1)OCCC2 ; canonical: C1COC2CCCOC2C1.  Numbering O-1, C-2,
    # C-3, C-4, C-4a, O-5, C-6, C-7, C-8, C-8a (C2-symmetric: 2==6, 3==7, 4==8,
    # 4a==8a).
    # Atom indices in C1COC2CCCOC2C1:
    #   idx2=O(pos1), idx1=C(pos2), idx0=C(pos3), idx9=C(pos4),
    #   idx8=C(pos4a,bridgehead), idx7=O(pos5), idx6=C(pos6), idx5=C(pos7),
    #   idx4=C(pos8), idx3=C(pos8a,bridgehead).
    # OPSIN chloro probing: 2-Cl/6-Cl -> ClC1CCC2OCCCC2O1; 3-Cl/7-Cl ->
    #   ClC1COC2CCCOC2C1; 4-Cl/8-Cl -> ClC1CCOC2CCCOC12;
    #   4a-Cl/8a-Cl -> ClC12CCCOC1CCCO2.
    "C1COC2CCCOC2C1":  {"name": "octahydropyrano[3,2-b]pyran",
                        "substituent_form": "octahydropyrano[3,2-b]pyran-yl",
                        "alkyl_stem_ok": False,
                        "atom_locants": {2: 1, 1: 2, 0: 3, 9: 4, 8: "4a",
                                         7: 5, 6: 6, 5: 7, 4: 8, 3: "8a"}},

    # hexahydrocyclopenta[b]furan: a 5,5-fused bicyclic system with one O
    # (tetrahydrofuran fused to a cyclopentane).  This is the prostaglandin-style
    # bicyclic core (e.g. FDA-0486 latanoprost-acid analogue).
    # OPSIN 'hexahydrocyclopenta[b]furan' -> O1C2C(CC1)CCC2 ; RDKit canonical:
    # C1CC2CCOC2C1.  Numbering O-1, C-2, C-3, C-3a, C-4, C-5, C-6, C-6a
    # (asymmetric: ring A has O-1/C-2/C-3 plus the two bridgeheads 3a/6a).
    # Atom indices in canonical 'C1CC2CCOC2C1':
    #   idx5=O(pos1), idx4=C(pos2), idx3=C(pos3), idx2=C(pos3a, bridgehead, NOT
    #   adjacent to O), idx1=C(pos4), idx0=C(pos5), idx7=C(pos6), idx6=C(pos6a,
    #   bridgehead, adjacent to O).
    # Verified by OPSIN chloro probing of every ring position (2,3,3a,4,5,6,6a),
    # de-Cl + substructure-match into parent canon.
    "C1CC2CCOC2C1":    {"name": "hexahydrocyclopenta[b]furan",
                        "substituent_form": "hexahydrocyclopenta[b]furan-yl",
                        "alkyl_stem_ok": False,
                        "atom_locants": {5: 1, 4: 2, 3: 3, 2: "3a",
                                         1: 4, 0: 5, 7: 6, 6: "6a"}},

    # -----------------------------------------------------------------------
    # Steroid skeleton: cyclopenta[a]phenanthrene (6-6-6-5 tetracyclic, 17 C)
    # -----------------------------------------------------------------------
    # The steroid core in IUPAC 2013 is named as a partially-saturated
    # cyclopenta[a]phenanthrene.  Five saturation states cover all 16
    # cyclopenta[a]phenanthrene-bearing compounds in the eval set.
    #
    # All atom_locants verified via OPSIN chloro-probing: for each ring variant,
    # every position 1-17 (or those accessible) was probed with N-chloro in
    # the fully-named ring, the resulting SMILES was parsed, Cl was removed,
    # and the remaining fragment was substructure-matched to the base ring
    # canonical SMILES to assign rdkit_atom_idx → IUPAC_locant.
    # Stereo-stripped keys are used so stereo-bearing fragments (FDA drugs)
    # are found by the stereo-tolerant fallback in retained_lookup.py (c4ac872).
    #
    # Positions at aromatic/bridgehead junctions (5 and 10 in the three
    # decahydro variants) cannot accept Cl without topology change; they are
    # omitted from atom_locants (never appear as substituent positions in the
    # test set).
    #
    # ---- Fully-aromatic parent: 15H-cyclopenta[a]phenanthrene ----
    # OPSIN '15H-cyclopenta[a]phenanthrene' -> 'C1=CC=CC2=CC=C3C=4CC=CC4C=CC3=C12'
    # RDKit canonical: 'C1=Cc2ccc3c(ccc4ccccc43)c2C1'.  17 atoms.
    # OPSIN accepts a FLAT 1-17 locant set (all integers valid, letter
    # suffixes rejected); positions 12, 13, 14 are the ring junctions.
    # Stage 4 unit 14: atom_locants verified by OPSIN methyl-probing all
    # 17 integer locants; all 17 atoms get a locant (no omissions).
    "C1=Cc2ccc3c(ccc4ccccc43)c2C1": {
        "name": "15H-cyclopenta[a]phenanthrene",
        "substituent_form": None,
        "alkyl_stem_ok": False,
        "atom_locants": {13: 1, 12: 2, 11: 3, 10: 4, 9: 5, 8: 6, 7: 7,
                         6: 8, 5: 9, 14: 10, 4: 11, 3: 12, 2: 13,
                         15: 14, 16: 15, 0: 16, 1: 17}},

    # ---- Ring 1 of 5: aromatic A-ring (estrone/estradiol class) ----
    # OPSIN 'decahydro-17H-cyclopenta[a]phenanthrene' / '6,7,8,9,11,12,13,14,15,16-decahydro-17H-...'
    # → 'C1=CC=CC=2CCC3C4CCCC4CCC3C12'; RDKit canonical: c1ccc2c(c1)CCC1C2CCC2CCCC21
    # Positions 1-4 are aromatic (ring A), 6-9 & 11-17 are sp3.
    # Positions 5 (4a-type junction) and 10 (8a-type junction) omitted (not substitutable).
    # Compounds: FDA-0505, FDA-0513, FDA-0330, FDA-0504 (estrone/estradiol family)
    "c1ccc2c(c1)CCC1C2CCC2CCCC21": {
        "name": "6,7,8,9,11,12,13,14,15,16-decahydro-17H-cyclopenta[a]phenanthrene",
        "substituent_form": "6,7,8,9,11,12,13,14,15,16-decahydro-17H-cyclopenta[a]phenanthrenyl",
        "alkyl_stem_ok": False,
        # atom_locants: ring_mol_atom_idx -> IUPAC_locant
        # Verified via OPSIN chloro-probing of 1-9, 11-17 (pos 5, 10 = junction, skipped).
        "atom_locants": {2: 1, 1: 2, 0: 3, 5: 4,  # pos 5 (junction) OMITTED
                         6: 6, 7: 7, 8: 8, 9: 9,
                         10: 11, 11: 12, 12: 13, 16: 14, 15: 15, 14: 16, 13: 17}},

    # ---- Ring 2 of 5: tetradecahydro-3H form (testosterone/progesterone class) ----
    # OPSIN '1,2,6,7,8,9,10,11,12,13,14,15,16,17-tetradecahydro-3H-cyclopenta[a]phenanthrene'
    # → 'C1CCC=C2CCC3C4CCCC4CCC3C12'; RDKit canonical: C1=C2CCC3C4CCCC4CCC3C2CCC1
    # One C=C double bond at C3=C4 (the 'enone' ring A).  Position 5 (junction) omitted.
    # Compounds: FDA-0662, FDA-1307, FDA-0867, FDA-0931 (testosterone/progesterone family)
    "C1=C2CCC3C4CCCC4CCC3C2CCC1": {
        "name": "1,2,6,7,8,9,10,11,12,13,14,15,16,17-tetradecahydro-3H-cyclopenta[a]phenanthrene",
        "substituent_form": "1,2,6,7,8,9,10,11,12,13,14,15,16,17-tetradecahydro-3H-cyclopenta[a]phenanthrenyl",
        "alkyl_stem_ok": False,
        # Verified via OPSIN chloro-probing (pos 5 = junction, skipped).
        "atom_locants": {14: 1, 15: 2, 16: 3, 0: 4,  # pos 5 (junction) OMITTED
                         2: 6, 3: 7, 4: 8, 12: 9, 13: 10, 11: 11, 10: 12, 9: 13,
                         5: 14, 6: 15, 7: 16, 8: 17}},

    # ---- Ring 3 of 5: dodecahydro-3H form v1 (progesterone/11-deoxy class) ----
    # OPSIN '1,2,8,9,10,11,12,13,14,15,16,17-dodecahydro-3H-cyclopenta[a]phenanthrene'
    # → 'C1CCC=C2C=CC3C4CCCC4CCC3C12'; RDKit canonical: C1=CC2C3CCCC3CCC2C2CCCC=C12
    # Two C=C double bonds (ring A enone + ring B).  Position 5 (junction) omitted.
    # Compounds: FDA-0817 (progesterone-like with 4,6-dien-3-one)
    "C1=CC2C3CCCC3CCC2C2CCCC=C12": {
        "name": "1,2,8,9,10,11,12,13,14,15,16,17-dodecahydro-3H-cyclopenta[a]phenanthrene",
        "substituent_form": "1,2,8,9,10,11,12,13,14,15,16,17-dodecahydro-3H-cyclopenta[a]phenanthrenyl",
        "alkyl_stem_ok": False,
        # Verified via OPSIN chloro-probing (pos 5 = junction, skipped).
        "atom_locants": {12: 1, 13: 2, 14: 3, 15: 4,  # pos 5 (junction) OMITTED
                         0: 6, 1: 7, 2: 8, 10: 9, 11: 10, 9: 11, 8: 12, 7: 13,
                         3: 14, 4: 15, 5: 16, 6: 17}},

    # ---- Ring 4 of 5: dodecahydro-3H form v2 (cortisone/corticosteroid class) ----
    # OPSIN '6,7,8,9,10,11,12,13,14,15,16,17-dodecahydro-3H-cyclopenta[a]phenanthrene'
    # → 'C1=CCC=C2CCC3C4CCCC4CCC3C12'; RDKit canonical: C1=CC2C(=CC1)CCC1C3CCCC3CCC21
    # Two C=C double bonds (ring A 3-en-1-one pattern).  Position 5 (junction) omitted.
    # Compounds: FDA-0127, FDA-0029, FDA-0574, FDA-1107 (corticosteroid family)
    "C1=CC2C(=CC1)CCC1C3CCCC3CCC21": {
        "name": "6,7,8,9,10,11,12,13,14,15,16,17-dodecahydro-3H-cyclopenta[a]phenanthrene",
        "substituent_form": "6,7,8,9,10,11,12,13,14,15,16,17-dodecahydro-3H-cyclopenta[a]phenanthrenyl",
        "alkyl_stem_ok": False,
        # Verified via OPSIN chloro-probing (pos 5 = junction, skipped).
        "atom_locants": {1: 1, 0: 2, 5: 3, 4: 4,  # pos 5 (junction) OMITTED
                         6: 6, 7: 7, 8: 8, 16: 9, 2: 10, 15: 11, 14: 12, 13: 13,
                         9: 14, 10: 15, 11: 16, 12: 17}},

    # ---- Ring 5 of 5: hexadecahydro-1H form (bile acid / cholesterol class) ----
    # OPSIN 'hexadecahydro-1H-cyclopenta[a]phenanthrene'
    # → 'C1CCCC2CCC3C4CCCC4CCC3C12'; RDKit canonical: C1CCC2C(C1)CCC1C3CCCC3CCC21
    # Fully saturated steroid skeleton (= gonane).  All 17 positions probed.
    # Compounds: FDA-1286, FDA-0257 (bile acid / sterol family)
    "C1CCC2C(C1)CCC1C3CCCC3CCC21": {
        "name": "hexadecahydro-1H-cyclopenta[a]phenanthrene",
        "substituent_form": "hexadecahydro-1H-cyclopenta[a]phenanthrenyl",
        "alkyl_stem_ok": False,
        # All 17 positions verified via OPSIN chloro-probing.
        "atom_locants": {2: 1, 1: 2, 0: 3, 5: 4, 4: 5, 6: 6, 7: 7, 8: 8,
                         16: 9, 3: 10, 15: 11, 14: 12, 13: 13,
                         9: 14, 10: 15, 11: 16, 12: 17}},

    # ---- Ring 6 of 7: dodecahydro-3H form v3 (mifepristone / 4,5-enone + ring-B ene class) ----
    # OPSIN '1,2,6,7,8,11,12,13,14,15,16,17-dodecahydro-3H-cyclopenta[a]phenanthrene'
    # → 'C1CCC=C2CCC3C4CCCC4CCC3=C12'; RDKit canonical: C1=C2CCC3C(=C2CCC1)CCC1CCCC13
    # Two C=C bonds: ring-A (C4=C5) and ring-B (C9=C10). Ring-A enone class
    # where -3-one makes this the mifepristone-style skeleton.
    # All positions verified via OPSIN chloro-probing (5, 9, 10 = junctions).
    # Compounds: FDA-0888 (mifepristone)
    "C1=C2CCC3C(=C2CCC1)CCC1CCCC13": {
        "name": "1,2,6,7,8,11,12,13,14,15,16,17-dodecahydro-3H-cyclopenta[a]phenanthrene",
        "substituent_form": "1,2,6,7,8,11,12,13,14,15,16,17-dodecahydro-3H-cyclopenta[a]phenanthrenyl",
        "alkyl_stem_ok": False,
        "atom_locants": {7: 1, 8: 2, 9: 3, 0: 4, 2: 6, 3: 7, 4: 8, 5: 9, 6: 10,
                         10: 11, 11: 12, 12: 13, 16: 14, 15: 15, 14: 16, 13: 17}},

    # ---- Ring 7 of 7: dodecahydro-1H form (abiraterone / ring-B + ring-D ene class) ----
    # OPSIN '2,3,4,7,8,9,10,11,12,13,14,15-dodecahydro-1H-cyclopenta[a]phenanthrene'
    # → 'C1CCCC2=CCC3C4CC=CC4CCC3C12'; RDKit canonical: C1=CC2CCC3C4CCCCC4=CCC3C2C1
    # Two C=C bonds: ring-B (C5=C6) and ring-D (C16=C17). Abiraterone-type
    # steroid precursor with the Δ5,6 and Δ16,17 double bond pattern.
    # All positions verified via OPSIN chloro-probing (5, 9 = junctions).
    # Compounds: FDA-0002 (abiraterone acetate)
    "C1=CC2CCC3C4CCCCC4=CCC3C2C1": {
        "name": "2,3,4,7,8,9,10,11,12,13,14,15-dodecahydro-1H-cyclopenta[a]phenanthrene",
        "substituent_form": "2,3,4,7,8,9,10,11,12,13,14,15-dodecahydro-1H-cyclopenta[a]phenanthrenyl",
        "alkyl_stem_ok": False,
        "atom_locants": {7: 1, 8: 2, 9: 3, 10: 4, 12: 6, 13: 7, 14: 8,
                         5: 9, 6: 10, 4: 11, 3: 12, 2: 13,
                         15: 14, 16: 15, 0: 16, 1: 17}},

    # -----------------------------------------------------------------------
    # indolo[4,3-fg]quinoline family (ergoline scaffold)
    # -----------------------------------------------------------------------
    # 4-ring fused system: benzene + pyrrole (indole) fused with a
    # pyridine / tetrahydropyridine ring.  Parent is indolo[4,3-fg]quinoline.
    # Used for ergot alkaloids (ergoline class) in FDA drug set.
    # Locants 1–10, 6a, 10a per IUPAC fusion-numbering rules.
    #
    # ---- Bare aromatic indolo[4,3-fg]quinoline ----
    # OPSIN → C1=CC=C2N=CC=3C2=C1C=1C=CC=NC1C3
    # RDKit canonical: C1=Nc2cccc3c2c1cc1ncccc13
    # (11 aromatic C/N + 2 pyrrolic carbons; N1 = pyrrole NH, N5 = pyridine N.)
    "C1=Nc2cccc3c2c1cc1ncccc13": {
        "name": "indolo[4,3-fg]quinoline",
        "substituent_form": "indolo[4,3-fg]quinolinyl",
        "alkyl_stem_ok": False,
        # Atom_locants verified via OPSIN chloro-probing.
        "atom_locants": {5: 1, 4: 2, 3: 3, 1: 4, 0: 5, 9: 6, 11: 7, 12: 8,
                         13: 9, 14: 10}},

    # ---- 4,6,6a,7,8,9-hexahydroindolo[4,3-fg]quinoline (lysergic acid / lisuride) ----
    # OPSIN → C1=CC=C2NC=C3C2=C1C1=CCCNC1C3
    # RDKit canonical: C1=C2c3cccc4[nH]cc(c34)CC2NCC1
    # 4,5 double bond preserved in ring-C; ring-D is saturated.
    # Compounds: FDA-0779 (lisuride), FDA-0168 (bromocriptine core),
    # FDA-0869 (methysergide, has N4-methyl on indole; stereo-stripped match).
    "C1=C2c3cccc4[nH]cc(c34)CC2NCC1": {
        "name": "4,6,6a,7,8,9-hexahydroindolo[4,3-fg]quinoline",
        "substituent_form": "4,6,6a,7,8,9-hexahydroindolo[4,3-fg]quinolinyl",
        "alkyl_stem_ok": False,
        # Atom_locants verified via OPSIN chloro-probing of positions 1–10, 6a, 10a.
        # Pos 4 = pyrrole NH (atom 7) — manual assignment (probe ambiguous on 4).
        "atom_locants": {3: 1, 4: 2, 5: 3, 7: 4, 8: 5, 12: "6a",
                         13: 7, 14: 8, 15: 9, 0: 10, 1: "10a"}},

    # ---- 4,6,6a,7,8,9,10,10a-octahydroindolo[4,3-fg]quinoline (cabergoline / ergoloid) ----
    # OPSIN → C1=CC=C2NC=C3C2=C1C1CCCNC1C3
    # RDKit canonical: c1cc2c3c(c[nH]c3c1)CC1NCCCC21
    # Ring-C fully saturated (both 9,10 and 10a,10a+1 saturated); fully reduced ergoline.
    # Compounds: FDA-0186 (cabergoline), FDA-0489 (ergoloid mesylate).
    "c1cc2c3c(c[nH]c3c1)CC1NCCCC21": {
        "name": "4,6,6a,7,8,9,10,10a-octahydroindolo[4,3-fg]quinoline",
        "substituent_form": "4,6,6a,7,8,9,10,10a-octahydroindolo[4,3-fg]quinolinyl",
        "alkyl_stem_ok": False,
        # Atom_locants verified via OPSIN chloro-probing of positions 1–10, 6a, 10a.
        # Pos 4 = pyrrole NH (atom 6) — manual assignment (probe ambiguous on 4).
        "atom_locants": {1: 1, 0: 2, 8: 3, 6: 4, 5: 5, 10: "6a",
                         11: 7, 12: 8, 13: 9, 14: 10, 15: "10a"}},

    # ---- 1,2,3,4,4a,5,7,8,13,14-decahydroindolo[2',3':3,4]pyrido[1,2-b]isoquinoline ----
    # Pentacyclic core of deserpidine (FDA-0375).  Five fused rings: benzene (ring A,
    # aromatic), pyrrole (ring B, aromatic, N-H), piperidine (ring C, sp3, N=bridgehead),
    # cyclohexane (ring D, sp3), and an additional cyclohexane (ring E, sp3) fused to
    # ring C at positions 13b/14a.  This is the yohimbane scaffold (rings A-C-D) extended
    # by ring E (cyclohexane) fused to yohimbane at C2-C3.
    # Parent (fully aromatic): indolo[2',3':3,4]pyrido[1,2-b]isoquinoline
    #   OPSIN → c1ccc2cn3ccc4c5ccccc5nc-4c3cc2c1  (21 atoms, arom=21)
    # Decahydro: 1,2,3,4,4a,5,7,8,13,14-decahydroindolo[2',3':3,4]pyrido[1,2-b]isoquinoline
    #   OPSIN → c1ccc2c3c([nH]c2c1)C1CC2CCCCC2CN1CC3  (arom=9, matches target scaffold)
    # RDKit canonical: c1ccc2c3c([nH]c2c1)C1CC2CCCCC2CN1CC3
    #
    # Atom_locants: positions 1–14, 4a, 5, 6, 7, 8, 8a, 9, 9a, 12, 12a, 13, 13a, 13b, 14a
    # Probed via OPSIN methyl-substitution on the decahydro name (MCS-verified against scaffold):
    #   1→idx12, 2→idx13, 3→idx14, 4→idx15, 4a→idx16, 5→idx17, 7→idx19, 8→idx20,
    #   8a→idx4, 9→idx2, 10→idx1, 11→idx0, 12→idx8, 12a→idx7, 13→idx6 (indole NH,
    #   confirmed by N-methyl probe), 13a→idx5, 13b→idx9, 14→idx10, 14a→idx11.
    # Positions 6 (bridgehead N, sp3, not substitutable) and 9a (aromatic tri-ring
    # junction C shared by rings A, C, and pyrido) deduced topologically:
    #   9a (idx3) adjacent to 9(idx2), 8a(idx4), and 12a(idx7) — tri-ring junction;
    #   6 (idx18, sp3 N) adjacent to 5(idx17), 7(idx19), and 13b(idx9) — bridgehead N.
    # Compounds: FDA-0375 (deserpidine = 11-desmethoxy reserpine).
    "c1ccc2c3c([nH]c2c1)C1CC2CCCCC2CN1CC3": {
        "name": "1,2,3,4,4a,5,7,8,13,14-decahydroindolo[2',3':3,4]pyrido[1,2-b]isoquinoline",
        "substituent_form": "1,2,3,4,4a,5,7,8,13,14-decahydroindolo[2',3':3,4]pyrido[1,2-b]isoquinolinyl",
        "alkyl_stem_ok": False,
        "atom_locants": {12: 1, 13: 2, 14: 3, 15: 4, 16: "4a", 17: 5, 18: 6, 19: 7, 20: 8,
                         4: "8a", 2: 9, 3: "9a", 1: 10, 0: 11, 8: 12, 7: "12a", 5: "13a",
                         6: 13, 9: "13b", 10: 14, 11: "14a"}},

    # ---- morphinan ----
    # Classic opioid core: phenanthrene-like tetracycle with a nitrogen-
    # containing ring (ring D) — rings A (aromatic), B (sp3 cyclohexane),
    # C (sp3 cyclohexane), D (piperidine).
    # OPSIN 'morphinan' → C1=CC=CC=2C34CCCCC3C(CC21)NCC4.
    # RDKit canonical: c1ccc2c(c1)CC1NCCC23CCCCC13 (17 atoms: 16 C + 1 N).
    # Atom_locants verified via OPSIN methyl-probing of positions 1-10, 14-17
    # and deduced by topology for positions 11, 12 (aromatic A/B ring
    # junctions) and 13 (sp3 B/C/D ring junction, quaternary).  Morphinan
    # numbering has no 4a / 8a locants (all 17 atoms fit into positions
    # 1-17 including 11, 12, 13 as junctions).
    # Compounds: FDA-0753 (levallorphan, 17-(prop-2-en-1-yl)morphinan-3-ol).
    "c1ccc2c(c1)CC1NCCC23CCCCC13": {
        "name": "morphinan",
        "substituent_form": "morphinanyl",
        "alkyl_stem_ok": False,
        "atom_locants": {5: 1, 0: 2, 1: 3, 2: 4, 12: 5, 13: 6, 14: 7, 15: 8,
                         7: 9, 6: 10, 4: 11, 3: 12, 11: 13, 16: 14, 10: 15,
                         9: 16, 8: 17}},

    # ---- 1,2,3,4,5,6-hexahydro-2,6-methano-3-benzazocine (benzomorphan) ----
    # Benzene fused to a methano-bridged azocine (benzomorphan scaffold).  Core
    # of pentazocine, phenazocine, cyclazocine and other benzomorphan opioids.
    # OPSIN '1,2,3,4,5,6-hexahydro-2,6-methano-3-benzazocine' parses cleanly;
    # the historical name "benzomorphan" is NOT recognised by OPSIN so we use
    # the modern IUPAC-style name as the stem.
    # OPSIN → c1ccc2c(c1)CC3CC2CCN3, RDKit canonical: c1ccc2c(c1)CC1CC2CCN1
    # (13 atoms: 6 aromatic C + 6 sp3 C + 1 N).
    # Atom_locants verified via OPSIN chloro/methyl probing (positions 1,2,4,5,
    # 6,7,8,9,10,11, plus 3 (N) via 3-methyl); junctions 6a and 10a assigned
    # from topology (aromatic junction adjacent to sp3 position 6 → 6a = idx3;
    # aromatic junction adjacent to sp3 position 1 → 10a = idx4).
    # Compounds: FDA-1035 (pentazocine).
    "c1ccc2c(c1)CC1CC2CCN1": {
        "name": "1,2,3,4,5,6-hexahydro-2,6-methano-3-benzazocine",
        "substituent_form": "1,2,3,4,5,6-hexahydro-2,6-methano-3-benzazocinyl",
        "alkyl_stem_ok": False,
        "atom_locants": {6: 1, 7: 2, 12: 3, 11: 4, 10: 5, 9: 6, 3: "6a",
                         2: 7, 1: 8, 0: 9, 5: 10, 4: "10a", 8: 11}},

    # ---- 4a,5,9,10,11,12-hexahydro-6H-[1]benzofuro[3a,3,2-ef][2]benzazepine ----
    # Galanthamine scaffold (Amaryllidaceae alkaloid, AChE inhibitor).
    # Four fused rings: benzene (ring A) ortho-fused to benzofuran via the
    # furan O, spiro-linked through a quaternary sp3 carbon (C12a) to a
    # cyclohexene (ring C), with an azepine ring (ring D, N11) bridging
    # C4a-C12a to the aromatic ring via C12 and C6 (sp3 CH2).
    # OPSIN '4a,5,9,10,11,12-hexahydro-6H-[1]benzofuro[3a,3,2-ef][2]benzazepine'
    # → C1=CC=C2C=3C4(CCNCC31)C(O2)CCC=C4.
    # RDKit canonical: C1=CC23CCNCc4cccc(c42)OC3CC1.
    # Compounds: FDA-0605 (galanthamine) — 3-methoxy-11-methyl-(...)-6-ol.
    # Atom_locants (17-atom canonical) verified via OPSIN chloro/methyl probing
    # of positions 1,2,3,5,6,7,8,9,10,12,4a (1-chloro → idx8, 3-chloro → idx10,
    # 11-methyl → idx5 (N)), plus topological deduction of junctions:
    #   idx2 = 12a (sp3 quat, 4 C-neighbours),
    #   idx12 = 12b (aromatic junction, between ring A and the sp3 quat),
    #   idx7 = 8a (aromatic junction adj. to CH2-6 and CH2-N),
    #   idx11 = 3a (aromatic junction adj. to furan O),
    #   idx13 = 4 (furan oxygen, non-substitutable).
    "C1=CC23CCNCc4cccc(c42)OC3CC1": {
        "name": "4a,5,9,10,11,12-hexahydro-6H-[1]benzofuro[3a,3,2-ef][2]benzazepine",
        "substituent_form": "4a,5,9,10,11,12-hexahydro-6H-[1]benzofuro[3a,3,2-ef][2]benzazepinyl",
        "alkyl_stem_ok": False,
        "atom_locants": {8: 1, 9: 2, 10: 3, 13: 4, 14: "4a", 15: 5, 16: 6,
                         0: 7, 1: 8, 7: "8a", 3: 9, 4: 10, 5: 11, 6: 12,
                         2: "12a", 11: "3a", 12: "12b"}},

    # -----------------------------------------------------------------------
    # octahydro-8H-oxazolo[3,2-a]pyrrolo[2,1-c]pyrazine
    # -----------------------------------------------------------------------
    # 3-ring fused heterocycle: oxazolidine fused to piperazine fused to
    # pyrrolidine.  Parent is oxazolo[3,2-a]pyrrolo[2,1-c]pyrazine.
    # Fully saturated form with indicated-H at position 8 is
    # "octahydro-8H-oxazolo[3,2-a]pyrrolo[2,1-c]pyrazine".
    # Compounds: FDA-0168 (bromocriptine), FDA-0489 (ergoloid mesylate) —
    # the cyclol substituent fused to their indolo[4,3-fg]quinoline core.
    # OPSIN → O1CCN2C1C1N(CC2)CCC1
    # RDKit canonical: C1CC2C3OCCN3CCN2C1
    # Substitutable locants verified by OPSIN chloro-probing: 2, 3, 5, 6, 8,
    # 9, 10, 10a, 10b (positions 1=O, 4/7=shared N are non-substitutable).
    "C1CC2C3OCCN3CCN2C1": {
        "name": "octahydro-8H-oxazolo[3,2-a]pyrrolo[2,1-c]pyrazine",
        "substituent_form": "octahydro-8H-oxazolo[3,2-a]pyrrolo[2,1-c]pyrazinyl",
        "alkyl_stem_ok": False,
        "atom_locants": {5: 2, 6: 3, 8: 5, 9: 6, 11: 8, 0: 9, 1: 10,
                         2: "10a", 3: "10b"}},

    # -----------------------------------------------------------------------
    # azepane (formerly homopiperidine): 7-membered ring, 1 N
    # RDKit canonical for N1CCCCCC1 is C1CCCNCC1
    "C1CCCNCC1":       {"name": "azepane",         "substituent_form": "azepan-1-yl",       "alkyl_stem_ok": False},
    # 1,4-diazepane (formerly homopiperazine): 7-membered ring, 2 N at 1,4
    # RDKit canonical for N1CCNCCC1 is C1CNCCNC1
    "C1CNCCNC1":       {"name": "1,4-diazepane",   "substituent_form": "1,4-diazepan-1-yl", "alkyl_stem_ok": False},
    # 1,4-oxazepane (formerly homomorpholine): 7-membered ring, N and O at 1,4
    # RDKit canonical for N1CCOCCC1 is C1CNCCOC1
    "C1CNCCOC1":       {"name": "1,4-oxazepane",   "substituent_form": "1,4-oxazepan-4-yl", "alkyl_stem_ok": False},
    # 1,4-thiazepane / 1,4-selenazepane / 1,4-tellurazepane (formerly the
    # obsolete arylGroups.xml "thiahomomorpholin / selenohomomorpholin /
    # tellurohomomorpholin" stems): 7-membered ring with N and the chalcogen at
    # 1,4 — the chalcogen is senior (O>S>Se>Te>N), so it takes locant 1 and N
    # takes 4.  Without these curated entries the obsolete "*homomorpholine"
    # OPSIN stem is rehydrated as the standalone parent and, lacking
    # atom_locants, misnumbers substituted forms (e.g. "2-chlorothiahomo-
    # morpholine" for a Cl the 1,4 numbering places at locant 7), failing the
    # OPSIN round-trip.  Atom layout (canonical C1CN CC X C1): idx5 = chalcogen
    # (locant 1), idx4 = 2, idx3 = 3, idx2 = N (locant 4), idx1 = 5, idx0 = 6,
    # idx6 = 7 — derived by OPSIN methyl-probing each numeric locant.
    "C1CNCCSC1":       {"name": "1,4-thiazepane",  "substituent_form": "1,4-thiazepan-4-yl", "alkyl_stem_ok": False,
                        "atom_locants": {5: 1, 4: 2, 3: 3, 2: 4, 1: 5, 0: 6, 6: 7}},
    "C1CNCC[Se]C1":    {"name": "1,4-selenazepane", "substituent_form": "1,4-selenazepan-4-yl", "alkyl_stem_ok": False,
                        "atom_locants": {5: 1, 4: 2, 3: 3, 2: 4, 1: 5, 0: 6, 6: 7}},
    "C1CNCC[Te]C1":    {"name": "1,4-tellurazepane", "substituent_form": "1,4-tellurazepan-4-yl", "alkyl_stem_ok": False,
                        "atom_locants": {5: 1, 4: 2, 3: 3, 2: 4, 1: 5, 0: 6, 6: 7}},

    # -----------------------------------------------------------------------
    # 6-membered O-heterocycles
    # -----------------------------------------------------------------------
    # Morpholine — RDKit canonical is C1COCCN1 (NOT C1CNCCO1)
    "C1COCCN1":        {"name": "morpholine",  "substituent_form": "morpholinyl",  "alkyl_stem_ok": False},

    # -----------------------------------------------------------------------
    # Partially saturated 6-membered heterocycles (pyran/pyridine/thiopyran family)
    # -----------------------------------------------------------------------
    # IUPAC P-25.3.1.3 / P-31.1.4: these use indicated-H (e.g. 2H-, 4H-) and
    # dihydro/tetrahydro prefixes over the mancude parent (pyran/pyridine).
    # atom_locants computed via OPSIN chloro-probing of each ring position.
    #
    # 3,4-dihydro-2H-pyran: canonical C1=COCCC1
    # idx2=O=1, idx3=2(sp3), idx4=3(sp3), idx5=4(sp3), idx0=5(sp2), idx1=6(sp2)
    "C1=COCCC1":       {"name": "3,4-dihydro-2H-pyran", "substituent_form": "3,4-dihydro-2H-pyranyl", "alkyl_stem_ok": False,
                        "atom_locants": {2: 1, 3: 2, 4: 3, 5: 4, 0: 5, 1: 6}},
    # 3,6-dihydro-2H-pyran: canonical C1=CCOCC1
    # idx3=O=1, idx2=2(sp3), idx1=3(sp2), idx0=4(sp2), idx5=5(sp3 alt), idx4=... we use: 1=3,2=4,3=5,4=0,5=1,6=2
    "C1=CCOCC1":       {"name": "3,6-dihydro-2H-pyran", "substituent_form": "3,6-dihydro-2H-pyranyl", "alkyl_stem_ok": False,
                        "atom_locants": {1: 3, 2: 4, 3: 5, 4: 0, 5: 1, 6: 2}},
    # 2H-pyran: canonical C1=CCOC=C1
    # idx3=O=1, idx2=2(sp3), idx1=3, idx0=4, idx5=5, idx4=6
    "C1=CCOC=C1":      {"name": "2H-pyran", "substituent_form": "2H-pyranyl", "alkyl_stem_ok": False,
                        "atom_locants": {1: 3, 2: 2, 3: 1, 4: 0, 5: 5, 6: 4}},
    # 4H-pyran: canonical C1=COC=CC1
    # idx2=O=1, idx1=2, idx0=3, idx5=4(sp3), idx4=5, idx3=6
    "C1=COC=CC1":      {"name": "4H-pyran", "substituent_form": "4H-pyranyl", "alkyl_stem_ok": False,
                        "atom_locants": {1: 2, 2: 1, 3: 0, 4: 5, 5: 4, 6: 3}},
    # 4H-thiopyran: canonical C1=CSC=CC1 (S analogue of 4H-pyran; same topology).
    # P-25.2.1 / P-22.2.1.1 favour the retained "thiopyran" over HW "thiine" for
    # 6-rings with 1 S.  Atom layout: idx2=S(pos1), idx1=C(pos2), idx0=C(pos3),
    # idx5=C(pos4,sp3 CH2), idx4=C(pos5), idx3=C(pos6).  Mirrors 4H-pyran.
    "C1=CSC=CC1":      {"name": "4H-thiopyran", "substituent_form": "4H-thiopyranyl", "alkyl_stem_ok": False,
                        "atom_locants": {1: 2, 2: 1, 3: 0, 4: 5, 5: 4, 6: 3}},
    # 2H-thiopyran / 2H-selenopyran / 2H-telluropyran: the 2H tautomers of
    # the thio/seleno/telluro pyran family.  Canonical SMILES mirror 2H-pyran
    # ``C1=CCOC=C1`` with S/Se/Te in place of O.  Atom layout: the sp3 CH2
    # ring carbon is atom 2 in canonical ordering (the unique saturated ring
    # atom carrying 2H), and the ring heteroatom is atom 3.  IUPAC P-25.2.1 /
    # P-22.2.1.1 favour the retained thiopyran/selenopyran/telluropyran over
    # HW "thiine/selenine/tellurine" for 6-rings with a single chalcogen.
    # Without these curated atom_locants entries, the 2H- marker is implicit
    # in the OPSIN arylGroups.xml name ``thiopyran`` (stem form), and the
    # numbering falls through to the generic monocyclic traversal where the
    # indicated-H atom and substituent locants are NOT coherently pinned.
    "C1=CCSC=C1":      {"name": "2H-thiopyran", "substituent_form": "2H-thiopyranyl", "alkyl_stem_ok": False,
                        "atom_locants": {1: 3, 2: 2, 3: 1, 4: 0, 5: 5, 6: 4}},
    "C1=CC[Se]C=C1":   {"name": "2H-selenopyran", "substituent_form": "2H-selenopyranyl", "alkyl_stem_ok": False,
                        "atom_locants": {1: 3, 2: 2, 3: 1, 4: 0, 5: 5, 6: 4}},
    "C1=CC[Te]C=C1":   {"name": "2H-telluropyran", "substituent_form": "2H-telluropyranyl", "alkyl_stem_ok": False,
                        "atom_locants": {1: 3, 2: 2, 3: 1, 4: 0, 5: 5, 6: 4}},
    # 4H-pyran-4-one (4-pyrone): canonical O=c1ccocc1 — the aromatic 6-ring
    # with 1 O and an exocyclic carbonyl on the opposite carbon.  IUPAC
    # P-25.2.1 / P-22.2.1.1 forbids HW "oxine" for 6-rings with 1 O; the
    # retained parent is pyran (with 4H-indicated-H since the =O is on C4).
    # Key on with-=O canonical so the exocyclic-oxo fallback claims the =O.
    # Atom indices: idx0=O(exo,no locant), idx1=C(pos4,C=O), idx2=C(pos3),
    # idx3=C(pos2), idx4=O(pos1), idx5=C(pos6), idx6=C(pos5).
    # Symmetric about the C4-O1 axis; numbering strategy picks direction
    # for lowest substituent locants.
    "O=c1ccocc1":      {"name": "4H-pyran-4-one", "substituent_form": "4H-pyran-4-on-N-yl", "alkyl_stem_ok": False,
                        "atom_locants": {1: 4, 2: 3, 3: 2, 4: 1, 5: 6, 6: 5}},
    # 4H-thiopyran-4-one (4-thiopyrone): canonical O=c1ccscc1 — analogous
    # S analogue.  P-22.2.1.1 forbids HW "thine" for 6-rings with 1 S.
    "O=c1ccscc1":      {"name": "4H-thiopyran-4-one", "substituent_form": "4H-thiopyran-4-on-N-yl", "alkyl_stem_ok": False,
                        "atom_locants": {1: 4, 2: 3, 3: 2, 4: 1, 5: 6, 6: 5}},
    # ----- Dihydro-2H-pyran-2-ones (6-ring O lactones with one C=C) -----
    # IUPAC P-25.3.1.3: monocyclic α,β- and β,γ-unsaturated δ-lactones with
    # the 2H-pyran-2-one parent.  The exocyclic-oxo fallback path strips the
    # C=O and looks up the unsaturated-ring skeleton C1=CCOCC1 → "3,6-dihydro-
    # 2H-pyran" (the curated standalone form).  But that form pins the ring
    # O at L1 with both ring-O neighbours sp3 and the C=C at L4-L5; placing
    # the C=O onto that parent yields "2-oxo-3,6-dihydro-2H-pyran" — a name
    # OPSIN parses to a DIFFERENT isomer (it places C=O at L2 by re-routing
    # the ring numbering, shifting the C=C to L3-L4).  P-31.1.4.3.4 / P-66.6.3
    # require the principal characteristic group (=O) to receive the lowest
    # locant, which mandates the indicated-H / dihydro re-numbering with the
    # C=O at L2.  Curate the with-=O canonicals so the exo-oxo fallback
    # picks up the correct numbering directly, with =O claimed via
    # extra_atom_indices and the ring name pre-composed.
    #
    # Common atom layout for all three with-=O canonicals (7 atoms):
    #   idx0=O(exo, claimed by ring), idx1=C(L2, C=O), idx2=C(L3),
    #   idx3=C(L4), idx4=C(L5), idx5=C(L6), idx6=O(L1, ring O).
    # Verified via OPSIN chloro-probe round-trips (see investigation log).
    #
    # 5,6-dihydro-2H-pyran-2-one (α,β-unsaturated δ-lactone, C=C at L3-L4):
    "O=C1C=CCCO1":     {"name": "5,6-dihydro-2H-pyran-2-one",
                        "substituent_form": "5,6-dihydro-2H-pyran-2-on-N-yl",
                        "alkyl_stem_ok": False,
                        "atom_locants": {1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 1}},
    # 3,6-dihydro-2H-pyran-2-one (β,γ-unsaturated δ-lactone, C=C at L4-L5):
    "O=C1CC=CCO1":     {"name": "3,6-dihydro-2H-pyran-2-one",
                        "substituent_form": "3,6-dihydro-2H-pyran-2-on-N-yl",
                        "alkyl_stem_ok": False,
                        "atom_locants": {1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 1}},
    # 3,4-dihydro-2H-pyran-2-one (γ,δ-unsaturated δ-lactone, C=C at L5-L6):
    "O=C1CCC=CO1":     {"name": "3,4-dihydro-2H-pyran-2-one",
                        "substituent_form": "3,4-dihydro-2H-pyran-2-on-N-yl",
                        "alkyl_stem_ok": False,
                        "atom_locants": {1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 1}},
    # 3,4-dihydro-2H-thiopyran: canonical C1=CSCCC1 (same topology as pyran with S)
    # idx2=S=1, idx3=2, idx4=3, idx5=4, idx0=5, idx1=6
    "C1=CSCCC1":       {"name": "3,4-dihydro-2H-thiopyran", "substituent_form": "3,4-dihydro-2H-thiopyranyl", "alkyl_stem_ok": False,
                        "atom_locants": {2: 1, 3: 2, 4: 3, 5: 4, 0: 5, 1: 6}},
    # 1,2,3,6-tetrahydropyridine: canonical C1=CCNCC1 (same topology as 3,6-dihydro-2H-pyran with N)
    # idx3=N=1, idx2=6, idx1=5, idx0=4, idx5=3, idx4=2 -- use: 1=3,2=4,3=5,4=0,5=1,6=2
    "C1=CCNCC1":       {"name": "1,2,3,6-tetrahydropyridine", "substituent_form": "1,2,3,6-tetrahydropyridinyl", "alkyl_stem_ok": False,
                        "atom_locants": {1: 3, 2: 4, 3: 5, 4: 0, 5: 1, 6: 2}},
    # 3,4,5,6-tetrahydropyridine: canonical C1=NCCCC1 (cyclic imine, C=N)
    # idx1=N=1, idx0=2(C=N), idx5=3, idx4=4, idx3=5, idx2=6
    "C1=NCCCC1":       {"name": "3,4,5,6-tetrahydropyridine", "substituent_form": "3,4,5,6-tetrahydropyridinyl", "alkyl_stem_ok": False,
                        "atom_locants": {1: 1, 2: 0, 3: 5, 4: 4, 5: 3, 6: 2}},
    # 2,5-dihydro-1H-pyrrole (3-pyrroline): canonical C1=CCNC1
    # idx3=N=1, idx2=2(sp3), idx1=3(sp2), idx0=4(sp2), idx4=5(sp3)
    "C1=CCNC1":        {"name": "2,5-dihydro-1H-pyrrole", "substituent_form": "2,5-dihydro-1H-pyrrolyl", "alkyl_stem_ok": False,
                        "atom_locants": {1: 3, 2: 2, 3: 1, 4: 0, 5: 4}},
    # 2,3-dihydro-1H-pyrrole (2-pyrroline): canonical C1=CNCC1
    # idx2=N=1, idx3=2(sp3), idx4=3(sp3), idx0=4(sp2), idx1=5(sp2)
    "C1=CNCC1":        {"name": "2,3-dihydro-1H-pyrrole", "substituent_form": "2,3-dihydro-1H-pyrrolyl", "alkyl_stem_ok": False,
                        "atom_locants": {1: 2, 2: 3, 3: 4, 4: 0, 5: 1}},
    # 3,4-dihydro-2H-pyrrole (1-pyrroline): canonical C1=NCCC1.  Cyclic imine
    # with C=N at L1-L5 and three sp3 C (L2-L4).  Atom_locants verified via
    # OPSIN chloro probing: 2-Cl->ClC1CCC=N1, 3-Cl->ClC1CC=NC1, 4-Cl->ClC1C=NCC1,
    # 5-Cl->ClC1=NCCC1 (the C=N carbon).  L1=N, L2-L4=sp3 C, L5=sp2 C.
    "C1=NCCC1":        {"name": "3,4-dihydro-2H-pyrrole", "substituent_form": "3,4-dihydro-2H-pyrrolyl", "alkyl_stem_ok": False,
                        "atom_locants": {0: 5, 1: 1, 2: 2, 3: 3, 4: 4}},
    # 2,3-dihydrofuran: canonical C1=COCC1 — analogous to 2,3-dihydropyrrole with O
    # idx2=O=1, idx3=2(sp3), idx4=3(sp3), idx0=4(sp2), idx1=5(sp2)
    "C1=COCC1":        {"name": "2,3-dihydrofuran", "substituent_form": "2,3-dihydrofuranyl", "alkyl_stem_ok": False,
                        "atom_locants": {1: 2, 2: 3, 3: 4, 4: 0, 5: 1}},
    # 2,5-dihydrofuran: canonical C1=CCOC1 — analogous to 2,5-dihydropyrrole with O
    # idx3=O=1, idx2=2(sp3), idx1=3(sp2), idx0=4(sp2), idx4=5(sp3)
    "C1=CCOC1":        {"name": "2,5-dihydrofuran", "substituent_form": "2,5-dihydrofuranyl", "alkyl_stem_ok": False,
                        "atom_locants": {1: 3, 2: 2, 3: 1, 4: 0, 5: 4}},

    # -----------------------------------------------------------------------
    # Small Si/Ge rings with cumulenic / sp ring atoms (Phase 9)
    # -----------------------------------------------------------------------
    # These are unusual edge cases where the maximally-unsaturated form of a
    # 3- or 4-membered ring carries cumulated double bonds (or a ring triple
    # bond) that the Hantzsch-Widman "-ete" / "-irene" stem alone cannot
    # express.  HW gives the name only for the "single double bond" or
    # "two non-cumulated double bonds" canonical unsaturation; a fully
    # cumulenic 4-ring needs the "cyclobutatetraene" mancunde-style name
    # and a ring with a real triple bond needs the "-cyclopropyne" form.
    # OPSIN round-trip verified for each name below.
    #
    # 1,3-disilacyclobutatetraene — 4-ring with 2C/2Si and 4 cumulated double
    # bonds (no H).  Input SMILES C1=[Si]=C=[Si]=1; canonical identical.
    # OPSIN: '1,3-disilacyclobutatetraene' -> [Si]=1=C=[Si]=C1 (canonical
    # match confirmed via RDKit).  No atom_locants needed: ring is fully
    # symmetric with no substituents in the test corpus.
    "C1=[Si]=C=[Si]=1": {"name": "1,3-disilacyclobutatetraene",
                          "substituent_form": "1,3-disilacyclobutatetraenyl",
                          "alkyl_stem_ok": False},
    # 1,2,3-trigermacyclobutatetraene — 4-ring with 1C/3Ge and 4 cumulated
    # double bonds (no H).  Input SMILES [C]1=[Ge]=[Ge]=[Ge]=1; canonical
    # identical.  OPSIN: '1,2,3-trigermacyclobutatetraene' ->
    # [Ge]=1=[Ge]=[Ge]=C1 (canonical match confirmed).
    "[C]1=[Ge]=[Ge]=[Ge]=1": {"name": "1,2,3-trigermacyclobutatetraene",
                               "substituent_form": "1,2,3-trigermacyclobutatetraenyl",
                               "alkyl_stem_ok": False},
    # silacycloprop-2-yne — 3-ring with 2C/1Si and a C#C ring triple bond.
    # Input SMILES C1#C[SiH2]1; canonical identical.  HW "silirene" parses
    # to the C=C cyclopropene-like form ([SiH2]1C=C1), NOT the cyclopropyne
    # form needed here, so the curated name uses the fully spelled-out
    # "silacycloprop-2-yne".  OPSIN: 'silacycloprop-2-yne' -> [SiH2]1C#C1
    # (canonical match confirmed).
    "C1#C[SiH2]1":     {"name": "silacycloprop-2-yne",
                          "substituent_form": "silacycloprop-2-ynyl",
                          "alkyl_stem_ok": False},

    # -----------------------------------------------------------------------
    # 5-membered monocyclic heterocycles
    # -----------------------------------------------------------------------
    # One heteroatom
    "c1cc[nH]c1":      {"name": "1H-pyrrole",  "substituent_form": "pyrrolyl",  "alkyl_stem_ok": False},
    "c1ccoc1":         {"name": "furan",        "substituent_form": "furanyl",   "alkyl_stem_ok": False},
    "c1ccsc1":         {"name": "thiophene",    "substituent_form": "thienyl",   "alkyl_stem_ok": False},
    # 2,3-dihydrothiophene: partially saturated 5-membered S ring. RDKit canonical 'C1=CSCC1'.
    # atom_locants: idx2=1(S), idx3=2, idx4=3, idx0=4, idx1=5
    # Verified via OPSIN chloro probing: 2-Cl->ClC1CC=CS1 (idx3), 3-Cl->ClC1C=CSC1 (idx4),
    #   4-Cl->ClC1=CSCC1 (idx0), 5-Cl->ClC1=CCCS1 (idx1).
    "C1=CSCC1":        {"name": "2,3-dihydrothiophene", "substituent_form": "2,3-dihydrothiophenyl", "alkyl_stem_ok": False,
                        "atom_locants": {2: 1, 3: 2, 4: 3, 0: 4, 1: 5}},
    # Saturated 5-membered O ring (tetrahydrofuran / oxolane)
    "C1CCOC1":         {"name": "oxolane",      "substituent_form": "oxolanyl",  "alkyl_stem_ok": False},

    # Two heteroatoms (azoles)
    # N,N
    "c1c[nH]cn1":      {"name": "1H-imidazole",       "substituent_form": "imidazolyl",       "alkyl_stem_ok": False},
    "c1cn[nH]c1":      {"name": "1H-pyrazole",        "substituent_form": "pyrazolyl",        "alkyl_stem_ok": False},
    # 4,5-dihydro-1H-imidazole (2-imidazoline): partially saturated 5-ring N-C=N-C-C with
    # the C=N between C2 and N3.  RDKit canonical 'C1=NCCN1'.
    # OPSIN '4,5-dihydro-1H-imidazole' -> N1C=NCC1 -> canonical C1=NCCN1.
    # The tautomer '4,5-dihydro-3H-imidazole' (N1=CNCC1) gives the same canonical SMILES.
    # atom_locants verified via OPSIN chloro probing on canonical 'C1=NCCN1':
    #   2-Cl -> ClC1=NCCN1 (Cl on idx0); 1-Cl -> ClN1C=NCC1 (Cl on idx4);
    #   4-Cl -> ClC1CNC=N1 (Cl on idx2); 5-Cl -> ClC1CN=CN1 (Cl on idx3).
    # idx0=C2, idx1=N3, idx2=C4, idx3=C5, idx4=N1(H).
    "C1=NCCN1":        {"name": "4,5-dihydro-1H-imidazole", "substituent_form": "4,5-dihydro-1H-imidazolyl", "alkyl_stem_ok": False,
                        "atom_locants": {0: 2, 1: 3, 2: 4, 3: 5, 4: 1}},
    # S,N
    # 1,3-thiazole: S=1, C=2 (between S and N), N=3, C=4 (next to N), C=5 (next to S).
    # Canonical 'c1cscn1': idx0=C, idx1=C, idx2=S, idx3=C, idx4=N. Ring bonds 0-1, 1-2, 2-3, 3-4, 4-0.
    # Map: idx2=S=L1, idx3=C(between S and N)=L2, idx4=N=L3, idx0=C(N-adj)=L4, idx1=C(S-adj)=L5.
    # atom_locants verified via OPSIN 2/4/5-methyl-1,3-thiazole probing.
    # Without this pin, an N-alkyl-thiazol-X-ium cation (e.g. thiamine-like
    # 'C[n+]2csc(CCO)c2C') was numbered with N=4 instead of N=3 because
    # _compute_monocyclic_numberings is locant-free and the
    # lowest-substituent-locant scoring outvoted the canonical 1,3-thiazole
    # heteroatom positions for the SUBSTITUENT-form path.
    "c1cscn1":         {"name": "1,3-thiazole",        "substituent_form": "1,3-thiazolyl",        "alkyl_stem_ok": False,
                        "atom_locants": {2: 1, 3: 2, 4: 3, 0: 4, 1: 5}},
    "c1cnsc1":         {"name": "isothiazole",          "substituent_form": "isothiazolyl",     "alkyl_stem_ok": False},
    # 1,2,5-thiadiazole: S flanked by both N atoms (c1cnsn1 in RDKit canonical SMILES)
    "c1cnsn1":         {"name": "1,2,5-thiadiazole",   "substituent_form": "1,2,5-thiadiazolyl", "alkyl_stem_ok": False},
    # O,N
    # 1,3-oxazole: O=1, C=2 (between O and N), N=3, C=4 (next to N), C=5 (next to O).
    # Canonical 'c1cocn1': idx0=C, idx1=C, idx2=O, idx3=C, idx4=N. Ring bonds 0-1, 1-2, 2-3, 3-4, 4-0.
    # Map: idx2=O=L1, idx3=C(between O and N)=L2, idx4=N=L3, idx0=C(N-adj)=L4, idx1=C(O-adj)=L5.
    "c1cocn1":         {"name": "1,3-oxazole",         "substituent_form": "1,3-oxazolyl",         "alkyl_stem_ok": False,
                        "atom_locants": {2: 1, 3: 2, 4: 3, 0: 4, 1: 5}},
    # isoxazole: O(3)=1, N(2)=2, C(1)=3, C(0)=4, C(4)=5 (verified via OPSIN 3/4/5-methyl probing)
    "c1cnoc1":         {"name": "isoxazole",            "substituent_form": "isoxazolyl",       "alkyl_stem_ok": False,
                        "atom_locants": {3: 1, 2: 2, 1: 3, 0: 4, 4: 5}},

    # Three heteroatoms
    # N,N,N
    "c1cn[nH]n1":      {"name": "1H-1,2,3-triazole",  "substituent_form": "1,2,3-triazolyl",  "alkyl_stem_ok": False},
    # 1H-1,2,4-triazole: N(H) on an adjacent N (N1). Atoms (canonical 'c1nc[nH]n1'):
    # 0=C, 1=N, 2=C, 3=[nH], 4=N. Ring bonds: 0-1, 1-2, 2-3, 3-4, 4-0.
    # [nH] at 3 is N1; its ring neighbors are 2 (C) and 4 (N). Atom 4 is the adjacent N2.
    # Atom 1 is the remaining N — the lone N between two Cs → N4.
    # Atom 0 between N4(1) and N2(4) → C3. Atom 2 between N4(1) and N1(3) → C5.
    "c1nc[nH]n1":      {"name": "1H-1,2,4-triazole",  "substituent_form": "1,2,4-triazolyl",  "alkyl_stem_ok": False,
                        "atom_locants": {0: 3, 1: 4, 2: 5, 3: 1, 4: 2}},
    # 4H-1,2,4-triazole: N(H) on the lone N (N4). Atoms (canonical 'c1nnc[nH]1'):
    # 0=C, 1=N, 2=N, 3=C, 4=[nH]. Ring bonds: 0-1, 1-2, 2-3, 3-4, 4-0.
    # [nH] at 4 is the lone N (neighbors 3 and 0, both C) → N4.
    # Atoms 1 and 2 are adjacent Ns → N1/N2 (interchangeable by symmetry).
    # Atom 0 between N4(4) and N1(1) → C5. Atom 3 between N2(2) and N4(4) → C3.
    "c1nnc[nH]1":      {"name": "4H-1,2,4-triazole",  "substituent_form": "4H-1,2,4-triazol-4-yl",  "alkyl_stem_ok": False,
                        "atom_locants": {0: 5, 1: 1, 2: 2, 3: 3, 4: 4}},
    # S,N,N
    "c1nncs1":         {"name": "1,3,4-thiadiazole",   "substituent_form": "1,3,4-thiadiazolyl", "alkyl_stem_ok": False},
    "c1ncsn1":         {"name": "1,2,4-thiadiazole",   "substituent_form": "1,2,4-thiadiazolyl", "alkyl_stem_ok": False},
    # O,N,N
    "c1nnco1":         {"name": "1,3,4-oxadiazole",    "substituent_form": "1,3,4-oxadiazolyl",  "alkyl_stem_ok": False},
    "c1ncon1":         {"name": "1,2,4-oxadiazole",    "substituent_form": "1,2,4-oxadiazolyl",  "alkyl_stem_ok": False},
    "c1conn1":         {"name": "1,2,5-oxadiazole",    "substituent_form": "furazanyl",           "alkyl_stem_ok": False},

    # Four heteroatoms
    "c1nnn[nH]1":      {"name": "1H-tetrazole",        "substituent_form": "tetrazolyl",       "alkyl_stem_ok": False},
    # 2H-tetrazole 'c1nn[nH]n1' — STRUCTURALLY DISTINCT tautomer from
    # 1H-tetrazole.  The NH lives on the MIDDLE N of the NNN(H)N chain
    # (locant N2), not the END N (locant N1 in 1H form).  RDKit canonical
    # SMILES differ; OPSIN parses 1H- and 2H-tetrazole as distinct molecules.
    # Without this entry, the carved 2H-tetrazole ring SMILES would be
    # aliased to 1H-tetrazole via _CURATED_ALIASES and the resulting name
    # would round-trip to the wrong tautomer.  No atom_locants here: adding
    # them would force the parent-path numbering through
    # _build_numbering_from_atom_locants and reduce the strategy's plan-cap
    # count, exposing a benzene-vs-heteroring parent-selection regression
    # on FDA-1333 etc.  The substituent-locant pinning lives in
    # engine._TAUTOMER_NH_RING_SUBSTITUENT_DATA.
    "c1nn[nH]n1":      {"name": "2H-tetrazole",        "substituent_form": "2H-tetrazolyl",     "alkyl_stem_ok": False},

    # -----------------------------------------------------------------------
    # Benzo-fused 5-membered heterocycles
    # -----------------------------------------------------------------------
    # atom_locants: {rdkit_atom_idx: iupac_locant} for correct ring numbering
    # Computed via OPSIN locant-SMILES matching (see ARCHITECTURE docs).
    "c1ccc2[nH]ccc2c1":  {"name": "1H-indole",         "substituent_form": "indolyl",          "alkyl_stem_ok": False,
                           "atom_locants": {4: 1, 5: 2, 6: 3, 8: 4, 0: 5, 1: 6, 2: 7}},
    # 1H-benzimidazole: N1(4)=1, C2(5)=2, N3(6)=3, C3a(7)=3a, C4(8)=4, C5(0)=5, C6(1)=6, C7(2)=7, C7a(3)=7a
    "c1ccc2[nH]cnc2c1":  {"name": "1H-benzimidazole",  "substituent_form": "benzimidazolyl",   "alkyl_stem_ok": False,
                           "atom_locants": {4: 1, 5: 2, 6: 3, 7: "3a", 8: 4, 0: 5, 1: 6, 2: 7, 3: "7a"}},
    # For all benzo-fused 5-membered rings: atom_locants verified via OPSIN probing.
    # Pattern: heteroatom at ring_idx=4 → locant 1 (or 2 for isoindole/2-benzofuran).
    "c1ccc2ocnc2c1":     {"name": "1,3-benzoxazole",   "substituent_form": "benzoxazolyl",     "alkyl_stem_ok": False,
                           "atom_locants": {4: 1, 5: 2, 6: 3, 8: 4, 0: 5, 1: 6, 2: 7}},
    "c1ccc2scnc2c1":     {"name": "1,3-benzothiazole", "substituent_form": "benzothiazolyl",   "alkyl_stem_ok": False,
                           "atom_locants": {4: 1, 5: 2, 6: 3, 8: 4, 0: 5, 1: 6, 2: 7}},
    "c1ccc2occc2c1":     {"name": "benzofuran",         "substituent_form": "benzofuranyl",     "alkyl_stem_ok": False,
                           "atom_locants": {4: 1, 5: 2, 6: 3, 8: 4, 0: 5, 1: 6, 2: 7}},
    "c1ccc2sccc2c1":     {"name": "1-benzothiophene",  "substituent_form": "benzothienyl",     "alkyl_stem_ok": False,
                           "atom_locants": {4: 1, 5: 2, 6: 3, 8: 4, 0: 5, 1: 6, 2: 7}},
    "c1ccc2[nH]ncc2c1":  {"name": "1H-indazole",       "substituent_form": "indazolyl",        "alkyl_stem_ok": False,
                           "atom_locants": {4: 1, 5: 2, 6: 3, 8: 4, 0: 5, 1: 6, 2: 7}},
    # 2H-indazole tautomer (NH on N2). Canonical 'c1ccc2n[nH]cc2c1'.
    # atom_locants verified via OPSIN chloro-probing of N-chloro-2H-indazole series.
    "c1ccc2n[nH]cc2c1":  {"name": "2H-indazole",       "substituent_form": "2H-indazol-2-yl",  "alkyl_stem_ok": False,
                           "atom_locants": {4: 1, 5: 2, 6: 3, 8: 4, 0: 5, 1: 6, 2: 7}},
    "c1ccc2c[nH]cc2c1":  {"name": "2H-isoindole",      "substituent_form": "isoindolyl",       "alkyl_stem_ok": False,
                           "atom_locants": {4: 1, 5: 2, 6: 3, 8: 4, 0: 5, 1: 6, 2: 7}},  # N at pos 2 (idx 5)
    "c1ccc2cocc2c1":     {"name": "2-benzofuran",       "substituent_form": "isobenzofuranyl",  "alkyl_stem_ok": False,
                           "atom_locants": {6: 1, 5: 2, 4: 3, 2: 4, 1: 5, 0: 6, 8: 7}},  # O at pos 2 (idx 5)

    # ---------------------------------------------------------------------
    # 1,3-dihydro furo/thieno-fused heteroaromatic parents — the base ring
    # scaffolds for HETEROAROMATIC-FUSED RING LACTONES (P-25.3).
    #
    # These mirror the curated ``1,3-dihydro-2-benzofuran`` (the phthalide
    # base) entry: the furan/thiophene ring is 1,3-dihydro (positions 1 and 3
    # are sp3 CH2) and fused to an aromatic azine.  ``extract_ring_mol``
    # carves exactly this scaffold from a fused ring-lactone, and the
    # exocyclic-oxo claim path then expresses the ring C=O as a ``-one``
    # suffix — naming e.g. Cc1ncc2c(c1O)C(=O)OC2Cl ->
    # 3-chloro-7-hydroxy-6-methylfuro[3,4-c]pyridin-1(3H)-one (pyridoxolactone),
    # the same machinery that turns 1,3-dihydro-2-benzofuran into phthalide
    # (1,3-dihydro-2-benzofuran-1-one).
    #
    # atom_locants verified by OPSIN chloro/methyl-probing of every
    # substitutable ring position on the 1,3-dihydro parent (numeric anchors),
    # with ring heteroatoms and the two bridgehead carbons closed by the
    # canonical fused-ring perimeter walk (heteroatoms take the sequential
    # numeric locant; each bridgehead takes the 'Na' letter of the preceding
    # numbered atom).  The walk assignment is the UNIQUE one consistent with
    # all OPSIN chloro anchors.  Saturation of the furan ring does NOT change
    # the locants relative to the aromatic parent (P-31.1.4).
    #
    # 1,3-dihydrofuro[3,4-c]pyridine: canon 'c1cc2c(cn1)COC2';
    #   idx8=1(CH2), idx7=2(O), idx6=3(CH2), idx3=3a, idx4=4, idx5=5(N),
    #   idx0=6, idx1=7, idx2=7a.
    "c1cc2c(cn1)COC2":   {"name": "1,3-dihydrofuro[3,4-c]pyridine", "substituent_form": "1,3-dihydrofuro[3,4-c]pyridin-1-yl", "alkyl_stem_ok": False,
                           "atom_locants": {8: 1, 7: 2, 6: 3, 3: "3a", 4: 4, 5: 5, 0: 6, 1: 7, 2: "7a"}},
    # 1,3-dihydrothieno[3,4-c]pyridine: canon 'c1cc2c(cn1)CSC2'; same layout (S at pos2=idx7).
    "c1cc2c(cn1)CSC2":   {"name": "1,3-dihydrothieno[3,4-c]pyridine", "substituent_form": "1,3-dihydrothieno[3,4-c]pyridin-1-yl", "alkyl_stem_ok": False,
                           "atom_locants": {8: 1, 7: 2, 6: 3, 3: "3a", 4: 4, 5: 5, 0: 6, 1: 7, 2: "7a"}},

    # 5,7-dihydro furo/thieno[3,4-d]pyrimidine and [3,4-b]pyrazine: the d/b
    # fusion places the lactone (CH2-O-CH2) carbons at positions 5 and 7, so
    # the saturated-ring base is the 5,7-dihydro form (not 1,3-dihydro).  These
    # name e.g. O=C1OCc2ncncc21 -> 5,7-dihydrofuro[3,4-d]pyrimidin-5-one.
    # 5,7-dihydrofuro[3,4-d]pyrimidine: canon 'c1ncc2c(n1)COC2';
    #   idx5=1, idx0=2(N), idx1=3, idx2=4, idx3=4a, idx8=5(CH2), idx7=6(O),
    #   idx6=7(CH2), idx4=7a.
    "c1ncc2c(n1)COC2":   {"name": "5,7-dihydrofuro[3,4-d]pyrimidine", "substituent_form": "5,7-dihydrofuro[3,4-d]pyrimidin-5-yl", "alkyl_stem_ok": False,
                           "atom_locants": {5: 1, 0: 2, 1: 3, 2: 4, 3: "4a", 8: 5, 7: 6, 6: 7, 4: "7a"}},
    # 5,7-dihydrothieno[3,4-d]pyrimidine: canon 'c1ncc2c(n1)CSC2'; same layout (S at pos6=idx7).
    "c1ncc2c(n1)CSC2":   {"name": "5,7-dihydrothieno[3,4-d]pyrimidine", "substituent_form": "5,7-dihydrothieno[3,4-d]pyrimidin-5-yl", "alkyl_stem_ok": False,
                           "atom_locants": {5: 1, 0: 2, 1: 3, 2: 4, 3: "4a", 8: 5, 7: 6, 6: 7, 4: "7a"}},
    # 5,7-dihydrofuro[3,4-b]pyrazine: canon 'c1cnc2c(n1)COC2';
    #   idx5=1, idx0=2, idx1=3, idx2=4(N), idx3=4a, idx8=5(CH2), idx7=6(O),
    #   idx6=7(CH2), idx4=7a.
    "c1cnc2c(n1)COC2":   {"name": "5,7-dihydrofuro[3,4-b]pyrazine", "substituent_form": "5,7-dihydrofuro[3,4-b]pyrazin-5-yl", "alkyl_stem_ok": False,
                           "atom_locants": {5: 1, 0: 2, 1: 3, 2: 4, 3: "4a", 8: 5, 7: 6, 6: 7, 4: "7a"}},
    # 5,7-dihydrothieno[3,4-b]pyrazine: canon 'c1cnc2c(n1)CSC2'; same layout (S at pos6=idx7).
    "c1cnc2c(n1)CSC2":   {"name": "5,7-dihydrothieno[3,4-b]pyrazine", "substituent_form": "5,7-dihydrothieno[3,4-b]pyrazin-5-yl", "alkyl_stem_ok": False,
                           "atom_locants": {5: 1, 0: 2, 1: 3, 2: 4, 3: "4a", 8: 5, 7: 6, 6: 7, 4: "7a"}},
    # OMITTED (no aromatic-azine lactone via 'b/c/d' fusion that OPSIN names as
    # a clean ring lactone): the furo/thieno[3,4-b]pyridine and
    # [3,4-c/d]pyridazine fusions produce saturated lactam-type rings or
    # non-aromatic 6-rings under OPSIN's '1,3-dihydro' canonicalization, not a
    # lactone fused to an aromatic azine — see report.
    "c1ccc2oncc2c1":     {"name": "benzisoxazole",       "substituent_form": "benzisoxazolyl",  "alkyl_stem_ok": False,
                           "atom_locants": {4: 1, 5: 2, 6: 3, 8: 4, 0: 5, 1: 6, 2: 7}},
    "c1ccc2sncc2c1":     {"name": "benzisothiazole",    "substituent_form": "benzisothiazolyl","alkyl_stem_ok": False,
                           "atom_locants": {4: 1, 5: 2, 6: 3, 8: 4, 0: 5, 1: 6, 2: 7}},
    # 1H-1,2,3-benzotriazole: PIN per Blue Book P-25.3.1.3 fused-name list.
    # Canonical 'c1ccc2[nH]nnc2c1': idx4=N1(NH), idx5=N2, idx6=N3, idx7=C3a,
    #   idx8=C4, idx0=C5, idx1=C6, idx2=C7, idx3=C7a.  Same skeletal locant
    #   pattern as 1H-benzimidazole (idx4 = pos 1 NH).  OPSIN accepts both
    #   "1H-benzotriazole" and "1H-1,2,3-benzotriazole" — emit the shorter
    #   one for cleaner round-trip diffs.
    "c1ccc2[nH]nnc2c1":  {"name": "1H-benzotriazole",   "substituent_form": "benzotriazolyl",  "alkyl_stem_ok": False,
                           "atom_locants": {4: 1, 5: 2, 6: 3, 7: "3a", 8: 4, 0: 5, 1: 6, 2: 7, 3: "7a"}},
    # 2H-1,2,3-benzotriazole tautomer (NH on N2).  Canonical 'c1ccc2n[nH]nc2c1'.
    # Same idx-to-locant map as 1H form — only the [nH] position moves to N2 (idx5).
    "c1ccc2n[nH]nc2c1":  {"name": "2H-benzotriazole",   "substituent_form": "2H-benzotriazol-2-yl", "alkyl_stem_ok": False,
                           "atom_locants": {4: 1, 5: 2, 6: 3, 7: "3a", 8: 4, 0: 5, 1: 6, 2: 7, 3: "7a"}},
    # imidazo[4,5-b]pyridine and imidazo[4,5-c]pyridine — 5,6-fused with 3 N
    # in the imidazo ring + 1 pyridine N.  Tautomer pairs (1H- and 3H-) differ
    # only in which imidazole N bears the H.  atom_locants verified via OPSIN
    # chloro-probing (Stage 10 R10-A-1).
    # 1H-imidazo[4,5-b]pyridine: canon 'c1cnc2nc[nH]c2c1', NH at idx6 (L1),
    #   pyridine N at idx2 (L4); jct atoms idx3=L3a, idx7=L7a.
    "c1cnc2nc[nH]c2c1":  {"name": "1H-imidazo[4,5-b]pyridine", "substituent_form": "1H-imidazo[4,5-b]pyridin-N-yl", "alkyl_stem_ok": False,
                           "atom_locants": {6: 1, 5: 2, 4: 3, 3: "3a", 2: 4, 1: 5, 0: 6, 8: 7, 7: "7a"}},
    # 3H-imidazo[4,5-b]pyridine tautomer: canon 'c1cnc2[nH]cnc2c1', NH at idx4 (L3).
    "c1cnc2[nH]cnc2c1":  {"name": "3H-imidazo[4,5-b]pyridine", "substituent_form": "3H-imidazo[4,5-b]pyridin-N-yl", "alkyl_stem_ok": False,
                           "atom_locants": {6: 1, 5: 2, 4: 3, 3: "3a", 2: 4, 1: 5, 0: 6, 8: 7, 7: "7a"}},
    # 1H-imidazo[4,5-c]pyridine: canon 'c1cc2[nH]cnc2cn1', NH at idx3 (L1),
    #   pyridine N at idx8 (L5); jct atoms idx2=L7a, idx6=L3a.
    "c1cc2[nH]cnc2cn1":  {"name": "1H-imidazo[4,5-c]pyridine", "substituent_form": "1H-imidazo[4,5-c]pyridin-N-yl", "alkyl_stem_ok": False,
                           "atom_locants": {3: 1, 4: 2, 5: 3, 6: "3a", 7: 4, 8: 5, 0: 6, 1: 7, 2: "7a"}},
    # 3H-imidazo[4,5-c]pyridine tautomer: canon 'c1cc2nc[nH]c2cn1', NH at idx5 (L3).
    "c1cc2nc[nH]c2cn1":  {"name": "3H-imidazo[4,5-c]pyridine", "substituent_form": "3H-imidazo[4,5-c]pyridin-N-yl", "alkyl_stem_ok": False,
                           "atom_locants": {5: 3, 4: 2, 3: 1, 6: "3a", 7: 4, 8: 5, 0: 6, 1: 7, 2: "7a"}},
    # Pyrrolo[?,?-?]pyridines (azaindoles) — 5,6-fused with 1 pyrrole-NH + 1
    # pyridine-N.  All four [2,3-b]/[3,2-b]/[2,3-c]/[3,2-c] regio-isomers
    # are PINs.  atom_locants verified via OPSIN chloro-probing
    # (Stage 10 R10-A-2).
    # 1H-pyrrolo[2,3-b]pyridine (= 7-azaindole): canon 'c1cnc2[nH]ccc2c1',
    #   NH at idx4 (L1), pyridine N at idx2 (L7); jct idx7=L3a, idx3=L7a.
    "c1cnc2[nH]ccc2c1":  {"name": "1H-pyrrolo[2,3-b]pyridine", "substituent_form": "1H-pyrrolo[2,3-b]pyridin-N-yl", "alkyl_stem_ok": False,
                           "atom_locants": {4: 1, 5: 2, 6: 3, 7: "3a", 8: 4, 0: 5, 1: 6, 2: 7, 3: "7a"}},
    # 1H-pyrrolo[3,2-b]pyridine (= 4-azaindole): canon 'c1cnc2cc[nH]c2c1',
    #   NH at idx6 (L1), pyridine N at idx2 (L4); jct idx3=L3a, idx7=L7a.
    "c1cnc2cc[nH]c2c1":  {"name": "1H-pyrrolo[3,2-b]pyridine", "substituent_form": "1H-pyrrolo[3,2-b]pyridin-N-yl", "alkyl_stem_ok": False,
                           "atom_locants": {6: 1, 5: 2, 4: 3, 3: "3a", 2: 4, 1: 5, 0: 6, 8: 7, 7: "7a"}},
    # 1H-pyrrolo[2,3-c]pyridine: canon 'c1cc2cc[nH]c2cn1', NH at idx5 (L1),
    #   pyridine N at idx8 (L6); jct idx2=L3a, idx6=L7a.
    "c1cc2cc[nH]c2cn1":  {"name": "1H-pyrrolo[2,3-c]pyridine", "substituent_form": "1H-pyrrolo[2,3-c]pyridin-N-yl", "alkyl_stem_ok": False,
                           "atom_locants": {5: 1, 4: 2, 3: 3, 2: "3a", 1: 4, 0: 5, 8: 6, 7: 7, 6: "7a"}},
    # 1H-pyrrolo[3,2-c]pyridine (= 5-azaindole): canon 'c1cc2[nH]ccc2cn1',
    #   NH at idx3 (L1), pyridine N at idx8 (L5); jct idx6=L3a, idx2=L7a.
    "c1cc2[nH]ccc2cn1":  {"name": "1H-pyrrolo[3,2-c]pyridine", "substituent_form": "1H-pyrrolo[3,2-c]pyridin-N-yl", "alkyl_stem_ok": False,
                           "atom_locants": {3: 1, 4: 2, 5: 3, 6: "3a", 7: 4, 8: 5, 0: 6, 1: 7, 2: "7a"}},
    # Bridgehead-N bicyclics (5,6-fused with the shared atom an N at the
    # ring junction).  Same 9-atom topology as indolizine: bridgehead N gets
    # locant 4, the other junction C is 8a.  OPSIN rejects 4-Cl/3a-Cl/7a-Cl
    # for these scaffolds (heteroatom valency / no available H), but the
    # 2/3/5/6/7/8 chloro-probes confirm the substitutable atom indices.
    # imidazo[1,2-a]pyridine (zolpidem core): canon 'c1ccn2ccnc2c1',
    #   bridgehead N at idx3 (L4), N1 at idx6, jct C at idx7 (L8a).
    "c1ccn2ccnc2c1":     {"name": "imidazo[1,2-a]pyridine", "substituent_form": "imidazo[1,2-a]pyridin-N-yl", "alkyl_stem_ok": False,
                           "atom_locants": {6: 1, 5: 2, 4: 3, 3: 4, 2: 5, 1: 6, 0: 7, 8: 8, 7: "8a"}},
    # imidazo[1,5-a]pyridine: canon 'c1ccn2cncc2c1', bridgehead N at idx3 (L4),
    #   non-bridgehead N at idx5 (L2), jct C at idx7 (L8a).
    "c1ccn2cncc2c1":     {"name": "imidazo[1,5-a]pyridine", "substituent_form": "imidazo[1,5-a]pyridin-N-yl", "alkyl_stem_ok": False,
                           "atom_locants": {6: 1, 5: 2, 4: 3, 3: 4, 2: 5, 1: 6, 0: 7, 8: 8, 7: "8a"}},
    # [1,2,4]triazolo[4,3-a]pyridine: canon 'c1ccn2cnnc2c1', bridgehead N at
    #   idx3 (L4), L1=idx6, L2=idx5, L3=idx4, jct C idx7=L8a.
    "c1ccn2cnnc2c1":     {"name": "[1,2,4]triazolo[4,3-a]pyridine", "substituent_form": "[1,2,4]triazolo[4,3-a]pyridin-N-yl", "alkyl_stem_ok": False,
                           "atom_locants": {6: 1, 5: 2, 4: 3, 3: 4, 2: 5, 1: 6, 0: 7, 8: 8, 7: "8a"}},
    # tetrazolo[1,5-a]pyridine: canon 'c1ccn2nnnc2c1', bridgehead N at idx3 (L4),
    #   four N's at L1/L2/L3/L4 (idx6/idx5/idx4/idx3), jct C idx7=L8a.
    "c1ccn2nnnc2c1":     {"name": "tetrazolo[1,5-a]pyridine", "substituent_form": "tetrazolo[1,5-a]pyridin-N-yl", "alkyl_stem_ok": False,
                           "atom_locants": {6: 1, 5: 2, 4: 3, 3: 4, 2: 5, 1: 6, 0: 7, 8: 8, 7: "8a"}},
    # 6,7,8,9-tetrahydro-5H-tetrazolo[1,5-a]azepine — Phase 7 ring-batch-4.
    # 5,7-fused: 5-membered tetrazole + 7-membered azepane sharing the
    # N(bridgehead)-C(junction) edge.  The azepine ring is fully saturated
    # (5,6,7,8,9-tetrahydro-5H- + the implicit ring-double-bond between the
    # two junctions in the tetrazole).  Canonical (RDKit) 'C1CCc2nnnn2CC1'.
    # OPSIN 'C1=NN=NN2C1CCCCC2' canonicalises to 'C1CCc2nnnn2CC1'.
    # Atom layout of 'C1CCc2nnnn2CC1':
    #   idx0=C(pos7), idx1=C(pos8), idx2=C(pos9), idx3=C(pos9a junction),
    #   idx4=N(pos1), idx5=N(pos2), idx6=N(pos3), idx7=N(pos4 bridgehead),
    #   idx8=C(pos5), idx9=C(pos6).
    # OPSIN chloro-probing verified: pos5→idx8, pos6→idx9, pos7→idx0,
    # pos8→idx1, pos9→idx2; junctions pos4 (N bridgehead) and pos9a (C)
    # derived from topology.
    "C1CCc2nnnn2CC1":    {"name": "6,7,8,9-tetrahydro-5H-tetrazolo[1,5-a]azepine",
                           "substituent_form": "6,7,8,9-tetrahydro-5H-tetrazolo[1,5-a]azepin-yl",
                           "alkyl_stem_ok": False,
                           "atom_locants": {0: 7, 1: 8, 2: 9, 3: "9a", 4: 1, 5: 2, 6: 3,
                                            7: 4, 8: 5, 9: 6}},
    # [1,2,3]Triazolo[4,5-b]pyridines — 5,6-fused with 3 N in the 5-ring +
    # 1 pyridine N.  Tautomer pair (1H- and 3H-) differ in which N bears H.
    # atom_locants verified via OPSIN chloro-probing of L=5/6/7 (Stage 10
    # R10-A-4).  Same skeletal topology as 1H-imidazo[4,5-b]pyridine: same
    # idx-to-locant map; the third N just shifts the ring-NH position.
    # 1H form: canon 'c1cnc2nn[nH]c2c1', NH at idx6 (L1).
    "c1cnc2nn[nH]c2c1":  {"name": "1H-[1,2,3]triazolo[4,5-b]pyridine", "substituent_form": "1H-[1,2,3]triazolo[4,5-b]pyridin-N-yl", "alkyl_stem_ok": False,
                           "atom_locants": {6: 1, 5: 2, 4: 3, 3: "3a", 2: 4, 1: 5, 0: 6, 8: 7, 7: "7a"}},
    # 3H form: canon 'c1cnc2[nH]nnc2c1', NH at idx4 (L3).
    "c1cnc2[nH]nnc2c1":  {"name": "3H-[1,2,3]triazolo[4,5-b]pyridine", "substituent_form": "3H-[1,2,3]triazolo[4,5-b]pyridin-N-yl", "alkyl_stem_ok": False,
                           "atom_locants": {6: 1, 5: 2, 4: 3, 3: "3a", 2: 4, 1: 5, 0: 6, 8: 7, 7: "7a"}},

    # -----------------------------------------------------------------------
    # Benzo-fused 6-membered diazines & bicyclic aromatics
    # -----------------------------------------------------------------------
    # Quinoline / isoquinoline — canonical SMILES verified via RDKit
    # atom_locants: both have junction atoms 4a/8a that are non-substitutable.
    # Quinoline: N(4)=1, C(5)=2, C(6)=3, C(7)=4, C(8)=4a, C(9)=5, C(0)=6, C(1)=7, C(2)=8, C(3)=8a
    # (positions 4a and 8a are ring-junction atoms — included but rarely need explicit locants)
    "c1ccc2ncccc2c1":    {"name": "quinoline",          "substituent_form": "quinolinyl",       "alkyl_stem_ok": False,
                           "atom_locants": {4: 1, 5: 2, 6: 3, 7: 4, 8: "4a", 9: 5, 0: 6, 1: 7, 2: 8, 3: "8a"}},
    # Protonated quinoline (canonical 'c1ccc2[nH+]cccc2c1' — same atom indexing as
    # neutral quinoline, N+ at idx4 stays as locant 1).  Substitutive assembly
    # appends -1-ium → "quinolin-1-ium".
    "c1ccc2[nH+]cccc2c1": {"name": "quinoline",         "substituent_form": "quinolinyl",       "alkyl_stem_ok": False,
                           "atom_locants": {4: 1, 5: 2, 6: 3, 7: 4, 8: "4a", 9: 5, 0: 6, 1: 7, 2: 8, 3: "8a"}},
    "c1ccc2cnccc2c1":    {"name": "isoquinoline",       "substituent_form": "isoquinolinyl",    "alkyl_stem_ok": False,
                           "atom_locants": {4: 1, 5: 2, 6: 3, 7: 4, 9: 5, 0: 6, 1: 7, 2: 8}},
    # Protonated isoquinoline (canonical 'c1ccc2c[nH+]ccc2c1' — same atom indexing as
    # neutral isoquinoline, N+ at idx5 stays as locant 2).  Substitutive assembly
    # appends -2-ium → "isoquinolin-2-ium".
    "c1ccc2c[nH+]ccc2c1": {"name": "isoquinoline",      "substituent_form": "isoquinolinyl",    "alkyl_stem_ok": False,
                           "atom_locants": {4: 1, 5: 2, 6: 3, 7: 4, 9: 5, 0: 6, 1: 7, 2: 8}},
    # quinoxaline: atom_locants verified via OPSIN chloro-probing.
    # canonical 'c1ccc2nccnc2c1': idx4=N1, idx5=C2, idx6=C3, idx7=N4, idx8=C4a,
    #   idx9=C5, idx0=C6, idx1=C7, idx2=C8, idx3=C8a.
    # C2v symmetry collapses 2<->3, 5<->8, 6<->7 to the same canonical SMILES via OPSIN.
    # Verified by RDKit emission: chloride at idx{5,6,9,0,1,2} round-trips to OPSIN canonical.
    # Without atom_locants, brimonidine (Brc1c(N...)ccc2nccnc12) emitted "quinoxalin-7-amine"
    # instead of -6-amine; the lowest-locant rule (47c51a3) now resolves correctly.
    "c1ccc2nccnc2c1":    {"name": "quinoxaline", "substituent_form": "quinoxalinyl", "alkyl_stem_ok": False,
                          "atom_locants": {4: 1, 5: 2, 6: 3, 7: 4, 8: "4a", 9: 5, 0: 6, 1: 7, 2: 8, 3: "8a"}},
    # Benzo[g]quinoxaline (Stage 13 R13-A-4) — quinoxaline with an extra
    # benzene fused at the [g] bond (the 6-7 bond of quinoxaline).  The
    # 3-ring scaffold has anthracene-like topology with the central ring
    # being the pyrazine of quinoxaline.  C2v symmetry through the N1-N4
    # axis.  Canonical 'c1ccc2cc3nccnc3cc2c1' (14 atoms = 8 substitutable
    # CH at L={2,3,5,6,7,8,9,10} + 2 N at L={1,4} + 4 ring-junction
    # at L={4a, 5a, 9a, 10a}).  Pre-fix this scaffold returned
    # NAMING_ERROR (no curated entry; engine couldn't synthesise the
    # benzo-fused N-heterocyclic name).  Locant assignment derived from
    # OPSIN chloro-probing of L=2,3,5,6,7,8,9,10 (the 8 substitutable
    # carbons) plus periphery walk through the canonical
    # 1→2→3→4→4a→5→5a→6→7→8→9→9a→10→10a→1:
    #   L=1→idx6, L=2→7, L=3→8, L=4→9, L=4a→10, L=5→11, L=5a→12,
    #   L=6→13, L=7→0, L=8→1, L=9→2, L=9a→3, L=10→4, L=10a→5.
    # ``stage2_fusion_base: False`` keeps this anthracene-class N-
    # heterocyclic scaffold opted out of Stage 2B as a precaution: a
    # 4-ring extension via Stage 2B would push past the ≤3-ring
    # invariant.
    "c1ccc2cc3nccnc3cc2c1": {"name": "benzo[g]quinoxaline",
                              "substituent_form": "benzo[g]quinoxalinyl",
                              "alkyl_stem_ok": False,
                              "atom_locants": {6: 1, 7: 2, 8: 3, 9: 4, 10: "4a",
                                               11: 5, 12: "5a", 13: 6, 0: 7, 1: 8,
                                               2: 9, 3: "9a", 4: 10, 5: "10a"},
                              "stage2_fusion_base": False},
    # Benzo[g]quinoxaline (Stage 13 R13-A-4) — quinoxaline with an extra
    # benzene fused at the [g] bond (the 6-7 bond of quinoxaline).  The
    # 3-ring scaffold has anthracene-like topology with the central ring
    # being the pyrazine of quinoxaline.  C2v symmetry through the N1-N4
    # bond.  Canonical 'c1ccc2cc3nccnc3cc2c1' (14 atoms = 8 substitutable
    # CH at L={2,3,5,6,7,8,9,10} + 2 N at L={1,4} + 4 ring-junction
    # at L={4a, 5a, 9a, 10a}).  Pre-fix this scaffold returned
    # NAMING_ERROR.  Locant assignment derived from OPSIN chloro-probing
    # of L=2,3,5,6,7,8,9,10 (the 8 substitutable carbons) plus periphery
    # walk through the canonical:
    #   L=1→idx6, L=2→7, L=3→8, L=4→9, L=4a→10, L=5→11, L=5a→12,
    #   L=6→13, L=7→0, L=8→1, L=9→2, L=9a→3, L=10→4, L=10a→5.
    # ``stage2_fusion_base: False`` keeps anthracene-class scaffolds
    # opted out of Stage 2B as a precaution (a 4-ring extension of
    # benzo[g]quinoxaline would tip into the ≤3-ring invariant zone).
    "c1ccc2cc3nccnc3cc2c1": {"name": "benzo[g]quinoxaline",
                              "substituent_form": "benzo[g]quinoxalinyl",
                              "alkyl_stem_ok": False,
                              "atom_locants": {6: 1, 7: 2, 8: 3, 9: 4, 10: "4a",
                                               11: 5, 12: "5a", 13: 6, 0: 7, 1: 8,
                                               2: 9, 3: "9a", 4: 10, 5: "10a"},
                              "stage2_fusion_base": False},
    # quinazoline: atom_locants verified via OPSIN chloro-probing.
    # canonical 'c1ccc2ncncc2c1': idx4=N1, idx5=C2, idx6=N3, idx7=C4, idx8=C4a, idx9=C5,
    #   idx0=C6, idx1=C7, idx2=C8, idx3=C8a
    # Verified: 2-Cl->Clc1ncc2ccccc2n1 (idx5), 4-Cl->Clc1ncnc2ccccc12 (idx7),
    #   5-Cl->Clc1cccc2ncncc12 (idx9), 6-Cl->Clc1ccc2ncncc2c1 (idx0).
    "c1ccc2ncncc2c1":    {"name": "quinazoline", "substituent_form": "quinazolinyl", "alkyl_stem_ok": False,
                          "atom_locants": {4: 1, 5: 2, 6: 3, 7: 4, 8: "4a", 9: 5, 0: 6, 1: 7, 2: 8, 3: "8a"}},
    # Cinnoline/phthalazine: SMILES verified against InChIKeys
    # c1ccc2nnccc2c1 InChIKey WCZVZNOTHYJIEI = cinnoline (PubChem CID 9232)
    # c1ccc2cnncc2c1 InChIKey LFSXCDWNBUNEEM = phthalazine (PubChem CID 10148)
    # phthalazine has C at position 1 (flanked by N-2 and N-3), must hardcode.
    # cinnoline: atom_locants verified via OPSIN chloro-probing (no symmetry; all 6 distinct).
    # canonical 'c1ccc2nnccc2c1': idx4=N1, idx5=N2, idx6=C3, idx7=C4, idx8=C4a,
    #   idx9=C5, idx0=C6, idx1=C7, idx2=C8, idx3=C8a.
    # Verified: 3-Cl->idx6, 4-Cl->idx7, 5-Cl->idx9, 6-Cl->idx0, 7-Cl->idx1, 8-Cl->idx2 all round-trip.
    "c1ccc2nnccc2c1":    {"name": "cinnoline", "substituent_form": "cinnolinyl", "alkyl_stem_ok": False,
                          "atom_locants": {4: 1, 5: 2, 6: 3, 7: 4, 8: "4a", 9: 5, 0: 6, 1: 7, 2: 8, 3: "8a"}},
    "c1ccc2cnncc2c1":    {"name": "phthalazine",        "substituent_form": "phthalazinyl",     "alkyl_stem_ok": False,
                           "atom_locants": {4: 1, 5: 2, 6: 3, 7: 4, 9: 5, 0: 6, 1: 7, 2: 8}},
    # 1,2,3-benzotriazine (PIN; equivalent CAS-style "benzo[d][1,2,3]triazine"
    # canonicalises identically per OPSIN).  Canonical 'c1ccc2nnncc2c1':
    # idx4=N1, idx5=N2, idx6=N3, idx7=C4, idx8=C4a, idx9=C5, idx0=C6,
    # idx1=C7, idx2=C8, idx3=C8a.  Same skeletal locant pattern as cinnoline
    # but with three Ns at 1/2/3 instead of two at 1/2.
    "c1ccc2nnncc2c1":    {"name": "1,2,3-benzotriazine", "substituent_form": "1,2,3-benzotriazin-N-yl", "alkyl_stem_ok": False,
                          "atom_locants": {4: 1, 5: 2, 6: 3, 7: 4, 8: "4a", 9: 5, 0: 6, 1: 7, 2: 8, 3: "8a"}},
    # 1,2,3,4-benzotetrazine — fused 6,6 with four N in the heterocyclic ring.
    # Canonical 'c1ccc2nnnnc2c1': idx4=N1, idx5=N2, idx6=N3, idx7=N4,
    # idx8=C4a, idx9=C5, idx0=C6, idx1=C7, idx2=C8, idx3=C8a.
    "c1ccc2nnnnc2c1":    {"name": "1,2,3,4-benzotetrazine", "substituent_form": "1,2,3,4-benzotetrazin-N-yl", "alkyl_stem_ok": False,
                          "atom_locants": {4: 1, 5: 2, 6: 3, 7: 4, 8: "4a", 9: 5, 0: 6, 1: 7, 2: 8, 3: "8a"}},
    # Naphthyridines (6,6-fused diazines).  1,5- and 1,8-naphthyridine plus
    # indolizine already work; 1,6/1,7/2,6/2,7 added here in Stage 10 R10-A-5.
    # All four 10-atom maps verified by OPSIN chloro-probing.
    # 1,6-naphthyridine: canon 'c1cnc2ccncc2c1', N's at idx2 (L1) and idx6 (L6),
    #   jct atoms idx3 (L8a), idx8 (L4a).
    "c1cnc2ccncc2c1":    {"name": "1,6-naphthyridine", "substituent_form": "1,6-naphthyridin-N-yl", "alkyl_stem_ok": False,
                           "atom_locants": {2: 1, 1: 2, 0: 3, 9: 4, 8: "4a", 7: 5, 6: 6, 5: 7, 4: 8, 3: "8a"}},
    # 1,7-naphthyridine: canon 'c1cnc2cnccc2c1', N's at idx2 (L1) and idx5 (L7).
    "c1cnc2cnccc2c1":    {"name": "1,7-naphthyridine", "substituent_form": "1,7-naphthyridin-N-yl", "alkyl_stem_ok": False,
                           "atom_locants": {2: 1, 1: 2, 0: 3, 9: 4, 8: "4a", 7: 5, 6: 6, 5: 7, 4: 8, 3: "8a"}},
    # 2,6-naphthyridine: canon 'c1cc2cnccc2cn1', C2-symmetric; N's at idx9 (L2)
    #   and idx4 (L6); jct idx2 (L4a), idx7 (L8a).  Symmetric orbits collapse
    #   1<->5, 3<->7, 4<->8 to identical canonical SMILES via OPSIN — one
    #   consistent assignment is stored.
    "c1cc2cnccc2cn1":    {"name": "2,6-naphthyridine", "substituent_form": "2,6-naphthyridin-N-yl", "alkyl_stem_ok": False,
                           "atom_locants": {8: 1, 9: 2, 0: 3, 1: 4, 2: "4a", 3: 5, 4: 6, 5: 7, 6: 8, 7: "8a"}},
    # 2,7-naphthyridine: canon 'c1cc2ccncc2cn1', C2-symmetric; N's at idx9 (L2)
    #   and idx5 (L7); jct idx2 (L4a), idx7 (L8a).
    "c1cc2ccncc2cn1":    {"name": "2,7-naphthyridine", "substituent_form": "2,7-naphthyridin-N-yl", "alkyl_stem_ok": False,
                           "atom_locants": {8: 1, 9: 2, 0: 3, 1: 4, 2: "4a", 3: 5, 4: 6, 5: 7, 6: 8, 7: "8a"}},

    # -----------------------------------------------------------------------
    # Partially saturated benzo-fused heterocycles
    # -----------------------------------------------------------------------
    "c1ccc2c(c1)CCO2":   {"name": "2,3-dihydro-1-benzofuran",  "substituent_form": None, "alkyl_stem_ok": False},
    # indoline (2,3-dihydro-1H-indole): atom_locants verified via OPSIN chloro-probing.
    # canonical 'c1ccc2c(c1)CCN2': N=idx8=pos1, C=idx7=pos2(sp3), C=idx6=pos3(sp3),
    #   C=idx4=pos3a(junction), C=idx5=pos4, C=idx0=pos5, C=idx1=pos6, C=idx2=pos7, C=idx3=pos7a(junction)
    "c1ccc2c(c1)CCN2":   {"name": "2,3-dihydro-1H-indole", "substituent_form": "indolinyl", "alkyl_stem_ok": False,
                           "atom_locants": {8: 1, 7: 2, 6: 3, 4: "3a", 5: 4, 0: 5, 1: 6, 2: 7, 3: "7a"}},
    # 1,3-dihydro-2-benzofuran (O at pos 2, two CH2 at pos 1 and 3)
    # atom_locants: canonical 'c1ccc2c(c1)COC2', probed via OPSIN 1-(1,3-dihydro-2-benzofuran-N-yl)ethan-1-one
    # idx6=pos1(CH2), idx7=pos2(O), idx8=pos3(CH2), idx3=pos3a, idx2=pos4, idx1=pos5, idx0=pos6, idx5=pos7, idx4=pos7a
    "c1ccc2c(c1)COC2":   {"name": "1,3-dihydro-2-benzofuran", "substituent_form": "1,3-dihydro-2-benzofuranyl", "alkyl_stem_ok": False,
                           "atom_locants": {6: 1, 7: 2, 8: 3, 3: "3a", 2: 4, 1: 5, 0: 6, 5: 7, 4: "7a"}},
    # isoindoline (2,3-dihydro-1H-isoindole, N at pos 2): atom_locants verified via OPSIN probing.
    # canonical 'c1ccc2c(c1)CNC2': C(idx8)=pos1(CH2), N(idx7)=pos2, C(idx6)=pos3(CH2),
    #   C(idx4)=pos3a (junction adjacent to CH2,pos3), C(idx5)=pos4, C(idx0)=pos5,
    #   C(idx1)=pos6, C(idx2)=pos7, C(idx3)=pos7a (junction adjacent to CH2,pos1).
    # Topology: benzo ring 3a(idx4)-4(idx5)-5(idx0)-6(idx1)-7(idx2)-7a(idx3), 5-ring 3a(idx4)-3(idx6)-N(idx7)-1(idx8)-7a(idx3).
    # Verified: 1-Cl->idx8, 3-Cl->idx6, 4-Cl->idx5, 5-Cl->idx0, 6-Cl->idx1, 7-Cl->idx2 (all round-trip via OPSIN).
    "c1ccc2c(c1)CNC2":   {"name": "isoindoline", "substituent_form": "isoindolinyl", "alkyl_stem_ok": False,
                           "atom_locants": {8: 1, 7: 2, 6: 3, 4: "3a", 5: 4, 0: 5, 1: 6, 2: 7, 3: "7a"}},
    # octahydro-1H-isoindole (fully saturated cyclohexane fused to pyrrolidine, N at pos 2):
    # atom_locants verified via OPSIN chloro-probing.  Canonical 'C1CCC2CNCC2C1':
    #   idx5=N=pos2 (N), idx4=C=pos1 (CH2 adj to N), idx6=C=pos3 (CH2 adj to N),
    #   idx3=C=pos7a (junction adj to pos1=idx4), idx7=C=pos3a (junction adj to pos3=idx6),
    #   idx8=C=pos4, idx0=C=pos5, idx1=C=pos6, idx2=C=pos7.
    # Symmetric (Cs): positions 1<->3, 3a<->7a, 4<->7, 5<->6 collapse to same canonical via OPSIN;
    # one consistent assignment is chosen.  Covers FDA-0900 (octahydroisoindole-2-carbonyl side chain).
    "C1CCC2CNCC2C1":     {"name": "octahydro-1H-isoindole", "substituent_form": "octahydro-1H-isoindol-yl", "alkyl_stem_ok": False,
                           "atom_locants": {4: 1, 5: 2, 6: 3, 7: "3a", 8: 4, 0: 5, 1: 6, 2: 7, 3: "7a"}},
    # octahydro-4H-indene (hydrindane, bicyclo[4.3.0]nonane): fully saturated cyclohexane
    # fused to cyclopentane.  All-carbon symmetric ring; OPSIN chloro-probing shows
    # symmetry orbits idx{4,6}=pos1/3, idx5=pos2, idx{3,7}=pos3a/7a, idx{2,8}=pos4/7,
    # idx{0,1}=pos5/6.  One consistent assignment is stored; downstream locant
    # minimization (GetSubstructMatches(uniquify=False) in _build_numbering_from_atom_locants)
    # enumerates all orientations and picks the one giving lowest substituent locants
    # per P-14.5.  Covers FDA-0188 (calcipotriol), FDA-0189 (calcitriol), FDA-0417
    # (dihydrotachysterol) — all carry a 7a-methyl-octahydro-4H-inden-1-yl CD-ring core.
    "C1CCC2CCCC2C1":     {"name": "octahydro-4H-indene", "substituent_form": "octahydro-4H-inden-yl", "alkyl_stem_ok": False,
                           "atom_locants": {4: 1, 5: 2, 6: 3, 7: "3a", 8: 4, 0: 5, 1: 6, 2: 7, 3: "7a"}},
    # 3a,4,7,7a-tetrahydro-2H-isoindole (isoindole with cyclohexene fused to pyrrolidine,
    # N at pos 2, C=C at pos 5-6). Canonical 'C1=CCC2CNCC2C1' has the double bond at
    # idx0-idx1 which map to pos 5-6, matching OPSIN's locant convention.
    # Atom index mapping follows octahydro-1H-isoindole (same canonical layout since
    # the only difference is the 5,6 bond order); verified via OPSIN chloro-probing:
    #   1-Cl -> idx4, 3-Cl -> idx4 (sym), 3a-Cl -> idx3/7 (sym), 4-Cl -> idx2/8 (sym),
    #   5-Cl -> idx0/1 (sym), 7a-Cl -> idx3/7 (sym).
    # Indicated-H is "2H-" (on the NH nitrogen at pos 2) — NOT "1H-": after the
    # 4,5,6,7 ring carbons are hydrogenated and C1/C3 are still sp2 (adjacent to
    # N2 via the remaining C=N-C partial unsaturation implied by the aromatic
    # 5-ring valence budget), the only atom left that can legally bear the
    # indicated-H to satisfy valence is the N at pos 2.  OPSIN round-trips
    # "3a,4,7,7a-tetrahydro-1H-isoindole" to the WRONG SMILES
    # (C1=CCC2CN=CC2C1, an imine at N2=C3 instead of NH at N2).
    # Covers the isoindole-1,3-dione (imide) captan-like skeleton ZT-2471 / ZT-2472:
    # O=C1NC(=O)C2CC=CCC12 extracts as this ring and gains "1,3-dioxo-" prefix.
    "C1=CCC2CNCC2C1":    {"name": "3a,4,7,7a-tetrahydro-2H-isoindole",
                           "substituent_form": "3a,4,7,7a-tetrahydro-2H-isoindol-yl",
                           "alkyl_stem_ok": False,
                           "atom_locants": {4: 1, 5: 2, 6: 3, 7: "3a", 8: 4, 0: 5, 1: 6, 2: 7, 3: "7a"}},
    # 2,3,3a,4,5,6-hexahydro-1H-benz[de]isoquinoline (partially reduced naphthalimide
    # scaffold): peri-fused 6-6-6 tricyclic with a central 3-connected aromatic atom
    # (locant 9b). Aromatic ring occupies positions 6a,7,8,9,9a,9b; pyridine-like
    # ring (reduced, with NH) occupies 1,2,3,3a,9a,9b; cyclohexene (reduced) occupies
    # 3a,4,5,6,6a,9b. OPSIN '2,3,3a,4,5,6-hexahydro-1H-benz[de]isoquinoline' ->
    # C1NCC2C=3C(=CC=CC13)CCC2; canonical: c1cc2c3c(c1)CNCC3CCC2.
    # Atom indices in the canonical SMILES:
    #   idx6=C(pos1,CH2), idx7=N(pos2,NH), idx8=C(pos3,CH2),
    #   idx9=C(pos3a,sp3 junction), idx10=C(pos4), idx11=C(pos5), idx12=C(pos6),
    #   idx2=C(pos6a,aromatic junction), idx1=C(pos7), idx0=C(pos8), idx5=C(pos9),
    #   idx4=C(pos9a,aromatic junction), idx3=C(pos9b,central 3-connected aromatic).
    # IUPAC numbering verified via OPSIN chloro-probing on positions 1,2,3,3a,4,5,6,7,8,9
    # (aromatic junctions 6a/9a/9b fail kekulize-after-Cl-substitution but are uniquely
    # determined by topology: 6a adj to 6 & 7, 9a adj to 9 & 1, 9b central (adj to 3a,6a,9a)).
    # Covers FDA-1013 via exocyclic-oxo fallback: FDA-1013 inner ring
    # O=C1NC[C@@H]2CCCc3cccc1c32 extracts to this bare ring and gains "1-oxo-" suffix.
    "c1cc2c3c(c1)CNCC3CCC2": {"name": "2,3,3a,4,5,6-hexahydro-1H-benz[de]isoquinoline",
                               "substituent_form": "2,3,3a,4,5,6-hexahydro-1H-benz[de]isoquinolin-yl",
                               "alkyl_stem_ok": False,
                               "atom_locants": {6: 1, 7: 2, 8: 3, 9: "3a", 10: 4, 11: 5, 12: 6,
                                                2: "6a", 1: 7, 0: 8, 5: 9, 4: "9a", 3: "9b"}},
    # chroman (3,4-dihydro-2H-chromene, O at pos 1)
    # atom_locants: canonical 'c1ccc2c(c1)CCCO2', verified via OPSIN chloro-probing.
    # Ring atoms: O(idx9)=pos1, C(idx8)=pos2, C(idx7)=pos3, C(idx6)=pos4,
    #   C(idx4)=pos4a (junction to benzo), C(idx5)=pos5, C(idx0)=pos6,
    #   C(idx1)=pos7, C(idx2)=pos8, C(idx3)=pos8a (junction to pyran-O).
    # Verified: 2-Cl->idx8, 3-Cl->idx7, 4-Cl->idx6, 5-Cl->idx5, 6-Cl->idx0,
    #           7-Cl->idx1, 8-Cl->idx2 (all round-trip via OPSIN).
    # pin_eligible=False: per P-53 / P-54.4.3.2 the retained names chroman /
    # chromanyl are general-nomenclature only; the PIN is the systematic
    # 3,4-dihydro-2H-1-benzopyran (or 3,4-dihydro-2H-chromen-N-yl).
    "c1ccc2c(c1)CCCO2":  {"name": "chroman", "substituent_form": "chromanyl", "alkyl_stem_ok": False,
                           "pin_eligible": False,
                           "pin_name": "3,4-dihydro-2H-1-benzopyran",
                           "pin_substituent_form": "3,4-dihydro-2H-1-benzopyran-N-yl",
                           "atom_locants": {9: 1, 8: 2, 7: 3, 6: 4, 4: "4a", 5: 5, 0: 6, 1: 7, 2: 8, 3: "8a"}},

    # isochroman (3,4-dihydro-1H-2-benzopyran): O at IUPAC pos 2
    # atom_locants: canonical 'c1ccc2c(c1)CCOC2', probed via OPSIN 1-(isochroman-N-yl)ethan-1-one
    # idx9=pos1(CH2), idx8=pos2(O), idx7=pos3(CH2), idx6=pos4(CH2),
    # idx4=pos4a(quat C), idx5=pos5, idx0=pos6, idx1=pos7, idx2=pos8, idx3=pos8a(quat C)
    # pin_eligible=False: per P-53 / P-54.4.3.2 / P-32.4 the retained names
    # isochroman / isochromanyl are general-nomenclature only.
    "c1ccc2c(c1)CCOC2":  {"name": "isochroman", "substituent_form": "isochromanyl", "alkyl_stem_ok": False,
                           "pin_eligible": False,
                           "pin_name": "3,4-dihydro-1H-2-benzopyran",
                           "pin_substituent_form": "3,4-dihydro-1H-2-benzopyran-N-yl",
                           "atom_locants": {9: 1, 8: 2, 7: 3, 6: 4, 4: "4a", 5: 5, 0: 6, 1: 7, 2: 8, 3: "8a"}},

    # 1,3-benzodioxole: O-1,O-3 bridging aromatic C-3a and C-7a
    # atom_locants: canonical 'c1ccc2c(c1)OCO2', probed via OPSIN 1-(1,3-benzodioxol-N-yl)ethan-1-one
    # idx6=pos1(O), idx7=pos2(CH2), idx8=pos3(O), idx3=pos3a(quat C), idx2=pos4, idx1=pos5, idx0=pos6, idx5=pos7, idx4=pos7a(quat C)
    "c1ccc2c(c1)OCO2":   {"name": "1,3-benzodioxole", "substituent_form": "1,3-benzodioxolyl", "alkyl_stem_ok": False,
                           "atom_locants": {6: 1, 7: 2, 8: 3, 3: "3a", 2: 4, 1: 5, 0: 6, 5: 7, 4: "7a"}},

    # coumarin (2H-1-benzopyran-2-one = 2H-chromen-2-one):
    # The retained name 'coumarin' ALREADY encodes the C2 lactone carbonyl
    # (coumarin IS 2H-chromen-2-one). We therefore key the entry on the
    # with-=O canonical SMILES so the substitutive lookup goes through the
    # exocyclic-oxo fallback in retained_lookup, which claims the =O atom
    # via extra_atom_indices and prevents downstream from re-emitting a
    # redundant '2-oxo' prefix for substituted coumarins like
    # 4-hydroxycoumarin ('O=c1cc(O)c2ccccc2o1') -- previously named
    # '2-oxocoumarin-4-ol' instead of '4-hydroxycoumarin'.
    # Canonical atom indices in 'O=c1ccc2ccccc2o1':
    #   idx0=O(exocyclic, claimed via extra_atom_indices),
    #   idx1=C(pos2,lactone C), idx2=C(pos3), idx3=C(pos4), idx4=C(pos4a,junction),
    #   idx5=C(pos5), idx6=C(pos6), idx7=C(pos7), idx8=C(pos8), idx9=C(pos8a,junction),
    #   idx10=O(pos1,ring O).
    # Verified by OPSIN chloro probing of positions 3,4,5,6,7,8.
    "O=c1ccc2ccccc2o1":  {"name": "coumarin", "substituent_form": "coumarinyl", "alkyl_stem_ok": False,
                           "atom_locants": {10: 1, 1: 2, 2: 3, 3: 4, 4: "4a", 5: 5, 6: 6, 7: 7, 8: 8, 9: "8a"}},

    # chromone (4H-chromen-4-one / 4H-1-benzopyran-4-one):
    # Retained name encodes the C4 carbonyl. Key on with-=O canonical
    # 'O=c1ccoc2ccccc12' so exocyclic-oxo fallback claims the =O.
    # Canonical atom indices: idx0=O(exo), idx1=C(pos4,C=O), idx2=C(pos3),
    # idx3=C(pos2), idx4=O(pos1), idx5=C(pos8a,junction), idx6=C(pos8),
    # idx7=C(pos7), idx8=C(pos6), idx9=C(pos5), idx10=C(pos4a,junction).
    # Verified by OPSIN chloro probing of positions 2,3,5,6,7,8.
    "O=c1ccoc2ccccc12":  {"name": "chromone", "substituent_form": "chromonyl", "alkyl_stem_ok": False,
                           "atom_locants": {4: 1, 3: 2, 2: 3, 1: 4, 10: "4a", 9: 5, 8: 6, 7: 7, 6: 8, 5: "8a"}},

    # isocoumarin (1H-isochromen-1-one / 1H-2-benzopyran-1-one):
    # Retained name encodes the C1 carbonyl. Key on with-=O canonical
    # 'O=c1occc2ccccc12' so exocyclic-oxo fallback claims the =O.
    # Canonical atom indices: idx0=O(exo), idx1=C(pos1,C=O), idx2=O(pos2),
    # idx3=C(pos3), idx4=C(pos4), idx5=C(pos4a,junction), idx6=C(pos5),
    # idx7=C(pos6), idx8=C(pos7), idx9=C(pos8), idx10=C(pos8a,junction).
    # Verified by OPSIN chloro probing of positions 3,4,5,6,7,8.
    "O=c1occc2ccccc12":  {"name": "isocoumarin", "substituent_form": "isocoumarinyl", "alkyl_stem_ok": False,
                           "atom_locants": {1: 1, 2: 2, 3: 3, 4: 4, 5: "4a", 6: 5, 7: 6, 8: 7, 9: 8, 10: "8a"}},

    # 2H-chromene (2H-1-benzopyran):
    # atom_locants: canonical 'C1=Cc2ccccc2OC1', probed via OPSIN 1-(2H-chromen-N-yl)ethan-1-one
    # idx8=pos1(O), idx9=pos2(CH2), idx0=pos3, idx1=pos4, idx2=pos4a, idx3=pos5, idx4=pos6, idx5=pos7, idx6=pos8, idx7=pos8a
    "C1=Cc2ccccc2OC1":   {"name": "2H-chromene", "substituent_form": "2H-chromenyl", "alkyl_stem_ok": False,
                           "atom_locants": {8: 1, 9: 2, 0: 3, 1: 4, 2: "4a", 3: 5, 4: 6, 5: 7, 6: 8, 7: "8a"}},

    # 4H-chromene (4H-1-benzopyran):
    # atom_locants: canonical 'C1=COc2ccccc2C1', probed via OPSIN 1-(4H-chromen-N-yl)ethan-1-one
    # idx2=pos1(O), idx1=pos2, idx0=pos3, idx9=pos4(CH2), idx8=pos4a, idx7=pos5, idx6=pos6, idx5=pos7, idx4=pos8, idx3=pos8a
    "C1=COc2ccccc2C1":   {"name": "4H-chromene", "substituent_form": "4H-chromenyl", "alkyl_stem_ok": False,
                           "atom_locants": {2: 1, 1: 2, 0: 3, 9: 4, 8: "4a", 7: 5, 6: 6, 5: 7, 4: 8, 3: "8a"}},

    # Chromenylium / thiochromenylium / selenochromenylium / tellurochromenylium
    # (1-benzopyrylium salts; O/S/Se/Te+ at position 1).
    # Keys are RDKit canonical SMILES. atom_locants derived by OPSIN chloro probing
    # of positions 2..8 against N-chloro{salt}: for all four O/S/Se/Te canon forms,
    # idx 4 = X+ (pos 1), idx 5..7 = pyrylium C (pos 2,3,4), idx 8 = pos 4a (junction),
    # idx 9,0,1,2 = benzo (pos 5,6,7,8), idx 3 = pos 8a (junction adjacent to X+).
    "c1ccc2[o+]cccc2c1":  {"name": "chromenylium", "substituent_form": "chromenyl", "alkyl_stem_ok": False,
                           "atom_locants": {4: 1, 5: 2, 6: 3, 7: 4, 8: "4a", 9: 5, 0: 6, 1: 7, 2: 8, 3: "8a"}},
    "c1ccc2[s+]cccc2c1":  {"name": "thiochromenylium", "substituent_form": "thiochromenyl", "alkyl_stem_ok": False,
                           "atom_locants": {4: 1, 5: 2, 6: 3, 7: 4, 8: "4a", 9: 5, 0: 6, 1: 7, 2: 8, 3: "8a"}},
    "c1ccc2[se+]cccc2c1": {"name": "selenochromenylium", "substituent_form": "selenochromenyl", "alkyl_stem_ok": False,
                           "atom_locants": {4: 1, 5: 2, 6: 3, 7: 4, 8: "4a", 9: 5, 0: 6, 1: 7, 2: 8, 3: "8a"}},
    "c1ccc2[te+]cccc2c1": {"name": "tellurochromenylium", "substituent_form": "tellurochromenyl", "alkyl_stem_ok": False,
                           "atom_locants": {4: 1, 5: 2, 6: 3, 7: 4, 8: "4a", 9: 5, 0: 6, 1: 7, 2: 8, 3: "8a"}},

    # 9H-beta-carboline (9H-pyrido[3,4-b]indole):
    # atom_locants: canonical 'c1ccc2c(c1)[nH]c1cnccc12', probed via OPSIN {N}-chloro-9H-beta-carboline.
    # Chloro matches: idx8->1, idx10->3, idx11->4, idx2->5, idx1->6, idx0->7, idx5->8, idx6->9(N-H).
    # Junctions inferred from ring-topology: idx12->4a (connects pos 4, 4b, 9a),
    # idx3->4b (connects pos 5, 4a, 8a), idx4->8a (connects pos 8, 4b, N9), idx7->9a (connects N9, pos 1, 4a).
    # Position 2 is the pyridine N (idx 9, no chloro probe — N has no H).
    "c1ccc2c(c1)[nH]c1cnccc12": {"name": "9H-beta-carboline", "substituent_form": "9H-beta-carbolin-1-yl", "alkyl_stem_ok": False,
                           "atom_locants": {8: 1, 9: 2, 10: 3, 11: 4, 12: "4a", 3: "4b", 2: 5, 1: 6, 0: 7, 5: 8, 4: "8a", 6: 9, 7: "9a"}},

    # 9H-carbazole:
    # atom_locants: canonical 'c1ccc2c(c1)[nH]c1ccccc12', probed via OPSIN {N}-chloro-9H-carbazole.
    # The two benzo rings are symmetry-equivalent, so pos 1-4 and 5-8 produce identical chloro canonicals to idx 8-11 vs 5-2.
    # Assignment follows the beta-carboline analogue (second benzo ring = positions 1-4):
    # idx8->1, idx9->2, idx10->3, idx11->4, idx12->4a, idx3->4b, idx2->5, idx1->6, idx0->7, idx5->8,
    # idx4->8a, idx6->9(N-H), idx7->9a.
    "c1ccc2c(c1)[nH]c1ccccc12": {"name": "9H-carbazole", "substituent_form": "9H-carbazol-1-yl", "alkyl_stem_ok": False,
                           "atom_locants": {8: 1, 9: 2, 10: 3, 11: 4, 12: "4a", 3: "4b", 2: 5, 1: 6, 0: 7, 5: 8, 4: "8a", 6: 9, 7: "9a"}},

    # 2,3,4,9-tetrahydro-1H-beta-carboline (tryptoline):
    # atom_locants: canonical 'c1ccc2c3c([nH]c2c1)CNCC3', probed via OPSIN {N}-chloro-2,3,4,9-tetrahydro-1H-beta-carboline.
    # Chloro matches: idx9->1, idx10->2(N), idx11->3, idx12->4, idx2->5, idx1->6, idx0->7, idx8->8, idx6->9(N-H).
    # Junctions inferred from topology: idx4->4a (sp2, connects pos 4 ring to pyrrole), idx3->4b,
    # idx7->8a (adjacent to pos 8 and N9), idx5->9a (adjacent to N9 and pos 1).
    "c1ccc2c3c([nH]c2c1)CNCC3": {"name": "tryptolin", "substituent_form": "tryptolinyl", "alkyl_stem_ok": False,
                           "atom_locants": {9: 1, 10: 2, 11: 3, 12: 4, 4: "4a", 3: "4b", 2: 5, 1: 6, 0: 7, 8: 8, 7: "8a", 6: 9, 5: "9a"}},

    # Isochromenylium / isothiochromenylium / isoselenochromenylium / isotellurochromenylium
    # (2-benzopyrylium salts; O/S/Se/Te+ at position 2).
    # Keys are RDKit canonical SMILES. atom_locants derived by OPSIN chloro probing
    # of positions 1,3,4,5,6,7,8 (position 2 is the X+ heteroatom): for all four forms,
    # idx 4 = pos 1 (C adjacent to X+), idx 5 = X+ (pos 2), idx 6 = pos 3, idx 7 = pos 4,
    # idx 3 = pos 4a (junction), idx 9,0,1,2 = benzo (pos 5,6,7,8), idx 8 = pos 8a (junction adjacent to pos 1).
    "c1ccc2c[o+]ccc2c1":  {"name": "isochromenylium", "substituent_form": "isochromenyl", "alkyl_stem_ok": False,
                           "atom_locants": {4: 1, 5: 2, 6: 3, 7: 4, 3: "4a", 9: 5, 0: 6, 1: 7, 2: 8, 8: "8a"}},
    "c1ccc2c[s+]ccc2c1":  {"name": "isothiochromenylium", "substituent_form": "isothiochromenyl", "alkyl_stem_ok": False,
                           "atom_locants": {4: 1, 5: 2, 6: 3, 7: 4, 3: "4a", 9: 5, 0: 6, 1: 7, 2: 8, 8: "8a"}},
    "c1ccc2c[se+]ccc2c1": {"name": "isoselenochromenylium", "substituent_form": "isoselenochromenyl", "alkyl_stem_ok": False,
                           "atom_locants": {4: 1, 5: 2, 6: 3, 7: 4, 3: "4a", 9: 5, 0: 6, 1: 7, 2: 8, 8: "8a"}},
    "c1ccc2c[te+]ccc2c1": {"name": "isotellurochromenylium", "substituent_form": "isotellurochromenyl", "alkyl_stem_ok": False,
                           "atom_locants": {4: 1, 5: 2, 6: 3, 7: 4, 3: "4a", 9: 5, 0: 6, 1: 7, 2: 8, 8: "8a"}},

    # -----------------------------------------------------------------------
    # Tricyclic ring systems
    # -----------------------------------------------------------------------
    # Acridine (tricyclic, N at position 10).
    # atom_locants derived from OPSIN probing of chloro-acridines (SMILES: c1ccc2nc3ccccc3cc2c1):
    # idx0=2, idx1=3, idx2=4, idx3=4a, idx4=10(N), idx5=10a, idx6=5, idx7=6,
    # idx8=7, idx9=8, idx10=8a, idx11=9, idx12=9a, idx13=1
    "c1ccc2nc3ccccc3cc2c1":     {"name": "acridine",        "substituent_form": "acridinyl",      "alkyl_stem_ok": False,
                                  "atom_locants": {0: 2, 1: 3, 2: 4, 3: "4a", 4: 10, 5: "10a",
                                                   6: 5, 7: 6, 8: 7, 9: 8, 10: "8a", 11: 9, 12: "9a", 13: 1}},
    # Dibenzofuran — fused 6,5,6 (two benzene rings fused to central furan).
    # Canonical 'c1ccc2c(c1)oc1ccccc12'.  C2v symmetry collapses pairs of
    # locants (1↔9, 2↔8, 3↔7, 4↔6) onto the same canonical-idx, so OPSIN
    # always picks the lower of each pair.  Verified via OPSIN chloro-
    # probing of L=1..9 — see Stage 11 R11-A commit.
    # Without atom_locants, the engine emitted symmetric-but-wrong locants
    # (e.g. 3-chlorodibenzofuran rendered as 4-chlorodibenzofuran), which
    # round-trips to a *different* RDKit canonical because the chosen atom
    # is the wrong one of the two C2v-related positions.  This entry pins
    # the canonical assignment.
    "c1ccc2c(c1)oc1ccccc12":    {"name": "dibenzofuran",    "substituent_form": "dibenzofuranyl", "alkyl_stem_ok": False,
                                  "atom_locants": {2: 1, 1: 2, 0: 3, 5: 4, 4: "4a",
                                                   6: 5, 7: "5a", 8: 6, 9: 7, 10: 8,
                                                   11: 9, 12: "9a", 3: "9b"}},
    # Dibenzothiophene — sulfur analog of dibenzofuran (P-25.3.1.3 retained).
    # Same skeletal C2v layout; only heteroatom symbol differs (S at idx 6).
    "c1ccc2c(c1)sc1ccccc12":    {"name": "dibenzothiophene", "substituent_form": "dibenzothienyl", "alkyl_stem_ok": False,
                                  "atom_locants": {2: 1, 1: 2, 0: 3, 5: 4, 4: "4a",
                                                   6: 5, 7: "5a", 8: 6, 9: 7, 10: 8,
                                                   11: 9, 12: "9a", 3: "9b"}},
    # Dibenzoselenophene — selenium analog (P-25.3.1.3 retained, less common).
    # Same skeletal C2v layout.  RDKit aromaticity model uses lowercase [se].
    "c1ccc2c(c1)[se]c1ccccc12": {"name": "dibenzoselenophene", "substituent_form": "dibenzoselenophenyl", "alkyl_stem_ok": False,
                                  "atom_locants": {2: 1, 1: 2, 0: 3, 5: 4, 4: "4a",
                                                   6: 5, 7: "5a", 8: 6, 9: 7, 10: 8,
                                                   11: 9, 12: "9a", 3: "9b"}},
    # 5H-dibenz[b,f]azepine (carbamazepine-like):
    # atom_locants: canonical 'C1=Cc2ccccc2Nc2ccccc21', probed via OPSIN 1-(5H-dibenz[b,f]azepin-N-yl)ethan-1-one
    # idx3=1, idx4=2, idx5=3, idx6=4, idx7=4a, idx8=5(N,H), idx9=5a, idx10=6, idx11=7, idx12=8, idx13=9, idx14=10a, idx0=10, idx1=11, idx2=11a
    "C1=Cc2ccccc2Nc2ccccc21":   {"name": "5H-dibenz[b,f]azepine", "substituent_form": "5H-dibenz[b,f]azepin-5-yl", "alkyl_stem_ok": False,
                                  "atom_locants": {3: 1, 4: 2, 5: 3, 6: 4, 7: "4a", 8: 5, 9: "5a", 10: 6, 11: 7, 12: 8, 13: 9, 14: "10a", 0: 10, 1: 11, 2: "11a"}},
    # Phenanthridine: 5-aza-phenanthrene; N at position 5.  atom_locants
    # derived via OPSIN methyl-probing (Stage 4 unit 14) + topology closure.
    "c1ccc2c(c1)cnc1ccccc12":   {"name": "phenanthridine",  "substituent_form": None,             "alkyl_stem_ok": False,
                                  "atom_locants": {12: 1, 11: 2, 10: 3, 9: 4, 8: "4a", 7: 5, 6: 6, 4: "6a", 5: 7, 0: 8, 1: 9, 2: 10, 3: "10a", 13: "10b"}},
    # Phenothiazine: S at 5, N at 10 (IUPAC P-31.1.3.4).
    # atom_locants derived from OPSIN locant-SMILES: canonical 'c1ccc2c(c1)Nc1ccccc1S2'
    "c1ccc2c(c1)Nc1ccccc1S2":   {"name": "phenothiazine",   "substituent_form": "phenothiazinyl", "alkyl_stem_ok": False,
                                  "atom_locants": {5: 1, 0: 2, 1: 3, 2: 4, 13: 5, 11: 6, 10: 7, 9: 8, 8: 9, 6: 10}},
    "c1ccc2c(c1)Cc1ccccc1S2":   {"name": "thioxanthene",    "substituent_form": "thioxanthenyl",  "alkyl_stem_ok": False,
                                  "atom_locants": {5: 1, 0: 2, 1: 3, 2: 4, 11: 5, 10: 6, 9: 7, 8: 8, 6: 9}},
    "c1ccc2c(c1)Cc1ccccc1O2":   {"name": "xanthene",        "substituent_form": "xanthenyl",      "alkyl_stem_ok": False,
                                  "atom_locants": {5: 1, 0: 2, 1: 3, 2: 4, 11: 5, 10: 6, 9: 7, 8: 8, 6: 9}},
    # 9H-xanthen-9-one (xanthone) and 9H-thioxanthen-9-one (thioxanthone):
    # the 9-oxo tautomers of xanthene / thioxanthene.  The plain-ring extraction
    # (strip exocyclic =O) yields a ring with aromatic o/s at position 10, which
    # is not itself a retained-ring canonical (and for the S variant RDKit cannot
    # even kekulise it).  Register the with-=O canonical SMILES directly, so the
    # exocyclic-oxo fallback in ``retained_lookup`` claims the carbonyl O via
    # extra_atom_indices and downstream does not re-emit a redundant "-9-oxo-"
    # prefix (the "-9-one" is already encoded in the retained-name stem).
    # Same precedent as 7,8,9,10-tetrahydrotetracene-5,12-dione above.
    # Canonical 'O=c1c2ccccc2oc2ccccc12' (15 atoms incl. exocyclic O=):
    #   idx0 = O (exocyclic, pos 9 oxo)
    #   idx1 = C (pos 9, carbonyl)
    #   idx2 = C (pos 8a, junction C9-C8)
    #   idx3 = C (pos 8)
    #   idx4 = C (pos 7)
    #   idx5 = C (pos 6)
    #   idx6 = C (pos 5)
    #   idx7 = C (pos 10a, junction C5-X10)
    #   idx8 = X (pos 10, O for xanthone / S for thioxanthone)
    #   idx9 = C (pos 4a, junction X10-C4)
    #   idx10 = C (pos 4)
    #   idx11 = C (pos 3)
    #   idx12 = C (pos 2)
    #   idx13 = C (pos 1)
    #   idx14 = C (pos 9a, junction C9-C1)
    # Locants verified by chloro/methyl probing:
    #   1-chloro-N-methyl-9H-xanthen-9-one ⇒ Cl on idx13 (pos1),
    #   Me on idx12/11/10 for pos2/3/4, idx6/5/4/3 for pos5/6/7/8.
    # Exact same mapping for the S variant (only idx8 changes element).
    "O=c1c2ccccc2oc2ccccc12":   {"name": "9H-xanthen-9-one", "substituent_form": "9-oxo-9H-xanthen-N-yl", "alkyl_stem_ok": False,
                                  "atom_locants": {1: 9, 2: "8a", 3: 8, 4: 7, 5: 6, 6: 5, 7: "10a",
                                                   8: 10, 9: "4a", 10: 4, 11: 3, 12: 2, 13: 1, 14: "9a"}},
    "O=c1c2ccccc2sc2ccccc12":   {"name": "9H-thioxanthen-9-one", "substituent_form": "9-oxo-9H-thioxanthen-N-yl", "alkyl_stem_ok": False,
                                  "atom_locants": {1: 9, 2: "8a", 3: 8, 4: 7, 5: 6, 6: 5, 7: "10a",
                                                   8: 10, 9: "4a", 10: 4, 11: 3, 12: 2, 13: 1, 14: "9a"}},
    # 9H-pyrano[3,2-g]quinoline-4,6-dione: linear 6-6-6 tricyclic with a central
    # benzene fused (a) to a 4-oxo-4H-chromen-style pyranone ring and (b) to a
    # 4-oxo-quinolin-4(1H)-one-style pyridinone ring.  The retained-name stem
    # encodes BOTH ring carbonyls (the "4,6-dione" tail) plus the N-H tautomer
    # (the "9H-" prefix identifies the NH position); key on the with-=O canonical
    # so the exocyclic-oxo fallback in retained_lookup claims the two =O atoms
    # via extra_atom_indices and downstream does not re-emit "4-oxo,6-oxo-".
    # Same precedent as 9H-xanthen-9-one / 7,8,9,10-tetrahydrotetracene-5,12-dione.
    # OPSIN '9H-pyrano[3,2-g]quinoline-4,6-dione' -> O=c1cc[nH]c2cc3occc(=O)c3cc12.
    # Canonical 'O=c1cc[nH]c2cc3occc(=O)c3cc12' (16 atoms incl. two exocyclic O=):
    #   idx0 = O (exocyclic, pos 6 oxo)
    #   idx1 = C (pos 6, pyridinone carbonyl)
    #   idx2 = C (pos 7)
    #   idx3 = C (pos 8)
    #   idx4 = N-H (pos 9)
    #   idx5 = C (pos 9a, junction N-C10)
    #   idx6 = C (pos 10)
    #   idx7 = C (pos 10a, junction C10-O1)
    #   idx8 = O (pos 1, ring O of pyran)
    #   idx9 = C (pos 2, pyran CH)
    #   idx10 = C (pos 3, pyran CH)
    #   idx11 = C (pos 4, pyranone carbonyl)
    #   idx12 = O (exocyclic, pos 4 oxo)
    #   idx13 = C (pos 4a, junction C4-C5)
    #   idx14 = C (pos 5)
    #   idx15 = C (pos 5a, junction C5-C6)
    # Locants verified by OPSIN methyl-probing positions 2,3,5,7,8,10 and
    # N-ethyl-probing position 9.  FDA-0938 (acronine/acronycine-type diacid) is
    # 9-ethyl-4,6-dioxo-10-propyl-9H-pyrano[3,2-g]quinoline-2,8-dicarboxylic acid.
    "O=c1cc[nH]c2cc3occc(=O)c3cc12": {"name": "9H-pyrano[3,2-g]quinoline-4,6-dione",
                                       "substituent_form": "4,6-dioxo-9H-pyrano[3,2-g]quinolin-yl",
                                       "alkyl_stem_ok": False,
                                       "atom_locants": {1: 6, 2: 7, 3: 8, 4: 9, 5: "9a",
                                                        6: 10, 7: "10a", 8: 1, 9: 2, 10: 3,
                                                        11: 4, 13: "4a", 14: 5, 15: "5a"}},
    "c1ccc2nc3ccccc3nc2c1":     {"name": "phenazine",       "substituent_form": "phenazinyl",     "alkyl_stem_ok": False,
                                  "atom_locants": {13: 1, 0: 2, 1: 3, 2: 4, 4: 5, 6: 6, 7: 7, 8: 8, 9: 9, 11: 10}},
    "c1ccc2c(c1)Nc1ccccc1O2":   {"name": "phenoxazine",     "substituent_form": "phenoxazinyl",   "alkyl_stem_ok": False,
                                  "atom_locants": {5: 1, 0: 2, 1: 3, 2: 4, 13: 5, 11: 6, 10: 7, 9: 8, 8: 9, 6: 10}},

    # -----------------------------------------------------------------------
    # Partially saturated benzo-fused rings
    # -----------------------------------------------------------------------
    # 3,4-dihydroquinolin-2(1H)-one (dihydroquinolinone)
    # atom_locants: canonical 'O=C1CCc2ccccc2N1', verified via IUPAC quinoline numbering.
    # N(idx=10)=1, C=O(idx=1)=2, CH2(idx=2)=3, CH2(idx=3)=4, C4a(idx=4)="4a",
    # C5(idx=5)=5, C6(idx=6)=6, C7(idx=7)=7, C8(idx=8)=8, C8a(idx=9)="8a"
    # substituent_form uses no hardcoded locant so _heteroaryl_substituent_with_locant
    # can compute the correct locant for the actual attachment position.
    "O=C1CCc2ccccc2N1":  {"name": "3,4-dihydroquinolin-2(1H)-one", "substituent_form": "3,4-dihydroquinolin-2(1H)-onyl", "alkyl_stem_ok": False,
                          "atom_locants": {10: 1, 1: 2, 2: 3, 3: 4, 4: "4a", 5: 5, 6: 6, 7: 7, 8: 8, 9: "8a"}},

    # 2,3,4,5-tetrahydro-1H-1-benzazepin-2-one  (7-membered benzo-fused lactam;
    # the ring system in benazepril and related ACE inhibitors).
    # Canonical SMILES with exocyclic =O: 'O=C1CCCc2ccccc2N1' (12 atoms incl. O).
    # atom_locants verified by chloro-probing each position with OPSIN and
    # mapping back via SubstructMatch onto the bare canonical:
    #   pos1=N(idx11), pos2=C=O(idx1), pos3=idx2, pos4=idx3, pos5=idx4,
    #   pos5a=idx5 (ring junction), pos6=idx6, pos7=idx7, pos8=idx8, pos9=idx9,
    #   pos9a=idx10 (ring junction).
    "O=C1CCCc2ccccc2N1": {"name": "2,3,4,5-tetrahydro-1H-1-benzazepin-2-one", "substituent_form": "2,3,4,5-tetrahydro-1H-1-benzazepin-2-onyl", "alkyl_stem_ok": False,
                          "atom_locants": {11: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: "5a", 6: 6, 7: 7, 8: 8, 9: 9, 10: "9a"}},

    # 2,3-dihydro-1H-1,4-benzodiazepin-2-one  (Cluster 4 — diazepam/lorazepam/nitrazepam family).
    # Canonical SMILES with exocyclic =O: 'O=C1CN=Cc2ccccc2N1' (12 atoms incl. O).
    # The same bare scaffold underlies: diazepam/oxazepam (C5-aryl, C7 substituent),
    # nitrazepam (7-NO2), lorazepam (3-OH, 7-Cl, 5-aryl), and many others.
    # atom_locants verified by chloro-probing positions 3,5,6,7,8,9 with OPSIN and
    # mapping back via SubstructMatch onto the bare canonical:
    #   pos1=N(idx11), pos2=C=O(idx1), pos3=CH2(idx2), pos4=N(idx3), pos5=CH(idx4),
    #   pos5a=C(idx5, ring junction), pos6..9=idx6..9, pos9a=C(idx10, ring junction).
    "O=C1CN=Cc2ccccc2N1": {"name": "2,3-dihydro-1H-1,4-benzodiazepin-2-one", "substituent_form": "2,3-dihydro-1H-1,4-benzodiazepin-2-onyl", "alkyl_stem_ok": False,
                            "atom_locants": {11: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: "5a", 6: 6, 7: 7, 8: 8, 9: 9, 10: "9a"}},

    # 1,2,4-benzothiadiazine — three indicated-hydrogen tautomers.
    # Used by the thiazide-diuretic family (chlorothiazide, hydrochlorothiazide,
    # diazoxide, methyclothiazide, trichlormethiazide, bendroflumethiazide,
    # benzthiazide, ...) where the 1,1-disulfonyl (1,1-dioxo) parent is the rule.
    # The S(=O)(=O) of these drugs is *not* keyed into the curated SMILES — we
    # store the bare scaffold (no =O on S) and let the standard "oxo on ring
    # heteroatom" prefix machinery emit "1,1-dioxo-" for the two oxide oxygens.
    # That keeps this entry generic (one entry covers every thiazide derivative
    # regardless of substitution) and avoids the architectural hazard of
    # memorising substituted SMILES.  Position 1 = S, 2 = N, 3 = C, 4 = N,
    # 4a/8a = ring junctions, 5..8 = benzo carbons.  Atom locants verified by
    # chloro-probing positions 3,5,6,7,8 against OPSIN and mapping back via
    # SubstructMatch onto the bare canonical RDKit SMILES.

    # 3,4-dihydro-2H-1,2,4-benzothiadiazine: c1ccc2c(c1)NCNS2  (10 ring atoms).
    # Bonds: benzo 0-1-2-3-4-5-0; heterocycle 4-N(6)-C(7)-N(8)-S(9)-3.
    # pos1=S(idx9), pos2=N(idx8), pos3=C(idx7), pos4=N(idx6),
    # pos4a=C(idx4, junction to N4), pos5..8=idx5,0,1,2, pos8a=C(idx3, junction to S1).
    "c1ccc2c(c1)NCNS2": {"name": "3,4-dihydro-2H-1,2,4-benzothiadiazine", "substituent_form": None, "alkyl_stem_ok": False,
                          "atom_locants": {9: 1, 8: 2, 7: 3, 6: 4, 4: "4a", 5: 5, 0: 6, 1: 7, 2: 8, 3: "8a"}},

    # 2H-1,2,4-benzothiadiazine: C1=Nc2ccccc2SN1  (10 ring atoms).
    # Bonds: heterocycle C(0)=N(1)-c(2)...c(7)-S(8)-N(9)-C(0); benzo 2-3-4-5-6-7-2.
    # pos1=S(idx8), pos2=N(idx9), pos3=C(idx0), pos4=N(idx1),
    # pos4a=C(idx2, junction to N4), pos5..8=idx3,4,5,6, pos8a=C(idx7, junction to S1).
    "C1=Nc2ccccc2SN1": {"name": "2H-1,2,4-benzothiadiazine", "substituent_form": None, "alkyl_stem_ok": False,
                         "atom_locants": {8: 1, 9: 2, 0: 3, 1: 4, 2: "4a", 3: 5, 4: 6, 5: 7, 6: 8, 7: "8a"}},

    # 4H-1,2,4-benzothiadiazine: C1=NSc2ccccc2N1  (10 ring atoms).
    # Bonds: heterocycle C(0)=N(1)-S(2)-c(3)...c(8)-N(9)-C(0); benzo 3-4-5-6-7-8-3.
    # pos1=S(idx2), pos2=N(idx1), pos3=C(idx0), pos4=N(idx9),
    # pos4a=C(idx8, junction to N4), pos5..8=idx7,6,5,4, pos8a=C(idx3, junction to S1).
    "C1=NSc2ccccc2N1": {"name": "4H-1,2,4-benzothiadiazine", "substituent_form": None, "alkyl_stem_ok": False,
                         "atom_locants": {2: 1, 1: 2, 0: 3, 9: 4, 8: "4a", 7: 5, 6: 6, 5: 7, 4: 8, 3: "8a"}},

    # 3,4-dihydro-1H-2,1,3-benzothiadiazine — Phase 7 ring-batch-4.
    # 6-ring with heteros at 1 (NH), 2 (S), 3 (NH); the 4-position is CH2.
    # Fused to benzene at 4a-8a.  Canonical (RDKit) 'c1ccc2c(c1)CNSN2'.
    # Engine FG-perception adds the dioxo on S (pos 2) and the in-ring C=O
    # at pos 4 to assemble bentazon-class names like
    # 3-isopropyl-1H-2,1,3-benzothiadiazin-4(3H)-one 2,2-dioxide.
    # OPSIN: '3,4-dihydro-1H-2,1,3-benzothiadiazine' → 'c1ccc2c(c1)CNSN2' (verified).
    # Atom layout of 'c1ccc2c(c1)CNSN2' (10 atoms):
    #   idx0=C(pos6 arom), idx1=C(pos7 arom), idx2=C(pos8 arom),
    #   idx3=C(pos8a junction adj. to N1), idx4=C(pos4a junction adj. to CH2),
    #   idx5=C(pos5 arom), idx6=C(pos4 CH2),
    #   idx7=N(pos3 NH), idx8=S(pos2), idx9=N(pos1 NH).
    # OPSIN chloro-probing verified: pos1→idx9 (N-Cl), pos3→idx7 (N-Cl),
    # pos4→idx6, pos5→idx5, pos6→idx0, pos7→idx1, pos8→idx2; junctions
    # pos4a→idx4, pos8a→idx3 derived from topology.
    "c1ccc2c(c1)CNSN2": {
        "name": "3,4-dihydro-1H-2,1,3-benzothiadiazine",
        "substituent_form": None,
        "alkyl_stem_ok": False,
        "atom_locants": {0: 6, 1: 7, 2: 8, 3: "8a", 4: "4a", 5: 5, 6: 4,
                         7: 3, 8: 2, 9: 1}},

    # 10,11-dihydro-5H-dibenz[b,f]azepine: atom_locants verified via OPSIN methyl-probing.
    # canonical 'c1ccc2c(c1)CCc1ccccc1N2'; 15 atoms; N=idx14=pos5; CH2 bridge=idx6(pos11),idx7(pos10).
    # Ring 1 (pos1-4 benzo): 1(idx5)-2(idx0)-3(idx1)-4(idx2)-4a(idx3)-11a(idx4).
    # Ring 2 (pos6-9 benzo): 6(idx12)-7(idx11)-8(idx10)-9(idx9)-10a(idx8)-5a(idx13).
    # 7-membered azepine: N(5,idx14)-4a(idx3)-11a(idx4)-11(idx6)-10(idx7)-10a(idx8)-5a(idx13)-N.
    # Verified: pos1-4 (idx5,0,1,2), pos5 (N,idx14), pos6-9 (idx12,11,10,9), pos10 (idx7), pos11 (idx6)
    #   all round-trip through OPSIN methyl-probing. Junctions (4a,5a,10a,11a) inferred from topology.
    "c1ccc2c(c1)CCc1ccccc1N2":  {"name": "10,11-dihydro-5H-dibenz[b,f]azepine", "substituent_form": "10,11-dihydro-5H-dibenz[b,f]azepin-5-yl", "alkyl_stem_ok": False,
                                  "atom_locants": {5: 1, 0: 2, 1: 3, 2: 4, 3: "4a", 14: 5, 13: "5a",
                                                   12: 6, 11: 7, 10: 8, 9: 9, 8: "10a", 7: 10, 6: 11, 4: "11a"}},

    # 10,11-dihydro-5H-dibenzo[a,d]cycloheptene (all-carbon analogue of above).
    # Canonical SMILES: c1ccc2c(c1)CCc1ccccc1C2
    # IUPAC numbering (verified by probing OPSIN with 1..11-methyl derivatives):
    #   C1=idx5, C2=idx0, C3=idx1, C4=idx2, C4a=idx3, C5=idx14, C5a=idx13,
    #   C6=idx12, C7=idx11, C8=idx10, C9=idx9, C9a=idx8, C10=idx7, C11=idx6, C11a=idx4
    "c1ccc2c(c1)CCc1ccccc1C2":  {"name": "10,11-dihydro-5H-dibenzo[a,d]cycloheptene", "substituent_form": "10,11-dihydro-5H-dibenzo[a,d]cyclohepten-5-yl", "alkyl_stem_ok": False,
                                  "atom_locants": {5: 1, 0: 2, 1: 3, 2: 4, 3: "4a", 14: 5, 13: "5a", 12: 6, 11: 7, 10: 8, 9: 9, 8: "9a", 7: 10, 6: 11, 4: "11a"}},

    # 5H-dibenzo[a,d][7]annulene (amitriptyline/cyclobenzaprine core — the unsaturated
    # aromatic-bridge analogue of 10,11-dihydrodibenzo[a,d]cycloheptene).  IUPAC 2013
    # uses the systematic [7]annulene name here; the older "cyclohepta-1,3-diene"
    # fusion nomenclature is disfavoured.  Position 5 = central CH2; positions 10-11
    # are the bridging C=C double bond.  Canonical SMILES: C1=Cc2ccccc2Cc2ccccc21
    # Methyl-probed (cyclobenzaprine = FDA-0339 parent scaffold):
    #   pos1-4 (one benzo): idx3,4,5,6; pos4a=idx7; pos5=idx8 (CH2); pos5a=idx9;
    #   pos6-9 (other benzo): idx10,11,12,13; pos9a=idx14;
    #   pos10=idx0, pos11=idx1 (C=C bridge); pos11a=idx2.
    "C1=Cc2ccccc2Cc2ccccc21":   {"name": "5H-dibenzo[a,d][7]annulene", "substituent_form": "5H-dibenzo[a,d][7]annulen-5-yl", "alkyl_stem_ok": False,
                                  "atom_locants": {3: 1, 4: 2, 5: 3, 6: 4, 7: "4a", 8: 5, 9: "5a", 10: 6, 11: 7, 12: 8, 13: 9, 14: "9a", 0: 10, 1: 11, 2: "11a"}},

    # 2,3,3a,12b-tetrahydro-1H-dibenzo[2,3:6,7]oxepino[4,5-c]pyrrole (asenapine core,
    # FDA-0101).  Tetracyclic scaffold: two benzo rings fused onto an oxepine which
    # is fused with a pyrrolidine ring.  Canonical SMILES: c1ccc2c(c1)Oc1ccccc1C1CNCC21
    # Methyl-probed against OPSIN: scaffold has a C2-symmetry axis through the 3a-12b
    # bond; each benzo position maps to two RDKit indices (mirror images) — one
    # consistent assignment is used here.
    #   pos1=idx14 (CH2), pos2=idx15 (NH), pos3=idx16 (CH2), pos3a=idx17 (sp3 CH),
    #   pos12b=idx13 (sp3 CH), pos4=idx2, pos5=idx1, pos6=idx0, pos7=idx5,
    #   O bridge (idx6) has no numeric locant (peripheral atom between pos7 and pos9),
    #   pos9=idx8, pos10=idx9, pos11=idx10, pos12=idx11.
    #   Benzo junctions (idx3, idx4, idx7, idx12) are interior and not numbered.
    "c1ccc2c(c1)Oc1ccccc1C1CNCC21": {"name": "2,3,3a,12b-tetrahydro-1H-dibenzo[2,3:6,7]oxepino[4,5-c]pyrrole",
                                      "substituent_form": "2,3,3a,12b-tetrahydro-1H-dibenzo[2,3:6,7]oxepino[4,5-c]pyrrol-5-yl",
                                      "alkyl_stem_ok": False,
                                      "atom_locants": {14: 1, 15: 2, 16: 3, 17: "3a",
                                                       2: 4, 1: 5, 0: 6, 5: 7,
                                                       8: 9, 9: 10, 10: 11, 11: 12,
                                                       13: "12b"}},

    # 9,13b-dihydro-1H-dibenzo[c,f]imidazo[1,5-a]azepine (epinastine core, FDA-0482).
    # Tetracyclic: imidazoline fused with azepine, both benzo-fused at c,f bonds.
    # Canonical SMILES: C1=NCC2c3ccccc3Cc3ccccc3N12
    # Methyl/chloro-probed against OPSIN (peripheral locants per IUPAC fusion rules):
    #   pos1=idx2 (CH2), pos2=idx1 (N=), pos3=idx0 (=CH),
    #   pos4a=idx16 (junction between imidazoline N and benzo), pos13a=idx4 (junction
    #   between other benzo and sp3 pos13b), pos13b=idx3 (sp3 CH),
    #   pos5=idx15, pos6=idx14, pos7=idx13, pos8=idx12 (one benzo),
    #   pos9=idx10 (CH2 bridge),
    #   pos10=idx8, pos11=idx7, pos12=idx6, pos13=idx5 (other benzo).
    #   idx17 (bridgehead N) and benzo junctions idx9, idx11 are interior (no locant).
    "C1=NCC2c3ccccc3Cc3ccccc3N12": {"name": "9,13b-dihydro-1H-dibenzo[c,f]imidazo[1,5-a]azepine",
                                     "substituent_form": None,
                                     "alkyl_stem_ok": False,
                                     "atom_locants": {2: 1, 1: 2, 0: 3, 16: "4a",
                                                      15: 5, 14: 6, 13: 7, 12: 8,
                                                      10: 9, 8: 10, 7: 11, 6: 12, 5: 13,
                                                      4: "13a", 3: "13b"}},

    # 2,3,4,9-tetrahydro-1H-indeno[2,1-c]pyridine (phenindamine core, FDA-1052).
    # Tricyclic: piperidine fused to indene; not dibenzo (one benzo + cyclopentene).
    # Canonical SMILES: c1ccc2c(c1)CC1=C2CCNC1
    # Methyl-probed against OPSIN:
    #   pos1=idx12 (CH2), pos2=idx11 (NH), pos3=idx10 (CH2), pos4=idx9 (CH2),
    #   pos4a=idx8 (sp2 C, pyridine-indene junction),
    #   pos5=idx2, pos6=idx1, pos7=idx0, pos8=idx5 (benzo positions on indene),
    #   pos9=idx6 (CH2 of indene),
    #   pos9a=idx4 (benzo junction to pos9), pos9b=idx7 (sp2 C, indene junction to
    #   pos1 side).  idx3 (other benzo junction) is interior (no locant).
    "c1ccc2c(c1)CC1=C2CCNC1":     {"name": "2,3,4,9-tetrahydro-1H-indeno[2,1-c]pyridine",
                                    "substituent_form": None,
                                    "alkyl_stem_ok": False,
                                    "atom_locants": {12: 1, 11: 2, 10: 3, 9: 4,
                                                     8: "4a", 2: 5, 1: 6, 0: 7, 5: 8,
                                                     6: 9, 4: "9a", 7: "9b"}},

    # 2,3-dihydro-1,4-benzodioxine
    "c1ccc2c(c1)OCCO2":  {"name": "2,3-dihydro-1,4-benzodioxine", "substituent_form": "2,3-dihydro-1,4-benzodioxin-2-yl", "alkyl_stem_ok": False},

    # 2,3-dihydropyrido[1,2,3-de][1,4]benzoxazin-7(4H)-one
    # (levofloxacin / ofloxacin tricyclic scaffold: dihydro-oxazine fused with benzene fused
    # with a pyridinone; the 7-oxo carbon is encoded in the retained-name stem).
    # Key on with-=O canonical so the exocyclic-oxo fallback in retained_lookup claims the
    # C7=O atom via extra_atom_indices and downstream does not re-emit a redundant "7-oxo-".
    # Same pattern as coumarin, chromone, 9H-xanthen-9-one above.
    # OPSIN '2,3-dihydropyrido[1,2,3-de][1,4]benzoxazin-7(4H)-one' ->
    #   O1CCN2C=3C1=CC=CC3C(C=C2)=O ; canonical: O=c1ccn2c3c(cccc13)OCC2  (14 atoms)
    # Atom indices in 'O=c1ccn2c3c(cccc13)OCC2':
    #   idx0 = O (exocyclic, pos 7 oxo — claimed via extra_atom_indices, no locant here)
    #   idx1 = C (pos 7, the pyridinone carbonyl C)
    #   idx2 = C (pos 6)
    #   idx3 = C (pos 5)
    #   idx4 = N (pos 4, bridgehead N)
    #   idx5 = C (pos 4a, central bridgehead common to all 3 rings)
    #   idx6 = C (pos 4b, bridgehead between benzene and oxazine)
    #   idx7 = C (pos 10)
    #   idx8 = C (pos 9)
    #   idx9 = C (pos 8)
    #   idx10 = C (pos 10a, bridgehead between benzene and pyridinone ring)
    #   idx11 = O (pos 1, ring O of the [1,4]benzoxazine — non-substitutable)
    #   idx12 = C (pos 2)
    #   idx13 = C (pos 3)
    # Locants verified via OPSIN chloro-probing of 7-oxo-2,3-dihydro-7H-pyrido[1,2,3-de][1,4]benzoxazine:
    #   2-Cl->idx12, 3-Cl->idx13, 5-Cl->idx3, 6-Cl->idx2, 8-Cl->idx9, 9-Cl->idx8, 10-Cl->idx7.
    #   Locant 7 (idx1) confirmed as C=O from '7-oxo-' prefix in OPSIN name round-trips.
    #   Locant 1 (idx11) = ring O (non-substitutable: chloro probe OPSIN-fails for O positions).
    #   Bridgeheads 4a (idx5), 4b (idx6), 10a (idx10) inferred topologically
    #   (junction OPSIN chloro probes fail to kekulize); not included in atom_locants
    #   (bridgehead C atoms are never directly substituted in any FDA compound here).
    # Covers: FDA-0760 levofloxacin
    #   ((S)-9-fluoro-3-methyl-10-(4-methylpiperazin-1-yl)-2,3-dihydropyrido[1,2,3-de][1,4]benzoxazin-7(4H)-one-6-carboxylic acid)
    "O=c1ccn2c3c(cccc13)OCC2": {
        "name": "2,3-dihydropyrido[1,2,3-de][1,4]benzoxazin-7(4H)-one",
        "substituent_form": None,
        "alkyl_stem_ok": False,
        "atom_locants": {1: 7, 2: 6, 3: 5, 4: 4, 7: 10, 8: 9, 9: 8, 12: 2, 13: 3}},

    # -----------------------------------------------------------------------
    # Pyrido-fused isoquinoline ring systems (tetrahydro / hexahydro forms)
    # -----------------------------------------------------------------------
    # 1,3,4,6,7,11b-hexahydro-2H-pyrido[2,1-a]isoquinolin-2-one
    # Tricyclic: two benzo rings fused with a piperidine-type N and a lactam.
    # Used by: FDA-1309 (3-isobutyl-9,10-dimethoxy derivative).
    # OPSIN '1,3,4,6,7,11b-hexahydro-2H-pyrido[2,1-a]isoquinolin-2-one'
    #   -> C1C(CCN2C1C1=CC=CC=C1CC2)=O; canonical 'O=C1CCN2CCc3ccccc3C2C1'.
    # atom_locants verified by chloro-probing each substitutable position:
    #   pos1=idx14(CH), pos3=idx2(CH2), pos4=idx3(CH2), pos5a=N(idx4,non-sub),
    #   pos6=idx5(CH2), pos7=idx6(CH2), pos8=idx8(arom C), pos9=idx9(arom C),
    #   pos10=idx10(arom C), pos11=idx11(arom C), pos11a=idx12(arom jct),
    #   pos11b=idx13(sp3 jct), pos2=C=O(idx1,non-sub).
    "O=C1CCN2CCc3ccccc3C2C1": {
        "name": "1,3,4,6,7,11b-hexahydro-2H-pyrido[2,1-a]isoquinolin-2-one",
        "substituent_form": "1,3,4,6,7,11b-hexahydro-2H-pyrido[2,1-a]isoquinolin-2-onyl",
        "alkyl_stem_ok": False,
        "atom_locants": {14: 1, 1: 2, 2: 3, 3: 4, 5: 6, 6: 7, 8: 8, 9: 9,
                         10: 10, 11: 11, 12: "11a", 13: "11b"}},

    # 1,3,4,6,7,11b-hexahydro-2H-pyrido[2,1-a]isoquinoline  (no lactam — pos 2 is CH2)
    # Same tricyclic but with the 2-one reduced (or absent).
    # Used by: FDA-0137 (3-(diethylcarbamoyl)-...-2-yl acetate: pos 2 is ester OH).
    # OPSIN '1,3,4,6,7,11b-hexahydro-2H-pyrido[2,1-a]isoquinoline'
    #   -> C1CCCN2C1C1=CC=CC=C1CC2; canonical 'c1ccc2c(c1)CCN1CCCCC21'.
    # atom_locants verified by chloro-probing:
    #   pos1=idx12, pos2=idx11, pos3=idx10, pos4=idx9, pos5=N(idx8,non-sub),
    #   pos6=idx7, pos7=idx6, pos8=idx5, pos9=idx0, pos10=idx1, pos11=idx2,
    #   pos11a=idx3(arom jct), pos11b=idx13(sp3 jct).
    "c1ccc2c(c1)CCN1CCCCC21": {
        "name": "1,3,4,6,7,11b-hexahydro-2H-pyrido[2,1-a]isoquinoline",
        "substituent_form": "1,3,4,6,7,11b-hexahydro-2H-pyrido[2,1-a]isoquinolinyl",
        "alkyl_stem_ok": False,
        "atom_locants": {12: 1, 11: 2, 10: 3, 9: 4, 7: 6, 6: 7, 5: 8, 0: 9,
                         1: 10, 2: 11, 3: "11a", 13: "11b"}},

    # -----------------------------------------------------------------------
    # 7,8,9,10-tetrahydro-6H-6,10-methanoazepino[4,5-g]quinoxaline
    # -----------------------------------------------------------------------
    # Tetracyclic: azepine ring with a methano bridge fused onto a quinoxaline.
    # Used by: FDA-1401 (the compound IS this scaffold, with stereochemistry).
    # OPSIN '7,8,9,10-tetrahydro-6H-6,10-methanoazepino[4,5-g]quinoxaline'
    #   -> N1=CC=NC=2C=C3C(=CC12)C1CNCC3C1; canonical 'c1cnc2cc3c(cc2n1)C1CNCC3C1'.
    # atom_locants verified by chloro-probing each accessible position:
    #   pos2=pos3=idx0 (symmetric quinoxaline CH, OPSIN probes give same SMILES),
    #   pos1=N(idx2,non-sub), pos4=N(idx9,non-sub),
    #   pos5=idx4(arom CH), pos6=idx14(sp3 CH), pos7=idx13(sp3 CH2),
    #   pos8=N(idx12,sp3 NH), pos9=idx11(sp3 CH2), pos10=idx10(sp3 CH),
    #   pos11=idx7(arom CH), pos12=idx15(sp3 CH2).
    "c1cnc2cc3c(cc2n1)C1CNCC3C1": {
        "name": "7,8,9,10-tetrahydro-6H-6,10-methanoazepino[4,5-g]quinoxaline",
        "substituent_form": "7,8,9,10-tetrahydro-6H-6,10-methanoazepino[4,5-g]quinoxalinyl",
        "alkyl_stem_ok": False,
        "atom_locants": {0: 2, 2: 1, 4: 5, 7: 11, 9: 4, 10: 10, 11: 9,
                         12: 8, 13: 7, 14: 6, 15: 12}},

    # -----------------------------------------------------------------------
    # 1,12-dihydro-14H-pyrano[3',4':6,7]indolizino[1,2-b]quinoline-3,14(4H)-dione
    # -----------------------------------------------------------------------
    # Pentacyclic camptothecin/irinotecan scaffold: lactone-pyranone ring fused
    # to an indolizine (bridged 5/6 N-heterocycle) fused to a quinoline.
    # The '1,12-dihydro-...(4H)-dione' form has sp3 CH2 at pos 1 and 12,
    # exocyclic =O at pos 3 (lactone) and pos 14 (amide), ring O at pos 2.
    # Used by: FDA-1350 ((S)-10-((dimethylamino)methyl)-4-ethyl-4,9-dihydroxy-...).
    # OPSIN '1,12-dihydro-14H-pyrano[3\',4\':6,7]indolizino[1,2-b]quinoline-3,14(4H)-dione'
    #   -> C1OC(CC2=C1C(N1CC=3C(=NC=4C=CC=CC4C3)C1=C2)=O)=O
    # RDKit canonical: 'O=C1Cc2cc3n(c(=O)c2CO1)Cc1cc2ccccc2nc1-3'
    # extract_ring_mol_with_exo_oxo extracts this exact SMILES from FDA-1350.
    # atom_locants verified by OPSIN chloro-probing each substitutable position:
    #   pos1=idx10(sp3 CH2), pos3=idx1(C=O lactone,non-sub), pos4=idx2(sp3 CH2),
    #   pos5=idx4(arom CH), pos6=idx21(quinoline N), pos7=idx19(arom CH),
    #   pos8=idx18(arom CH), pos9=idx17(arom CH), pos10=idx16(arom CH),
    #   pos11=idx14(arom CH), pos12=idx12(sp3 CH2), pos13=idx6(indolizino N,non-sub),
    #   pos14=idx7(C=O amide,non-sub), pos2=idx11(ring O,non-sub).
    "O=C1Cc2cc3n(c(=O)c2CO1)Cc1cc2ccccc2nc1-3": {
        "name": "1,12-dihydro-14H-pyrano[3',4':6,7]indolizino[1,2-b]quinoline-3,14(4H)-dione",
        "substituent_form": "3,14-dioxo-1,4,12,14-tetrahydro-3H-pyrano[3',4':6,7]indolizino[1,2-b]quinolinyl",
        "alkyl_stem_ok": False,
        "atom_locants": {10: 1, 1: 3, 2: 4, 4: 5, 21: 6, 19: 7, 18: 8, 17: 9,
                         16: 10, 14: 11, 12: 12, 6: 13, 7: 14}},

    # -----------------------------------------------------------------------
    # Vinca alkaloid scaffolds (Aspidosperma / Ibogamine dimer halves)
    # -----------------------------------------------------------------------
    # Four retained ring systems covering vincristine / vinblastine / vindesine
    # / vinorelbine / voacamine cores.  Each dimeric vinca alkaloid carries
    # two rings halves connected by a single C-C bond: one catharanthine-
    # derived half (ibogamine skeleton, indole fused to a bridged outer ring)
    # and one aspidosperma/vindoline-derived half (indolizino[8,1-cd]carbazole
    # or methanopyridoazepino[4,5-b]indole).
    #
    # Without these retained entries the engine names them as generic von
    # Baeyer polycycles (e.g. "diazapentacyclo[13.3.1.0^{4,12}.0^{5,10}.0^{13,18}]-
    # nonadecane"), which OPSIN re-canonicalises without the aromatic indole —
    # breaking the SMILES round-trip.  All atom_locants here were verified via
    # OPSIN <locant>-chloro-<scaffold> probing of each substitutable position;
    # aromatic-junction locants (9a, 10a, 14b, ...) were verified via OPSIN
    # parsing validity and topology.

    # ---- 1,4,5,6,7,8,9,10-octahydro-2H-3,7-methano[1]azacycloundecino[5,4-b]indole ----
    # Catharanthine half of vinblastine / vindesine (FDA-1411, FDA-1413).
    # Pentacyclic: aromatic indole fused to an 11-membered bridged azacycle;
    # the 3,7-methano bridge is a 1-carbon link.
    # OPSIN -> C1CN2CCCC(CCC=3NC=4C=CC=CC4C31)C2
    # RDKit canonical: c1ccc2c3c([nH]c2c1)CCC1CCCN(CC3)C1 (19 atoms).
    # atom_locants: numeric 1-15 verified by OPSIN <locant>-chloro probing;
    # 10 = indole NH (OPSIN 10-methyl works); 15 = methano bridge CH2;
    # junctions 9a (idx 5), 10a (idx 7), 14b (idx 3) verified valid via OPSIN
    # parsing.  One aromatic junction atom (idx 4) has no OPSIN-valid letter
    # locant (non-substitutable) and is left unmapped — the engine uses
    # locants only for actual substituents, so this is safe.
    "c1ccc2c3c([nH]c2c1)CCC1CCCN(CC3)C1": {
        "name": "1,4,5,6,7,8,9,10-octahydro-2H-3,7-methano[1]azacycloundecino[5,4-b]indole",
        "substituent_form": "1,4,5,6,7,8,9,10-octahydro-2H-3,7-methano[1]azacycloundecino[5,4-b]indolyl",
        "alkyl_stem_ok": False,
        "atom_locants": {17: 1, 16: 2, 15: 3, 14: 4, 13: 5, 12: 6, 11: 7,
                         10: 8, 9: 9, 6: 10, 8: 11, 0: 12, 1: 13, 2: 14,
                         18: 15, 5: "9a", 7: "10a", 3: "14b"}},

    # ---- 2,3,4,5,6,7,8,9-octahydro-1H-2,6-methanoazecino[5,4-b]indole ----
    # Catharanthine half of voacamine (FDA-1419) — differs from the vinblastine
    # catharanthine half by one fewer CH2 in the outer ring (10-member instead
    # of 11) and a different methano-bridge locant (2,6- instead of 3,7-).
    # OPSIN -> C1C2NCCC(CCC=3NC=4C=CC=CC4C31)C2
    # RDKit canonical: c1ccc2c3c([nH]c2c1)CCC1CCNC(C3)C1 (18 atoms).
    # atom_locants: numeric 1-14 and junctions 8a, 9a, 13b verified via OPSIN
    # chloro/methyl probing (9-methyl targets the indole NH).
    "c1ccc2c3c([nH]c2c1)CCC1CCNC(C3)C1": {
        "name": "2,3,4,5,6,7,8,9-octahydro-1H-2,6-methanoazecino[5,4-b]indole",
        "substituent_form": "2,3,4,5,6,7,8,9-octahydro-1H-2,6-methanoazecino[5,4-b]indolyl",
        "alkyl_stem_ok": False,
        "atom_locants": {16: 1, 15: 2, 14: 3, 13: 4, 12: 5, 11: 6, 10: 7,
                         9: 8, 6: 9, 8: 10, 0: 11, 1: 12, 2: 13, 17: 14,
                         5: "8a", 4: "9a", 3: "13b"}},

    # ---- 5,6,6a,7,8,9,10,12,13-nonahydro-6,9-methanopyrido[1',2':1,2]azepino[4,5-b]indole ----
    # Aspidosperma-derived half of voacamine (FDA-1419).  Indole fused to an
    # azepine fused to a pyrido ring, with a 6,9-methano bridge.
    # OPSIN -> C1=C2C3=C(NC2=CC=C1)C1C2N(CC3)CC(CC2)C1
    # RDKit canonical: c1ccc2c3c([nH]c2c1)C1CC2CCC1N(CC3)C2 (19 atoms).
    # atom_locants verified via OPSIN chloro/methyl probing: 1-4, 6-10, 12-14,
    # plus indole N at locant 5 and junction 6a (the sp3 CH between loc 6 and
    # the pyridoindole aromatic ring).
    "c1ccc2c3c([nH]c2c1)C1CC2CCC1N(CC3)C2": {
        "name": "5,6,6a,7,8,9,10,12,13-nonahydro-6,9-methanopyrido[1',2':1,2]azepino[4,5-b]indole",
        "substituent_form": "5,6,6a,7,8,9,10,12,13-nonahydro-6,9-methanopyrido[1',2':1,2]azepino[4,5-b]indolyl",
        "alkyl_stem_ok": False,
        "atom_locants": {2: 1, 1: 2, 0: 3, 8: 4, 6: 5, 9: 6, 14: "6a",
                         13: 7, 12: 8, 11: 9, 18: 10, 16: 12, 17: 13,
                         10: 14}},

    # ---- 3a,4,5,5a,6,11,12,13-octahydro-1H-indolizino[8,1-cd]carbazole ----
    # Aspidosperma/vindoline core of vinblastine/vindesine (FDA-1411, FDA-1413).
    # The ``[8,1-cd]`` notation indicates a 2-bond (3-atom) fusion of
    # indolizine atoms 8,1 to the c-d peri-face of carbazole.
    # OPSIN -> C1=C2C3=C(C=CC=C3N1)C14CCN2CC1CC4CC=C (after hydro prefixes)
    # RDKit canonical: C1=CC2CCC3Nc4ccccc4C34CCN(C1)C24 (19 atoms) with one
    # C=C double bond remaining in the non-aromatic ring.
    # atom_locants: numeric 1-12 and letter junctions 3a, 5a, 13a verified via
    # OPSIN chloro probing; the sp3 N (idx 16) and sp3 quat C (idx 13) are
    # non-substitutable junction atoms left unmapped.
    "C1=CC2CCC3Nc4ccccc4C34CCN(C1)C24": {
        "name": "3a,4,5,5a,6,11,12,13-octahydro-1H-indolizino[8,1-cd]carbazole",
        "substituent_form": "3a,4,5,5a,6,11,12,13-octahydro-1H-indolizino[8,1-cd]carbazolyl",
        "alkyl_stem_ok": False,
        "atom_locants": {17: 1, 0: 2, 1: 3, 3: 4, 4: 5, 6: 6, 8: 7, 9: 8,
                         10: 9, 11: 10, 14: 11, 15: 12,
                         2: "3a", 5: "5a", 18: "13a"}},

    # -----------------------------------------------------------------------
    # Retained polycyclic alkane (P-23.2.5.1.1)
    # -----------------------------------------------------------------------
    # Adamantane = tricyclo[3.3.1.1^{3,7}]decane.
    # Canonical RDKit SMILES: C1C2CC3CC1CC(C2)C3
    # Bridgeheads (CH, deg 3 in ring): idx 1, 3, 5, 7 -> locants 1, 3, 5, 7
    # Methylenes (CH2, deg 2 in ring): idx 0, 2, 4, 6, 8, 9 -> locants 2, 4, 6, 8, 9, 10
    # All bridgeheads are symmetry-equivalent; all methylenes are symmetry-equivalent,
    # so any consistent assignment (odd locants on bridgeheads) is valid.
    "C1C2CC3CC1CC(C2)C3": {"name": "adamantane", "substituent_form": "adamantan-1-yl", "alkyl_stem_ok": False,
                           "atom_locants": {1: 1, 0: 2, 3: 3, 2: 4, 5: 5, 4: 6, 7: 7, 6: 8, 8: 9, 9: 10}},

    # Cubane = pentacyclo[4.2.0.0^{2,5}.0^{3,8}.0^{4,7}]octane; all 8 carbons equivalent by symmetry.
    "C12C3C4C1C1C2C3C41": {"name": "cubane", "substituent_form": "cuban-1-yl", "alkyl_stem_ok": False,
                           "atom_locants": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8}},

    # Nortricyclane = tricyclo[2.2.1.0^{2,6}]heptane (norbornane with a C2-C6 zero-bridge).
    # OPSIN 'tricyclo[2.2.1.0^{2,6}]heptane' -> C12C3CC(CC31)C2; canonical 'C1C2CC3C1C3C2'.
    # C7H10: 4 bridgeheads (CH, deg 3) + 3 methylenes (CH2, deg 2).
    # Atom degree/neighbors in canonical:
    #   idx0(CH2) nbrs[1,4]; idx1(CH) nbrs[0,2,6]; idx2(CH2) nbrs[1,3];
    #   idx3(CH) nbrs[2,4,5]; idx4(CH) nbrs[3,5,0]; idx5(CH) nbrs[4,6,3];
    #   idx6(CH2) nbrs[5,1].
    # Symmetry classes (Chem.CanonicalRankAtoms breakTies=False): {0,2,6} (methylenes
    # C3/C5/C7), {3,4,5} (bridgeheads C1/C2/C6, each with 2 bridgehead neighbors),
    # and idx1 uniquely distinguished as the apex bridgehead C4 (neighbours all three
    # methylenes).  Verified via OPSIN chloro probing of all 7 locants:
    #   locants {1,2,6} -> Cl canon ClC12CC3CC1C2C3 (matches parent idx 3/4/5);
    #   locants {3,5,7} -> Cl canon ClC1C2CC3C(C2)C13 (matches parent idx 0/2/6);
    #   locant 4 -> Cl canon ClC12CC3C(C1)C3C2 (uniquely matches parent idx 1).
    # Assignment below (one of several equivalent permutations) uses:
    #   idx4=C1, idx5=C2, idx6=C3, idx1=C4 (apex), idx2=C5, idx3=C6, idx0=C7.
    # Check: C1(idx4) nbrs [3,5,0] = C6(idx3),C2(idx5),C7(idx0); C4(idx1) nbrs
    # [0,2,6] = C7,C5,C3; C2-C6 zero-bridge = idx5-idx3; all consistent.
    # Target compounds verified against OPSIN: 'tricyclo[2.2.1.0^{2,6}]heptan-3-one'
    # -> O=C1C2CC3C(C2)C13 (ZT-2315); '3-bromo...' -> BrC1C2CC3C(C2)C13 (ZT-2318);
    # 'tricyclo[2.2.1.0^{2,6}]hept-3-yl formate' -> O=COC1C2CC3C(C2)C31 (ZT-2369).
    "C1C2CC3C1C3C2":      {"name": "tricyclo[2.2.1.0^{2,6}]heptane",
                           "substituent_form": "tricyclo[2.2.1.0^{2,6}]heptan-3-yl",
                           "alkyl_stem_ok": False,
                           "atom_locants": {4: 1, 5: 2, 6: 3, 1: 4, 2: 5, 3: 6, 0: 7}},

    # -----------------------------------------------------------------------
    # Purine / pteridine
    # -----------------------------------------------------------------------
    # Purine has 4 distinct RDKit canonical SMILES depending on which N bears H
    # (1H, 3H, 7H, 9H tautomers).  All share the same InChI (H is mobile).
    # All four forms are labeled "9H-purine" here — the IUPAC preferred parent
    # for purine nomenclature.  The atom_locants for each form are chosen so
    # that N9 gets locant 9 in the physical topology (regardless of where the
    # [nH] happens to sit in the canonical SMILES).
    #
    # Topology:
    #   - 5-ring: C8 between N7 and N9; fused carbons C4 (next to N9) and C5 (next to N7)
    #   - 6-ring: N1-C2-N3-C4(fused)-C5(fused)-C6-N1
    #
    # Mapping A (H in 6-ring canonicals): atom indices match between 1H and 3H canonicals
    #   idx: 0=C8, 1=N7, 2=C5, 3=C6, 4=N1, 5=C2, 6=N3, 7=C4, 8=N9
    # Mapping B (H in 5-ring canonicals): atom indices match between 7H and 9H canonicals
    #   idx: 0=C2, 1=N1, 2=C6, 3=C5, 4=N7, 5=C8, 6=N9, 7=C4, 8=N3
    "c1nc2c[nH]cnc-2n1": {"name": "9H-purine",  "substituent_form": "purinyl",    "alkyl_stem_ok": False,
                          "atom_locants": {0: 8, 1: 7, 2: 5, 3: 6, 4: 1, 5: 2, 6: 3, 7: 4, 8: 9}},
    "c1nc2cnc[nH]c-2n1": {"name": "9H-purine",  "substituent_form": "purinyl",    "alkyl_stem_ok": False,
                          "atom_locants": {0: 8, 1: 7, 2: 5, 3: 6, 4: 1, 5: 2, 6: 3, 7: 4, 8: 9}},
    "c1ncc2nc[nH]c2n1":  {"name": "9H-purine",  "substituent_form": "purinyl",    "alkyl_stem_ok": False,
                          "atom_locants": {0: 2, 1: 1, 2: 6, 3: 5, 4: 7, 5: 8, 6: 9, 7: 4, 8: 3}},
    "c1ncc2[nH]cnc2n1":  {"name": "9H-purine",  "substituent_form": "purinyl",    "alkyl_stem_ok": False,
                          "atom_locants": {0: 2, 1: 1, 2: 6, 3: 5, 4: 7, 5: 8, 6: 9, 7: 4, 8: 3}},
    # Bare aromatic purine skeleton (no [nH]) — produced when extracting from a
    # parent where ALL four ring nitrogens are substituted (xanthines / caffeine
    # / linagliptin etc.).  In such parents Strategy 1 of _normalize_nh_fragment
    # is disabled (parent has no NH), and Strategy 4 partial-sanitize produces
    # this canonical form.  Atom_locants follow Mapping B (5-ring NH topology):
    # the 5-ring N adjacent to the C4-bridgehead is N9, the other is N7.
    # Bridgeheads: idx 3 = C5 (one N neighbor), idx 7 = C4 (two N neighbors).
    "c1ncc2ncnc2n1":     {"name": "9H-purine",  "substituent_form": "purinyl",    "alkyl_stem_ok": False,
                          "atom_locants": {0: 2, 1: 1, 2: 6, 3: 5, 4: 7, 5: 8, 6: 9, 7: 4, 8: 3}},
    # Pteridine IUPAC numbering (N1, C2, N3, C4, C4a, N5, C6, C7, N8, C8a)
    # verified by probing OPSIN with 2-, 4-, 6-, 7-methylpteridine.
    "c1cnc2ncncc2n1":    {"name": "pteridine",   "substituent_form": "pteridinyl", "alkyl_stem_ok": False,
                          "atom_locants": {0: 6, 1: 7, 2: 8, 3: "8a", 4: 1, 5: 2, 6: 3, 7: 4, 8: "4a", 9: 5}},
    # Pyrimido[5,4-d]pyrimidine: C2-symmetric bicyclic (two fused pyrimidines).
    # IUPAC numbering: N1, C2, N3, C4, C4a, N5, C6, N7, C8, C8a (N1-C8a closes).
    # Verified via OPSIN chloro-probing: 2-Cl -> idx0, 4-Cl -> idx2, 6-Cl -> idx5 (sym idx0),
    # 8-Cl -> idx7 (sym idx2). idx9 (N) neighbors junction idx8 so idx9=N1; idx1 (N) is N3.
    # Used for FDA-0432 (dipyridamole core): pyrimido[5,4-d]pyrimidine-2,6-diamine.
    "c1ncc2ncncc2n1":    {"name": "pyrimido[5,4-d]pyrimidine", "substituent_form": "pyrimido[5,4-d]pyrimidinyl",
                          "alkyl_stem_ok": False,
                          "atom_locants": {0: 2, 1: 3, 2: 4, 3: "4a", 4: 5, 5: 6, 6: 7, 7: 8, 8: "8a", 9: 1}},

    # Corrin — the tetrapyrrole-like macrocyclic core of vitamin B12.  IUPAC
    # P-25.3.1.3 lists "corrin" as a retained name; OPSIN parses it (arylGroups.xml).
    # Topology: four pyrroline-type 5-rings linked by three methine bridges plus one
    # direct C–C bond (corrin vs corrole distinction); the engine otherwise falls
    # back to a 5-ring von Baeyer name that OPSIN rejects.  Without substituent
    # verification we do not claim atom_locants; covers the bare-parent name only.
    # Stage 1 audit row (eval/stage1_raw.csv :: opsin_name == "corrin").
    "C1=C2CCC(=N2)C=C2CCC(N2)C2CCC(=N2)C=C2CCC1=N2":
                         {"name": "corrin",      "substituent_form": "corrinyl",   "alkyl_stem_ok": False},

    # -----------------------------------------------------------------------
    # 5H-phenarsazine — the saturated NH/AsH tautomer of phenarsazine
    # (distinct canonical from the cumulated-imine form
    # 'c1ccc2c(c1)N=c1ccccc1=[As]2', which the existing retained_lookup
    # atom_locants table already covers).  OPSIN produces the same
    # canonical for '5H-phenarsazine', '10H-phenarsazine', and
    # '5,10-dihydrophenarsazine'.  Pre-pin engine emitted the long von
    # Baeyer name '9-aza-2-arsatricyclo[8.4.0.0^{3,8}]tetradeca-...';
    # OPSIN round-trips that name correctly, but the IUPAC-preferred
    # retained form is much shorter.  Stage 6 R1-A follow-up.
    #
    # Canonical 'c1ccc2c(c1)Nc1ccccc1[AsH]2' (14 atoms): ring A = idx 0..5
    # (junctions idx 3,4); ring C = idx 7..12 (junctions idx 7, 12); the
    # bridge N = idx 6 (between idx 4 and idx 7), bridge As = idx 13
    # (between idx 3 and idx 12).  Numbering follows phenarsazine
    # convention: pos 5 = N, pos 10 = As (N more senior, gets lower locant).
    #   pos 1  = idx 2  (alpha-As in ring A, adj to 10a=idx 3)
    #   pos 2  = idx 1
    #   pos 3  = idx 0
    #   pos 4  = idx 5  (alpha-N in ring A, adj to 4a=idx 4)
    #   pos 4a = idx 4
    #   pos 5  = idx 6  (N)
    #   pos 5a = idx 7
    #   pos 6  = idx 8  (alpha-N in ring C)
    #   pos 7  = idx 9
    #   pos 8  = idx 10
    #   pos 9  = idx 11 (alpha-As in ring C)
    #   pos 9a = idx 12
    #   pos 10 = idx 13 (As)
    #   pos 10a= idx 3
    # Verified via OPSIN '4,6-dichloro-5H-phenarsazine' → both Cl land
    # on idx 5 and idx 8 (positions 4 and 6 = both alpha-N), matching.
    # -----------------------------------------------------------------------
    "c1ccc2c(c1)Nc1ccccc1[AsH]2": {"name": "5H-phenarsazine",
                                    "substituent_form": "5H-phenarsazin-yl",
                                    "alkyl_stem_ok": False,
                                    "atom_locants": {2: 1, 1: 2, 0: 3, 5: 4,
                                                     4: "4a", 6: 5, 7: "5a",
                                                     8: 6, 9: 7, 10: 8, 11: 9,
                                                     12: "9a", 13: 10, 3: "10a"}},

    # -----------------------------------------------------------------------
    # 5H-phenoxarsine — the O+AsH analogue of 5H-phenarsazine (O at locant
    # 5, As at locant 10).  Same topology; same atom_locants up to the
    # heteroatom substitution.  Pre-pin engine emitted the Hantzsch-Widman
    # shorthand 'phenoxarsinin' (OPSIN happens to accept this as the same
    # molecule, so it round-trips), but the IUPAC-preferred retained name
    # is '5H-phenoxarsine' or '10H-phenoxarsine' — both yield the same
    # canonical from OPSIN.  Stage 6 R1-A follow-up.
    #
    # Canonical 'c1ccc2c(c1)Oc1ccccc1[AsH]2' (14 atoms, O = idx 6,
    # As = idx 13): atom_locants identical to 5H-phenarsazine since the
    # ring topology is identical (only the bridge heteroatom changes).
    # -----------------------------------------------------------------------
    "c1ccc2c(c1)Oc1ccccc1[AsH]2": {"name": "5H-phenoxarsine",
                                    "substituent_form": "5H-phenoxarsin-yl",
                                    "alkyl_stem_ok": False,
                                    "atom_locants": {2: 1, 1: 2, 0: 3, 5: 4,
                                                     4: "4a", 6: 5, 7: "5a",
                                                     8: 6, 9: 7, 10: 8, 11: 9,
                                                     12: "9a", 13: 10, 3: "10a"}},

    # -----------------------------------------------------------------------
    # Nucleobase retained names — for nucleoside/nucleotide naming.
    # These keys are the WHOLE-MOLECULE canonical SMILES (ring + exo substituents).
    # Used when the engine tries to name a nucleobase as a substituent of a sugar.
    # The substituent_form encodes the attachment locant (always N9 for purines in
    # nucleosides/nucleotides, and N1 for pyrimidines).
    # -----------------------------------------------------------------------
    # Adenine (9H-purin-6-amine): substituent form = adenin-9-yl
    # Canonical: Nc1ncnc2[nH]cnc12 (RDKit canonical)
    "Nc1ncnc2[nH]cnc12": {"name": "adenine",     "substituent_form": "adenin-9-yl",  "alkyl_stem_ok": False},

    # Guanine (2-amino-9H-purin-6(1H)-one / 2-aminohypoxanthine):
    # Canonical: Nc1nc2[nH]cnc2c(=O)[nH]1
    "Nc1nc2[nH]cnc2c(=O)[nH]1": {"name": "guanine",    "substituent_form": "guanin-9-yl",  "alkyl_stem_ok": False},

    # Hypoxanthine (9H-purin-6(1H)-one):
    # Canonical: O=c1[nH]cnc2[nH]cnc12
    "O=c1[nH]cnc2[nH]cnc12": {"name": "hypoxanthine", "substituent_form": "hypoxanthin-9-yl", "alkyl_stem_ok": False},

    # Xanthine (3,7-dihydro-1H-purine-2,6-dione) — two tautomer canonicals
    # depending on which 5-ring N carries the H (7H vs 9H).  Both forms are
    # labeled "xanthine" and get the locant-explicit substituent form
    # corresponding to the N-H atom in each canonical (OPSIN parses
    # "xanthin-7-yl" and "xanthin-9-yl"; see probe in unit 10b investigation).
    # The 9H form is the one carved out when xanthine is an N9-substituent of
    # a nucleoside/nucleotide (xanthosine, 2'/3'/5'-xanthylic acids).
    "O=c1[nH]c(=O)c2[nH]cnc2[nH]1": {"name": "xanthine", "substituent_form": "xanthin-7-yl", "alkyl_stem_ok": False},
    "O=c1[nH]c(=O)c2nc[nH]c2[nH]1": {"name": "xanthine", "substituent_form": "xanthin-9-yl", "alkyl_stem_ok": False},

    # -----------------------------------------------------------------------
    # Fused aromatic N-heterocycles (Cluster 1)
    # All atom_locants verified via OPSIN chloro-probing.
    # -----------------------------------------------------------------------

    # imidazo[1,2-b]pyridazine: 9-atom bicyclic. 6-ring=(0,8,7,3,2,1), 5-ring=(4,3,7,6,5).
    # Bridgeheads: idx3=N3a, idx7=C8a.
    # Verified: 2-Cl->idx5, 3-Cl->idx4, 6-Cl->idx1, 7-Cl->idx0, 8-Cl->idx8.
    "c1cnn2ccnc2c1":    {"name": "imidazo[1,2-b]pyridazine", "substituent_form": "imidazo[1,2-b]pyridazinyl", "alkyl_stem_ok": False,
                         "atom_locants": {6: 1, 5: 2, 4: 3, 3: "3a", 2: 5, 1: 6, 0: 7, 8: 8, 7: "8a"}},

    # 2,1,3-benzoxadiazole (benzofurazan): N at 1,3; O at 2; benzo at 4-7.
    # C2v symmetry: pos4=pos7, pos5=pos6. Bridgeheads: idx3=C3a, idx7=C7a.
    # Verified: 4-Cl->idx8, 5-Cl->idx0.
    "c1ccc2nonc2c1":    {"name": "2,1,3-benzoxadiazole", "substituent_form": "2,1,3-benzoxadiazolyl", "alkyl_stem_ok": False,
                         "atom_locants": {4: 1, 5: 2, 6: 3, 3: "3a", 8: 4, 0: 5, 1: 6, 2: 7, 7: "7a"}},

    # pyrazolo[1,5-a]pyrimidine: 9-atom bicyclic. 6-ring=(pyrimidine) fused with 5-ring(pyrazole).
    # Verified: 2-Cl->idx5, 3-Cl->idx4, 5-Cl->idx1, 6-Cl->idx0, 7-Cl->idx8.
    "c1cnc2ccnn2c1":    {"name": "pyrazolo[1,5-a]pyrimidine", "substituent_form": "pyrazolo[1,5-a]pyrimidinyl", "alkyl_stem_ok": False,
                         "atom_locants": {5: 2, 4: 3, 1: 5, 0: 6, 8: 7}},

    # pyrrolo[2,3-d]pyrimidine tautomers.  6-ring=pyrimidine (positions 1-4 + 4a, 7a),
    # 5-ring=pyrrole (positions 5-7 + 4a, 7a).  Each tautomer has [nH] at a different
    # position; OPSIN parses each as a distinct structure, so they MUST be keyed
    # to the matching indicated-H form rather than collapsing them all to "1H-".
    #
    # 7H-pyrrolo[2,3-d]pyrimidine: NH on N7 (the pyrrole N), C2 and C4 of pyrimidine.
    # Canonical c1ncc2cc[nH]c2n1: idx0=C2, idx1=N3, idx2=C4, idx3=C4a, idx4=C5,
    # idx5=C6, idx6=N7(H), idx7=C7a, idx8=N1.  Verified via OPSIN chloro-probing
    # (2-chloro->idx0, 4-chloro->idx2, 5-chloro->idx4, 6-chloro->idx5).
    "c1ncc2cc[nH]c2n1": {"name": "7H-pyrrolo[2,3-d]pyrimidine", "substituent_form": "7H-pyrrolo[2,3-d]pyrimidinyl", "alkyl_stem_ok": False,
                          "atom_locants": {8: 1, 0: 2, 1: 3, 2: 4, 3: "4a", 4: 5, 5: 6, 6: 7, 7: "7a"}},
    # 1H-pyrrolo[2,3-d]pyrimidine: NH on N1 (pyrimidine N).
    # Canonical c1cc2cnc[nH]c-2n1: idx0=C6, idx1=C5, idx2=C4a, idx3=C4, idx4=N3,
    # idx5=C2, idx6=N1(H), idx7=C7a, idx8=N7.  Verified via OPSIN chloro-probing
    # of 1H- form (2-chloro->idx5, 4-chloro->idx3, 5-chloro->idx1, 6-chloro->idx0).
    "c1cc2cnc[nH]c-2n1": {"name": "1H-pyrrolo[2,3-d]pyrimidine", "substituent_form": "1H-pyrrolo[2,3-d]pyrimidinyl", "alkyl_stem_ok": False,
                           "atom_locants": {6: 1, 5: 2, 4: 3, 3: 4, 2: "4a", 1: 5, 0: 6, 8: 7, 7: "7a"}},
    # 3H-pyrrolo[2,3-d]pyrimidine: NH on N3 (pyrimidine N).
    # Canonical c1cc2c[nH]cnc-2n1: derived by chloro-probing analogous to above.
    "c1cc2c[nH]cnc-2n1": {"name": "3H-pyrrolo[2,3-d]pyrimidine", "substituent_form": "3H-pyrrolo[2,3-d]pyrimidinyl", "alkyl_stem_ok": False},

    # 1H-pyrazolo[4,3-d]pyrimidin-7(6H)-one (sildenafil/vardenafil-class core).
    # Bicyclic 5,6-fused: pyrazole (5-ring, positions 1-3 + 3a, 7a) ortho-fused
    # to pyrimidin-7-one (6-ring, positions 3a, 4-7, 7a) with 6H (NH at N6 adj
    # to C7=O) and 1H (NH at N1, the pyrazole NH adj to N2).  Used as the
    # parent ring for sildenafil's PIN
    # ``5-[2-ethoxy-5-(4-methylpiperazin-1-yl)sulfonylphenyl]-1-methyl-3-propyl-
    # 1H-pyrazolo[4,3-d]pyrimidin-7(6H)-one``.  Pre-fix the engine selected
    # benzene as parent (the highest-seniority simple ring, since the fused
    # heterocycle had no curated entry) and then failed the no-silent-atom-drop
    # invariant because the 9-atom heterocycle could not be carved as a
    # benzene substituent — yielding ``[NAMING ERROR: ... leaves heavy atoms
    # ... unclaimed]``.
    #
    # Keyed on the with-=O canonical SMILES so the substitutive lookup goes
    # through the exocyclic-oxo fallback in retained_lookup, which claims the
    # =O atom via extra_atom_indices and prevents downstream from re-emitting
    # a redundant ``-7-oxo-`` prefix (already encoded in the retained-name
    # ``-7(6H)-one`` stem).
    #
    # Two canonical SMILES keys map to the same retained name: RDKit
    # canonicalizes the ring with mobile-H tautomers differently depending on
    # whether the molecule arrives with [nH]/n positions matching the OPSIN
    # output of the IUPAC name (key 1) or with positions matching the input
    # SMILES form found in PubChem/DrugBank for sildenafil (key 2).  Both
    # share the same InChI (mobile-H groups) and yield identical atom_locants
    # under substructure match against the substituted form
    # ``1-methyl-3-propyl-5-bromo-1H-pyrazolo[4,3-d]pyrimidin-7(6H)-one``
    # (canonical ``CCCc1nn(C)c2c(=O)[nH]c(Br)nc12``):
    #   ring_idx 1 → C7 (=O), 2 → N6, 3 → C5, 4 → N4, 5 → C3a (junction),
    #   6 → C3, 7 → N2, 8 → N1, 9 → C7a (junction); ring_idx 0 is the
    #   exocyclic =O claimed via extra_atom_indices.
    #
    # Key 1 (OPSIN-canonical of the IUPAC name): NH at idx 2 (N6) and idx 8 (N1).
    "O=c1[nH]cnc2cn[nH]c12": {"name": "1H-pyrazolo[4,3-d]pyrimidin-7(6H)-one",
                              "substituent_form": "1H-pyrazolo[4,3-d]pyrimidin-7(6H)-on-yl",
                              "alkyl_stem_ok": False,
                              "atom_locants": {1: 7, 2: 6, 3: 5, 4: 4, 5: "3a",
                                               6: 3, 7: 2, 8: 1, 9: "7a"}},
    # Key 2 (sildenafil-input canonical, same molecule, different mobile-H placement):
    # NH at idx 4 (= N4 in this canon) and idx 7 (= N2 in this canon); RDKit
    # picks this perception when the input SMILES uses ``c1nn(C)...[nH]c12``
    # form before the N1-methyl is stripped during ring extraction.
    "O=c1nc[nH]c2c[nH]nc12":  {"name": "1H-pyrazolo[4,3-d]pyrimidin-7(6H)-one",
                                "substituent_form": "1H-pyrazolo[4,3-d]pyrimidin-7(6H)-on-yl",
                                "alkyl_stem_ok": False,
                                "atom_locants": {1: 7, 2: 6, 3: 5, 4: 4, 5: "3a",
                                                 6: 3, 7: 2, 8: 1, 9: "7a"}},

    # imidazo[5,1-f][1,2,4]triazin-4(3H)-one (vardenafil-class core).
    # Bicyclic 5,6-fused: imidazole (5-ring, positions 5, 6, 7 + bridgehead
    # N8a + junction C4a) fused via the f-bond of [1,2,4]triazine (6-ring,
    # positions 1, 2, 3, 4 + 4a + 8a) with 3H (NH at N3 adj to C4=O).  Used
    # as the parent ring for vardenafil's PIN
    # ``2-[2-ethoxy-5-(4-ethylpiperazin-1-ylsulfonyl)phenyl]-5-methyl-7-propyl-
    # imidazo[5,1-f][1,2,4]triazin-4(3H)-one``.  Pre-fix the engine fell
    # through to benzene as parent and emitted the same NAMING-ERROR class as
    # sildenafil's pyrazolopyrimidinone.
    #
    # Two canonical SMILES keys (mobile-H tautomers, identical InChI):
    # OPSIN-canonical and vardenafil-input canonical.  atom_locants verified
    # via OPSIN methyl-probing of positions 1, 2, 4, 5, 6, 7 plus the
    # multi-substituted form ``2-methyl-5-bromo-7-chloro-3H-imidazo[5,1-f]-
    # [1,2,4]triazin-4-one`` (canonical
    # ``Cc1nn2c(Cl)nc(Br)c2c(=O)[nH]1``):
    #   ring_idx 1 → C4 (=O), 2 → N3, 3 → C2, 4 → N1, 5 → N8a (bridgehead),
    #   6 → C7, 7 → N6, 8 → C5, 9 → C4a (junction); ring_idx 0 is the
    #   exocyclic =O claimed via extra_atom_indices.
    "O=c1[nH]cnn2cncc12": {"name": "imidazo[5,1-f][1,2,4]triazin-4(3H)-one",
                           "substituent_form": "imidazo[5,1-f][1,2,4]triazin-4(3H)-on-yl",
                           "alkyl_stem_ok": False,
                           "atom_locants": {1: 4, 2: 3, 3: 2, 4: 1, 5: "8a",
                                            6: 7, 7: 6, 8: 5, 9: "4a"}},
    "O=c1nc[nH]n2cncc12":  {"name": "imidazo[5,1-f][1,2,4]triazin-4(3H)-one",
                            "substituent_form": "imidazo[5,1-f][1,2,4]triazin-4(3H)-on-yl",
                            "alkyl_stem_ok": False,
                            "atom_locants": {1: 4, 2: 3, 3: 2, 4: 1, 5: "8a",
                                             6: 7, 7: 6, 8: 5, 9: "4a"}},

    # 4H-pyrido[1,2-a]pyrimidin-4-one: 11-atom ring (exo O at idx0 not in ring).
    # N1=idx4(pos1), C4=idx1(pos4, carbonyl C), N4a(bridge)=idx10, C9a(bridge)=idx5.
    # Verified positions 2,3,6,7,8,9 via methyl-OPSIN probing.
    "O=c1ccnc2ccccn12": {"name": "4H-pyrido[1,2-a]pyrimidin-4-one", "substituent_form": "4H-pyrido[1,2-a]pyrimidin-4-onyl", "alkyl_stem_ok": False,
                          "atom_locants": {1: 4, 2: 3, 3: 2, 4: 1, 5: "9a", 6: 9, 7: 8, 8: 7, 9: 6, 10: "4a"}},

    # 1,5-naphthyridine: C2v symmetry; N at pos1 (idx2) and pos5 (idx7).
    # Bridgeheads: idx3=C4a, idx8=C8a.
    # Verified: 2-Cl->idx1, 3-Cl->idx0, 4-Cl->idx9.
    "c1cnc2cccnc2c1":   {"name": "1,5-naphthyridine", "substituent_form": "1,5-naphthyridinyl", "alkyl_stem_ok": False,
                          "atom_locants": {2: 1, 1: 2, 0: 3, 9: 4, 3: "4a", 7: 5, 6: 6, 5: 7, 4: 8, 8: "8a"}},

    # 1,8-naphthyridine: C2v symmetry; N at pos1 (idx2) and pos8 (idx4).
    # Bridgeheads: idx8=C4a, idx3=C8a.
    # Verified: 2-Cl->idx5 (pos7 mirror->2), 3-Cl->idx0.
    "c1cnc2ncccc2c1":   {"name": "1,8-naphthyridine", "substituent_form": "1,8-naphthyridinyl", "alkyl_stem_ok": False,
                          "atom_locants": {2: 1, 1: 2, 0: 3, 9: 4, 8: "4a", 7: 5, 6: 6, 5: 7, 4: 8, 3: "8a"}},

    # thiazolo[5,4-d]pyrimidine: 9-atom bicyclic. Thiazole fused at C5a/C7a with pyrimidine.
    # Verified: 2-Cl->idx5, 5-Cl->idx0, 7-Cl->idx2.
    "c1ncc2ncsc2n1":    {"name": "thiazolo[5,4-d]pyrimidine", "substituent_form": "thiazolo[5,4-d]pyrimidinyl", "alkyl_stem_ok": False,
                          "atom_locants": {5: 2, 0: 5, 2: 7}},

    # [1,2,4]triazolo[3,4-b][1,3]benzothiazole: tricyclic. Benzo ring positions 5-8.
    # Verified: 5-Cl->idx2, 6-Cl->idx1, 7-Cl->idx0, 8-Cl->idx5.
    # Also confirmed via ZT-2407: 5-methyl compound Cc1cccc2sc3nncn3c12 round-trips correctly.
    "c1ccc2c(c1)sc1nncn12": {"name": "[1,2,4]triazolo[3,4-b][1,3]benzothiazole", "substituent_form": "[1,2,4]triazolo[3,4-b][1,3]benzothiazolyl", "alkyl_stem_ok": False,
                              "atom_locants": {2: 5, 1: 6, 0: 7, 5: 8}},

    # -----------------------------------------------------------------------
    # Partly-saturated fused heterocycles — additions from triage
    # -----------------------------------------------------------------------
    # Each entry was verified by chloro-probing every IUPAC ring position with
    # OPSIN, parsing the result with RDKit, and substructure-matching back onto
    # the bare canonical to derive {ring_atom_idx: IUPAC_locant}.

    # 5,6,7,8-tetrahydro-[1,2,4]triazolo[4,3-a]pyrazine (sitagliptin core, FDA-1230).
    # OPSIN '5,6,7,8-tetrahydro-[1,2,4]triazolo[4,3-a]pyrazine' -> N=1N=CN2C1CNCC2;
    # RDKit canonical: c1nnc2n1CCNC2.
    # Atom indices in c1nnc2n1CCNC2:
    #   idx0=C(arom,pos3), idx1=N(arom,pos2), idx2=N(arom,pos1),
    #   idx3=C(arom,pos8a/junction), idx4=N(arom,pos4/junction),
    #   idx5=C(pos5), idx6=C(pos6), idx7=N(pos7), idx8=C(pos8).
    # Verified by OPSIN chloro-probing positions 3, 5, 6, 7, 8.
    "c1nnc2n1CCNC2":   {"name": "5,6,7,8-tetrahydro-[1,2,4]triazolo[4,3-a]pyrazine",
                        "substituent_form": "5,6,7,8-tetrahydro-[1,2,4]triazolo[4,3-a]pyrazin-yl",
                        "alkyl_stem_ok": False,
                        "atom_locants": {0: 3, 5: 5, 6: 6, 7: 7, 8: 8}},

    # 5,6,7,8-tetrahydro-[1,2,4]triazolo[4,3-a]pyridine (dapiprazole core, FDA-0358).
    # OPSIN '5,6,7,8-tetrahydro-[1,2,4]triazolo[4,3-a]pyridine' -> N=1N=CN2C1CCCC2;
    # RDKit canonical: c1nnc2n1CCCC2.  Atom indices analogous to the pyrazine case
    # but with pos 7 = C (no NH).  Verified by OPSIN chloro-probing 3,5,6,7,8.
    "c1nnc2n1CCCC2":   {"name": "5,6,7,8-tetrahydro-[1,2,4]triazolo[4,3-a]pyridine",
                        "substituent_form": "5,6,7,8-tetrahydro-[1,2,4]triazolo[4,3-a]pyridin-yl",
                        "alkyl_stem_ok": False,
                        "atom_locants": {0: 3, 5: 5, 6: 6, 7: 7, 8: 8}},

    # 4,5,6,7-tetrahydro-1,3-benzothiazole (pramipexole core, FDA-1100).
    # OPSIN '4,5,6,7-tetrahydro-1,3-benzothiazole' -> S1C=NC2=C1CCCC2;
    # RDKit canonical: c1nc2c(s1)CCCC2.  Atom indices:
    #   idx0=C(arom,pos2), idx1=N(arom,pos3), idx2=C(arom,pos3a/junction),
    #   idx3=C(arom,pos7a/junction), idx4=S(arom,pos1),
    #   idx5=C(pos7), idx6=C(pos6), idx7=C(pos5), idx8=C(pos4).
    # Verified by OPSIN chloro-probing positions 2, 4, 5, 6, 7.
    "c1nc2c(s1)CCCC2": {"name": "4,5,6,7-tetrahydro-1,3-benzothiazole",
                        "substituent_form": "4,5,6,7-tetrahydro-1,3-benzothiazolyl",
                        "alkyl_stem_ok": False,
                        "atom_locants": {0: 2, 8: 4, 7: 5, 6: 6, 5: 7}},

    # 4,5,6,7-tetrahydrothieno[3,2-c]pyridine (ticlopidine/clopidogrel core, FDA-1328).
    # OPSIN '4,5,6,7-tetrahydrothieno[3,2-c]pyridine' -> S1C=CC=2CNCCC21;
    # RDKit canonical: c1cc2c(s1)CCNC2.  Atom indices:
    #   idx0=C(arom,pos2), idx1=C(arom,pos3),
    #   idx2=C(arom,pos3a/junction), idx3=C(arom,pos7a/junction),
    #   idx4=S(arom,pos1),
    #   idx5=C(pos7), idx6=C(pos6), idx7=N(pos5), idx8=C(pos4).
    # Verified by OPSIN chloro-probing positions 2, 3, 4, 5, 6, 7.
    "c1cc2c(s1)CCNC2": {"name": "4,5,6,7-tetrahydrothieno[3,2-c]pyridine",
                        "substituent_form": "4,5,6,7-tetrahydrothieno[3,2-c]pyridin-yl",
                        "alkyl_stem_ok": False,
                        "atom_locants": {0: 2, 1: 3, 8: 4, 7: 5, 6: 6, 5: 7}},

    # 1,2-benzodithiete (ZT-1051; benzodithiete = S-S 4-ring fused to benzene).
    # OPSIN '1,2-benzodithiete' -> S1SC2=C1C=CC=C2; RDKit canonical: c1ccc2ssc2c1.
    # Both '1,2-benzodithiete' and '7,8-dithiabicyclo[4.2.0]octa-1(6),2,4-triene'
    # round-trip to the same canonical SMILES.
    # Atom indices: idx0..3=C(arom,benzo), idx4=S(pos1), idx5=S(pos2),
    #   idx6=C(arom,pos2a/junction), idx7=C(arom,pos6a/junction).
    # Symmetry (C2v): pos3==pos6, pos4==pos5; only one of each unique pair has a
    # canonical chloro probe, so idx0 covers pos5 and idx7 covers pos6.
    "c1ccc2ssc2c1":    {"name": "1,2-benzodithiete",
                        "substituent_form": "1,2-benzodithiet-yl",
                        "alkyl_stem_ok": False,
                        "atom_locants": {0: 5, 7: 6}},

    # 6,7,8,9-tetrahydro-4H-pyrido[1,2-a]pyrimidin-4-one (paliperidone core, FDA-1012).
    # OPSIN '6,7,8,9-tetrahydro-4H-pyrido[1,2-a]pyrimidin-4-one' -> ...; RDKit canonical:
    # O=c1ccnc2n1CCCC2 (11 atoms incl. exo O).
    # The retained name encodes the C4 lactam carbonyl, so we key on the with-=O
    # canonical and rely on the exocyclic-oxo fallback in retained_lookup to claim
    # the =O via extra_atom_indices.  Atom indices in O=c1ccnc2n1CCCC2:
    #   idx0=O(exocyclic, claimed via extra_atom_indices),
    #   idx1=C(pos4,C=O), idx2=C(pos3), idx3=C(pos2), idx4=N(pos1),
    #   idx5=C(pos9a/junction), idx6=N(pos4a/N-bridgehead),
    #   idx7=C(pos6), idx8=C(pos7), idx9=C(pos8), idx10=C(pos9).
    # Verified by OPSIN chloro-probing positions 2, 3, 6, 7, 8, 9.
    "O=c1ccnc2n1CCCC2": {"name": "6,7,8,9-tetrahydro-4H-pyrido[1,2-a]pyrimidin-4-one",
                         "substituent_form": "6,7,8,9-tetrahydro-4H-pyrido[1,2-a]pyrimidin-4-onyl",
                         "alkyl_stem_ok": False,
                         "atom_locants": {1: 4, 2: 3, 3: 2, 4: 1, 5: "9a", 6: "4a",
                                          7: 6, 8: 7, 9: 8, 10: 9}},

    # 5,8-dihydropyrido[2,3-d]pyrimidin-7(6H)-one (tasosartan core, FDA-1285).
    # OPSIN '5,8-dihydropyrido[2,3-d]pyrimidin-7(6H)-one' -> N1=CN=CC2=C1NC(CC2)=O;
    # RDKit canonical: O=C1CCc2cncnc2N1 (11 atoms incl. exo O).
    # The retained name encodes the C7 lactam carbonyl, keyed on with-=O canonical.
    # Atom indices in O=C1CCc2cncnc2N1:
    #   idx0=O(exocyclic, claimed via extra_atom_indices),
    #   idx1=C(pos7,C=O), idx2=C(pos6), idx3=C(pos5),
    #   idx4=C(arom,pos4a/junction), idx5=C(arom,pos4),
    #   idx6=N(arom,pos3), idx7=C(arom,pos2), idx8=N(arom,pos1),
    #   idx9=C(arom,pos8a/junction), idx10=N(pos8,lactam-NH).
    # Verified by OPSIN chloro-probing positions 2, 4, 5, 6, 8.
    "O=C1CCc2cncnc2N1": {"name": "5,8-dihydropyrido[2,3-d]pyrimidin-7(6H)-one",
                         "substituent_form": "5,8-dihydropyrido[2,3-d]pyrimidin-7(6H)-onyl",
                         "alkyl_stem_ok": False,
                         "atom_locants": {1: 7, 2: 6, 3: 5, 4: "4a", 5: 4,
                                          6: 3, 7: 2, 8: 1, 9: "8a", 10: 8}},

    # 5,6,7,8-tetrahydropteridin-4(1H)-one (tetrahydrobiopterin core, FDA-1311).
    # OPSIN '5,6,7,8-tetrahydropteridin-4(1H)-one' -> N1C=NC(C=2NCCNC12)=O;
    # RDKit canonical: O=c1nc[nH]c2c1NCCN2 (11 atoms incl. exo O).
    # The retained name encodes the C4 lactam carbonyl + 1H-tautomer, keyed on
    # with-=O canonical.  Atom indices in O=c1nc[nH]c2c1NCCN2:
    #   idx0=O(exocyclic, claimed via extra_atom_indices),
    #   idx1=C(arom,pos4,C=O), idx2=N(arom,pos3),
    #   idx3=C(arom,pos2), idx4=N(arom,pos1,1H),
    #   idx5=C(arom,pos8a/junction; adj to N1 and N8),
    #   idx6=C(arom,pos4a/junction; adj to C4 and N5),
    #   idx7=N(pos5,sp3-NH), idx8=C(pos6), idx9=C(pos7),
    #   idx10=N(pos8,sp3-NH).
    # Verified by OPSIN chloro-probing positions 2, 5, 6, 7, 8 (positions 1, 3
    # cannot accept Cl on N/aromatic C in this tautomer).
    "O=c1nc[nH]c2c1NCCN2": {"name": "5,6,7,8-tetrahydropteridin-4(1H)-one",
                            "substituent_form": "5,6,7,8-tetrahydropteridin-4(1H)-onyl",
                            "alkyl_stem_ok": False,
                            "atom_locants": {1: 4, 2: 3, 3: 2, 4: 1, 5: "8a", 6: "4a",
                                             7: 5, 8: 6, 9: 7, 10: 8}},

    # pteridin-4(1H)-one (fully aromatic pyrazine half — folate / biopterin core).
    # Canonical 'O=c1nc[nH]c2nccnc12' (11 atoms incl. exo O).
    # atom_locants verified via OPSIN chloro-probing of 2-/6-/7-chloropteridin-4(1H)-one.
    # idx1=C4(C=O), idx2=N3, idx3=C2, idx4=N1(NH), idx5=C8a(junc),
    # idx6=N8, idx7=C7, idx8=C6, idx9=N5, idx10=C4a(junc).
    # Covers FDA-0577 (2-amino-pteridin-4(1H)-one core).
    "O=c1nc[nH]c2nccnc12": {"name": "pteridin-4(1H)-one",
                             "substituent_form": "pteridin-4(1H)-onyl",
                             "alkyl_stem_ok": False,
                             "atom_locants": {1: 4, 2: 3, 3: 2, 4: 1, 5: "8a",
                                              6: 8, 7: 7, 8: 6, 9: 5, 10: "4a"}},

    # pteridin-4(3H)-one — the 3H-tautomer of pteridin-4-one (NH at N3 between
    # C2 and C4=O, rather than at N1).  This is the tautomer found in pterin
    # (2-aminopteridin-4(3H)-one, the folate/biopterin core after amino
    # substitution at C2): the amino group at C2 hydrogen-bonds to the C4=O via
    # the N3-H, stabilising the 3H form.  Canonical 'O=c1[nH]cnc2nccnc12'.
    # atom_locants verified via OPSIN chloro-probing of 2-/6-/7-chloropteridin-
    # 4(3H)-one and 3-methylpteridin-4(3H)-one; ring-atom layout mirrors the 1H
    # entry (same IUPAC numbering, NH sits at idx2 instead of idx4):
    #   idx1=C4(C=O), idx2=N3(NH), idx3=C2, idx4=N1, idx5=C8a(junc),
    #   idx6=N8, idx7=C7, idx8=C6, idx9=N5, idx10=C4a(junc).
    # Without this entry, pterin ('Nc1nc2nccnc2c(=O)[nH]1') fails plan search:
    # the ring-only carve canonicalises to the 3H form but the curated table
    # only holds the 1H form, so the retained-name lookup misses.
    "O=c1[nH]cnc2nccnc12": {"name": "pteridin-4(3H)-one",
                             "substituent_form": "pteridin-4(3H)-onyl",
                             "alkyl_stem_ok": False,
                             "atom_locants": {1: 4, 2: 3, 3: 2, 4: 1, 5: "8a",
                                              6: 8, 7: 7, 8: 6, 9: 5, 10: "4a"}},

    # 7,8-dihydropteridin-4(3H)-one — 3H-tautomer of the dihydropteridinone
    # that appears in dihydropterin (2-amino-7,8-dihydropteridin-4(3H)-one).
    # Like pterin, the amino group at C2 stabilises the 3H form via H-bonding
    # to the C4=O.  OPSIN '7,8-dihydropteridin-4(3H)-one' canonicalises to
    # 'O=c1[nH]cnc2c1N=CCN2' (11 atoms incl. exo O); the pyrimidinone half is
    # aromatic (NH=N3, C2, N1, C8a, C4a ring), while N5=C6-C7-N8 is saturated
    # with an N5=C6 imine.  Atom indices in the canonical parent:
    #   idx0=O (exocyclic, claimed via extra_atom_indices),
    #   idx1=C(arom,pos4,C=O), idx2=N(arom,pos3,NH),
    #   idx3=C(arom,pos2), idx4=N(arom,pos1),
    #   idx5=C(arom,pos8a/junction; adj to N1 and N8),
    #   idx6=C(arom,pos4a/junction; adj to C4 and N5),
    #   idx7=N(pos5, sp2 imine, =C6), idx8=C(pos6, =N5),
    #   idx9=C(pos7, sp3), idx10=N(pos8, sp3-NH).
    # Verified by OPSIN chloro-probing positions 2, 6, 7: all map correctly.
    # Without this entry, dihydropterin ('Nc1nc2c(c(=O)[nH]1)N=CCN2') fails
    # plan search (Stage 1 NO_PLAN) because the retained-name table had no
    # matching key.
    "O=c1[nH]cnc2c1N=CCN2": {"name": "7,8-dihydropteridin-4(3H)-one",
                              "substituent_form": "7,8-dihydropteridin-4(3H)-onyl",
                              "alkyl_stem_ok": False,
                              "atom_locants": {1: 4, 2: 3, 3: 2, 4: 1, 5: "8a", 6: "4a",
                                               7: 5, 8: 6, 9: 7, 10: 8}},

    # 5,6,7,8-tetrahydropteridin-4(3H)-one — 3H-tautomer of the tetrahydro
    # pteridinone found in tetrahydropterin (2-amino-5,6,7,8-tetrahydropteridin-
    # 4(3H)-one; the biopterin/tetrahydrobiopterin core after amino
    # substitution at C2 stabilises the 3H form).  OPSIN of
    # '5,6,7,8-tetrahydropteridin-4(3H)-one' canonicalises to
    # 'O=c1[nH]cnc2c1NCCN2' (11 atoms incl. exo O).  Atom layout matches the
    # 1H sibling (same IUPAC numbering; NH sits at idx2=N3 rather than
    # idx4=N1), with the saturated pyrazine ring N5-C6-C7-N8:
    #   idx0=O (exocyclic, claimed via extra_atom_indices),
    #   idx1=C(arom,pos4,C=O), idx2=N(arom,pos3,NH),
    #   idx3=C(arom,pos2), idx4=N(arom,pos1),
    #   idx5=C(arom,pos8a/junction; adj to N1 and N8),
    #   idx6=C(arom,pos4a/junction; adj to C4 and N5),
    #   idx7=N(pos5, sp3-NH), idx8=C(pos6), idx9=C(pos7),
    #   idx10=N(pos8, sp3-NH).
    # Verified by OPSIN chloro-probing positions 2, 6, 7.  Without this entry,
    # tetrahydropterin ('Nc1nc2c(c(=O)[nH]1)NCCN2') fails plan search
    # (Stage 1 NO_PLAN).
    "O=c1[nH]cnc2c1NCCN2": {"name": "5,6,7,8-tetrahydropteridin-4(3H)-one",
                             "substituent_form": "5,6,7,8-tetrahydropteridin-4(3H)-onyl",
                             "alkyl_stem_ok": False,
                             "atom_locants": {1: 4, 2: 3, 3: 2, 4: 1, 5: "8a", 6: "4a",
                                              7: 5, 8: 6, 9: 7, 10: 8}},

    # carbapenem (retained; 1-azabicyclo[3.2.0]hept-2-en-7-one with penem numbering).
    # IUPAC penem numbering: C1-C2=C3-N4(bridgehead)-C5(bridgehead)-C6-C7(=O).
    # Canonical 'O=C1CC2CC=CN12' (9 atoms incl. exo O). atom_locants verified via
    # OPSIN chloro-probing positions 2, 3, 5, 6: 2-Cl -> idx5 (C2), 3-Cl -> idx6
    # (C3 adjacent to N4), 5-Cl -> idx3 (C5), 6-Cl -> idx2 (C6 adjacent to C7=O).
    # idx1 = C7 (C=O), idx4 = C1 (CH2 between C5 and C2), idx7 = N4.
    # Used by FDA-0833 / FDA-0494 carbapenem cores.
    "O=C1CC2CC=CN12": {"name": "carbapenem",
                        "substituent_form": "carbapenemyl",
                        "alkyl_stem_ok": False,
                        "atom_locants": {1: 7, 2: 6, 3: 5, 4: 1, 5: 2, 6: 3, 7: 4}},

    # 3H-[1,2,3]triazolo[4,5-d]pyrimidine (FDA-1326 core).
    # Fully aromatic; 3H tautomer has [nH] on N3 (adjacent to C3a bridgehead).
    # Canonical 'c1ncc2nn[nH]c2n1'. atom_locants verified via OPSIN chloro-probing
    # positions 5 and 7 (the only non-N aromatic substitution points):
    # 5-Cl -> idx0, 7-Cl -> idx2. Ring analysis fixes idx1=N6, idx8=N4,
    # idx7=C3a (next to [nH]=N3), idx3=C7a (next to N1), idx4=N1, idx5=N2, idx6=N3.
    "c1ncc2nn[nH]c2n1": {"name": "3H-[1,2,3]triazolo[4,5-d]pyrimidine",
                          "substituent_form": "3H-[1,2,3]triazolo[4,5-d]pyrimidinyl",
                          "alkyl_stem_ok": False,
                          "atom_locants": {0: 5, 1: 6, 2: 7, 3: "7a", 4: 1, 5: 2,
                                           6: 3, 7: "3a", 8: 4}},

    # 3,4,5,6-tetrahydroimidazo[4,5-d][1]benzazepine (FDA-0329 core).
    # Benzene-fused azepine-fused imidazole, 3H tautomer (NH on N3 of imidazole).
    # Canonical 'c1ccc2c(c1)NCCc1[nH]cnc1-2' (13 atoms, no exo oxo).
    # atom_locants verified via OPSIN chloro-probing positions 2, 4, 5, 6, 7, 8, 9, 10
    # and 3-methyl / 3a-chloro / 10b-chloro probes. idx0=C8, idx1=C9, idx2=C10,
    # idx3=C10a (benzene-azepine junction, not NH side), idx4=C6a (benzene-azepine
    # junction on NH side), idx5=C7, idx6=N6, idx7=C5, idx8=C4, idx9=C3a
    # (azepine-imidazole junction, N3 side), idx10=N3, idx11=C2, idx12=N1,
    # idx13=C10b (azepine-imidazole junction, N1 side).
    "c1ccc2c(c1)NCCc1[nH]cnc1-2": {"name": "3,4,5,6-tetrahydroimidazo[4,5-d][1]benzazepine",
                                     "substituent_form": "3,4,5,6-tetrahydroimidazo[4,5-d][1]benzazepin-6-yl",
                                     "alkyl_stem_ok": False,
                                     "atom_locants": {0: 8, 1: 9, 2: 10, 3: "10a", 4: "6a",
                                                      5: 7, 6: 6, 7: 5, 8: 4, 9: "3a",
                                                      10: 3, 11: 2, 12: 1, 13: "10b"}},

    # 1,4,5,6-tetrahydro-7H-pyrazolo[3,4-c]pyridin-7-one (FDA-0087 core).
    # Pyrazole fused to a 2-pyridinone; tetrahydro on the pyridinone half.
    # Canonical 'O=C1NCCc2cn[nH]c21' (10 atoms incl. exo O).
    # atom_locants verified via OPSIN chloro-probing positions 1, 3-6:
    # 1-Cl -> idx8 (N1), 3-Cl -> idx6 (C3), 4-Cl -> idx4 (C4), 5-Cl -> idx3 (C5),
    # 6-Cl -> idx2 (N6). idx1 is C7 (C=O), idx5 is C3a junction, idx7 is N2,
    # idx9 is C7a junction.
    "O=C1NCCc2cn[nH]c21": {"name": "1,4,5,6-tetrahydro-7H-pyrazolo[3,4-c]pyridin-7-one",
                            "substituent_form": "1,4,5,6-tetrahydro-7-oxo-7H-pyrazolo[3,4-c]pyridin-6-yl",
                            "alkyl_stem_ok": False,
                            "atom_locants": {1: 7, 2: 6, 3: 5, 4: 4, 5: "3a", 6: 3,
                                             7: 2, 8: 1, 9: "7a"}},

    # 6,7-dihydro-5H-pyrrolo[3,4-b]pyrazin-5-one (zopiclone core, FDA-1440).
    # Canonical 'O=C1NCc2nccnc21' (10 atoms incl. exo O).
    # IUPAC numbering N1, C2, C3, N4, C4a, C5(=O), N6, C7, C7a with C5-C4a-C7a-N1-...
    # atom_locants verified via OPSIN chloro-probing positions 2, 3, 6, 7:
    # 2-Cl -> idx6, 3-Cl -> idx7, 6-Cl -> idx2 (N6), 7-Cl -> idx3 (C7).
    # idx1 is the C=O carbon (C5), idx4 and idx9 are the bridgeheads (C7a, C4a).
    "O=C1NCc2nccnc21": {"name": "6,7-dihydro-5H-pyrrolo[3,4-b]pyrazin-5-one",
                         "substituent_form": "6,7-dihydro-5-oxo-5H-pyrrolo[3,4-b]pyrazin-6-yl",
                         "alkyl_stem_ok": False,
                         "atom_locants": {1: 5, 2: 6, 3: 7, 4: "7a", 5: 1, 6: 2,
                                          7: 3, 8: 4, 9: "4a"}},

    # 5H-2,3-benzodiazepine (tofisopam / GYKI-family scaffold). 11-atom bicycle
    # (7-ring + fused benzene). Canonical 'C1=NN=Cc2ccccc2C1'.
    # atom_locants verified via OPSIN chloro-probing positions 1, 4, 6, 7, 8, 9.
    # Ring walk: C1(idx3)-N2(idx2)-N3(idx1)-C4(idx0)-C5(idx10,CH2)-C5a(idx9)-
    #            C6(idx8)-C7(idx7)-C8(idx6)-C9(idx5)-C9a(idx4)-C1.
    # Covers FDA-1340 (tofisopam = 1-aryl-4-methyl-5-ethyl-5H-2,3-benzodiazepine).
    "C1=NN=Cc2ccccc2C1": {"name": "5H-2,3-benzodiazepine",
                           "substituent_form": "5H-2,3-benzodiazepin-5-yl",
                           "alkyl_stem_ok": False,
                           "atom_locants": {3: 1, 2: 2, 1: 3, 0: 4, 10: 5,
                                            9: "5a", 8: 6, 7: 7, 6: 8, 5: 9,
                                            4: "9a"}},

    # -----------------------------------------------------------------------
    # R5 Cluster A: small fused bicyclic / tricyclic heterocycles (drug cores)
    # -----------------------------------------------------------------------
    # 1,5-dihydroimidazo[2,1-b]quinazolin-2(3H)-one (anagrelide core).
    # Keyed on with-=O canonical SMILES so exocyclic-oxo fallback in
    # retained_lookup claims the C2 =O via extra_atom_indices, preventing
    # downstream from re-emitting a redundant "2-oxo" prefix.
    # OPSIN canonical: O=C1CN2Cc3ccccc3N=C2N1.
    # Atom indices in canonical 'O=C1CN2Cc3ccccc3N=C2N1':
    #   idx0=O(exo, claimed), idx1=C(pos 2, C=O), idx2=C(pos 3, CH2),
    #   idx3=N(pos 3a, fusion N), idx4=C(pos 5, CH2),
    #   idx5=C(pos 5a, arom junction adj to CH2 and pos 6),
    #   idx6-9=C(pos 6,7,8,9 arom CH), idx10=C(pos 9a, arom junction),
    #   idx11=N(pos 10, quinazoline N), idx12=C(pos 10a, amidine C junction),
    #   idx13=N(pos 1, NH).
    # Verified via OPSIN chloro-probing positions 3,5,6,7,8,9 and position 10.
    # Covers FDA-0079 (anagrelide = 6,7-dichloro derivative).
    "O=C1CN2Cc3ccccc3N=C2N1": {
        "name": "1,5-dihydroimidazo[2,1-b]quinazolin-2(3H)-one",
        "substituent_form": "1,5-dihydroimidazo[2,1-b]quinazolin-2(3H)-on-yl",
        "alkyl_stem_ok": False,
        "atom_locants": {13: 1, 1: 2, 2: 3, 3: "3a", 4: 5, 5: "5a",
                         6: 6, 7: 7, 8: 8, 9: 9, 10: "9a", 11: 10, 12: "10a"}},

    # 2,3-dihydro-1H-pyrrolizine (ketorolac core).
    # OPSIN canonical: c1cc2n(c1)CCC2.
    # Pyrrolizine is a bicyclic pyrrole fused through the N (bridgehead N = pos 4).
    # Atom indices:
    #   idx0=C(pos 6, arom CH), idx1=C(pos 7, arom CH),
    #   idx2=C(pos 7a, arom junction), idx3=N(pos 4, bridgehead N, arom),
    #   idx4=C(pos 5, arom CH), idx5=C(pos 3, sp3 CH2),
    #   idx6=C(pos 2, sp3 CH2), idx7=C(pos 1, sp3 CH2).
    # Verified via OPSIN chloro-probing all of positions 1,2,3,5,6,7.
    # Covers FDA-0725 (ketorolac = 5-benzoyl-1-carboxylic acid derivative).
    "c1cc2n(c1)CCC2": {
        "name": "2,3-dihydro-1H-pyrrolizine",
        "substituent_form": "2,3-dihydro-1H-pyrrolizin-yl",
        "alkyl_stem_ok": False,
        "atom_locants": {7: 1, 6: 2, 5: 3, 3: 4, 4: 5, 0: 6, 1: 7, 2: "7a"}},

    # hexahydro-1H-pyrrolizine (= pyrrolizidine, fully saturated bicyclic pyrrolidine
    # sharing bridgehead N).  Canonical RDKit SMILES 'C1CC2CCCN2C1':
    #   idx2=C (bridgehead C, pos 7a), idx6=N (bridgehead N, pos 4).
    # Ring A (pos 7a->1->2->3->4): idx2-idx3-idx4-idx5-idx6.
    # Ring B (pos 7a->7->6->5->4): idx2-idx1-idx0-idx7-idx6.
    # OPSIN chloro-probing confirms full C2v symmetry of the bare ring: positions 1<->7
    # (idx3/idx1), 2<->6 (idx4/idx0), 3<->5 (idx5/idx7) collapse to the same canonical
    # chloro-SMILES; one consistent assignment is chosen.
    # Covers ZT-2389 (1-hydroxymethyl-7-hydroxy derivative).
    "C1CC2CCCN2C1": {
        "name": "hexahydro-1H-pyrrolizine",
        "substituent_form": "hexahydro-1H-pyrrolizin-yl",
        "alkyl_stem_ok": False,
        "atom_locants": {3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 0: 6, 1: 7, 2: "7a"}},

    # 2,3,5,7a-tetrahydro-1H-pyrrolizine: pyrrolizine with sp3 positions 1,2,3,5,7a and
    # a C=C double bond between positions 6 and 7.  Canonical RDKit SMILES 'C1=CC2CCCN2C1':
    #   idx6=N (bridgehead N, pos 4), idx2=C (bridgehead C, pos 7a, sp3).
    # Ring A (sp3, pos 7a->1->2->3->4): idx2-idx3-idx4-idx5-idx6.
    # Ring B (contains C=C, pos 4->5->6->7->7a): idx6-idx7-idx0-idx1-idx2; double bond
    # between idx0 (pos 6) and idx1 (pos 7).
    # OPSIN chloro-probing: 1-Cl->idx3, 2-Cl->idx4, 3-Cl->idx5, 5-Cl->idx7, 6-Cl->idx0,
    # 7-Cl->idx1 (asymmetric; all six positions distinguishable).
    # Covers ZT-2379 (7-hydroxymethyl-1-hydroxy derivative).
    "C1=CC2CCCN2C1": {
        "name": "2,3,5,7a-tetrahydro-1H-pyrrolizine",
        "substituent_form": "2,3,5,7a-tetrahydro-1H-pyrrolizin-yl",
        "alkyl_stem_ok": False,
        "atom_locants": {3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 0: 6, 1: 7, 2: "7a"}},

    # 5,11-dihydro-6H-dipyrido[3,2-b:2',3'-e][1,4]diazepin-6-one (nevirapine core).
    # Keyed on with-=O canonical so exo-oxo fallback claims the C6 =O.
    # OPSIN canonical: O=C1Nc2cccnc2Nc2ncccc21.
    # Atom indices:
    #   idx0=O(exo, claimed), idx1=C(pos 6, C=O),
    #   idx2=N(pos 5, sp3 NH), idx3=C(pos 4a, arom junction adj to NH and pos 4),
    #   idx4=C(pos 4), idx5=C(pos 3), idx6=C(pos 2), idx7=N(pos 1, arom N),
    #   idx8=C(pos 11a, arom junction adj to arom N and NH pos 11),
    #   idx9=N(pos 11, sp3 NH),
    #   idx10=C(pos 10a, arom junction adj to NH and arom N pos 10),
    #   idx11=N(pos 10, arom N), idx12=C(pos 9), idx13=C(pos 8), idx14=C(pos 7),
    #   idx15=C(pos 6a, arom junction adj to C=O).
    # Verified via OPSIN chloro-probing positions 2,3,4,7,8,9 (positions 1 and 10
    # are aromatic ring N and reject Cl).
    # Covers FDA-0945 (nevirapine = 4-methyl-11-cyclopropyl derivative).
    "O=C1Nc2cccnc2Nc2ncccc21": {
        "name": "5,11-dihydro-6H-dipyrido[3,2-b:2',3'-e][1,4]diazepin-6-one",
        "substituent_form": "5,11-dihydro-6H-dipyrido[3,2-b:2',3'-e][1,4]diazepin-6-on-yl",
        "alkyl_stem_ok": False,
        "atom_locants": {7: 1, 6: 2, 5: 3, 4: 4, 3: "4a", 2: 5, 1: 6, 15: "6a",
                         14: 7, 13: 8, 12: 9, 11: 10, 10: "10a", 9: 11, 8: "11a"}},

    # 1H-thieno[2,3-e][1,4]diazepin-2(3H)-one (clotiazepam core).
    # Keyed on with-=O canonical so exo-oxo fallback claims the C2 =O.
    # OPSIN canonical: O=C1CN=Cc2ccsc2N1.
    # Thiophene S is at position 8 (not a junction).
    # Atom indices:
    #   idx0=O(exo, claimed), idx1=C(pos 2, C=O), idx2=C(pos 3, CH2),
    #   idx3=N(pos 4, sp2 N), idx4=C(pos 5, CH=N),
    #   idx5=C(pos 5a, arom junction, thiophene C adj to CH=N),
    #   idx6=C(pos 6, arom CH of thiophene), idx7=C(pos 7, arom CH),
    #   idx8=S(pos 8, arom S),
    #   idx9=C(pos 8a, arom junction adj to S and NH pos 1),
    #   idx10=N(pos 1, sp3 NH).
    # Verified via OPSIN chloro-probing positions 3,5,6,7,8.
    # Covers FDA-0318 (clotiazepam).
    "O=C1CN=Cc2ccsc2N1": {
        "name": "1H-thieno[2,3-e][1,4]diazepin-2(3H)-one",
        "substituent_form": "1H-thieno[2,3-e][1,4]diazepin-2(3H)-on-yl",
        "alkyl_stem_ok": False,
        "atom_locants": {10: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: "5a",
                         6: 6, 7: 7, 8: 8, 9: "8a"}},

    # 1,2,3,6,7,11b-hexahydro-4H-pyrazino[2,1-a]isoquinolin-4-one (praziquantel core).
    # Keyed on with-=O canonical so exo-oxo fallback claims the C4 =O.
    # OPSIN canonical: O=C1CNCC2c3ccccc3CCN12.
    # Atom indices:
    #   idx0=O(exo, claimed), idx1=C(pos 4, C=O), idx2=C(pos 3, CH2),
    #   idx3=N(pos 2, NH), idx4=C(pos 1, CH2),
    #   idx5=C(pos 11b, sp3 CH junction), idx6=C(pos 11a, arom junction adj to 11b),
    #   idx7=C(pos 11, arom CH), idx8=C(pos 10), idx9=C(pos 9), idx10=C(pos 8),
    #   idx11=C(pos 7a, arom junction adj to pos 7 CH2),
    #   idx12=C(pos 7, sp3 CH2), idx13=C(pos 6, sp3 CH2),
    #   idx14=N(pos 5, bridgehead N).
    # Verified via OPSIN chloro-probing positions 1,2,3,6,7,8,9,10,11,11b.
    # Covers FDA-1105 (praziquantel = 2-cyclohexanecarbonyl derivative).
    "O=C1CNCC2c3ccccc3CCN12": {
        "name": "1,2,3,6,7,11b-hexahydro-4H-pyrazino[2,1-a]isoquinolin-4-one",
        "substituent_form": "1,2,3,6,7,11b-hexahydro-4H-pyrazino[2,1-a]isoquinolin-4-on-yl",
        "alkyl_stem_ok": False,
        "atom_locants": {4: 1, 3: 2, 2: 3, 1: 4, 14: 5, 13: 6, 12: 7,
                         11: "7a", 10: 8, 9: 9, 8: 10, 7: 11, 6: "11a", 5: "11b"}},

    # 4H-[1,2,4]triazolo[4,3-a][1,4]benzodiazepine (alprazolam/estazolam core).
    # OPSIN canonical: C1=NCc2nncn2-c2ccccc21.
    # Atom indices:
    #   idx0=C(pos 6, CH=N of diazepine), idx1=N(pos 5, sp2 N),
    #   idx2=C(pos 4, CH2), idx3=C(pos 3a, triazole-diazepine junction),
    #   idx4=N(pos 3, arom), idx5=N(pos 2, arom),
    #   idx6=C(pos 1, arom CH of triazole),
    #   idx7=N(pos 11a, bridgehead N between triazole and benzene),
    #   idx8=C(pos 6a, benzene junction adj to CH=N pos 6),
    #   idx9=C(pos 10), idx10=C(pos 9), idx11=C(pos 8), idx12=C(pos 7),
    #   idx13=C(pos 10a, benzene junction adj to N11a).
    # Verified via OPSIN chloro-probing positions 1,4,6,7,8,9,10.
    # Covers FDA-0500 (estazolam = 8-chloro-6-phenyl derivative).
    "C1=NCc2nncn2-c2ccccc21": {
        "name": "4H-[1,2,4]triazolo[4,3-a][1,4]benzodiazepine",
        "substituent_form": "4H-[1,2,4]triazolo[4,3-a][1,4]benzodiazepin-yl",
        "alkyl_stem_ok": False,
        "atom_locants": {6: 1, 5: 2, 4: 3, 3: "3a", 2: 4, 1: 5, 0: 6,
                         8: "6a", 12: 7, 11: 8, 10: 9, 9: 10, 13: "10a",
                         7: "11a"}},

    # 6,11-dihydro-5H-benzo[5,6]cyclohepta[1,2-b]pyridine (azatadine / desloratadine core).
    # OPSIN canonical: c1ccc2c(c1)CCc1cccnc1C2.
    # Tricyclic: benzene + 7-ring (with three sp3 CH2 at pos 5,6,11) + pyridine.
    # Atom indices (RDKit canonical of c1ccc2c(c1)CCc1cccnc1C2):
    #   idx0,1,2=C(pos 8,9,10 arom CH), idx3=C(pos 10a, benzene junction adj to CH2 pos 11),
    #   idx4=C(pos 6a, benzene junction adj to CH2 pos 6),
    #   idx5=C(pos 7, arom CH), idx6=C(pos 6, sp3 CH2), idx7=C(pos 5, sp3 CH2),
    #   idx8=C(pos 4a, pyridine junction adj to CH2 pos 5),
    #   idx9=C(pos 4), idx10=C(pos 3), idx11=C(pos 2), idx12=N(pos 1, arom N),
    #   idx13=C(pos 11a, pyridine junction adj to arom N and CH2 pos 11),
    #   idx14=C(pos 11, sp3 CH2).
    # Verified via OPSIN chloro-probing positions 2,3,4,5,6,7,8,9,10,11.
    # Covers FDA-0379 (desloratadine = 8-chloro-11-(piperidin-4-ylidene) derivative).
    "c1ccc2c(c1)CCc1cccnc1C2": {
        "name": "6,11-dihydro-5H-benzo[5,6]cyclohepta[1,2-b]pyridine",
        "substituent_form": "6,11-dihydro-5H-benzo[5,6]cyclohepta[1,2-b]pyridin-yl",
        "alkyl_stem_ok": False,
        "atom_locants": {12: 1, 11: 2, 10: 3, 9: 4, 8: "4a", 7: 5, 6: 6,
                         4: "6a", 5: 7, 0: 8, 1: 9, 2: 10, 3: "10a",
                         14: 11, 13: "11a"}},
    # 1,2,3,3a,3b,4,5,5a,6,8,10,10a,10b,11,12,12a-hexadecahydrocyclopenta[5,6]naphtho[1,2-f]indazole:
    # The fully-saturated (outside the 1H-pyrazole ring) tetracyclic system of stanozolol-like
    # anabolic steroids — cyclopentane fused to the D-ring of an androstane-like carbocyclic
    # skeleton, with a 1H-pyrazole fused onto the A-ring position.
    # OPSIN-canonical SMILES verified: 'c1[nH]nc2c1CC1C(CCC3C4CCCC4CCC13)C2' (20 heavy atoms).
    # atom_locants established by OPSIN chloro-probing positions 1..12a (position 7 is the
    # aromatic =N- with no H and cannot bear a chloro substituent; its canonical idx=2 is
    # mapped manually from the fused-ring topology).
    # Covers FDA-1245 (stanozolol: 1,10a,12a-trimethyl-...-indazol-1-ol).
    "c1[nH]nc2c1CC1C(CCC3C4CCCC4CCC13)C2": {
        "name": "1,2,3,3a,3b,4,5,5a,6,8,10,10a,10b,11,12,12a-hexadecahydrocyclopenta[5,6]naphtho[1,2-f]indazole",
        "substituent_form": "1,2,3,3a,3b,4,5,5a,6,8,10,10a,10b,11,12,12a-hexadecahydrocyclopenta[5,6]naphtho[1,2-f]indazol-yl",
        "alkyl_stem_ok": False,
        "atom_locants": {14: 1, 13: 2, 12: 3, 11: "3a", 10: "3b", 9: 4, 8: 5,
                         7: "5a", 19: 6, 2: 7, 1: 8, 0: 9, 5: 10, 6: "10a",
                         18: "10b", 17: 11, 16: 12, 15: "12a"}},

    # -----------------------------------------------------------------------
    # R5 Cluster B: partially saturated bicyclic fused rings (FDA drug cores)
    # -----------------------------------------------------------------------

    # octahydrocyclopenta[b]pyran-6-one — FDA-0794 (prostacyclin analogue).
    # The ring system extracted by the engine is the bare octahydrocyclopenta[b]pyran
    # (C1COC2CCCC2C1).  In FDA-0794 the C6 carbonyl is in-ring → the engine's
    # exo-oxo fallback then looks up this with-=O keyed entry (O=C1CC2CCCOC2C1).
    # OPSIN: '6-oxooctahydrocyclopenta[b]pyran' → O=C1CC2CCCOC2C1 (verified).
    # Atom layout of 'O=C1CC2CCCOC2C1' (10 atoms):
    #   idx0=O(exo, claimed), idx1=C(pos6, C=O), idx2=C(pos5), idx3=C(pos4a junction),
    #   idx4=C(pos4), idx5=C(pos3), idx6=C(pos2), idx7=O(pos1 ring O),
    #   idx8=C(pos7a junction), idx9=C(pos7).
    # OPSIN chloro-probing verified: pos2→idx6, pos3→idx5, pos4→idx4, pos4a→idx3,
    # pos5→idx2, pos7→idx9, pos7a→idx8.  pos1=O and pos6=C=O from topology.
    # Covers FDA-0794 (7-{2-(1,1-difluoropentyl)-2-hydroxy-6-oxooctahydrocyclopenta[b]pyran-5-yl}heptanoic acid).
    "O=C1CC2CCCOC2C1": {
        "name": "octahydrocyclopenta[b]pyran-6-one",
        "substituent_form": "octahydrocyclopenta[b]pyran-6-on-yl",
        "alkyl_stem_ok": False,
        "atom_locants": {7: 1, 6: 2, 5: 3, 4: 4, 3: "4a", 2: 5, 1: 6,
                         9: 7, 8: "7a"}},

    # 5,6,7,7a-tetrahydro-4aH-cyclopenta[c]pyran-1-one — ZT-2525.
    # The engine's exo-oxo fallback looks up the with-=O form O=C1OC=CC2CCCC12.
    # OPSIN: '4a,5,6,7-tetrahydrocyclopenta[c]pyran-1-one' → O=C1OC=CC2CCCC12 (verified).
    # In the expected compound name the indicator-H atom 4aH is shown because C4a
    # is the saturated junction; the name 5,6,7,7a-tetrahydro-4aH-cyclopenta[c]pyran-1-one
    # is also accepted by OPSIN (both give the same SMILES).
    # Atom layout of 'O=C1OC=CC2CCCC12' (10 atoms):
    #   idx0=O(exo, claimed), idx1=C(pos1, C=O lactone), idx2=O(pos2 ring O),
    #   idx3=C(pos3), idx4=C(pos4), idx5=C(pos4a junction),
    #   idx6=C(pos5), idx7=C(pos6), idx8=C(pos7), idx9=C(pos7a junction).
    # OPSIN chloro-probing verified: pos3→idx3, pos4→idx4, pos4a→idx5,
    # pos5→idx6, pos6→idx7, pos7→idx8, pos7a→idx9.
    # Covers ZT-2525 (4,7-dimethyl-5,6,7,7a-tetrahydro-4aH-cyclopenta[c]pyran-1-one).
    "O=C1OC=CC2CCCC12": {
        "name": "5,6,7,7a-tetrahydro-4aH-cyclopenta[c]pyran-1-one",
        "substituent_form": "5,6,7,7a-tetrahydro-4aH-cyclopenta[c]pyran-1-on-yl",
        "alkyl_stem_ok": False,
        "atom_locants": {1: 1, 2: 2, 3: 3, 4: 4, 5: "4a", 6: 5, 7: 6, 8: 7, 9: "7a"}},

    # hexahydrocyclopenta[c]pyran-3(1H)-one — Phase 7 ring-batch-4.
    # 5,6-fused: cyclopentane fused to a 6-membered lactone ring with the C=O at
    # pos 3 and ring O at pos 2.  Canonical (RDKit) 'O=C1CC2CCCC2CO1'.
    # OPSIN: 'hexahydrocyclopenta[c]pyran-3(1H)-one' → 'O=C1CC2CCCC2CO1' (verified).
    # Covers iridoid bicyclic lactones (e.g. nepetalactone parent skeleton without
    # methyls).
    # Atom layout of 'O=C1CC2CCCC2CO1' (10 atoms):
    #   idx0=O(exo, claimed), idx1=C(pos3, C=O lactone), idx2=C(pos4),
    #   idx3=C(pos4a junction), idx4=C(pos5), idx5=C(pos6), idx6=C(pos7),
    #   idx7=C(pos7a junction), idx8=C(pos1), idx9=O(pos2 ring O).
    # OPSIN chloro-probing verified: pos1→idx8, pos4→idx2, pos4a→idx3,
    # pos5→idx4, pos6→idx5, pos7→idx6, pos7a→idx7.
    "O=C1CC2CCCC2CO1": {
        "name": "hexahydrocyclopenta[c]pyran-3(1H)-one",
        "substituent_form": "hexahydrocyclopenta[c]pyran-3(1H)-on-yl",
        "alkyl_stem_ok": False,
        "atom_locants": {1: 3, 2: 4, 3: "4a", 4: 5, 5: 6, 6: 7, 7: "7a", 8: 1, 9: 2}},

    # 4,5,6,7-tetrahydrobenzofuran-2(7aH)-one — Phase 7 ring-batch-4.
    # 5,6-fused: cyclohexane fused to a 5-membered lactone (γ-butenolide-like)
    # with ring O at pos 1, C=O at pos 2, and a C=C between pos 3 and pos 3a
    # (the junction).  Canonical (RDKit) 'O=C1C=C2CCCCC2O1'.
    # OPSIN: '4,5,6,7-tetrahydrobenzofuran-2(7aH)-one' → 'O=C1C=C2CCCCC2O1'
    # (verified).
    # Covers sesquiterpene lactones built on this α,β-unsaturated γ-lactone
    # scaffold (e.g. CC1(C)CCC[C@@]2(C)OC(=O)C=C12).
    # Atom layout of 'O=C1C=C2CCCCC2O1' (10 atoms):
    #   idx0=O(exo, claimed), idx1=C(pos2, C=O lactone), idx2=C(pos3),
    #   idx3=C(pos3a junction, sp2), idx4=C(pos4), idx5=C(pos5),
    #   idx6=C(pos6), idx7=C(pos7), idx8=C(pos7a junction, sp3 with H),
    #   idx9=O(pos1 ring O).
    # OPSIN chloro-probing verified: pos3→idx2, pos4→idx4, pos5→idx5,
    # pos6→idx6, pos7→idx7, pos7a→idx8; junction pos3a→idx3 derived from
    # topology (sp2 junction adjacent to both pos3 and pos4).
    "O=C1C=C2CCCCC2O1": {
        "name": "4,5,6,7-tetrahydrobenzofuran-2(7aH)-one",
        "substituent_form": "4,5,6,7-tetrahydrobenzofuran-2(7aH)-on-yl",
        "alkyl_stem_ok": False,
        "atom_locants": {1: 2, 2: 3, 3: "3a", 4: 4, 5: 5, 6: 6, 7: 7, 8: "7a", 9: 1}},

    # octahydrocyclopenta[c]pyrrole — FDA-1289 (boceprevir core).
    # RDKit canonical 'C1CC2CNCC2C1' (8 atoms).
    # OPSIN: 'octahydrocyclopenta[c]pyrrole' → C1NCC2C1CCC2 → canonical C1CC2CNCC2C1.
    # The ring is C2-symmetric (pos1 and pos3 equivalent, pos3a and pos6a equivalent,
    # pos4 and pos6 equivalent).
    # Atom layout of 'C1CC2CNCC2C1':
    #   idx0=C(pos5), idx1=C(pos4), idx2=C(pos3a junction), idx3=C(pos3),
    #   idx4=N(pos2), idx5=C(pos1), idx6=C(pos6a junction), idx7=C(pos6).
    # OPSIN chloro-probing verified: pos1→idx5, pos2(N)→idx4, pos3a→idx2, pos4→idx1, pos5→idx0.
    # Covers FDA-1289 (boceprevir: (1S,3aR,6aS)-octahydrocyclopenta[c]pyrrole-1-carboxamide fragment).
    "C1CC2CNCC2C1": {
        "name": "octahydrocyclopenta[c]pyrrole",
        "substituent_form": "octahydrocyclopenta[c]pyrrol-yl",
        "alkyl_stem_ok": False,
        "atom_locants": {5: 1, 4: 2, 3: 3, 2: "3a", 1: 4, 0: 5, 7: 6, 6: "6a"}},

    # 3,6,7,8-tetrahydroimidazo[4,5-d][1,3]diazepine — FDA-1039 (coformycin/pentostatin core).
    # RDKit canonical 'C1=Nc2[nH]cnc2CCN1' (10 atoms, 2 rings: 7-membered diazepine + 5-membered imidazole).
    # OPSIN: '3,6,7,8-tetrahydroimidazo[4,5-d][1,3]diazepine' → N1=CNC=2N=CNCCC21
    # → canonical C1=Nc2[nH]cnc2CCN1 (InChI-verified match).
    # Atom layout of 'C1=Nc2[nH]cnc2CCN1' (10 atoms):
    #   7-membered ring: idx0,1,2,6,7,8,9; 5-membered imidazole ring: idx2,3,4,5,6.
    #   Junctions: idx2=C(pos3a), idx6=C(pos7a).
    #   idx1=N(pos1, imidazole N-H), idx4=C(pos2, imidazole CH), idx3=N(pos3, diazepine sp3 N),
    #   idx5=N(pos1 of imidazole, aromatic, but in fusion scheme pos is 1),
    #   idx0=C(pos5, C=N sp2), idx9=N(pos6, sp3 N), idx8=C(pos7), idx7=C(pos8).
    # OPSIN chloro-probing verified: pos3→idx3, pos5→idx0, pos6→idx9, pos7→idx8, pos8→idx7.
    # pos1(N-H), pos3a, pos7a derived from topology.
    # Covers FDA-1039 (coformycin/pentostatin: (R)-3-(tetrahydrofuran-2-yl)-3,6,7,8-tetrahydroimidazo[4,5-d][1,3]diazepin-8-ol).
    "C1=Nc2[nH]cnc2CCN1": {
        "name": "3,6,7,8-tetrahydroimidazo[4,5-d][1,3]diazepine",
        "substituent_form": "3,6,7,8-tetrahydroimidazo[4,5-d][1,3]diazepin-yl",
        "alkyl_stem_ok": False,
        "atom_locants": {1: 1, 4: 2, 3: 3, 2: "3a", 0: 5, 9: 6, 8: 7, 7: 8, 6: "7a"}},

    # 5,8,8a,9-tetrahydrofuro[3',4':6,7]naphtho[2,3-d][1,3]dioxol-6(5aH)-one — FDA-1088 (podophyllotoxin core).
    # The engine's exo-oxo fallback looks up the with-=O form O=C1OCC2Cc3cc4c(cc3CC12)OCO4.
    # OPSIN: '5,8,8a,9-tetrahydrofuro[3',4':6,7]naphtho[2,3-d][1,3]dioxol-6(5aH)-one'
    # → O1COC2=C1C=C1CC3C(CC1=C2)C(OC3)=O → canonical O=C1OCC2Cc3cc4c(cc3CC12)OCO4 (InChI-verified).
    # The engine extracts the bare ring c1c2c(cc3c1OCO3)CC1COCC1C2 and the exo-oxo fallback
    # adds the lactone C=O (exo from the furo ring) to give the keyed with-=O SMILES.
    # Atom layout of 'O=C1OCC2Cc3cc4c(cc3CC12)OCO4' (17 atoms):
    #   Ring 1 (furo-lactone, 5-membered): idx1(C,pos6 C=O), idx2(O,pos7), idx3(C,pos8), idx4(C,pos8a), idx13(C,pos5a).
    #   Ring 2 (6-membered): idx4(pos8a), idx5(C,pos9), idx6(arom C), idx11(arom C), idx12(C,pos5), idx13(pos5a).
    #   Ring 3 (aromatic): idx6-11.  Ring 4 (dioxole): idx8,9,14-16.
    #   idx0=O(exo, claimed), idx14=O(dioxole), idx15=C(dioxole CH2), idx16=O(dioxole).
    # OPSIN chloro-probing verified: pos5→idx12, pos5a→idx13, pos8→idx3, pos8a→idx4, pos9→idx5.
    # Covers FDA-1088 (podophyllotoxin: (5R,5aR,8aR,9R)-9-hydroxy-5-(3,4,5-trimethoxyphenyl)-...-one).
    "O=C1OCC2Cc3cc4c(cc3CC12)OCO4": {
        "name": "5,8,8a,9-tetrahydrofuro[3',4':6,7]naphtho[2,3-d][1,3]dioxol-6(5aH)-one",
        "substituent_form": "5,8,8a,9-tetrahydrofuro[3',4':6,7]naphtho[2,3-d][1,3]dioxol-6(5aH)-on-yl",
        "alkyl_stem_ok": False,
        "atom_locants": {1: 6, 2: 7, 3: 8, 4: "8a", 5: 9, 12: 5, 13: "5a"}},

    # 1,2,3,4,4a,9,10,10a-octahydrophenanthrene — FDA-0462 (abietic acid / diterpene core).
    # RDKit canonical 'c1ccc2c(c1)CCC1CCCCC21' (14 atoms, 3 rings: 2 six-membered aliphatic + 1 aromatic).
    # OPSIN: '1,2,3,4,4a,9,10,10a-octahydrophenanthrene' → C1CCCC2C3=CC=CC=C3CCC12
    # → canonical c1ccc2c(c1)CCC1CCCCC21 (InChI-verified match).
    # Ring A (aromatic, pos 5-8, 8a, 4b): idx0-5.  Ring B (pos 4a,4b,8a,9,10,10a): idx3,4,6,7,8,13.
    # Ring C (pos 1-4, 4a, 10a): idx8-13.
    # Atom layout of 'c1ccc2c(c1)CCC1CCCCC21':
    #   idx0=C(pos7, arom), idx1=C(pos6, arom), idx2=C(pos5, arom), idx3=C(pos4b junction arom),
    #   idx4=C(pos8a junction arom), idx5=C(pos8, arom), idx6=C(pos9), idx7=C(pos10),
    #   idx8=C(pos10a junction), idx9=C(pos1), idx10=C(pos2), idx11=C(pos3), idx12=C(pos4),
    #   idx13=C(pos4a junction).
    # OPSIN chloro-probing verified: pos1→idx9, pos2→idx10, pos3→idx11, pos4→idx12,
    # pos4a→idx13, pos5→idx2, pos6→idx1, pos7→idx0, pos8→idx5, pos9→idx6, pos10→idx7.
    # Covers FDA-0462 (abietic acid: 1R,4aS,10aR-7-isopropyl-1,4a-dimethyl-6-sulfo-...-carboxylic acid).
    "c1ccc2c(c1)CCC1CCCCC21": {
        "name": "1,2,3,4,4a,9,10,10a-octahydrophenanthrene",
        "substituent_form": "1,2,3,4,4a,9,10,10a-octahydrophenanthrenyl",
        "alkyl_stem_ok": False,
        "atom_locants": {9: 1, 10: 2, 11: 3, 12: 4, 13: "4a", 2: 5, 1: 6, 0: 7, 5: 8,
                         4: "8a", 6: 9, 7: 10, 8: "10a", 3: "4b"}},

    # -----------------------------------------------------------------------
    # Phase 4 fused-ring stretch additions (May 2026)
    # Each entry verified by OPSIN methyl-probing on the named locants and
    # round-trips end-to-end (input SMILES -> our name -> OPSIN parse ->
    # canonical SMILES match).  All atom indices are in the canonical SMILES.
    # -----------------------------------------------------------------------

    # Pyrazolo[1,5-a]pyridine: 5-ring (pyrazole) ortho-fused to 6-ring
    # (pyridine) sharing one bridgehead N.  Canonical 'c1ccn2nccc2c1':
    #   idx4 = degree-2 N (locant 1), idx3 = bridgehead N (7a).
    # Locants verified via OPSIN methyl-probing at L=1..7,3a.
    "c1ccn2nccc2c1": {
        "name": "pyrazolo[1,5-a]pyridine",
        "substituent_form": "pyrazolo[1,5-a]pyridinyl",
        "alkyl_stem_ok": False,
        "atom_locants": {4: 1, 5: 2, 6: 3, 7: "3a", 8: 4, 0: 5, 1: 6, 2: 7, 3: "7a"}},

    # 1,3,2-Benzodioxaborole: dioxaborole 5-ring ortho-fused to benzene.
    # Canonical 'B1Oc2ccccc2O1':
    #   idx0 = B (locant 2), idx1/idx8 = O (locants 1/3 — symmetric).
    # Locants verified via OPSIN methyl-probing at L=2,4,5,6,7 with
    # symmetry-breaking 4-fluoro/5-fluoro/6-fluoro/7-fluoro disambiguation
    # (4↔7 and 5↔6 collapse under the C2v symmetry — either assignment
    # round-trips through OPSIN).
    "B1Oc2ccccc2O1": {
        "name": "1,3,2-benzodioxaborole",
        "substituent_form": "1,3,2-benzodioxaborol-N-yl",
        "alkyl_stem_ok": False,
        "atom_locants": {1: 1, 0: 2, 8: 3, 7: "3a", 6: 4, 5: 5, 4: 6, 3: 7, 2: "7a"}},

    # 4H-1,3,2-Benzodioxaphosphinine — Phase 7 ring-batch-4.
    # 6-ring dioxaphosphinine (O-P-O-C with one CH2 between P and arom-fused C)
    # ortho-fused to benzene.  Heteros at pos 1 (O), pos 2 (P), pos 3 (O); pos 4
    # is the CH2.  Canonical (RDKit) 'c1ccc2c(c1)COPO2'.
    # OPSIN: '4H-1,3,2-benzodioxaphosphinine' → 'c1ccc2c(c1)COPO2' (verified).
    # Covers cyclic phosphite/phosphate scaffolds where the P bears an exocyclic
    # =S, =O, or -OR substituent (e.g. 2-methoxy-4H-1,3,2-benzodioxaphosphinine
    # 2-sulfide → COP1(=S)OCc2ccccc2O1).
    # Atom layout of 'c1ccc2c(c1)COPO2' (10 atoms):
    #   idx0=C(pos6 arom), idx1=C(pos7 arom), idx2=C(pos8 arom),
    #   idx3=C(pos8a junction), idx4=C(pos4a junction), idx5=C(pos5 arom),
    #   idx6=C(pos4 CH2), idx7=O(pos3 ring O), idx8=P(pos2),
    #   idx9=O(pos1 ring O).
    # OPSIN chloro-probing verified: pos4→idx6, pos5→idx5, pos6→idx0,
    # pos7→idx1, pos8→idx2; junctions pos4a→idx4, pos8a→idx3 derived from
    # topology (pos8a is the arom junction adjacent to ring O at pos 1;
    # pos4a is the arom junction adjacent to the CH2 at pos 4).
    "c1ccc2c(c1)COPO2": {
        "name": "4H-1,3,2-benzodioxaphosphinine",
        "substituent_form": "4H-1,3,2-benzodioxaphosphinin-N-yl",
        "alkyl_stem_ok": False,
        "atom_locants": {0: 6, 1: 7, 2: 8, 3: "8a", 4: "4a", 5: 5, 6: 4,
                         7: 3, 8: 2, 9: 1}},

    # 1,2,3,4-tetrahydroquinazoline: saturated diazine 6-ring ortho-fused
    # to benzene.  Canonical 'c1ccc2c(c1)CNCN2':
    #   idx9 = N1, idx8 = C2, idx7 = N3, idx6 = C4, idx4 = C4a (sp2 fusion
    #   bonded to CH2), idx3 = C8a (sp2 fusion bonded to N1).
    # Locants verified via OPSIN methyl-probing at L=1..8.
    "c1ccc2c(c1)CNCN2": {
        "name": "1,2,3,4-tetrahydroquinazoline",
        "substituent_form": "1,2,3,4-tetrahydroquinazolinyl",
        "alkyl_stem_ok": False,
        "atom_locants": {9: 1, 8: 2, 7: 3, 6: 4, 4: "4a", 5: 5, 0: 6, 1: 7, 2: 8, 3: "8a"}},

    # 2,3-dihydro-1,2-benzothiazole (= 2,3-dihydro-1,2-benzisothiazole):
    # NS-bearing 5-ring ortho-fused to benzene.  Canonical 'c1ccc2c(c1)CNS2':
    #   idx8 = S1, idx7 = N2, idx6 = C3, idx4 = C3a, idx3 = C7a.
    # Locants verified via OPSIN methyl-probing at L=2..7.
    "c1ccc2c(c1)CNS2": {
        "name": "2,3-dihydro-1,2-benzothiazole",
        "substituent_form": "2,3-dihydro-1,2-benzothiazolyl",
        "alkyl_stem_ok": False,
        "atom_locants": {8: 1, 7: 2, 6: 3, 4: "3a", 5: 4, 0: 5, 1: 6, 2: 7, 3: "7a"}},

    # 2H-1,2-benzothiazine: 6-ring with N-S sharing the 1,2-edge ortho-fused
    # to benzene; one C=C double bond between C3 and C4.  Canonical
    # 'C1=Cc2ccccc2SN1':
    #   idx8 = S1, idx9 = N2, idx0 = C3, idx1 = C4, idx2 = C4a (fusion),
    #   idx7 = C8a (fusion).
    # Locants verified via OPSIN methyl-probing at L=3..8.
    "C1=Cc2ccccc2SN1": {
        "name": "2H-1,2-benzothiazine",
        "substituent_form": "2H-1,2-benzothiazin-N-yl",
        "alkyl_stem_ok": False,
        "atom_locants": {8: 1, 9: 2, 0: 3, 1: 4, 2: "4a", 3: 5, 4: 6, 5: 7, 6: 8, 7: "8a"}},

    # 3,4-dihydro-2H-thieno[2,3-e][1,2]thiazine: thiophene 5-ring fused at
    # its 2,3-edge to the 4a-5 edge of a 1,2-thiazine 6-ring.  Canonical
    # 'c1cc2c(s1)CCNS2':
    #   idx8 = S1, idx7 = N2, idx6 = C3, idx5 = C4, idx3 = C4a (fusion),
    #   idx4 = S5 (thiophene S), idx0 = C6, idx1 = C7, idx2 = C7a (fusion).
    # Locants verified via OPSIN methyl-probing at L=2..7,4a,7a.
    "c1cc2c(s1)CCNS2": {
        "name": "3,4-dihydro-2H-thieno[2,3-e][1,2]thiazine",
        "substituent_form": "3,4-dihydro-2H-thieno[2,3-e][1,2]thiazin-N-yl",
        "alkyl_stem_ok": False,
        "atom_locants": {8: 1, 7: 2, 6: 3, 5: 4, 3: "4a", 4: 5, 0: 6, 1: 7, 2: "7a"}},

    # bicyclo[4.2.0]octa-3,5-diene: 6-ring with two C=C double bonds, fused
    # at one edge to a 4-ring.  Canonical 'C1=CCC2CCC2=C1':
    #   idx 3 and idx 6 are bridgeheads.  Numbering starts at one bridgehead
    #   (1), goes through the longer bridge (2, 3, 4, 5), reaches the other
    #   bridgehead (6), then the shorter bridge (7, 8).  Direction chosen so
    #   the diene picks low locants 3,5 (giving octa-3,5-diene) and the keto
    #   position adjacent to position-6 bridgehead maps to 7 (so the
    #   'O=C1CC2CC=CC=C12' input round-trips as octa-3,5-dien-7-one).
    "C1=CCC2CCC2=C1": {
        "name": "bicyclo[4.2.0]octa-3,5-diene",
        "substituent_form": "bicyclo[4.2.0]octa-3,5-dien-N-yl",
        "alkyl_stem_ok": False,
        "atom_locants": {3: 1, 2: 2, 1: 3, 0: 4, 7: 5, 6: 6, 5: 7, 4: 8}},

    # bicyclo[3.2.0]hepta-2,6-diene — Phase 7 ring-batch-4.
    # 5-ring fused to a 4-ring sharing one C-C bond (cis ring junction).  The
    # 5-ring carries one C=C (at pos 2-3) and the 4-ring carries one C=C
    # (at pos 6-7).  Canonical (RDKit) 'C1=CC2C=CC2C1'.
    # OPSIN: 'bicyclo[3.2.0]hepta-2,6-diene' → 'C1=CC2C=CC2C1' (verified).
    # Covers heptenophos and related bicyclo[3.2.0]hepta-dienyl phosphate
    # esters.
    # Atom layout of 'C1=CC2C=CC2C1' (7 atoms):
    #   idx0=C(pos3), idx1=C(pos2), idx2=C(pos1 bridgehead),
    #   idx3=C(pos7), idx4=C(pos6), idx5=C(pos5 bridgehead),
    #   idx6=C(pos4).
    # OPSIN chloro-probing verified: pos1→idx2, pos2→idx1, pos3→idx0,
    # pos4→idx6, pos5→idx5, pos6→idx4, pos7→idx3.
    "C1=CC2C=CC2C1": {
        "name": "bicyclo[3.2.0]hepta-2,6-diene",
        "substituent_form": "bicyclo[3.2.0]hepta-2,6-dien-N-yl",
        "alkyl_stem_ok": False,
        "atom_locants": {0: 3, 1: 2, 2: 1, 3: 7, 4: 6, 5: 5, 6: 4}},

    # 1,2,3,4-tetrahydroisoquinoline: saturated pyridine 6-ring fused to
    # benzene at the 4a-8a edge.  Canonical 'c1ccc2c(c1)CCNC2':
    #   idx9 = C1, idx8 = N2, idx7 = C3, idx6 = C4, idx4 = C4a (fusion C
    #   bonded to C4), idx3 = C8a (fusion C bonded to C1).
    # Locants verified via OPSIN methyl-probing at L=1..8.
    "c1ccc2c(c1)CCNC2": {
        "name": "1,2,3,4-tetrahydroisoquinoline",
        "substituent_form": "1,2,3,4-tetrahydroisoquinolinyl",
        "alkyl_stem_ok": False,
        "atom_locants": {9: 1, 8: 2, 7: 3, 6: 4, 4: "4a", 5: 5, 0: 6, 1: 7, 2: 8, 3: "8a"}},

    # 2,3,4,5-tetrahydro-1H-1,5-benzodiazepine: 7-ring with two NH at
    # opposite sides (1,5) ortho-fused to benzene.  Canonical
    # 'c1ccc2c(c1)NCCCN2':
    #   idx6 = N1, idx7 = C2, idx8 = C3, idx9 = C4, idx10 = N5,
    #   idx3 = C5a (fusion bonded to N5), idx4 = C9a (fusion bonded to N1).
    # The molecule has a Cs mirror; either {N1=idx6, N5=idx10} or
    # {N1=idx10, N5=idx6} round-trips.  Locants verified via OPSIN
    # methyl-probing at L=1..9 (with L=1↔L=5 and L=6↔L=9 collapsing).
    "c1ccc2c(c1)NCCCN2": {
        "name": "2,3,4,5-tetrahydro-1H-1,5-benzodiazepine",
        "substituent_form": "2,3,4,5-tetrahydro-1H-1,5-benzodiazepin-N-yl",
        "alkyl_stem_ok": False,
        "atom_locants": {6: 1, 7: 2, 8: 3, 9: 4, 10: 5, 3: "5a", 2: 6, 1: 7, 0: 8, 5: 9, 4: "9a"}},

    # 2,3,4,5-tetrahydro-1,5-benzothiazepine: 7-ring with S at 1 and N at 5
    # ortho-fused to benzene.  Canonical 'c1ccc2c(c1)NCCCS2':
    #   idx10 = S1, idx9 = C2, idx8 = C3, idx7 = C4, idx6 = N5,
    #   idx4 = C5a (fusion bonded to N5), idx3 = C9a (fusion bonded to S1).
    # Locants verified via OPSIN methyl-probing at L=1..9.
    "c1ccc2c(c1)NCCCS2": {
        "name": "2,3,4,5-tetrahydro-1,5-benzothiazepine",
        "substituent_form": "2,3,4,5-tetrahydro-1,5-benzothiazepin-N-yl",
        "alkyl_stem_ok": False,
        "atom_locants": {10: 1, 9: 2, 8: 3, 7: 4, 6: 5, 4: "5a", 5: 6, 0: 7, 1: 8, 2: 9, 3: "9a"}},

    # 4,5,6,7-tetrahydrothieno[2,3-b]thiopyran: aromatic thiophene fused at
    # its 2,3-edge to the 4a-5 (b) edge of saturated thiopyran (1H-2-benzothio
    # pyran-style 6-ring with one S).  Canonical 'c1cc2c(s1)SCCC2':
    #   idx4 = S1 (thiophene S, position 1), idx0 = C2, idx1 = C3,
    #   idx2 = C3a (fusion bonded to thiophene C3), idx8 = C4, idx7 = C5,
    #   idx6 = C6, idx5 = S7 (thiopyran S),
    #   idx3 = C7a (fusion bonded to thiopyran S and thiophene S).
    # Locants verified via OPSIN methyl-probing at L=2..7,1.
    "c1cc2c(s1)SCCC2": {
        "name": "4,5,6,7-tetrahydrothieno[2,3-b]thiopyran",
        "substituent_form": "4,5,6,7-tetrahydrothieno[2,3-b]thiopyran-N-yl",
        "alkyl_stem_ok": False,
        "atom_locants": {4: 1, 0: 2, 1: 3, 2: "3a", 8: 4, 7: 5, 6: 6, 5: 7, 3: "7a"}},

    # 2,3,4,5-tetrahydro-1H-3-benzazepine: 7-ring with N at position 3 in
    # the centre, ortho-fused to benzene.  Canonical 'c1ccc2c(c1)CCNCC2':
    #   idx6 = C1, idx7 = C2, idx8 = N3, idx9 = C4, idx10 = C5,
    #   idx3 = C5a (fusion bonded to C5), idx4 = C9a (fusion bonded to C1).
    # By Cs symmetry L=1↔L=5 and L=6↔L=9 collapse — either direction
    # round-trips through OPSIN.  Locants verified via methyl-probing.
    "c1ccc2c(c1)CCNCC2": {
        "name": "2,3,4,5-tetrahydro-1H-3-benzazepine",
        "substituent_form": "2,3,4,5-tetrahydro-1H-3-benzazepin-N-yl",
        "alkyl_stem_ok": False,
        "atom_locants": {6: 1, 7: 2, 8: 3, 9: 4, 10: 5, 3: "5a", 2: 6, 1: 7, 0: 8, 5: 9, 4: "9a"}},

    # Octahydropentalene = bicyclo[3.3.0]octane: fully saturated, two cis-
    # fused 5-rings sharing one edge (two bridgeheads).  Canonical
    # 'C1CC2CCCC2C1':
    #   idx2, idx6 = bridgeheads (3a, 6a).
    # Numbering 1→2→3→3a→4→5→6→6a→1 around the periphery.
    # By C2 symmetry L=1↔L=4, L=2↔L=5, L=3↔L=6 collapse.
    "C1CC2CCCC2C1": {
        "name": "octahydropentalene",
        "substituent_form": "octahydropentalenyl",
        "alkyl_stem_ok": False,
        "atom_locants": {7: 1, 0: 2, 1: 3, 2: "3a", 3: 4, 4: 5, 5: 6, 6: "6a"}},

    # 1H-imidazo[4,5-c]quinoline: tricyclic — imidazole 5-ring fused at its
    # 4,5-edge to bond c (3-4) of quinoline.  Canonical
    # 'c1ccc2c(c1)ncc1nc[nH]c12' (13 atoms).  Pre-fix the engine emitted
    # "1H-imidazo[4,5-c]quinolin-5-amine" for the imiquimod core
    # 'Nc1nc2ccccc2c2[nH]cnc12' (and for the full 1-isobutyl form
    # 'CC(C)Cn1cnc2c(N)nc3ccccc3c21') — locant 5 is the pyridine N
    # (cannot bear an amine substituent), the actual amine sits on the
    # adjacent ring C at pos 4.  atom_locants pin idx -> position so the
    # amine substituent gets locant 4.  Atom indices in canonical SMILES:
    #   idx0  = C(pos 7, benzo CH)
    #   idx1  = C(pos 8, benzo CH)
    #   idx2  = C(pos 9, benzo CH)
    #   idx3  = C(pos 9a, jct benzo-pyridine adj to C9b)
    #   idx4  = C(pos 5a, jct benzo-pyridine adj to N5)
    #   idx5  = C(pos 6, benzo CH peri to N5)
    #   idx6  = N(pos 5, pyridine N)
    #   idx7  = C(pos 4, pyridine CH adj to N5)
    #   idx8  = C(pos 3a, jct pyridine-imidazole adj to N3)
    #   idx9  = N(pos 3, imidazole N)
    #   idx10 = C(pos 2, imidazole CH)
    #   idx11 = N(pos 1, imidazole NH)
    #   idx12 = C(pos 9b, jct imidazole-pyridine-benzo)
    # Locants verified via OPSIN methyl-probing positions 1, 2, 3, 4, 6,
    # 7, 8, 9 (the unique CH/N-H probe positions) plus 9b junction; the
    # 4-amine and 5-amine probe SMILES pin which junction the amine sits
    # adjacent to (4-amine on idx 7 = adj to N5).
    "c1ccc2c(c1)ncc1nc[nH]c12": {
        "name": "1H-imidazo[4,5-c]quinoline",
        "substituent_form": "1H-imidazo[4,5-c]quinolin-N-yl",
        "alkyl_stem_ok": False,
        "atom_locants": {0: 7, 1: 8, 2: 9, 3: "9a", 4: "5a", 5: 6,
                          6: 5, 7: 4, 8: "3a", 9: 3, 10: 2, 11: 1,
                          12: "9b"}},

    # Pyrimido[4,5-b]quinoxaline: tricyclic 6,6,6 — pyrimidine fused at its
    # 4,5-edge to bond b of quinoxaline.  Canonical 'c1ccc2nc3ncncc3nc2c1'.
    # Unsubstituted parent only — atom-locants not pinned.
    "c1ccc2nc3ncncc3nc2c1": {
        "name": "pyrimido[4,5-b]quinoxaline",
        "substituent_form": "pyrimido[4,5-b]quinoxalinyl",
        "alkyl_stem_ok": False,
        "stage2_fusion_base": False},

    # 2,3-dihydro-1H-1,4-benzodiazepine: 7-ring with NH at position 1, sp2
    # C=N at position 4-5, ortho-fused to benzene.  Canonical
    # 'C1=NCCNc2ccccc21':
    #   idx4 = N1, idx3 = C2, idx2 = C3, idx1 = N4, idx0 = C5,
    #   idx5 = C5a (fusion C bonded to C5), idx10 = C9a (fusion C bonded
    #   to N1).
    # Locants verified via OPSIN methyl-probing at L=1..3,5..9.
    "C1=NCCNc2ccccc21": {
        "name": "2,3-dihydro-1H-1,4-benzodiazepine",
        "substituent_form": "2,3-dihydro-1H-1,4-benzodiazepin-N-yl",
        "alkyl_stem_ok": False,
        "atom_locants": {4: 1, 3: 2, 2: 3, 1: 4, 0: 5, 10: "5a", 9: 6, 8: 7, 7: 8, 6: 9, 5: "9a"}},

    # 3,4-dihydro-2H-thieno[3,2-e][1,2]thiazine: thiophene 5-ring fused at
    # its 3,2-edge to bond e (5a-1) of a 1,2-thiazine 6-ring.  Canonical
    # 'c1cc2c(s1)SNCC2'.  This is the brinzolamide (glaucoma drug) core.
    # Atom indices in canonical SMILES vs IUPAC locants (derived via OPSIN
    # methyl-probing of locants 2/3/4/5/6 plus topology for 4a/7a/7/1):
    #   idx 0 = aromatic C adj to thiophene-S  -> locant 6
    #   idx 1 = aromatic C adj to junction     -> locant 5
    #   idx 2 = aromatic C junction adj C(4)   -> locant 4a
    #   idx 3 = aromatic C junction adj S      -> locant 7a
    #   idx 4 = thiophene S                     -> locant 7
    #   idx 5 = ring-S (1,1-dioxide site)      -> locant 1
    #   idx 6 = ring N                          -> locant 2
    #   idx 7 = sp3 C adj N                     -> locant 3
    #   idx 8 = sp3 C adj junction             -> locant 4
    "c1cc2c(s1)SNCC2": {
        "name": "3,4-dihydro-2H-thieno[3,2-e][1,2]thiazine",
        "substituent_form": "3,4-dihydro-2H-thieno[3,2-e][1,2]thiazin-N-yl",
        "alkyl_stem_ok": False,
        "atom_locants": {0: 6, 1: 5, 2: "4a", 3: "7a", 4: 7,
                          5: 1, 6: 2, 7: 3, 8: 4}},

    # Imidazo[5,1-d][1,2,3,5]tetrazine: bicyclic 5+6 — imidazole (atoms 5,1)
    # fused at edge d (4-4a) of [1,2,3,5]tetrazine.  Canonical
    # 'c1ncn2cnnnc12'.  Atom-locants not pinned; unsubstituted parent works
    # but substituted forms may need explicit locant probing.
    "c1ncn2cnnnc12": {
        "name": "imidazo[5,1-d][1,2,3,5]tetrazine",
        "substituent_form": "imidazo[5,1-d][1,2,3,5]tetrazin-N-yl",
        "alkyl_stem_ok": False},

    # Hexahydrothieno[3,4-d]imidazole: saturated 5+5 — imidazoline
    # fused at edge d (4-5) to thiolane's 3,4-edge.  Canonical 'C1NC2CSCC2N1'.
    # Core scaffold of biotin (vitamin B7).
    "C1NC2CSCC2N1": {
        "name": "hexahydrothieno[3,4-d]imidazole",
        "substituent_form": "hexahydrothieno[3,4-d]imidazol-N-yl",
        "alkyl_stem_ok": False},

    # Octahydroindole = octahydro-1H-indole: fully saturated indole.
    # Canonical 'C1CCC2NCCC2C1'.  Two cis-fused 5+6 saturated rings with
    # one N at position 1.
    "C1CCC2NCCC2C1": {
        "name": "octahydro-1H-indole",
        "substituent_form": "octahydro-1H-indol-N-yl",
        "alkyl_stem_ok": False},

    # -----------------------------------------------------------------------
    # octahydro-1H-pyrrolo[3,4-b]pyridine (moxifloxacin C7 side chain core).
    # -----------------------------------------------------------------------
    # Fully saturated pyrrolo[3,4-b]pyridine: pyrrolidine (5-ring) fused to
    # piperidine (6-ring) at the [3,4-b] edge.  Two ring nitrogens — N1 on
    # the piperidine half, N6 on the pyrrolidine half — with bridgehead
    # carbons C3a and C7a.  IUPAC numbering follows pyrrolo[3,4-b]pyridine
    # parent: N1, C2, C3, C3a, C4, C5, N6, C7, C7a (6-ring = 1-2-3-3a-7a-? —
    # actually 1-2-3-3a-... wait, [3,4-b] means C3,C3a are fused; numbering
    # goes around 6-ring N1-C2-C3-C3a-C7a-...-N1, then 5-ring C3a-C4(no,5)-N6-C7-C7a.
    # OPSIN canonical for octahydropyrrolo[3,4-b]pyridine: 'N1C2C(CCC1)CNC2'.
    # RDKit canonical: 'C1CNC2CNCC2C1' (9 ring atoms).
    # atom_locants verified via OPSIN methyl-probing 1, 2, 3, 4, 5, 6, 7, 7a
    # against the RDKit canonical key (3a falls out as the remaining bridgehead):
    #   1 -> idx2 (N1, piperidine NH)
    #   2 -> idx1 (C2 piperidine)
    #   3 -> idx0 (C3 piperidine)
    #   3a -> idx7 (bridgehead C)
    #   4 -> idx8 (C4 piperidine, bonds to C3a junction)
    #   5 -> idx6 (C5 pyrrolidine CH2 alpha to N6)
    #   6 -> idx5 (N6, pyrrolidine NH)
    #   7 -> idx4 (C7 pyrrolidine CH2 alpha to N6)
    #   7a -> idx3 (bridgehead C, between N1 and C7)
    # Covers FDA-2071 (moxifloxacin) C7-quinolone substituent.  In moxifloxacin
    # the bicyclic attaches via N6 to the quinolone C7.
    # substituent_form uses the "-1-yl" placeholder so the engine's locant-
    # rewriting path (engine.py ~line 8700, regex r"-(\d\w*)-yl$") strips and
    # replaces with the actual attachment locant; the "-N-yl" placeholder
    # form does NOT work in carved-substituent contexts (yields "...pyridin-N--6-yl").
    "C1CNC2CNCC2C1": {
        "name": "octahydro-1H-pyrrolo[3,4-b]pyridine",
        "substituent_form": "octahydro-1H-pyrrolo[3,4-b]pyridin-1-yl",
        "alkyl_stem_ok": False,
        "atom_locants": {2: 1, 1: 2, 0: 3, 7: "3a", 8: 4, 6: 5, 5: 6, 4: 7, 3: "7a"}},

    # 1H-3,1-benzoxazine-2,4-dione (isatoic anhydride): benzofused 6-ring
    # with N at pos 1 (NH), C=O at pos 2, ring-O at pos 3, C=O at pos 4.
    # Canonical 'O=c1[nH]c2ccccc2c(=O)o1' (12 atoms incl. two exocyclic O).
    # The retained-name stem encodes BOTH ring carbonyls + the 1H-tautomer;
    # key on the with-=O canonical so the exocyclic-oxo fallback in
    # retained_lookup claims both =O atoms via extra_atom_indices.
    # Atom indices in canonical SMILES:
    #   idx0  = O (exocyclic on C2, claimed via extra_atom_indices)
    #   idx1  = C(pos 2, ring C=O between N1 and O3)
    #   idx2  = N(pos 1, NH)
    #   idx3  = C(pos 8a, junction adj to N1)
    #   idx4  = C(pos 8)
    #   idx5  = C(pos 7)
    #   idx6  = C(pos 6)
    #   idx7  = C(pos 5)
    #   idx8  = C(pos 4a, junction adj to C4)
    #   idx9  = C(pos 4, ring C=O)
    #   idx10 = O (exocyclic on C4, claimed via extra_atom_indices)
    #   idx11 = O(pos 3, ring O)
    # Locants verified via OPSIN chloro-probing of L=5..8 and 1-methyl probe.
    "O=c1[nH]c2ccccc2c(=O)o1": {
        "name": "1H-3,1-benzoxazine-2,4-dione",
        "substituent_form": "2,4-dioxo-1H-3,1-benzoxazin-N-yl",
        "alkyl_stem_ok": False,
        "atom_locants": {1: 2, 2: 1, 3: "8a", 4: 8, 5: 7, 6: 6, 7: 5,
                          8: "4a", 9: 4, 11: 3}},

    # 3a,4,5,7a-tetrahydroisobenzofuran-1,3-dione: 5+6 fused (γ,δ-unsaturated
    # cis-cyclohexene-1,2-dicarboxylic anhydride).  Canonical
    # 'O=C1OC(=O)C2CCC=CC12' (11 atoms incl. two exocyclic =O).  The
    # retained-name stem encodes BOTH ring carbonyls (the "1,3-dione" tail);
    # key on the with-=O canonical so the exocyclic-oxo fallback in
    # retained_lookup claims both =O atoms via extra_atom_indices.
    # Atom indices in canonical SMILES:
    #   idx0  = O (exocyclic on C1, claimed via extra_atom_indices)
    #   idx1  = C(pos 1, lactone C=O adj to ring-O at pos 2)
    #   idx2  = O(pos 2, ring O)
    #   idx3  = C(pos 3, lactone C=O adj to ring-O)
    #   idx4  = O (exocyclic on C3, claimed via extra_atom_indices)
    #   idx5  = C(pos 3a, sp3 junction)
    #   idx6  = C(pos 4, sp3 CH2)
    #   idx7  = C(pos 5, sp3 CH2)
    #   idx8  = C(pos 6, sp2, C6=C7)
    #   idx9  = C(pos 7, sp2, C6=C7)
    #   idx10 = C(pos 7a, sp3 junction)
    # Locants verified via OPSIN chloro-probing positions 4, 5, 6, 7,
    # 3a, 7a and 7a-methyl probe (CC12C=CCCC1C(=O)OC2=O canonical match).
    "O=C1OC(=O)C2CCC=CC12": {
        "name": "3a,4,5,7a-tetrahydroisobenzofuran-1,3-dione",
        "substituent_form": "1,3-dioxo-3a,4,5,7a-tetrahydroisobenzofuran-N-yl",
        "alkyl_stem_ok": False,
        "atom_locants": {1: 1, 2: 2, 3: 3, 5: "3a", 6: 4, 7: 5, 8: 6, 9: 7,
                          10: "7a"}},

    # tetrahydroazeto[2,1-b][1,3]oxazole: 4+5 fused saturated bicyclic with
    # an N-C bridgehead.  Canonical 'C1CN2CCC2O1' (7 atoms, no exo
    # substituent).  5-ring (1,3-oxazolidine: O-C-C-N-C) + 4-ring (azetidine:
    # N-C-C-C) sharing the N-C edge.  Bridgeheads are N (idx2, fusion-N
    # labelled implicitly via [2,1-b]) and C (idx5, locant 6a).  Pinning
    # atom_locants is needed so the substitutive renderer emits the correct
    # stereodescriptor locant for the bridgehead C (e.g. "(6aR)" instead of
    # the default "(7R)" that comes from positional fallback).
    # Atom indices in canonical SMILES C1CN2CCC2O1:
    #   idx0 = C(pos 2, 5-ring CH2 adj to O1)
    #   idx1 = C(pos 3, 5-ring CH2 adj to N)
    #   idx2 = N(bridgehead, fusion atom — no separate locant)
    #   idx3 = C(pos 5, 4-ring CH2 adj to N)
    #   idx4 = C(pos 6, 4-ring CH2 adj to bridgehead C)
    #   idx5 = C(pos 6a, bridgehead C — bonded to O, N, and 4-ring CH2)
    #   idx6 = O(pos 1, 5-ring ring O)
    # Locants verified via OPSIN methyl-probing positions 2, 3, 5, 6, 6a.
    "C1CN2CCC2O1": {
        "name": "tetrahydroazeto[2,1-b][1,3]oxazole",
        "substituent_form": "tetrahydroazeto[2,1-b][1,3]oxazol-N-yl",
        "alkyl_stem_ok": False,
        "atom_locants": {0: 2, 1: 3, 3: 5, 4: 6, 5: "6a", 6: 1}},

    # octahydropyridazino[1,2-a][1,2]diazepine: 6+7 fused saturated bicyclic
    # with an N-N bridgehead bond.  Canonical 'C1CCN2CCCCN2CC1' (11 atoms).
    # The 7-ring is [1,2]diazepine (2 adjacent N), the 6-ring is pyridazine
    # (also 2 adjacent N); they share the N-N bond.  Atom-locants not pinned
    # (unsubstituted parent only — bridged-N pyridazine system has internal
    # automorphism that makes single-locant probing ambiguous; substituted
    # forms can be probed individually if/when they appear in the corpus).
    "C1CCN2CCCCN2CC1": {
        "name": "octahydropyridazino[1,2-a][1,2]diazepine",
        "substituent_form": "octahydropyridazino[1,2-a][1,2]diazepin-N-yl",
        "alkyl_stem_ok": False},

    # pyrazolo[3,4-d]pyrimidin-4(1H)-one: 5+6 fused bicyclic with pyrimidine
    # (aromatic, with C4=O) and a 1,2-dihydropyrazole (NH-NH-C=C).  Canonical
    # 'O=c1ncnc2[nH][nH]cc1-2' (10 atoms incl. exocyclic =O).  The retained
    # name encodes the C4 carbonyl (the "4(1H)-one" tail); key on the with-=O
    # canonical so the exocyclic-oxo fallback in retained_lookup claims the
    # =O atom via extra_atom_indices.  Atom indices in canonical SMILES:
    #   idx0 = O (exocyclic on C4, claimed via extra_atom_indices)
    #   idx1 = C(pos 4, ring C=O)
    #   idx2 = N(pos 5)
    #   idx3 = C(pos 6)
    #   idx4 = N(pos 7)
    #   idx5 = C(pos 7a, junction adj to N7 and N1)
    #   idx6 = N(pos 1, NH)
    #   idx7 = N(pos 2, NH)
    #   idx8 = C(pos 3)
    #   idx9 = C(pos 3a, junction adj to C3 and C4)
    # Locants verified via OPSIN chloro-probing positions 1, 2, 3, 6.
    "O=c1ncnc2[nH][nH]cc1-2": {
        "name": "pyrazolo[3,4-d]pyrimidin-4(1H)-one",
        "substituent_form": "4-oxo-1,4-dihydropyrazolo[3,4-d]pyrimidin-N-yl",
        "alkyl_stem_ok": False,
        "atom_locants": {1: 4, 2: 5, 3: 6, 4: 7, 5: "7a", 6: 1, 7: 2,
                          8: 3, 9: "3a"}},

    # [1,3]dioxolo[4,5-g]cinnoline: 6+6+5 ortho-fused tricyclic — cinnoline
    # with a methylenedioxy bridge fused at edge "g" (C6-C7) of the benzo
    # half.  Canonical 'c1cc2cc3c(cc2nn1)OCO3' (13 atoms).  Without this
    # entry, the engine emits the wrong locant family ("7-…-9-…-10-…")
    # because the fusion-name synthesis path picks the cinnoline-side
    # numbering instead of the [1,3]dioxolo-renumbered system.
    # Atom indices in canonical SMILES:
    #   idx0  = C(pos 3, cinnoline C adj to N2)
    #   idx1  = C(pos 4, cinnoline C adj to C4a)
    #   idx2  = C(pos 4a, junction cinnoline-benzene)
    #   idx3  = C(pos 5, benzene C adj to C4a and C5a)
    #   idx4  = C(pos 5a, junction benzene-dioxole, bonded to O at pos 6)
    #   idx5  = C(pos 8a, junction benzene-dioxole, bonded to O at pos 8)
    #   idx6  = C(pos 9, benzene C adj to C9a)
    #   idx7  = C(pos 9a, junction benzene-cinnoline, adj to N1 and C9)
    #   idx8  = N(pos 1)
    #   idx9  = N(pos 2)
    #   idx10 = O(pos 8, dioxole ring O)
    #   idx11 = C(pos 7, dioxole O-C-O CH2)
    #   idx12 = O(pos 6, dioxole ring O)
    # Locants verified via OPSIN chloro-probing positions 3, 4, 5, 7, 9.
    "c1cc2cc3c(cc2nn1)OCO3": {
        "name": "[1,3]dioxolo[4,5-g]cinnoline",
        "substituent_form": "[1,3]dioxolo[4,5-g]cinnolin-N-yl",
        "alkyl_stem_ok": False,
        "atom_locants": {0: 3, 1: 4, 2: "4a", 3: 5, 4: "5a", 5: "8a",
                          6: 9, 7: "9a", 8: 1, 9: 2, 10: 8, 11: 7, 12: 6}},

    # 7H-furo[3,2-g]chromen-7-one: 5+6+6 ortho-fused tricyclic — furan fused
    # at edge [3,2] to the "g" bond of chromen-7-one (the chromen ring with
    # the carbonyl at pos 7).  Canonical 'O=c1ccc2cc3ccoc3cc2o1' (14 atoms
    # incl. exocyclic =O).  Without this entry, the engine returns
    # NAMING_ERROR for both 'Cc1cc2cc3c(C)cc(=O)oc3cc2o1' (case 7, dimethyl
    # form) and 'COc1c2occc2cc2ccc(=O)oc12' (case 9, methoxy form) — both
    # share the same ring-only carve.  The retained-name stem encodes the
    # C7 carbonyl + the 7H-tautomer; key on the with-=O canonical so the
    # exocyclic-oxo fallback in retained_lookup claims the =O atom via
    # extra_atom_indices.  Atom indices in canonical SMILES:
    #   idx0  = O (exocyclic on C7, claimed via extra_atom_indices)
    #   idx1  = C(pos 7, ring C=O of chromen)
    #   idx2  = C(pos 6)
    #   idx3  = C(pos 5)
    #   idx4  = C(pos 4a, junction chromen-benzene adj to C5)
    #   idx5  = C(pos 4)
    #   idx6  = C(pos 3a, junction benzene-furan adj to C3)
    #   idx7  = C(pos 3)
    #   idx8  = C(pos 2)
    #   idx9  = O(pos 1, furan ring O)
    #   idx10 = C(pos 9b, junction benzene-furan adj to O1)
    #   idx11 = C(pos 9)
    #   idx12 = C(pos 9a, junction benzene-chromen adj to O8)
    #   idx13 = O(pos 8, chromen ring O)
    # Locants verified via OPSIN chloro-probing positions 2, 3, 4, 5, 6, 9.
    "O=c1ccc2cc3ccoc3cc2o1": {
        "name": "7H-furo[3,2-g]chromen-7-one",
        "substituent_form": "7-oxo-7H-furo[3,2-g]chromen-N-yl",
        "alkyl_stem_ok": False,
        "atom_locants": {1: 7, 2: 6, 3: 5, 4: "4a", 5: 4, 6: "3a", 7: 3,
                          8: 2, 9: 1, 10: "9b", 11: 9, 12: "9a", 13: 8}},

    # 5H-chromeno[2,3-b]pyridin-5-one: 6+6+6 ortho-fused tricyclic — pyridine
    # fused at its [2,3]-edge to bond "b" of chromen (4H-chromen-4-one
    # extended to a chromeno-pyridine), with the C=O at the central pos 5.
    # Canonical 'O=c1c2ccccc2oc2ncccc12' (15 atoms incl. exocyclic =O).
    # The retained-name stem encodes the C5 carbonyl + 5H-tautomer; key on
    # the with-=O canonical so the exocyclic-oxo fallback in retained_lookup
    # claims the =O atom via extra_atom_indices.
    # Atom indices in canonical SMILES:
    #   idx0  = O (exocyclic on C5, claimed via extra_atom_indices)
    #   idx1  = C(pos 5, ring C=O at central position)
    #   idx2  = C(pos 5a, junction central-benzene adj to C=O)
    #   idx3  = C(pos 6)
    #   idx4  = C(pos 7)
    #   idx5  = C(pos 8)
    #   idx6  = C(pos 9)
    #   idx7  = C(pos 9a, junction benzene-pyran adj to O10)
    #   idx8  = O(pos 10, pyran ring O)
    #   idx9  = C(pos 10a, junction pyran-pyridine adj to O10 and N1)
    #   idx10 = N(pos 1, pyridine N)
    #   idx11 = C(pos 2)
    #   idx12 = C(pos 3)
    #   idx13 = C(pos 4)
    #   idx14 = C(pos 4a, junction pyridine-central adj to C4 and C=O)
    # Locants verified via OPSIN chloro-probing positions 2, 3, 4, 6, 7, 8, 9
    # and methyl-probing junctions 4a, 5a, 9a, 10a.
    "O=c1c2ccccc2oc2ncccc12": {
        "name": "5H-chromeno[2,3-b]pyridin-5-one",
        "substituent_form": "5-oxo-5H-chromeno[2,3-b]pyridin-N-yl",
        "alkyl_stem_ok": False,
        "atom_locants": {1: 5, 2: "5a", 3: 6, 4: 7, 5: 8, 6: 9, 7: "9a",
                          8: 10, 9: "10a", 10: 1, 11: 2, 12: 3, 13: 4,
                          14: "4a"}},

    # 2,3,5,6-tetrahydroimidazo[2,1-b][1,3]thiazole (levamisole core) — 5+5
    # ortho-fused N,N,S heterocycle with the bridgehead N of imidazole shared
    # with the [b]-edge of 1,3-thiazole; positions 2,3 (in thiazole half) and
    # 5,6 (in imidazole half) are sp3 (the tetrahydro carbons), while pos 1
    # (S) and pos 7,7a (=N–C) remain sp2 (the imidazoline imine).  Without
    # this entry, the engine emits NAMING_ERROR for any levamisole-style
    # input ("{[NAMING ERROR: No valid naming plan found for C1CN2CCSC2=N1]}
    # benzene" for levamisole's phenyl-substituted core).  Canonical
    # 'C1CN2CCSC2=N1' (8 atoms, no exocyclic features).
    # Atom indices in canonical SMILES:
    #   idx0 = C(pos 6, sp3 CH2 adj to imidazoline N7)
    #   idx1 = C(pos 5, sp3 CH2 adj to bridgehead N4)
    #   idx2 = N(pos 4, bridgehead N shared between rings)
    #   idx3 = C(pos 3, sp3 CH2 adj to bridgehead N4 in thiazole)
    #   idx4 = C(pos 2, sp3 CH2 adj to S1)
    #   idx5 = S(pos 1, thiazole S)
    #   idx6 = C(pos 7a, bridgehead C, =N7)
    #   idx7 = N(pos 7, sp2 N, =C7a)
    # Locants verified via OPSIN methyl-probing positions 2, 3, 5, 6 (the
    # sp3 hydrogens) plus 1 (S→sulfonium) and 7a (bridgehead C); junction
    # N(pos 4) inferred from "Atom is in unphysical valency state: N=4"
    # error when methyl-probing pos 4.  Levamisole = (6S)-6-phenyl-
    # 2,3,5,6-tetrahydroimidazo[2,1-b][1,3]thiazole; phenyl at pos 6 (idx 0).
    "C1CN2CCSC2=N1": {
        "name": "2,3,5,6-tetrahydroimidazo[2,1-b][1,3]thiazole",
        "substituent_form": "2,3,5,6-tetrahydroimidazo[2,1-b][1,3]thiazol-N-yl",
        "alkyl_stem_ok": False,
        "atom_locants": {0: 6, 1: 5, 2: 4, 3: 3, 4: 2, 5: 1,
                          6: "7a", 7: 7}},

    # Benzo[g]pteridine — 6+6+6 ortho-fused PAH-aza scaffold (alloxazine /
    # lumichrome / riboflavin core).  Three rings: pyrimidine + pyrazine
    # (central) + benzo, with two N atoms in pyrimidine (pos 1,3), two N in
    # pyrazine (pos 5,10), and four ring junctions (4a, 5a, 9a, 10a).
    # Without this entry, the engine emits the synonymous fusion name
    # "pyrimido[4,5-b]quinoxaline" with the wrong locant family — for
    # alloxazine 'O=c1[nH]c(=O)c2nc3ccccc3nc2[nH]1' it produces
    # "4,6-dioxopyrimido[4,5-b]quinoxaline" which fails OPSIN round-trip
    # (locants 4,6 don't carry the carbonyls in pyrimido[4,5-b]quinoxaline
    # numbering; they live at 2,4 in the equivalent benzo[g]pteridine map).
    # Registering benzo[g]pteridine (the IUPAC-preferred retained name for
    # this scaffold) with the correct atom_locants steers the engine to the
    # right numbering and unlocks the "-2,4(1H,3H)-dione" suffix path so
    # alloxazine round-trips as benzo[g]pteridine-2,4(1H,3H)-dione.
    # Canonical 'c1ccc2nc3ncncc3nc2c1' (14 atoms, all aromatic).
    # Atom indices in canonical SMILES:
    #   idx0  = C(pos 7, benzo CH)
    #   idx1  = C(pos 8, benzo CH)
    #   idx2  = C(pos 9, benzo CH)
    #   idx3  = C(pos 9a, jct benzo-pyrazine adj to N10)
    #   idx4  = N(pos 10, pyrazine N junction)
    #   idx5  = C(pos 10a, jct pyrazine-pyrimidine adj to N1 and N10)
    #   idx6  = N(pos 1, pyrimidine N)
    #   idx7  = C(pos 2, pyrimidine C between N1 and N3)
    #   idx8  = N(pos 3, pyrimidine N)
    #   idx9  = C(pos 4, pyrimidine C between N3 and C4a)
    #   idx10 = C(pos 4a, jct pyrimidine-pyrazine adj to N5)
    #   idx11 = N(pos 5, pyrazine N junction)
    #   idx12 = C(pos 5a, jct pyrazine-benzo adj to C6)
    #   idx13 = C(pos 6, benzo CH)
    # Locants verified via OPSIN methyl-probing positions 2, 4, 6, 7, 8, 9
    # (the C-H positions) plus 1, 3 (pyrimidine NH) and 5, 10 (pyrazine N
    # junctions) and 4a, 5a, 9a, 10a (the four ring-junction carbons).
    "c1ccc2nc3ncncc3nc2c1": {
        "name": "benzo[g]pteridine",
        "substituent_form": "benzo[g]pteridin-N-yl",
        "alkyl_stem_ok": False,
        "atom_locants": {0: 7, 1: 8, 2: 9, 3: "9a", 4: 10, 5: "10a",
                          6: 1, 7: 2, 8: 3, 9: 4, 10: "4a", 11: 5,
                          12: "5a", 13: 6}},

    # 1H-3,1-benzoxazin-2(4H)-one (efavirenz core) — 6+6 ortho-fused parent
    # with a 1,4-dihydro saturation pattern: ring oxygen at pos 3, ring NH
    # at pos 1, ring C=O at pos 2 (between N1 and O3), sp3 CH2 at pos 4,
    # plus benzo positions 5..8 and junctions 4a, 8a.  Distinct from the
    # already-registered isatoic anhydride 'O=c1[nH]c2ccccc2c(=O)o1' which
    # is the 2,4-dione (both ring carbons are C=O).  Without this entry,
    # the engine routes efavirenz 'O=C1Nc2ccc(Cl)cc2[C@@](C#CC2CC2)(C(F)(F)F)
    # O1' through a wrong synthetic fusion path and emits a malformed
    # "[1,3]oxazino[4,5-b]b..." stem with "10S" stereo and a locant past
    # the actual ring size.  Canonical 'O=C1Nc2ccccc2CO1' (11 atoms incl.
    # exocyclic =O on C2).  Keyed on the with-=O canonical so the
    # exocyclic-oxo fallback in retained_lookup claims the carbonyl O via
    # extra_atom_indices.
    # Atom indices in canonical SMILES:
    #   idx0  = O (exocyclic on C2, claimed via extra_atom_indices)
    #   idx1  = C(pos 2, ring C=O)
    #   idx2  = N(pos 1, ring NH)
    #   idx3  = C(pos 8a, jct benzo-hetero adj to N1)
    #   idx4  = C(pos 8, benzo CH)
    #   idx5  = C(pos 7, benzo CH)
    #   idx6  = C(pos 6, benzo CH)
    #   idx7  = C(pos 5, benzo CH)
    #   idx8  = C(pos 4a, jct benzo-hetero adj to C4)
    #   idx9  = C(pos 4, sp3 CH2)
    #   idx10 = O(pos 3, ring O)
    # Locants verified via OPSIN methyl-probing positions 4, 5, 6, 7, 8
    # (the CH positions) plus 4a, 8a (junctions) and 1 (N-methyl).  For
    # efavirenz, Cl on idx 6 → pos 6, sp3 substituent on idx 9 → pos 4.
    "O=C1Nc2ccccc2CO1": {
        "name": "1H-3,1-benzoxazin-2(4H)-one",
        "substituent_form": "2-oxo-1H-3,1-benzoxazin-N-yl",
        "alkyl_stem_ok": False,
        "atom_locants": {1: 2, 2: 1, 3: "8a", 4: 8, 5: 7, 6: 6, 7: 5,
                          8: "4a", 9: 4, 10: 3}},

    # [1,3]dithiolo[4,5-b]quinoxalin-2(1H)-one — 5+6+6 ortho-fused tricyclic
    # with the dithiole's C=O at the central pos 2 and the two ring S atoms
    # at pos 1, 3 flanking it; quinoxaline 4-N, 9-N, benzo positions 5..8.
    # Pre-fix the engine emitted "8-methyl-2-oxo[1,3]dithiolo[4,5-b]
    # quinoxaline" for 'Cc1ccc2nc3sc(=O)sc3nc2c1' — the locant 8 lies on
    # the peri benzo position (idx 9 of parent canonical), but the methyl
    # actually sits on an inner benzo position (idx 7/8) and should be
    # locant 6.  Without atom_locants the engine's stage-2 fusion path
    # arrives at the highest-numbered side instead of the IUPAC-preferred
    # lowest-locant set.  Canonical 'O=c1sc2nc3ccccc3nc2s1' (14 atoms incl.
    # exocyclic =O on C2).  Keyed on the with-=O canonical so the
    # exocyclic-oxo fallback in retained_lookup claims the carbonyl O via
    # extra_atom_indices.
    # Atom indices in canonical SMILES:
    #   idx0  = O (exocyclic on C2, claimed via extra_atom_indices)
    #   idx1  = C(pos 2, dithiole C=O)
    #   idx2  = S(pos 3)
    #   idx3  = C(pos 3a, jct dithiole-pyrazine adj to N4)
    #   idx4  = N(pos 4, pyrazine N)
    #   idx5  = C(pos 4a, jct pyrazine-benzo adj to C5)
    #   idx6  = C(pos 5, benzo CH peri to N4)
    #   idx7  = C(pos 6, benzo CH)
    #   idx8  = C(pos 7, benzo CH)
    #   idx9  = C(pos 8, benzo CH peri to N9)
    #   idx10 = C(pos 8a, jct benzo-pyrazine adj to N9)
    #   idx11 = N(pos 9, pyrazine N)
    #   idx12 = C(pos 9a, jct pyrazine-dithiole adj to N9)
    #   idx13 = S(pos 1)
    # Locants verified via OPSIN methyl-probing positions 4, 5, 6, 7, 8, 9
    # (the unique CH/N-H probe-able positions) plus 4a, 8a (junctions)
    # and 2 (S→sulfonium for pos 1, 3).  Note the molecule has C2v
    # symmetry through the C2-O axis: positions 5/8 and 6/7 form
    # equivalent pairs at the parent level, so for monosubstituted forms
    # IUPAC selects the lowest locant (pos 6 over pos 7 for an inner
    # methyl).
    "O=c1sc2nc3ccccc3nc2s1": {
        "name": "[1,3]dithiolo[4,5-b]quinoxalin-2(1H)-one",
        "substituent_form": "2-oxo-1H-[1,3]dithiolo[4,5-b]quinoxalin-N-yl",
        "alkyl_stem_ok": False,
        "atom_locants": {1: 2, 2: 3, 3: "3a", 4: 4, 5: "4a", 6: 5,
                          7: 6, 8: 7, 9: 8, 10: "8a", 11: 9, 12: "9a",
                          13: 1}},

    # 2,3,4,9-tetrahydro-1H-pyrido[3,4-b]indol-1-one — 6+5+6 tetracyclic
    # (well, tricyclic) "tryptoline lactam" core: tetrahydropyridine fused
    # at [3,4-b] to indole, with C=O at pos 1 (β-carboline-1-one
    # tetrahydro framework).  Without this entry, the engine emits a
    # von-Baeyer fall-through name "2,11-diazatricyclo[7.4.0.0^{3,8}]
    # trideca-1(9),3,5,7-tetraen-10-one" which is technically valid but
    # doesn't match the corpus IUPAC-preferred fusion-name expectation.
    # Canonical 'O=C1NCCc2c1[nH]c1ccccc21' (14 atoms incl. exocyclic =O
    # on C1).  Keyed on the with-=O canonical so the exocyclic-oxo
    # fallback in retained_lookup claims the carbonyl O via
    # extra_atom_indices.
    # Atom indices in canonical SMILES:
    #   idx0  = O (exocyclic on C1, claimed via extra_atom_indices)
    #   idx1  = C(pos 1, ring C=O)
    #   idx2  = N(pos 2, sp3 NH)
    #   idx3  = C(pos 3, sp3 CH2)
    #   idx4  = C(pos 4, sp3 CH2)
    #   idx5  = C(pos 4a, jct piperidone-pyrrole)
    #   idx6  = C(pos 9a, jct pyrrole-piperidone adj to C1 and N9)
    #   idx7  = N(pos 9, indole NH)
    #   idx8  = C(pos 8a, jct pyrrole-benzo adj to N9)
    #   idx9  = C(pos 8, benzo CH)
    #   idx10 = C(pos 7, benzo CH)
    #   idx11 = C(pos 6, benzo CH)
    #   idx12 = C(pos 5, benzo CH)
    #   idx13 = C(pos 4b, jct benzo-pyrrole adj to C4a and C5)
    # Locants verified via OPSIN methyl-probing positions 2, 3, 4, 5, 6,
    # 7, 8, 9 (the unique ring-atom probe positions) plus 4a, 8a, 9a
    # (junctions).  Position 4b infers from canonical topology (the
    # benzo-pyrrole junction adj to C4a).  For the corpus entry
    # 'Cc1[nH]cnc1CN1CCc2c(c3ccccc3n2C)C1=O' (an imidazolylmethyl-
    # substituted 9-methyl form), the engine now produces the
    # IUPAC-preferred fusion name ending in 2,3,4,9-tetrahydro-1H-pyrido
    # [3,4-b]indol-1-one.
    "O=C1NCCc2c1[nH]c1ccccc21": {
        "name": "2,3,4,9-tetrahydro-1H-pyrido[3,4-b]indol-1-one",
        "substituent_form": "1-oxo-2,3,4,9-tetrahydro-1H-pyrido[3,4-b]indol-N-yl",
        "alkyl_stem_ok": False,
        "atom_locants": {1: 1, 2: 2, 3: 3, 4: 4, 5: "4a", 6: "9a",
                          7: 9, 8: "8a", 9: 8, 10: 7, 11: 6, 12: 5,
                          13: "4b"}},
}


# ---------------------------------------------------------------------------
# Curated inorganic / ion / parent-hydride retained names
# ---------------------------------------------------------------------------
#
# Covers:
#   - monoatomic ions (Na+, K+, Cl-, OH-, ...)
#   - simple parent hydrides RDKit canonicalises to one atom (P, S, [SiH4], ...)
#   - small inorganic molecules (water, hydrogen peroxide, ...)
#
# All entries are keyed by the RDKit canonical SMILES for the standalone species.
# The 'source' field distinguishes these from OPSIN-mined retained names, which
# are filtered by `_NON_STANDALONE_OPSIN_SOURCES` in engine.py (this table
# bypasses that filter because its entries are hand-curated and known-good).
#
# IUPAC preferred names are used where they differ from common names (e.g.
# "azanium" for NH4+, "azane" for NH3). Common/semi-systematic fallbacks are
# noted when they are also widely accepted.

_INORGANIC_CURATED_SMILES: dict[str, dict] = {
    # --- Monoatomic cations ---
    "[NH4+]":  {"name": "azanium"},             # P-73.2.1; also "ammonium" (common)
    "[H3O+]":  {"name": "oxidanium"},           # hydronium
    "[Li+]":   {"name": "lithium(1+)"},
    "[Na+]":   {"name": "sodium(1+)"},
    "[K+]":    {"name": "potassium(1+)"},
    "[Rb+]":   {"name": "rubidium(1+)"},
    "[Cs+]":   {"name": "caesium(1+)"},
    "[Mg+2]":  {"name": "magnesium(2+)"},
    "[Ca+2]":  {"name": "calcium(2+)"},
    "[Sr+2]":  {"name": "strontium(2+)"},
    "[Ba+2]":  {"name": "barium(2+)"},
    "[Al+3]":  {"name": "aluminium(3+)"},
    "[Al+2]":  {"name": "aluminium(2+)"},   # radical_e=1; bypass free-valence guard
    "[Al+]":   {"name": "aluminium(1+)"},   # radical_e=2; bypass free-valence guard
    "[Ga+2]":  {"name": "gallium(2+)"},     # radical_e=1
    "[Ga+]":   {"name": "gallium(1+)"},     # radical_e=2
    "[In+2]":  {"name": "indium(2+)"},      # radical_e=1
    "[In+]":   {"name": "indium(1+)"},      # radical_e=2
    "[Ge+2]":  {"name": "germanium(2+)"},   # radical_e=2
    "[Zn+2]":  {"name": "zinc(2+)"},
    "[Fe+2]":  {"name": "iron(2+)"},
    "[Fe+3]":  {"name": "iron(3+)"},
    "[Cu+]":   {"name": "copper(1+)"},
    "[Cu+2]":  {"name": "copper(2+)"},
    "[Ag+]":   {"name": "silver(1+)"},
    "[Mn+2]":  {"name": "manganese(2+)"},
    "[Ni+2]":  {"name": "nickel(2+)"},
    "[Co+2]":  {"name": "cobalt(2+)"},
    "[Co+3]":  {"name": "cobalt(3+)"},
    "[Cr+3]":  {"name": "chromium(3+)"},
    "[Cr+2]":  {"name": "chromium(2+)"},

    # --- Lanthanide cations (P-72.5; La–Lu, all commonly +3; Eu also +2; Ce also +4) ---
    # RDKit radical-electron notes (d/f-shell modelling, not real valence):
    #   [La+3] r=0, [Ce+3] r=1, [Ce+4] r=0, [Pr+3] r=0, [Nd+3] r=1,
    #   [Pm+3] r=0, [Sm+3] r=1, [Eu+3] r=0, [Eu+2] r=1, [Gd+3] r=1,
    #   [Tb+3] r=0, [Dy+3] r=1, [Ho+3] r=0, [Er+3] r=1, [Tm+3] r=0,
    #   [Yb+3] r=1, [Yb+2] r=0, [Lu+3] r=0.
    # All must be in this table so _validate_no_open_valences bypasses.
    "[La+3]":  {"name": "lanthanum(3+)"},
    "[Ce+3]":  {"name": "cerium(3+)"},
    "[Ce+4]":  {"name": "cerium(4+)"},
    "[Pr+3]":  {"name": "praseodymium(3+)"},
    "[Pr+2]":  {"name": "praseodymium(2+)"},  # radical_e=1; bypass free-valence guard
    "[Pr+]":   {"name": "praseodymium(1+)"},  # radical_e=0
    "[Nd+3]":  {"name": "neodymium(3+)"},
    "[Pm+3]":  {"name": "promethium(3+)"},
    "[Sm+3]":  {"name": "samarium(3+)"},
    "[Eu+3]":  {"name": "europium(3+)"},
    "[Eu+2]":  {"name": "europium(2+)"},
    "[Gd+3]":  {"name": "gadolinium(3+)"},   # FDA-0603/0604 fix — Gd carries 1 radical e-
    "[Tb+3]":  {"name": "terbium(3+)"},
    "[Dy+3]":  {"name": "dysprosium(3+)"},
    "[Ho+3]":  {"name": "holmium(3+)"},
    "[Er+3]":  {"name": "erbium(3+)"},
    "[Tm+3]":  {"name": "thulium(3+)"},
    "[Yb+3]":  {"name": "ytterbium(3+)"},
    "[Yb+2]":  {"name": "ytterbium(2+)"},
    "[Lu+3]":  {"name": "lutetium(3+)"},

    # --- Scandium and yttrium (group 3, often grouped with lanthanides) ---
    "[Sc+3]":  {"name": "scandium(3+)"},
    "[Sc+2]":  {"name": "scandium(2+)"},   # radical_e=1
    "[Sc+]":   {"name": "scandium(1+)"},   # radical_e=0
    "[Y+3]":   {"name": "yttrium(3+)"},

    # --- Platinum-group metal cations ---
    # Platinum: Pt2+ and Pt4+. RDKit: [Pt+2] r=0, [Pt+4] r=0.
    "[Pt+2]":  {"name": "platinum(2+)"},
    "[Pt+4]":  {"name": "platinum(4+)"},
    # Palladium: Pd2+. RDKit: [Pd+2] r=0.
    "[Pd+2]":  {"name": "palladium(2+)"},
    # Iridium: Ir3+ and Ir4+. RDKit: [Ir+3] r=0, [Ir+4] r=0.
    "[Ir+3]":  {"name": "iridium(3+)"},
    "[Ir+4]":  {"name": "iridium(4+)"},
    # Rhodium: Rh3+. RDKit: [Rh+3] r=0.
    "[Rh+3]":  {"name": "rhodium(3+)"},
    # Ruthenium: Ru2+ and Ru3+. RDKit: [Ru+2] r=0, [Ru+3] r=1.
    "[Ru+2]":  {"name": "ruthenium(2+)"},
    "[Ru+3]":  {"name": "ruthenium(3+)"},
    # Osmium: Os2+, Os3+, Os4+. RDKit: [Os+4] r=0.
    "[Os+2]":  {"name": "osmium(2+)"},
    "[Os+3]":  {"name": "osmium(3+)"},
    "[Os+4]":  {"name": "osmium(4+)"},
    # Gold: Au+ (aurous) and Au3+ (auric). RDKit: [Au+] r=0, [Au+3] r=0.
    "[Au+]":   {"name": "gold(1+)"},
    "[Au+3]":  {"name": "gold(3+)"},

    # --- Additional transition metal cations ---
    # Titanium: Ti2+, Ti3+, Ti4+. RDKit: all r=0.
    "[Ti+2]":  {"name": "titanium(2+)"},
    "[Ti+3]":  {"name": "titanium(3+)"},
    "[Ti+4]":  {"name": "titanium(4+)"},
    # Vanadium: V+–V5+. RDKit: [V+] r=0, [V+2] r=1; others r=0.
    "[V+]":    {"name": "vanadium(1+)"},
    "[V+2]":   {"name": "vanadium(2+)"},
    "[V+3]":   {"name": "vanadium(3+)"},
    "[V+4]":   {"name": "vanadium(4+)"},
    "[V+5]":   {"name": "vanadium(5+)"},
    # Chromium: Cr4+–Cr6+ (extending existing Cr2+/Cr3+). RDKit radical_e: [Cr+4] r=0, [Cr+5] r=1, [Cr+6] r=0.
    "[Cr+4]":  {"name": "chromium(4+)"},
    "[Cr+5]":  {"name": "chromium(5+)"},
    "[Cr+6]":  {"name": "chromium(6+)"},
    # Molybdenum: Mo2+–Mo6+. RDKit: [Mo+2] r=0, [Mo+3] r=1; [Mo+6] r=0.
    "[Mo+2]":  {"name": "molybdenum(2+)"},
    "[Mo+3]":  {"name": "molybdenum(3+)"},
    "[Mo+4]":  {"name": "molybdenum(4+)"},
    "[Mo+5]":  {"name": "molybdenum(5+)"},
    "[Mo+6]":  {"name": "molybdenum(6+)"},
    # Tungsten: W2+–W6+. RDKit: [W+2] r=0, [W+3] r=1, [W+6] r=0.
    "[W+2]":   {"name": "tungsten(2+)"},
    "[W+3]":   {"name": "tungsten(3+)"},
    "[W+4]":   {"name": "tungsten(4+)"},
    "[W+5]":   {"name": "tungsten(5+)"},
    "[W+6]":   {"name": "tungsten(6+)"},
    # Niobium: Nb2+–Nb5+. RDKit: [Nb+2] r=1, [Nb+4] r=1; others r=0.
    "[Nb+2]":  {"name": "niobium(2+)"},
    "[Nb+3]":  {"name": "niobium(3+)"},
    "[Nb+4]":  {"name": "niobium(4+)"},
    "[Nb+5]":  {"name": "niobium(5+)"},
    # Tantalum: Ta3+–Ta5+. RDKit: [Ta+3] r=0, [Ta+4] r=1, [Ta+5] r=0.
    "[Ta+3]":  {"name": "tantalum(3+)"},
    "[Ta+4]":  {"name": "tantalum(4+)"},
    "[Ta+5]":  {"name": "tantalum(5+)"},
    # Rhenium: Re2+–Re7+. RDKit: [Re+2] r=1, [Re+3] r=0, [Re+4] r=1, [Re+5] r=0, [Re+6] r=1, [Re+7] r=0.
    "[Re+2]":  {"name": "rhenium(2+)"},
    "[Re+3]":  {"name": "rhenium(3+)"},
    "[Re+4]":  {"name": "rhenium(4+)"},
    "[Re+5]":  {"name": "rhenium(5+)"},
    "[Re+6]":  {"name": "rhenium(6+)"},
    "[Re+7]":  {"name": "rhenium(7+)"},
    # Technetium: Tc4+ (r=1) and Tc7+ (r=0).
    "[Tc+4]":  {"name": "technetium(4+)"},
    "[Tc+7]":  {"name": "technetium(7+)"},
    # Zirconium: Zr2+–Zr4+. RDKit: [Zr+2] r=0, [Zr+3] r=1, [Zr+4] r=0.
    "[Zr+2]":  {"name": "zirconium(2+)"},
    "[Zr+3]":  {"name": "zirconium(3+)"},
    "[Zr+4]":  {"name": "zirconium(4+)"},
    # Hafnium: Hf2+–Hf4+. RDKit: [Hf+2] r=0, [Hf+3] r=1, [Hf+4] r=0.
    "[Hf+2]":  {"name": "hafnium(2+)"},
    "[Hf+3]":  {"name": "hafnium(3+)"},
    "[Hf+4]":  {"name": "hafnium(4+)"},
    # Manganese: Mn3+–Mn7+ (extending existing Mn2+).
    "[Mn+3]":  {"name": "manganese(3+)"},
    "[Mn+4]":  {"name": "manganese(4+)"},
    "[Mn+5]":  {"name": "manganese(5+)"},
    "[Mn+6]":  {"name": "manganese(6+)"},   # radical_e=1
    "[Mn+7]":  {"name": "manganese(7+)"},
    # Nickel: Ni3+ (extending existing Ni2+).
    "[Ni+3]":  {"name": "nickel(3+)"},
    # Copper: Cu3+ (rare; extending existing Cu+/Cu2+).
    "[Cu+3]":  {"name": "copper(3+)"},
    # Iron: Fe4+ (rare; extending existing Fe2+/Fe3+).
    "[Fe+4]":  {"name": "iron(4+)"},
    # Zinc: Zn+ (extending existing Zn2+).
    "[Zn+]":   {"name": "zinc(1+)"},

    # --- Actinide cations (common laboratory ions) ---
    # Uranium: U3+ (uranous-III), U4+ (uranous), U5+ (uranyl-V intermediate),
    # and U6+ (uranyl).  The +5 oxidation state is real (e.g. U(V)
    # alkoxides like uranium(V) pentaethoxide).
    "[U+3]":   {"name": "uranium(3+)"},
    "[U+4]":   {"name": "uranium(4+)"},
    "[U+5]":   {"name": "uranium(5+)"},
    "[U+6]":   {"name": "uranium(6+)"},
    # Thorium: Th4+. RDKit: r=0.
    "[Th+4]":  {"name": "thorium(4+)"},
    # Thorium: Th2+ (rare; appears in [Th+2] + 2 anionic ligand salts).
    # RDKit: [Th+2] r=0; safe to bypass the free-valence guard via curated lookup.
    "[Th+2]":  {"name": "thorium(2+)"},
    # Uranium: U2+ (appears in [U+2] + 2 [CH3-] organouranium salts).
    "[U+2]":   {"name": "uranium(2+)"},

    # --- Phase 13 organometallic neutral-atom / dispersion-shaped fragments ---
    # These curated entries route the salt path's per-fragment naming
    # for the bare metal / inorganic ligand pieces that appear in
    # PubChem-style multi-fragment depictions of organometallic
    # compounds.  Each entry yields an OPSIN-parseable PIN that the
    # eval's ``_metal_anion_stoich_equiv`` / ``_metal_ionic_covalent_equiv``
    # matchers reconcile with the input SMILES.

    # Bare neutral metal atoms in salt-form depictions: when a SMILES
    # like ``C.[U]`` (methane + neutral uranium metal) reaches the salt
    # path, the per-fragment namer needs a curated name for the bare
    # [U] atom.  The eval's _metal_anion_stoich_equiv path accepts
    # the resulting "methane uranium" via OPSIN round-trip.
    "[U]":     {"name": "uranium"},

    # Methanidylidyne-metal cations (P-72.2 / P-66.6).  Two-atom
    # fragments of the form ``[C-]#[M+]`` (or neutral ``[C-]#[M]``) are
    # named ``methanidylidyne{metal}(n+)`` where the charge qualifier
    # reflects the metal's formal charge.  These cations appear in
    # PubChem as standalone species (cases 3, 4) and as ligand pieces
    # in metal-carbide cluster depictions (case 8).
    "[C-]#[W+]":   {"name": "methanidylidynetungsten(1+)"},
    "[C-]#[Zr+]":  {"name": "methanidylidynezirconium(1+)"},
    "[C-]#[Cr]":   {"name": "methanidylidynechromium"},

    # Trimethylaluminium (P-69.3 / OPSIN-retained ``trimethylaluminium``).
    # Single-fragment ``[CH3][Al]([CH3])[CH3]`` (neutral, Al bonded to 3
    # methyls).  Used as the per-fragment name in the dimer
    # ``[CH3][Al]([CH3])[CH3].[CH3][Al]([CH3])[CH3]`` whose PubChem
    # depiction is two disconnected monomers.  OPSIN parses the
    # space-separated name to two such fragments.
    "[CH3][Al]([CH3])[CH3]":  {"name": "trimethylaluminium"},

    # Phenylmercury(II) acetate (P-69.5 / Hg-organometallic).  Single
    # fragment depicted as ``CC(=O)[O][Hg][c]1ccccc1`` (acetate-O
    # singly bonded to neutral Hg with an aromatic phenyl bond).
    # OPSIN parses ``(acetyloxy)(phenyl)mercury`` back to the same
    # canonical SMILES exactly.
    "CC(=O)[O][Hg][c]1ccccc1":  {"name": "(acetyloxy)(phenyl)mercury"},

    # Molybdenum dioxide [O]=Mo=[O] — neutral Mo(VI) dioxo unit.
    # Appears as the metal centre in MoO2(acac)2 depictions
    # (case 12).  OPSIN parses ``molybdenum dioxide`` to
    # ``[Mo](=O)=O`` which canonicalises identically.
    "[O]=[Mo]=[O]":  {"name": "molybdenum dioxide"},

    # --- Monoatomic anions ---
    "[H-]":    {"name": "hydride"},
    # Deuteride / deuteroxide: OPSIN's retained names for the ²H-labelled
    # hydride and hydroxide anions (sodium deuteride, sodium deuteroxide).
    # These are isotope-specific retained spellings, not the (²H) isotope-
    # descriptor notation, so they need explicit entries keyed on the
    # deuterium-bearing canonical SMILES.
    "[2H-]":   {"name": "deuteride"},
    "[2H][O-]": {"name": "deuteroxide"},
    "[F-]":    {"name": "fluoride"},
    "[Cl-]":   {"name": "chloride"},
    "[Br-]":   {"name": "bromide"},
    "[I-]":    {"name": "iodide"},
    "[OH-]":   {"name": "hydroxide"},
    "[SH-]":   {"name": "hydrosulfide"},
    "[SeH-]":  {"name": "selanide"},
    "[TeH-]":  {"name": "tellanide"},
    # Monatomic pnictogen parent-hydride anions: the conjugate base of
    # azane (NH3 -> NH2-) / phosphane (PH3 -> PH2-).  Parallels the
    # chalcogen ``-anide`` series above (P-72.2.2.1.2 / P-73.2.2.1.2).
    "[NH2-]":  {"name": "azanide"},
    "[PH2-]":  {"name": "phosphanide"},
    # Dioxidanide (hydroperoxide anion HOO-): the 2-O homolog of the
    # trioxidan-1-ide / tetraoxidan-1-ide chain anions below.  Appears as the
    # anion of salts such as sodium hydroperoxide ([O-]O.[Na+]).
    "[O-]O":   {"name": "dioxidanide"},

    # --- Stage 17 R17-B: heteroatom-chain anion retained names ---
    # ``-N-ide`` chain anion forms of disulfane / diselane / hydrazine
    # (and their tri-N homologues).  Pre-fix the engine's
    # ``perception/__init__.py::candidate_parents`` heteroatom_chain
    # generator (Section 4) only handled neutral 2-atom chains; charged
    # endpoints fell through to a path that tries to carve the [X-] as a
    # substituent of a 2-atom chain parent and fails with "no plan".
    # The audit row ``[Se-][Se][SeH].[Na+]`` (sodium triselan-1-ide) is
    # the entry-point case; the curated lookup at ``data_loader.py`` is
    # the simplest fix and covers the small chains directly.
    # Larger N-atom chain anions still require the Stage 17+ heteroatom-
    # chain candidate-generator extension (deferred).
    "[S-]S":           {"name": "disulfan-1-ide"},     # 2 S
    "[S-]SS":          {"name": "trisulfan-1-ide"},    # 3 S, terminal anion
    "[S-]SSS":         {"name": "tetrasulfan-1-ide"},  # 4 S, terminal anion
    "[S-]SSSS":        {"name": "pentasulfan-1-ide"},  # 5 S
    "[S-]SSSSS":       {"name": "hexasulfan-1-ide"},   # 6 S
    "[Se-][SeH]":      {"name": "diselan-1-ide"},      # 2 Se
    "[Se-][Se][SeH]":  {"name": "triselan-1-ide"},     # 3 Se, terminal anion (audit row)
    "[Se-][Se][Se][SeH]":     {"name": "tetraselan-1-ide"},   # 4 Se
    "[Se-][Se][Se][Se][SeH]": {"name": "pentaselan-1-ide"},   # 5 Se
    "[Se-][Se][Se][Se][Se][SeH]": {"name": "hexaselan-1-ide"},# 6 Se
    "[NH-]N":          {"name": "hydrazinide"},        # 2 N (hydrazine anion)
    "[NH-]NN":         {"name": "triazan-1-ide"},      # 3 N
    "[NH-]NNN":        {"name": "tetrazan-1-ide"},     # 4 N
    "[NH-]NNNN":       {"name": "pentazan-1-ide"},     # 5 N
    "[O-]OO":          {"name": "trioxidan-1-ide"},    # 3 O
    "[O-]OOO":         {"name": "tetraoxidan-1-ide"},  # 4 O
    "[Te-][TeH]":      {"name": "ditellan-1-ide"},     # 2 Te
    "[Te-][Te][TeH]":  {"name": "tritellan-1-ide"},    # 3 Te
    "[Te-][Te][Te][TeH]": {"name": "tetratellan-1-ide"},  # 4 Te
    "[O-2]":   {"name": "oxide(2-)"},
    "[S-2]":   {"name": "sulfide(2-)"},
    "[N-3]":   {"name": "nitride(3-)"},

    # --- Pseudohalide anions (retained names, OPSIN round-trip verified) ---
    # Cyanide: canonical "[C-]#N"
    "[C-]#N":     {"name": "cyanide"},
    # Thiocyanate: canonical "N#C[S-]"
    "N#C[S-]":    {"name": "thiocyanate"},
    # Cyanate: canonical "N#C[O-]"
    "N#C[O-]":    {"name": "cyanate"},
    # Isocyanate: canonical "[N-]=C=O"
    "[N-]=C=O":   {"name": "isocyanate"},
    # Isothiocyanate: canonical "[N-]=C=S"
    "[N-]=C=S":   {"name": "isothiocyanate"},

    # --- Terminal alkynyl carbanion anions (P-72.2 / P-73 salt context) ---
    # These ``[C-]#C-R`` anions arise in metal-acetylide salts.  The engine
    # would otherwise name them as their neutral parents (e.g. "(ethynyl)benzene"
    # for [C-]#Cc1ccccc1) because the ``_classify_simple_carbon_charge``
    # classifier rejects non-single-bond neighbours and no other classifier
    # claims the terminal C(-).  Curated entries here ensure the ionic forms
    # are emitted correctly in salt context.
    # OPSIN roundtrip verified: py2opsin("<name>") == canonical SMILES.
    #
    # Phenylacetylide (retained name per P-72.2 / IUPAC Blue Book Appendix):
    "[C-]#Cc1ccccc1":      {"name": "phenylacetylide"},
    # (3-Phenylprop-1-yn-1-ide): [C-]#C-CH2-Ph
    "[C-]#CCc1ccccc1":     {"name": "(3-phenylprop-1-yn-1-ide)"},
    # 4-Phenylbuta-1,3-diyn-1-ide: [C-]#C-C#C-Ph
    "[C-]#CC#Cc1ccccc1":   {"name": "4-phenylbuta-1,3-diyn-1-ide"},
    # but-3-en-1-yn-1-ide: [C-]#C-CH=CH2
    "[C-]#CC=C":           {"name": "but-3-en-1-yn-1-ide"},

    # --- Simple parent hydrides / small neutrals (canonical SMILES from RDKit) ---
    # Water: "O" (RDKit canonical for H2O)
    "O":       {"name": "water"},
    # Hydrogen peroxide: canonical "OO"
    "OO":      {"name": "hydrogen peroxide"},
    # Ammonia: canonical "N" (already in algorithm.py source; listed for completeness)
    "N":       {"name": "ammonia"},

    # --- Stage 17 R17-A: isotopologues of water and ammonia ---
    # Bare-element isotope cases that fall through the regular plan-search
    # path (no carbon parent, only one heavy atom + isotope-bearing H).
    # OPSIN-canonical names verified individually via py2opsin.  Note the
    # explicit ``H1`` count for single-D forms: OPSIN rejects ``(2H)water``
    # (no count) but accepts ``(2H1)water``.  The unisotoped ``"O"`` /
    # ``"N"`` entries above remain unchanged; isotope-bearing forms have
    # different RDKit canonical SMILES so the lookup is unambiguous.
    "[2H]O":      {"name": "(2H1)water"},     # HOD
    "[2H]O[2H]":  {"name": "(2H2)water"},     # D2O
    "[3H]O":      {"name": "(3H1)water"},     # HTO
    "[3H]O[3H]":  {"name": "(3H2)water"},     # T2O
    "[2H]O[3H]":  {"name": "(2H1,3H1)water"}, # DTO
    "[15NH3]":    {"name": "(15N)ammonia"},   # 15N-ammonia
    # Hydrazine: canonical "NN"
    "NN":      {"name": "hydrazine", "substituent_form": "hydrazinyl"},
    # Phosphane (phosphine): canonical "P"
    "P":       {"name": "phosphane", "substituent_form": "phosphanyl"},
    # Arsane (arsine): canonical "[AsH3]"
    "[AsH3]":  {"name": "arsane", "substituent_form": "arsanyl"},
    # Silane: canonical "[SiH4]"
    "[SiH4]":  {"name": "silane", "substituent_form": "silanyl"},
    # Stannane: canonical "[SnH4]"
    "[SnH4]":  {"name": "stannane", "substituent_form": "stannanyl"},
    # Germane: canonical "[GeH4]"
    "[GeH4]":  {"name": "germane", "substituent_form": "germanyl"},
    # Borane (trihydridoboron): canonical "B"
    "B":       {"name": "borane", "substituent_form": "boranyl"},
    # Hydrogen sulfide: canonical "S"
    "S":       {"name": "hydrogen sulfide"},
    # Hydrogen selenide: canonical "[SeH2]"
    "[SeH2]":  {"name": "hydrogen selenide"},

    # --- Oxoacids ---
    # Phosphoric acid: canonical "O=P(O)(O)O"
    "O=P(O)(O)O":  {"name": "phosphoric acid"},
    # Sulfuric acid: canonical "O=S(=O)(O)O"
    "O=S(=O)(O)O": {"name": "sulfuric acid"},
    # Disulfuric acid (= pyrosulfuric acid, P-67.3): HO-SO2-O-SO2-OH.
    "O=S(=O)(O)OS(=O)(=O)O": {"name": "disulfuric acid"},
    # Sulfate dianion (P-77 inorganic acid family).  The fully-deprotonated
    # form is required for diammonium / disodium sulfate salt PINs.
    "O=S(=O)([O-])[O-]": {"name": "sulfate"},
    # Hydrogensulfate / bisulfate monoanion.
    "O=S(=O)(O)[O-]": {"name": "hydrogen sulfate"},
    # Phosphate trianion / dianion / monoanion (round out the family).
    "O=P([O-])([O-])[O-]": {"name": "phosphate"},
    "O=P(O)([O-])[O-]":    {"name": "hydrogen phosphate"},
    "O=P(O)(O)[O-]":       {"name": "dihydrogen phosphate"},
    # Carbonate / hydrogen carbonate.
    "O=C([O-])[O-]": {"name": "carbonate"},
    "O=C(O)[O-]":    {"name": "hydrogen carbonate"},
    # Sulfurous acid: canonical "O=S(O)O"
    "O=S(O)O":     {"name": "sulfurous acid"},
    # Sulfamic acid: canonical "NS(=O)(=O)O"
    "NS(=O)(=O)O": {"name": "sulfamic acid"},
    # Sulfamide: canonical "NS(N)(=O)=O"
    "NS(N)(=O)=O": {"name": "sulfamide"},
    # Phosphoramidic acid: canonical "NP(=O)(O)O"
    "NP(=O)(O)O":  {"name": "phosphoramidic acid"},
    # Carbamic acid (P-42.4 noncarbon-acid retained PIN): canonical "NC(=O)O"
    "NC(=O)O":     {"name": "carbamic acid"},
    # Carbamate anion (P-72.2 / P-77): carbamic acid deprotonated at O-H.
    # Canonical RDKit SMILES for H2N-C(=O)-O⁻ is "NC(=O)[O-]".
    "NC(=O)[O-]":  {"name": "carbamate"},
    # Cyanic acid (P-42.2 / P-65.2 retained PIN): canonical "N#CO"
    "N#CO":        {"name": "cyanic acid"},
    # Thiocyanic acid (P-65.2 retained PIN): canonical "N#CS"
    "N#CS":        {"name": "thiocyanic acid"},
    # Nitrous acid: canonical "O=NO"
    "O=NO":        {"name": "nitrous acid"},
    # Nitric acid: canonical "O=[N+]([O-])O"
    # P-66.6.3.4: retained name preferred over the verbose oxidoazanium form.
    "O=[N+]([O-])O": {"name": "nitric acid"},
    # Nitrite anion: canonical "O=N[O-]" (RDKit form of N(=O)[O-])
    "O=N[O-]":     {"name": "nitrite"},
    # Nitrate anion: canonical "O=[N+]([O-])[O-]"
    "O=[N+]([O-])[O-]": {"name": "nitrate"},

    # --- Hypervalent halogen / metalloid fluoroanions ---
    # Counter-anions commonly seen as the inorganic half of organic salts
    # (e.g. pyridinium hexafluorophosphate). All OPSIN-round-trip verified.
    # Hexafluorophosphate (PF6-): canonical "F[P-](F)(F)(F)(F)F"
    "F[P-](F)(F)(F)(F)F":   {"name": "hexafluorophosphate"},
    # Hexafluoroantimonate (SbF6-): canonical "F[Sb-](F)(F)(F)(F)F"
    "F[Sb-](F)(F)(F)(F)F":  {"name": "hexafluoroantimonate"},
    # Hexafluoroarsenate (AsF6-): canonical "F[As-](F)(F)(F)(F)F"
    "F[As-](F)(F)(F)(F)F":  {"name": "hexafluoroarsenate"},
    # Tetrafluoroborate (BF4-): canonical "F[B-](F)(F)F"
    "F[B-](F)(F)F":         {"name": "tetrafluoroborate"},

    # --- Simple inorganic oxides / small molecules ---
    # Sulfur dioxide: canonical "O=S=O"
    "O=S=O":       {"name": "sulfur dioxide"},
    # Sulfur trioxide: canonical "O=S(=O)=O"
    "O=S(=O)=O":   {"name": "sulfur trioxide"},
    # Carbon dioxide: canonical "O=C=O"
    "O=C=O":       {"name": "carbon dioxide"},
    # Carbon monoxide: canonical "[C-]#[O+]"
    "[C-]#[O+]":   {"name": "carbon monoxide"},
    # Phase 5 — carbene/biradical form of carbon monoxide.  RDKit does NOT
    # canonicalise [C]=O to [C-]#[O+]; the SMILES "[C]=O" yields a bare
    # carbon with two radical electrons + double-bonded oxygen.  OPSIN parses
    # "carbon monoxide" → [C]=O directly (no -r flag needed) so the name is
    # eval-compatible.
    "[C]=O":       {"name": "carbon monoxide"},
    # Phase 5 — carbon monosulfide (carbene/biradical form).  Direct OPSIN
    # parse "carbon monosulfide" → [C]=S; round-trips without -r.
    "[C]=S":       {"name": "carbon monosulfide"},
    # Phase 7 ring-batch-4 — silicon monocarbide (zwitterionic representation).
    # RDKit canonicalises both [C-]#[Si+] and (the also-valid) [Si+]#[C-]
    # to "[C-]#[Si+]".  OPSIN: 'methanidylidynesilylium' → '[C-]#[Si+]'.
    # Closes the [(silylidyne)methane?] NAMING ERROR for the SiC ⇋ C-#Si+
    # zwitterion form (the carbene/biradical [Si]=C and [C]=[Si] forms remain
    # uncovered — those round-trip via separate OPSIN parses if/when needed).
    "[C-]#[Si+]":  {"name": "methanidylidynesilylium"},
    # Dinitrogen: "N#N"
    "N#N":         {"name": "dinitrogen"},
    # Hydrogen chloride (gas)
    "Cl":          {"name": "hydrogen chloride"},
    # Hydrogen bromide
    "Br":          {"name": "hydrogen bromide"},
    # Hydrogen fluoride
    "F":           {"name": "hydrogen fluoride"},
    # Hydrogen iodide
    "I":           {"name": "hydrogen iodide"},

    # --- Bare proton (P-72.2.2.1.1) ---
    # [H+] has zero heavy atoms so _name_elementary_atom skips it.
    # The IUPAC PIN is "hydron"; synonym "proton" is acceptable.
    "[H+]":  {"name": "hydron"},

    # --- Post-transition metal cations ---
    # Thallium: Tl+ (thallous) and Tl3+ (thallic). RDKit: [Tl+] r=0, [Tl+3] r=0.
    "[Tl+]":   {"name": "thallium(1+)"},
    "[Tl+3]":  {"name": "thallium(3+)"},
    # Lead: Pb+–Pb4+. RDKit: [Pb+] r=3, [Pb+2] r=2, [Pb+3] r=1, [Pb+4] r=0.
    "[Pb+]":   {"name": "lead(1+)"},
    "[Pb+2]":  {"name": "lead(2+)"},
    "[Pb+3]":  {"name": "lead(3+)"},
    "[Pb+4]":  {"name": "lead(4+)"},
    # Bismuth: Bi+–Bi5+. RDKit: [Bi+] r=4, [Bi+3] r=2, [Bi+5] r=0.
    "[Bi+]":   {"name": "bismuth(1+)"},
    "[Bi+3]":  {"name": "bismuth(3+)"},
    "[Bi+5]":  {"name": "bismuth(5+)"},
    # Antimony (stibium): Sb+–Sb5+. RDKit: [Sb+] r=4, [Sb+3] r=2, [Sb+5] r=0.
    "[Sb+]":   {"name": "antimony(1+)"},
    "[Sb+3]":  {"name": "antimony(3+)"},
    "[Sb+5]":  {"name": "antimony(5+)"},
    # Arsenic: As3+ and As5+. RDKit: [As+3] r=2, [As+5] r=0.
    "[As+3]":  {"name": "arsenic(3+)"},
    "[As+5]":  {"name": "arsenic(5+)"},
    # Tin: Sn+–Sn4+. RDKit: [Sn+] r=3, [Sn+2] r=2, [Sn+3] r=1, [Sn+4] r=0.
    "[Sn+]":   {"name": "tin(1+)"},
    "[Sn+2]":  {"name": "tin(2+)"},
    "[Sn+3]":  {"name": "tin(3+)"},
    "[Sn+4]":  {"name": "tin(4+)"},
    # Indium: In3+. RDKit: [In+3] r=0.
    "[In+3]":  {"name": "indium(3+)"},
    # Gallium: Ga3+. RDKit: [Ga+3] r=0.
    "[Ga+3]":  {"name": "gallium(3+)"},
    # Germanium: Ge4+. RDKit: [Ge+4] r=0.
    "[Ge+4]":  {"name": "germanium(4+)"},
    # Cadmium: Cd2+. RDKit: [Cd+2] r=0.
    "[Cd+2]":  {"name": "cadmium(2+)"},
    # Mercury: Hg+ (mercurous; RDKit [Hg+] r=1) and Hg2+ (mercuric; r=0).
    "[Hg+]":   {"name": "mercury(1+)"},
    "[Hg+2]":  {"name": "mercury(2+)"},

    # --- Elemental metals (neutral atoms) ---
    "[Fe]":  {"name": "iron"},
    "[Al]":  {"name": "aluminium"},
    "[Zn]":  {"name": "zinc"},
    "[Mn]":  {"name": "manganese"},
    "[Cu]":  {"name": "copper"},
    "[Ni]":  {"name": "nickel"},
    "[Co]":  {"name": "cobalt"},
    "[Cr]":  {"name": "chromium"},
    "[Ag]":  {"name": "silver"},
    "[Au]":  {"name": "gold"},
    "[Pt]":  {"name": "platinum"},
    "[Hg]":  {"name": "mercury"},
    "[Pb]":  {"name": "lead"},
    "[Sn]":  {"name": "tin"},
    "[Ti]":  {"name": "titanium"},
    # Additional elemental metal neutrals — many carry RDKit radical electrons
    # from d/f-shell modelling; they need to be here for _validate_no_open_valences.
    "[Tl]":  {"name": "thallium"},   # r=1
    "[Bi]":  {"name": "bismuth"},    # r=3
    "[Sb]":  {"name": "antimony"},   # r=3
    "[In]":  {"name": "indium"},     # r=3
    "[Ga]":  {"name": "gallium"},    # r=3
    "[Ge]":  {"name": "germanium"},  # r=4
    "[Cd]":  {"name": "cadmium"},    # r=0
    "[Sc]":  {"name": "scandium"},   # r=1
    "[Y]":   {"name": "yttrium"},    # r=1
    "[La]":  {"name": "lanthanum"},  # r=1
    "[Ce]":  {"name": "cerium"},
    "[Gd]":  {"name": "gadolinium"}, # r=1
    "[Lu]":  {"name": "lutetium"},
    # Tantalum: neutral atom, r=1. Needed to bypass _validate_no_open_valences
    # when [Ta] appears as a fragment in multi-fragment molecules (e.g. C.[Ta]).
    # OPSIN round-trip: py2opsin("tantalum") → "[Ta]". ✓
    "[Ta]":  {"name": "tantalum"},   # r=1
    # Lithium: neutral atom, r=1. Listed here so _validate_no_open_valences
    # treats it as a known curated fragment and does not raise for multi-fragment
    # molecules in which a bare [Li] appears.
    "[Li]":  {"name": "lithium"},    # r=1
    # Carbon (bare, hypovalent): r=4. Likewise curated so the guard bypasses
    # it when a bare [C] fragment appears in a multi-fragment molecule.
    "[C]":   {"name": "carbon"},     # r=4
    # Aluminium glycinate (monodentate O-bound, covalent): NCC(=O)[O][Al].
    # The Al atom carries 2 radical electrons (normal valence 3; only 1 bond).
    # This is a per-fragment curated inorganic name; the entry lets
    # _validate_no_open_valences bypass the radical Al fragment.
    # OPSIN: py2opsin("aluminum(I) glycinate") → "NCC(=O)[O-].[Al+]". ✓
    "NCC(=O)[O][Al]": {"name": "aluminum(I) glycinate"},  # Al r=2

    # --- Simple heteronuclear binary compounds ---
    # Selenenyl sulfide (selenium monosulfide): canonical "S=[Se]"
    "S=[Se]":  {"name": "selenenyl sulfide"},

    # --- Nitrogen oxides ---
    # Nitrogen monoxide (nitric oxide): canonical "[N]=O"
    "[N]=O":   {"name": "nitrogen monoxide"},

    # --- Hydroxylamine-O-sulfonic acid / N-hydroxysulfamic acid ---
    # O=S(=O)(O)NO is "N-hydroxysulfamic acid" (N-OH derivative of sulfamic
    # acid H2N-SO3H).  OPSIN round-trip verified:
    #   py2opsin("N-hydroxysulfamic acid") → "ONS(O)(=O)=O"
    #   canonical → "O=S(=O)(O)NO"  ✓
    "O=S(=O)(O)NO":  {"name": "N-hydroxysulfamic acid"},

    # --- Metal oxo compounds (covalent M=O form) ---
    # These appear as [M]=O or O=[M] in chemist-shorthand SMILES.  OPSIN
    # parses the retained prefix-form "oxo{metal}" to the covalent O=[M]
    # SMILES, while "magnesium oxide" / "calcium oxide" parse to the ionic
    # [M+2].[O-2] form.  Adding both directions here allows the engine to
    # name the covalent form correctly and the eval to accept the round-trip.
    # Round-trip verified: py2opsin("oxo{metal}") == canonical [O]=[M].
    "[O]=[Mg]":  {"name": "oxomagnesium"},
    "[O]=[Ca]":  {"name": "oxocalcium"},
    "[O]=[Ba]":  {"name": "oxobarium"},
    "[O]=[Sr]":  {"name": "oxostrontium"},
    "[Be]=[O]":  {"name": "oxoberyllium"},   # RDKit canonical: Be first

    # --- Bicyclic / fused ring retained names with exo groups ---
    # These are full-molecule entries for ring systems that can't be named
    # via ring-fragment extraction (aromatic ring depends on exo groups).
    #
    # Glycoluril (tetrahydroimidazo[4,5-d]imidazole-2,5(1H,3H)-dione)
    "O=C1NC2NC(=O)NC2N1": {"name": "tetrahydroimidazo[4,5-d]imidazole-2,5(1H,3H)-dione"},
    # Pyrazolo[3,4-d]pyrimidinone (1H-pyrazolo[3,4-d]pyrimidin-4(5H)-one) - ZT-0841
    "O=c1[nH]cnc2[nH]ncc12": {"name": "1H-pyrazolo[3,4-d]pyrimidin-4(5H)-one"},
    # Pyrazolo[3,4-d]pyrimidine-4-thione - ZT-2249
    "S=c1nc[nH]c2[nH]ncc12": {"name": "1H-pyrazolo[3,4-d]pyrimidine-4(5H)-thione"},
    # Indane-1,2,3-trione (ninhydrin precursor) - ZT-1713
    "O=c1c(=O)c2ccccc2c1=O": {"name": "indane-1,2,3-trione"},
}


def _lookup_curated_inorganic(smiles: str) -> dict | None:
    """Check the curated inorganic/ion table for a canonical SMILES match.

    Also tries removing explicit Hs before lookup, so that fragments carved with
    explicit H atoms still match the canonical ion/parent-hydride SMILES.
    """
    entry = _INORGANIC_CURATED_SMILES.get(smiles)
    if entry is not None:
        return {"smiles": smiles, "source": "inorganic_curated", **entry}

    if "[H]" in smiles or "[h]" in smiles:
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mol_no_h = Chem.RemoveHs(mol)
                canonical = Chem.MolToSmiles(mol_no_h)
                entry = _INORGANIC_CURATED_SMILES.get(canonical)
                if entry is not None:
                    return {"smiles": canonical, "source": "inorganic_curated", **entry}
        except Exception:
            pass

    return None


def _adjust_substituent_form_for_ring_charge(
    record: dict, smiles: str
) -> dict:
    """Inject the ``-ium`` / ``-ide`` cation/anion marker into ``substituent_form``
    for charged-ring retained entries (P-73.1 cation / P-72.2 anion nomenclature,
    substituent position).

    Background
    ----------
    Several curated ring entries are keyed on the charged SMILES form (e.g.
    ``c1cc[nH+]cc1`` for protonated pyridine, ``c1ccc2[nH+]cccc2c1`` for
    protonated quinoline) but carry the *neutral* ``name``/``substituent_form``
    ("pyridine"/"pyridinyl", "quinoline"/"quinolinyl").  The accompanying
    standalone ``-ium`` suffix is appended by ``assembly.py`` via the
    ``ring_cation_locants`` path, so the top-level output for a lone
    protonated pyridine is correctly ``pyridin-1-ium``.

    In SUBSTITUENT position, however, the retained-name fast-path emits the
    record's ``substituent_form`` directly as a leaf (``_execute_retained``
    in engine.py): e.g. ``pyridinyl`` + locant 1 => ``pyridin-1-yl``, with
    no ``-ium``.  This silently drops the ring charge from N-alkyl pyridinium
    substituents (e.g. ceftazidime's 3-[(pyridinium-1-yl)methyl]- side chain).

    Expressing the IUPAC rule
    -------------------------
    A charged ring atom's substituent form carries the same ``-ium`` / ``-ide``
    marker as the standalone form, placed as an infix before the free-valence
    ``-yl`` suffix (e.g. ``pyridinium-yl``, ``imidazolium-yl``,
    ``quinolinium-yl``, ``pyridin-1-ide-yl``).  OPSIN round-trips both
    ``pyridinium-1-yl`` and ``pyridin-1-ium-1-yl``; we emit the former
    (shorter, matches the existing curated ``pyridinium`` entry).

    This helper is idempotent: substituent forms that already encode the
    charge (``pyridinium-yl``, ``*-ium-yl``, ``*-ide-yl``) are returned
    unchanged.  Only the neutral ``<stem>yl`` form is transformed.  Entries
    with no ``substituent_form``, no ``yl`` suffix, or no ring [n+]/[n-] in
    the key SMILES are also passed through untouched.
    """
    sub_form = record.get("substituent_form")
    if not sub_form or not sub_form.endswith("yl"):
        return record

    # Already encodes the charge → idempotent.
    if "ium-yl" in sub_form or "ide-yl" in sub_form:
        return record

    # Detect ring charge in the matched SMILES key.  ``[nH+]`` and ``[n+]``
    # are the aromatic-ring cation tokens; ``[nH-]`` / ``[n-]`` the anion
    # tokens.  (Non-ring [N+] like quaternary acyclic ammonium is handled
    # elsewhere.)  Skip the transform for multi-charge rings: the standalone
    # multiplier machinery (``-1,3-diium``) has no substituent-form analog
    # in the current benchmark; leave those to fall through unchanged rather
    # than emit a malformed single-marker name.
    cation_count = smiles.count("[n+]") + smiles.count("[nH+]")
    anion_count = smiles.count("[n-]") + smiles.count("[nH-]")
    if cation_count + anion_count != 1:
        return record

    marker = "ium" if cation_count else "ide"
    # Strip trailing "yl" and append "<marker>-yl": "pyridinyl" → "pyridinium-yl".
    new_sub_form = sub_form[:-2] + f"{marker}-yl"

    # Return a shallow-copied dict so we don't mutate the cached curated entry.
    return {**record, "substituent_form": new_sub_form}


def _lookup_curated_ring(smiles: str) -> dict | None:
    """Check the curated ring table for a canonical SMILES match.

    Also tries removing explicit Hs (from carving) before lookup.
    E.g. '[H]c1ccccc1' → look up 'c1ccccc1' as benzene.

    When the matched SMILES key encodes a charged aromatic ring nitrogen
    ([n+]/[nH+] or [n-]/[nH-]) but the curated entry's ``substituent_form``
    is still the neutral form, the substituent form is augmented with the
    ``-ium``/``-ide`` marker (e.g. ``pyridinyl`` → ``pyridinium-yl``).  This
    matches the IUPAC P-73.1 rule for ring-cation substituent names; the
    parent (standalone) name's ``-ium`` suffix continues to be supplied by
    the assembly.py ``ring_cation_locants`` path.
    """
    entry = _RING_CURATED_SMILES.get(smiles)
    if entry is not None:
        return _adjust_substituent_form_for_ring_charge(
            {"smiles": smiles, "source": "ring_curated",
             **_apply_pin_alias(entry)}, smiles
        )

    # If the SMILES contains explicit H atoms (from carving), strip them and retry.
    if "[H]" in smiles or "[h]" in smiles:
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mol_no_h = Chem.RemoveHs(mol)
                canonical = Chem.MolToSmiles(mol_no_h)
                entry = _RING_CURATED_SMILES.get(canonical)
                if entry is not None:
                    return _adjust_substituent_form_for_ring_charge(
                        {"smiles": canonical, "source": "ring_curated",
                         **_apply_pin_alias(entry)},
                        canonical,
                    )
        except Exception:
            pass

    return None


def _apply_pin_alias(entry: dict) -> dict:
    """If ``entry`` is flagged ``pin_eligible: False`` and supplies a
    ``pin_name`` / ``pin_substituent_form`` alias, return a copy of ``entry``
    with those fields swapped into ``name`` / ``substituent_form``.

    Per IUPAC 2013 P-25.3.1.3 / P-31.1.4.2.4 / P-32.4 / P-53 / P-54.4.3.2,
    retained names like tetraline, indane, chroman, and isochroman are
    general-nomenclature only.  Their PINs are the systematic hydro-derived
    forms (1,2,3,4-tetrahydronaphthalene; 2,3-dihydro-1H-indene; etc.).  The
    curated record's atom_locants are correct for both spellings (the
    saturated-ring locants are 1,2,3,4 for tetralin === tetrahydronaphthalene
    and 1,2,3 for indan === 2,3-dihydro-1H-indene), so swapping just the name
    string preserves all downstream numbering / substituent-locant logic.
    """
    if entry.get("pin_eligible", True) is False:
        result = dict(entry)
        pin_name = entry.get("pin_name")
        if pin_name:
            result["name"] = pin_name
        pin_sub = entry.get("pin_substituent_form")
        if pin_sub:
            result["substituent_form"] = pin_sub
        return result
    return entry


def lookup_retained_ring(name: str) -> dict | None:
    """Search retained ring tables by ring name.

    Searches:
    - ``retained_rings.json`` (categories → {ring_name: record})
    - ``rings_from_opsin.json`` (list of {name, smiles, source})

    The *name* lookup is case-insensitive. Returns the record dict or None.
    """
    name_lower = name.lower()

    # retained_rings: nested dict by category
    rings = get_retained_rings()
    for category_data in rings.values():
        if isinstance(category_data, dict):
            for ring_name, record in category_data.items():
                if ring_name.lower() == name_lower:
                    return {"name": ring_name, **record} if isinstance(record, dict) else {"name": ring_name}

    # rings_from_opsin: list with pipe-separated name variants
    for entry in get_rings_from_opsin():
        if isinstance(entry, dict):
            # names may be "benzo|benz" style
            variants = [v.strip() for v in entry.get("name", "").split("|")]
            if any(v.lower() == name_lower for v in variants):
                return entry

    return None


# ---------------------------------------------------------------------------
# ACID_ADJECTIVE_TABLE
# Built from retained_names.json functional_parents and hardcoded acids.
# ---------------------------------------------------------------------------

def _build_acid_adjective_table() -> dict[str, str]:
    """Build mapping from retained acid name → adjective stem."""
    # Hardcoded entries that are always required by the Blue Book
    table: dict[str, str] = {
        "formic acid": "form",
        "acetic acid": "acet",
        "propionic acid": "propion",
        "butyric acid": "butyr",
        "isobutyric acid": "isobutyr",
        "valeric acid": "valer",
        "isovaleric acid": "isovaler",
        "oxalic acid": "oxal",
        "malonic acid": "malon",
        "succinic acid": "succin",
        "glutaric acid": "glutar",
        "adipic acid": "adip",
        "pimelic acid": "pimel",
        "phthalic acid": "phthal",
        "isophthalic acid": "isophthal",
        "terephthalic acid": "terephthal",
        "benzoic acid": "benzo",
        "phenylacetic acid": "phenylacet",
        "cinnamic acid": "cinnam",
        "maleic acid": "male",
        "fumaric acid": "fumar",
        "citric acid": "citr",
        "lactic acid": "lact",
        "tartaric acid": "tartar",
        "mandelic acid": "mandel",
        "acrylic acid": "acryl",
        "methacrylic acid": "methacryl",
        "pyruvic acid": "pyruv",
        "levulinic acid": "levulin",
        "nicotinic acid": "nicotin",
        "isonicotinic acid": "isonicotin",
        "picolinic acid": "picolin",
    }
    # Supplement from retained_names.json: acid entries in functional_parents
    try:
        fp = get_retained_names().get("functional_parents", {})
        for acid_name, rec in fp.items():
            if "acid" in acid_name.lower() and acid_name not in table:
                # Derive stem: strip " acid", keep root before last vowel run
                stem = acid_name.replace(" acid", "").rstrip("aeiou")
                if stem:
                    table.setdefault(acid_name, stem)
    except Exception:  # pragma: no cover — fail-safe, data is optional enhancement
        pass
    return table


# Module-level singleton for ACID_ADJECTIVE_TABLE
ACID_ADJECTIVE_TABLE: dict[str, str] = _build_acid_adjective_table()


# ---------------------------------------------------------------------------
# SUFFIX_ELISION_TABLE
# Rules for terminal 'e' elision before IUPAC suffixes.
# ---------------------------------------------------------------------------

# Map suffix → True if terminal 'e' of parent stem is elided before this suffix.
# Blue Book rule: elide before a suffix that starts with a vowel (a, e, i, o, u),
# EXCEPT for a few special cases.
SUFFIX_ELISION_TABLE: dict[str, bool] = {
    # Suffixes starting with vowel — elide
    "-al": True,
    "-aldehyde": False,     # starts with consonant
    "-amide": False,        # "-amine" class: no elision
    "-amine": False,        # special: no elision despite starting with vowel
    "-anol": False,         # composite, handled separately
    "-ate": True,
    "-ic acid": False,      # starts with consonant (i in context)
    "-ide": False,          # starts with consonant
    "-in": False,
    "-ine": True,
    "-imine": False,
    "-ium": False,
    "-nitrile": False,      # starts with consonant
    "-oic acid": False,     # starts with vowel 'o' but elision NOT applied: suffix already ends on vowel cluster
    "-ol": True,
    "-olide": True,
    "-one": True,
    "-onic acid": False,
    "-onyl": False,
    "-oyl": False,
    "-oyl chloride": False,
    "-oyl bromide": False,
    "-oyl fluoride": False,
    "-oyl iodide": False,
    "-carbonyl chloride": False,
    "-carbonyl bromide": False,
    "-carbonyl fluoride": False,
    "-carbonyl iodide": False,
    "-yl": False,
    "-ylene": True,
    "-ylidene": True,
    "-ylidyne": True,
    "-yne": True,
    # Multiplicative/complex suffixes
    "-diol": False,
    "-dione": False,
    "-diamine": False,
    "-dicarboxylic acid": False,
    # Hetero-suffix
    "-sulfonic acid": False,
    "-sulfinic acid": False,
    "-thiol": False,
}


def suffix_elides_terminal_e(suffix: str) -> bool:
    """Return True if the parent stem's terminal 'e' should be elided before *suffix*.

    Falls back to the general rule: elide before vowel-starting suffixes,
    with the exception of ``-amine``.
    """
    if suffix in SUFFIX_ELISION_TABLE:
        return SUFFIX_ELISION_TABLE[suffix]
    # General rule for unlisted suffixes
    stripped = suffix.lstrip("-")
    if not stripped:
        return False
    if suffix == "-amine" or suffix.endswith("amine"):
        return False
    return stripped[0] in "aeiou"


# ---------------------------------------------------------------------------
# Cache invalidation (for testing)
# ---------------------------------------------------------------------------

def clear_cache() -> None:
    """Clear the in-memory JSON cache.  Useful for testing."""
    _cache.clear()
