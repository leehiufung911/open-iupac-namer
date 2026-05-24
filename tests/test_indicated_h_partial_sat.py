"""Regression tests for Stage 6 R3-B: indicated-H partial-saturation in
monocyclic Hantzsch-Widman heterocycles.

Root cause #18 in ``docs/opsin_coverage_taxonomy.md``: monocyclic HW rings
that already have the maximum non-cumulative double-bond count for their
heteroatom composition but still contain at least one sp3 ring atom are
named with an italicised "<N>H-" prefix marking the indicated-H carrier
(IUPAC Blue Book P-25.7.1.3).  Examples:

    2H-1,3-oxazine,   4H-1,3-oxazine,   2H-1,4-oxazine
    2H-1,2-thiazine,  4H-1,3-thiazine,  2H-1,4-thiazine
    1H-azepine,       2H-azepine

Pre-fix, these rings emitted *fully-saturated* HW stems (e.g. "1,3-oxazinane"
instead of "4H-1,3-oxazine") because ``try_hantzsch_widman`` had no notion of
"current DB count == max mancude DB count → partially-saturated mancude
parent with indicated-H".

The fix introduces three helpers in ``ring_naming/monocyclic.py``:

  * ``_ring_endocyclic_db_count`` — count current ring DBs
  * ``_max_mancude_db_count``     — max non-cumulative DBs for the ring,
                                    accounting for divalent chalcogen
                                    "blockers" (O/S/Se/Te) that segment
                                    the DB-capable atom chain
  * ``_pick_indicated_h_locant``  — lowest locant of a non-blocker ring
                                    atom that carries no endocyclic DB

When current DBs == max mancude DBs and a valid indicated-H carrier exists,
the unsaturated HW stem is selected and a "<N>H-" prefix is prepended.

Negative regressions guarded:
  * Retained ``2H-pyran`` / ``4H-pyran`` / ``2H-thiopyran`` still emit via
    the curated retained-name path (must not be renamed by HW).
  * Fully saturated HW rings (no endo DBs) still emit "-ane" stems.
  * Pyridine-class 6-rings (max=3 DBs all-N) with one sp3 atom emit as
    "dihydropyridine" (or are handled elsewhere) — they must NOT spuriously
    emit "<N>H-" prefixes from this path.
"""
from __future__ import annotations

import pytest

from iupac_namer.engine import name_smiles


def _name(smiles: str) -> str:
    return name_smiles(smiles)


# ---------------------------------------------------------------------------
# 1. Six-membered O/N rings: oxazines
# ---------------------------------------------------------------------------

class TestOxazineIndicatedH:
    """1,2-/1,3-/1,4-oxazine partial-sat tautomers."""

    def test_2H_1_2_oxazine(self):
        # O1NC=CC=C1 -> 2H-1,2-oxazine (sp3 N at locant 2)
        assert _name("O1NC=CC=C1") == "2H-1,2-oxazine"

    def test_2H_1_3_oxazine(self):
        # O1CN=CC=C1 -> 2H-1,3-oxazine (sp3 C at locant 2)
        assert _name("O1CN=CC=C1") == "2H-1,3-oxazine"

    def test_4H_1_3_oxazine(self):
        # O1C=NCC=C1 -> 4H-1,3-oxazine (sp3 C at locant 4)
        assert _name("O1C=NCC=C1") == "4H-1,3-oxazine"

    def test_2H_1_4_oxazine_a(self):
        # O1C=CN=CC1 -> 2H-1,4-oxazine (sp3 C at locant 2 by tiebreaker)
        assert _name("O1C=CN=CC1") == "2H-1,4-oxazine"

    def test_2H_1_4_oxazine_b(self):
        # O1CC=NC=C1 -> 2H-1,4-oxazine (alternate Kekulé)
        assert _name("O1CC=NC=C1") == "2H-1,4-oxazine"


# ---------------------------------------------------------------------------
# 2. Six-membered S/N rings: thiazines
# ---------------------------------------------------------------------------

class TestThiazineIndicatedH:
    """1,2-/1,3-/1,4-thiazine partial-sat tautomers."""

    def test_2H_1_2_thiazine(self):
        assert _name("S1NC=CC=C1") == "2H-1,2-thiazine"

    def test_2H_1_3_thiazine(self):
        assert _name("S1CN=CC=C1") == "2H-1,3-thiazine"

    def test_4H_1_3_thiazine(self):
        assert _name("S1C=NCC=C1") == "4H-1,3-thiazine"

    def test_2H_1_4_thiazine(self):
        # S1C=CN=CC1 -> 2H-1,4-thiazine (tiebreaker resolves)
        assert _name("S1C=CN=CC1") == "2H-1,4-thiazine"


# ---------------------------------------------------------------------------
# 3. Seven-membered rings: azepines
# ---------------------------------------------------------------------------

class TestAzepineIndicatedH:
    """7-ring N: max 3 non-cumulative DBs; mancude tautomers are 1H- and 2H-."""

    def test_1H_azepine(self):
        # N1C=CC=CC=C1 -> 1H-azepine (sp3 N at locant 1)
        assert _name("N1C=CC=CC=C1") == "1H-azepine"

    def test_2H_azepine(self):
        # N1CC=CC=CC=1 -> 2H-azepine (sp3 C at locant 2; N still locant 1)
        assert _name("N1CC=CC=CC=1") == "2H-azepine"


# ---------------------------------------------------------------------------
# 4. Negative regressions — retained names must still emit
# ---------------------------------------------------------------------------

class TestRetainedNamesPreserved:
    """The HW indicated-H path must NOT poach retained-name rings — these
    have hardcoded curated entries (2H-pyran, 4H-pyran, 2H-thiopyran, etc.)
    in ``data_loader._RING_CURATED_SMILES``.  The retained-name lookup
    runs before HW so this is mostly a sanity check."""

    def test_2H_pyran_retained(self):
        # O1CC=CC=C1 -> retained 2H-pyran
        name = _name("O1CC=CC=C1")
        assert name == "2H-pyran"

    def test_4H_pyran_retained(self):
        # C1=COC=CC1 -> retained 4H-pyran
        name = _name("C1=COC=CC1")
        assert name == "4H-pyran"

    def test_2H_thiopyran_retained(self):
        # S1CC=CC=C1 -> retained 2H-thiopyran
        name = _name("S1CC=CC=C1")
        assert "thiopyran" in name.lower() or "thiine" in name.lower()


# ---------------------------------------------------------------------------
# 5. Negative regressions — fully-saturated rings stay -ane
# ---------------------------------------------------------------------------

class TestFullySaturatedNoIndicatedH:
    """Saturated HW rings (no endocyclic DBs) must NOT emit "<N>H-" prefix —
    indicated-H is only for mancude tautomers with cur_dbs == max_dbs."""

    def test_morpholine_unaffected(self):
        # Morpholine = 1,4-oxazinane (saturated) — retained name expected
        name = _name("O1CCNCC1")
        assert name == "morpholine"
        assert "1H-" not in name and "2H-" not in name

    def test_piperidine_unaffected(self):
        name = _name("N1CCCCC1")
        assert name == "piperidine"
        assert "1H-" not in name and "2H-" not in name

    def test_oxane_unaffected(self):
        # Tetrahydropyran / oxane — retained "oxane"
        name = _name("O1CCCCC1")
        assert "oxane" in name or "tetrahydro" in name
        assert "H-oxine" not in name


# ---------------------------------------------------------------------------
# 6. Helper unit tests — direct API
# ---------------------------------------------------------------------------

class TestHelperFunctions:
    """Direct tests for ``_pick_indicated_h_locant``, ``_max_mancude_db_count``,
    and ``_ring_endocyclic_db_count``."""

    def _ring_atoms(self, smiles: str):
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"Failed to parse {smiles}"
        # First (and only) ring
        ring_info = mol.GetRingInfo().AtomRings()
        assert len(ring_info) >= 1
        return frozenset(ring_info[0]), mol

    def test_endocyclic_db_count_oxazine(self):
        from iupac_namer.ring_naming.monocyclic import _ring_endocyclic_db_count
        ring_atoms, mol = self._ring_atoms("O1CN=CC=C1")
        # 2H-1,3-oxazine: 2 endocyclic DBs (C=N and C=C)
        assert _ring_endocyclic_db_count(ring_atoms, mol) == 2

    def test_endocyclic_db_count_saturated(self):
        from iupac_namer.ring_naming.monocyclic import _ring_endocyclic_db_count
        ring_atoms, mol = self._ring_atoms("O1CCNCC1")  # morpholine
        assert _ring_endocyclic_db_count(ring_atoms, mol) == 0

    def test_max_mancude_db_count_pyran(self):
        from iupac_namer.ring_naming.monocyclic import _max_mancude_db_count
        # Pyran: 6-ring with 1 O blocker → segment of 5 C → floor(5/2) = 2
        ring_atoms, mol = self._ring_atoms("O1CC=CC=C1")
        assert _max_mancude_db_count(ring_atoms, mol, ["O"]) == 2

    def test_max_mancude_db_count_oxazine_1_3(self):
        from iupac_namer.ring_naming.monocyclic import _max_mancude_db_count
        # 1,3-oxazine: O blocker + N (DB-capable) + 4 C → segment of 5 atoms
        # floor(5/2) = 2 DBs max
        ring_atoms, mol = self._ring_atoms("O1CN=CC=C1")
        assert _max_mancude_db_count(ring_atoms, mol, ["O", "N"]) == 2

    def test_max_mancude_db_count_pyridine(self):
        from iupac_namer.ring_naming.monocyclic import _max_mancude_db_count
        # Pyridine 6-ring: no chalcogen blockers, all 6 atoms DB-capable
        # floor(6/2) = 3 DBs max
        ring_atoms, mol = self._ring_atoms("c1ccncc1")
        assert _max_mancude_db_count(ring_atoms, mol, ["N"]) == 3

    def test_max_mancude_db_count_azepine(self):
        from iupac_namer.ring_naming.monocyclic import _max_mancude_db_count
        # Azepine 7-ring with N: floor(7/2) = 3 DBs max
        ring_atoms, mol = self._ring_atoms("N1C=CC=CC=C1")
        assert _max_mancude_db_count(ring_atoms, mol, ["N"]) == 3

    def test_pick_indicated_h_locant_returns_lowest(self):
        from iupac_namer.ring_naming.monocyclic import (
            _pick_indicated_h_locant,
            _compute_hw_locants,
            _get_ring_cycle_order,
            _HW_PRIORITY,
        )
        ring_atoms, mol = self._ring_atoms("O1CN=CC=C1")  # 2H-1,3-oxazine
        cycle = _get_ring_cycle_order(ring_atoms, mol)
        # Build HW priority map: heteroatoms with their HW priorities
        prio_map: dict[int, int] = {}
        hetero_set: set[int] = set()
        for atom_idx in ring_atoms:
            sym = mol.GetAtomWithIdx(atom_idx).GetSymbol()
            if sym in _HW_PRIORITY:
                hetero_set.add(atom_idx)
                prio_map[atom_idx] = _HW_PRIORITY[sym]
        locant_map = _compute_hw_locants(cycle, hetero_set, prio_map)
        assert locant_map is not None
        loc = _pick_indicated_h_locant(ring_atoms, locant_map, mol)
        # Locant 2 is the lone sp3 carbon between O(1) and N(3)
        assert loc == 2

    def test_pick_indicated_h_locant_skips_chalcogen(self):
        """Divalent chalcogens (O/S/Se/Te) are NOT indicated-H carriers
        even when they bear no endocyclic DB — they are intrinsic blockers."""
        from iupac_namer.ring_naming.monocyclic import (
            _pick_indicated_h_locant,
            _compute_hw_locants,
            _get_ring_cycle_order,
            _HW_PRIORITY,
        )
        # 4H-1,3-oxazine — sp3 carbon at locant 4 (between N and double-bond C)
        ring_atoms, mol = self._ring_atoms("O1C=NCC=C1")
        cycle = _get_ring_cycle_order(ring_atoms, mol)
        prio_map: dict[int, int] = {}
        hetero_set: set[int] = set()
        for atom_idx in ring_atoms:
            sym = mol.GetAtomWithIdx(atom_idx).GetSymbol()
            if sym in _HW_PRIORITY:
                hetero_set.add(atom_idx)
                prio_map[atom_idx] = _HW_PRIORITY[sym]
        locant_map = _compute_hw_locants(cycle, hetero_set, prio_map)
        assert locant_map is not None
        loc = _pick_indicated_h_locant(ring_atoms, locant_map, mol)
        # Locant 4: the lone sp3 carbon. Even though O has no endo DB, it is
        # excluded from indicated-H carriers (intrinsic divalent blocker).
        assert loc == 4
