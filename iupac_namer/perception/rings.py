"""
iupac_namer/perception/rings.py

RingAnalysis subsystem — Subsystem 4 of the Perception layer.

Detects ring systems in an RDKit mol, groups them, classifies each as
monocyclic / fused / bridged / spiro, and computes structural descriptors
(bridge sizes, spiro sizes, fusion info, heteroatom positions).

This module produces ONLY structural descriptors.  It does NOT name rings —
ring naming happens later in the ring naming module during plan generation.

See ARCHITECTURE_PERCEPTION.md §Subsystem 4 for the full spec.
"""

from __future__ import annotations

import logging
from collections import Counter, deque
from typing import TYPE_CHECKING

from iupac_namer.types import FusionInfo, HeteroPosition, RingSystem

if TYPE_CHECKING:
    from iupac_namer.perception.atoms import AtomAnalysis

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RingAnalysis
# ---------------------------------------------------------------------------


class RingAnalysis:
    """Ring system analysis for an RDKit molecule.

    Groups rings into ring systems (connected components sharing atoms),
    classifies each as monocyclic / fused / bridged / spiro, and computes
    structural descriptors required by the naming engine.

    Parameters
    ----------
    mol:
        An RDKit ``Mol`` object (after sanitisation).
    atom_analysis:
        The :class:`~iupac_namer.perception.atoms.AtomAnalysis` for this mol.
    """

    def __init__(self, mol: object, atom_analysis: "AtomAnalysis") -> None:
        self._mol = mol
        self._atoms = atom_analysis
        self._ring_systems: tuple[RingSystem, ...] = self._analyze()

    # ------------------------------------------------------------------
    # Main analysis pipeline
    # ------------------------------------------------------------------

    def _analyze(self) -> tuple[RingSystem, ...]:
        """Build a RingSystem for every connected ring system in the molecule."""
        ri = self._mol.GetRingInfo()  # type: ignore[attr-defined]
        if ri.NumRings() == 0:
            return ()

        # Get SSSR rings as frozensets of atom indices
        raw_rings: list[frozenset[int]] = [
            frozenset(ring) for ring in ri.AtomRings()
        ]

        # Group rings sharing atoms into ring systems
        system_groups = self._group_rings_into_systems(raw_rings)

        result: list[RingSystem] = []
        for group_rings in system_groups:
            all_atoms = frozenset().union(*group_rings)
            rs = self._build_ring_system(group_rings, all_atoms)
            result.append(rs)

        return tuple(result)

    # ------------------------------------------------------------------
    # Ring grouping
    # ------------------------------------------------------------------

    def _group_rings_into_systems(
        self, rings: list[frozenset[int]]
    ) -> list[list[frozenset[int]]]:
        """Group rings sharing atoms into ring systems using union-find.

        Two rings are in the same system if they share at least one atom.
        Returns a list of groups, each group being a list of ring atom-sets.
        """
        n = len(rings)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(n):
            for j in range(i + 1, n):
                if rings[i] & rings[j]:  # shared atoms
                    union(i, j)

        groups: dict[int, list[frozenset[int]]] = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(rings[i])

        return list(groups.values())

    # ------------------------------------------------------------------
    # Ring system construction
    # ------------------------------------------------------------------

    def _build_ring_system(
        self, rings: list[frozenset[int]], all_atoms: frozenset[int]
    ) -> RingSystem:
        """Classify a group of rings and construct a RingSystem descriptor."""
        ring_size = len(all_atoms)

        if len(rings) == 1:
            # Monocyclic
            ring = rings[0]
            ordered = self._order_ring_atoms(ring)
            heteroatoms = self._detect_heteroatoms(ordered)
            aromatic = self._is_ring_aromatic(ring)
            return RingSystem(
                atom_indices=all_atoms,
                rings=(ring,),
                type="monocyclic",
                aromatic=aromatic,
                bridge_sizes=None,
                spiro_sizes=None,
                fusion_info=None,
                heteroatoms=heteroatoms,
                ring_size=ring_size,
            )

        # Polycyclic — classify and compute descriptors
        ring_type, ambiguous, alt_type = self._classify_ring_system(rings, all_atoms)
        aromatic = self._is_system_aromatic(all_atoms)
        heteroatoms = self._detect_system_heteroatoms(all_atoms)

        fusion_info: FusionInfo | None = None
        bridge_sizes: tuple[int, ...] | None = None
        spiro_sizes: tuple[int, ...] | None = None
        alt_bridge_sizes: tuple[int, ...] | None = None
        secondary_bridges: tuple[tuple[tuple[int, int], tuple[int, ...]], ...] | None = None

        if ring_type == "fused":
            fusion_info = self._compute_fusion_info(rings)
            if ambiguous:
                alt_bridge_sizes = self._compute_bridge_sizes(rings, all_atoms)
            else:
                # IUPAC P-23.2.5: ortho-fused bicycles containing a 3-membered
                # ring (e.g. cyclohexene oxide → 7-oxabicyclo[4.1.0]heptane,
                # norcarane → bicyclo[4.1.0]heptane) are named via von Baeyer
                # nomenclature with a 0-atom bridge.  Compute the VB alt so the
                # ring naming layer can fall back to bicyclo[N.1.0] when no
                # retained name matches.  Restricted to 2-SSSR-ring systems
                # whose smallest ring is <= 3 atoms to avoid producing VB names
                # for general fused arenes (where retained naming is preferred
                # and a VB fallback would only mask retained-name failures).
                #
                # Macrocycle extension: a 2-ring fused system whose larger ring
                # is a macrocycle (>= 8 atoms) and smaller ring is non-aromatic
                # also gets a VB alternate.  No retained name covers e.g.
                # macrocyclic peptides with a fused proline residue (which would
                # produce a `bicyclo[N.3.0]` von Baeyer name), so without this
                # alternate the molecule fails entirely.  Aromatic-fused
                # macrocycles (annulenes) are excluded so retained-name lookups
                # for benzo-fused arenes are not bypassed.
                if len(rings) == 2 and min(len(r) for r in rings) <= 3:
                    vb_sizes = self._compute_bridge_sizes(rings, all_atoms)
                    if vb_sizes:
                        alt_bridge_sizes = vb_sizes
                        alt_type = "bridged"
                        ambiguous = True
                elif (
                    len(rings) == 2
                    and max(len(r) for r in rings) >= 8
                    and not self._is_system_aromatic(all_atoms)
                ):
                    vb_sizes = self._compute_bridge_sizes(rings, all_atoms)
                    if vb_sizes:
                        alt_bridge_sizes = vb_sizes
                        alt_type = "bridged"
                        ambiguous = True
                elif (
                    len(rings) >= 3
                    and not self._is_system_aromatic(all_atoms)
                ):
                    # Non-aromatic polycyclic-fused systems with no retained name
                    # and no systematic fused-ring name MUST be expressed in von
                    # Baeyer form.  Examples:
                    #   - cyclic lipopeptides with multiple fused proline residues
                    #     in a macrocycle backbone
                    #   - corticosteroid cores with the steroid scaffold ortho-fused
                    #     to a 1,3-dioxolane (FDA-0172 budesonide-like substituent
                    #     after the C20 ketone is carved away — the resulting
                    #     5-ring fused steroid+dioxolane has no retained name and
                    #     no OPSIN-acceptable fused or bis(oxy) substituent form)
                    # The retained-name lookup runs first and out-scores VB at the
                    # strategy layer (1M vs ~k), so offering VB alt here does not
                    # displace decalin / cyclopenta[a]phenanthrene / etc.  It only
                    # provides a fallback when nothing else fits.
                    # Use vb_decompose which handles ortho-fused multi-ring systems.
                    from iupac_namer.ring_naming.vb_decompose import (
                        decompose_ring_system,
                    )
                    decomps = decompose_ring_system(all_atoms, self._mol)
                    if decomps:
                        best = decomps[0]
                        alt_bridge_sizes = best.main_bridge_sizes
                        alt_type = "bridged"
                        ambiguous = True
                else:
                    # IUPAC P-23.2.5: fused polycyclic systems with an atom
                    # shared among 3+ rings (e.g. tricyclo[5.2.1.0^{4,10}]
                    # dec-2-ene) are angularly-fused cages that retained-ring
                    # naming cannot cover — they require von Baeyer
                    # nomenclature with a 0-atom secondary bridge.  Detect
                    # this pattern and expose the VB descriptor as an
                    # alternate type.
                    atom_ring_count: Counter[int] = Counter()
                    for r in rings:
                        for a in r:
                            atom_ring_count[a] += 1
                    if any(c >= 3 for c in atom_ring_count.values()):
                        from iupac_namer.ring_naming.vb_decompose import (
                            decompose_ring_system,
                        )
                        decomps = decompose_ring_system(all_atoms, self._mol)
                        if decomps:
                            best = decomps[0]
                            alt_bridge_sizes = best.main_bridge_sizes
                            alt_type = "bridged"
                            ambiguous = True

        elif ring_type == "bridged":
            # For tricyclic+ (rank >= 3) use the structured decomposition that
            # produces both main bridge sizes and explicit secondary bridges.
            # For bicyclic (rank == 2) fall back to the legacy path since
            # behaviour is already correct.
            structured = self._try_structured_decomposition(all_atoms)
            if structured is not None:
                main_sizes, sec = structured
                bridge_sizes = main_sizes
                secondary_bridges = sec
            else:
                bridge_sizes = self._compute_bridge_sizes(rings, all_atoms)
            if ambiguous:
                # Could also be read as fused
                pass

        elif ring_type == "spiro":
            spiro_sizes = self._compute_spiro_sizes(rings, all_atoms)

        return RingSystem(
            atom_indices=all_atoms,
            rings=tuple(rings),
            type=ring_type,
            aromatic=aromatic,
            bridge_sizes=bridge_sizes,
            spiro_sizes=spiro_sizes,
            fusion_info=fusion_info,
            heteroatoms=heteroatoms,
            ring_size=ring_size,
            classification_ambiguous=ambiguous,
            alternate_type=alt_type,
            alternate_bridge_sizes=alt_bridge_sizes,
            secondary_bridges=secondary_bridges,
        )

    def _try_structured_decomposition(
        self, all_atoms: frozenset[int]
    ) -> tuple[tuple[int, ...], tuple[tuple[tuple[int, int], tuple[int, ...]], ...]] | None:
        """Use VBDecomposition for rank >= 3 bridged systems.

        Returns ``(main_bridge_sizes_desc, secondary_bridges)`` or ``None``
        to signal "use legacy path".
        """
        from iupac_namer.ring_naming.vb_decompose import (
            _circuit_rank,
            decompose_ring_system,
        )
        try:
            rank = _circuit_rank(all_atoms, self._mol)
        except Exception:  # defensive
            return None
        if rank < 3:
            return None
        decomps = decompose_ring_system(all_atoms, self._mol)
        if not decomps:
            return None
        best = decomps[0]
        return best.main_bridge_sizes, best.secondary_bridges

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _classify_ring_system(
        self, rings: list[frozenset[int]], all_atoms: frozenset[int]
    ) -> tuple[str, bool, str | None]:
        """Classify a polycyclic ring system.

        Returns
        -------
        (type, ambiguous, alternate_type)
            type: "fused", "bridged", or "spiro"
            ambiguous: True when fused/bridged distinction is unclear
            alternate_type: the other classification when ambiguous
        """
        has_spiro = False   # pair sharing exactly 1 atom (true articulation point)
        has_fused = False   # pair sharing exactly 2 atoms AND bonded
        has_bridged = False # pair sharing 3+ atoms OR 2 atoms but unbonded

        for i, r1 in enumerate(rings):
            for r2 in rings[i + 1:]:
                shared = r1 & r2
                n_shared = len(shared)
                if n_shared == 0:
                    continue
                elif n_shared == 1:
                    # Only count as spiro if this atom is a true articulation point
                    # in the ring-atom subgraph (not just an SSSR artefact).
                    candidate = next(iter(shared))
                    if self._is_articulation_point(all_atoms, candidate):
                        has_spiro = True
                elif n_shared == 2:
                    # Check if the two shared atoms are bonded (fused edge)
                    shared_list = list(shared)
                    bond = self._mol.GetBondBetweenAtoms(  # type: ignore[attr-defined]
                        shared_list[0], shared_list[1]
                    )
                    if bond is not None:
                        has_fused = True
                    else:
                        # Shared 2 atoms but no direct bond between them → bridged
                        has_bridged = True
                else:
                    # 3+ shared atoms → bridged
                    has_bridged = True

        # Spiro takes priority (pure spiro or mixed spiro+fused)
        if has_spiro and not has_bridged:
            return ("spiro", False, None)

        # Bridged takes priority over fused
        if has_bridged:
            # If there are also fused junctions, it may be ambiguous
            ambiguous = has_fused
            alt = "fused" if ambiguous else None
            return ("bridged", ambiguous, alt)

        if has_fused:
            return ("fused", False, None)

        # Fallback (should not happen for valid ring groups)
        return ("fused", False, None)

    def _is_articulation_point(self, all_ring_atoms: frozenset[int], candidate: int) -> bool:
        """Return True if removing candidate disconnects the ring-atom subgraph.

        Used to distinguish true spiro atoms from SSSR artefacts in
        bridged/fused polycyclics where two SSSR rings happen to share
        exactly one atom without the compound being truly spiro.
        """
        remaining = all_ring_atoms - {candidate}
        if len(remaining) < 2:
            return True
        start = next(iter(remaining))
        visited: set[int] = {start}
        queue = deque([start])
        mol = self._mol
        while queue:
            curr = queue.popleft()
            for nb in mol.GetAtomWithIdx(curr).GetNeighbors():  # type: ignore[attr-defined]
                nb_idx = nb.GetIdx()
                if nb_idx in remaining and nb_idx not in visited:
                    visited.add(nb_idx)
                    queue.append(nb_idx)
        return len(visited) < len(remaining)

    # ------------------------------------------------------------------
    # Bridged systems — bridge size computation
    # ------------------------------------------------------------------

    def _compute_bridge_sizes(
        self, rings: list[frozenset[int]], all_atoms: frozenset[int]
    ) -> tuple[int, ...]:
        """Compute bridge sizes for von Baeyer nomenclature.

        Finds the primary bridgehead pair and enumerates all simple paths
        between them within the ring atom set, where each bridge path does NOT
        pass through other bridgehead atoms.  Bridge size = number of atoms on
        each bridge path excluding the two principal bridgeheads.

        For norbornane (bicyclo[2.2.1]heptane): (2, 2, 1).
        For DABCO (bicyclo[2.2.2]octane with 2N): (2, 2, 2).

        Algorithm:
        1. Find candidate bridgehead atoms (in 2+ SSSR rings).
        2. For each candidate bridgehead pair (bh1, bh2):
           a. Compute bridges = simple paths from bh1 to bh2 that do NOT
              pass through other bridgeheads.
           b. Check that all ring atoms are covered (bridges + bridgeheads).
           c. Score = number of bridges (more is better for polycyclics).
        3. Select the pair with most bridges covering all atoms.
        """
        # Count how many SSSR rings each atom appears in
        atom_ring_count: Counter[int] = Counter()
        for ring in rings:
            for atom in ring:
                atom_ring_count[atom] += 1

        # Bridgehead atoms: in 2 or more SSSR rings
        all_bridgeheads = sorted(
            {a for a, c in atom_ring_count.items() if c >= 2}
        )

        if len(all_bridgeheads) < 2:
            return ()

        # Principal bridgeheads: atoms in the MOST rings (highest ring count).
        # For bicyclo[2.2.2] systems (e.g. DABCO), these are the 2 atoms in 3 rings.
        # For bicyclo[2.2.1] systems (norbornane), the 2 atoms in 3 rings are the
        # bridgeheads; all other bridgehead-like atoms (in 2 rings) are on bridges.
        max_ring_count = max(atom_ring_count[a] for a in all_bridgeheads)
        # Try principal bridgeheads first (highest ring count), then fallback to all
        candidate_bhs_by_priority = []
        # Priority 1: atoms in max rings
        priority1 = sorted({a for a in all_bridgeheads if atom_ring_count[a] == max_ring_count})
        candidate_bhs_by_priority.append(priority1)
        # Priority 2: all bridgeheads (fallback)
        if priority1 != all_bridgeheads:
            candidate_bhs_by_priority.append(all_bridgeheads)

        n_total = len(all_atoms)

        for candidate_bhs in candidate_bhs_by_priority:
            bh_set = set(candidate_bhs)
            # "Other bridgeheads" = atoms NOT selected as principal bridgeheads
            # but still in 2+ rings; bridges must not pass through these.
            non_bridge_bhs = set(all_bridgeheads) - bh_set  # intermediate bridgeheads to exclude

            best_result: tuple[int, ...] | None = None
            best_score: tuple = (-1,)

            for i in range(len(candidate_bhs)):
                for j in range(i + 1, len(candidate_bhs)):
                    bh1, bh2 = candidate_bhs[i], candidate_bhs[j]
                    # Bridges must not pass through other PRINCIPAL bridgeheads
                    # (allow bridge paths to pass through atoms in exactly 2 rings,
                    #  which are ordinary bridge atoms that happen to appear in SSSR)
                    other_principal = bh_set - {bh1, bh2}
                    paths = self._find_all_simple_paths(
                        bh1, bh2, all_atoms, exclude_intermediate=other_principal
                    )
                    if not paths:
                        continue
                    # Check all atoms covered: union of paths should include all ring atoms
                    covered: set[int] = set()
                    for p in paths:
                        covered.update(p)
                    if len(covered) != n_total:
                        continue
                    sizes = sorted(
                        (len(p) - 2 for p in paths),
                        reverse=True,
                    )
                    n_paths = len(paths)
                    min_bridge = min(sizes)
                    # Score: more bridges first, then min bridge size (prefer balanced)
                    score = (n_paths, min_bridge)
                    if score > best_score:
                        best_score = score
                        best_result = tuple(sizes)

            if best_result is not None:
                return best_result

        # Final fallback: allow paths through other bridgeheads (legacy behavior).
        best_paths_fb: list[list[int]] = []
        best_min_fb: int = -1
        for i in range(len(all_bridgeheads)):
            for j in range(i + 1, len(all_bridgeheads)):
                bh1, bh2 = all_bridgeheads[i], all_bridgeheads[j]
                paths = self._find_all_simple_paths(
                    bh1, bh2, all_atoms, exclude_intermediate=set()
                )
                if not paths:
                    continue
                sizes = [len(p) - 2 for p in paths]
                min_bridge = min(sizes)
                n_paths = len(paths)
                best_n = len(best_paths_fb)
                if n_paths > best_n or (n_paths == best_n and min_bridge > best_min_fb):
                    best_paths_fb = paths
                    best_min_fb = min_bridge
        return tuple(sorted((len(p) - 2 for p in best_paths_fb), reverse=True))

    def _find_all_simple_paths(
        self,
        start: int,
        end: int,
        allowed: frozenset[int],
        exclude_intermediate: set[int],
    ) -> list[list[int]]:
        """Find all simple paths from start to end staying within allowed atoms.

        Intermediate nodes cannot be in exclude_intermediate (bridgeheads).
        Returns a list of paths (each path is a list of atom indices including
        start and end).
        """
        mol = self._mol
        all_paths: list[list[int]] = []
        # DFS with stack: (current_atom, path_so_far)
        stack: list[tuple[int, list[int]]] = [(start, [start])]

        while stack:
            curr, path = stack.pop()
            if curr == end:
                all_paths.append(path)
                continue
            for nb in mol.GetAtomWithIdx(curr).GetNeighbors():  # type: ignore[attr-defined]
                nb_idx = nb.GetIdx()
                if nb_idx not in allowed:
                    continue
                if nb_idx in path:
                    continue
                # Intermediate nodes cannot be bridgeheads (except the destination)
                if nb_idx != end and nb_idx in exclude_intermediate:
                    continue
                stack.append((nb_idx, path + [nb_idx]))

        return all_paths

    # ------------------------------------------------------------------
    # Spiro systems — spiro size computation
    # ------------------------------------------------------------------

    def _compute_spiro_sizes(
        self, rings: list[frozenset[int]], all_atoms: frozenset[int]
    ) -> tuple[int, ...]:
        """Compute spiro ring sizes for spiro nomenclature.

        For spiro[4.5]decane: (4, 5) — sizes are atoms in each ring
        minus 1 (not counting the spiro atom), sorted ascending.

        For polyspirans with multiple spiro atoms: returns sizes in chain order.
        """
        # Find spiro atoms (atoms shared between exactly 2 rings AND are
        # articulation points in the ring atom graph)
        atom_ring_count: Counter[int] = Counter()
        for ring in rings:
            for atom in ring:
                atom_ring_count[atom] += 1

        spiro_atoms = {
            a for a, c in atom_ring_count.items()
            if c >= 2 and self._is_articulation_point(all_atoms, a)
        }

        if not spiro_atoms:
            # Fallback for simple spiro: find atom in exactly 2 rings
            spiro_atoms = {a for a, c in atom_ring_count.items() if c >= 2}

        if len(spiro_atoms) == 1:
            # Simple spiro: two rings connected at one atom
            spiro_atom = next(iter(spiro_atoms))
            # Find the two components around the spiro atom
            ring_sizes = sorted(
                len(r) - 1 for r in rings if spiro_atom in r
            )
            return tuple(ring_sizes)

        # Polyspiro: find the chain of rings and spiro atoms
        # Build the connectivity: which rings are connected through which spiro atoms
        ring_graph: dict[int, set[int]] = {i: set() for i in range(len(rings))}
        for i in range(len(rings)):
            for j in range(i + 1, len(rings)):
                shared = rings[i] & rings[j]
                if shared & spiro_atoms:
                    ring_graph[i].add(j)
                    ring_graph[j].add(i)

        # Order rings in chain for systematic spiro naming
        # Start from a ring with only one connection (end of chain)
        chain_order = self._order_spiro_chain(rings, ring_graph, spiro_atoms)

        # Ring sizes minus 1 for each ring in chain order
        spiro_sizes = tuple(len(rings[i]) - 1 for i in chain_order)
        return spiro_sizes

    def _order_spiro_chain(
        self,
        rings: list[frozenset[int]],
        ring_graph: dict[int, set[int]],
        spiro_atoms: set[int],
    ) -> list[int]:
        """Return ring indices in spiro chain order (for polyspiro naming).

        Starts from a terminal ring (degree 1 in the ring connectivity graph)
        and traverses the chain.
        """
        # Find terminal rings (degree 1)
        terminals = [i for i, nbrs in ring_graph.items() if len(nbrs) == 1]
        if not terminals:
            # Cycle: start anywhere
            start = 0
        else:
            start = terminals[0]

        # BFS/DFS through ring_graph
        order: list[int] = []
        visited: set[int] = set()
        queue: deque[int] = deque([start])
        while queue:
            i = queue.popleft()
            if i in visited:
                continue
            visited.add(i)
            order.append(i)
            for j in ring_graph[i]:
                if j not in visited:
                    queue.append(j)

        return order

    # ------------------------------------------------------------------
    # Fusion info computation
    # ------------------------------------------------------------------

    def _compute_fusion_info(self, rings: list[frozenset[int]]) -> FusionInfo:
        """Compute FusionInfo for a fused ring system.

        Identifies all ring-pair shared edges (bonds shared between two rings).
        """
        shared_edges: list[tuple[frozenset[int], frozenset[int]]] = []
        fusion_atoms_list: list[tuple[int, int]] = []
        mol = self._mol

        for i in range(len(rings)):
            for j in range(i + 1, len(rings)):
                shared = rings[i] & rings[j]
                if len(shared) == 2:
                    shared_list = list(shared)
                    bond = mol.GetBondBetweenAtoms(  # type: ignore[attr-defined]
                        shared_list[0], shared_list[1]
                    )
                    if bond is not None:
                        shared_edges.append((rings[i], rings[j]))
                        fusion_atoms_list.append(
                            (shared_list[0], shared_list[1])
                        )

        return FusionInfo(
            shared_edges=tuple(shared_edges),
            fusion_atoms=tuple(fusion_atoms_list),
        )

    # ------------------------------------------------------------------
    # Heteroatom detection
    # ------------------------------------------------------------------

    def _detect_heteroatoms(
        self, ring_atoms_ordered: list[int]
    ) -> tuple[HeteroPosition, ...]:
        """Detect non-carbon atoms in a ring with their ring positions.

        Parameters
        ----------
        ring_atoms_ordered:
            Atom indices in ring traversal order (0-indexed positions).
        """
        heteros: list[HeteroPosition] = []
        for pos, atom_idx in enumerate(ring_atoms_ordered):
            info = self._atoms[atom_idx]
            if info.element != "C":
                heteros.append(
                    HeteroPosition(
                        atom_idx=atom_idx,
                        element=info.element,
                        locant=None,  # locant assigned by ring naming, not perception
                    )
                )
        return tuple(heteros)

    def _detect_system_heteroatoms(
        self, all_atoms: frozenset[int]
    ) -> tuple[HeteroPosition, ...] | None:
        """Detect heteroatoms in any ring system (monocyclic or polycyclic)."""
        heteros: list[HeteroPosition] = []
        for atom_idx in sorted(all_atoms):
            info = self._atoms[atom_idx]
            if info.element != "C":
                heteros.append(
                    HeteroPosition(
                        atom_idx=atom_idx,
                        element=info.element,
                        locant=None,
                    )
                )
        return tuple(heteros) if heteros else None

    # ------------------------------------------------------------------
    # Aromaticity
    # ------------------------------------------------------------------

    def _is_ring_aromatic(self, ring: frozenset[int]) -> bool:
        """Return True if all atoms in the ring are aromatic."""
        return all(self._atoms[idx].aromatic for idx in ring)

    def _is_system_aromatic(self, all_atoms: frozenset[int]) -> bool:
        """Return True if all atoms in the system are aromatic."""
        return all(self._atoms[idx].aromatic for idx in all_atoms)

    # ------------------------------------------------------------------
    # Ring atom ordering
    # ------------------------------------------------------------------

    def _order_ring_atoms(self, ring: frozenset[int]) -> list[int]:
        """Return ring atoms in traversal order (BFS from lowest-index atom).

        The ordering is consistent for a given ring but not guaranteed to
        match any canonical IUPAC numbering — that is computed later by
        the numbering module.
        """
        if not ring:
            return []

        mol = self._mol
        ring_set = ring
        start = min(ring)  # lowest atom index as starting point

        ordered: list[int] = [start]
        visited: set[int] = {start}

        curr = start
        while len(ordered) < len(ring):
            # Walk to adjacent ring atom not yet visited
            found = False
            for nb in mol.GetAtomWithIdx(curr).GetNeighbors():  # type: ignore[attr-defined]
                nb_idx = nb.GetIdx()
                if nb_idx in ring_set and nb_idx not in visited:
                    ordered.append(nb_idx)
                    visited.add(nb_idx)
                    curr = nb_idx
                    found = True
                    break
            if not found:
                # Disconnected from rest — add remaining in index order
                for idx in sorted(ring_set - visited):
                    ordered.append(idx)
                    visited.add(idx)
                break

        return ordered

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def ring_systems(self) -> tuple[RingSystem, ...]:
        """All ring systems found in the molecule."""
        return self._ring_systems

    @property
    def has_rings(self) -> bool:
        """True if the molecule has at least one ring."""
        return len(self._ring_systems) > 0

    def ring_system_for_atom(self, atom_idx: int) -> RingSystem | None:
        """Return the RingSystem containing atom_idx, or None if not in a ring."""
        for rs in self._ring_systems:
            if atom_idx in rs.atom_indices:
                return rs
        return None

    def all_ring_atoms(self) -> frozenset[int]:
        """Return the set of all atom indices that belong to any ring."""
        if not self._ring_systems:
            return frozenset()
        result: frozenset[int] = frozenset()
        for rs in self._ring_systems:
            result = result | rs.atom_indices
        return result

    def detect_ring_unsaturation(
        self, ring_system: RingSystem, numbering: object
    ) -> tuple:
        """Detect non-aromatic double/triple bonds in a ring parent.

        Aromatic bonds are NOT reported (the ring name encodes aromaticity).
        Returns UnsaturationInfix objects.

        This is a stub — fully implemented when numbering support exists.
        """
        return ()

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"RingAnalysis({len(self._ring_systems)} ring systems, "
            f"has_rings={self.has_rings})"
        )
