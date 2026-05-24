"""
iupac_namer/perception/chains.py

ChainFinding subsystem — Subsystem 7 of the Perception layer.

Finds candidate parent chains in an RDKit molecule.  A chain is an acyclic
sequence of heavy atoms (currently: carbon atoms for standard substitutive
nomenclature; heteroatom chains handled by replacement nomenclature in Phase 3).

Key responsibilities:
  - Build an acyclic subgraph (non-ring atoms connected by non-ring bonds)
  - Find all longest simple paths (candidate principal chains, P-44.3)
  - Filter by PCG anchor atoms when requested
  - Detect chain unsaturation (double and triple bonds)
  - Return CandidateParent objects ready for the strategy layer

See ARCHITECTURE_PERCEPTION.md §Subsystem 7 for the full spec.
See ARCHITECTURE_DATA_STRUCTURES.md for CandidateParent and UnsaturationInfix.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from iupac_namer.types import CandidateParent, Locant, UnsaturationInfix

if TYPE_CHECKING:
    from iupac_namer.perception.atoms import AtomAnalysis
    from iupac_namer.perception.rings import RingAnalysis

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Multiplier helper
# ---------------------------------------------------------------------------


def _multiplier(count: int) -> str | None:
    """Return the multiplicative prefix for *count* identical unsaturations.

    Returns ``None`` for count == 1 (no multiplier needed).  Delegates to
    ``data_loader.get_multiplier`` so counts above 10 (undeca/dodeca/
    trideca/...) resolve correctly; the previous local table only went up
    to 10 and fell back to ``f"{count}a"`` for larger chains (e.g. lycopene's
    13 double bonds rendered as ``13aene`` instead of ``tridecaene``).
    """
    if count <= 1:
        return None
    from iupac_namer.data_loader import get_multiplier
    return get_multiplier(count, complex=False)


# ---------------------------------------------------------------------------
# ChainFinding
# ---------------------------------------------------------------------------


class ChainFinding:
    """Find candidate parent chains in an RDKit molecule.

    Parameters
    ----------
    mol:
        An RDKit ``Mol`` object (after sanitisation).
    atom_analysis:
        The :class:`~iupac_namer.perception.atoms.AtomAnalysis` for this mol.
    ring_analysis:
        The :class:`~iupac_namer.perception.rings.RingAnalysis` for this mol.
    """

    def __init__(
        self,
        mol: object,
        atom_analysis: "AtomAnalysis",
        ring_analysis: "RingAnalysis",
    ) -> None:
        self._mol = mol
        self._atoms = atom_analysis
        self._rings = ring_analysis

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_candidate_chains(
        self,
        pcg_anchors: tuple[int, ...] = (),
        required_atom: int | None = None,
    ) -> list[CandidateParent]:
        """Find candidate parent chains.

        Parameters
        ----------
        pcg_anchors:
            If non-empty, only return chains that contain or are directly
            adjacent to at least one anchor atom.  An anchor is typically
            the defining atom of a principal characteristic group (PCG)
            such as the carboxyl carbon of -COOH.
        required_atom:
            If provided, also include a single-atom chain for this atom
            if it is in the acyclic graph but not covered by any
            max-length chain candidate.  Used in SUBSTITUENT mode to
            ensure the attachment atom always has a valid parent candidate
            even when a longer disconnected chain exists elsewhere (e.g.
            N-ethyl group longer than the exo-methyl attachment point on
            a pyrrolidine ring).

        Returns
        -------
        list[CandidateParent]
            All candidate chains of maximal length, longest first.
            May be empty for all-ring molecules (e.g. cyclohexane).
            When *required_atom* is given and not covered, a one-atom
            chain for that atom is appended at the end.
        """
        graph = self._build_acyclic_graph()
        if not graph:
            return []

        longest_paths = self._find_longest_paths(graph)
        if not longest_paths:
            return []

        # De-duplicate: treat A-B-C and C-B-A as the same path.
        unique_paths = self._deduplicate_paths(longest_paths)

        # Optionally filter by PCG anchor adjacency.
        if pcg_anchors:
            unique_paths = self._filter_by_anchors(unique_paths, pcg_anchors)

        # P-44.3a: when 2+ PCG anchors are graph nodes (carbon anchors of e.g.
        # carboxylic acid, nitrile, aldehyde) and no max-length chain contains
        # ALL of them, we must also yield shorter candidate chains that span
        # all anchors.  IUPAC ranks the chain with the most PCG suffix groups
        # (Band 4) above the longest chain (Band 3), so the strategy will then
        # correctly pick the all-anchor-spanning chain — producing e.g.
        # "2-ethylpropanedioic acid" for HOOC-CH(Et)-COOH instead of the
        # malformed "butanoic acid-2-carboxylic acid" (which is what happens
        # when only the longest 4C chain — containing only ONE COOH C —
        # survives the candidate filter).
        anchor_paths_to_add: list[list[int]] = []
        if pcg_anchors and len(pcg_anchors) >= 2:
            graph_anchors = [a for a in pcg_anchors if a in graph]
            if len(graph_anchors) >= 2:
                # Check whether some existing max-length path already contains
                # all graph-resident anchors.  If so, the strategy can already
                # pick it on its own and no extra candidate is needed.
                anchor_set_g = set(graph_anchors)
                already_covered = any(
                    anchor_set_g <= set(p) for p in unique_paths
                )
                if not already_covered:
                    spanning = self._find_longest_path_through_anchors(
                        graph, anchor_set_g
                    )
                    if spanning:
                        anchor_paths_to_add.extend(spanning)

        # P-44.1.1 (Phase 8 P-4 audit): even with a SINGLE PCG anchor, the
        # parent chain must include the chain carbon that bears that PCG, so
        # the suffix can be expressed on the parent.  Band-4 +2.0 for a
        # terminal anchor beats the +0.1-per-extra-atom band-3 length bonus,
        # so the chain that bears the PCG always wins on score once it is
        # enumerated.
        #
        # Two anchor flavours:
        #   (a) C-anchor FGs (alcohol, aldehyde, COOH-as-suffix, nitrile)
        #       — anchor IS a graph carbon.  Add a path through it.
        #   (b) Heteroatom-anchor FGs (amine: N, thiol: S, etc.) — anchor
        #       is not a graph carbon, but its only chain neighbour (the
        #       C-atom bonded to N/S/...) MUST be on the parent.  Treat
        #       that neighbour as the effective chain anchor.
        #
        # Without this branch, NCC(CC)CCCC produced the wrong PIN
        # "3-(aminomethyl)heptane" instead of "2-ethylhexan-1-amine".
        if pcg_anchors and len(pcg_anchors) == 1:
            graph_anchors: list[int] = []
            for a in pcg_anchors:
                if a in graph:
                    graph_anchors.append(a)
                else:
                    # Heteroatom anchor: pull the chain-graph C-neighbour
                    # if there is exactly one.  Skip cases where 0 or 2+
                    # neighbours are in graph (ambiguous).
                    if a < len(self._atoms):
                        ainfo = self._atoms[a]
                        chain_nbrs = [n for n in ainfo.neighbors if n in graph]
                        if len(chain_nbrs) == 1:
                            graph_anchors.append(chain_nbrs[0])
            if len(graph_anchors) == 1:
                anchor = graph_anchors[0]
                already_covered = any(anchor in p for p in unique_paths)
                if not already_covered:
                    spanning = self._find_longest_path_through_anchors(
                        graph, {anchor}
                    )
                    if spanning:
                        anchor_paths_to_add.extend(spanning)

        candidates = []
        for path in unique_paths:
            cp = self._make_candidate(path)
            candidates.append(cp)

        # Sort by length descending (all equal after dedup, but keep API contract).
        candidates.sort(key=lambda c: -c.length)

        # Append anchor-spanning shorter chains AFTER the longest-path
        # candidates.  De-duplicate against the existing candidate set by
        # atom_indices.
        if anchor_paths_to_add:
            seen_atom_sets = {c.atom_indices for c in candidates}
            for path in anchor_paths_to_add:
                cp = self._make_candidate(path)
                if cp.atom_indices not in seen_atom_sets:
                    candidates.append(cp)
                    seen_atom_sets.add(cp.atom_indices)

        # If a required_atom is specified and not covered by any max-length chain
        # candidate, add a single-atom chain for that atom.  This handles the
        # case where a short exo-chain atom (e.g. the exo-methyl on a ring C)
        # is the substituent attachment point but a longer disconnected chain
        # (e.g. N-ethyl) is the only max-length chain.  Without this fallback
        # no plan can be generated and the substituent naming fails.
        if required_atom is not None and required_atom in graph:
            covered = any(required_atom in c.atom_indices for c in candidates)
            if not covered:
                single_atom_cp = self._make_candidate([required_atom])
                candidates.append(single_atom_cp)

        return candidates

    def detect_chain_unsaturation(
        self,
        chain_atoms: list[int],
        numbering: object = None,
    ) -> tuple[UnsaturationInfix, ...]:
        """Detect double and triple bonds between consecutive chain atoms.

        Parameters
        ----------
        chain_atoms:
            Ordered list of atom indices along the chain (start to end).
        numbering:
            If provided, use locants from ``numbering.atom_to_locant``.
            If ``None``, use 1-indexed positions along *chain_atoms*
            (position 1 = chain_atoms[0]).

        Returns
        -------
        tuple[UnsaturationInfix, ...]
            Sorted by first locant.  Empty if chain is fully saturated.
        """
        double_locs: list[Locant] = []
        triple_locs: list[Locant] = []

        atom_to_locant: dict[int, Locant] | None = None
        if numbering is not None:
            atom_to_locant = numbering.atom_to_locant  # type: ignore[attr-defined]

        for i in range(len(chain_atoms) - 1):
            a1 = chain_atoms[i]
            a2 = chain_atoms[i + 1]
            bond_type = self._atoms.get_bond_type(a1, a2)

            if bond_type not in ("double", "triple"):
                continue

            # Determine bond locant: the lower-numbered end of the bond.
            # IUPAC P-31.1.2.1: the locant cited is that of the first carbon
            # in the double/triple bond, i.e. the lower locant of the two atoms.
            if atom_to_locant is not None:
                loc_a1 = atom_to_locant.get(a1)
                loc_a2 = atom_to_locant.get(a2)
                if loc_a1 is not None and loc_a2 is not None:
                    locant = min(loc_a1, loc_a2)
                elif loc_a1 is not None:
                    locant = loc_a1
                elif loc_a2 is not None:
                    locant = loc_a2
                else:
                    locant = Locant.numeric(i + 1)
            else:
                locant = Locant.numeric(i + 1)

            if bond_type == "double":
                double_locs.append(locant)
            else:  # triple
                triple_locs.append(locant)

        infixes: list[UnsaturationInfix] = []

        if double_locs:
            double_locs.sort()
            mult = _multiplier(len(double_locs))
            infixes.append(
                UnsaturationInfix(
                    type="en",
                    locants=tuple(double_locs),
                    multiplier=mult,
                )
            )

        if triple_locs:
            triple_locs.sort()
            mult = _multiplier(len(triple_locs))
            infixes.append(
                UnsaturationInfix(
                    type="yn",
                    locants=tuple(triple_locs),
                    multiplier=mult,
                )
            )

        # Sort infixes by their first locant (en before yn when at same position)
        infixes.sort(key=lambda inf: (inf.locants[0] if inf.locants else Locant.numeric(0)))

        return tuple(infixes)

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_acyclic_graph(self) -> dict[int, set[int]]:
        """Build an adjacency list of non-ring CARBON atoms.

        Only non-ring carbon atoms are nodes.  Edges exist between two non-ring
        carbon atoms that share a bond.  Carbon atoms adjacent to ring atoms are
        included as nodes (they become chain terminals that sit next to a ring).

        Heteroatoms (N, O, S, P, etc.) are NOT included in the graph.
        Acyclic substitutive chains consist only of carbon atoms per IUPAC P-31.
        Heteroatom chains (replacement nomenclature) are handled separately in
        Phase 3.

        Returns
        -------
        dict[int, set[int]]
            Adjacency list: atom_idx -> set of neighbouring atom indices
            (all within the carbon-only acyclic subgraph).
        """
        ring_atoms = self._rings.all_ring_atoms()

        # Include only non-ring CARBON atoms (skip H and all heteroatoms)
        graph: dict[int, set[int]] = {}
        for atom_info in self._atoms:
            if atom_info.element != "C":
                continue
            if atom_info.idx not in ring_atoms:
                graph[atom_info.idx] = set()

        # Add edges between non-ring carbon atoms
        for atom_idx in graph:
            atom_info = self._atoms[atom_idx]
            for neighbor_idx in atom_info.neighbors:
                if neighbor_idx in graph:
                    graph[atom_idx].add(neighbor_idx)

        return graph

    # ------------------------------------------------------------------
    # Path finding
    # ------------------------------------------------------------------

    def _find_longest_paths(self, graph: dict[int, set[int]]) -> list[list[int]]:
        """Find all longest simple paths in the acyclic subgraph.

        Algorithm:
        1. Partition the graph into connected components.
        2. For each component, identify terminal atoms (degree 0 or 1) and
           atoms adjacent to ring atoms (chain-end terminals at ring boundary).
        3. DFS from each terminal within its component to enumerate all simple
           paths; collect the longest paths within that component.
        4. Return the union of per-component longest paths.

        Treating each component independently is critical when the acyclic
        carbon graph is disconnected (e.g., two separate alkyl chains separated
        by a heteroatom bridge).  A global-max approach would discard shorter
        components that nonetheless contain the PCG anchor atom.

        Returns
        -------
        list[list[int]]
            All per-component longest paths (may include forward/reverse
            duplicates — caller should de-duplicate).
        """
        if not graph:
            return []

        ring_atoms = self._rings.all_ring_atoms()

        # ------------------------------------------------------------------
        # Step 1: find connected components via BFS
        # ------------------------------------------------------------------
        visited: set[int] = set()
        components: list[set[int]] = []
        for seed in graph:
            if seed in visited:
                continue
            component: set[int] = set()
            queue = [seed]
            while queue:
                node = queue.pop()
                if node in component:
                    continue
                component.add(node)
                for nb in graph[node]:
                    if nb not in component:
                        queue.append(nb)
            components.append(component)
            visited |= component

        # ------------------------------------------------------------------
        # Step 2–4: per-component longest-path search
        # ------------------------------------------------------------------
        all_longest: list[list[int]] = []

        for component in components:
            # Build sub-graph restricted to this component
            subgraph = {n: graph[n] & component for n in component}

            # Terminal atoms: degree 0 or 1 in the subgraph
            terminals: list[int] = []
            for node, neighbors in subgraph.items():
                if len(neighbors) <= 1:
                    terminals.append(node)

            # Also include atoms adjacent to ring atoms
            ring_adjacent: set[int] = set()
            for atom_idx in subgraph:
                atom_info = self._atoms[atom_idx]
                for neighbor_idx in atom_info.neighbors:
                    if neighbor_idx in ring_atoms:
                        ring_adjacent.add(atom_idx)

            terminal_set = set(terminals) | ring_adjacent
            terminals = list(terminal_set)

            if not terminals:
                # Isolated node
                terminals = list(subgraph.keys())[:1]

            comp_max = 0
            comp_longest: list[list[int]] = []

            for start in terminals:
                paths = self._dfs_all_paths_from(subgraph, start)
                for path in paths:
                    path_len = len(path)
                    if path_len > comp_max:
                        comp_max = path_len
                        comp_longest = [path]
                    elif path_len == comp_max:
                        comp_longest.append(path)

            all_longest.extend(comp_longest)

        return all_longest

    def _dfs_all_paths_from(
        self, graph: dict[int, set[int]], start: int
    ) -> list[list[int]]:
        """Enumerate all simple paths starting from *start* via DFS.

        Parameters
        ----------
        graph:
            Adjacency list of the acyclic subgraph.
        start:
            Starting atom index.

        Returns
        -------
        list[list[int]]
            Every simple path (list of atom indices) reachable from *start*.
            Includes trivial single-node path [start].
        """
        all_paths: list[list[int]] = []
        # Stack entries: (current_node, path_so_far, visited_set)
        stack: list[tuple[int, list[int], frozenset[int]]] = [
            (start, [start], frozenset([start]))
        ]

        while stack:
            curr, path, visited = stack.pop()
            extended = False
            for neighbor in graph[curr]:
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    new_visited = visited | {neighbor}
                    stack.append((neighbor, new_path, new_visited))
                    extended = True
            # If we couldn't extend, this is a terminal path
            if not extended:
                all_paths.append(path)
            else:
                # Also record the current path as a candidate
                # (in case the longer paths are not actually longer
                #  due to branching — the DFS will extend further)
                # We DON'T record intermediate paths here; only leaves are
                # valid paths (simplest correct implementation).
                pass

        # If graph has a single node, the stack processing above leaves
        # all_paths empty (no stack push happens).  Handle that edge case.
        if not all_paths and len(graph) == 1:
            all_paths = [[start]]

        return all_paths

    # ------------------------------------------------------------------
    # De-duplication and filtering
    # ------------------------------------------------------------------

    def _deduplicate_paths(self, paths: list[list[int]]) -> list[list[int]]:
        """Remove reversed duplicates from *paths*.

        Treats path A-B-C and C-B-A as the same chain.  Keeps the
        lexicographically smaller of the two (by first element).

        Returns
        -------
        list[list[int]]
            De-duplicated paths.
        """
        seen: set[tuple[int, ...]] = set()
        unique: list[list[int]] = []
        for path in paths:
            key = tuple(path) if path[0] <= path[-1] else tuple(reversed(path))
            if key not in seen:
                seen.add(key)
                unique.append(path)
        return unique

    def _find_longest_path_through_anchors(
        self,
        graph: dict[int, set[int]],
        anchor_set: set[int],
    ) -> list[list[int]]:
        """Find the longest simple path(s) containing every atom in *anchor_set*.

        Used by ``find_candidate_chains`` to add a shorter candidate chain that
        spans 2+ PCG anchors when no max-length path covers them all.  The
        downstream strategy ranks chains with more PCGs on the parent (Band 4)
        above the longest chain (Band 3), so providing the all-anchor-spanning
        chain lets the engine emit "<X>anedioic acid" forms instead of the
        malformed "<X>oic acid-N-carboxylic acid" concatenation.

        If no simple path contains all anchors (the anchors are in different
        connected components, or only one connected via the acyclic carbon
        graph), returns an empty list — the caller will fall back to the
        longest-path candidates.  Triacid / branched poly-PCG cases (3+ PCG
        anchors where one must be a branch) are not handled here; they need
        a separate prefix-fallback fix in the suffix-emission code so the
        non-chain anchor renders as a "carboxy" prefix rather than a
        "-N-carboxylic acid" appendage.  See TODO in engine.py.

        Returns
        -------
        list[list[int]]
            All longest simple paths in *graph* that contain every atom in
            *anchor_set*.  Each path is canonicalised so it can be compared
            against other paths.  Empty if no such path exists.
        """
        if not anchor_set:
            return []
        # Sanity: every anchor must be a graph node.
        if not anchor_set <= set(graph.keys()):
            return []

        # All anchors must lie in the same connected component for a simple
        # path through them to exist.
        anchors_list = list(anchor_set)
        seed = anchors_list[0]
        visited: set[int] = {seed}
        stack = [seed]
        while stack:
            n = stack.pop()
            for nb in graph[n]:
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        if not anchor_set <= visited:
            return []

        # DFS from every node and prune paths that don't contain all anchors.
        best_len = 0
        best_paths: list[list[int]] = []
        max_n = len(graph)
        # Cap to avoid combinatorial blow-up on huge components.
        if max_n > 60:
            return []

        for start in graph:
            stack2: list[tuple[int, list[int], frozenset[int]]] = [
                (start, [start], frozenset([start]))
            ]
            while stack2:
                curr, path, vis = stack2.pop()
                # Check whether this path is a candidate (contains all anchors).
                if anchor_set <= set(path):
                    plen = len(path)
                    if plen > best_len:
                        best_len = plen
                        best_paths = [path]
                    elif plen == best_len:
                        best_paths.append(path)
                # Extend.
                for nb in graph[curr]:
                    if nb not in vis:
                        stack2.append((nb, path + [nb], vis | {nb}))

        # De-duplicate (path and its reverse).
        unique = self._deduplicate_paths(best_paths) if best_paths else []
        return unique

    def _filter_by_anchors(
        self, paths: list[list[int]], pcg_anchors: tuple[int, ...]
    ) -> list[list[int]]:
        """Return only paths that contain or are adjacent to a PCG anchor.

        A path is "related to" an anchor if:
        - The anchor atom is IN the path (terminal FG), OR
        - The anchor atom is bonded to an atom IN the path (non-terminal FG,
          e.g. a -COOH attached to a ring, where the COOH carbon is exocyclic).

        Parameters
        ----------
        paths:
            Candidate chain paths (list of atom index lists).
        pcg_anchors:
            Atom indices of all PCG anchor atoms.

        Returns
        -------
        list[list[int]]
            Filtered paths.  If no path passes the filter, returns all paths
            (fall-back to avoid empty candidate lists).
        """
        anchor_set = set(pcg_anchors)

        # Build adjacency set: all atoms bonded to any anchor
        anchor_neighbors: set[int] = set()
        for anchor_idx in anchor_set:
            if anchor_idx < len(self._atoms):
                atom_info = self._atoms[anchor_idx]
                anchor_neighbors.update(atom_info.neighbors)

        related_paths = []
        for path in paths:
            path_set = set(path)
            # Check if any anchor is on the path or adjacent to path
            if path_set & anchor_set:
                related_paths.append(path)
            elif path_set & anchor_neighbors:
                related_paths.append(path)

        # Fallback: if no paths matched, return all (don't silently drop everything)
        if not related_paths:
            logger.debug(
                "ChainFinding: pcg_anchor filter matched no paths; "
                "returning all %d candidate(s).",
                len(paths),
            )
            return paths

        return related_paths

    # ------------------------------------------------------------------
    # CandidateParent construction
    # ------------------------------------------------------------------

    def _make_candidate(self, chain_atoms: list[int]) -> CandidateParent:
        """Build a :class:`~iupac_namer.types.CandidateParent` from an ordered chain.

        Parameters
        ----------
        chain_atoms:
            Ordered list of atom indices (start to end of chain).

        Returns
        -------
        CandidateParent
            Structural descriptor ready for the strategy layer.
        """
        unsaturation = self.detect_chain_unsaturation(chain_atoms)

        return CandidateParent(
            atom_indices=frozenset(chain_atoms),
            type="chain",
            length=len(chain_atoms),
            ring_system=None,
            unsaturation=unsaturation if unsaturation else None,
            element=None,
            lambda_value=None,
        )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        ring_count = len(self._rings.ring_systems)
        graph = self._build_acyclic_graph()
        return (
            f"ChainFinding("
            f"acyclic_atoms={len(graph)}, "
            f"ring_systems={ring_count})"
        )
