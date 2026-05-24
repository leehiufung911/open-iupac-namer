"""
iupac_namer/ring_naming/__init__.py

Ring naming package — converts RingSystem structural descriptors into
NamedParent objects that the engine uses during plan generation.

Main API:
    name_ring_system(candidate, mol) -> list[NamedParent]

Design: ring naming is a NAMING decision, not a perception decision.
Perception produces structural descriptors (ring type, sizes, heteroatom
positions). Ring naming uses those to generate named parent candidates.
Strategy scores the resulting plans.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from iupac_namer.types import NamedParent

if TYPE_CHECKING:
    from iupac_namer.types import CandidateParent, RingSystem

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def name_ring_system(
    candidate: "CandidateParent",
    mol,
) -> list[NamedParent]:
    """Generate NamedParent candidates for a ring system.

    Returns all valid naming options. Strategy will score them.
    For most rings: 1-2 options (retained + systematic, or just one).

    Parameters
    ----------
    candidate:
        CandidateParent with ring_system populated.
    mol:
        RDKit Mol for the full molecule (used for SMILES extraction).
    """
    ring_system = candidate.ring_system
    if ring_system is None:
        return []

    results: list[NamedParent] = []

    # --- 1. Try retained name first (highest priority) ---
    try:
        from iupac_namer.ring_naming.retained_lookup import try_retained_name
        retained = try_retained_name(ring_system, mol)
        if retained:
            results.extend(retained)
    except Exception as e:
        logger.warning("Retained ring name lookup failed: %s", e)

    # --- 2. Generate systematic name(s) ---
    rs_type = ring_system.type
    try:
        match rs_type:
            case "monocyclic":
                from iupac_namer.ring_naming.monocyclic import name_monocyclic
                results.extend(name_monocyclic(ring_system, candidate, mol))
            case "fused":
                from iupac_namer.ring_naming.fused import name_fused
                results.extend(name_fused(ring_system, candidate, mol))
                # Stage 5: methylenedioxy-bridge on retained polycyclic base.
                # Tight guard — only fires when the ring system contains
                # an exact O-CH2-O dioxolane attached to two adjacent atoms
                # of an otherwise retained polycyclic base (e.g. saturated
                # cyclopenta[a]phenanthrene).  See
                # iupac_namer/ring_naming/methylenedioxy_bridge.py for
                # guard details and the architectural rationale.
                from iupac_namer.ring_naming.methylenedioxy_bridge import (
                    name_methylenedioxy_bridge,
                )
                results.extend(
                    name_methylenedioxy_bridge(ring_system, candidate, mol)
                )
            case "bridged":
                from iupac_namer.ring_naming.bridged import name_bridged
                results.extend(name_bridged(ring_system, candidate, mol))
                # Spiro-polycyclic-fused (P-24.5): a "bridged" system that
                # actually contains an articulation atom splitting it into a
                # bridged polycyclic partner + monocyclic partner is named as
                # spiro[<poly>-<loc>,<loc>'-<mono>].  Offer both so strategy
                # can prefer spiro when the partner topology fits.
                from iupac_namer.ring_naming.spiro import name_polycyclic_spiro
                results.extend(name_polycyclic_spiro(ring_system, candidate, mol))
                # Benzo-fused-bridged-bicyclic (e.g. dezocine's
                # 5,11-methanobenzocyclodecene): when the bridged system
                # contains exactly one ortho-fused benzene, the IUPAC-preferred
                # name uses "benzo<cyclo[N]ene>" parent with hydro and
                # methano/ethano bridge prefixes.  Pure VB tricyclo[...] would
                # collapse the aromatic ring into a saturated one and fail the
                # OPSIN round-trip.
                from iupac_namer.ring_naming.benzo_fused_bridged import (
                    name_benzo_fused_bridged,
                )
                results.extend(
                    name_benzo_fused_bridged(ring_system, candidate, mol)
                )
            case "spiro":
                from iupac_namer.ring_naming.spiro import name_spiro
                results.extend(name_spiro(ring_system, candidate, mol))
            case _:
                logger.debug("Unknown ring type: %s", rs_type)
    except Exception as e:
        logger.warning("Systematic ring naming failed for type=%s: %s", rs_type, e)

    # --- 3. Handle ambiguous classification ---
    if ring_system.classification_ambiguous and ring_system.alternate_type:
        alt_type = ring_system.alternate_type
        try:
            _generate_alternate(ring_system, candidate, mol, alt_type, results)
        except Exception as e:
            logger.debug("Alternate classification naming failed: %s", e)

    # Deduplicate by name string.  When two sources produce the SAME name,
    # prefer the one that carries pre-computed ``numbering_options`` (a fixed
    # heteroatom-/locant-pinned numbering) over one with none.  Several OPSIN-
    # extracted saturated heterocycle entries (e.g. "1,4-oxazepane") arrive via
    # the retained-lookup with NO numbering_options and would otherwise shadow
    # the systematic Hantzsch-Widman parent, which DOES pin the heteroatom-
    # determined numbering.  Without the pinned numbering the engine's generic
    # numbering pass minimises substituent locants against the heteroatom
    # constraint (e.g. emitting "2-chloro-1,4-oxazepane" for a Cl the cited 1,4
    # numbering actually places at 7), which fails the OPSIN round-trip.
    by_name: dict[str, NamedParent] = {}
    order: list[str] = []
    for np in results:
        existing = by_name.get(np.name)
        if existing is None:
            by_name[np.name] = np
            order.append(np.name)
        elif not existing.numbering_options and np.numbering_options:
            # Upgrade to the variant that pins a numbering.
            by_name[np.name] = np

    return [by_name[n] for n in order]


def _generate_alternate(
    ring_system: "RingSystem",
    candidate: "CandidateParent",
    mol,
    alt_type: str,
    results: list[NamedParent],
) -> None:
    """Generate names for the alternate ring classification."""
    if alt_type == "bridged" and ring_system.alternate_bridge_sizes:
        from iupac_namer.types import RingSystem as RS
        # When the fused→bridged alt path is taken for a true tricyclic+
        # cage system, recompute secondary bridges from the decomposition so
        # the bridged-naming layer can emit ``tricyclo[...]`` / higher forms
        # with correct superscript locants.
        alt_secondary = None
        if ring_system.type == "fused":
            try:
                from iupac_namer.ring_naming.vb_decompose import (
                    _circuit_rank,
                    decompose_ring_system,
                )
                if _circuit_rank(ring_system.atom_indices, mol) >= 3:
                    decomps = decompose_ring_system(
                        ring_system.atom_indices, mol
                    )
                    if decomps:
                        alt_secondary = decomps[0].secondary_bridges
            except Exception:  # defensive — fall back to bicyclic alt
                alt_secondary = None
        alt_rs = RS(
            atom_indices=ring_system.atom_indices,
            rings=ring_system.rings,
            type="bridged",
            aromatic=ring_system.aromatic,
            bridge_sizes=ring_system.alternate_bridge_sizes,
            spiro_sizes=None,
            fusion_info=None,
            heteroatoms=ring_system.heteroatoms,
            ring_size=ring_system.ring_size,
            secondary_bridges=alt_secondary,
        )
        from iupac_namer.types import CandidateParent as CP
        alt_candidate = CP(
            atom_indices=candidate.atom_indices,
            type="bridged",
            length=candidate.length,
            ring_system=alt_rs,
            unsaturation=candidate.unsaturation,
            element=candidate.element,
            lambda_value=candidate.lambda_value,
        )
        from iupac_namer.ring_naming.bridged import name_bridged
        results.extend(name_bridged(alt_rs, alt_candidate, mol))


# ---------------------------------------------------------------------------
# Re-export numbering helper for engine use
# ---------------------------------------------------------------------------

def get_ring_numberings(ring_system: "RingSystem", mol, named_parent: NamedParent):
    """Compute ring numberings for a named parent.  Convenience wrapper."""
    from iupac_namer.ring_naming.numbering import compute_ring_numberings
    return compute_ring_numberings(ring_system, mol, named_parent)


__all__ = ["name_ring_system", "get_ring_numberings"]
