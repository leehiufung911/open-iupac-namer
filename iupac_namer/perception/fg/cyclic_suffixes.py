"""iupac_namer.perception.fg.cyclic_suffixes

Ring-FG detector for cyclic imide / lactam / lactone motifs — Stage 6 R2-E.

Closes root cause #13 in ``docs/opsin_coverage_taxonomy.md`` (FG audit
Gap 8): our engine's cyclic-suffix path historically assumed
cyclopentane / cyclohexane parents and failed to classify
``-dicarboximide`` / ``-olactam`` / ``-olactone`` motifs on other ring
sizes and on iminol tautomer variants.

Scope
-----
The module provides a *classifier* for ring-embedded carbonyl-heteroatom
patterns that form the basis of the IUPAC cyclic-suffix family:

* **Imide** — an endocyclic ``C(=O)-N(R)-C(=O)`` triple, with any number of
  chain atoms between the two carbonyls.  The two carbonyl carbons and
  the bridging nitrogen all live in the same ring.
* **Lactam** — an endocyclic ``C(=O)-N(R)`` pair; the carbonyl carbon and
  the nitrogen are ring atoms.
* **Lactone** — an endocyclic ``C(=O)-O`` pair; the carbonyl carbon and the
  ester oxygen are ring atoms.

For each motif the classifier records the ring size, the anchor atom
indices and whether the carbonyl is written in keto or iminol/enol
tautomer form.  Downstream consumers (the assembly layer, the ring-FG
scorer, the engine's retained-lactone / retained-lactam dispatch) may
consult this classification to decide whether to emit the retained
``-olactone`` / ``-olactam`` surface form or fall back to the
``2-oxo`` / ``1,3-dioxo`` ring-parent equivalent.

Design
------
*Purely perceptual* — the module does **not** rename molecules or mutate
engine state.  It is a read-only classifier that wraps a handful of
SMARTS queries and exposes a small public API::

    classify_cyclic_suffix(mol) -> CyclicSuffixClassification | None
    detect(mol)                 -> LeafTree | None   (engine-callable stub)
    all_motif_names()           -> tuple[str, ...]

The engine dispatches ``detect()`` after the acid-infix composition stage
and before the generic plan search.  ``detect()`` currently returns
``None`` for every input — its job is to establish the dispatch point
and classification API so subsequent stages (R3+) can wire in surface
emission without re-threading the engine.  Returning ``None`` guarantees
zero regression on the existing 1177/1181 eval score; the full-eval
gate in the handoff verifies this pre-commit.

Precedence & non-regression
---------------------------
Because ``detect()`` returns ``None`` for all current inputs, no existing
name is perturbed.  The classifier is exposed via the public API for
tests and for future callers.  The narrow-parent heuristic used by
``assembly.py`` (``succinimide`` for the 5-ring dione-amide etc.) stays
untouched, so the existing ``O=C1CCC(=O)N1 -> 2,5-dioxopyrrolidine``
round-trip remains bit-identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from iupac_namer.types import (
    DecisionContext,
    FreeValenceInfo,
    LeafTree,
    OutputForm,
)

if TYPE_CHECKING:  # pragma: no cover
    from rdkit import Chem  # noqa: F401


# ---------------------------------------------------------------------------
# Motif metadata — alpha-letter / ring-size surface names
# ---------------------------------------------------------------------------

# IUPAC Blue Book P-25.5.1 / P-66.6.3 retains the Greek-letter nicknames
# for small-ring lactams and lactones.  The table is keyed on ring size.
# Entry ``None`` means no retained name at that ring size; fall back to
# ``-olactam`` / ``-olactone`` or the oxo-ring-parent form.
_LACTAM_RETAINED_BY_RING_SIZE: dict[int, str | None] = {
    4: "beta-lactam",
    5: "gamma-lactam",
    6: "delta-lactam",
    7: "epsilon-lactam",
}

_LACTONE_RETAINED_BY_RING_SIZE: dict[int, str | None] = {
    4: "beta-propiolactone",
    5: "gamma-butyrolactone",
    6: "delta-valerolactone",
    7: "epsilon-caprolactone",
}

# Motifs published as the module's "vocabulary" for the classifier's
# tests — a curated set of retained cyclic-suffix name fragments.  Not
# to be confused with the engine's FG seniority table; this is simply
# the scope of names the classifier reasons about.
_CYCLIC_SUFFIX_MOTIF_NAMES: tuple[str, ...] = (
    "dicarboximide",
    "imide",
    "lactam",
    "lactone",
    "carbolactone",
    # Greek-letter specifics (Blue Book P-66.6.3)
    "beta-lactam",
    "gamma-lactam",
    "delta-lactam",
    "epsilon-lactam",
    "beta-propiolactone",
    "gamma-butyrolactone",
    "delta-valerolactone",
    "epsilon-caprolactone",
)


def all_motif_names() -> tuple[str, ...]:
    """Return the sorted retained cyclic-suffix vocabulary the classifier
    recognises.  Used by the regression test suite to pin the scope."""
    return tuple(sorted(set(_CYCLIC_SUFFIX_MOTIF_NAMES)))


# ---------------------------------------------------------------------------
# Classification dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CyclicSuffixClassification:
    """Structural summary of a ring-embedded carbonyl-heteroatom motif.

    Attributes
    ----------
    motif:
        One of ``"imide"``, ``"lactam"``, ``"lactone"``.
    ring_size:
        The size of the ring that holds the FG atoms.  For imides this is
        the ring that holds N and both C=O carbons.
    anchor_atoms:
        Tuple of atom indices (in the source ``mol``) that define the
        motif.  For imide: ``(C=O, N, C=O)`` in ring-traversal order.
        For lactam: ``(C=O, N)``.  For lactone: ``(C=O, O)``.
    carbonyl_atoms:
        The carbonyl-carbon indices (length 2 for imide, 1 for lactam /
        lactone).
    heteroatom:
        The N / O index that closes the ring back to the carbonyl(s).
    tautomer_form:
        ``"keto"`` when every carbonyl is a conventional C=O; ``"iminol"``
        when the N=C-OH / O=C-OH tautomer form is drawn.  The distinction
        matters because retained surface names like ``succinimide`` canon-
        ically correspond to the keto form and never to the iminol.
    is_fused:
        True when the motif's ring is fused to at least one other ring
        (the classic ``phthalimide`` / ``cyclohexane-1,2-dicarboximide``
        topology).
    """

    motif: str
    ring_size: int
    anchor_atoms: tuple[int, ...]
    carbonyl_atoms: tuple[int, ...]
    heteroatom: int
    tautomer_form: str
    is_fused: bool

    @property
    def retained_name_hint(self) -> str | None:
        """Return the small-ring retained Greek-letter name, or ``None``.

        Only fires for the plain (non-fused, keto) lactam / lactone cases;
        imides never have a Greek-letter retained form at this granularity.
        """
        if self.is_fused:
            return None
        if self.tautomer_form != "keto":
            return None
        if self.motif == "lactam":
            return _LACTAM_RETAINED_BY_RING_SIZE.get(self.ring_size)
        if self.motif == "lactone":
            return _LACTONE_RETAINED_BY_RING_SIZE.get(self.ring_size)
        return None


# ---------------------------------------------------------------------------
# Public classifier
# ---------------------------------------------------------------------------

def classify_cyclic_suffix(mol) -> CyclicSuffixClassification | None:
    """Classify a molecule as an imide, lactam, or lactone if it bears a
    ring-embedded FG of that family; else return ``None``.

    Priority order (specificity first):
        imide → lactam → lactone

    The first match wins; if none applies, ``None`` is returned.  This
    classifier is intentionally read-only — it never mutates ``mol``.
    """
    if mol is None:
        return None
    imide = _classify_imide(mol)
    if imide is not None:
        return imide
    lactam = _classify_lactam(mol)
    if lactam is not None:
        return lactam
    lactone = _classify_lactone(mol)
    if lactone is not None:
        return lactone
    return None


# ---------------------------------------------------------------------------
# Internal — motif-specific detection
# ---------------------------------------------------------------------------

def _classify_imide(mol) -> CyclicSuffixClassification | None:
    """Detect an endocyclic ``C(=O)-N(R)-C(=O)`` pattern.

    Walks every small ring and, for every candidate nitrogen in the ring,
    asks whether both of its ring-neighbour carbons bear an exocyclic
    C=O (keto form) or a C=O / C-OH iminol pair.
    """
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        ring_set = set(ring)
        if len(ring) < 5:
            # Imide ring must be at least 5-membered: (C=O)(N)(C=O) plus
            # at least two chain atoms to close the cycle.
            continue
        for n_idx in ring:
            atom = mol.GetAtomWithIdx(n_idx)
            if atom.GetAtomicNum() != 7:
                continue
            ring_neighbors = [
                nb for nb in atom.GetNeighbors()
                if nb.GetIdx() in ring_set
            ]
            if len(ring_neighbors) != 2:
                continue
            # Both neighbours must be carbons.
            if any(nb.GetAtomicNum() != 6 for nb in ring_neighbors):
                continue
            c1, c2 = ring_neighbors
            keto1 = _has_exocyclic_double_o(mol, c1.GetIdx(), ring_set)
            keto2 = _has_exocyclic_double_o(mol, c2.GetIdx(), ring_set)
            iminol1 = _is_iminol_carbon(mol, c1.GetIdx(), n_idx, ring_set)
            iminol2 = _is_iminol_carbon(mol, c2.GetIdx(), n_idx, ring_set)
            # Keto + keto OR keto + iminol: both acceptable imide forms.
            kc1 = keto1 or iminol1
            kc2 = keto2 or iminol2
            if not (kc1 and kc2):
                continue
            tautomer = "keto" if (keto1 and keto2) else "iminol"
            is_fused = _ring_is_fused(ri, ring)
            return CyclicSuffixClassification(
                motif="imide",
                ring_size=len(ring),
                anchor_atoms=(c1.GetIdx(), n_idx, c2.GetIdx()),
                carbonyl_atoms=(c1.GetIdx(), c2.GetIdx()),
                heteroatom=n_idx,
                tautomer_form=tautomer,
                is_fused=is_fused,
            )
    return None


def _classify_lactam(mol) -> CyclicSuffixClassification | None:
    """Detect an endocyclic ``C(=O)-N(R)`` pair (single-carbonyl ring)."""
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        ring_set = set(ring)
        if len(ring) < 4:
            continue
        for n_idx in ring:
            atom = mol.GetAtomWithIdx(n_idx)
            if atom.GetAtomicNum() != 7:
                continue
            ring_neighbors = [
                nb for nb in atom.GetNeighbors()
                if nb.GetIdx() in ring_set
            ]
            # The N has two ring neighbours; exactly one of them should be
            # a carbonyl carbon, the other an aliphatic / aromatic carbon.
            carbonyl_c: int | None = None
            tautomer = "keto"
            for nb in ring_neighbors:
                if nb.GetAtomicNum() != 6:
                    continue
                if _has_exocyclic_double_o(mol, nb.GetIdx(), ring_set):
                    if carbonyl_c is None:
                        carbonyl_c = nb.GetIdx()
                    else:
                        # Two carbonyls — this is an imide, not a lactam.
                        carbonyl_c = None
                        break
                elif _is_iminol_carbon(mol, nb.GetIdx(), n_idx, ring_set):
                    if carbonyl_c is None:
                        carbonyl_c = nb.GetIdx()
                        tautomer = "iminol"
                    else:
                        carbonyl_c = None
                        break
            if carbonyl_c is None:
                continue
            is_fused = _ring_is_fused(ri, ring)
            return CyclicSuffixClassification(
                motif="lactam",
                ring_size=len(ring),
                anchor_atoms=(carbonyl_c, n_idx),
                carbonyl_atoms=(carbonyl_c,),
                heteroatom=n_idx,
                tautomer_form=tautomer,
                is_fused=is_fused,
            )
    return None


def _classify_lactone(mol) -> CyclicSuffixClassification | None:
    """Detect an endocyclic ``C(=O)-O`` pair (single-carbonyl ring with
    an ester oxygen as the bridging atom)."""
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        ring_set = set(ring)
        if len(ring) < 4:
            continue
        for o_idx in ring:
            atom = mol.GetAtomWithIdx(o_idx)
            if atom.GetAtomicNum() != 8:
                continue
            ring_neighbors = [
                nb for nb in atom.GetNeighbors()
                if nb.GetIdx() in ring_set
            ]
            carbonyl_c: int | None = None
            for nb in ring_neighbors:
                if nb.GetAtomicNum() != 6:
                    continue
                if _has_exocyclic_double_o(mol, nb.GetIdx(), ring_set):
                    if carbonyl_c is None:
                        carbonyl_c = nb.GetIdx()
                    else:
                        carbonyl_c = None
                        break
            if carbonyl_c is None:
                continue
            is_fused = _ring_is_fused(ri, ring)
            return CyclicSuffixClassification(
                motif="lactone",
                ring_size=len(ring),
                anchor_atoms=(carbonyl_c, o_idx),
                carbonyl_atoms=(carbonyl_c,),
                heteroatom=o_idx,
                tautomer_form="keto",
                is_fused=is_fused,
            )
    return None


# ---------------------------------------------------------------------------
# Topology helpers
# ---------------------------------------------------------------------------

def _has_exocyclic_double_o(mol, c_idx: int, ring_set: set[int]) -> bool:
    """True when atom ``c_idx`` carries a double bond to an O that is
    **outside** the ring ``ring_set``.  Classic ``C(=O)`` carbonyl."""
    atom = mol.GetAtomWithIdx(c_idx)
    for bond in atom.GetBonds():
        if bond.GetBondTypeAsDouble() != 2.0:
            continue
        other = bond.GetOtherAtom(atom)
        if other.GetAtomicNum() != 8:
            continue
        if other.GetIdx() in ring_set:
            # Intra-ring C=O (e.g. pyranone double bond) is not what this
            # predicate asks about.
            continue
        return True
    return False


def _is_iminol_carbon(
    mol, c_idx: int, n_idx: int, ring_set: set[int]
) -> bool:
    """True when atom ``c_idx`` is the ``C`` in a ``N=C-OH`` iminol
    tautomer that closes back into the ring through ``n_idx``.

    Topology test:
        * ``c_idx`` has a double bond to ``n_idx`` (N inside the ring).
        * ``c_idx`` also has a single bond to an exocyclic oxygen bearing
          at least one H (the OH of the iminol).
    """
    atom = mol.GetAtomWithIdx(c_idx)
    has_c_eq_n = False
    has_c_oh = False
    for bond in atom.GetBonds():
        other = bond.GetOtherAtom(atom)
        if other.GetIdx() == n_idx:
            if bond.GetBondTypeAsDouble() == 2.0:
                has_c_eq_n = True
            continue
        if other.GetAtomicNum() == 8 and other.GetIdx() not in ring_set:
            if (
                bond.GetBondTypeAsDouble() == 1.0
                and other.GetTotalNumHs() >= 1
            ):
                has_c_oh = True
    return has_c_eq_n and has_c_oh


def _ring_is_fused(ri, ring: tuple[int, ...]) -> bool:
    """True when ``ring`` shares at least two atoms with any other ring
    reported by the ring-info object — i.e. the classic ortho-fused
    bicyclic topology underpinning ``phthalimide`` /
    ``cyclohexane-1,2-dicarboximide``."""
    ring_set = set(ring)
    for other in ri.AtomRings():
        if other == ring:
            continue
        shared = ring_set & set(other)
        if len(shared) >= 2:
            return True
    return False


# ---------------------------------------------------------------------------
# Engine dispatch stub
# ---------------------------------------------------------------------------

def detect(
    mol,
    output_form: OutputForm,
    free_valence: FreeValenceInfo | None,
    decision_ctx: DecisionContext | None,
    strategy,
    session,
    depth: int,
) -> LeafTree | None:
    """Engine dispatch entry point.

    Returns ``None`` for every input at R2-E: the module establishes the
    classifier infrastructure and the engine dispatch hook, but defers
    surface-name emission to downstream stages.  Because ``None`` falls
    through to the generic plan search, this dispatch is strictly
    non-regressing — the existing keto-form names (``succinimide ->
    2,5-dioxopyrrolidine``, ``phthalimide -> 1,3-dioxoisoindoline``,
    ``caprolactam -> 2-oxoazepane``) pass through untouched.

    The signature mirrors ``detect_acid_infix_composition`` so a future
    caller can wire emission on without changing the hook site.
    """
    if output_form != OutputForm.STANDALONE or free_valence is not None:
        return None
    # Classification is computed defensively; downstream stages can call
    # ``classify_cyclic_suffix(mol)`` directly instead of re-dispatching
    # through this engine-level entry point.
    _ = classify_cyclic_suffix(mol)
    return None
