"""Naming strategy implementations.

A NamingStrategy controls:
  - Which plans are structurally valid (accept_plan)
  - How plans are ranked (score_plan)
  - How eagerly the engine searches (max_plans_hint, good_enough_score)
  - Whether additive nomenclature is used (accept_additive)
  - Retained name policy

v13 spec: ARCHITECTURE_STRATEGY.md
"""
from __future__ import annotations

from iupac_namer.types import (
    NamingPlan, RetainedPlan, SubstitutivePlan, FunctionalClassPlan,
    MultiplicativePlan, RingAssemblyPlan, ReplacementPlan, AdditivePlan,
    InterpretationQuery, PlanComplexity, OutputForm,
)

# IUPAC P-31.1.2.2 / P-14.5: heteroatom element priority for ring numbering.
# Lower value = higher priority (gets lowest locant first).
# O > S > Se > Te > N > P > B > Si > Ge > Sn
_HETERO_ELEMENT_PRIORITY: dict[str, int] = {
    "O":  1,
    "S":  2,
    "Se": 3,
    "Te": 4,
    "N":  5,
    "P":  6,
    "B":  7,
    "Si": 8,
    "Ge": 9,
    "Sn": 10,
}


# ---------------------------------------------------------------------------
# Retained-name stereo capability (Stage 6 R1-I)
# ---------------------------------------------------------------------------
#
# Many retained ring names cannot express ring-fusion or ring-junction
# stereochemistry in the word itself.  The canonical case is ``decalin`` —
# cis-decahydronaphthalene and trans-decahydronaphthalene both map to
# the same retained stem, so emitting the bare word silently drops stereo.
#
# When a stereogenic molecule would take a stereo-opaque retained stem we
# disqualify the retained-shortcut plan at the engine level
# (``_generate_retained_plans``) and in the ring-named-parent enumerator
# (``_name_parent_candidates``).  Full structural naming then takes over
# and (where topologically possible) attaches stereo descriptors via the
# relaxed ``_collect_stereo_descriptors`` guard.
def _steroid_stems_encoding_stereo() -> frozenset[str]:
    """Lazy import avoids circular dependency at module load."""
    try:
        from iupac_namer.natural_products import STEROID_STEMS
        return STEROID_STEMS
    except Exception:
        return frozenset()


_RETAINED_NAMES_ENCODING_STEREO: frozenset[str] = (
    # Stage 6 R2-A: steroid biochemical tetracycle stems (androstane,
    # pregnane, cholestane, …) re-emit via
    # ``iupac_namer.natural_products.steroid.try_steroid_stem_name`` and
    # carry implicit stereo at the ring junctions + optional 5α/5β
    # descriptor.  OPSIN reconstructs the original stereo on round-trip,
    # so these names encode stereo even though the greek-letter
    # descriptor may be absent for the base case.
    _steroid_stems_encoding_stereo()
    | frozenset({
        # Stage 6 R1-E follow-up: flavonoid / diterpene retained pins whose
        # bare IUPAC name (no stereo prefix) round-trips through OPSIN to a
        # stereo-specified canonical SMILES.  Verified via py2opsin — each
        # name maps to a canonical SMILES containing @ tags for implicit
        # (2R,3R) flavanonol / (2S) flavanone / (4aS,4bS,8aS,10aR) abietane
        # stereochemistry.
        "taxifolin",
        "naringenin",
        "aromadendrin",
        "abietamide",
        # Phase 3 P-64.2: 'chalcone' is the retained PIN for
        # (E)-1,3-diphenylprop-2-en-1-one. The retained name carries the E
        # configuration implicitly — OPSIN parses 'chalcone' to a SMILES
        # with /C=C/ trans double-bond stereo, so the bare retained name
        # encodes the E descriptor without an explicit (E)- prefix.
        "chalcone",
        # Phase 12 nucleotide-triphosphate: "adenosine 5'-triphosphate" is
        # the retained PIN for ATP; the biochemical word "adenosine"
        # implicitly carries the (2R,3R,4S,5R)-β-D-ribofuranosyl
        # stereochemistry, and OPSIN parses the bare name to a fully
        # stereo-specified canonical SMILES (verified: ribose @ tags
        # preserved on round-trip; the triphosphate P-centre stereo is
        # symmetric/achiral on both engine output and target).  Without
        # this entry the stereo-drop gate disqualifies the retained
        # shortcut and the engine falls back to a substitutive name with
        # malformed P-centre chirality descriptors that OPSIN cannot parse.
        "adenosine 5'-triphosphate",
    })
)


def _retained_name_encodes_stereo(name: str) -> bool:
    """Return True iff *name* inherently expresses stereo.

    A name is stereo-capable when it either:
      - includes greek-letter stereo locants (5α-, 17β-),
      - includes explicit (R)/(S)/(E)/(Z) descriptors, or
      - is listed in ``_RETAINED_NAMES_ENCODING_STEREO``.
    """
    if not name:
        return False
    lowered = name.lower()
    if "\u03b1" in name or "\u03b2" in name or "alpha-" in lowered or "beta-" in lowered:
        return True
    for descriptor in ("(r)-", "(s)-", "(e)-", "(z)-", "(r,", "(s,", "(e,", "(z,"):
        if descriptor in lowered:
            return True
    # Locant-prefixed stereo descriptors: "(4Z)-", "(2R,3S)-", "(1E,3E)-".
    # The leading paren followed by digits (and optional commas) and an
    # R/S/E/Z descriptor still encodes stereo even though the literal
    # patterns above don't match.
    import re as _re
    if _re.search(r"\(\d+[a-z]?[rsez]\b", lowered):
        return True
    if name in _RETAINED_NAMES_ENCODING_STEREO:
        return True
    return False


def retained_plan_would_drop_stereo(match_name: str, mol) -> bool:
    """Return True when emitting *match_name* for *mol* would silently
    discard stereo information.

    Fires when:
      - the retained name is NOT stereo-capable, AND
      - the molecule has at least one informative stereo marker
        (chiral tag on any atom, or E/Z flag on any bond).

    Bridgehead / ring-fusion tetrahedrals where RDKit stores the chiral
    tag but cannot assign a CIP descriptor (because the bridgeheads are
    CIP-equivalent by symmetry, as in cis-decalin) ALSO count as
    informative — the retained name cannot distinguish cis and trans.
    """
    if mol is None:
        return False
    if _retained_name_encodes_stereo(match_name):
        return False
    try:
        from rdkit import Chem
    except ImportError:
        return False
    chi_unspec = Chem.ChiralType.CHI_UNSPECIFIED
    stereo_none = Chem.BondStereo.STEREONONE
    for atom in mol.GetAtoms():
        if atom.GetChiralTag() != chi_unspec:
            return True
    for bond in mol.GetBonds():
        if bond.GetStereo() != stereo_none:
            return True
    return False


class NamingStrategy:
    """Base strategy. IUPACCanonical is the primary concrete implementation.

    All methods have sensible defaults; subclasses override as needed.
    """

    def interpretation_query(self, mol) -> InterpretationQuery:
        """Return preference hints for interpretation search order."""
        return InterpretationQuery(
            preferred_decomp_types=None,
            preferred_parent_type=None,
            suppress_functional_class=False,
            max_results=3,
        )

    def accept_plan(self, plan: NamingPlan) -> bool:
        """Hard structural validity check.

        Returns False if *plan* violates a rule of this naming system.
        This is a FAST check — no floating-point arithmetic.
        Plans that pass are guaranteed structurally valid under this strategy.
        """
        return True

    def accept_additive(self, additive_groups) -> bool:
        """Strategy veto for additive nomenclature (v13 Issue 20).

        Called before plan search when additive groups are detected.
        Return False to force substitutive naming instead.
        """
        return True

    def score_plan(self, plan: NamingPlan) -> float:
        """Rate the structural commitments encoded in *plan*.

        Higher score = preferred. Only called on plans where accept_plan
        returned True.

        CONTRACT: MUST be pure with respect to (plan, strategy_config).
        MUST NOT read or depend on DecisionContext.
        """
        if isinstance(plan, RetainedPlan):
            return 1_000_000.0   # retained names always preferred over systematic

        if isinstance(plan, SubstitutivePlan):
            score = 0.0
            # Prefer having a PCG (suffix-eligible FG)
            if plan.pcg_type and plan.pcg_instances:
                seniority = plan.pcg_instances[0].get_property("seniority", 9999)
                score += (10_000 - seniority) * 10_000
            # Prefer longer parent chains (P-44.3b)
            score += plan.named_parent.candidate.length * 100
            # Prefer lower locant set for suffix positions (P-14.4)
            if plan.suffix_groups:
                locant_sum = sum(
                    (loc._numeric_value or 0)
                    for sg in plan.suffix_groups
                    for loc in sg.locants
                )
                score -= locant_sum
            # Prefer lower locant set for prefix positions (P-14.4 tie-break).
            # This ensures that when suffix locants are equal (e.g., no PCG in
            # substituent mode), the numbering that gives lower locants to
            # substituents is preferred.  Weight is smaller than suffix scoring
            # so suffix criteria dominate.
            if plan.prefix_assignments:
                from iupac_namer.types import TerminalPrefix, BridgingPrefix
                prefix_locant_sum = 0.0
                for pa in plan.prefix_assignments:
                    if hasattr(pa, "locant") and pa.locant is not None:
                        lv = getattr(pa.locant, "_numeric_value", None)
                        if lv is not None:
                            prefix_locant_sum += lv
                    elif hasattr(pa, "locants"):
                        for loc in pa.locants:
                            lv = getattr(loc, "_numeric_value", None)
                            if lv is not None:
                                prefix_locant_sum += lv
                score -= prefix_locant_sum * 0.01
            return score

        return 0.0

    def max_plans_hint(self, complexity: PlanComplexity | None = None) -> int:
        """Stop generating plans after this many accepted candidates."""
        return 50

    def good_enough_score(self) -> float:
        """If a plan scores at or above this value, stop searching immediately.

        Retained names score 1_000_000, so this threshold means "stop if we
        found a retained name". Set to float('-inf') to disable early exit.
        """
        return 1_000_000.0   # stop if we found a retained name

    def retained_name_policy(self) -> str:
        """ALWAYS_IF_AVAILABLE, NEVER, or PREFER."""
        return "ALWAYS_IF_AVAILABLE"

    def cache_key(self) -> str:
        """Identity string for memoisation. Same key = same naming decisions."""
        return "base"


class IUPACCanonical(NamingStrategy):
    """IUPAC canonical naming strategy — 2013 Blue Book preference rules.

    Scoring uses magnitude bands (v13 E1) so higher-priority criteria
    always dominate lower-priority ones:

      Band 4: PCG seniority (P-65)      × 10_000
      Band 3: Parent selection (P-44)   × 100
      Band 2: Numbering quality         × 1
      Band 1: Naming method/style       × 0.01

    Retained names score 1_000_000 (above all systematic names).
    """

    def interpretation_query(self, mol) -> InterpretationQuery:
        return InterpretationQuery(
            preferred_decomp_types=None,
            preferred_parent_type=None,
            suppress_functional_class=False,
            max_results=1,   # IUPACCanonical: one best interpretation
        )

    # Map FC subtype to the FG type it covers. An FC plan is only accepted
    # if no FG with strictly higher seniority (lower seniority number) than
    # the covered FG is present in the interpretation.
    _FC_SUBTYPE_TO_FG: dict[str, str] = {
        "ester": "ester",
        "carbamate": "carbamate",
        "acyl_isothiocyanate": "acyl_isothiocyanate",
        "thioester": "thioester",
        "thionoester": "thionoester",
        "dithioester": "dithioester",
        "thionocarbamate": "thionocarbamate",
        "dithiocarbamate": "dithiocarbamate",
        "carbamothioate": "carbamothioate",
        "symmetric_diester": "ester",
        "polyester": "ester",
    }

    def accept_additive(self, additive_groups) -> bool:
        """Gate P-oxide additive nomenclature.

        Per IUPAC P-64.4, ``phosphane oxide`` (additive) is the PIN for
        trialkyl P=O — e.g. ``trimethylphosphane oxide`` for
        ``CP(=O)(C)C``.  But for phosphate-style P=O where every
        non-``=O`` neighbour is an O/N/S linker (ester / amide / thio
        family), the substitutive form ``tri(methoxy)(oxo)phosphane``
        and friends is the PIN — additive would compete with the
        phosphoric-acid retained ester family and OPSIN's parsing of
        ``trimethoxyphosphane oxide`` is fragile when P is a substituent
        on an organic parent.

        Rule: accept P-oxide additive only when ALL non-``=O`` heavy
        neighbours of the P centre are *acyclic* carbons.  Aryl-/ring-
        bound P falls back to the substitutive form because the ring
        would win as parent (the trailing "oxide" otherwise attaches
        to the wrong stem).  Phosphate / phosphoric-amide / thio
        analogues stay substitutive because at least one neighbour is
        non-carbon.

        N-oxide additive nomenclature is always accepted (pyridine
        1-oxide, trimethylamine N-oxide, etc.).
        """
        for ag in additive_groups:
            if ag.get("center_element") != "P":
                continue
            non_oxide = ag.get("non_oxide_neighbor_elements")
            if non_oxide is None:
                # Backwards-compat: detection didn't enrich this group;
                # fall back to the conservative reject.
                return False
            # Only accept additive if every non-oxide neighbour is carbon.
            if not all(elem == "C" for elem in non_oxide):
                return False
            # And only when none of those carbon neighbours sit inside
            # a ring — otherwise the ring will win as parent and the
            # trailing "oxide" would attach to the wrong scaffold (e.g.
            # triphenylphosphine oxide must take the substitutive form
            # ``[(oxo)diphenylphosphanyl]benzene`` because benzene is
            # the parent).
            if ag.get("non_oxide_neighbor_aromatic"):
                return False
            if ag.get("non_oxide_neighbor_in_ring"):
                return False
        return True

    def accept_plan(self, plan: NamingPlan) -> bool:
        """Hard structural reject for FC plans that violate IUPAC rules."""
        if isinstance(plan, FunctionalClassPlan):
            # Hard reject: intramolecular FC (lactones, cyclic anhydrides)
            if plan.decomposition.intramolecular:
                return False

            # Hard reject: per IUPAC P-73 (cation nomenclature) outranks
            # P-66 (ester functional class) in suffix selection.  When any
            # piece of an ester FC decomposition contains a ring-embedded
            # [N+] atom, that piece must take a cation suffix (e.g. -ium)
            # which cannot be expressed via FC ester syntax.  Substitutive
            # naming with the ring-cation parent is the only correct route.
            pieces = plan.decomposition.pieces or ()
            for piece in pieces:
                if piece.mol is None:
                    continue
                for atom in piece.mol.GetAtoms():
                    if (atom.GetSymbol() == "N"
                            and atom.GetFormalCharge() == 1
                            and atom.IsInRing()):
                        return False

            # Hard reject: FC is only valid when the covered FG is the
            # most senior suffix-eligible FG in the molecule. If a more
            # senior FG is present (e.g., a free -COOH alongside an ester),
            # substitutive naming must be used.
            subtype = plan.decomposition.subtype
            covered_fg_type = self._FC_SUBTYPE_TO_FG.get(subtype)
            if covered_fg_type is None:
                return False  # unsupported FC subtype
            interp = plan.interpretation
            if interp is None:
                return False
            # Seniority of the FC-covered FG (from any instance of that type)
            covered_seniority: int | None = None
            for fg in interp.fgs:
                if fg.type == covered_fg_type:
                    covered_seniority = fg.get_property("seniority", 9999)
                    break
            if covered_seniority is None:
                return False
            # Reject if there is a strictly more senior FG (lower seniority
            # number) than the FC-covered type.
            for fg in interp.fgs:
                if fg.type == covered_fg_type:
                    continue
                if not fg.suffix_eligible:
                    continue
                sen = fg.get_property("seniority", 9999)
                if sen < covered_seniority:
                    return False
        return True

    def score_plan(self, plan: NamingPlan) -> float:
        """Score using magnitude bands (v13 E1)."""
        match plan:
            case RetainedPlan():
                return 1_000_000.0   # always preferred over systematic

            case SubstitutivePlan():
                score = 0.0
                # Band 4: PCG seniority (0–30) × 10_000
                score += self._pcg_seniority_score(plan.pcg_type, plan.pcg_instances) * 10_000
                # Band 4 (cation): ring-embedded [N+] on parent backbone
                # gets cation-as-PCG seniority per P-73.  Because IUPAC P-73
                # (cations) outranks P-66 (esters), the bonus must lift the
                # substitutive plan above the FC ester score (400_000).  Only
                # active when a ring N+ is actually claimed by the parent;
                # other plans (no ring N+, or ring N+ as substituent) stay
                # at their normal seniority band.
                #
                # P-43.2 / P-41 guard: a parent candidate bearing a senior
                # acid-class PCG (carboxylic acid, sulfonic acid, etc. —
                # Blue Book class 7, seniority < 800) outranks a bare onium
                # ring in principal-parent selection.  In that case the
                # acid-bearing plan must win; suppress the cation bonus here
                # so the acid-PCG plan (score ~293k) beats the cation plan.
                # This implements P-43.2: "a chain bearing the principal
                # characteristic group as suffix outranks an onium cation
                # ring that expresses only the charge."
                if plan.parent_ring_cation_atoms:
                    interp = plan.interpretation
                    has_senior_acid_pcg = (
                        interp is not None
                        and any(
                            fg.suffix_eligible
                            and fg.get_property("seniority", 9999) < 800
                            for fg in interp.fgs
                        )
                    )
                    if not has_senior_acid_pcg:
                        score += 500_000.0
                # Band 3: parent selection (0–99) × 100
                score += self._parent_selection_score(plan) * 100
                # Band 2: numbering quality
                score += self._numbering_score(plan) * 1
                # Band 2 (ring-seniority): a retained ring system name is senior
                # to a systematic von Baeyer construction of the SAME ring
                # system (P-31.1.4.3 / ring seniority; P-31.1.3 prefers retained
                # names where they exist).  The plain method-style preference
                # (Band 1, ×0.01) is too weak to express this: a retained ring's
                # FIXED IUPAC numbering can put heteroatoms at higher locants
                # than the von Baeyer competitor's free numbering, and that
                # heteroatom-locant penalty lives in Band 2 (×1, weight 0.4 per
                # locant) — so it can swamp the +1.0 Band-1 retained credit and
                # let an (often unparseable) von Baeyer name win.  Corrin is the
                # canonical case: its mandated N-locants 21–24 lose to the von
                # Baeyer 20–23 by ~1.4 in Band 2.  Promote the retained-vs-
                # systematic preference to Band-2 magnitude so it dominates the
                # heteroatom-locant differential, while staying well below the
                # Band-3 minimum step (0.1 parent-atom × 100 = 10) so it can
                # NEVER override parent-selection seniority (a genuinely more
                # senior / longer ring or chain still wins).  Gated on
                # naming_method == "retained" AND a ring parent, so it only
                # strengthens the already-correct retained preference and is
                # constant across all retained plans (never reorders them).
                score += self._retained_ring_seniority_score(plan.named_parent) * 1
                # Band 1: naming method style × 0.01
                score += self._naming_method_score(plan.named_parent) * 0.01
                return score

            case FunctionalClassPlan():
                # FC plans that passed accept_plan score above substitutive
                return 400_000.0

            case MultiplicativePlan():
                return 450_000.0

            case RingAssemblyPlan():
                return 420_000.0

            case ReplacementPlan():
                score = 0.0
                score += self._pcg_seniority_score(
                    plan.pcg.type if plan.pcg else None,
                    (plan.pcg,) if plan.pcg else (),
                ) * 10_000
                return score

            case _:
                return 0.0

    def max_plans_hint(self, complexity: PlanComplexity | None = None) -> int:
        if complexity is None:
            return 20
        est = complexity.estimated_plans
        if est <= 20:
            return 20
        return min(80, max(20, est // 2))

    def good_enough_score(self) -> float:
        return 1_000_000.0   # stop on retained name

    def cache_key(self) -> str:
        return "iupac"

    # ------------------------------------------------------------------
    # Score components (v13 E1)
    # ------------------------------------------------------------------

    def _pcg_seniority_score(self, pcg_type: str | None, pcg_instances) -> float:
        """P-65.1 seniority score. Returns 0.0–30.0.

        None (no PCG) returns 0. Higher = more senior FG class.

        Having ANY PCG (suffix group) scores much higher than having no PCG.
        Among PCG types, lower seniority number = more senior = higher score.
        Actual seniority values in functional_groups.json range from ~700–9999
        (lower = more senior per P-65.1 Blue Book table).
        """
        if not pcg_type or not pcg_instances:
            return 0.0
        try:
            fg = pcg_instances[0] if pcg_instances else None
            if fg is None:
                return 0.0
            seniority = fg.get_property("seniority", 9999)
            # Any PCG present gets a base score of 20.0.
            # More senior classes (lower seniority number) get up to 10 extra pts.
            # seniority 700 (most senior in our data) → ~20 + 10*(1-700/10000) ≈ 29.3
            # seniority 2000 → ~20 + 10*(1-2000/10000) = 28.0
            # seniority 9999 → ~20 + 10*(1-9999/10000) ≈ 20.0
            # All are >> 0 (no PCG), which is 0.0.
            seniority_bonus = max(0.0, 10.0 * (1.0 - seniority / 10_000.0))
            return 20.0 + seniority_bonus
        except Exception:
            return 20.0  # default: any PCG is strongly preferred

    def _parent_selection_score(self, plan: SubstitutivePlan) -> float:
        """P-44 criteria in strict priority order. Returns 0.0–99.0.

        Priority order (each band dominates all lower bands):
          Band 4 (×1.0): max principal characteristic groups on parent (P-44.3a)
          Band 3 (×0.1): max parent chain length (P-44.3b) — DOMINATES locants
          Band 2 (×0.001): max number of multiple bonds (P-44.3c)
          Band 1 (×0.0001): max number of substituents (P-44.3d)

        Note: lowest-locant criterion (P-14.5) is handled in _numbering_score,
        NOT here.  Parent chain selection (P-44) is always decided BEFORE
        numbering direction.  Mixing the two causes shorter chains to
        "win" via low-locant bonuses when they shouldn't.
        """
        score = 0.0
        candidate = plan.named_parent.candidate

        # Band 4 (highest priority): PCG anchors on parent chain (P-44.3a)
        # Give 2.0 pts for terminal FG (anchor directly IN parent) or 1.0 for
        # nonterminal FG (anchor BONDED TO parent, not in it).
        # Unreachable suffixes (no locant) score 0.
        #
        # For RING parents: IUPAC P-44.3a treats a ring + exo-COOH nonterminal
        # suffix the same as a chain terminal suffix in terms of PCG count.  A
        # ring (≥3 atoms, e.g. benzene=6 → +0.6 in band 3) combined with a
        # nonterminal COOH (1.0) beats a 1-carbon chain (0.1) with terminal COOH
        # (2.0): ring = 1.0+0.6 = 1.6 < 2.1.  To fix this, we give ring nonterminal
        # the same 2.0 score as terminal — the chain-length band 3 then correctly
        # selects the ring (6 atoms) over the 1C chain.  Ring candidates have
        # candidate.type != "chain".
        #
        # For CHAIN parents: keeping terminal=2.0 > nonterminal=1.0 ensures that
        # dicarboxylic acids prefer the chain where BOTH COOHs are terminal (score
        # 4.0) over a chain where one is nonterminal (score 3.0).  This is correct
        # per P-44.3a: the chain with the most PCGs in terminal position is best.
        #
        # Max expected: ~4 instances → max 8.0 pts (range 0–8 in band 4)
        if plan.suffix_groups:
            parent_atom_set = candidate.atom_indices
            is_ring_parent = (candidate.type != "chain"
                              and candidate.ring_system is not None)
            # Only the diazene (N=N) heteroatom-chain parent gets ring-exo-
            # equivalent scoring for nonterminal PCGs.  Hydrazine (N-N) and
            # the others remain at the chain-nonterminal +1.0 — that keeps
            # NNC=O ("formohydrazide" / "hydrazinecarbaldehyde") naming
            # correct, where a different (smaller) parent is preferred.
            is_diazene_chain_parent = (
                candidate.type == "heteroatom_chain"
                and getattr(candidate, "element", None) == "N=N"
            )
            for sg in plan.suffix_groups:
                if not sg.locants:
                    continue  # unreachable: no score
                if sg.fg.anchor in parent_atom_set:
                    score += 2.0  # terminal: anchor directly on parent
                elif is_ring_parent:
                    score += 2.0  # ring nonterminal: exo-COOH/CN on ring
                elif is_diazene_chain_parent:
                    # diazene (N=N) parent with an exo-PCG anchor (e.g.
                    # -C(=O)NH2 bonded to one of the heteroatoms, naming
                    # "diazene-1-carboxamide") behaves like a ring-exo PCG:
                    # the PCG carbon is one bond outside the parent skeleton
                    # but diazene is the senior parent per IUPAC for N=N
                    # backbones.  Score at the same +2.0 as ring-exo to
                    # prevent a 1-carbon (methanamide) parent from beating
                    # the diazene parent on band-4.
                    score += 2.0
                else:
                    score += 1.0  # chain nonterminal: anchor bonded to chain
        elif (
            plan.named_parent.naming_method == "retained"
            and plan.pcg_type
            and plan.pcg_instances
            and candidate.ring_system is not None
        ):
            # A retained-name parent whose stem already encodes the PCG (e.g.
            # "4-pyrazolone" encodes the C=O via extra_atom_indices) has
            # suffix_groups=() but is still expressing the PCG on-parent.
            # Credit it at the same band-4 level as an explicit on-parent
            # suffix, so the retained plan competes fairly with a systematic
            # plan of the same ring that emits the PCG as a -one suffix.
            #
            # Guard: only credit when the FG is *fully claimed* by the parent
            # — i.e. every atom of the FG (anchor + =O etc.) lies inside
            # candidate.atom_indices.  Otherwise this branch would give a
            # free +2.0 to any retained-ring plan with a PCG hanging off it
            # (e.g. acetylbenzene would beat 1-phenylethanone by giving
            # benzene a spurious on-parent PCG credit).
            parent_atom_set = candidate.atom_indices
            # Pre-composed retained-ring detection (Phase 8 — P-31.1):
            # data-file retained names like ``5-pyrazolone`` / ``urazol``
            # / ``phthalhydrazide`` embed a suffix-form ending in the
            # stem itself, which lexically locks the suffix slot so a
            # separate PCG (amine, alcohol) cannot be glued on.  The
            # engine routes such PCGs to the prefix list, leaving
            # ``suffix_groups=()``.  When the PCG is structurally on
            # the ring (anchor in ring_system.atom_indices) — exactly the
            # ring-exo / ring-terminal positions a systematic ring parent
            # would credit at +2.0 in the suffix branch above — credit
            # the retained plan equivalently so the pre-composed retained
            # form (e.g. ``3-amino-5-pyrazolone``) competes fairly with
            # the systematic alternative (``5-oxo-1H-pyrazol-3-amine``).
            ring_atom_set = (
                candidate.ring_system.atom_indices
                if candidate.ring_system is not None
                else frozenset()
            )
            is_precomposed_retained = getattr(
                plan.named_parent,
                "precomposed_retained_no_suffix",
                False,
            )
            for fg in plan.pcg_instances:
                if fg.atoms <= parent_atom_set:
                    score += 2.0
                elif (
                    is_precomposed_retained
                    and bool(fg.atoms & ring_atom_set)
                ):
                    # Ring-exo PCG on a pre-composed retained ring: at
                    # least one FG atom (typically the carbon bearing
                    # the substituent) sits in the ring while the
                    # anchor sits outside.  Mirror the ring-nonterminal
                    # +2.0 that the suffix branch applies to systematic
                    # ring parents (P-44.3a).
                    score += 2.0

        # Band 3: parent chain length (P-44.3b)
        # Each chain atom contributes 0.1 pts.
        # One extra carbon atom (+0.1) always beats any band-2 or band-1 bonus.
        # A 40-atom chain → 4.0 pts; a 3-atom chain → 0.3 pts.
        #
        # Exception — heteroatom parent hydrides (P, Si, B, As, Ge, Sn):
        # IUPAC P-68 mandates that the heteroatom is always the parent; all
        # carbon chains hanging off it are substituents.  Give a fixed bonus
        # of 50.0 that beats any realistic carbon chain (< 500 atoms × 0.1).
        if candidate.type == "heteroatom_center":
            score += 50.0
        elif candidate.type == "heteroatom_chain":
            # IUPAC mandates N-N (hydrazine) and S-S (disulfane) as parent
            # when the molecule is a substituted hydrazine/disulfane.
            # Give a fixed bonus of 0.9 — large enough to beat any pure carbon
            # chain (length × 0.1 per atom, so 0.9 beats chains up to 9 atoms),
            # but less than the PCG-on-parent advantage (+2.0) so that a carbon
            # chain or ring with a PCG anchor ON/BONDED-TO it still wins over an
            # N-N chain where the PCG anchor is only BONDED to the N-N.
            score += 0.9
        else:
            score += candidate.length * 0.1

        # Ring systems score via their length × 0.1 (same as chains).
        # With ring nonterminal PCG scoring 2.0 (same as terminal), a ring
        # with a nonterminal COOH suffix (ring=6 → +0.6) beats a 1C chain with a
        # terminal COOH (chain=1 → +0.1): 2.6 > 2.1.  Correct per P-44.3a+b.
        # heteroatom_chain parents (bonus 0.9) beat rings of < 9 atoms without a
        # PCG; rings with a PCG nonterminal suffix score 2.0+length×0.1 > 0.9.

        # P-44.4.1.2 ring-class seniority: heterocyclic ring systems are
        # senior to carbocyclic ring systems of the same skeletal size.
        # Without this, a phenyl-pyridine biaryl picks benzene as parent
        # (both rings size 6, length tie) and emits "(pyridin-2-yl)benzene"
        # rather than the spec PIN "2-phenylpyridine".  The bonus is
        # smaller than one chain atom (0.1) so it never overrides P-44.1
        # (max skeletal atoms / chain length).
        if (candidate.type != "chain"
                and candidate.type != "heteroatom_center"
                and candidate.type != "heteroatom_chain"
                and candidate.ring_system is not None):
            rs = candidate.ring_system
            has_hetero = bool(getattr(rs, "heteroatoms", None))
            if has_hetero:
                score += 0.05

            # P-44.1.2: a ring or ring system is senior to a chain when
            # the molecule has no principal characteristic group (PCG)
            # to break the tie via on-parent FG seniority.  Engine
            # previously emitted ``1-phenylnonane`` for C6H5-(CH2)8-CH3
            # because the 9-carbon chain (length × 0.1 = 0.9) outscored
            # the 6-atom benzene ring (0.6).  Apply a fixed +5.0 bonus
            # that beats any realistic chain-length advantage but only
            # in the no-PCG case, so PCG-on-chain plans (which carry the
            # +2.0 band-4 anchor credit) still beat ring-as-parent for
            # genuine chain-PCG molecules like decanoic acid.
            if not plan.suffix_groups and not plan.pcg_instances:
                score += 5.0

        # Band 2: number of multiple bonds in chain (P-44.3c)
        # Max ~10 double/triple bonds → max 0.01 pts.
        if plan.unsaturation:
            score += len(plan.unsaturation) * 0.001

        # Band 1: number of substituents (P-44.3d)
        # Max ~10 substituents → max 0.001 pts.
        score += len(plan.prefix_assignments) * 0.0001

        return min(99.0, score)

    def _numbering_score(self, plan: SubstitutivePlan) -> float:
        """Reward numberings that give lower locants (P-14.5, P-14.4).

        Priority order (IUPAC P-31.1.2.2 + P-14.5):
          0. For heterocyclic rings: lowest locant set to heteroatoms (FIRST)
          1. Lowest locant set to suffix groups (PCG)
          2. Lowest locant set to detachable prefixes (substituents)
          3. First point of difference in the locant set

        Returns a score: lower locant sum → higher value.
        Heteroatom locants dominate suffix locants; suffix locants dominate prefix.

        MAGNITUDE CONSTRAINT: This band-2 score must stay well below 10 in
        absolute value so that the band-3 minimum step (0.1 chain atom × 100 =
        10) always dominates the numbering preference.  This enforces IUPAC P-44
        (longest chain first) before P-14.5 (lowest locants).

        Sub-band magnitudes within band-2 (×1 multiplier from caller):
          - Heteroatom locants: ×0.4 (dominates suffix + prefix combined)
          - Suffix locants: ×0.1 (max single locant ~40 → -4.0)
          - Prefix locants: ×0.01 (max sum ~100 → -1.0)
        """
        # --- Heteroatom locant score (P-31.1.2.2) ---
        # For monocyclic heterocyclic rings, heteroatoms must be given the
        # lowest possible locants.  Within that, higher-priority elements
        # (O > S > N, by IUPAC P-14.5) get lower locants first.
        #
        # Scoring: group heteroatoms by element priority and compute the
        # weighted sum of locants, with exponentially decreasing weight per
        # priority group.  Same-element atoms (e.g. two Ns in piperazine)
        # share the same weight and their locants are summed — so both
        # N=1,N=4 and N=4,N=1 score equally (both sum to 5).
        #
        # Example — morpholine (O=1,N=4 vs N=1,O=4):
        #   O=1,N=4: O_sum=1 × 0.4  + N_sum=4 × 0.04  = 0.56  (BETTER)
        #   N=1,O=4: O_sum=4 × 0.4  + N_sum=1 × 0.04  = 1.64  (WORSE)
        heteroatom_score = 0.0
        ring_system = plan.named_parent.candidate.ring_system
        if ring_system is not None and ring_system.heteroatoms:
            atom_to_loc = plan.numbering.atom_to_locant
            # Group heteroatoms by element priority
            # (same element → same group → summed equally)
            groups: dict[int, float] = {}
            for hp in ring_system.heteroatoms:
                if hp.atom_idx in atom_to_loc:
                    loc_val = atom_to_loc[hp.atom_idx]._numeric_value
                    if loc_val:
                        prio = _HETERO_ELEMENT_PRIORITY.get(hp.element, 99)
                        groups[prio] = groups.get(prio, 0.0) + loc_val
            if groups:
                # Assign exponentially decreasing weight per priority level
                weight = 0.4
                weighted_locant_sum = 0.0
                for prio in sorted(groups):
                    weighted_locant_sum += groups[prio] * weight
                    weight *= 0.1
                heteroatom_score = -weighted_locant_sum

        # Suffix locants (highest priority after heteroatom constraint)
        suffix_locants = sorted(
            loc._numeric_value or 0
            for sg in plan.suffix_groups
            for loc in sg.locants
        )
        # Unsaturation locants (P-31.1.2.2: lowest locants to double/triple bonds
        # after suffix FG, before detachable prefixes)
        unsat_locants_raw: list[int] = []
        for inf in plan.unsaturation:
            for loc in inf.locants:
                v = loc._numeric_value or 0
                unsat_locants_raw.append(v)

        # For systematic monocyclic rings with stored bond pairs, compute the
        # ring double-bond locants from the current numbering (P-31.1.2.2).
        # This allows the strategy to correctly prefer the direction with lowest
        # locants for the ring double bonds, consistent with IUPAC rules.
        np = plan.named_parent
        if np.ring_unsaturation_bonds:
            from iupac_namer.ring_naming.monocyclic import (
                compute_ring_unsaturation_locants_from_numbering,
            )
            ring_dbl, ring_tri = compute_ring_unsaturation_locants_from_numbering(
                np.ring_unsaturation_bonds,
                plan.numbering.atom_to_locant,
            )
            unsat_locants_raw.extend(ring_dbl)
            unsat_locants_raw.extend(ring_tri)

        unsat_locants = sorted(unsat_locants_raw)
        # Prefix locants (lower priority within band 2)
        # TerminalPrefix has .locant (single Locant or None)
        # BridgingPrefix has .locants (tuple of Locant)
        prefix_locants_raw: list[int] = []
        prefix_prime_count = 0  # number of detachable-prefix locants carrying a "'" suffix
        for pa in plan.prefix_assignments:
            if hasattr(pa, 'locant') and pa.locant is not None:
                v = pa.locant._numeric_value
                if v:
                    prefix_locants_raw.append(v)
                    if "'" in getattr(pa.locant, 'suffix', ''):
                        prefix_prime_count += 1
            elif hasattr(pa, 'locants'):
                for loc in pa.locants:
                    if loc is not None:
                        v = loc._numeric_value
                        if v:
                            prefix_locants_raw.append(v)
                            if "'" in getattr(loc, 'suffix', ''):
                                prefix_prime_count += 1
        prefix_locants = sorted(prefix_locants_raw)

        if not suffix_locants and not unsat_locants and not prefix_locants and not heteroatom_score:
            return 0.0

        # Suffix: ×0.1 per locant unit (max single locant ~40 → -4.0)
        suffix_score = -sum(suffix_locants) * 0.1
        # Unsaturation: ×0.05 per locant unit — after suffix, before prefix
        unsat_score = -sum(unsat_locants) * 0.05
        # Prefix: ×0.01 per locant unit (max sum ~100 → -1.0)
        prefix_score = -sum(prefix_locants) * 0.01
        # Prime tiebreak: when the numeric locant is equal (e.g. 2 vs 2'),
        # unprimed ring-assembly locants are preferred (P-14.5 first-point-of-
        # difference: unprimed is lower than primed).  Weight is well below
        # any other sub-band so it only breaks ties.
        prefix_prime_score = -prefix_prime_count * 0.00001

        # P-45.5 alphanumerical-locant tiebreak: when the prefix-locant set
        # is symmetric (both numbering directions give the same sorted
        # locants), the alphabetically-first detachable prefix takes the
        # lower locant.  Use a coarse signature for the prefix name
        # because we don't run the full substituent-naming recursion at
        # plan-score time.  Single-atom halogen / hydroxyl / amino / etc.
        # cases (the typical tiebreak shape — RBrCl-style chains) are
        # handled directly from the substituent atom's element.  Compound
        # substituents share a fallback signature ("z" — sorted last).
        # Weight (0.0001) is well below the prime tiebreak so it only
        # fires when no higher-priority sub-band breaks the tie.
        alpha_first_score = 0.0
        try:
            named_prefix_locs: list[tuple[str, int]] = []
            for pa in plan.prefix_assignments:
                # Use the FG type as the alphabetic key when available —
                # for halogens / oxo / hydroxy / amino / cyano / nitro
                # the type IS the prefix name ("bromo", "chloro", "oxo",
                # ...).  Compound substituents (ring/chain carved
                # fragments) don't have an FG; fall back to "z" so they
                # are sorted last.
                fg = getattr(pa, "fg", None)
                key = fg.type if fg is not None and isinstance(fg.type, str) else "z"
                if hasattr(pa, "locant") and pa.locant is not None:
                    v = pa.locant._numeric_value
                    if v:
                        named_prefix_locs.append((key, v))
                elif hasattr(pa, "locants"):
                    for loc in pa.locants:
                        if loc is None:
                            continue
                        v = loc._numeric_value
                        if v:
                            named_prefix_locs.append((key, v))
            if named_prefix_locs:
                _alpha_first_key = min(t[0] for t in named_prefix_locs)
                _alpha_locs = [
                    v for k, v in named_prefix_locs if k == _alpha_first_key
                ]
                if _alpha_locs:
                    alpha_first_score = -min(_alpha_locs) * 0.0001
        except Exception:
            alpha_first_score = 0.0

        return (heteroatom_score
                + suffix_score
                + unsat_score
                + prefix_score
                + prefix_prime_score
                + alpha_first_score)

    def _naming_method_score(self, named_parent) -> float:
        """Preference ordering for naming methods (Band 1, × 0.01).

        Retained names are preferred over systematic for ring parents
        (P-31.1.3: use retained names where they exist).
        For chain parents, systematic is preferred (deterministic naming).
        """
        rs_type = (named_parent.candidate.ring_system.type
                   if named_parent.candidate.ring_system else None)
        is_ring = rs_type is not None

        if is_ring:
            # For ring parents: retained > HW > systematic
            # Retained ring names (isoxazole, pyridine, etc.) are IUPAC-preferred
            # over systematic HW names (P-31.1.3). The score gap must be large
            # enough to beat any realistic band-2 locant difference (up to ~4.0
            # for suffix locants on a 10-membered ring). With ×0.01 multiplier,
            # a gap of 50.0 here → 0.5 advantage, which beats any suffix locant
            # difference so retained names are unconditionally preferred.
            method_ranks = {
                "retained": 100.0,
                "hantzsch_widman": 50.0,
                # Skeletal a-prefix replacement (P-23.2.5 / P-31.1.3): standard
                # for saturated heterocyclic monocycles too large for HW.
                # Must rank above carbocyclic "systematic" so a heterocycle is
                # never silently named as a carbocycle.
                "replacement": 40.0,
                # spiro_polycyclic (P-24.5): a polycyclic-partnered spiro name
                # is preferred over plain von_baeyer when the ring system has
                # an articulation atom that truly splits it into spiro partners
                # (see name_polycyclic_spiro).  Ranked well above von_baeyer so
                # the spiro form wins even when the VB numbering happens to
                # give marginally lower substituent locants on the bare
                # carbocyclic skeleton.  The bare VB skeleton is actually
                # wrong for these systems (it ignores the spiro topology);
                # we only emit name_polycyclic_spiro when the split is
                # architecturally valid.
                "spiro_polycyclic": 40.0,
                # Stage 5: methylenedioxy-bridge on retained polycyclic base
                # (e.g. 16,17-methylenedioxy-hexadecahydro-1H-cyclopenta[a]
                # phenanthrene).  Ranked above spiro_polycyclic (40) so a
                # genuine methylenedioxy-bridged steroid emits the canonical
                # IUPAC bridge form instead of an articulation-split polyspiro
                # name; still below retained (100) / HW (50) so benzodioxol /
                # [1,3]dioxol retained fused names on monocyclic bases always
                # win.  Module: ring_naming/methylenedioxy_bridge.py.
                "methylenedioxy_bridge": 45.0,
                # Stage 3 fused-hetero with a hydro- prefix (hexahydro-
                # [1,3]dioxolo[4,5-b]benzene etc.).  Ranked above von_baeyer
                # so saturated dioxolo/dithiolo heterocycles prefer the
                # fused systematic form over a bare VB decomposition.  Still
                # below HW/replacement/retained so those always win for
                # skeletons that have a better name.
                "fused_hetero_hydro": 5.0,
                "von_baeyer": 1.2,
                "spiro_systematic": 1.0,
                "systematic": 0.9,   # systematic ring name is last resort
                "heteroatom_hydride": 0.8,
            }
        elif named_parent.candidate.type in ("heteroatom_center", "heteroatom_chain"):
            # For heteroatom-center/chain parents (phosphane, silane, hydrazine,
            # disulfane, etc.): heteroatom_parent method is the only valid method;
            # give it a slight boost above chain "systematic" so the heteroatom
            # parent wins over a same-length carbon chain.
            method_ranks = {
                "heteroatom_parent": 1.5,
            }
        else:
            # For chain parents: systematic is always preferred
            method_ranks = {
                "systematic": 1.0,
                "retained": 0.9,
                "heteroatom_hydride": 0.6,
            }
        return method_ranks.get(named_parent.naming_method, 0.5)

    # Band-2-magnitude credit for a retained ring system name over the
    # systematic von Baeyer construction of the same ring system.  See the
    # call site in score_plan for the full rationale (P-31.1.4.3 / P-31.1.3).
    #
    # Magnitude 5.0 (× Band-2 weight 1) is chosen to sit ABOVE the maximum
    # plausible heteroatom-locant differential between a retained ring's fixed
    # numbering and a systematic competitor (corrin's is ~1.4; even a heavily
    # hetero-substituted macrocycle stays under a few units at the 0.4 weight)
    # yet BELOW the Band-3 minimum step (0.1 parent-atom × 100 = 10), so it can
    # never override parent-selection seniority (P-44).
    _RETAINED_RING_SENIORITY_CREDIT: float = 5.0

    def _retained_ring_seniority_score(self, named_parent) -> float:
        """Return the retained-ring seniority credit (Band 2) for this parent.

        Scoped to retained names whose ring system's PRIMARY classification is
        ``bridged`` (RingSystem.type == "bridged").  von Baeyer nomenclature
        (P-23) is the systematic method specifically for bridged ring systems,
        so a primary-bridged retained ring name (corrin, etc.) is exactly what
        competes with — and is senior to (P-31.1.4.3) — a von Baeyer
        construction of the same skeleton.  The credit is what lets the
        retained bridged name overcome the Band-2 heteroatom-locant penalty
        its FIXED numbering may incur.

        Gated on the PRIMARY ``type`` only, NOT ``alternate_type``: ordinary
        fused systems (acridan, etc.) are classified ``type="fused"`` with an
        ``alternate_type="bridged"`` flag, but their PIN is the systematic
        form (acridan is a pin-ineligible general-nomenclature retained name
        whose PIN is the von Baeyer azatricyclo construction), so they must NOT
        receive this Band-2 promotion.  Likewise monocyclic retained rings
        (imidazole / triazole — whose systematic competitors are HW /
        replacement, and whose anion / substituent-ring locants would be
        wrongly perturbed) stay governed by the Band-1 method preference.
        Constant across all qualifying plans, so it never reorders retained
        bridged plans relative to each other.
        """
        if named_parent.naming_method != "retained":
            return 0.0
        rs = named_parent.candidate.ring_system
        if rs is None:
            return 0.0
        if getattr(rs, "type", None) != "bridged":
            return 0.0
        return self._RETAINED_RING_SENIORITY_CREDIT
