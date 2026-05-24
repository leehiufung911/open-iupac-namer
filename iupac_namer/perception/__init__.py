"""
iupac_namer/perception/__init__.py

Perception facade — lazily initializes the seven subsystems on first access.

Subsystem dependency DAG:
    AtomAnalysis ──► StereoAnalysis
         │
         ├──► RingAnalysis ──► FGDetection (includes deconfliction)
         │         │
         │         ├──► SymmetryAnalysis
         │         │
         │         └──► ChainFinding
         │
         └──► FragmentAnalysis (independent — uses only RDKit mol)

Each subsystem has its own module, takes explicit dependencies in __init__,
and is independently constructable and testable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

from iupac_namer.perception.atoms import AtomAnalysis
from iupac_namer.perception.fragments import FragmentAnalysis
from iupac_namer.perception.rings import RingAnalysis
from iupac_namer.perception.stereo import StereoAnalysis
from iupac_namer.perception.symmetry import SymmetryAnalysis

if TYPE_CHECKING:
    from iupac_namer.perception.chains import ChainFinding
    from iupac_namer.perception.fg_detection import FGDetection
    from iupac_namer.types import (
        CandidateParent,
        Interpretation,
        InterpretationQuery,
        RetainedMatch,
    )


class Perception:
    """Facade over seven perception subsystems.

    Subsystems are lazily initialized on first access.  The dependency DAG is
    preserved: accessing ``fgs`` triggers ``atoms`` and ``rings`` construction
    if not yet built.  Retained-name checks do not trigger ring/FG/chain
    subsystems, avoiding wasted work for simple molecules.

    Each subsystem is independently constructable for unit testing.
    """

    def __init__(self, mol: object) -> None:
        self._mol = mol
        self._atoms: AtomAnalysis | None = None
        self._stereo: StereoAnalysis | None = None
        self._fragments: FragmentAnalysis | None = None
        self._rings: RingAnalysis | None = None
        self._fgs: object | None = None
        self._symmetry: SymmetryAnalysis | None = None
        self._chains: object | None = None

    # ------------------------------------------------------------------
    # Subsystem 1 — Atom-level Analysis
    # ------------------------------------------------------------------

    @property
    def atoms(self) -> AtomAnalysis:
        """Per-atom structural info (element, valence, bonds, ring membership)."""
        if self._atoms is None:
            self._atoms = AtomAnalysis(self._mol)
        return self._atoms

    # ------------------------------------------------------------------
    # Subsystem 2 — Stereo Analysis
    # ------------------------------------------------------------------

    @property
    def stereo(self) -> StereoAnalysis:
        """Stereocenter detection — tetrahedral R/S and double-bond E/Z."""
        if self._stereo is None:
            self._stereo = StereoAnalysis(self._mol, self.atoms)
        return self._stereo

    # ------------------------------------------------------------------
    # Subsystem 3 — Fragment Detection
    # ------------------------------------------------------------------

    @property
    def fragments(self) -> FragmentAnalysis:
        """Disconnected-fragment detection (salts, multi-component species)."""
        if self._fragments is None:
            self._fragments = FragmentAnalysis(self._mol)
        return self._fragments

    # ------------------------------------------------------------------
    # Subsystem 4 — Ring System Analysis
    # ------------------------------------------------------------------

    @property
    def rings(self) -> RingAnalysis:
        """Ring system analysis — monocyclic, fused, bridged, spiro."""
        if self._rings is None:
            self._rings = RingAnalysis(self._mol, self.atoms)
        return self._rings

    # ------------------------------------------------------------------
    # Subsystem 5 — Functional Group Detection
    # ------------------------------------------------------------------

    @property
    def fgs(self) -> "FGDetection":
        """Functional group detection with SMARTS matching and 3-pass deconfliction."""
        if self._fgs is None:
            from iupac_namer.perception.fg_detection import FGDetection
            self._fgs = FGDetection(self._mol, self.atoms, self.rings)
        return self._fgs  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Subsystem 6 — Symmetry Analysis
    # ------------------------------------------------------------------

    @property
    def symmetry(self) -> SymmetryAnalysis:
        """Symmetry analysis — ring assembly and multiplicative candidates."""
        if self._symmetry is None:
            self._symmetry = SymmetryAnalysis(self._mol, self.atoms, self.rings)
        return self._symmetry

    # ------------------------------------------------------------------
    # Subsystem 7 — Chain Finding
    # ------------------------------------------------------------------

    @property
    def chains(self) -> "ChainFinding":
        """Chain finding — acyclic candidate parent chains."""
        if self._chains is None:
            from iupac_namer.perception.chains import ChainFinding
            self._chains = ChainFinding(self._mol, self.atoms, self.rings)
        return self._chains  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Interpretation generation (lazy, query-steered)
    # ------------------------------------------------------------------

    def interpretations(
        self, query: "InterpretationQuery"
    ) -> Iterator["Interpretation"]:
        """Yield Interpretation objects in best-first order relative to query.

        CONTRACT: This is a generator FUNCTION (not a stored iterator).
        Each call creates an independent generator.  Calling interpretations()
        twice produces two independent sequences, both starting from the
        beginning.  The generator is lazy — work happens on ``next()``.

        For molecules with no ambiguity points, yields exactly one
        Interpretation (the common case).  For molecules with ambiguity
        points, yields one per combination (product of options), capped at
        ``query.max_results``.
        """
        from iupac_namer.types import Interpretation
        from itertools import product as iproduct

        fgs = self.fgs.detected_fgs
        ambiguity_points = self.fgs.ambiguity_points
        ring_systems = self.rings.ring_systems
        stereocenters = self.stereo.stereocenters
        symmetry_groups = self.symmetry.symmetry_groups

        if not ambiguity_points:
            # Single interpretation — the overwhelmingly common case.
            yield Interpretation(
                fgs=fgs,
                ambiguity_choices=(),
                ring_systems=ring_systems,
                stereocenters=stereocenters,
                symmetry_groups=symmetry_groups,
            )
            return

        # Generate one interpretation per combination of ambiguity choices.
        # Pre-compute which atoms are "ambiguous" so we can filter the base FG list.
        ambiguous_atoms: set[int] = set()
        for ap in ambiguity_points:
            ambiguous_atoms.update(ap.atoms)

        base_fgs = tuple(fg for fg in fgs if not (fg.atoms & ambiguous_atoms))

        choice_ranges = [range(len(ap.options)) for ap in ambiguity_points]
        count = 0

        for combo in iproduct(*choice_ranges):
            if count >= query.max_results:
                return

            chosen_fgs: list = []
            seen_fg_keys: set = set()
            choices: list[tuple[int, int]] = []
            for ap_idx, opt_idx in enumerate(combo):
                for fg in ambiguity_points[ap_idx].options[opt_idx].fgs:
                    # Dedupe: the same FG can appear across multiple ambiguity
                    # points when its atom set spans several ambiguous atoms
                    # (e.g. a ketone match [#6][CX3](=O)[#6] whose two alpha
                    # carbons each conflict with a separate alcohol FG).
                    # Adding it once per AP would produce a phantom duplicate
                    # suffix (e.g. "butane-2,2-dione" instead of one "-one").
                    key = (fg.type, fg.atoms, fg.anchor)
                    if key in seen_fg_keys:
                        continue
                    seen_fg_keys.add(key)
                    chosen_fgs.append(fg)
                choices.append((ap_idx, opt_idx))

            yield Interpretation(
                fgs=base_fgs + tuple(chosen_fgs),
                ambiguity_choices=tuple(choices),
                ring_systems=ring_systems,
                stereocenters=stereocenters,
                symmetry_groups=symmetry_groups,
            )
            count += 1

    # ------------------------------------------------------------------
    # Retained name matching (stub — full implementation in Phase 1.4)
    # ------------------------------------------------------------------

    def retained_matches(
        self, mol: object, output_form: object
    ) -> Iterator["RetainedMatch"]:
        """Yield retained name matches valid for this molecule and output form.

        Interpretation-independent — called once before the interpretation loop.

        Full implementation is deferred to Phase 1.4.  This stub yields nothing,
        so the engine falls through to systematic naming for all molecules.
        """
        return
        yield  # make this a generator (stub yields nothing)

    # ------------------------------------------------------------------
    # Candidate parent generation (lazy, PCG-parameterized)
    # ------------------------------------------------------------------

    def candidate_parents(
        self,
        interpretation: "Interpretation",
        pcg_anchors: tuple[int, ...] = (),
        required_atom: "int | None" = None,
    ) -> Iterator["CandidateParent"]:
        """Yield CandidateParent objects interleaving chain and ring candidates.

        Parameters
        ----------
        interpretation:
            The current Interpretation (FG assignments, ring systems, etc.).
        pcg_anchors:
            Anchor atom indices of ALL instances of the PCG type.  Candidates
            must be RELATED TO at least one anchor: the anchor is either ON the
            parent (terminal FG) or BONDED TO a parent atom (non-terminal FG).
        required_atom:
            If provided and not covered by any max-length chain candidate,
            a single-atom chain candidate is also yielded for this atom.
            Used in SUBSTITUENT mode to guarantee a valid parent candidate
            for the attachment atom even when a longer disconnected chain
            exists elsewhere in the molecule.

        Notes
        -----
        Full interleaving of chain vs ring candidates with proper scoring will
        be implemented in Phase 2.  For now, chains come first, then rings.
        """
        from iupac_namer.types import CandidateParent

        # 0. Acyclic N+ azanium heteroatom_center candidates (P-73.2 /
        # P-62.3.1).  Yielded BEFORE rings so the +50 heteroatom_center
        # bonus reaches the scorer before the plan-budget cap fills with
        # ring-numbering variants.  Without this, ``c1ccccc1[NH3+]``
        # picks benzene as parent and emits ``(azaniumyl)benzene``
        # instead of the spec ``phenylazanium`` / ``anilinium`` form.
        _prefix_fg_atoms_pre: frozenset[int] = frozenset(
            idx
            for fg in interpretation.fgs
            if not fg.suffix_eligible
            for idx in fg.atoms
        )
        for atom_info in self.atoms:
            if atom_info.element != "N":
                continue
            if atom_info.charge != 1:
                continue
            if atom_info.in_ring:
                continue
            if atom_info.idx in _prefix_fg_atoms_pre:
                continue
            yield CandidateParent(
                atom_indices=frozenset({atom_info.idx}),
                type="heteroatom_center",
                length=1,
                ring_system=None,
                unsaturation=None,
                element="N+",
                lambda_value=None,
            )

        # 1. Chain candidates — filter out candidates entirely composed of
        #    carbons internal to a prefix-only FG.  The motivating case is
        #    methyl isocyanate CH3-N=C=O: the central C of -N=C=O is part
        #    of the isocyanato FG (its only bonds go to other FG atoms),
        #    yet the chain finder treats it as a 1-carbon parent candidate
        #    on equal footing with the methyl carbon.  Letting it through
        #    would steer the strategy into a substitutive
        #    "(methylimino)(oxo)methane" name instead of the spec PIN
        #    "isocyanatomethane" / "methyl isocyanate" (P-61.8).
        #
        #    Internal carbons are FG atoms that are NOT the attachment
        #    context (the SMARTS [#6] anchor that connects the FG to the
        #    parent chain).  For most prefix-only groups this set is empty
        #    (no internal C); only -N=C=O / -N=C=S contain a structural
        #    carbon between heteroatoms.
        _internal_prefix_fg_carbons: set[int] = set()
        for fg in interpretation.fgs:
            if fg.suffix_eligible:
                continue
            attachment_atoms = fg.get_property("attachment_context")
            if attachment_atoms is None:
                # No attachment_context recorded — skip filtering for this
                # FG to avoid mis-classifying parent carbons as internal.
                continue
            for atom_idx in fg.atoms:
                if atom_idx in attachment_atoms:
                    continue
                atom = self._mol.GetAtomWithIdx(atom_idx)  # type: ignore[attr-defined]
                if atom.GetAtomicNum() == 6:
                    _internal_prefix_fg_carbons.add(atom_idx)

        for cand in self.chains.find_candidate_chains(
            pcg_anchors, required_atom=required_atom
        ):
            if (
                _internal_prefix_fg_carbons
                and cand.atom_indices
                and cand.atom_indices <= _internal_prefix_fg_carbons
            ):
                # Entire candidate chain is internal-FG carbons; skip.
                continue
            yield cand

        # 2. Ring candidates (from ring systems detected in this interpretation)
        # Order ring candidates so the senior parent comes first per IUPAC
        # P-44 (max skeletal atoms = first criterion; among equal sizes,
        # P-44.4.1.2 prefers heterocyclic over carbocyclic).  Without this
        # ordering the plan-budget cap can drop the senior candidate
        # before scoring, leading to e.g. ``(naphthalen-2-yl)benzene``
        # instead of ``2-phenylnaphthalene``, or
        # ``(pyridin-2-yl)benzene`` instead of ``2-phenylpyridine``.
        # Tie-break: keep insertion order (stable sort).
        ordered_rs = sorted(
            interpretation.ring_systems,
            # P-44.4.1.2: heterocyclic ring systems are senior to
            # carbocyclic ones (FIRST), then by max skeletal atoms (P-44.1).
            key=lambda rs: (0 if rs.heteroatoms else 1, -rs.ring_size),
        )
        for rs in ordered_rs:
            yield CandidateParent(
                atom_indices=rs.atom_indices,
                type=rs.type,
                length=rs.ring_size,
                ring_system=rs,
                unsaturation=None,   # computed later with numbering
                element=None,
                lambda_value=None,
            )

        # 3. Heteroatom-center candidates (P, Si, B, As, Ge, Sn as single-atom
        #    acyclic neutral centers — named as phosphane, silane, borane, etc.)
        #
        # P-66.6.3: acid suffixes (phosphonic, boronic) take precedence over the
        # parent hydride name (phosphane, borane).  When a P or B atom is already
        # the *anchor* of a suffix-eligible FG in this interpretation, suppress
        # the heteroatom_center candidate — the FG fully accounts for that atom.
        _HETEROATOM_PARENT_ELEMENTS = frozenset({"P", "Si", "B", "As", "Ge", "Sn"})
        # Stage 22 R22-B: Bi, Sb, Pb get heteroatom_center candidates ONLY
        # when the molecule also contains a ring system.  This unblocks the
        # carved aryl-on-heavy-element SUBSTITUENT path (e.g. ``[PbH3][c]1ccccc1``
        # carved from ``c1cc[c]([PbH2][c]2ccccc2)cc1`` — the ring wins as parent
        # in the standalone case, and the heteroatom_center is the parent for
        # the carved substituent recursion).  Gating on ring-presence preserves
        # the R18-A test guards: ``C[PbH3]`` stays ``(plumbyl)methane`` /
        # ``methylplumbane`` form via the carbon chain parent, and
        # ``[PbH3][PbH3]`` stays ``diplumbane`` via the heteroatom_chain parent.
        _R22B_RING_GATED_ELEMENTS = frozenset({"Bi", "Sb", "Pb"})
        _has_ring_system = bool(interpretation.ring_systems)
        _suffix_fg_anchors: frozenset[int] = frozenset(
            fg.anchor for fg in interpretation.fgs if fg.suffix_eligible
        )
        for atom_info in self.atoms:
            if atom_info.element in _HETEROATOM_PARENT_ELEMENTS:
                pass
            elif atom_info.element in _R22B_RING_GATED_ELEMENTS:
                # Phase 3 R35: Bi/Sb/Pb heteroatom_center is allowed when
                # either (a) the molecule has a ring system (R22-B) OR
                # (b) the centre carries ≥2 heavy substituents (multi-
                # organyl Bi/Sb/Pb has no carbon-chain parent that can
                # capture all branches, so the substitutive plan-search
                # fails — admitting the heteroatom_center lets it win
                # cleanly).  R18-A guards on ``C[PbH3]`` etc. (single
                # heavy neighbour) are preserved by the ≥2 gate.
                heavy_nb_count = sum(
                    1 for nb_idx in atom_info.neighbors
                    if self.atoms[nb_idx].element != "H"
                )
                if not (_has_ring_system or heavy_nb_count >= 2):
                    continue
            else:
                continue
            if atom_info.in_ring:
                continue
            if atom_info.charge != 0:
                continue
            # Skip this atom if it's already claimed as the anchor of a
            # suffix-eligible FG (e.g. phosphonic_acid, boronic_acid).
            # Those FG suffixes take priority over the parent-hydride name.
            if atom_info.idx in _suffix_fg_anchors:
                continue
            yield CandidateParent(
                atom_indices=frozenset({atom_info.idx}),
                type="heteroatom_center",
                length=1,
                ring_system=None,
                unsaturation=None,
                element=atom_info.element,
                lambda_value=None,
            )

        # 3b. Charged-N (azanium) parent hydrides — already yielded above
        #     in section 0 (moved earlier in Phase 3 R31 so the candidate
        #     reaches the scorer before the plan-budget fills with ring-
        #     numbering variants).  Keep ``_prefix_fg_atoms`` available
        #     for downstream sections (3c sulfanium uses the same gate).
        _prefix_fg_atoms: frozenset[int] = frozenset(
            idx
            for fg in interpretation.fgs
            if not fg.suffix_eligible
            for idx in fg.atoms
        )

        # 3c. Charged-S (sulfanium) parent hydrides — S+ acyclic atoms.
        #     Trivalent sulfonium cations R-[S+](R')-R'' (or R-[SH+]-R')
        #     are named as "sulfanium" parent hydrides (P-66.6.5 / P-73.2.2.1),
        #     directly analogous to azanium for N+.  We use element="S+" to
        #     distinguish from neutral S.  Ring-embedded S+ (e.g. thiopyrylium)
        #     is excluded here (in_ring check) and would be handled by ring_naming.
        #
        # Exclusion: if the S+ atom is a member of any prefix-only FG, skip it.
        # (Symmetric with the N+ check in 3b — see Stage 15 R15-A note.)
        for atom_info in self.atoms:
            if atom_info.element != "S":
                continue
            if atom_info.charge != 1:
                continue
            if atom_info.in_ring:
                continue
            if atom_info.idx in _prefix_fg_atoms:
                continue
            yield CandidateParent(
                atom_indices=frozenset({atom_info.idx}),
                type="heteroatom_center",
                length=1,
                ring_system=None,
                unsaturation=None,
                element="S+",
                lambda_value=None,
            )

        # 3d. Charged group-15/16 (P/As/Sb/O/Se/Te) parent-hydride cations.
        #     Mirrors the N+ (azanium) and S+ (sulfanium) blocks above for the
        #     remaining main-group heteroatoms: an acyclic cation whose bond
        #     count is one MORE than the element's standard valence is the
        #     "-ium" H-addition cation (P-73.2.2.1.1) — phosphanium ([PH4+]),
        #     oxidanium ([OH3+] / H3O+), arsanium, stibanium, selanium,
        #     tellanium, and their substituted forms (e.g. trimethyloxidanium,
        #     tetramethylphosphanium).  The valence gate excludes the H-removal
        #     "-ylium" cations (e.g. silylium [SiH3+], sulfanylium [SH+]) which
        #     take a different suffix and are left to other paths; ring-embedded
        #     cations (pyrylium, thiopyrylium) are excluded by the in_ring check.
        _HYDRIDE_CATION_STD_VALENCE = {
            "P": 3, "As": 3, "Sb": 3, "Bi": 3, "O": 2, "Se": 2, "Te": 2,
            # Halogen "-ium" H-addition cations (P-73.2.2.1.1): protonated
            # halogenanes — fluoranium ([FH2+]), chloranium, bromanium,
            # iodanium.  Standard valence 1; the cation's valence is std+1=2.
            # These are closed-shell (explicit H, no radical electron) so the
            # open-valence guard does not reject them.
            "F": 1, "Cl": 1, "Br": 1, "I": 1,
        }
        for atom_info in self.atoms:
            _std_val = _HYDRIDE_CATION_STD_VALENCE.get(atom_info.element)
            if _std_val is None:
                continue
            if atom_info.charge != 1:
                continue
            if atom_info.in_ring:
                continue
            if atom_info.idx in _prefix_fg_atoms:
                continue
            _cat_atom = self._mol.GetAtomWithIdx(atom_info.idx)  # type: ignore[attr-defined]
            if _cat_atom.GetTotalValence() != _std_val + 1:
                continue
            yield CandidateParent(
                atom_indices=frozenset({atom_info.idx}),
                type="heteroatom_center",
                length=1,
                ring_system=None,
                unsaturation=None,
                element=atom_info.element + "+",
                lambda_value=None,
            )

        # 3e. Charged group-14/15 (P/As/Sb/Si/Ge) parent-hydride anions.
        #     The "-ide" H-removal anions (P-73.2.2.1.2): phosphanide ([PH2-]),
        #     arsanide, stibanide, silanide ([SiH3-]), germanide — the anion
        #     centre's valence is one LESS than the element's standard valence.
        #     O/S/Se/Te anions ([O-]/[S-] → olate/thiolate, etc.) are claimed
        #     earlier by the charge-perception acidic-anion path, so they are
        #     deliberately excluded here.  Boron's H-addition -uide anion
        #     ([BH4-] boranuide) has its own classifier.
        _HYDRIDE_ANION_STD_VALENCE = {
            "P": 3, "As": 3, "Sb": 3, "Si": 4, "Ge": 4,
        }
        for atom_info in self.atoms:
            _std_val = _HYDRIDE_ANION_STD_VALENCE.get(atom_info.element)
            if _std_val is None:
                continue
            if atom_info.charge != -1:
                continue
            if atom_info.in_ring:
                continue
            if atom_info.idx in _prefix_fg_atoms:
                continue
            _an_atom = self._mol.GetAtomWithIdx(atom_info.idx)  # type: ignore[attr-defined]
            if _an_atom.GetTotalValence() != _std_val - 1:
                continue
            yield CandidateParent(
                atom_indices=frozenset({atom_info.idx}),
                type="heteroatom_center",
                length=1,
                ring_system=None,
                unsaturation=None,
                element=atom_info.element + "-",
                lambda_value=None,
            )

        # 3f. Charged group-14 (Si/Ge) parent-hydride "-ylium" cations
        #     (P-73.2.2.1.1).  These are the H-REMOVAL cations: a [SiH3+]
        #     silicon centre carries one FEWER bond than silane ([SiH4]),
        #     i.e. its valence is standard-1, exactly mirroring the 3e
        #     "-ide" H-removal anions but with +1 charge.  silylium ([SiH3+]),
        #     germylium ([GeH3+]) and their substituted forms (methylsilylium,
        #     trimethylsilylium, ...).  The valence==std-1 gate distinguishes
        #     these from the 3d "-ium" H-ADDITION cations (which require
        #     valence==std+1 and live in a different element set).  O/S/Se/Te
        #     "-ylium" cations are deliberately excluded — those hit the
        #     open-valence guard and are left unhandled (see CLAUDE.md scope).
        _YLIUM_CATION_STD_VALENCE = {
            "Si": 4, "Ge": 4, "Sn": 4, "Pb": 4,
        }
        for atom_info in self.atoms:
            _std_val = _YLIUM_CATION_STD_VALENCE.get(atom_info.element)
            if _std_val is None:
                continue
            if atom_info.charge != 1:
                continue
            if atom_info.in_ring:
                continue
            if atom_info.idx in _prefix_fg_atoms:
                continue
            _yl_atom = self._mol.GetAtomWithIdx(atom_info.idx)  # type: ignore[attr-defined]
            if _yl_atom.GetTotalValence() != _std_val - 1:
                continue
            yield CandidateParent(
                atom_indices=frozenset({atom_info.idx}),
                type="heteroatom_center",
                length=1,
                ring_system=None,
                unsaturation=None,
                element=atom_info.element + "+",
                lambda_value=None,
            )

        # 4. Heteroatom-chain candidates (N-N as hydrazine, S-S as disulfane,
        #    O-O as dioxidane/hydrogen peroxide, Se-Se as diselane, Te-Te as
        #    ditellane, plus the group-13/14/15 dimeric parent hydrides
        #    registered in Stage 18 R18-B: dibismuthane, distibane, diarsane,
        #    diphosphane, distannane, diplumbane, digermane, disilane).
        #    Both atoms must be acyclic, uncharged, and directly bonded by a
        #    single bond.
        _HETEROATOM_CHAIN_ELEMENTS = frozenset({
            "N", "S", "O", "Se", "Te",
            "P", "As", "Sb", "Bi",
            "Si", "Ge", "Sn", "Pb",
        })
        _VALID_HOMO_PAIRS = frozenset({
            ("N", "N"), ("S", "S"), ("O", "O"), ("Se", "Se"), ("Te", "Te"),
            ("P", "P"), ("As", "As"), ("Sb", "Sb"), ("Bi", "Bi"),
            ("Si", "Si"), ("Ge", "Ge"), ("Sn", "Sn"), ("Pb", "Pb"),
        })
        seen_pairs: set[frozenset[int]] = set()
        for atom_info in self.atoms:
            if atom_info.element not in _HETEROATOM_CHAIN_ELEMENTS:
                continue
            if atom_info.in_ring:
                continue
            if atom_info.charge != 0:
                continue
            for nb_idx in atom_info.neighbors:
                nb_info = self.atoms[nb_idx]
                if nb_info.element not in _HETEROATOM_CHAIN_ELEMENTS:
                    continue
                if nb_info.in_ring:
                    continue
                if nb_info.charge != 0:
                    continue
                # Single bonds: N-N (hydrazine), S-S (disulfane), O-O (dioxidane), etc.
                # Double bond N=N is also accepted as "diazene" (a separate
                # heteroatom-chain parent whose element is encoded as "N=N");
                # this allows ``H2N-N=N-naphthyl`` and similar diazene-based
                # PINs (e.g. "diazene-1-carboxamide") to use diazene as parent
                # rather than collapsing to a tiny carbon chain that forces
                # the diazenyl group into a substituent prefix.  N≡N
                # (diazonium) remains out of scope.
                bond_type = next(
                    (bt for other, bt in atom_info.bond_types if other == nb_idx),
                    None,
                )
                if bond_type not in ("single", "double"):
                    continue
                # Only N=N is currently supported on the unsaturated path; other
                # homo-pairs are restricted to single bonds.
                if bond_type == "double" and not (
                    atom_info.element == "N" and nb_info.element == "N"
                ):
                    continue
                pair_key = frozenset({atom_info.idx, nb_idx})
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                pair_elements = tuple(sorted([atom_info.element, nb_info.element]))
                if pair_elements not in _VALID_HOMO_PAIRS:
                    continue
                # Determine the canonical element string for this pair
                elem_pair = (atom_info.element, nb_info.element)
                canonical_pair = tuple(sorted(elem_pair))  # e.g. ("N","N"), ("S","S"), ("O","O")
                # Encode the N=N case via element="N=N" so the NamedParent
                # builder can route to "diazene" instead of "hydrazine".
                if bond_type == "double":
                    element_label = "N=N"
                else:
                    element_label = canonical_pair[0]
                yield CandidateParent(
                    atom_indices=frozenset({atom_info.idx, nb_idx}),
                    type="heteroatom_chain",
                    length=2,
                    ring_system=None,
                    unsaturation=None,
                    element=element_label,  # "N", "S", "O", or "N=N"
                    lambda_value=None,
                )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        n_atoms = len(self._atoms) if self._atoms is not None else "?"
        return (
            f"Perception(mol={self._mol!r}, "
            f"atoms_cached={self._atoms is not None}, "
            f"n_atoms={n_atoms})"
        )


__all__ = [
    "Perception",
    "AtomAnalysis",
    "FragmentAnalysis",
    "RingAnalysis",
    "StereoAnalysis",
    "SymmetryAnalysis",
]
