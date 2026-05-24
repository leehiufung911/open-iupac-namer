"""
iupac_namer/perception/fg_detection.py

Functional Group Detection — Subsystem 5 of the Perception layer.

Graph-based SMARTS matching with 3-pass atom-ownership deconfliction.

Algorithm:
  1. Run all SMARTS patterns against the molecule (suffix + prefix-only groups).
  2. Build raw DetectedFG objects from every match.
  3. Apply 3-pass deconfliction to resolve overlapping atom claims:
       Pass 1 — Subsumption removal (known parent/child FG pairs)
       Pass 2 — Greedy assignment by seniority; unknown overlaps → AmbiguityPoint
       Pass 3 — (no further processing currently needed)
  4. Detect N-oxide and P-oxide additive groups separately.

Dependencies: AtomAnalysis, RingAnalysis (both must be built before this).

See ARCHITECTURE_PERCEPTION.md §Subsystem 5 for the full spec.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from iupac_namer.data_loader import get_functional_groups
from iupac_namer.types import AmbiguityPoint, DetectedFG, FGFraming

# Registration marker for the ``perception.fg`` sub-package (Stage 6 R1-F).
# The ``acid_infix_composition`` module layers a table-driven fallback for
# OPSIN infixes not covered by ``data/functional_groups.json`` (``nitrid``,
# ``tellur``, ``isocyanid``, ``isotellurocyanatid``, ``tellurocyanatid``,
# ``ditelluroperox``, ``hydroxim`` plus the partially covered ``azid`` /
# ``selenocyanatid`` / ``isoselenocyanatid`` cluster).  The engine
# dispatches it directly; importing it here keeps the coverage story
# traceable from a single file.
from iupac_namer.perception.fg import acid_infix_composition as _acid_infix_composition  # noqa: F401

# Registration marker for the ``perception.fg`` sub-package (Stage 6 R1-B).
# The ``heteroelement_oxoacids`` module is dispatched directly by
# ``engine.name`` as a whole-molecule shortcut for mononuclear
# (HO)_n X(=O)_m acids (stiboric, telluric, perchloric, boric, chromic,
# permanganic, halogen oxyacids, etc.).  Importing it here keeps the
# coverage story traceable from a single file and ensures the lookup
# table is eagerly loaded alongside the SMARTS FG table.
from iupac_namer.perception.fg import heteroelement_oxoacids as _heteroelement_oxoacids  # noqa: F401

# Registration marker for the ``perception.fg`` sub-package (Stage 6 R2-F).
# The ``phosphorus_oxoacids`` module is dispatched directly by
# ``engine.name`` as a whole-molecule shortcut for polynuclear P-O-P
# chain acids (diphosphoric / triphosphoric) and direct-P-P-bond acids
# (hypodiphosphoric) - the polynuclear complement to R1-B's mononuclear
# table.  Importing here keeps the coverage story traceable from a
# single file.
from iupac_namer.perception.fg import phosphorus_oxoacids as _phosphorus_oxoacids  # noqa: F401

# Registration marker for the ``perception.fg`` sub-package (Stage 6 R2-E).
# The ``cyclic_suffixes`` module classifies ring-embedded
# imide / lactam / lactone motifs (root cause #13 in
# ``docs/opsin_coverage_taxonomy.md``; FG audit Gap 8).  The engine
# dispatches its ``detect`` entry point after the acid-infix composition
# stage and before the generic plan search so a future emission layer can
# wire the ``-dicarboximide`` / ``-olactam`` / ``-olactone`` surface forms
# on without perturbing the existing plan dispatch.
from iupac_namer.perception.fg import cyclic_suffixes as _cyclic_suffixes  # noqa: F401

# Registration marker for the perception layer (Stage 6 R2-B).
# The ``charge_perception`` module classifies -ylium / -ide / -uide /
# acylium / amidinium / diazonium motifs on charged inputs BEFORE the
# plan-search neutralizer can drop them (root cause #3 in
# ``docs/opsin_coverage_taxonomy.md``).  The engine dispatches it
# directly via a hook in ``engine.name``; the import here keeps the
# coverage story traceable from a single file.
from iupac_namer.perception import charge_perception as _charge_perception  # noqa: F401

# Registration marker for the perception layer (Stage 6 R3-A).
# The ``organometallic`` module classifies neutral cyclopentadienyl
# sandwich complexes (ferrocene, ruthenocene, …) by exact whole-
# molecule canonical-SMILES match against an immutable retained-name
# table (root cause #16 in ``docs/opsin_coverage_taxonomy.md``).  The
# engine dispatches it from ``name_smiles`` BEFORE the P-29.2 free-
# valence guard so radical-bearing metal centres (V, Rh, Pb, Nb, …)
# are accepted without weakening the guard.  Importing here keeps the
# coverage story traceable from a single file.
from iupac_namer.perception import organometallic as _organometallic  # noqa: F401

if TYPE_CHECKING:
    from iupac_namer.perception.atoms import AtomAnalysis
    from iupac_namer.perception.rings import RingAnalysis

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Subsumption table
# ---------------------------------------------------------------------------

# SUBSUMPTION_TABLE[(fg_a_type, fg_b_type)] = True means:
#   fg_a is more specific and subsumes fg_b.
#   When fg_a and fg_b overlap AND this entry is True, fg_b is removed.
SUBSUMPTION_TABLE: dict[tuple[str, str], bool] = {
    # Amide subsumes its component patterns
    ("amide", "ketone"): True,
    ("amide", "amine"): True,
    ("amide", "secondary_amine"): True,
    ("amide", "tertiary_amine"): True,
    # Carbamate subsumes ester + amine variants
    ("carbamate", "ester"): True,
    ("carbamate", "amine"): True,
    ("carbamate", "secondary_amine"): True,
    ("carbamate", "tertiary_amine"): True,
    ("carbamate", "ketone"): True,
    # Urea subsumes amine
    ("urea", "amine"): True,
    ("urea", "secondary_amine"): True,
    # Guanidino (substituent prefix) subsumes amine + imine variants:
    # H2N-C(=NH)-NH-R contains an NH2 (amine), =NH (imine), and the NH linker
    # which the secondary_amine SMARTS would otherwise claim.
    ("guanidino", "amine"): True,
    ("guanidino", "secondary_amine"): True,
    ("guanidino", "imine"): True,
    # Thioamide subsumes thione, thial (C=S part), and amine (N part)
    ("thioamide", "thione"): True,
    ("thioamide", "thial"): True,
    ("thioamide", "amine"): True,
    ("thioamide", "secondary_amine"): True,
    ("thioamide", "tertiary_amine"): True,
    # Sulfoxide / sulfone subsuming thioether
    ("sulfoxide", "thioether"): True,
    ("sulfone", "thioether"): True,
    # P-56.1 / P-63.4: peroxol (suffix) supersedes hydroperoxy (prefix-only)
    # at the same -OO-H site so the suffix wins when -OOH is the PCG.
    ("peroxol", "hydroperoxy"): True,
    # P-43 / P-65.3.1.2: peroxy carboxylic acid R-C(=O)-O-O-H supersedes
    # the peroxol + ketone/aldehyde split at the same R-C(=O)-OO-H site
    # so the combined PIN suffix '-peroxoic acid' is used (peroxyacetic
    # acid / ethaneperoxoic acid) rather than '1-oxoethaneperoxol'.
    ("peroxy_acid", "peroxol"): True,
    ("peroxy_acid", "ketone"): True,
    ("peroxy_acid", "aldehyde"): True,
    ("peroxy_acid", "hydroperoxy"): True,
    # P-43: selenoic Se-acid R-C(=O)-SeH supersedes the selenol + ketone/
    # aldehyde split at the same R-C(=O)-SeH site so the combined PIN
    # suffix '-selenoic Se-acid' is used.
    ("carboselenoic_Se_acid", "selenol"): True,
    ("carboselenoic_Se_acid", "ketone"): True,
    ("carboselenoic_Se_acid", "aldehyde"): True,
    # P-56.3: imidamide R-C(=NH)-NH2 supersedes the imine + amine pair so the
    # combined characteristic group is named with the single -imidamide suffix
    # instead of being split into '...imine' + '...amine' substituent forms.
    ("imidamide", "imine"): True,
    ("imidamide", "amine"): True,
    ("imidamide", "secondary_amine"): True,
    # Phase 4: sulfinic_acid (anchor on S) overlaps with the C-anchored
    # imidamide FG when an amidino C is bonded directly to S of -S(=O)-OH,
    # e.g. N=C(N)S(=O)O.  Sulfinic acid has higher seniority (703) so it
    # wins as the principal characteristic group; the imidamide pieces are
    # then released to constituent imine/amine prefixes, yielding e.g.
    # "1-amino-1-iminomethanesulfinic acid".  Architecturally the C and S
    # are different atoms — the SMARTS' [#6] in the sulfinic_acid pattern
    # pulls in the C as part of the FG match — so dropping the C-anchored
    # imidamide cleanly releases the C for substitutive naming.  Same rule
    # for the standalone imine FG when it sits on the alpha-C of a sulfinic
    # acid (PASS 2 ambiguity warning otherwise — semantic resolution is
    # already correct because sulfinic_acid wins the seniority race).
    ("sulfinic_acid", "imidamide"): True,
    ("sulfinic_acid", "imine"): True,
    ("sulfonic_acid", "imidamide"): True,
    ("sulfonic_acid", "imine"): True,
    # Sulfonamide subsumes amine patterns
    ("sulfonamide", "amine"): True,
    ("sulfonamide", "secondary_amine"): True,
    ("sulfonamide", "tertiary_amine"): True,
    # Sulfonate-ester-anion (sulfonatooxy) and sulfamate-anion (sulfonatoamino)
    # are more specific than sulfonamide for the anionic [O-] forms.
    # sulfonatoamino: R-NH-S(=O)(=O)[O-] — anionic N-sulfonate / sulfamate anion
    # sulfonatooxy:  R-O-S(=O)(=O)[O-]  — anionic O-sulfonate / sulfate ester anion
    ("sulfonatoamino", "sulfonamide"): True,
    ("sulfonatoamino", "secondary_amine"): True,
    # Hydroxamic acid subsumes amide + alcohol
    ("hydroxamic_acid", "amide"): True,
    ("hydroxamic_acid", "alcohol"): True,
    # Hydrazide subsumes amide + amine
    ("hydrazide", "amide"): True,
    ("hydrazide", "amine"): True,
    ("hydrazide", "secondary_amine"): True,
    # Acid subsumes simpler patterns
    ("carboxylic_acid", "alcohol"): True,
    ("carboxylic_acid", "ketone"): True,
    # Thio-acids subsume their component patterns
    ("carbothioic_O_acid", "alcohol"): True,
    ("carbothioic_O_acid", "phenol"): True,
    ("carbothioic_O_acid", "thione"): True,
    ("carbothioic_S_acid", "thiol"): True,
    ("carbothioic_S_acid", "ketone"): True,
    ("carbodithioic_acid", "thiol"): True,
    ("carbodithioic_acid", "thione"): True,
    # P-66.4.1: carboximidic acid R-C(=NH)-OH supersedes the imine (C=NH) +
    # alcohol/phenol (OH) split at the same C so the combined PIN suffix
    # '-imidic acid' / '-carboximidic acid' is used (ethanimidic acid) rather
    # than '1-hydroxy...imine'.  When the same C also bears an amine (NH2) /
    # secondary_amine the acid (class 7a) outranks the amide-family group so it
    # subsumes those too, yielding aminomethanimidic acid for carbamimidic acid.
    ("carboximidic_acid", "imine"): True,
    ("carboximidic_acid", "alcohol"): True,
    ("carboximidic_acid", "phenol"): True,
    ("carboximidic_acid", "amine"): True,
    ("carboximidic_acid", "secondary_amine"): True,
    ("carboximidic_acid", "imidamide"): True,
    # carbonimidothioic acid HO-C(=NH)-SH: the O-acid carboximidic_acid is the
    # PCG and the -SH on the same C is released to a sulfanyl prefix, so it
    # subsumes the thiol match (sulfanylmethanimidic acid).
    ("carboximidic_acid", "thiol"): True,
    # P-66.4.1: carboximidothioic acid R-C(=NH)-SH supersedes imine + thiol.
    # carboximidic_acid (O-acid, more senior, lower seniority number) wins the
    # shared C of carbonimidothioic acid HO-C(=NH)-SH, so carboximidothioic_acid
    # also subsumes carboximidic_acid's leftover alcohol/phenol when it is the
    # PCG on a pure -SH compound, and releases the OH to a hydroxy prefix
    # otherwise via the seniority race (727 < 728).
    ("carboximidothioic_acid", "imine"): True,
    ("carboximidothioic_acid", "thiol"): True,
    ("carboximidothioic_acid", "amine"): True,
    ("carboximidothioic_acid", "secondary_amine"): True,
    ("carboximidothioic_acid", "imidamide"): True,
    # Shared-carbon overlap of carbonimidothioic acid (-OH and -SH on one C):
    # the more-senior O-acid carboximidic_acid wins and subsumes the
    # carboximidothioic_acid match, releasing the -SH to a sulfanyl prefix
    # (sulfanylmethanimidic acid).  Both forms OPSIN-round-trip.
    ("carboximidic_acid", "carboximidothioic_acid"): True,
    # Thio-ester FC variants subsume their component patterns
    ("thioester", "ketone"): True,
    ("thionoester", "thione"): True,
    ("dithioester", "thione"): True,
    # Thiono- / dithio-carbamate variants subsume their component patterns.
    # Structurally these are N-substituted analogues of thionoester/dithioester,
    # so they subsume the corresponding FC subtype plus amine variants,
    # exactly mirroring the carbamate -> ester/amine subsumption above.
    ("thionocarbamate", "thionoester"): True,
    ("thionocarbamate", "thione"): True,
    ("thionocarbamate", "thioamide"): True,
    ("thionocarbamate", "amine"): True,
    ("thionocarbamate", "secondary_amine"): True,
    ("thionocarbamate", "tertiary_amine"): True,
    ("dithiocarbamate", "dithioester"): True,
    ("dithiocarbamate", "thione"): True,
    ("dithiocarbamate", "thioamide"): True,
    ("dithiocarbamate", "amine"): True,
    ("dithiocarbamate", "secondary_amine"): True,
    ("dithiocarbamate", "tertiary_amine"): True,
    # Carbamothioate (S-substituted thiocarbamate, R-S-C(=O)-NR2) subsumes
    # its component patterns — structurally an N-substituted thioester so
    # it subsumes thioester plus amide / amine variants.
    ("carbamothioate", "thioester"): True,
    ("carbamothioate", "ketone"): True,
    ("carbamothioate", "amide"): True,
    ("carbamothioate", "secondary_amide"): True,
    ("carbamothioate", "tertiary_amide"): True,
    ("carbamothioate", "amine"): True,
    ("carbamothioate", "secondary_amine"): True,
    ("carbamothioate", "tertiary_amine"): True,
    # Aldehyde subsumes alcohol (C=O vs C-OH confusion at the H level)
    ("aldehyde", "alcohol"): True,
    # Nitrile subsumes imine
    ("nitrile", "imine"): True,
    # Phenol vs alcohol — both are valid FG types; phenol is more specific
    ("phenol", "alcohol"): True,
    # Acyl isothiocyanate subsumes the isothiocyanato prefix group
    ("acyl_isothiocyanate", "isothiocyanato"): True,
    # Acyl halides subsume ketone and halogen (the C=O and C-X are part of the acyl halide)
    ("acyl_chloride", "ketone"): True,
    ("acyl_chloride", "aldehyde"): True,
    ("acyl_chloride", "chloro"): True,
    ("acyl_bromide", "ketone"): True,
    ("acyl_bromide", "aldehyde"): True,
    ("acyl_bromide", "bromo"): True,
    ("acyl_fluoride", "ketone"): True,
    ("acyl_fluoride", "aldehyde"): True,
    ("acyl_fluoride", "fluoro"): True,
    ("acyl_iodide", "ketone"): True,
    ("acyl_iodide", "aldehyde"): True,
    ("acyl_iodide", "iodo"): True,
    # Secondary/tertiary amide subsumes primary amide + amine patterns
    ("secondary_amide", "amide"): True,
    ("secondary_amide", "ketone"): True,
    ("secondary_amide", "secondary_amine"): True,
    ("tertiary_amide", "amide"): True,
    ("tertiary_amide", "secondary_amide"): True,
    ("tertiary_amide", "ketone"): True,
    ("tertiary_amide", "tertiary_amine"): True,
    # Phase 4 — secondary/tertiary thioamide mirror the amide variants.
    # Subsumes the primary thioamide pattern (overlaps because [NX3H2]
    # restriction now means primary won't match, but a stale match from
    # an earlier match-pass could still need explicit subsumption) plus
    # thione (C=S overlap) and the corresponding amine.
    ("secondary_thioamide", "thioamide"): True,
    ("secondary_thioamide", "thione"): True,
    ("secondary_thioamide", "thial"): True,
    ("secondary_thioamide", "amine"): True,
    ("secondary_thioamide", "secondary_amine"): True,
    ("tertiary_thioamide", "thioamide"): True,
    ("tertiary_thioamide", "secondary_thioamide"): True,
    ("tertiary_thioamide", "thione"): True,
    ("tertiary_thioamide", "thial"): True,
    ("tertiary_thioamide", "amine"): True,
    ("tertiary_thioamide", "tertiary_amine"): True,
}


# ---------------------------------------------------------------------------
# Suffix form extraction helper
# ---------------------------------------------------------------------------

def _extract_suffix_forms(fg_def: dict) -> dict[str, str]:
    """Extract suffix forms from a functional group definition dict.

    The JSON may use:
    - 'suffix_terminal' / 'suffix_nonterminal' (separate forms)
    - 'suffix' (single form used for both positions)

    Returns dict with 'terminal' and/or 'nonterminal' keys.
    """
    forms: dict[str, str] = {}
    if "suffix_terminal" in fg_def:
        forms["terminal"] = fg_def["suffix_terminal"]
    if "suffix_nonterminal" in fg_def:
        forms["nonterminal"] = fg_def["suffix_nonterminal"]
    if "suffix" in fg_def:
        # Single suffix used for both positions
        forms.setdefault("terminal", fg_def["suffix"])
        forms.setdefault("nonterminal", fg_def["suffix"])
    return forms


def _extract_prefix_form(fg_def: dict) -> str:
    """Extract the terminal prefix form.

    Preference order: explicit "prefix", then "prefix_terminal", then the
    FG name as a last resort.  The nonterminal form (if distinct) is
    returned by _extract_prefix_form_nonterminal and stored separately on
    DetectedFG.
    """
    if "prefix" in fg_def:
        return fg_def["prefix"]
    if "prefix_terminal" in fg_def:
        return fg_def["prefix_terminal"]
    return fg_def.get("name", "")


def _extract_prefix_form_nonterminal(fg_def: dict) -> str | None:
    """Extract the nonterminal prefix form, or None if it matches the terminal.

    Used when the FG anchor is a branch off the parent (P-66.6 / P-66.6.1).
    E.g. aldehyde: prefix_terminal="oxo", prefix_nonterminal="formyl".
    """
    return fg_def.get("prefix_nonterminal")


# ---------------------------------------------------------------------------
# FGDetection
# ---------------------------------------------------------------------------

class FGDetection:
    """Functional group detection with SMARTS matching and 3-pass deconfliction.

    Parameters
    ----------
    mol:
        RDKit Mol object (sanitised).
    atom_analysis:
        AtomAnalysis for this molecule.
    ring_analysis:
        RingAnalysis for this molecule.

    Attributes (via properties)
    ----------
    detected_fgs:
        Non-overlapping DetectedFG instances after deconfliction.
    ambiguity_points:
        AmbiguityPoint instances for unknown FG overlaps.
    additive_groups:
        N-oxide / P-oxide additive group info dicts.
    """

    def __init__(
        self,
        mol: object,
        atom_analysis: AtomAnalysis,
        ring_analysis: RingAnalysis,
    ) -> None:
        self._mol = mol
        self._atoms = atom_analysis
        self._rings = ring_analysis

        (
            self._detected_fgs,
            self._ambiguity_points,
            self._additive_groups,
        ) = self._analyze()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def detected_fgs(self) -> tuple[DetectedFG, ...]:
        """All non-overlapping DetectedFG instances after deconfliction."""
        return self._detected_fgs

    @property
    def ambiguity_points(self) -> tuple[AmbiguityPoint, ...]:
        """AmbiguityPoint instances for unknown FG overlaps."""
        return self._ambiguity_points

    @property
    def additive_groups(self) -> list[dict]:
        """N-oxide / P-oxide additive group info dicts."""
        return self._additive_groups

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def fgs_by_type(self, fg_type: str) -> tuple[DetectedFG, ...]:
        """Return all detected FGs of a given type string."""
        return tuple(fg for fg in self._detected_fgs if fg.type == fg_type)

    def suffix_eligible_fgs(self) -> tuple[DetectedFG, ...]:
        """Return FGs that can be expressed as a suffix."""
        return tuple(fg for fg in self._detected_fgs if fg.suffix_eligible)

    def fg_at_atom(self, atom_idx: int) -> DetectedFG | None:
        """Return the DetectedFG that claims atom_idx, or None."""
        for fg in self._detected_fgs:
            if atom_idx in fg.atoms:
                return fg
        return None

    # ------------------------------------------------------------------
    # Top-level analysis
    # ------------------------------------------------------------------

    def _analyze(
        self,
    ) -> tuple[tuple[DetectedFG, ...], tuple[AmbiguityPoint, ...], list[dict]]:
        """Run SMARTS matching, build raw FGs, deconflict, detect additives."""
        raw_matches = self._run_smarts_matching()
        raw_fgs = [self._build_detected_fg(raw) for raw in raw_matches]
        final_fgs, ambiguity_points = self._deconflict(raw_fgs)
        additive_groups = self._detect_additive_groups(final_fgs)
        return (
            tuple(final_fgs),
            tuple(ambiguity_points),
            additive_groups,
        )

    # ------------------------------------------------------------------
    # SMARTS matching
    # ------------------------------------------------------------------

    def _run_smarts_matching(self) -> list[dict]:
        """Run all FG SMARTS patterns against the molecule.

        Returns a list of raw match dicts ready for DetectedFG construction.
        Each dict has:
            name, atoms (frozenset), anchor (int), seniority (int),
            suffix_eligible (bool), suffix_forms (dict), prefix_form (str),
            elision (bool), definition (dict).
        """
        from rdkit import Chem  # local import — RDKit may not always be needed

        fg_data = get_functional_groups()
        raw_matches: list[dict] = []

        # Build per-ring atom sets for ring-sharing checks below.
        # ring_sets[i] = frozenset of atom indices in ring i.
        ring_info = self._mol.GetRingInfo()  # type: ignore[attr-defined]
        ring_sets: list[frozenset[int]] = [frozenset(r) for r in ring_info.AtomRings()]

        # Endocyclic-amide guard: FG types where we must check whether the
        # carbonyl C (SMARTS atom 0) and the amide N (SMARTS atom 2) are both
        # in the same ring.  If they are, the C=O is a ring-carbonyl (oxo/one),
        # NOT an amide FG — it must not be classified as such.
        _ENDOCYCLIC_AMIDE_TYPES = frozenset(
            {"amide", "secondary_amide", "tertiary_amide",
             "thioamide", "secondary_thioamide", "tertiary_thioamide"}
        )

        def _is_endocyclic_amide(fg_name: str, match: tuple[int, ...]) -> bool:
            """Return True if this amide match has C and N both in the same ring."""
            if fg_name not in _ENDOCYCLIC_AMIDE_TYPES:
                return False
            if len(match) < 3:
                return False
            c_idx = match[0]  # carbonyl C
            n_idx = match[2]  # amide N
            return any(c_idx in rset and n_idx in rset for rset in ring_sets)

        # IUPAC P-65.1: a carboxylate anion -C(=O)[O-] is the same principal-
        # characteristic-group class as its conjugate acid -C(=O)OH.  The JSON
        # SMARTS only matches the acid form (historical reasons); we augment
        # the carboxylic_acid matches here with the anion form so that
        # zwitterionic molecules (e.g. cephem antibiotics ceftaroline and
        # ceftazidime) expose the ring-carboxylate as a PCG candidate.  Anion
        # dispatch (SUFFIX_VARIANT_TABLE in assembly.py) rewrites '-oic acid'
        # to '-oate' and '-carboxylic acid' to '-carboxylate' at the ANION
        # OutputForm.
        #
        # Two restrictions that keep pre-change behaviour intact:
        #   (a) Only augment when the molecule is ONE connected fragment with
        #       net formal charge 0 (a zwitterion — the carboxylate anion is
        #       balanced by a cation site on the same species).  Classic
        #       salts are dispatched fragment-by-fragment through _name_salt;
        #       their anion fragments carry net charge -1 and are already
        #       handled by the salt-naming pipeline without needing an extra
        #       PCG candidate from this augmentation — adding one steers
        #       strategy into a salt-fragment name format that OPSIN cannot
        #       roundtrip.
        #   (b) Skip anion matches on a carbon that ALREADY has an acid-form
        #       match.  Bicarbonate HO-C(=O)-[O-] is the canonical case: the
        #       -OH arm already supplies the carboxylic_acid FG, and adding a
        #       second anion-arm FG at the same carbon would cause the
        #       deconflictor to keep both (different atom sets, same anchor
        #       type) and yield a spurious "methane-1,1-dioate" name.
        _carboxylate_anion_pattern = Chem.MolFromSmarts("[CX3](=O)[OX1H0-]")
        # IUPAC P-65.3 sulfonate anion: -S(=O)(=O)[O-] is the conjugate base of
        # sulfonic acid -S(=O)(=O)OH and shares the same PCG class. Same logic
        # as carboxylate above: augment match on charge-balanced single-fragment
        # zwitterions (e.g. monobactam aztreonam: ring-N-S(=O)(=O)[O-] balanced
        # by remote NH3+).
        _sulfonate_anion_pattern = Chem.MolFromSmarts("[#16X4](=O)(=O)[OX1H0-]")
        _n_frags = len(Chem.GetMolFrags(self._mol))
        _net_charge = sum(a.GetFormalCharge() for a in self._mol.GetAtoms())  # type: ignore[attr-defined]
        _augment_carboxylate_anion = _n_frags == 1 and _net_charge == 0
        _augment_sulfonate_anion = _n_frags == 1 and _net_charge == 0

        # --- Suffix-eligible groups ---
        for fg_def in fg_data.get("suffix_groups", []):
            smarts = fg_def.get("smarts", "")
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is None:
                logger.warning("Invalid SMARTS for FG %r: %r", fg_def.get("name"), smarts)
                continue

            matches = self._mol.GetSubstructMatches(pattern)  # type: ignore[attr-defined]
            # For carboxylic_acid on a single-fragment molecule, also include
            # -C(=O)[O-] matches (see the block comment above).
            if (
                fg_def.get("name") == "carboxylic_acid"
                and _carboxylate_anion_pattern is not None
                and _augment_carboxylate_anion
            ):
                # Carbon atoms already claimed by an acid-form match.
                _acid_carbons = {m[0] for m in matches}
                anion_matches = self._mol.GetSubstructMatches(  # type: ignore[attr-defined]
                    _carboxylate_anion_pattern
                )
                extra = tuple(m for m in anion_matches if m[0] not in _acid_carbons)
                if extra:
                    matches = tuple(matches) + extra
            # Same augmentation for sulfonic_acid: anion form -S(=O)(=O)[O-]
            # belongs to the same PCG class. Required for monobactams such as
            # aztreonam where the ring-N-sulfonate is the senior characteristic
            # group of an internal zwitterion.
            if (
                fg_def.get("name") == "sulfonic_acid"
                and _sulfonate_anion_pattern is not None
                and _augment_sulfonate_anion
            ):
                _acid_sulfurs = {m[0] for m in matches}
                anion_matches_s = self._mol.GetSubstructMatches(  # type: ignore[attr-defined]
                    _sulfonate_anion_pattern
                )
                extra_s = tuple(m for m in anion_matches_s if m[0] not in _acid_sulfurs)
                if extra_s:
                    matches = tuple(matches) + extra_s
            anchor_index = fg_def.get("anchor_index", 0)
            # Some FG types appear in the "suffix_groups" table (so they carry a
            # seniority rank used by FC decomposition) but are NOT named with a
            # substitutive suffix — esters are the canonical example; they are
            # named via the Functional Class path as "alkyl acylate".
            substitutive_suffix_ineligible = fg_def.get("name") in {"ester"}
            fg_name = fg_def["name"]
            # Carbamate-family FGs (R-O/S-C(=X)-N<) place the heteroatom-ester
            # part on the alkyl chain and the amide N at the LAST match atom.
            # The "alkyl ...carbamate" suffix form treats the molecule as an
            # ester of carbamic acid, which requires the N to be acyclic.  When
            # the N is a RING atom (e.g. a Boc-protected pyrrolidine), the only
            # valid form is the ring-N prefix "(alkyloxy)carbonyl"; expressing
            # the carbamate as a suffix would make the exocyclic alkyl-O carbon
            # the PCG and pull the ring numbering off the heteroatom (N must be
            # locant 1).  Mark such matches suffix-ineligible per-match so they
            # become prefixes on the ring parent (P-66.4 / P-31.1.4.3.4).
            _carbamate_family = {
                "carbamate", "thionocarbamate",
                "carbamothioate", "dithiocarbamate",
            }
            for match in matches:
                # Skip amide/thioamide matches where both C and N are ring atoms
                # in the same ring — those are endocyclic carbonyls (oxo/one),
                # not substituent-style amide functional groups.
                if _is_endocyclic_amide(fg_name, match):
                    logger.debug(
                        "Skipping endocyclic %s match at atoms %s (C and N share a ring)",
                        fg_name,
                        match,
                    )
                    continue
                match_suffix_ineligible = substitutive_suffix_ineligible
                if (
                    fg_name in _carbamate_family
                    and match
                    and self._mol.GetAtomWithIdx(match[-1]).IsInRing()
                ):
                    match_suffix_ineligible = True
                raw_matches.append(
                    {
                        "name": fg_name,
                        "atoms": frozenset(match),
                        # anchor_index selects the defining atom of the FG within the
                        # SMARTS match.  Most patterns have the FG atom at index 0,
                        # but patterns like [#6][CX3](=O)[#6] (ketone) need index 1
                        # (the carbonyl C), and patterns like [OX2H1][CX4] (alcohol)
                        # need index 1 (the C bearing the -OH).
                        "anchor": match[anchor_index],
                        "seniority": fg_def.get("seniority", 9999),
                        "suffix_eligible": not match_suffix_ineligible,
                        "suffix_forms": _extract_suffix_forms(fg_def),
                        "prefix_form": _extract_prefix_form(fg_def),
                        "prefix_form_nonterminal": _extract_prefix_form_nonterminal(fg_def),
                        "elision": fg_def.get("elision", False),
                        "definition": fg_def,
                    }
                )

        # --- Prefix-only groups ---
        for fg_def in fg_data.get("prefix_only_groups", []):
            smarts = fg_def.get("smarts", "")
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is None:
                logger.warning("Invalid SMARTS for FG %r: %r", fg_def.get("name"), smarts)
                continue

            matches = self._mol.GetSubstructMatches(pattern)  # type: ignore[attr-defined]
            # By convention, prefix-only group SMARTS use a generic [#6]
            # placeholder for the attachment-context carbon (the parent C
            # the FG is bonded to).  Pattern atoms written as plain "[#6]"
            # qualify; pattern atoms with valence/H constraints like
            # "[CX2]" or "[CX3;!R]" are FG-internal carbons (the central C
            # of -N=C=O, the central C of guanidino, etc.).
            attachment_smarts_indices: list[int] = []
            for sa_idx in range(pattern.GetNumAtoms()):
                sa = pattern.GetAtomWithIdx(sa_idx)
                if sa.GetAtomicNum() != 6:
                    continue
                # Plain "[#6]" placeholders have GetSmarts() == "[#6]".
                if sa.GetSmarts() == "[#6]":
                    attachment_smarts_indices.append(sa_idx)
            for match in matches:
                # Map every plain-[#6] placeholder to its molecule atom.
                attachment_atoms = frozenset(
                    match[i] for i in attachment_smarts_indices
                )
                raw_matches.append(
                    {
                        "name": fg_def["name"],
                        "atoms": frozenset(match),
                        "anchor": match[0],
                        "attachment_context": attachment_atoms,
                        "seniority": 9999,
                        "suffix_eligible": False,
                        "suffix_forms": {},
                        "prefix_form": _extract_prefix_form(fg_def),
                        "prefix_form_nonterminal": _extract_prefix_form_nonterminal(fg_def),
                        "elision": False,
                        "definition": fg_def,
                    }
                )

        return raw_matches

    # ------------------------------------------------------------------
    # DetectedFG construction
    # ------------------------------------------------------------------

    def _build_detected_fg(self, raw: dict) -> DetectedFG:
        """Build a DetectedFG from a raw match dict.

        Uses AtomAnalysis to determine terminal vs nonterminal status.
        """
        anchor = raw["anchor"]
        atom_info = self._atoms[anchor]

        # Terminal heuristic: anchor has degree <= 2 (at most 2 heavy-atom neighbours).
        # For carbon atoms specifically: degree 1 (methyl) or 2 (chain end) counts as terminal.
        # This is a simplified heuristic; the engine refines terminal/nonterminal
        # during plan generation when the parent chain is known.
        is_terminal = atom_info.degree <= 2

        # Ring membership of the anchor
        in_ring = atom_info.in_ring

        properties = (
            ("seniority", raw["seniority"]),
            ("terminal", is_terminal),
            ("in_ring", in_ring),
            ("elision", raw["elision"]),
            ("attachment_context", raw.get("attachment_context")),
        )

        suffix_forms = tuple(raw["suffix_forms"].items()) if raw["suffix_forms"] else ()

        return DetectedFG(
            type=raw["name"],
            atoms=raw["atoms"],
            anchor=anchor,
            properties=properties,
            suffix_eligible=raw["suffix_eligible"],
            suffix_forms=suffix_forms,
            prefix_form=raw["prefix_form"],
            prefix_form_nonterminal=raw.get("prefix_form_nonterminal"),
        )

    # ------------------------------------------------------------------
    # 3-pass deconfliction algorithm
    # ------------------------------------------------------------------

    def _deconflict(
        self,
        raw_fgs: list[DetectedFG],
    ) -> tuple[list[DetectedFG], list[AmbiguityPoint]]:
        """Apply 3-pass deconfliction to produce non-overlapping FG claims.

        PASS 1 — Subsumption removal:
            Known parent/child pairs: remove the child when it overlaps the parent.

        PASS 2 — Greedy assignment by seniority:
            Sort by seniority (lower = higher priority).
            Claim atoms greedily; unknown overlaps → AmbiguityPoint.
            Exception: if a second FG shares an anchor with an already-claimed
            FG of the SAME TYPE but has a different atom set, both are legitimate
            coexisting FGs at that anchor (geminal diol, sulfamide, etc.) — accept
            the second one and merge its atoms into claimed_atoms. Identical atom
            sets are silently dropped as duplicates.

        PASS 3 — (reserved; currently no additional processing needed)

        Returns (final_fgs, ambiguity_points).
        """
        # ---------- PASS 1: Subsumption removal ----------
        surviving = list(raw_fgs)
        to_remove: set[int] = set()

        for i, fg_a in enumerate(surviving):
            for j, fg_b in enumerate(surviving):
                if i == j or i in to_remove or j in to_remove:
                    continue
                # Only act when the two FGs share at least one atom
                if not (fg_a.atoms & fg_b.atoms):
                    continue
                if SUBSUMPTION_TABLE.get((fg_a.type, fg_b.type)):
                    to_remove.add(j)

        surviving = [fg for i, fg in enumerate(surviving) if i not in to_remove]

        # ---------- PASS 2: Greedy assignment by seniority ----------
        # Lower seniority number = higher priority in the Blue Book P-65 table
        surviving.sort(key=lambda fg: fg.get_property("seniority", 9999))

        claimed_atoms: set[int] = set()
        final_fgs: list[DetectedFG] = []
        ambiguity_points: list[AmbiguityPoint] = []

        for fg in surviving:
            anchor_claimed = fg.anchor in claimed_atoms

            if not anchor_claimed:
                # Anchor is free — accept this FG and claim its atoms
                claimed_atoms.update(fg.atoms)
                final_fgs.append(fg)
            else:
                # Anchor is already owned by an earlier FG
                conflicting: DetectedFG | None = None
                for existing in final_fgs:
                    if fg.anchor in existing.atoms:
                        conflicting = existing
                        break

                if conflicting is None:
                    # Anchor claimed, but we can't find the owner — accept anyway
                    claimed_atoms.update(fg.atoms)
                    final_fgs.append(fg)
                    continue

                # Same-type coexistence at a shared anchor (e.g. geminal diol
                # at one carbon). Two matches are legitimately distinct FGs
                # only when the SHARED atoms are exactly the anchor itself;
                # if they share more atoms than that, it's a single chemical
                # core that the SMARTS matched twice because the peripheral
                # arms are symmetric. Example: thiourea H2N-C(=S)-NH2 yields
                # two thioamide matches sharing both the C and the S — those
                # are not two independent thioamides, they're one C(=S)(N)(N)
                # core. Upstream urea/thiourea/sulfamide functional-parent
                # handlers (P-66.6.3) capture the clean (R)2N-C(=X)-N(R)2
                # case; when those don't apply (e.g. N-N hydrazide-style arm
                # as in thiosemicarbazones CC=NN-C(=S)-NH2) we must still
                # emit the FG only once — otherwise the downstream
                # substituent-naming walks the shared C=S twice and produces
                # a bogus "dicarbothioamide" / "1-thiocarbamoyl-…-1-
                # carbothioamide" form.
                #
                # Among the duplicates, prefer the match whose non-shared
                # atoms are MORE TERMINAL (fewer heavy neighbours outside
                # the FG). For thiosemicarbazone this keeps the match that
                # includes the terminal -NH2 rather than the hydrazinyl N.
                # Identical atom sets are true duplicates — drop.
                if conflicting.type == fg.type:
                    if fg.atoms == conflicting.atoms:
                        # True duplicate — silent drop.
                        continue
                    shared = fg.atoms & conflicting.atoms
                    # Legit geminal iff the only shared atom is the anchor.
                    if shared <= {fg.anchor}:
                        claimed_atoms.update(fg.atoms)
                        final_fgs.append(fg)
                        continue
                    # Shared core beyond the anchor — duplicate emission of
                    # one chemical motif. Keep exactly one match; choose
                    # the more-terminal one.
                    def _terminality(match_atoms: frozenset[int]) -> int:
                        """Sum of (# heavy neighbours outside the match) for each
                        non-shared atom. Lower = more terminal. Ties broken by
                        sorted atom order for determinism."""
                        score = 0
                        for idx in match_atoms - shared:
                            a = self._mol.GetAtomWithIdx(idx)  # type: ignore[attr-defined]
                            score += sum(
                                1 for nb in a.GetNeighbors()
                                if nb.GetAtomicNum() > 1 and nb.GetIdx() not in match_atoms
                            )
                        return score
                    new_score = _terminality(fg.atoms)
                    old_score = _terminality(conflicting.atoms)
                    if new_score < old_score or (
                        new_score == old_score
                        and tuple(sorted(fg.atoms)) < tuple(sorted(conflicting.atoms))
                    ):
                        # Replace the existing match with the newcomer.
                        logger.debug(
                            "Replacing same-type FG %r: newer match %s is more "
                            "terminal than %s (shared core %s)",
                            fg.type,
                            sorted(fg.atoms),
                            sorted(conflicting.atoms),
                            sorted(shared),
                        )
                        # Remove the old one from final_fgs and replace.
                        final_fgs.remove(conflicting)
                        # Shrink claimed_atoms to drop atoms that were only
                        # from the replaced match, then re-add the newcomer's.
                        claimed_atoms -= (conflicting.atoms - shared)
                        claimed_atoms.update(fg.atoms)
                        final_fgs.append(fg)
                    else:
                        logger.debug(
                            "Dropping duplicate same-type FG %r (shared core %s "
                            "beyond anchor %s; kept match is more terminal)",
                            fg.type,
                            sorted(shared),
                            fg.anchor,
                        )
                    continue

                pair = (conflicting.type, fg.type)
                reverse_pair = (fg.type, conflicting.type)

                if SUBSUMPTION_TABLE.get(pair) or SUBSUMPTION_TABLE.get(reverse_pair):
                    # Known relationship — the lower-priority FG is silently dropped.
                    pass
                else:
                    # Unknown overlap — record as AmbiguityPoint (v13 F1).
                    # Log a warning so developers can add a subsumption entry.
                    overlap_atoms = conflicting.atoms & fg.atoms
                    logger.warning(
                        "Unknown FG overlap: %r vs %r at atoms %s. "
                        "Treating as ambiguity. Consider adding a subsumption entry.",
                        conflicting.type,
                        fg.type,
                        overlap_atoms,
                    )
                    ambiguity_points.append(
                        AmbiguityPoint(
                            atoms=overlap_atoms,
                            options=(
                                FGFraming(
                                    fgs=(conflicting,),
                                    description=conflicting.type,
                                ),
                                FGFraming(
                                    fgs=(fg,),
                                    description=fg.type,
                                ),
                            ),
                            canonical_preference=0,
                        )
                    )

        # ---------- PASS 3: nitrile-on-acid-heteroatom subsumption ----------
        # A terminal nitrile carbon (-C≡N) bonded directly to a chalcogen/
        # pnictogen acid centre (S/Se/Te/P that anchors a more-senior
        # suffix-eligible -onic/-inic acid FG) is NOT an independent "cyano"
        # substituent: that single carbon is the acid parent's own carbon and
        # the ≡N is a nitrido (azanylidyne) substituent on it.  Keeping the
        # nitrile would emit "cyano…sulfonic acid", which OPSIN reads as a
        # -CH2- bridge between the cyano carbon and the acid (a phantom
        # carbon).  The sulfinic-acid SMARTS already pulls that carbon into
        # its match (so deconfliction drops the nitrile); the X(VI) acid
        # SMARTS do not, so normalise them here.  Drop such a nitrile so the
        # carbon becomes the acid's parent and the ≡N flows to the engine's
        # azanylidyne fallback (P-29.2).
        _ACID_HETERO_ANCHORS = {
            existing.anchor
            for existing in final_fgs
            if existing.suffix_eligible
            and existing.type in (
                "sulfonic_acid", "selenonic_acid", "telluronic_acid",
                "phosphonic_acid", "phosphinic_acid", "sulfonamide",
                "seleninic_acid",
            )
        }
        if _ACID_HETERO_ANCHORS:
            _kept: list[DetectedFG] = []
            for fg in final_fgs:
                if fg.type == "nitrile":
                    c_atom = self._mol.GetAtomWithIdx(fg.anchor)  # type: ignore[attr-defined]
                    heavy_nbrs = [
                        nb for nb in c_atom.GetNeighbors()
                        if nb.GetAtomicNum() > 1
                    ]
                    # Terminal nitrile carbon: exactly the ≡N and one acid
                    # heteroatom centre as heavy neighbours.
                    acid_nbrs = [
                        nb for nb in heavy_nbrs
                        if nb.GetIdx() in _ACID_HETERO_ANCHORS
                    ]
                    if len(heavy_nbrs) == 2 and len(acid_nbrs) == 1:
                        logger.debug(
                            "Subsuming nitrile at C%d into acid heteroatom %d "
                            "(carbon is acid parent; ≡N → azanylidyne)",
                            fg.anchor, acid_nbrs[0].GetIdx(),
                        )
                        continue  # drop the nitrile FG
                _kept.append(fg)
            final_fgs = _kept

        return final_fgs, ambiguity_points

    # ------------------------------------------------------------------
    # N-oxide and P-oxide detection
    # ------------------------------------------------------------------

    def _detect_additive_groups(
        self, detected_fgs: list[DetectedFG] | None = None
    ) -> list[dict]:
        """Detect N-oxide [N+]([O-]) and P-oxide P(=O) additive groups.

        These are handled via additive nomenclature rather than as ordinary FGs,
        because the base compound is named first (without the oxide oxygen) and
        the word "oxide" is appended.

        Parameters
        ----------
        detected_fgs:
            The deconflicted suffix-eligible FGs (from _deconflict).  If a P
            atom is already the *anchor* of a suffix-eligible FG (e.g.
            phosphonic_acid), it is NOT emitted as a P-oxide additive group —
            the acid suffix takes priority (IUPAC P-66.6.3).

        Returns a list of dicts, each with:
            type ('oxide'), center_atom (int), added_atom (int),
            center_element ('N' or 'P').
        """
        from rdkit import Chem  # local import

        # Build the set of P atom indices that are already accounted for by a
        # suffix-eligible FG.  These must NOT be treated as P-oxide centers.
        _suffix_p_anchors: frozenset[int] = frozenset(
            fg.anchor
            for fg in (detected_fgs or [])
            if fg.suffix_eligible and self._atoms[fg.anchor].element == "P"
        )

        additive: list[dict] = []

        # N-oxide: [N+X4][O-]  (quaternary N with negatively charged O)
        n_oxide_smarts = "[N+X4]([O-])"
        n_oxide_pattern = Chem.MolFromSmarts(n_oxide_smarts)
        if n_oxide_pattern is not None:
            matches = self._mol.GetSubstructMatches(n_oxide_pattern)  # type: ignore[attr-defined]
            for match in matches:
                additive.append(
                    {
                        "type": "oxide",
                        "center_atom": match[0],
                        "added_atom": match[1],
                        "center_element": "N",
                    }
                )

        # Aromatic N-oxide: [n+][O-] (aromatic nitrogen ring, e.g. pyridine N-oxide)
        n_oxide_ar_smarts = "[n+]([O-])"
        n_oxide_ar_pattern = Chem.MolFromSmarts(n_oxide_ar_smarts)
        if n_oxide_ar_pattern is not None:
            matches = self._mol.GetSubstructMatches(n_oxide_ar_pattern)  # type: ignore[attr-defined]
            for match in matches:
                # Avoid duplicates with aliphatic N-oxide pattern
                if not any(
                    d["center_atom"] == match[0] and d["center_element"] == "N"
                    for d in additive
                ):
                    additive.append(
                        {
                            "type": "oxide",
                            "center_atom": match[0],
                            "added_atom": match[1],
                            "center_element": "N",
                        }
                    )

        # P-oxide: [PX4](=O) (pentavalent P with one =O)
        # Skip P atoms that are already the anchor of a suffix-eligible FG
        # (e.g. phosphonic_acid): those are named via the acid suffix, not
        # as "phosphane oxide".
        p_oxide_smarts = "[PX4](=O)"
        p_oxide_pattern = Chem.MolFromSmarts(p_oxide_smarts)
        if p_oxide_pattern is not None:
            matches = self._mol.GetSubstructMatches(p_oxide_pattern)  # type: ignore[attr-defined]
            for match in matches:
                p_idx = match[0]
                if p_idx in _suffix_p_anchors:
                    continue  # P=O is part of phosphonic acid (etc.) — skip
                # Phase 3 R-P64.4: collect the heavy-atom neighbours of P
                # other than the additive O.  Used by strategy.accept_additive
                # to gate "phosphane oxide" — additive nomenclature is the
                # PIN for trialkyl P=O (e.g. trimethylphosphane oxide,
                # P-64.4) but NOT for phosphate-style esters (every non-=O
                # neighbour is an O linker → tri(methoxy)(oxo)phosphane
                # substitutively) and NOT when P is buried inside an
                # aromatic substituent (where the ring would win as parent
                # and the trailing "oxide" would attach to the wrong stem).
                p_atom = self._mol.GetAtomWithIdx(p_idx)
                non_oxide_elems: list[str] = []
                non_oxide_aromatic = False
                non_oxide_in_ring = False
                for nb in p_atom.GetNeighbors():
                    if nb.GetIdx() == match[1]:
                        continue
                    non_oxide_elems.append(nb.GetSymbol())
                    if nb.GetIsAromatic():
                        non_oxide_aromatic = True
                    if nb.IsInRing():
                        non_oxide_in_ring = True
                additive.append(
                    {
                        "type": "oxide",
                        "center_atom": p_idx,
                        "added_atom": match[1],
                        "center_element": "P",
                        "non_oxide_neighbor_elements": tuple(non_oxide_elems),
                        "non_oxide_neighbor_aromatic": non_oxide_aromatic,
                        "non_oxide_neighbor_in_ring": non_oxide_in_ring,
                    }
                )

        return additive

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"FGDetection("
            f"n_fgs={len(self._detected_fgs)}, "
            f"n_ambiguity={len(self._ambiguity_points)}, "
            f"n_additive={len(self._additive_groups)})"
        )
