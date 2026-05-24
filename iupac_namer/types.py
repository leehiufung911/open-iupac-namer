"""
iupac_namer/types.py
All typed dataclasses for the v13 architecture.

All dataclasses are frozen (immutable) except NamingSession.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Iterator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OutputForm(Enum):
    """What string form should this fragment produce?
    Also constrains which decompositions are valid for this fragment.

    Design rule (Principle 7): each OutputForm must change what STRING is
    produced. If two forms produce the same string, merge them. Express
    plan-only constraints elsewhere (on Decomposition eligibility or
    role annotations).
    """
    STANDALONE = auto()       # "ethanol", "acetic acid"
    SUBSTITUENT = auto()      # "ethyl", "propan-2-yl" -- suppresses FC decomposition
    ACID_STEM = auto()        # "acetate", "benzoate"
    ACYL = auto()             # "acetyl" -- suppresses FC decomposition
    ANION = auto()            # "ethanolate", "phenoxide"
    CATION = auto()           # "ethylium"
    PARENT_HYDRIDE = auto()   # "ethane" (no suffix) -- for multiplicative subunits


class SubstituentMethod(Enum):
    """Which of the two IUPAC methods for substituent naming (P-29.2)."""
    ALKYL = auto()      # Method (1): replace "-ane" with "-yl". Locant 1 omitted.
    ALKANYL = auto()    # Method (2): add "-yl" to parent name. Locant cited.


# ---------------------------------------------------------------------------
# FREE_VALENCE_SUFFIXES dict
# ---------------------------------------------------------------------------

# Maps (n_attachment_points, sorted_bond_orders_tuple) -> IUPAC suffix string
FREE_VALENCE_SUFFIXES: dict[tuple, str] = {
    (1, (1,)): "yl",        # monovalent single bond: ethyl, propyl
    (1, (2,)): "ylidene",   # monovalent double bond: methylidene
    (1, (3,)): "ylidyne",   # monovalent triple bond: methylidyne
    (2, (1, 1)): "diyl",    # divalent, two single bonds at different atoms
    (2, (2,)): "ylidene",   # divalent via double bond at one atom (same as monovalent double)
    (3, (1, 1, 1)): "triyl",
}


# ---------------------------------------------------------------------------
# Locant
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Locant:
    """A locant in IUPAC nomenclature.

    Numeric:     Locant.numeric(2)          -> "2"
    Compound:    Locant.numeric(4, "a")     -> "4a"   (fused ring junction atoms)
    Primed:      Locant.numeric(1, "'")     -> "1'"   (ring assembly second ring)
    Added-H:     Locant.numeric(1, "H")     -> "1H"   (indicated hydrogen)
    Heteroatom:  Locant.hetero("N")         -> "N"    (N-locant for amines/amides)
    Heteroatom:  Locant.hetero("O")         -> "O"    (O-locant for esters)
    Superscript: Locant.hetero("N", sup="2") -> "N2"  (multiple heteroatom positions)
    """
    label: str              # "2", "4a", "N", "O", "N2"
    is_numeric: bool        # True for integer-based, False for letter-based
    _numeric_value: int | None = None   # cached for numeric sorting
    suffix: str = ""        # "a", "'", "H" for compound locants

    @staticmethod
    def numeric(value: int, suffix: str = "") -> Locant:
        return Locant(
            label=f"{value}{suffix}",
            is_numeric=True,
            _numeric_value=value,
            suffix=suffix,
        )

    @staticmethod
    def hetero(element: str, sup: str = "") -> Locant:
        label = f"{element}{sup}" if sup else element
        return Locant(label=label, is_numeric=False, _numeric_value=None, suffix="")

    def __str__(self) -> str:
        return self.label

    def __lt__(self, other: Locant) -> bool:
        # Numeric locants sort before heteroatom locants (P-14.4)
        if self.is_numeric != other.is_numeric:
            return self.is_numeric  # numeric < heteroatom
        if self.is_numeric:
            if self._numeric_value != other._numeric_value:
                return self._numeric_value < other._numeric_value  # type: ignore[operator]
            return self.suffix < other.suffix
        # Heteroatom locants: alphabetical N < O < P < S
        return self.label < other.label


# ---------------------------------------------------------------------------
# Numbering
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Numbering:
    """A complete locant assignment for a named parent. Immutable.

    For chains: one of two directions (forward/reverse).
    For monocyclic rings: starting atom + direction (CW/CCW).
    For fused rings: fixed by fusion descriptor (one canonical numbering).
    For bridged rings: fixed by von Baeyer convention.
    For spiro rings: fixed by spiro convention.

    compute_numberings() pre-filters by P-14.4 rules and typically
    yields 1-3 options for any parent type. See candidate parent generation.
    """
    _assignments: tuple[tuple[int, Locant], ...]  # (atom_idx, locant) pairs, sorted
    locant_set: tuple[Locant, ...]                 # sorted, for lowest-set comparison

    @property
    def atom_to_locant(self) -> dict[int, Locant]:
        """Reconstructed on access. Cheap for typical parent sizes (<50 atoms)."""
        return dict(self._assignments)

    @property
    def locant_to_atom(self) -> dict[Locant, int]:
        return {loc: idx for idx, loc in self._assignments}


# ---------------------------------------------------------------------------
# AtomInfo
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AtomInfo:
    idx: int
    element: str            # any element, not just C/H/N/O
    atomic_num: int
    valence: int
    charge: int
    degree: int             # number of bonded neighbors
    in_ring: bool
    aromatic: bool
    neighbors: tuple[int, ...]
    coordination_number: int
    bond_types: tuple[tuple[int, str], ...]  # (neighbor_idx, bond_type) pairs
    isotope: int = 0        # RDKit GetIsotope() — 0 = natural / unspecified.
                            # 2/3 on H, 13/14 on C, 15 on N, etc.


# ---------------------------------------------------------------------------
# IsotopeLabel
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IsotopeLabel:
    """A single isotope label attached to a named parent position.

    Emitted as the IUPAC bracketed element prefix "(N-mass-symbol_count)" —
    e.g. ``(1-¹³C)`` or ``(²H4)``.  One label represents one (locant,
    element, mass_number) group; ``count`` folds multiple identically-placed
    atoms (three D's on C1 → ``count=3``) into a single bracket.
    """
    locant: "Locant | None"   # None == no explicit locant (whole-molecule or
                              # single-position parent)
    element: str              # IUPAC element symbol: "H", "C", "N", ...
    mass_number: int          # 2 for deuterium, 3 for tritium, 13/14 for C, ...
    count: int                # number of atoms of this isotope at this locant


# ---------------------------------------------------------------------------
# StereoCenter
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StereoCenter:
    atom_idx: int
    type: str               # "tetrahedral", "double_bond", "axial", "planar"
    descriptor: str | None  # "R", "S", "E", "Z" -- computed via CIP
    cip_priorities: tuple | None


# ---------------------------------------------------------------------------
# StereoDescriptor (used in plan/tree types below)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StereoDescriptor:
    """A stereo descriptor attached to a plan or tree node."""
    locant: Locant | None
    descriptor: str         # "R", "S", "E", "Z", "rel-R", etc.
    stereo_center: StereoCenter | None


# ---------------------------------------------------------------------------
# Fragment
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Fragment:
    atom_indices: frozenset[int]
    mol: Any                # RDKit mol for this fragment (not freezable)
    charge: int             # net charge


# ---------------------------------------------------------------------------
# Structural descriptors used by RingSystem
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FusionInfo:
    """Describes which rings share which edges in a fused system."""
    shared_edges: tuple[tuple[frozenset[int], frozenset[int]], ...]
    # each entry: (ring_a_atom_set, ring_b_atom_set) sharing an edge
    fusion_atoms: tuple[tuple[int, int], ...]  # (atom1, atom2) shared edge atoms


@dataclass(frozen=True)
class HeteroPosition:
    """A heteroatom position in a ring, for Hantzsch-Widman eligibility."""
    atom_idx: int
    element: str
    locant: Locant | None


# ---------------------------------------------------------------------------
# RingSystem
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RingSystem:
    atom_indices: frozenset[int]
    rings: tuple[frozenset[int], ...]    # individual rings in the system
    type: str                            # "monocyclic", "fused", "bridged", "spiro"
    aromatic: bool
    bridge_sizes: tuple[int, ...] | None # for von Baeyer: (2, 2, 1)
    spiro_sizes: tuple[int, ...] | None  # for spiro: (4, 5)
    fusion_info: FusionInfo | None       # for fused: which rings share which edges
    heteroatoms: tuple[HeteroPosition, ...] | None  # for Hantzsch-Widman eligibility
    ring_size: int                       # total atoms in the ring system

    # Ambiguous classification support:
    classification_ambiguous: bool = False  # True when fused/bridged boundary unclear
    alternate_type: str | None = None      # e.g., "bridged" when type="fused"
    alternate_bridge_sizes: tuple[int, ...] | None = None

    # Von Baeyer secondary bridges (tricyclic and larger):
    #   Each entry is ((atom_idx_a, atom_idx_b), interior_path) where:
    #     - atom_idx_a, atom_idx_b are the two endpoint atoms (already on
    #       the main bicyclic skeleton) that the secondary bridge connects
    #     - interior_path is a tuple of atom indices (possibly empty for
    #       a direct 0-atom bridge) traversing the bridge from a to b
    #   Bridge size = len(interior_path).  Listed in DESCENDING size order
    #   (IUPAC P-23.2.5).
    secondary_bridges: tuple[
        tuple[tuple[int, int], tuple[int, ...]], ...
    ] | None = None


# ---------------------------------------------------------------------------
# DetectedFG
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DetectedFG:
    type: str               # "carboxylic_acid", "alcohol", "ester", "ketone", ...
    atoms: frozenset[int]   # atoms this FG claims
    anchor: int             # defining atom (e.g., C of COOH)
    # properties stored as tuple of (key, value) pairs for hashability
    properties: tuple[tuple[str, Any], ...]
    suffix_eligible: bool   # can this FG be expressed as a suffix?
    suffix_forms: tuple[tuple[str, str], ...]  # (("terminal", "-oic acid"), ...) frozen pairs
    prefix_form: str        # "carboxy-", "oxo-", "hydroxy-", etc.
    # Nonterminal prefix form: used when the FG anchor is a branch off the parent
    # rather than part of the main chain.  E.g. aldehyde uses "oxo" when the
    # CHO carbon is the terminus of the parent chain, but "formyl" when it is
    # a pendant group (P-66.6 / P-66.6.1).  None means fall back to prefix_form.
    prefix_form_nonterminal: str | None = None

    def get_property(self, key: str, default: Any = None) -> Any:
        """Retrieve a property value by key (dict-like access for the frozen properties field)."""
        for k, v in self.properties:
            if k == key:
                return v
        return default

    def properties_dict(self) -> dict[str, Any]:
        """Return properties as a plain dict for convenient access."""
        return dict(self.properties)

    def suffix_forms_dict(self) -> dict[str, str]:
        """Return suffix_forms as a plain dict."""
        return dict(self.suffix_forms)


# ---------------------------------------------------------------------------
# AmbiguityPoint and FGFraming
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FGFraming:
    fgs: tuple[DetectedFG, ...]
    description: str            # "ester", "substituted alcohol", etc.


@dataclass(frozen=True)
class AmbiguityPoint:
    atoms: frozenset[int]
    options: tuple[FGFraming, ...]
    canonical_preference: int


# ---------------------------------------------------------------------------
# SymmetryGroup
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SymmetryGroup:
    """A set of identical substructures in the molecule."""
    subunit_atoms: tuple[frozenset[int], ...]   # each set = atoms of one copy
    subunit_mol: Any                             # RDKit mol of the canonical subunit
    linking_atoms: frozenset[int]                # atoms connecting the subunits
    linking_type: str                            # "direct_bond", "linking_group"
    linking_group_mol: Any | None                # if linking_type == "linking_group"
    multiplicity: int                            # how many identical subunits


# ---------------------------------------------------------------------------
# SuffixGroup
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SuffixGroup:
    fg: DetectedFG
    locants: tuple[Locant, ...]
    base_form: str          # "ol", "oic acid", "carboxylic acid", "al", etc.
                            # Determined by terminal vs nonterminal position.
                            # OutputForm variant applied in assembly.
    elides_terminal_e: bool # whether this suffix triggers vowel elision on parent stem
    added_indicated_h: tuple[Locant, ...] = ()
                            # P-31.1.4.2.4 / P-58.2.2 added indicated hydrogen.
                            # When a mancude retained ring (naphthalene, anthracene,
                            # etc.) carries a ring-embedded ketone/aldehyde PCG
                            # whose suffix locant breaks the aromaticity, the
                            # locant of the freshly-saturated adjacent atom is
                            # cited inline as ``(NH)`` between the suffix locant
                            # and the suffix tail.  Example:
                            # ``naphthalen-1(2H)-one`` — locants=(1,),
                            # added_indicated_h=(2,).


# ---------------------------------------------------------------------------
# UnsaturationInfix
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UnsaturationInfix:
    type: str               # "en", "yn"
    locants: tuple[Locant, ...]
    multiplier: str | None  # "di", "tri" for multiple double/triple bonds


# ---------------------------------------------------------------------------
# CandidateParent and NamedParent
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CandidateParent:
    """Structural facts only -- produced by perception. No naming yet."""
    atom_indices: frozenset[int]
    type: str                           # "chain", "monocyclic", "fused", "bridged",
                                        # "spiro", "heteroatom_center"
    length: int
    ring_system: RingSystem | None      # structural descriptor (bridge sizes, fusion, etc.)
    unsaturation: tuple[UnsaturationInfix, ...] | None
    element: str | None                 # for heteroatom parents: "P", "Si", etc.
    lambda_value: int | None            # for hypervalent heteroatoms: lambda5, etc.


@dataclass(frozen=True)
class NamedParent:
    """CandidateParent + naming decisions. Created during plan generation."""
    candidate: CandidateParent
    name: str                           # "bicyclo[2.2.1]heptane" or "norbornane"
    stem: str                           # "bicyclo[2.2.1]heptan" or "norbornan"
                                        # (for Method 2 / suffix attachment)
    alkyl_stem: str | None              # "bicyclo[2.2.1]hept" or "norborn"
                                        # (for Method 1 / -ane replacement)
                                        # None if Method 1 is not applicable
    naming_method: str                  # "systematic", "retained", "hantzsch_widman",
                                        # "von_baeyer", "spiro_systematic", "heteroatom_hydride"
    indicated_hydrogen: tuple[Locant, ...] | None
    numbering_options: tuple[Numbering, ...]  # valid numberings for this named parent
    ring_unsaturation_bonds: tuple[tuple[int, int, str], ...] | None = None
                                        # for systematic monocyclic rings with unsaturation:
                                        # tuple of (atom1_idx, atom2_idx, bond_type)
                                        # where bond_type is "double" or "triple".
                                        # Used to recompute locants after the IUPAC ring
                                        # numbering is selected.  None for all other parents.
    added_indicated_h_atoms: tuple[int, ...] | None = None
                                        # P-31.1.4.2.4 / P-58.2.2 added-indicated-H atoms.
                                        # Full-mol atom indices of the freshly-saturated
                                        # ring atoms whose locants are cited inline as
                                        # ``(NH)`` after the suffix locant — e.g. for
                                        # ``naphthalen-1(2H)-one`` the atom is the sp3
                                        # CH2 adjacent to the ring ketone.  None for
                                        # parents that do not require added-IH.
    precomposed_retained_no_suffix: bool = False
                                        # P-31.1 (Phase 8 — pyrazolone family):
                                        # set True for OPSIN data-table retained
                                        # entries whose stem already lexically
                                        # incorporates a suffix-form ending
                                        # (e.g. ``5-pyrazolone`` → ``-one``,
                                        # ``urazole`` → ``-zole``,
                                        # ``phthalhydrazide`` → ``-hydrazide``).
                                        # OPSIN refuses to glue a separable PCG
                                        # suffix onto these stems
                                        # (``5-pyrazolon-3-amine`` is rejected,
                                        # but ``3-amino-5-pyrazolone`` parses).
                                        # The substitutive-plan generator uses
                                        # this flag to drop every PCG instance
                                        # from the suffix slot, forcing the FG
                                        # to be expressed as a prefix.  Curated
                                        # retained entries (cephem, sulfolene,
                                        # …) leave this flag False because
                                        # their atom-locants metadata supports
                                        # appending suffixes like
                                        # ``cephem-4-carboxylate``.


# ---------------------------------------------------------------------------
# RetainedMatch
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RingDescriptor:
    """Describes a matched ring for parent_hydride retained matches."""
    ring_system: RingSystem
    retained_name: str
    substituent_form: str | None


@dataclass(frozen=True)
class RetainedMatch:
    name: str
    smiles: str                     # canonical SMILES this name maps to
    scope: str                      # "exact_molecule" | "parent_hydride"
    valid_output_forms: frozenset[OutputForm]
    substituent_form: str | None    # e.g., "phenyl" for benzene, "naphthyl" for naphthalene
    ring_descriptor: RingDescriptor | None  # for parent_hydride ring matches


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Decomposition:
    """A structural decomposition of the molecule into nameable pieces.
    Used by FC, multiplicative, and ring assembly paths.
    Substitutive naming does not need a Decomposition -- the plan itself
    encodes the parent/substituent split."""
    type: str                               # "functional_class",
                                            # "multiplicative", "ring_assembly"
    subtype: str | None                     # for FC: "ester", "anhydride", etc.
    pieces: tuple[Fragment, ...] | None     # for FC: the disconnected fragments
    symmetry_group: SymmetryGroup | None    # for multiplicative/ring_assembly
    locants: tuple[str, ...] | None         # for ring_assembly (multiplicative resolves in execution)
    root_atoms: frozenset[int]              # atoms forming the parent/backbone
    intramolecular: bool = False            # True for lactones, cyclic anhydrides, etc.


# ---------------------------------------------------------------------------
# FC decomposition helpers (lazy, uses RDKit)
# ---------------------------------------------------------------------------

def _build_ester_decomposition(fg: DetectedFG, mol: Any) -> Decomposition | None:
    """Build a Decomposition for an ester FG.

    Identifies:
        - acyl C (the sp2 carbon with =O and -O-)
        - carbonyl O (the =O)
        - alkyl O (the -O- linking acyl C to alkyl C)
        - alkyl C (the C bonded to alkyl O, not the acyl C)

    Then virtually cuts the alkyl-C-to-alkyl-O bond to get two connected
    components (acid side and alcohol side), and wraps each as a Fragment.

    Returns None if the expected structure can't be resolved.
    """
    # Find acyl C, alkyl O, alkyl C in the FG atom set.
    atoms = set(fg.atoms)
    acyl_c = None
    alkyl_o = None
    alkyl_c = None
    carbonyl_o = None

    for a_idx in atoms:
        atom = mol.GetAtomWithIdx(a_idx)
        if atom.GetSymbol() != "C":
            continue
        if atom.GetHybridization().__str__() != "SP2":
            continue
        # Check if this is the acyl C: double-bond to O and single-bond to O
        has_double_o = False
        has_single_o_to_c = False
        single_o_idx = None
        dbl_o_idx = None
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            if other.GetSymbol() != "O":
                continue
            bt = bond.GetBondTypeAsDouble()
            if bt == 2.0:
                has_double_o = True
                dbl_o_idx = other.GetIdx()
            elif bt == 1.0:
                # This O must be bonded to another C to qualify as ester O
                for nbr in other.GetNeighbors():
                    if nbr.GetIdx() != atom.GetIdx() and nbr.GetSymbol() == "C":
                        has_single_o_to_c = True
                        single_o_idx = other.GetIdx()
                        break
        if has_double_o and has_single_o_to_c:
            acyl_c = a_idx
            alkyl_o = single_o_idx
            carbonyl_o = dbl_o_idx
            break

    if acyl_c is None or alkyl_o is None:
        return None

    # Find alkyl C — the non-acyl C neighbor of alkyl_o
    alkyl_o_atom = mol.GetAtomWithIdx(alkyl_o)
    for nbr in alkyl_o_atom.GetNeighbors():
        if nbr.GetIdx() != acyl_c and nbr.GetSymbol() == "C":
            alkyl_c = nbr.GetIdx()
            break

    if alkyl_c is None:
        return None

    # Detect intramolecular: the alkyl C to alkyl O bond is in a ring
    # (if yes, this is a lactone -- strategy will reject).
    cut_bond = mol.GetBondBetweenAtoms(alkyl_c, alkyl_o)
    if cut_bond is None:
        return None
    intramolecular = bool(cut_bond.IsInRing())

    # Walk connected components virtually (without cutting the bond):
    # BFS from acyl_c excluding the alkyl_c-to-alkyl_o edge.
    n = mol.GetNumAtoms()
    forbidden_edge = frozenset({alkyl_c, alkyl_o})

    def _bfs(start: int) -> frozenset[int]:
        visited = {start}
        stack = [start]
        while stack:
            cur = stack.pop()
            atom = mol.GetAtomWithIdx(cur)
            for bond in atom.GetBonds():
                other = bond.GetOtherAtomIdx(cur)
                # Skip the alkyl_c-alkyl_o bond (virtual cut)
                if frozenset({cur, other}) == forbidden_edge:
                    continue
                if other not in visited:
                    visited.add(other)
                    stack.append(other)
        return frozenset(visited)

    acid_side = _bfs(acyl_c)
    alcohol_side = _bfs(alkyl_c)

    # In the intermolecular case, the two sides should be disjoint.
    # If overlap exists, this is ring-bound (intramolecular) — we still
    # emit with intramolecular=True and let strategy reject.
    if acid_side & alcohol_side:
        intramolecular = True

    # Sanity: together they must cover every heavy atom.
    total = acid_side | alcohol_side
    if not intramolecular and len(total) != n:
        # Disconnected starting fragments? Shouldn't happen for a single mol.
        return None

    # Build Fragment objects. We store atom_indices as frozensets and
    # keep a reference to the parent mol — the actual cut/carve happens
    # in carve_fc_fragments during execution.
    acid_frag = Fragment(
        atom_indices=acid_side,
        mol=mol,
        charge=0,
    )
    alcohol_frag = Fragment(
        atom_indices=alcohol_side,
        mol=mol,
        charge=0,
    )

    return Decomposition(
        type="functional_class",
        subtype="ester",
        pieces=(acid_frag, alcohol_frag),
        symmetry_group=None,
        locants=None,
        root_atoms=frozenset({acyl_c, alkyl_o, alkyl_c}),
        intramolecular=intramolecular,
    )


def _build_thio_ester_decomposition(
    fg: DetectedFG, mol: Any, subtype: str
) -> Decomposition | None:
    """Build a Decomposition for a thio-ester variant FG.

    Handles three subtypes (all share the same shape):
        thioester:    R-C(=O)-S-R'   (acid side = carbothioic S-acid)
        thionoester:  R-C(=S)-O-R'   (acid side = carbothioic O-acid)
        dithioester:  R-C(=S)-S-R'   (acid side = carbodithioic acid)

    The cut is between the alkyl C (alcohol side) and the bridging
    chalcogen atom (S or O, depending on subtype).

    Parameters
    ----------
    fg:
        The matched DetectedFG whose atoms span [acyl_c, chalc1, bridge, alkyl_c].
    mol:
        The RDKit molecule.
    subtype:
        One of "thioester", "thionoester", "dithioester".

    Returns
    -------
    A Decomposition with pieces=(acid_frag, alcohol_frag) and root_atoms
    that include acyl_c, bridge, and alkyl_c.  Or None if the expected
    structure can't be resolved.
    """
    # Identify acyl_c, bridge, alkyl_c from the FG atom set plus molecule topology.
    # Acyl C: sp2 C with exactly one =X (X=O or S depending on subtype),
    # and one single bond to a 2-coordinate Y (Y=S or O depending on subtype).
    if subtype == "thioester":
        dbl_sym, brg_sym = "O", "S"
    elif subtype == "thionoester":
        dbl_sym, brg_sym = "S", "O"
    elif subtype == "dithioester":
        dbl_sym, brg_sym = "S", "S"
    else:
        return None

    atoms = set(fg.atoms)
    acyl_c = None
    bridge = None
    alkyl_c = None

    for a_idx in atoms:
        atom = mol.GetAtomWithIdx(a_idx)
        if atom.GetSymbol() != "C":
            continue
        if atom.GetHybridization().__str__() != "SP2":
            continue
        dbl_idx = None
        brg_idx = None
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            sym = other.GetSymbol()
            bt = bond.GetBondTypeAsDouble()
            if sym == dbl_sym and bt == 2.0 and dbl_idx is None:
                dbl_idx = other.GetIdx()
            elif sym == brg_sym and bt == 1.0 and brg_idx is None:
                # Must be connected to another C to qualify as bridge
                for nbr in other.GetNeighbors():
                    if nbr.GetIdx() != a_idx and nbr.GetSymbol() == "C":
                        brg_idx = other.GetIdx()
                        break
        if dbl_idx is not None and brg_idx is not None:
            acyl_c = a_idx
            bridge = brg_idx
            break

    if acyl_c is None or bridge is None:
        return None

    # alkyl C: the non-acyl C neighbor of the bridge
    bridge_atom = mol.GetAtomWithIdx(bridge)
    for nbr in bridge_atom.GetNeighbors():
        if nbr.GetIdx() != acyl_c and nbr.GetSymbol() == "C":
            alkyl_c = nbr.GetIdx()
            break

    if alkyl_c is None:
        return None

    # Detect intramolecular (thiolactone/O-thionolactone/dithiolactone).
    cut_bond = mol.GetBondBetweenAtoms(alkyl_c, bridge)
    if cut_bond is None:
        return None
    intramolecular = bool(cut_bond.IsInRing())

    n = mol.GetNumAtoms()
    forbidden_edge = frozenset({alkyl_c, bridge})

    def _bfs(start: int) -> frozenset[int]:
        visited = {start}
        stack = [start]
        while stack:
            cur = stack.pop()
            atom = mol.GetAtomWithIdx(cur)
            for bond in atom.GetBonds():
                other = bond.GetOtherAtomIdx(cur)
                if frozenset({cur, other}) == forbidden_edge:
                    continue
                if other not in visited:
                    visited.add(other)
                    stack.append(other)
        return frozenset(visited)

    acid_side = _bfs(acyl_c)
    alcohol_side = _bfs(alkyl_c)

    if acid_side & alcohol_side:
        intramolecular = True

    total = acid_side | alcohol_side
    if not intramolecular and len(total) != n:
        return None

    acid_frag = Fragment(
        atom_indices=acid_side,
        mol=mol,
        charge=0,
    )
    alcohol_frag = Fragment(
        atom_indices=alcohol_side,
        mol=mol,
        charge=0,
    )

    return Decomposition(
        type="functional_class",
        subtype=subtype,
        pieces=(acid_frag, alcohol_frag),
        symmetry_group=None,
        locants=None,
        root_atoms=frozenset({acyl_c, bridge, alkyl_c}),
        intramolecular=intramolecular,
    )


def _build_carbamate_decomposition(fg: DetectedFG, mol: Any) -> Decomposition | None:
    """Build a Decomposition for a carbamate FG: R-O-C(=O)-NR'R''.

    The cut is between the alkyl C (alcohol side) and the bridging O,
    exactly mirroring _build_ester_decomposition.  The acid side is
    H2N-C(=O)-OH (carbamic acid) with N-substituents; the alcohol side is the
    R group that gives the "alkyl" part of the name ("butyl carbamate", etc.).

    Pieces:
        pieces[0] — carbamic_acid side (acyl C + =O + bridging O + N + N-subs)
        pieces[1] — alcohol side (alkyl C and its substituents)

    Returns None if the expected structure can't be resolved.
    """
    atoms = set(fg.atoms)
    acyl_c = None
    alkyl_o = None
    carbonyl_o = None
    n_atom = None

    # Find the acyl C: sp2 C with =O and -O- and -N
    for a_idx in atoms:
        atom = mol.GetAtomWithIdx(a_idx)
        if atom.GetSymbol() != "C":
            continue
        if atom.GetHybridization().__str__() != "SP2":
            continue
        has_double_o = False
        has_single_o = False
        has_n = False
        single_o_idx = None
        dbl_o_idx = None
        n_idx = None
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            bt = bond.GetBondTypeAsDouble()
            sym = other.GetSymbol()
            if sym == "O" and bt == 2.0:
                has_double_o = True
                dbl_o_idx = other.GetIdx()
            elif sym == "O" and bt == 1.0:
                has_single_o = True
                single_o_idx = other.GetIdx()
            elif sym == "N":
                has_n = True
                n_idx = other.GetIdx()
        if has_double_o and has_single_o and has_n:
            acyl_c = a_idx
            alkyl_o = single_o_idx
            carbonyl_o = dbl_o_idx
            n_atom = n_idx
            break

    if acyl_c is None or alkyl_o is None or n_atom is None:
        return None

    # Find alkyl C — the non-acyl C neighbor of alkyl_o
    alkyl_c = None
    alkyl_o_atom = mol.GetAtomWithIdx(alkyl_o)
    for nbr in alkyl_o_atom.GetNeighbors():
        if nbr.GetIdx() != acyl_c:
            alkyl_c = nbr.GetIdx()
            break

    if alkyl_c is None:
        return None

    # Detect intramolecular (ring-bound O-C(=O)-N)
    cut_bond = mol.GetBondBetweenAtoms(alkyl_c, alkyl_o)
    if cut_bond is None:
        return None
    intramolecular = bool(cut_bond.IsInRing())

    # Ring-embedded carbamate nitrogen (e.g. piperidine-1-carboxylate,
    # morpholine-4-carboxylate, solifenacin's tetrahydroisoquinoline).
    # When the N belongs to a ring that the carbamate C=O is not part of,
    # the two N-substituents are connected through the ring, so splitting
    # into "N-[sub1]-N-[sub2]-carbamate" would duplicate the ring backbone.
    # Flag as intramolecular so strategy.accept_plan rejects the FC plan
    # and the substitutive path (ring parent + carbamate/ester suffix) wins.
    n_atom_obj = mol.GetAtomWithIdx(n_atom)
    if n_atom_obj.IsInRing():
        from rdkit.Chem import GetSymmSSSR
        for ring in GetSymmSSSR(mol):
            ring_atoms = set(ring)
            if n_atom in ring_atoms and acyl_c not in ring_atoms:
                intramolecular = True
                break

    n = mol.GetNumAtoms()
    forbidden_edge = frozenset({alkyl_c, alkyl_o})

    def _bfs(start: int) -> frozenset:
        visited = {start}
        stack = [start]
        while stack:
            cur = stack.pop()
            atom = mol.GetAtomWithIdx(cur)
            for bond in atom.GetBonds():
                other = bond.GetOtherAtomIdx(cur)
                if frozenset({cur, other}) == forbidden_edge:
                    continue
                if other not in visited:
                    visited.add(other)
                    stack.append(other)
        return frozenset(visited)

    carbamic_side = _bfs(acyl_c)
    alcohol_side = _bfs(alkyl_c)

    if carbamic_side & alcohol_side:
        intramolecular = True

    total = carbamic_side | alcohol_side
    if not intramolecular and len(total) != n:
        return None

    carbamic_frag = Fragment(atom_indices=carbamic_side, mol=mol, charge=0)
    alcohol_frag = Fragment(atom_indices=alcohol_side, mol=mol, charge=0)

    return Decomposition(
        type="functional_class",
        subtype="carbamate",
        pieces=(carbamic_frag, alcohol_frag),
        symmetry_group=None,
        locants=None,
        root_atoms=frozenset({acyl_c, alkyl_o, alkyl_c, n_atom}),
        intramolecular=intramolecular,
    )


def _build_thio_carbamate_decomposition(
    fg: DetectedFG, mol: Any, subtype: str
) -> Decomposition | None:
    """Build a Decomposition for a thiono- or dithio-carbamate FG.

    Handles two subtypes (both N-substituted analogues of a thio-ester):
        thionocarbamate:  R2N-C(=S)-O-R'   (parent = carbamothioic O-acid)
        dithiocarbamate:  R2N-C(=S)-S-R'   (parent = carbamodithioic acid)

    The cut is between the alkyl C (alcohol/thiol side) and the bridging
    chalcogen atom (O for thionocarbamate, S for dithiocarbamate), exactly
    mirroring the carbamate builder.

    Pieces:
        pieces[0] — carbamoyl side (acyl C + =S + bridging chalc + N + N-subs)
        pieces[1] — R' side (alkyl/aryl C and its substituents)

    Returns None if the expected structure can't be resolved.
    """
    if subtype == "thionocarbamate":
        brg_sym = "O"
    elif subtype == "dithiocarbamate":
        brg_sym = "S"
    else:
        return None

    atoms = set(fg.atoms)
    acyl_c = None
    bridge = None
    n_atom = None

    # Find the acyl C: sp2 C with =S, single bond to bridge chalcogen, and single bond to N.
    for a_idx in atoms:
        atom = mol.GetAtomWithIdx(a_idx)
        if atom.GetSymbol() != "C":
            continue
        if atom.GetHybridization().__str__() != "SP2":
            continue
        has_double_s = False
        brg_idx = None
        n_idx = None
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            bt = bond.GetBondTypeAsDouble()
            sym = other.GetSymbol()
            if sym == "S" and bt == 2.0:
                has_double_s = True
            elif sym == brg_sym and bt == 1.0 and brg_idx is None:
                # Must connect to another C to be the bridge
                for nbr in other.GetNeighbors():
                    if nbr.GetIdx() != a_idx and nbr.GetSymbol() == "C":
                        brg_idx = other.GetIdx()
                        break
            elif sym == "N" and bt == 1.0 and n_idx is None:
                n_idx = other.GetIdx()
        if has_double_s and brg_idx is not None and n_idx is not None:
            acyl_c = a_idx
            bridge = brg_idx
            n_atom = n_idx
            break

    if acyl_c is None or bridge is None or n_atom is None:
        return None

    # alkyl C: the non-acyl C neighbor of the bridge
    bridge_atom = mol.GetAtomWithIdx(bridge)
    alkyl_c = None
    for nbr in bridge_atom.GetNeighbors():
        if nbr.GetIdx() != acyl_c and nbr.GetSymbol() == "C":
            alkyl_c = nbr.GetIdx()
            break

    if alkyl_c is None:
        return None

    # Detect intramolecular (ring-bound).
    cut_bond = mol.GetBondBetweenAtoms(alkyl_c, bridge)
    if cut_bond is None:
        return None
    intramolecular = bool(cut_bond.IsInRing())

    # Ring-embedded nitrogen: see _build_carbamate_decomposition for the
    # rationale. Flag as intramolecular so FC is rejected and substitutive
    # naming takes over.
    n_atom_obj = mol.GetAtomWithIdx(n_atom)
    if n_atom_obj.IsInRing():
        from rdkit.Chem import GetSymmSSSR
        for ring in GetSymmSSSR(mol):
            ring_atoms = set(ring)
            if n_atom in ring_atoms and acyl_c not in ring_atoms:
                intramolecular = True
                break

    n_total = mol.GetNumAtoms()
    forbidden_edge = frozenset({alkyl_c, bridge})

    def _bfs(start: int) -> frozenset[int]:
        visited = {start}
        stack = [start]
        while stack:
            cur = stack.pop()
            atom = mol.GetAtomWithIdx(cur)
            for bond in atom.GetBonds():
                other = bond.GetOtherAtomIdx(cur)
                if frozenset({cur, other}) == forbidden_edge:
                    continue
                if other not in visited:
                    visited.add(other)
                    stack.append(other)
        return frozenset(visited)

    carbamoyl_side = _bfs(acyl_c)
    alcohol_side = _bfs(alkyl_c)

    if carbamoyl_side & alcohol_side:
        intramolecular = True

    total = carbamoyl_side | alcohol_side
    if not intramolecular and len(total) != n_total:
        return None

    carbamoyl_frag = Fragment(atom_indices=carbamoyl_side, mol=mol, charge=0)
    alcohol_frag = Fragment(atom_indices=alcohol_side, mol=mol, charge=0)

    return Decomposition(
        type="functional_class",
        subtype=subtype,
        pieces=(carbamoyl_frag, alcohol_frag),
        symmetry_group=None,
        locants=None,
        root_atoms=frozenset({acyl_c, bridge, alkyl_c, n_atom}),
        intramolecular=intramolecular,
    )


def _build_carbamothioate_decomposition(
    fg: DetectedFG, mol: Any
) -> Decomposition | None:
    """Build a Decomposition for an S-substituted carbamothioate FG:
    R2N-C(=O)-S-R'.

    Subtype = "carbamothioate".  Surface form: ``S-R' N-R,N-R'-carbamothioate``
    (P-66.6.5.5).  The cut is between the alkyl C (the R' side) and the
    bridging sulfur, exactly mirroring the carbamate / thionocarbamate
    builders but with the bridge element being S and the carbonyl C bearing
    a =O (not =S).

    Pieces:
        pieces[0] — carbamoyl side (acyl C + =O + bridging S + N + N-subs)
        pieces[1] — R' side (alkyl/aryl C and its substituents)

    Returns None if the expected structure can't be resolved.
    """
    atoms = set(fg.atoms)
    acyl_c = None
    bridge = None
    n_atom = None

    # Find the acyl C: sp2 C with =O, single bond to bridge S, and single bond to N.
    for a_idx in atoms:
        atom = mol.GetAtomWithIdx(a_idx)
        if atom.GetSymbol() != "C":
            continue
        if atom.GetHybridization().__str__() != "SP2":
            continue
        has_double_o = False
        brg_idx = None
        n_idx = None
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            bt = bond.GetBondTypeAsDouble()
            sym = other.GetSymbol()
            if sym == "O" and bt == 2.0:
                has_double_o = True
            elif sym == "S" and bt == 1.0 and brg_idx is None:
                # Must connect to another C to be the bridge
                for nbr in other.GetNeighbors():
                    if nbr.GetIdx() != a_idx and nbr.GetSymbol() == "C":
                        brg_idx = other.GetIdx()
                        break
            elif sym == "N" and bt == 1.0 and n_idx is None:
                n_idx = other.GetIdx()
        if has_double_o and brg_idx is not None and n_idx is not None:
            acyl_c = a_idx
            bridge = brg_idx
            n_atom = n_idx
            break

    if acyl_c is None or bridge is None or n_atom is None:
        return None

    # alkyl C: the non-acyl C neighbor of the bridge
    bridge_atom = mol.GetAtomWithIdx(bridge)
    alkyl_c = None
    for nbr in bridge_atom.GetNeighbors():
        if nbr.GetIdx() != acyl_c and nbr.GetSymbol() == "C":
            alkyl_c = nbr.GetIdx()
            break

    if alkyl_c is None:
        return None

    # Detect intramolecular (ring-bound).
    cut_bond = mol.GetBondBetweenAtoms(alkyl_c, bridge)
    if cut_bond is None:
        return None
    intramolecular = bool(cut_bond.IsInRing())

    # Ring-embedded nitrogen: see _build_carbamate_decomposition for the
    # rationale.  Flag as intramolecular so the FC plan is rejected and
    # substitutive naming takes over.
    n_atom_obj = mol.GetAtomWithIdx(n_atom)
    if n_atom_obj.IsInRing():
        from rdkit.Chem import GetSymmSSSR
        for ring in GetSymmSSSR(mol):
            ring_atoms = set(ring)
            if n_atom in ring_atoms and acyl_c not in ring_atoms:
                intramolecular = True
                break

    n_total = mol.GetNumAtoms()
    forbidden_edge = frozenset({alkyl_c, bridge})

    def _bfs(start: int) -> frozenset[int]:
        visited = {start}
        stack = [start]
        while stack:
            cur = stack.pop()
            atom = mol.GetAtomWithIdx(cur)
            for bond in atom.GetBonds():
                other = bond.GetOtherAtomIdx(cur)
                if frozenset({cur, other}) == forbidden_edge:
                    continue
                if other not in visited:
                    visited.add(other)
                    stack.append(other)
        return frozenset(visited)

    carbamoyl_side = _bfs(acyl_c)
    alcohol_side = _bfs(alkyl_c)

    if carbamoyl_side & alcohol_side:
        intramolecular = True

    total = carbamoyl_side | alcohol_side
    if not intramolecular and len(total) != n_total:
        return None

    carbamoyl_frag = Fragment(atom_indices=carbamoyl_side, mol=mol, charge=0)
    alcohol_frag = Fragment(atom_indices=alcohol_side, mol=mol, charge=0)

    return Decomposition(
        type="functional_class",
        subtype="carbamothioate",
        pieces=(carbamoyl_frag, alcohol_frag),
        symmetry_group=None,
        locants=None,
        root_atoms=frozenset({acyl_c, bridge, alkyl_c, n_atom}),
        intramolecular=intramolecular,
    )


def _build_symmetric_diester_decomposition(
    ester_fgs: list["DetectedFG"], mol: Any
) -> "Decomposition | None":
    """Build a Decomposition for a symmetric diester (e.g. diallyl oxalate).

    A symmetric diester has two ester groups sharing a common diacid backbone,
    both esterified with the same alkyl/aryl group R.  The canonical form is
    ``R-O-C(=O)-[acid_chain]-C(=O)-O-R``.

    Algorithm
    ---------
    1. For every pair of ester FGs, identify their acyl_c and alkyl_c atoms.
    2. Check that the two acyl_c atoms are connected (direct or via a chain of
       carbons with no heteroatoms — the diacid backbone).
    3. Extract the two R groups (alkyl_c side of each ester) as canonical SMILES.
       If they are NOT identical, reject (not symmetric).
    4. Build a diacid backbone molecule by virtual-cutting both alkyl_c-to-O
       bonds and replacing the dummy atoms with OH.
    5. Return a Decomposition with:
           subtype="symmetric_diester"
           pieces=(acid_frag, r_frag)   — acid_frag covers the diacid backbone,
                                          r_frag covers ONE of the two identical
                                          R groups (both sides).
           root_atoms = {acyl_c_1, alkyl_o_1, alkyl_c_1,
                         acyl_c_2, alkyl_o_2, alkyl_c_2}

    Returns None if the structure cannot be cleanly carved as a symmetric diester.
    """
    from rdkit import Chem

    if len(ester_fgs) < 2:
        return None

    # -----------------------------------------------------------------------
    # Step 1: find acyl_c / alkyl_o / alkyl_c for each ester FG.
    # -----------------------------------------------------------------------
    def _parse_ester_fg(fg: "DetectedFG"):
        """Return (acyl_c, alkyl_o, alkyl_c) or None."""
        atoms = set(fg.atoms)
        for a_idx in atoms:
            atom = mol.GetAtomWithIdx(a_idx)
            if atom.GetSymbol() != "C":
                continue
            if atom.GetHybridization().__str__() != "SP2":
                continue
            dbl_o_idx = None
            sng_o_idx = None
            for bond in atom.GetBonds():
                other = bond.GetOtherAtom(atom)
                if other.GetSymbol() != "O":
                    continue
                bt = bond.GetBondTypeAsDouble()
                if bt == 2.0 and dbl_o_idx is None:
                    dbl_o_idx = other.GetIdx()
                elif bt == 1.0 and sng_o_idx is None:
                    # Single-bond O must connect to a non-acyl carbon to be
                    # the ester (alkyl) oxygen.
                    for nbr in other.GetNeighbors():
                        if nbr.GetIdx() != a_idx and nbr.GetSymbol() in ("C", "c"):
                            sng_o_idx = other.GetIdx()
                            break
            if dbl_o_idx is not None and sng_o_idx is not None:
                # alkyl_c: C neighbor of alkyl O that is not the acyl C
                alkyl_o_atom = mol.GetAtomWithIdx(sng_o_idx)
                for nbr in alkyl_o_atom.GetNeighbors():
                    if nbr.GetIdx() != a_idx and nbr.GetAtomicNum() == 6:
                        return (a_idx, sng_o_idx, nbr.GetIdx())
        return None

    # Collect valid (acyl_c, alkyl_o, alkyl_c) tuples for all ester FGs
    parsed: list[tuple[int, int, int]] = []
    for fg in ester_fgs:
        result = _parse_ester_fg(fg)
        if result is not None:
            parsed.append(result)

    if len(parsed) < 2:
        return None

    # -----------------------------------------------------------------------
    # Step 2: try all pairs; find one where acyl_c atoms are connected through
    # a carbon-only path (the diacid backbone).
    # -----------------------------------------------------------------------
    def _diacid_backbone(acyl_c1: int, acyl_c2: int) -> frozenset[int] | None:
        """BFS from acyl_c1 to acyl_c2 through heavy atoms excluding ester O atoms.

        Only carbon atoms are allowed in the backbone (no heteroatoms bridging
        the two acyl carbons — those would indicate a non-standard diacid).
        The path may be a direct bond (oxalate: C-C) or through methylenes
        (malonate, succinate, ...).

        Returns the frozenset of backbone atom indices (including both acyl_c
        endpoints) or None if no such path exists.
        """
        # BFS from acyl_c1, constrained to C atoms, looking for acyl_c2.
        # We must not cross through the ester oxygens (alkyl_o) or the carbonyl
        # oxygens — but those are not C atoms, so the C-only constraint handles it.
        visited: set[int] = {acyl_c1}
        stack: list[list[int]] = [[acyl_c1]]
        while stack:
            path = stack.pop()
            current = path[-1]
            atom = mol.GetAtomWithIdx(current)
            for bond in atom.GetBonds():
                nbr_idx = bond.GetOtherAtomIdx(current)
                if nbr_idx in visited:
                    continue
                nbr_atom = mol.GetAtomWithIdx(nbr_idx)
                if nbr_atom.GetAtomicNum() != 6:
                    continue  # only carbon backbone
                new_path = path + [nbr_idx]
                if nbr_idx == acyl_c2:
                    return frozenset(new_path)
                visited.add(nbr_idx)
                stack.append(new_path)
        return None

    best_triple: tuple | None = None  # (parsed[i], parsed[j], backbone)
    for i in range(len(parsed)):
        for j in range(i + 1, len(parsed)):
            ac1, ao1, rc1 = parsed[i]
            ac2, ao2, rc2 = parsed[j]
            backbone = _diacid_backbone(ac1, ac2)
            if backbone is not None:
                best_triple = (parsed[i], parsed[j], backbone)
                break
        if best_triple is not None:
            break

    if best_triple is None:
        return None

    (ac1, ao1, rc1), (ac2, ao2, rc2), backbone = best_triple

    # -----------------------------------------------------------------------
    # Step 3: check symmetry — both R groups must have identical canonical SMILES.
    # -----------------------------------------------------------------------
    def _r_group_smiles(alkyl_c: int, alkyl_o: int) -> str | None:
        """Return canonical SMILES of the R group attached at alkyl_c."""
        from rdkit import Chem as _Chem
        bond = mol.GetBondBetweenAtoms(alkyl_c, alkyl_o)
        if bond is None:
            return None
        fragmented = _Chem.FragmentOnBonds(
            mol, [bond.GetIdx()], addDummies=True, dummyLabels=[(0, 0)]
        )
        frag_atom_lists = _Chem.GetMolFrags(fragmented, asMols=False)
        frag_mols_list = _Chem.GetMolFrags(fragmented, asMols=True, sanitizeFrags=False)
        r_frag_idx = None
        for fi, orig_indices in enumerate(frag_atom_lists):
            if alkyl_c in orig_indices:
                r_frag_idx = fi
                break
        if r_frag_idx is None:
            return None
        rw = _Chem.RWMol(frag_mols_list[r_frag_idx])
        for a in rw.GetAtoms():
            if a.GetAtomicNum() == 0:
                a.SetAtomicNum(1)
                a.SetNoImplicit(False)
        try:
            _Chem.SanitizeMol(rw)
        except Exception:
            pass
        try:
            no_h = _Chem.RemoveHs(rw.GetMol())
        except Exception:
            no_h = rw.GetMol()
        return _Chem.MolToSmiles(no_h)

    smi1 = _r_group_smiles(rc1, ao1)
    smi2 = _r_group_smiles(rc2, ao2)
    if smi1 is None or smi2 is None or smi1 != smi2:
        return None  # not symmetric

    # -----------------------------------------------------------------------
    # Step 3b: check that the diacid backbone is unsubstituted.
    #
    # The backbone atoms are the carbon chain from ac1 to ac2 (inclusive).
    # Each INTERIOR backbone carbon (not ac1 or ac2) must have degree 2
    # within the backbone.  If it has additional bonds to non-backbone,
    # non-H atoms, the diacid has substituents (e.g. 2-propylmalonate).
    # In that case we do NOT use the symmetric diester path — the regular
    # single-ester FC path handles the compound better.
    # -----------------------------------------------------------------------
    backbone_set = set(backbone)  # {ac1, ..., ac2}
    for b_atom_idx in backbone_set:
        if b_atom_idx in (ac1, ac2):
            continue  # terminal acyl_c: they carry =O and -O- bonds, OK
        b_atom = mol.GetAtomWithIdx(b_atom_idx)
        for bond in b_atom.GetBonds():
            nbr_idx = bond.GetOtherAtomIdx(b_atom_idx)
            nbr = mol.GetAtomWithIdx(nbr_idx)
            if nbr_idx in backbone_set:
                continue  # within backbone
            if nbr.GetAtomicNum() == 1:
                continue  # H (implicit or explicit)
            # Non-backbone heavy atom attached to interior backbone carbon
            # → substituted backbone; abort symmetric diester path.
            return None

    # -----------------------------------------------------------------------
    # Step 4: build Fragment objects.
    # acid_side = backbone + both carbonyl Os + both alkyl Os
    # r_side = R group of either alkyl_c (they're identical; pick rc1)
    # -----------------------------------------------------------------------
    # Collect all ester FG atoms that belong to the acid side (both FG atom sets
    # minus the two alkyl_c atoms).
    all_fg_atoms: set[int] = set()
    for fg in ester_fgs:
        all_fg_atoms.update(fg.atoms)
    # The acid side = entire molecule minus both alkyl_c atoms' R groups
    # We define it by BFS from ac1 excluding both (alkyl_c, alkyl_o) edges.
    n = mol.GetNumAtoms()
    forbidden: set[frozenset] = {
        frozenset({rc1, ao1}),
        frozenset({rc2, ao2}),
    }

    def _bfs_acid(start: int) -> frozenset[int]:
        visited: set[int] = {start}
        stack = [start]
        while stack:
            cur = stack.pop()
            atom = mol.GetAtomWithIdx(cur)
            for bond in atom.GetBonds():
                other = bond.GetOtherAtomIdx(cur)
                if frozenset({cur, other}) in forbidden:
                    continue
                if other not in visited:
                    visited.add(other)
                    stack.append(other)
        return frozenset(visited)

    acid_side = _bfs_acid(ac1)
    r_side_1 = frozenset(set(range(n)) - acid_side)  # rc1 + its tail

    # Detect intramolecular (ring-bound ester) — reject
    bond1 = mol.GetBondBetweenAtoms(rc1, ao1)
    bond2 = mol.GetBondBetweenAtoms(rc2, ao2)
    if bond1 is None or bond2 is None:
        return None
    if bond1.IsInRing() or bond2.IsInRing():
        return True  # type: ignore  # will be caught — flag intramolecular
    intramolecular = bool(bond1.IsInRing() or bond2.IsInRing())

    # Sanity: acid_side + r_side_1 must together cover both R groups correctly
    # (r_side_1 = rc1 group, rc2 group is inside acid_side — that's wrong).
    # We need to carve BOTH R groups from acid side separately.
    # Actually the pieces[1] is only ONE r group; the assembler emits "di-" prefix.
    # Both rc1 and rc2 must NOT be in acid_side after the cut.
    # Let me verify: acid_side starts from ac1 excluding both forbidden edges.
    # So acid_side does not include rc1 or rc2. r_side_1 = atoms not in acid_side
    # = rc1 group + rc2 group. We only need ONE for pieces[1] since they're identical.
    # Carve just rc1's group (from rc1, not crossing ao2/rc2 bond).
    r_side_rc1: set[int] = set()
    stack2 = [rc1]
    while stack2:
        cur = stack2.pop()
        if cur in r_side_rc1:
            continue
        r_side_rc1.add(cur)
        atom = mol.GetAtomWithIdx(cur)
        for bond in atom.GetBonds():
            other = bond.GetOtherAtomIdx(cur)
            if frozenset({cur, other}) in forbidden:
                continue
            if other not in r_side_rc1 and other not in acid_side:
                stack2.append(other)
    r_side_rc1_fset = frozenset(r_side_rc1)

    acid_frag = Fragment(atom_indices=acid_side, mol=mol, charge=0)
    r_frag = Fragment(atom_indices=r_side_rc1_fset, mol=mol, charge=0)

    return Decomposition(
        type="functional_class",
        subtype="symmetric_diester",
        pieces=(acid_frag, r_frag),
        symmetry_group=None,
        locants=None,
        root_atoms=frozenset({ac1, ao1, rc1, ac2, ao2, rc2}),
        intramolecular=intramolecular,
    )


def _build_polyester_decomposition(
    ester_fgs: list["DetectedFG"], mol: Any
) -> "Decomposition | None":
    """Build a Decomposition for a fully-esterified poly-acid (P-65.6.3.3.2).

    Generalises the symmetric-diester path to arbitrary parent acid skeletons
    (aromatic, heterocyclic, branched aliphatic) and to *mixed* alkyl groups.

    A poly-ester has two or more ester groups whose acyl carbons all sit on a
    single connected parent skeleton, and where EVERY acid-like group on that
    parent is esterified (no free -COOH / carboxylate).  Such a structure is
    named as the functional-class ester of the parent poly-acid:

        ``<alkyl word(s)> <parent>...dicarboxylate / ...dioate``

    e.g. ``ethyl methyl benzene-1,3-dicarboxylate``,
    ``dimethyl benzene-1,2-dicarboxylate``,
    ``trimethyl butane-1,1,3-tricarboxylate``.

    The acid skeleton is named by the engine's ordinary machinery (it produces
    ``benzene-1,3-dicarboxylate`` etc.); this builder only carves the structure
    and records, per ester, the acyl carbon (so assembly can attach a locant to
    the alkyl word when the parent positions are not symmetry-equivalent).

    Returns None when the structure is not a clean, fully-esterified poly-ester
    (partial esters, lactones, mixed acid components, single ester) so that the
    caller falls back to the substitutive / monoester path.
    """
    from rdkit import Chem

    if len(ester_fgs) < 2:
        return None

    def _parse_ester_fg(fg: "DetectedFG"):
        """Return (acyl_c, alkyl_o, alkyl_c) or None for a -C(=O)-O-R ester."""
        atoms = set(fg.atoms)
        for a_idx in atoms:
            atom = mol.GetAtomWithIdx(a_idx)
            if atom.GetSymbol() != "C":
                continue
            if atom.GetHybridization().__str__() != "SP2":
                continue
            dbl_o_idx = None
            sng_o_idx = None
            for bond in atom.GetBonds():
                other = bond.GetOtherAtom(atom)
                if other.GetSymbol() != "O":
                    continue
                bt = bond.GetBondTypeAsDouble()
                if bt == 2.0 and dbl_o_idx is None:
                    dbl_o_idx = other.GetIdx()
                elif bt == 1.0 and sng_o_idx is None:
                    for nbr in other.GetNeighbors():
                        if nbr.GetIdx() != a_idx and nbr.GetSymbol() in ("C", "c"):
                            sng_o_idx = other.GetIdx()
                            break
            if dbl_o_idx is not None and sng_o_idx is not None:
                alkyl_o_atom = mol.GetAtomWithIdx(sng_o_idx)
                for nbr in alkyl_o_atom.GetNeighbors():
                    if nbr.GetIdx() != a_idx and nbr.GetAtomicNum() == 6:
                        return (a_idx, sng_o_idx, nbr.GetIdx())
        return None

    parsed: list[tuple[int, int, int]] = []
    for fg in ester_fgs:
        result = _parse_ester_fg(fg)
        if result is not None:
            parsed.append(result)
    if len(parsed) < 2:
        return None

    acyl_cs = {p[0] for p in parsed}
    alkyl_os = {p[1] for p in parsed}
    alkyl_cs = {p[2] for p in parsed}
    # Cut bonds = each alkyl_c -- alkyl_o pair.
    cut_pairs = {frozenset({ao, ac}) for (_acyl, ao, ac) in parsed}

    # -----------------------------------------------------------------------
    # The acid skeleton is the connected component containing the acyl carbons
    # after removing the alkyl-C--alkyl-O ester bonds.  BFS from one acyl_c.
    # -----------------------------------------------------------------------
    def _acid_skeleton(start: int) -> frozenset[int]:
        visited = {start}
        stack = [start]
        while stack:
            cur = stack.pop()
            atom = mol.GetAtomWithIdx(cur)
            for bond in atom.GetBonds():
                other = bond.GetOtherAtomIdx(cur)
                if frozenset({cur, other}) in cut_pairs:
                    continue
                if other not in visited:
                    visited.add(other)
                    stack.append(other)
        return frozenset(visited)

    acid_side = _acid_skeleton(parsed[0][0])

    # All acyl carbons must lie on the same skeleton (single acid component).
    if not acyl_cs.issubset(acid_side):
        return None
    # The alkyl oxygens are part of the acid side (the -O- stays with -COOH).
    if not alkyl_os.issubset(acid_side):
        return None
    # The alkyl carbons (R groups) must NOT be on the acid side; otherwise the
    # cut did not separate the alcohol component (intramolecular / cyclic ester).
    if alkyl_cs & acid_side:
        return None

    # -----------------------------------------------------------------------
    # Reject if the acid skeleton carries a FREE acid group (partial ester):
    # any carboxyl/sulfonic/etc. -OH or -[O-] that is NOT one of our ester
    # oxygens.  Per P-65.6.3.3.5 a partial ester's PIN is substitutive
    # (free acid + alkoxycarbonyl prefixes), so we must NOT claim it here.
    #
    # Detection: scan acid-side carbons that bear a =O AND a single-bonded O.
    # If that single-bonded O is terminal (-OH or -O[-], i.e. not bridging to
    # another heavy atom) then it is a free acid arm -> reject.
    # -----------------------------------------------------------------------
    for c_idx in acid_side:
        c_atom = mol.GetAtomWithIdx(c_idx)
        if c_atom.GetAtomicNum() != 6:
            continue
        has_dbl_o = False
        free_acid_o = False
        for bond in c_atom.GetBonds():
            other = bond.GetOtherAtom(c_atom)
            if other.GetAtomicNum() != 8:
                continue
            bt = bond.GetBondTypeAsDouble()
            if bt == 2.0:
                has_dbl_o = True
            elif bt == 1.0:
                # Terminal O (only neighbour is this carbon) -> free acid arm.
                o_heavy_nbrs = [
                    nb for nb in other.GetNeighbors() if nb.GetAtomicNum() != 1
                ]
                if len(o_heavy_nbrs) == 1:
                    free_acid_o = True
        if has_dbl_o and free_acid_o:
            return None

    # -----------------------------------------------------------------------
    # Reject intramolecular (cyclic ester / lactone): any ester C-O cut bond
    # that lies in a ring would split a ring -> not an intermolecular ester.
    # -----------------------------------------------------------------------
    for (_acyl, ao, ac) in parsed:
        bond = mol.GetBondBetweenAtoms(ao, ac)
        if bond is None or bond.IsInRing():
            return None

    n = mol.GetNumAtoms()
    r_side = frozenset(set(range(n)) - acid_side)
    if not r_side:
        return None

    acid_frag = Fragment(atom_indices=acid_side, mol=mol, charge=0)
    r_frag = Fragment(atom_indices=r_side, mol=mol, charge=0)

    # root_atoms encodes every ester core atom; carve_fc_fragments rebuilds the
    # (acyl_c, alkyl_o, alkyl_c) triples from it.
    root: set[int] = set()
    for (acyl, ao, ac) in parsed:
        root.update((acyl, ao, ac))

    return Decomposition(
        type="functional_class",
        subtype="polyester",
        pieces=(acid_frag, r_frag),
        symmetry_group=None,
        locants=None,
        root_atoms=frozenset(root),
        intramolecular=False,
    )


def _build_acyl_isothiocyanate_decomposition(fg: "DetectedFG", mol: Any) -> "Decomposition | None":
    """Build a Decomposition for an acyl isothiocyanate FG: R-C(=O)-N=C=S.

    Cut bond: acyl_C -- N.
    Pieces:
        pieces[0] — acid side (R + acyl_C + =O), named as ACID_STEM -> "benzoyl"
    The N=C=S atoms become the fixed class word "isothiocyanate" in assembly.
    """
    atoms = set(fg.atoms)
    acyl_c = None
    n_atom = None

    # Find acyl C: sp2 C in FG with =O and -N= neighbours
    for a_idx in atoms:
        atom = mol.GetAtomWithIdx(a_idx)
        if atom.GetSymbol() != "C":
            continue
        if atom.GetHybridization().__str__() != "SP2":
            continue
        has_double_o = False
        has_n = False
        n_idx = None
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            bt = bond.GetBondTypeAsDouble()
            sym = other.GetSymbol()
            if sym == "O" and bt == 2.0:
                has_double_o = True
            elif sym == "N":
                has_n = True
                n_idx = other.GetIdx()
        if has_double_o and has_n:
            acyl_c = a_idx
            n_atom = n_idx
            break

    if acyl_c is None or n_atom is None:
        return None

    # Verify N is in FG atoms (i.e., part of N=C=S)
    if n_atom not in atoms:
        return None

    # Cut bond: acyl_C -- N
    cut_bond = mol.GetBondBetweenAtoms(acyl_c, n_atom)
    if cut_bond is None or cut_bond.IsInRing():
        return None

    # BFS acid_side from acyl_C, excluding the acyl_C-N bond
    n_total = mol.GetNumAtoms()
    forbidden_edge = frozenset({acyl_c, n_atom})

    def _bfs(start: int) -> frozenset:
        visited: set[int] = {start}
        stack = [start]
        while stack:
            cur = stack.pop()
            atom_cur = mol.GetAtomWithIdx(cur)
            for bond in atom_cur.GetBonds():
                other = bond.GetOtherAtomIdx(cur)
                if frozenset({cur, other}) == forbidden_edge:
                    continue
                if other not in visited:
                    visited.add(other)
                    stack.append(other)
        return frozenset(visited)

    acid_side = _bfs(acyl_c)
    ncs_side = _bfs(n_atom)

    # Must be cleanly split
    if acid_side & ncs_side:
        return None
    if len(acid_side) + len(ncs_side) != n_total:
        return None

    acid_frag = Fragment(atom_indices=acid_side, mol=mol, charge=0)

    return Decomposition(
        type="functional_class",
        subtype="acyl_isothiocyanate",
        pieces=(acid_frag,),   # only acid piece; NCS is the class word
        symmetry_group=None,
        locants=None,
        root_atoms=frozenset({acyl_c, n_atom}),
        intramolecular=False,
    )


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Interpretation:
    fgs: tuple[DetectedFG, ...]
    ambiguity_choices: tuple[tuple[int, int], ...]  # (ambiguity_point_idx, option_idx)
    ring_systems: tuple[RingSystem, ...]            # shared across interpretations
    stereocenters: tuple[StereoCenter, ...]         # shared across interpretations
    symmetry_groups: tuple[SymmetryGroup, ...]      # shared across interpretations

    def decomposition_candidates(self, mol: Any = None) -> Iterator[Decomposition]:
        """Yield FC, multiplicative, ring assembly decompositions.
        Substitutive is always available and handled directly by
        SubstitutivePath -- it does not appear here.

        `mol` is the RDKit molecule corresponding to this interpretation.
        It is required to generate FC decompositions (which need to carve
        pieces and inspect ring bonds). If None, no FC decompositions are
        yielded."""
        if mol is None:
            return
        # --- Functional Class: esters (P-66.6) ---
        # For each detected ester FG, emit an FC decomposition with the
        # acid and alcohol fragments. Intramolecular (lactone) cases are
        # flagged on the decomposition; strategy rejects them (Phase 2d
        # handles intermolecular only).
        for fg in self.fgs:
            if fg.type != "ester":
                continue
            decomp = _build_ester_decomposition(fg, mol)
            if decomp is not None:
                yield decomp
        # --- Functional Class: carbamates (P-66.6) ---
        for fg in self.fgs:
            if fg.type != "carbamate":
                continue
            decomp = _build_carbamate_decomposition(fg, mol)
            if decomp is not None:
                yield decomp
        # --- Functional Class: acyl isothiocyanates ---
        for fg in self.fgs:
            if fg.type != "acyl_isothiocyanate":
                continue
            decomp = _build_acyl_isothiocyanate_decomposition(fg, mol)
            if decomp is not None:
                yield decomp
        # --- Functional Class: thio-ester variants (P-65.6.3) ---
        for fg in self.fgs:
            if fg.type not in ("thioester", "thionoester", "dithioester"):
                continue
            decomp = _build_thio_ester_decomposition(fg, mol, fg.type)
            if decomp is not None:
                yield decomp
        # --- Functional Class: thiono- / dithio-carbamates (P-66.6.5.5) ---
        for fg in self.fgs:
            if fg.type not in ("thionocarbamate", "dithiocarbamate"):
                continue
            decomp = _build_thio_carbamate_decomposition(fg, mol, fg.type)
            if decomp is not None:
                yield decomp
        # --- Functional Class: carbamothioate (S-substituted thiocarbamate,
        #     R2N-C(=O)-S-R').  Same shape as carbamate but with S replacing
        #     the bridging O — reuses _build_carbamothioate_decomposition.
        for fg in self.fgs:
            if fg.type != "carbamothioate":
                continue
            decomp = _build_carbamothioate_decomposition(fg, mol)
            if decomp is not None:
                yield decomp
        # --- Functional Class: poly-ester / mixed ester (P-65.6.3.3.2) ---
        # When two or more ester FGs share a single parent acid skeleton and
        # that parent is FULLY esterified, name the whole thing as the
        # functional-class ester of the parent poly-acid:
        #   ethyl methyl benzene-1,3-dicarboxylate, dimethyl butanedioate, ...
        # The general polyester path subsumes the older symmetric-diester path
        # (it also handles aromatic / heterocyclic / branched backbones and
        # mixed alkyl groups).  Fall back to the symmetric path only if the
        # general one declines, to preserve any edge cases it covered.
        ester_fgs = [fg for fg in self.fgs if fg.type == "ester"]
        if len(ester_fgs) >= 2:
            decomp = _build_polyester_decomposition(ester_fgs, mol)
            if decomp is None:
                decomp = _build_symmetric_diester_decomposition(ester_fgs, mol)
            if decomp is not None:
                yield decomp


# ---------------------------------------------------------------------------
# InterpretationQuery
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InterpretationQuery:
    """Strategy's preference hints. Controls generation order, not validity."""
    preferred_decomp_types: tuple[str, ...] | None  # "functional_class", "multiplicative", ...
    preferred_parent_type: str | None               # "chain", "ring", "heteroatom_center"
    suppress_functional_class: bool                 # force substitutive reading
    max_results: int                                # stop after N valid interpretations

    def with_override(self, **kwargs) -> InterpretationQuery:
        """Return a copy with fields overridden."""
        return dataclasses.replace(self, **kwargs)


# ---------------------------------------------------------------------------
# PrefixAssignment -- Union Type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TerminalPrefix:
    """A substituent with one attachment point to the parent."""
    fg: DetectedFG | None           # the FG being expressed as prefix (None for simple substituent)
    substituent_atoms: frozenset[int]
    attachment_bond: tuple[int, int]  # (parent_atom_idx, substituent_atom_idx)
    attachment_bond_order: int        # 1=yl, 2=ylidene, 3=ylidyne
    locant: Locant | None             # locant on the parent where this attaches
    output_form: OutputForm           # SUBSTITUENT for prefixes
    role: str                         # "substituent", "demoted_fg"
    # fragment_mol is NOT stored on the plan -- it's carved during execution.


@dataclass(frozen=True)
class BridgingPrefix:
    """A substituent with two+ attachment points to the parent (e.g., -O-, -CH2-, -NH-)."""
    fg: DetectedFG | None
    substituent_atoms: frozenset[int]
    attachment_bonds: tuple[tuple[int, int], ...]  # ((parent1, sub1), (parent2, sub2))
    attachment_bond_orders: tuple[int, ...]         # bond orders at each attachment
    locants: tuple[Locant, ...]                     # locant on parent at each attachment
    output_form: OutputForm                         # SUBSTITUENT
    role: str


PrefixAssignment = TerminalPrefix | BridgingPrefix


# ---------------------------------------------------------------------------
# FreeValenceInfo
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FreeValenceInfo:
    """Describes how a substituent fragment attaches to its parent.
    Determined by the caller (parent plan's PrefixAssignment).
    Propagated through to assembly for suffix rendering.
    Included in cache key (bond_orders only -- method is a naming decision).
    """
    bond_orders: tuple[int, ...]        # one entry per attachment point
                                         # (1,) -> monovalent single bond
                                         # (2,) -> monovalent double bond
                                         # (3,) -> monovalent triple bond
                                         # (1, 1) -> divalent, different atoms
    method: SubstituentMethod           # Method (1) or (2) -- naming decision
    attachment_atoms_in_fragment: tuple[int, ...] | None
        # Atom indices in the carved fragment where the free valences are.
        # Used by compute_numberings to constrain locant assignment.
        # None before carving (plan phase).
    elide_locant_one: bool = True
        # When True (default), locant "1" is omitted from the -yl suffix for
        # ALKANYL method (e.g. "pyrimidinyl" instead of "pyrimidin-1-yl").
        # Set to False for ring substituents where the attachment is at a
        # heteroatom: OPSIN needs the explicit locant to pick the right atom
        # (e.g. "pyrimidin-1-yl" vs "pyrimidinyl" which OPSIN may misparse).

    @property
    def is_monovalent(self) -> bool:
        return len(self.bond_orders) == 1

    @property
    def suffix(self) -> str:
        """Determine the IUPAC free-valence suffix per Table 3.4."""
        n = len(self.bond_orders)
        sig = tuple(sorted(self.bond_orders, reverse=True))
        return FREE_VALENCE_SUFFIXES.get((n, sig), f"{n}yl")


# ---------------------------------------------------------------------------
# DecisionContext (forward-referenced NamingPlan resolved below)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DecisionContext:
    """Why is this fragment being named? Informational for tracing and debugging.

    CONTRACT: Strategy's score_plan() MUST NOT read or depend on DecisionContext.
    Cache correctness depends on (smiles, output_form, fv_bond_orders) being
    sufficient. DecisionContext is NOT part of the cache key.
    """
    role: str                           # "acid_part", "alcohol_part", "substituent",
                                        # "multiplicative_subunit", "ring_assembly_unit",
                                        # "salt_ion", "demoted_fg"
    parent_plan: Any                    # NamingPlan | None -- typed as Any to avoid circularity
    depth: int


# ---------------------------------------------------------------------------
# Supporting plan types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReplacementPrefix:
    """An 'a' prefix replacing a carbon atom with a heteroatom."""
    element: str          # "O", "N", "S", "P", "Si", etc.
    locant: Locant
    a_prefix: str         # "oxa", "aza", "thia", "phospha", "sila", etc.


@dataclass(frozen=True)
class AdditiveGroup:
    """An atom/group added to the parent."""
    type: str             # "oxide", "sulfide", "selenide", "imide"
    locant: Locant
    multiplier: str | None  # "di" for dioxide, etc.


# ---------------------------------------------------------------------------
# NamingPlan -- Union Type (forward declarations via Any for self-refs)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlanBase:
    """Fields shared by all plan types."""
    interpretation: Interpretation | None   # None for retained names
    stereo_descriptors: tuple[StereoDescriptor, ...] | None


@dataclass(frozen=True)
class RetainedPlan(PlanBase):
    """Use a retained (trivial/semi-systematic) name for the whole molecule."""
    match: RetainedMatch


@dataclass(frozen=True)
class SubstitutivePlan(PlanBase):
    """Standard substitutive nomenclature (P-31).

    Does not carry a Decomposition -- the plan itself encodes the
    parent/substituent split through named_parent + prefix_assignments.
    """
    named_parent: NamedParent
    numbering: Numbering
    pcg_type: str | None                    # FG type string, not instance
    pcg_instances: tuple[DetectedFG, ...]   # all instances of PCG type
    suffix_groups: tuple[SuffixGroup, ...]
    unsaturation: tuple[UnsaturationInfix, ...]
    prefix_assignments: tuple[PrefixAssignment, ...]
    indicated_hydrogen: tuple[Locant, ...] | None
    # Ring-embedded cation atoms (currently: ring [N+]) that lie on the
    # parent backbone.  Populated by SubstitutivePath.generate_plans by
    # intersecting candidate.atom_indices with mol-detected ring-N+ atoms.
    # Drives the cation-as-PCG band in IUPACCanonical.score_plan so that
    # cation suffixes (P-73) outrank ester FC (P-66) when present.
    parent_ring_cation_atoms: frozenset[int] = frozenset()


@dataclass(frozen=True)
class FunctionalClassPlan(PlanBase):
    """Functional class nomenclature (esters, anhydrides, etc.)."""
    decomposition: Decomposition            # carries the FC split info
    fragment_roles: tuple[tuple[str, int], ...]  # (role_name, fragment_idx) pairs
    fragment_output_forms: tuple[tuple[str, OutputForm], ...]


@dataclass(frozen=True)
class MultiplicativePlan(PlanBase):
    """Multiplicative nomenclature (P-51.3) -- identical subunits + linking group."""
    decomposition: Decomposition            # carries the symmetry group
    linking_group: str | None
    multiplier: str                         # "bis", "tris", etc.
    linking_atom_indices: tuple[int, ...]   # atom indices where links attach


@dataclass(frozen=True)
class RingAssemblyPlan(PlanBase):
    """Ring assembly nomenclature (P-28.4) -- identical rings directly bonded."""
    decomposition: Decomposition            # carries the symmetry group
    multiplier: str                         # "bi", "ter", etc.
    locants: tuple[str, ...]


@dataclass(frozen=True)
class ReplacementPlan(PlanBase):
    """Replacement ('a') nomenclature (P-15.4, P-22.1).

    Heteroatoms in chains/rings are expressed as 'a' prefixes:
    oxa (O), aza (N), thia (S), phospha (P), sila (Si), etc.
    The parent is the all-carbon skeleton; heteroatoms are replacements.
    """
    carbon_parent: NamedParent
    replacements: tuple[ReplacementPrefix, ...]
    numbering: Numbering
    pcg: DetectedFG | None
    suffix_groups: tuple[SuffixGroup, ...]
    unsaturation: tuple[UnsaturationInfix, ...]
    prefix_assignments: tuple[PrefixAssignment, ...]
    indicated_hydrogen: tuple[Locant, ...] | None


@dataclass(frozen=True)
class AdditivePlan(PlanBase):
    """Additive nomenclature (P-68.3).

    Names compounds with added atoms/groups:
    - N-oxide: pyridine 1-oxide, trimethylamine oxide
    - P-oxide: triphenylphosphane oxide
    """
    parent_plan: Any                        # NamingPlan -- typed as Any to avoid circularity
    additions: tuple[AdditiveGroup, ...]


# Union type for type-safe dispatch
NamingPlan = (RetainedPlan | SubstitutivePlan | FunctionalClassPlan
              | MultiplicativePlan | RingAssemblyPlan
              | ReplacementPlan | AdditivePlan)


# ---------------------------------------------------------------------------
# Choice
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Choice:
    """A naming decision recorded on a NameTree for tracing and alignment."""
    type: str               # "retained", "substitutive", "functional_class",
                            # "multiplicative", "ring_assembly", "replacement",
                            # "additive", "salt"
    detail: str             # human-readable description of the specific choice


# ---------------------------------------------------------------------------
# PrefixEntry and MergedPrefix
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PrefixEntry:
    """A named substituent prefix, ready for assembly.
    Created during execution (after recursive naming produces the tree)."""
    tree: Any               # NameTree -- typed as Any to avoid circularity
    locants: tuple[Locant, ...]       # always tuple, even for single locant
    # multiplier is NOT stored here -- it's computed during assembly's
    # merge_identical_prefixes step.


@dataclass(frozen=True)
class MergedPrefix:
    """Result of grouping identical PrefixEntry trees in assembly."""
    name: str                          # assembled prefix string
    locants: tuple[Locant, ...]        # combined from all entries
    multiplier: str | None             # "di", "tri", etc. None if count=1
    sort_name: str                     # derived via derive_sort_name()
    needs_brackets: bool               # True if compound prefix


# ---------------------------------------------------------------------------
# NameTree -- Typed Union IR
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TreeBase:
    """Fields shared by all tree types. Frozen -- immutable after creation.
    Use with_warnings() to produce a new tree with additional warnings."""
    output_form: OutputForm
    free_valence: FreeValenceInfo | None
    choices_made: tuple[Choice, ...]
    decision_ctx: DecisionContext | None
    validity_warnings: tuple[str, ...] | None

    def with_warnings(self, *new_warnings: str) -> TreeBase:
        """Return a copy of this tree with additional validity warnings appended."""
        existing = self.validity_warnings or ()
        return dataclasses.replace(self, validity_warnings=existing + new_warnings)


@dataclass(frozen=True)
class LeafTree(TreeBase):
    """A terminal name string (retained name, simple substituent, etc.)."""
    text: str

    def with_warnings(self, *new_warnings: str) -> LeafTree:
        existing = self.validity_warnings or ()
        return dataclasses.replace(self, validity_warnings=existing + new_warnings)


@dataclass(frozen=True)
class SaltTree(TreeBase):
    """Disconnected ionic species. Ions already in correct order."""
    ion_trees: tuple[NameTree, ...]

    def with_warnings(self, *new_warnings: str) -> SaltTree:
        existing = self.validity_warnings or ()
        return dataclasses.replace(self, validity_warnings=existing + new_warnings)


@dataclass(frozen=True)
class FunctionalClassTree(TreeBase):
    """Functional class name (ester, anhydride, etc.)."""
    subtype: str
    pieces: tuple[tuple[str, NameTree], ...]  # (role, subtree) pairs, immutable
    # For subtype="polyester": one entry per "alcohol_N" role, recording
    # (role, locant_str_or_None, symmetry_rank).  locant_str is the parent-acid
    # locant of that ester position (e.g. "1", "3"); symmetry_rank groups
    # topologically interchangeable positions.  Assembly uses these to decide
    # whether to format "ethyl methyl ..." vs "1-ethyl 3-methyl ...".
    polyester_alkyl_locants: tuple[tuple[str, str | None, int], ...] | None = None

    def with_warnings(self, *new_warnings: str) -> FunctionalClassTree:
        existing = self.validity_warnings or ()
        return dataclasses.replace(self, validity_warnings=existing + new_warnings)


@dataclass(frozen=True)
class SubstitutiveTree(TreeBase):
    """Standard substitutive name."""
    named_parent: NamedParent
    numbering: Numbering
    suffix_groups: tuple[SuffixGroup, ...]
    unsaturation: tuple[UnsaturationInfix, ...]
    prefixes: tuple[PrefixEntry, ...]   # unordered -- assembly sorts alphabetically
    stereo_descriptors: tuple[StereoDescriptor, ...] | None
    indicated_hydrogen: tuple[Locant, ...] | None
    # Locants of ring-embedded [N+] atoms in the parent backbone, populated
    # only when output_form == CATION.  Drives the ring-cation -ium suffix
    # (P-73.1) in assembly: e.g. piperidine + (1,) → "piperidin-1-ium",
    # azabicyclo[3.2.1]octane + (8,) → "azabicyclo[3.2.1]octan-8-ium".
    # None means "no ring-cation -ium suffix to append" (acyclic N+, neutral
    # parent, or the parent name already encodes the cation, e.g. retained
    # "pyridinium").
    ring_cation_locants: tuple[Locant, ...] | None = None
    # Locants of ring-embedded aromatic [n-] atoms in the parent backbone.
    # Drives the ring-anion -ide suffix (P-72.2 / P-73) in assembly: e.g.
    # "1,3-dimethyl-2,6-dioxo-2,3,6,7-tetrahydro-1H-purin-7-ide" for the
    # N7-deprotonated theophylline anion.  None means "no ring-anion -ide
    # suffix to append" (neutral parent, or the charge is non-ring).
    ring_anion_locants: tuple[Locant, ...] | None = None
    # Isotope labels attached to parent backbone atoms (Stage 6 R1-D).
    # Assembled as the IUPAC bracketed element prefix "(N-¹³C)" / "(²H₄)"
    # emitted between stereo and indicated-hydrogen.  ``None`` and the
    # empty tuple are equivalent; ``None`` is the default so engine code
    # paths that do not populate this field remain unchanged.
    isotope_labels: tuple[IsotopeLabel, ...] | None = None
    # P-14.3.4.4 (omission of locants that are unique by symmetry): True when
    # there is exactly ONE substituent on the parent (a single prefix, or the
    # sole PCG suffix) and every parent position at which that substituent could
    # attach is in one graph-symmetry class.  When True, assembly omits the lone
    # substituent's locant — generalising the all-carbon-monocyclic special
    # case to fused (``chlorocoronene``) and heterocyclic
    # (``pyrazinecarboxylic acid``) parents.  Computed in the engine from the
    # RDKit mol (perception.symmetry.single_substituent_locant_forced_by_symmetry);
    # default False so untouched code paths keep their locants.
    single_substituent_positions_all_equivalent: bool = False

    def with_warnings(self, *new_warnings: str) -> SubstitutiveTree:
        existing = self.validity_warnings or ()
        return dataclasses.replace(self, validity_warnings=existing + new_warnings)


@dataclass(frozen=True)
class MultiplicativeTree(TreeBase):
    """Multiplicative name (identical subunits + linking group)."""
    subunit: NameTree
    linking_group: str | None
    multiplier: str
    locants: tuple[str, ...]

    def with_warnings(self, *new_warnings: str) -> MultiplicativeTree:
        existing = self.validity_warnings or ()
        return dataclasses.replace(self, validity_warnings=existing + new_warnings)


@dataclass(frozen=True)
class RingAssemblyTree(TreeBase):
    """Ring assembly name (identical rings directly bonded)."""
    ring_unit: NameTree
    multiplier: str
    locants: tuple[str, ...]

    def with_warnings(self, *new_warnings: str) -> RingAssemblyTree:
        existing = self.validity_warnings or ()
        return dataclasses.replace(self, validity_warnings=existing + new_warnings)


@dataclass(frozen=True)
class ReplacementTree(TreeBase):
    """Replacement ('a') nomenclature name."""
    carbon_parent: NamedParent
    replacements: tuple[ReplacementPrefix, ...]
    numbering: Numbering
    suffix_groups: tuple[SuffixGroup, ...]
    unsaturation: tuple[UnsaturationInfix, ...]
    prefixes: tuple[PrefixEntry, ...]
    stereo_descriptors: tuple[StereoDescriptor, ...] | None
    indicated_hydrogen: tuple[Locant, ...] | None

    def with_warnings(self, *new_warnings: str) -> ReplacementTree:
        existing = self.validity_warnings or ()
        return dataclasses.replace(self, validity_warnings=existing + new_warnings)


@dataclass(frozen=True)
class AdditiveTree(TreeBase):
    """Additive nomenclature name (N-oxide, P-oxide, etc.)."""
    parent_tree: NameTree
    additions: tuple[AdditiveGroup, ...]

    def with_warnings(self, *new_warnings: str) -> AdditiveTree:
        existing = self.validity_warnings or ()
        return dataclasses.replace(self, validity_warnings=existing + new_warnings)


@dataclass(frozen=True)
class ErrorTree(TreeBase):
    """Naming failed for this fragment."""
    message: str

    def with_warnings(self, *new_warnings: str) -> ErrorTree:
        existing = self.validity_warnings or ()
        return dataclasses.replace(self, validity_warnings=existing + new_warnings)


NameTree = (LeafTree | SaltTree | FunctionalClassTree | SubstitutiveTree
            | MultiplicativeTree | RingAssemblyTree
            | ReplacementTree | AdditiveTree | ErrorTree)


# ---------------------------------------------------------------------------
# NamingSession -- the ONE mutable dataclass
# ---------------------------------------------------------------------------

@dataclass
class NamingSession:
    """Created once per top-level name() call. Shared across recursion.
    NOT frozen -- this is the one mutable data structure (session-scoped)."""
    cache: dict[tuple, Any] = field(default_factory=dict)  # NameTree values
    max_depth: int = 24
    _plan_seq: int = 0

    def _make_key(self, smiles: str, output_form: OutputForm,
                  fv_bond_orders: tuple[int, ...],
                  attachment_indices: tuple[int, ...] | None) -> tuple:
        """Build the cache key.

        attachment_indices: indices of attachment atoms within the FRAGMENT
        (not the parent molecule). These positions determine naming --
        propan-1-yl (attachment at atom 0) vs propan-2-yl (attachment at atom 1).
        For non-substituent forms (STANDALONE, etc.), this is None.

        attachment_indices are in CANONICAL atom ordering of the fragment
        (guaranteed by carve_substituent's canonical index normalization).
        """
        return (smiles, output_form, fv_bond_orders,
                attachment_indices or ())

    def cache_lookup(self, smiles: str, output_form: OutputForm,
                     fv_bond_orders: tuple[int, ...],
                     attachment_indices: tuple[int, ...] | None = None
                     ) -> Any:  # NameTree | None
        return self.cache.get(
            self._make_key(smiles, output_form, fv_bond_orders,
                           attachment_indices))

    def cache_store(self, smiles: str, output_form: OutputForm,
                    fv_bond_orders: tuple[int, ...], tree: Any,  # NameTree
                    attachment_indices: tuple[int, ...] | None = None):
        self.cache[self._make_key(smiles, output_form, fv_bond_orders,
                                  attachment_indices)] = tree

    def next_seq(self) -> int:
        self._plan_seq += 1
        return self._plan_seq


# ---------------------------------------------------------------------------
# PlanComplexity
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlanComplexity:
    """Estimated plan space size, for adaptive cap."""
    n_suffix_eligible_fgs: int
    n_candidate_parents: int
    n_ring_naming_options: int  # systematic + retained per ring parent

    @property
    def estimated_plans(self) -> int:
        """Coarse upper bound on plan count."""
        return (max(1, self.n_suffix_eligible_fgs + 1)  # +1 for None PCG
                * max(1, self.n_candidate_parents)
                * max(1, self.n_ring_naming_options)
                * 2)  # numbering directions


# ---------------------------------------------------------------------------
# Type alias re-exports (for documentation and import convenience)
# ---------------------------------------------------------------------------

# These are union types defined inline above as Python type aliases.
# Re-stated here for clarity:
#   PrefixAssignment = TerminalPrefix | BridgingPrefix
#   NamingPlan = RetainedPlan | SubstitutivePlan | FunctionalClassPlan | ...
#   NameTree = LeafTree | SaltTree | FunctionalClassTree | ...

__all__ = [
    # Enums
    "OutputForm",
    "SubstituentMethod",
    # Locant & Numbering
    "Locant",
    "Numbering",
    # Atom-level types
    "AtomInfo",
    "StereoCenter",
    "StereoDescriptor",
    # Structural types
    "Fragment",
    "FusionInfo",
    "HeteroPosition",
    "RingSystem",
    # FG types
    "DetectedFG",
    "FGFraming",
    "AmbiguityPoint",
    # Symmetry
    "SymmetryGroup",
    # Suffix/unsaturation
    "SuffixGroup",
    "UnsaturationInfix",
    # Parent types
    "CandidateParent",
    "NamedParent",
    # Retained match
    "RingDescriptor",
    "RetainedMatch",
    # Decomposition / Interpretation
    "Decomposition",
    "Interpretation",
    "InterpretationQuery",
    # Prefix assignment union
    "TerminalPrefix",
    "BridgingPrefix",
    "PrefixAssignment",
    # Supporting plan types
    "ReplacementPrefix",
    "AdditiveGroup",
    # FreeValence
    "FreeValenceInfo",
    "FREE_VALENCE_SUFFIXES",
    # Decision context
    "DecisionContext",
    # Plan types
    "PlanBase",
    "RetainedPlan",
    "SubstitutivePlan",
    "FunctionalClassPlan",
    "MultiplicativePlan",
    "RingAssemblyPlan",
    "ReplacementPlan",
    "AdditivePlan",
    "NamingPlan",
    # Choice
    "Choice",
    # Prefix entry/merged
    "PrefixEntry",
    "MergedPrefix",
    # Tree types
    "TreeBase",
    "LeafTree",
    "SaltTree",
    "FunctionalClassTree",
    "SubstitutiveTree",
    "MultiplicativeTree",
    "RingAssemblyTree",
    "ReplacementTree",
    "AdditiveTree",
    "ErrorTree",
    "NameTree",
    # Session & complexity
    "NamingSession",
    "PlanComplexity",
]
