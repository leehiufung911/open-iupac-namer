# IUPAC Naming Engine: Data Structures Reference

**v13 -- merged from v12 + v13 delta**

This document defines all typed dataclasses used at layer boundaries. All data structures are frozen (immutable) unless noted otherwise. For context on how these are used, see `ARCHITECTURE_OVERVIEW.md` and the layer-specific architecture documents.

---

## Locant

```python
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
    def numeric(value: int, suffix: str = "") -> 'Locant':
        return Locant(
            label=f"{value}{suffix}",
            is_numeric=True,
            _numeric_value=value,
            suffix=suffix,
        )

    @staticmethod
    def hetero(element: str, sup: str = "") -> 'Locant':
        label = f"{element}{sup}" if sup else element
        return Locant(label=label, is_numeric=False, _numeric_value=None, suffix="")

    def __str__(self) -> str:
        return self.label

    def __lt__(self, other: 'Locant') -> bool:
        # Numeric locants sort before heteroatom locants (P-14.4)
        if self.is_numeric != other.is_numeric:
            return self.is_numeric  # numeric < heteroatom
        if self.is_numeric:
            if self._numeric_value != other._numeric_value:
                return self._numeric_value < other._numeric_value
            return self.suffix < other.suffix
        # Heteroatom locants: alphabetical N < O < P < S
        return self.label < other.label
```

---

## Numbering

```python
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
```

---

## SuffixGroup

**Note: v13 revision.** The `form` field from v12 is renamed to `base_form` to clarify its role in the two-stage suffix resolution system. Stage 1 (`_compute_suffixes`) determines the base form (terminal vs nonterminal). Stage 2 (`render_suffixes` in assembly) applies the OutputForm variant transform. See `ARCHITECTURE_ENGINE.md` (suffix computation) and `ARCHITECTURE_ASSEMBLY.md` (SUFFIX_VARIANT_TABLE).

```python
@dataclass(frozen=True)
class SuffixGroup:
    fg: DetectedFG
    locants: tuple[Locant, ...]
    base_form: str          # "ol", "oic acid", "carboxylic acid", "al", etc.
                            # Determined by terminal vs nonterminal position.
                            # OutputForm variant applied in assembly.
    elides_terminal_e: bool # whether this suffix triggers vowel elision on parent stem
```

Note: The multiplier (e.g., "di", "tri") is NOT stored on SuffixGroup. It is derived from the count of suffix groups sharing the same base_form during assembly's grouping step.

---

## UnsaturationInfix

```python
@dataclass(frozen=True)
class UnsaturationInfix:
    type: str               # "en", "yn"
    locants: tuple[Locant, ...]
    multiplier: str | None  # "di", "tri" for multiple double/triple bonds
```

---

## AtomInfo

```python
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
```

No element restrictions. This is the foundation for future organometallic support.

---

## StereoCenter

```python
@dataclass(frozen=True)
class StereoCenter:
    atom_idx: int
    type: str               # "tetrahedral", "double_bond", "axial", "planar"
    descriptor: str | None  # "R", "S", "E", "Z" -- computed via CIP
    cip_priorities: tuple | None
```

---

## Fragment

```python
@dataclass(frozen=True)
class Fragment:
    atom_indices: frozenset[int]
    mol: object             # RDKit mol for this fragment
    charge: int             # net charge
```

Fragment detection is consumed directly by the engine for salt handling. Salts are handled before interpretation generation.

---

## RingSystem

```python
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
```

When `classification_ambiguous=True`, the ring naming module generates `NamedParent` candidates for BOTH classifications, and strategy scores the full plans. This turns a perception uncertainty into a search problem.

**Important:** Perception produces structural descriptors (bridge sizes, fusion edges, heteroatom positions). It does NOT name rings. Ring naming happens in the ring naming module during plan generation. See `ARCHITECTURE_ENGINE.md`.

---

## DetectedFG

```python
@dataclass(frozen=True)
class DetectedFG:
    type: str               # "carboxylic_acid", "alcohol", "ester", "ketone", ...
    atoms: frozenset[int]   # atoms this FG claims
    anchor: int             # defining atom (e.g., C of COOH)
    properties: dict        # type-specific (e.g., terminal=True, in_ring=True)
    suffix_eligible: bool   # can this FG be expressed as a suffix?
    suffix_forms: dict[str, str]  # "terminal": "-oic acid", "nonterminal": "-carboxylic acid"
    prefix_form: str        # "carboxy-", "oxo-", "hydroxy-", etc.
```

---

## AmbiguityPoint and FGFraming

```python
@dataclass(frozen=True)
class AmbiguityPoint:
    atoms: frozenset[int]
    options: tuple[FGFraming, ...]
    canonical_preference: int

@dataclass(frozen=True)
class FGFraming:
    fgs: tuple[DetectedFG, ...]
    description: str            # "ester", "substituted alcohol", etc.
```

An ambiguity point is a region where multiple valid FG framings exist. Most molecules have zero ambiguity points -- one unambiguous FG framing. Total interpretation count = product of options at ambiguity points only. See `ARCHITECTURE_PERCEPTION.md` for the deconfliction algorithm.

---

## SymmetryGroup

```python
@dataclass(frozen=True)
class SymmetryGroup:
    """A set of identical substructures in the molecule."""
    subunit_atoms: tuple[frozenset[int], ...]   # each set = atoms of one copy
    subunit_mol: object                          # RDKit mol of the canonical subunit
    linking_atoms: frozenset[int]                # atoms connecting the subunits
    linking_type: str                            # "direct_bond", "linking_group"
    linking_group_mol: object | None             # if linking_type == "linking_group"
    multiplicity: int                            # how many identical subunits
```

Disambiguation: `linking_type == "direct_bond"` -> ring assembly candidate. `linking_type == "linking_group"` -> multiplicative candidate. Both decompositions are yielded; strategy scores the plans.

---

## CandidateParent and NamedParent

```python
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
```

**Two stem variants** (v13 note) support the two-method substituent naming system (P-29.2):

- `stem` -- for Method (2) and general suffix attachment. Ends at the consonant before terminal "e": "ethan", "propan", "cyclohexan", "naphthalen".
- `alkyl_stem` -- for Method (1) only. The stem with "-ane"/"-ene"/"-yne" stripped entirely: "eth", "prop", "cyclohex". `None` for parents where Method (1) is not applicable (fused rings, polycyclic, heteroatom centers).

**Method (1) applicability:**
- Saturated acyclic chains: always applicable (alkyl_stem = chain stem without "-ane")
- Monocyclic saturated rings: applicable (cyclohexane -> cyclohex)
- Fused/bridged/spiro rings: NOT applicable (always Method 2)
- Heteroatom parent hydrides: generally NOT applicable (phosphane -> phosphanyl, not phosphyl)

---

## Interpretation

```python
@dataclass(frozen=True)
class Interpretation:
    fgs: tuple[DetectedFG, ...]
    ambiguity_choices: tuple[tuple[int, int], ...]  # (ambiguity_point_idx, option_idx)
    ring_systems: tuple[RingSystem, ...]            # shared across interpretations
    stereocenters: tuple[StereoCenter, ...]         # shared across interpretations
    symmetry_groups: tuple[SymmetryGroup, ...]      # shared across interpretations

    def decomposition_candidates(self) -> Iterator[Decomposition]:
        """Yield FC, multiplicative, ring assembly decompositions.
        Substitutive is always available and handled directly by
        SubstitutivePath -- it does not appear here."""
        ...
```

---

## InterpretationQuery

```python
@dataclass(frozen=True)
class InterpretationQuery:
    """Strategy's preference hints. Controls generation order, not validity."""
    preferred_decomp_types: tuple[str, ...] | None  # "functional_class", "multiplicative", ...
    preferred_parent_type: str | None               # "chain", "ring", "heteroatom_center"
    suppress_functional_class: bool                 # force substitutive reading
    max_results: int                                # stop after N valid interpretations

    def with_override(self, **kwargs) -> 'InterpretationQuery':
        """Return a copy with fields overridden."""
        ...
```

---

## RetainedMatch

```python
@dataclass(frozen=True)
class RetainedMatch:
    name: str
    smiles: str                     # canonical SMILES this name maps to
    scope: str                      # "exact_molecule" | "parent_hydride"
    valid_output_forms: frozenset[OutputForm]
    substituent_form: str | None    # e.g., "phenyl" for benzene, "naphthyl" for naphthalene
    ring_descriptor: RingDescriptor | None  # for parent_hydride ring matches
```

---

## Decomposition

```python
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
```

---

## PrefixAssignment -- Union Type

```python
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
```

---

## PrefixEntry and MergedPrefix

**v13 definition.** PrefixEntry is created during execution (after recursive naming produces the tree). MergedPrefix is the result of assembly's deduplication step.

```python
@dataclass(frozen=True)
class PrefixEntry:
    """A named substituent prefix, ready for assembly.
    Created during execution (after recursive naming produces the tree)."""
    tree: NameTree
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
```

Usage in execution code:

```python
# TerminalPrefix -> single-element tuple
prefixes.append(PrefixEntry(
    tree=sub_tree,
    locants=(pa.locant,) if pa.locant is not None else (),
))

# BridgingPrefix -> multi-element tuple
prefixes.append(PrefixEntry(
    tree=sub_tree,
    locants=pa.locants,
))
```

---

## NamingPlan -- Union Type

All plan types. The union enables type-safe dispatch throughout the engine. See `ARCHITECTURE_ENGINE.md` for how each plan type is generated and executed.

```python
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
    replacements: tuple['ReplacementPrefix', ...]
    numbering: Numbering
    pcg: DetectedFG | None
    suffix_groups: tuple[SuffixGroup, ...]
    unsaturation: tuple[UnsaturationInfix, ...]
    prefix_assignments: tuple['PrefixAssignment', ...]
    indicated_hydrogen: tuple[Locant, ...] | None

@dataclass(frozen=True)
class AdditivePlan(PlanBase):
    """Additive nomenclature (P-68.3).

    Names compounds with added atoms/groups:
    - N-oxide: pyridine 1-oxide, trimethylamine oxide
    - P-oxide: triphenylphosphane oxide
    """
    parent_plan: 'NamingPlan'
    additions: tuple['AdditiveGroup', ...]

# Union type for type-safe dispatch
NamingPlan = (RetainedPlan | SubstitutivePlan | FunctionalClassPlan
              | MultiplicativePlan | RingAssemblyPlan
              | ReplacementPlan | AdditivePlan)
```

**v13 note on SubstitutivePlan:** The v12 version had a single `pcg: DetectedFG | None` field. v13 replaces this with `pcg_type: str | None` and `pcg_instances: tuple[DetectedFG, ...]` to group identical FG types and avoid generating redundant plans for molecules like hexanedioic acid (two -COOH groups). See `ARCHITECTURE_ENGINE.md` section on PCG grouping.

---

## Supporting Plan Types

```python
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
```

---

## NameTree -- Typed Union IR

Name trees are a union of typed variants. All are frozen (immutable). Use `with_warnings()` to produce a new tree with additional warnings (copy-on-write).

```python
@dataclass(frozen=True)
class TreeBase:
    """Fields shared by all tree types. Frozen -- immutable after creation.
    Use with_warnings() to produce a new tree with additional warnings."""
    output_form: OutputForm
    free_valence: FreeValenceInfo | None
    choices_made: tuple[Choice, ...]
    decision_ctx: DecisionContext | None
    validity_warnings: tuple[str, ...] | None

@dataclass(frozen=True)
class LeafTree(TreeBase):
    """A terminal name string (retained name, simple substituent, etc.)."""
    text: str

@dataclass(frozen=True)
class SaltTree(TreeBase):
    """Disconnected ionic species. Ions already in correct order."""
    ion_trees: tuple['NameTree', ...]

@dataclass(frozen=True)
class FunctionalClassTree(TreeBase):
    """Functional class name (ester, anhydride, etc.)."""
    subtype: str
    pieces: tuple[tuple[str, 'NameTree'], ...]  # (role, subtree) pairs, immutable

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

@dataclass(frozen=True)
class MultiplicativeTree(TreeBase):
    """Multiplicative name (identical subunits + linking group)."""
    subunit: 'NameTree'
    linking_group: str | None
    multiplier: str
    locants: tuple[str, ...]

@dataclass(frozen=True)
class RingAssemblyTree(TreeBase):
    """Ring assembly name (identical rings directly bonded)."""
    ring_unit: 'NameTree'
    multiplier: str
    locants: tuple[str, ...]

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

@dataclass(frozen=True)
class AdditiveTree(TreeBase):
    """Additive nomenclature name (N-oxide, P-oxide, etc.)."""
    parent_tree: 'NameTree'
    additions: tuple[AdditiveGroup, ...]

@dataclass(frozen=True)
class ErrorTree(TreeBase):
    """Naming failed for this fragment."""
    message: str

NameTree = (LeafTree | SaltTree | FunctionalClassTree | SubstitutiveTree
            | MultiplicativeTree | RingAssemblyTree
            | ReplacementTree | AdditiveTree | ErrorTree)
```

`choices_made` enables:
- **Explain this name:** narrate each decision
- **Enumerate valid names:** branch at each choice point
- **Alignment:** compare choice lists between NameTrees

---

## OutputForm

```python
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
```

Note: `ACID_NAME` was removed (present in v10). Anhydride acid components use `STANDALONE`; assembly handles adjective extraction via `_acid_to_adjective()`.

---

## FreeValenceInfo and SubstituentMethod

```python
class SubstituentMethod(Enum):
    """Which of the two IUPAC methods for substituent naming (P-29.2)."""
    ALKYL = auto()      # Method (1): replace "-ane" with "-yl". Locant 1 omitted.
    ALKANYL = auto()    # Method (2): add "-yl" to parent name. Locant cited.

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

    @property
    def is_monovalent(self) -> bool:
        return len(self.bond_orders) == 1

    @property
    def suffix(self) -> str:
        """Determine the IUPAC free-valence suffix per Table 3.4."""
        n = len(self.bond_orders)
        sig = tuple(sorted(self.bond_orders, reverse=True))
        return FREE_VALENCE_SUFFIXES.get((n, sig), f"{n}yl")
```

---

## DecisionContext

```python
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
    parent_plan: NamingPlan | None
    depth: int
```

---

## NamingSession

**v13 note:** Cache key uses canonical attachment indices, guaranteed by the explicit canonical index normalization in `carve_substituent`. See `ARCHITECTURE_ENGINE.md` (fragment carving).

```python
@dataclass
class NamingSession:
    """Created once per top-level name() call. Shared across recursion.
    NOT frozen -- this is the one mutable data structure (session-scoped)."""
    cache: dict[tuple, NameTree] = field(default_factory=dict)
    max_depth: int = 10
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
                     ) -> NameTree | None:
        return self.cache.get(
            self._make_key(smiles, output_form, fv_bond_orders,
                          attachment_indices))

    def cache_store(self, smiles: str, output_form: OutputForm,
                    fv_bond_orders: tuple[int, ...], tree: NameTree,
                    attachment_indices: tuple[int, ...] | None = None):
        self.cache[self._make_key(smiles, output_form, fv_bond_orders,
                                  attachment_indices)] = tree

    def next_seq(self) -> int:
        self._plan_seq += 1
        return self._plan_seq
```

---

## PlanComplexity

```python
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
```

---

## Choice

```python
@dataclass(frozen=True)
class Choice:
    """A naming decision recorded on a NameTree for tracing and alignment."""
    type: str               # "retained", "substitutive", "functional_class",
                            # "multiplicative", "ring_assembly", "replacement",
                            # "additive", "salt"
    detail: str             # human-readable description of the specific choice
```

`Choice` objects are accumulated in `TreeBase.choices_made` and enable name explanation, enumeration of valid alternatives, and alignment comparison between trees.
