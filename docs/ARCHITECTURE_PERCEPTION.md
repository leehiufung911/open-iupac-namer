# Perception Layer

**v13 — merged from v12 + v13 delta**

Detailed specification of the Perception layer. For high-level overview, see `ARCHITECTURE_OVERVIEW.md`. For type definitions, see `ARCHITECTURE_DATA_STRUCTURES.md`.

---

## Role

**Input:** RDKit mol object + `InterpretationQuery` from strategy
**Output:** Lazy iterator of `Interpretation` objects, best-first relative to query

Perception is the largest layer (~3000-3500 lines). It generates *structural readings* of the molecule. It never makes naming decisions. Strategy steers generation order via `InterpretationQuery`.

---

## Architecture: Facade Over 7 Subsystems

Perception is a facade over seven independently testable subsystems, each in its own module, with explicit dependencies forming a DAG:

```
AtomAnalysis ──► StereoAnalysis
     │
     ├──► RingAnalysis ──► FGDetection (includes deconfliction)
     │         │
     │         ├──► SymmetryAnalysis
     │         │
     │         └──► ChainFinding
     │
     └──► FragmentAnalysis (independent — uses only RDKit mol)
```

```python
class Perception:
    """Facade. Each subsystem is independently constructable and testable.
    Subsystems are lazily initialized on first access."""

    def __init__(self, mol):
        self._mol = mol
        self._atoms = None
        self._stereo = None
        self._fragments = None
        self._rings = None
        self._fgs = None
        self._symmetry = None
        self._chains = None

    @property
    def atoms(self) -> AtomAnalysis:
        if self._atoms is None:
            self._atoms = AtomAnalysis(self._mol)
        return self._atoms

    @property
    def stereo(self) -> StereoAnalysis:
        if self._stereo is None:
            self._stereo = StereoAnalysis(self._mol, self.atoms)
        return self._stereo

    @property
    def fragments(self) -> FragmentAnalysis:
        if self._fragments is None:
            self._fragments = FragmentAnalysis(self._mol)
        return self._fragments

    @property
    def rings(self) -> RingAnalysis:
        if self._rings is None:
            self._rings = RingAnalysis(self._mol, self.atoms)
        return self._rings

    @property
    def fgs(self) -> FGDetection:
        if self._fgs is None:
            self._fgs = FGDetection(self._mol, self.atoms, self.rings)
        return self._fgs

    @property
    def symmetry(self) -> SymmetryAnalysis:
        if self._symmetry is None:
            self._symmetry = SymmetryAnalysis(self._mol, self.atoms, self.rings)
        return self._symmetry

    @property
    def chains(self) -> ChainFinding:
        if self._chains is None:
            self._chains = ChainFinding(self._mol, self.atoms, self.rings)
        return self._chains
```

Subsystems are lazily initialized on first access. The dependency DAG is preserved — accessing `fgs` triggers `atoms` and `rings` construction if not yet built. Retained-name checks do not trigger ring/FG/chain subsystems, avoiding wasted work for simple molecules.

Each subsystem:
- Has its own file: `perception/atoms.py`, `perception/rings.py`, etc.
- Has its own unit tests
- Takes explicit dependencies in `__init__`, not the whole Perception object
- Is buildable and testable before the engine exists

---

## Subsystem 1 — Atom-level Analysis (~200 lines)

Computed once, cached, element-agnostic:

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

## Subsystem 2 — Stereocenter Detection (~400 lines)

Computed once, cached:

```python
@dataclass(frozen=True)
class StereoCenter:
    atom_idx: int
    type: str               # "tetrahedral", "double_bond", "axial", "planar"
    descriptor: str | None  # "R", "S", "E", "Z" — computed via CIP
    cip_priorities: tuple | None
```

---

## Subsystem 3 — Fragment Detection (~100 lines)

For salts, disconnected species:

```python
@dataclass(frozen=True)
class Fragment:
    atom_indices: frozenset[int]
    mol: object             # RDKit mol for this fragment
    charge: int             # net charge
```

Fragment detection is consumed directly by the engine for salt handling. Salts are handled before interpretation generation.

---

## Subsystem 4 — Ring System Analysis (~600 lines)

Structural descriptors only — no naming:

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

When `classification_ambiguous=True`, the ring naming module generates NamedParent candidates for BOTH classifications, and strategy scores the full plans.

**Important:** Perception produces structural descriptors (bridge sizes, fusion edges, heteroatom positions). It does NOT name rings. Ring naming happens in the ring naming module during plan generation.

```python
def detect_ring_unsaturation(self, ring_system: RingSystem,
                              numbering: Numbering
                              ) -> tuple[UnsaturationInfix, ...]:
    """Detect non-aromatic unsaturation in a ring parent.
    Aromatic bonds are NOT reported (the ring name encodes aromaticity).
    Returns: Tuple of UnsaturationInfix. Empty for fully saturated
             or fully aromatic rings.
    """
```

---

## Subsystem 5 — Functional Group Detection (~600 lines)

Graph-based, extensible, with atom-ownership deconfliction:

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

### Atom-ownership deconfliction algorithm (v13 revised: F1)

When multiple FG patterns match overlapping atoms, deconfliction resolves which FG "owns" which atoms. This runs once per interpretation.

**Input:** list of raw FG matches (may overlap)
**Output:** list of DetectedFG with non-overlapping atom claims

```
PASS 1 — Subsumption removal:
  If FG_A's matched atoms are a strict superset of FG_B's in the overlap
  region, and FG_A is a more specific pattern, remove FG_B.
  Implementation: SUBSUMPTION_TABLE[fg_type_a][fg_type_b] = True.
  Known True entries:
    amide > ketone, amide > amine, carbamate > ester, carbamate > amine,
    carbamic acid > carboxylic acid, urea > amine, thioamide > thioketone,
    sulfoxide > thioether, sulfone > thioether, sulfonamide > sulfone + amine,
    phosphonate > phosphine

PASS 2 — Greedy assignment by seniority (v13 REVISED):
  For remaining overlaps (partial, neither subsumes the other):
  Sort FGs by seniority (P-65 table, descending).
  For each FG in order:
    Claim all unclaimed atoms in its match.
    If ANY of its anchor atoms are already claimed:
      Check: is there a SUBSUMPTION_TABLE entry for this pair?
        YES, and says "compatible" → allow sharing
        YES, and says "conflict" → mark as AmbiguityPoint
        NO ENTRY → mark as AmbiguityPoint (DEFAULT TO AMBIGUITY)

PASS 3 — Conflict resolution:
  For each conflict from Pass 2:
    Create an AmbiguityPoint with both framings.
```

**v13 change (F1):** Unknown overlaps default to ambiguity instead of greedy assignment. Safe — worst case, perception generates more interpretations, but never silently picks the wrong FG. Log a warning when this happens.

```python
if not SUBSUMPTION_TABLE.get((fg_a.type, fg_b.type)):
    logger.warning(
        f"Unknown FG overlap: {fg_a.type} vs {fg_b.type} "
        f"at atoms {fg_a.atoms & fg_b.atoms}. "
        f"Treating as ambiguity. Consider adding a subsumption entry."
    )
```

---

## Subsystem 6 — Symmetry Detection (~300 lines)

For multiplicative + ring assembly:

```python
@dataclass(frozen=True)
class SymmetryGroup:
    subunit_atoms: tuple[frozenset[int], ...]
    subunit_mol: object
    linking_atoms: frozenset[int]
    linking_type: str                # "direct_bond", "linking_group"
    linking_group_mol: object | None
    multiplicity: int
```

Ring assembly vs multiplicative disambiguation: `linking_type == "direct_bond"` → ring assembly candidate. `linking_type == "linking_group"` → multiplicative candidate. Both decompositions are yielded; strategy scores the plans.

---

## Subsystem 7 — Chain Finding (~400 lines)

Part of candidate parent generation, separated for testability.

```python
def detect_chain_unsaturation(self, chain_atoms: frozenset[int],
                               numbering: Numbering
                               ) -> tuple[UnsaturationInfix, ...]:
    """Detect double and triple bonds along a chain parent.
    Aromatic bonds NOT reported as unsaturation.
    Returns: Tuple of UnsaturationInfix, sorted by locant.
    """
```

---

## Ambiguity Points

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

Most molecules have zero ambiguity points. Total interpretation count = product of options at ambiguity points only.

---

## Candidate Parent Generation (lazy, PCG-parameterized)

**v13 change (A2):** `candidate_parents` now takes `pcg_anchors` (tuple of ALL instances of the PCG type), not a single `pcg_anchor`. The anchor filter uses an adjacency check:

```python
def candidate_parents(self, interpretation: Interpretation,
                      pcg_anchors: tuple[int, ...] = ()
                      ) -> Iterator[CandidateParent]:
    """Yield parent structure candidates.

    pcg_anchors: anchor atom indices of ALL instances of the PCG type.
    Candidates must be RELATED TO at least one anchor: the anchor is
    either ON the parent (terminal FG) or BONDED TO a parent atom
    (non-terminal FG, e.g., -COOH on a ring where the COOH carbon
    is exocyclic).

    Three categories, yielded interleaved:
    1. Chain parents (containing/related to pcg_anchors)
    2. Ring parents (containing/related to pcg_anchors)
    3. Heteroatom parents
    """
```

---

## Decomposition Enumeration (lazy, per-interpretation)

```python
@dataclass(frozen=True)
class Decomposition:
    type: str                               # "functional_class", "multiplicative",
                                            # "ring_assembly"
    subtype: str | None                     # for FC: "ester", "anhydride", etc.
    pieces: tuple[Fragment, ...] | None     # for FC
    symmetry_group: SymmetryGroup | None    # for multiplicative/ring_assembly
    locants: tuple[str, ...] | None         # for ring_assembly
    root_atoms: frozenset[int]
    intramolecular: bool = False            # True for lactones, cyclic anhydrides
```

FC decomposition detection uses SMARTS patterns with a connectivity check for intramolecular cases.

---

## Interpretation Generation (lazy, query-steered)

**v13 change (B2):** Interpretation iterator contract specified:

```python
class Perception:
    def interpretations(self, query: InterpretationQuery) -> Iterator[Interpretation]:
        """Yield interpretations in best-first order relative to query.

        CONTRACT: This is a generator FUNCTION (not a stored iterator).
        Each call creates an independent generator. Calling interpretations()
        twice produces two independent sequences, both starting from the
        beginning. The generator is lazy — work happens on next().

        This means the function is restartable but NOT rewindable within
        a single generator instance.
        """
```

### InterpretationQuery

```python
@dataclass(frozen=True)
class InterpretationQuery:
    preferred_decomp_types: tuple[str, ...] | None
    preferred_parent_type: str | None
    suppress_functional_class: bool
    max_results: int

    def with_override(self, **kwargs) -> 'InterpretationQuery':
        """Return a copy with fields overridden."""
```

---

## Retained Name Matching

```python
@dataclass(frozen=True)
class RetainedMatch:
    name: str
    smiles: str
    scope: str                      # "exact_molecule" | "parent_hydride"
    valid_output_forms: frozenset[OutputForm]
    substituent_form: str | None    # e.g., "phenyl" for benzene
    ring_descriptor: RingDescriptor | None

def retained_matches(self, mol, output_form: OutputForm) -> Iterator[RetainedMatch]:
    """Only yield matches valid for this molecule AND output form.
    Interpretation-independent — called once before the interpretation loop."""
```

**Two levels of matching:**

1. **Top-level** (in `name()`, before interpretation loop): `retained_matches()` checks the WHOLE MOLECULE against the table. Only `scope="exact_molecule"` entries produce `RetainedPlan` candidates.

2. **Ring naming module** (during plan generation): For each `CandidateParent` that is a ring system, checks `scope="parent_hydride"` entries.

**v13 change (H1):** Retained name matching runs at EVERY recursion level. When engine recurses on a carved benzene fragment with `OutputForm.SUBSTITUENT`:
- Finds `RetainedMatch(name="phenyl", smiles="c1ccccc1", scope="exact_molecule", valid_output_forms={SUBSTITUENT})`
- Produces `RetainedPlan` → `LeafTree(text="phenyl")`
- `good_enough_score` triggers early termination

This is the fast path for common ring substituents (phenyl, naphthyl, furyl, thienyl, pyridyl).

---

## Fragment Extraction (utility module, ~250 lines)

Called by path handler `execute()` methods during execution phase.

### carve_substituent

```python
def carve_substituent(mol, substituent_atoms, attachment_bond
                      ) -> tuple[Mol, int, int]:
    """Extract a substituent fragment as a standalone mol.

    Returns: (fragment_mol, attachment_atom_in_fragment, attachment_bond_order)

    v13 CHANGE (G1): Explicit canonical index normalization.
    After FragmentOnBonds + dummy replacement, the fragment is canonicalized
    via RDKit's RenumberAtoms using the canonical ordering. The attachment
    atom index is mapped through this renumbering. This ensures that for
    any two carving operations that produce the same canonical SMILES,
    the attachment index is identical.
    """
    # ... produce raw_fragment ...
    canonical_order = Chem.CanonicalRankAtoms(raw_fragment)
    reorder_map = {old: new for new, old in enumerate(canonical_order)}
    fragment_mol = Chem.RenumberAtoms(raw_fragment, canonical_order)
    canonical_attachment = reorder_map[raw_attachment_idx]
    return (fragment_mol, canonical_attachment, bond_order)
```

### carve_bridging_substituent

```python
def carve_bridging_substituent(mol, substituent_atoms, attachment_bonds
                                ) -> tuple[Mol, list[int], list[int]]:
    """Extract a bridging substituent (-CH2-, -O-, -NH-).
    Returns: (fragment_mol, attachment_atoms_in_fragment, bond_orders)
    Same canonical normalization as carve_substituent.
    """
```

### carve_fc_fragments

```python
def carve_fc_fragments(mol, decomp: Decomposition) -> dict[str, Mol]:
    """Split a molecule at functional-class boundaries.
    For each role, produce a standalone mol with satisfied valences."""
```

### strip_additive_atoms (v13 addition: C1)

```python
def strip_additive_atoms(mol: Mol,
                         additive_groups: list[AdditiveGroupInfo]
                         ) -> tuple[Mol, dict[int, int]]:
    """Remove additive atoms from a molecule, producing the parent molecule.

    For N-oxide [N+]([O-]):
      Remove the O atom. Adjust N charge from +1 to 0.

    For P-oxide P(=O):
      Remove the O atom and the P=O bond. P valence adjusts.

    Returns:
        (parent_mol, atom_map)
        atom_map: {old_idx_in_parent_mol: old_idx_in_original_mol}

    Implementation: RWMol editing — remove atoms in reverse index order.
    """
```

Since additive detection and stripping happen BEFORE plan search (per v13 B3), the engine never sees the additive atoms. The parent plan operates on a clean molecule.

---

## Perception/Strategy Boundary

- **Perception resolves:** atom-ownership within each interpretation, ring structural descriptors, candidate parents
- **Strategy resolves:** naming priority (which FG becomes PCG), retained vs systematic ring names
- **The bridge:** InterpretationQuery controls generation order (preference hints)

---

## File Organization

```
perception/
    __init__.py     # Perception facade class
    atoms.py        # AtomAnalysis (~200 lines)
    stereo.py       # StereoAnalysis (~400 lines)
    fragments.py    # FragmentAnalysis (~100 lines)
    rings.py        # RingAnalysis (~700 lines, includes unsaturation detection)
    fg_detection.py # FGDetection (~600 lines, includes 3-pass deconfliction)
    symmetry.py     # SymmetryAnalysis (~300 lines)
    chains.py       # ChainFinding (~500 lines, includes unsaturation detection)
    retained.py     # Retained name matching (~100 lines)
    extraction.py   # Fragment carving utilities (~250 lines)
```
