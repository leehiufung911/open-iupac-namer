# Strategy Layer

**v13 — merged from v12 + v13 delta**

Detailed specification of the Strategy layer. For high-level overview, see `ARCHITECTURE_OVERVIEW.md`. For type definitions, see `ARCHITECTURE_DATA_STRUCTURES.md`.

---

## Role

**Input:** Questions from the engine + plan candidates
**Output:** Scores, accept/reject decisions, preference hints

Strategy scores structural commitments. It does NOT score concrete substituent names (determined during recursive execution). IUPAC preference rules (P-44, P-65) are structural criteria, not string-based. Alphabetical prefix ordering is a deterministic formatting rule in assembly.

---

## NamingStrategy Protocol

```python
class NamingStrategy(Protocol):

    def interpretation_query(self, mol) -> InterpretationQuery:
        """Preference hints for interpretation search order."""

    def accept_plan(self, plan: NamingPlan) -> bool:
        """Hard structural validity check. Returns False if this plan violates
        a rule of this naming system.

        Examples:
        - IUPACCanonical rejects intramolecular FC plans
        - CASStrategy rejects certain retained names IUPAC allows

        accept_plan is a FAST check (no floating-point scoring).
        Plans that pass are guaranteed structurally valid under this strategy.
        """

    def accept_additive(self, additive_groups) -> bool:
        """Strategy veto for additive nomenclature (v13 addition: Issue 20).
        Called before plan search when additive groups are detected.
        Returns False to force substitutive naming instead."""

    def score_plan(self, plan: NamingPlan) -> float:
        """Rate the structural commitments. Higher = preferred.
        Only called on plans where accept_plan returned True.

        CONTRACT: This method MUST be pure with respect to (plan, strategy_config).
        It MUST NOT read or depend on DecisionContext. Cache correctness depends
        on (canonical_smiles, output_form, free_valence_bond_orders) being a
        sufficient cache key.
        """

    def max_plans_hint(self, complexity: PlanComplexity | None = None) -> int:
        """Stop generating plans after this many accepted candidates."""

    def good_enough_score(self) -> float:
        """If a plan scores at or above this, stop searching."""

    def retained_name_policy(self) -> RetainedPolicy:
        """ALWAYS_IF_AVAILABLE, NEVER, or PREFER."""

    def cache_key(self) -> str:
        """Identity for memoization. Same key = same naming decisions."""
```

---

## Scoring Architecture (v13 revised: E1)

Strategy scoring has two distinct problems:
1. **Within a plan type** (e.g., two substitutive plans): P-44/P-65 criteria in strict priority order
2. **Across plan types** (e.g., substitutive vs FC): depends on what FGs are present

### Within-type: Magnitude Bands

```python
class IUPACCanonical(NamingStrategy):

    # Magnitude bands ensure higher-priority criteria ALWAYS dominate:
    #
    # Band 4: PCG seniority (P-65)     x 10_000
    # Band 3: Parent selection (P-44)  x 100
    # Band 2: Numbering quality        x 1
    # Band 1: Naming method/style      x 0.01
```

### Cross-type: accept_plan + Comparable Ranges

The key rule: **FC plans are only valid when the FC-type FG is the most senior FG class in the molecule.** If there's a more senior FG (e.g., COOH is more senior than ester), the FC plan for the ester is REJECTED by `accept_plan`, not merely scored lower.

```python
    def accept_plan(self, plan):
        match plan:
            case FunctionalClassPlan():
                # Hard reject: intramolecular FC for canonical naming
                if plan.decomposition.intramolecular:
                    return False
                # Hard reject: FC doesn't handle most senior FG
                most_senior = self._most_senior_fg_type(plan.interpretation.fgs)
                fc_handles = self._fc_covered_fg_types(plan)
                if most_senior and most_senior not in fc_handles:
                    return False
                return True
            case ReplacementPlan():
                return True
            case _:
                return True

    def score_plan(self, plan):
        """All plan types return scores in shared range [0, 500_000].
        Retained plans score above this range."""
        match plan:
            case RetainedPlan():
                return 1_000_000   # always preferred over systematic names

            case SubstitutivePlan():
                score = 0.0
                score += self._pcg_seniority_score(plan.pcg_type) * 10_000
                score += self._parent_selection_score(plan) * 100
                score += self._numbering_score(plan.numbering) * 1
                score += self._naming_method_score(plan.named_parent) * 0.01
                return score

            case FunctionalClassPlan():
                # FC plans that reach scoring have passed accept_plan.
                return 400_000 + self._fc_quality_score(plan) * 100

            case ReplacementPlan():
                score = 0.0
                score += self._pcg_seniority_score(plan.pcg_type) * 10_000
                score += self._parent_selection_score(plan) * 100
                score += self._numbering_score(plan.numbering) * 1
                return score

            case MultiplicativePlan():
                return 450_000 + self._multiplicative_quality(plan) * 100

            case RingAssemblyPlan():
                return 420_000 + self._ring_assembly_quality(plan) * 100
```

### Score Component Details

```python
    def _pcg_seniority_score(self, pcg_type: str | None) -> float:
        """P-65.1 seniority table. Returns 0.0-30.0.
        None (no PCG) returns 0. Higher = more senior."""

    def _parent_selection_score(self, plan) -> float:
        """P-44 criteria, applied in strict priority order. Returns 0.0-99.0.
        Sub-criteria weighted so earlier criteria dominate later ones:
          3a: n_pcg_anchors_on_parent x 9.0  (max ~4 -> 36)
          3b: parent_length x 0.5            (max ~40 -> 20)
          3c: n_multiple_bonds x 3.0         (max ~5 -> 15)
          3d: n_substituents x 2.0           (max ~6 -> 12)
          3e: lowest_pcg_locant_inv x 1.0    (max ~10 -> 10)
        """

    CANONICAL_OPTIMUM = 1_000_000  # retained name for exact molecule

    def good_enough_score(self):
        return self.CANONICAL_OPTIMUM
```

### Key Scoring Properties

1. Within substitutive: magnitude bands guarantee PCG seniority always dominates parent selection, which always dominates numbering.
2. Cross-type: `accept_plan` eliminates structurally invalid plans. Surviving FC plans score above substitutive because FC IS preferred when the FC FG class is the most senior.
3. Retained names always win (when valid).
4. A wrong-PCG substitutive plan can never outscore a right-PCG plan (10K band gap).

---

## Plan Complexity and Adaptive Caps

```python
@dataclass(frozen=True)
class PlanComplexity:
    n_suffix_eligible_fgs: int
    n_candidate_parents: int
    n_ring_naming_options: int

    @property
    def estimated_plans(self) -> int:
        return (max(1, self.n_suffix_eligible_fgs + 1)
                * max(1, self.n_candidate_parents)
                * max(1, self.n_ring_naming_options)
                * 2)  # numbering directions
```

---

## Concrete Strategies

### IUPACCanonical

```python
class IUPACCanonical(NamingStrategy):
    """2013 Blue Book preference rules at every decision point."""

    def interpretation_query(self, mol):
        return InterpretationQuery(max_results=1)

    def max_plans_hint(self, complexity=None):
        if complexity is None:
            return 20
        est = complexity.estimated_plans
        if est <= 20:
            return 20
        return min(80, max(20, est // 2))

    def cache_key(self):
        return "iupac"
```

### AlignedStrategy (future)

```python
class AlignedStrategy(NamingStrategy):
    """Name a molecule so its name structurally resembles reference names.
    Uses OPSIN for name parsing (not required for canonical naming)."""

    def __init__(self, reference_names: list[str]):
        self.reference_names = reference_names
        self.ref_patterns = [parse_naming_pattern(n) for n in reference_names]

    def good_enough_score(self):
        return 0.95

    def max_plans_hint(self, complexity=None):
        return 50  # search more broadly for alignment

    def cache_key(self):
        return f"aligned:{':'.join(sorted(self.reference_names))}"
```

### CASStrategy (future)

```python
class CASStrategy(NamingStrategy):
    """CAS naming conventions. May accept intramolecular FC that IUPAC rejects."""

    def cache_key(self):
        return "cas"
```

---

## Supporting Types

### CandidateParent vs NamedParent

Perception produces structural candidates. Plan generation names them.

```python
@dataclass(frozen=True)
class CandidateParent:
    atom_indices: frozenset[int]
    type: str                  # "chain", "monocyclic", "fused", "bridged", "spiro",
                               # "heteroatom_center"
    length: int
    ring_system: RingSystem | None
    unsaturation: tuple[UnsaturationInfix, ...] | None
    element: str | None        # for heteroatom parents
    lambda_value: int | None   # for hypervalent heteroatoms

@dataclass(frozen=True)
class NamedParent:
    candidate: CandidateParent
    name: str              # "bicyclo[2.2.1]heptane" or "norbornane"
    stem: str              # "bicyclo[2.2.1]heptan" (for Method 2 / suffix)
    alkyl_stem: str | None # "bicyclo[2.2.1]hept" (for Method 1 / -ane replace)
    naming_method: str     # "systematic", "retained", "hantzsch_widman", etc.
    indicated_hydrogen: tuple[Locant, ...] | None
    numbering_options: tuple[Numbering, ...]
```

Two stem variants support two-method substituent naming (P-29.2):
- `stem` — Method (2) and suffix attachment. Ends at consonant before terminal "e"
- `alkyl_stem` — Method (1). Strip "-ane"/"-ene"/"-yne" entirely. `None` when not applicable (fused, polycyclic, heteroatom centers)

### Ring Naming Package

```
ring_naming/
    __init__.py           ~50 lines    public API: name_parent_candidates()
    common.py            ~100 lines    shared types, stem tables
    monocyclic.py        ~300 lines    cycloalkane, Hantzsch-Widman
    fused.py             ~800 lines    fusion naming, component tables
    bridged.py           ~400 lines    von Baeyer, bridge sizes
    spiro.py             ~200 lines    spiro naming
    retained_lookup.py   ~150 lines    retained ring name table
    numbering.py         ~250 lines    ring numbering per type
Total: ~2250 lines
```

### Numbering Generation

```python
def compute_numberings(named_parent, pcg, fgs, free_valence=None
                       ) -> Iterator[Numbering]:
    """Yield valid numberings, pre-filtered by P-14.4.
    Never yields more than ~6 numberings for any parent type."""
```
