# Build Plan

**v13 — merged from v12 + v13 delta**

Build sequence, phase targets, and implementation strategy. For high-level overview, see `ARCHITECTURE_OVERVIEW.md`.

---

## What Carries Over from v1

- **Data tables (copy directly):** chain stems, multipliers, retained names, SMARTS patterns, context transforms, element seniority, FG definitions, element parent hydrides. JSON/dicts — portable as-is. Located in `data/`.
- **Algorithmic logic (reimplement into correct layer):** chain finding, ring perception, numbering, elision, CIP assignment, suffix/prefix computation. Old code is the spec. Most functions translate nearly line-for-line but land in different layers.
- **Test suite (use directly):** The ~1181 OPSIN round-trip tests are the ground truth.

## What Does NOT Carry Over

- **Ad-hoc bail-out guards.** Plan scoring makes them unnecessary.
- **Dispatch ordering.** Replaced by plan scoring.
- **Pipeline IR.** Replaced by typed NamingPlan union + NameTree union.
- **Special class handlers.** Absorbed into perception + path handlers.

---

## Build Sequence

### Phase 1: Foundation

```
1.1 Define all data structures
    - Typed plan/tree unions with frozen TreeBase
    - Locant with heteroatom support
    - Extended cache key with attachment_indices
    - PrefixEntry, MergedPrefix (v13 A3)
    - ReplacementPlan, AdditivePlan, ReplacementTree, AdditiveTree
    - PlanComplexity
    - OutputForm, FreeValenceInfo, SubstituentMethod

1.2 Port data tables from existing codebase
    - Chain stems, multipliers, retained names
    - FG detection patterns (SMARTS)
    - ACID_ADJECTIVE_TABLE
    - REPLACEMENT_A_PREFIXES
    - Free-valence suffix table (Table 3.4)
    - FG subsumption table
    - FG seniority table (P-65)
    - SUFFIX_VARIANT_TABLE (v13 D3)
    - SUFFIX_ELISION_TABLE

1.3 Build perception subsystems incrementally (with lazy construction):
    1.3a AtomAnalysis + FragmentAnalysis (no dependencies)
    1.3b RingAnalysis (depends on AtomAnalysis; includes ring unsaturation)
    1.3c ChainFinding (depends on AtomAnalysis + RingAnalysis;
         includes chain unsaturation)
    1.3d FGDetection with 3-pass deconfliction + N-oxide/P-oxide detection
         (depends on AtomAnalysis + RingAnalysis)
         v13 F1: Unknown overlap → AmbiguityPoint
    1.3e StereoAnalysis (depends on AtomAnalysis)
    1.3f SymmetryAnalysis (depends on AtomAnalysis + RingAnalysis)
    1.3g Perception facade (lazy @property init) + interpretation generation
         v13 B2: Generator function contract

1.4 Build fragment extraction utilities
    - carve_substituent (with canonical index normalization, v13 G1)
    - carve_bridging_substituent
    - carve_fc_fragments
    - strip_additive_atoms (v13 C1)

1.5 Build assembly (NameTree -> string)
    - derive_sort_name (v13 I1: corrected regex)
    - merge_identical_prefixes (v13 A4: bracket rules)
    - SUFFIX_VARIANT_TABLE + resolve_suffix_variant (v13 D3)
    - Table-driven _acid_to_adjective
    - Replacement prefix rendering
    - Additive name rendering
    - Two-method substituent rendering
    - Free-valence suffix table
    - Elision (table-driven)
    - Prefix ordering, FC templates, multiplicative/ring assembly frames

1.6 Build engine skeleton
    - name() -> search -> execute -> recurse
    - NamingSession with extended cache key
    - Three-tier plan generation (v13 B1)
    - Additive as pre-search wrapper (v13 B3)
    - with_warnings copy-on-write
    - Saved-best-tree retry loop
    - Adaptive plan cap
    - Class-based handler registry

1.7 Build ring_naming/ package
    1.7a ring_naming/common.py + ring_naming/retained_lookup.py
    1.7b ring_naming/monocyclic.py (Hantzsch-Widman + systematic)
    1.7c ring_naming/bridged.py (von Baeyer)
    1.7d ring_naming/spiro.py
    1.7e ring_naming/fused.py (largest — start early)
    1.7f ring_naming/numbering.py

1.8 Unit tests for each layer independently
```

### Phase 2: Core Naming (iterative, test-driven)

Budget ~55% of total effort. Each sub-phase: implement, test, debug, commit with score.

```
2a. Minimal substitutive path (chain-only, single FG)
    - Chain parent finding + numbering
    - Single-FG suffix attachment
    - Fragment carving for simple substituents
    - Method (1) alkyl-type substituent names
    - N-locant support for amides/amines
    - v13 A1: Outside-in prefix discovery
    - Run OPSIN suite on chain-only subset
    Target:  Pessimistic 20%  Realistic 28%  Optimistic 35%

2b. Ring parents
    - Ring naming package (all sub-modules)
    - Ring numbering
    - Both alkyl_stem and stem variants
    - Run full suite
    Target:  Pessimistic 35%  Realistic 43%  Optimistic 52%

2c. Multi-FG molecules
    - v13 A2: PCG grouping by type
    - Prefix assignment for demoted FGs
    - Prefix deduplication + derive_sort_name
    - Method (2) alkanyl-type substituent names
    - Bridging substituent detection and naming
    - Adaptive plan cap (PlanComplexity)
    - Run full suite
    Target:  Pessimistic 47%  Realistic 55%  Optimistic 63%

2d. Functional class path
    - Ester, anhydride, acid halide decomposition
    - FC detection patterns + intramolecular flag
    - carve_fc_fragments (intermolecular path)
    - v13 E1: Two-level scoring with accept_plan
    - Run full suite
    Target:  Pessimistic 54%  Realistic 62%  Optimistic 70%

2e. Multiplicative + ring assembly
    - Symmetry detection
    - Multiplicative handler (locants resolved in execution)
    - Ring assembly handler
    - Run full suite
    Target:  Pessimistic 58%  Realistic 66%  Optimistic 74%

2f. Strategy scoring (P-44, P-65)
    - Full canonical preference scoring in IUPACCanonical
    - Magnitude bands (v13 E1)
    - Substituent method scoring (Method 1 vs 2 preference)
    - Run full suite
    Target:  Pessimistic 64%  Realistic 72%  Optimistic 80%
```

### Phase 3: Completeness + Replacement Nomenclature

```
3a. Replacement nomenclature path handler
    - ReplacementPlan, ReplacementTree, ReplacementPath
    - REPLACEMENT_A_PREFIXES table
    - Internal heteroatom detection
    - Replacement prefix assembly
    - Should address ~29% of current failures

3b. Stereochemistry (CIP, descriptors, rendering)
3c. Retained name expansion (table coverage)
3d. Ambiguity points + interpretation generation
3e. OutputForm decomposition constraints (full enforcement)
3f. Heteroatom parent hydrides (phosphane, silane, borane, etc.)
3g. Error recovery (child failure -> retry)
3h. Salt handling
3i. Run test suite

Target:  Pessimistic 82%  Realistic 88%  Optimistic 93%
```

### Phase 4: Extensions

```
4a. AlignedStrategy (OPSIN-backed name parsing + pattern similarity)
4b. CASStrategy
4c. Additive nomenclature path handler (N-oxide, P-oxide)
4d. Conjunctive nomenclature path handler
4e. Organometallic perception + coordination path handler
```

---

## Line Count Estimates

```
Perception (~3200-3800 lines):
  AtomAnalysis         ~200
  StereoAnalysis       ~400
  FragmentAnalysis     ~100
  RingAnalysis         ~700
  FGDetection          ~600
  SymmetryAnalysis     ~300
  ChainFinding         ~500
  Interpretation gen   ~200
  Retained matching    ~100
  Fragment extraction  ~250

Ring Naming (~2250 lines):
  common + retained    ~300
  monocyclic           ~300
  fused                ~800
  bridged              ~400
  spiro                ~200
  numbering            ~250

Strategy (~1000 lines):
  IUPACCanonical       ~500
  AlignedStrategy      ~300
  CASStrategy          ~200

Engine (~1400 lines):
  Session, search, dispatch, recursion, path handlers

Assembly (~850 lines):
  All tree-type assembly, prefix ordering, elision, FC templates

Total: ~8700-9300 lines
```

---

## Testing Methodology

- **OPSIN round-trip eval** (1181 compounds): primary metric
- **Unit tests per subsystem**: perception, assembly, ring naming each testable independently
- **Commit convention**: `Score XXX/1181 (XX.X%): description`
- **Delta tracking**: eval compares to previous run automatically

---

## Known Limitations

1. Coarse plan ordering may evaluate more plans than necessary for unusual molecules
2. Recursion can produce unexpected sub-names (mitigated by OutputForm constraints)
3. AlignedStrategy depends on OPSIN (runtime dependency, not for canonical)
4. Retained name coverage determines the floor
5. Ring naming depends on correct perception structural descriptors
6. Isotope descriptors not initially supported
7. Radicals and zwitterions not explicitly handled
8. Indicated hydrogen rules complex for fused/bridged heterocycles
9. Strategy scoring is structural, not string-based
10. Intramolecular FC detected but rejected by canonical strategy
11. Subsumption table completeness (v13 F1: unknown → ambiguity mitigates)
12. Locant canonicalization depends on RDKit version (pin it)
13. Prefix sort key edge cases for unusual naming forms
