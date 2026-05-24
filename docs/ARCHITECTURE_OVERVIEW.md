# IUPAC Naming Engine: Architecture Overview

**v13 -- merged from v12 + v13 delta**

High-level overview of the chem-namer-2 architecture. For detailed type definitions, see `ARCHITECTURE_DATA_STRUCTURES.md`. For layer-specific details, see the corresponding architecture documents.

---

## Design Principles

1. **Naming is a sequence of coupled choices at multiple levels.** Every valid IUPAC name is the result of choosing a structural interpretation, then a naming plan within it, then assembling a string. The canonical name is one specific path through this space. Every other valid name is another path.

2. **Three phases of resolution, each with a clear contract.**
   - *Plan*: structural commitment. Which FG is the PCG, which structure is the parent, which numbering, retained vs systematic ring name, suffix forms, substituent method (alkyl vs alkanyl). Plans are scored as a unit by strategy. Plans do NOT contain concrete substituent names -- those require recursion.
   - *Execution*: recursive realization. Names sub-fragments by recursively calling the engine, producing concrete NameTree children. Deterministic given a plan + strategy.
   - *Assembly*: deterministic formatting. Alphabetical prefix ordering, spelling, elision, hyphenation. All structural decisions have been made upstream; assembly applies only deterministic formatting rules.

3. **Perception generates, strategy steers.** Perception knows what's structurally valid. Strategy knows what's preferred. Perception never makes naming decisions. Strategy never reasons about chemistry. The bridge is a query/hint mechanism that controls search order. Corollary: ring *naming* (retained vs systematic) is a naming decision, so it belongs in plan generation, not perception. Perception produces structural descriptors; the ring naming module turns them into named parents.

4. **One engine, many strategies.** Canonical, CAS, aligned naming are all the same engine with different strategy objects.

5. **Naming paths are first-class and extensible.** Substitutive, functional class, multiplicative, ring assembly -- each is a registered path handler. New nomenclature categories are added without modifying the engine core.

6. **Element-agnostic infrastructure.** Atom-level analysis, the engine, NameTree, and assembly make no assumptions about which elements are present. Organic-specific knowledge lives in FG detection patterns and strategy scoring -- the layers that are easiest to extend.

7. **OutputForm constrains the full pipeline, not just assembly.** When the engine recurses on a fragment with a given OutputForm, that form constrains which decompositions and interpretations are valid for the fragment, not just what string it produces. Each OutputForm must change what *string* is produced, not just what plans are valid. If two forms produce the same string, they should be merged. Express plan-only constraints elsewhere (on Decomposition eligibility or role annotations).

8. **The engine names one molecule at a time.** Higher-level tools (series naming, SAR table formatting) are built on top of the engine, not inside it. The engine's contract: one molecule in, one name out (or one molecule in, several candidate names out).

9. **Typed data structures at layer boundaries.** Plans and name trees are union types, not god objects with optional fields. Each path handler creates and receives its own plan/tree type. The type system enforces which fields exist for which naming path.

10. **Session-scoped state, no globals.** Caches and recursion tracking live in a session object created per top-level call. No module-level mutable state.

11. **Immutable shared data structures.** Plans, name trees, numberings, named parents, and candidate parents are frozen dataclasses. They are shared across plan candidates and cached entries. Mutations would corrupt other plans or cached results. Build once, share freely. Warning attachment on trees uses copy-on-write via `with_warnings()` -- the original tree is never mutated.

---

## System Diagram

```
Strategy ---> InterpretationQuery
                  |  ("I want X kind of interpretation")
                  |
                  v
+----------------------------------+
|       Perception (facade)        |  "what are the valid structural readings?"
|                                  |
|  AtomAnalysis ---> StereoAnalysis|
|       |                          |
|       +---> RingAnalysis ---> FGDetection
|       |         |                |
|       |         +---> SymmetryAnalysis
|       |         |                |
|       |         +---> ChainFinding
|       |                          |
|       +---> FragmentAnalysis     |
|                                  |
|  Output: Iterator[Interpretation] (lazy, query-steered, best-first)
+---------------+------------------+
                |  (typically consume 1-3)
                v
+----------------------------------+     +---------------------+
|            Engine                |<--->|      Strategy        |
|                                  |     |                      |
|  Salt check (pre-interpretation) |     |  score_plan()        |
|  Retained name check (deduped)   |     |  accept_plan()       |
|  Plan generator (lazy) ----------+---->|  accept_additive()   |
|  Per-plan early termination      |     |  good_enough_score() |
|  Execute best plan               |     |                      |
|  Recurse on sub-fragments -------+---->+---------------------+
|  (with session memoization)      |     +---------------------+
|  Retry on child failure          |     |  Fragment Carving    |
|                                  |     |                      |
|  Replacement Path                |     |  carve_substituent   |
|  Additive Path                   |     |  carve_bridging_sub  |
+---------------+------------------+     |  carve_fc_fragments  |
                |  NameTree                |  strip_additive_atoms|
                |  (typed union)         +---------------------+
                v
+----------------------------------+
|           Assembly               |  "what string does this tree produce?"
|                                  |
|  Deterministic string building   |
|  Two-method substituent stems    |
|  Free-valence suffix rendering   |
|  Prefix dedup + alphabetical     |
|  FC template formatting          |
|  Elision (table-driven)          |
+----------------------------------+
                |
                v
          "ethyl acetate"
```

---

## Layer Summaries

### Perception

Perception is a facade over seven independently testable subsystems: AtomAnalysis, StereoAnalysis, FragmentAnalysis, RingAnalysis, FGDetection, SymmetryAnalysis, and ChainFinding. Subsystems are lazily initialized (a retained name check never triggers ring or chain analysis). Perception takes an RDKit mol and an `InterpretationQuery`, and yields a lazy iterator of `Interpretation` objects in best-first order. It resolves structural facts -- atom ownership, ring descriptors, candidate parents -- but never makes naming decisions. See `ARCHITECTURE_PERCEPTION.md` for full details.

### Strategy

Strategy scores structural commitments in plans: PCG seniority (P-65), parent selection (P-44), numbering quality, suffix form, naming method. It uses two-level scoring: magnitude bands within a plan type guarantee higher-priority criteria always dominate, while `accept_plan()` handles cross-type arbitration (e.g., rejecting FC plans when a more senior FG exists). Three concrete strategies are defined: `IUPACCanonical`, `AlignedStrategy`, and `CASStrategy`. See `ARCHITECTURE_STRATEGY.md` for full details.

### Engine

The engine generates plans, asks strategy to score them, executes the best one, and recurses on sub-fragments. All mutable state lives in a `NamingSession` scoped to one top-level call. Path handlers (substitutive, functional class, multiplicative, ring assembly, replacement) are registered via a class-based registry and instantiated per use. Additive nomenclature is a pre-plan-search wrapper, not a plan tier. See `ARCHITECTURE_ENGINE.md` for full details.

### Assembly

Assembly is deterministic. It takes a `NameTree` (the typed union IR produced by the engine) and produces a final name string. All structural decisions have already been made upstream. Assembly handles: alphabetical prefix ordering, prefix deduplication with multipliers, two-method substituent stem/suffix rendering, vowel elision, stereo descriptor rendering, FC template formatting, replacement prefix rendering, additive name rendering, and output form suffix variant transforms via `SUFFIX_VARIANT_TABLE`. See `ARCHITECTURE_ASSEMBLY.md` for full details.

---

## Recursion Model

The engine names one molecule at a time. When a plan requires naming a sub-fragment (a substituent, an FC acid part, a multiplicative subunit), the engine recursively calls `name()` on a carved fragment molecule with an appropriate `OutputForm` and `FreeValenceInfo`. Each recursive call goes through the full pipeline: perception, plan search, strategy scoring, execution, and assembly.

Session memoization prevents redundant work: three identical ethyl groups produce "ethyl" computed once. The cache key is `(canonical_smiles, output_form, fv_bond_orders, attachment_indices)`. Recursion depth is bounded by `NamingSession.max_depth` (default 10); exceeding it produces a graceful `ErrorTree`, not a crash.

---

## OutputForm Enum

```python
class OutputForm(Enum):
    STANDALONE = auto()       # "ethanol", "acetic acid"
    SUBSTITUENT = auto()      # "ethyl", "propan-2-yl" -- suppresses FC decomposition
    ACID_STEM = auto()        # "acetate", "benzoate"
    ACYL = auto()             # "acetyl" -- suppresses FC decomposition
    ANION = auto()            # "ethanolate", "phenoxide"
    CATION = auto()           # "ethylium"
    PARENT_HYDRIDE = auto()   # "ethane" (no suffix) -- for multiplicative subunits
```

Each OutputForm changes what *string* is produced (Principle 7). `SUBSTITUENT` and `ACYL` also suppress functional-class decompositions in recursive calls.

---

## Preprocessor Chain

Before plan search, `name()` runs a chain of preprocessor checks in this order:

1. **Cache check** -- return immediately if this (smiles, output_form, fv_bond_orders, attachment_indices) was already named in this session.
2. **Depth check** -- return `ErrorTree` if recursion depth exceeds `max_depth`.
3. **Salt check** -- if the molecule has multiple disconnected fragments, name each ion separately via `_name_salt()`.
4. **Additive check** -- if additive groups (N-oxide, P-oxide) are detected and `strategy.accept_additive()` approves, strip them, name the parent on the modified molecule, and wrap in `AdditiveTree`.
5. **Retained name check** -- match the whole molecule against the retained name table (Tier 0, interpretation-independent).
6. **Plan search** -- generate and score plans via three-tier generation (Tier 0: retained, Tier 1: substitutive + replacement, Tier 2: decomposition-based FC/multiplicative/ring assembly).

See `ARCHITECTURE_ENGINE.md` for the full specification of each step.

---

## Further Reading

| Document | Contents |
|----------|----------|
| `ARCHITECTURE_DATA_STRUCTURES.md` | All typed dataclasses: Locant, Numbering, SuffixGroup, Fragment, DetectedFG, NamingPlan union, NameTree union, OutputForm, FreeValenceInfo, etc. |
| `ARCHITECTURE_PERCEPTION.md` | Perception facade, 7 subsystems, atom-ownership deconfliction, candidate parent generation, interpretation generation |
| `ARCHITECTURE_STRATEGY.md` | NamingStrategy protocol, two-level scoring, concrete strategies, naming pattern extraction, PlanComplexity |
| `ARCHITECTURE_ENGINE.md` | `name()` function, plan search, path handlers, fragment carving, suffix computation, error handling |
| `ARCHITECTURE_ASSEMBLY.md` | `assemble()` dispatch, substitutive/FC/replacement/additive assembly, prefix sort key, bracket insertion, SUFFIX_VARIANT_TABLE |
| `ARCHITECTURE_BUILD_PLAN.md` | Build phases, known limitations, where knowledge lives, complexity bounds, reference documents, v12-to-v13 changelog |
