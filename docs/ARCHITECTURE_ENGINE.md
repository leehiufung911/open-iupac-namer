# Engine Layer

**v13 — merged from v12 + v13 delta**

Detailed specification of the Engine layer including all path handlers. For high-level overview, see `ARCHITECTURE_OVERVIEW.md`. For type definitions, see `ARCHITECTURE_DATA_STRUCTURES.md`.

---

## Role

The engine generates plans, asks the strategy to score them, executes the best one, and recurses. It knows about naming path types as structural categories. It does NOT know about specific functional groups, elements, or preference rules.

---

## NamingSession

All mutable state lives in a session object, scoped to one top-level `name()` call:

```python
@dataclass
class NamingSession:
    cache: dict[tuple, NameTree] = field(default_factory=dict)
    max_depth: int = 10
    _plan_seq: int = 0

    def _make_key(self, smiles, output_form, fv_bond_orders, attachment_indices):
        # attachment_indices are already in canonical ordering
        # (guaranteed by carve_substituent/carve_bridging_substituent per v13 G1)
        return (smiles, output_form, fv_bond_orders,
                attachment_indices or ())

    def cache_lookup(self, smiles, output_form, fv_bond_orders,
                     attachment_indices=None) -> NameTree | None:
        return self.cache.get(
            self._make_key(smiles, output_form, fv_bond_orders, attachment_indices))

    def cache_store(self, smiles, output_form, fv_bond_orders, tree,
                    attachment_indices=None):
        self.cache[self._make_key(smiles, output_form, fv_bond_orders,
                                  attachment_indices)] = tree

    def next_seq(self) -> int:
        self._plan_seq += 1
        return self._plan_seq
```

Cache key is `(smiles, output_form, fv_bond_orders, attachment_indices)`. `attachment_indices` distinguishes propan-1-yl from propan-2-yl. SubstituentMethod is NOT in the key (determined by attachment position + strategy, which is fixed per session). DecisionContext is explicitly NOT part of the cache key.

---

## Core Function: name()

```python
def name(mol, strategy: NamingStrategy,
         output_form: OutputForm = OutputForm.STANDALONE,
         free_valence: FreeValenceInfo | None = None,
         decision_ctx: DecisionContext | None = None,
         _session: NamingSession | None = None,
         _depth: int = 0) -> NameTree:

    if _session is None:
        _session = NamingSession()

    smiles = canonical_smiles(mol)
    fv_bond_orders = free_valence.bond_orders if free_valence else ()
    attachment_indices = (
        free_valence.attachment_atoms_in_fragment
        if free_valence and free_valence.attachment_atoms_in_fragment
        else None
    )

    cached = _session.cache_lookup(smiles, output_form, fv_bond_orders,
                                    attachment_indices)
    if cached is not None:
        return cached

    if _depth > _session.max_depth:
        return ErrorTree(message=f"Max recursion depth exceeded", ...)

    perception = Perception(mol)

    # --- Salt check (pre-interpretation) ---
    fragments = perception.fragments.detect()
    if len(fragments) > 1:
        tree = _name_salt(...)
        _session.cache_store(smiles, output_form, fv_bond_orders, tree,
                             attachment_indices)
        return tree

    # --- Additive check (pre-interpretation, v13 B3) ---
    additive_groups = perception.fgs.detect_additive_groups()
    if additive_groups and strategy.accept_additive(additive_groups):
        parent_mol, atom_map = strip_additive_atoms(mol, additive_groups)
        parent_tree = name(parent_mol, strategy, OutputForm.STANDALONE,
                           free_valence=None, ...)
        return AdditiveTree(
            parent_tree=parent_tree,
            additions=_build_additive_groups(additive_groups, perception, atom_map),
            ...
        )

    # --- Normal plan search ---
    query = strategy.interpretation_query(mol)
    if output_form in (OutputForm.SUBSTITUENT, OutputForm.ACYL):
        query = query.with_override(suppress_functional_class=True)

    ranked_plans = _search_plans(perception, mol, output_form, query,
                                 strategy, _session)

    if not ranked_plans:
        return ErrorTree(message=f"No valid naming plan found for {smiles}", ...)

    # --- Execute best plan; retry on child failure ---
    best_tree = None
    best_score = float('-inf')

    for score, _seq, plan in reversed(ranked_plans):
        tree = execute_plan(plan, mol, strategy, output_form, free_valence,
                            decision_ctx, _session, _depth)
        if not _has_error_children(tree):
            _session.cache_store(smiles, output_form, fv_bond_orders, tree,
                                 attachment_indices)
            return tree
        if score > best_score:
            best_score = score
            best_tree = tree

    warned_tree = with_warnings(best_tree,
        ("All plans had sub-fragment errors; returning best attempt",))
    _session.cache_store(smiles, output_form, fv_bond_orders, warned_tree,
                         attachment_indices)
    return warned_tree
```

---

## Three-Tier Plan Generation (v13 revised: B1)

```python
def _generate_all_plans(perception, mol, output_form, query, strategy, session):
    """Yield (score, seq, plan) tuples.

    Three tiers:
      Tier 0 — Retained name check (interpretation-independent)
      Tier 1 — Always-available path handlers (per interpretation):
               Substitutive + Replacement
      Tier 2 — Decomposition-based path handlers (per interpretation):
               Functional class, multiplicative, ring assembly

    Additive is NOT a plan tier — it's a pre-search wrapper (v13 B3).
    """
    max_plans = _DEFAULT_MAX_PLANS

    # --- Tier 0: Retained names ---
    yield from _generate_retained_plans(perception, mol, output_form,
                                        strategy, session)

    # --- Tier 1 + Tier 2: per-interpretation ---
    for i, interpretation in enumerate(perception.interpretations(query)):

        # Compute complexity lazily on first interpretation (v13 B2)
        if i == 0:
            complexity = _estimate_complexity(interpretation, perception)
            max_plans = strategy.max_plans_hint(complexity)

        # Tier 1a: Substitutive
        yield from _generate_from_handler(
            "substitutive", None, interpretation, perception, strategy, session)

        # Tier 1b: Replacement (self-filters when no internal heteroatoms)
        yield from _generate_from_handler(
            "replacement", None, interpretation, perception, strategy, session)

        # Tier 2: Decomposition-based
        for decomp in interpretation.decomposition_candidates():
            yield from _generate_from_handler(
                decomp.type, decomp, interpretation, perception, strategy, session)

        if session.plan_count >= max_plans:
            return


def _generate_from_handler(handler_name, decomp, interpretation,
                           perception, strategy, session):
    handler_cls = _PATH_HANDLERS.get(handler_name)
    if handler_cls is None:
        return
    handler = handler_cls()
    for plan in handler.generate_plans(decomp, interpretation, perception, strategy):
        if not strategy.accept_plan(plan):
            continue
        score = strategy.score_plan(plan)
        yield (score, session.next_seq(), plan)
```

### Plan Search with Early Termination

```python
def _search_plans(perception, mol, output_form, query, strategy, session):
    ranked_plans = []
    good_enough = strategy.good_enough_score()

    for score, seq, plan in _generate_all_plans(perception, mol, output_form,
                                                 query, strategy, session):
        insort(ranked_plans, (score, seq, plan))
        if score >= good_enough:
            break  # Found good enough — stop searching immediately

    return ranked_plans
```

`good_enough_score()` is checked after every single plan insertion, not just at interpretation boundaries.

---

## Path Handler Registry

```python
_PATH_HANDLERS: dict[str, type[PathHandler]] = {}  # stores CLASSES, not instances

def register_path(decomp_type: str):
    def decorator(cls):
        _PATH_HANDLERS[decomp_type] = cls
        return cls
    return decorator

class PathHandler(Protocol):
    def generate_plans(self, decomp, interpretation, perception, strategy
                       ) -> Iterator[NamingPlan]: ...
    def execute(self, plan, mol, strategy, output_form, free_valence,
                decision_ctx, session, depth) -> NameTree: ...
```

---

## Path Handler: Substitutive (v13 revised: A1, A2)

### Plan Generation

```python
@register_path("substitutive")
class SubstitutivePath:

    def generate_plans(self, decomp, interpretation, perception, strategy):
        # v13 A2: Group suffix-eligible FGs by type, iterate over types
        eligible = [fg for fg in interpretation.fgs if fg.suffix_eligible]
        type_groups: dict[str, list[DetectedFG]] = {}
        for fg in eligible:
            type_groups.setdefault(fg.type, []).append(fg)

        sorted_types = sorted(type_groups.keys(),
                              key=lambda t: -FG_SENIORITY.get(t, 0))
        pcg_options = [(t, type_groups[t]) for t in sorted_types] + [(None, [])]

        for pcg_type, pcg_instances in pcg_options:
            pcg_anchors = tuple(fg.anchor for fg in pcg_instances)

            for candidate in perception.candidate_parents(
                    interpretation, pcg_anchors=pcg_anchors):

                # v13 A2: Adjacency-based anchor filter
                if pcg_anchors:
                    anchor_set = set(pcg_anchors)
                    on_parent = candidate.atom_indices & anchor_set
                    if not on_parent:
                        bonded_to_parent = any(
                            neighbor in candidate.atom_indices
                            for anchor in pcg_anchors
                            for neighbor in perception.atoms[anchor].neighbors
                        )
                        if not bonded_to_parent:
                            continue

                for named_parent in name_parent_candidates(candidate, strategy):
                    for numbering in compute_numberings(named_parent,
                                                         pcg_instances,
                                                         interpretation.fgs):
                        suffix_groups = self._compute_suffixes(
                            pcg_type, pcg_instances, named_parent,
                            numbering, interpretation.fgs)
                        prefix_assignments = self._compute_prefix_assignments(
                            interpretation, pcg_type, pcg_instances,
                            named_parent, numbering, ...)

                        yield SubstitutivePlan(...)
```

### Suffix Resolution — Two-Stage (v13 D1, D2)

**Stage 1 (`_compute_suffixes`):** Determine base suffix form (terminal vs nonterminal):

```python
def _compute_suffixes(self, pcg_type, pcg_instances, named_parent,
                      numbering, all_fgs) -> tuple[SuffixGroup, ...]:
    parent_atoms = named_parent.candidate.atom_indices
    parent_terminal_atoms = _terminal_atoms(named_parent)

    for fg in pcg_instances:
        is_terminal = fg.anchor in parent_terminal_atoms

        if is_terminal:
            form = fg.suffix_forms.get("terminal", fg.suffix_forms.get("default"))
            locant = numbering.atom_to_locant.get(fg.anchor)
        else:
            form = fg.suffix_forms.get("nonterminal", fg.suffix_forms.get("default"))
            # v13 D2/Issue 19: Non-terminal locant from PARENT NEIGHBOR
            parent_neighbor = _find_parent_neighbor(fg.anchor, parent_atoms, mol)
            locant = numbering.atom_to_locant.get(parent_neighbor)

        suffix_groups.append(SuffixGroup(
            fg=fg,
            locants=(locant,) if locant is not None else (),
            base_form=form,
            elides_terminal_e=SUFFIX_ELISION_TABLE.get(form, False),
        ))

def _find_parent_neighbor(anchor_idx, parent_atoms, mol) -> int | None:
    """Find the parent atom bonded to an off-parent FG anchor."""
    atom = mol.GetAtomWithIdx(anchor_idx)
    for neighbor in atom.GetNeighbors():
        if neighbor.GetIdx() in parent_atoms:
            return neighbor.GetIdx()
    return None

def _terminal_atoms(named_parent) -> frozenset[int]:
    """Chain: first and last. Ring: empty. Heteroatom center: the center."""
```

**Stage 2 (`render_suffixes`):** Apply OutputForm variant — lives in assembly (see ARCHITECTURE_ASSEMBLY.md).

### Substituent Discovery — Outside-In (v13 A1)

```python
def _compute_prefix_assignments(self, interpretation, pcg_type, pcg_instances,
                                named_parent, numbering, mol):
    """Discover ALL substituent groups on the parent.

    Algorithm — outside-in:
    1. parent_atoms = named_parent.candidate.atom_indices
       suffix_atoms = union of (fg.atoms - parent_atoms) for each PCG instance
    2. claimed = parent_atoms | suffix_atoms
       remaining = all atoms - claimed
    3. Partition remaining into connected components (flood-fill within remaining)
    4. For each component:
       a. Find attachment bonds (component ↔ parent_atoms)
       b. 1 attachment → TerminalPrefix; 2+ → BridgingPrefix
       c. If component contains demoted FG → role = "demoted_fg"
       d. Record locant(s) on parent
    """
```

### Execution

```python
    def execute(self, plan, mol, strategy, output_form, free_valence,
                decision_ctx, session, depth):
        prefixes = []
        for pa in plan.prefix_assignments:
            match pa:
                case TerminalPrefix():
                    fragment_mol, attachment_idx, bond_order = carve_substituent(
                        mol, pa.substituent_atoms, pa.attachment_bond)
                    sub_tree = name(
                        fragment_mol, strategy, pa.output_form,
                        free_valence=FreeValenceInfo(
                            bond_orders=(bond_order,),
                            method=_select_substituent_method(...),
                            attachment_atoms_in_fragment=(attachment_idx,)),
                        _session=session, _depth=depth + 1,
                    )
                    prefixes.append(PrefixEntry(
                        tree=sub_tree,
                        locants=(pa.locant,) if pa.locant is not None else (),
                    ))
                case BridgingPrefix():
                    # ... similar, always Method (2) ...

        return SubstitutiveTree(
            named_parent=plan.named_parent,
            numbering=plan.numbering,
            suffix_groups=plan.suffix_groups,
            unsaturation=plan.unsaturation,
            prefixes=tuple(prefixes),
            ...
        )
```

---

## Path Handler: Functional Class

```python
@register_path("functional_class")
class FunctionalClassPath:

    def generate_plans(self, decomp, interpretation, perception, strategy):
        template = FUNCTIONAL_CLASS_TEMPLATES[decomp.subtype]
        if template.symmetric:
            yield self._make_plan(decomp, interpretation, template.roles)
        else:
            for role_assignment in template.role_permutations(decomp.pieces):
                yield self._make_plan(decomp, interpretation, role_assignment)

    def execute(self, plan, mol, strategy, output_form, free_valence,
                decision_ctx, session, depth):
        fragment_mols = carve_fc_fragments(mol, plan.decomposition)
        pieces = {}
        for role, form in plan.fragment_output_forms:
            frag_mol = fragment_mols[role]
            pieces[role] = name(frag_mol, strategy, form,
                               _session=session, _depth=depth + 1)
        return FunctionalClassTree(subtype=plan.decomposition.subtype,
                                    pieces=tuple((r, t) for r, t in pieces.items()), ...)
```

---

## Path Handler: Multiplicative (P-51.3)

```python
@register_path("multiplicative")
class MultiplicativePath:

    def generate_plans(self, decomp, interpretation, perception, strategy):
        sym = decomp.symmetry_group
        for linking_atoms in self._enumerate_linking_positions(sym, perception):
            yield MultiplicativePlan(
                decomposition=decomp,
                multiplier=MULTIPLICATIVE_PREFIXES[sym.multiplicity],
                linking_atom_indices=linking_atoms, ...)

    def execute(self, plan, mol, strategy, output_form, free_valence,
                decision_ctx, session, depth):
        sym = plan.decomposition.symmetry_group
        subunit_tree = name(sym.subunit_mol, strategy, OutputForm.STANDALONE,
                            _session=session, _depth=depth + 1)
        locants = self._resolve_locants(plan.linking_atom_indices, subunit_tree)
        return MultiplicativeTree(subunit=subunit_tree, multiplier=plan.multiplier,
                                   locants=locants, ...)
```

---

## Path Handler: Ring Assembly (P-28.4)

```python
@register_path("ring_assembly")
class RingAssemblyPath:

    def generate_plans(self, decomp, interpretation, perception, strategy):
        sym = decomp.symmetry_group
        yield RingAssemblyPlan(
            decomposition=decomp,
            multiplier=ASSEMBLY_PREFIXES[sym.multiplicity],
            locants=decomp.locants, ...)

    def execute(self, plan, mol, strategy, output_form, free_valence,
                decision_ctx, session, depth):
        ring_tree = name(sym.subunit_mol, strategy, OutputForm.STANDALONE,
                         _session=session, _depth=depth + 1)
        return RingAssemblyTree(ring_unit=ring_tree, multiplier=plan.multiplier,
                                 locants=plan.locants, ...)
```

---

## Path Handler: Replacement Nomenclature (P-15.4, P-22.1)

```python
REPLACEMENT_A_PREFIXES = {
    "O": "oxa", "S": "thia", "Se": "selena", "Te": "tellura",
    "N": "aza", "P": "phospha", "As": "arsa", "Si": "sila",
    "Ge": "germa", "Sn": "stanna", "B": "bora", ...
}

@register_path("replacement")
class ReplacementPath:

    def generate_plans(self, decomp, interpretation, perception, strategy):
        """Self-filters: returns immediately when no candidate parents have
        internal heteroatoms. Calling unconditionally is cheap."""
        # For each PCG type x candidate parent containing internal heteroatoms:
        # 1. Construct all-carbon skeleton
        # 2. Name carbon parent
        # 3. Enumerate replacement prefixes
        # 4. Combine with PCG/suffix/prefix as in substitutive
        ...

    def execute(self, plan, mol, strategy, output_form, free_valence,
                decision_ctx, session, depth):
        """Substituent naming identical to SubstitutivePath.execute."""
        ...
```

---

## Path Handler: Additive (P-68.3) — Pre-Search Wrapper (v13 B3)

Additive nomenclature (N-oxide, P-oxide) is NOT a plan tier — it wraps another plan. Detection and stripping happen BEFORE plan search:

```python
# In name():
additive_groups = perception.fgs.detect_additive_groups()
if additive_groups and strategy.accept_additive(additive_groups):
    parent_mol, atom_map = strip_additive_atoms(mol, additive_groups)
    parent_tree = name(parent_mol, strategy, OutputForm.STANDALONE, ...)
    return AdditiveTree(parent_tree=parent_tree,
                         additions=_build_additive_groups(...), ...)
```

This solves: (1) no undefined `_generate_parent_plans`, (2) parent named on modified molecule, (3) additive doesn't compete in plan scoring.

---

## Salt Handling

```python
def _name_salt(fragments, strategy, output_form, ...):
    """Ordering rules (P-16.3.4): cations > anions > neutral,
    alphabetical within same charge class."""
```

---

## OutputForm System

```python
class OutputForm(Enum):
    STANDALONE = auto()       # "ethanol", "acetic acid"
    SUBSTITUENT = auto()      # "ethyl", "propan-2-yl" — suppresses FC
    ACID_STEM = auto()        # "acetate", "benzoate"
    ACYL = auto()             # "acetyl" — suppresses FC
    ANION = auto()            # "ethanolate", "phenoxide"
    CATION = auto()           # "ethylium"
    PARENT_HYDRIDE = auto()   # "ethane" (no suffix)
```

### OutputForm Contracts

- **STANDALONE** — PCG is most senior suffix-eligible FG. FC allowed.
- **SUBSTITUENT** — No PCG (all FGs as prefixes). Free-valence suffix. FC suppressed. Method (1) or (2) per P-29.2.
- **ACID_STEM** — Suffix is "-ate"/"-oate" form. FC allowed.
- **ACYL** — Suffix is acyl form "-yl". FC suppressed.
- **ANION** — Suffix "-ate"/"-ide".
- **CATION** — Suffix "-ium"/"-ylium".
- **PARENT_HYDRIDE** — No PCG, no suffix. Just parent + unsaturation + terminal vowel.

---

## Error Handling

### Child Failure Recovery

When recursive call produces ErrorTree:
1. `_has_error_children` detects it
2. Engine tries next-best plan from `ranked_plans`
3. If all fail, applies `with_warnings()` (copy-on-write) to best tree

### Error Escalation

- Leaf errors → ErrorTree, engine retries with systematic plan
- Child errors → engine retries parent with different plan
- Plan-level errors → ErrorTree, escalates to caller
- Depth errors → ErrorTree with diagnostic message

### Validation

Post-execution `validate_tree` checks: all atoms accounted for, no double-claims, locants consistent. Violations produce warnings, not hard errors.
