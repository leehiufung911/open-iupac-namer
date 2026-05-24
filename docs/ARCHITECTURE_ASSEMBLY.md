# Assembly Layer

**v13 — merged from v12 + v13 delta**

Detailed specification of the Assembly layer. For high-level overview, see `ARCHITECTURE_OVERVIEW.md`. For type definitions, see `ARCHITECTURE_DATA_STRUCTURES.md`.

---

## Role

**Input:** NameTree (typed union)
**Output:** Final name string

Assembly is deterministic. All structural decisions have been made upstream. Assembly applies formatting rules: alphabetical prefix ordering, prefix deduplication with multipliers, two-method substituent stem/suffix rendering, vowel elision, stereo descriptor rendering, FC template formatting, and string concatenation.

---

## Top-Level Dispatch

```python
def assemble(tree: NameTree) -> str:
    match tree:
        case LeafTree():
            return format_for_output_form(tree.text, tree.output_form)
        case SaltTree():
            return " ".join(assemble(ion) for ion in tree.ion_trees)
        case FunctionalClassTree():
            return _assemble_fc(tree)
        case SubstitutiveTree():
            return _assemble_substitutive(tree)
        case MultiplicativeTree():
            subunit_name = assemble(tree.subunit)
            locant_str = ",".join(tree.locants) + "-" if tree.locants else ""
            linking = tree.linking_group + "-" if tree.linking_group else ""
            return f"{locant_str}{linking}{tree.multiplier}({subunit_name})"
        case RingAssemblyTree():
            ring_name = assemble(tree.ring_unit)
            locant_str = ",".join(tree.locants) + "-" if tree.locants else ""
            return f"{locant_str}{tree.multiplier}{ring_name}"
        case ReplacementTree():
            return _assemble_replacement(tree)
        case AdditiveTree():
            return _assemble_additive(tree)
        case ErrorTree():
            return f"[NAMING ERROR: {tree.message}]"
```

---

## Substitutive Assembly

```python
def _assemble_substitutive(tree: SubstitutiveTree) -> str:
    parts = []

    # 1. Stereo descriptors: "(2R,3S)-"
    if tree.stereo_descriptors:
        parts.append(render_stereo(tree.stereo_descriptors))

    # 2. Indicated hydrogen: "1H-"
    if tree.indicated_hydrogen:
        parts.append(render_indicated_h(tree.indicated_hydrogen))

    # 3. Prefixes: assemble each, deduplicate, order alphabetically
    if tree.prefixes:
        assembled_prefixes = []
        for pe in tree.prefixes:
            prefix_name = assemble(pe.tree)
            assembled_prefixes.append((prefix_name, pe.locants))
        merged = merge_identical_prefixes(assembled_prefixes)
        merged.sort(key=lambda m: m.sort_name)
        parts.append(render_merged_prefixes(merged))

    # 4. Parent stem (method-dependent)
    fv = tree.free_valence
    if fv is not None and fv.is_monovalent and fv.method == SubstituentMethod.ALKYL:
        parts.append(tree.named_parent.alkyl_stem or tree.named_parent.stem)
    else:
        parts.append(tree.named_parent.stem)

    # 5. Unsaturation infixes: "-2-en-", "-2,4-dien-"
    if tree.unsaturation:
        parts.append(render_unsaturation(tree.unsaturation))

    # 6. Suffix: FG suffix, free-valence suffix, or terminal vowel
    if tree.suffix_groups:
        parts.append(render_suffixes(tree.suffix_groups, tree.output_form))
    elif fv is not None and any(o > 0 for o in fv.bond_orders):
        parts.append(render_free_valence_suffix(fv, tree.numbering))
    else:
        parts.append(terminal_vowel(tree.named_parent, tree.output_form))

    return elide("".join(parts))
```

---

## Prefix Assembly

### PrefixEntry and MergedPrefix (v13 A3)

```python
@dataclass(frozen=True)
class PrefixEntry:
    """A named substituent prefix, ready for assembly.
    Created during execution (after recursive naming produces the tree)."""
    tree: NameTree
    locants: tuple[Locant, ...]       # always tuple, even for single locant

@dataclass(frozen=True)
class MergedPrefix:
    """Result of grouping identical PrefixEntry trees in assembly."""
    name: str
    locants: tuple[Locant, ...]
    multiplier: str | None             # "di", "tri", etc.
    sort_name: str
    needs_brackets: bool
```

### Bracket Insertion Rules (v13 A4)

Per P-16.3.3 and P-16.3.4:

```python
def merge_identical_prefixes(entries) -> list[MergedPrefix]:
    """Group entries by assembled name, produce multiplied entries.

    Rules:
    1. Simple prefix, count=1: no brackets, no multiplier.
       "methyl", locants=(2,) -> "2-methyl"

    2. Simple prefix, count>1: "di"/"tri"/"tetra" multiplier, no brackets.
       "methyl", count=2 -> "2,4-dimethyl"

    3. Compound prefix, count=1: enclosing brackets, no multiplier.
       "2-chloroethyl" -> "(2-chloroethyl)"

    4. Compound prefix, count>1: "bis"/"tris"/"tetrakis" WITH brackets.
       "2-chloroethyl", count=2 -> "bis(2-chloroethyl)"

    Detection of "compound" prefix:
    - Contains digits followed by hyphen (locant pattern): compound
    - Contains internal brackets: compound
    - Is in the simple-prefix allowlist (methyl, ethyl, phenyl, etc.): simple
    - Default: compound (safe — extra brackets never wrong)
    """
```

### Sort Key Derivation (v13 revised: I1)

```python
def derive_sort_name(prefix_name: str) -> str:
    """Derive alphabetical sort key for a prefix name (P-14.5)."""
    s = prefix_name

    # Step 1: strip outermost enclosing marks (including {})
    while ((s.startswith("(") and s.endswith(")")) or
           (s.startswith("[") and s.endswith("]")) or
           (s.startswith("{") and s.endswith("}"))):
        s = s[1:-1]

    # Step 2: strip multiplicative prefixes (longest-first)
    MULT_PREFIXES = [
        "tetrakis", "pentakis", "hexakis", "heptakis",
        "octakis", "nonakis", "decakis",
        "tetra", "penta", "hexa", "hepta", "octa", "nona", "deca",
        "tris", "bis",
        "tri", "di",
    ]
    for m in MULT_PREFIXES:
        if s.startswith(m):
            rest = s[len(m):]
            if rest and (rest[0].isalpha() or rest[0] in "(["):
                s = rest
                if s.startswith("(") and s.endswith(")"):
                    s = s[1:-1]
                elif s.startswith("[") and s.endswith("]"):
                    s = s[1:-1]
                break

    # Step 3: strip leading locant-hyphen patterns
    import re
    # First strip stereodescriptor prefix: "(2R,3S)-" or "(E)-"
    s = re.sub(r"^\([^)]*\)-", "", s)
    # Then strip locant-hyphen (includes H for indicated hydrogen, P for phosphorus)
    s = re.sub(r"^[0-9NOPSH,'^]+(?:,[0-9NOPSH,'^]+)*-", "", s)

    # Step 4: lowercase
    return s.lower()
```

v13 changes from v12: Fixed regex (removed double-escaping), added H and P to locant character class, added stereodescriptor stripping, strip `{}` brackets.

Non-detachable prefixes (cyclo, iso, neo, sec-, tert-) are RETAINED — they are structural, not multiplicative.

---

## Suffix Assembly

### SUFFIX_VARIANT_TABLE (v13 D3)

```python
SUFFIX_VARIANT_TABLE: dict[tuple[str, OutputForm], str] = {
    # Carboxylic acid variants
    ("oic acid",          OutputForm.STANDALONE):   "oic acid",
    ("oic acid",          OutputForm.ACID_STEM):    "ate",
    ("oic acid",          OutputForm.ACYL):         "oyl",
    ("oic acid",          OutputForm.ANION):        "oate",
    ("carboxylic acid",   OutputForm.STANDALONE):   "carboxylic acid",
    ("carboxylic acid",   OutputForm.ACID_STEM):    "carboxylate",
    ("carboxylic acid",   OutputForm.ACYL):         "carbonyl",
    ("carboxylic acid",   OutputForm.ANION):        "carboxylate",

    # Alcohol variants
    ("ol",                OutputForm.STANDALONE):   "ol",
    ("ol",                OutputForm.ANION):        "olate",

    # Aldehyde variants
    ("al",                OutputForm.STANDALONE):   "al",
    ("al",                OutputForm.ACYL):         "oyl",

    # Ketone variants
    ("one",               OutputForm.STANDALONE):   "one",

    # Amine variants
    ("amine",             OutputForm.STANDALONE):   "amine",
    ("amine",             OutputForm.CATION):       "aminium",

    # ... etc. for each FG x OutputForm combination
}

def resolve_suffix_variant(base_form: str, output_form: OutputForm) -> str:
    return SUFFIX_VARIANT_TABLE.get((base_form, output_form), base_form)

def render_suffixes(suffix_groups, output_form: OutputForm) -> str:
    for sg in suffix_groups:
        rendered_form = resolve_suffix_variant(sg.base_form, output_form)
        # ... locant + multiplier + rendered_form ...
```

**Ownership summary:**
- Stage 1 (`_compute_suffixes` in engine): structure → terminal vs nonterminal
- Stage 2 (`render_suffixes` in assembly): output form → -oic acid vs -ate vs -oyl
- Neither stage knows about the other's concerns

---

## Functional Class Assembly

```python
ACID_ADJECTIVE_TABLE: dict[str, str] = {
    "formic acid": "formic", "acetic acid": "acetic",
    "benzoic acid": "benzoic", "oxalic acid": "oxalic",
    "malonic acid": "malonic", "succinic acid": "succinic",
    "glutaric acid": "glutaric", "maleic acid": "maleic",
    "fumaric acid": "fumaric", "phthalic acid": "phthalic",
    # ... ~22 entries
}

def _acid_to_adjective(acid_name: str) -> tuple[str, str | None]:
    """Table-first, systematic-second (strip " acid"), warning-third."""

def _assemble_fc(tree: FunctionalClassTree) -> str:
    filled = {role: assemble(subtree) for role, subtree in tree.pieces}
    match tree.subtype:
        case "ester":
            return f"{filled['alcohol']} {filled['acid']}"
        case "anhydride":
            adj1, _ = _acid_to_adjective(filled["acid1"])
            adj2, _ = _acid_to_adjective(filled["acid2"])
            if adj1 == adj2:
                return f"{adj1} anhydride"
            return f"{min(adj1, adj2)} {max(adj1, adj2)} anhydride"
        case "acid_halide":
            return f"{filled['acid']} {filled['halide']}"
        case "thioester":
            return f"{filled['thiol']} {filled['acid']}"
```

---

## Replacement Assembly

```python
def _assemble_replacement(tree: ReplacementTree) -> str:
    """Structure: [prefixes][replacement_prefixes][parent_stem][unsaturation][suffix]

    Example: 2,5,8-trioxadecane, 4-aza-2-oxaheptane"""
    # Same prefix assembly as substitutive, then:
    # Replacement prefixes sorted by locant, grouped by element for multiplier
    ...
```

---

## Additive Assembly

```python
def _assemble_additive(tree: AdditiveTree) -> str:
    """Pattern: "{parent_name} {locant}-{multiplier}{type}"
    Examples: pyridine 1-oxide, triphenylphosphane oxide"""
    parent_name = assemble(tree.parent_tree)
    addition_parts = []
    for ag in tree.additions:
        loc_str = f"{ag.locant}-" if ag.locant.is_numeric else ""
        mult_str = ag.multiplier or ""
        addition_parts.append(f"{loc_str}{mult_str}{ag.type}")
    return f"{parent_name} {' '.join(addition_parts)}"
```

---

## Free-Valence Suffix Rendering

```python
FREE_VALENCE_SUFFIXES = {
    (1, (1,)):    "yl",
    (1, (2,)):    "ylidene",
    (1, (3,)):    "ylidyne",
    (2, (1, 1)):  "diyl",
    (3, (1, 1, 1)): "triyl",
    (2, (2, 1)):  "ylylidene",
    (4, (1, 1, 1, 1)): "tetrayl",
}

def render_free_valence_suffix(fv, numbering) -> str:
    """Method (1): just suffix, no locant. Method (2): locants + suffix."""
```

---

## Elision and Terminal Vowel

- Terminal "e" elision driven by `SuffixGroup.elides_terminal_e`, not heuristic
- "-ol" after "methan" → "methanol"
- "-oic acid" after "butan" → "butanoic acid" (no elision)

```python
def terminal_vowel(named_parent, output_form) -> str:
    if output_form == OutputForm.SUBSTITUENT:
        return ""  # free-valence suffix handled separately
    return "e"
```

---

## What Assembly Does

- Render stereo descriptors, indicated hydrogen
- Assemble prefix subtrees to get concrete names
- Merge identical prefixes with multipliers
- Order prefixes alphabetically by sort key
- Render parent stem (method-dependent)
- Render unsaturation infixes
- Render suffix groups with OutputForm variant transform
- Render free-valence suffixes
- Render FC templates (with anhydride adjective extraction)
- Render replacement 'a' prefixes
- Render additive suffixes
- Render multiplicative/ring assembly frames
- Apply elision

## What Assembly Does NOT Do

- Choose suffix forms (plan)
- Choose retained vs systematic names (plan)
- Choose substituent method (plan)
- Resolve FG priorities (strategy)
- Assign output forms (engine)
- Reason about ring topology (NamedParent carries stems)
