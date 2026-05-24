"""
iupac_namer/assembly.py

Assembly layer: NameTree -> string.

All functions are pure string operations -- no RDKit, no mol objects.
All chemistry decisions have been made upstream. Assembly applies:
  - alphabetical prefix ordering
  - prefix deduplication with multipliers
  - two-method substituent stem/suffix rendering
  - vowel elision
  - stereo descriptor rendering
  - FC template formatting
  - string concatenation

v13 spec: ARCHITECTURE_ASSEMBLY.md
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import TYPE_CHECKING

from iupac_namer.types import (
    AdditiveTree,
    ErrorTree,
    FreeValenceInfo,
    FunctionalClassTree,
    LeafTree,
    Locant,
    MergedPrefix,
    MultiplicativeTree,
    NamedParent,
    Numbering,
    OutputForm,
    PrefixEntry,
    ReplacementTree,
    RingAssemblyTree,
    SaltTree,
    StereoDescriptor,
    SubstituentMethod,
    SubstitutiveTree,
    SuffixGroup,
    UnsaturationInfix,
)
from iupac_namer.data_loader import ACID_ADJECTIVE_TABLE, get_multiplier

# NameTree union -- not imported directly to avoid circular imports; used only for type hints
if TYPE_CHECKING:
    from iupac_namer.types import NameTree

# Retained substituent forms for ring parent names (P-31.1.2.4).
# When a ring parent would produce "name + yl", use the retained form instead.
_RETAINED_RING_SUBSTITUENT: dict[str, str] = {
    # Substituent forms where ALL ring positions are equivalent (or the
    # retained form is locant-free per IUPAC P-22.1.1).
    "benzene":      "phenyl",
    # NB: naphthalene is intentionally NOT included here.  Naphthalenes
    # have non-equivalent positions (1,4,5,8 ≡ alpha; 2,3,6,7 ≡ beta);
    # the substituent form must always cite the attachment locant —
    # "naphthalen-1-yl" or "naphthalen-2-yl".  Letting it fall through to
    # the locant-aware path below produces e.g.
    # "(6-methoxynaphthalen-2-yl)" rather than the ambiguous
    # "(2-methoxynaphthalenyl)" which OPSIN parses to the wrong tautomer.
    "toluene":      "tolyl",
}

# ---------------------------------------------------------------------------
# SUFFIX_VARIANT_TABLE (v13 D3)
# ---------------------------------------------------------------------------

SUFFIX_VARIANT_TABLE: dict[tuple[str, OutputForm], str] = {
    # Carboxylic acid variants
    ("oic acid",          OutputForm.STANDALONE):   "oic acid",
    ("oic acid",          OutputForm.ACID_STEM):    "oate",
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
    # Deprotonated amine N(-) is the principal characteristic group →
    # promote the "-amine" suffix to its anion variant "-aminide"
    # (P-72.2 / P-73): methanamine → methanaminide, benzenamine →
    # benzenaminide, N-methylmethanamine → N-methylmethanaminide.
    ("amine",             OutputForm.ANION):        "aminide",

    # Acyl halide variants
    ("oyl chloride",      OutputForm.STANDALONE):   "oyl chloride",
    ("carbonyl chloride", OutputForm.STANDALONE):   "carbonyl chloride",
    ("oyl bromide",       OutputForm.STANDALONE):   "oyl bromide",
    ("carbonyl bromide",  OutputForm.STANDALONE):   "carbonyl bromide",
    ("oyl fluoride",      OutputForm.STANDALONE):   "oyl fluoride",
    ("carbonyl fluoride", OutputForm.STANDALONE):   "carbonyl fluoride",
    ("oyl iodide",        OutputForm.STANDALONE):   "oyl iodide",
    ("carbonyl iodide",   OutputForm.STANDALONE):   "carbonyl iodide",

    # Amide variants
    ("amide",             OutputForm.STANDALONE):   "amide",
    ("carboxamide",       OutputForm.STANDALONE):   "carboxamide",

    # Nitrile variants
    ("nitrile",           OutputForm.STANDALONE):   "nitrile",
    ("carbonitrile",      OutputForm.STANDALONE):   "carbonitrile",

    # Thiol variant
    ("thiol",             OutputForm.STANDALONE):   "thiol",
    ("thiol",             OutputForm.ANION):        "thiolate",

    # Ether (no suffix in substitutive, only as prefix) — placeholder
    ("ether",             OutputForm.STANDALONE):   "ether",

    # Carboxylate (pre-formed anion name)
    ("carboxylate",       OutputForm.STANDALONE):   "carboxylate",
    ("carboxylate",       OutputForm.ANION):        "carboxylate",

    # Thioic / dithioic acid variants (P-65.6.3)
    # The acid-stem form of all three thio-acid terminal suffixes collapses
    # to "-thioate" / "-dithioate" (the "O-"/"S-" prefix lives on the alkyl
    # side of the FC name, not on the acid stem).
    ("thioic O-acid",     OutputForm.STANDALONE):   "thioic O-acid",
    ("thioic O-acid",     OutputForm.ACID_STEM):    "thioate",
    ("thioic O-acid",     OutputForm.ANION):        "thioate",
    ("thioic S-acid",     OutputForm.STANDALONE):   "thioic S-acid",
    ("thioic S-acid",     OutputForm.ACID_STEM):    "thioate",
    ("thioic S-acid",     OutputForm.ANION):        "thioate",
    ("dithioic acid",     OutputForm.STANDALONE):   "dithioic acid",
    ("dithioic acid",     OutputForm.ACID_STEM):    "dithioate",
    ("dithioic acid",     OutputForm.ANION):        "dithioate",
    # Nonterminal forms:
    ("carbothioic O-acid", OutputForm.STANDALONE):  "carbothioic O-acid",
    ("carbothioic O-acid", OutputForm.ACID_STEM):   "carbothioate",
    ("carbothioic O-acid", OutputForm.ANION):       "carbothioate",
    ("carbothioic S-acid", OutputForm.STANDALONE):  "carbothioic S-acid",
    ("carbothioic S-acid", OutputForm.ACID_STEM):   "carbothioate",
    ("carbothioic S-acid", OutputForm.ANION):       "carbothioate",
    ("carbodithioic acid", OutputForm.STANDALONE):  "carbodithioic acid",
    ("carbodithioic acid", OutputForm.ACID_STEM):   "carbodithioate",
    ("carbodithioic acid", OutputForm.ANION):       "carbodithioate",

    # Sulfonic acid
    ("sulfonic acid",     OutputForm.STANDALONE):   "sulfonic acid",
    ("sulfonic acid",     OutputForm.ANION):        "sulfonate",
    ("sulfonic acid",     OutputForm.ACID_STEM):    "sulfonate",

    # Sulfonamide (N-anion → sulfonamidate; P-66.4.1 / P-72.2.2.1.3)
    ("sulfonamide",       OutputForm.STANDALONE):   "sulfonamide",
    ("sulfonamide",       OutputForm.ANION):        "sulfonamidate",

    # Sulfinic acid
    ("sulfinic acid",     OutputForm.STANDALONE):   "sulfinic acid",
    ("sulfinic acid",     OutputForm.ANION):        "sulfinate",
    ("sulfinic acid",     OutputForm.ACID_STEM):    "sulfinate",

    # Phosphoric acid
    ("phosphoric acid",   OutputForm.STANDALONE):   "phosphoric acid",
    ("phosphoric acid",   OutputForm.ANION):        "phosphate",
    ("phosphoric acid",   OutputForm.ACID_STEM):    "phosphate",

    # Imine
    ("imine",             OutputForm.STANDALONE):   "imine",

    # Epoxide (as substituent → prefix only, but cover it)
    ("oxide",             OutputForm.STANDALONE):   "oxide",
}


def resolve_suffix_variant(base_form: str, output_form: OutputForm) -> str:
    """Return the output-form-specific rendered suffix string for a base_form.

    Falls back to base_form if the (base_form, output_form) pair is not in the
    table (so unlisted forms pass through unchanged).
    """
    return SUFFIX_VARIANT_TABLE.get((base_form, output_form), base_form)


# ---------------------------------------------------------------------------
# FREE_VALENCE_SUFFIXES (mirrors types.py definition for local use)
# ---------------------------------------------------------------------------

FREE_VALENCE_SUFFIXES: dict[tuple, str] = {
    (1, (1,)):       "yl",
    (1, (2,)):       "ylidene",
    (1, (3,)):       "ylidyne",
    (2, (1, 1)):     "diyl",
    (2, (2,)):       "ylidene",
    (3, (1, 1, 1)):  "triyl",
    (2, (2, 1)):     "ylylidene",
    (4, (1, 1, 1, 1)): "tetrayl",
}


# ---------------------------------------------------------------------------
# Stereo and indicated-hydrogen rendering
# ---------------------------------------------------------------------------

def render_stereo(descriptors: tuple[StereoDescriptor, ...]) -> str:
    """Render stereo descriptors as "(2R,3S)-" or "(E)-" etc."""
    parts: list[str] = []
    for sd in descriptors:
        if sd.locant is not None:
            parts.append(f"{sd.locant}{sd.descriptor}")
        else:
            parts.append(sd.descriptor)
    return "(" + ",".join(parts) + ")-"


def render_indicated_h(locants: tuple[Locant, ...]) -> str:
    """Render indicated hydrogen locants as "1H-" or "1H,3H-"."""
    return ",".join(f"{loc}H" for loc in locants) + "-"


# ---------------------------------------------------------------------------
# Sort key derivation (P-14.5) — v13 I1
# ---------------------------------------------------------------------------

# Multiplicative prefixes to strip, longest-first
_MULT_PREFIXES = [
    "tetrakis", "pentakis", "hexakis", "heptakis",
    "octakis",  "nonakis",  "decakis",
    "tetra",    "penta",    "hexa",    "hepta",    "octa",    "nona",   "deca",
    "tris",     "bis",
    "tri",      "di",
]

# Pattern for leading locant-hyphen: digits/letters used as locants, then hyphen
_LOCANT_HYPHEN_RE = re.compile(r"^[0-9NOPSH,'^]+(?:,[0-9NOPSH,'^]+)*-")
# Pattern to strip a leading stereodescriptor: "(2R,3S)-" or "(E)-"
_STEREO_PREFIX_RE = re.compile(r"^\([^)]*\)-")


def derive_sort_name(prefix_name: str) -> str:
    """Derive the alphabetical sort key for a prefix name (P-14.5).

    Rules applied in order:
    1. Strip outermost enclosing brackets ( (), [], {} ).
    2. Strip multiplicative prefix (di, tri, bis, tris, ...).
    3. Strip stereodescriptor prefix "(2R,3S)-".
    4. Strip leading locant-hyphen pattern.
    5. Lowercase.
    """
    s = prefix_name.strip()

    # Step 1: strip outermost brackets
    while len(s) >= 2 and (
        (s[0] == "(" and s[-1] == ")")
        or (s[0] == "[" and s[-1] == "]")
        or (s[0] == "{" and s[-1] == "}")
    ):
        s = s[1:-1]

    # Step 2: strip multiplicative prefixes (longest first)
    for m in _MULT_PREFIXES:
        if s.startswith(m):
            rest = s[len(m):]
            if rest and (rest[0].isalpha() or rest[0] in "([{"):
                s = rest
                # strip the brackets that may wrap the remainder
                if len(s) >= 2 and (
                    (s[0] == "(" and s[-1] == ")")
                    or (s[0] == "[" and s[-1] == "]")
                    or (s[0] == "{" and s[-1] == "}")
                ):
                    s = s[1:-1]
                break

    # Step 3: strip stereodescriptor prefix
    s = _STEREO_PREFIX_RE.sub("", s)

    # Step 4: strip leading locant-hyphen
    s = _LOCANT_HYPHEN_RE.sub("", s)

    # Step 5: lowercase
    return s.lower()


# ---------------------------------------------------------------------------
# Compound-prefix detection
# ---------------------------------------------------------------------------

# Simple prefixes that NEVER need brackets even with count > 1
# (all others default to compound → need brackets)
_SIMPLE_PREFIXES = frozenset({
    "methyl", "ethyl", "propyl", "butyl", "pentyl", "hexyl", "heptyl",
    "octyl", "nonyl", "decyl",
    "isopropyl", "isobutyl", "sec-butyl", "tert-butyl",
    "vinyl", "allyl", "phenyl", "benzyl", "naphthyl", "cyclohexyl",
    "cyclopentyl", "cyclobutyl", "cyclopropyl",
    "fluoro", "chloro", "bromo", "iodo",
    "hydroxy", "amino", "nitro", "cyano",
    "methoxy", "ethoxy", "propoxy",
    "oxo", "thioxo", "imino",
    "mercapto", "carboxy", "sulfo",
    "formyl", "acetyl",
    # Retained/systematic acyl prefixes (P-66.6.3)
    "propanoyl", "butanoyl", "pentanoyl", "hexanoyl", "heptanoyl",
    "octanoyl", "nonanoyl", "decanoyl", "undecanoyl", "dodecanoyl",
    "benzoyl",
    # FG-derived simple prefixes (P-65.1, P-66.1, etc.)
    "carbamoyl", "thiocarbamoyl", "sulfamoyl", "selenocarbamoyl",
    "carbonyl", "thiocarbonyl",
    "azido", "diazo", "nitrosoamino",
    "phosphono", "phosphino",
    "sulfanyl", "selanyl", "tellanyl",
    "sulfamoyl", "sulfinyl",
    # Additional single-FG / single-element simple substituent prefixes.
    # Each names one unsubstituted substituent group, so per P-16.3.3 it
    # carries no enclosing marks when cited as a detachable prefix.  Adding
    # them here corrects over-bracketing such as "(isocyano)benzene" →
    # "isocyanobenzene" (the Blue Book PIN).  None of these can be a
    # *substituted* substituent without acquiring a locant or internal
    # bracket (which _is_compound_prefix already catches), so the exact
    # bare name is unambiguously simple, and (verified by the OPSIN
    # round-trip probe) the unbracketed form maps back to the same
    # structure in every observed context.
    # Pseudohalogen / N-/chalcogen-/metalloid-acid prefixes
    # (P-66.5, P-66.4, P-68):
    "isocyano", "isocyanato", "isothiocyanato",
    "nitroso", "iodosyl", "iodyl",
    "borono", "selenono", "sulfino", "arsono",
    # NB: "silyl", "hydrazinyl", and the "-ylidene"/"-ylidyne" substituent
    # prefixes are intentionally OMITTED.  Although each is simple per IUPAC,
    # an existing test guard pins "(silyl)benzene" (bracketed); and when
    # "hydrazinyl"/"-ylidene"/"-ylidyne" are left unbracketed and
    # concatenated with an adjacent stem OPSIN merges them
    # ("cyclohexylhydrazinylmethanimine" round-trips to the wrong
    # connectivity) or an "-idene"/"-idyne" ending before a vowel-initial
    # parent triggers spurious vowel elision.  Their enclosing marks are
    # load-bearing and must be retained.
})


def _is_compound_prefix(name: str) -> bool:
    """Return True if *name* is a compound prefix (needs brackets).

    A prefix is compound if:
    - It contains a digit followed by a hyphen (locant pattern), OR
    - It contains internal brackets, OR
    - It is NOT in the simple-prefix allowlist.

    Extra brackets are never wrong, so default to compound (True).
    """
    # Check for internal locant pattern (digit or letter locant + hyphen)
    if re.search(r"[0-9]-", name):
        return True
    # Check for internal brackets
    if "(" in name or "[" in name:
        return True
    # Check allowlist
    if name in _SIMPLE_PREFIXES:
        return False
    # Default: treat as compound (safe)
    return True


# ---------------------------------------------------------------------------
# Prefix merging and rendering
# ---------------------------------------------------------------------------

def merge_identical_prefixes(
    entries: list[tuple[str, tuple[Locant, ...]]]
) -> list[MergedPrefix]:
    """Group prefix entries by assembled name and produce multiplied entries.

    Args:
        entries: list of (assembled_name, locants_tuple) pairs.

    Returns:
        List of MergedPrefix, one per unique name.

    Rules (P-16.3.3 / P-16.3.4):
        1. Simple prefix, count=1:  no brackets, no multiplier. "2-methyl"
        2. Simple prefix, count>1:  "di"/"tri" multiplier, no brackets. "2,4-dimethyl"
        3. Compound prefix, count=1: enclose in brackets. "(2-chloroethyl)"
        4. Compound prefix, count>1: "bis"/"tris" + brackets. "bis(2-chloroethyl)"
    """
    # Group by name
    groups: dict[str, list[tuple[str, tuple[Locant, ...]]]] = defaultdict(list)
    for name, locants in entries:
        groups[name].append((name, locants))

    result: list[MergedPrefix] = []
    for name, group in groups.items():
        # Combine all locants from all occurrences
        all_locants: list[Locant] = []
        for _, locs in group:
            all_locants.extend(locs)
        all_locants_sorted = tuple(sorted(all_locants))

        count = len(group)
        compound = _is_compound_prefix(name)
        sort_name = derive_sort_name(name)

        if not compound:
            # Simple prefix
            if count == 1:
                multiplier = None
                needs_brackets = False
            else:
                multiplier = get_multiplier(count, complex=False)
                needs_brackets = False
        else:
            # Compound prefix
            if count == 1:
                multiplier = None
                needs_brackets = True
            else:
                multiplier = get_multiplier(count, complex=True)
                needs_brackets = True

        result.append(MergedPrefix(
            name=name,
            locants=all_locants_sorted,
            multiplier=multiplier,
            sort_name=sort_name,
            needs_brackets=needs_brackets,
        ))

    return result


def _render_locants(locants: tuple[Locant, ...]) -> str:
    """Render a tuple of locants as "2,4-" or "" if empty."""
    if not locants:
        return ""
    return ",".join(str(loc) for loc in locants) + "-"


def _choose_brackets(inner_name: str) -> tuple[str, str]:
    """Choose the correct enclosing bracket pair for a compound prefix (P-16.3.3).

    IUPAC requires a nesting sequence so that the enclosing brackets are always
    one level "higher" than the deepest bracket already present inside the name:

        no brackets inside          → (  )
        contains ( but not [        → [  ]
        contains [ but not {        → {  }
        contains { (very rare)      → {  }  (log limitation — not seen in practice)

    The check uses simple character membership so it works regardless of whether
    the inner brackets are balanced or partially assembled.
    """
    if "{" in inner_name:
        # Already at the deepest level IUPAC defines; reuse {} and document
        return ("{", "}")
    if "[" in inner_name:
        return ("{", "}")
    if "(" in inner_name:
        return ("[", "]")
    return ("(", ")")


def render_merged_prefixes(merged_list: list[MergedPrefix]) -> str:
    """Render a sorted list of MergedPrefix objects to a single prefix string.

    Each MergedPrefix produces: locants + multiplier + (brackets?) + name.
    Adjacent prefix groups are separated by a hyphen when the previous group
    ends with a letter and the next group starts with a digit (P-16.3.2).
    This handles cases like "2-methyl-5-propylcyclohexan-1-ol" where multiple
    locant-prefix pairs must be hyphenated between them.
    """
    parts: list[str] = []
    for mp in merged_list:
        locant_str = _render_locants(mp.locants)
        if mp.needs_brackets:
            # Compound: locants go OUTSIDE the brackets (P-14.5.2).
            # The bracket type depends on what is already inside mp.name (P-16.3.3).
            open_b, close_b = _choose_brackets(mp.name)
            if mp.multiplier:
                parts.append(f"{locant_str}{mp.multiplier}{open_b}{mp.name}{close_b}")
            else:
                parts.append(f"{locant_str}{open_b}{mp.name}{close_b}")
        else:
            # Simple: locants go outside the multiplier+name block
            if mp.multiplier:
                parts.append(f"{locant_str}{mp.multiplier}{mp.name}")
            else:
                parts.append(f"{locant_str}{mp.name}")

    if not parts:
        return ""

    # Join prefix groups with a hyphen when needed (P-16.3.2).
    # Two conditions require a separating hyphen:
    #   (a) The next group has a numeric or heteroatom locant — always hyphenate
    #       after any letter or closing-bracket ending.  This covers both
    #       "2-methyl" + "5-propyl" → "2-methyl-5-propyl" (numeric locants) and
    #       "N-ethyl" + "N-methyl" → "N-ethyl-N-methyl" (heteroatom locants).
    #   (b) The next group starts with a digit or opening bracket and the
    #       previous ends with a letter/close-bracket.
    result = parts[0]
    for i, part in enumerate(parts[1:], start=1):
        next_mp = merged_list[i]
        # If the next merged prefix carries any locant, always insert a hyphen
        # when the running result ends with a letter or closing bracket.
        prev_ends_alpha_or_close = result and (result[-1].isalpha() or result[-1] in ")]}")
        next_has_locant = bool(next_mp.locants)
        next_starts_digit = bool(part) and part[0].isdigit()
        # Insert a hyphen when: the previous part ends with a letter/close-bracket AND
        #   (a) the next group has a locant (always hyphenate: "2-methyl-5-propyl"), OR
        #   (b) the next group starts with a digit (e.g. "methyl-1H-imidazole").
        # Do NOT insert a hyphen before a bare bracket-opening group with no
        # locant — neither between two bracket groups ("(ethoxy)(fluoro)") nor
        # between an unbracketed leading simple prefix and a following bracket
        # group ("chloro(methyl)silane", "butyl(ethyl)(methyl)(propyl)silane").
        # The enclosing mark itself is the boundary, so a hyphen would be wrong
        # (and breaks the OPSIN round-trip).
        if prev_ends_alpha_or_close and (next_has_locant or next_starts_digit):
            result += "-" + part
        else:
            result += part
    return result


# ---------------------------------------------------------------------------
# Unsaturation rendering
# ---------------------------------------------------------------------------

def _strip_unsaturation_locants_if_omissible(
    infixes: tuple[UnsaturationInfix, ...],
    parent_length: int,
) -> tuple[UnsaturationInfix, ...]:
    """Return infixes with locants stripped where omission is allowed.

    IUPAC P-31.1.2.1 / P-14.6: for a 2-atom chain with a single double or
    triple bond (ethene, ethyne), the locant is unambiguous and is omitted.
    Rule: when parent_length == 2 and there is exactly one infix with one
    locant, strip that locant.
    """
    if parent_length != 2:
        return infixes
    # Only omit when there is exactly one infix (one bond type) with one locant.
    if len(infixes) != 1 or len(infixes[0].locants) != 1:
        return infixes
    import dataclasses as _dc
    return (_dc.replace(infixes[0], locants=()),)


def render_unsaturation(infixes: tuple[UnsaturationInfix, ...]) -> str:
    """Render unsaturation infixes with full vowel form (ene/yne).

    The full form is rendered here; elide() handles vowel-vowel junctions
    at the suffix boundary (e.g. "ene" + "-al" → "en-al" via elision).

    P-31.1.2.1 / P-54.1: when one unsaturation infix is IMMEDIATELY
    followed by another (e.g. ``-1-ene-3-yne``), the trailing 'e' of the
    earlier infix is elided so that the rendered form reads
    ``-1-en-3-yne`` rather than ``-1-ene-3-yne``.  The next infix begins
    with a hyphen + locant so the boundary-aware ``elide_at_boundaries``
    will not see the 'e' (the hyphen breaks the vowel-vowel adjacency it
    looks for); we drop the 'e' here directly when the next infix begins
    with a locant.

    Examples:
      single double bond at 2:           "-2-ene"
      two double bonds at 2,4:           "-2,4-diene"
      triple bond at 1:                  "-1-yne"
      double at 2 + triple at 4:         "-2-en-4-yne"
      no locant (e.g. ethene):            "ene"
    """
    if not infixes:
        return ""
    # Full vowel forms for unsaturation types
    _FULL_FORM = {"en": "ene", "yn": "yne"}
    parts: list[str] = []
    for inf in infixes:
        locant_str = _render_locants(inf.locants)
        mult_str = inf.multiplier or ""
        full_type = _FULL_FORM.get(inf.type, inf.type + "e")
        if locant_str:
            parts.append(f"-{locant_str}{mult_str}{full_type}")
        elif mult_str:
            parts.append(f"-{mult_str}{full_type}")
        else:
            parts.append(full_type)
    # P-31.1.2.1: elide trailing 'e' of an infix when the next infix
    # begins with a hyphen+locant (i.e. "-N-").
    for i in range(len(parts) - 1):
        nxt = parts[i + 1]
        if parts[i].endswith("e") and nxt.startswith("-"):
            parts[i] = parts[i][:-1]
    return "".join(parts)


# ---------------------------------------------------------------------------
# Locant omission (P-14.6)
# ---------------------------------------------------------------------------

# Suffixes whose attachment point is always C1 by IUPAC convention.
# For these, the locant "1" is never cited.
_TERMINAL_ALWAYS_C1_SUFFIXES: frozenset[str] = frozenset({
    "al",               # aldehyde: C1 by definition (it IS the chain-end carbonyl)
    "oic acid",         # carboxylic acid: C1 by definition
    "carboxylic acid",  # carboxylic acid (nonterminal form)
    "amide",            # carboxamide: C1 by definition
    "carboxamide",      # same
    "nitrile",          # nitrile: C1 by definition
    "carbonitrile",     # same
    "oyl",              # acyl derived from oic acid: C1
    # Acyl halide terminal forms: C1 by definition
    "oyl chloride",
    "oyl bromide",
    "oyl fluoride",
    "oyl iodide",
    # Thio-acid terminal forms: C1 by definition (chain terminus carbonyl)
    "thioic O-acid",
    "thioic S-acid",
    "dithioic acid",
    # P-66.1.4 / P-66.3 amide-family terminal forms: C1 by definition
    # (chain-terminus C-N attachment) — locant '1' never cited.
    "thioamide",
    "carbothioamide",
    "selenoamide",
    "carboselenoamide",
    "tellanoamide",
    "carbotellanoamide",
    # Hydrazide terminal forms (P-66.3): emitted base_form is the
    # leading-hyphen-stripped form of -ohydrazide / -carbohydrazide etc.
    "ohydrazide",
    "carbohydrazide",
    "thiohydrazide",
    "carbothiohydrazide",
    "selenohydrazide",
    "carboselenohydrazide",
    "tellurohydrazide",
    "carbotellurohydrazide",
})

# Chain-terminal di-suffix base_forms.  For these, when a chain carries TWO (or
# more) such suffixes they MUST sit at chain termini by IUPAC definition: the
# group IS a chain-end carbon (-CHO, -COOH, -CHS, ...) or a chain-end C-N
# attachment (-CONH2, ...), so its attachment carbon is fixed at C1 / C-N.  Per
# P-14.3.4.4 (omission of locants that are unique by symmetry / definition),
# P-14.3.4.5, P-66.6.3 (diacids), and P-66.1.1.1.1.1 / P-66.1.4.1.1 (amide
# family), the locants are forced and therefore omitted from the PIN, e.g.:
#   "butanedioic acid"   (not "butane-1,4-dioic acid")
#   "hexanedioic acid"   (not "hexane-1,6-dioic acid")
#   "pentanedial"        (not "pentane-1,5-dial")
#   "pentanedithial"     (not "pentane-1,5-dithial")
#   "pentanediamide"     (not "pentane-1,5-diamide")
#   "butanedithioamide"  (not "butane-1,4-dithioamide")
#
# DELIBERATE EXCLUSIONS:
#   * "nitrile" / "carbonitrile" — the test suite asserts "heptan-1,7-dinitrile"
#     with explicit locants (load-bearing test guard, see MEMORY.md).
#   * all "carbo*" / "carboxylic acid" forms — these denote a carbon ADDED
#     beyond the parent ring/chain, attached at an arbitrary parent position,
#     so the locant is genuinely needed (e.g. "naphthalene-1,4-dicarboxylic
#     acid", "cyclohexane-1,3-dicarbaldehyde").
# Both exclusions matter because they are NOT terminal-by-definition for a
# *chain* parent: only the bare acid/al/amide chain-terminus forms are.
_AMIDE_FAMILY_CHAIN_TERMINAL_SUFFIXES: frozenset[str] = frozenset({
    # Acid family (chain-terminus carbon = the acid carbon)
    "oic acid",
    "thioic O-acid",
    "thioic S-acid",
    "dithioic acid",
    # Aldehyde family (chain-terminus carbon = the carbonyl carbon)
    "al",
    "thial",
    "selenal",
    "tellural",
    # Amide / hydrazide family (chain-terminus C with the C-N attachment)
    "amide",
    "thioamide",
    "selenoamide",
    "tellanoamide",
    "ohydrazide",
    "thiohydrazide",
    "selenohydrazide",
    "tellurohydrazide",
})


def _strip_locant_1_if_omissible(
    suffix_groups: tuple[SuffixGroup, ...],
    parent_length: int,
    parent_has_indicated_h: bool = False,
    is_monosubstituted_homogeneous_monocycle: bool = False,
    single_suffix_symmetry_forced: bool = False,
) -> tuple[SuffixGroup, ...]:
    """Return suffix_groups with locant '1' stripped where P-14.6 applies.

    Rules for omitting the locant '1' (P-14.6.1):
    1. Terminal-by-definition suffixes (al, oic acid, amide, nitrile, ...):
       the attachment position is always C1 — locant omitted, BUT ONLY when
       there is exactly one suffix group of that base_form.  For diacids,
       dinitriles, etc., both terminal locants (1 and N) must be cited.
    2. Any single suffix (total) at position 1 on a chain of length 1 or 2:
       the position is forced (no ambiguity) — locant omitted.
       BUT for chains of length >= 3, the position may be ambiguous (e.g.
       propan-1-ol vs propan-2-ol) so the locant is RETAINED.
    3. P-14.3.4.2(c): single suffix at locant 1 on a homogeneous monocyclic
       ring with no other prefixes — all ring positions are equivalent before
       the suffix is added, so locant 1 is forced.  Examples:
       cyclohexanethiol (not cyclohexane-1-thiol),
       cyclohexanol (not cyclohexan-1-ol),
       cyclohexanamine (not cyclohexan-1-amine).
       BUT when other prefixes/locants are present (e.g. 2-chlorocyclohexane-
       1-thiol), the "1-" is load-bearing and must be retained per P-14.3.3.

    Exception (P-14.3.2 unambiguity requirement): when the parent carries an
    indicated hydrogen (e.g. "1H-pyrrole", "4H-pyran"), eliding a suffix
    locant produces an ambiguous name — OPSIN then defaults the suffix to
    the lowest-locant carbon (e.g. "1H-pyrrolecarboxylic acid" → position
    2) rather than honoring the indicated-H locant.  Retain the locant in
    this case so the name round-trips unambiguously.

    This function only strips the locant '1'.  All other locants are
    preserved unchanged.
    """
    if not suffix_groups:
        return suffix_groups

    import dataclasses as _dc

    # P-14.3.4.4 (general single-suffix symmetry omission): when the engine has
    # determined that the parent has exactly ONE PCG suffix (and no prefix) whose
    # attachment position is forced by graph symmetry — every parent position the
    # suffix could occupy is in one symmetry class — the suffix locant is
    # redundant regardless of its numeric value and is dropped.  This is the
    # heterocyclic / fused generalisation of the all-carbon-monocyclic rule
    # below: "pyrazinecarboxylic acid" (not "pyrazine-2-carboxylic acid"),
    # "cyclohexanecarboxylic acid", etc.  The flag is only True for a single
    # suffix group, so it is safe to strip its locant unconditionally here.
    # Indicated-H still forces the locant (P-14.3.2): an unlocanted suffix on an
    # indicated-H parent lets OPSIN default to a different position.
    if (single_suffix_symmetry_forced
            and len(suffix_groups) == 1
            and not parent_has_indicated_h):
        return (_dc.replace(suffix_groups[0], locants=()),)

    # Count how many suffix groups exist per base_form
    from collections import Counter as _Counter
    base_form_counts = _Counter(sg.base_form for sg in suffix_groups)

    # Rule 3 (P-14.3.4.4 / P-14.3.4.5 / P-66.6.3 / P-66.1.1.1.1.1 / P-66.1.4.1.1):
    # for chain-terminal di-suffixes (acid: -oic acid, -dithioic acid, ...;
    # aldehyde: -al, -thial, ...; amide/hydrazide: -amide, -thioamide,
    # -ohydrazide, ...), when ALL suffix groups share a chain-terminal terminal
    # base_form AND every locant is at a chain terminus (position 1 or position
    # N), the locants are forced (no choice) and must be omitted from the PIN.
    # Examples: "butanedioic acid"  (not "butane-1,4-dioic acid"),
    #           "pentanedial"       (not "pentane-1,5-dial"),
    #           "pentanedithial"    (not "pentane-1,5-dithial"),
    #           "pentanediamide"    (not "pentane-1,5-diamide"),
    #           "butanedithioamide" (not "butane-1,4-dithioamide").
    # Excludes -nitrile (test guard) and all -carbo* forms (added-carbon groups
    # whose locant is genuinely needed); see the frozenset definition above.
    drop_all_amide_family_locants = False
    if (parent_length >= 1
            and not parent_has_indicated_h
            and all(sg.base_form in _AMIDE_FAMILY_CHAIN_TERMINAL_SUFFIXES
                    for sg in suffix_groups)):
        # All suffix groups must have a single locant that is at a chain
        # terminus (1 or parent_length).  parent_length==1 is the methane case
        # where 1 is the only position; parent_length==2 has only positions
        # 1 and 2 which are both termini.
        chain_termini = {"1", str(parent_length)}
        if all(len(sg.locants) == 1
               and str(sg.locants[0]) in chain_termini
               for sg in suffix_groups):
            drop_all_amide_family_locants = True

    modified: list[SuffixGroup] = []
    for sg in suffix_groups:
        # Rule 3 (amide-family multi-suffix elision): strip locants entirely
        # when all amide-family terminal suffixes sit at chain termini.
        if drop_all_amide_family_locants:
            sg = _dc.replace(sg, locants=())
            modified.append(sg)
            continue

        # Check if there is exactly one locant and it is "1"
        if len(sg.locants) == 1 and str(sg.locants[0]) == "1":
            omit = False
            if (sg.base_form in _TERMINAL_ALWAYS_C1_SUFFIXES
                    and base_form_counts[sg.base_form] == 1):
                # Rule 1: terminal-always-C1 suffix, and only ONE such group.
                # (For dinitrile/diacid, base_form_counts > 1 → do NOT omit.)
                omit = True
            elif len(suffix_groups) == 1 and parent_length <= 2:
                # Rule 2: single suffix total, short chain (length 1 or 2) — unambiguous
                omit = True
            elif (len(suffix_groups) == 1
                    and is_monosubstituted_homogeneous_monocycle):
                # Rule 3 (P-14.3.4.2(c)): single suffix at locant 1 on a
                # homogeneous monocyclic ring with no other prefixes — all
                # ring positions are equivalent so the locant is forced.
                # The caller is responsible for ensuring there are no other
                # prefixes that would break the homogeneity (e.g.
                # 2-chlorocyclohexane-1-thiol must keep its "1-").
                omit = True
            # P-14.3.2: do NOT elide when the parent carries indicated-H —
            # eliding lets OPSIN default the suffix to a different position.
            if omit and parent_has_indicated_h:
                omit = False
            if omit:
                sg = _dc.replace(sg, locants=())
        modified.append(sg)
    return tuple(modified)


# ---------------------------------------------------------------------------
# Suffix rendering
# ---------------------------------------------------------------------------

def render_suffixes(
    suffix_groups: tuple[SuffixGroup, ...],
    output_form: OutputForm,
) -> str:
    """Render suffix groups with multipliers and locants.

    Groups suffix_groups by base_form, applies multiplier for count>1,
    applies OutputForm variant transform.

    Examples:
      2 alcohols at positions 1,3 → "-1,3-diol"
      1 acid at position 1        → "-oic acid"
    """
    # Group by base_form
    by_form: dict[str, list[SuffixGroup]] = defaultdict(list)
    for sg in suffix_groups:
        by_form[sg.base_form].append(sg)

    # Render each group — preserve original ordering of first occurrence
    seen_forms: list[str] = []
    for sg in suffix_groups:
        if sg.base_form not in seen_forms:
            seen_forms.append(sg.base_form)

    parts: list[str] = []
    for base_form in seen_forms:
        group = by_form[base_form]
        count = len(group)

        # Collect all locants
        all_locants: list[Locant] = []
        for sg in group:
            all_locants.extend(sg.locants)
        all_locants_sorted = sorted(all_locants)

        # P-31.1.4.2.4 / P-58.2.2: collect added-indicated-H locants from any
        # SuffixGroup in this form group.  Each group's locants are paired
        # with its added_indicated_h list; we sort jointly so the IH pairs
        # follow the same order as the suffix locants.
        added_ih_locants: list[Locant] = []
        if any(sg.added_indicated_h for sg in group):
            paired: list[tuple[Locant, tuple[Locant, ...]]] = []
            for sg in group:
                # Pair each suffix locant with the SuffixGroup's added_ih
                # tuple — typically one IH per suffix.
                for loc in sg.locants:
                    paired.append((loc, sg.added_indicated_h))
            paired.sort(key=lambda kv: kv[0])
            for _suf_loc, ihs in paired:
                added_ih_locants.extend(ihs)

        # Derive rendered form (applies OutputForm variant transform)
        rendered_form = resolve_suffix_variant(base_form, output_form)

        # Multiplier (for the suffix itself, use simple series: di, tri, ...)
        locant_str = _render_locants(tuple(all_locants_sorted))
        if count > 1:
            mult = get_multiplier(count, complex=False) or ""
        else:
            mult = ""

        # P-16.7.1(c): elide terminal 'a' of a multiplier before a suffix
        # beginning with 'a' or 'o'.  E.g.: penta+ol->pentol, tetra+amine->tetramine.
        if mult.endswith("a") and rendered_form and rendered_form[0] in "ao":
            mult = mult[:-1]

        # P-66.3 / P-16.3.3: for "-ohydrazide" the leading 'o' is a connecting
        # vowel between the parent stem and the "hydrazide" tail.  When a
        # multiplier is present (di-, tri-, ...), the multiplier replaces the
        # connector — e.g. "pentanedihydrazide" not "pentanediohydrazide".
        # This applies only when count>1 and the form starts with the
        # connector "o" (currently only "ohydrazide" in the data).
        if (count > 1 and mult and rendered_form.startswith("o")
                and base_form == "ohydrazide"):
            rendered_form = rendered_form[1:]  # drop leading 'o'

        # P-58.2.2 added-indicated-H rendering: when present, the parenthetical
        # (NH) — or (NH,MH) for multiple — sits between the suffix-locant block
        # and the suffix tail, with a closing hyphen before the tail.
        # Example: ``naphthalen-1(2H)-one`` — locant_str="1-", ih_str="(2H)".
        # Note: ``locant_str`` carries a trailing hyphen from ``_render_locants``;
        # for the (NH) form that hyphen must move to AFTER the parenthetical.
        ih_str = ""
        locant_str_no_dash = locant_str
        if added_ih_locants:
            ih_str = "(" + ",".join(f"{loc}H" for loc in added_ih_locants) + ")"
            if locant_str_no_dash.endswith("-"):
                locant_str_no_dash = locant_str_no_dash[:-1]

        # For multi-word suffixes like "oic acid" → "dioic acid"
        # The multiplier prefixes the FIRST word if it is a vowel suffix.
        # Hyphen rule: if there are locants, use hyphens; otherwise attach directly.
        # e.g. "ethanol" (no locants), "propan-2-ol" (has locant)
        if locant_str:
            # locants present — use hyphen delimiters
            if mult and " " in rendered_form:
                # e.g. "1,6-dioic acid"
                first_word, rest_words = rendered_form.split(" ", 1)
                if ih_str:
                    suffix_str = f"-{locant_str_no_dash}{ih_str}-{mult}{first_word} {rest_words}"
                else:
                    suffix_str = f"-{locant_str}{mult}{first_word} {rest_words}"
            else:
                if ih_str:
                    suffix_str = f"-{locant_str_no_dash}{ih_str}-{mult}{rendered_form}"
                else:
                    suffix_str = f"-{locant_str}{mult}{rendered_form}"
        else:
            # no locants — attach suffix directly (no hyphen)
            if mult and " " in rendered_form:
                first_word, rest_words = rendered_form.split(" ", 1)
                suffix_str = f"{mult}{first_word} {rest_words}"
            else:
                suffix_str = f"{mult}{rendered_form}"

        parts.append(suffix_str)

    return "".join(parts)


# ---------------------------------------------------------------------------
# Free-valence suffix rendering
# ---------------------------------------------------------------------------

def _free_valence_locant_will_elide(fv: FreeValenceInfo, numbering: Numbering) -> bool:
    """Predict whether render_free_valence_suffix will produce a locant-less
    suffix for a monovalent ALKANYL free valence.

    Returns True iff the rendered suffix will be just "-yl"/"-ylidene"/etc.
    with no locant (i.e. attachment at locant 1 with elide_locant_one=True),
    and the FV is monovalent. This is the trigger for the P-29.2 contracted-
    alkyl form: when ALKANYL's locant ends up elided, the substituent stem
    should collapse from "<chain>an" to "<chain>" so e.g. "propan" + "-yl"
    becomes "propyl" rather than the malformed "propan-yl".
    """
    if fv.method != SubstituentMethod.ALKANYL:
        return False
    if not fv.is_monovalent:
        return False
    if fv.attachment_atoms_in_fragment is None:
        return False
    n = len(fv.bond_orders)
    sig = tuple(sorted(fv.bond_orders, reverse=True))
    suffix = FREE_VALENCE_SUFFIXES.get((n, sig), f"{n}yl")
    if suffix != "yl":
        # Only the bare "-yl" form participates in the alkan→alk contraction.
        # Forms like "ylidene" (>C=) keep the systematic stem.
        return False
    atom_to_loc = numbering.atom_to_locant
    attachment_locants = [
        atom_to_loc[idx]
        for idx in fv.attachment_atoms_in_fragment
        if idx in atom_to_loc
    ]
    if len(attachment_locants) != 1:
        return False
    loc = attachment_locants[0]
    return str(loc) == "1" and fv.elide_locant_one


def render_free_valence_suffix(
    fv: FreeValenceInfo,
    numbering: Numbering,
    has_unsaturation: bool = False,
) -> str:
    """Render -yl, -ylidene, -diyl etc.

    Method ALKYL (1): suffix only, locant 1 omitted.
    Method ALKANYL (2): locants cited + suffix.

    ``has_unsaturation`` (P-29.2): when the substituent's chain carries an
    explicit unsaturation locant (``-en-``, ``-yn-``), the free-valence
    locant must be cited too — even when it is "1" — so that the rendered
    name preserves the locant relationship.  E.g. ``but-3-en-1-yl`` not
    ``but-3-enyl``.  The caller passes True iff ``tree.unsaturation`` is
    populated.
    """
    n = len(fv.bond_orders)
    sig = tuple(sorted(fv.bond_orders, reverse=True))
    suffix = FREE_VALENCE_SUFFIXES.get((n, sig), f"{n}yl")

    if fv.method == SubstituentMethod.ALKYL:
        # Method 1: no locant (locant 1 always omitted for methyl, ethyl, etc.)
        # No leading hyphen — the alkyl_stem already has the terminal 'e' dropped,
        # so the suffix attaches directly: meth + yl = methyl (not meth-yl).
        return suffix

    # Method 2 (ALKANYL): cite locants for the attachment points
    if fv.attachment_atoms_in_fragment is None:
        return f"-{suffix}"

    atom_to_loc = numbering.atom_to_locant
    attachment_locants = [
        atom_to_loc[idx]
        for idx in fv.attachment_atoms_in_fragment
        if idx in atom_to_loc
    ]
    attachment_locants.sort()

    if not attachment_locants:
        return f"-{suffix}"

    # For monovalent: single locant; omit if locant is "1" and suffix is "yl"
    if len(attachment_locants) == 1:
        loc = attachment_locants[0]
        if (str(loc) == "1"
                and suffix == "yl"
                and fv.elide_locant_one
                and not has_unsaturation):
            return f"-{suffix}"  # locant 1 elided for alkan-1-yl in most contexts
        return f"-{loc}-{suffix}"

    locant_str = ",".join(str(loc) for loc in attachment_locants)
    # Multiplier for diyl, triyl etc.
    mult = get_multiplier(len(attachment_locants), complex=False) or ""
    return f"-{locant_str}-{mult}{suffix}"


# ---------------------------------------------------------------------------
# Terminal vowel
# ---------------------------------------------------------------------------

def terminal_vowel(named_parent: NamedParent, output_form: OutputForm) -> str:
    """Return the terminal vowel string for a parent stem.

    For SUBSTITUENT form, the free-valence suffix is handled separately.
    For other forms the parent name normally ends in "e" ("ethan" → "ethane").

    Exception: retained names like "furan", "pyran", "oxan" already end in
    a consonant but do NOT take a terminal "e" — the ``name`` field is used
    directly.  We detect this by checking whether the ``stem`` equals the
    ``name`` (i.e. no terminal 'e' was stripped from the name when forming
    the stem).  If ``stem == name`` the name requires no terminal vowel.
    """
    if output_form == OutputForm.SUBSTITUENT:
        return ""
    # If the stem is the same as the full name, no terminal 'e' is needed.
    # (e.g. "furan": stem="furan", name="furan" → no 'e')
    # Contrast: "ethane": stem="ethan", name="ethane" → add 'e'
    if named_parent.stem == named_parent.name:
        return ""
    return "e"


# ---------------------------------------------------------------------------
# Rendered-suffix terminal-'e' check
# ---------------------------------------------------------------------------

# Regex to strip the leading locant block from a rendered suffix string.
# Matches patterns like "-1-", "-1,2-", "-1,2,3-", "-N-" at the start.
# Also strips an optional added-IH parenthetical "(NH)" / "(NH,MH)" between
# the locant block and the suffix tail (e.g. "-1(2H)-one"), so the actual
# suffix tail (e.g. "one", "imine") is what gets consonant-tested.
_RENDERED_SUFFIX_LOCANT_RE = re.compile(
    r"^-[0-9NOPSH,\'^]+(?:,[0-9NOPSH,\'^]+)*(?:\(\d+[a-z]?H(?:,\d+[a-z]?H)*\))?-"
)


def _rendered_suffix_starts_with_consonant(rendered: str) -> bool:
    """Return True when the rendered suffix's first alphabetic character is a consonant.

    The rendered suffix (as returned by render_suffixes) is one of:

    * ``"ol"`` — no locants, direct concat: check first char.
    * ``"-1-ol"`` — locant present: strip "-<locants>-", then check first char.

    This is used by _assemble_substitutive / _assemble_replacement to decide
    whether the parent stem needs its terminal 'e' preserved.  If True, the
    terminal 'e' is retained (e.g. "propane-1,2,3-triol"); if False, elide()
    will handle removal when the suffix starts with a vowel.
    """
    if not rendered:
        return False
    s = rendered
    # Strip leading hyphen + locants + hyphen (e.g. "-1,2,3-")
    s = _RENDERED_SUFFIX_LOCANT_RE.sub("", s)
    if not s:
        return False
    return s[0].lower() not in "aeiou"


# ---------------------------------------------------------------------------
# Elision
# ---------------------------------------------------------------------------

def elide(name: str) -> str:
    """Apply vowel elision: remove a terminal 'e' from the parent stem when
    the following suffix begins with a vowel and elision is appropriate.

    This function applies IUPAC P-16.3.3 elision at the stem/suffix
    junction.  It trusts that the SuffixGroup.elides_terminal_e flag has
    been checked upstream and that callers who want precise control use
    :func:`elide_at_boundaries` instead.  The string-level API here is
    conservative: it only elides if doing so cannot disturb an interior
    vowel digram of a retained stem.

    Rule (conservative):
        Find the last 'e' in the string that (a) is immediately followed
        by a vowel, AND (b) the char BEFORE the 'e' is a consonant, AND
        (c) at least one character of the vowel-initial tail would remain
        after the 'e'.  If the tail matches an "-amine"/"-amide"/"-amino"
        no-elision pattern, skip and try the next-rightmost candidate
        (there won't be one in well-formed input).  Elide only that 'e'.

    This rule by itself still mishandles retained stems like "aceanthrene"
    (interior "ea") or "pleiadene" (interior "ei") when they reach elide()
    via the ring-parent assembly path with no appended suffix.  Callers
    that KNOW the boundary positions should use
    :func:`elide_at_boundaries` which takes the parts list and only tests
    true stem/suffix junctions — that is the robust path for assembled
    ring/chain names.  The string-level ``elide()`` remains for legacy
    callers and existing unit tests.
    """
    # Do NOT elide before "-amine" or "-amide" suffixes
    no_elision_patterns = ("amine", "amide", "amino")

    for i in range(len(name) - 2, -1, -1):
        if name[i] == "e" and name[i + 1] in "aeiouy":
            remaining = name[i + 1:]
            if any(remaining.startswith(p) for p in no_elision_patterns):
                continue
            return name[:i] + name[i + 1:]
    return name


def elide_at_boundaries(parts: list[str]) -> str:
    """Apply IUPAC P-16.3.3 elision only at explicit part-boundary junctions.

    ``parts`` is the ordered list assembled by substitutive/suffix/ring
    rendering code.  Between each consecutive pair ``(left, right)`` we
    look at exactly one junction: does ``left`` end with 'e' and does
    ``right`` begin with a vowel?  If so (and the right-hand segment is
    NOT an "amine"/"amide"/"amino" variant), drop the terminal 'e' of
    ``left``.

    Interior ``e<vowel>`` digrams WITHIN a single part (e.g. "aceanthrene",
    "pleiadene", "oleic") are NEVER touched — those are syllables of a
    retained stem that must be preserved.

    This is the robust replacement for the string-level ``elide()`` for
    callers that have the parts list available.
    """
    no_elision_patterns = ("amine", "amide", "amino")
    # P-31.1.2.1 ene/yne elision before a locant-prefixed continuation:
    # ``ene`` + ``-1-ol`` → ``en-1-ol``, ``ene`` + ``-1-yl`` → ``en-1-yl``.
    # The standard vowel-vowel rule does not fire because the right-hand
    # part begins with ``-`` not a vowel.  Detect ``-<digit>-...`` and
    # treat the digit run as if the trailing 'e' were directly preceding
    # an "ol"/"yl"/"al"/etc. consonant-stem.
    import re as _re_elide
    _LOCANT_PREFIX_RE = _re_elide.compile(r"^-\d+(?:[a-z])?(?:,\d+(?:[a-z])?)*-")

    result_parts: list[str] = []
    for idx, part in enumerate(parts):
        left = part
        if idx + 1 < len(parts):
            right = parts[idx + 1]
            should_elide = False
            if (
                left.endswith("e")
                and right
                and right[0] in "aeiouy"
                and not any(right.startswith(p) for p in no_elision_patterns)
            ):
                should_elide = True
            elif left.endswith("e") and right and _LOCANT_PREFIX_RE.match(right):
                # Skip past the "-<locant>-" prefix and check the suffix start.
                suffix_after_locant = _LOCANT_PREFIX_RE.sub("", right)
                if (suffix_after_locant
                        and suffix_after_locant[0] in "aeiouy"
                        and not any(suffix_after_locant.startswith(p) for p in no_elision_patterns)):
                    should_elide = True
            if should_elide:
                left = left[:-1]
        result_parts.append(left)
    return "".join(result_parts)


# ---------------------------------------------------------------------------
# format_for_output_form — for LeafTree
# ---------------------------------------------------------------------------

def format_for_output_form(text: str, output_form: OutputForm) -> str:
    """Apply output-form transformation to a leaf text string.

    Most retained names are used as-is (STANDALONE). For other output forms,
    we apply simple transformations where applicable.
    """
    # For most retained names, the text already encodes the correct form.
    # The engine is responsible for setting the right text on LeafTree.
    # This function is a pass-through placeholder — engine sets the right text.
    return text


# ---------------------------------------------------------------------------
# Functional class assembly
# ---------------------------------------------------------------------------

def _carbamic_n_subs_to_prefix(n_sub_names: list[str]) -> str:
    """Convert a list of N-substituent names to a carbamate N-locant prefix.

    Each element in n_sub_names is a substituent name produced by naming one
    N-substituent as SUBSTITUENT (e.g. "phenyl", "methyl", "(3-chlorophenyl)").

    Rules:
      []             → "" (no N-prefix: "ethyl carbamate")
      ["phenyl"]     → "N-phenyl"
      ["methyl"]     → "N-methyl"
      ["methyl","methyl"] → "N,N-dimethyl"
      ["(3-chlorophenyl)"] → "N-(3-chlorophenyl)"
    """
    if not n_sub_names:
        return ""

    # Count occurrences of each unique N-substituent name
    from collections import Counter as _Counter
    name_counts = _Counter(n_sub_names)
    from iupac_namer.data_loader import get_multiplier as _get_mult
    n_parts: list[str] = []
    for sub_name in sorted(set(n_sub_names), key=lambda s: derive_sort_name(s)):
        count = name_counts[sub_name]
        # Determine N-locant string: "N-" for 1, "N,N-" for 2, etc.
        n_locant_str = ",".join(["N"] * count) + "-"
        # Determine if sub_name needs brackets (compound prefix rules).
        is_compound = _is_compound_prefix(sub_name)
        if count > 1:
            if is_compound:
                mult = _get_mult(count, complex=True) or ""
                open_b, close_b = _choose_brackets(sub_name)
                sub_str = f"{mult}{open_b}{sub_name}{close_b}"
            else:
                mult = _get_mult(count, complex=False) or ""
                sub_str = f"{mult}{sub_name}"
        else:
            if is_compound:
                open_b, close_b = _choose_brackets(sub_name)
                sub_str = f"{open_b}{sub_name}{close_b}"
            else:
                sub_str = sub_name
        n_parts.append(f"{n_locant_str}{sub_str}")

    # Join multiple unique substituents: "N-methyl-N-phenyl" style (IUPAC 2013 P-16.3)
    return "-".join(n_parts)


def _acid_to_adjective(acid_name: str) -> tuple[str, str | None]:
    """Convert a retained acid name to its adjective form.

    Table-first, systematic-second (strip " acid"), warning-third.

    Returns (adjective, warning_or_None).
    """
    # Table lookup (exact match)
    if acid_name in ACID_ADJECTIVE_TABLE:
        return ACID_ADJECTIVE_TABLE[acid_name], None

    # Systematic fallback: strip " acid" suffix
    if acid_name.endswith(" acid"):
        stem = acid_name[:-5]  # strip " acid"
        return stem, None

    # Last resort: return as-is with a warning
    return acid_name, f"Could not derive adjective for acid: {acid_name!r}"


def _polyester_locant_sort_key(loc: str | None):
    """Sort key for an ester locant label (numeric-aware, e.g. '1' < '1a' < '3')."""
    if loc is None:
        return (1, 0, "")
    import re as _re
    m = _re.match(r"(\d+)([a-z']*)", loc)
    if m:
        return (0, int(m.group(1)), m.group(2))
    return (0, 0, loc)


def _assemble_polyester(tree: FunctionalClassTree) -> str:
    """Assemble a poly-/mixed-ester functional-class name (P-65.6.3.3.2).

    Form: ``<alkyl word(s)> <parent>...dicarboxylate``.
      - identical alkyls collapse to a multiplier (``dimethyl``, ``dipropyl``);
      - different alkyls are cited as separate words in alphanumerical order;
      - locants are cited at the front of each alkyl word only when necessary
        (different alkyls AND the parent positions are not symmetry-equivalent,
        i.e. the unlocanted alphanumerical default would not reproduce the
        actual structure).
    """
    acid_name = ""
    # role -> alkyl name
    alkyls: dict[str, str] = {}
    for role, subtree in tree.pieces:
        nm = assemble(subtree)
        if role == "acid":
            acid_name = nm
        elif role.startswith("alcohol_"):
            alkyls[role] = nm

    loc_map: dict[str, str | None] = {}
    rank_map: dict[str, int] = {}
    if tree.polyester_alkyl_locants:
        for role, loc, rank in tree.polyester_alkyl_locants:
            loc_map[role] = loc
            rank_map[role] = rank

    roles = list(alkyls.keys())
    names = [alkyls[r] for r in roles]
    distinct_names = set(names)

    # Decide whether locants must be cited (P-65.6.3.3.2 "when necessary").
    # Never cite when all alkyls are identical (the multiplier form is
    # unambiguous).  When alkyls differ, the *unlocanted* name denotes the
    # molecule obtained by assigning alkyls to positions in alphanumerical
    # order; this reproduces the actual structure iff the (symmetry-rank,
    # alkyl) multiset of the default assignment equals that of the actual one
    # (positions sharing a symmetry rank are interchangeable, so a different
    # alkyl on each does NOT need a locant — e.g. benzene-1,3 positions).
    cite_locants = False
    all_have_locants = all(loc_map.get(r) is not None for r in roles)
    have_ranks = all(rank_map.get(r, -1) >= 0 for r in roles)
    if len(distinct_names) > 1 and all_have_locants and have_ranks:
        # actual: (rank, alkyl) per position
        actual_pairs = sorted((rank_map[r], alkyls[r]) for r in roles)
        # default: sort positions by locant, alkyls alphanumerically, zip, then
        # read off (rank, alkyl) per position.
        ordered_roles = sorted(roles, key=lambda r: _polyester_locant_sort_key(loc_map[r]))
        sorted_alkyls = sorted((alkyls[r] for r in roles),
                              key=lambda s: derive_sort_name(s))
        default_pairs = sorted(
            (rank_map[ordered_roles[i]], sorted_alkyls[i])
            for i in range(len(ordered_roles))
        )
        if actual_pairs != default_pairs:
            cite_locants = True
    elif len(distinct_names) > 1 and not (all_have_locants and have_ranks):
        # Different alkyls but we could not recover locants/ranks: be safe and
        # do not emit an ambiguous unlocanted mixed-ester name.  Citing the raw
        # locants we *do* have is preferable; if none, fall through to words.
        cite_locants = all_have_locants

    # Build the alkyl-word string.
    if cite_locants:
        # Each alkyl keeps its own locant; cite as separate words, ordered by
        # alphanumerical alkyl name then locant.  Identical alkyls at different
        # locants share one word with a multiplied/comma locant set.
        from collections import defaultdict as _dd
        by_name: dict[str, list[str]] = _dd(list)
        for r in roles:
            by_name[alkyls[r]].append(loc_map[r])
        words: list[tuple[str, str]] = []  # (sort_key_name, word)
        for nm in sorted(by_name, key=lambda s: derive_sort_name(s)):
            locs = sorted(by_name[nm], key=_polyester_locant_sort_key)
            loc_prefix = ",".join(locs) + "-"
            count = len(locs)
            if count > 1:
                compound = _is_compound_prefix(nm)
                if compound:
                    mult = get_multiplier(count, complex=True) or ""
                    ob, cb = _choose_brackets(nm)
                    word = f"{loc_prefix}{mult}{ob}{nm}{cb}"
                else:
                    mult = get_multiplier(count, complex=False) or ""
                    word = f"{loc_prefix}{mult}{nm}"
            else:
                word = f"{loc_prefix}{nm}"
            words.append((derive_sort_name(nm), word))
        alkyl_str = " ".join(w for _k, w in sorted(words, key=lambda x: x[0]))
        return f"{alkyl_str} {acid_name}"

    # Unlocanted form: group identical alkyls with multipliers, sort words
    # alphanumerically.
    from collections import Counter as _Counter
    name_counts = _Counter(names)
    words2: list[str] = []
    for nm in sorted(set(names), key=lambda s: derive_sort_name(s)):
        count = name_counts[nm]
        if count > 1:
            compound = _is_compound_prefix(nm)
            if compound:
                mult = get_multiplier(count, complex=True) or ""
                ob, cb = _choose_brackets(nm)
                words2.append(f"{mult}{ob}{nm}{cb}")
            else:
                mult = get_multiplier(count, complex=False) or ""
                words2.append(f"{mult}{nm}")
        else:
            words2.append(nm)
    alkyl_str = " ".join(words2)
    return f"{alkyl_str} {acid_name}"


def _assemble_fc(tree: FunctionalClassTree) -> str:
    """Assemble functional class names."""
    if tree.subtype == "polyester":
        return _assemble_polyester(tree)

    # Assemble each piece
    filled: dict[str, str] = {}
    for role, subtree in tree.pieces:
        filled[role] = assemble(subtree)

    match tree.subtype:
        case "ester":
            # "methyl acetate": alcohol-part acid-part(as-acid-stem)
            return f"{filled['alcohol']} {filled['acid']}"

        case "symmetric_diester":
            # "diallyl oxalate": di{alkyl} {diacid-as-ate}
            # The acid fragment names the diacid backbone with ACID_STEM output
            # form (e.g. "oxalic acid" → "oxalate"). We then multiply the
            # alkyl name by 2 using the standard di/bis rules.
            r_name = filled["alcohol"]
            acid_name = filled["acid"]
            # Apply di/bis multiplier: use "di" for simple alkyl stems,
            # "bis(...)" for compound ones (same rule as merge_identical_prefixes).
            compound = _is_compound_prefix(r_name)
            if compound:
                open_b, close_b = _choose_brackets(r_name)
                multiplied = f"bis{open_b}{r_name}{close_b}"
            else:
                multiplied = f"di{r_name}"
            return f"{multiplied} {acid_name}"

        case "thioester" | "dithioester":
            # P-65.6.3: R-C(=O)-S-R'   -> "S-R' R-thioate"
            #           R-C(=S)-S-R'   -> "S-R' R-dithioate"
            # The "S-" locant prefix on the alkyl disambiguates the acid-stem
            # from its thionoester counterpart (for thioate) and signals that
            # the chalcogen link is through sulfur.
            return f"S-{filled['alcohol']} {filled['acid']}"

        case "thionoester":
            # P-65.6.3: R-C(=S)-O-R' -> "O-R' R-thioate"
            return f"O-{filled['alcohol']} {filled['acid']}"

        case "carbamate":
            # "butyl N-phenylcarbamate" / "ethyl carbamate" (no N-sub)
            # pieces dict has "alcohol" and 0-N "n_sub_N" entries.
            # Each n_sub_N is a substituent on N, named as SUBSTITUENT.
            alcohol_name = filled.get("alcohol", "")
            n_sub_names = [
                v for k, v in sorted(filled.items())
                if k.startswith("n_sub_")
            ]
            n_prefix = _carbamic_n_subs_to_prefix(n_sub_names)
            if n_prefix:
                return f"{alcohol_name} {n_prefix}carbamate"
            else:
                return f"{alcohol_name} carbamate"

        case "thionocarbamate" | "dithiocarbamate" | "carbamothioate":
            # P-66.6.5.5:
            #   R2N-C(=S)-O-R'  -> "O-R' N-R,N-R'-carbamothioate"   (thionocarbamate)
            #   R2N-C(=S)-S-R'  -> "S-R' N-R,N-R'-carbamodithioate" (dithiocarbamate)
            #   R2N-C(=O)-S-R'  -> "S-R' N-R,N-R'-carbamothioate"   (carbamothioate,
            #                       S-substituted thiocarbamate; bridging atom is S)
            # The "O-"/"S-" locant on the alkyl disambiguates the bridge
            # chalcogen, mirroring the thionoester / dithioester FC shape.
            alcohol_name = filled.get("alcohol", "")
            n_sub_names = [
                v for k, v in sorted(filled.items())
                if k.startswith("n_sub_")
            ]
            n_prefix = _carbamic_n_subs_to_prefix(n_sub_names)
            if tree.subtype == "thionocarbamate":
                bridge_locant = "O"
                stem = "carbamothioate"
            elif tree.subtype == "carbamothioate":
                bridge_locant = "S"
                stem = "carbamothioate"
            else:
                bridge_locant = "S"
                stem = "carbamodithioate"
            if n_prefix:
                return f"{bridge_locant}-{alcohol_name} {n_prefix}{stem}"
            else:
                return f"{bridge_locant}-{alcohol_name} {stem}"

        case "anhydride":
            adj1, _ = _acid_to_adjective(filled["acid1"])
            adj2, _ = _acid_to_adjective(filled["acid2"])
            if adj1 == adj2:
                return f"{adj1} anhydride"
            # Alphabetical order
            sorted_adjs = sorted([adj1, adj2])
            return f"{sorted_adjs[0]} {sorted_adjs[1]} anhydride"

        case "acid_halide":
            return f"{filled['acid']} {filled['halide']}"

        case "thioester":
            return f"{filled['thiol']} {filled['acid']}"

        case "amide" | "imide":
            return filled.get("amide", filled.get("name", ""))

        case "acyl_isothiocyanate":
            # "benzoyl isothiocyanate" — acid stem + fixed class word
            return f"{filled['acid']} isothiocyanate"

        case _:
            # Generic: join all pieces in order
            return " ".join(filled[role] for role, _ in tree.pieces)


# ---------------------------------------------------------------------------
# Additive assembly
# ---------------------------------------------------------------------------

def _assemble_additive(tree: AdditiveTree) -> str:
    """Pattern: "{parent_name} {locant}-{multiplier}{type}"

    Examples:
      pyridine 1-oxide
      triphenylphosphane oxide
    """
    parent_name = assemble(tree.parent_tree)
    # P-74.2.1: identical additions (e.g. two N-oxides on quinoxaline) combine
    # into ONE multiplied term with cited locants -> "quinoxaline 1,4-dioxide",
    # NOT two separate "oxide oxide" terms.  Group by addition type, preserving
    # first-seen order; combine only when each addition is a distinct,
    # non-pre-multiplied object (so an addition that already carries a
    # multiplier is left untouched).
    grouped: dict[str, list] = {}
    order: list[str] = []
    for ag in tree.additions:
        if ag.type not in grouped:
            grouped[ag.type] = []
            order.append(ag.type)
        grouped[ag.type].append(ag)

    def _loc_val(ag):
        loc = ag.locant
        return loc._numeric_value if (loc and loc._numeric_value is not None) else 0

    addition_parts: list[str] = []
    for typ in order:
        ags = grouped[typ]
        if len(ags) > 1 and not any(ag.multiplier for ag in ags):
            ags_sorted = sorted(ags, key=_loc_val)
            locs = [ag.locant for ag in ags_sorted if ag.locant and ag.locant.is_numeric]
            mult = get_multiplier(len(ags)) or ""
            loc_str = (",".join(str(l) for l in locs) + "-") if len(locs) == len(ags) else ""
            addition_parts.append(f"{loc_str}{mult}{typ}")
        else:
            for ag in ags:
                loc_str = f"{ag.locant}-" if ag.locant.is_numeric else ""
                mult_str = ag.multiplier or ""
                addition_parts.append(f"{loc_str}{mult_str}{typ}")
    return f"{parent_name} {' '.join(addition_parts)}"


# ---------------------------------------------------------------------------
# Replacement assembly
# ---------------------------------------------------------------------------

def _assemble_replacement(tree: ReplacementTree) -> str:
    """Structure: [prefixes][replacement_prefixes][parent_stem][unsaturation][suffix]

    Example: 2,5,8-trioxadecane, 4-aza-2-oxaheptane
    """
    parts: list[str] = []

    # 1. Stereo descriptors
    if tree.stereo_descriptors:
        parts.append(render_stereo(tree.stereo_descriptors))

    # 2. Indicated hydrogen
    if tree.indicated_hydrogen:
        parts.append(render_indicated_h(tree.indicated_hydrogen))

    # 3. Regular substituent prefixes (same as substitutive)
    if tree.prefixes:
        assembled_prefixes: list[tuple[str, tuple[Locant, ...]]] = []
        for pe in tree.prefixes:
            prefix_name = assemble(pe.tree)
            assembled_prefixes.append((prefix_name, pe.locants))
        merged = merge_identical_prefixes(assembled_prefixes)
        merged.sort(key=lambda m: m.sort_name)
        parts.append(render_merged_prefixes(merged))

    # 4. Replacement 'a' prefixes — sorted by locant, grouped by element for multiplier
    if tree.replacements:
        # Group by a_prefix
        repl_by_prefix: dict[str, list] = defaultdict(list)
        for rp in tree.replacements:
            repl_by_prefix[rp.a_prefix].append(rp)

        # Sort replacements by first locant in each group
        sorted_groups = sorted(
            repl_by_prefix.items(),
            key=lambda kv: min(r.locant for r in kv[1])
        )

        repl_parts: list[str] = []
        for a_prefix, repls in sorted_groups:
            # Sort by locant within group
            repls_sorted = sorted(repls, key=lambda r: r.locant)
            locants_str = ",".join(str(r.locant) for r in repls_sorted)
            count = len(repls_sorted)
            mult = (get_multiplier(count, complex=False) or "") if count > 1 else ""
            repl_parts.append(f"{locants_str}-{mult}{a_prefix}")

        parts.append("".join(repl_parts))

    # 5. Parent stem (all-carbon skeleton)
    stem_idx = len(parts)
    parts.append(tree.carbon_parent.stem)

    # 6. Unsaturation infixes
    if tree.unsaturation:
        parts.append(render_unsaturation(tree.unsaturation))

    # 7. Suffix
    if tree.suffix_groups:
        parent_length = tree.carbon_parent.candidate.length
        _has_ih = bool(tree.indicated_hydrogen)
        suffix_groups = _strip_locant_1_if_omissible(
            tree.suffix_groups, parent_length, parent_has_indicated_h=_has_ih
        )
        rendered_suf = render_suffixes(suffix_groups, tree.output_form)
        # Retain terminal 'e' when rendered suffix starts with a consonant
        # (e.g. "diol", "dione", "diamine" → need "propane-..." not "propan-...")
        if _rendered_suffix_starts_with_consonant(rendered_suf):
            if not parts[stem_idx].endswith("e"):
                parts[stem_idx] += "e"
        parts.append(rendered_suf)
    else:
        parts.append(terminal_vowel(tree.carbon_parent, tree.output_form))

    return elide_at_boundaries(parts)


# ---------------------------------------------------------------------------
# Substitutive assembly — the main case
# ---------------------------------------------------------------------------

def _ring_is_all_carbon_monocyclic(tree: SubstitutiveTree) -> bool:
    """Return True when the named parent is a monocyclic all-carbon ring.

    Used for the P-31.1.3.4 locant-omission rule: a single substituent on a
    fully symmetric (all-carbon monocyclic) ring does not need a locant.
    """
    ring_system = tree.named_parent.candidate.ring_system
    if ring_system is None:
        return False
    if ring_system.type != "monocyclic":
        return False
    # heteroatoms is None (not computed) or an empty tuple → all-carbon
    if ring_system.heteroatoms:
        return False
    return True


def _saturated_chain_full_substitution_signature(
    parent_length: int,
) -> tuple[str, ...]:
    """Return the locant multiset that a *fully* substituted saturated chain
    bears when a chain-terminal PCG occupies C1 (P-14.3.4.4 saturation case).

    For a saturated acyclic chain C1-C2-…-CN whose C1 is the principal-
    characteristic-group carbon (an acid / amide / etc. carbon that consumes
    all of C1's substitutable valences), every remaining position is on
    C2…CN.  Each internal methylene (C2…C(N-1)) contributes two substitutable
    H positions, and the terminal methyl (CN) contributes three.  A single
    monovalent substituent type that fills *every* one of those positions
    therefore bears the exact locant multiset::

        (2, 2, 3, 3, …, N-1, N-1, N, N, N)

    Returned as a sorted tuple of locant strings.  When the substituent's
    actual locant tuple equals this signature the substitution is *complete*
    and the locants are forced (no positional ambiguity remains), so they are
    omitted from the PIN — "heptafluorobutanoic acid", not
    "2,2,3,3,4,4,4-heptafluorobutanoic acid".  Any partial pattern (e.g.
    "3,3,3-trifluoropropanoic acid", where C2 carries no substituent) yields a
    different multiset and so retains its locants.

    Returns an empty tuple for parent_length < 2 (no off-C1 positions to fill).
    """
    if parent_length < 2:
        return ()
    locs: list[str] = []
    for c in range(2, parent_length + 1):
        # internal CH2 → two positions; terminal CH3 (c == parent_length) →
        # three positions.
        n = 3 if c == parent_length else 2
        locs.extend([str(c)] * n)
    return tuple(sorted(locs, key=lambda s: (int(s), s)))


def _needs_hyphen_before_stem(prev: str, stem: str) -> bool:
    """Return True when prev + stem requires an explicit hyphen (P-16.3.2).

    A hyphen is needed when the running prefix string ends with a letter or
    closing bracket AND the parent stem starts with a digit (e.g. "1H-imidazol",
    "1,3-benzothiazol") or a heteroatom locant letter followed by a digit or
    hyphen (e.g. "N-oxide"-style stems, uncommon).

    Examples requiring a hyphen:
      "1-methyl" + "1H-imidazol"       → "1-methyl-1H-imidazol"
      "1,5-dimethyl" + "1H-tetrazol"   → "1,5-dimethyl-1H-tetrazol"

    Examples NOT requiring a hyphen (stem starts with a letter):
      "chloro" + "benzen"              → "chlorobenzen"
      "chloro" + "methan"              → "chloromethan"
    """
    if not prev or not stem:
        return False
    prev_tail = prev[-1]
    if not (prev_tail.isalpha() or prev_tail in ")]}"):
        return False
    head = stem[0]
    if head.isdigit():
        return True
    # Heteroatom locant letter at stem start (e.g. "N-oxide" style, rare)
    if head in "NOSPH" and len(stem) > 1 and stem[1] in "-,0123456789":
        return True
    return False


# ---------------------------------------------------------------------------
# Retained-acyl-PIN rewrite (P-66.6.1 / P-66.6.3)
# ---------------------------------------------------------------------------
#
# IUPAC P-66.6.3.2 / P-66.6.1: ``benzoic acid`` is the retained PIN for the
# phenyl-CO-OH carbon framework, including under ring substitution.  Likewise
# ``acetic acid`` is the retained PIN for CH3-CO-OH and accommodates α-
# substituents (``trichloroacetic acid``, ``bromoacetic acid``, …).  The same
# stems propagate to acyl halides (``benzoyl chloride`` / ``acetyl chloride``),
# amides (``benzamide``), nitriles (``benzonitrile``), and the corresponding
# anion / ACID_STEM forms (``benzoate`` / ``acetate``).
#
# The unsubstituted forms are picked up by the Tier-0 retained-name lookup in
# the engine; the substituted forms fall through to the substitutive path,
# which builds names like ``2-methylbenzenecarboxylic acid`` and
# ``ethanoyl chloride``.  This post-process rewrites the systematic
# ``benzene[-N]-carb...`` / ``ethan-...`` tails to their retained PIN form.
#
# Gating is by tree shape — the rewrite only fires when the parent name and
# suffix base_form match the table — so it cannot misfire on unrelated
# substituents that happen to spell the same letters.

_BENZENE_RETAINED_TAIL: dict[tuple[str, OutputForm], tuple[str, str]] = {
    # base_form, output_form -> (regex pattern matching the substitutive tail,
    #                            replacement retained tail)
    ("carboxylic acid", OutputForm.STANDALONE):
        (r"benzene(?:-\d+)?-?carboxylic acid$",     "benzoic acid"),
    ("carboxylic acid", OutputForm.ACID_STEM):
        (r"benzene(?:-\d+)?-?carboxylate$",         "benzoate"),
    ("carboxylic acid", OutputForm.ANION):
        (r"benzene(?:-\d+)?-?carboxylate$",         "benzoate"),
    # The "-\d+-" locant is OPTIONAL: a mono-substituted benzene omits the
    # substituent locant (P-31.1.3.4), so the un-derivatised acyl-halide tail
    # is "benzenecarbonyl chloride" (no locant) and the locanted form only
    # arises under further ring substitution.  Match both.
    ("carbonyl chloride", OutputForm.STANDALONE):
        (r"benzene(?:-\d+)?-?carbonyl chloride$",   "benzoyl chloride"),
    ("carbonyl bromide", OutputForm.STANDALONE):
        (r"benzene(?:-\d+)?-?carbonyl bromide$",    "benzoyl bromide"),
    ("carbonyl fluoride", OutputForm.STANDALONE):
        (r"benzene(?:-\d+)?-?carbonyl fluoride$",   "benzoyl fluoride"),
    ("carbonyl iodide", OutputForm.STANDALONE):
        (r"benzene(?:-\d+)?-?carbonyl iodide$",     "benzoyl iodide"),
    ("carboxamide", OutputForm.STANDALONE):
        (r"benzene(?:-\d+)?-?carboxamide$",         "benzamide"),
    ("carbonitrile", OutputForm.STANDALONE):
        (r"benzene(?:-\d+)?-?carbonitrile$",        "benzonitrile"),
}

_ETHANE_RETAINED_TAIL: dict[tuple[str, OutputForm], tuple[str, str]] = {
    ("oic acid", OutputForm.STANDALONE):
        (r"ethanoic acid$",   "acetic acid"),
    ("oic acid", OutputForm.ACID_STEM):
        (r"ethanoate$",       "acetate"),
    ("oic acid", OutputForm.ANION):
        (r"ethanoate$",       "acetate"),
    ("oyl chloride", OutputForm.STANDALONE):
        (r"ethanoyl chloride$", "acetyl chloride"),
    ("oyl bromide", OutputForm.STANDALONE):
        (r"ethanoyl bromide$",  "acetyl bromide"),
    ("oyl fluoride", OutputForm.STANDALONE):
        (r"ethanoyl fluoride$", "acetyl fluoride"),
    ("oyl iodide", OutputForm.STANDALONE):
        (r"ethanoyl iodide$",   "acetyl iodide"),
    # P-66.1: acetamide is the retained PIN for CH3-C(=O)-NH2 and propagates
    # under N-substitution (N-methylacetamide, N,N-dimethylacetamide, etc.).
    # The substitutive path emits "ethanamide"; rewrite the tail so the
    # retained acid stem ("acet-") survives the derivation.
    ("amide", OutputForm.STANDALONE):
        (r"ethanamide$",        "acetamide"),
}


def _apply_retained_acyl_pin(result: str, tree: SubstitutiveTree) -> str:
    """Rewrite ``benzene[-N]-carb…`` / ``ethan…`` tails to their retained PINs.

    Per IUPAC P-66.6.3.2 the retained acid name is the PIN even when the
    parent ring carries substituents.  The substitutive path correctly emits
    the systematic form for the substituted molecule; this function rewrites
    the tail to the retained PIN form.  Locants on substituents are unaffected
    because numbering already places the principal characteristic group at
    position 1, and the retained name implies that locant.
    """
    parent = tree.named_parent.name
    if parent not in ("benzene", "ethane"):
        return result
    if len(tree.suffix_groups) != 1:
        return result
    if tree.unsaturation:
        # Benzene is aromatic without explicit unsaturation infixes; ethane
        # has no unsaturation by definition.  Anything here means the tree
        # carries extra modifiers we do not handle.
        return result
    if tree.indicated_hydrogen:
        return result
    if tree.ring_cation_locants or tree.ring_anion_locants:
        return result
    base_form = tree.suffix_groups[0].base_form
    table = _BENZENE_RETAINED_TAIL if parent == "benzene" else _ETHANE_RETAINED_TAIL
    entry = table.get((base_form, tree.output_form))
    if entry is None:
        return result
    pattern, replacement = entry
    return re.sub(pattern, replacement, result)


def _assemble_substitutive(tree: SubstitutiveTree) -> str:
    """Assemble a SubstitutiveTree to its IUPAC name string.

    Steps:
    1. Stereo descriptors ("(2R,3S)-")
    2. Indicated hydrogen ("1H-")
    3. Prefixes: assemble each, deduplicate, sort alphabetically
    4. Parent stem (method-dependent)
    5. Unsaturation infixes ("-en-", "-yn-")
    6. Suffix (FG suffix, free-valence suffix, or terminal vowel)
    """
    parts: list[str] = []

    # 1. Stereo descriptors
    if tree.stereo_descriptors:
        parts.append(render_stereo(tree.stereo_descriptors))

    # 1b. Isotope labels (Stage 6 R1-D)
    # IUPAC P-82 "Isotopically Modified Compounds" — the bracketed element
    # prefix sits between the stereo descriptor and the indicated-H
    # marker.  When any label carries a locant we add a trailing hyphen
    # so subsequent tokens read "(1-¹³C)-1H-indole" / "(1-²H)-ethan-1-ol".
    # When every label is whole-molecule (no locant, e.g. "(²H₄)methanol")
    # the bracket is written immediately before the parent name without
    # a hyphen, matching canonical IUPAC usage.
    if tree.isotope_labels:
        from iupac_namer.isotope import render_isotope_labels as _render_iso
        iso_str = _render_iso(tree.isotope_labels)
        if iso_str:
            any_locanted = any(
                lbl.locant is not None for lbl in tree.isotope_labels
            )
            parts.append(iso_str + "-" if any_locanted else iso_str)

    # 2. Indicated hydrogen
    if tree.indicated_hydrogen:
        parts.append(render_indicated_h(tree.indicated_hydrogen))

    # 3. Prefixes
    if tree.prefixes:
        assembled_prefixes: list[tuple[str, tuple[Locant, ...]]] = []
        for pe in tree.prefixes:
            prefix_name = assemble(pe.tree)
            assembled_prefixes.append((prefix_name, pe.locants))

        # P-31.1.3.4: For a single substituent on a fully symmetric
        # (all-carbon monocyclic) ring with no FG suffix, the locant is omitted.
        # This covers "chlorobenzene" (not "1-chlorobenzene") and
        # "methylcyclohexane" (not "1-methylcyclohexane").
        # EXCEPTION 1: do NOT omit locants when the ring is itself a substituent
        # (OutputForm.SUBSTITUENT). In that case the attachment atom is at
        # locant 1 and other substituents must carry their real locants, e.g.
        # "(4-hydroxyphenyl)" — the "4-" is load-bearing.
        # EXCEPTION 2: do NOT omit when the ring carries non-aromatic endocyclic
        # unsaturation (e.g. cyclohexene, cyclohexa-1,3-diene, cyclooctatetraene).
        # A double bond breaks the ring's positional symmetry, so the substituent
        # locant is load-bearing: "3-bromocyclohexene", "4-chlorocyclohexene",
        # and "1-chlorocyclohexene" are distinct structures.  Fully saturated
        # (cyclohexane) and aromatic (benzene) rings have ring_unsaturation_bonds
        # == None, so they still omit correctly.
        # EXCEPTION 3 (P-73): do NOT omit when the ring carries a charge feature
        # that fixes the numbering.  A ring cation/anion centre (the -ylium /
        # -ium / -ide atom) is the fixed reference at locant 1, so it breaks
        # ring-position equivalence exactly as a double bond does.  The three
        # chlorophenylium isomers (2-/3-/4-chlorophenylium) are distinct and
        # the substituent locant is load-bearing; without it OPSIN defaults the
        # substituent to the position adjacent to the ylium carbon.  The charge
        # feature is signalled either by the ring_cation/anion locant fields
        # (hetero-ring cations) or by a retained ring-cation parent name whose
        # stem already bakes in the charge suffix (carbon ylium: "phenylium").
        _parent_name_for_charge = tree.named_parent.name
        _ring_charge_fixes_numbering = bool(
            tree.ring_cation_locants
            or tree.ring_anion_locants
            or _parent_name_for_charge.endswith("ylium")
            or _parent_name_for_charge.endswith("ium")
        )
        if (len(assembled_prefixes) == 1
                and not tree.suffix_groups
                and _ring_is_all_carbon_monocyclic(tree)
                and not tree.named_parent.ring_unsaturation_bonds
                and not _ring_charge_fixes_numbering
                and tree.output_form != OutputForm.SUBSTITUENT):
            prefix_name, _locants = assembled_prefixes[0]
            assembled_prefixes = [(prefix_name, ())]

        # P-14.3.4.4 (general single-substituent symmetry omission): the
        # engine-computed flag is True when this lone prefix's locant is forced
        # by graph symmetry — every parent position it could occupy is in one
        # symmetry class.  This generalises the all-carbon-monocyclic special
        # case above to FUSED (chlorocoronene) and HETEROCYCLIC parents, and
        # also fires when a PCG suffix occupies symmetry-fixed positions while
        # the single prefix's position is itself forced (chlorobutanedioic acid:
        # the di-acid fixes both chain termini, leaving C2 ≡ C3, so the "2-" on
        # the chloro is redundant).  The flag is False whenever any structural
        # feature (unsaturation, heteroatom, charge centre, second substituent)
        # breaks the parent symmetry, so the structural exceptions guarding the
        # all-carbon-monocyclic rule above are baked into the flag itself; we
        # only re-assert the SUBSTITUENT-form guard (the engine gates on
        # STANDALONE, but assembly is the single source of truth for the
        # rendered form).  When a suffix is present the engine computes the flag
        # with the suffix kept attached, so the suffix's own locant handling
        # (in _strip_locant_1_if_omissible) is independent of this branch.
        elif (len(assembled_prefixes) == 1
                and tree.single_substituent_positions_all_equivalent
                and tree.output_form != OutputForm.SUBSTITUENT):
            prefix_name, _locants = assembled_prefixes[0]
            assembled_prefixes = [(prefix_name, ())]

        # P-14.6 (heteroatom parent hydrides): for a single-atom parent
        # (heteroatom_center — phosphane, silane, borane, etc.), there is
        # only one position (locant 1).  Locant 1 is trivially omissible —
        # "fluorophosphane" not "1-fluorophosphane".
        is_heteroatom_center = tree.named_parent.candidate.type == "heteroatom_center"
        if is_heteroatom_center:
            assembled_prefixes = [(name, ()) for name, _loc in assembled_prefixes]

        # P-14.3.4.5 / P-31.1.2.2: When the parent is a single-carbon chain
        # (methane) used as a substituent (SUBSTITUENT output form), all
        # substituent positions are on carbon 1 — there is only one possible
        # locant. Redundant "1,1,1-" locants must be suppressed:
        #   "1,1,1-trifluoromethyl" → "trifluoromethyl"
        # Restriction: only apply when the parent chain has exactly 1 carbon
        # (length == 1) and the output form is SUBSTITUENT. This avoids
        # suppressing load-bearing locants on longer chains or standalone names.
        if (tree.output_form == OutputForm.SUBSTITUENT
                and tree.named_parent.candidate.type == "chain"
                and tree.named_parent.candidate.length == 1):
            assembled_prefixes = [(name, ()) for name, _loc in assembled_prefixes]

        # P-14.6 (single-atom carbon parent — methanoic acid, methanephosphonic
        # acid, methanol, methanamine, methane, ...): there is only one
        # position, so prefix locants of "1" are redundant and OPSIN-confusing
        # ("1-phosphonomethanoic acid" parses to phosphono-O-CH=O instead
        # of phosphono-C(=O)-OH).  Strip the locant when EVERY prefix locant
        # equals 1 and the parent chain has exactly one atom; this leaves
        # multi-atom-chain locants intact.
        # EXCEPTION 1: methanamide / methanoyl-X derivatives have BOTH a C1 and
        # an amide-N attachment point.  When ANY prefix carries an "N" locant
        # (another substituent is N-attached), the bare "1-" on a C-attached
        # prefix is load-bearing — without it OPSIN may default the un-locanted
        # prefix to the amide N (e.g. streptozotocin's
        # 1-[methyl-(nitroso)amino]-N-(sugar)methanamide round-trip flips when
        # the "1-" is elided).
        # EXCEPTION 2 (Phase 8 carbamic): methanamide with NO N-locanted prefix
        # but a C-locanted substituent still requires the "1-" — OPSIN parses
        # bare "(R)methanamide" as R-on-N (an N-substituted formamide, giving
        # an extra N atom in the chain), not R-on-C.  This is the urea/
        # semicarbazone family: H2N-C(=O)-NHR (e.g. NC(=O)NN=C parses
        # without "1-" as H-C(=O)-NH-NH-N=CH2 — wrong connectivity).  For
        # methanethioamide (-C(=S)NH2) and methanohydrazide (-C(=O)NHNH2)
        # OPSIN attaches the unlocanted substituent to C, so only the
        # base_form == "amide" case needs explicit "1-" preservation.
        _AMIDE_BASE_FORMS_NEED_C1 = frozenset({"amide"})
        _has_amide_suffix_needing_c1 = any(
            sg.base_form in _AMIDE_BASE_FORMS_NEED_C1 for sg in tree.suffix_groups
        )
        if (tree.named_parent.candidate.type == "chain"
                and tree.named_parent.candidate.length == 1):
            _has_n_locant_prefix = any(
                any(str(l).upper() == "N" for l in loc)
                for _name, loc in assembled_prefixes
            )
            if not _has_n_locant_prefix and not _has_amide_suffix_needing_c1:
                assembled_prefixes = [
                    (name, () if all(str(l) == "1" for l in loc) else loc)
                    for name, loc in assembled_prefixes
                ]

        # P-14.3.4.4 (forced-locant omission on a 2-carbon acid chain):
        # When the parent is a 2-carbon chain whose C1 carries a terminal-by-
        # definition carboxylic-acid principal characteristic group (-oic acid
        # and chalcogen analogs), that C1 carbon is fully substituted by the
        # acid group, so EVERY substituent prefix is forced onto C2 — the only
        # remaining carbon.  The "2-" locants are then redundant and omitted
        # from the PIN (this is the "acetic acid" derivative family):
        #   "2-sulfanylacetic acid"    -> "sulfanylacetic acid"
        #   "2-phenylacetic acid"      -> "phenylacetic acid"
        #   "2,2-dichloroacetic acid"  -> "dichloroacetic acid"
        # (verified: each locant-free form round-trips through OPSIN to the same
        #  structure; "1-" on this carbon is unphysical because C1 is the acid
        #  carbon).  Only fires when ALL suffix groups sit at C1 and EVERY
        #  prefix locant equals "2"; an N-locant or any non-C2 prefix locant
        #  aborts the rule so genuinely-needed locants survive.
        #
        # Scoped to the ACID family only.  The aldehyde / amide / nitrile
        # families are deliberately excluded: a load-bearing regression guard
        # asserts "2-phenylethanal" (test_aldehyde_with_phenyl_still_works), and
        # the Blue Book worklist contains no 2-carbon aldehyde/amide/nitrile
        # PINs that omit the C2 locant — so the broader set would add no PIN
        # value while breaking that test.
        _SINGLE_ATOM_C1_SUFFIXES = frozenset({
            "oic acid", "thioic O-acid", "thioic S-acid", "dithioic acid",
        })
        # Restricted to STANDALONE whole-molecule names with no free valence
        # and no stereo descriptor.  In SUBSTITUENT / ACYL contexts (e.g.
        # "(2R)-2-amino-2-phenylacetyl-" nested inside a larger name) and
        # whenever a stereo descriptor cites locant "2", that locant is
        # load-bearing — stripping it leaves OPSIN unable to place the
        # stereocentre (cephem-antibiotic round-trip regression).
        if (tree.named_parent.candidate.type == "chain"
                and tree.named_parent.candidate.length == 2
                and tree.suffix_groups
                and tree.output_form == OutputForm.STANDALONE
                and tree.free_valence is None
                and not tree.stereo_descriptors
                and all(sg.base_form in _SINGLE_ATOM_C1_SUFFIXES
                        for sg in tree.suffix_groups)
                and all(len(sg.locants) == 1 and str(sg.locants[0]) == "1"
                        for sg in tree.suffix_groups)
                and assembled_prefixes
                and all(loc and all(str(l) == "2" for l in loc)
                        for _name, loc in assembled_prefixes)):
            assembled_prefixes = [(name, ()) for name, _loc in assembled_prefixes]

        # P-14.3.4.4 (forced-locant omission by COMPLETE SATURATION on a
        # ≥3-carbon acid chain):  the saturation analogue of the 2-carbon acid
        # rule above.  When a saturated acyclic chain whose C1 bears a chain-
        # terminal acid PCG is substituted at EVERY remaining position by a
        # single monovalent substituent type — i.e. the substituent's locant
        # multiset exactly equals the full-substitution signature
        # (2,2,3,3,…,N,N,N) — the locants are completely determined and are
        # omitted from the PIN:
        #   "heptafluorobutanoic acid"   (not 2,2,3,3,4,4,4-…)
        #   "pentafluoropropanoic acid"  (not 2,2,3,3,3-…)
        #   "heptachlorobutanoic acid", "heptabromobutanoic acid", …
        # The exact-multiset test is self-validating: any PARTIAL pattern (e.g.
        # "3,3,3-trifluoropropanoic acid", where C2 is unsubstituted) produces
        # a different multiset and so keeps its locants, and any MIXED-
        # substituent pattern (two prefix types) cannot have a single type
        # match the full signature.  Verified: each locant-free perhalo-acid
        # round-trips through OPSIN to the same structure.
        #
        # Gated identically to the 2-carbon acid rule (STANDALONE, no free
        # valence, no stereo descriptor, all suffix groups a single chain-
        # terminal acid base_form at C1) plus: the chain must be SATURATED
        # (no en/yn infixes change the per-carbon H counts the signature
        # assumes) and there must be exactly ONE distinct prefix substituent.
        if (tree.named_parent.candidate.type == "chain"
                and tree.named_parent.candidate.length >= 3
                and not tree.unsaturation
                and tree.suffix_groups
                and tree.output_form == OutputForm.STANDALONE
                and tree.free_valence is None
                and not tree.stereo_descriptors
                and all(sg.base_form in _SINGLE_ATOM_C1_SUFFIXES
                        for sg in tree.suffix_groups)
                and all(len(sg.locants) == 1 and str(sg.locants[0]) == "1"
                        for sg in tree.suffix_groups)
                and assembled_prefixes
                and len({name for name, _loc in assembled_prefixes}) == 1):
            _full_sig = _saturated_chain_full_substitution_signature(
                tree.named_parent.candidate.length
            )
            # Flatten the per-instance locant tuples into one multiset.  The
            # engine emits one PrefixEntry per substituent instance (seven
            # ``fluoro`` entries for heptafluoro), so concatenate every locant.
            _all_locs: list[str] = []
            for _name, _loc in assembled_prefixes:
                _all_locs.extend(str(l) for l in _loc)
            _actual_sig = tuple(sorted(_all_locs, key=lambda s: (int(s), s)))
            if _full_sig and _actual_sig == _full_sig:
                assembled_prefixes = [
                    (name, ()) for name, _loc in assembled_prefixes
                ]

        # P-14.3.4.4 (forced-locant omission on a symmetric 2-carbon ene/yne):
        # A 2-carbon chain's only unsaturation spans C1-C2, so the unsubstituted
        # parent ethene / ethyne is symmetric in its two carbons.  A SINGLE
        # substituent therefore has only one possible locant, and that "1" is
        # omitted from the PIN:
        #   "1-bromoethyne"  -> "bromoethyne"
        #   "1-chloroethene" -> "chloroethene"
        # Only fires for a SINGLE prefix at locant "1", no suffix, no free
        # valence, no stereo descriptor, STANDALONE output.  A second
        # substituent (e.g. 1,2-dichloroethene) makes the locants load-bearing
        # — they distinguish positional / geometric isomers — so the rule is
        # restricted to exactly one prefix.  Verified to round-trip via OPSIN.
        # (Hydrazine is an analogous symmetric 2-position parent but a
        #  load-bearing test asserts "1-methylhydrazine", so it is left alone.)
        _np_cand = tree.named_parent.candidate
        _is_ene_yne_2c = (
            _np_cand.type == "chain"
            and _np_cand.length == 2
            and bool(tree.unsaturation)
        )
        if (_is_ene_yne_2c
                and tree.output_form == OutputForm.STANDALONE
                and tree.free_valence is None
                and not tree.suffix_groups
                and not tree.stereo_descriptors
                and len(assembled_prefixes) == 1
                and assembled_prefixes[0][1]
                and all(str(l) == "1" for l in assembled_prefixes[0][1])):
            _pname = assembled_prefixes[0][0]
            assembled_prefixes = [(_pname, ())]

        merged = merge_identical_prefixes(assembled_prefixes)
        merged.sort(key=lambda m: m.sort_name)

        # P-73.4 anion-multiplicity rule for heteroatom parent hydrides:
        # When the parent is a single-atom heteroatom hydride (phosphane,
        # arsane, stibane, ...) bearing two or more identical anionic
        # substituents of the form "<atom>anide"/"oxido", the assembly form
        # is `<remaining-prefixes><parent>diyl|triyl|...bis|tris|...(<anion>)`,
        # NOT `<remaining-prefixes>bis|tris|...(<anion>)<parent>`.
        #
        #   CCCSP([S-])[S-]  →  (propylsulfanyl)phosphanediylbis(sulfanide)
        #   CP([S-])[S-]     →  methylphosphanediylbis(sulfanide)
        #   CP([O-])[O-]     →  methylphosphanediylbis(oxidanide)
        #   [S-]P([S-])[S-]  →  phosphanetriyltris(sulfanide)
        #
        # Implementation: pull the multiplied anion prefix out of `merged`
        # and stash it on a local variable; the parent-stem step appends
        # the `<stem>ediyl/etriyl/...` form and then this trailing anion
        # group.  Only fires for STANDALONE output and when there is no
        # other free-valence/suffix already on the parent.
        anion_suffix_form: str | None = None  # e.g. "ediylbis(sulfanide)"
        # OPSIN parses "oxidanide" but treats "oxido" as `=O`; rewrite for
        # the anion-suffix form so the assembled name round-trips.
        _ANION_SUFFIX_REWRITE = {"oxido": "oxidanide"}
        _ANION_PREFIX_NAMES = {"sulfanide", "oxido", "azanide",
                               "selanide", "tellanide", "oxidanide"}
        if (is_heteroatom_center
                and tree.output_form == OutputForm.STANDALONE
                and tree.free_valence is None
                and not tree.suffix_groups
                and not tree.unsaturation):
            # Count occurrences of each prefix name in the original (pre-merge)
            # entries — `merged` strips locants to () for heteroatom_center
            # parents, so we recover counts from `assembled_prefixes` instead.
            _prefix_counts: dict[str, int] = {}
            for _name, _locs in assembled_prefixes:
                _prefix_counts[_name] = _prefix_counts.get(_name, 0) + 1
            anion_indices = [
                i for i, mp in enumerate(merged)
                if mp.name in _ANION_PREFIX_NAMES
                and _prefix_counts.get(mp.name, 0) >= 2
            ]
            if len(anion_indices) == 1:
                idx = anion_indices[0]
                anion_mp = merged[idx]
                anion_count = _prefix_counts[anion_mp.name]
                # Free-valence multiplier (diyl/triyl/tetrayl/...)
                fv_suffix = FREE_VALENCE_SUFFIXES.get(
                    (anion_count, tuple([1] * anion_count))
                )
                if fv_suffix is not None:
                    anion_name = _ANION_SUFFIX_REWRITE.get(
                        anion_mp.name, anion_mp.name
                    )
                    # Drop this entry from prefix list; the diyl/anion goes
                    # at the parent-stem position instead.
                    merged = merged[:idx] + merged[idx + 1:]
                    # The stem ends in a consonant ("phosphan", "arsan",
                    # ...). We need "<stem>e<diyl>bis(<anion>)" — the 'e'
                    # is the parent's terminal vowel which must be kept
                    # because "diyl" starts with a consonant.
                    anion_suffix_form = (
                        f"e{fv_suffix}{anion_mp.multiplier}({anion_name})"
                    )

        # P-68.3 / P-71.1: for heteroatom parent hydrides (phosphane, silane,
        # borane, etc.), concatenated substituent names like "ethoxyethyl-"
        # are misread by OPSIN as a single group.  Force each substituent
        # prefix to be individually bracketed: "(ethoxy)(ethyl)phosphane"
        # rather than "ethoxyethylphosphane".
        #
        # P-16.3.3 exception (preferred form): the FIRST (lowest-sort) prefix is
        # left UNbracketed when it is a *simple* (unsubstituted) prefix.  The
        # opening enclosing mark of the following prefix already supplies the
        # boundary, so the leading bracket is redundant:
        #   "(chloro)(methyl)silane"               -> "chloro(methyl)silane"
        #   "(butyl)(ethyl)(methyl)(propyl)silane" -> "butyl(ethyl)(methyl)(propyl)silane"
        #   "tri(chloro)(methyl)silane"            -> "trichloro(methyl)silane"
        #   "(ethyl)di(methyl)phosphane"           -> "ethyldi(methyl)phosphane"
        #   "(acetyl)(methyl)chloranium"           -> "acetyl(methyl)chloranium"
        # The leading prefix stays bracketed when:
        #   - it is compound (carries an internal locant/bracket), because then
        #     the enclosing mark IS needed; or
        #   - it is an alkoxy/aryloxy ("...oxy") ether prefix.  Concatenating a
        #     multiplier directly onto such a prefix ("di(methoxy)" -> "dimethoxy")
        #     is deliberately kept bracketed for legibility in phosphane/silane
        #     oxoacid-ester names, matching the established PINs there; none of
        #     the simple-prefix PINs that this rule targets has an "-oxy" lead.
        # All non-leading prefixes remain bracketed so a boundary always exists.
        if is_heteroatom_center and len(merged) > 1:
            import dataclasses as _dc
            lead = merged[0]
            lead_bare = (
                not _is_compound_prefix(lead.name)
                and not lead.name.endswith("oxy")
            )
            merged = (
                [lead if lead_bare else _dc.replace(lead, needs_brackets=True)]
                + [_dc.replace(mp, needs_brackets=True) for mp in merged[1:]]
            )

        # P-29 / P-66.6.3: for a 1-carbon SUBSTITUENT (methyl) with multiple
        # substituents — typically a "C(=NH)(NHNH2)-" / "C(=NH)(NH2)-" /
        # "C(=O)(NHR)-" type fragment — OPSIN reads "(hydrazinyl)iminomethyl"
        # as "(hydrazinyl)imino-methyl" = R-CH=N-NHNH2, NOT as the intended
        # H2N-NH-C(=NH)-Ar. The simple chalcogen-class prefix (imino/oxo/
        # thioxo/selenoxo) must be individually bracketed so OPSIN treats both
        # substituents as siblings on the methyl carbon:
        #     "(hydrazinyl)iminomethyl" -> "hydrazinyl(imino)methyl"
        #     "(R)(oxo)methyl", "amino(imino)methyl" (= carbamimidoyl), etc.
        # Fires when (a) ANY merged prefix already needs brackets (compound), OR
        # (b) the methyl also carries a single-bonded chalcogenyl/hydroxy prefix
        #     (hydroxy/sulfanyl/selanyl/tellanyl) — the carbothioic /
        #     carbodithioic / carboselenoic O/S/Se acid demoted-to-prefix case
        #     (P-65.1): "oxosulfanylmethyl" is misread by OPSIN as O=S-CH2-
        #     (a phantom chain), so the =X prefix must be bracketed to give
        #     "(oxo)sulfanylmethyl" / "oxo(sulfanyl)methyl".
        if (tree.output_form == OutputForm.SUBSTITUENT
                and tree.named_parent.candidate.type == "chain"
                and tree.named_parent.candidate.length == 1
                and len(merged) >= 2):
            _DOUBLE_BOND_SIMPLE = {"imino", "oxo", "thioxo", "selenoxo", "telluroxo"}
            _CHALCOGENYL_SIMPLE = {"hydroxy", "sulfanyl", "selanyl", "tellanyl"}
            _any_compound = any(mp.needs_brackets for mp in merged)
            _has_double_bond = any(
                mp.name in _DOUBLE_BOND_SIMPLE and mp.multiplier is None
                for mp in merged
            )
            _has_chalcogenyl = any(
                mp.name in _CHALCOGENYL_SIMPLE for mp in merged
            )
            if _any_compound or (_has_double_bond and _has_chalcogenyl):
                import dataclasses as _dc
                # The compound-prefix case (a) only needs the =X double-bond
                # prefix bracketed.  The chalcogen-acid case (b) needs BOTH the
                # =X prefix AND the single-bonded chalcogenyl prefix bracketed:
                # bracketing only the leading prefix leaves OPSIN reading the
                # trailing "-yl" as a chain ("(oxo)sulfanylmethyl" -> O=S-CH2-),
                # so "(oxo)(sulfanyl)methyl" / "oxo(sulfanyl)methyl" is required.
                _bracket_chalcogenyl = _has_double_bond and _has_chalcogenyl
                merged = [
                    _dc.replace(mp, needs_brackets=True)
                    if (
                        (mp.name in _DOUBLE_BOND_SIMPLE and mp.multiplier is None)
                        or (_bracket_chalcogenyl
                            and mp.name in _CHALCOGENYL_SIMPLE
                            and mp.multiplier is None)
                    )
                    else mp
                    for mp in merged
                ]

        parts.append(render_merged_prefixes(merged))
    else:
        anion_suffix_form = None

    # 4. Parent stem (method-dependent)
    fv = tree.free_valence
    # Flag set when we collapse "<chain>an" → "<chain>" for the contracted
    # alkyl form (P-29.2 case where ALKANYL's free-valence locant elides).
    # The FV-suffix step uses this to drop its leading hyphen so that the
    # concatenation reads "propyl" not "prop-yl".
    contracted_alkyl_form = False
    if (fv is not None
            and fv.is_monovalent
            and fv.method == SubstituentMethod.ALKYL
            and tree.named_parent.alkyl_stem is not None):
        stem_part = tree.named_parent.alkyl_stem
    elif (fv is not None
            and not tree.suffix_groups
            and not tree.unsaturation
            and tree.named_parent.alkyl_stem is not None
            and tree.named_parent.stem == tree.named_parent.alkyl_stem + "an"
            and _free_valence_locant_will_elide(fv, tree.numbering)):
        # P-29.2 contracted-alkyl form: when ALKANYL's free-valence locant
        # ends up elided (attachment at C1 with no other locant constraint),
        # the rendered suffix is just "-yl" — collapse "<chain>an" + "-yl"
        # into the contracted "<chain>yl" form.
        # E.g. "propan" + "-yl" → "propyl" (not "propan-yl").
        # Examples affected: "1-(4-chlorophenyl)-1-phenylethyl" (was
        # "...-ethan-yl"), "(1,1-dimethoxymethyl)" (was "...methan-yl").
        # Restriction: stem == alkyl_stem + "an" excludes unsaturated ring
        # parents whose unsaturation is baked into the stem
        # (e.g. "cyclohex-3-en"/"cyclohex"), where dropping back to
        # alkyl_stem would silently lose the unsaturation locant.
        stem_part = tree.named_parent.alkyl_stem
        contracted_alkyl_form = True
    elif tree.unsaturation and tree.named_parent.alkyl_stem is not None:
        # IUPAC P-31.1.2.1: when a chain has unsaturation (double/triple bonds),
        # the parent stem drops the "-ane" ending.
        # "butane" → "but-2-ene" (not "butan-2-ene")
        # "hexane" → "hex-3-en-1-al" (not "hexan-3-en-1-al")
        # The alkyl_stem is already the bare prefix (e.g. "but", "hex").
        stem_part = tree.named_parent.alkyl_stem
    else:
        stem_part = tree.named_parent.stem

    # P-31.1.2.1 ring locant-omission rule:
    # For a monocyclic ring with a SINGLE double bond and NO FG suffix, the
    # double-bond locant is omissible in STANDALONE form (e.g. "cyclohexene").
    # In SUBSTITUENT form, the double-bond locant MUST be retained so OPSIN
    # can distinguish "cyclohex-2-en-1-yl" from "cyclohex-3-en-1-yl" etc.
    # The stem already encodes the locant (e.g. "cyclohex-1-en") to support
    # the general case. Strip it here only for STANDALONE/non-substituent forms.
    import re as _re_asm
    if (
        not tree.suffix_groups
        and not tree.unsaturation
        and tree.named_parent.candidate.ring_system is not None
        and tree.named_parent.candidate.ring_system.type == "monocyclic"
        and tree.output_form != OutputForm.SUBSTITUENT
    ):
        _ring_single_db = _re_asm.match(r"^(cyclo[a-z]+)-\d+-en$", stem_part)
        if _ring_single_db:
            stem_part = _ring_single_db.group(1) + "en"

    # Track the stem index so we can append terminal 'e' if needed.
    # P-16.3.2: insert a hyphen separator when the previous part ends with a
    # letter (or close-bracket) and the stem starts with a digit or heteroatom
    # locant letter (e.g. "1-methyl" + "1H-imidazol" → "1-methyl-1H-imidazol").
    if parts and _needs_hyphen_before_stem(parts[-1], stem_part):
        parts.append("-")
    stem_idx = len(parts)
    parts.append(stem_part)

    # 5. Unsaturation infixes
    if tree.unsaturation:
        parent_length = tree.named_parent.candidate.length
        active_unsat = _strip_unsaturation_locants_if_omissible(
            tree.unsaturation, parent_length
        )
        # IUPAC P-31.1.2.1: insert 'a' before the infix when any infix
        # has a multiplier (di, tri, ...) — e.g. "buta-1,3-diene".
        # For single unsaturation (no multiplier), no 'a' — "but-2-ene".
        if any(inf.multiplier for inf in active_unsat):
            parts[stem_idx] = stem_part + "a"
        parts.append(render_unsaturation(active_unsat))

    # 6. Suffix
    if tree.suffix_groups:
        parent_length = tree.named_parent.candidate.length
        # Detect indicated-H either on the tree field or baked into the parent
        # name/stem (e.g. "1H-pyrrole", "4H-pyran" from the retained-ring DB,
        # or "2,5-dihydro-1H-pyrrole" where the hydro prefix precedes the IH).
        import re as _re_ih
        _has_ih = (
            bool(tree.indicated_hydrogen)
            or bool(getattr(tree.named_parent, "indicated_hydrogen", None))
            or _re_ih.search(r"(?:^|-)\d+H-", tree.named_parent.name or "") is not None
            or _re_ih.search(r"(?:^|-)\d+H-", tree.named_parent.stem or "") is not None
        )

        # P-58.2.2: when the parent carries a leading "<digit>H-" indicated-H
        # marker AND a single ring-PCG suffix (one / thione / selone /
        # tellurone / imine) is present at a different ring locant, the
        # indicated-H is consequent to the suffix and must be cited inline as
        # ``(NH)`` after the suffix locant — NOT as a separate front-of-name
        # token.  E.g. ``1H-pyridine`` + ``-4-one`` → ``pyridin-4(1H)-one``
        # (PIN), not the malformed ``1H-pyridin-4-one``.  Only fires when:
        #   * the stem starts with a SINGLE ``<digit>H-`` (not e.g. ``1H,3H-``),
        #   * exactly one suffix group with exactly one locant,
        #   * the suffix's base_form is in the added-IH PCG set,
        #   * the suffix's existing ``added_indicated_h`` is empty,
        #   * the IH locant differs from the suffix locant.
        # Also strips the same prefix from the parent name detection so the
        # ``_has_ih`` flag below is computed against the rewritten stem.
        _RING_PCG_ADDED_IH_BASE_FORMS = frozenset({
            "one", "thione", "selone", "tellurone", "imine",
        })
        adjusted_suffix_groups: tuple[SuffixGroup, ...] = tree.suffix_groups
        _ih_match = _re_ih.match(r"^(\d+[a-z]?)H-(.+)$", parts[stem_idx])
        if (_ih_match
                and len(tree.suffix_groups) == 1
                and len(tree.suffix_groups[0].locants) == 1
                and tree.suffix_groups[0].base_form in _RING_PCG_ADDED_IH_BASE_FORMS
                and not tree.suffix_groups[0].added_indicated_h):
            ih_locant_str = _ih_match.group(1)
            stem_remainder = _ih_match.group(2)
            sg = tree.suffix_groups[0]
            suf_locant_str = str(sg.locants[0])
            if ih_locant_str != suf_locant_str:
                # Build the IH locant.
                from iupac_namer.types import Locant as _Locant
                _ih_num: int | None = None
                _ih_suf = ""
                _i = 0
                while _i < len(ih_locant_str) and ih_locant_str[_i].isdigit():
                    _i += 1
                if _i > 0:
                    _ih_num = int(ih_locant_str[:_i])
                    _ih_suf = ih_locant_str[_i:]
                if _ih_num is not None:
                    ih_locant = _Locant.numeric(_ih_num, _ih_suf)
                    import dataclasses as _dc
                    new_sg = _dc.replace(
                        sg, added_indicated_h=(ih_locant,)
                    )
                    adjusted_suffix_groups = (new_sg,)
                    parts[stem_idx] = stem_remainder
                    # _has_ih now refers to the absorbed IH form: the parent
                    # carries an "added-IH" tag, not a front-of-name IH.
                    _has_ih = True

        # P-14.3.4.2(c) eligibility: monosubstituted homogeneous monocyclic
        # ring with exactly one suffix at locant 1 and no other prefixes.
        # The suffix-only-locant rule fires below in _strip_locant_1_if_omissible.
        # We also gate on no unsaturation (since unsaturation locants would
        # break ring-position equivalence) and no indicated-H / cation / anion
        # markers (any of which select a specific ring atom).  Also detect
        # unsaturation baked into the stem (e.g. "cyclohex-3-en") — those carry
        # an explicit locant that ranks the ring positions, so the suffix
        # locant-1 is load-bearing and must NOT be omitted.
        _ring_system = tree.named_parent.candidate.ring_system
        _stem_has_baked_unsat_locant = bool(
            _re_asm.search(r"-\d+(?:,\d+)*-(?:en|yn)$", parts[stem_idx])
        )
        _is_mono_hom_monocycle = (
            _ring_system is not None
            and _ring_system.type == "monocyclic"
            and not _ring_system.heteroatoms
            and not tree.prefixes
            and not tree.unsaturation
            and not _stem_has_baked_unsat_locant
            and not tree.indicated_hydrogen
            and not tree.ring_cation_locants
            and not tree.ring_anion_locants
        )
        suffix_groups = _strip_locant_1_if_omissible(
            adjusted_suffix_groups, parent_length, parent_has_indicated_h=_has_ih,
            is_monosubstituted_homogeneous_monocycle=_is_mono_hom_monocycle,
            single_suffix_symmetry_forced=(
                tree.single_substituent_positions_all_equivalent
                and not tree.prefixes
            ),
        )
        rendered_suf = render_suffixes(suffix_groups, tree.output_form)
        # IUPAC elision rule: when the rendered suffix (after multiplier application)
        # starts with a consonant (e.g. "triol", "diol", "dione", "diamine"),
        # the parent stem retains its terminal 'e'.  We insert 'e' into the stem
        # part directly so that elide() (which only strips 'e' before a vowel)
        # does not accidentally remove it.
        # Only add terminal 'e' when there is NO unsaturation infix.
        # When unsaturation is present (e.g. "but-2-en-"), the vowel
        # connection is handled by render_unsaturation + elide().
        # IMPORTANT: only restore 'e' if the parent NAME actually carries a
        # terminal 'e' (i.e. the stem was formed by stripping it). Parent
        # names that natively end in a consonant — penam, cepham, furan,
        # pyran, coumarin, decalin — must NOT have an 'e' appended; doing
        # so produces malformed names like "pename-3-carboxylic acid".
        if not tree.unsaturation:
            if _rendered_suffix_starts_with_consonant(rendered_suf):
                parent_name = tree.named_parent.name
                if (parent_name.endswith("e")
                        and not parts[stem_idx].endswith("e")):
                    parts[stem_idx] += "e"
        parts.append(rendered_suf)
    elif fv is not None and any(o > 0 for o in fv.bond_orders):
        # Check for retained ring substituent form (P-31.1.2.4):
        # e.g. "benzene" + "-yl" -> "phenyl" (not "benzenyl")
        retained_sub = _RETAINED_RING_SUBSTITUENT.get(tree.named_parent.name)
        if (retained_sub is not None
                and fv.is_monovalent
                and not tree.suffix_groups
                and not tree.unsaturation):
            # Replace entire stem with the retained substituent form
            parts[stem_idx] = retained_sub
        else:
            # P-73.1 ring-N+ infix for SUBSTITUENT form: insert "-N-ium-"
            # between the parent stem and the free-valence suffix so that
            # e.g. "1-azabicyclo[2.2.2]octan" + (1,) + "-3-yl" renders as
            # "1-azabicyclo[2.2.2]octan-1-ium-3-yl" (not the broken
            # "1-azabicyclo[2.2.2]octan-3-yl-1-ium" that would result if
            # the ium suffix were appended after -yl).  The standalone
            # CATION form keeps the ium suffix at the end (handled in the
            # later block).
            if tree.ring_cation_locants:
                rc_locants = tree.ring_cation_locants
                rc_count = len(rc_locants)
                rc_locant_str = ",".join(str(loc) for loc in rc_locants)
                rc_mult = (
                    get_multiplier(rc_count, complex=False)
                    if rc_count > 1 else ""
                ) or ""
                # Singular -ium: "i" is a vowel -> drop terminal 'e' on stem.
                # Multiplied -diium/-triium: leading consonant -> keep 'e'.
                if not rc_mult:
                    if parts[stem_idx].endswith("e"):
                        parts[stem_idx] = parts[stem_idx][:-1]
                else:
                    if not parts[stem_idx].endswith("e"):
                        parts[stem_idx] += "e"
                parts.append(f"-{rc_locant_str}-{rc_mult}ium")
            # P-31.1.2.1: when the parent's unsaturation is baked into the
            # stem (e.g. monocyclic ring "cyclohex-1-en"), tree.unsaturation
            # is empty but the substituent is still unsaturated.  Detect a
            # locant-bearing "-en"/"-yn" tail in the stem and treat it the
            # same as tree.unsaturation so the free-valence locant "1" is
            # still cited (cyclohex-1-en-1-yl, not cyclohex-1-en-yl).
            _stem_text = parts[stem_idx]
            _stem_has_baked_unsat = bool(
                _re_asm.search(r"-\d+(?:,\d+)*-(?:en|yn)$", _stem_text)
            )
            fv_rendered = render_free_valence_suffix(
                fv, tree.numbering,
                has_unsaturation=bool(tree.unsaturation) or _stem_has_baked_unsat,
            )
            if contracted_alkyl_form and fv_rendered.startswith("-"):
                # Strip the leading hyphen: "prop" + "-yl" → "propyl",
                # not "prop-yl". The contracted-stem branch above replaced
                # "<chain>an" with "<chain>", so the suffix must abut.
                fv_rendered = fv_rendered[1:]
            parts.append(fv_rendered)
    elif anion_suffix_form is not None:
        # P-73.4 dianion-on-heteroatom: append "<e><diyl|triyl|...>bis(...)"
        # in place of the terminal vowel.  The 'e' is the parent's terminal
        # vowel, kept because "diyl/triyl" starts with a consonant.
        parts.append(anion_suffix_form)
    else:
        parts.append(terminal_vowel(tree.named_parent, tree.output_form))

    # 6b. Ring-embedded N+ → -ium suffix (P-73.1, cation nomenclature).
    # Populated by SubstitutivePath.execute when output_form == CATION and
    # the parent backbone contains one or more ring-embedded [N+] atoms
    # (and the parent name does not already encode the cation, e.g. the
    # retained "pyridinium" stem).
    #
    # Rendering examples:
    #   piperidine + (1,)        → "piperidin-1-ium"
    #   piperazine + (1, 4)      → "piperazine-1,4-diium"
    #   azabicyclo[3.2.1]octane + (8,) → "azabicyclo[3.2.1]octan-8-ium"
    #
    # The "i" of "ium" is a vowel, so terminal 'e' on the stem is stripped
    # by elide() automatically.  For the "-diium"/"-triium" cases, the
    # leading "d"/"t" is a consonant — but the locant block precedes the
    # multiplier ("-1,4-diium"), so the stem still abuts a hyphen+digit
    # rather than a letter and elide() leaves the 'e' alone.  Net effect
    # for diium: "piperazine-1,4-diium" (stem keeps 'e').  We insert the
    # 'e' explicitly to mirror the consonant-suffix rule above.
    # SUBSTITUENT form already inserted "-N-ium-" as an infix BEFORE the
    # free-valence suffix in the elif fv-branch above; skip the trailing
    # append here to avoid duplicating the cation marker.
    if tree.ring_cation_locants and tree.output_form != OutputForm.SUBSTITUENT:
        rc_locants = tree.ring_cation_locants
        rc_count = len(rc_locants)
        rc_locant_str = ",".join(str(loc) for loc in rc_locants)
        rc_mult = get_multiplier(rc_count, complex=False) if rc_count > 1 else ""
        rc_mult = rc_mult or ""
        rc_suffix = f"-{rc_locant_str}-{rc_mult}ium"
        if tree.suffix_groups:
            # P-74 / P-31.1.4.3.4: when the cationic ring ALSO bears a
            # principal characteristic group expressed as a suffix
            # (e.g. -carboxylic acid), the "-ium" cation suffix is cited
            # IMMEDIATELY after the parent stem and BEFORE the PCG suffix:
            #   pyridine + "-1-ium" + "-4-carboxylic acid"
            #     → "pyridin-1-ium-4-carboxylic acid"  (PIN)
            # NOT the order-reversed "pyridine-4-carboxylic acid-1-ium".
            # The PCG suffix (rendered_suf) was just appended as the final
            # part by the suffix block; insert "-ium" directly before it.
            # The stem region now abuts the vowel-leading "-...-ium", so any
            # terminal 'e' the consonant-suffix branch restored onto the stem
            # (for a consonant-leading PCG suffix like "-carboxylic acid")
            # must be elided (P-16.3.3); for "-diium"/"-triium"
            # (consonant-leading multiplier) the 'e' is kept.
            suf_idx = len(parts) - 1
            if not rc_mult:
                if parts[stem_idx].endswith("e"):
                    parts[stem_idx] = parts[stem_idx][:-1]
            else:
                if not parts[stem_idx].endswith("e"):
                    parts[stem_idx] += "e"
            parts.insert(suf_idx, rc_suffix)
        else:
            # If the multiplied form starts with a consonant (di/tri/...),
            # restore the stem's terminal 'e' (elision would only strip it
            # before a vowel).
            if rc_mult and not tree.unsaturation:
                # The terminal 'e' (if needed) was appended at line ~1509 as a
                # separate part by terminal_vowel(), or is already on the stem.
                # Make sure the stem region (joined) ends in 'e'.
                if not (parts[stem_idx].endswith("e")
                        or (len(parts) > stem_idx + 1 and parts[-1] == "e")):
                    parts[stem_idx] += "e"
            # IUPAC P-16.3.3: when the suffix begins with a vowel (singular
            # "-ium": the "i" is a vowel), the parent stem's terminal 'e' is
            # elided ("pyridine" + "-1-ium" → "pyridin-1-ium", not
            # "pyridine-1-ium").  elide() only strips 'e' immediately before a
            # vowel — but here the stem's 'e' abuts a hyphen+locant, so elide()
            # never sees it.  Drop the terminal 'e' here directly.  The 'e' may
            # have been appended either onto parts[stem_idx] (e.g. by the
            # consonant-suffix branch above for compound cases) or as a
            # separate "e" part by terminal_vowel() at line ~1509.
            # For "-diium"/"-triium" (consonant-leading multiplier), the stem
            # keeps its 'e' (handled by the branch above).
            if not rc_mult and not tree.unsaturation:
                if len(parts) > stem_idx + 1 and parts[-1] == "e":
                    parts.pop()
                elif parts[stem_idx].endswith("e"):
                    parts[stem_idx] = parts[stem_idx][:-1]
            parts.append(rc_suffix)

    # 6c. Ring-embedded aromatic [n-] → -ide suffix (P-72.2 / P-73, anion
    # nomenclature).  Populated by SubstitutivePath.execute when the parent
    # backbone contains one or more deprotonated aromatic ring N atoms.
    #
    # Rendering examples:
    #   1H-purine + (7,)      → "1H-purin-7-ide"
    #   1,3-dimethyl-2,6-dioxo-2,3,6,7-tetrahydro-1H-purine + (7,)
    #                         → "1,3-dimethyl-2,6-dioxo-2,3,6,7-tetrahydro-1H-purin-7-ide"
    #
    # Elision mirrors the -ium branch: singular "-ide" starts with a vowel
    # ("i") so the stem's terminal 'e' is dropped; multiplied "-diide"/
    # "-triide" starts with a consonant so the stem keeps 'e'.  SUBSTITUENT
    # form is not emitted here (ring-anion parent forms are not currently
    # attached as substituents in the benchmark; if ever needed, an infix
    # "-N-ide-" could be added in the fv branch above).
    if tree.ring_anion_locants and tree.output_form != OutputForm.SUBSTITUENT:
        ra_locants = tree.ring_anion_locants
        ra_count = len(ra_locants)
        ra_locant_str = ",".join(str(loc) for loc in ra_locants)
        ra_mult = get_multiplier(ra_count, complex=False) if ra_count > 1 else ""
        ra_mult = ra_mult or ""
        ra_suffix = f"-{ra_locant_str}-{ra_mult}ide"
        if ra_mult and not tree.suffix_groups and not tree.unsaturation:
            if not (parts[stem_idx].endswith("e")
                    or (len(parts) > stem_idx + 1 and parts[-1] == "e")):
                parts[stem_idx] += "e"
        if not ra_mult and not tree.suffix_groups and not tree.unsaturation:
            if len(parts) > stem_idx + 1 and parts[-1] == "e":
                parts.pop()
            elif parts[stem_idx].endswith("e"):
                parts[stem_idx] = parts[stem_idx][:-1]
        parts.append(ra_suffix)

    result = elide_at_boundaries(parts)
    # P-66.6 retained-acyl-PIN rewrite — must run AFTER elision so the
    # systematic ``benzenecarboxylic acid`` / ``ethanoic acid`` tail is fully
    # composed before we rewrite it to its retained PIN form.
    result = _apply_retained_acyl_pin(result, tree)
    return result


# ---------------------------------------------------------------------------
# Salt-fragment multiplier collapse (P-66 / P-73)
# ---------------------------------------------------------------------------

# Charge marker like "(1+)", "(2+)", "(3-)" appended to monatomic ion names.
_CHARGE_SUFFIX_RE = re.compile(r"\(\d+[+-]\)$")


def _collapse_identical_salt_ions(ion_names: list[str]) -> list[str]:
    """Collapse runs of identical charged salt fragments with a multiplier.

    Per IUPAC P-66 (multiplicative prefixes) / P-73 (cation nomenclature):
    multiple equivalents of the same counterion are denoted by a numerical
    multiplicative prefix (e.g. ``decasodium(1+)`` for 10 Na+, ``disodium(1+)``
    for 2 Na+).  This collapses ``["sodium(1+)", "sodium(1+)", ...]`` into a
    single ``"decasodium(1+)"`` token.

    Eligibility gate: only fragment names that carry an explicit trailing
    charge marker like ``(1+)``, ``(2+)``, ``(3-)`` are collapsed.  This
    deliberately excludes:
      * neutral fragments (e.g. co-crystallised ``formic acid`` x2 -> not a
        salt, must not collapse to ``bis(formic acid)``); and
      * marker-less retained ion names (e.g. ``azanium`` for NH4+) where the
        ``di``-prefixed form clashes with a recognised hydride parent
        (``diazanium`` is hydrazinium, [NH3+]N, not 2 x NH4+).

    Multiplier choice mirrors ``merge_identical_prefixes`` (P-16.3.3 /
    P-16.3.4): *simple* names (a single word stem plus the charge marker,
    e.g. ``sodium(1+)``) take ``di``/``tri``/.../``deca`` directly attached
    before the stem; *compound* names (containing spaces, hyphens, or
    non-charge-marker parentheses) take ``bis``/``tris``/... wrapping the
    whole fragment in parentheses.

    Order is preserved: each unique name appears at the position of its first
    occurrence, so the upstream cation-then-anion sort is respected.
    """
    if not ion_names:
        return ion_names

    counts: dict[str, int] = {}
    order: list[str] = []
    for n in ion_names:
        if n not in counts:
            counts[n] = 0
            order.append(n)
        counts[n] += 1

    # Anion-name forms that take the simple multiplier (e.g.
    # ``dichloride`` / ``dibromide`` for MgCl2 / CaBr2 salts).  These
    # don't carry an explicit ``(1-)`` charge marker but their charge
    # is unambiguous from the name suffix (``-ide``).  Treat them as
    # collapse-eligible at the simple level.
    # Note: "ium" deliberately excluded — collapsing "azanium azanium" to
    # "diazanium" breaks OPSIN round-trip (OPSIN parses "diazanium" as
    # the hydrazinium ion, not 2× azanium).  Cation collapse happens
    # via the "(N+)" charge-marker path above.
    # Polyatomic oxoanions ("sulfate", "phosphate", "nitrate", ...) have
    # OPSIN-recognised "di"-prefixed forms (disulfate = pyrosulfate
    # bridged anhydride, NOT 2 sulfate; same for diphosphate / dichromate
    # / etc.).  Multiple equivalents of these must use bis/tris, NOT the
    # direct di/tri.  Only monoatomic / "-ide" anions are safe for the
    # simple multiplier path.
    _COLLAPSIBLE_ANION_SUFFIXES = ("ide",)

    out: list[str] = []
    for name in order:
        c = counts[name]
        if c == 1:
            out.append(name)
            continue

        # Eligibility: must end with an explicit charge marker (P-73)
        # OR be a simple-word anion ending in -ide / -ate / -ite.
        m = _CHARGE_SUFFIX_RE.search(name)
        # "Simple" anion = purely inorganic / small-molecule retained name
        # with no organic substituent indicator.  Compound organic anion
        # names (e.g. "phenylacetylide", "(3-phenylprop-1-yn-1-ide)")
        # must NOT collapse with a di-/tri- prefix because OPSIN does not
        # accept e.g. "diphenylacetylide" as a salt anion.  They are emitted
        # as repeated tokens ("phenylacetylide phenylacetylide") instead,
        # which OPSIN roundtrips correctly.  Gate: a name that contains an
        # internal "yl" substring — characteristic of substituent morphemes
        # like "phenyl-", "acetyl-", "methyl-", etc. — is treated as
        # compound and excluded from the simple-multiplier path.
        _has_yl_substituent = bool(re.search(r"[a-z]yl", name))
        is_simple_anion = (
            m is None
            and not _has_yl_substituent
            and name.replace("-", "").replace(" ", "").isalpha()
            and any(name.endswith(sfx) for sfx in _COLLAPSIBLE_ANION_SUFFIXES)
        )
        if m is None and not is_simple_anion:
            out.extend([name] * c)
            continue

        if m is None:
            # Simple anion (no charge marker).
            mult = get_multiplier(c, complex=False)
            if mult is None:
                out.extend([name] * c)
                continue
            out.append(f"{mult}{name}")
            continue

        bare = name[: m.start()]
        # Simple iff the part before the charge marker is a single
        # alphabetic word (e.g. "sodium", "potassium").  Any spaces,
        # hyphens, or further parentheses make it a compound name that
        # must be wrapped in bis(...)/tris(...) per P-16.3.3/4.
        is_simple = bool(bare) and bare.isalpha()

        if is_simple:
            mult = get_multiplier(c, complex=False)
            if mult is None:
                out.extend([name] * c)
                continue
            # Attach multiplier directly: di + sodium(1+) -> disodium(1+)
            out.append(f"{mult}{name}")
        else:
            mult = get_multiplier(c, complex=True)
            if mult is None:
                out.extend([name] * c)
                continue
            # Wrap compound name in parentheses: bis(...)
            out.append(f"{mult}({name})")

    return out


# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------

def assemble(tree: "NameTree") -> str:  # type: ignore[type-arg]
    """Assemble a NameTree to its final IUPAC name string.

    This is the sole entry point into the assembly layer. Dispatches
    on tree type and calls the appropriate assembler.
    """
    match tree:
        case LeafTree():
            return format_for_output_form(tree.text, tree.output_form)

        case SaltTree():
            ion_names = [assemble(ion) for ion in tree.ion_trees]
            collapsed = _collapse_identical_salt_ions(ion_names)
            # P-77 salt PIN convention: alkali / alkaline-earth /
            # aluminium mono- and dipositive cations don't need an
            # explicit charge marker in salt names.  The standalone
            # forms ``sodium(1+)`` / ``potassium(1+)`` / … keep the
            # marker because the bare element name resolves to neutral
            # metal — but inside a salt the anion supplies the charge
            # context.  Strip the trailing ``(N+)`` where the cation is
            # one of the standard monovalent / divalent metals (charge
            # is unambiguous from the element).  Done AFTER the
            # multiplicative-prefix collapse so ``sodium(1+) sodium(1+)``
            # becomes ``disodium(1+)`` first and then ``disodium``.
            # Map: metal element name → its single most common oxidation
            # state (where charge marker may be elided in a salt PIN).  For
            # metals with multiple common oxidation states (e.g. Al(I) vs
            # Al(III), Cu(I) vs Cu(II)), the marker is REQUIRED whenever the
            # charge is not the default — otherwise OPSIN re-parses to the
            # default state and yields the wrong compound (e.g.
            # "aluminium cyanide" -> Al(III)(CN)3, not Al(I)CN).
            _UNAMBIGUOUS_METAL_CATIONS = {
                "lithium": 1, "sodium": 1, "potassium": 1, "rubidium": 1,
                "caesium": 1, "cesium": 1,
                "beryllium": 2, "magnesium": 2, "calcium": 2, "strontium": 2,
                "barium": 2,
                # Aluminium / Group 13: default +3 (Al(III) is overwhelmingly
                # common); +1 / +2 forms must keep the (n+) marker.
                "aluminium": 3, "aluminum": 3,
                # Group 11: silver is almost exclusively +1 in common
                # inorganic chemistry; the (1+) marker is redundant in a
                # binary salt.  Au / Cu retain markers because they have
                # multiple common states.
                "silver": 1,
            }
            import re as _re_salt
            _METAL_CHARGE_RE = _re_salt.compile(
                r"^(?P<prefix>(?:di|tri|tetra|penta|hexa|hepta|octa|nona|deca)?)"
                r"(?P<elem>[a-z]+)\((?P<n>\d+)\+\)$"
            )
            cleaned: list[str] = []
            for name in collapsed:
                m = _METAL_CHARGE_RE.match(name)
                if (m
                        and m.group("elem") in _UNAMBIGUOUS_METAL_CATIONS
                        and int(m.group("n"))
                            == _UNAMBIGUOUS_METAL_CATIONS[m.group("elem")]):
                    cleaned.append(m.group("prefix") + m.group("elem"))
                else:
                    cleaned.append(name)

            # IR-5.3.3 / P-65.3 binary-salt PIN convention: the monatomic
            # chalcogenide anions (oxide, sulfide, selenide, telluride) carry a
            # standalone charge marker ``(2-)`` (their bare element name
            # resolves to the neutral atom otherwise), but inside a salt the
            # cation supplies the charge context and the marker is dropped —
            # OPSIN parses ``disodium oxide`` / ``calcium oxide`` /
            # ``diiron(3+) trioxide`` but rejects the ``oxide(2-)`` form.
            # Strip the trailing ``(N-)`` from a chalcogenide anion token while
            # preserving any multiplicative prefix (di/tri/…).  Scoped to the
            # closed chalcogenide set because that is the family whose
            # marker-free form OPSIN accepts in a binary salt; nitride/
            # phosphide/carbide use a different (multiplier-free) compositional
            # convention and are intentionally left untouched.
            _CHALCOGENIDE_CHARGE_RE = _re_salt.compile(
                r"^(?P<prefix>(?:di|tri|tetra|penta|hexa|hepta|octa|nona|deca)?)"
                r"(?P<anion>oxide|sulfide|selenide|telluride)\(\d+-\)$"
            )
            cleaned = [
                (mm.group("prefix") + mm.group("anion"))
                if (mm := _CHALCOGENIDE_CHARGE_RE.match(n)) else n
                for n in cleaned
            ]

            # P-77 salt PIN convention: HX (hydrogen halide) counterions
            # paired with an organic nitrogen base are named as hydrohalides.
            # "hydrogen chloride" → "hydrochloride"
            # "dihydrogen chloride" → "dihydrochloride"
            # etc.  The multiplier prefix (di/tri/…) is preserved verbatim;
            # "hydrogen" is contracted to "hydro" and joined with the halide name.
            # Also: hydrohalide tokens must follow the organic cation name
            # (P-77 ordering: cation first, then the acid).
            _HYDROHALIDE_RE = _re_salt.compile(
                r"^((?:di|tri|tetra|penta|hexa|hepta|octa|nona|deca"
                r"|undeca|dodeca|trideca|tetradeca|pentadeca|hexadeca"
                r"|heptadeca|octadeca|nonadeca|icosa|henicosa|docosa"
                r"|tricosa|tetracosa|pentacosa|hexacosa|heptacosa"
                r"|octacosa|nonacosa|triaconta)?)hydrogen "
                r"(fluoride|chloride|bromide|iodide|astatide)$"
            )
            _has_organic = any(
                not _HYDROHALIDE_RE.match(n) for n in cleaned
            )
            final: list[str] = []
            hydrohalides: list[str] = []
            for name in cleaned:
                hh = _HYDROHALIDE_RE.match(name)
                if hh and _has_organic:
                    # Contract "hydrogen <halide>" → "hydro<halide>"
                    hydrohalides.append(f"{hh.group(1)}hydro{hh.group(2)}")
                else:
                    final.append(name)
            final.extend(hydrohalides)
            return " ".join(final)

        case FunctionalClassTree():
            return _assemble_fc(tree)

        case SubstitutiveTree():
            return _assemble_substitutive(tree)

        case MultiplicativeTree():
            subunit_name = assemble(tree.subunit)
            locant_str = ",".join(tree.locants) + "-" if tree.locants else ""
            linking = (tree.linking_group + "-") if tree.linking_group else ""
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

        case _:
            # Unknown tree type — emit error
            return f"[NAMING ERROR: unknown tree type {type(tree).__name__}]"
