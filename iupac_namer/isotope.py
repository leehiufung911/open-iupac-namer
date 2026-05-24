"""
iupac_namer/isotope.py

Isotope-label perception and emission (Stage 6 R1-D).

IUPAC 2013 Blue Book — "Isotopically Modified Compounds" — defines two
equivalent notations for isotopic labels.  This module supports the IUPAC
bracketed-element style (the audit canonical):

    (²H)methanol              — one deuterium on methanol
    (²H₄)methanol             — four deuteria (perdeuteromethanol)
    (¹³C)methane              — single ¹³C
    (1-¹³C)ethan-1-ol         — locanted label at C1 of ethanol
    (1-¹⁵N)-1H-indole         — locanted on N1 of 1H-indole
    (1R)-(1-²H)ethan-1-ol     — combined stereo + isotope

Labels are collected per (locant, element, mass_number) and grouped so that
three deuteria on the same carbon produce a single bracket with count=3.

The module exposes two entry points:

1.  ``collect_isotope_labels(mol, atom_to_locant, fg_anchor_map=None)`` —
    walks the molecule and returns a tuple of :class:`IsotopeLabel` ready
    to be stashed on a :class:`SubstitutiveTree`.
2.  ``render_isotope_labels(labels)`` — formats a label tuple to the
    assembly-ready string ``"(²H)-"`` / ``"(²H₄)"`` / ``"(1-¹³C)-"`` etc.

The module does not touch the mol and does not mutate state — all
functions are pure.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping

from iupac_namer.types import IsotopeLabel, Locant


# ---------------------------------------------------------------------------
# Unicode superscript/subscript digit translation
# ---------------------------------------------------------------------------
#
# IUPAC isotope labels cite the mass number in superscript and the atom
# count in subscript.  Unicode has a dedicated code-point for every digit
# (U+2070, U+00B9, U+00B2, U+00B3, U+2074..U+2079 for superscript;
# U+2080..U+2089 for subscript) so we can emit "¹³C", "²H₄", "¹⁴C", "¹⁵N"
# directly.

_SUPERSCRIPT_DIGITS: dict[str, str] = {
    "0": "\u2070",
    "1": "\u00b9",
    "2": "\u00b2",
    "3": "\u00b3",
    "4": "\u2074",
    "5": "\u2075",
    "6": "\u2076",
    "7": "\u2077",
    "8": "\u2078",
    "9": "\u2079",
}

_SUBSCRIPT_DIGITS: dict[str, str] = {
    "0": "\u2080",
    "1": "\u2081",
    "2": "\u2082",
    "3": "\u2083",
    "4": "\u2084",
    "5": "\u2085",
    "6": "\u2086",
    "7": "\u2087",
    "8": "\u2088",
    "9": "\u2089",
}


def _superscript(n: int) -> str:
    """Return *n* rendered in Unicode superscript digits."""
    return "".join(_SUPERSCRIPT_DIGITS[d] for d in str(n))


def _subscript(n: int) -> str:
    """Return *n* rendered in Unicode subscript digits."""
    return "".join(_SUBSCRIPT_DIGITS[d] for d in str(n))


# ---------------------------------------------------------------------------
# collect_isotope_labels
# ---------------------------------------------------------------------------


def collect_isotope_labels(
    mol: object,
    atom_to_locant: Mapping[int, Locant],
    fg_anchor_map: Mapping[int, Locant] | None = None,
) -> tuple[IsotopeLabel, ...]:
    """Extract IUPAC isotope labels from *mol* for the given parent.

    For each atom with ``GetIsotope() > 0``:

    * Element ``"H"`` → the label targets the atom's single heavy neighbour,
      not the hydrogen itself (deuterium and tritium are *hydrogen isotopes
      on the parent position*).  If the neighbour lacks a locant (e.g. the
      H is attached to an off-parent atom), the label is dropped.
    * Any other element → the label targets the atom itself.  If the atom
      is not in ``atom_to_locant`` (not a parent backbone atom) the label
      is dropped.

    The optional ``fg_anchor_map`` extends the set of label-addressable
    atoms to include suffix-FG atoms that are structurally owned by the
    parent (e.g. the oxygen of an "-ol" alcohol, the nitrogens of an
    "-amine").  Each FG atom maps to the locant of its anchor so that
    ``(²H₄)methanol`` captures both the methyl-CD3 and the hydroxyl-OD
    deuteriums under locant 1.

    Labels are grouped by ``(locant, element, mass_number)`` and ``count``
    equals the number of isotope-bearing atoms in that bucket.

    The returned tuple is sorted by ``(locant-as-string, mass_number,
    element)`` for deterministic output.
    """
    # bucket: (locant, element, mass) -> count
    buckets: dict[tuple[Locant | None, str, int], int] = defaultdict(int)

    for atom in mol.GetAtoms():                      # type: ignore[attr-defined]
        iso = atom.GetIsotope()
        if iso == 0:
            continue
        element = atom.GetSymbol()
        if element == "H":
            # Attach the label to the heavy neighbour (CAS/IUPAC convention:
            # D/T label sits on the *heavy* position, not the H index).
            neighbours = list(atom.GetNeighbors())
            if len(neighbours) != 1:
                # Unbound or multiply-bonded H (rare) — can't locate the
                # parent position, skip silently.
                continue
            target_idx = neighbours[0].GetIdx()
        else:
            target_idx = atom.GetIdx()

        # Primary lookup: parent-backbone locant.  Fall-back: suffix-FG
        # anchor locant (e.g. the O of methanol maps back to C1).
        locant = atom_to_locant.get(target_idx)
        if locant is None and fg_anchor_map is not None:
            locant = fg_anchor_map.get(target_idx)
            # Heteroatom-locant rule (P-82.2.3.1.1 / Blue Book isotope ref):
            # a D/T (or other isotope) sitting directly on a suffix-FG
            # heteroatom — the -OH oxygen, -SH sulfur, -NH nitrogen — is
            # cited with the italic element-symbol locant ("O-2H", "S-2H",
            # "N-2H"), NOT the numeric locant of the anchor carbon.  The
            # fg_anchor_map maps those heteroatoms onto their anchor's
            # numeric locant for backbone purposes; override that here when
            # the labelled position is itself the heteroatom (i.e. an
            # isotopic H whose sole heavy neighbour is the FG heteroatom).
            if locant is not None:
                target_atom = mol.GetAtomWithIdx(target_idx)  # type: ignore[attr-defined]
                target_sym = target_atom.GetSymbol()
                if (element == "H"
                        and target_sym != "C"
                        and target_idx not in atom_to_locant):
                    locant = Locant.hetero(target_sym)
        # If the target atom has no addressable locant on this parent,
        # drop the label rather than emitting a guess.
        if locant is None:
            continue

        key = (locant, element, iso)
        buckets[key] += 1

    def _sort_key(item: tuple[tuple[Locant | None, str, int], int]) -> tuple[str, int, str]:
        # ``sorted(buckets.items(), key=_sort_key)`` hands us a
        # ``((locant, element, mass), count)`` item — pull the key tuple out.
        loc, elem, mass = item[0]
        loc_str = "" if loc is None else str(loc)
        # Pad numeric locants for natural order: "10" > "2"
        try:
            loc_sort = (0, int(loc_str)) if loc_str.isdigit() else (1, loc_str)
        except ValueError:
            loc_sort = (1, loc_str)
        return (f"{loc_sort[0]}:{loc_sort[1]!s:>10}", mass, elem)

    return tuple(
        IsotopeLabel(locant=loc, element=elem, mass_number=mass, count=count)
        for (loc, elem, mass), count in sorted(buckets.items(), key=_sort_key)
    )


# ---------------------------------------------------------------------------
# render_isotope_labels
# ---------------------------------------------------------------------------


def render_isotope_label(label: IsotopeLabel) -> str:
    """Render a single IsotopeLabel as its inner bracket text.

    Stage 16 R16-B switched the renderer from unicode super/subscripts
    (``²H``, ``¹³C``, ``²H₄``) to ASCII (``2H``, ``13C``, ``2H4``)
    because OPSIN's parser does not accept unicode digits in isotope
    labels — names like ``(1-¹³C)methane`` and ``(²H₄)methanol`` were
    OPSIN-unparseable, so isotopologue inputs (``[13CH4]``,
    ``[2H][2H][2H][2H]C``, etc.) couldn't round-trip even though the
    engine produced a structurally-correct name.

    For count > 1 with an explicit locant, OPSIN expects a VERBOSE
    locant list (``(1,1,1,1-2H4)methane``) rather than a count subscript
    on a single locant (``(1-2H4)methane`` — rejected by OPSIN).  The
    whole-molecule form ``(2H4)methane`` (no locants, count subscript
    on element) is accepted.  We therefore preserve the locant list
    when one is present and the count > 1.

    Examples:
        IsotopeLabel(None, "H", 2, 1)  -> "2H"
        IsotopeLabel(None, "H", 2, 4)  -> "2H4"
        IsotopeLabel("1",  "C", 13, 1) -> "1-13C"
        IsotopeLabel("2",  "C", 13, 1) -> "2-13C"
        IsotopeLabel(None, "N", 15, 1) -> "15N"
        IsotopeLabel("1",  "H", 2, 4)  -> "1,1,1,1-2H4"
    """
    mass = str(label.mass_number)
    count_str = str(label.count) if label.count > 1 else ""
    body = f"{mass}{label.element}{count_str}"
    if label.locant is None:
        return body
    if label.count == 1:
        return f"{label.locant}-{body}"
    # count > 1 with an explicit locant: OPSIN requires the locant
    # repeated for each isotope atom.  All atoms share the same locant
    # because perception bucketed them together — emit the locant list.
    locant_list = ",".join([str(label.locant)] * label.count)
    return f"{locant_list}-{body}"


def render_isotope_labels(labels: Iterable[IsotopeLabel]) -> str:
    """Render a sequence of IsotopeLabel as the IUPAC bracket prefix.

    Multiple labels are combined inside a single pair of parentheses,
    comma-separated.  Returns the empty string when no labels are
    supplied (so the caller can unconditionally concatenate).

    Examples:
        ()                                       -> ""
        (IsotopeLabel(None,"H",2,4),)            -> "(²H₄)"
        (IsotopeLabel("1","C",13,1),)            -> "(1-¹³C)"
        (IsotopeLabel("1","H",2,1),
         IsotopeLabel("2","C",13,1),)            -> "(1-²H,2-¹³C)"
    """
    labels = tuple(labels)
    if not labels:
        return ""
    body = ",".join(render_isotope_label(lbl) for lbl in labels)
    return f"({body})"


__all__ = [
    "collect_isotope_labels",
    "render_isotope_label",
    "render_isotope_labels",
]
