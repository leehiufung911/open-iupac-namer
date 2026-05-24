"""
iupac_namer/perception/fg/chalcogen_acid_modifiers.py
=====================================================

Generative composer for *modified* organic chalcogen oxoacids — the
``-sulfinic`` / ``-sulfonic`` / ``-seleninic`` / ``-selenonic`` /
``-tellurinic`` / ``-telluronic`` acid family with functional-replacement
infixes (P-65.3.1 / P-66, OPSIN ``infixes.xml``).

The plain acids and a handful of single-modifier variants (``sulfinimidic``,
``sulfonothioic S-acid`` …) are already enumerated as individual SMARTS
entries in ``data/functional_groups.json``.  This module COMPUTES the name
for the **combinatorial remainder** that the static table does not list —
seleno / telluro / peroxo replacements, Se/Te-base imido / hydrazono forms,
and any combination of these — directly from the perceived structure.

Naming model (P-65.3.1)
-----------------------
An organic chalcogen oxoacid centre ``X`` (X = S, Se, Te) bonded to exactly
one carbon carries:

* ``tier`` doubly-bonded *oxo positions* — one for the ``-in-`` acids
  (sulfinic / seleninic / tellurinic), two for the ``-on-`` acids
  (sulfonic / selenonic / telluronic);
* one singly-bonded *acidic position* (the ``-OH`` of the parent acid).

Functional replacement modifies these positions:

* an oxo position ``=O`` may become ``=S`` (thio), ``=Se`` (seleno),
  ``=Te`` (telluro), ``=NH`` (imido), or ``=N-NH2`` (hydrazono);
* the acidic position ``-OH`` may become ``-SH`` / ``-SeH`` / ``-TeH``
  (the chalcogen-acid tautomer, flagged by the italic ``S-`` / ``Se-`` /
  ``Te-acid`` element locant) or ``-O-OH`` (peroxo).

The name is the base acid name with the replacement infixes spliced in
ahead of the ``ic acid`` ending, cited in the canonical functional-
replacement order ``peroxo, hydrazono, imido, thio, seleno, telluro`` with
``di`` / ``tri`` multipliers for repeats and the standard ``o`` connecting
vowel (elided before another ``o``).

Dispatch
--------
``engine.name`` calls :func:`compute_name` in the whole-molecule oxoacid
shortcut region, BEFORE the free-valence guard (the hypervalent S/Se/Te
centre may carry RDKit-modelled radical electrons) and before the generic
plan search (so it pre-empts a wrong substitutive name such as
``(selanylsulfonyl)ethane`` for the Se-acid tautomer).  It returns ``None``
for anything that is not a modified organic chalcogen oxoacid, and — by
design — for the exact signatures already covered by the static SMARTS
table, so those keep their existing (PIN-spelled) handling untouched.

No molecule-specific branches: every decision keys on a structural feature
(centre element, oxo tier, replacement element, acidic-position element).
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Structural tables
# ---------------------------------------------------------------------------

# Centre element -> (tier-1 base stem, tier-2 base stem).  The base stems are
# the part of the parent acid name before the ``ic acid`` ending.
_BASE_STEM = {
    "S": {1: "sulfin", 2: "sulfon"},
    "Se": {1: "selenin", 2: "selenon"},
    "Te": {1: "tellurin", 2: "telluron"},
}

# Canonical infix citation order and the rendered base of each infix (the
# multiplier and connecting ``o`` are added by the renderer).
_INFIX_ORDER = ("peroxo", "hydrazono", "imido", "thio", "seleno", "telluro")
_INFIX_BASE = {
    "peroxo": "peroxo",
    "hydrazono": "hydrazon",
    "imido": "imid",
    "thio": "thio",
    "seleno": "seleno",
    "telluro": "telluro",
}

_MULT = {1: "", 2: "di", 3: "tri", 4: "tetra", 5: "penta"}
_VOWELS = frozenset("aeiou")

# Element of an acidic-position atom -> the italic element-locant designator
# word ("" for oxygen, which is the unmarked default).
_ACID_DESIGNATOR = {"O": "", "S": "S-acid", "Se": "Se-acid", "Te": "Te-acid"}

# Element symbols that may sit at an oxo / acidic position via functional
# replacement (besides oxygen / nitrogen).
_CHALCOGENS = frozenset({"S", "Se", "Te"})


# Signatures already enumerated in ``data/functional_groups.json`` — those
# keep their existing (statically PIN-spelled) handling, so this generator
# declines them.  Each key is
# ``(centre_element, sorted oxo-position element/role tuple, acidic-element)``
# where the oxo role is the atom symbol ("O"/"S"/"N") with "NN" marking the
# hydrazono =N-NH2 oxo.  (Peroxo never appears here — no static peroxo entry.)
_DATA_COVERED: frozenset[tuple] = frozenset({
    # plain acids
    ("S", ("O",), "O"), ("S", ("O", "O"), "O"),
    ("Se", ("O",), "O"), ("Se", ("O", "O"), "O"),
    ("Te", ("O",), "O"), ("Te", ("O", "O"), "O"),
    # imido / diimido (S, Se)
    ("S", ("N",), "O"), ("S", ("N", "O"), "O"), ("S", ("N", "N"), "O"),
    ("Se", ("N", "O"), "O"), ("Se", ("N", "N"), "O"),
    # thio tautomers (S only, in data)
    ("S", ("S",), "O"),                 # sulfinothioic O-acid
    ("S", ("O", "O"), "S"),             # sulfonothioic S-acid
    # hydrazono (S)
    ("S", ("NN",), "O"), ("S", ("NN", "O"), "O"),
})


class _Decline(Exception):
    """Internal: the molecule is not a composer-eligible modified acid."""


@dataclass(frozen=True)
class _Centre:
    idx: int
    element: str
    carbon_idx: int
    tier: int
    # oxo-position roles, each one of: "O","S","Se","Te" (chalcogen oxo),
    # "imido" (=NH), "hydrazono" (=N-NH2)
    oxo_roles: tuple[str, ...]
    acidic_element: str            # element of the acidic-position atom
    acidic_is_peroxy: bool         # acidic position is -O-OH (peroxo)


# ---------------------------------------------------------------------------
# Perception
# ---------------------------------------------------------------------------

def _perceive(mol) -> _Centre:
    from rdkit import Chem  # local import

    if mol.GetNumAtoms() == 0:
        raise _Decline("empty")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise _Decline("multi-fragment")
    if Chem.GetFormalCharge(mol) != 0:
        raise _Decline("charged")

    # Locate the single S/Se/Te acid centre bonded to exactly one carbon.
    centres = [
        a for a in mol.GetAtoms()
        if a.GetSymbol() in _CHALCOGENS and not a.IsInRing()
    ]
    # Pick centres that are bonded to a carbon (the organic acid centre); a
    # second chalcogen elsewhere (e.g. a thioether substituent) is allowed
    # only if it is NOT itself an acid centre, which the strict neighbour
    # classification below enforces by raising on anything unexpected.
    acid_centres = [
        a for a in centres
        if any(nb.GetSymbol() == "C" for nb in a.GetNeighbors())
        and not _is_pure_substituent_chalcogen(mol, a)
    ]
    if len(acid_centres) != 1:
        raise _Decline("not exactly one acid centre")
    centre = acid_centres[0]
    element = centre.GetSymbol()
    c_idx = centre.GetIdx()
    if centre.GetFormalCharge() != 0:
        raise _Decline("charged centre")

    carbon_neighbours = [nb for nb in centre.GetNeighbors()
                         if nb.GetSymbol() == "C"]
    if len(carbon_neighbours) != 1:
        raise _Decline("centre must bind exactly one carbon")
    carbon_idx = carbon_neighbours[0].GetIdx()
    if mol.GetBondBetweenAtoms(c_idx, carbon_idx).GetBondTypeAsDouble() != 1.0:
        raise _Decline("non-single bond to carbon")

    oxo_roles: list[str] = []
    acidic_element: str | None = None
    acidic_is_peroxy = False

    for nb in centre.GetNeighbors():
        ni = nb.GetIdx()
        if ni == carbon_idx:
            continue
        bond = mol.GetBondBetweenAtoms(c_idx, ni)
        order = bond.GetBondTypeAsDouble()
        sym = nb.GetSymbol()

        if order == 2.0:
            # oxo position
            if sym == "O" and nb.GetDegree() == 1 and nb.GetFormalCharge() == 0:
                oxo_roles.append("O")
            elif sym in _CHALCOGENS and nb.GetDegree() == 1 \
                    and nb.GetFormalCharge() == 0:
                oxo_roles.append(sym)
            elif sym == "N":
                role = _classify_imido_nitrogen(mol, nb, c_idx)
                if role is None:
                    raise _Decline("unexpected =N substituent")
                oxo_roles.append(role)
            else:
                raise _Decline("unexpected oxo position")
        elif order == 1.0:
            # acidic position: terminal -OH/-SH/-SeH/-TeH or -O-OH peroxy
            if acidic_element is not None:
                raise _Decline("more than one acidic position")
            if sym == "O" and nb.GetDegree() == 2 and nb.GetFormalCharge() == 0 \
                    and nb.GetTotalNumHs() == 0:
                # -O-OH peroxy bridge
                others = [x for x in nb.GetNeighbors() if x.GetIdx() != c_idx]
                if len(others) != 1 or others[0].GetSymbol() != "O":
                    raise _Decline("non-peroxy bridge at acidic position")
                term = others[0]
                if term.GetDegree() != 1 or term.GetTotalNumHs() != 1 \
                        or term.GetFormalCharge() != 0:
                    raise _Decline("bad terminal peroxy")
                acidic_element = "O"
                acidic_is_peroxy = True
            elif (sym == "O" or sym in _CHALCOGENS) and nb.GetDegree() == 1 \
                    and nb.GetTotalNumHs() == 1 and nb.GetFormalCharge() == 0:
                acidic_element = sym
            else:
                raise _Decline("unexpected acidic position")
        else:
            raise _Decline("unexpected bond order at centre")

    tier = len(oxo_roles)
    if tier not in (1, 2):
        raise _Decline("tier must be 1 or 2")
    if acidic_element is None:
        raise _Decline("no acidic position")

    return _Centre(
        idx=c_idx, element=element, carbon_idx=carbon_idx, tier=tier,
        oxo_roles=tuple(oxo_roles), acidic_element=acidic_element,
        acidic_is_peroxy=acidic_is_peroxy,
    )


def _is_pure_substituent_chalcogen(mol, atom) -> bool:
    """A chalcogen that is plainly a substituent (thioether / selenoether /
    thiol etc.), not an acid centre: degree<=2, no double bonds to O/N, and
    no terminal =O/=N — i.e. it carries no acid-defining oxo position."""
    has_oxo = False
    for nb in atom.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(atom.GetIdx(), nb.GetIdx())
        if bond.GetBondTypeAsDouble() == 2.0 and nb.GetSymbol() in ("O", "N"):
            has_oxo = True
        if bond.GetBondTypeAsDouble() == 2.0 and nb.GetSymbol() in _CHALCOGENS:
            has_oxo = True
    return not has_oxo


def _classify_imido_nitrogen(mol, n_atom, centre_idx) -> str | None:
    """Return "imido" for =N-H / =N-R or "hydrazono" for =N-NH2, else None."""
    if n_atom.GetFormalCharge() != 0:
        return None
    others = [x for x in n_atom.GetNeighbors() if x.GetIdx() != centre_idx]
    if not others:
        # =NH (no heavy neighbour) — imido
        return "imido" if n_atom.GetTotalNumHs() >= 1 else None
    if len(others) != 1:
        return None
    other = others[0]
    bond = mol.GetBondBetweenAtoms(n_atom.GetIdx(), other.GetIdx())
    if bond.GetBondTypeAsDouble() != 1.0:
        return None
    if other.GetSymbol() == "N" and other.GetDegree() == 1 \
            and other.GetTotalNumHs() == 2 and other.GetFormalCharge() == 0:
        return "hydrazono"
    # =N-C / =N-other: N-substituted imido is out of this generator's scope
    # (it needs an N-substituent prefix; decline so plan search handles it).
    return None


# ---------------------------------------------------------------------------
# Modifier accounting
# ---------------------------------------------------------------------------

def _data_signature(c: _Centre) -> tuple:
    """The (element, sorted-oxo-roles, acidic-element) key used to detect
    signatures already enumerated in the static SMARTS table.  Hydrazono is
    keyed as "NN" and imido as "N" to match ``_DATA_COVERED``."""
    role_map = {"imido": "N", "hydrazono": "NN"}
    oxo = tuple(sorted(role_map.get(r, r) for r in c.oxo_roles))
    # Peroxy is never in the static SMARTS table, so mark it distinctly to
    # keep it out of the (plain-acid) "O" signature class.
    acidic = "OO" if c.acidic_is_peroxy else c.acidic_element
    return (c.element, oxo, acidic)


def _modifier_counts(c: _Centre) -> dict[str, int]:
    """Count each functional-replacement infix across oxo + acidic positions."""
    counts: dict[str, int] = {k: 0 for k in _INFIX_ORDER}
    chalc_infix = {"S": "thio", "Se": "seleno", "Te": "telluro"}
    for role in c.oxo_roles:
        if role == "O":
            continue
        if role == "imido":
            counts["imido"] += 1
        elif role == "hydrazono":
            counts["hydrazono"] += 1
        elif role in chalc_infix:
            counts[chalc_infix[role]] += 1
    # acidic position
    if c.acidic_is_peroxy:
        counts["peroxo"] += 1
    elif c.acidic_element in chalc_infix:
        counts[chalc_infix[c.acidic_element]] += 1
    return counts


# ---------------------------------------------------------------------------
# Name rendering
# ---------------------------------------------------------------------------

def _render_modified_stem(base_stem: str, counts: dict[str, int]) -> str:
    """Splice the replacement infixes into ``base_stem`` and add ``ic acid``.

    Canonical order (``_INFIX_ORDER``), ``di``/``tri`` multipliers for
    repeats, ``o`` connecting vowel elided before another ``o`` and before a
    vowel-initial infix base (``imid``)."""
    body = base_stem
    for tok in _INFIX_ORDER:
        n = counts.get(tok, 0)
        if not n:
            continue
        core = _MULT.get(n, "") + _INFIX_BASE[tok]
        if not core:
            continue
        # connecting 'o' elides before a vowel-initial core (e.g. 'imid')
        connector = "" if core[0] in _VOWELS else "o"
        frag = connector + core
        # collapse a doubled 'o' at the junction (othio + oseleno -> othioseleno)
        if frag and body.endswith("o") and frag.startswith("o"):
            frag = frag[1:]
        body += frag
    return body + "ic acid"


def _base_acid_name(mol, centre: _Centre, strategy, session, depth) -> str | None:
    """Rebuild ``mol`` with the chalcogen centre as its *plain* oxoacid
    (``tier`` × =O plus one -OH, carbon parent intact) and name it, returning
    the base acid surface form (e.g. ``"propane-1-sulfinic acid"``)."""
    from rdkit import Chem
    from iupac_namer.engine import name as _recursive_name
    from iupac_namer.assembly import assemble
    from iupac_namer.types import OutputForm

    rw = Chem.RWMol(mol)
    c_idx = centre.idx

    # Atoms to delete: everything reachable from the centre that is NOT the
    # carbon parent side.  Identify the acid-side atoms (all neighbours of the
    # centre except the carbon, plus their terminal hangers-on like the second
    # peroxy O or the hydrazono terminal N).
    acid_side: set[int] = set()
    for nb in mol.GetAtomWithIdx(c_idx).GetNeighbors():
        if nb.GetIdx() == centre.carbon_idx:
            continue
        acid_side.add(nb.GetIdx())
        for nb2 in nb.GetNeighbors():
            if nb2.GetIdx() != c_idx and nb2.GetAtomicNum() > 1:
                acid_side.add(nb2.GetIdx())

    to_delete = sorted(acid_side, reverse=True)
    centre_new = c_idx - sum(1 for d in to_delete if d < c_idx)
    for idx in to_delete:
        rw.RemoveAtom(idx)
    if centre_new < 0 or centre_new >= rw.GetNumAtoms():
        return None

    centre_atom = rw.GetAtomWithIdx(centre_new)
    centre_atom.SetNoImplicit(True)
    centre_atom.SetNumExplicitHs(0)
    centre_atom.SetFormalCharge(0)
    # Attach the plain oxoacid environment: tier × =O, one -OH.
    for _ in range(centre.tier):
        o = rw.AddAtom(Chem.Atom(8))
        rw.AddBond(centre_new, o, Chem.BondType.DOUBLE)
    oh = rw.AddAtom(Chem.Atom(8))
    rw.AddBond(centre_new, oh, Chem.BondType.SINGLE)
    rw.GetAtomWithIdx(oh).SetNumExplicitHs(1)
    try:
        m = rw.GetMol()
        Chem.SanitizeMol(m)
    except Exception:
        return None

    try:
        tree = _recursive_name(
            m, strategy, OutputForm.STANDALONE, free_valence=None,
            decision_ctx=None, _session=session, _depth=depth + 1,
        )
        nm = assemble(tree)
    except Exception:
        return None
    if not nm or "NAMING ERROR" in nm:
        return None
    return nm


# Base acid surface words keyed by (element, tier) — used to locate and strip
# the ``…ic acid`` ending of the recursively-named base acid.
_BASE_WORD = {
    ("S", 1): "sulfinic acid", ("S", 2): "sulfonic acid",
    ("Se", 1): "seleninic acid", ("Se", 2): "selenonic acid",
    ("Te", 1): "tellurinic acid", ("Te", 2): "telluronic acid",
}


def _compose(mol, centre: _Centre, strategy, session, depth) -> str | None:
    counts = _modifier_counts(centre)
    if not any(counts.values()):
        return None  # plain acid — let the static FG table handle it

    base_word = _BASE_WORD[(centre.element, centre.tier)]
    base_name = _base_acid_name(mol, centre, strategy, session, depth)
    if base_name is None or not base_name.endswith(base_word):
        return None
    prefix_part = base_name[: -len(base_word)]   # e.g. "propane-1-"
    base_stem = base_word[: -len("ic acid")]     # e.g. "sulfin"

    modified = _render_modified_stem(base_stem, counts)

    designator = ""
    if not centre.acidic_is_peroxy:
        designator = _ACID_DESIGNATOR.get(centre.acidic_element, "")
        if designator == "" and centre.acidic_element != "O":
            return None  # unknown acidic element
    name = f"{prefix_part}{modified}"
    if designator:
        # replace the trailing "acid" with the element-locant designator word
        assert name.endswith("ic acid")
        name = name[: -len("acid")] + designator
    return name


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_name(mol, strategy=None, session=None, depth: int = 0) -> str | None:
    """Compute the IUPAC name of a modified organic chalcogen oxoacid, or
    ``None`` when *mol* is not one (or is a signature the static SMARTS table
    already covers)."""
    if mol is None:
        return None
    try:
        centre = _perceive(mol)
    except _Decline:
        return None
    except Exception:  # pragma: no cover — never break the engine
        return None

    if _data_signature(centre) in _DATA_COVERED:
        return None

    if strategy is None:
        from iupac_namer.strategy import IUPACCanonical
        strategy = IUPACCanonical()

    try:
        return _compose(mol, centre, strategy, session, depth)
    except _Decline:
        return None
    except Exception:  # pragma: no cover
        return None
