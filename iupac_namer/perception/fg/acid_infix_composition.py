"""
iupac_namer/perception/fg/acid_infix_composition.py
===================================================

Table-driven acid-infix dispatcher (Stage 6 R1-F, root-cause #10 from the
OPSIN coverage audit).

OPSIN's ``infixes.xml`` encodes 28 atomic replacement rules that turn a simple
acid (``C(=O)O``, ``S(=O)(=O)O``, ``P(=O)(O)O`` …) into a modified acid name
such as ``ethanenitridic acid`` or ``acetyl isotellurocyanate``.  Our engine
already covers the 15 rules that round-trip through the standard FG entries in
``data/functional_groups.json`` (``-oyl chloride``, ``-amide``, ``-ohydrazide``
etc.).  This module closes the remaining 13 by detecting the post-replacement
graph directly and emitting a pre-composed ``LeafTree`` name.

Design notes
------------

* The rule set is loaded once from ``data/infix_rules.json`` (derived from
  OPSIN's ``infixes.xml``) and cached.  Rules with ``covered_via`` set are
  skipped at detection time — they are already handled by the native plan
  search.
* The engine dispatches this module **after** the specialised functional
  parent handlers (urea, thiourea, sulfamide, fulminic …) and **before** the
  generic plan search.  Matching is intentionally narrow (exact post-
  replacement graph with no other senior FG present) so the detector is a
  targeted safety net, never a pre-empt of a correct native name.
* For class-word form (``acetyl isocyanide``) we rename the acyl side by
  rebuilding the parent as the plain carboxylic acid and running that through
  the engine as ``STANDALONE``, then transforming ``-oic acid`` / ``-carboxylic
  acid`` / retained acid names (``acetic acid``, ``benzoic acid``, …) into
  their acyl equivalents.
* For embedded-stem form (``phosphoronitridic acid``, ``ethanotelluroic
  acid``, ``ethanohydroximic acid``) we splice the infix into the acid-form
  name.

The module must NOT regress existing infix handling; the smoke tests in
``tests/test_acid_infix_composition.py`` pin the before/after coverage for
all 28 rules.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from iupac_namer.types import (
    Choice,
    DecisionContext,
    FreeValenceInfo,
    LeafTree,
    OutputForm,
)

# ---------------------------------------------------------------------------
# Rule loading
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
_RULES_CACHE: list["InfixRule"] | None = None


@dataclass(frozen=True)
class InfixRule:
    """One row of ``data/infix_rules.json``.

    ``opsin_from`` / ``opsin_to`` retain the raw pattern strings from OPSIN's
    ``infixes.xml`` so the match logic stays transparent to future updates.
    """

    alias: str
    opsin_from: str
    opsin_to: str
    replaces_oh: bool
    class_word: str | None
    ic_stem: str | None
    applicable_to: tuple[str, ...]
    covered_via: str | None = None

    @property
    def is_handled_natively(self) -> bool:
        """Whether the existing plan search already names this infix."""
        return self.covered_via is not None


def load_infix_rules() -> list[InfixRule]:
    """Load and cache the infix rule table from ``data/infix_rules.json``."""
    global _RULES_CACHE
    if _RULES_CACHE is None:
        path = _DATA_DIR / "infix_rules.json"
        with path.open(encoding="utf-8") as fh:
            raw = json.load(fh)
        rules: list[InfixRule] = []
        for entry in raw.get("rules", ()):
            rules.append(
                InfixRule(
                    alias=entry["alias"],
                    opsin_from=entry["opsin_from"],
                    opsin_to=entry["opsin_to"],
                    replaces_oh=bool(entry["replaces_oh"]),
                    class_word=entry.get("class_word"),
                    ic_stem=entry.get("ic_stem"),
                    applicable_to=tuple(entry.get("applicable_to", ())),
                    covered_via=entry.get("covered_via"),
                )
            )
        _RULES_CACHE = rules
    return _RULES_CACHE


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------


def detect_acid_infix_composition(
    mol,
    output_form: OutputForm,
    free_valence: FreeValenceInfo | None,
    decision_ctx: DecisionContext | None,
    strategy,
    session,
    depth: int,
) -> LeafTree | None:
    """Return a ``LeafTree`` with a pre-composed acid-infix name, or ``None``
    to fall through to the regular plan search.

    Only fires for ``OutputForm.STANDALONE`` with no free valence — the acid
    analogue names are always top-level structures.
    """
    if output_form != OutputForm.STANDALONE or free_valence is not None:
        return None
    rules = [r for r in load_infix_rules() if not r.is_handled_natively]
    for rule in rules:
        leaf = _try_rule(rule, mol, decision_ctx, strategy, session, depth)
        if leaf is not None:
            return leaf
    return None


def _try_rule(
    rule: InfixRule,
    mol,
    decision_ctx: DecisionContext | None,
    strategy,
    session,
    depth: int,
) -> LeafTree | None:
    alias = rule.alias
    if alias in _CLASS_WORD_PATTERNS:
        return _detect_class_word_rule(
            rule, mol, decision_ctx, strategy, session, depth
        )
    if alias == "nitrid":
        return _detect_nitrid(mol, decision_ctx)
    if alias == "tellur":
        return _detect_tellur(mol, decision_ctx, strategy, session, depth)
    if alias == "hydroxim":
        return _detect_hydroxim(mol, decision_ctx, strategy, session, depth)
    if alias == "ditelluroperox":
        return _detect_ditelluroperox(
            mol, decision_ctx, strategy, session, depth
        )
    return None


# ---------------------------------------------------------------------------
# Graph-carving helpers
# ---------------------------------------------------------------------------


def _reach_avoiding(mol, start: int, blocked: set[int]) -> frozenset[int]:
    """BFS over the heavy-atom graph starting at ``start``, never crossing an
    atom in ``blocked``.  Used to collect the R group of an acyl parent."""
    seen: set[int] = {start}
    frontier = [start]
    while frontier:
        nxt: list[int] = []
        for idx in frontier:
            for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
                if nb.GetAtomicNum() == 1:
                    continue
                if nb.GetIdx() in blocked:
                    continue
                if nb.GetIdx() in seen:
                    continue
                seen.add(nb.GetIdx())
                nxt.append(nb.GetIdx())
        frontier = nxt
    return frozenset(seen)


def _rebuild_as_acid(mol, carbonyl_c_idx: int, keep_atoms: Iterable[int]):
    """Return an RDKit mol representing R-C(=O)-OH where R is the subgraph
    given by ``keep_atoms`` (excluding the carbonyl carbon).

    The post-deletion index of the carbonyl C is pre-computed by counting how
    many atoms with a smaller original index will be removed; this avoids
    brittle "find a sp2-C with missing =O" heuristics that failed on aromatic
    rings.  After deletion we attach a fresh =O and -OH, and sanitise.
    """
    from rdkit import Chem

    rw = Chem.RWMol(mol)
    keep = set(keep_atoms) | {carbonyl_c_idx}
    to_delete_idx = sorted(
        [a.GetIdx() for a in rw.GetAtoms() if a.GetIdx() not in keep],
        reverse=True,
    )
    carbonyl_c_new = carbonyl_c_idx - sum(
        1 for d in to_delete_idx if d < carbonyl_c_idx
    )
    for idx in to_delete_idx:
        rw.RemoveAtom(idx)
    if carbonyl_c_new < 0 or carbonyl_c_new >= rw.GetNumAtoms():
        return None
    oxo = rw.AddAtom(Chem.Atom(8))
    rw.AddBond(carbonyl_c_new, oxo, Chem.BondType.DOUBLE)
    oh = rw.AddAtom(Chem.Atom(8))
    rw.AddBond(carbonyl_c_new, oh, Chem.BondType.SINGLE)
    rw.GetAtomWithIdx(oh).SetNumExplicitHs(1)
    try:
        Chem.SanitizeMol(rw)
    except Exception:
        return None
    return rw.GetMol()


def _name_as_acid(mol, strategy, session, depth) -> str | None:
    """Drive the engine on a constructed acid fragment and return the
    assembled standalone name."""
    from iupac_namer.engine import name as _recursive_name

    try:
        tree = _recursive_name(
            mol,
            strategy,
            OutputForm.STANDALONE,
            free_valence=None,
            decision_ctx=None,
            _session=session,
            _depth=depth + 1,
        )
    except Exception:
        return None
    from iupac_namer.assembly import assemble

    try:
        return assemble(tree)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Acid-name → acyl-name transform (retained-aware)
# ---------------------------------------------------------------------------

_RETAINED_ACYL_MAP: dict[str, str] = {
    "acetic acid": "acetyl",
    "benzoic acid": "benzoyl",
    "formic acid": "formyl",
    "propionic acid": "propionyl",
    "butyric acid": "butyryl",
    "oxalic acid": "oxalyl",
}


def _acid_to_acyl(acid_name: str) -> str | None:
    """Convert a rendered carboxylic acid name to its acyl equivalent.

    Recognises retained names (``acetic acid`` → ``acetyl``) and systematic
    forms (``-oic acid`` → ``-oyl``, ``-carboxylic acid`` → ``-carbonyl``).
    Returns ``None`` when the input does not end with a known acid suffix.
    """
    if not acid_name:
        return None
    for retained, acyl in _RETAINED_ACYL_MAP.items():
        if acid_name == retained:
            return acyl
        if acid_name.endswith(retained):
            return acid_name[: -len(retained)] + acyl
    if acid_name.endswith("oic acid"):
        return acid_name[: -len("oic acid")] + "oyl"
    if acid_name.endswith("carboxylic acid"):
        return acid_name[: -len("carboxylic acid")] + "carbonyl"
    return None


# ---------------------------------------------------------------------------
# Class-word rules: R-C(=O)-<tail>
# ---------------------------------------------------------------------------

_CLASS_WORD_PATTERNS: dict[str, tuple[str, str]] = {
    "isocyanid":         ("[NX2+;H0]#[CX1-]",      "isocyanide"),
    "isotellurocyanatid":("[NX2]=[CX2]=[Te]",      "isotellurocyanate"),
    "tellurocyanatid":   ("[TeX2][CX2]#[NX1]",     "tellurocyanate"),
    "isoselenocyanatid": ("[NX2]=[CX2]=[Se]",      "isoselenocyanate"),
    "selenocyanatid":    ("[SeX2][CX2]#[NX1]",     "selenocyanate"),
    "azid":              ("[NX2]=[NX2+]=[NX1-]",   "azide"),
    "cyanatid":          ("[OX2][CX2]#[NX1]",      "cyanate"),
    "cyanid":            ("[CX2]#[NX1]",           "cyanide"),
    "isocyanatid":       ("[NX2]=[CX2]=[OX1]",     "isocyanate"),
    "thiocyanatid":      ("[SX2][CX2]#[NX1]",      "thiocyanate"),
}


def _detect_class_word_rule(
    rule: InfixRule,
    mol,
    decision_ctx,
    strategy,
    session,
    depth,
) -> LeafTree | None:
    """Detect ``R-C(=O)-<tail>`` and emit ``<acyl> <class-word>``.

    Guards:

    * Exactly ONE carbonyl-C anchor must match.
    * No OTHER acid FG (``-COOH``, ``-COCl``, ``-CONH2``, …) on a different C.
    * Every heavy atom of the molecule must be accounted for.
    """
    from rdkit import Chem

    tail_smarts, class_word = _CLASS_WORD_PATTERNS[rule.alias]
    full_smarts = f"[#6;!$(C=O)][CX3](=O){tail_smarts}"
    patt = Chem.MolFromSmarts(full_smarts)
    if patt is None:
        return None
    matches = mol.GetSubstructMatches(patt, useChirality=False)
    if len(matches) != 1:
        return None
    match = matches[0]
    r_atom, acyl_c, carbonyl_o = match[0], match[1], match[2]
    tail_atoms = frozenset(match[3:])

    heavy_indices = {
        a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1
    }
    r_side = _reach_avoiding(mol, r_atom, {acyl_c})
    covered = r_side | {acyl_c, carbonyl_o} | tail_atoms
    if covered != heavy_indices:
        return None

    other_acyl = Chem.MolFromSmarts(
        "[CX3](=O)[OX2H1,FX1,ClX1,BrX1,IX1,NX3]"
    )
    if other_acyl is not None:
        other_matches = [
            m for m in mol.GetSubstructMatches(other_acyl) if m[0] != acyl_c
        ]
        if other_matches:
            return None

    acid_mol = _rebuild_as_acid(mol, acyl_c, r_side)
    if acid_mol is None:
        return None
    acid_name = _name_as_acid(acid_mol, strategy, session, depth)
    if acid_name is None or "NAMING ERROR" in (acid_name or ""):
        return None
    acyl_name = _acid_to_acyl(acid_name)
    if acyl_name is None:
        return None

    return LeafTree(
        output_form=OutputForm.STANDALONE,
        free_valence=None,
        choices_made=(Choice(
            type="acid_infix_composition",
            detail=f"{rule.alias}: {acyl_name} {class_word}",
        ),),
        decision_ctx=decision_ctx,
        validity_warnings=None,
        text=f"{acyl_name} {class_word}",
    )


# ---------------------------------------------------------------------------
# Embedded-stem rules
# ---------------------------------------------------------------------------


def _detect_nitrid(mol, decision_ctx) -> LeafTree | None:
    """``#O:#N`` — triple-bond N in place of =O.  Covers the phosphoronitridic
    case (``N#P(OH)(OH)``).  Plain nitriles (``R-C#N``) are named by the
    retained / nitrile FG entry — we do not displace those."""
    from rdkit import Chem

    patt = Chem.MolFromSmarts("[NX1]#[PX3]([OX2H1])([OX2H1])")
    if patt is None:
        return None
    matches = mol.GetSubstructMatches(patt)
    if not matches:
        return None
    match = matches[0]
    heavy = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
    if heavy != set(match):
        return None
    return LeafTree(
        output_form=OutputForm.STANDALONE,
        free_valence=None,
        choices_made=(Choice(
            type="acid_infix_composition",
            detail="nitrid/phosphoronitridic",
        ),),
        decision_ctx=decision_ctx,
        validity_warnings=None,
        text="phosphoronitridic acid",
    )


def _compose_embedded_stem(acid_name: str, infix: str) -> str | None:
    """Splice the given infix between an acid's chain stem and its ``ic acid``
    tail.  Retained names are first normalised to their systematic counterparts
    because OPSIN only accepts the systematic infix form."""
    if not acid_name:
        return None
    _retained_to_systematic = {
        "acetic acid":   "ethanoic acid",
        "formic acid":   "methanoic acid",
        "propionic acid":"propanoic acid",
        "butyric acid":  "butanoic acid",
        "benzoic acid":  "benzenecarboxylic acid",
    }
    for retained, systematic in _retained_to_systematic.items():
        if acid_name == retained:
            acid_name = systematic
            break
        if acid_name.endswith(retained):
            acid_name = acid_name[: -len(retained)] + systematic
            break
    # Strip only ``ic acid`` (never ``oic acid``) so systematic ``ethanoic
    # acid`` yields stem ``ethano`` — cleanly giving ``ethanotelluroic acid``
    # with a single 'o' vowel.
    if acid_name.endswith("carboxylic acid"):
        stem = acid_name[: -len("carboxylic acid")]
        return f"{stem}{infix}carboxylic acid"
    if acid_name.endswith("ic acid"):
        stem = acid_name[: -len("ic acid")]
        return f"{stem}{infix}ic acid"
    return None


def _detect_tellur(mol, decision_ctx, strategy, session, depth) -> LeafTree | None:
    """``=O,-O:[TeH?]`` — replace ``=O`` or ``-OH`` with ``=Te``/``-TeH``.
    Emits ``{stem}telluroic acid``."""
    from rdkit import Chem

    shapes = (
        ("R-C(O)=[Te]",
         Chem.MolFromSmarts("[#6;!$(C=O)][CX3]([OX2H1])=[Te]")),
        # Terminal Te-H has TotalDegree=2 (counts the H) but heavy-atom
        # degree D=1 — probe by D/H rather than X.
        ("R-C(=O)[TeH]",
         Chem.MolFromSmarts("[#6;!$(C=O)][CX3](=O)[Te;D1;H1]")),
    )
    for label, patt in shapes:
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt)
        if len(matches) != 1:
            continue
        match = matches[0]
        r_idx, c_idx = match[0], match[1]
        het_1, het_2 = match[2], match[3]
        heavy = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
        r_side = _reach_avoiding(mol, r_idx, {c_idx})
        if (r_side | {c_idx, het_1, het_2}) != heavy:
            continue
        acid_mol = _rebuild_as_acid(mol, c_idx, r_side)
        if acid_mol is None:
            continue
        acid_name = _name_as_acid(acid_mol, strategy, session, depth)
        if acid_name is None or "NAMING ERROR" in acid_name:
            continue
        composed = _compose_embedded_stem(acid_name, "telluro")
        if composed is None:
            continue
        return LeafTree(
            output_form=OutputForm.STANDALONE,
            free_valence=None,
            choices_made=(Choice(
                type="acid_infix_composition",
                detail=f"tellur: {label}",
            ),),
            decision_ctx=decision_ctx,
            validity_warnings=None,
            text=composed,
        )
    return None


def _detect_hydroxim(mol, decision_ctx, strategy, session, depth) -> LeafTree | None:
    """``=O:=NO`` — ``=N-OH`` in place of ``=O``.  Retired form, still
    accepted by OPSIN.  Emits ``{stem}hydroximic acid``."""
    from rdkit import Chem

    patt = Chem.MolFromSmarts("[#6;!$(C=O)][CX3](=[NX2][OX2H1])[OX2H1]")
    if patt is None:
        return None
    matches = mol.GetSubstructMatches(patt)
    if len(matches) != 1:
        return None
    match = matches[0]
    r_idx, c_idx = match[0], match[1]
    n_idx, no_o_idx, oh_idx = match[2], match[3], match[4]
    heavy = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
    r_side = _reach_avoiding(mol, r_idx, {c_idx})
    if (r_side | {c_idx, n_idx, no_o_idx, oh_idx}) != heavy:
        return None
    acid_mol = _rebuild_as_acid(mol, c_idx, r_side)
    if acid_mol is None:
        return None
    acid_name = _name_as_acid(acid_mol, strategy, session, depth)
    if acid_name is None or "NAMING ERROR" in acid_name:
        return None
    composed = _compose_embedded_stem(acid_name, "hydroxim")
    if composed is None:
        return None
    return LeafTree(
        output_form=OutputForm.STANDALONE,
        free_valence=None,
        choices_made=(Choice(
            type="acid_infix_composition",
            detail="hydroxim",
        ),),
        decision_ctx=decision_ctx,
        validity_warnings=None,
        text=composed,
    )


def _detect_ditelluroperox(
    mol, decision_ctx, strategy, session, depth
) -> LeafTree | None:
    """``-O-:[Te|2][Te|2]`` — replace the ``-O-`` bridge with ``Te-Te``.
    Emits ``{stem}ditelluroperoxoic acid``."""
    from rdkit import Chem

    patt = Chem.MolFromSmarts("[#6;!$(C=O)][CX3](=O)[Te;D2][Te;D1;H1]")
    if patt is None:
        return None
    matches = mol.GetSubstructMatches(patt)
    if len(matches) != 1:
        return None
    match = matches[0]
    r_idx, c_idx = match[0], match[1]
    o_idx, te1_idx, te2_idx = match[2], match[3], match[4]
    heavy = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
    r_side = _reach_avoiding(mol, r_idx, {c_idx})
    if (r_side | {c_idx, o_idx, te1_idx, te2_idx}) != heavy:
        return None
    acid_mol = _rebuild_as_acid(mol, c_idx, r_side)
    if acid_mol is None:
        return None
    acid_name = _name_as_acid(acid_mol, strategy, session, depth)
    if acid_name is None or "NAMING ERROR" in acid_name:
        return None
    composed = _compose_embedded_stem(acid_name, "ditelluroperoxo")
    if composed is None:
        return None
    return LeafTree(
        output_form=OutputForm.STANDALONE,
        free_valence=None,
        choices_made=(Choice(
            type="acid_infix_composition",
            detail="ditelluroperox",
        ),),
        decision_ctx=decision_ctx,
        validity_warnings=None,
        text=composed,
    )


# ---------------------------------------------------------------------------
# Diagnostic helper
# ---------------------------------------------------------------------------


def infix_coverage_matrix() -> dict[str, str]:
    """Return ``{alias: status}`` where ``status ∈ {"native", "composed",
    "skipped"}``."""
    composed = set(_CLASS_WORD_PATTERNS.keys()) | {
        "nitrid", "tellur", "hydroxim", "ditelluroperox",
    }
    out: dict[str, str] = {}
    for rule in load_infix_rules():
        if rule.is_handled_natively:
            out[rule.alias] = "native"
        elif rule.alias in composed:
            out[rule.alias] = "composed"
        else:
            out[rule.alias] = "skipped"
    return out
