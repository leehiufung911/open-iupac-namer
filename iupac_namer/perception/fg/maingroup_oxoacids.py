"""iupac_namer.perception.fg.maingroup_oxoacids

GENERATIVE main-group oxoacid namer (Stage 15 — oxoacid coverage gap).

This module replaces the whole-molecule SMILES->name *lookup* approach of
``heteroelement_oxoacids.py`` / ``phosphorus_oxoacids.py`` with a
structure-driven *generator*.  Given a molecule whose skeleton is a
main-group oxoacid — one or more central atoms X (B, N, P, As, Sb, S, Se,
Te, or a halogen) each bonded only to terminal ``=O`` (oxo), ``-OH``
(hydroxy), anionic ``[O-]`` (oxido), ``-O-X`` anhydride bridges, ``-O-O-``
peroxy links / terminal hydroperoxy, direct ``X-X`` bonds, and (for
sulfur) bridging chalcogen atoms — it COMPUTES the IUPAC name from the
perceived structural features:

* the central element and how many single-bonded acidic O positions it
  carries (the *tier*: -or-/-on-/-in- for pnictogens; element root for
  chalcogens/boron),
* whether the centre bears an oxo (``-ic`` vs ``-ous``),
* the polynuclear count (di/tri/...) and whether the centres are joined
  by ``-O-`` (anhydride) or directly (``hypo`` prefix),
* a leading ``peroxy`` modifier when a peroxide ``-O-O-`` link is present,
* and the anion / partial-anion modifiers (``-ate`` / ``-ite`` for fully
  deprotonated, ``hydrogen``-prefix for partial, ``per...ate`` for the
  highest halogen oxidation state).

Because every decision branches on a *structural feature* (element,
single-O count, oxo presence, bridge type, chain length) and never on an
exact known-molecule SMILES, the generator extends automatically to every
member of each family — e.g. a tetra-/penta-selenic chain needs no new
code or data.

Citations
---------
IUPAC 2013 Recommendations (Blue Book):

* P-67.1.1.1 — mononuclear noncarbon oxoacid preselected names (the
  -or/-on/-in tier scheme and the element roots in
  ``data/maingroup_oxoacid_roots.json``).
* P-67.2.1 — di-/polynuclear preselected names (``di``/``tri`` prefixes
  for ``-O-`` bridged chains; ``hypo`` + ``di`` for direct ``X-X`` bonds;
  the ``thion`` series for sulfur–sulfur-bridged chains).
* P-67.1.2.3.1 / P-67.1.2.3.3 — ``peroxo``/``peroxy`` functional
  replacement for ``-O-O-`` links.
* P-65.3 / P-72 — salt and anion (``-ate``/``-ite``/``hydrogen``) naming.

Dispatch
--------
``engine.name`` calls :func:`compute_name` in the oxoacid shortcut region
(after the curated-inorganic and charge-perception dispatchers, before
plan search).  The generator self-gates: it returns ``None`` for anything
that is not a pure main-group oxoacid skeleton (carbon present, ring
membership, unexpected substituents, free valences, multi-fragment
inputs), so substituted derivatives and esters fall through to the
ordinary substitutive machinery untouched.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from rdkit import Chem  # noqa: F401


# ---------------------------------------------------------------------------
# Structural data table (element -> acid roots).  Loaded once; never mutated.
# ---------------------------------------------------------------------------

_TABLE_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data"
    / "maingroup_oxoacid_roots.json"
)

_ROOTS: dict | None = None


def _roots() -> dict:
    global _ROOTS
    if _ROOTS is None:
        with open(_TABLE_PATH, encoding="utf-8") as f:
            _ROOTS = json.load(f)
    return _ROOTS


# Multiplying prefixes for the polynuclear count.  Local copy keeps this
# module self-contained; values match data/multipliers.json (P-14.2).
_MULT = {
    1: "", 2: "di", 3: "tri", 4: "tetra", 5: "penta",
    6: "hexa", 7: "hepta", 8: "octa", 9: "nona", 10: "deca",
}

# Elements eligible to be acid *centres*.  Halogens are handled by a
# separate code path (oxo count == central formal charge).
_PNICT_CHALC_B = frozenset({"N", "P", "As", "Sb", "B", "S", "Se", "Te"})
_HALOGENS = frozenset({"F", "Cl", "Br", "I"})
_CENTRE_ELEMENTS = _PNICT_CHALC_B | _HALOGENS

# Chalcogen elements that may serve as a *bridge* atom in a sulfur thionic
# chain (P-67.2.1 ``trithionic``/``tetrathionic`` … : -S- links).
_CHALC_BRIDGE = frozenset({"S", "Se", "Te"})


# ---------------------------------------------------------------------------
# Perception data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Centre:
    """A single acid centre and its oxygen-environment counts."""
    idx: int
    element: str
    n_oxo: int          # =O (or charge-balancing [O-] on a cationic centre)
    n_oh: int           # terminal -OH
    n_oxide: int        # terminal anionic [O-] (true anion, not oxo)
    n_hydro: int        # substitutable H directly on the centre
    n_peroxy_oh: int    # terminal -O-OH hydroperoxy
    n_peroxy_oxide: int  # terminal -O-O[-] anionic peroxy
    amido: int          # -NH2 substituent count (sulfamic family)


class _PerceptionFailure(Exception):
    """Raised internally when the molecule is not a clean oxoacid skeleton."""


# ---------------------------------------------------------------------------
# Perception
# ---------------------------------------------------------------------------

def _perceive(mol):
    """Decompose *mol* into acid centres, bridges and modifiers.

    Returns a dict describing the skeleton, or raises ``_PerceptionFailure``
    if *mol* is not a pure main-group oxoacid (so the caller returns None).
    """
    from rdkit import Chem  # local import

    # --- Global gating: single fragment, no carbon, no rings, no charge on C.
    if mol.GetNumAtoms() == 0:
        raise _PerceptionFailure("empty")
    # Reject anything with carbon — that's an organic derivative / ester,
    # which the ordinary substitutive machinery handles.
    for a in mol.GetAtoms():
        if a.GetAtomicNum() == 6:
            raise _PerceptionFailure("carbon present")
        if a.GetIsAromatic() or a.IsInRing():
            raise _PerceptionFailure("ring / aromatic")
        if a.GetNumRadicalElectrons() and a.GetAtomicNum() not in (
            # heavy main-group atoms may show RDKit radical electrons in the
            # hypervalent valence model; that is fine.  But a genuine radical
            # (open valence) on O/N should disqualify us.
            7, 15, 33, 51, 5, 16, 34, 52, 9, 17, 35, 53,
        ):
            raise _PerceptionFailure("radical")

    centres: list[_Centre] = []
    centre_idxs: set[int] = set()

    # Identify candidate centre atoms.  A terminal nitrogen (-NH2, degree 1,
    # neutral) is an amido SUBSTITUENT (sulfamic family, P-67.1.2.6), not a
    # centre, even though N is a centre-eligible element.
    amido_n: set[int] = set()
    for a in mol.GetAtoms():
        if a.GetSymbol() not in _CENTRE_ELEMENTS:
            continue
        if (a.GetSymbol() == "N" and a.GetDegree() == 1
                and a.GetFormalCharge() == 0 and a.GetTotalNumHs() == 2):
            amido_n.add(a.GetIdx())
            continue
        centre_idxs.add(a.GetIdx())

    if not centre_idxs:
        raise _PerceptionFailure("no centre element")

    # A chalcogen (S/Se/Te) atom that is a pure bridge (degree 2, both bonds
    # single, no oxo/oh, neighbours are themselves chalcogen or a centre) is
    # NOT a centre — it is a thionic bridge.  Re-classify those.
    bridge_chalc: set[int] = set()
    for idx in list(centre_idxs):
        a = mol.GetAtomWithIdx(idx)
        if a.GetSymbol() not in _CHALC_BRIDGE:
            continue
        nbrs = a.GetNeighbors()
        if a.GetDegree() != 2:
            continue
        if any(mol.GetBondBetweenAtoms(idx, n.GetIdx()).GetBondTypeAsDouble() != 1.0
               for n in nbrs):
            continue
        # All single bonds, degree 2, no terminal O on it -> bridge.
        if all(n.GetSymbol() not in ("O",) for n in nbrs):
            bridge_chalc.add(idx)
            centre_idxs.discard(idx)

    if not centre_idxs:
        raise _PerceptionFailure("no centre after bridge reclassification")

    # All centres must be the same element (mixed-element acids are out of
    # generative scope — they fall through to substitutive naming).
    centre_elems = {mol.GetAtomWithIdx(i).GetSymbol() for i in centre_idxs}
    if len(centre_elems) != 1:
        raise _PerceptionFailure("mixed centre elements")
    element = next(iter(centre_elems))

    # Track which O atoms are consumed so we can verify every atom is claimed.
    claimed: set[int] = set()

    # --- Classify each centre's local environment. ---
    o_bridges: list[tuple[int, int, int]] = []  # (centreA, centreB, O idx)
    peroxy_bridges: list[tuple[int, int, int, int]] = []  # cA,cB,O1,O2
    direct_bonds: list[tuple[int, int]] = []  # (centreA, centreB) X-X
    chalc_bridges: list[tuple] = []  # (cA, cB, tuple-of-bridge-atom-idxs)

    for idx in sorted(centre_idxs):
        a = mol.GetAtomWithIdx(idx)
        claimed.add(idx)
        n_oxo = n_oh = n_oxide = n_perox_oh = n_perox_ox = amido = 0
        cationic = a.GetFormalCharge() > 0
        oxo_credit = a.GetFormalCharge() if cationic else 0
        for nb in a.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(idx, nb.GetIdx())
            order = bond.GetBondTypeAsDouble()
            sym = nb.GetSymbol()
            ni = nb.GetIdx()

            if sym == "O":
                deg = nb.GetDegree()
                chg = nb.GetFormalCharge()
                nh = nb.GetTotalNumHs()
                if order == 2.0 and deg == 1:
                    n_oxo += 1
                    claimed.add(ni)
                elif order == 1.0 and deg == 1:
                    if chg == -1:
                        # On a cationic centre the [O-] is a charge-balancing
                        # oxo (e.g. [O-][Cl+2] = chloric); spend the credit.
                        if oxo_credit > 0:
                            n_oxo += 1
                            oxo_credit -= 1
                        else:
                            n_oxide += 1
                    elif nh == 1 and chg == 0:
                        n_oh += 1
                    else:
                        raise _PerceptionFailure("unexpected terminal O")
                    claimed.add(ni)
                elif order == 1.0 and deg == 2:
                    # Bridge oxygen: -O-(centre) anhydride, -O-O- peroxy, or
                    # terminal hydroperoxy -O-OH.
                    other = [x for x in nb.GetNeighbors() if x.GetIdx() != idx][0]
                    osym = other.GetSymbol()
                    if osym in _CENTRE_ELEMENTS and other.GetIdx() in centre_idxs:
                        # plain -O- anhydride bridge between two centres
                        b = tuple(sorted((idx, other.GetIdx())))
                        if (b[0], b[1], ni) not in o_bridges:
                            o_bridges.append((b[0], b[1], ni))
                        claimed.add(ni)
                    elif osym == "O":
                        # peroxide link: idx-O-O-?
                        o2 = other
                        o2_other = [x for x in o2.GetNeighbors()
                                    if x.GetIdx() != ni]
                        claimed.add(ni)
                        claimed.add(o2.GetIdx())
                        if not o2_other:
                            # -O-OH or -O-O[-] terminal peroxy
                            if o2.GetFormalCharge() == -1:
                                n_perox_ox += 1
                            elif o2.GetTotalNumHs() == 1:
                                n_perox_oh += 1
                            else:
                                raise _PerceptionFailure("bad terminal peroxy")
                        else:
                            o3 = o2_other[0]
                            if (o3.GetSymbol() in _CENTRE_ELEMENTS
                                    and o3.GetIdx() in centre_idxs):
                                b = tuple(sorted((idx, o3.GetIdx())))
                                key = (b[0], b[1], ni, o2.GetIdx())
                                rev = (b[0], b[1], o2.GetIdx(), ni)
                                if key not in peroxy_bridges and rev not in peroxy_bridges:
                                    peroxy_bridges.append(key)
                            else:
                                raise _PerceptionFailure("bad peroxy bridge")
                    else:
                        raise _PerceptionFailure("unexpected bridge O")
                else:
                    raise _PerceptionFailure("unexpected O bonding")

            elif sym == element and ni in centre_idxs:
                # direct X-X bond
                if order != 1.0:
                    raise _PerceptionFailure("multi-bond X-X")
                db = tuple(sorted((idx, ni)))
                if db not in direct_bonds:
                    direct_bonds.append(db)

            elif ni in bridge_chalc:
                # bridging chalcogen (thionic chain): walk the run of bridge
                # atoms (S-S, S-S-S, ...) to the far centre, recording every
                # bridge atom.  P-67.2.1 trithionic (1)/tetrathionic (2)/...
                chain: list[int] = []
                prev, cur = idx, ni
                while cur in bridge_chalc:
                    chain.append(cur)
                    br = mol.GetAtomWithIdx(cur)
                    nxts = [x for x in br.GetNeighbors() if x.GetIdx() != prev]
                    if len(nxts) != 1:
                        raise _PerceptionFailure("branched chalcogen bridge")
                    prev, cur = cur, nxts[0].GetIdx()
                far = mol.GetAtomWithIdx(cur)
                if far.GetSymbol() == element and far.GetIdx() in centre_idxs:
                    b = tuple(sorted((idx, far.GetIdx())))
                    rec = (b[0], b[1], tuple(sorted(chain)))
                    if rec not in chalc_bridges:
                        chalc_bridges.append(rec)
                    for cidx in chain:
                        claimed.add(cidx)
                else:
                    raise _PerceptionFailure("bad chalcogen bridge")

            elif sym == "N" and nb.GetDegree() == 1 and order == 1.0:
                # -NH2 amido substituent (sulfamic / amidosulfonic family).
                if nb.GetTotalNumHs() == 2 and nb.GetFormalCharge() == 0:
                    amido += 1
                    claimed.add(ni)
                else:
                    raise _PerceptionFailure("non-amido N substituent")

            else:
                raise _PerceptionFailure(f"unexpected substituent {sym}")

        n_hydro = a.GetTotalNumHs()
        centres.append(_Centre(
            idx=idx, element=element, n_oxo=n_oxo, n_oh=n_oh,
            n_oxide=n_oxide, n_hydro=n_hydro, n_peroxy_oh=n_perox_oh,
            n_peroxy_oxide=n_perox_ox, amido=amido,
        ))

    # Verify every atom was claimed (no silent atom drops, CLAUDE.md #2).
    for a in mol.GetAtoms():
        if a.GetIdx() not in claimed and a.GetIdx() not in centre_idxs:
            raise _PerceptionFailure("unclaimed atom")

    return {
        "element": element,
        "centres": centres,
        "o_bridges": o_bridges,
        "peroxy_bridges": peroxy_bridges,
        "direct_bonds": direct_bonds,
        "chalc_bridges": chalc_bridges,
        "centre_idxs": centre_idxs,
    }


# ---------------------------------------------------------------------------
# Name computation
# ---------------------------------------------------------------------------

def _tier_count(c: _Centre) -> int:
    """Number of single-bonded acidic O positions on the centre.

    -OH, [O-] anion, anhydride/peroxy bridge each count once.  This is the
    P-67 'tier' that selects -or-/-on-/-in- for pnictogens and the element
    root (sulfur vs ...) for chalcogens.  Bridges and direct X-X bonds are
    added back in the caller because they live on the skeleton, not the
    centre dataclass.
    """
    return c.n_oh + c.n_oxide + c.n_peroxy_oh + c.n_peroxy_oxide


def _mononuclear_root_and_suffix(element: str, tier: int, oxo_count: int):
    """Return (root, suffix, scheme) for a single centre, or None.

    ``oxo_count`` is the number of oxo groups (=O / charge-balanced) on the
    centre.  For chalcogens the -ic/-ous split is by oxo *count* (2 vs 1);
    for pnictogens it is by oxo *presence* (>=1 vs 0).
    """
    roots = _roots()["elements"].get(element)
    if roots is None:
        return None
    scheme = roots["scheme"]
    tdata = roots["tiers"].get(str(tier))
    if tdata is None:
        return None
    if scheme == "chalcogen":
        # S/Se/Te: -ic has exactly 2 oxo (highest OS), -ous has exactly 1.
        # A different oxo count (e.g. an "-ene" or hypervalent form) is not a
        # standard tier acid -> fall through to substitutive naming.
        if oxo_count == 2:
            suffix = "ic"
        elif oxo_count == 1:
            suffix = "ous"
        else:
            return None
        return tdata["ic"], suffix, scheme
    if scheme == "boron":
        # Boron has no lower oxidation state and bears no oxo: always the
        # -ic-form root.  Any oxo disqualifies (not a boron oxoacid form).
        if oxo_count != 0:
            return None
        return tdata["ic"], "ic", scheme
    # pnictogen: -or/-on/-in baked into the per-tier root; the standard -ic
    # form has EXACTLY one oxo and the -ous form has none.  More than one oxo
    # is an "-ene"/meta form (e.g. phosphenic O=P(=O)OH) outside this scheme.
    if oxo_count == 1:
        return tdata["ic"], "ic", scheme
    if oxo_count == 0:
        return tdata["ous"], "ous", scheme
    return None


def _anion_suffix(acid_suffix: str) -> str:
    """Map an acid suffix to its anion suffix (P-65.3 / P-72)."""
    return {"ic": "ate", "ous": "ite"}[acid_suffix]


def _compute(per: dict) -> str | None:
    element = per["element"]
    centres: list[_Centre] = per["centres"]
    n = len(centres)
    o_bridges = per["o_bridges"]
    peroxy_bridges = per["peroxy_bridges"]
    direct_bonds = per["direct_bonds"]
    chalc_bridges = per["chalc_bridges"]

    # --- Halogen oxoacids: distinct rule (oxo == central charge). ---
    if element in _HALOGENS:
        return _compute_halogen(per)

    # Per-centre skeleton connectivity (bridges + direct bonds touching each
    # centre).  Each such link occupied a -OH position in the parent acid,
    # so it counts toward the tier.
    skel_deg = {c.idx: 0 for c in centres}
    for a, b, _o in o_bridges:
        skel_deg[a] += 1
        skel_deg[b] += 1
    for a, b, _o1, _o2 in peroxy_bridges:
        skel_deg[a] += 1
        skel_deg[b] += 1
    for a, b in direct_bonds:
        skel_deg[a] += 1
        skel_deg[b] += 1
    for a, b, _br in chalc_bridges:
        skel_deg[a] += 1
        skel_deg[b] += 1

    # Tier = single-O on centre + amido (each replaced an -OH) + skeleton
    # links.  All centres must agree (P-67.2.1 retains only symmetric chains).
    tiers = {c.idx: _tier_count(c) + c.amido + skel_deg[c.idx] for c in centres}
    if len(set(tiers.values())) != 1:
        return None  # asymmetric tier — not a retained polynuclear acid
    tier = next(iter(tiers.values()))

    # Oxo count: chalcogens need the per-centre count (2 -> -ic, 1 -> -ous);
    # all centres must agree.
    oxo_counts = {c.n_oxo for c in centres}
    if len(oxo_counts) != 1:
        return None
    oxo_count = next(iter(oxo_counts))

    rs = _mononuclear_root_and_suffix(element, tier, oxo_count)
    if rs is None:
        return None
    root, suffix, scheme = rs
    anion_root = _roots()["elements"][element].get("anion_root")

    # For MONONUCLEAR pnictogen / boron acids the standard tier acid has a
    # FIXED number of substitutable H on the centre: n_hydro == 3 - tier (an
    # H takes the place of each missing -OH).  Forms with a different H count
    # (e.g. arsenenous As(=O)OH, 0 H at tier 1) are non-standard "-ene" acids
    # and fall through to substitutive naming.  Polynuclear pnictogen acids
    # are skipped here: OPSIN's reference SMILES for the bridged forms (e.g.
    # diarsonic [As](=O)(O)O[As](=O)O) leaves the centre H implicit and RDKit
    # reads n_hydro==0, so the bridge connectivity (already constrained to a
    # symmetric chain) is the reliable signal instead.  Chalcogens carry no
    # substitutable H in any of these acids.
    if scheme in ("pnictogen", "boron"):
        if n == 1 and any(c.n_hydro != 3 - tier for c in centres):
            return None
    elif scheme == "chalcogen":
        if any(c.n_hydro != 0 for c in centres):
            return None

    # --- Linkage classification across the chain. ---
    n_links = len(o_bridges) + len(peroxy_bridges) + len(direct_bonds) + len(chalc_bridges)
    if n == 1:
        if n_links != 0:
            return None
        linkage = "mono"
    else:
        # Must form a single linear chain: exactly n-1 links.
        if n_links != n - 1:
            return None
        if direct_bonds and (o_bridges or chalc_bridges):
            return None  # mixed linkage chain — not a retained name
        if len(peroxy_bridges) > 1:
            return None  # only single peroxy link supported (P-67.2.1)
        if peroxy_bridges and n != 2:
            return None
        if chalc_bridges and element != "S":
            return None  # thionic series is sulfur-only in P-67.2.1
        if direct_bonds:
            linkage = "direct"        # hypo + di
        elif peroxy_bridges:
            linkage = "peroxy_bridge"  # peroxydi
        elif chalc_bridges:
            linkage = "thionic"       # tri/tetra-thionic
        else:
            linkage = "anhydride"     # di/tri

    # --- Anion / hydrogen modifiers. ---
    total_oh = sum(c.n_oh + c.n_peroxy_oh for c in centres)
    total_oxide = sum(c.n_oxide + c.n_peroxy_oxide for c in centres)
    if total_oh + total_oxide == 0:
        return None  # no acidic position at all — not an acid/anion
    amido = sum(c.amido for c in centres)

    # Terminal peroxy hydroperoxy/peroxide present anywhere -> leading
    # peroxy modifier (mononuclear peroxy acid, e.g. peroxysulfuric).
    has_terminal_peroxy = any(c.n_peroxy_oh or c.n_peroxy_oxide for c in centres)

    # --- Thionic sulfur series: dithionic/trithionic/... (special root). ---
    # P-67.2.1: HO-SO2-SO2-OH = dithionic acid (direct S-S);
    # HO-SO2-S-SO2-OH = trithionic; +1 bridge S each step.  -ous when the
    # centres carry a single oxo (dithionous: HO-SO-SO-OH).
    if element == "S" and linkage in ("thionic", "direct") and amido == 0:
        n_bridge_atoms = sum(len(rec[2]) for rec in chalc_bridges)
        s_total = n + n_bridge_atoms
        if s_total > 5:
            return None  # OPSIN vocabulary stops at pentathionic
        thion_root = _roots()["elements"]["S"]["thionic"]
        th_suffix = "ic" if oxo_count >= 2 else "ous"
        prefix = _MULT.get(s_total)
        if prefix is None:
            return None
        return _apply_anion(
            mult=prefix, peroxy="", acid_root=thion_root, suffix=th_suffix,
            anion_root=thion_root, total_oh=total_oh, total_oxide=total_oxide,
            amido=amido,
        )

    # --- Build the core acid name components. ---
    peroxy_part = ""
    mult = ""
    if linkage == "mono":
        pass
    elif linkage == "anhydride":
        mult = _MULT[n]
    elif linkage == "direct":
        # Boron's tier-3 direct-bond di-acid is retained as "hypoboric" (no
        # infix "di"); the systematic "hypodiboric" is not in OPSIN's
        # vocabulary.  The tier-2 analogue, by contrast, IS "hypodiboronic"
        # (with "di"), and tier-1 ("hypo(di)borinic") has no OPSIN form, so
        # only the tier-3 "bor" root drops the "di".
        peroxy_part = "hypo"
        if element == "B" and n == 2 and root == "bor":
            mult = ""
        elif element == "B" and root == "borin":
            return None  # no OPSIN-parseable hypodiborinic form
        else:
            mult = _MULT[n]
    elif linkage == "peroxy_bridge":
        mult = _MULT[n]
        peroxy_part = "peroxy"
    else:
        return None

    # Leading "peroxy" for a terminal -OOH (mononuclear peroxy acid).
    if has_terminal_peroxy and linkage == "mono":
        peroxy_part = "peroxy"
    elif has_terminal_peroxy:
        return None  # peroxy + polynuclear terminal combination unsupported

    return _apply_anion(
        mult=mult, peroxy=peroxy_part, acid_root=root, suffix=suffix,
        anion_root=anion_root, total_oh=total_oh, total_oxide=total_oxide,
        amido=amido,
    )


def _apply_anion(*, mult: str, peroxy: str, acid_root: str, suffix: str,
                 anion_root: str | None, total_oh: int, total_oxide: int,
                 amido: int) -> str | None:
    """Assemble the final name from components and the anion state.

    The acid name is ``<peroxy><mult><acid_root><suffix>``; the anion swaps
    ``<acid_root><suffix>`` for ``<anion_root><ate|ite>``.  An ``amido``
    prefix (sulfamic family, P-67.1.2.6) leads the whole name.
    """
    amido_prefix = "amido" * amido if amido else ""

    if total_oxide == 0:
        # Neutral acid.
        return f"{amido_prefix}{peroxy}{mult}{acid_root}{suffix} acid"

    # Anion forms need an explicit anion root.  Pnictogen / boron anions are
    # out of generative scope (irregular -ite stems) -> fall through.
    if anion_root is None:
        return None
    anion = _anion_suffix(suffix)
    stem = f"{peroxy}{mult}{anion_root}"
    if total_oh == 0:
        return f"{amido_prefix}{stem}{anion}"
    # Partial anion: "hydrogen ...ate/...ite" per remaining -OH.
    h_prefix = "hydrogen" if total_oh == 1 else _MULT.get(total_oh, "") + "hydrogen"
    return f"{amido_prefix}{h_prefix} {stem}{anion}"


def _compute_halogen(per: dict) -> str | None:
    """Halogen oxoacids (F/Cl/Br/I): hypo-/-ous/-ic/per- by oxo count.

    P-67.1.1.1 / P-67.1.2.2.  Single centre only (no polynuclear halogen
    oxoacids in the retained set).
    """
    centres: list[_Centre] = per["centres"]
    if len(centres) != 1:
        return None
    if per["o_bridges"] or per["direct_bonds"] or per["peroxy_bridges"] or per["chalc_bridges"]:
        return None
    c = centres[0]
    if c.amido:
        return None
    root = _roots()["halogens"].get(c.element)
    if root is None:
        return None
    n_oxo = c.n_oxo
    n_oh = c.n_oh
    n_oxide = c.n_oxide
    # ortho forms (e.g. orthoperiodic I(OH)5(=O) / orthoperiodate I(O-)6):
    # 5 or 6 single-O positions on iodine.
    single_o = n_oh + n_oxide
    if single_o >= 5:
        # orthoper- form (only iodine in practice).  oxo count high.
        if c.element != "I":
            return None
        base = f"orthoper{root}ic"
        return _apply_anion_halogen(base, "ic", n_oh, n_oxide)
    if single_o != 1:
        return None  # ordinary halogen oxoacids carry exactly one acidic O
    # oxo count determines tier: 0 hypo-ous, 1 -ous, 2 -ic, 3 per-ic.
    if n_oxo == 0:
        base, suffix = f"hypo{root}ous", "ous"
    elif n_oxo == 1:
        base, suffix = f"{root}ous", "ous"
    elif n_oxo == 2:
        base, suffix = f"{root}ic", "ic"
    elif n_oxo == 3:
        base, suffix = f"per{root}ic", "ic"
    else:
        return None
    return _apply_anion_halogen(base, suffix, n_oh, n_oxide)


def _apply_anion_halogen(base: str, suffix: str, n_oh: int, n_oxide: int):
    if n_oxide == 0:
        return f"{base} acid"
    if n_oh == 0:
        anion = _anion_suffix(suffix)
        stem = base[: -len(suffix)]
        return f"{stem}{anion}"
    # partial anion
    anion = _anion_suffix(suffix)
    stem = base[: -len(suffix)]
    h = "hydrogen" if n_oh == 1 else _MULT.get(n_oh, "") + "hydrogen"
    return f"{h} {stem}{anion}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_name(mol) -> str | None:
    """Compute the IUPAC name of a whole-molecule main-group oxoacid.

    Returns the name as a plain string, or ``None`` when *mol* is not a
    pure main-group oxoacid skeleton (in which case the engine falls
    through to its ordinary naming machinery).
    """
    if mol is None:
        return None
    try:
        per = _perceive(mol)
    except _PerceptionFailure:
        return None
    except Exception:  # pragma: no cover — defensive; never break the engine
        return None
    try:
        return _compute(per)
    except _PerceptionFailure:
        return None
    except Exception:  # pragma: no cover
        return None


# ---------------------------------------------------------------------------
# Carbon-substituted nitrogen oxoacids (azonic / azinic / azonous) and their
# anions (azonate / azinate / azonite) — P-67.1.1.2 / P-72.2
# ---------------------------------------------------------------------------
#
# The pure (carbon-free) parent acids and anions are handled by
# ``compute_name`` above.  This second generator covers the *organyl*-
# substituted members, named per P-67.1.1.2 ("Acids with hydrogen atoms
# attached to the central atom may be substituted by organyl groups and
# preferred IUPAC names are formed in this manner.") and their (P-72.2)
# ``-ate``/``-ite`` anions.  The N centre is either:
#
#   * a tetravalent ``[N+]`` carrying exactly one charge-balancing terminal
#     ``[O-]`` (the N->O oxido oxide) — the ``-ic`` series (azonic / azinic);
#     OR
#   * a neutral trivalent ``N`` with no oxido — the ``-ous`` series
#     (azonous; the azinous tier-1 form is not in OPSIN's vocabulary).
#
# Beyond the oxido it carries ``tier`` acidic-oxygen positions — each either a
# terminal ``-OH`` (neutral acid position) or a deprotonated terminal ``[O-]``
# (anion position) — one or more organyl substituents (carbon-bonded, each
# replacing a parent ``N-H``), and the remaining valence filled by H.  The
# tier selects the root (1 -> azinic, 2 -> azonic); the oxido presence selects
# the series (-ic vs -ous); the deprotonation state selects the suffix:
#
#   * all positions protonated (-OH only)        -> ``<prefix><root> acid``
#   * all positions deprotonated ([O-] only)     -> ``<prefix><stem><ate|ite>``
#   * partial (mix of -OH and [O-])              -> ``[di…]hydrogen <…><ate|ite>``
#
# Standard substitutable-H bookkeeping (P-67.1.1.2): on the [N+] centre the
# oxido takes one valence so oxido + tier + organyl + H == 4; on the neutral
# centre tier + organyl + H == 3 directly.  Both reduce to
# ``tier + organyl + H == 3``:
#   azonic / azonous (tier 2): 1 substitutable H -> exactly 1 organyl, 0 H;
#   azinic           (tier 1): 2 substitutable H -> 1..2 organyl, (2-organyl) H.
# (Azoric / tier 3 has no substitutable H on N — its "substituted" forms are
# O-esters, out of scope here, so tier 3 is rejected.)
#
# Each organyl group is carved and named with the engine's ordinary
# substituent machinery, then assembled with the standard P-16.3.3/4
# multiplier and enclosing-mark rules.  No molecule-specific branches: the
# central-atom pattern (oxido + tier acidic-O + organyl count + deprotonation
# state) is the sole gate.

# Acid root keyed by tier (count of acidic-O positions on the N centre) and
# series.  The -ic roots are the P-67.1.1.1 preselected names; the -ous root
# (azonous) exists only at tier 2 (azinous is not in OPSIN's vocabulary).
# Tier 3 (azoric) is absent because it has no substitutable H.
_N_TIER_ROOT = {1: "azinic", 2: "azonic"}
_N_TIER_ROOT_OUS = {2: "azonous"}


def _carve_and_name_organyl(mol, n_idx: int, c_idx: int) -> str | None:
    """Carve the organyl substituent rooted at *c_idx* and return its name.

    Uses the engine's standard substituent naming so every organyl group
    (alkyl, aryl, substituted, unsaturated, cyclic) is handled uniformly.
    Returns ``None`` if naming fails.
    """
    from iupac_namer.engine import (
        carve_substituent,
        _select_substituent_method,
        _fvi_elide_locant_one,
        name as engine_name,
    )
    from iupac_namer.types import FreeValenceInfo, OutputForm
    from iupac_namer.strategy import IUPACCanonical
    from iupac_namer.assembly import assemble

    # Flood-fill the organyl fragment from c_idx, never crossing back to N.
    visited = {c_idx}
    stack = [c_idx]
    while stack:
        cur = stack.pop()
        for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
            ni = nb.GetIdx()
            if nb.GetAtomicNum() == 1 or ni == n_idx or ni in visited:
                continue
            visited.add(ni)
            stack.append(ni)
    sub_atoms = frozenset(visited)

    sub_mol, sub_att, _ = carve_substituent(mol, sub_atoms, (n_idx, c_idx))
    bond = mol.GetBondBetweenAtoms(n_idx, c_idx)
    fv = FreeValenceInfo(
        bond_orders=(int(bond.GetBondTypeAsDouble()),),
        method=_select_substituent_method(sub_mol, sub_att),
        attachment_atoms_in_fragment=(sub_att,),
        elide_locant_one=_fvi_elide_locant_one(sub_mol, sub_att),
    )
    sub_tree = engine_name(
        sub_mol, IUPACCanonical(), OutputForm.SUBSTITUENT, free_valence=fv,
    )
    sub_name = assemble(sub_tree)
    if not sub_name or "[NAMING ERROR" in sub_name or "NAMING ERROR" in sub_name:
        return None
    return sub_name


def compute_substituted_n_oxoacid_name(mol) -> str | None:
    """Name a carbon-substituted nitrogen oxoacid or its anion.

    Returns the IUPAC name (e.g. ``"ethylazonic acid"``,
    ``"dimethylazinic acid"``, ``"methyl(phenyl)azinic acid"``,
    ``"ethylazonate"``, ``"hydrogen ethylazonate"``, ``"methylazinate"``,
    ``"ethylazonite"``) or ``None`` when *mol* is not an organyl-substituted
    azonic/azinic/azonous acid or anion skeleton.

    P-67.1.1.2 (substitution of mononuclear noncarbon oxoacids on the
    central atom) / P-72.2 (their ``-ate``/``-ite`` anions).
    """
    if mol is None:
        return None
    try:
        return _compute_substituted_n_oxoacid(mol)
    except _PerceptionFailure:
        return None
    except Exception:  # pragma: no cover — never break the engine
        return None


def _compute_substituted_n_oxoacid(mol) -> str | None:
    from rdkit import Chem  # noqa: F401  (local import parity with module)

    if mol.GetNumAtoms() == 0:
        return None

    # Single fragment only.
    if len(Chem.GetMolFrags(mol)) != 1:
        return None

    # Closed-shell only — never name a genuine open-shell radical (CLAUDE.md
    # free-valence guard).  Heavy-atom hypervalence is fine, but any radical
    # electron disqualifies us so the engine's guard keeps its authority.
    if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        return None

    # Locate the single nitrogen acid centre.  Either a cationic [N+] with
    # exactly one charge-balancing terminal [O-] (the oxido / N->O oxide,
    # putting it in the -ic series) or a neutral N with no oxido (-ous series).
    n_atoms = [a for a in mol.GetAtoms() if a.GetSymbol() == "N"]
    if len(n_atoms) != 1:
        return None
    n_atom = n_atoms[0]
    n_charge = n_atom.GetFormalCharge()
    if n_charge not in (0, 1):
        return None
    if n_atom.IsInRing() or n_atom.GetIsAromatic():
        return None

    n_terminal_oxide = 0  # terminal [O-] (oxido AND/OR deprotonated acid O)
    n_oh = 0              # terminal -OH (protonated acid position)
    organyl_carbons: list[int] = []
    n_idx = n_atom.GetIdx()

    for nb in n_atom.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(n_idx, nb.GetIdx())
        order = bond.GetBondTypeAsDouble()
        sym = nb.GetSymbol()
        if sym == "O":
            # Every acid / oxido oxygen is a terminal single-bonded O here; a
            # double-bonded =O (e.g. nitric acid O=[N+]([O-])O) is a different
            # (genuine-oxo) skeleton handled elsewhere.
            if order != 1.0 or nb.GetDegree() != 1:
                return None
            chg = nb.GetFormalCharge()
            nh = nb.GetTotalNumHs()
            if chg == -1 and nh == 0:
                n_terminal_oxide += 1
            elif chg == 0 and nh == 1:
                n_oh += 1
            else:
                return None
        elif sym == "C":
            # Organyl substituent attached by a single bond (sp3/sp2/aromatic
            # carbon all valid; the substituent namer handles unsaturation).
            if order != 1.0:
                return None
            organyl_carbons.append(nb.GetIdx())
        else:
            # Any other heteroatom on N (e.g. -NH2 amido, halide, S) is a
            # different family — fall through to ordinary naming.
            return None

    # Split the terminal [O-] into oxido vs deprotonated-acid positions.  On a
    # cationic [N+] exactly one [O-] balances the charge (the N->O oxido,
    # marking the -ic series); the remaining [O-] are deprotonated acid
    # positions.  On a neutral N there is no oxido (-ous series) and every
    # [O-] is a deprotonated acid position.
    if n_charge == 1:
        if n_terminal_oxide < 1:
            return None  # cationic centre needs the charge-balancing oxido
        has_oxido = True
        acid_oxide = n_terminal_oxide - 1
    else:
        has_oxido = False
        acid_oxide = n_terminal_oxide

    # Acidic positions (the tier) = protonated -OH + deprotonated [O-].
    tier = n_oh + acid_oxide
    root = (_N_TIER_ROOT if has_oxido else _N_TIER_ROOT_OUS).get(tier)
    if root is None:
        return None

    # Must carry at least one organyl group — the carbon-free parents/anions
    # are handled by ``compute_name``.
    if not organyl_carbons:
        return None

    n_h = n_atom.GetTotalNumHs()

    # Substitutable-H bookkeeping (P-67.1.1.2): on the [N+] centre the oxido
    # takes one valence (oxido + tier + organyl + H == 4); on the neutral
    # centre tier + organyl + H == 3 directly.  Both reduce to
    # ``tier + organyl + H == 3``.  Verify the substitution count matches the
    # parent's substitutable-H budget:
    #   azonic/azonous (tier 2): 1 substitutable H -> exactly 1 organyl, 0 H.
    #   azinic         (tier 1): 2 substitutable H -> 1..2 organyl, (2-org) H.
    n_org = len(organyl_carbons)
    if tier + n_org + n_h != 3:
        return None
    max_organyl = 3 - tier  # substitutable-H budget on the parent
    if not 1 <= n_org <= max_organyl:
        return None

    # Name each organyl substituent with the engine's standard machinery.
    sub_names: list[str] = []
    for c_idx in organyl_carbons:
        nm = _carve_and_name_organyl(mol, n_idx, c_idx)
        if nm is None:
            return None
        sub_names.append(nm)

    # Assemble the prefix block with standard P-16.3.3/4 multiplier and
    # enclosing-mark rules, alphabetised (P-14.5.2).  No locants: all
    # substituents sit on the single central atom.
    from iupac_namer.assembly import (
        merge_identical_prefixes,
        render_merged_prefixes,
    )

    merged = merge_identical_prefixes([(s, ()) for s in sub_names])
    merged.sort(key=lambda mp: mp.sort_name)
    prefix = render_merged_prefixes(merged)
    if not prefix:
        return None

    # --- Deprotonation state selects acid / anion / partial-anion suffix
    # (P-72.2).  acid_oxide deprotonated positions, n_oh remaining -OH. ---
    if acid_oxide == 0:
        # Fully protonated -> neutral acid.
        return f"{prefix}{root} acid"

    # Anion: swap the acid suffix for -ate (-ic) / -ite (-ous).  The stem is
    # the root with its acid suffix stripped (azonic -> azon, azonous -> azon).
    acid_suffix = "ic" if has_oxido else "ous"
    anion = _anion_suffix(acid_suffix)
    stem = root[: -len(acid_suffix)]
    if n_oh == 0:
        # Fully deprotonated anion.
        return f"{prefix}{stem}{anion}"
    # Partial anion: "[di…]hydrogen <prefix><stem><anion>" per remaining -OH.
    h_prefix = "hydrogen" if n_oh == 1 else _MULT.get(n_oh, "") + "hydrogen"
    return f"{h_prefix} {prefix}{stem}{anion}"


# ---------------------------------------------------------------------------
# Esters of mononuclear main-group oxoacids — P-67.1.3.2 / P-65.6.3
# ---------------------------------------------------------------------------
#
# Esters of the mononuclear noncarbon oxoacids (phosphoric, phosphorous,
# phosphonic, phosphinic, sulfuric, sulfurous, boric, arsoric, selenic, …)
# are named by *functional class* nomenclature (P-67.1.3.2: "named in the
# same way as esters of organic acids, see P-65.6.3.2"):
#
#     <organyl-O-ester word(s)>  [<hydrogen word(s)>]  <acid-anion stem>
#
#   * each O-organyl ester group (X-O-R) is cited as a SEPARATE WORD, the
#     organyl named by the ordinary substituent machinery (alkyl/aryl/…),
#     in alphanumerical order, with a multiplying prefix for identical
#     groups (P-65.6.3.3.2.1);
#   * each remaining free -OH acid position is denoted by the word
#     "hydrogen" (with di/tri… multiplier) — partial esters, P-67.1.3.2;
#   * a carbon substituent bonded DIRECTLY to the centre (phosphonate /
#     phosphinate / etc.) becomes a substituent prefix glued to the anion
#     stem, e.g.  CH3-P(=O)(OEt)2 -> "diethyl methylphosphonate";
#   * the anion stem is the anion form of the parent acid name (-ic -> -ate,
#     -ous -> -ite), computed from the SAME structural tier/oxo logic the
#     parent-acid generator uses, with the ester -O-R and direct-C positions
#     reckoned into the tier exactly as -OH and substitutable-H are.
#
# The gate is purely structural: a single main-group centre whose only
# neighbours are =O (oxo), -OH, [O-], -O-R organyl esters, direct -C
# substituents, and (for the phosphonate/phosphinate P-H forms) H — with at
# least one -O-R ester group.  Anything else (anhydride / peroxy bridges,
# halide or amido or thio substituents, polynuclear chains, free valences,
# multiple fragments, charge) returns None so the molecule falls through to
# the ordinary substitutive machinery untouched.

# Irregular anion stems that are NOT a plain ``root + ate/ite`` (P-67.1.1.1
# preselected names whose anion form drops/changes the acid root).  Only
# phosphorus tier-3 is irregular: phosphoric acid -> phosphate (not
# "phosphorate"), phosphorous acid -> phosphite.  Every other element/tier
# anion stem (phosphonate, phosphinate, sulfate, borate, arsorate, selenate,
# …) is the regular concatenation and needs no entry here.
_IRREGULAR_ANION_STEM = {
    ("P", "phosphor"): "phosph",
}


def _anion_stem(element: str, root: str) -> str:
    """Return the anion-name stem for an *element*/*root* acid (no suffix).

    Chalcogens carry an explicit ``anion_root`` in the data table (sulfur ->
    "sulf", etc.); pnictogen tier-3 phosphorus is irregular ("phosphor" ->
    "phosph"); every other root concatenates regularly.
    """
    chalc_root = _roots()["elements"].get(element, {}).get("anion_root")
    if chalc_root is not None:
        return chalc_root
    return _IRREGULAR_ANION_STEM.get((element, root), root)


def _render_ester_words(organyl_names: list[str]) -> str:
    """Render the ester organyl groups as space-separated words (P-65.6.3).

    Identical organyl groups merge under a multiplying prefix (di/tri or
    bis/tris with enclosing marks for compound names); different groups are
    cited as separate words in alphanumerical order.  Returns e.g.
    ``"trimethyl"``, ``"ethyl methyl phenyl"``, ``"bis(2-chloroethyl)"``.
    """
    from iupac_namer.assembly import (
        merge_identical_prefixes,
        _choose_brackets,
    )

    merged = merge_identical_prefixes([(s, ()) for s in organyl_names])
    merged.sort(key=lambda mp: mp.sort_name)
    words: list[str] = []
    for mp in merged:
        if mp.needs_brackets:
            open_b, close_b = _choose_brackets(mp.name)
            body = f"{open_b}{mp.name}{close_b}"
        else:
            body = mp.name
        words.append(f"{mp.multiplier}{body}" if mp.multiplier else body)
    return " ".join(words)


def compute_oxoacid_ester_name(mol) -> str | None:
    """Name an ester of a mononuclear main-group oxoacid (P-67.1.3.2).

    Returns the IUPAC functional-class name (e.g. ``"trimethyl phosphate"``,
    ``"dimethyl sulfate"``, ``"diethyl methylphosphonate"``,
    ``"methyl dihydrogen phosphate"``) or ``None`` when *mol* is not an
    O-organyl ester of a mononuclear main-group oxoacid skeleton.
    """
    if mol is None:
        return None
    try:
        return _compute_oxoacid_ester(mol)
    except _PerceptionFailure:
        return None
    except Exception:  # pragma: no cover — never break the engine
        return None


def _compute_oxoacid_ester(mol) -> str | None:
    from rdkit import Chem  # local import (parity with module)

    if mol.GetNumAtoms() == 0:
        return None
    # Single fragment only (salts/partial-ester salts route through the salt
    # machinery, which can call the parent-acid anion namer separately).
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    # Whole molecule must be neutral — a net charge means an anion/salt form,
    # out of this generator's (neutral-ester) scope.
    if Chem.GetFormalCharge(mol) != 0:
        return None
    # Closed shell only — never name a genuine open-shell radical (preserve the
    # engine's free-valence guard authority).  Heavy-atom hypervalence aside,
    # any radical electron on a light atom disqualifies us.
    for a in mol.GetAtoms():
        if a.GetNumRadicalElectrons() and a.GetAtomicNum() not in (
            7, 15, 33, 51, 5, 16, 34, 52, 9, 17, 35, 53,
        ):
            return None

    # --- Locate the single acid centre. ---
    centre_atoms = [a for a in mol.GetAtoms()
                    if a.GetSymbol() in _PNICT_CHALC_B and not a.IsInRing()]
    if len(centre_atoms) != 1:
        return None
    centre = centre_atoms[0]
    if centre.GetIsAromatic():
        return None
    element = centre.GetSymbol()
    c_idx = centre.GetIdx()

    # Centre must be neutral.  The cationic charge-separated representations
    # (nitric-acid esters CH3-O-[N+](=O)[O-] -> "methyl nitrate") are a
    # distinct high-oxo "meta" pnictogen family that the parent-acid generator
    # also declines (``_mononuclear_root_and_suffix`` rejects >1 oxo on a
    # pnictogen); they are out of this generator's scope.
    if centre.GetFormalCharge() != 0:
        return None

    n_oxo = n_oh = n_oxide = 0
    ester_roots: list[int] = []     # carbon atoms of each -O-R organyl
    ester_oxys: list[int] = []      # the bridging O of each -O-R (for carving)
    direct_carbons: list[int] = []  # carbons bonded DIRECTLY to the centre

    for nb in centre.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(c_idx, nb.GetIdx())
        order = bond.GetBondTypeAsDouble()
        sym = nb.GetSymbol()
        ni = nb.GetIdx()

        if sym == "O":
            deg = nb.GetDegree()
            chg = nb.GetFormalCharge()
            nh = nb.GetTotalNumHs()
            if order == 2.0 and deg == 1 and chg == 0:
                n_oxo += 1
            elif order == 1.0 and deg == 1 and chg == 0 and nh == 1:
                n_oh += 1
            elif order == 1.0 and deg == 1 and chg == -1:
                n_oxide += 1
            elif order == 1.0 and deg == 2 and chg == 0 and nh == 0:
                # bridging O: must be an ester -O-R (R rooted at carbon).
                other = [x for x in nb.GetNeighbors()
                         if x.GetIdx() != c_idx][0]
                if other.GetAtomicNum() != 6:
                    # -O-X anhydride / -O-O- peroxy / -O-Si etc.: out of scope.
                    return None
                ester_oxys.append(ni)
                ester_roots.append(other.GetIdx())
            else:
                return None
        elif sym == "C" and order == 1.0:
            direct_carbons.append(ni)
        else:
            # Halide, S/Se/Te bridge, amido N, second centre, multi-bond C,
            # etc. — a thio/amido/halido ester or polynuclear acid, out of the
            # neutral O-ester generator's scope.
            return None

    # Must be a genuine O-organyl ester: at least one -O-R group.
    if not ester_roots:
        return None
    # The centre may carry a net charge model artefact? guarded above.

    # --- Tier / oxo / scheme via the parent-acid root logic. ---
    # Tier = count of single-bonded acidic O positions = free -OH + anionic
    # [O-] + esterified -O-R.  Direct-C substituents fill substitutable-H
    # positions and do NOT count toward the tier (mirrors phosphonic /
    # phosphinic where a C replaces the parent's substitutable H).
    tier = n_oh + n_oxide + len(ester_roots)
    rs = _mononuclear_root_and_suffix(element, tier, n_oxo)
    if rs is None:
        return None
    root, suffix, scheme = rs

    # Substitutable-position bookkeeping: on the standard tier acid the centre
    # carries (max_valence_positions - tier) substitutable H, each of which a
    # direct-C substituent (or a remaining H) fills.  For pnictogens / boron
    # the parent has 3 such positions beyond the oxo; for chalcogens none.
    n_h = centre.GetTotalNumHs()
    if scheme in ("pnictogen", "boron"):
        if len(direct_carbons) + n_h != 3 - tier:
            return None
    elif scheme == "chalcogen":
        # Chalcogen acids carry no substitutable H or C beyond the tier O / oxo
        # (sulfuric / sulfurous etc.).  A direct-C centre is a sulfonic /
        # sulfinic acid (carbon-bonded chalcogen) — a different family handled
        # by the substitutive suffix machinery, so decline here.
        if direct_carbons or n_h != 0:
            return None

    # Anion stem from the acid root (P-65.6.1 / P-72).
    anion = _anion_suffix(suffix)
    stem = _anion_stem(element, root)

    # --- Name the ester organyl groups (each cited as a SEPARATE WORD). ---
    # P-65.6.3.3.2.1: identical organyl groups are merged with a multiplying
    # prefix (di/tri/bis/tris…); DIFFERENT organyl groups are separate words
    # in alphanumerical order ("ethyl methyl phenyl phosphate").
    from iupac_namer.assembly import merge_identical_prefixes

    ester_names: list[str] = []
    for o_idx, r_idx in zip(ester_oxys, ester_roots):
        nm = _carve_and_name_organyl(mol, o_idx, r_idx)
        if nm is None:
            return None
        ester_names.append(nm)

    ester_block = _render_ester_words(ester_names)
    if not ester_block:
        return None

    # --- Name the direct-C substituents (glued prefix on the anion stem). ---
    # A carbon bonded directly to the centre is a substituent of the acid
    # itself (methylphosphonate, dimethylphosphinate); cite it as an ordinary
    # detachable prefix concatenated (no space) to the anion stem.
    stem_prefix = ""
    if direct_carbons:
        from iupac_namer.assembly import render_merged_prefixes

        c_names: list[str] = []
        for r_idx in direct_carbons:
            nm = _carve_and_name_organyl(mol, c_idx, r_idx)
            if nm is None:
                return None
            c_names.append(nm)
        c_merged = merge_identical_prefixes([(s, ()) for s in c_names])
        c_merged.sort(key=lambda mp: mp.sort_name)
        stem_prefix = render_merged_prefixes(c_merged)
        if not stem_prefix:
            return None

    # --- "hydrogen" word(s) for the remaining free -OH (partial ester). ---
    # An anionic [O-] position in a neutral whole molecule was guarded out
    # above (net charge != 0), so only neutral free -OH contributes here.
    hydrogen_word = ""
    if n_oh == 1:
        hydrogen_word = "hydrogen "
    elif n_oh > 1:
        mult = _MULT.get(n_oh)
        if mult is None:
            return None
        hydrogen_word = f"{mult}hydrogen "

    return f"{ester_block} {hydrogen_word}{stem_prefix}{stem}{anion}"
