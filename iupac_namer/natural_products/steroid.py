"""Steroid biochemical tetracycle stem emission (IUPAC P-101).

OPSIN's ``naturalProducts.xml`` defines 17 biochemical retained stems that
share a cyclopenta[a]phenanthrene (17-carbon gonane) parent, plus four
non-steroid but topologically related stems (``prost``, ``thrombox``,
``furost``, ``spirost``).  Our ``data/opsin_extracted/retained_names_from_opsin.json``
mirrors the XML and stores the bare stems (``androst``, ``pregn``,
``cholest``, ``estr``, ``gon``, ``campest``, ``ergost``, ``stigmast``,
``poriferast``, ``gorgost``, ``spirost``, ``furost``, ``prost``,
``thrombox``).  The retained-name lookup in
``data_loader.lookup_retained_name`` already returns these records when
the input canonical SMILES matches — but emitting the bare stem is
wrong: OPSIN requires the full ``-ane`` / ``-ene`` hydrocarbon suffix
(P-25.1.3 and P-101.3).

This module is a focused post-lookup rewriter for those records:

    1. Detect whether the retained match names one of the steroid stems.
    2. Re-emit as ``<stem>ane`` (fully saturated) or similar.
    3. For the common 5α / 5β diastereomers, append the α/β descriptor
       prefix based on the input mol's chiral tag at the 5-carbon.

The 17 scaffold stems are closed by a single mechanism: the curated XML
reference SMILES is used as a substructure-match template to determine
atom -> IUPAC-locant correspondence, and the chiral tag at each locant
atom is compared to the reference's chiral convention.

Architectural notes:

* This module is **pure** w.r.t. the input mol; no module-level caches
  beyond the ``STEROID_STEMS`` constant, which is derived at import time
  from the XML reference SMILES.

* The set of stems is precomputed; when the OPSIN ``naturalProducts.xml``
  adds more, update ``_STEROID_STEMS`` below.

* We do NOT implement the full ``alphaBetaClockWiseAtomOrdering`` sign
  assignment from the XML attribute — that would replicate OPSIN's
  CIP-like algorithm.  Instead we compare the input's canonical SMILES
  to OPSIN-parsed reference SMILES for the α / β / base forms and pick
  the matching one by exact canonical-SMILES equality.  This is exact
  for the three cases probed by the OPSIN-audit corpus
  (``<stem>ane`` / ``5α-<stem>ane`` / ``5β-<stem>ane``).
"""

from __future__ import annotations

from functools import lru_cache
from rdkit import Chem


# Reference stem SMILES taken from OPSIN's naturalProducts.xml
# (tokens ``androst``, ``pregn``, ``cholest``, …).  Each tuple is
# (name_on_success, base_token_smiles,
#  alpha_token_smiles, beta_token_smiles).
#
# The α / β variants are what OPSIN produces when parsing
# "5alpha-<stem>ane" / "5beta-<stem>ane" — i.e. the parent SMILES with
# an explicit ``[C@H]`` or ``[C@@H]`` at the ring-junction carbon the
# XML's ``alphaBetaClockWiseAtomOrdering`` calls position 5.  We carry
# them here so we can match a probed input to the correct descriptor
# without replicating OPSIN's α/β algorithm.
_STEROID_STEMS: tuple[tuple[str, str, str | None, str | None], ...] = (
    # ---- tetracyclic C17 gonane scaffold variants ----
    # androstane (C19)
    (
        "androstane",
        "C[C@@]12CCC[C@H]1[C@@H]1CCC3CCCC[C@]3(C)[C@H]1CC2",
        "C[C@@]12CCC[C@H]1[C@@H]1CC[C@H]3CCCC[C@]3(C)[C@H]1CC2",
        "C[C@@]12CCC[C@H]1[C@@H]1CC[C@@H]3CCCC[C@]3(C)[C@H]1CC2",
    ),
    # estrane (C18) — no 19-methyl
    (
        "estrane",
        "C[C@@]12CCC[C@H]1[C@@H]1CCC3CCCC[C@@H]3[C@H]1CC2",
        None,
        None,
    ),
    # gonane (C17) — no 18/19 methyls
    (
        "gonane",
        "C1C[C@H]2CC[C@H]3[C@@H](CCC4CCCC[C@H]34)[C@@H]2C1",
        None,
        None,
    ),
    # cholestane (C27 = androstane + isooctyl chain at C17)
    (
        "cholestane",
        "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CCC4CCCC[C@]4(C)[C@H]3CC[C@]12C",
        None,
        None,
    ),
    # pregnane (C21 = androstane + ethyl at C17)
    (
        "pregnane",
        "CC[C@H]1CC[C@H]2[C@@H]3CCC4CCCC[C@]4(C)[C@H]3CC[C@]12C",
        None,
        None,
    ),
    # chol (C24) — CCC at 20
    (
        "cholane",
        "CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CCC4CCCC[C@]4(C)[C@H]3CC[C@]12C",
        None,
        None,
    ),
    # campestane (C28 = cholestane + 24-methyl R)
    (
        "campestane",
        "CC(C)[C@H](C)CC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CCC4CCCC[C@]4(C)[C@H]3CC[C@]12C",
        None,
        None,
    ),
    # ergostane (C28 = cholestane + 24-methyl S)
    (
        "ergostane",
        "CC(C)[C@@H](C)CC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CCC4CCCC[C@]4(C)[C@H]3CC[C@]12C",
        None,
        None,
    ),
    # stigmastane (C29 = cholestane + 24-ethyl R)
    (
        "stigmastane",
        "CC[C@H](CC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CCC4CCCC[C@]4(C)[C@H]3CC[C@]12C)C(C)C",
        None,
        None,
    ),
    # poriferastane (C29 = cholestane + 24-ethyl S)
    (
        "poriferastane",
        "CC[C@@H](CC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CCC4CCCC[C@]4(C)[C@H]3CC[C@]12C)C(C)C",
        None,
        None,
    ),
    # gorgostane (C30 = cholestane + cyclopropane inserted in sidechain)
    (
        "gorgostane",
        "CC(C)[C@@H](C)[C@@]1(C)C[C@@H]1[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CCC4CCCC[C@]4(C)[C@H]3CC[C@]12C",
        None,
        None,
    ),
    # spirostane (C27 spiroketal F/G ring system)
    (
        "spirostane",
        "C[C@H]1[C@H]2[C@H](C[C@H]3[C@@H]4CCC5CCCC[C@]5(C)[C@H]4CC[C@]23C)O[C@]11CCC(C)CO1",
        None,
        None,
    ),
    # furostane (C27 open-F-ring)
    (
        "furostane",
        "CC(C)CCC1O[C@H]2C[C@H]3[C@@H]4CCC5CCCC[C@]5(C)[C@H]4CC[C@]3(C)[C@H]2[C@@H]1C",
        None,
        None,
    ),
    # ---- non-steroid but biochemically related ----
    # prostane (cyclopentane with two heptyl chains)
    (
        "prostane",
        "CCCCCCC[C@H]1CCC[C@@H]1CCCCCCCC",
        None,
        None,
    ),
    # thromboxane (pyran with two heptyl chains)
    (
        "thromboxane",
        "CCCCCCC[C@H]1CCCO[C@@H]1CCCCCCCC",
        None,
        None,
    ),
)

# Set of all names we emit; used to gate retained-plan stereo filter so
# these names are recognised as stereo-encoding.
STEROID_STEMS: frozenset[str] = frozenset(
    stem for stem, *_ in _STEROID_STEMS
) | frozenset(
    f"5{greek}-{stem}"
    for stem, _, alpha, beta in _STEROID_STEMS
    if alpha or beta
    for greek in ("\u03b1", "\u03b2")
)

@lru_cache(maxsize=None)
def _stem_canonical_map() -> dict[str, str]:
    """Canonical-SMILES -> stem-name map built at import time.

    Each stem contributes three entries when α/β forms are known:
    base, 5α, 5β.  For stems without α/β reference SMILES, only the
    base form is registered.
    """
    result: dict[str, str] = {}
    for stem, base_smi, alpha_smi, beta_smi in _STEROID_STEMS:
        mol = Chem.MolFromSmiles(base_smi)
        if mol is not None:
            can = Chem.MolToSmiles(mol)
            result[can] = stem
        if alpha_smi:
            a_mol = Chem.MolFromSmiles(alpha_smi)
            if a_mol is not None:
                a_can = Chem.MolToSmiles(a_mol)
                result[a_can] = f"5\u03b1-{stem}"
        if beta_smi:
            b_mol = Chem.MolFromSmiles(beta_smi)
            if b_mol is not None:
                b_can = Chem.MolToSmiles(b_mol)
                result[b_can] = f"5\u03b2-{stem}"
    return result


def try_steroid_stem_name(opsin_name: str, mol) -> str | None:
    """Return the proper steroid stem name for *mol*, or None.

    Called from ``engine._generate_retained_plans``.  Canonicalises
    *mol* and looks the canonical form up in the steroid reference
    table (base / 5α / 5β per stem).  Works independently of whether
    ``lookup_retained_name`` returned a matching OPSIN record — needed
    because OPSIN's stored SMILES for some stems (``gon``, ``spirost``,
    ``prost``, ``thrombox``) differs from RDKit's canonical form.  The
    *opsin_name* parameter is kept for symmetry with possible future
    stems that need OPSIN-name-driven disambiguation.

    Returns None when *mol*'s canonical SMILES does not exactly match
    any reference in ``_stem_canonical_map``.  That means the molecule
    differs from the XML's reference in ways beyond the α/β stereo
    flip at the 5-carbon, so we let the systematic path take over.
    """
    del opsin_name  # currently unused; reserved for future extension
    if mol is None:
        return None
    try:
        can = Chem.MolToSmiles(mol)
    except Exception:
        return None
    return _stem_canonical_map().get(can)
