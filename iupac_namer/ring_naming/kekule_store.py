"""
iupac_namer/ring_naming/kekule_store.py

Kekulé-disambiguation rewrite table for retained ring names.

Background
----------
RDKit canonical SMILES for aromatic/partly-saturated rings aromatises the
bond pattern, so two distinct kekulé tautomers map to the same canonical
key.  Examples:

    1H-indene  (C1 sp3, C2=C3)          }
    2H-indene  (C2 sp3, C1=C3)          }  canonical: C1=Cc2ccccc2C1
    ind-1-ene  (C3 sp3, C1=C2)          }
    ind-2-ene  (C1 sp3, C2=C3)          }

    1H-perimidine  (N1 NH, C2=N3)       }  canonical: C1=Nc2cccc3cccc(c23)N1
    3H-perimidine  (N3 NH, N1=C2)       }

These tautomers have DIFFERENT ring-numbering conventions even though they
share a canonical SMILES.  When our engine emits ``ind-1-ene`` for a
molecule whose atom_locants table is pinned to 1H-indene numbering (sp3
at locant 1, benzene 4/5/6/7 on the C=C side), OPSIN parses the name
assuming the ind-1-ene convention (sp3 at 3, benzene 4/5/6/7 on the CH2
side) and the round-trip produces a mirror-imaged isomer.

Fix
----
For each canonical SMILES whose default emitted name is at odds with the
convention the ring's ``atom_locants`` table is pinned to, supply the
*correct* name/substituent form.  The lookup table is keyed on the
canonical SMILES produced by ``Chem.MolToSmiles`` and is consulted from
``retained_lookup.try_retained_name`` after the primary match.

The rewrite is *name-only* and does not touch ``atom_locants``: the
existing atom_locants are preserved because they already reflect the
tautomer the rewritten name picks.  No atoms are dropped, no guards are
relaxed.

Each entry is a ``KekuleRewrite`` namedtuple:

    KekuleRewrite(
        name="1H-indene",
        substituent_form="1H-inden-N-yl",   # optional
    )

The helper ``maybe_rewrite_for_kekule`` returns the rewrite when
applicable or ``None`` otherwise, leaving ambiguous/unknown canonicals
untouched (no regression risk).

See ``docs/opsin_coverage_taxonomy.md`` (root-cause #1) and
``docs/opsin_audit_rings.md`` top-20 table for the probes that motivated
each entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class KekuleRewrite:
    """Name rewrite applied to a retained-ring match for kekulé consistency."""

    # The IUPAC name to emit (e.g. "1H-indene", "3H-perimidine").
    name: str
    # Substituent form ("1H-inden-N-yl"), if any.  ``None`` means no
    # substituent-form override — the builder will derive one from ``name``
    # via the usual stem-stripping heuristics.
    substituent_form: Optional[str] = None


# ---------------------------------------------------------------------------
# Rewrite table.  Keyed by canonical SMILES (Chem.MolToSmiles output).
#
# Only enter a rewrite when the *current* emitted name's locant convention
# does NOT match the ``atom_locants`` table in
# ``retained_lookup._OPSIN_RING_ATOM_LOCANTS`` (or curated entry).  The
# rewrite makes the two agree, restoring SMILES round-trip fidelity.
#
# Every entry is covered by a regression test in
# ``tests/test_kekule_store.py``.
# ---------------------------------------------------------------------------

_REWRITES: dict[str, KekuleRewrite] = {
    # ------------------------------------------------------------------
    # indene — atom_locants pin sp3 CH2 (atom 8) to locant 1, C=C to
    # locants 2/3.  That is the 1H-indene convention.  OPSIN parses
    # ``ind-1-ene`` as "C1=C2 double bond, C3 sp3" — a 180°-rotated
    # numbering whose benzo ring 4/5/6/7 locants reflect across the
    # sp3 atom.  Emitting ``1H-indene`` realigns OPSIN with our table.
    # Covers WRONG rows for tokens: inden, indeno, ind-1-en, ind-2-en.
    # ------------------------------------------------------------------
    "C1=Cc2ccccc2C1": KekuleRewrite(
        name="1H-indene",
        substituent_form="1H-inden-1-yl",
    ),

    # ------------------------------------------------------------------
    # perimidine — atom_locants pin N (atom 1, no H) to locant 1 and
    # NH (atom 12) to locant 3.  That is the 3H-perimidine convention.
    # OPSIN's default ``perimidine`` (or ``perimidin``) places the NH
    # at locant 1 (the 1H tautomer), which differs on any asymmetric
    # substitution probe — giving N=CN vs NC=N round-trip mismatches.
    # Emitting ``3H-perimidine`` pins OPSIN to the tautomer whose
    # numbering matches our atom_locants.
    # Covers WRONG rows for tokens: perimidin, perimidino.
    # ------------------------------------------------------------------
    "C1=Nc2cccc3cccc(c23)N1": KekuleRewrite(
        name="3H-perimidine",
        substituent_form="3H-perimidin-1-yl",
    ),
}


def maybe_rewrite_for_kekule(canonical_smiles: str) -> Optional[KekuleRewrite]:
    """Return a :class:`KekuleRewrite` for ``canonical_smiles`` if one is
    registered, else ``None``.

    ``canonical_smiles`` is the output of ``Chem.MolToSmiles`` on the
    ring-mol the retained-name lookup matched against.  The rewrite is
    applied only when a match is present — if the canonical is unknown
    (the vast majority of rings), the caller falls back to the default
    emitted name and numbering.
    """
    return _REWRITES.get(canonical_smiles)
