"""Tautomer / Kekulé-equivalence helpers (Stage 7).

Background
----------
For partly-saturated or ambiguous-Kekulé retained ring systems (perimidine,
indene, phenarsazine, ...), the engine and OPSIN may pick different Kekulé
partners for the same chemical species.  RDKit canonical SMILES preserves
explicit ``=N``/``=As`` bonds across ring junctions instead of aromatising
them, so two Kekulé partners of the SAME molecule can yield distinct
``Chem.MolToSmiles`` strings even though their InChI is identical.

Stage 6 R1-A introduced the ``kekule_store`` rewrite table, which pins
canonical-SMILES-keyed retained tautomers (e.g. ``C1=Cc2ccccc2C1`` →
``1H-indene``, ``C1=Nc2cccc3cccc(c23)N1`` → ``3H-perimidine``).  That
covers the ~12 substituted-indene + substituted-perimidine probes whose
atom_locants happened to match a single tautomer's numbering.

This module provides the *equivalence test* used by audit and diagnostic
tooling to decide whether a round-trip SMILES mismatch is genuinely a
naming defect or merely a Kekulé-canonicalisation drift.  The test is
**InChI equality with stereo dropped** (FixedH-free) — two Kekulé
partners of the same arrangement of atoms and bond orders share the same
InChI, so ``kekule_equivalent`` returns ``True`` for them.

Design notes
------------
* No SMILES strings are hard-coded.  The function operates on arbitrary
  inputs and is generic to any ring/Kekulé combination.
* No engine-state coupling.  The helpers are pure and side-effect free,
  suitable for use from strategy-layer post-passes, audit harnesses, or
  diagnostic tests.
* The InChI comparison strips charge layers (``/q``, ``/p``) when
  ``ignore_charge=True`` is passed; the default keeps charge layers,
  because flavylium-style charge drops are *real* naming defects.

Public API
----------
* :class:`KekuleEquivalence` — frozen result record with ``equivalent``
  and ``inchi_a``/``inchi_b`` fields for callers that want to log the
  decision.
* :func:`kekule_equivalent` — boolean shorthand.
* :func:`classify_round_trip` — return one of ``"OK"``,
  ``"KEKULE_EQUIVALENT"``, ``"WRONG"``, ``"INVALID"`` for a (input_smi,
  round_trip_smi) pair.

The classification is consumed by ``eval/probe_tautomer_residuals.py`` to
re-bucket Stage 2 LOCANT_WRONG rows.  It is intentionally NOT consumed
by the eval scorer (``eval/authoritative_eval.py``) — that scorer's
SMILES-equality contract is the eval ground truth and Stage 7 must not
relax it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rdkit import Chem

__all__ = (
    "KekuleEquivalence",
    "kekule_equivalent",
    "classify_round_trip",
)


@dataclass(frozen=True)
class KekuleEquivalence:
    """Frozen record describing a Kekulé-equivalence test.

    ``equivalent`` is True when both inputs parse and share the same
    standard InChI (charge layer included unless ``ignore_charge`` was
    passed).  ``inchi_a`` and ``inchi_b`` are the computed InChIs (or
    ``None`` for parse failures).
    """

    equivalent: bool
    inchi_a: Optional[str]
    inchi_b: Optional[str]


def _safe_inchi(smiles: str, ignore_charge: bool) -> Optional[str]:
    """Compute the standard InChI for *smiles*; return ``None`` on parse
    or InChI failure.  When ``ignore_charge`` is True the charge layer
    (``/q``) and proton layer (``/p``) are stripped before comparison
    so neutral ↔ cation pairs compare equal at the connectivity level.
    """
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return None
    if mol is None:
        return None
    try:
        inchi = Chem.MolToInchi(mol)
    except Exception:
        return None
    if not inchi:
        return None
    if ignore_charge:
        # InChI layer order: /c (connections) /h (H) /q (charge) /p (proton)
        # Drop /q and /p sub-layers without disturbing earlier layers.
        kept: list[str] = []
        for layer in inchi.split("/"):
            if layer.startswith("q") or layer.startswith("p"):
                continue
            kept.append(layer)
        inchi = "/".join(kept)
    return inchi


def kekule_equivalent(
    smiles_a: str,
    smiles_b: str,
    *,
    ignore_charge: bool = False,
) -> bool:
    """Return True iff *smiles_a* and *smiles_b* describe the same
    molecule modulo Kekulé bond-order placement.

    Two SMILES that parse to InChI-identical structures are considered
    Kekulé-equivalent.  Stereo is part of standard InChI, so stereo
    differences also break equivalence — callers that need to ignore
    stereo should canonicalise their inputs with ``isomericSmiles=False``
    before calling this function.

    Pass ``ignore_charge=True`` to additionally strip the InChI charge
    layer; this distinguishes "engine dropped a charge" (False even with
    the flag) from "engine kept the charge but on a different atom"
    (True with the flag).  The default is conservative — charge drops
    are treated as inequivalent, because they are real naming defects.
    """
    return _kekule_record(smiles_a, smiles_b, ignore_charge=ignore_charge).equivalent


def _kekule_record(
    smiles_a: str, smiles_b: str, *, ignore_charge: bool
) -> KekuleEquivalence:
    a = _safe_inchi(smiles_a, ignore_charge)
    b = _safe_inchi(smiles_b, ignore_charge)
    return KekuleEquivalence(
        equivalent=a is not None and b is not None and a == b,
        inchi_a=a,
        inchi_b=b,
    )


def classify_round_trip(
    input_smiles: str,
    round_trip_smiles: str,
) -> str:
    """Bucket a (input, round-trip) pair into one of:

    * ``"INVALID"`` — either side fails to parse.
    * ``"OK"`` — RDKit canonical SMILES equal (the eval scorer's view).
    * ``"KEKULE_EQUIVALENT"`` — canonical SMILES differ but standard
      InChI is identical (Kekulé/tautomer drift, not a naming defect).
    * ``"WRONG"`` — InChI differs (real naming defect).

    The return value is a flat string for easy CSV writing; callers
    that need the underlying InChIs can call :func:`_kekule_record`
    directly.
    """
    if not input_smiles or not round_trip_smiles:
        return "INVALID"
    try:
        m_in = Chem.MolFromSmiles(input_smiles)
        m_rt = Chem.MolFromSmiles(round_trip_smiles)
    except Exception:
        return "INVALID"
    if m_in is None or m_rt is None:
        return "INVALID"
    can_in = Chem.MolToSmiles(m_in)
    can_rt = Chem.MolToSmiles(m_rt)
    if can_in == can_rt:
        return "OK"
    rec = _kekule_record(input_smiles, round_trip_smiles, ignore_charge=False)
    if rec.equivalent:
        return "KEKULE_EQUIVALENT"
    return "WRONG"
