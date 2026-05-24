"""iupac_namer.perception.fg.phosphorus_oxoacids

Polynuclear phosphorus oxoacid perception - Stage 6 R2-F.

Closes root cause #12 in ``docs/opsin_coverage_taxonomy.md`` and Gap 3 in
``docs/opsin_audit_fg.md``: the engine has no generator for P-O-P chain
acids (``diphosphoric`` / ``pyrophosphoric`` / ``triphosphoric``) nor for
P-P-direct-bond chain acids (``hypodiphosphoric``).  The mononuclear
``phosphoric acid`` and ``phosphonic acid`` cases work through
``data_loader._INORGANIC_CURATED_SMILES`` and the R1-B
``heteroelement_oxoacids`` table respectively; this module is the
polynuclear complement.

Design
------
Table-driven whole-molecule lookup, mirroring the architecture of the
R1-B ``heteroelement_oxoacids`` subsystem (same file-ownership and
dispatch pattern).  The table is loaded once at import time from
``data/polynuclear_p_oxoacids.json`` and keyed on the RDKit canonical
SMILES of the standalone molecule.

Dispatch happens in ``engine.name`` immediately after the R1-B
heteroelement-oxoacid shortcut: see the ``Polynuclear phosphorus oxoacid
whole-molecule shortcut`` block in ``engine.py``.  Because the lookup
is by *canonical* SMILES the match is exact - so the engine never
misfires on a substituted derivative (e.g. a methyl diphosphate ester
``COP(=O)(O)OP(=O)(O)O``) and no existing mononuclear path is clobbered
(``O=P(O)(O)O`` is simply not in our table).

Scope - what this handler does NOT cover
----------------------------------------

* **Cyclic polyphosphoric acids** (``cyclotriphosphoric`` /
  ``metaphosphoric``).  OPSIN 2.8.0 cannot parse either name, so we
  cannot round-trip-verify an emission; omitting them keeps the
  architecture clean without touching assembly.
* **Partial-anion salts.**  The fully deprotonated anions (``diphosphate``
  / ``triphosphate`` / ``hypodiphosphate``) already round-trip through
  the descriptive ``{[bis(oxido)(oxo)phosphanyl]oxy}...`` form produced
  by the ordinary plan machinery, so they need no new path here.
* **The ``hypodiphosphonic`` / ``hypodiphosphorous`` / ``diphosphonic``
  descriptive analogues.**  OPSIN accepts our existing substitutive
  output for those (``[(hydroxy)(oxo)phosphanyl](hydroxy)(oxo)phosphane``
  and the likes), so the round-trip is already COVERED.
* **Phosphoronitridic acid** ``N#P(O)O``.  Closed by the R1-F
  ``acid_infix_composition`` infix dispatcher; this module does not
  re-implement it.

Precedence
----------
Runs on the whole-molecule canonical SMILES *before* plan search, and
*after* the mononuclear R1-B table.  The effective "polynuclear wins
only when it's the sole functional ensemble" semantics fall out of the
exact-SMILES lookup key, not from an explicit seniority rule.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from rdkit import Chem  # noqa: F401


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_TABLE_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data"
    / "polynuclear_p_oxoacids.json"
)


def _load_table() -> dict[str, str]:
    """Load the canonical-SMILES -> name table from JSON.

    Stored once in a module-level cache; never mutated at runtime so it
    respects the "frozen data structures" principle in CLAUDE.md.
    """
    with open(_TABLE_PATH, encoding="utf-8") as f:
        payload = json.load(f)
    entries = payload.get("entries", {})
    if not isinstance(entries, dict):  # pragma: no cover
        raise RuntimeError(
            f"polynuclear_p_oxoacids.json: 'entries' must be an object, got "
            f"{type(entries).__name__}"
        )
    return dict(entries)


# Module-level cache. Populated on first access.
_TABLE: dict[str, str] | None = None


def get_table() -> dict[str, str]:
    """Return the module-level canonical-SMILES -> name table.

    Lazy so that simply importing this module does not read from disk.
    """
    global _TABLE
    if _TABLE is None:
        _TABLE = _load_table()
    return _TABLE


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def lookup_name(canonical_smiles: str) -> str | None:
    """Return the IUPAC name for a whole-molecule polynuclear P oxoacid.

    Parameters
    ----------
    canonical_smiles:
        The RDKit canonical SMILES of the *entire* molecule (one fragment,
        all atoms).  The caller is responsible for canonicalising via
        ``Chem.MolToSmiles`` before handing it to us.

    Returns
    -------
    The IUPAC preferred name as a plain string, or None if this molecule
    is not in the polynuclear-P-oxoacid table.
    """
    if not canonical_smiles:
        return None
    return get_table().get(canonical_smiles)


def lookup_mol(mol) -> str | None:
    """Convenience: canonicalise a Mol and look it up.

    Useful in engine dispatch where the caller already has the Mol but
    not the canonical SMILES string.
    """
    try:
        from rdkit import Chem  # local import: lets this module be loaded without RDKit
    except ImportError:  # pragma: no cover
        return None
    if mol is None:
        return None
    canonical = Chem.MolToSmiles(mol)
    return lookup_name(canonical)


def is_registered(canonical_smiles: str) -> bool:
    """Cheap membership test for callers that only care about existence."""
    return canonical_smiles in get_table()


def all_names() -> tuple[str, ...]:
    """Sorted tuple of all names in the table - used by the test suite."""
    return tuple(sorted(set(get_table().values())))
