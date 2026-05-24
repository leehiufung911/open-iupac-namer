"""iupac_namer.perception.fg.heteroelement_oxoacids

Mononuclear heteroelement oxoacid perception - Stage 6 R1-B.

Closes the Stage-5 coverage gap "heavy-heteroatom mononuclear oxoacids
absent" (cross-class root cause #2 in ``docs/opsin_coverage_taxonomy.md``).
Covers:

- **Pnictogens:** stiboric / stibonic / stibinic / stibonous acid and
  stibonite anion; phosphonic and phosphinic acid (filling round-trip
  gaps); arsinic acid.
- **Chalcogens:** selenic / selenous / selenate / selenite; telluric /
  tellurous / tellurate / tellurite / orthotelluric acid.
- **d-block:** chromic / dichromic acid; manganic / permanganic acid;
  technetic / pertechnetic acid; rhenic / perrhenic acid; perruthenic acid;
  and the lower-oxidation '-ous' series chromous / dichromous / manganous /
  permanganous / technetous / pertechnetous / rhenous / perrhenous /
  perruthenous acid (P-67 / IR-8 traditional names; one fewer terminal =O
  than the matching '-ic' acid, each keyed on its own canonical SMILES).
- **Sulfur amido:** sulfinamous acid (H2N-S-OH, the oxo-free lower form of
  the sulfamic/sulfinamic family — the generative namer covers the oxo>=1
  amidosulfuric/amidosulfurous members but not this oxo=0 retained name).
- **Main-group:** boric acid.
- **Halogen oxyacids:** hypochlorous / chlorous / chloric / perchloric;
  hypobromous / bromous / bromic / perbromic; hypoiodous / iodous /
  iodic / periodic / orthoperiodic; and every matching hypo-/-ate /
  -ite / -per-ate anion.

Design
------
Table-driven whole-molecule lookup, mirroring the architecture of
``data_loader._INORGANIC_CURATED_SMILES`` but isolated to its own file to
keep ownership of this functional-group subsystem crisp.  The table is
loaded once at import time from ``data/heteroelement_oxoacids.json`` and
keyed on the RDKit canonical SMILES of the standalone molecule.

Dispatch happens in ``engine.name`` immediately after the single-atom
shortcut: see the ``Heteroelement oxoacid whole-molecule shortcut`` block
in ``engine.py``.  Because the lookup is by *canonical* SMILES the match
is exact - so the engine never misfires on a substituted derivative
(e.g. a Sb-bearing ring fragment), and no existing retained path
(sulfuric, phosphoric, arsenic acid in ``_INORGANIC_CURATED_SMILES``) is
clobbered - those SMILES are simply not in our table.

Precedence vs carboxylic acid
-----------------------------
This dispatcher runs on the whole-molecule canonical SMILES before any
plan search.  It therefore cannot fire on a molecule that contains a
carboxylic acid, an ester, or any other principal characteristic group
in addition to the heteroelement oxoacid: the resulting canonical
SMILES would never equal one of our small-molecule keys.  The effective
"X-ic acid wins only when it's the sole PCG" semantics thus fall out of
the lookup key, not from an explicit seniority rule - exactly the
behaviour IUPAC P-66.6.1 prescribes for this class.
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
    / "heteroelement_oxoacids.json"
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
            f"heteroelement_oxoacids.json: 'entries' must be an object, got "
            f"{type(entries).__name__}"
        )
    # Freeze as a tuple-backed dict copy; no further mutation permitted.
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
    """Return the IUPAC name for a whole-molecule heteroelement oxoacid.

    Parameters
    ----------
    canonical_smiles:
        The RDKit canonical SMILES of the *entire* molecule (one fragment,
        all atoms).  The caller is responsible for canonicalising via
        ``Chem.MolToSmiles`` before handing it to us.

    Returns
    -------
    The IUPAC preferred name as a plain string, or None if this molecule
    is not in the heteroelement-oxoacid table.
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
