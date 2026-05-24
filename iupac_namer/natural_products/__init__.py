"""Natural-product naming helpers.

Subsystems in this package handle retained natural-product stems whose
emission needs logic beyond a simple SMILES -> name lookup.

Currently wired:
    * ``steroid`` — biochemical tetracycle stems (``androst``, ``pregn``,
      ``cholest``, etc.) re-emitted as ``<stem>ane`` / ``<stem>-N-ene`` /
      ``<stem>a-N,M-diene`` per IUPAC P-101, with α/β descriptor emission
      for ring-junction stereocentres that OPSIN's ``naturalProducts.xml``
      pins via ``alphaBetaClockWiseAtomOrdering``.
"""

from iupac_namer.natural_products.steroid import (
    STEROID_STEMS,
    try_steroid_stem_name,
)

__all__ = ["STEROID_STEMS", "try_steroid_stem_name"]
