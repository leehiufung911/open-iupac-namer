"""iupac_namer.perception.fg — fine-grained functional-group subsystems.

Each submodule here owns a narrow slice of functional-group / acid perception
that is too specialised to live in the top-level ``fg_detection.py`` SMARTS
loop.  Modules are dispatched directly by ``engine.name()`` (whole-molecule
shortcuts) or looked up by ``fg_detection.FGDetection`` via a registration
hook.

Current members:

- ``acid_infix_composition`` (Stage 6 R1-F):
  Table-driven dispatcher for OPSIN ``infixes.xml`` rules that the native
  ``functional_groups.json`` table does not round-trip (``nitrid``, ``tellur``,
  ``isocyanid``, ``isotellurocyanatid``, ``tellurocyanatid``,
  ``ditelluroperox``, ``hydroxim`` plus the partially covered ``azid`` /
  ``selenocyanatid`` / ``isoselenocyanatid`` cluster).
"""

from __future__ import annotations
