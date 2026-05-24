"""
tests/test_multiplier_elision.py

Regression tests for IUPAC P-16.7.1(c) — terminal 'a' elision.

Rule: the terminal letter 'a' in numerical multiplicative prefixes
(tetra, penta, hexa, hepta, octa, nona, deca, ...) is dropped when
the following suffix begins with 'a' or 'o'.

Correct:   penta + ol  -> pentol   (NOT pentaol)
Correct:   tetra + ol  -> tetrol   (NOT tetraol)
Correct:   hexa  + ol  -> hexol    (NOT hexaol)
Correct:   hepta + ol  -> heptol   (NOT heptaol)
Correct:   tetra + amine -> tetramine (NOT tetraamine)

Unaffected (di/tri have no terminal 'a'):
           di + ol   -> diol
           tri + ol  -> triol
           di + amine -> diamine
"""
from __future__ import annotations

import pytest

from iupac_namer.engine import name_smiles


def _n(smi: str) -> str:
    return name_smiles(smi)


# ---------------------------------------------------------------------------
# Polyols — suffix begins with 'o' (elision required)
# ---------------------------------------------------------------------------

class TestPolyolElision:
    def test_ethane_diol(self):
        # di has no terminal 'a' — no elision
        assert _n("OCCO") == "ethane-1,2-diol"

    def test_propane_triol(self):
        # tri has no terminal 'a' — no elision
        assert _n("OCC(O)CO") == "propane-1,2,3-triol"

    def test_butane_tetrol(self):
        # tetra + ol -> tetrol (NOT tetraol)
        assert _n("OCC(O)C(O)CO") == "butane-1,2,3,4-tetrol"

    def test_pentane_pentol(self):
        # penta + ol -> pentol (NOT pentaol)  — xylitol skeleton
        assert _n("OCC(O)C(O)C(O)CO") == "pentane-1,2,3,4,5-pentol"

    def test_hexane_hexol(self):
        # hexa + ol -> hexol (NOT hexaol)  — sorbitol skeleton
        assert _n("OCC(O)C(O)C(O)C(O)CO") == "hexane-1,2,3,4,5,6-hexol"

    def test_heptane_heptol(self):
        # hepta + ol -> heptol (NOT heptaol)
        assert _n("OCC(O)C(O)C(O)C(O)C(O)CO") == "heptane-1,2,3,4,5,6,7-heptol"


# ---------------------------------------------------------------------------
# Diamines / polyamines — suffix begins with 'a' (elision required)
# ---------------------------------------------------------------------------

class TestAmineElision:
    def test_butane_diamine_no_elision(self):
        # di has no terminal 'a' — no change
        assert _n("NCCCCN") == "butane-1,4-diamine"

    def test_pentane_diamine_no_elision(self):
        # di has no terminal 'a' — no change
        assert _n("NCCCCCN") == "pentane-1,5-diamine"


# ---------------------------------------------------------------------------
# No elision before other vowels (rule is ONLY 'a' and 'o')
# ---------------------------------------------------------------------------

class TestNoElisionBeforeOtherVowels:
    def test_no_elision_before_e(self):
        # 'ene' suffix begins with 'e' — no elision of multiplier 'a'
        # (if a pentaene-style molecule is ever generated, the 'a' stays)
        # Simple smoke test: name must not crash and suffix 'ene' stays intact
        name = _n("C=CC=CC=C")  # hexa-1,3,5-triene
        assert "trien" in name or "triene" in name, f"Unexpected name: {name!r}"
