"""Regression tests for multi-heteroatom Hantzsch-Widman rings with several
mixed chalcogens (S / Se / Te / O) — P-22.1.2 / P-22.1.3.3 / P-31.1.4 HW.

The HW path in ``iupac_namer/ring_naming/monocyclic.py`` already handles rings
that carry two-to-four mixed chalcogens across ring sizes 3-8, in both the
saturated and the mancude (maximally non-cumulative unsaturated) regimes.  This
module locks that behaviour in:

  * heteroatom seniority for locant assignment and prefix ordering is
    O > S > Se > Te (``_HW_PRIORITY``), so e.g. ``O1CCSCC1`` is ``1,4-oxathiane``
    (oxa before thia) and ``[Se]1[Te]CCCC1`` is ``1,2-selenatellurane`` (selena
    before tellura);
  * the ``-thia- / -selena- / -tellura-`` replacement prefixes are emitted with
    HW elision (``oxa`` + ``dithiane`` → ``oxadithiane``);
  * the HW stem follows ring size + saturation (``-irane/-etane/-olane/-ane``
    saturated; ``-ine/-ole`` mancude; ``-epane/-ocane`` for 7/8 saturated);
  * mancude carrier-free rings get the unsaturated stem (``1,4-dithiine``) and
    mancude rings with an sp3 carrier get the indicated-H prefix
    (``3H-1,2-dithiole``).

"Correct" is defined as: the engine emits the expected IUPAC string AND that
string round-trips through OPSIN to the input structure.  A name that parses
back to a different molecule is wrong even if it is a legal IUPAC string.

These cases were each verified parseable through OPSIN 2.8.0 before being
pinned here (the SMILES column is the canonical OPSIN output for the name).
"""
from __future__ import annotations

import os

os.environ.setdefault(
    "JAVA_HOME",
    os.environ.get("JAVA_HOME", ""),
)
os.environ["PATH"] = (
    os.environ["JAVA_HOME"] + "/bin" + os.pathsep + os.environ.get("PATH", "")
)

import pytest
from rdkit import Chem

from iupac_namer.engine import name_smiles


def _canonical(smiles: str) -> str | None:
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(m) if m is not None else None


def _opsin_round_trip(name: str) -> str | None:
    try:
        from py2opsin import py2opsin
    except ImportError:  # pragma: no cover
        pytest.skip("py2opsin not installed")
    out = py2opsin(name)
    if not out:
        return None
    return _canonical(out)


# ---------------------------------------------------------------------------
# 1. Saturated multi-chalcogen HW rings — 2 mixed chalcogens
# ---------------------------------------------------------------------------

SATURATED_DI = [
    ("O1CCSCC1", "1,4-oxathiane"),          # O > S seniority: oxa before thia
    ("O1CC[Se]CC1", "1,4-oxaselenane"),     # O > Se
    ("S1CC[Se]CC1", "1,4-thiaselenane"),    # S > Se
    ("[Se]1CC[Se]CC1", "1,4-diselenane"),   # two Se → diselena
    ("[Te]1CC[Te]CC1", "1,4-ditellurane"),  # two Te → ditellura
    ("S1[Te]CCCC1", "1,2-thiatellurane"),   # S > Te, adjacent
    ("[Se]1[Te]CCCC1", "1,2-selenatellurane"),  # Se > Te, adjacent
]


# ---------------------------------------------------------------------------
# 2. Saturated multi-chalcogen HW rings — 3+ chalcogens
# ---------------------------------------------------------------------------

SATURATED_POLY = [
    ("S1SSCCC1", "1,2,3-trithiane"),
    ("[Se]1[Se][Se]CCC1", "1,2,3-triselenane"),
    ("O1SSCCC1", "1,2,3-oxadithiane"),       # O@1, two S@2,3
    ("S1CSCSC1", "1,3,5-trithiane"),         # alternating S
    ("O1[Se][Se]CCC1", "1,2,3-oxadiselenane"),
    ("S1SCSSC1", "1,2,4,5-tetrathiane"),
    ("S1SSSSS1", "1,2,3,4,5,6-hexathiane"),
    ("[Se]1[Se][Se][Se][Se][Se]1", "1,2,3,4,5,6-hexaselenane"),
    ("O1SSSCC1", "1,2,3,4-oxatrithiane"),    # O@1, three S@2,3,4
]


# ---------------------------------------------------------------------------
# 3. Smaller rings (3/4/5-membered) with mixed chalcogens
# ---------------------------------------------------------------------------

SMALL_RING = [
    # 5-ring
    ("S1CSCC1", "1,3-dithiolane"),
    ("O1CSCC1", "1,3-oxathiolane"),
    ("O1C[Se]CC1", "1,3-oxaselenolane"),
    ("[Se]1[Se]C[Se]C1", "1,2,4-triselenolane"),
    ("S1[Se][Se]CC1", "1,2,3-thiadiselenolane"),   # S > Se
    # 4-ring
    ("S1SCC1", "1,2-dithietane"),
    ("S1CSC1", "1,3-dithietane"),
    ("[Se]1[Se]CC1", "1,2-diselenetane"),
    ("O1SCC1", "1,2-oxathietane"),
    # 3-ring
    ("S1SC1", "1,2-dithiirane"),
    ("O1SC1", "1,2-oxathiirane"),
    ("S1[Se]C1", "1,2-thiaselenirane"),
]


# ---------------------------------------------------------------------------
# 4. Larger rings (7/8-membered) with mixed chalcogens
# ---------------------------------------------------------------------------

LARGE_RING = [
    ("S1CCSCCC1", "1,4-dithiepane"),
    ("O1CCSCCC1", "1,4-oxathiepane"),
    ("S1SSCCCC1", "1,2,3-trithiepane"),
    ("S1CCCSCCC1", "1,5-dithiocane"),
]


# ---------------------------------------------------------------------------
# 5. Mancude (unsaturated) multi-chalcogen HW rings
# ---------------------------------------------------------------------------

MANCUDE = [
    # carrier-free mancude 6-rings → unsaturated stem
    ("S1C=CSC=C1", "1,4-dithiine"),
    ("O1C=CSC=C1", "1,4-oxathiine"),
    ("[Se]1C=C[Se]C=C1", "1,4-diselenine"),
    ("S1SC=CC=C1", "1,2-dithiine"),
    # mancude 5-rings with an sp3 carrier → indicated-H prefix
    ("S1SCC=C1", "3H-1,2-dithiole"),
    ("S1CSC=C1", "2H-1,3-dithiole"),
    ("O1CSC=C1", "2H-1,3-oxathiole"),
    ("[Se]1[Se][Se]C=C1", "1,2,3-triselenole"),
]


# ---------------------------------------------------------------------------
# 6. The task's literal closed-ring analogues
# ---------------------------------------------------------------------------

TASK_ANALOGUES = [
    # 4-ring, 3 Se + 1 S → seniority puts S@1, the three Se@2,3,4
    ("[Se]1[Se][Se]S1", "1,2,3,4-thiatriselenetane"),
    # 6-ring O + 2 S → oxa before dithia; O@1 forces 1,2,6 numbering
    ("C1CSOSC1", "1,2,6-oxadithiane"),
]


ALL_CASES = (
    SATURATED_DI + SATURATED_POLY + SMALL_RING + LARGE_RING
    + MANCUDE + TASK_ANALOGUES
)


@pytest.mark.parametrize("smi,expected", ALL_CASES)
def test_multi_chalcogen_hw_name(smi, expected):
    """Engine emits the expected HW name (seniority, prefixes, stem, locants)."""
    assert name_smiles(smi) == expected


@pytest.mark.parametrize("smi,_expected", ALL_CASES)
def test_multi_chalcogen_hw_round_trip(smi, _expected):
    """Every emitted name parses back through OPSIN to the input structure."""
    got = name_smiles(smi)
    assert not got.startswith("[NAMING ERROR"), f"Namer failed for {smi}: {got}"
    assert _opsin_round_trip(got) == _canonical(smi), (
        f"Round-trip mismatch for {smi}: name={got!r}"
    )


class TestSeniorityOrdering:
    """Heteroatom seniority O > S > Se > Te governs both locant 1 placement
    and the prefix order (P-22.1.3.3 / P-31.1.2.2)."""

    def test_oxa_before_thia(self):
        # O takes locant 1; prefix order oxa-thia (not thia-oxa).
        assert name_smiles("O1CCSCC1") == "1,4-oxathiane"

    def test_thia_before_selena(self):
        assert name_smiles("S1CC[Se]CC1") == "1,4-thiaselenane"

    def test_selena_before_tellura(self):
        assert name_smiles("[Se]1[Te]CCCC1") == "1,2-selenatellurane"

    def test_senior_gets_locant_one(self):
        # In O + 2S the oxygen (senior) must take locant 1 -> 1,2,6 set.
        assert name_smiles("C1CSOSC1") == "1,2,6-oxadithiane"
