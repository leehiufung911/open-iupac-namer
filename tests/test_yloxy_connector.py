"""
tests/test_yloxy_connector.py

Regression tests for the -yloxy connector fix (Cluster 4).

When a substituent has the shape <ring>-O- (a ring O-ether attached at a ring atom),
assembly must emit <ring-stem>-N-yloxy, NOT <ring-stem>-N-oxy.

The distinction (P-63.6.1.1):
- Simple alkyl ethers: the "-yl" is elided into the contraction (methyl→methoxy,
  ethyl→ethoxy, propyl→propoxy, butyl→butoxy, propan-2-yl→propan-2-yloxy*).
- Locant-bearing ring substituents: the "-yl" must be retained, giving "-yloxy"
  (pyridin-4-yl→pyridin-4-yloxy, oxan-4-yl→oxan-4-yloxy).

*propan-2-yloxy is the IUPAC form; "isopropyloxy" or "isopropoxy" are also
 accepted retained names but the systematic form uses "-yloxy".
"""
from __future__ import annotations

import pytest

from iupac_namer.engine import name_smiles


# ---------------------------------------------------------------------------
# Regression: ring O-ether substituents must produce -yloxy (not -oxy)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected_fragment", [
    # Pyridine as ring O-ether substituent (chain-acid PCG → chain wins as parent)
    ("OC(=O)CCOc1ccncc1", "pyridin-4-yloxy"),
    # Pyridine as ring O-ether substituent (amine PCG → chain wins as parent)
    ("NCCOc1ccncc1", "pyridin-4-yloxy"),
])
def test_ring_yloxy_produced(smiles, expected_fragment):
    """Ring O-ether substituents must emit -yloxy, not -oxy."""
    result = name_smiles(smiles)
    assert expected_fragment in result, (
        f"Expected '{expected_fragment}' in name of {smiles!r}, got: {result!r}"
    )


@pytest.mark.parametrize("smiles,rejected_fragment", [
    # These must NOT contain the bare -oxy form (without -yl-)
    ("OC(=O)CCOc1ccncc1", "pyridin-4-oxy"),
])
def test_ring_bare_oxy_absent(smiles, rejected_fragment):
    """Ring O-ether substituents must NOT emit bare -oxy (missing -yl-)."""
    result = name_smiles(smiles)
    assert rejected_fragment not in result, (
        f"Must not contain '{rejected_fragment}' in name of {smiles!r}, got: {result!r}"
    )


# ---------------------------------------------------------------------------
# Regressions: simple alkyl ether contractions must still work
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected_fragment", [
    # methyl → methoxy (not methyloxy)
    ("COC", "methoxy"),
    # ethyl → ethoxy
    ("CCOC", "methoxy"),   # methoxyethane: the methoxy is on the methane side
    # ethyl ether on benzene: ethoxybenzene
    ("CCOc1ccccc1", "ethoxy"),
    # propyl ether
    ("CCCOc1ccccc1", "propoxy"),
    # butyl ether
    ("CCCCOc1ccccc1", "butoxy"),
])
def test_simple_alkyl_ether_contraction(smiles, expected_fragment):
    """Simple aliphatic ethers must still contract: methyl→methoxy, ethyl→ethoxy, etc."""
    result = name_smiles(smiles)
    assert expected_fragment in result, (
        f"Expected '{expected_fragment}' in name of {smiles!r}, got: {result!r}"
    )


@pytest.mark.parametrize("smiles,must_not_contain", [
    # methyl ether must NOT appear as "methyloxy"
    ("COC", "methyloxy"),
    ("CCOC", "ethyloxy"),
    ("CCOc1ccccc1", "ethyloxy"),
])
def test_simple_alkyl_not_yloxy(smiles, must_not_contain):
    """Simple aliphatic ethers must NOT use the expanded '-yloxy' form."""
    result = name_smiles(smiles)
    assert must_not_contain not in result, (
        f"Must not contain '{must_not_contain}' in name of {smiles!r}, got: {result!r}"
    )
