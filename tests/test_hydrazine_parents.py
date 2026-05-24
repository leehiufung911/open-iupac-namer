"""
tests/test_hydrazine_parents.py

Unit tests for hydrazine (N-N) as parent hydride and hydrazinyl as substituent form.
Covers: standalone, mono-substituted, di-substituted, fully-substituted,
        hydrazinyl as ring substituent, and non-regression for baseline cases.

These tests verify both Fix 1 (hydrazinyl substituent_form) and Fix 2
(N-N heteroatom_chain parent hydride).
"""
from __future__ import annotations

import pytest

from iupac_namer.engine import name_smiles


@pytest.mark.parametrize("smi,expected", [
    # --- Fix 2: N-N as standalone parent ---
    ("NN", "hydrazine"),                          # must still match retained name

    # --- Fix 2: N-N as 2-atom parent with substituents ---
    ("CNN", "1-methylhydrazine"),                 # methyl on N1
    ("CNNC", "1,2-dimethylhydrazine"),            # methyl on both N atoms
    ("CN(C)N(C)C", "1,1,2,2-tetramethylhydrazine"),  # fully substituted

    # --- Fix 1: N-N as substituent (hydrazinyl form) ---
    # OPSIN accepts (hydrazinyl)benzene — benzene wins via PCG seniority
    ("NNc1ccccc1", "(hydrazinyl)benzene"),

    # --- Fix 1 + Fix 2: benzyl hydrazine — C attachment to ring ---
    # N-N parent is not used here (attachment via C); benzene parent + hydrazinylmethyl.
    # The methyl carbon has only one position (locant 1), so locant "1-" is redundant
    # per P-14.3.4.5. Fix B (1-carbon SUBSTITUENT locant suppression) removes it.
    # OPSIN round-trip confirms: [(hydrazinyl)methyl]benzene → NNCc1ccccc1 ✓
    ("NNCc1ccccc1", "[(hydrazinyl)methyl]benzene"),

    # --- Regression: unsubstituted parent hydrides must still work ---
    ("P", "phosphane"),
    ("[SiH4]", "silane"),
    ("B", "borane"),

    # --- Regression: PCG-bearing carbon chain beats N-N parent ---
    ("NNC=O", "methanohydrazide"),         # hydrazide PCG on methane chain wins
    ("NNC(=O)C", "ethanohydrazide"),       # propanohydrazide pattern

    # --- Regression: standard chain/ring naming unaffected ---
    ("CCO", "ethanol"),
    ("CC(=O)c1ccccc1", "1-phenylethanone"),
    ("CC(O)c1ccccc1", "1-phenylethanol"),
])
def test_hydrazine_parent(smi: str, expected: str) -> None:
    """Name SMILES *smi* and check the result matches *expected*."""
    result = name_smiles(smi)
    assert result == expected, (
        f"SMILES {smi!r}: expected {expected!r}, got {result!r}"
    )
