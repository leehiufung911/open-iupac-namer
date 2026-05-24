"""Tests for P-16.3.2 hyphen insertion between prefix section and digit-led parent stem.

When a substituted retained ring's parent stem begins with a digit
(e.g. "1H-imidazol", "1H-tetrazol") a hyphen must be inserted between the
rendered prefix section and the stem.  Without the fix the names would be
produced as "1-methyl1H-imidazole" (missing hyphen).
"""
import pytest
from iupac_namer.engine import name_smiles


@pytest.mark.parametrize("smiles,expected", [
    # N-methyl-1H-imidazole: prefix "1-methyl" + stem "1H-imidazol" → needs hyphen
    ("Cn1ccnc1", "1-methyl-1H-imidazole"),
    # 1,5-dimethyl-1H-tetrazole: two methyls + digit-led stem
    ("Cc1nnnn1C", "1,5-dimethyl-1H-tetrazole"),
    # Unsubstituted 1H-benzimidazole: no prefix, no hyphen issue — just check it works
    ("c1ccc2[nH]cnc2c1", "1H-benzimidazole"),
    # chlorobenzene: stem "benzen" starts with a letter → no extra hyphen
    ("ClC1=CC=CC=C1", "chlorobenzene"),
    # Chlorobenzene: prefix "chloro" + stem "benzen" → no spurious hyphen (letter-to-letter)
    ("Clc1ccccc1", "chlorobenzene"),
])
def test_stem_hyphenation(smiles, expected):
    """Confirm hyphen (or lack thereof) between prefix section and parent stem."""
    result = name_smiles(smiles)
    assert result == expected, f"For {smiles!r}: got {result!r}, expected {expected!r}"


def test_helper_needs_hyphen_before_stem():
    """Unit-test the helper directly."""
    from iupac_namer.assembly import _needs_hyphen_before_stem

    # Digit-led stem after letter: needs hyphen
    assert _needs_hyphen_before_stem("1-methyl", "1H-imidazol") is True
    assert _needs_hyphen_before_stem("dimethyl", "1H-tetrazol") is True
    assert _needs_hyphen_before_stem("chloro", "1,3-benzothiazol") is True

    # Letter-led stem: no hyphen needed
    assert _needs_hyphen_before_stem("chloro", "benzen") is False
    assert _needs_hyphen_before_stem("methyl", "cyclohexan") is False

    # Empty inputs: no hyphen
    assert _needs_hyphen_before_stem("", "1H-imidazol") is False
    assert _needs_hyphen_before_stem("methyl", "") is False

    # Previous ends with digit (not letter): no hyphen
    assert _needs_hyphen_before_stem("1-methyl-3", "1H-imidazol") is False
