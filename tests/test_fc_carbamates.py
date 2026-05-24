"""
tests/test_fc_carbamates.py

Functional Class (FC) path for carbamates (R-O-C(=O)-N(R')(R'')).
Expected outputs verified against OPSIN canonical names.
"""
import logging
import pytest

logging.disable(logging.WARNING)

from iupac_namer.engine import name_smiles


@pytest.mark.parametrize("smiles,expected", [
    # Unsubstituted N (NH2)
    ("CCOC(=O)N",         "ethyl carbamate"),
    # N-monosubstituted
    ("CCOC(=O)NC",        "ethyl N-methylcarbamate"),
    ("CCCCOC(=O)Nc1ccccc1", "butyl N-phenylcarbamate"),
    ("CC(C)OC(=O)Nc1cccc(Cl)c1", "propan-2-yl N-(3-chlorophenyl)carbamate"),
    # N,N-disubstituted
    ("CCOC(=O)N(C)C",     "ethyl N,N-dimethylcarbamate"),
])
def test_carbamate_fc_naming(smiles, expected):
    result = name_smiles(smiles)
    assert result == expected, f"SMILES={smiles!r}: got {result!r}, expected {expected!r}"
