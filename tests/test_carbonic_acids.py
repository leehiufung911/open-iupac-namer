"""Tests for the GENERATIVE polynuclear carbonic-acid namer (P-65.2.3).

Exercises ``iupac_namer.perception.fg.carbonic_acids.compute_carbonic_acid_name``
and the engine dispatch hook that consumes it.  Carbonic acid and the di-,
tri-, and tetracarbonic acids are functional parent compounds with retained
preferred IUPAC names; their chalcogen / imido / hydrazono / peroxy
functional-replacement analogues (P-65.2.3.1.2) are named with replacement
prefixes, position locants, and the chain numbering "consecutively from one
end to the other, starting from and ending at a carbon atom"
(P-65.2.3.1.1).

Every name asserted here was verified to round-trip through OPSIN 2.8.0
(name -> SMILES -> same RDKit canonical) when the namer was built.  All the
1-/1,3-/1,1,3,3-/penta- examples are the Blue Book P-65.2.3.1.2.1/.2 worked
examples.  Because the generator branches only on structural features (chain
length, per-site element, bond order, bridge topology) it covers the whole
family, not a fixed set of molecules.
"""

from __future__ import annotations

import pytest
from rdkit import Chem

from iupac_namer.engine import name_smiles
from iupac_namer.perception.fg.carbonic_acids import (
    compute_carbonic_acid_name,
)


def _name(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None, f"bad test SMILES {smi!r}"
    return compute_carbonic_acid_name(mol)


# ---------------------------------------------------------------------------
# Unmodified retained polyacid parents (P-65.2)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smi,expected", [
    ("O=C(O)OC(=O)O", "dicarbonic acid"),
    ("O=C(O)OC(=O)OC(=O)O", "tricarbonic acid"),
    ("O=C(O)OC(=O)OC(=O)OC(=O)O", "tetracarbonic acid"),
])
def test_parent_polyacids(smi: str, expected: str) -> None:
    assert _name(smi) == expected


# ---------------------------------------------------------------------------
# Imido (=NH) replacement — Blue Book P-65.2.3.1.2.1 worked examples
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smi,expected", [
    ("N=C(O)OC(=O)O", "1-imidodicarbonic acid"),
    ("O=C(O)NC(=O)O", "2-imidodicarbonic acid"),
    ("N=C(O)OC(=N)O", "1,3-diimidodicarbonic acid"),
    ("N=C(O)NC(=N)NC(=N)O", "1,2,3,4,5-pentaimidotricarbonic acid"),
    ("N=C(O)NC(=N)NC(=N)NC(=N)O",
     "1,2,3,4,5,6,7-heptaimidotetracarbonic acid"),
])
def test_imido_replacement(smi: str, expected: str) -> None:
    assert _name(smi) == expected


# ---------------------------------------------------------------------------
# Thio (S / =S) replacement — including the all-positions "penta" form whose
# locants are omitted (P-65.2.3.1.2.2)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smi,expected", [
    ("O=C(O)SC(=O)O", "2-thiodicarbonic acid"),
    ("S=C(S)OC(=S)S", "1,1,3,3-tetrathiodicarbonic acid"),
    ("S=C(S)SC(=S)S", "pentathiodicarbonic acid"),
    ("O=C(O)SC(=O)SC(=O)O", "2,4-dithiotricarbonic acid"),
])
def test_thio_replacement(smi: str, expected: str) -> None:
    assert _name(smi) == expected


# ---------------------------------------------------------------------------
# Hydrazono (=N-NH2) and peroxy (-O-O-) replacement
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smi,expected", [
    ("NN=C(O)OC(=O)O", "1-hydrazonodicarbonic acid"),
    ("O=C(O)OC(=O)OO", "1-peroxydicarbonic acid"),
    ("O=C(OO)OC(=O)OO", "1,3-diperoxydicarbonic acid"),
    ("O=C(O)OOC(=O)O", "2-peroxydicarbonic acid"),
])
def test_hydrazono_peroxy_replacement(smi: str, expected: str) -> None:
    assert _name(smi) == expected


# ---------------------------------------------------------------------------
# Mixed-prefix chains — prefixes cited in alphabetical order (P-65.2.3.1.1)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smi,expected", [
    # imido on the doubly-bonded C1, thio bridge at position 2.
    ("N=C(O)SC(=O)O", "1-imido-2-thiodicarbonic acid"),
    # imido on both terminal C, thio bridge in the middle.
    ("N=C(O)SC(=N)O", "1,3-diimido-2-thiodicarbonic acid"),
])
def test_mixed_prefixes_alphabetical(smi: str, expected: str) -> None:
    assert _name(smi) == expected


# ---------------------------------------------------------------------------
# Selenium / tellurium replacement (the scheme generalises by element)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smi,expected", [
    ("[Se]=C([SeH])[Se]C(=[Se])[SeH]", "pentaselenodicarbonic acid"),
])
def test_selenium_replacement(smi: str, expected: str) -> None:
    assert _name(smi) == expected


# ---------------------------------------------------------------------------
# Lowest-locant direction (P-14.5) — asymmetric chains numbered from the end
# that gives the replacement set the lowest locants
# ---------------------------------------------------------------------------

def test_lowest_locant_direction() -> None:
    # 1-imidotricarbonic acid: =NH on a terminal carbon.  Numbering from the
    # imido end gives locant 1; from the other end it would be 5.
    assert _name("N=C(O)OC(=O)OC(=O)O") == "1-imidotricarbonic acid"


# ---------------------------------------------------------------------------
# Negative cases — the generator declines (returns None) so the molecule
# falls through to the ordinary machinery untouched.  No false positives.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smi", [
    "O=C(O)O",          # mononuclear carbonic acid (P-65.2.1 scheme)
    "N=C(O)O",          # mononuclear carbonimidic acid
    "CC(=O)OC(C)=O",    # acetic anhydride (C-C bonds, not a carbonic chain)
    "O=C(O)C(=O)O",     # oxalic acid (direct C-C, no -O- bridge)
    "O=C(O)CC(=O)O",    # malonic acid
    "O=C(OC)OC",        # dimethyl carbonate (ester, mononuclear)
    "O=C(O)OC(=O)OC",   # mixed acid/methyl ester (out of scope)
    "NC(=O)OC(=O)O",    # NH2 (amido) replacement — P-65.2.3.1.4 (out of scope)
    "ClC(=O)OC(=O)O",   # Cl (chlorido) replacement — P-65.2.3.1.3 (out of scope)
    "c1ccccc1",         # benzene
])
def test_declines_non_carbonic(smi: str) -> None:
    assert _name(smi) is None


# ---------------------------------------------------------------------------
# End-to-end engine dispatch: the whole-molecule name is the carbonic PIN.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smi,expected", [
    ("N=C(O)OC(=O)O", "1-imidodicarbonic acid"),
    ("S=C(S)OC(=S)S", "1,1,3,3-tetrathiodicarbonic acid"),
    ("O=C(O)OC(=O)O", "dicarbonic acid"),
    ("O=C(OO)OC(=O)OO", "1,3-diperoxydicarbonic acid"),
])
def test_engine_dispatch(smi: str, expected: str) -> None:
    assert name_smiles(smi) == expected
