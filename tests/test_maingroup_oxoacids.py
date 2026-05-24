"""Tests for the GENERATIVE main-group oxoacid namer (Stage 15).

Exercises ``iupac_namer.perception.fg.maingroup_oxoacids.compute_name`` and
the two engine dispatch hooks that consume it.  Unlike the lookup-table
subsystems (``heteroelement_oxoacids`` / ``phosphorus_oxoacids``), this namer
COMPUTES names from structural features, so the assertions cover whole
families — mononuclear / polynuclear chains, anhydride vs direct (hypo) vs
thionic vs peroxy linkages, the -or/-on/-in pnictogen tiers, chalcogen and
halogen schemes, and the anion / hydrogen modifiers.

All names asserted here were verified to round-trip through OPSIN 2.8.0
(name -> SMILES -> same RDKit canonical) when the namer was built; see
P-67.1.1.1 / P-67.2.1 for the IUPAC citations.
"""

from __future__ import annotations

import pytest
from rdkit import Chem

from iupac_namer.engine import name_smiles
from iupac_namer.perception.fg.maingroup_oxoacids import (
    compute_name,
    compute_substituted_n_oxoacid_name,
)


def _name(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None, f"bad test SMILES {smi!r}"
    return compute_name(mol)


def _name_subst_n(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None, f"bad test SMILES {smi!r}"
    return compute_substituted_n_oxoacid_name(mol)


# ---------------------------------------------------------------------------
# Mononuclear acids — pnictogen tier scheme (-or/-on/-in, ic/ous)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smi,expected", [
    ("O=[As](O)(O)O", "arsoric acid"),
    ("O=[AsH](O)O", "arsonic acid"),
    ("O=[AsH2]O", "arsinic acid"),
    ("O[As](O)O", "arsorous acid"),
    ("O[AsH]O", "arsonous acid"),
    ("O[AsH2]", "arsinous acid"),
    ("[O-][NH+](O)O", "azonic acid"),
    ("[O-][N+](O)(O)O", "nitroric acid"),
    ("ON(O)O", "azorous acid"),
    ("ONO", "azonous acid"),
])
def test_mononuclear_pnictogen(smi: str, expected: str) -> None:
    assert _name(smi) == expected


# ---------------------------------------------------------------------------
# Mononuclear acids — chalcogen and boron schemes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smi,expected", [
    ("O=[Se](=O)(O)O", "selenic acid"),
    ("O=[Se](O)O", "selenous acid"),
    ("O=[Te](=O)(O)O", "telluric acid"),
    ("O=[Te](O)O", "tellurous acid"),
    ("OB(O)O", "boric acid"),
    ("OBO", "boronic acid"),
    ("OB", "borinic acid"),
])
def test_mononuclear_chalcogen_boron(smi: str, expected: str) -> None:
    assert _name(smi) == expected


# ---------------------------------------------------------------------------
# Polynuclear chains — anhydride (di/tri), direct (hypo)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smi,expected", [
    ("O=[Se](=O)(O)O[Se](=O)(=O)O", "diselenic acid"),
    ("O=[Se](O)O[Se](=O)O", "diselenous acid"),
    ("O=[Se](=O)(O)O[Se](=O)(=O)O[Se](=O)(=O)O", "triselenic acid"),
    ("O=[Te](=O)(O)O[Te](=O)(=O)O", "ditelluric acid"),
    ("OB(O)OB(O)O", "diboric acid"),
    ("OB(O)B(O)O", "hypoboric acid"),
    ("OBBO", "hypodiboronic acid"),
    ("O=[Se](=O)(O)[Se](=O)(=O)O", "hypodiselenic acid"),
    ("O=[As](O)(O)[As](=O)(O)O", "hypodiarsoric acid"),
    ("O=[As](O)O[As](=O)O", "diarsonic acid"),
    ("[O]=[Sb]([OH])[O][Sb](=[O])[OH]", "distibonic acid"),
])
def test_polynuclear_chains(smi: str, expected: str) -> None:
    assert _name(smi) == expected


# ---------------------------------------------------------------------------
# Sulfur thionic series + peroxy + amido
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smi,expected", [
    ("O=S(=O)(O)S(=O)(=O)O", "dithionic acid"),
    ("O=S(O)S(=O)O", "dithionous acid"),
    ("O=S(=O)(O)SS(=O)(=O)O", "trithionic acid"),
    ("O=S(=O)(O)SSS(=O)(=O)O", "tetrathionic acid"),
    ("O=S(=O)(O)SSSS(=O)(=O)O", "pentathionic acid"),
    ("O=S(=O)(O)OO", "peroxysulfuric acid"),
    ("O=S(=O)(O)OOS(=O)(=O)O", "peroxydisulfuric acid"),
    ("NS(=O)(=O)O", "amidosulfuric acid"),
    ("NS(=O)O", "amidosulfurous acid"),
])
def test_thionic_peroxy_amido(smi: str, expected: str) -> None:
    assert _name(smi) == expected


# ---------------------------------------------------------------------------
# Anion / hydrogen forms
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smi,expected", [
    ("O=S(=O)([O-])O", "hydrogen sulfate"),
    ("O=S([O-])O", "hydrogen sulfite"),
    ("O=S([O-])[O-]", "sulfite"),
    ("O=S(=O)([O-])OS(=O)(=O)[O-]", "disulfate"),
    ("O=S([O-])OS(=O)[O-]", "disulfite"),
    ("O=S(=O)([O-])OOS(=O)(=O)[O-]", "peroxydisulfate"),
    ("NS(=O)(=O)[O-]", "amidosulfate"),
])
def test_anion_forms(smi: str, expected: str) -> None:
    assert _name(smi) == expected


# ---------------------------------------------------------------------------
# Halogen oxoacids and anions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smi,expected", [
    ("OCl", "hypochlorous acid"),
    ("[O-][Cl+]O", "chlorous acid"),
    ("[O-][Cl+2]([O-])O", "chloric acid"),
    ("[O-][Cl+3]([O-])([O-])O", "perchloric acid"),
    ("[O-][Cl+2]([O-])[O-]", "chlorate"),
    ("[O-]Cl", "hypochlorite"),
    ("[O-][I+]([O-])([O-])([O-])([O-])[O-]", "orthoperiodate"),
])
def test_halogen_oxoacids(smi: str, expected: str) -> None:
    assert _name(smi) == expected


# ---------------------------------------------------------------------------
# Self-gating: must NOT fire on non-oxoacid skeletons
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smi", [
    "CCO",                 # carbon present
    "CC(=O)O",             # carboxylic acid
    "OS(=O)(=O)c1ccccc1",  # benzenesulfonic acid (ring/carbon)
    "O=P(=O)O",            # phosphenic — non-standard oxo count
    "O=[As]O",             # arsenenous — non-standard mononuclear H
    "[CH3]",               # carbon radical
])
def test_does_not_fire_on_non_oxoacid(smi: str) -> None:
    mol = Chem.MolFromSmiles(smi)
    assert compute_name(mol) is None


# ---------------------------------------------------------------------------
# End-to-end engine integration (names emitted by the dispatch hooks)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smi,expected", [
    ("O=[Se](=O)(O)O[Se](=O)(=O)O", "diselenic acid"),
    ("O=S(=O)([O-])O", "hydrogen sulfate"),
    ("O=S(=O)(O)SS(=O)(=O)O", "trithionic acid"),
    ("O=[As](O)O[As](=O)O", "diarsonic acid"),     # via pre-guard dispatch
    ("OBO", "boronic acid"),
])
def test_engine_integration(smi: str, expected: str) -> None:
    assert name_smiles(smi) == expected


def test_engine_radical_guard_intact() -> None:
    """A genuine non-oxoacid radical must still be rejected by the guard."""
    with pytest.raises(ValueError):
        name_smiles("[O][O]")


# ---------------------------------------------------------------------------
# Carbon-substituted nitrogen oxoacids: ACIDS (P-67.1.1.2)
# ---------------------------------------------------------------------------
# Unchanged behaviour — the acid forms must keep naming after the anion
# extension landed.  These are substituent-prefix PINs (OPSIN also accepts the
# conjunctive "ethaneazonic acid" form for the same structure).

@pytest.mark.parametrize("smi,expected", [
    # azonic: 1 organyl, oxido, 2 -OH
    ("CC[N+]([O-])(O)O", "ethylazonic acid"),
    ("[O-][N+](O)(O)c1ccccc1", "phenylazonic acid"),
    # azinic: 1 organyl + N-H, oxido, 1 -OH
    ("C[NH+]([O-])O", "methylazinic acid"),
    # azinic: 2 organyl, oxido, 1 -OH
    ("C[N+](C)([O-])O", "dimethylazinic acid"),
    # azonous (neutral N, -ous series): 1 organyl, 2 -OH
    ("CCN(O)O", "ethylazonous acid"),
])
def test_substituted_n_oxoacid_acids(smi: str, expected: str) -> None:
    assert _name_subst_n(smi) == expected


# ---------------------------------------------------------------------------
# Carbon-substituted nitrogen oxoacid ANIONS (P-72.2): azonate / azinate /
# azonite, plus the partial "hydrogen ...ate" mono-anion.
# ---------------------------------------------------------------------------
# Every expected name below was verified to round-trip through OPSIN 2.8.0
# (name -> SMILES -> same RDKit canonical) when the anion extension was built.

@pytest.mark.parametrize("smi,expected", [
    # --- azonate: fully-deprotonated dianion (oxido + 2 [O-], 1 organyl) ---
    ("CC[N+]([O-])([O-])[O-]", "ethylazonate"),
    ("C[N+]([O-])([O-])[O-]", "methylazonate"),
    ("C(CCCC)[N+]([O-])([O-])[O-]", "pentylazonate"),
    ("CC(C)(C)[N+]([O-])([O-])[O-]", "(2-methylpropan-2-yl)azonate"),
    ("[O-][N+]([O-])([O-])c1ccccc1", "phenylazonate"),
    ("[O-][N+]([O-])([O-])Cc1ccccc1", "(phenylmethyl)azonate"),
    ("[O-][N+]([O-])([O-])C1CCCCC1", "cyclohexylazonate"),
    ("C1(=CC=CC2=CC=CC=C12)[N+]([O-])([O-])[O-]", "(naphthalen-1-yl)azonate"),
    ("C=C[N+]([O-])([O-])[O-]", "(ethenyl)azonate"),
    ("[O-][N+]([O-])([O-])CCCl", "(2-chloroethyl)azonate"),
    # --- azonate partial mono-anion (oxido + 1 -OH + 1 [O-]) ---
    ("CC[N+]([O-])(O)[O-]", "hydrogen ethylazonate"),
    ("C[N+]([O-])(O)[O-]", "hydrogen methylazonate"),
    ("[O-][N+](O)([O-])c1ccccc1", "hydrogen phenylazonate"),
    # --- azinate: mono-anion (oxido + 1 [O-], 1 organyl + N-H) ---
    ("C[NH+]([O-])[O-]", "methylazinate"),
    ("CC[NH+]([O-])[O-]", "ethylazinate"),
    ("[O-][NH+]([O-])c1ccccc1", "phenylazinate"),
    ("[O-][NH+]([O-])C1CCCCC1", "cyclohexylazinate"),
    # --- azinate: mono-anion (oxido + 1 [O-], 2 organyl) ---
    ("C[N+]([O-])([O-])C", "dimethylazinate"),
    ("CC[N+]([O-])([O-])CC", "diethylazinate"),
    ("C[N+]([O-])([O-])c1ccccc1", "methylphenylazinate"),
    ("[O-][N+]([O-])(c1ccccc1)c1ccccc1", "diphenylazinate"),
    # --- azonite: -ous anion (neutral N, no oxido, 2 [O-], 1 organyl) ---
    ("CCN([O-])[O-]", "ethylazonite"),
    ("CN([O-])[O-]", "methylazonite"),
    ("[O-]N([O-])c1ccccc1", "phenylazonite"),
    ("[O-]N([O-])C1CCCCC1", "cyclohexylazonite"),
])
def test_substituted_n_oxoacid_anions(smi: str, expected: str) -> None:
    assert _name_subst_n(smi) == expected


@pytest.mark.parametrize("smi,expected", [
    # Engine integration — these route through the pre-guard dispatch hook.
    ("CC[N+]([O-])([O-])[O-]", "ethylazonate"),
    ("C[NH+]([O-])[O-]", "methylazinate"),
    ("CCN([O-])[O-]", "ethylazonite"),
    ("CC[N+]([O-])(O)[O-]", "hydrogen ethylazonate"),
])
def test_substituted_n_oxoacid_anion_engine(smi: str, expected: str) -> None:
    assert name_smiles(smi) == expected


@pytest.mark.parametrize("smi", [
    "CCO",                      # carbon-only, no N centre
    "C[N+](C)(C)C",            # quaternary ammonium, no acidic O / oxido
    "CN",                       # methylamine, no oxido / acidic O
    "O=N(=O)[O-]",             # genuine-oxo nitro skeleton (not single-bond O)
    "[CH3]",                   # carbon radical
])
def test_substituted_n_oxoacid_does_not_fire(smi: str) -> None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        pytest.skip(f"unparseable test SMILES {smi!r}")
    assert compute_substituted_n_oxoacid_name(mol) is None
