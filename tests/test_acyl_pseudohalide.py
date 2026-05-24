"""Tests for the acyl-pseudohalide functional-class dispatcher (P-65.3.1 / P-66).

Carboxylic-acyl pseudohalides ``R-C(=O)-X`` where X is one of isothiocyanate
(``-N=C=S``), isocyanate (``-N=C=O``), isocyanide (``-[N+]#[C-]``), cyanate
(``-O-C#N``) or cyanide (``-C#N``) are named ``{acyl} {class-word}`` exactly
like acid halides (``acetyl chloride``).  Diacyl forms sharing one R backbone
use the diacyl stem with a multiplied (identical) or alphabetically-ordered
(mixed) class word.

Before the dedicated dispatcher:

* mono isothiocyanate mis-rendered as ``acetic acid isothiocyanate`` because
  the ACYL output form falls back to the acid name for retained stems;
* diacyl forms double-counted the carbonyl carbon of one acyl group as a
  substituent on the other's chain
  (``oxalyl diisothiocyanate`` → ``2-(isothiocyanatocarbonyl)ethanoyl
  isothiocyanate``).

The string-only assertions run in any dev environment (no OPSIN needed).
"""

from __future__ import annotations

import pytest

from iupac_namer.engine import name_smiles


class TestMonoAcylPseudohalide:
    """``R-C(=O)-X`` → ``{acyl} {class-word}``."""

    @pytest.mark.parametrize(
        "smiles,expected",
        [
            ("CC(=O)N=C=S",              "acetyl isothiocyanate"),
            ("CC(=O)N=C=O",              "acetyl isocyanate"),
            ("CC(=O)[N+]#[C-]",          "acetyl isocyanide"),
            ("CC(=O)OC#N",               "acetyl cyanate"),
            ("CC(=O)C#N",                "acetyl cyanide"),
            ("O=C(N=C=S)c1ccccc1",       "benzoyl isothiocyanate"),
            ("O=C(N=C=O)c1ccccc1",       "benzoyl isocyanate"),
            ("CCC(=O)N=C=S",             "propanoyl isothiocyanate"),
            ("C1(CCCCC1)C(=O)N=C=S",     "cyclohexanecarbonyl isothiocyanate"),
        ],
    )
    def test_mono(self, smiles, expected):
        assert name_smiles(smiles) == expected


class TestDiacylPseudohalide:
    """``X-C(=O)-R-C(=O)-X'`` → ``{diacyl} di{word}`` / mixed alphabetical."""

    @pytest.mark.parametrize(
        "smiles,expected",
        [
            # Identical tails → multiplied class word; carbonyl carbons become
            # part of the diacid parent (no double-counting).
            ("O=C(N=C=S)C(=O)N=C=S",         "oxalyl diisothiocyanate"),
            ("O=C(N=C=O)C(=O)N=C=O",         "oxalyl diisocyanate"),
            ("O=C(OC#N)C(=O)OC#N",           "oxalyl dicyanate"),
            ("O=C([N+]#[C-])C(=O)[N+]#[C-]", "oxalyl diisocyanide"),
            ("O=C(C#N)C(=O)C#N",             "oxalyl dicyanide"),
            ("O=C(N=C=S)CC(=O)N=C=S",        "propanedioyl diisothiocyanate"),
            ("O=C(N=C=S)CCC(=O)N=C=S",       "butanedioyl diisothiocyanate"),
            # Mixed tails → alphabetical "word1 word2".
            ("O=C(N=C=S)CCC(=O)[N+]#[C-]",   "butanedioyl isocyanide isothiocyanate"),
            ("O=C(N=C=S)C(=O)N=C=O",         "oxalyl isocyanate isothiocyanate"),
        ],
    )
    def test_diacyl(self, smiles, expected):
        assert name_smiles(smiles) == expected


class TestDoesNotFireOnOtherClasses:
    """The dispatcher must defer to the substitutive pipeline when the acyl
    group is not a genuine carboxylic acyl (carbamoyl / carbamic), or when a
    senior characteristic group on the backbone is the principal group."""

    @pytest.mark.parametrize(
        "smiles,expected",
        [
            # Carbamoyl backbone (N-C(=O)-): amide is the principal group, the
            # pseudohalide is a prefix.  Must NOT become "carbamyl isocyanate".
            ("NC(=O)N=C=O",  "1-isocyanatomethanamide"),
            ("N#CC(N)=O",    "1-cyanomethanamide"),
            # Plain acid / nitrile / ester unaffected.
            ("CC(=O)O",      "acetic acid"),
            ("CC#N",         "acetonitrile"),
        ],
    )
    def test_defers(self, smiles, expected):
        assert name_smiles(smiles) == expected
