"""
Tests for Stage 6 R2-G chalcogen anion FGs + Se/Te analogs of
sulfonate/sulfonamide/sulfoxide + aromatic selone/tellone.

Targets Gap 6 and Gap 7 of `docs/opsin_audit_fg.md` plus the Se/Te
selenoxide/tellurone cluster called out by natural-products audit b.2.

The acceptance criterion for each probe is a SMILES → IUPAC round-trip
via OPSIN that canonicalises back to the same InChI layer as the input.
For unit-level ease we assert the *name text* the engine emits; the
authoritative eval harness carries the round-trip check.
"""
from __future__ import annotations

import pytest
from iupac_namer.engine import name_smiles as name


# ---------------------------------------------------------------------------
# Gap 6 — tellone on acyclic and saturated-cyclic parents
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "smiles, expected_suffix",
    [
        # ethane-1-tellone  — engine elides locant 1 on terminal ethane → "ethanetellone"
        ("CC=[Te]",           "tellone"),
        # pentane-1-tellone
        ("CCCCC=[Te]",        "tellone"),
        # propan-2-tellone — internal H0 carbon in acyclic chain
        ("CC(C)=[Te]",        "tellone"),
        # cyclohexane-1-tellone (terminal C=Te on a ring)
        ("[Te]=C1CCCCC1",     "tellone"),
    ],
)
def test_gap6_tellone(smiles: str, expected_suffix: str) -> None:
    out = name(smiles)
    assert expected_suffix in out, f"{smiles} → {out}  (want substring {expected_suffix!r})"


# ---------------------------------------------------------------------------
# Gap 7 — chalcogen anion suffix FGs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "smiles, expected_suffix",
    [
        # ethane-1-sulfenate  (CS-O-)
        ("CCS[O-]",                  "sulfenate"),
        # pentane-1-sulfenate
        ("CCCCCS[O-]",               "sulfenate"),
        # cyclohexane-1-sulfenate (ring + S-O- branch)
        ("[O-]SC1CCCCC1",            "sulfenate"),
        # ethane-1-sulfenothioate  (CS-S-)
        ("CCS[S-]",                  "sulfenothioate"),
        # ethane-1-sulfenoselenoate (CS-Se-)
        ("CCS[Se-]",                 "sulfenoselenoate"),
        # ethane-1-tellurosulfenate (CS-Te-) — OPSIN accepts `tellurosulfenate`
        # as the chain-anion name; `sulfenotelluroate` is NOT accepted.
        ("CCS[Te-]",                 "tellurosulfenate"),
        # ethane-1-tellurolate  (C-Te-)
        ("CC[Te-]",                  "tellurolate"),
    ],
)
def test_gap7_chalcogen_anions(smiles: str, expected_suffix: str) -> None:
    out = name(smiles)
    assert expected_suffix in out, f"{smiles} → {out}  (want substring {expected_suffix!r})"


# ---------------------------------------------------------------------------
# Natural b.2 — Se/Te analogs of sulfonate/sulfonamide/acid families
# (selenoxide/telluroxide/selenone/tellurone FUNCTIONAL-CLASS forms are named
#  via the `methylseleninyl` / `methyltellurinyl` / `methylselenonyl` /
#  `methyltelluronyl` substituent prefixes; that path lives in the engine's
#  sulfonyl/sulfinyl special-case and is out of this agent's scope.  What we
#  DO cover here: the suffix/anion forms reachable through SMARTS matching.)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "smiles, expected_substring",
    [
        # ethane-1-selenonamide (R-Se(=O)(=O)-NH2)
        ("CC[Se](=O)(=O)N",        "selenonamide"),
        # ethane-1-selenonate (anion form)
        ("CC[Se](=O)(=O)[O-]",     "selenonate"),
        # ethane-1-telluronate (anion form)
        ("CC[Te](=O)(=O)[O-]",     "telluronate"),
        # ethane-1-seleninate (anion form)
        ("CC[Se](=O)[O-]",         "seleninate"),
        # ethane-1-tellurinate (anion form)
        ("CC[Te](=O)[O-]",         "tellurinate"),
        # ethanetelluronic acid
        ("CC[Te](=O)(=O)O",        "telluronic"),
        # ethanetellurinic acid
        ("CC[Te](=O)O",            "tellurinic"),
    ],
)
def test_gap6_se_te_acid_analogs(smiles: str, expected_substring: str) -> None:
    out = name(smiles)
    assert expected_substring in out, f"{smiles} → {out}  (want substring {expected_substring!r})"


# ---------------------------------------------------------------------------
# Regression: previously-working selone names must still render
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "smiles, expected_substring",
    [
        # ethane-1-selone / propan-2-selone / cyclohexan-1-selone all COVERED
        # before this change via selenoxo prefix; keep one form as regression.
        ("CC(C)=[Se]",          "seleno"),
        ("[Se]=C1CCCCC1",       "seleno"),
    ],
)
def test_selone_regression(smiles: str, expected_substring: str) -> None:
    out = name(smiles)
    # The engine may emit either the new "-selone" suffix or the legacy
    # "selenoxo" prefix; both forms round-trip via OPSIN.  Accept either.
    assert ("selone" in out) or ("selenoxo" in out) or (expected_substring in out), (
        f"{smiles} → {out}"
    )
