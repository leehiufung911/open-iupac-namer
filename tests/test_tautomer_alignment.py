"""Regression tests for Stage 7 tautomer / Kekulé-equivalence alignment.

Covers:

* :func:`iupac_namer.perception.tautomer_alignment.kekule_equivalent` and
  :func:`classify_round_trip` — pure helpers used by the audit tooling.
* The Stage 6 R1-A ``kekule_store`` rewrites for ``1H-indene`` and
  ``3H-perimidine``.  These were the load-bearing fixes that closed the
  12 (of 22) Stage 2 LOCANT_WRONG residuals listed under the
  HANDOFF.md "tautomer / indicated-H drift" class.  Each substituted
  probe is asserted to round-trip identically through OPSIN.

* The phenarsazine 3/4/6/7 substituent positions — these probes
  emit a chemically-correct name whose OPSIN round-trip differs only in
  Kekulé bond-order placement (``=N``/``=As`` cross-ring partners).  The
  test asserts the round-trip is ``KEKULE_EQUIVALENT`` (InChI matches),
  documenting the residual as not-a-naming-defect rather than masking
  it.
"""

from __future__ import annotations

import os
import tempfile

import pytest


# ---------------------------------------------------------------------------
# Pure-helper tests — no Java dependency.
# ---------------------------------------------------------------------------

def test_kekule_equivalent_aromatic_partners() -> None:
    from iupac_namer.perception.tautomer_alignment import kekule_equivalent
    # Two Kekulé partners of *substituted* phenarsazine differ in
    # cross-ring =N/=As bond placement but describe the same molecule.
    # (Unsubstituted phenarsazine is D2h-symmetric and the two partners
    # canonicalise to the same SMILES; using a 3-Me probe breaks the
    # symmetry so the canonical SMILES diverge while InChI matches.)
    a = "Cc1ccc2c(c1)N=c1ccccc1=[As]2"
    b = "Cc1ccc2c(c1)=Nc1ccccc1[As]=2"
    assert kekule_equivalent(a, b)


def test_kekule_equivalent_distinct_isomers() -> None:
    from iupac_namer.perception.tautomer_alignment import kekule_equivalent
    # 4-methylindene and 7-methylindene are NOT Kekulé-equivalent — the
    # methyl is on a different atom.
    a = "Cc1cccc2c1C=CC2"  # 4-methyl-1H-indene
    b = "Cc1cccc2c1CC=C2"  # 7-methyl-1H-indene
    assert not kekule_equivalent(a, b)


def test_classify_round_trip_ok() -> None:
    from iupac_namer.perception.tautomer_alignment import classify_round_trip
    a = "Cc1cccc2c1C=CC2"
    assert classify_round_trip(a, a) == "OK"


def test_classify_round_trip_kekule_equivalent() -> None:
    from iupac_namer.perception.tautomer_alignment import classify_round_trip
    # Substituted phenarsazine: a 3-methyl probe and the corresponding
    # ``=N``/``=As`` mirror Kekulé partner.  These canonicalise to
    # different RDKit SMILES but share the same standard InChI — the
    # exact pattern that lights up the Stage 2 phenarsazine residual.
    a = "Cc1ccc2c(c1)N=c1ccccc1=[As]2"
    b = "Cc1ccc2c(c1)=Nc1ccccc1[As]=2"
    assert classify_round_trip(a, b) == "KEKULE_EQUIVALENT"


def test_classify_round_trip_wrong() -> None:
    from iupac_namer.perception.tautomer_alignment import classify_round_trip
    a = "Cc1cccc2c1C=CC2"
    b = "Cc1cccc2c1CC=C2"  # different atom — not Kekulé equivalent
    assert classify_round_trip(a, b) == "WRONG"


def test_classify_round_trip_invalid() -> None:
    from iupac_namer.perception.tautomer_alignment import classify_round_trip
    assert classify_round_trip("CCO", "") == "INVALID"
    assert classify_round_trip("", "CCO") == "INVALID"
    assert classify_round_trip("not a smiles", "CCO") == "INVALID"


def test_kekule_equivalent_charge_drop_default_strict() -> None:
    from iupac_namer.perception.tautomer_alignment import kekule_equivalent
    # Charge drops are real defects (flavylium → chromenyl-benzene).
    # Default behaviour: charge drop ⇒ NOT equivalent.
    cation = "c1ccc2[o+]cccc2c1"  # chromenylium
    neutral = "c1ccc2occc(=O)c12"  # chromen-4(4H)-one (different skeleton)
    assert not kekule_equivalent(cation, neutral)


# ---------------------------------------------------------------------------
# Stage 6 R1-A kekule_store regression tests — require Java/OPSIN.
# ---------------------------------------------------------------------------

def _java_available() -> bool:
    java_home = os.environ.get("JAVA_HOME")
    return bool(java_home) and os.path.exists(java_home)


_JAVA_REQUIRED = pytest.mark.skipif(
    not _java_available(),
    reason="JAVA_HOME not set or invalid; skipping OPSIN round-trip tests",
)


def _round_trip_canonical(name: str) -> str | None:
    """Name → SMILES → RDKit canonical SMILES."""
    # py2opsin uses CWD-local temp files; cd into a fresh dir to avoid
    # the Windows ``[WinError 32] py2opsin_temp_input.txt`` race that
    # surfaces under concurrent invocation (cf. tests/conftest.py).
    cwd = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="kekule_test_")
    try:
        os.chdir(tmp)
        from py2opsin import py2opsin
        from rdkit import Chem
        smi = py2opsin(name, output_format="SMILES")
    finally:
        os.chdir(cwd)
    if not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


# (engine_input_smiles, expected_engine_name)
INDENE_PROBES = [
    ("CC1=CCc2ccccc21",           "3-methyl-1H-indene"),
    ("Cc1cccc2c1C=CC2",           "4-methyl-1H-indene"),
    ("Cc1ccc2c(c1)C=CC2",         "5-methyl-1H-indene"),
    ("Cc1ccc2c(c1)CC=C2",         "6-methyl-1H-indene"),
    ("Cc1cccc2c1CC=C2",           "7-methyl-1H-indene"),
    ("CC1C=Cc2ccccc21",           "1-methyl-1H-indene"),
]

PERIMIDINE_PROBES = [
    ("Cc1ccc2cccc3c2c1N=CN3",     "9-methyl-3H-perimidine"),
    ("Cc1cc2c3c(cccc3c1)NC=N2",   "8-methyl-3H-perimidine"),
    ("Cc1ccc2c3c(cccc13)NC=N2",   "7-methyl-3H-perimidine"),
    ("Cc1ccc2c3c(cccc13)N=CN2",   "6-methyl-3H-perimidine"),
    ("Cc1cc2c3c(cccc3c1)N=CN2",   "5-methyl-3H-perimidine"),
    ("Cc1ccc2cccc3c2c1NC=N3",     "4-methyl-3H-perimidine"),
]


@_JAVA_REQUIRED
@pytest.mark.parametrize("input_smi,expected_name", INDENE_PROBES)
def test_indene_kekule_store_round_trips(input_smi: str, expected_name: str) -> None:
    """Engine names round-trip to canonically-identical SMILES.

    The Stage 6 R1-A ``kekule_store`` rewrites the default ``ind-1-ene``
    name to ``1H-indene`` for canonical SMILES ``C1=Cc2ccccc2C1``,
    aligning the locant convention with our atom_locants table.  Each
    of the 6 substituent positions on indene was a Stage 2 LOCANT_WRONG
    row; with the rewrite, all 6 round-trip cleanly.
    """
    from iupac_namer.engine import name_smiles
    from rdkit import Chem
    name = name_smiles(input_smi)
    assert name == expected_name, f"engine emitted {name!r}, expected {expected_name!r}"
    rt_canon = _round_trip_canonical(name)
    in_canon = Chem.MolToSmiles(Chem.MolFromSmiles(input_smi))
    assert rt_canon == in_canon, (
        f"round-trip mismatch for {expected_name!r}: "
        f"input {in_canon!r} vs round-trip {rt_canon!r}"
    )


@_JAVA_REQUIRED
@pytest.mark.parametrize("input_smi,expected_name", PERIMIDINE_PROBES)
def test_perimidine_kekule_store_round_trips(input_smi: str, expected_name: str) -> None:
    """Stage 6 R1-A perimidine rewrite: ``perimidine`` → ``3H-perimidine``.

    OPSIN's default ``perimidine`` parses to the 1H tautomer (NH at the
    locant where our atom_locants pin the no-H N); the rewrite pins the
    name to the 3H form so OPSIN materialises the matching Kekulé
    partner.  6 substituted probes round-trip cleanly post-rewrite.
    """
    from iupac_namer.engine import name_smiles
    from rdkit import Chem
    name = name_smiles(input_smi)
    assert name == expected_name, f"engine emitted {name!r}, expected {expected_name!r}"
    rt_canon = _round_trip_canonical(name)
    in_canon = Chem.MolToSmiles(Chem.MolFromSmiles(input_smi))
    assert rt_canon == in_canon, (
        f"round-trip mismatch for {expected_name!r}: "
        f"input {in_canon!r} vs round-trip {rt_canon!r}"
    )


# ---------------------------------------------------------------------------
# Phenarsazine documented residual: KEKULE_EQUIVALENT, not WRONG.
# ---------------------------------------------------------------------------

PHENARSAZINE_KEKULE_EQUIV_PROBES = [
    ("Cc1ccc2c(c1)N=c1ccccc1=[As]2",  "3-methylphenarsazin"),
    ("Cc1cccc2c1N=c1ccccc1=[As]2",    "4-methylphenarsazin"),
    ("Cc1cccc2c1=Nc1ccccc1[As]=2",    "6-methylphenarsazin"),
    ("Cc1ccc2c(c1)=Nc1ccccc1[As]=2",  "7-methylphenarsazin"),
]


@_JAVA_REQUIRED
@pytest.mark.parametrize("input_smi,expected_name", PHENARSAZINE_KEKULE_EQUIV_PROBES)
def test_phenarsazine_round_trip_is_kekule_equivalent(
    input_smi: str, expected_name: str
) -> None:
    """Phenarsazine 3/4/6/7-Me round-trips to a Kekulé-equivalent partner.

    The engine emits the chemically-correct locant (verified against
    OPSIN's own canonical-form-of-name SMILES at the InChI level), but
    OPSIN's reverse-parse picks the *opposite* Kekulé partner of the
    central N/As cross-ring double bonds.  RDKit canonical SMILES does
    not aromatise these explicit cross-ring ``=N``/``=As`` bonds, so the
    canonical strings differ.  Standard InChI normalises Kekulé partners
    and confirms the molecules are identical.

    This test pins the diagnosis: the residual is a SMILES-canonicalisation
    artefact, not a naming defect.  Any attempt to "fix" it by emitting
    a different name would name the wrong molecule.
    """
    from iupac_namer.engine import name_smiles
    from iupac_namer.perception.tautomer_alignment import classify_round_trip
    name = name_smiles(input_smi)
    assert name == expected_name, f"engine emitted {name!r}, expected {expected_name!r}"
    cwd = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="phenarsazine_test_")
    try:
        os.chdir(tmp)
        from py2opsin import py2opsin
        rt_smi = py2opsin(name, output_format="SMILES")
    finally:
        os.chdir(cwd)
    assert rt_smi, f"py2opsin returned empty SMILES for {name!r}"
    bucket = classify_round_trip(input_smi, rt_smi)
    # KEKULE_EQUIVALENT documents the diagnosis; OK is also acceptable
    # if a future fix aligns Kekulé partners.  WRONG would mean the
    # name names a different molecule and should fail loudly.
    assert bucket in ("OK", "KEKULE_EQUIVALENT"), (
        f"phenarsazine residual classification regressed to {bucket!r}: "
        f"input {input_smi!r} round-trips via {name!r} to {rt_smi!r}"
    )
