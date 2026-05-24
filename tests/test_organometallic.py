"""Regression tests for Stage 6 R3-A metallocene perception.

Pins the behaviour of ``iupac_namer.perception.organometallic`` and
the engine dispatch hook in ``name_smiles``.  Each test runs the full
SMILES -> name -> OPSIN -> canonical SMILES round-trip; the OPSIN
agreement is the load-bearing assertion (matches the Stage 4
authoritative-eval gate).
"""

from __future__ import annotations

import pytest
from rdkit import Chem

from iupac_namer.engine import name_smiles
from iupac_namer.perception.organometallic import (
    MetalloceneClassification,
    classify_metallocene,
    detect,
    _METALLOCENE_PINS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canon(smiles: str | None) -> str | None:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def _opsin_canon(name: str) -> str | None:
    """OPSIN(name) -> canonical SMILES, or ``None`` if unparseable."""
    from py2opsin import py2opsin

    smi = py2opsin(name)
    return _canon(smi) if smi else None


# ---------------------------------------------------------------------------
# Classifier sanity
# ---------------------------------------------------------------------------


def test_classify_ferrocene_canonical_form() -> None:
    cls = classify_metallocene(
        Chem.MolFromSmiles("[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1")
    )
    assert cls is not None
    assert cls.retained_name == "ferrocene"
    # Two ring atom sets, exactly five atoms each.
    assert len(cls.ring_atom_sets) == 2
    assert all(len(s) == 5 for s in cls.ring_atom_sets)


def test_classify_ferrocene_alternate_input_form() -> None:
    """Both Kekule and aromatic SMILES inputs canonicalise the same."""
    cls = classify_metallocene(
        Chem.MolFromSmiles("[CH-]1C=CC=C1.[CH-]1C=CC=C1.[Fe+2]")
    )
    assert cls is not None
    assert cls.retained_name == "ferrocene"


def test_classify_returns_none_for_non_sandwich() -> None:
    # Plain ethanol.
    assert classify_metallocene(Chem.MolFromSmiles("CCO")) is None
    # An iron salt with an aromatic ring but no Cp- pair.
    assert (
        classify_metallocene(
            Chem.MolFromSmiles("[Fe+3].c1ccc(O)cc1")
        )
        is None
    )


def test_classify_returns_none_for_substituted_metallocene() -> None:
    """Substituted (1,1'-dimethylferrocene-style) sandwiches are out of
    scope for R3-A and must fall through unchanged."""
    cls = classify_metallocene(
        Chem.MolFromSmiles("[Fe+2].Cc1cc[cH-]c1.c1cc[cH-]c1")
    )
    assert cls is None


def test_classify_returns_none_for_non_mol() -> None:
    assert classify_metallocene(None) is None


def test_detect_emits_leaf_with_retained_name() -> None:
    leaf = detect(Chem.MolFromSmiles("[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1"))
    assert leaf is not None
    assert leaf.text == "ferrocene"


def test_detect_returns_none_for_unmatched() -> None:
    assert detect(Chem.MolFromSmiles("CCO")) is None


# ---------------------------------------------------------------------------
# Engine integration: SMILES -> name -> OPSIN canonical round-trip
# ---------------------------------------------------------------------------


METALLOCENE_PROBES: list[tuple[str, str]] = [
    ("[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1", "ferrocene"),
    ("[Ru+2].c1cc[cH-]c1.c1cc[cH-]c1", "ruthenocene"),
    ("[Os+2].c1cc[cH-]c1.c1cc[cH-]c1", "osmocene"),
    ("[Co+2].c1cc[cH-]c1.c1cc[cH-]c1", "cobaltocene"),
    ("[Ni+2].c1cc[cH-]c1.c1cc[cH-]c1", "nickelocene"),
    ("[Rh+2].c1cc[cH-]c1.c1cc[cH-]c1", "rhodocene"),
    ("[Cr+2].c1cc[cH-]c1.c1cc[cH-]c1", "chromocene"),
    ("[V+2].c1cc[cH-]c1.c1cc[cH-]c1", "vanadocene"),
    ("[Ti+2].c1cc[cH-]c1.c1cc[cH-]c1", "titanocene"),
    ("[Mo+2].c1cc[cH-]c1.c1cc[cH-]c1", "molybdocene"),
    ("[Pb+2].c1cc[cH-]c1.c1cc[cH-]c1", "plumbocene"),
    ("[Zr+2].c1cc[cH-]c1.c1cc[cH-]c1", "zirconocene"),
    ("[Nb+2].c1cc[cH-]c1.c1cc[cH-]c1", "niobocene"),
]


@pytest.mark.parametrize("smiles,expected_name", METALLOCENE_PROBES)
def test_engine_emits_retained_metallocene_name(
    smiles: str, expected_name: str
) -> None:
    assert name_smiles(smiles) == expected_name


def test_all_metallocenes_round_trip_through_opsin() -> None:
    """OPSIN(our_name) canonicalises to the same SMILES as the input.

    Run as a single test (rather than parametrised) so the OPSIN
    invocation is batched into one Java call — py2opsin's temp-file
    handling has a race when many tests invoke it in close succession
    on Windows, and a single batched call sidesteps the issue.
    """
    from py2opsin import py2opsin

    expected_names = []
    canonical_inputs = []
    for smi, expected in METALLOCENE_PROBES:
        emitted = name_smiles(smi)
        assert emitted == expected
        expected_names.append(emitted)
        canonical_inputs.append(_canon(smi))

    opsin_smiles = py2opsin(expected_names, output_format="SMILES")
    assert len(opsin_smiles) == len(expected_names)
    for (smi, expected), out_smi in zip(METALLOCENE_PROBES, opsin_smiles):
        assert out_smi, f"OPSIN unparseable for {expected!r}"
        assert _canon(out_smi) == _canon(smi), (
            f"Round-trip mismatch for {expected!r}: "
            f"input={_canon(smi)!r} OPSIN={_canon(out_smi)!r}"
        )


def test_pinned_table_internal_consistency() -> None:
    """Every pin's key canonicalises to itself (i.e., the table key IS the
    canonical SMILES; protects against accidental hand-mutated entries)."""
    for canonical_key, name in _METALLOCENE_PINS.items():
        m = Chem.MolFromSmiles(canonical_key)
        assert m is not None, f"unparseable pin key: {canonical_key!r}"
        recanon = Chem.MolToSmiles(m)
        assert recanon == canonical_key, (
            f"pin key not canonical: {canonical_key!r} -> {recanon!r}"
        )
        assert isinstance(name, str) and name


def test_substituted_metallocene_now_in_scope_after_R22E() -> None:
    """Stage 22 R22-E: substituted metallocenes (mono-atomic substituents,
    at most one per Cp ring) ARE now in scope.  The engine emits the
    retained ``-ocene`` parent rather than falling through to the
    salt path's ``iron(2+) methylcyclopentadienide cyclopentadienide``.

    This test was inverted in R22-E to reflect the new closed scope.
    The strict pin classifier ``classify_metallocene`` still returns
    ``None`` for substituted forms (per
    ``test_classify_returns_none_for_substituted_metallocene``); the
    new behaviour is that the engine ``detect`` then routes via the
    R22-E path before deferring.
    """
    name_out = name_smiles("[Fe+2].Cc1cc[cH-]c1.c1cc[cH-]c1")
    assert name_out == "methylferrocene"


# ---------------------------------------------------------------------------
# Metal carbonyl complexes (coordination nomenclature, P-68.3)
# ---------------------------------------------------------------------------
#
# Disconnected SMILES like ``[C]=O.[C]=O.[C]=O.[C]=O.[C]=O.[Fe]`` are
# coordination compound representations.  The engine must emit the IUPAC
# coordination name ``{N}carbonyl{metal}`` rather than the generic salt
# name ``pentacarbon monoxide iron``.
#
# Round-trip through OPSIN: OPSIN parses the concatenated form
# (``pentacarbonyliron``) but returns a *connected* organometallic
# canonical SMILES (``O=[C]=[Fe](=[C]=O)...``), which is structurally
# different from the disconnected input.  We therefore pin only the
# emitted name, not the OPSIN round-trip.


CARBONYL_PROBES: list[tuple[str, str]] = [
    # Pure homoleptic carbonyls
    ("[C]=O.[C]=O.[C]=O.[C]=O.[C]=O.[Fe]",         "pentacarbonyliron"),
    ("[C]=O.[C]=O.[C]=O.[C]=O.[Fe]",                "tetracarbonyliron"),
    ("[C]=O.[C]=O.[C]=O.[C]=O.[Ni]",                "tetracarbonylnickel"),
    ("[C]=O.[C]=O.[C]=O.[C]=O.[C]=O.[C]=O.[Cr]",   "hexacarbonylchromium"),
    ("[C]=O.[C]=O.[C]=O.[C]=O.[C]=O.[C]=O.[Mo]",   "hexacarbonylmolybdenum"),
    ("[C]=O.[C]=O.[C]=O.[C]=O.[C]=O.[C]=O.[W]",    "hexacarbonyltungsten"),
    # With halide counterions
    ("[Br-].[Br-].[C]=O.[C]=O.[C]=O.[C]=O.[C]=O.[Mn+2]",
     "pentacarbonylmanganese dibromide"),
]


@pytest.mark.parametrize("smiles,expected_name", CARBONYL_PROBES)
def test_engine_emits_carbonyl_name(smiles: str, expected_name: str) -> None:
    """The engine must produce the coordination name for metal carbonyls."""
    assert name_smiles(smiles) == expected_name


def test_carbonyl_does_not_fire_for_non_co_fragments() -> None:
    """Molecules with non-CO, non-halide fragments fall through to salt path."""
    # Ring ligand present — not a pure carbonyl complex.
    name = name_smiles("[C]=O.[C]=O.[C]=O.[Cr].c1ccsc1")
    assert "carbonyl" not in name or "tricarbon" in name  # falls to salt path


def test_carbonyl_does_not_fire_for_charge_mismatch() -> None:
    """Metal charge must match halide count; mismatches fall through."""
    # [Fe] is charge 0 but [Br-] would need +1 metal — falls to salt path.
    name = name_smiles("[Br-].[C]=O.[C]=O.[Fe]")
    assert "carbonyliron" not in name  # charge mismatch: 0 ≠ 1


# ---------------------------------------------------------------------------
# Phase 11 — d-block coordination complex (cisplatin-shape) tests.
# ---------------------------------------------------------------------------


DBLOCK_COORD_PROBES: list[tuple[str, str]] = [
    # Cisplatin-shape Pt(IV) with 2 amino + 2 chloro.
    ("[NH2][Pt]([NH2])([Cl])[Cl]", "diaminoplatinum(IV) dichloride"),
    # Pd analogue, dichloride.
    ("[NH2][Pd]([NH2])([Cl])[Cl]", "diaminopalladium(IV) dichloride"),
    # Pd dibromide.
    ("[NH2][Pd]([NH2])([Br])[Br]", "diaminopalladium(IV) dibromide"),
    # Pt difluoride.
    ("[NH2][Pt]([NH2])([F])[F]",   "diaminoplatinum(IV) difluoride"),
    # Pt diiodide.
    ("[NH2][Pt]([NH2])([I])[I]",   "diaminoplatinum(IV) diiodide"),
    # 2-coordinate Pt(II): 1 amino + 1 chloro.
    ("[NH2][Pt][Cl]",              "aminoplatinum(II) chloride"),
]


@pytest.mark.parametrize("smiles,expected_name", DBLOCK_COORD_PROBES)
def test_engine_emits_dblock_coordination_name(
    smiles: str, expected_name: str
) -> None:
    """The engine must produce the coordination PIN for cisplatin-shape complexes."""
    assert name_smiles(smiles) == expected_name


@pytest.mark.parametrize("smiles,expected_name", DBLOCK_COORD_PROBES)
def test_dblock_coordination_round_trips_through_opsin(
    smiles: str, expected_name: str
) -> None:
    """Each emitted PIN must round-trip through OPSIN to the input SMILES."""
    assert _opsin_canon(expected_name) == _canon(smiles)


def test_dblock_coord_defers_for_charged_metal() -> None:
    """Charged metal centre is out of scope for this dispatcher."""
    # [Pt+] is a charged centre — defer to the regular pipeline.
    name = name_smiles("[NH2][Pt+]([NH2])([Cl])[Cl]")
    assert "platinum" not in name or "NAMING ERROR" in name


def test_dblock_coord_defers_for_mixed_halides() -> None:
    """Mixed Cl/Br halides on the same metal are out of scope."""
    name = name_smiles("[NH2][Pt]([NH2])([Cl])[Br]")
    assert "diamino" not in name  # dispatcher returned None, fell through


def test_dblock_coord_defers_for_non_supported_metal() -> None:
    """Cisplatin-shape Co coordination is out of scope (Co not in pinned set)."""
    name = name_smiles("[NH2][Co]([NH2])([Cl])[Cl]")
    assert "diamino" not in name  # falls through


# ---------------------------------------------------------------------------
# Phase 11 — sodium-nitroprusside-class pentacyano(nitroso) coordination.
# ---------------------------------------------------------------------------


NITROPRUSSIDE_PROBES: list[tuple[str, str]] = [
    # Bare [Fe(CN)5(NO)]^2- anion → "pentacyano(nitroso)iron(IV)".
    (
        "N#[C][Fe-2]([C]#N)([C]#N)([C]#N)([C]#N)[N]=O",
        "pentacyano(nitroso)iron(IV)",
    ),
    # Disodium salt → "disodium pentacyano(nitroso)iron(IV)".
    (
        "[Na+].[Na+].N#[C][Fe-2]([C]#N)([C]#N)([C]#N)([C]#N)[N]=O",
        "disodium pentacyano(nitroso)iron(IV)",
    ),
    # Dipotassium salt → "dipotassium pentacyano(nitroso)iron(IV)".
    (
        "[K+].[K+].N#[C][Fe-2]([C]#N)([C]#N)([C]#N)([C]#N)[N]=O",
        "dipotassium pentacyano(nitroso)iron(IV)",
    ),
]


@pytest.mark.parametrize("smiles,expected_name", NITROPRUSSIDE_PROBES)
def test_engine_emits_nitroprusside_name(
    smiles: str, expected_name: str
) -> None:
    """The engine must produce the pentacyano(nitroso)iron(IV) coordination PIN."""
    assert name_smiles(smiles) == expected_name


@pytest.mark.parametrize("smiles,expected_name", NITROPRUSSIDE_PROBES)
def test_nitroprusside_round_trips_through_opsin(
    smiles: str, expected_name: str
) -> None:
    """Each emitted PIN must round-trip through OPSIN to the input SMILES."""
    assert _opsin_canon(expected_name) == _canon(smiles)


def test_nitroprusside_defers_for_wrong_metal_charge() -> None:
    """Anion charge other than -2 on Fe is out of scope (returns None)."""
    # [Fe-3] would imply oxidation state III/different charge balance —
    # not the nitroprusside shape.
    name = name_smiles("N#[C][Fe-3]([C]#N)([C]#N)([C]#N)([C]#N)[N]=O")
    assert "pentacyano(nitroso)iron" not in name


def test_nitroprusside_defers_for_unsupported_metal() -> None:
    """[Mn-2] in the same coordination shape is out of scope (Mn not pinned)."""
    name = name_smiles("N#[C][Mn-2]([C]#N)([C]#N)([C]#N)([C]#N)[N]=O")
    assert "pentacyano(nitroso)" not in name


def test_nitroprusside_defers_for_charge_imbalanced_salt() -> None:
    """A single Na+ alongside the [Fe-2] anion is charge-imbalanced (defer)."""
    name = name_smiles("[Na+].N#[C][Fe-2]([C]#N)([C]#N)([C]#N)([C]#N)[N]=O")
    assert "sodium pentacyano(nitroso)iron" not in name


# ---------------------------------------------------------------------------
# Exotic binary metal salts (IR-5 / P-65.3): dichalcogenide / tritide /
# fulminate anions and the multiplier-free nitride compositional convention.
# ---------------------------------------------------------------------------

# (input SMILES, expected exact surface name).  Each round-trips through OPSIN
# to the same canonical SMILES (verified in the batched test below).
EXOTIC_SALT_PROBES = [
    # Dichalcogenide(2-) dianions (peroxide class, X-X bonded) — OPSIN's
    # compositional ``dioxide`` / ``disulfide`` / ... form.
    ("[Na+].[Na+].[O-][O-]", "disodium dioxide"),
    ("[K+].[K+].[O-][O-]", "dipotassium dioxide"),
    ("[Ca+2].[O-][O-]", "calcium dioxide"),
    ("[Ba+2].[O-][O-]", "barium dioxide"),
    ("[Na+].[Na+].[S-][S-]", "disodium disulfide"),
    ("[Ba+2].[S-][S-]", "barium disulfide"),
    ("[Ca+2].[S-][S-]", "calcium disulfide"),
    ("[Na+].[Na+].[Se-][Se-]", "disodium diselenide"),
    ("[Ba+2].[Se-][Se-]", "barium diselenide"),
    ("[Na+].[Na+].[Te-][Te-]", "disodium ditelluride"),
    ("[Ba+2].[Te-][Te-]", "barium ditelluride"),
    # Tritium hydride anion (tritide), incl. the di- collapse.
    ("[3H-].[Na+]", "sodium tritide"),
    ("[3H-].[K+]", "potassium tritide"),
    ("[3H-].[3H-].[Ca+2]", "calcium ditritide"),
    # Fulminate-family pseudohalide anions.
    ("[O-][N+]#[C-].[Na+]", "sodium fulminate"),
    ("[O-][N+]#[C-].[K+]", "potassium fulminate"),
    ("[S-][N+]#[C-].[Na+]", "sodium thiofulminate"),
    ("[Se-][N+]#[C-].[Na+]", "sodium selenofulminate"),
    ("[Te-][N+]#[C-].[Na+]", "sodium tellurofulminate"),
    # Multiplier-free nitride compositional convention.
    ("[Mg+2].[Mg+2].[Mg+2].[N-3].[N-3]", "magnesium nitride"),
    ("[Li+].[Li+].[Li+].[N-3]", "lithium nitride"),
    ("[N-3].[Na+].[Na+].[Na+]", "sodium nitride"),
    ("[K+].[K+].[K+].[N-3]", "potassium nitride"),
    ("[Ca+2].[Ca+2].[Ca+2].[N-3].[N-3]", "calcium nitride"),
    ("[N-3].[N-3].[Sr+2].[Sr+2].[Sr+2]", "strontium nitride"),
]


@pytest.mark.parametrize("smiles,expected_name", EXOTIC_SALT_PROBES)
def test_engine_emits_exotic_salt_name(smiles: str, expected_name: str) -> None:
    assert name_smiles(smiles) == expected_name


def test_exotic_salts_round_trip_through_opsin() -> None:
    """OPSIN(our_name) canonicalises back to the input (single batched call)."""
    from py2opsin import py2opsin

    expected_names = []
    for smi, expected in EXOTIC_SALT_PROBES:
        emitted = name_smiles(smi)
        assert emitted == expected, f"{smi}: {emitted!r} != {expected!r}"
        expected_names.append(emitted)

    opsin_smiles = py2opsin(expected_names, output_format="SMILES")
    assert len(opsin_smiles) == len(expected_names)
    for (smi, expected), out_smi in zip(EXOTIC_SALT_PROBES, opsin_smiles):
        assert out_smi, f"OPSIN unparseable for {expected!r}"
        assert _canon(out_smi) == _canon(smi), (
            f"Round-trip mismatch for {expected!r}: "
            f"input={_canon(smi)!r} OPSIN={_canon(out_smi)!r}"
        )


def test_nitride_defers_for_out_of_scope_metal() -> None:
    """Barium / transition-metal nitrides have no compositional PIN: the
    multiplier-free ``{metal} nitride`` form is NOT emitted (no wrong name)."""
    # Ba3N2 — OPSIN cannot parse "barium nitride", so we must not emit it.
    assert name_smiles("[Ba+2].[Ba+2].[Ba+2].[N-3].[N-3]") != "barium nitride"


def test_dichalcogenide_distinct_from_monatomic_oxide() -> None:
    """The X-X-bonded dianion ([O-][O-] -> dioxide) is distinct from a lone
    monatomic oxide ([O-2] -> oxide): the O-O bond presence disambiguates."""
    assert name_smiles("[Na+].[Na+].[O-][O-]") == "disodium dioxide"
    assert name_smiles("[O-2].[Na+].[Na+]") == "disodium oxide"


# ---------------------------------------------------------------------------
# Simple transition / coinage-metal organyls (IR-5 / P-69).
# ---------------------------------------------------------------------------

# (input SMILES, expected exact surface name).
METAL_ORGANYL_PROBES = [
    # Substitutive neutral mono-organyl.
    ("[CH3][Cu]", "methylcopper"),
    ("[Cu]c1ccccc1", "phenylcopper"),
    ("[Cu]C#CC", "(prop-1-yn-1-yl)copper"),
    ("C#C[Cu]", "(ethynyl)copper"),
    ("[CH3][Ag]", "methylsilver"),
    ("[CH3][Au]", "methylgold"),
    ("[CH3][Mn]", "methylmanganese"),
    ("[Fe]c1ccccc1", "phenyliron"),
    ("[CH3][Ti]", "methyltitanium"),
    # Substitutive cation (charge marker retained).
    ("[Hg+]C", "methylmercury(1+)"),
    ("[Hg+]c1ccccc1", "phenylmercury(1+)"),
    ("[Cu+]C", "methylcopper(1+)"),
    ("[Cu+2]C", "methylcopper(2+)"),
    # Substitutive multi-organyl.
    ("[CH3][Cu][CH3]", "dimethylcopper"),
    ("[CH3][Fe]([CH3])[CH3]", "trimethyliron"),
    # Additive hydrido coordination form.
    ("[ReH2]c1ccc2ccccc2c1", "dihydrido(naphthalen-2-yl)rhenium"),
    ("[ReH3]c1ccccc1", "trihydrido(phenyl)rhenium"),
    ("[CuH]c1ccccc1", "hydrido(phenyl)copper"),
    ("[CH3][ReH2]", "dihydrido(methyl)rhenium"),
]


@pytest.mark.parametrize("smiles,expected_name", METAL_ORGANYL_PROBES)
def test_engine_emits_metal_organyl_name(smiles: str, expected_name: str) -> None:
    assert name_smiles(smiles) == expected_name


def test_metal_organyls_round_trip_through_opsin() -> None:
    """OPSIN(our_name) canonicalises back to the input (single batched call)."""
    from py2opsin import py2opsin

    expected_names = []
    for smi, expected in METAL_ORGANYL_PROBES:
        emitted = name_smiles(smi)
        assert emitted == expected, f"{smi}: {emitted!r} != {expected!r}"
        expected_names.append(emitted)

    opsin_smiles = py2opsin(expected_names, output_format="SMILES")
    assert len(opsin_smiles) == len(expected_names)
    for (smi, expected), out_smi in zip(METAL_ORGANYL_PROBES, opsin_smiles):
        assert out_smi, f"OPSIN unparseable for {expected!r}"
        assert _canon(out_smi) == _canon(smi), (
            f"Round-trip mismatch for {expected!r}: "
            f"input={_canon(smi)!r} OPSIN={_canon(out_smi)!r}"
        )


def test_metal_organyl_defers_for_neutral_group12_mono() -> None:
    """Neutral group-12 mono/di-organyls stay with the existing
    _detect_simple_organometallic path (still correctly named)."""
    assert name_smiles("C[Hg]") == "methylmercury"
    assert name_smiles("[CH3][Zn]") == "methylzinc"
    assert name_smiles("[CH3][Cd]") == "methylcadmium"
