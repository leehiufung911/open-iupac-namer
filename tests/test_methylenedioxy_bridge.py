"""Stage 5 — unit tests for methylenedioxy-bridge naming on retained
polycyclic bases (``ring_naming/methylenedioxy_bridge.py``).

These tests pin three kinds of invariants:

  1. POSITIVE: a synthetic methylenedioxy-bridged steroid skeleton
     (16,17-methylenedioxy-hexadecahydro-1H-cyclopenta[a]phenanthrene)
     is correctly detected and named by the Stage 5 module.

  2. NEGATIVE / GUARD: the guard does NOT fire on the wrong topologies:
       * amcinonide's acetonide (middle C has 0 H, 2 C neighbors — NOT a
         methylene) must be ignored so its existing polyspiro-articulation
         name stays unchanged;
       * benzodioxole (monocyclic benzene base — handled by existing
         retained fused dioxolo path) must be ignored;
       * simple 1,3-dioxolane alone (no polycyclic base) must be ignored.

  3. RING-LEVEL: the detector helper ``_find_methylenedioxy_ring``
     identifies the correct 5-ring and rejects near-misses (e.g. sulfur
     bridges, O-C(R)(R')-O ketals).
"""

from __future__ import annotations

import os

# Pin OPSIN's JAVA_HOME the same way authoritative_eval.py does (tests that
# use py2opsin rely on this; skip gracefully when py2opsin is absent).
os.environ.setdefault(
    "JAVA_HOME",
    os.environ.get("JAVA_HOME", ""),
)
os.environ["PATH"] = (
    os.environ["JAVA_HOME"] + "/bin" + os.pathsep + os.environ.get("PATH", "")
)

from rdkit import Chem

from iupac_namer.perception.atoms import AtomAnalysis
from iupac_namer.perception.rings import RingAnalysis
from iupac_namer.ring_naming.methylenedioxy_bridge import (
    _base_ring_system_and_locants,
    _find_methylenedioxy_ring,
    name_methylenedioxy_bridge,
)
from iupac_namer.types import CandidateParent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ring_system(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    aa = AtomAnalysis(mol)
    ra = RingAnalysis(mol, aa)
    return mol, ra.ring_systems[0]


def _candidate_for(rs):
    return CandidateParent(
        atom_indices=rs.atom_indices,
        type=rs.type,
        length=len(rs.atom_indices),
        ring_system=rs,
        unsaturation=None,
        element=None,
        lambda_value=None,
    )


# Canonical target: the saturated steroid skeleton with an O-CH2-O bridge
# spanning the D-ring's 16,17 positions.  This is exactly the OPSIN-verified
# target from Stage 4's positive probe (test_opsin_accepts_methylenedioxy_
# bridge_form in tests/test_fused_ring_hetero_4ring_base.py).
METHYLENEDIOXY_STEROID = "C1CCC2C(C1)CCC1C2CCC2C1CC1OCOC12"


# ---------------------------------------------------------------------------
# 1. Positive: a genuine methylenedioxy-bridged steroid is detected.
# ---------------------------------------------------------------------------


def test_methylenedioxy_steroid_detected():
    """``_find_methylenedioxy_ring`` must identify the 5-ring O-CH2-O of
    the synthetic methylenedioxy-bridged cyclopenta[a]phenanthrene."""
    mol, rs = _ring_system(METHYLENEDIOXY_STEROID)
    assert rs.type == "fused"
    found = _find_methylenedioxy_ring(rs, mol)
    assert found is not None, (
        "expected to detect the O-CH2-O dioxolane ring"
    )
    bridge_ring, o1, ch2, o2 = found
    assert len(bridge_ring) == 5
    # CH2 must be the sp3 carbon with exactly 2 H and 2 heavy neighbors.
    ch2_atom = mol.GetAtomWithIdx(ch2)
    assert ch2_atom.GetAtomicNum() == 6
    assert ch2_atom.GetTotalNumHs() == 2
    assert ch2_atom.GetDegree() == 2
    # Both oxygens are neutral, non-aromatic, and bond only to CH2 + one C.
    for oi in (o1, o2):
        oa = mol.GetAtomWithIdx(oi)
        assert oa.GetAtomicNum() == 8
        assert oa.GetFormalCharge() == 0
        assert not oa.GetIsAromatic()
        nbrs = list(oa.GetNeighbors())
        assert len(nbrs) == 2
        assert ch2 in (n.GetIdx() for n in nbrs)


def test_methylenedioxy_steroid_emits_canonical_name():
    """The full naming path emits the expected
    ``16,17-methylenedioxy-hexadecahydro-1H-cyclopenta[a]phenanthrene``."""
    mol, rs = _ring_system(METHYLENEDIOXY_STEROID)
    cand = _candidate_for(rs)
    out = name_methylenedioxy_bridge(rs, cand, mol)
    assert len(out) == 1, f"expected 1 NamedParent, got {out}"
    np = out[0]
    assert np.naming_method == "methylenedioxy_bridge"
    assert np.name == (
        "16,17-methylenedioxy-hexadecahydro-1H-cyclopenta[a]phenanthrene"
    ), f"got {np.name!r}"


def test_methylenedioxy_steroid_name_roundtrips_through_opsin():
    """Positive OPSIN probe: the Stage 5 output for the synthetic steroid
    must round-trip to the correct skeletal topology."""
    try:
        from py2opsin import py2opsin
    except ImportError:
        import pytest
        pytest.skip("py2opsin not installed")

    mol, rs = _ring_system(METHYLENEDIOXY_STEROID)
    cand = _candidate_for(rs)
    out = name_methylenedioxy_bridge(rs, cand, mol)
    assert out, "expected Stage 5 output"
    name = out[0].name
    smi = py2opsin(name)
    assert smi, f"OPSIN rejected Stage 5 output {name!r}"
    got = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    expected = Chem.MolToSmiles(Chem.MolFromSmiles(METHYLENEDIOXY_STEROID))
    assert got == expected, (
        f"OPSIN round-trip mismatch: expected {expected!r}, got {got!r}"
    )


# ---------------------------------------------------------------------------
# 2. Negative: acetonide / ketal / non-CH2 middle carbon must NOT match.
# ---------------------------------------------------------------------------


def test_amcinonide_not_matched_as_methylenedioxy():
    """Amcinonide (FDA-0054) has an ACETONIDE (O-C(cyclopentyl)-O), not a
    methylenedioxy bridge.  The middle C of its dioxolane has 0 H and two
    ring-C neighbors (the spiro cyclopentane), so the Stage 5 guard must
    refuse — amcinonide's existing polyspiro-articulation name stays."""
    amcinonide_smi = (
        "CC(=O)OCC(=O)[C@@]12OC3(CCCC3)O[C@@H]1C[C@H]1[C@@H]3CCC4=CC(=O)C=C"
        "[C@]4(C)[C@@]3(F)[C@@H](O)C[C@@]12C"
    )
    mol, rs = _ring_system(amcinonide_smi)
    cand = _candidate_for(rs)
    out = name_methylenedioxy_bridge(rs, cand, mol)
    assert out == [], (
        f"Stage 5 must NOT fire on amcinonide's acetonide; got {out!r}"
    )


def test_benzodioxole_monocyclic_base_rejected():
    """Benzodioxole's base is a MONOCYCLIC benzene — the existing retained
    fused ``1,3-benzodioxol`` path is the IUPAC-preferred form.  Stage 5
    must refuse (guard: ``len(base_rs.rings) < 2`` check inside
    ``_base_ring_system_and_locants``)."""
    mol, rs = _ring_system("c1ccc2c(c1)OCO2")
    cand = _candidate_for(rs)
    out = name_methylenedioxy_bridge(rs, cand, mol)
    assert out == [], (
        f"Stage 5 must NOT fire on benzodioxole (monocyclic base); "
        f"got {out!r}"
    )


def test_spiro_dioxolane_rejected():
    """A spiro dioxolane (e.g. 1,4-dioxaspiro[4.5]decane, the cyclohexanone
    ethylene ketal) does NOT have an O-CH2-O bridge — its dioxolane has
    an O-CH2-CH2-O pattern (ethylene glycol ketal).  Guard must refuse."""
    # 1,4-dioxaspiro[4.5]decane: SMILES OCC(OCC1)(CCCCC1)  — cyclohexanone
    # ketal. Actually canonical: C1CCC2(CC1)OCCO2
    mol, rs = _ring_system("C1CCC2(CC1)OCCO2")
    found = _find_methylenedioxy_ring(rs, mol)
    assert found is None, (
        "ethylene ketal (O-CH2-CH2-O) must not match the O-CH2-O guard"
    )


def test_thio_analogue_rejected():
    """A thiomethylenedioxy analogue (O-CH2-S instead of O-CH2-O) must
    NOT be picked up as a methylenedioxy bridge."""
    # A synthetic cyclopentane[...]steroid-like structure with S
    # (won't canonicalize to any retained name — we only test the low-level
    # detector refuses the O,S mixed ring).
    mol, rs = _ring_system("C1CCC2C(C1)CCC1C2CCC2C1CC1OCSC12")
    found = _find_methylenedioxy_ring(rs, mol)
    assert found is None, (
        "O-CH2-S ring must not match the O-CH2-O guard"
    )


# ---------------------------------------------------------------------------
# 3. Integration: Stage 5 output ranks above spiro_polycyclic.
# ---------------------------------------------------------------------------


def test_strategy_rank_ordering():
    """The strategy's naming-method rank table must place
    ``methylenedioxy_bridge`` strictly between the retained/HW tier and the
    spiro_polycyclic tier so real methylenedioxy-bridged steroids win over
    articulation-split polyspiro but retained fused names still win over
    Stage 5."""
    from iupac_namer.strategy import IUPACCanonical
    from iupac_namer.types import CandidateParent as CP, NamedParent

    # Build a dummy ring system (type="fused") so _naming_method_score
    # returns the ring-branch lookup.  Use zero-atom frozen sets; the
    # method_ranks dict is looked up by naming_method string only.
    from iupac_namer.types import RingSystem as RS
    rs = RS(
        atom_indices=frozenset(),
        rings=(),
        type="fused",
        aromatic=False,
        bridge_sizes=None,
        spiro_sizes=None,
        fusion_info=None,
        heteroatoms=None,
        ring_size=0,
    )
    cand = CP(
        atom_indices=frozenset(),
        type="fused",
        length=0,
        ring_system=rs,
        unsaturation=None,
        element=None,
        lambda_value=None,
    )
    def _np(method: str) -> NamedParent:
        return NamedParent(
            candidate=cand,
            name="",
            stem="",
            alkyl_stem=None,
            naming_method=method,
            indicated_hydrogen=None,
            numbering_options=(),
        )
    strat = IUPACCanonical()
    r_retained = strat._naming_method_score(_np("retained"))
    r_hw = strat._naming_method_score(_np("hantzsch_widman"))
    r_methylene = strat._naming_method_score(_np("methylenedioxy_bridge"))
    r_spiropoly = strat._naming_method_score(_np("spiro_polycyclic"))
    r_vb = strat._naming_method_score(_np("von_baeyer"))
    r_sys = strat._naming_method_score(_np("systematic"))

    # retained (100) > HW (50) > methylenedioxy_bridge (45) > spiro_polycyclic (40).
    # Stage 5 sits one tier below HW so pure retained/HW wins over Stage 5 when
    # both apply (e.g. a monocyclic benzene base's benzodioxol retained form).
    # HW does not apply to polycyclic bases, so Stage 5 still wins for the
    # steroid case where HW would return no candidate.
    assert r_retained > r_methylene, (r_retained, r_methylene)
    assert r_hw > r_methylene, (r_hw, r_methylene)
    assert r_methylene > r_spiropoly, (r_methylene, r_spiropoly)
    assert r_methylene > r_vb, (r_methylene, r_vb)
    assert r_methylene > r_sys, (r_methylene, r_sys)
