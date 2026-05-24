"""Stage 4 architectural foundation: 4-ring retained bases (e.g.
cyclopenta[a]phenanthrene) fused to a [1,3]-dihetero smaller component.

These tests pin down the current behaviour of the gate-lift +
completeness guard added to ``fused.py``:

  1. The 5-ring total cap (``len(rings) > 5``) admits a 5-ring fused
     system (1 dihetero smaller + 4-ring base) into the Stage 1/2/3
     pipeline so future Stage 4 work (full peri-locant inference for
     retained 4-ring bases) can hook in without touching the gate.

  2. The completeness guard in ``_select_multi_base_numbering`` rejects
     retained-name numberings that are MISSING junction locants (the
     cyclopenta[a]phenanthrene retained entry's ``atom_locants`` omits
     positions 4a/4b/8a/10a/11a).  Without this guard ``_fusion_letter_
     from_string_locants`` would compute a deep alphabetical letter
     ('o', 'p', ...) that OPSIN rejects when round-tripping.

  3. As a consequence, the polycyclic partner of FDA-0054 amcinonide
     (steroid + ring-A enone fused to 16,17-dioxolane) currently emits
     no fused-hetero candidate, so the existing VB/polyspiro path keeps
     producing the spiro[7,9-dioxapentacyclo[...]] form that round-trips
     cleanly through OPSIN.  When Stage 4 lands (peri-locant inference +
     re-numbering for 4-ring retained bases) the same gate will let the
     canonical ``[1,3]dioxolo[4,5-b]<dodecahydro-cyclopenta[a]phenanth-
     rene>`` form flow through this pipeline.
"""
from __future__ import annotations

import os

# Pin OPSIN's JAVA_HOME the same way authoritative_eval.py does.
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
from iupac_namer.ring_naming.fused import (
    _identify_smaller_and_base,
    _select_multi_base_numbering,
    name_fused,
)
from iupac_namer.types import CandidateParent


# Polycyclic partner of FDA-0054 amcinonide after polyspiro articulation:
# steroid skeleton (3 cyclohexane + 1 cyclopentane) with ring-A enone
# fused at the 16,17 edge (D-ring of cyclopenta[a]phenanthrene) to a
# 16,17-dioxolane.  5 rings total.
AMCINONIDE_PARTNER_SMILES = "C1=CC2C(=CC1)CCC1C2CCC2C1CC1OCOC12"


def _ring_system(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    aa = AtomAnalysis(mol)
    ra = RingAnalysis(mol, aa)
    rs = ra.ring_systems[0]
    return mol, rs


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


# ---------------------------------------------------------------------------
# 1. Gate lift: 5-ring fused system enters the smaller+base classifier.
# ---------------------------------------------------------------------------

def test_5ring_fused_system_passes_size_gate():
    """The (lifted) total-rings-≤5 gate admits a 5-ring fused system so
    the smaller+base classifier can run on it.  Previously the hard
    ``len(rings) > 3`` gate short-circuited at this size."""
    mol, rs = _ring_system(AMCINONIDE_PARTNER_SMILES)
    assert rs.type == "fused"
    assert len(rs.rings) == 5
    triple = _identify_smaller_and_base(
        rs.rings, rs.fusion_info.fusion_atoms, mol
    )
    assert triple is not None, "smaller+base identification should succeed"
    smaller, base_rings, edge = triple
    # Smaller is the dioxolane (5-ring with 2 O + middle C between them).
    assert len(smaller) == 5
    smaller_elems = {mol.GetAtomWithIdx(a).GetSymbol() for a in smaller}
    assert smaller_elems == {"C", "O"}
    # Base is the 4-ring steroid (cyclopenta[a]phenanthrene-type).
    assert len(base_rings) == 4


# ---------------------------------------------------------------------------
# 2. Completeness guard: incomplete numberings are rejected.
# ---------------------------------------------------------------------------

def test_amcinonide_partner_emits_no_fused_candidate_yet():
    """Stage 4 hasn't landed: the cyclopenta[a]phenanthrene retained entry
    omits junction locants, so the completeness guard rejects the Stage 3
    multi-base path and ``name_fused`` returns no candidate.  This is the
    safe behaviour — the existing polyspiro/VB form still wins."""
    mol, rs = _ring_system(AMCINONIDE_PARTNER_SMILES)
    cand = _candidate_for(rs)
    parents = name_fused(rs, cand, mol)
    # name_fused should NOT emit a fused-hetero candidate for this base
    # until Stage 4 lands the peri-locant inference.
    assert parents == [], (
        f"expected no fused-hetero candidate (Stage 4 not yet landed); "
        f"got {[p.name for p in parents]}"
    )


def test_completeness_guard_rejects_incomplete_numbering():
    """``_select_multi_base_numbering`` must reject a numbering whose
    label dict misses any base atom.  For cyclopenta[a]phenanthrene the
    retained entry's ``atom_locants`` covers only 15 of 17 ring atoms
    (junction positions 5 and 10 / 4a 4b 8a 10a 11a peri-locants are
    omitted), so the multi-base lookup must return None."""
    mol, rs = _ring_system(AMCINONIDE_PARTNER_SMILES)
    triple = _identify_smaller_and_base(
        rs.rings, rs.fusion_info.fusion_atoms, mol
    )
    assert triple is not None
    _smaller, base_rings, edge = triple
    sel = _select_multi_base_numbering(
        base_rings, mol, edge, force_aromatic=False
    )
    assert sel is None, (
        f"completeness guard must reject incomplete cyclopenta[a]phenanthrene "
        f"numbering; got {sel!r}"
    )


# ---------------------------------------------------------------------------
# 3. Existing 2-ring multi-base bases still work (regression guard).
# ---------------------------------------------------------------------------

def test_naphthalene_base_still_produces_full_numbering():
    """Stage 2B with naphthalene base must still pass — the retained
    entry for naphthalene includes ALL 10 atom_locants (1..8, 4a, 8a)
    so the completeness guard is satisfied and the fusion letter
    computation succeeds."""
    # 2,3-methylenedioxynaphthalene-style: dioxolo on naphthalene.
    smi = "c1ccc2cc3c(cc2c1)OCO3"
    mol, rs = _ring_system(smi)
    triple = _identify_smaller_and_base(
        rs.rings, rs.fusion_info.fusion_atoms, mol
    )
    assert triple is not None
    _smaller, base_rings, edge = triple
    sel = _select_multi_base_numbering(
        base_rings, mol, edge, force_aromatic=False
    )
    assert sel is not None, "naphthalene-base Stage 2B regression"
    base_name, _numbering, letter = sel
    assert base_name == "naphthalene"
    assert letter == "b"


# ---------------------------------------------------------------------------
# 4. Stage 4 investigation: OPSIN-probe findings.
# ---------------------------------------------------------------------------
#
# These tests lock in the architectural finding from April 2026: for
# amcinonide's polycyclic partner (16,17-dioxolane fused to the D-ring of
# cyclopenta[a]phenanthrene), OPSIN does NOT accept any
# ``[1,3]dioxolo[4,5-X]cyclopenta[a]phenanthrene`` form that round-trips to
# the correct skeletal topology.  The canonical IUPAC form is instead a
# methylenedioxy-bridge (``16,17-methylenedioxyhexadecahydro-1H-
# cyclopenta[a]phenanthrene``), which is handled by a different naming
# pathway.
#
# Consequently, Stage 4 cannot ship a ``name_fused``-based canonical name
# for amcinonide's partner — the ``fused.py`` pipeline is architecturally
# the wrong entry point.  The completeness guard therefore stays in place
# as a safety net.  These tests pin that finding so a future refactor
# doesn't silently lift the guard without confirming the OPSIN gap closes.


def test_opsin_rejects_all_fusion_letters_for_amcinonide_shape():
    """OPSIN probe evidence: no ``[1,3]dioxolo[4,5-X]cyclopenta[a]phenanthrene``
    letter (a..n) gives a skeleton matching amcinonide's polycyclic
    partner.  This is why Stage 4 via ``name_fused`` is a dead end for
    amcinonide; the methylenedioxy-bridge path is the canonical form.

    Requires py2opsin + Java (the test env pins both above).
    """
    try:
        from py2opsin import py2opsin
    except ImportError:
        import pytest
        pytest.skip("py2opsin not installed")

    # Fully-saturate the partner skeleton for topology-only comparison.
    m = Chem.MolFromSmiles(AMCINONIDE_PARTNER_SMILES)
    from rdkit.Chem import RWMol, BondType
    rw = RWMol(m)
    for b in rw.GetBonds():
        b.SetBondType(BondType.SINGLE)
    for atom in rw.GetAtoms():
        atom.SetIsAromatic(False)
    Chem.SanitizeMol(rw)
    target_canon = Chem.MolToSmiles(rw)

    matches: list[str] = []
    for letter in "abcdefghijklmn":
        name = (
            f"hexadecahydro-[1,3]dioxolo[4,5-{letter}]"
            "cyclopenta[a]phenanthrene"
        )
        smi = py2opsin(name)
        if not smi:
            continue
        got = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
        if got == target_canon:
            matches.append(f"[4,5-{letter}]")
    assert matches == [], (
        f"Stage 4 architectural assumption broken: OPSIN now accepts "
        f"{matches} as the amcinonide-partner fused form.  Revisit "
        f"``name_fused`` to emit this canonical form."
    )


def test_opsin_accepts_methylenedioxy_bridge_form():
    """Positive probe: the canonical IUPAC form for amcinonide's partner
    via OPSIN is ``16,17-methylenedioxyhexadecahydro-1H-
    cyclopenta[a]phenanthrene`` (methylenedioxy BRIDGE on the saturated
    steroid skeleton), not a ring-fusion.  This test pins that OPSIN
    round-trips the bridge form to the expected skeleton.
    """
    try:
        from py2opsin import py2opsin
    except ImportError:
        import pytest
        pytest.skip("py2opsin not installed")

    m = Chem.MolFromSmiles(AMCINONIDE_PARTNER_SMILES)
    from rdkit.Chem import RWMol, BondType
    rw = RWMol(m)
    for b in rw.GetBonds():
        b.SetBondType(BondType.SINGLE)
    for atom in rw.GetAtoms():
        atom.SetIsAromatic(False)
    Chem.SanitizeMol(rw)
    target_canon = Chem.MolToSmiles(rw)

    name = "16,17-methylenedioxyhexadecahydro-1H-cyclopenta[a]phenanthrene"
    smi = py2opsin(name)
    assert smi, f"OPSIN rejected canonical bridge form {name!r}"
    got = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    assert got == target_canon, (
        f"OPSIN bridge form doesn't round-trip to amcinonide-partner "
        f"skeleton: expected {target_canon!r}, got {got!r}"
    )


def test_opsin_accepts_fused_dioxolo_on_other_4ring_bases():
    """Positive probe: OPSIN DOES accept fused dioxolo on other 4-ring
    retained aromatic bases (phenanthrene, anthracene, chrysene, pyrene).
    This documents that Stage 4 could still be useful for non-steroid
    4-ring bases if/when the eval set includes them — the amcinonide gap
    is specific to cyclopenta[a]phenanthrene's peri-numbering topology.
    """
    try:
        from py2opsin import py2opsin
    except ImportError:
        import pytest
        pytest.skip("py2opsin not installed")

    # Each of these is accepted by OPSIN as a fully-saturated fused form.
    accepted_probes = [
        "decahydro-[1,3]dioxolo[4,5-a]phenanthrene",
        "decahydro-[1,3]dioxolo[4,5-a]anthracene",
        "decahydro-[1,3]dioxolo[4,5-a]chrysene",
    ]
    for name in accepted_probes:
        smi = py2opsin(name)
        assert smi, f"OPSIN unexpectedly rejected {name!r}"
