"""Regression tests for Stage 6 R3-C: ``[N]annulene`` systematic naming.

Root cause #20 in ``docs/opsin_coverage_taxonomy.md``: Blue Book P-22.1.4
defines ``[N]annulene`` as the systematic name of a monocyclic, fully
conjugated, all-carbon ring of size N (≥ 8).  Two structural realisations
arrive from RDKit:

  * 4n+2 sizes (10, 14, 18, 22, 26, …) — the ring is detected as aromatic;
    every ring atom carries the aromatic flag and the explicit DB count is
    zero.
  * 4n   sizes (8, 12, 16, 20, 24, …)  — the ring is Kekulé; every ring atom
    participates in exactly one endocyclic double bond and there are N/2
    endocyclic DBs total.

Pre-fix, the carbocyclic systematic path emitted ``cycloXane`` (e.g.
``cyclodocosane`` for [22]annulene) for the 4n+2 ring sizes that lacked a
curated retained entry, because ``_detect_ring_unsaturation`` short-circuits
to empty lists for aromatic rings.  The fix adds ``_try_annulene_name`` in
``iupac_namer/ring_naming/monocyclic.py``: when the ring qualifies as a
fully-conjugated even monocycle of size ≥ 8, the systematic monocyclic path
emits ``[N]annulene`` directly, never falling into the saturated branch.

The retained-name lookup (data_loader curated entries for sizes 10/14/16/18/20)
runs higher up in ``name_monocyclic`` and continues to win for those sizes.
``_try_annulene_name`` is the uniform systematic fallback for the remaining
even sizes (e.g. 8, 12, 22, 24, 26, 28, 30) and matches the retained name
for the curated sizes (so dedup keeps either copy — the names are identical).
"""
from __future__ import annotations

import pytest

from iupac_namer.engine import name_smiles


def _name(smiles: str) -> str:
    return name_smiles(smiles)


def _build_kekule(n: int) -> str:
    """Build the all-cis kekulé SMILES for an [N]annulene (even N)."""
    bonds = []
    for i in range(n - 1):
        bonds.append("=" if i % 2 == 0 else "")
    return "C1" + "".join(b + "C" for b in bonds) + "1"


# ---------------------------------------------------------------------------
# 1. 4n+2 (Hückel-aromatic) annulenes — RDKit aromatizes the ring
# ---------------------------------------------------------------------------

class TestHuckelAnnulenes:
    """Aromatic [N]annulene where N ≡ 2 (mod 4): 10, 14, 18, 22, 26."""

    def test_10_annulene(self):
        # c1ccccccccc1 / C1=CC=CC=CC=CC=C1 — both canonicalize to the
        # aromatic SMILES; the curated retained entry already maps this.
        assert _name(_build_kekule(10)) == "[10]annulene"

    def test_14_annulene(self):
        assert _name(_build_kekule(14)) == "[14]annulene"

    def test_18_annulene(self):
        assert _name(_build_kekule(18)) == "[18]annulene"

    def test_22_annulene(self):
        # Pre-fix this emitted "cyclodocosane" because no retained entry
        # existed for the 22-aromatic SMILES.  This is the load-bearing case.
        assert _name(_build_kekule(22)) == "[22]annulene"

    def test_26_annulene(self):
        # No curated retained entry — must come from the systematic path.
        assert _name(_build_kekule(26)) == "[26]annulene"


# ---------------------------------------------------------------------------
# 2. 4n (Kekulé) annulenes — RDKit keeps explicit double bonds
# ---------------------------------------------------------------------------

class TestKekuleAnnulenes:
    """Non-Hückel [N]annulene where N ≡ 0 (mod 4): 8, 12, 16, 20, 24."""

    def test_8_annulene(self):
        # P-54.2: cycloocta-1,3,5,7-tetraene is the PIN for the 8-ring
        # fully-unsaturated carbocycle; [8]annulene is general-nomenclature
        # only.  Both forms are OPSIN-accepted; the engine emits the spec
        # PIN (cycloocta-1,3,5,7-tetraene).
        assert _name(_build_kekule(8)) == "cycloocta-1,3,5,7-tetraene"

    def test_12_annulene(self):
        assert _name(_build_kekule(12)) == "[12]annulene"

    def test_16_annulene(self):
        # Curated retained entry exists for the kekulé canonical form.
        assert _name(_build_kekule(16)) == "[16]annulene"

    def test_20_annulene(self):
        assert _name(_build_kekule(20)) == "[20]annulene"

    def test_24_annulene(self):
        assert _name(_build_kekule(24)) == "[24]annulene"


# ---------------------------------------------------------------------------
# 3. Negative cases — non-annulene rings must NOT be renamed
# ---------------------------------------------------------------------------

class TestAnnuleneGuards:
    """Inputs that look annulene-adjacent but must keep their normal names."""

    def test_benzene_unaffected(self):
        # 6-ring is benzene, NOT [6]annulene — guard: N ≥ 8.
        assert _name("c1ccccc1") == "benzene"

    def test_cyclohexane_unaffected(self):
        # Saturated 6-ring → cyclohexane.
        assert _name("C1CCCCC1") == "cyclohexane"

    def test_cyclooctane_unaffected(self):
        # Fully-saturated 8-ring is NOT an annulene.
        assert _name("C1CCCCCCC1") == "cyclooctane"

    def test_cyclooctene_unaffected(self):
        # 8-ring with ONE double bond → cyclooctene, not [8]annulene.
        assert _name("C1=CCCCCCC1") == "cyclooctene"

    def test_cyclodocosane_saturated(self):
        # Fully-saturated 22-ring stays "cyclodocosane".
        assert _name("C1CCCCCCCCCCCCCCCCCCCCC1") == "cyclodocosane"

    def test_odd_ring_no_annulene(self):
        # 9-membered fully-conjugated rings do NOT receive an annulene
        # name (annulenes are even-membered).  The 9-ring with one sp3
        # carbon (canonical ``[9]annulene`` from OPSIN) round-trips via
        # the systematic cyclo-non-X-tetraene path; not the annulene
        # branch.  We only verify here that the engine does NOT emit an
        # ``[9]annulene`` name for an odd ring.
        smi = "C1C=CC=CC=CC=C1"  # 9-ring, 4 endo DBs, 1 sp3 carbon
        out = _name(smi)
        assert "annulene" not in out

    def test_heterocycle_unaffected(self):
        # 8-membered heterocycle (1-aza analogue) must NOT be named
        # ``[8]annulene`` — it has a heteroatom and goes through the HW /
        # replacement path.
        smi = "N1=CC=CC=CC=C1"  # 8-ring, 1 N, fully conjugated
        out = _name(smi)
        assert "annulene" not in out


# ---------------------------------------------------------------------------
# 4. OPSIN round-trip parity (the eval gold)
# ---------------------------------------------------------------------------

class TestAnnuleneOPSINRoundTrip:
    """Each emitted [N]annulene name must round-trip via OPSIN to canonical
    SMILES matching the input."""

    @pytest.mark.parametrize("n", [8, 10, 12, 14, 16, 18, 20, 22, 24])
    def test_round_trip(self, n):
        from rdkit import Chem
        from py2opsin import py2opsin
        import os
        import tempfile

        # py2opsin race-around-temp-file workaround
        cwd = os.getcwd()
        tmp = tempfile.mkdtemp(prefix="opsin_annulene_test_")
        try:
            os.chdir(tmp)
            smi = _build_kekule(n)
            in_canon = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
            engine_name = _name(smi)
            opsin_smis = py2opsin([engine_name], output_format="SMILES")
            assert opsin_smis and opsin_smis[0], (
                f"OPSIN returned no SMILES for {engine_name!r}"
            )
            opsin_canon = Chem.MolToSmiles(Chem.MolFromSmiles(opsin_smis[0]))
            assert in_canon == opsin_canon, (
                f"Round-trip mismatch for N={n}: input={in_canon!r}, "
                f"engine={engine_name!r}, opsin={opsin_canon!r}"
            )
        finally:
            os.chdir(cwd)
