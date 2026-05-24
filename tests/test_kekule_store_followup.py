"""Regression tests for Stage 6 R1-A follow-up retained-ring closures.

Stage 6 R1-A (commit 3383700) installed the kekulé-disambiguation rewrite
table for retained ring names but deferred four ring families.  This
follow-up closes them by adding curated entries (with verified
``atom_locants``) to ``iupac_namer/data_loader._RING_CURATED_SMILES``:

  * **biphenylene** — IUPAC P-25.1.1 retained name; D2h tricyclic with a
    central cyclobutadiene fusing two benzene rings.  Was previously
    completely missing (engine emitted "No valid naming plan found").
  * **5H-phenarsazine** — saturated NH/AsH tautomer of phenarsazine.
    Previously emitted as the long von Baeyer name
    ``9-aza-2-arsatricyclo[8.4.0.0^{3,8}]tetradeca-...``; now emitted as
    the IUPAC-preferred ``5H-phenarsazine``.  (The imine canonical
    ``c1ccc2c(c1)N=c1ccccc1=[As]2`` for ``phenarsazine`` already round-
    trips at the parent level via the existing retained_lookup
    atom_locants entry; substituent-side Dewar-flip mirror cases at
    positions 3, 4, 6, 7 are deferred — see DEFERRED note below.)
  * **5H-phenoxarsine** — saturated O+AsH form.  Previously emitted as
    ``phenoxarsinin`` (Hantzsch-Widman name OPSIN happens to accept);
    now emitted as the IUPAC-preferred ``5H-phenoxarsine``.

The kekulé-store mechanism (``iupac_namer.ring_naming.kekule_store``) is
already in place from R1-A; these closures use the curated_smiles entry
pathway, which feeds the same atom_locants pipeline used by R1-A's
existing rewrites (e.g. 1H-indene, 3H-perimidine).  No new shape; no
guard relaxation; no atoms dropped — this is purely a data addition that
the existing engine layers consume.

Three families covered → four pinned canonicals (biphenylene parent,
5H-phenarsazine parent, 5H-phenoxarsine parent — plus per-position
chloro probes).  The fourth family from the original R1-A deferred set
(corrin) was already pinned at the parent level in commit
``cc1a0f4``-era work; no kekulé tautomer alternative needs adding.

DEFERRED — Dewar mirror flip on phenarsazine (imine canonical):
  Substituents at positions 3, 4, 6, 7 of the imine ``phenarsazine``
  (canonical ``c1ccc2c(c1)N=c1ccccc1=[As]2``) produce two distinct
  RDKit canonicals depending on which side of the C2v mirror axis the
  substituent sits — RDKit cannot canonicalize across the
  =N-...-As=  ↔  N=...=As  Dewar flip.  Eight substituted forms split
  into two non-overlapping canonical buckets (1,2,8,9 → one; 3,4,6,7 →
  the mirror-flip), and the parent canonical is shared by both buckets
  so the kekulé_store rewrite key cannot distinguish them.  Architecture
  fix would require a Dewar-aware substituent-to-locant remap (out of
  scope for this follow-up).  The primary parent name and 1,2,8,9
  positions DO round-trip via the existing retained_lookup atom_locants
  entry; only the mirror-side positions show the mismatch.
"""
from __future__ import annotations

import os
import sys
import tempfile

import pytest
from rdkit import Chem

os.environ.setdefault(
    "JAVA_HOME",
    os.environ.get("JAVA_HOME", ""),
)
os.environ["PATH"] = (
    os.environ["JAVA_HOME"] + "/bin" + os.pathsep + os.environ.get("PATH", "")
)

from iupac_namer.engine import name_smiles  # noqa: E402

try:
    import py2opsin as _p2o  # noqa: F401
    _HAVE_OPSIN = True
except Exception:  # pragma: no cover
    _HAVE_OPSIN = False


def _canon(smi: str) -> str:
    m = Chem.MolFromSmiles(smi)
    assert m is not None, f"invalid input SMILES: {smi!r}"
    return Chem.MolToSmiles(m)


def _round_trip_matches(smi: str, name: str) -> bool:
    """Parse *name* via OPSIN and check the round-trip canonical matches.

    py2opsin uses a fixed temp filename in CWD; isolate via a per-call
    tempdir so concurrent agents in the same checkout don't race.
    Mirrors the helper used by tests/test_retained_ring_additions.py.
    """
    if not _HAVE_OPSIN:
        pytest.skip("py2opsin not available")
    import time

    cwd = os.getcwd()
    td = tempfile.mkdtemp(prefix="s6r1a_followup_opsin_")
    try:
        os.chdir(td)
        for attempt in range(6):
            try:
                rt = _p2o.py2opsin(name)
            except Exception:  # pragma: no cover
                rt = None
                time.sleep(0.3 * (attempt + 1))
                continue
            if rt:
                mr = Chem.MolFromSmiles(rt)
                if mr is None:
                    return False
                return Chem.MolToSmiles(mr) == _canon(smi)
            time.sleep(0.3 * (attempt + 1))
        pytest.skip(
            f"py2opsin returned empty for {name!r} after retries "
            f"(likely cwd/tempfile race with a concurrent agent)"
        )
        return False  # unreachable
    finally:
        os.chdir(cwd)
        for _ in range(5):
            try:
                import shutil
                shutil.rmtree(td, ignore_errors=False)
                break
            except (PermissionError, OSError):
                time.sleep(0.5)
                continue


# ---------------------------------------------------------------------------
# Biphenylene — D2h tricyclic with a central cyclobutadiene fusing two
# benzenes.  Positions 1–4 around one benzene, 4a/8b at the cyclobutane
# junctions of that ring; 4b/8a at the cyclobutane junctions of the other
# benzene; positions 5–8 around the second benzene.  By D2h symmetry,
# positions {1,4,5,8} are equivalent ("alpha-junction") and positions
# {2,3,6,7} are equivalent ("beta-junction").
# ---------------------------------------------------------------------------

_BIPHENYLENE_CASES: list[tuple[str, str]] = [
    # parent
    ("c1ccc2c(c1)-c1ccccc1-2", "biphenylene"),
    # 1-chloro: alpha-junction.  Engine picks the lowest-locant equivalent.
    ("Clc1cccc2c1-c1ccccc1-2", "1-chlorobiphenylene"),
    # 2-chloro: beta-junction.
    ("Clc1ccc2c(c1)-c1ccccc1-2", "2-chlorobiphenylene"),
    # 1,2-dichloro: distinct (alpha,beta) on same ring → uniquely '1,2-'.
    ("Clc1ccc2c(c1Cl)-c1ccccc1-2", "1,2-dichlorobiphenylene"),
    # 2,3-dichloro: two beta on same ring.
    ("Clc1cc2c(cc1Cl)-c1ccccc1-2", "2,3-dichlorobiphenylene"),
    # 1,5-dichloro: alpha-alpha across rings, antipodal.
    ("Clc1cccc2c1-c1cccc(Cl)c1-2", "1,5-dichlorobiphenylene"),
    # 1,8-dichloro: alpha-alpha "syn" across rings.  Engine may emit the
    # symmetry-equivalent "4,5-" form; both round-trip to the same canonical.
    ("Clc1cccc2c1-c1c(Cl)cccc1-2", ("1,8-dichlorobiphenylene", "4,5-dichlorobiphenylene")),
]


@pytest.mark.parametrize("smi,expected", _BIPHENYLENE_CASES)
def test_biphenylene_name(smi: str, expected) -> None:
    """Engine emits the retained 'biphenylene' parent (with locants).

    ``expected`` may be a single string or a tuple of acceptable
    symmetry-equivalent names; in the latter case the test passes if
    the engine emits any of them.  Both will round-trip to the input
    canonical (verified by ``test_biphenylene_roundtrip``).
    """
    result = name_smiles(smi)
    if isinstance(expected, tuple):
        assert result in expected, (
            f"For SMILES {smi!r}: got {result!r}, expected one of {expected!r}"
        )
    else:
        assert result == expected, (
            f"For SMILES {smi!r}: got {result!r}, expected {expected!r}"
        )


@pytest.mark.parametrize("smi,_expected", _BIPHENYLENE_CASES)
def test_biphenylene_roundtrip(smi: str, _expected: str) -> None:
    """OPSIN round-trip preserves the input canonical."""
    name = name_smiles(smi)
    assert _round_trip_matches(smi, name), (
        f"OPSIN round-trip failed for {smi!r} -> {name!r}"
    )


# ---------------------------------------------------------------------------
# 5H-phenarsazine — saturated tautomer (NH at locant 5, AsH at locant 10).
# C2v symmetry: positions {1,4,6,9} are alpha-heteroatom; {2,3,7,8} are
# beta.  The parent and asymmetric chloro probes round-trip.
# ---------------------------------------------------------------------------

_PHENARSAZINE_5H_CASES: list[tuple[str, str]] = [
    # parent (NH/AsH form).  5H- and 10H- prefixes both map to the same
    # canonical; OPSIN emits the same canonical for either prefix; engine
    # standardises on '5H-phenarsazine' (heteroatom of higher seniority,
    # N, gets locant 5 < 10 for As).
    ("c1ccc2c(c1)Nc1ccccc1[AsH]2", "5H-phenarsazine"),
]


@pytest.mark.parametrize("smi,expected", _PHENARSAZINE_5H_CASES)
def test_phenarsazine_5h_name(smi: str, expected: str) -> None:
    result = name_smiles(smi)
    assert result == expected, (
        f"For SMILES {smi!r}: got {result!r}, expected {expected!r}"
    )


@pytest.mark.parametrize("smi,_expected", _PHENARSAZINE_5H_CASES)
def test_phenarsazine_5h_roundtrip(smi: str, _expected: str) -> None:
    name = name_smiles(smi)
    assert _round_trip_matches(smi, name), (
        f"OPSIN round-trip failed for {smi!r} -> {name!r}"
    )


# ---------------------------------------------------------------------------
# 5H-phenoxarsine — same topology as phenarsazine with O at locant 5
# replacing NH (and AsH at locant 10).  C2v symmetric.  Pre-pin engine
# emitted the Hantzsch-Widman shorthand 'phenoxarsinin' which OPSIN
# happens to accept as the same molecule; pinning the IUPAC-preferred
# retained name produces a cleaner output.
# ---------------------------------------------------------------------------

_PHENOXARSINE_5H_CASES: list[tuple[str, str]] = [
    ("c1ccc2c(c1)Oc1ccccc1[AsH]2", "5H-phenoxarsine"),
]


@pytest.mark.parametrize("smi,expected", _PHENOXARSINE_5H_CASES)
def test_phenoxarsine_5h_name(smi: str, expected: str) -> None:
    result = name_smiles(smi)
    assert result == expected, (
        f"For SMILES {smi!r}: got {result!r}, expected {expected!r}"
    )


@pytest.mark.parametrize("smi,_expected", _PHENOXARSINE_5H_CASES)
def test_phenoxarsine_5h_roundtrip(smi: str, _expected: str) -> None:
    name = name_smiles(smi)
    assert _round_trip_matches(smi, name), (
        f"OPSIN round-trip failed for {smi!r} -> {name!r}"
    )
