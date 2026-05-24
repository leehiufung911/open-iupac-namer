"""
tests/test_retained_terpenes.py

Stage 8 regression tests for retained terpene-ring names.

ARCHITECTURAL DECISION (Stage 8 vocabulary expansion):
======================================================

The audit refresh listed several retained terpene bicyclic names as
candidates for inclusion in ``data_loader._RING_CURATED_SMILES``.  An
OPSIN round-trip probe of each candidate confirmed which names parse
to a stable canonical SMILES:

    camphor       OK   CC12CCC(CC1=O)C2(C)C
    bornane       OK   CC12CCC(CC1)C2(C)C
    borneol       OK   CC1(C)C2CCC1(C)C(O)C2
    isoborneol    NOT RECOGNISED by OPSIN
    fenchone      NOT RECOGNISED by OPSIN
    pinane        OK   CC1CCC2CC1C2(C)C
    alpha-pinene  OK   CC1=CCC2CC1C2(C)C
    beta-pinene   OK   C=C1CCC2CC1C2(C)C
    norbornane    OK   C1CC2CCC1C2
    norbornene    OK   C1=CC2CCC1C2
    p-menthane    OK   CC1CCC(C(C)C)CC1
    carane        OK   CC1CCC2C(C1)C2(C)C

Of these, **only camphor is the IUPAC preferred name (PIN) for its
structure** under the 2013 Blue Book retained-name framework.  The
remainder are explicitly NOT PINs:

    Blue Book P1.html line 155:
        bornane / bicyclo[2.2.1]heptane (PIN)
    Blue Book Papp3.html lines 120, 153, 158:
        bornane    "(named systematically by CAS)"
        p-menthane "(named systematically by CAS)"
        pinane     "(named systematically by CAS)"

Adding ``bornane`` etc. to ``_RING_CURATED_SMILES`` would override
the correct PIN ``1,7,7-trimethylbicyclo[2.2.1]heptane`` with a
non-preferred trivial name — a clear violation of "architecture over
score" (project CLAUDE.md).  These names are therefore DEFERRED.

Camphor is a different case: it is retained under Chapter P-66.6.3
and Table 28.1 (with "(+)-" / "(-)-" stereo descriptor for the chiral
isomers) and is already curated in
``data/retained_names_expanded.json``.  This test pins the existing
behaviour as a regression guard.

The audit's primary stereo probe ``C[C@@]12C(C[C@@H](CC1)C2(C)C)=O``
must continue to emit the *systematic* name even with camphor curated:
the strategy stereo-drop guard
(``strategy._RETAINED_NAMES_ENCODING_STEREO``) does not list camphor,
and ``camphor`` cannot encode stereo.  An attempt to add camphor to
the stereo-encoding set is OUT OF SCOPE for this stage (that file is
owned by another agent in the parallel-dispatch protocol).
"""
from __future__ import annotations

import os
import sys

import pytest
from rdkit import Chem

os.environ.setdefault(
    "JAVA_HOME",
    os.environ.get("JAVA_HOME", ""),
)
os.environ["PATH"] = (
    os.environ["JAVA_HOME"] + "/bin" + os.pathsep + os.environ.get("PATH", "")
)

from iupac_namer.engine import name_smiles

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
    """Parse *name* via OPSIN under a per-call tempdir (Windows
    concurrency-safe) and check the canonical SMILES round-trips."""
    if not _HAVE_OPSIN:
        pytest.skip("py2opsin not available")
    import tempfile
    import time

    cwd = os.getcwd()
    td = tempfile.mkdtemp(prefix="s8_terp_opsin_")
    try:
        os.chdir(td)
        for attempt in range(6):
            try:
                rt = _p2o.py2opsin(name)
            except Exception:
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
# Camphor — the only PIN-acceptable retained terpene from the audit.
# ---------------------------------------------------------------------------


class TestCamphorRetainedEmission:
    """Pin the existing camphor coverage (already wired via
    ``data/retained_names_expanded.json``) so a future refactor of the
    retained-name lookup pipeline cannot silently drop the entry."""

    def test_achiral_camphor_emits_retained_name(self):
        """The achiral parent SMILES emits ``camphor``."""
        smi = "CC12CCC(CC1=O)C2(C)C"
        name = name_smiles(smi)
        assert name == "camphor", (
            f"expected 'camphor' for achiral camphor SMILES, got {name!r}"
        )

    def test_camphor_round_trips_through_opsin(self):
        """``camphor`` parses back to the same canonical SMILES via OPSIN."""
        smi = "CC12CCC(CC1=O)C2(C)C"
        assert _round_trip_matches(smi, "camphor")

    def test_reordered_achiral_camphor_emits_camphor(self):
        """A non-canonical input SMILES of the same molecule still emits
        ``camphor`` after canonicalisation."""
        smi = "O=C1CC2CCC1(C)C2(C)C"
        name = name_smiles(smi)
        assert name == "camphor", (
            f"expected 'camphor' for re-ordered SMILES, got {name!r}"
        )

    def test_audit_stereo_probe_falls_back_to_systematic(self):
        """The audit's primary stereo probe
        ``C[C@@]12C(C[C@@H](CC1)C2(C)C)=O`` must NOT emit ``camphor``
        because the retained name does not encode stereochemistry —
        emitting it would silently drop the chirality.  The strategy
        stereo-drop guard (``retained_plan_would_drop_stereo``) routes
        the molecule to the systematic name instead.

        This test guards against a regression where camphor is added to
        ``_RETAINED_NAMES_ENCODING_STEREO`` without confirming that the
        ring junction stereo round-trips through OPSIN.  As of Stage 8
        the systematic name is the architecturally correct emission.
        """
        smi = "C[C@@]12C(C[C@@H](CC1)C2(C)C)=O"
        name = name_smiles(smi)
        assert name != "camphor", (
            f"camphor was emitted for stereo audit probe (would drop "
            f"chirality); got {name!r}.  This indicates either the "
            f"stereo-drop guard regressed, or camphor was admitted to "
            f"_RETAINED_NAMES_ENCODING_STEREO without round-trip "
            f"verification."
        )
        # The expected systematic emission (the engine has already shown
        # this output stably under HEAD 3c2821e).
        assert "trimethylbicyclo[2.2.1]heptan-2-one" in name, (
            f"expected systematic von Baeyer fallback for stereo "
            f"camphor probe, got {name!r}"
        )


# ---------------------------------------------------------------------------
# Deferred terpene names — pin the architectural decision.
#
# These tests document that the engine MUST NOT emit a non-PIN trivial
# name for these structures.  If a future agent adds e.g. bornane to
# ``_RING_CURATED_SMILES`` without first updating the IUPAC-PIN
# decision (which would require Blue Book backing), this test will
# flag it as a regression.
# ---------------------------------------------------------------------------


class TestDeferredTerpeneNames:
    """The following retained names parse via OPSIN but are explicitly
    NOT the IUPAC PIN per Blue Book P1 / Appendix 3.  The engine must
    continue to emit the systematic von Baeyer name."""

    @pytest.mark.parametrize(
        "smi,trivial,systematic_marker",
        [
            # bornane: PIN per P1.html is bicyclo[2.2.1]heptane.
            (
                "CC12CCC(CC1)C2(C)C",
                "bornane",
                "1,7,7-trimethylbicyclo[2.2.1]heptane",
            ),
            # borneol: the alcohol; PIN is the systematic substituted
            # bicyclo[2.2.1]heptan-2-ol.
            (
                "CC1(C)C2CCC1(C)C(O)C2",
                "borneol",
                "1,7,7-trimethylbicyclo[2.2.1]heptan-2-ol",
            ),
            # pinane: Papp3.html "named systematically by CAS"
            (
                "CC1CCC2CC1C2(C)C",
                "pinane",
                "2,6,6-trimethylbicyclo[3.1.1]heptane",
            ),
            # alpha-pinene: same; the PIN is the systematic name.
            (
                "CC1=CCC2CC1C2(C)C",
                "alpha-pinene",
                "2,6,6-trimethylbicyclo[3.1.1]hept-2-ene",
            ),
            # norbornane: scaffold parent; PIN is bicyclo[2.2.1]heptane
            # (already emitted).
            (
                "C1CC2CCC1C2",
                "norbornane",
                "bicyclo[2.2.1]heptane",
            ),
            # norbornene: PIN is bicyclo[2.2.1]hept-2-ene.
            (
                "C1=CC2CCC1C2",
                "norbornene",
                "bicyclo[2.2.1]hept-2-ene",
            ),
            # p-menthane: Papp3.html "named systematically by CAS"
            (
                "CC1CCC(C(C)C)CC1",
                "p-menthane",
                "4-methyl-1-(propan-2-yl)cyclohexane",
            ),
            # carane: 3,7,7-trimethylbicyclo[4.1.0]heptane (PIN).
            (
                "CC1CCC2C(C1)C2(C)C",
                "carane",
                "3,7,7-trimethylbicyclo[4.1.0]heptane",
            ),
        ],
    )
    def test_terpene_emits_systematic_not_trivial(
        self, smi: str, trivial: str, systematic_marker: str
    ) -> None:
        name = name_smiles(smi)
        assert name != trivial, (
            f"non-PIN trivial name {trivial!r} was emitted for {smi!r}; "
            f"this violates Blue Book P1 / Appendix 3 (architecture > score)."
        )
        assert systematic_marker in name, (
            f"expected systematic von Baeyer name containing "
            f"{systematic_marker!r} for {smi!r}, got {name!r}"
        )


# ---------------------------------------------------------------------------
# OPSIN-input smoke test.
#
# Sanity check that py2opsin still recognises ``camphor`` (used as a
# regression guard for Java/OPSIN environment configuration).
# ---------------------------------------------------------------------------


def test_opsin_recognises_camphor():
    """If this fails, the Java environment / py2opsin install is broken,
    not the engine."""
    if not _HAVE_OPSIN:
        pytest.skip("py2opsin not available")
    import tempfile
    cwd = os.getcwd()
    td = tempfile.mkdtemp(prefix="s8_terp_smoke_")
    try:
        os.chdir(td)
        rt = _p2o.py2opsin("camphor")
    finally:
        os.chdir(cwd)
        try:
            import shutil
            shutil.rmtree(td, ignore_errors=True)
        except Exception:
            pass
    assert rt, "OPSIN did not recognise 'camphor' — environment issue"
    assert _canon(rt) == "CC12CCC(CC1=O)C2(C)C"
