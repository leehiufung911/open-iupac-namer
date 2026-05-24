"""FR-5.2 / FR-5.3 (P-25.3.3) peripheral-numbering tests.

These pin the behaviour of iupac_namer.ring_naming.fr_orientation and the
mono-hetero fused-naming path it feeds: a SUBSTITUTED furo/thieno/pyrrolo-fused
heteroaromatic must receive substituent locants consistent with the emitted
fusion name (i.e. it round-trips through OPSIN to the same structure).

Before this module the engine numbered the fused parent 1..N with the generic
peripheral walk, producing impossible locants like ``9-chlorofuro[2,3-b]pyridine``
(position 9 does not exist).  The cases below all round-tripped WRONG (or to an
out-of-range locant) prior to the FR-5.3 numbering.
"""
import os

os.environ.setdefault(
    "JAVA_HOME",
    os.environ.get("JAVA_HOME", ""),
)
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin" + os.pathsep + os.environ.get("PATH", "")

import pytest
from rdkit import Chem

from iupac_namer.engine import name_smiles
from iupac_namer.ring_naming.fr_orientation import (
    compute_peripheral_numbering,
    compute_peripheral_numberings,
)

from tests.audit._audit_helpers import assert_round_trip


# (canonical-ish SMILES, human label) — every entry round-trips through OPSIN to
# the same structure with the correct FR-5.3 substituent locant.
SUBSTITUTED_ROUND_TRIP = [
    # furo[2,3-b]pyridine, chloro at every peripheral C-H (OPSIN locants 2..6)
    ("Clc1cc2cccnc2o1", "2-chlorofuro[2,3-b]pyridine"),
    ("Clc1coc2ncccc12", "3-chlorofuro[2,3-b]pyridine"),
    ("Clc1ccnc2occc12", "4-chlorofuro[2,3-b]pyridine"),
    ("Clc1cnc2occc2c1", "5-chlorofuro[2,3-b]pyridine"),
    ("Clc1ccc2ccoc2n1", "6-chlorofuro[2,3-b]pyridine"),
    # thieno[2,3-d]pyrimidine
    ("Clc1ncc2ccsc2n1", "4-chlorothieno[2,3-d]pyrimidine"),
    # furo[3,4-c]pyridine family (the eval-target parent)
    ("Cc1ncc2c(c1O)C(=O)OC2Cl",
     "3-chloro-7-hydroxy-6-methyl-1,3-dihydrofuro[3,4-c]pyridin-1-one"),
    # ---- FR-5.2 2D-orientation tie-break (P-25.3.3.1.2 (f)) ----
    # Near-symmetric fused mirrors that tie on every FR-5.3 atom-level criterion
    # and are resolved only by the FR-5.2 preferred orientation (indicated H).
    # Before the tie-break these round-tripped to the WRONG mirror locant for the
    # asymmetric (pyrrolo) parents; the symmetric (furo/thieno) parents are
    # automorphism-equivalent and round-trip under either choice.
    #   pyrrolo[3,4-b]pyrazine — the asymmetric NH parent, all peripheral C-H
    ("ClC1=CN=C2C(N1)=CN=C2", "2-chloropyrrolo[3,4-b]pyrazine"),
    ("ClC=1N=C2C(NC1)=CN=C2", "3-chloropyrrolo[3,4-b]pyrazine"),
    ("ClC1=NC=C2NC=CN=C21", "5-chloropyrrolo[3,4-b]pyrazine"),
    ("ClN1C=C2N=CC=NC2=C1", "6-chloropyrrolo[3,4-b]pyrazine"),
    ("CC1=NC=C2NC=CN=C21", "5-methylpyrrolo[3,4-b]pyrazine"),
    ("ClN1C=2C(=NCC1)C=NC2", "1-chloro-2,3-dihydro-1H-pyrrolo[3,4-b]pyrazine"),
    #   pyrrolo[3,4-c]pyridazine — asymmetric NH parent
    ("ClN1N=C2C(C=C1)=CN=C2", "2-chloropyrrolo[3,4-c]pyridazine"),
    #   furo / thieno [3,4-b]pyrazine & [3,4-d]pyridazine — symmetric mirrors
    ("ClC1=NN=CC=2C1=COC2", "1-chlorofuro[3,4-d]pyridazine"),
    ("ClC=1C=2C(C=NN1)=COC2", "4-chlorofuro[3,4-d]pyridazine"),
    ("ClC=1OC=C2N=CC=NC21", "5-chlorofuro[3,4-b]pyrazine"),
    ("ClC=1OC=C2C1N=CC=N2", "7-chlorofuro[3,4-b]pyrazine"),
    ("ClC=1SC=C2N=CC=NC21", "5-chlorothieno[3,4-b]pyrazine"),
    ("ClC1=NN=CC=2C1=CSC2", "1-chlorothieno[3,4-d]pyridazine"),
    ("ClC=1OC=C2N=NC=CC21", "5-chlorofuro[3,4-c]pyridazine"),
]


@pytest.mark.parametrize("smiles,label", SUBSTITUTED_ROUND_TRIP,
                         ids=[lbl for _, lbl in SUBSTITUTED_ROUND_TRIP])
def test_substituted_fused_round_trip(smiles, label):
    assert_round_trip(smiles)


def _atom_to_label(smiles: str) -> dict[int, str]:
    mol = Chem.MolFromSmiles(smiles)
    nb = compute_peripheral_numbering(frozenset(range(mol.GetNumAtoms())), mol)
    assert nb is not None, f"no peripheral numbering for {smiles!r}"
    return {a: l.label for a, l in nb._assignments}


def test_furo_2_3_b_pyridine_locants_have_junction_suffixes():
    """furo[2,3-b]pyridine: O=1, furan C at 2,3, junctions 3a/7a, pyridine
    C at 4,5,6, N=7 — junction atoms must carry the ``Na`` suffix, never plain
    integers like 8/9."""
    a2l = _atom_to_label("c1cnc2occc2c1")  # bare furo[2,3-b]pyridine
    labels = set(a2l.values())
    assert "3a" in labels and "7a" in labels
    # No locant above 7 (the old generic walk produced 8/9).
    assert all(
        int("".join(ch for ch in v if ch.isdigit())) <= 7 for v in labels
    )


def test_oxygen_takes_lowest_locant():
    """The furan O is the most senior heteroatom and takes locant 1."""
    a2l = _atom_to_label("c1cnc2occc2c1")
    mol = Chem.MolFromSmiles("c1cnc2occc2c1")
    o_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "O"]
    assert a2l[o_atoms[0]] == "1"


def test_near_symmetric_skeleton_resolved_by_indicated_hydrogen():
    """A near-symmetric skeleton (pyrrolo[3,4-b]pyrazine) used to yield two tied
    mirror numberings (the bare-skeleton FR-5.3 rules tie on every atom-level
    criterion).  The FR-5.2 2D-orientation tie-break — implemented as
    P-25.3.3.1.2 (f), low locants to indicated hydrogen — now resolves it to a
    single numbering in which the indicated-hydrogen N (the ``NH``) takes the
    lower of the two mirror locants, matching OPSIN's preferred orientation.
    """
    mol = Chem.MolFromSmiles("c1c[nH]c2cncc-2n1")  # pyrrolo[3,4-b]pyrazine
    ai = frozenset(range(mol.GetNumAtoms()))
    nbs = compute_peripheral_numberings(ai, mol)
    # Resolved to a single numbering (was 2 before the FR-5.2 tie-break).
    assert len(nbs) == 1
    a2l = {a: l.label for a, l in nbs[0]._assignments}

    # Heteroatom locant set is unchanged by the tie-break: {1, 4, 6}.
    het = sorted(
        int("".join(c for c in a2l[a] if c.isdigit()))
        for a in range(mol.GetNumAtoms())
        if mol.GetAtomWithIdx(a).GetSymbol() == "N"
    )
    assert het == [1, 4, 6]

    # The indicated-hydrogen nitrogen (the one carrying the ``NH``) is given the
    # LOWER of its two possible mirror locants — verify the chosen numbering's
    # indicated-H locant is <= the mirror's.
    nh = next(
        a for a in ai
        if mol.GetAtomWithIdx(a).GetSymbol() == "N"
        and mol.GetAtomWithIdx(a).GetTotalNumHs() > 0
    )
    chosen_nh_locant = int("".join(c for c in a2l[nh] if c.isdigit()))
    # The two mirror locants for any heteroatom here are {1, 6} (1 + 6 - 1 = 6).
    # The tie-break must select the numbering giving the NH the lower locant.
    assert chosen_nh_locant == min(het)  # NH lands on locant 1, not its mirror 6


def test_symmetric_furo_skeleton_returns_single_representative():
    """furo[3,4-b]pyrazine has a true molecular mirror symmetry: the two mirror
    numberings are automorphism-equivalent (no indicated hydrogen to break the
    tie).  The module returns a single deterministic representative — either is
    correct because any substituent receives the same locant under both."""
    mol = Chem.MolFromSmiles("c1cnc2c(n1)coc2")  # furo[3,4-b]pyrazine
    nbs = compute_peripheral_numberings(frozenset(range(mol.GetNumAtoms())), mol)
    assert len(nbs) == 1


def test_monocyclic_returns_empty():
    """A monocyclic ring is not this module's job."""
    mol = Chem.MolFromSmiles("c1ccncc1")  # pyridine
    assert compute_peripheral_numberings(
        frozenset(range(mol.GetNumAtoms())), mol
    ) == ()
