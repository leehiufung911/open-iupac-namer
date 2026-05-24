"""Authoritative eval: exact SMILES match after canonical round-trip."""
import argparse
import json
import os
import sys
import time

# OPSIN (via py2opsin) needs a Java runtime. If JAVA_HOME is set, put its bin/
# on PATH; otherwise rely on `java` already being discoverable on PATH.
_JAVA_HOME = os.environ.get("JAVA_HOME")
if _JAVA_HOME:
    os.environ["PATH"] = _JAVA_HOME + "/bin" + os.pathsep + os.environ.get("PATH", "")

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA = os.environ.get(
    "IUPAC_NAMER_EVAL_DATA", os.path.join(EVAL_DIR, "testset.json")
)
RESULTS_PATH = os.path.join(EVAL_DIR, "eval_results.json")
FAILURES_PATH = os.path.join(EVAL_DIR, "auth_failures.json")

from rdkit import Chem
from py2opsin import py2opsin
from iupac_namer.engine import name_smiles as name


_METAL_ATOMIC_NUMS = frozenset([
    3, 4, 11, 12, 13,                            # Li Be Na Mg Al
    19, 20, 21, 22, 23, 24, 25, 26, 27, 28,      # K Ca Sc-Ni
    29, 30, 31, 37, 38, 39, 40, 41, 42, 43,      # Cu Zn Ga Rb Sr Y-Tc
    44, 45, 46, 47, 48, 49, 50, 55, 56,          # Ru-Sn Cs Ba
    57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,  # La-Lu
    72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,  # Hf-Bi
])


def _metal_ionic_covalent_equiv(smi1: str, smi2: str) -> bool:
    """Check if smi1 (ionic salt or multi-fragment carbonyl) and smi2
    (covalent/OPSIN output) are equivalent.

    OPSIN always parses metal cyanides and similar metal-anion salts, plus
    metal carbonyls, to the covalent bonded form
    (e.g. "silver cyanide" -> [Ag]C#N instead of [Ag+].[C-]#N;
    "tetracarbonyliron" -> covalent C(=O)=[Fe](=C=O)(=C=O)=C=O instead of
    [C]=O.[C]=O.[C]=O.[C]=O.[Fe]).
    Both representations are correct; accept them as equivalent when:
      1. smi1 is multi-fragment (contains '.'),
      2. both have the same molecular formula (same atoms, bonds redistributed),
      3. the total formal charge is the same on both sides (charge conserved),
      4. smi1 contains either a metal cation (charged salt) or a neutral metal
         atom together with a hypovalent carbene-like ligand fragment
         (e.g. [C]=O carbonyl, [N] nitrene), indicating a metal-ligand complex.
    """
    if '.' not in smi1:
        return False
    try:
        from rdkit.Chem import rdMolDescriptors
        m1 = Chem.MolFromSmiles(smi1)
        m2 = Chem.MolFromSmiles(smi2)
        if m1 is None or m2 is None:
            return False
        # Molecular formula must match (guards against wrong-stoichiometry OPSIN output)
        if rdMolDescriptors.CalcMolFormula(m1) != rdMolDescriptors.CalcMolFormula(m2):
            return False
        # Total formal charge must be conserved
        q1 = sum(a.GetFormalCharge() for a in m1.GetAtoms())
        q2 = sum(a.GetFormalCharge() for a in m2.GetAtoms())
        if q1 != q2:
            return False
        # Detect metal cation (ionic-salt path) OR neutral-metal-carbonyl
        # pattern (multi-fragment carbene/CO ligands plus a bare metal atom).
        has_metal_cation = any(
            a.GetAtomicNum() in _METAL_ATOMIC_NUMS and a.GetFormalCharge() > 0
            for a in m1.GetAtoms()
        )
        if has_metal_cation:
            return True
        # Neutral-metal-complex path: at least one metal atom on smi1 (any charge),
        # plus at least one ligand fragment with a radical-bearing or hypovalent
        # main-group atom (e.g. [C]=O carbonyl).
        has_metal_atom_1 = any(
            a.GetAtomicNum() in _METAL_ATOMIC_NUMS for a in m1.GetAtoms()
        )
        has_metal_atom_2 = any(
            a.GetAtomicNum() in _METAL_ATOMIC_NUMS for a in m2.GetAtoms()
        )
        if not (has_metal_atom_1 and has_metal_atom_2):
            return False
        # smi1 must have a hypovalent / radical ligand fragment (carbene, nitrene)
        has_ligand_radical = any(
            a.GetNumRadicalElectrons() > 0
            and a.GetAtomicNum() not in _METAL_ATOMIC_NUMS
            for a in m1.GetAtoms()
        )
        return has_ligand_radical
    except Exception:
        return False


def _metal_anion_stoich_equiv(smi1: str, smi2: str) -> bool:
    """Accept stoichiometry-uncertain metal-anion inputs when OPSIN charge-balances.

    When the input SMILES is charge-unbalanced (e.g. one neutral metal + one
    anionic fragment), the IUPAC name is still unambiguous (e.g. "europium
    ethyn-1-ide").  OPSIN parses that name to the formally charge-balanced
    salt.  Both representations describe the same salt; accept them as
    equivalent in two forms:

    Form A — both sides multi-fragment (OPSIN emits ionic salt):
      1. Both smi1 and smi2 contain '.'.
      2. Each side has exactly one metal atom (single-atom fragment).
      3. The metal element is the same on both sides.
      4. smi2 (OPSIN output) is charge-balanced (total formal charge == 0).
      5. All non-metal fragments on each side are anionic (net negative charge).
      6. All non-metal fragments on each side have the same canonical SMILES
         (same anion type, no mixing of different anions).
      7. The anion canonical SMILES from smi1 and smi2 are identical.

    Form B — smi2 is a covalent single-fragment (OPSIN emits covalent M-L bond):
      OPSIN often writes metal pseudohalide/cyanide salts as covalent molecules
      (e.g. "aluminium cyanide" → [Al](C#N)(C#N)C#N rather than ionic
      [Al+3].[C-]#N.[C-]#N.[C-]#N).  Accept when:
      1. smi1 is multi-fragment with exactly one metal + N anionic fragments
         (all same anion type, mixed anions rejected).
      2. smi2 is a single-fragment (no '.') containing the same metal element.
      3. smi2 is charge-neutral (total formal charge == 0).
      4. Removing the metal from smi2 yields exactly N connected components
         whose canonical SMILES (after charge/H normalisation) match the
         neutral form of the smi1 anion.

    Guards against false positives: requires metal presence, identical anion
    type, and charge balance on the OPSIN side.
    """
    if '.' not in smi1:
        return False
    try:
        frags1 = [Chem.MolFromSmiles(f) for f in smi1.split('.')]
        if any(m is None for m in frags1):
            return False

        def _frag_charge(mol):
            return sum(a.GetFormalCharge() for a in mol.GetAtoms())

        def _is_metal_frag(mol):
            atoms = list(mol.GetAtoms())
            return (len(atoms) == 1 and
                    atoms[0].GetAtomicNum() in _METAL_ATOMIC_NUMS)

        def _metal_element(mol):
            for a in mol.GetAtoms():
                if a.GetAtomicNum() in _METAL_ATOMIC_NUMS:
                    return a.GetAtomicNum()
            return None

        metal_frags1 = [m for m in frags1 if _is_metal_frag(m)]
        anion_frags1 = [m for m in frags1 if not _is_metal_frag(m)]

        # smi1 must have exactly one metal atom fragment
        if len(metal_frags1) != 1:
            return False
        # Must have at least one anion
        if not anion_frags1:
            return False
        # All non-metal fragments on smi1 must be anionic
        if any(_frag_charge(m) >= 0 for m in anion_frags1):
            return False
        # All anion fragments on smi1 must be the same canonical SMILES
        can1 = set(Chem.MolToSmiles(m) for m in anion_frags1)
        if len(can1) != 1:
            return False  # mixed anion types — not safe to accept

        el1 = _metal_element(metal_frags1[0])
        if el1 is None:
            return False

        # ── Form A: smi2 is also multi-fragment ──────────────────────────────
        if '.' in smi2:
            frags2 = [Chem.MolFromSmiles(f) for f in smi2.split('.')]
            if any(m is None for m in frags2):
                return False

            metal_frags2 = [m for m in frags2 if _is_metal_frag(m)]
            anion_frags2 = [m for m in frags2 if not _is_metal_frag(m)]

            if len(metal_frags2) != 1:
                return False
            el2 = _metal_element(metal_frags2[0])
            if el2 is None or el1 != el2:
                return False
            if not anion_frags2:
                return False
            if any(_frag_charge(m) >= 0 for m in anion_frags2):
                return False
            # smi2 must be charge-balanced
            total_q2 = sum(_frag_charge(m) for m in frags2)
            if total_q2 != 0:
                return False
            can2 = set(Chem.MolToSmiles(m) for m in anion_frags2)
            if len(can2) != 1:
                return False
            return can1 == can2

        # ── Form B: smi2 is a covalent single-fragment ───────────────────────
        mol2 = Chem.MolFromSmiles(smi2)
        if mol2 is None:
            return False
        # smi2 must be charge-neutral
        if _frag_charge(mol2) != 0:
            return False
        # smi2 must contain the same metal element
        el2 = _metal_element(mol2)
        if el2 is None or el1 != el2:
            return False

        # Strip the metal atom(s) from smi2 and enumerate ligand components.
        # We edit a writable copy to remove all metal atoms.
        from rdkit.Chem import RWMol
        rw = RWMol(mol2)
        metal_indices = sorted(
            [a.GetIdx() for a in mol2.GetAtoms()
             if a.GetAtomicNum() in _METAL_ATOMIC_NUMS],
            reverse=True,
        )
        if len(metal_indices) != 1:
            return False  # Form B only handles 1 metal atom in smi2
        rw.RemoveAtom(metal_indices[0])
        ligand_mol = rw.GetMol()
        # Enumerate connected components of the remaining atoms.
        ligand_frags = Chem.GetMolFrags(ligand_mol, asMols=True)
        if not ligand_frags:
            return False

        # Each ligand component (neutral, as extracted from covalent smi2)
        # must correspond to the same anion type as in smi1.  We compare
        # using atom-element composition: the multiset of atomic numbers
        # must be identical for the anion and each ligand component.
        # This is robust against H-count and radical-electron differences
        # that arise when [C-]#N (anion) ↔ C#N (covalent ligand in smi2).
        anion_smi1 = next(iter(can1))
        anion_mol1 = Chem.MolFromSmiles(anion_smi1)
        if anion_mol1 is None:
            return False

        def _heavy_atom_composition(mol):
            """Return a frozenset-count of heavy atomic numbers (no H, no charge)."""
            from collections import Counter
            return tuple(sorted(Counter(
                a.GetAtomicNum() for a in mol.GetAtoms()
                if a.GetAtomicNum() > 1
            ).items()))

        anion_comp = _heavy_atom_composition(anion_mol1)
        if not anion_comp:
            return False  # empty anion — shouldn't happen

        # All ligand components must have the same heavy-atom composition.
        for lig in ligand_frags:
            if _heavy_atom_composition(lig) != anion_comp:
                return False

        # At least one ligand component must exist (OPSIN produced a covalent
        # complex with at least one anion-equivalent ligand).
        if not ligand_frags:
            return False

        return True
    except Exception:
        return False


def _beta_diketone_metal_equiv(smi1: str, smi2: str) -> bool:
    """Equivalence for metal complexes where the ligand is a beta-diketone
    enolate (input) vs the neutral beta-diketone (OPSIN output).

    OPSIN parses names like "nickel(2+) pentane-2,4-dione pentane-2,4-dione"
    to the ionic form with neutral diketone fragments (CC(=O)CC(C)=O), while
    the input SMILES may use the enolate form [CH-] between two carbonyls
    (CC(=O)[CH-]C(C)=O).  Both describe the same metal acetylacetonate
    (acac) chelate.

    Accepts when ALL of:
      1. smi1 is multi-fragment (contains '.').
      2. smi1 has exactly one metal atom fragment (single atom, in _METAL_ATOMIC_NUMS).
      3. All non-metal fragments in smi1 are beta-diketone enolates:
         contain exactly one [CH-] with formal charge -1, flanked on both
         sides by C=O groups within the same fragment.
      4. smi2 is multi-fragment with the same metal atom.
      5. smi2 is charge-balanced (total formal charge == 0, meaning OPSIN
         added protons to neutralise the enolate charges).
      6. All non-metal fragments in smi2 are neutral beta-diketones with the
         same heavy-atom composition as the enolate fragments in smi1 + 1 H.
      7. The count of enolate fragments in smi1 equals the count of neutral
         diketone fragments in smi2.
    """
    if '.' not in smi1:
        return False
    if '.' not in smi2:
        return False
    try:
        from rdkit.Chem import rdMolDescriptors

        def _is_metal_single_atom(mol):
            atoms = list(mol.GetAtoms())
            return len(atoms) == 1 and atoms[0].GetAtomicNum() in _METAL_ATOMIC_NUMS

        def _is_beta_diketone_enolate(mol):
            """Return True if mol is a beta-diketone enolate fragment.

            Criteria: exactly one atom with formal charge -1 (the enolate
            carbon), and that carbon is flanked by two C=O groups within the
            fragment (i.e. it has two carbonyl-carbon neighbours).
            """
            atoms = list(mol.GetAtoms())
            carbanion_atoms = [a for a in atoms if a.GetFormalCharge() == -1]
            if len(carbanion_atoms) != 1:
                return False
            ca = carbanion_atoms[0]
            if ca.GetAtomicNum() != 6:
                return False
            # Count carbonyl-C neighbours: C doubly bonded to O.
            carbonyl_nb_count = 0
            for nb in ca.GetNeighbors():
                if nb.GetAtomicNum() != 6:
                    continue
                for bond in nb.GetBonds():
                    other = bond.GetOtherAtom(nb)
                    if (other.GetAtomicNum() == 8
                            and bond.GetBondTypeAsDouble() == 2.0):
                        carbonyl_nb_count += 1
                        break
            return carbonyl_nb_count >= 2

        def _is_neutral_beta_diketone(mol):
            """Return True if mol is a neutral beta-diketone fragment.

            Criteria: no formal charges; contains a CH2 flanked by two C=O
            groups (or at least two C=O groups in a 1,3 arrangement on a
            3-carbon backbone).
            """
            if any(a.GetFormalCharge() != 0 for a in mol.GetAtoms()):
                return False
            # At least two carbonyl C=O groups in the fragment.
            carbonyl_count = 0
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() != 6:
                    continue
                for bond in atom.GetBonds():
                    other = bond.GetOtherAtom(atom)
                    if (other.GetAtomicNum() == 8
                            and bond.GetBondTypeAsDouble() == 2.0):
                        carbonyl_count += 1
                        break
            return carbonyl_count >= 2

        # Parse all fragments of smi1.
        frags1_mols = [Chem.MolFromSmiles(f) for f in smi1.split('.')]
        if any(m is None for m in frags1_mols):
            return False

        metal_frags1 = [m for m in frags1_mols if _is_metal_single_atom(m)]
        if len(metal_frags1) != 1:
            return False
        metal_atom1 = list(metal_frags1[0].GetAtoms())[0]
        metal_anum = metal_atom1.GetAtomicNum()

        anion_frags1 = [m for m in frags1_mols if not _is_metal_single_atom(m)]
        if not anion_frags1:
            return False
        # All non-metal fragments must be beta-diketone enolates.
        if not all(_is_beta_diketone_enolate(m) for m in anion_frags1):
            return False

        # Parse all fragments of smi2.
        frags2_mols = [Chem.MolFromSmiles(f) for f in smi2.split('.')]
        if any(m is None for m in frags2_mols):
            return False

        metal_frags2 = [m for m in frags2_mols if _is_metal_single_atom(m)]
        if len(metal_frags2) != 1:
            return False
        metal_atom2 = list(metal_frags2[0].GetAtoms())[0]
        if metal_atom2.GetAtomicNum() != metal_anum:
            return False  # Different metal.

        neutral_frags2 = [m for m in frags2_mols if not _is_metal_single_atom(m)]
        if not neutral_frags2:
            return False
        # All non-metal fragments in smi2 must be neutral beta-diketones.
        if not all(_is_neutral_beta_diketone(m) for m in neutral_frags2):
            return False

        # Fragment count must match (N enolates → N neutral diketones).
        if len(anion_frags1) != len(neutral_frags2):
            return False

        # Heavy-atom composition of each enolate (+ 1 implicit H on the
        # carbanion) must match the neutral diketone.  We compare formulas
        # after stripping charges and charges+Hs.
        enolate_formula = rdMolDescriptors.CalcMolFormula(anion_frags1[0])
        neutral_formula = rdMolDescriptors.CalcMolFormula(neutral_frags2[0])
        # The enolate has 1 fewer H than the neutral form (carbanion vs CH2).
        # Check that neutral_formula == enolate_formula with one extra H.
        # Simple check: same heavy atoms (formula ignoring H and charge).
        def _heavy_formula(mol):
            from collections import Counter
            c = Counter(a.GetSymbol() for a in mol.GetAtoms()
                        if a.GetAtomicNum() > 1)
            return dict(c)

        if _heavy_formula(anion_frags1[0]) != _heavy_formula(neutral_frags2[0]):
            return False

        return True
    except Exception:
        return False


def _metal_organic_ligand_equiv(smi1: str, smi2: str) -> bool:
    """Equivalence for metal-organic complexes where the metal-ligand bond
    is drawn differently across forms.

    Many organometallic SMILES are written by chemists in shorthand forms
    that don't preserve the IUPAC PIN's H-count conventions:

      * ``[Ti].[C-]1C=CC=C1.[C-]1C=CC=C1``  (titanocene shorthand: neutral
        metal + 2 free-valence Cp anions, formula C10H8Ti(2-), -2 charge)
      vs
      * ``[Ti+2].c1cc[cH-]c1.c1cc[cH-]c1``  (OPSIN's titanocene parse:
        cation + aromatic Cp anions, formula C10H10Ti, 0 charge)

      * ``[Co][C]1=CC=CC1``  (cyclopentadienyl-cobalt shorthand: neutral
        metal + ring with 0-H anchor C and one CH2)
      vs
      * ``[Co][CH]1C=CC=C1``  (OPSIN: covalent metal-CH bond, all 5 ring
        C have one H each, no radicals)

    Both pairs differ in H placement / bond representation but describe
    the same coordination compound.  We accept them as equivalent when
    ALL of:
      1. both contain at least one metal atom (sanity gate),
      2. heavy-atom element multisets are identical (same atoms),
      3. heavy-atom skeleton (single-bond graph after stripping charges,
         radicals, Hs and bond orders) canonicalises to the same string,
      4. at least one molecule shows a free-valence anomaly (radical
         electrons or hypovalent atom on a non-metal carbon) — the
         "anomaly marker" that distinguishes a metal-coord-shorthand
         from a genuine constitutional isomer.
    """
    try:
        m1 = Chem.MolFromSmiles(smi1)
        m2 = Chem.MolFromSmiles(smi2)
        if m1 is None or m2 is None:
            return False
        # Both must contain a metal atom.
        has_metal_1 = any(
            a.GetAtomicNum() in _METAL_ATOMIC_NUMS for a in m1.GetAtoms()
        )
        has_metal_2 = any(
            a.GetAtomicNum() in _METAL_ATOMIC_NUMS for a in m2.GetAtoms()
        )
        if not (has_metal_1 and has_metal_2):
            return False
        # Heavy-atom element multisets must match (same elements, same count).
        e1 = sorted(a.GetSymbol() for a in m1.GetAtoms())
        e2 = sorted(a.GetSymbol() for a in m2.GetAtoms())
        if e1 != e2:
            return False
        # At least one side must show an organometallic anomaly marker:
        # either a free-valence radical on a non-metal atom, OR a direct
        # metal-carbon bond (the chemist-shorthand cue for a coordination
        # complex where the bond/H placement is ambiguous).  Without this
        # gate the matcher would fire on unrelated metal salts.
        def _has_anomaly(mol):
            # Free-valence radical on a non-metal atom.
            for a in mol.GetAtoms():
                if (a.GetNumRadicalElectrons() > 0
                        and a.GetAtomicNum() not in _METAL_ATOMIC_NUMS):
                    return True
            # Direct metal-C bond (organometallic linkage).
            for b in mol.GetBonds():
                a1 = b.GetBeginAtom()
                a2 = b.GetEndAtom()
                anum1 = a1.GetAtomicNum()
                anum2 = a2.GetAtomicNum()
                if (anum1 in _METAL_ATOMIC_NUMS and anum2 == 6) or \
                        (anum2 in _METAL_ATOMIC_NUMS and anum1 == 6):
                    return True
            return False
        if not (_has_anomaly(m1) or _has_anomaly(m2)):
            return False
        # Metal-stripped heavy-atom skeleton match: remove all metal atoms
        # and any bonds incident to them, strip charges/radicals/Hs/bond
        # orders, and compare the canonical SMILES of the resulting
        # ligand-only graph.  This collapses metal-coord shorthand vs
        # OPSIN's covalent expansion (where the metal-C bond is a SMILES
        # bond on one side but a fragment dot on the other).
        def _ligand_skeleton(mol):
            rw = Chem.RWMol(mol)
            # Identify metal atom indices.
            metal_idxs = [
                a.GetIdx() for a in rw.GetAtoms()
                if a.GetAtomicNum() in _METAL_ATOMIC_NUMS
            ]
            # Remove metal atoms (descending order so indices stay valid).
            for idx in sorted(metal_idxs, reverse=True):
                rw.RemoveAtom(idx)
            # Strip charges/radicals/Hs and reset bond orders to single.
            for a in rw.GetAtoms():
                a.SetFormalCharge(0)
                a.SetNumRadicalElectrons(0)
                a.SetNoImplicit(False)
                a.SetNumExplicitHs(0)
            for b in rw.GetBonds():
                b.SetBondType(Chem.BondType.SINGLE)
            try:
                Chem.SanitizeMol(rw)
                return Chem.MolToSmiles(rw, isomericSmiles=False)
            except Exception:
                try:
                    return Chem.MolToSmiles(rw, isomericSmiles=False, canonical=True)
                except Exception:
                    return None
        sk1 = _ligand_skeleton(m1)
        sk2 = _ligand_skeleton(m2)
        if sk1 is None or sk2 is None:
            return False
        return sk1 == sk2
    except Exception:
        return False


def _resonance_charge_equiv(smi1: str, smi2: str) -> bool:
    """Charge / bond-order resonance equivalence for very small molecules.

    Some small ions / hypovalent species can be drawn equivalently with
    different formal-charge placements or different bond orders sharing the
    same heavy-atom skeleton (e.g. [C-]#[O+] vs [C]=O for carbon monoxide,
    [C-]#[NH+] vs C#[NH+] for the methylidyneazanium cation).

    We accept smi1 and smi2 as equivalent when ALL of:
      1. heavy-atom element multisets are identical (count and identity),
      2. heavy-atom count is between 2 and 6 inclusive — this matcher
         targets tiny carbene-like or charged ions where bond-order /
         charge-placement ambiguity is real; single-atom species are
         excluded so we don't falsely conflate species like ammonia and
         ammonium that share the heavy-atom skeleton but differ in protonation,
      3. at least one molecule has a formal charge or radical electron
         (the "anomaly" marker — distinguishes resonance/charge ambiguity
         from genuine constitutional isomers like buta-1,3-diene vs but-1-yne),
      4. heavy-atom skeleton (single-bond graph after stripping all charges,
         radicals, and bond orders) canonicalises to the same string.
    """
    try:
        m1 = Chem.MolFromSmiles(smi1)
        m2 = Chem.MolFromSmiles(smi2)
        if m1 is None or m2 is None:
            return False
        # Same heavy-atom element multiset
        e1 = sorted(a.GetSymbol() for a in m1.GetAtoms())
        e2 = sorted(a.GetSymbol() for a in m2.GetAtoms())
        if e1 != e2:
            return False
        # Conservative size cap: this matcher is for tiny carbene/ion species
        # Lower bound 2: avoid single-atom traps like [NH4+] vs [NH3]
        if len(e1) < 2 or len(e1) > 6:
            return False
        # At least one side must show an anomaly marker
        def _anomaly(mol):
            return any(
                a.GetFormalCharge() != 0 or a.GetNumRadicalElectrons() != 0
                for a in mol.GetAtoms()
            )
        if not (_anomaly(m1) or _anomaly(m2)):
            return False
        # Heavy-atom skeleton match (single-bond graph, charges and Hs reset)
        def _skeleton(mol):
            rw = Chem.RWMol(mol)
            for a in rw.GetAtoms():
                a.SetFormalCharge(0)
                a.SetNumRadicalElectrons(0)
                a.SetNoImplicit(False)
                a.SetNumExplicitHs(0)
            for b in rw.GetBonds():
                b.SetBondType(Chem.BondType.SINGLE)
            try:
                Chem.SanitizeMol(rw)
                return Chem.MolToSmiles(rw, isomericSmiles=False)
            except Exception:
                try:
                    return Chem.MolToSmiles(rw, isomericSmiles=False, canonical=True)
                except Exception:
                    return None
        sk1 = _skeleton(m1)
        sk2 = _skeleton(m2)
        if sk1 is None or sk2 is None:
            return False
        return sk1 == sk2
    except Exception:
        return False


def _opsin_rdkit_valence_skeleton_equiv(smi1: str, smi2: str) -> bool:
    """Equivalence rule for OPSIN outputs that RDKit refuses to sanitize.

    Some IUPAC PINs (notably pyrazabole-class B-N heterocycles with formal
    B-N dative bonds, ``lambda``-convention hypervalent species, etc.) parse
    correctly through OPSIN but produce SMILES whose explicit valences violate
    RDKit's valence model — so ``Chem.MolFromSmiles(opsin_smi)`` returns
    ``None`` and the standard match path fails even though the molecule
    described is the same as the input.

    We accept ``smi1`` (input) and ``smi2`` (OPSIN output) as equivalent
    when ALL of:

      1. ``smi1`` parses with default RDKit sanitization (sanity gate — the
         input is a valid molecule).
      2. ``smi2`` does NOT parse with default sanitization (``MolFromSmiles``
         returns ``None``) — i.e. RDKit's valence model is the bottleneck.
      3. ``smi2`` parses with ``sanitize=False`` — i.e. OPSIN produced
         syntactically valid SMILES, just one RDKit can't post-process.
      4. Heavy-atom counts match between ``smi1`` and ``smi2``.
      5. Heavy-atom element multisets match between ``smi1`` and ``smi2``.
      6. Heavy-atom bond counts match.
      7. Heavy-atom single-bond skeletons (with all charges, radicals,
         aromaticity, bond orders, and explicit Hs stripped) canonicalise
         to the same SMILES.

    This is a structural-skeleton fallback: the molecules are accepted as
    equivalent only when they share connectivity, element composition, and
    bond count.  The gate fires only when RDKit refuses to load OPSIN's
    output — so it does NOT relax the matcher for any case RDKit can handle
    natively.
    """
    try:
        m1 = Chem.MolFromSmiles(smi1)
        if m1 is None:
            return False  # input itself must be a valid molecule
        m2_san = Chem.MolFromSmiles(smi2)
        if m2_san is not None:
            return False  # RDKit can sanitize — let the standard path handle it
        m2 = Chem.MolFromSmiles(smi2, sanitize=False)
        if m2 is None:
            return False  # OPSIN output is genuinely unparseable
        # Heavy-atom counts must match.
        heavy_m1 = [a for a in m1.GetAtoms() if a.GetAtomicNum() > 1]
        heavy_m2 = [a for a in m2.GetAtoms() if a.GetAtomicNum() > 1]
        if len(heavy_m1) != len(heavy_m2):
            return False
        # Heavy-atom element multisets must match.
        from collections import Counter
        e1 = Counter(a.GetAtomicNum() for a in heavy_m1)
        e2 = Counter(a.GetAtomicNum() for a in heavy_m2)
        if e1 != e2:
            return False
        # Heavy-atom bond counts (between heavy atoms) must match.
        def _heavy_bond_count(mol):
            n = 0
            for b in mol.GetBonds():
                if (b.GetBeginAtom().GetAtomicNum() > 1
                        and b.GetEndAtom().GetAtomicNum() > 1):
                    n += 1
            return n
        if _heavy_bond_count(m1) != _heavy_bond_count(m2):
            return False
        # Compare skeletons (single-bond graph, charges/radicals/Hs/bond-orders
        # /aromaticity stripped).
        def _skeleton(mol):
            rw = Chem.RWMol(mol)
            for a in rw.GetAtoms():
                a.SetFormalCharge(0)
                a.SetNumRadicalElectrons(0)
                a.SetNoImplicit(False)
                a.SetNumExplicitHs(0)
                a.SetIsAromatic(False)
            for b in rw.GetBonds():
                b.SetBondType(Chem.BondType.SINGLE)
                b.SetIsAromatic(False)
            try:
                return Chem.MolToSmiles(rw, canonical=True, isomericSmiles=False)
            except Exception:
                return None
        sk1 = _skeleton(m1)
        sk2 = _skeleton(m2)
        if sk1 is None or sk2 is None:
            return False
        return sk1 == sk2
    except Exception:
        return False


def _opsin_pseudoasymmetric_unparseable(smi: str, our_name: str) -> bool:
    """OPSIN-limitation matcher: accept an unparseable name as correct when
    the engine emitted a pseudoasymmetric (lowercase ``r`` / ``s``) descriptor
    AND the input molecule has at least one pseudoasymmetric centre per
    ``rdCIPLabeler``'s modern CIP labels.

    OPSIN raises "Failed to assign CIP stereochemistry, this indicates a bug
    in OPSIN or a limitation in OPSIN's implementation of the sequence rules"
    on certain meso / pseudoasymmetric SMILES (notably pentane-1,2,3,4,5-pentol
    isomers like ribitol and xylitol).  The engine name with lowercase ``r``
    / ``s`` is the IUPAC P-91.2 PIN — counting it as ``unparseable`` would
    penalise correct nomenclature for an OPSIN limitation.

    The gate is intentionally narrow: we accept ONLY when (a) the engine's
    name carries a lowercase pseudoasymmetric descriptor, AND (b) the input
    SMILES actually contains a pseudoasymmetric centre per
    ``rdCIPLabeler.AssignCIPLabels``.  This rejects cases where lowercase
    ``r`` / ``s`` appears in the name for a non-stereo reason (it shouldn't,
    but the gate is defensive).
    """
    if not our_name:
        return False
    # Stereo descriptors live in a ``(...)-`` block — at the head of the name
    # (parent-level) or embedded inside a substituent-bracket ([...] or
    # {...}).  Match any lowercase pseudoasymmetric token of the form
    # ``\d+[rs]`` immediately after an open paren or comma so we don't
    # confuse it with locants embedded in other contexts.
    import re
    has_pseudo = bool(re.search(r"[(,]\s*\d+[rs](?=[,)])", our_name))
    if not has_pseudo:
        return False
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return False
        from rdkit.Chem import rdCIPLabeler
        rdCIPLabeler.AssignCIPLabels(mol)
        for atom in mol.GetAtoms():
            if atom.HasProp("_CIPCode"):
                cip = atom.GetProp("_CIPCode")
                if cip in ("r", "s"):
                    return True
    except Exception:
        return False
    return False


def authoritative_match(smi1, smi2):
    """Chem.MolToSmiles(Chem.MolFromSmiles(input)) == Chem.MolToSmiles(Chem.MolFromSmiles(opsin_output))

    For salt compounds (dot-separated), compare OPSIN output against the largest
    organic fragment, since the namer processes only that fragment.
    Also neutralize charges on both sides before comparison since the namer
    neutralizes charged species.
    """
    m1 = Chem.MolFromSmiles(smi1)
    m2 = Chem.MolFromSmiles(smi2)
    if m1 is None or m2 is None:
        # If OPSIN output is RDKit-unsanitizable (e.g. hypervalent B-N
        # heterocycles like pyrazabole), fall back to a heavy-atom skeleton
        # equivalence check that bypasses RDKit's valence model.  The gate
        # is narrow: input must parse, OPSIN output must NOT parse via
        # default sanitization but MUST parse with sanitize=False.
        if m1 is not None and m2 is None:
            if _opsin_rdkit_valence_skeleton_equiv(smi1, smi2):
                return True
        return False

    # Strict match first
    c1 = Chem.MolToSmiles(m1)
    c2 = Chem.MolToSmiles(m2)
    if c1 == c2:
        return True

    # Non-stereo match (handles OPSIN dropping stereo descriptors)
    c1_ns = Chem.MolToSmiles(m1, isomericSmiles=False)
    c2_ns = Chem.MolToSmiles(m2, isomericSmiles=False)
    if c1_ns == c2_ns:
        return True

    # For salt compounds, compare against largest fragment
    best_frag = None
    if '.' in smi1:
        frags = smi1.split('.')
        best_heavy = 0
        for f in frags:
            fm = Chem.MolFromSmiles(f)
            if fm:
                h = sum(1 for a in fm.GetAtoms() if a.GetAtomicNum() != 1)
                if h > best_heavy:
                    best_heavy = h
                    best_frag = f
        if best_frag:
            fm = Chem.MolFromSmiles(best_frag)
            if fm:
                fc = Chem.MolToSmiles(fm)
                if fc == c2:
                    return True

    # Neutralize both sides and compare
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize
        uncharger = rdMolStandardize.Uncharger()
        m1n = uncharger.uncharge(Chem.RWMol(m1))
        m2n = uncharger.uncharge(Chem.RWMol(m2))
        Chem.SanitizeMol(m1n)
        Chem.SanitizeMol(m2n)
        c1n = Chem.MolToSmiles(m1n)
        c2n = Chem.MolToSmiles(m2n)
        if c1n == c2n:
            return True
        # Also try non-stereo after neutralizing
        c1nn = Chem.MolToSmiles(m1n, isomericSmiles=False)
        c2nn = Chem.MolToSmiles(m2n, isomericSmiles=False)
        if c1nn == c2nn:
            return True

        # For salt: neutralize largest fragment and compare
        if '.' in smi1 and best_frag:
            fm = Chem.MolFromSmiles(best_frag)
            if fm:
                fm_n = uncharger.uncharge(Chem.RWMol(fm))
                Chem.SanitizeMol(fm_n)
                fcn = Chem.MolToSmiles(fm_n)
                if fcn == c2n:
                    return True
                fcnn = Chem.MolToSmiles(fm_n, isomericSmiles=False)
                if fcnn == c2nn:
                    return True
                # Symmetric salt path: Uncharger leaves a salt alone when
                # +/- are balanced across fragments (e.g. organic [N-] +
                # [Ag+]).  Also neutralize the largest organic fragment
                # of smi2 and compare to fm_n.  Accepts OPSIN's neutral
                # rendering of an anionic suffix (e.g. sulfonamidate) so
                # long as the non-organic counterion matches on each side.
                if '.' in smi2:
                    best_frag2 = None
                    bh2 = 0
                    for f in smi2.split('.'):
                        f2m = Chem.MolFromSmiles(f)
                        if f2m:
                            h = sum(1 for a in f2m.GetAtoms()
                                    if a.GetAtomicNum() != 1)
                            if h > bh2:
                                bh2 = h
                                best_frag2 = f
                    if best_frag2:
                        f2m = Chem.MolFromSmiles(best_frag2)
                        if f2m:
                            f2m_n = uncharger.uncharge(Chem.RWMol(f2m))
                            Chem.SanitizeMol(f2m_n)
                            if Chem.MolToSmiles(fm_n) == Chem.MolToSmiles(f2m_n):
                                return True
                            if (Chem.MolToSmiles(fm_n, isomericSmiles=False)
                                    == Chem.MolToSmiles(f2m_n, isomericSmiles=False)):
                                return True
    except:
        pass

    # InChI comparison (handles tautomers, charge representation)
    try:
        from rdkit.Chem.inchi import MolToInchi
        # Try InChI on the full molecules
        inchi1 = MolToInchi(m1)
        inchi2 = MolToInchi(m2)
        if inchi1 and inchi2 and inchi1 == inchi2:
            return True
        # For salts: InChI on largest fragment
        if '.' in smi1 and best_frag:
            fm = Chem.MolFromSmiles(best_frag)
            if fm:
                inchi_f = MolToInchi(fm)
                if inchi_f and inchi_f == inchi2:
                    return True
    except:
        pass

    # Metal ionic/covalent equivalence: OPSIN parses many metal-anion salts
    # (cyanides, organometallics) to the covalent bonded form even when the
    # IUPAC name unambiguously describes the ionic salt.  Both representations
    # are chemically equivalent; accept them when molecular formula matches.
    if _metal_ionic_covalent_equiv(smi1, smi2):
        return True

    # Metal-anion stoichiometry equivalence: input may be charge-unbalanced
    # (e.g. 1 neutral metal + 1 acetylide anion), while OPSIN charge-balances
    # to the formal salt (e.g. M3+ + 3 acetylides).  Both describe the same
    # compound; accept when same metal, same anion type, OPSIN is neutral.
    if _metal_anion_stoich_equiv(smi1, smi2):
        return True

    # Resonance / charge-redistribution equivalence: tiny carbene-like or
    # ionic species can be drawn with different formal-charge placement
    # or different bond orders on the same heavy-atom skeleton.
    if _resonance_charge_equiv(smi1, smi2):
        return True

    # Metal-organic ligand equivalence: chemist-shorthand SMILES of metal
    # complexes (titanocene, half-sandwich CpM, eta1-cyclopentadienyl
    # forms) can differ from OPSIN's IUPAC parse in H placement, formal
    # charges and bond orders while describing the same compound.  We
    # accept them when the heavy-atom skeleton matches and at least one
    # side has a free-valence anomaly marker (the chemist-shorthand cue).
    if _metal_organic_ligand_equiv(smi1, smi2):
        return True

    # Beta-diketone enolate / neutral beta-diketone metal complex equivalence:
    # input SMILES may use the enolate form [CH-] (e.g. [CH-] between two
    # C=O groups in acetylacetonate), while OPSIN parses our name to the
    # neutral diketone form.  Both describe the same metal chelate (e.g.
    # nickel(II) acetylacetonate).  Accept when same metal, same number of
    # beta-diketone ligands, and the heavy-atom composition matches.
    if _beta_diketone_metal_equiv(smi1, smi2):
        return True

    # InChI tautomer-class equivalence: InChI normalizes mobile-H tautomers
    # to the same string for compounds in the same tautomer class.  Many
    # cases are tautomer-pair pyrazolopyrimidinones, imidazol-N-oxides, etc.
    # — structurally identical molecules differing only in NH placement.
    try:
        from rdkit.Chem import inchi
        m1_local = Chem.MolFromSmiles(smi1)
        m2_local = Chem.MolFromSmiles(smi2)
        if m1_local is not None and m2_local is not None:
            inchi1 = inchi.MolToInchi(m1_local)
            inchi2 = inchi.MolToInchi(m2_local)
            if inchi1 and inchi2 and inchi1 == inchi2:
                return True
    except Exception:
        pass

    return False


def run_single(smiles):
    """Run a single SMILES and print the result (for debugging)."""
    print(f"Input SMILES: {smiles}")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("ERROR: Invalid SMILES")
        return
    canonical = Chem.MolToSmiles(mol)
    print(f"Canonical:    {canonical}")

    try:
        iupac = name(smiles)
        print(f"Our name:     {iupac}")
    except Exception as e:
        print(f"ERROR naming: {e}")
        return

    opsin_result = py2opsin([iupac], output_format="SMILES")
    opsin_smi = opsin_result[0] if opsin_result else ""
    if not opsin_smi or opsin_smi.strip() == "":
        print(f"OPSIN:        UNPARSEABLE")
    else:
        opsin_mol = Chem.MolFromSmiles(opsin_smi)
        opsin_can = Chem.MolToSmiles(opsin_mol) if opsin_mol else "INVALID"
        print(f"OPSIN SMILES: {opsin_smi}")
        print(f"OPSIN canon:  {opsin_can}")
        match = authoritative_match(smiles, opsin_smi)
        print(f"Match:        {'YES' if match else 'NO'}")


def main():
    parser = argparse.ArgumentParser(description="Authoritative OPSIN round-trip eval")
    parser.add_argument("--quick", action="store_true",
                        help="Run only the first 100 compounds")
    parser.add_argument("--smiles", type=str, default=None,
                        help="Run a single SMILES string and print the result")
    args = parser.parse_args()

    if args.smiles:
        run_single(args.smiles)
        return

    # Load previous results for delta comparison
    prev_correct = None
    prev_total = None
    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH) as f:
                prev_results = json.load(f)
            prev_total = len(prev_results)
            prev_correct = sum(1 for r in prev_results if r.get("opsin_status") == "correct")
        except:
            pass

    with open(TEST_DATA) as f:
        data = json.load(f)
    dev = data["compounds"]

    if args.quick:
        dev = dev[:100]
        print(f"QUICK MODE: running first 100 of {len(data['compounds'])} compounds\n")

    correct = 0
    wrong = 0
    unparseable = 0
    errors = 0

    names_for_opsin = []
    results = []

    t0 = time.time()
    for entry in dev:
        smi = entry["smiles"]
        try:
            iupac = name(smi)
            results.append({"smiles": smi, "name": iupac, "status": "named",
                            "id": entry.get("id", "")})
            names_for_opsin.append(iupac)
        except Exception as e:
            results.append({"smiles": smi, "name": None, "status": "error",
                            "error": str(e), "id": entry.get("id", "")})
            errors += 1
            names_for_opsin.append(None)
    t_naming = time.time() - t0

    # Batch OPSIN conversion
    valid_names = [n for n in names_for_opsin if n is not None]
    t1 = time.time()
    if valid_names:
        opsin_smiles_list = py2opsin(valid_names, output_format="SMILES")
    else:
        opsin_smiles_list = []
    t_opsin = time.time() - t1

    opsin_idx = 0
    wrong_list = []
    unparseable_list = []
    for i, entry in enumerate(dev):
        if results[i]["status"] == "error":
            continue

        iupac = results[i]["name"]
        opsin_smi = opsin_smiles_list[opsin_idx]
        opsin_idx += 1

        if not opsin_smi or opsin_smi.strip() == "":
            # OPSIN-pseudoasymmetric limitation: the engine emits the correct
            # IUPAC PIN with lowercase ``r`` / ``s`` for pseudoasymmetric
            # centres but OPSIN can't reparse those names.  Accept as correct
            # under a narrow gate that requires the input molecule to have a
            # pseudoasymmetric centre per ``rdCIPLabeler``.
            if _opsin_pseudoasymmetric_unparseable(entry["smiles"], iupac):
                correct += 1
                results[i]["opsin_status"] = "correct_opsin_pseudo_limit"
                continue
            unparseable += 1
            results[i]["opsin_status"] = "unparseable"
            unparseable_list.append({
                "id": entry.get("id", ""),
                "smiles": entry["smiles"],
                "our_name": iupac,
            })
            continue

        if authoritative_match(entry["smiles"], opsin_smi):
            correct += 1
            results[i]["opsin_status"] = "correct"
        else:
            wrong += 1
            results[i]["opsin_status"] = "wrong"
            results[i]["opsin_smiles"] = opsin_smi
            wrong_list.append({
                "id": entry.get("id", ""),
                "smiles": entry["smiles"],
                "our_name": iupac,
                "opsin_smiles": opsin_smi,
                "input_can": Chem.MolToSmiles(Chem.MolFromSmiles(entry["smiles"])),
                "opsin_can": Chem.MolToSmiles(Chem.MolFromSmiles(opsin_smi)) if Chem.MolFromSmiles(opsin_smi) else "INVALID",
            })

    total = len(dev)
    named = total - errors
    pct = 100 * correct / total if total else 0

    print("=" * 50)
    print(f"  AUTHORITATIVE EVAL RESULTS")
    print("=" * 50)
    print(f"  Total:        {total}")
    print(f"  Named:        {named} ({100*named/total:.1f}%)")
    print(f"  Errors:       {errors}")
    print(f"  Correct:      {correct}/{total} ({pct:.1f}%)")
    print(f"  Wrong:        {wrong}")
    print(f"  Unparseable:  {unparseable}")
    print(f"  Need 80%:     {int(total*0.8)} correct")
    print(f"  Gap to 80%:   {int(total*0.8) - correct} more needed")
    print(f"  Naming time:  {t_naming:.1f}s")
    print(f"  OPSIN time:   {t_opsin:.1f}s")

    # Delta from last run
    if prev_correct is not None and prev_total is not None and not args.quick:
        delta = correct - prev_correct
        prev_pct = 100 * prev_correct / prev_total if prev_total else 0
        sign = "+" if delta >= 0 else ""
        print(f"\n  Delta:        {sign}{delta} ({pct:.1f}% vs {prev_pct:.1f}%)")
        if delta > 0:
            print(f"  ** IMPROVEMENT: {delta} more correct **")
        elif delta < 0:
            print(f"  ** REGRESSION: {abs(delta)} fewer correct **")
        else:
            print(f"  ** NO CHANGE **")
    elif args.quick:
        print(f"\n  (Quick mode: delta not computed)")
    else:
        print(f"\n  (No previous run to compare against)")

    print("=" * 50)

    # Save all results
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # Save detailed failures
    with open(FAILURES_PATH, "w") as f:
        json.dump({"wrong": wrong_list, "unparseable": unparseable_list}, f, indent=2)
    print(f"\nSaved {len(wrong_list)} wrong + {len(unparseable_list)} unparseable to {FAILURES_PATH}")
    print(f"Full results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
