"""
iupac_namer/ring_naming/common.py

Shared utilities for the ring naming package.

Provides:
- extract_ring_mol:  carve ring atoms into an isolated RDKit Mol
- get_ring_smiles:   canonical SMILES for the ring submolecule
- AROMATIC_CARBOCYCLE_NAMES: stem lookup for all-carbon aromatic rings
- SATURATED_CARBOCYCLE_NAMES: stem lookup for saturated monocarbocycles
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rdkit import Chem

from iupac_namer.data_loader import get_chain_stem

if TYPE_CHECKING:
    from iupac_namer.types import RingSystem

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ring submolecule extraction
# ---------------------------------------------------------------------------

def _normalize_nh_fragment(
    frag_smiles: str,
    *,
    allow_nh_insert: bool = True,
) -> Chem.Mol | None:
    """Try to produce a valid Mol from a ring fragment SMILES that may have
    aromatic N atoms without explicit H (because a substituent was stripped).

    When ``MolFragmentToSmiles`` carves off a ring from an N-substituted
    molecule it produces bare aromatic 'n' atoms that RDKit cannot kekulize
    (e.g. 'c1cncn1' for an N-methyl imidazole ring).  This function retries
    by inserting [nH] at each bare-n position until a valid molecule is found.

    Also handles the reverse case: a ring fragment that already contains [nH]
    (from a keto/amino tautomer of an aromatic ring, e.g. cytosine's pyrimidine
    ring) may fail to kekulize because the explicit H makes the ring
    non-aromatic in isolation.  In that case we try stripping each [nH] -> n
    to recover the fully aromatic parent SMILES (e.g. 'c1cnc[nH]c1' → 'c1cncnc1'
    = pyrimidine).

    ``allow_nh_insert`` (default True) controls Strategy 1 (insert [nH] at a
    bare-n position).  Set to False when the caller knows the parent ring has
    no NH on any extracted atom — inserting a phantom NH then would corrupt
    locant lookup against curated tables that key the [nH] position to a
    specific IUPAC locant (e.g. all-N-substituted purines like xanthines).

    Returns a sanitized RDKit Mol or None.
    """
    # Aromatic ring anion/cation (theophyllin-7-ide, imidazolate, pyrrolate,
    # N-methylpyridinium, 1,3-dialkyl-1H-imidazol-3-ium, ...): when the
    # fragment contains [n-] or [n+], rewrite for scaffold lookup.  The
    # ring's IUPAC parent identity is independent of whether a given N is
    # protonated, deprotonated, or methylated-to-cation — they all occupy
    # the same lone-pair slot in the aromatic π-system.  The charge is
    # tracked separately on the full molecule and emitted as a `-N-ide` /
    # `-N-ium` suffix at assembly time (see engine.py ring_anion_locants /
    # ring_cation_locants).
    #
    # Two substitution variants try to recover the neutral parent:
    #   (a) [n-]/[n+] → [nH]  — works for 5-ring azoles (pyrrole, imidazole,
    #       triazole, tetrazole, purine-core) where the parent tautomer has
    #       an [nH] at the charged slot.
    #   (b) [n-]/[n+] → n (no H)  — works for 6-ring azines (pyridine,
    #       pyrimidine, pyridazine, pyrazine, quinoline, isoquinoline,
    #       purine-non-NH-slot) where the parent has bare aromatic N and
    #       the charge is carried by a protonation/N-alkylation outside the
    #       π-bookkeeping.
    # Even when the raw frag_smiles parses (e.g. ``c1cc[n+]cc1`` for
    # pyridinium), we still prefer the neutralized scaffold so retained-name
    # lookup matches the parent-ring key (e.g. ``c1ccncc1`` = pyridine).
    if "[n-]" in frag_smiles or "[n+]" in frag_smiles:
        neut_with_h = frag_smiles.replace("[n-]", "[nH]").replace("[n+]", "[nH]")
        m = Chem.MolFromSmiles(neut_with_h)
        if m is not None:
            return m
        neut_bare = frag_smiles.replace("[n-]", "n").replace("[n+]", "n")
        m = Chem.MolFromSmiles(neut_bare)
        if m is not None:
            return m

    # Stage 21 R21-A: Restore implicit-H on aromatic C atoms whose
    # heavy substituent was carved out, so the ring's canonical SMILES
    # matches the unsubstituted parent's curated entry.
    #
    # Background: ``MolFragmentToSmiles`` produces ``c1c[c]ccc1`` (with
    # explicit ``[c]`` for the formerly-substituted atom) when the parent
    # SMILES has the substituted ring atom in bracketed form (e.g.
    # ``[BiH2][c]1ccccc1`` — phenylbismuthane).  RDKit canonicalises
    # ``c1c[c]ccc1`` to ``[c]1ccccc1`` (preserving the noImplicit
    # property on the bracketed atom), which doesn't match the curated
    # benzene entry ``c1ccccc1``; the engine then falls through to the
    # systematic ``cyclohexane`` branch, silently dropping aromaticity.
    #
    # Phenylsilane (``[SiH3]c1ccccc1``) doesn't trigger this because
    # RDKit canonicalises the Si-attached aromatic C as ``c`` (no
    # brackets).  Why the heavier elements (Bi/Pb/As) leave the bracket
    # form on the substituted ring C is an RDKit canonicalisation
    # idiosyncrasy.
    #
    # The simplest fix is a string-level rewrite: ``[c]`` → ``c`` and
    # ``[C]`` → ``C`` in the fragment SMILES before parse.  An aromatic
    # C with an unfilled bond (the carved free valence) needs an
    # implicit H to satisfy aromatic valence; the bracket form pinned
    # the H count to 0, so removing the brackets lets RDKit re-add it.
    if "[c]" in frag_smiles or "[C]" in frag_smiles:
        rewritten = frag_smiles.replace("[c]", "c").replace("[C]", "C")
        mol = Chem.MolFromSmiles(rewritten)
        if mol is not None:
            return mol

    mol = Chem.MolFromSmiles(frag_smiles)
    if mol is not None:
        return mol

    # Locate all standalone lowercase 'n' characters (skip bracket atoms)
    bare_n_positions: list[int] = []
    nh_spans: list[tuple[int, int]] = []  # (start, end) of each [nH] token
    i = 0
    while i < len(frag_smiles):
        c = frag_smiles[i]
        if c == "[":
            try:
                j = frag_smiles.index("]", i)
                token = frag_smiles[i : j + 1]
                if token.lower() in ("[nh]", "[nh+]"):
                    nh_spans.append((i, j + 1))
                i = j + 1
            except ValueError:
                i += 1
        elif c == "n":
            bare_n_positions.append(i)
            i += 1
        else:
            i += 1

    # Strategy 1: insert [nH] at each bare-n position.
    # Skipped when caller knows the parent has no NH (e.g. fully N-substituted
    # purines / xanthines): a phantom NH would corrupt curated atom_locants
    # lookup which is keyed on the H position.
    if allow_nh_insert:
        for pos in bare_n_positions:
            candidate = frag_smiles[:pos] + "[nH]" + frag_smiles[pos + 1:]
            mol = Chem.MolFromSmiles(candidate)
            if mol is not None:
                return mol

    # Strategy 2: strip [nH] -> n at each existing [nH] position, or strip ALL.
    # Useful for rings like cytosine whose fragment is 'c1cnc[nH]c1'
    # (the [nH] makes the 6-ring non-aromatic; removing it gives 'c1cncnc1'
    # = pyrimidine, which is the parent ring system for retention lookup).
    # Also handles thymine-type fragments with two [nH] groups: 'c1c[nH]c[nH]c1'
    # -> strip all -> 'c1cncnc1' = pyrimidine.

    # First, try stripping ALL [nH] at once (handles ≥2 [nH] groups efficiently).
    if nh_spans:
        stripped_all = frag_smiles.replace("[nH]", "n").replace("[NH]", "N")
        mol = Chem.MolFromSmiles(stripped_all)
        if mol is not None:
            return mol

    # Then try stripping one [nH] at a time (may succeed when stripping all does not).
    for start, end in nh_spans:
        candidate = frag_smiles[:start] + "n" + frag_smiles[end:]
        mol = Chem.MolFromSmiles(candidate)
        if mol is not None:
            return mol

    # Strategy 3: for fragments with bare 'o' (aromatic O, e.g. lactone ring extracted
    # without its exo =O, like coumarin → 'c1ccc2occcc2c1'), try partial sanitization
    # (skip kekulization).  This produces a SMILES-comparable ring mol even though
    # RDKit won't do full Kekulé assignment.
    if 'o' in frag_smiles and '[o' not in frag_smiles.lower():
        try:
            m = Chem.MolFromSmiles(frag_smiles, sanitize=False)
            if m is not None:
                Chem.SanitizeMol(
                    m,
                    Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
                )
                return m
        except Exception:
            pass

    # Strategy 4: partial sanitization for fully-N-substituted aromatic rings.
    # When the parent has no NH on the ring atoms (e.g. all-N-substituted purine
    # in xanthines like caffeine / linagliptin), the fragment SMILES has bare
    # aromatic n's that won't kekulize.  Skip kekulization to produce a stable
    # canonical SMILES (e.g. 'c1ncc2ncnc2n1' for the bare aromatic purine
    # skeleton) that the curated lookup can target with a dedicated entry.
    # Only triggered when [nH] insertion is disabled — otherwise Strategy 1
    # already handled the case.
    if not allow_nh_insert and bare_n_positions:
        try:
            m = Chem.MolFromSmiles(frag_smiles, sanitize=False)
            if m is not None:
                Chem.SanitizeMol(
                    m,
                    Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
                )
                return m
        except Exception:
            pass

    return None


def _neutralize_ring_charged_n(ring_mol: Chem.Mol) -> Chem.Mol | None:
    """If ``ring_mol`` contains an aromatic [n-] or [n+], return a neutralized
    copy where each such N becomes a neutral aromatic N with one explicit H
    ([nH]).  Returns None if ``ring_mol`` has no charged aromatic N.

    Rationale: the IUPAC parent ring identity (purine, imidazole, pyridine,
    pyrrole, triazole, ...) is independent of whether a given ring N is
    neutral, deprotonated, or N-alkylated-to-cation — they all occupy the
    same lone-pair slot and share the same curated scaffold.  The charge is
    rendered via a separate ``-N-ide`` / ``-N-ium`` suffix at assembly time
    (P-72.2 / P-73 / P-66.6), not via the curated-key canonical SMILES.
    Neutralizing here lets the scaffold lookup match the existing [nH]
    tautomer entry (e.g. theophyllin-7-ide's ring matches 9H-purine;
    1,3-dimethyl-1H-imidazol-3-ium's ring matches 1H-imidazole).

    For [n+] the original atom already carries the substituent whose H is
    being added here as explicit — that's intentional: the curated scaffold
    (e.g. ``c1c[nH]cn1`` for imidazole) expects ONE [nH] per aromatic N
    slot.  ``_build_ring_mol_preserving_tautomer`` already handles picking
    the right indicated-H tautomer when multiple N's could carry the H;
    this helper is a fallback for the string-based extraction path that
    produces a ring fragment with bare charged atoms.
    """
    # Two passes: (a) neutralize + explicit [nH] (for 5-ring azoles where
    # the parent has [nH] at this slot), then (b) if sanitization fails,
    # neutralize without adding H (for 6-ring azines where the parent
    # aromatic N is bare).
    def _try(with_h: bool) -> Chem.Mol | None:
        rw = Chem.RWMol(ring_mol)
        has_charged = False
        for ai in range(rw.GetNumAtoms()):
            a = rw.GetAtomWithIdx(ai)
            if (a.GetAtomicNum() == 7 and a.GetIsAromatic()
                    and a.GetFormalCharge() in (-1, 1)):
                a.SetFormalCharge(0)
                if with_h:
                    a.SetNumExplicitHs(1)
                    a.SetNoImplicit(True)
                else:
                    a.SetNumExplicitHs(0)
                    a.SetNoImplicit(False)
                has_charged = True
        if not has_charged:
            return None
        candidate = rw.GetMol()
        try:
            Chem.SanitizeMol(candidate)
            return candidate
        except Exception:
            return None

    # Prefer the with-[nH] form (5-ring azoles) — curated keys for pyrrole /
    # imidazole / triazole / pyrazole / tetrazole / indole / purine all
    # encode an [nH].  Fall back to the bare-n form for 6-ring azines
    # (pyridine, pyrimidine, quinoline, ...) where the parent scaffold has
    # no indicated-H on any ring atom.
    result = _try(with_h=True)
    if result is not None:
        return result
    return _try(with_h=False)


# Backwards-compatibility alias for the pre-generalization name.  External
# callers (if any) may still import _neutralize_ring_anion; the extended
# helper handles both [n-] and [n+].
_neutralize_ring_anion = _neutralize_ring_charged_n


def extract_ring_mol(ring_system: "RingSystem", mol) -> Chem.Mol | None:
    """Extract ring atoms as an isolated RDKit molecule.

    Uses RDKit's MolFragmentToSmiles + MolFromSmiles to produce a fresh
    molecule with only the ring atoms (no surrounding substituents).

    For aromatic N-containing rings where the N is substituted (no H in the
    parent molecule), the fragment SMILES has bare 'n' atoms that cannot be
    kekulized.  In that case we try inserting [nH] at each such position to
    recover the parent ring's canonical form (e.g. 'c1cncn1' → 'c1c[nH]cn1'
    for 1H-imidazole).

    Tautomer preservation: for rings where indicated-H location depends on
    which ring N is substituted (e.g. 1H vs 4H-1,2,4-triazole — FDA-0804
    maraviroc), we FIRST try building the ring directly, putting an explicit
    H on each ring N that carried a substituent (or H) in the parent.  This
    preserves the actual tautomer identity so the curated lookup matches the
    right entry (1H- vs 4H-).  Falls back to the string-based ``_normalize_nh_fragment``
    path if direct construction fails.

    Ring charged-N neutralization: if the extracted ring_mol contains an
    aromatic [n-] or [n+] (deprotonated ring N or N-alkylated ring cation,
    e.g. theophyllin-7-ide, N-methylpyridinium, 1,3-dialkyl-1H-imidazol-3-ium),
    neutralize it to [nH] for scaffold lookup — the charge is tracked
    separately on the full molecule and emitted as a `-N-ide` / `-N-ium`
    suffix at assembly time.

    Returns None on failure.
    """
    atom_indices = sorted(ring_system.atom_indices)
    if not atom_indices:
        return None

    # Tautomer-preserving direct build.
    direct = _build_ring_mol_preserving_tautomer(mol, atom_indices)
    if direct is not None:
        return direct

    allow_nh_insert = _should_allow_nh_insert(ring_system, mol, atom_indices)
    try:
        frag_smiles = Chem.MolFragmentToSmiles(mol, atom_indices)
        if not frag_smiles:
            return None
        frag_mol = _normalize_nh_fragment(
            frag_smiles, allow_nh_insert=allow_nh_insert
        )
        if frag_mol is not None:
            # Neutralize any aromatic [n-] for scaffold lookup (see docstring).
            neutralized = _neutralize_ring_charged_n(frag_mol)
            if neutralized is not None:
                return neutralized
        return frag_mol
    except Exception:
        logger.debug("extract_ring_mol failed for %s", ring_system)
        return None


def _build_ring_mol_preserving_tautomer(mol, atom_indices: list[int]) -> Chem.Mol | None:
    """Build a ring-only RDKit Mol that preserves the tautomer identity.

    For every ring atom that is a nitrogen which (a) already carries an H in
    the parent, or (b) has a non-ring neighbor (i.e. is N-substituted in the
    parent), mark the carved ring atom with an explicit H so that after
    sanitization the canonical SMILES pins the correct tautomer.

    Rationale: ``MolFragmentToSmiles`` on an N-substituted aromatic ring drops
    the substituent and produces bare 'n' atoms that cannot be kekulized; the
    string-based [nH] insertion in ``_normalize_nh_fragment`` just picks the
    FIRST kekulizable position, which for triazoles/tetrazoles can silently
    flip the tautomer (1H ↔ 4H for 1,2,4-triazole).  Preserving the H on the
    substituted-N position avoids that flip.

    Returns None on sanitization failure so the caller can fall back.
    """
    ring_set = set(atom_indices)
    # Identify ring N's that are "substituted" (non-ring neighbor) or already [nH].
    # An aromatic [n-] or [n+] is treated as an NH target for the ring
    # scaffold — it occupies the same lone-pair slot as [nH] and the charge
    # is emitted via a separate -N-ide / -N-ium suffix at assembly time.
    # For [n+] with an external substituent (e.g. N-methylpyridinium's
    # C[n+]1ccccc1), the substituent gets stripped during the ring carve
    # and the atom becomes the neutral [nH] indicated-H position in the
    # parent scaffold.  Likewise a bare [n-] (e.g. pyrrolate's [n-]1cccc1)
    # maps to the indicated-H position of 1H-pyrrole.
    nh_targets: set[int] = set()
    for i in atom_indices:
        a = mol.GetAtomWithIdx(i)
        if a.GetAtomicNum() != 7:
            continue
        if a.GetFormalCharge() in (-1, 1) and a.GetTotalNumHs() == 0:
            nh_targets.add(i)
            continue
        # [NH+]/[NH-] with one or more H's AND an external substituent:
        # treat the same as the no-H charged case — the parent scaffold
        # carries one indicated-H per ring N slot, the protonation H is
        # tracked separately via the -N-ium / -N-ide suffix at assembly time.
        # Without this, atracurium-style N-alkyl-N-protonated ring cations
        # like C[NH+]1CCc2ccccc2C1 fall through to the default carve path
        # which produces a fragment SMILES with [NH2+] (the lost-substituent
        # H plus the protonation H concatenated by sanitization), and the
        # curated-ring lookup misses the parent THIQ scaffold.
        if a.GetFormalCharge() in (-1, 1) and a.GetTotalNumHs() > 0:
            has_ext_chg = any(
                nb.GetIdx() not in ring_set for nb in a.GetNeighbors()
            )
            if has_ext_chg:
                nh_targets.add(i)
                continue
        if a.GetFormalCharge() != 0:
            # Any other non-zero charge (e.g. [N--]): let the default path handle.
            continue
        if a.GetTotalNumHs() > 0:
            nh_targets.add(i)
            continue
        has_ext = any(nb.GetIdx() not in ring_set for nb in a.GetNeighbors())
        if has_ext:
            nh_targets.add(i)

    if not nh_targets:
        return None  # nothing to preserve; let the default path run.

    rw = Chem.RWMol(mol)
    # Remove non-ring atoms in reverse index order so remaining indices stay stable.
    to_remove = sorted([i for i in range(mol.GetNumAtoms()) if i not in ring_set], reverse=True)
    # Build old->new mapping for ring atoms (indices compact after removals).
    old_to_new: dict[int, int] = {}
    new_idx = 0
    for old in range(mol.GetNumAtoms()):
        if old in ring_set:
            old_to_new[old] = new_idx
            new_idx += 1
    for old in to_remove:
        rw.RemoveAtom(old)
    # Strip stereochemistry, explicit-H counts, and isotope info from ring atoms
    # so the carved fragment canonicalizes identically to the reference parent
    # ring (curated keys are stereo-free and use implicit H counts).
    for ai in range(rw.GetNumAtoms()):
        atom = rw.GetAtomWithIdx(ai)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(False)
        atom.SetIsotope(0)
        atom.SetAtomMapNum(0)
    for bi in range(rw.GetNumBonds()):
        bond = rw.GetBondWithIdx(bi)
        bond.SetStereo(Chem.BondStereo.STEREONONE)
    # Mark each target N with one explicit H (replacing the stripped
    # substituent).  For [n-] / [n+] targets also neutralize the formal
    # charge in the ring mol so the scaffold canonicalizes identically to
    # the neutral parent ring (curated keys are uncharged).  The charge
    # locant is tracked separately on the full molecule and emitted as an
    # ``-N-ide`` / ``-N-ium`` suffix at assembly time.
    for old in nh_targets:
        ni = old_to_new[old]
        atom = rw.GetAtomWithIdx(ni)
        atom.SetNumExplicitHs(1)
        atom.SetNoImplicit(True)
        if atom.GetFormalCharge() != 0:
            atom.SetFormalCharge(0)
    result = rw.GetMol()
    try:
        Chem.SanitizeMol(result)
    except Exception:
        return None
    return result


def _should_allow_nh_insert(
    ring_system: "RingSystem", mol, atom_indices: list[int]
) -> bool:
    """Decide whether to allow phantom [nH] insertion during fragment normalization.

    [nH] insertion (Strategy 1 of _normalize_nh_fragment) is necessary for
    monocyclic N-heterocycles whose curated table keys carry an [nH]
    (imidazole, pyrazole, tetrazole, etc.) — there is exactly one curated
    canonical and the substruct match against the parent is robust to the
    H position.

    It is HARMFUL for fused multi-N rings whose curated table has multiple
    tautomer keys with different atom_locants mappings (purine: 1H/3H share
    one mapping, 7H/9H share another).  When ALL ring N's are substituted in
    the parent (xanthines, all-N-substituted purines), arbitrarily inserting
    [nH] produces a key whose atom_locants are wrong for the actual
    substitution pattern.  In that case we disable [nH] insertion and let
    Strategy 4 (partial sanitize) produce the bare aromatic skeleton, which
    has its own dedicated curated entry.

    Heuristic: disable only for fused rings with ≥3 ring N's AND no NH on
    any ring atom in the parent.  Benzimidazole (2 N's), benzazoles, and
    indazoles have a single curated tautomer key, so [nH] insertion is safe;
    purines / xanthines (4 N's) have multiple tautomer keys with different
    atom_locants, so phantom NH insertion corrupts locant assignment.

    An aromatic ring [n-] or [n+] counts as an NH for this decision — it
    occupies the same tautomer slot as [nH] (same lone-pair contribution
    to the π system, same IUPAC locant) and the charge is rendered via a
    separate ``-N-ide`` / ``-N-ium`` suffix, not by the curated-key
    tautomer mapping.  Examples: theophyllin-7-ide ([n-] slot), a fused
    ring cation like 3-methylimidazo[4,5-b]pyridin-3-ium ([n+] slot).
    """
    if ring_system.type != "fused":
        return True
    n_count = 0
    for idx in atom_indices:
        a = mol.GetAtomWithIdx(idx)
        if a.GetAtomicNum() == 7:
            n_count += 1
            if a.GetTotalNumHs() > 0:
                return True  # at least one NH present — [nH] form is real
            if a.GetFormalCharge() in (-1, 1):
                return True  # [n-]/[n+] occupies an NH-equivalent tautomer slot
    if n_count < 3:
        return True  # too few N's for tautomer-locant ambiguity (e.g. benzimidazole)
    return False


def extract_ring_mol_with_exo_oxo(
    ring_system: "RingSystem", mol
) -> tuple[Chem.Mol, tuple[int, ...]] | None:
    """Extract ring atoms PLUS exocyclic =O on sp2 ring carbons (and, for
    cyclic sulfones/sulfoxides, =O on a ring sulfur).

    Many retained-name lookup keys for partially-saturated heterocyclic ketones
    / lactams (e.g. ``O=C1CCc2ccccc2N1`` for 3,4-dihydroquinolin-2(1H)-one)
    encode the carbonyl =O in the canonical SMILES key.  The default
    ``extract_ring_mol`` strips exocyclic atoms, producing keys that miss
    those entries (the substituent path's ring-only carve never matches the
    retained table).  This helper rebuilds the carbonyl-included form so the
    lookup can match.

    Also handles cyclic sulfones / sulfoxides (e.g. sulfolene =
    ``O=S1(=O)CC=CC1``) where the retained key embeds the SO / SO2 group on
    a ring S atom.  Ring S with one or two exocyclic =O neighbors is treated
    the same way as ring C with =O.

    Returns ``(ring_mol, exo_oxo_full_mol_indices)`` on success — the full-mol
    indices of the extra =O atoms are returned so the caller can mark them as
    claimed by the retained name.  Returns ``None`` if no exocyclic =O was
    found OR on failure (the caller should fall back to the default
    ``extract_ring_mol`` path).
    """
    atom_indices = sorted(ring_system.atom_indices)
    if not atom_indices:
        return None
    ring_set = set(atom_indices)

    # Only include =O on ring sulfur when the ring system is monocyclic
    # (sulfolene, sulfolane, thiirene-dioxide, ...).  For FUSED/BRIDGED ring
    # systems, pulling SO2 into the retained-name key would defeat the
    # penam lookup (key ``O=C1CC2SCCN12`` matches ``penam`` only when the
    # SO2 is NOT embedded in the key; the ``1,1-dioxopenam`` form is emitted
    # by a separate sulfone-dioxide decoration path).
    accept_ring_sulfur_oxo = (getattr(ring_system, "type", None) == "monocyclic")

    extra: list[int] = []
    for ring_idx in atom_indices:
        atom = mol.GetAtomWithIdx(ring_idx)
        # Accept ring carbons (ketone / lactam C=O) always, and ring sulfur
        # atoms (SO / SO2) only for monocyclic rings where the retained name
        # embeds the SO/SO2 group in the canonical key (e.g. sulfolene).
        atomic_num = atom.GetAtomicNum()
        if atomic_num == 6:
            pass
        elif atomic_num == 16 and accept_ring_sulfur_oxo:
            pass
        else:
            continue
        for nb in atom.GetNeighbors():
            if nb.GetIdx() in ring_set:
                continue
            if nb.GetAtomicNum() != 8:
                continue
            if nb.IsInRing():
                continue
            if nb.GetFormalCharge() != 0:
                continue
            bond = mol.GetBondBetweenAtoms(ring_idx, nb.GetIdx())
            if bond is None:
                continue
            if bond.GetBondTypeAsDouble() != 2.0:
                continue
            extra.append(nb.GetIdx())

    if not extra:
        return None  # nothing to add — caller's default extract is sufficient

    extended_indices = atom_indices + extra
    allow_nh_insert = _should_allow_nh_insert(ring_system, mol, atom_indices)
    try:
        frag_smiles = Chem.MolFragmentToSmiles(mol, extended_indices)
        if not frag_smiles:
            return None
        frag_mol = _normalize_nh_fragment(
            frag_smiles, allow_nh_insert=allow_nh_insert
        )
        if frag_mol is None:
            return None
        # Neutralize any aromatic [n-] for scaffold lookup (see extract_ring_mol).
        neutralized = _neutralize_ring_anion(frag_mol)
        if neutralized is not None:
            frag_mol = neutralized
        return frag_mol, tuple(extra)
    except Exception:
        logger.debug("extract_ring_mol_with_exo_oxo failed for %s", ring_system)
        return None


def extract_ring_mol_stripping_exo_oxo(
    ring_system: "RingSystem", mol
) -> "tuple[Chem.Mol, tuple[int, ...], tuple[int, ...]] | None":
    """Extract ring atoms with the exocyclic =O on sp2 ring carbons
    STRIPPED and the former C=O carbons converted to saturated CH.

    Used by the retained-name lookup to recover the "indicated-H" parent
    form of an in-ring ketone (e.g. ``O=c1ccocc1`` → ``C1=COC=CC1`` aka
    4H-pyran).  When the curated table holds the base retained parent
    (4H-pyran, 4H-thiopyran, ...) but not the decorated ketone form, the
    retained lookup can match the stripped canonical, verify that the
    indicated-H locant aligns with the former-C=O position, and emit the
    retained parent name — leaving the =O to be picked up by downstream
    ketone-FG detection as a ``-<N>-one`` suffix.

    Returns ``(stripped_ring_mol, exo_oxo_full_mol_indices,
    former_co_full_mol_indices)`` on success.  The former-C=O indices are
    returned in the SAME order as ``exo_oxo_full_mol_indices`` so callers
    can pair them up when checking locant alignment.

    Returns ``None`` if no exocyclic =O was found, or on sanitization
    failure.  Behaves conservatively: if the ring cannot be rebuilt as a
    valid neutral molecule after stripping, we return None and let the
    caller fall back to other lookup strategies.
    """
    atom_indices = sorted(ring_system.atom_indices)
    if not atom_indices:
        return None
    ring_set = set(atom_indices)

    # Collect (ring_C_idx, exo_O_idx) pairs.
    pairs: list[tuple[int, int]] = []
    for ring_idx in atom_indices:
        atom = mol.GetAtomWithIdx(ring_idx)
        if atom.GetAtomicNum() != 6:
            continue
        for nb in atom.GetNeighbors():
            if nb.GetIdx() in ring_set:
                continue
            if nb.GetAtomicNum() != 8:
                continue
            if nb.IsInRing():
                continue
            if nb.GetFormalCharge() != 0:
                continue
            bond = mol.GetBondBetweenAtoms(ring_idx, nb.GetIdx())
            if bond is None or bond.GetBondTypeAsDouble() != 2.0:
                continue
            pairs.append((ring_idx, nb.GetIdx()))

    if not pairs:
        return None

    # Strategy: copy the parent molecule into an RWMol, then delete every atom
    # that is NOT a ring atom AND NOT an exo-=O atom we intend to strip.  This
    # preserves atom-index identity for ring atoms (via an old→new map) so we
    # can track which ring carbons were the former C=O carriers.  Kekulize,
    # delete the exo-=O atoms, sanitize, and return.
    former_co_indices = [p[0] for p in pairs]
    exo_oxo_indices = [p[1] for p in pairs]

    keep_set = ring_set | set(exo_oxo_indices)

    rw = Chem.RWMol(mol)
    # Kekulize (clears aromatic bond flags) before we start deleting atoms.
    try:
        Chem.Kekulize(rw, clearAromaticFlags=True)
    except Exception:
        logger.debug("extract_ring_mol_stripping_exo_oxo: kekulize failed for %s",
                     ring_system)
        return None

    # Delete non-kept atoms in REVERSE index order so kept indices stay stable.
    # Build an old→new index map.
    n_total = rw.GetNumAtoms()
    old_to_new: dict[int, int] = {}
    new_idx = 0
    for old in range(n_total):
        if old in keep_set:
            old_to_new[old] = new_idx
            new_idx += 1
    for old in range(n_total - 1, -1, -1):
        if old not in keep_set:
            rw.RemoveAtom(old)

    # Now delete the exo-=O atoms from the reduced mol, in reverse index order.
    exo_new_indices = [old_to_new[o] for o in exo_oxo_indices]
    for pos in sorted(exo_new_indices, reverse=True):
        rw.RemoveAtom(pos)

    # Reset aromaticity / explicit-H flags on the remaining atoms and let
    # sanitization re-perceive the kekulized structure.
    for ai in range(rw.GetNumAtoms()):
        a = rw.GetAtomWithIdx(ai)
        a.SetIsAromatic(False)
        a.SetNumExplicitHs(0)
        a.SetNoImplicit(False)
        a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for bi in range(rw.GetNumBonds()):
        b = rw.GetBondWithIdx(bi)
        b.SetIsAromatic(False)
        b.SetStereo(Chem.BondStereo.STEREONONE)

    stripped = rw.GetMol()
    try:
        Chem.SanitizeMol(stripped)
    except Exception:
        logger.debug("extract_ring_mol_stripping_exo_oxo: sanitize failed for %s",
                     ring_system)
        return None

    return stripped, tuple(exo_oxo_indices), tuple(former_co_indices)


def get_ring_canonical_smiles(ring_system: "RingSystem", mol) -> str | None:
    """Return the canonical SMILES for a ring system, or None on failure."""
    ring_mol = extract_ring_mol(ring_system, mol)
    if ring_mol is None:
        return None
    try:
        return Chem.MolToSmiles(ring_mol)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Chain stem helpers (re-exported for convenience)
# ---------------------------------------------------------------------------

def chain_stem(n: int) -> str | None:
    """Return the chain stem for n carbons (e.g. 6 -> 'hex')."""
    return get_chain_stem(n)


# ---------------------------------------------------------------------------
# HeteroPosition element list from RingSystem
# ---------------------------------------------------------------------------

def heteroatom_elements(ring_system: "RingSystem") -> list[str]:
    """Return the list of heteroatom elements in ring order."""
    if not ring_system.heteroatoms:
        return []
    return [hp.element for hp in ring_system.heteroatoms]
