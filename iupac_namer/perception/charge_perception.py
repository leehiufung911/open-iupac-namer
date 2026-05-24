"""iupac_namer.perception.charge_perception
====================================================

Charge perception + ``-ylium`` / ``-ide`` / acylium / amidinium / diazonium
dispatcher (Stage 6 R2-B, root cause #3 in
``docs/opsin_coverage_taxonomy.md`` and Top-3 Gaps 4/7/13 in
``docs/opsin_audit_fg.md`` / ``docs/opsin_audit_hw_charge.md``).

Why this module exists
----------------------
Historically the engine's neutralizer sanitises away formal charges
*before* functional-group detection runs.  That throws away the very
clue the charge-suffix scorer would need to emit ``methylium`` /
``methanide`` / ``acetylium`` / ``ethanamidinium`` / ``ethane-1-diazonium``
and friends, so the engine ends up naming the neutralized skeleton
(``methane``, ``ethane``, ``1-oxoethane`` …) and the round-trip fails.

This module is a *perception classifier* with a tightly-scoped engine
hook.  It runs on the original (still-charged) RDKit ``Mol`` ahead of
the generic plan search and recognises a handful of well-defined
charge motifs:

* **alkanide / alkanylium** - aliphatic ``[C-]`` / ``[C+]`` carbon
  centres on a saturated chain or ring (e.g. ``[CH3+]`` -> methylium,
  ``[CH2-]CCCC`` -> pentan-1-ide, ``[CH+]1CCCCC1`` -> cyclohexan-1-ylium).
* **alkanediazonium** - acyclic ``[N+]#N`` terminal triplet bound to a
  parent carbon (``CC[N+]#N`` -> ethane-1-diazonium).
* **alkanamidinium** - the protonated-imine amidine cation
  ``R-C(=[NH2+])N`` (``CC(=[NH2+])N`` -> ethanamidinium).
* **acylium / carbonylium** - the carbon-centred acyl cation
  ``R-[C+]=O`` (``[C+](C)=O`` -> acetylium, ``C1(CCCCC1)[C+]=O`` ->
  cyclohexanecarbonylium).
* **boranuide** - ``[BH4-]``, the simplest known boron anion.

For every such motif we emit a ``LeafTree`` whose ``text`` is the
already-composed surface name; downstream assembly treats it as a
terminal string just like a retained-ring lookup.

What this module *does not* claim
---------------------------------
Charge motifs that the engine already names correctly are deliberately
left to the existing machinery:

* ring-embedded ``[N+]`` / ``[n+]`` cations (e.g. pyridinium,
  quinolinium) - handled by the substitutive ring-cation path
  (``ring_cation_locants``);
* ring-embedded aromatic ``[n-]`` anions - handled by the
  substitutive ring-anion path (``ring_anion_locants``);
* retained-name cations like ``pyrylium`` / ``phenylium`` /
  ``chromenylium`` - handled via the retained-ring SMILES table;
* monatomic salts (``[Na+]``, ``[Cl-]``) - handled in the salt path.

When ``classify_charges`` finds zero motifs it claims *nothing* and the
engine falls through to its existing dispatch.  That is the standard
non-regression contract: any compound that rendered correctly before
this module landed must still render correctly after, byte-identical.

Architectural notes
-------------------
* No module-level mutable state (caches, etc.) - all data is
  ``frozen=True`` and the dispatcher is pure-functional given the
  input mol.
* The classifier *never* mutates the input ``mol``; the per-motif
  emitters build fresh RDKit objects via ``Chem.RWMol`` for the
  neutralized parent surrogate that drives recursive naming.
* No silent atom drops: every returned ``ChargeClassification``
  enumerates the atom indices it claims, so a future extension can
  cross-check coverage against the engine's atom-claim ledger.
* Output names go through ``py2opsin`` round-trip in the test suite -
  every probe in the audit CSVs that this module closes is pinned to a
  ``classify_charges`` result and an end-to-end ``name_smiles`` ->
  OPSIN -> canonical-SMILES round-trip.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Literal

from iupac_namer.types import (
    Choice,
    DecisionContext,
    FreeValenceInfo,
    LeafTree,
    OutputForm,
)


@lru_cache(maxsize=1)
def _retained_carbocyclic_ring_cation_smiles() -> frozenset[str]:
    """Canonical SMILES of single-ring all-carbon cations that carry a
    *retained* ``-ylium`` PIN (phenylium, cyclopentadienylium).

    Built from the OPSIN-mined ring table so ``_classify_aromatic_ring_cation``
    defers to the retained-ring path for these (and any future retained
    ring-cation entries) instead of emitting the systematic
    ``cyclo<stem>-...-ylium`` form.  The cache is process-wide and read-only,
    so no session-scoped mutable state is introduced.
    """
    from rdkit import Chem
    from iupac_namer.data_loader import get_rings_from_opsin

    out: set[str] = set()
    for entry in get_rings_from_opsin():
        smi = entry.get("smiles")
        if not smi:
            continue
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        charged = [a for a in m.GetAtoms() if a.GetFormalCharge() != 0]
        if len(charged) != 1:
            continue
        c = charged[0]
        if c.GetAtomicNum() != 6 or c.GetFormalCharge() != 1:
            continue
        ri = m.GetRingInfo()
        rings = ri.AtomRings()
        if len(rings) != 1:
            continue
        ring_set = set(rings[0])
        if any(m.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in ring_set):
            continue
        if any(
            a.GetIdx() not in ring_set
            for a in m.GetAtoms()
            if a.GetAtomicNum() > 1
        ):
            continue
        out.add(Chem.MolToSmiles(m))
    return frozenset(out)

# ---------------------------------------------------------------------------
# Public data structure
# ---------------------------------------------------------------------------


SuffixHint = Literal[
    "ylium",        # carbocation / single-atom cation (alkanylium, methanide-equivalent +)
    "ide",          # carbanion / single-atom anion (alkanide)
    "uide",         # hyper-coordinated anion ([BH4-] -> boranuide)
    "acylium",      # R-[C+]=O - carbon-centred acyl cation
    "amidinium",    # R-C(=[NH2+])N - amidine cation
    "diazonium",    # R-[N+]#N - terminal diazonium
    # ---- Stage 7 extensions ----
    "diylium",      # multi-locant carbocation: propane-1,3-diylium etc.
    "diide",        # multi-locant carbanion: propane-1,3-diide etc.
    "mixed_id_ylium",   # hybrid +/-: butan-1-id-4-ylium etc.
    "polyacylium",  # bis-acyl cation: oxalylium / malonylium / ...dioylium
    "aminylium",    # R-[NH+] (radical-cation amino): pentan-1-aminylium
    "iminylium",    # R=[N+]  (radical-cation imino): pentan-1-iminylium
    "amidylium",    # R-C(=O)[NH+] (radical-cation amido): ethan-1-amidylium
    # ---- Stage 7 follow-up: aryl-substituted chromenylium (flavylium family) ----
    "aryl_chromenylium",  # 2-aryl-chromenylium (flavylium / N'-methylflavylium)
]

ChargeSign = Literal["+", "-"]


@dataclass(frozen=True)
class ChargeClassification:
    """Structural summary of one charge-suffix motif.

    Attributes
    ----------
    site_atom_indices:
        Tuple of atom indices in the *original* mol that the motif
        claims.  The classifier guarantees no two classifications
        share an atom on the same input mol; if a future motif would
        overlap with one already claimed we drop the later one.
    charge_sign:
        ``"+"`` for cations, ``"-"`` for anions.
    suffix_hint:
        One of the strings in :data:`SuffixHint`.
    locant:
        Numeric locant on the parent skeleton, ``None`` for retained
        / single-atom motifs where no locant is emitted.
    parent_smiles:
        Canonical SMILES of the *neutralized* parent surrogate that
        the engine will be driven against to obtain the parent
        substring.  ``None`` for motifs whose surface name is hard-
        coded (``methylium``, ``boranuide``).
    surface_name:
        The fully-composed name string the engine should emit, *if*
        the classifier already knows the answer.  When set, the
        emitter short-circuits and uses this string verbatim;
        otherwise it falls back to driving the engine on
        ``parent_smiles`` and splicing the suffix.
    """

    site_atom_indices: tuple[int, ...]
    charge_sign: ChargeSign
    suffix_hint: str
    locant: int | None = None
    parent_smiles: str | None = None
    surface_name: str | None = None
    # ---- Stage 7 additions ----
    # Number of RDKit "radical electrons" the classifier explicitly
    # claims at site atoms.  For closed-shell motifs this is 0 (the
    # default), preserving R2-B's wire-format byte-for-byte.  For the
    # radical-cation motifs (aminylium / iminylium / amidylium) RDKit
    # reports 2 radical electrons on the cation N because the N+ is
    # one H short of a closed-shell ammonium; explicitly enumerating
    # them here is what lets the engine route past
    # ``_validate_no_open_valences`` without relaxing that guard.
    radical_count: int = 0
    # The charge magnitude per site atom (parallel to
    # ``site_atom_indices`` for the new multi-charge motifs).  R2-B
    # motifs ignore this and rely on ``charge_sign`` directly.
    site_charges: tuple[int, ...] = ()

    @property
    def is_cation(self) -> bool:
        return self.charge_sign == "+"

    @property
    def is_anion(self) -> bool:
        return self.charge_sign == "-"


# ---------------------------------------------------------------------------
# Top-level classifier
# ---------------------------------------------------------------------------


def classify_charges(mol) -> tuple[ChargeClassification, ...]:
    """Walk ``mol`` and return every charge motif this module recognises.

    The classifier is *non-mutating*: it never neutralises atoms or
    edits bond orders on the input mol.  Each returned classification
    records the atom indices it claims (relative to ``mol``).

    Empty tuple return value means "no recognised motif" and is the
    standard signal to fall through to the existing engine dispatch.
    Crucially this is also what we return for every charge motif the
    rest of the engine already names correctly (ring-N+, retained
    cations like pyrylium, monatomic salts, etc.).
    """
    if mol is None:
        return ()
    # Cheap pre-filter: zero formal charges -> nothing to classify.
    if not any(a.GetFormalCharge() != 0 for a in mol.GetAtoms()):
        return ()

    claimed: set[int] = set()
    out: list[ChargeClassification] = []

    # Order matters: most-specific first so a later, broader rule
    # cannot steal an atom from a more-specific one.
    #
    # Stage 7 ordering note: the radical-cation classifiers
    # (aminylium / iminylium / amidylium) and the polyacylium classifier
    # are each more specific than the closed-shell carbon ylium / ide
    # families, so they run first.  The multi-charge polycation /
    # polyanion classifier runs last among the carbon classifiers
    # because it intentionally does not match single-site cases (those
    # belong to ``_classify_simple_carbon_charge``).
    for fn in (
        # ---- Stage 7 follow-up: 2-aryl-chromenylium (flavylium family) ----
        # Runs FIRST so the aryl-substituted [o+] motif is claimed before
        # any other classifier and BEFORE the engine's PCG selector can
        # decompose the molecule into (chromen-2-yl)benzene by picking the
        # neutral phenyl as parent and silently dropping the [O+].  The
        # bare chromenylium (no 2-aryl) is left untouched here and the
        # existing retained-ring lookup handles it as before.
        _classify_aryl_substituted_chromenylium,
        # ---- closed-shell carbocyclic ring cation (tropylium etc.) ----
        # Runs BEFORE simple_carbon_charge (which rejects aromatic /
        # multiple-bonded skeletons) so the conjugated ring cation is
        # claimed as an "-ylium" motif instead of falling through to the
        # plan-search neutralizer that drops the charge AND saturates the
        # ring (tropylium -> "cycloheptane").
        _classify_aromatic_ring_cation,
        # ---- sp-carbon anion (acetylide: [C-]#C -> ethyn-1-ide) ----
        # Runs BEFORE simple_carbon_charge so the triple-bond pattern is
        # claimed here (simple_carbon_charge rejects non-single bonds).
        _classify_alkynyl_anion,
        # ---- cyclopentadienyl anion ([C-]1C=CC=C1) ----
        # Has 1 radical electron, routes via detect_pre_validation.
        _classify_cyclopentadienyl_anion,
        # ---- R2-B closed-shell motifs (unchanged) ----
        _classify_amidinium,
        _classify_diazonium,
        # ---- Stage 7 polyacylium runs BEFORE _classify_acylium so a
        #      multi-[C+]=O molecule is claimed as a single polyacylium
        #      classification rather than n disjoint single-acylium ones.
        _classify_polyacylium,
        _classify_acylium,
        # ---- Stage 7 radical-cation N motifs ----
        _classify_amidylium,
        _classify_iminylium,
        _classify_aminylium,
        _classify_azaniumyl,
        # ---- P-72/P-73 polycharged / zwitterionic species ----
        # The carbanion zwitterion runs FIRST among the new motifs: it is the
        # only one with both a + and a - atom, so claiming it before any
        # single-sign classifier prevents the carbanion from being stolen by
        # ``_classify_simple_carbon_charge`` (which would drop the cation).
        _classify_carbanion_zwitterion,
        # 2-N backbone di-charge (diazene-diium / diazane-diide) before the
        # generic carbon classifiers (those reject N parents anyway, but
        # ordering keeps the claim explicit).
        _classify_dinitrogen_polycharge,
        # Single |q|==2 carbon bearing aromatic / hetero substituents — must
        # run before _classify_polycarbon_charge (which rejects aromatic atoms
        # and so would silently leave the dianion unclaimed).
        _classify_substituted_carbon_dianion,
        # Ring carbanion dianion on a carbocyclic (possibly partially-hydro)
        # parent — also rejected by _classify_polycarbon_charge's aromatic /
        # multiple-bond guards, so it runs ahead of it here.
        _classify_ring_polycarbanion,
        # ---- R2-B single-site simple ylium / ide ----
        _classify_simple_carbon_charge,
        # ---- Stage 7 multi-charge carbon (di/tri/tetra-ylium / -ide / mixed) ----
        _classify_polycarbon_charge,
        # ---- R2-B borohydride + substituted boranuide ----
        # Substituted form runs first because it covers the multi-heavy-atom
        # case; the bare [BH4-] classifier requires NumHeavyAtoms == 1 so
        # the two are mutually exclusive on the gate.
        _classify_substituted_boranuide,
        _classify_borohydride,
        # ---- Phase 4 carbamate anion: R-NH-C(=O)-O⁻ (before acidic-anion) ----
        _classify_carbamate_anion,
        # ---- P-72.2 / P-73 deprotonated amide N(-): R-C(=O)-[NH-] ----
        # Runs BEFORE the amine classifier so the acyl-bearing N is claimed as
        # an amide anion ({acyl}amide) rather than mis-read as an amine anion.
        _classify_amide_anion,
        # ---- P-72.2 / P-73 deprotonated amine N(-): R-[NH-] / R2-[N-] ----
        _classify_amine_anion,
        # ---- Phase 3 R4 acidic-anion (O⁻ / S⁻ on neutral parent) ----
        _classify_acidic_anion,
    ):
        for cls in fn(mol):
            if any(idx in claimed for idx in cls.site_atom_indices):
                continue
            out.append(cls)
            claimed.update(cls.site_atom_indices)
    return tuple(out)


# ---------------------------------------------------------------------------
# Engine dispatch entry
# ---------------------------------------------------------------------------


def detect(
    mol,
    output_form: OutputForm,
    free_valence: FreeValenceInfo | None,
    decision_ctx: DecisionContext | None,
    strategy,
    session,
    depth: int,
) -> LeafTree | None:
    """Engine entry point.

    Returns a fully-named ``LeafTree`` when one of the recognised
    charge motifs is the *only* charge feature on the molecule and we
    can render it with confidence.  Returns ``None`` to defer to the
    generic plan search otherwise.

    Gates (all must hold for a non-None return):

    * ``output_form == OutputForm.STANDALONE`` (substituent / acyl
      forms keep their existing pathways);
    * ``free_valence is None`` (free-valence atoms get their own
      P-29.2 error path);
    * ``classify_charges(mol)`` returns *exactly one* classification;
    * the molecule is single-fragment (salts go through the salt
      path which already handles per-fragment charge dispatch);
    * the classification's site atoms cover *every* charged atom in
      the mol (otherwise some other charge feature is unaccounted
      for and we must fall through).
    """
    if output_form != OutputForm.STANDALONE or free_valence is not None:
        return None

    # Single-fragment gate.  RDKit's GetMolFrags works on a sanitised mol;
    # multi-fragment salts are handled upstream in the salt dispatch.
    from rdkit import Chem

    try:
        if len(Chem.GetMolFrags(mol)) > 1:
            return None
    except Exception:
        return None

    # Curated inorganic / pseudohalide retained-name priority gate.
    # Multi-atom charged species like ``N#C[S-]`` (thiocyanate), ``[N-]=C=O``
    # (isocyanate), etc. are in ``_INORGANIC_CURATED_SMILES`` under their
    # correct retained names.  The acidic-anion classifier would otherwise
    # re-protonate the deprotonated atom and name the neutral parent, producing
    # "1-sulfanylmethanenitrile" instead of "thiocyanate".  Check the curated
    # table FIRST so the retained name takes precedence over the classifier.
    try:
        _smiles_for_curated = Chem.MolToSmiles(mol)
        from iupac_namer.data_loader import _lookup_curated_inorganic as _lic
        _curated = _lic(_smiles_for_curated)
        if _curated is not None and "name" in _curated:
            return LeafTree(
                output_form=output_form,
                free_valence=free_valence,
                choices_made=(Choice(
                    type="retained",
                    detail=f"inorganic_curated: {_curated['name']}",
                ),),
                decision_ctx=decision_ctx,
                validity_warnings=None,
                text=_curated["name"],
            )
    except Exception:
        pass

    classifications = classify_charges(mol)
    if len(classifications) != 1:
        return None
    cls = classifications[0]

    # Coverage gate: the classification must claim every formally-charged
    # atom in the mol so we never silently drop a charge.
    charged_idx = frozenset(
        a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() != 0
    )
    if not charged_idx.issubset(set(cls.site_atom_indices)):
        return None

    # Stage 7: closed-shell-only gate.  Radical-cation motifs route
    # through ``detect_pre_validation`` (which runs before the
    # free-valence guard); they must not also reach this post-guard
    # entry point or they would be double-claimed.
    if cls.radical_count != 0:
        return None
    # Stage 7: when site_charges is provided the per-site charges must
    # sum to the molecule's net formal charge so we never name a
    # fraction of a multi-charge motif.
    if cls.site_charges:
        net_charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
        if sum(cls.site_charges) != net_charge:
            return None

    text = _render(cls, mol, strategy=strategy, session=session, depth=depth)
    if text is None:
        return None

    return LeafTree(
        output_form=output_form,
        free_valence=free_valence,
        choices_made=(Choice(
            type="charge_perception",
            detail=f"{cls.suffix_hint}: {text}",
        ),),
        decision_ctx=decision_ctx,
        validity_warnings=None,
        text=text,
    )


def detect_pre_validation(
    mol,
    strategy,
    session,
    depth: int = 0,
) -> LeafTree | None:
    """Stage 7: pre-validation entry point for radical-cation motifs.

    Runs from ``name_smiles`` BEFORE ``_validate_no_open_valences`` so
    the radical electrons on the cation N (aminylium / iminylium /
    amidylium) are explicitly claimed instead of triggering the
    free-valence guard.  No guard relaxation: the guard still rejects
    every radical-bearing molecule that does not match a recognised
    closed-form (or curated-inorganic) motif.

    Returns ``None`` for any input that does not match a radical-cation
    motif so the existing dispatch (elementary atom, metallocene,
    free-valence guard, plan search) flows through unchanged.

    Gates (all must hold for a non-None return):

    * single-fragment (multi-fragment radical-cation salts are out of
      scope and properly belong to the salt dispatch);
    * exactly one classification, all its claimed radical electrons
      account for the entirety of ``GetNumRadicalElectrons`` across
      the mol (no silent free-valence drops);
    * all formal charges are covered by the classification's
      ``site_atom_indices`` (no silent charge drops);
    * the classification's ``radical_count > 0`` — closed-shell motifs
      should flow through the post-guard ``detect`` instead.
    """
    from rdkit import Chem

    if mol is None:
        return None
    try:
        if len(Chem.GetMolFrags(mol)) > 1:
            return None
    except Exception:
        return None

    classifications = classify_charges(mol)
    if len(classifications) != 1:
        return None
    cls = classifications[0]

    if cls.radical_count == 0:
        return None

    # Coverage gates: the claim must cover every charged atom AND every
    # radical-bearing atom in the mol, so the free-valence guard would
    # otherwise have nothing left to reject.
    charged_idx = frozenset(
        a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() != 0
    )
    if not charged_idx.issubset(set(cls.site_atom_indices)):
        return None
    total_radicals = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    if cls.radical_count != total_radicals:
        return None

    text = _render(cls, mol, strategy=strategy, session=session, depth=depth)
    if text is None:
        return None
    return LeafTree(
        output_form=OutputForm.STANDALONE,
        free_valence=None,
        choices_made=(Choice(
            type="charge_perception",
            detail=f"{cls.suffix_hint}: {text}",
        ),),
        decision_ctx=None,
        validity_warnings=None,
        text=text,
    )


# ---------------------------------------------------------------------------
# Per-motif classifiers
# ---------------------------------------------------------------------------


def _classify_amidinium(mol) -> Iterable[ChargeClassification]:
    """Detect ``R-C(=[NH2+])N`` - the protonated-imine amidine cation.

    The carbon centre has two N neighbours: one ``=[NH2+]`` (charge
    +1, double bond) and one ``-NH2`` (charge 0, single bond, at least
    one H).
    """
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6 or atom.GetFormalCharge() != 0:
            continue
        if atom.GetDegree() != 3:
            continue
        eq_n_plus: int | None = None
        s_n: int | None = None
        r_neighbour: int | None = None
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            if other.GetAtomicNum() == 7:
                if (
                    other.GetFormalCharge() == 1
                    and bond.GetBondTypeAsDouble() == 2.0
                    and eq_n_plus is None
                ):
                    eq_n_plus = other.GetIdx()
                    continue
                if (
                    other.GetFormalCharge() == 0
                    and bond.GetBondTypeAsDouble() == 1.0
                    and other.GetTotalNumHs() >= 1
                    and s_n is None
                ):
                    s_n = other.GetIdx()
                    continue
            if other.GetAtomicNum() == 6 and bond.GetBondTypeAsDouble() == 1.0:
                if r_neighbour is None:
                    r_neighbour = other.GetIdx()
                    continue
        if eq_n_plus is None or s_n is None or r_neighbour is None:
            continue
        # Guard: the =[NH2+] partner must carry exactly +1 charge; the
        # neutral N must be a primary amine (no other heavy bonds beyond
        # the link to the central carbon).  These guards keep us off
        # false hits on N-substituted guanidinium salts.
        n_plus_atom = mol.GetAtomWithIdx(eq_n_plus)
        if n_plus_atom.GetDegree() != 1 or n_plus_atom.GetTotalNumHs() != 2:
            continue
        n_neutral_atom = mol.GetAtomWithIdx(s_n)
        if n_neutral_atom.GetDegree() != 1:
            continue
        # Site atoms = central C + both Ns.
        site = (atom.GetIdx(), eq_n_plus, s_n)
        yield ChargeClassification(
            site_atom_indices=site,
            charge_sign="+",
            suffix_hint="amidinium",
            locant=None,
            # The parent surrogate is the carboxylic acid: replace the
            # amidinium tail with -C(=O)OH and rename.  Stored implicitly
            # in the emitter so we capture the central-C index here.
            parent_smiles=None,
            surface_name=None,
        )


def _classify_diazonium(mol) -> Iterable[ChargeClassification]:
    """Detect ``R-[N+]#N`` (acyclic, terminal triple bond).

    Pattern: an ``[N+]`` with degree 2, exactly one neighbour ``[N]``
    (charge 0, degree 1, terminal) connected by a triple bond, and one
    parent-carbon neighbour by a single bond.
    """
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 7 or atom.GetFormalCharge() != 1:
            continue
        if atom.IsInRing():
            continue
        if atom.GetDegree() != 2:
            continue
        triple_n: int | None = None
        parent_c: int | None = None
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            if (
                other.GetAtomicNum() == 7
                and other.GetFormalCharge() == 0
                and bond.GetBondTypeAsDouble() == 3.0
                and other.GetDegree() == 1
            ):
                triple_n = other.GetIdx()
                continue
            if (
                other.GetAtomicNum() == 6
                and bond.GetBondTypeAsDouble() == 1.0
            ):
                parent_c = other.GetIdx()
                continue
        if triple_n is None or parent_c is None:
            continue
        site = (atom.GetIdx(), triple_n)
        yield ChargeClassification(
            site_atom_indices=site,
            charge_sign="+",
            suffix_hint="diazonium",
            locant=None,
            parent_smiles=None,
            surface_name=None,
        )


def _classify_acylium(mol) -> Iterable[ChargeClassification]:
    """Detect ``R-[C+]=O`` (acyl cation).

    Pattern: a ``[C+]`` with no hydrogens that is double-bonded to an
    ``[O]`` of charge 0 and single-bonded to one parent neighbour
    (the R group).
    """
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6 or atom.GetFormalCharge() != 1:
            continue
        if atom.GetTotalNumHs() != 0:
            continue
        if atom.IsInRing():
            continue
        if atom.GetDegree() not in (1, 2):
            continue
        eq_o: int | None = None
        parent_idx: int | None = None
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            if (
                other.GetAtomicNum() == 8
                and other.GetFormalCharge() == 0
                and bond.GetBondTypeAsDouble() == 2.0
                and other.GetDegree() == 1
            ):
                eq_o = other.GetIdx()
                continue
            if bond.GetBondTypeAsDouble() == 1.0:
                parent_idx = other.GetIdx()
                continue
        if eq_o is None or parent_idx is None:
            continue
        site = (atom.GetIdx(), eq_o)
        yield ChargeClassification(
            site_atom_indices=site,
            charge_sign="+",
            suffix_hint="acylium",
            locant=None,
            parent_smiles=None,
            surface_name=None,
        )


def _classify_simple_carbon_charge(mol) -> Iterable[ChargeClassification]:
    """Detect aliphatic ``[CH_n+]`` / ``[CH_n-]`` on a saturated chain or ring.

    The test compounds in the audit CSVs are
    methylium / methanide (``[CH3+]`` / ``[CH3-]``),
    ethan-1-ylium / ethan-1-ide (``[CH2+]C`` / ``[CH2-]C``),
    pentan-1-ylium / pentan-1-ide (``[CH2+]CCCC`` / ``[CH2-]CCCC``),
    and cyclohexan-1-ylium / cyclohexan-1-ide
    (``[CH+]1CCCCC1`` / ``[CH-]1CCCCC1``).

    Pattern: exactly one charged carbon in the mol, |charge|=1, no
    heteroatom or aromatic neighbour, and every other heavy atom is
    a saturated sp3 carbon connected to it via single bonds.
    """
    charged_carbons = [
        a for a in mol.GetAtoms()
        if a.GetAtomicNum() == 6 and abs(a.GetFormalCharge()) == 1
    ]
    if len(charged_carbons) != 1:
        return
    c = charged_carbons[0]
    # Must not have free-valence (radical) - unrelated audit territory.
    if c.GetNumRadicalElectrons() != 0:
        return
    # All neighbours via single bond (no =O, no =N, etc).
    for bond in c.GetBonds():
        if bond.GetBondTypeAsDouble() != 1.0:
            return
        other = bond.GetOtherAtom(c)
        if other.GetAtomicNum() != 6:
            return
        if other.GetIsAromatic():
            return
    # The whole skeleton must be saturated all-carbon.
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6:
            return
        if atom.GetIsAromatic():
            return
    # All bonds between heavy atoms must be single.
    for bond in mol.GetBonds():
        if bond.GetBondTypeAsDouble() != 1.0:
            return
    sign: ChargeSign = "+" if c.GetFormalCharge() == 1 else "-"
    yield ChargeClassification(
        site_atom_indices=(c.GetIdx(),),
        charge_sign=sign,
        suffix_hint="ylium" if sign == "+" else "ide",
        locant=None,  # filled in by the emitter once parent is named
        parent_smiles=None,
        surface_name=None,
    )


def _classify_aromatic_ring_cation(mol) -> Iterable[ChargeClassification]:
    """Detect a closed-shell carbocyclic ring carbocation (``-ylium``).

    Covers the aromatic / conjugated carbocyclic cations whose only
    formal-charge atom is a single ring carbon ``[c+]`` / ``[C+]`` in an
    all-carbon ring, e.g.

    * ``c1ccc[cH+]cc1``     -> ``cyclohepta-2,4,6-trien-1-ylium`` (tropylium)
    * ``C1=C[CH+]1``        -> ``cycloprop-2-en-1-ylium`` (cyclopropenylium)
    * ``[CH+]1C=CC=C1``     -> ``cyclopenta-2,4-dien-1-ylium``
    * ``[C+]1=CC=CC=CC=C1`` -> ``cycloocta-1,3,5,7-tetraen-1-ylium``

    Why this classifier exists
    --------------------------
    The plan-search neutralizer drops the ring charge *and* saturates the
    ring (tropylium -> "cycloheptane"), discarding both the cation and the
    unsaturation — a badly WRONG result.  ``_classify_simple_carbon_charge``
    deliberately bails out on aromatic / multiple-bonded skeletons, so these
    fall through to plan search untouched.  This classifier claims the ring
    cation as an ``-ylium`` motif (P-73.2.2.1.1 H-removal cation) and the
    renderer composes the ``cyclo<stem>-<ene-locants>-<n>-1-ylium`` PIN by
    fixing the cationic carbon at locant 1 (the ylium suffix outranks the
    ene unsaturation for low locants).

    Detection contract (all must hold, else yields nothing):

    * exactly one formally-charged atom, a carbon with charge +1;
    * the molecule is a single all-carbon ring (no heteroatoms, no fused
      partners) and the cationic carbon is in that ring;
    * CLOSED-SHELL: zero radical electrons anywhere (the free-valence guard
      stays untouched — radical ring cations are out of scope);
    * the ring contains at least one C=C double bond (a fully-saturated ring
      cation ``[CH+]1CCCCC1`` is the ``cyclohexan-1-ylium`` case already
      handled by ``_classify_simple_carbon_charge``).
    """
    charged = [a for a in mol.GetAtoms() if a.GetFormalCharge() != 0]
    if len(charged) != 1:
        return
    c = charged[0]
    if c.GetAtomicNum() != 6 or c.GetFormalCharge() != 1:
        return
    if not c.IsInRing():
        return
    # Closed-shell only — radical ring cations route through the
    # free-valence guard, never here.
    if any(a.GetNumRadicalElectrons() != 0 for a in mol.GetAtoms()):
        return
    ri = mol.GetRingInfo()
    rings = ri.AtomRings()
    if len(rings) != 1:
        return  # fused / polycyclic ring cations are out of scope
    ring = rings[0]
    if c.GetIdx() not in ring:
        return
    # Every ring atom must be carbon (carbocyclic) and every heavy atom in
    # the molecule must belong to the ring (no exocyclic substituents — a
    # substituted ring cation needs the full substitutive locant machinery).
    ring_set = set(ring)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        if atom.GetIdx() not in ring_set:
            return
        if atom.GetAtomicNum() != 6:
            return
    # Defer to the retained ring-cation table when the cation has a retained
    # PIN (the benzene cation -> "phenylium"; the cyclopentadienyl cation ->
    # "cyclopentadienylium").  Those names are senior to the systematic
    # ``cyclo<stem>-...-ylium`` form, so the retained-ring path must own them
    # byte-identically.  Only ring cations *without* a retained ylium name
    # (cyclopropenylium / tropylium / cyclooctatetraenylium …) are claimed
    # here.  Built generatively from the OPSIN-mined ring table so any future
    # retained ring-cation entry is honoured without code change.
    from rdkit import Chem

    try:
        _canon = Chem.MolToSmiles(mol)
        if _canon in _retained_carbocyclic_ring_cation_smiles():
            return
    except Exception:
        pass

    # Require at least one ring C=C (saturated ring cation is the simple
    # cyclohexan-1-ylium case handled elsewhere).
    km = Chem.RWMol(mol)
    try:
        Chem.Kekulize(km, clearAromaticFlags=True)
    except Exception:
        return
    has_double = False
    for bond in km.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a1 in ring_set and a2 in ring_set and bond.GetBondTypeAsDouble() == 2.0:
            has_double = True
            break
    if not has_double:
        return
    yield ChargeClassification(
        site_atom_indices=(c.GetIdx(),),
        charge_sign="+",
        suffix_hint="aromatic_ring_ylium",
        locant=1,
        parent_smiles=None,
        surface_name=None,
    )


def _classify_alkynyl_anion(mol) -> Iterable[ChargeClassification]:
    """Detect ``[C-]#C`` — ethyn-1-ide (acetylide).

    Pattern:
    * exactly two heavy atoms, both carbon;
    * exactly one has formal charge -1 and zero H (sp carbanion);
    * the other is neutral with one H;
    * one triple bond between them;
    * no radical electrons.

    Emits a pre-cooked ``surface_name="ethyn-1-ide"`` so the renderer
    short-circuits without needing a recursive engine call.
    OPSIN round-trip: ``ethyn-1-ide`` -> ``[C-]#C`` ✓
    """
    if mol.GetNumHeavyAtoms() != 2:
        return
    atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() != 1]
    if len(atoms) != 2:
        return
    if any(a.GetAtomicNum() != 6 for a in atoms):
        return
    if any(a.GetNumRadicalElectrons() != 0 for a in atoms):
        return
    charged = [a for a in atoms if a.GetFormalCharge() == -1]
    neutral = [a for a in atoms if a.GetFormalCharge() == 0]
    if len(charged) != 1 or len(neutral) != 1:
        return
    # The charged C must have no H (sp anion) and the neutral C must have 1 H.
    if charged[0].GetTotalNumHs() != 0:
        return
    if neutral[0].GetTotalNumHs() != 1:
        return
    # Must be connected by a triple bond.
    bond = mol.GetBondBetweenAtoms(charged[0].GetIdx(), neutral[0].GetIdx())
    if bond is None or bond.GetBondTypeAsDouble() != 3.0:
        return
    yield ChargeClassification(
        site_atom_indices=(charged[0].GetIdx(),),
        charge_sign="-",
        suffix_hint="ide",
        locant=1,
        parent_smiles=None,
        surface_name="ethyn-1-ide",
    )


def _classify_cyclopentadienyl_anion(mol) -> Iterable[ChargeClassification]:
    """Detect ``[C-]1C=CC=C1`` — cyclopenta-2,4-dien-1-ide.

    Pattern:
    * exactly five heavy atoms, all carbon;
    * exactly one formally charged C with charge -1 and zero H
      (the unusual-valence sp2 carbanion that RDKit models with
      1 radical electron);
    * the ring is 5-membered and all-carbon;
    * total net charge -1, total radical electrons == 1 (the one on the C-).

    Uses ``radical_count=1`` so the classification is claimed by
    ``detect_pre_validation`` (which runs before the free-valence guard)
    and ``surface_name`` short-circuits the renderer.

    OPSIN parses ``cyclopenta-2,4-dien-1-ide`` as ``[CH-]1C=CC=C1`` (with
    1 H on the carbanion) which is a different tautomer with conventional
    valency; the bare-C form ``[C-]1C=CC=C1`` is an exotic species that
    shares the same name by chemical convention.
    """
    if mol.GetNumHeavyAtoms() != 5:
        return
    atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() != 1]
    if len(atoms) != 5:
        return
    if any(a.GetAtomicNum() != 6 for a in atoms):
        return
    charged = [a for a in atoms if a.GetFormalCharge() == -1]
    if len(charged) != 1:
        return
    c_minus = charged[0]
    if c_minus.GetTotalNumHs() != 0:
        return
    # The radical electron on [C-] in a ring.
    total_radicals = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    if c_minus.GetNumRadicalElectrons() != 1:
        return
    if total_radicals != 1:
        return
    # Must be in a 5-membered ring.
    if not c_minus.IsInRing():
        return
    ri = mol.GetRingInfo()
    five_rings = [r for r in ri.AtomRings() if len(r) == 5]
    if not five_rings:
        return
    # Verify c_minus is in one of the 5-membered rings and all atoms in that ring are C.
    c_minus_in_5ring = any(c_minus.GetIdx() in r for r in five_rings)
    if not c_minus_in_5ring:
        return
    # All other heavy atoms must be neutral.
    if any(a.GetFormalCharge() != 0 for a in atoms if a.GetIdx() != c_minus.GetIdx()):
        return
    yield ChargeClassification(
        site_atom_indices=(c_minus.GetIdx(),),
        charge_sign="-",
        suffix_hint="ide",
        locant=1,
        parent_smiles=None,
        surface_name="cyclopenta-2,4-dien-1-ide",
        radical_count=1,
    )


def _classify_borohydride(mol) -> Iterable[ChargeClassification]:
    """Detect ``[BH4-]`` - the boranuide anion.

    Single heavy atom, B, charge -1, four implicit/explicit hydrogens.
    """
    if mol.GetNumHeavyAtoms() != 1:
        return
    only = next((a for a in mol.GetAtoms() if a.GetAtomicNum() != 1), None)
    if only is None or only.GetAtomicNum() != 5:
        return
    if only.GetFormalCharge() != -1:
        return
    if only.GetTotalNumHs() != 4:
        return
    yield ChargeClassification(
        site_atom_indices=(only.GetIdx(),),
        charge_sign="-",
        suffix_hint="uide",
        locant=None,
        parent_smiles=None,
        surface_name="boranuide",
    )


def _classify_carbamate_anion(mol) -> Iterable[ChargeClassification]:
    """Detect the N-substituted carbamate anion motif: R-NH-C(=O)-O⁻.

    Per P-72.2 / P-77, ``N-Rcarbamate`` is the IUPAC PIN for the anion
    derived by deprotonating the O-H of N-substituted carbamic acid.

    Pattern (all must hold):

    * exactly one formally-charged atom (-1 on O, no H, one C neighbour)
    * the C neighbour has a =O and an N neighbour (amide pattern)
    * the N is NOT part of the carbamate C's ring (ring-embedded → fall through)
    * the N carries only C substituents (alkyl / aryl / H) — no P, S, etc.
    * the C has no other heavy-atom substituents beyond =O, O⁻, and N

    The bare ``NC(=O)[O-]`` case (no N-substituents) is already covered by
    the ``_INORGANIC_CURATED_SMILES`` lookup in ``detect()``; this classifier
    handles the *substituted* carbamate anion (one or two N-substituents).
    """
    charged_atoms = [a for a in mol.GetAtoms() if a.GetFormalCharge() != 0]
    if len(charged_atoms) != 1:
        return
    a = charged_atoms[0]
    if a.GetFormalCharge() != -1 or a.GetSymbol() != "O":
        return
    if a.GetTotalNumHs() != 0:
        return
    heavy_nbs = [nb for nb in a.GetNeighbors() if nb.GetAtomicNum() != 1]
    if len(heavy_nbs) != 1:
        return
    acyl_c = heavy_nbs[0]
    if acyl_c.GetAtomicNum() != 6:
        return
    bond = mol.GetBondBetweenAtoms(a.GetIdx(), acyl_c.GetIdx())
    if bond is None or bond.GetBondTypeAsDouble() != 1.0:
        return

    # Identify =O and N neighbours on the acyl C (besides the O⁻).
    oxo_nb = None
    n_nb = None
    for nb2 in acyl_c.GetNeighbors():
        if nb2.GetIdx() == a.GetIdx():
            continue
        b2 = mol.GetBondBetweenAtoms(acyl_c.GetIdx(), nb2.GetIdx())
        if b2 is None:
            continue
        if nb2.GetAtomicNum() == 8 and b2.GetBondTypeAsDouble() == 2.0:
            oxo_nb = nb2
        elif nb2.GetAtomicNum() == 7 and b2.GetBondTypeAsDouble() == 1.0:
            n_nb = nb2
        else:
            # Unexpected heavy neighbour (e.g. C-C, C-S…) → not carbamate
            return
    if oxo_nb is None or n_nb is None:
        return

    # The N must have at least one C-substituent (otherwise bare NH2 carbamate
    # is handled by the curated table).
    n_heavy_nbs = [nb for nb in n_nb.GetNeighbors() if nb.GetAtomicNum() != 1
                   and nb.GetIdx() != acyl_c.GetIdx()]
    if not n_heavy_nbs:
        return  # bare NH2 → curated table handles it

    # All N heavy-substituents must be carbon (no O, S, P…).
    for sub in n_heavy_nbs:
        if sub.GetAtomicNum() != 6:
            return

    # N must not be ring-embedded with the acyl C (lactamate / cyclic
    # carbamate → fall through to plan search).
    if n_nb.IsInRing():
        # Check if acyl_c and n_nb share any ring.
        ri = mol.GetRingInfo()
        for ring in ri.AtomRings():
            if acyl_c.GetIdx() in ring and n_nb.GetIdx() in ring:
                return

    # Claim the acyl C, =O, O⁻, and N atom.
    site_indices = (a.GetIdx(), acyl_c.GetIdx(), oxo_nb.GetIdx(), n_nb.GetIdx())
    yield ChargeClassification(
        site_atom_indices=site_indices,
        charge_sign="-",
        suffix_hint="carbamate_anion",
        locant=None,
        parent_smiles=None,
        surface_name=None,
    )


def _classify_amide_anion(mol) -> Iterable[ChargeClassification]:
    """Detect the deprotonated primary-amide nitrogen anion ``R-C(=O)-[NH-]``.

    Per IUPAC P-72.2 / P-73 the deprotonated amide N is the principal
    anionic characteristic group; the OPSIN-parseable PIN is the
    acyl-group name + ``amide`` (``acetylamide``, ``benzoylamide``,
    ``formylamide``, ``propanoylamide`` …).  The systematic ``-amidide``
    promotion of the parent amide is *not* an OPSIN-parseable form, so the
    renderer carves the corresponding acid, names its acyl group, and
    appends ``amide``.

    Gates (all must hold):

    * exactly one formally-charged atom in the molecule, charge -1, N, no
      radical electrons;
    * the N carries exactly one H and exactly one heavy neighbour (the
      acyl carbon) via a single bond — i.e. a *primary* amide anion.
      The N-substituted (secondary) amide anion ``R-C(=O)-[N-]-R'`` has
      no reliable OPSIN round-trip form and is deliberately left to fall
      through.
    * the heavy neighbour is a carbon bearing exactly one terminal ``=O``
      (the acyl carbonyl), so sulfonamide / phosphonamide N-anions
      (bonded to S / P, handled elsewhere) are excluded.
    """
    charged = [a for a in mol.GetAtoms() if a.GetFormalCharge() != 0]
    if len(charged) != 1:
        return
    n = charged[0]
    if n.GetAtomicNum() != 7 or n.GetFormalCharge() != -1:
        return
    if n.GetNumRadicalElectrons() != 0:
        return
    if n.GetTotalNumHs() != 1:
        return
    heavy_nbs = [nb for nb in n.GetNeighbors() if nb.GetAtomicNum() != 1]
    if len(heavy_nbs) != 1:
        return
    acyl_c = heavy_nbs[0]
    if acyl_c.GetAtomicNum() != 6:
        return
    bond = mol.GetBondBetweenAtoms(n.GetIdx(), acyl_c.GetIdx())
    if bond is None or bond.GetBondTypeAsDouble() != 1.0:
        return
    # The acyl carbon must bear exactly one terminal carbonyl =O (and no
    # ring membership — ring amides are lactam anions, out of scope here).
    if acyl_c.IsInRing():
        return
    oxo_count = 0
    for nb2 in acyl_c.GetNeighbors():
        if nb2.GetIdx() == n.GetIdx():
            continue
        b2 = mol.GetBondBetweenAtoms(acyl_c.GetIdx(), nb2.GetIdx())
        if b2 is None:
            continue
        if (nb2.GetAtomicNum() == 8 and b2.GetBondTypeAsDouble() == 2.0
                and nb2.GetDegree() == 1 and nb2.GetFormalCharge() == 0):
            oxo_count += 1
        elif b2.GetBondTypeAsDouble() != 1.0:
            # An additional multiple bond on the acyl C (=N, =S, …) is not
            # a plain amide → fall through.
            return
    if oxo_count != 1:
        return
    yield ChargeClassification(
        site_atom_indices=(n.GetIdx(),),
        charge_sign="-",
        suffix_hint="amide_anion",
        locant=None,
        parent_smiles=None,
        surface_name=None,
    )


def _classify_amine_anion(mol) -> Iterable[ChargeClassification]:
    """Detect the deprotonated amine nitrogen anion ``R-[NH-]`` / ``R2-[N-]``.

    Per IUPAC P-72.2 / P-73 the deprotonated amine N is the principal
    anionic characteristic group: the parent amine's ``-amine`` suffix is
    promoted to ``-aminide`` (methanamine → methanaminide, benzenamine →
    benzenaminide, N-methylmethanamine → N-methylmethanaminide).

    The renderer re-protonates the N to a neutral amine and drives the
    engine with ``OutputForm.ANION``; the SUFFIX_VARIANT_TABLE entry
    ``("amine", OutputForm.ANION) → "aminide"`` and the retained-amine
    ANION skip in ``_generate_retained_plans`` produce the systematic PIN.

    Gates (all must hold):

    * exactly one formally-charged atom in the molecule, charge -1, N, no
      radical electrons;
    * every heavy neighbour of the N is carbon, bonded by a single bond
      (so amide / sulfonamide / imide / nitrogen-anion-chain motifs,
      whose N is bonded to an acyl C / S / N, are NOT claimed here — the
      amide-anion classifier runs first for the acyl case);
    * none of the N's carbon neighbours is an acyl carbon (a C bearing a
      terminal ``=O``); that shape is the amide anion handled above.
    * the N has 0, 1, or 2 heavy (carbon) neighbours — primary or
      secondary amine anion (the all-H ``[NH2-]`` azanide and the
      tertiary ``R3N`` cannot be deprotonated, so they never reach here).
    """
    charged = [a for a in mol.GetAtoms() if a.GetFormalCharge() != 0]
    if len(charged) != 1:
        return
    n = charged[0]
    if n.GetAtomicNum() != 7 or n.GetFormalCharge() != -1:
        return
    if n.GetNumRadicalElectrons() != 0:
        return
    heavy_nbs = [nb for nb in n.GetNeighbors() if nb.GetAtomicNum() != 1]
    # At least one carbon neighbour (bare [NH2-] is the azanide handled by
    # the single-atom substituent table / curated path) and at most two.
    if not (1 <= len(heavy_nbs) <= 2):
        return
    for nb in heavy_nbs:
        if nb.GetAtomicNum() != 6:
            return
        bond = mol.GetBondBetweenAtoms(n.GetIdx(), nb.GetIdx())
        if bond is None or bond.GetBondTypeAsDouble() != 1.0:
            return
        # Reject acyl-carbon neighbours (amide anion → handled separately).
        for nb2 in nb.GetNeighbors():
            if nb2.GetIdx() == n.GetIdx():
                continue
            b2 = mol.GetBondBetweenAtoms(nb.GetIdx(), nb2.GetIdx())
            if (b2 is not None and b2.GetBondTypeAsDouble() == 2.0
                    and nb2.GetAtomicNum() == 8 and nb2.GetDegree() == 1
                    and nb2.GetFormalCharge() == 0):
                return
    yield ChargeClassification(
        site_atom_indices=(n.GetIdx(),),
        charge_sign="-",
        suffix_hint="amine_anion",
        locant=None,
        parent_smiles=None,
        surface_name=None,
    )


def _classify_acidic_anion(mol) -> Iterable[ChargeClassification]:
    """Detect a single negative charge on O or S where the deprotonation
    site corresponds to a recognised acidic group.

    Per IUPAC P-72.2, the spec PINs for these anions are formed by
    promoting the parent's principal characteristic group suffix to its
    ANION variant:

      * R-O⁻      (alkoxide / phenoxide)        → ``-olate``
      * R-S⁻      (thiolate)                    → ``-thiolate``
      * R-COO⁻    (carboxylate)                 → ``-oate`` / ``-ate``

    The classifier flags the structural shape; the renderer
    re-protonates the deprotonated atom and drives the engine on the
    resulting neutral parent with ``OutputForm.ANION``, which lets the
    standard suffix machinery (and the retained-name table) emit the
    correct anion PIN.

    Gates (all must hold):
      * exactly one formally-charged atom in the molecule
      * the charge is -1
      * the charged atom is O or S (carbanion / nitride / etc. flow
        through the existing simple-carbon-charge classifier)
      * the charged atom has exactly one heavy neighbour and zero H
        (carved deprotonation site)
      * for O⁻ on a carbonyl-C neighbour: the C must carry a =O so the
        re-protonation builds a carboxylic acid (not a hemiketal etc.)
      * for O⁻ / S⁻ on a non-carbonyl neighbour: the neighbour must be
        carbon (alcohol / thiol)
    """
    charged_atoms = [a for a in mol.GetAtoms() if a.GetFormalCharge() != 0]
    if not charged_atoms:
        return
    # P-65.1 / P-66 / P-72.2 multiplicative anions: 1..N homogeneous
    # deprotonation sites of the SAME acid-anion class (all alkoxide /
    # thiolate "olate", or all carboxylate).  Each site is validated by
    # ``_acidic_anion_site_kind``; when ≥2 are present a single
    # classification claims all of them and the renderer re-protonates
    # every site and drives the engine with OutputForm.ANION, so the
    # standard suffix machinery emits the multiplied anion suffix
    # (``benzene-1,2-dithiolate`` / ``butane-1,4-dithiolate``) rather
    # than the OPSIN-unparseable ``bis(sulfanide)`` substituent-prefix
    # fallback.
    site_kinds: dict[int, str] = {}
    for a in charged_atoms:
        kind = _acidic_anion_site_kind(mol, a)
        if kind is None:
            return  # a non-acidic-anion charged atom present -> defer
        site_kinds[a.GetIdx()] = kind
    # All sites must be the same class so a single uniform suffix applies.
    kinds = set(site_kinds.values())
    if len(kinds) != 1:
        return
    is_carboxylate = kinds == {"carboxylate"}
    site_indices = sorted(site_kinds)
    # Scope-narrowing gate: only fire when the deprotonation sites are the
    # ONLY acid-derived functional groups on the molecule.  Mixed
    # charged+neutral cases (a thiolate next to a neutral -SH, a
    # carboxylate next to a -OH, …) are deferred to the standard
    # plan-search path, which already handles them through
    # SUFFIX_VARIANT_TABLE + the salt-dispatch _choose_salt_fragment_form
    # gate; routing them through this fast path can produce salt-context
    # names OPSIN cannot round-trip (e.g. complex cation + ‐oate where the
    # cation ends in "-diol", which OPSIN parses as an ester
    # relationship).  The plan-search form (oxido/oxo prefix on a more
    # elaborate parent) is OPSIN-safe.
    for other in mol.GetAtoms():
        if other.GetIdx() in site_kinds:
            continue
        if other.GetSymbol() not in ("O", "S", "N"):
            continue
        # An -OH / -SH / -NH (neutral acidic H) is a competing FG.
        if other.GetTotalNumHs() == 0:
            continue
        return
    yield ChargeClassification(
        site_atom_indices=tuple(site_indices),
        charge_sign="-",
        suffix_hint="acidic_anion_carboxylate"
        if is_carboxylate else "acidic_anion_olate",
        locant=None,
        parent_smiles=None,
        surface_name=None,
    )


def _acidic_anion_site_kind(mol, a) -> str | None:
    """Classify a single charged atom as an acid-derived anion site.

    Returns ``"carboxylate"`` for ``-C(=O)-O⁻``, ``"olate"`` for an
    alkoxide / phenoxide / thiolate (``C-O⁻`` / ``C-S⁻`` with no carbonyl
    on the C), or ``None`` if the atom is not a recognised single-site
    acidic anion (so the caller defers to the generic plan search).

    Gates (all must hold):
      * charge is exactly -1
      * the atom is O or S with zero H (carved deprotonation site)
      * exactly one heavy neighbour, which is carbon, single-bonded
    """
    if a.GetFormalCharge() != -1:
        return None
    if a.GetSymbol() not in ("O", "S"):
        return None
    if a.GetTotalNumHs() != 0:
        return None
    heavy_nbs = [nb for nb in a.GetNeighbors() if nb.GetAtomicNum() != 1]
    if len(heavy_nbs) != 1:
        return None
    nb = heavy_nbs[0]
    if nb.GetAtomicNum() != 6:
        return None
    bond = mol.GetBondBetweenAtoms(a.GetIdx(), nb.GetIdx())
    if bond is None or bond.GetBondTypeAsDouble() != 1.0:
        return None
    # Distinguish carboxylate vs alkoxide/thiolate by checking for an
    # =O on the C neighbour (other than the charged O).
    if a.GetSymbol() == "O":
        for nb2 in nb.GetNeighbors():
            if nb2.GetIdx() == a.GetIdx():
                continue
            if nb2.GetAtomicNum() != 8:
                continue
            b2 = mol.GetBondBetweenAtoms(nb.GetIdx(), nb2.GetIdx())
            if b2 is not None and b2.GetBondTypeAsDouble() == 2.0:
                return "carboxylate"
    return "olate"


def _classify_substituted_boranuide(mol) -> Iterable[ChargeClassification]:
    """Detect ``R-[BH(3-n)-]`` - substituted boranuide.

    Per IUPAC P-72.3, ``[BH4-]`` (boranuide) is the parent anion; replacing
    one or more H atoms with carbon (or other) substituents gives names of
    the form ``methylboranuide`` / ``propylboranuide`` /
    ``dimethylboranuide`` etc.  Emit a single classification claiming the B
    atom; ``_render_substituted_boranuide`` recurses into each substituent
    via the engine's standard SUBSTITUENT path.

    Gating: exactly one B atom in the molecule with charge -1 and 1+
    heavy neighbours, totalling 4 (heavy + H), and no other charged atoms
    or open valences.  Stricter shapes (B-ring, B with multiple bonds, B
    with non-carbon heavy neighbours other than O/N/S that would have a
    distinct retained PIN) are intentionally left to the general anion
    dispatch.
    """
    boron_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 5]
    if len(boron_atoms) != 1:
        return
    b = boron_atoms[0]
    if b.GetFormalCharge() != -1:
        return
    if b.GetIsAromatic():
        return
    if b.IsInRing():
        return  # ring forms (borinanuide etc.) need P-72.4 skeletal path
    # Heavy neighbours: must be 1+ and the total (heavy + H) must be 4.
    heavy_nbs = [nb for nb in b.GetNeighbors() if nb.GetAtomicNum() != 1]
    if not heavy_nbs:
        return
    n_h = b.GetTotalNumHs()
    if len(heavy_nbs) + n_h != 4:
        return
    # All B-heavy bonds must be single (no =B/-B# multiple bonds — those
    # would be a different anion class).
    for nb in heavy_nbs:
        bond = mol.GetBondBetweenAtoms(b.GetIdx(), nb.GetIdx())
        if bond is None or bond.GetBondTypeAsDouble() != 1.0:
            return
    yield ChargeClassification(
        site_atom_indices=(b.GetIdx(),),
        charge_sign="-",
        suffix_hint="substituted_boranuide",
        locant=None,
        parent_smiles=None,
        surface_name=None,
    )


# ---------------------------------------------------------------------------
# Stage 7: radical-cation N motifs (aminylium / iminylium / amidylium)
# ---------------------------------------------------------------------------


def _classify_amidylium(mol) -> Iterable[ChargeClassification]:
    """Detect ``R-C(=O)[NH+]`` (radical-cation amido).

    The ``[NH+]`` carries 2 radical electrons in RDKit's accounting
    because it is one H short of a closed-shell ammonium and the
    additional unpaired electron is what distinguishes ``-amidylium``
    from the closed-shell amide ``-amide`` form.

    Pattern (atom indices in mol):
      C(=O)(R)-[NH+] with the central acyl C bonded to one R
      neighbour (carbon, single bond), one terminal ``=O`` (double
      bond), and one terminal ``[NH+]`` (single bond, charge +1, one
      H, two radical electrons).
    """
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6 or atom.GetFormalCharge() != 0:
            continue
        if atom.GetDegree() != 3:
            continue
        n_plus_idx: int | None = None
        o_idx: int | None = None
        r_idx: int | None = None
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            bt = bond.GetBondTypeAsDouble()
            if (
                other.GetAtomicNum() == 7
                and other.GetFormalCharge() == 1
                and bt == 1.0
                and other.GetDegree() == 1
                and other.GetTotalNumHs() == 1
                and other.GetNumRadicalElectrons() == 2
                and n_plus_idx is None
            ):
                n_plus_idx = other.GetIdx()
                continue
            if (
                other.GetAtomicNum() == 8
                and other.GetFormalCharge() == 0
                and bt == 2.0
                and other.GetDegree() == 1
                and o_idx is None
            ):
                o_idx = other.GetIdx()
                continue
            if (
                other.GetAtomicNum() == 6
                and bt == 1.0
                and r_idx is None
            ):
                r_idx = other.GetIdx()
                continue
        if n_plus_idx is None or o_idx is None or r_idx is None:
            continue
        site = (atom.GetIdx(), n_plus_idx, o_idx)
        yield ChargeClassification(
            site_atom_indices=site,
            charge_sign="+",
            suffix_hint="amidylium",
            locant=None,
            parent_smiles=None,
            surface_name=None,
            radical_count=2,
            site_charges=(0, 1, 0),
        )


def _classify_iminylium(mol) -> Iterable[ChargeClassification]:
    """Detect ``R=[N+]`` (radical-cation imino).

    Pattern: a parent C double-bonded to an ``[N+]`` of degree 1 with
    no H and 2 radical electrons (e.g. ``CC=[N+]`` from
    ``ethan-1-iminylium``).
    """
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 7 or atom.GetFormalCharge() != 1:
            continue
        if atom.GetDegree() != 1:
            continue
        if atom.GetTotalNumHs() != 0:
            continue
        if atom.GetNumRadicalElectrons() != 2:
            continue
        # Single bond — must be exactly one double bond to a C.
        bond = atom.GetBonds()[0]
        if bond.GetBondTypeAsDouble() != 2.0:
            continue
        carbon = bond.GetOtherAtom(atom)
        if carbon.GetAtomicNum() != 6:
            continue
        site = (atom.GetIdx(),)
        yield ChargeClassification(
            site_atom_indices=site,
            charge_sign="+",
            suffix_hint="iminylium",
            locant=None,
            parent_smiles=None,
            surface_name=None,
            radical_count=2,
            site_charges=(1,),
        )


def _classify_aminylium(mol) -> Iterable[ChargeClassification]:
    """Detect ``R-[NH+]`` (radical-cation amino).

    Pattern: an ``[NH+]`` of degree 1 with 1 H and 2 radical
    electrons, single-bonded to a parent atom (typically C, including
    aromatic ring carbons).  Distinguishes from ``=[N+]`` (iminylium,
    classified above) and from ``[NH3+]`` closed-shell ammonium.

    The check explicitly excludes the ``-C(=O)[NH+]`` shape so the
    amidylium classifier (registered earlier) is not undermined.
    """
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 7 or atom.GetFormalCharge() != 1:
            continue
        if atom.GetDegree() != 1:
            continue
        if atom.GetTotalNumHs() != 1:
            continue
        if atom.GetNumRadicalElectrons() != 2:
            continue
        bond = atom.GetBonds()[0]
        if bond.GetBondTypeAsDouble() != 1.0:
            continue
        # Parent-C must not itself be the acyl carbon of an
        # ``-C(=O)[NH+]`` amidylium motif (already classified).
        parent = bond.GetOtherAtom(atom)
        if parent.GetAtomicNum() == 6:
            has_acyl_o = any(
                bnd.GetBondTypeAsDouble() == 2.0
                and bnd.GetOtherAtom(parent).GetAtomicNum() == 8
                and bnd.GetOtherAtom(parent).GetFormalCharge() == 0
                and bnd.GetOtherAtom(parent).GetDegree() == 1
                for bnd in parent.GetBonds()
                if bnd.GetIdx() != bond.GetIdx()
            )
            if has_acyl_o:
                continue
        site = (atom.GetIdx(),)
        yield ChargeClassification(
            site_atom_indices=site,
            charge_sign="+",
            suffix_hint="aminylium",
            locant=None,
            parent_smiles=None,
            surface_name=None,
            radical_count=2,
            site_charges=(1,),
        )


def _classify_azaniumyl(mol) -> Iterable[ChargeClassification]:
    """Detect ``R-[NH2+]`` (azaniumyl radical-cation, P-73 / P-29.2).

    Pattern: an ``[NH2+]`` of degree 1 with 2 H and 1 radical electron,
    single-bonded to a parent atom (typically C).  This differs from the
    ``aminylium`` pattern ([NH+], 1H, 2 radEle) in both H-count and
    radical-electron count.

    The IUPAC name is ``{R}azaniumyl`` — e.g. ``C[NH2+]`` → "methylazaniumyl".
    OPSIN (allow_radicals=True) parses "methylazaniumyl" → C[NH2+]. ✓
    """
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 7 or atom.GetFormalCharge() != 1:
            continue
        if atom.GetDegree() != 1:
            continue
        if atom.GetTotalNumHs() != 2:
            continue
        if atom.GetNumRadicalElectrons() != 1:
            continue
        bond = atom.GetBonds()[0]
        if bond.GetBondTypeAsDouble() != 1.0:
            continue
        site = (atom.GetIdx(),)
        yield ChargeClassification(
            site_atom_indices=site,
            charge_sign="+",
            suffix_hint="azaniumyl",
            locant=None,
            parent_smiles=None,
            surface_name=None,
            radical_count=1,
            site_charges=(1,),
        )


# ---------------------------------------------------------------------------
# Stage 7 follow-up: 2-aryl-chromenylium (flavylium family)
# ---------------------------------------------------------------------------


# Canonical chromenylium core: aromatic [o+] in a 6,6-fused bicyclic with one
# pyrylium ring (containing the [o+]) and one benzene ring fused at 4a-8a.
# Atom-order template (see retained_lookup.py:643): O+ = pos 1, then pos 2 is
# the aromatic C ortho to [o+] and NOT in the benzo ring.
_CHROMENYLIUM_CORE_SMILES = "c1cc2ccccc2[o+]c1"


def _classify_aryl_substituted_chromenylium(mol) -> Iterable[ChargeClassification]:
    """Detect ``2-aryl-chromenylium`` (the flavylium family).

    Why this classifier exists
    --------------------------
    The retained-ring lookup happily resolves the bare chromenylium
    SMILES to the name ``chromenylium``, but the engine's PCG (parent
    class group) selector treats a 2-aryl-chromenylium molecule as
    *two* ring systems linked by a single bond (chromenylium + phenyl)
    and picks the neutral phenyl as parent, emitting the structurally
    correct but charge-dropping name ``(chromen-2-yl)benzene``.  The
    [O+] is silently neutralised in the process — a real naming defect.

    This classifier closes that gap by recognising the 2-aryl-
    chromenylium pattern *before* the plan search runs and emitting
    the OPSIN-compatible flavylium-family surface name directly.  No
    guard relaxation: the [o+] is explicitly enumerated in
    ``site_atom_indices`` so coverage gates in ``detect`` see the
    charge claimed.

    Detection contract
    ------------------
    Fires only when *all* of the following hold (otherwise yields nothing):

    * the molecule's only formal-charge atom is a single aromatic
      ``[o+]`` in a 6-membered ring fused to a benzene (i.e. the
      chromenylium core);
    * the chromenylium pos-2 carbon (the aromatic C ortho to ``[o+]``
      that is NOT part of the benzo ring) carries an exocyclic single
      bond to an aromatic carbon that is itself in a *separate*
      6-membered aromatic ring (the 2-aryl substituent);
    * the 2-aryl ring is a benzene (no heteroatoms, no fused partners
      of its own) — so the aryl is either unsubstituted phenyl
      (``flavylium``) or carries a single methyl (``2'-`` / ``3'-`` /
      ``4'-methylflavylium``).  Anything richer is left to the engine
      because OPSIN does not parse arbitrary primed-locant
      chromenylium derivatives unambiguously.
    * the chromenylium core itself (positions 3..8) carries no other
      exocyclic substituents; substituted-core flavylium variants are
      already named correctly by the existing substitutive flow without
      dropping the [O+].

    What this classifier *does not* claim
    -------------------------------------
    * bare chromenylium (no 2-aryl) — already handled by retained-ring
      lookup, byte-identical;
    * 2-methyl / 2-alkyl chromenylium variants — handled by the
      existing substitutive-name machinery that does not drop the
      [O+] for them;
    * 3-/4-substituted chromenyliums *with* a 2-aryl partner — these
      need primed locant + ring-locant combinations that OPSIN does
      not reliably round-trip; we leave them to fall through.
    """
    from rdkit import Chem

    # Cheap pre-filter: exactly one formal-charge atom and it must be O+.
    charged = [a for a in mol.GetAtoms() if a.GetFormalCharge() != 0]
    if len(charged) != 1:
        return
    o_plus = charged[0]
    if o_plus.GetAtomicNum() != 8 or o_plus.GetFormalCharge() != 1:
        return
    if not o_plus.GetIsAromatic() or not o_plus.IsInRing():
        return

    # Pre-filter: the only radicals (if any) must be 0 — flavylium is
    # closed-shell and the radical-cation pre-validation hook is the
    # wrong path for this motif.
    if any(a.GetNumRadicalElectrons() != 0 for a in mol.GetAtoms()):
        return

    # Substructure-match the chromenylium core SMILES against the mol.
    core = Chem.MolFromSmiles(_CHROMENYLIUM_CORE_SMILES)
    if core is None:
        return
    matches = mol.GetSubstructMatches(core)
    # Exactly one chromenylium core (a doubly-fused match would mean a
    # different ring system; the bare chromenylium covers itself).
    if len(matches) != 1:
        return
    core_atoms = matches[0]

    # Identify pos-2 in the core atom-order: in the SMILES
    # ``c1cc2ccccc2[o+]c1`` the [o+] is core_atoms[8] and pos-2 is
    # core_atoms[9] (the aromatic C between [o+] and the c1 ring-closure
    # that is *not* in the benzo fragment).  Verified by reading the
    # SMILES atom-by-atom: idx 0,1 = pos 4,3; idx 2..7 = benzo
    # (pos 4a,5,6,7,8,8a); idx 8 = O+ (pos 1); idx 9 = pos 2.
    if len(core_atoms) != 10:
        return
    pos2_idx = core_atoms[9]
    pos2_atom = mol.GetAtomWithIdx(pos2_idx)
    if not pos2_atom.GetIsAromatic() or pos2_atom.GetAtomicNum() != 6:
        return

    # Walk pos-2's exocyclic single bond and find the 2-aryl carbon.
    aryl_ipso_idx: int | None = None
    for bond in pos2_atom.GetBonds():
        other = bond.GetOtherAtom(pos2_atom)
        if other.GetIdx() in core_atoms:
            continue
        if bond.GetBondTypeAsDouble() != 1.0:
            return  # any other bond order disqualifies (architecturally cleaner)
        if other.GetAtomicNum() != 6 or not other.GetIsAromatic():
            return
        aryl_ipso_idx = other.GetIdx()
        break
    if aryl_ipso_idx is None:
        return  # bare chromenylium (no 2-substituent) — leave to retained-ring lookup

    # The 2-aryl substituent must be a *separate* benzene ring (not fused
    # to anything, no heteroatoms).  Find the ring it sits in.
    ring_info = mol.GetRingInfo()
    aryl_ring: tuple[int, ...] | None = None
    for ring in ring_info.AtomRings():
        if aryl_ipso_idx in ring and len(ring) == 6:
            # The aryl ring must be disjoint from the chromenylium core.
            if any(idx in core_atoms for idx in ring):
                continue
            # Every atom must be aromatic carbon.
            if not all(
                mol.GetAtomWithIdx(idx).GetAtomicNum() == 6
                and mol.GetAtomWithIdx(idx).GetIsAromatic()
                for idx in ring
            ):
                continue
            aryl_ring = tuple(ring)
            break
    if aryl_ring is None:
        return

    # The aryl ring atoms must each be in exactly one ring (no fusion to
    # other partners — flavylium derivatives with naphthyl, anthryl,
    # etc., 2-substituents are out of scope and OPSIN does not round-
    # trip them as ``flavylium`` derivatives anyway).
    for idx in aryl_ring:
        if ring_info.NumAtomRings(idx) != 1:
            return

    # The chromenylium core itself must be unsubstituted at pos 3..8;
    # the only allowed exocyclic bond off the core is the one at pos-2
    # leading into the aryl ring.  When pos 3/4 (or any benzo C) carries
    # a substituent the engine's existing flow correctly emits e.g.
    # ``3-methyl-2-phenylchromenylium`` *without* dropping the [O+], so
    # we must not steal that case here.
    for ci, mi in enumerate(core_atoms):
        if mi == pos2_idx:
            continue
        atom = mol.GetAtomWithIdx(mi)
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            if other.GetIdx() in core_atoms:
                continue
            # An exocyclic neighbor on a non-pos-2 core atom — out of scope.
            return

    # Now classify the substitution pattern on the aryl ring.
    # Allowed: bare phenyl (flavylium) or single methyl at 2'/3'/4' on
    # the aryl ring (2'/3'/4'-methylflavylium).  Anything else: leave it
    # to the existing engine flow.
    aryl_exo_substituents: list[tuple[int, int]] = []  # (aryl-ring-idx, exo-atom-idx)
    for idx in aryl_ring:
        ring_atom = mol.GetAtomWithIdx(idx)
        for bond in ring_atom.GetBonds():
            other = bond.GetOtherAtom(ring_atom)
            if other.GetIdx() in aryl_ring:
                continue
            if other.GetIdx() == pos2_idx:
                continue  # the bond back into the chromenylium
            aryl_exo_substituents.append((idx, other.GetIdx()))

    # Flavylium (bare 2-phenyl): no exo-substituents on the aryl ring.
    if len(aryl_exo_substituents) == 0:
        site = (o_plus.GetIdx(),)
        yield ChargeClassification(
            site_atom_indices=site,
            charge_sign="+",
            suffix_hint="aryl_chromenylium",
            locant=None,
            parent_smiles=None,
            surface_name="flavylium",
        )
        return

    # Single methyl on the aryl ring: emit 2'/3'/4'-methylflavylium.
    if len(aryl_exo_substituents) == 1:
        ring_idx, exo_idx = aryl_exo_substituents[0]
        exo_atom = mol.GetAtomWithIdx(exo_idx)
        # Must be a CH3: sp3 C, no further heavy bonds, 3 H (or implicit).
        if exo_atom.GetAtomicNum() != 6:
            return
        if exo_atom.GetIsAromatic():
            return
        if exo_atom.GetDegree() != 1:
            return
        if exo_atom.GetTotalNumHs() != 3:
            return
        if exo_atom.GetFormalCharge() != 0:
            return
        # Determine the primed locant: ipso = 1', then walk the ring to
        # find which aryl-ring atom holds the methyl.  The flavylium
        # numbering on the aryl ring is: ipso = 1', then 2'/3'/4' going
        # round (with 5'/6' on the other side).  Since methyl is a single
        # substituent, lowest-locant rule picks the smaller of the two
        # walks, giving 2'/3'/4'.
        try:
            primed = _flavylium_aryl_locant(mol, aryl_ring, aryl_ipso_idx, ring_idx)
        except Exception:
            return
        if primed is None:
            return
        site = (o_plus.GetIdx(),)
        yield ChargeClassification(
            site_atom_indices=site,
            charge_sign="+",
            suffix_hint="aryl_chromenylium",
            locant=None,
            parent_smiles=None,
            surface_name=f"{primed}'-methylflavylium",
        )
        return

    # Anything richer (multiple aryl substituents, non-methyl
    # substituents, etc.) is out of scope — fall through.
    return


def _flavylium_aryl_locant(
    mol,
    aryl_ring: tuple[int, ...],
    ipso_idx: int,
    target_idx: int,
) -> int | None:
    """Compute the primed locant (2/3/4) for a substituent on the
    2-aryl ring of a flavylium core.

    BFS distance from ``ipso_idx`` (which is 1') to ``target_idx`` along
    the 6-membered ring; locant = distance + 1.  In a 6-ring the BFS
    distances from a fixed atom are 0/1/1/2/2/3 — symmetric pairs give
    the same locant (there is only ever one canonical 2'/3'/4' for a
    single substituent), so this faithfully implements the lowest-locant
    rule for a singly-substituted phenyl.

    Returns ``None`` if the target is not on the ring or the distance
    falls outside [1, 3].
    """
    if target_idx == ipso_idx:
        return None
    ring_set = set(aryl_ring)
    adj: dict[int, list[int]] = {idx: [] for idx in aryl_ring}
    for idx in aryl_ring:
        atom = mol.GetAtomWithIdx(idx)
        for bond in atom.GetBonds():
            other_idx = bond.GetOtherAtom(atom).GetIdx()
            if other_idx in ring_set:
                adj[idx].append(other_idx)
    # BFS distance from ipso to target along the ring (max 3 in a 6-ring).
    seen = {ipso_idx: 0}
    queue = [ipso_idx]
    while queue:
        cur = queue.pop(0)
        d = seen[cur]
        for nb in adj[cur]:
            if nb in seen:
                continue
            seen[nb] = d + 1
            queue.append(nb)
    dist = seen.get(target_idx)
    if dist is None or dist < 1 or dist > 3:
        return None
    # Locant = distance + 1 (since ipso is 1', adjacent is 2', etc.).
    return dist + 1


# ---------------------------------------------------------------------------
# Stage 7: polyacylium (oxalylium / malonylium / butanedioylium / ...)
# ---------------------------------------------------------------------------


def _classify_polyacylium(mol) -> Iterable[ChargeClassification]:
    """Detect ``R(-[C+]=O)_n`` with ``n >= 2`` acyl cations.

    Pattern: every formally-charged atom in the mol is an acyl-cation
    carbon (``[C+]=O`` with no H, single-bonded to one parent atom),
    and there are at least two such carbons.  The classifier claims
    every cation C and every double-bond O that participates in the
    motif.

    The parent is a polycarboxylic acid (oxalic / malonic / succinic
    / glutaric / adipic / butanedioic / pentanedioic / hexanedioic)
    derivable by adding an OH off each [C+]=O carbon.
    """
    cation_carbons: list[int] = []
    paired_oxygens: list[int] = []
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() == 0:
            continue
        if atom.GetAtomicNum() != 6 or atom.GetFormalCharge() != 1:
            return  # not a polyacylium - bail out cleanly
        if atom.GetTotalNumHs() != 0:
            return
        if atom.IsInRing():
            return
        eq_o: int | None = None
        parent_neighbour: int | None = None
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            bt = bond.GetBondTypeAsDouble()
            if (
                other.GetAtomicNum() == 8
                and other.GetFormalCharge() == 0
                and bt == 2.0
                and other.GetDegree() == 1
            ):
                eq_o = other.GetIdx()
                continue
            if bt == 1.0:
                parent_neighbour = other.GetIdx()
                continue
        if eq_o is None or parent_neighbour is None:
            return
        cation_carbons.append(atom.GetIdx())
        paired_oxygens.append(eq_o)
    if len(cation_carbons) < 2:
        return
    site = tuple(cation_carbons + paired_oxygens)
    yield ChargeClassification(
        site_atom_indices=site,
        charge_sign="+",
        suffix_hint="polyacylium",
        locant=None,
        parent_smiles=None,
        surface_name=None,
        radical_count=0,
        site_charges=tuple([1] * len(cation_carbons) + [0] * len(paired_oxygens)),
    )


# ---------------------------------------------------------------------------
# Stage 7: multi-charge carbon (di / tri / tetra-ylium / -ide / mixed)
# ---------------------------------------------------------------------------


def _classify_polycarbon_charge(mol) -> Iterable[ChargeClassification]:
    """Detect multi-charge aliphatic carbon motifs.

    Closes the polycation/polyanion gap left open by R2-B's
    ``_classify_simple_carbon_charge`` (which intentionally bails out
    if more than one charged C is present).

    Covers (probed against OPSIN):

    * ``[CH2+]C[CH2+]``     -> ``propane-1,3-diylium``
    * ``[CH+]1CC[CH+]CC1`` -> ``cyclohexane-1,4-diylium``
    * ``[CH2-]C[CH2-]``     -> ``propane-1,3-diide``
    * ``[CH2-]CC[CH2+]``    -> ``butan-1-id-4-ylium``
    * ``[CH+3]``            -> ``methanetriylium`` (single atom, |q|>=2)
    * ``[CH2+2]``           -> ``methanediylium``
    * ``C[C+2]C``           -> ``propane-2,2-diylium``
    * ``[CH2+][CH+][CH+][CH2+]`` -> ``butane-1,2,3,4-tetraylium``

    Constraints (kept tight to avoid stealing motifs the rest of the
    engine handles):

    * pure-carbon parent (no heteroatoms, no aromatic atoms);
    * all bonds are single (so we never collide with conjugated
      cations / iminylium / acylium variants);
    * single-charged-atom |q| == 1 cases stay with R2-B's
      ``_classify_simple_carbon_charge``.
    """
    if any(atom.GetAtomicNum() != 6 for atom in mol.GetAtoms()):
        return
    if any(atom.GetIsAromatic() for atom in mol.GetAtoms()):
        return
    for bond in mol.GetBonds():
        if bond.GetBondTypeAsDouble() != 1.0:
            return
    charged_atoms = [a for a in mol.GetAtoms() if a.GetFormalCharge() != 0]
    if not charged_atoms:
        return
    # |q|==1 single-charged-atom case belongs to the simple classifier.
    if len(charged_atoms) == 1 and abs(charged_atoms[0].GetFormalCharge()) == 1:
        return
    for a in charged_atoms:
        if a.GetNumRadicalElectrons() != 0:
            return
    site_indices = tuple(a.GetIdx() for a in charged_atoms)
    site_charges = tuple(a.GetFormalCharge() for a in charged_atoms)
    has_pos = any(c > 0 for c in site_charges)
    has_neg = any(c < 0 for c in site_charges)
    if has_pos and has_neg:
        suffix = "mixed_id_ylium"
        sign: ChargeSign = "+"
    elif has_pos:
        suffix = "diylium"
        sign = "+"
    else:
        suffix = "diide"
        sign = "-"
    yield ChargeClassification(
        site_atom_indices=site_indices,
        charge_sign=sign,
        suffix_hint=suffix,
        locant=None,
        parent_smiles=None,
        surface_name=None,
        radical_count=0,
        site_charges=site_charges,
    )


# ---------------------------------------------------------------------------
# P-72 / P-73: polycharged / zwitterionic species
# ---------------------------------------------------------------------------
#
# Four structural families closed here, each generalising a P-72/P-73 rule
# rather than pinning a molecule:
#
#  (A) carbanion zwitterion  — a senior ``[C-]`` anion (suffix ``-ide``) that
#      coexists with a cationic onium centre expressed as a substituent
#      prefix (``...ammoniumyl`` / ``...oxoniumyl`` / ``...sulfaniumyl``).
#      P-73/P-74: anions outrank cations for suffix citation, so the carbanion
#      takes the suffix and the cation becomes a detachable prefix.
#      Example: ``C[N+](C)(C)[C-](C)C`` -> ``2-(trimethylammoniumyl)propan-2-ide``.
#
#  (B) substituted single-carbon dianion — one ``[C-2]`` carbon carrying only
#      substituents (no carbon backbone of its own) names as
#      ``<prefixes>methanediide``.
#      Example: ``[C-2](c1ccccc1)c1ccccc1`` -> ``diphenylmethanediide``.
#
#  (C) di-charged 2-atom nitrogen backbone — a diazane (N-N) or diazene (N=N)
#      hydride whose two N atoms (or one N at |q|=2) carry like charges, named
#      ``<prefixes>diazane-1,1-diide`` / ``<prefixes>diazene-1,2-diium``.
#      Examples: ``CC(=O)N[N-2]`` -> ``acetyldiazane-1,1-diide``;
#                ``C[N+](C)=[N+](C)C`` -> ``tetramethyldiazene-1,2-diium``.
#
#  (D) ring polycarbanion — two or more like-charged ring carbanions on a
#      (possibly partially hydrogenated) carbocyclic ring, named
#      ``<parent>-x,y-diide`` where the neutralised parent is whatever the
#      engine names the de-charged ring as.
#      Example: ``[CH-]1C=C[CH-]C2=CC=CC=C12`` ->
#               ``1,4-dihydronaphthalene-1,4-diide``.
#
# All four are CLOSED-SHELL (GetNumRadicalElectrons() == 0); the free-valence
# guard is untouched.

# Onium cation centre -> substituent-prefix base name (the ``-yl`` form of the
# parent cation, P-73.2.2.1.1 / Table 7.3).  Used by the carbanion-zwitterion
# renderer to express the cation as a detachable prefix.
_ONIUM_PREFIX_BASE: dict[str, str] = {
    "N": "ammoniumyl",   # quaternary N+ -> ...ammoniumyl (azaniumyl synonym)
    "O": "oxidaniumyl",  # R3O+ -> ...oxidaniumyl
    "S": "sulfaniumyl",  # R3S+ -> ...sulfaniumyl
    "P": "phosphaniumyl",
    "Se": "selaniumyl",
    "As": "arsaniumyl",
}


def _classify_carbanion_zwitterion(mol) -> Iterable[ChargeClassification]:
    """Detect a carbanion ``[C-]`` zwitterion with a cationic onium centre.

    Structural contract (all must hold):

    * net molecular charge is 0 (true zwitterion — equal +/- charge);
    * exactly one negatively-charged atom, a carbon with charge -1 and no
      radical electrons (the senior ``-ide`` anion);
    * exactly one positively-charged atom, an onium centre whose element is
      in :data:`_ONIUM_PREFIX_BASE` (N/O/S/P/Se/As), charge +1, not in a
      ring, single-bonded to the carbanion's parent carbon skeleton via a
      single bond, and bearing no H (a fully-substituted onium that names as
      a ``...iumyl`` substituent prefix);
    * the cation centre attaches *directly* to the carbanion carbon (so the
      carbanion's parent carries the cation as a substituent at the anion
      locant) — keeps the renderer's locant logic exact.

    The carbanion + its all-carbon neighbourhood form the parent; the onium
    centre and everything beyond it become the substituent prefix.

    Per P-73/P-74 the anion is senior and takes the suffix; the cation is the
    detachable prefix.  This is exactly OPSIN's PIN for the family.
    """
    charged = [a for a in mol.GetAtoms() if a.GetFormalCharge() != 0]
    if len(charged) != 2:
        return
    net = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    if net != 0:
        return
    neg = [a for a in charged if a.GetFormalCharge() == -1]
    pos = [a for a in charged if a.GetFormalCharge() == 1]
    if len(neg) != 1 or len(pos) != 1:
        return
    c_minus = neg[0]
    onium = pos[0]
    if c_minus.GetAtomicNum() != 6:
        return
    if c_minus.GetNumRadicalElectrons() != 0:
        return
    if onium.GetNumRadicalElectrons() != 0:
        return
    if onium.GetSymbol() not in _ONIUM_PREFIX_BASE:
        return
    if onium.IsInRing():
        return
    if onium.GetTotalNumHs() != 0:
        return
    # The onium must attach directly to the carbanion carbon via a single bond.
    bond = mol.GetBondBetweenAtoms(c_minus.GetIdx(), onium.GetIdx())
    if bond is None or bond.GetBondTypeAsDouble() != 1.0:
        return
    # Every other heavy bond on the carbanion carbon must be to carbon (its
    # parent skeleton) — so the carbanion's parent is a hydrocarbon and the
    # only non-carbon neighbour is the onium cation.
    for nb in c_minus.GetNeighbors():
        if nb.GetIdx() == onium.GetIdx():
            continue
        if nb.GetAtomicNum() != 6:
            return
    yield ChargeClassification(
        site_atom_indices=(c_minus.GetIdx(), onium.GetIdx()),
        charge_sign="-",
        suffix_hint="carbanion_zwitterion",
        locant=None,
        parent_smiles=None,
        surface_name=None,
        radical_count=0,
        site_charges=(-1, 1),
    )


def _classify_substituted_carbon_dianion(mol) -> Iterable[ChargeClassification]:
    """Detect a single ``[C-2]`` carbon carrying only substituents.

    The charged carbon has charge -2, zero H, no radical electrons, and every
    heavy neighbour is a substituent (the carbon itself is the whole parent
    skeleton — methane).  ``[C-2](Ph)Ph`` -> ``diphenylmethanediide``.

    Scope is deliberately narrow: this classifier fires ONLY when the lone
    |q|==2 carbon carries at least one substituent that
    ``_classify_polycarbon_charge`` cannot handle — i.e. an aromatic atom or a
    heteroatom is present somewhere in the molecule.  Pure-aliphatic-carbon
    dianions (``[CH2-2]`` bare, ``[C-2](C)C``, ``[CH-]CC[CH-]`` two-site) stay
    with the existing all-carbon polycarbon path so their established output
    (``methanediide`` / ``propane-2,2-diide`` / ...) is byte-for-byte
    unchanged.
    """
    charged = [a for a in mol.GetAtoms() if a.GetFormalCharge() != 0]
    if len(charged) != 1:
        return
    c = charged[0]
    if c.GetAtomicNum() != 6 or c.GetFormalCharge() != -2:
        return
    if c.GetTotalNumHs() != 0 or c.GetNumRadicalElectrons() != 0:
        return
    if c.IsInRing():
        return
    heavy_nbs = [nb for nb in c.GetNeighbors() if nb.GetAtomicNum() != 1]
    if not heavy_nbs:
        return  # bare [CH2-2] handled by _classify_polycarbon_charge
    # Each neighbour bond must be single (substituents off a methane centre).
    for nb in heavy_nbs:
        bond = mol.GetBondBetweenAtoms(c.GetIdx(), nb.GetIdx())
        if bond is None or bond.GetBondTypeAsDouble() != 1.0:
            return
    # Defer pure-aliphatic-carbon dianions to the existing polycarbon path.
    has_aromatic = any(a.GetIsAromatic() for a in mol.GetAtoms())
    has_hetero = any(a.GetAtomicNum() not in (1, 6) for a in mol.GetAtoms())
    if not has_aromatic and not has_hetero:
        return
    yield ChargeClassification(
        site_atom_indices=(c.GetIdx(),),
        charge_sign="-",
        suffix_hint="substituted_carbon_dianion",
        locant=None,
        parent_smiles=None,
        surface_name=None,
        radical_count=0,
        site_charges=(-2,),
    )


def _classify_dinitrogen_polycharge(mol) -> Iterable[ChargeClassification]:
    """Detect a di-charged 2-nitrogen backbone (diazane / diazene).

    Two structural shapes, both with a single N-N bond core:

    * **diazene-diium**: two adjacent N atoms joined by a *double* bond, each
      with charge +1, no H, degree 3 (=N plus two substituents).  Parent =
      diazene (N=N); suffix ``-1,2-diium``.
      ``C[N+](C)=[N+](C)C`` -> ``tetramethyldiazene-1,2-diium``.

    * **diazane-diide**: two adjacent N atoms joined by a *single* bond where
      one N carries charge -2 (no H) and the other is neutral.  Parent =
      diazane (N-N); suffix ``-1,1-diide`` on the dianionic N.
      ``CC(=O)N[N-2]`` -> ``acetyldiazane-1,1-diide``.

    Both N atoms (and any per-site charge) are claimed; everything off the two
    N atoms becomes a substituent prefix.  Closed-shell only.
    """
    if any(a.GetNumRadicalElectrons() != 0 for a in mol.GetAtoms()):
        return
    nitrogens = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    # ---- diazene-diium: two N+ joined by N=N ----
    n_plus = [a for a in nitrogens if a.GetFormalCharge() == 1]
    charged_all = [a for a in mol.GetAtoms() if a.GetFormalCharge() != 0]
    if (
        len(n_plus) == 2
        and len(charged_all) == 2
        and all(a.GetFormalCharge() == 1 for a in charged_all)
    ):
        a0, a1 = n_plus
        bond = mol.GetBondBetweenAtoms(a0.GetIdx(), a1.GetIdx())
        if (
            bond is not None
            and bond.GetBondTypeAsDouble() == 2.0
            and a0.GetTotalNumHs() == 0
            and a1.GetTotalNumHs() == 0
            and not a0.IsInRing()
            and not a1.IsInRing()
        ):
            yield ChargeClassification(
                site_atom_indices=(a0.GetIdx(), a1.GetIdx()),
                charge_sign="+",
                suffix_hint="diazene_diium",
                locant=None,
                parent_smiles=None,
                surface_name=None,
                radical_count=0,
                site_charges=(1, 1),
            )
            return
    # ---- diazane-diide: one N(-2) single-bonded to a neutral N ----
    n_dianion = [a for a in nitrogens if a.GetFormalCharge() == -2]
    if len(n_dianion) == 1 and len(charged_all) == 1:
        nd = n_dianion[0]
        if nd.GetTotalNumHs() != 0 or nd.IsInRing():
            return
        n_neighbours = [
            nb for nb in nd.GetNeighbors() if nb.GetAtomicNum() == 7
        ]
        if len(n_neighbours) != 1:
            return
        n_other = n_neighbours[0]
        if n_other.GetFormalCharge() != 0 or n_other.IsInRing():
            return
        bond = mol.GetBondBetweenAtoms(nd.GetIdx(), n_other.GetIdx())
        if bond is None or bond.GetBondTypeAsDouble() != 1.0:
            return
        # The dianionic N must have no heavy neighbour other than the partner N
        # (its two charges replace its two N-H bonds).
        nd_heavy = [nb for nb in nd.GetNeighbors() if nb.GetAtomicNum() != 1]
        if len(nd_heavy) != 1:
            return
        yield ChargeClassification(
            site_atom_indices=(nd.GetIdx(), n_other.GetIdx()),
            charge_sign="-",
            suffix_hint="diazane_diide",
            locant=None,
            parent_smiles=None,
            surface_name=None,
            radical_count=0,
            site_charges=(-2, 0),
        )
        return


def _classify_ring_polycarbanion(mol) -> Iterable[ChargeClassification]:
    """Detect 2+ like-charged ring carbanions on a carbocyclic parent.

    Generalises ``_classify_polycarbon_charge`` (which rejects aromatic atoms
    and multiple bonds) to ring carbanion dianions whose neutralised parent
    the engine can name directly (e.g. a partially-hydrogenated fused ring).

    Contract:

    * every charged atom is a ring carbon with charge -1 and no radical
      electrons (≥ 2 of them);
    * the whole molecule is carbocyclic (all heavy atoms are carbon) — keeps
      us clear of heteroatom rings the engine numbers differently;
    * neutralising each charged carbon yields a parent the engine names;
    * the per-site ring locants are taken from that parent's numbering.

    ``[CH-]1C=C[CH-]C2=CC=CC=C12`` -> ``1,4-dihydronaphthalene-1,4-diide``.
    """
    if any(a.GetNumRadicalElectrons() != 0 for a in mol.GetAtoms()):
        return
    if any(a.GetAtomicNum() != 6 for a in mol.GetAtoms()):
        return
    charged = [a for a in mol.GetAtoms() if a.GetFormalCharge() != 0]
    if len(charged) < 2:
        return
    if any(a.GetFormalCharge() != -1 for a in charged):
        return
    if any(not a.IsInRing() for a in charged):
        return
    site_indices = tuple(a.GetIdx() for a in charged)
    yield ChargeClassification(
        site_atom_indices=site_indices,
        charge_sign="-",
        suffix_hint="ring_polycarbanion",
        locant=None,
        parent_smiles=None,
        surface_name=None,
        radical_count=0,
        site_charges=tuple(-1 for _ in charged),
    )


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------


_RETAINED_ACYL_BASE: dict[str, str] = {
    "acetic acid": "acet",
    "benzoic acid": "benzo",
    "formic acid": "form",
    "propionic acid": "propanoyl-but-stem-handled-systematically",
    "butyric acid": "butyryl-but-stem-handled-systematically",
    "oxalic acid": "oxalo-handled-systematically",
}

# Retained acid -> retained acylium surface name (the only forms OPSIN
# parses unambiguously into the acylium SMILES used as ground truth).
_RETAINED_ACID_TO_ACYLIUM: dict[str, str] = {
    "acetic acid": "acetylium",
    "benzoic acid": "benzoylium",
    "formic acid": "formylium",
}


def _render(
    cls: ChargeClassification,
    mol,
    strategy,
    session,
    depth: int,
) -> str | None:
    """Render the surface name for a single classification."""
    if cls.surface_name is not None:
        return cls.surface_name
    if cls.suffix_hint in ("ylium", "ide"):
        return _render_simple_carbon(cls, mol, strategy, session, depth)
    if cls.suffix_hint == "aromatic_ring_ylium":
        return _render_aromatic_ring_cation(cls, mol, strategy, session, depth)
    if cls.suffix_hint == "diazonium":
        return _render_diazonium(cls, mol, strategy, session, depth)
    if cls.suffix_hint == "amidinium":
        return _render_amidinium(cls, mol, strategy, session, depth)
    if cls.suffix_hint == "acylium":
        return _render_acylium(cls, mol, strategy, session, depth)
    # ---- Stage 7 motifs ----
    if cls.suffix_hint == "aminylium":
        return _render_aminylium(cls, mol, strategy, session, depth)
    if cls.suffix_hint == "azaniumyl":
        return _render_azaniumyl(cls, mol, strategy, session, depth)
    if cls.suffix_hint == "iminylium":
        return _render_iminylium(cls, mol, strategy, session, depth)
    if cls.suffix_hint == "amidylium":
        return _render_amidylium(cls, mol, strategy, session, depth)
    if cls.suffix_hint in ("diylium", "diide", "mixed_id_ylium"):
        return _render_polycarbon(cls, mol, strategy, session, depth)
    # ---- P-72/P-73 polycharged / zwitterionic ----
    if cls.suffix_hint == "carbanion_zwitterion":
        return _render_carbanion_zwitterion(cls, mol, strategy, session, depth)
    if cls.suffix_hint == "substituted_carbon_dianion":
        return _render_substituted_carbon_dianion(cls, mol, strategy, session, depth)
    if cls.suffix_hint == "diazene_diium":
        return _render_diazene_diium(cls, mol, strategy, session, depth)
    if cls.suffix_hint == "diazane_diide":
        return _render_diazane_diide(cls, mol, strategy, session, depth)
    if cls.suffix_hint == "ring_polycarbanion":
        return _render_ring_polycarbanion(cls, mol, strategy, session, depth)
    if cls.suffix_hint == "polyacylium":
        return _render_polyacylium(cls, mol, strategy, session, depth)
    if cls.suffix_hint == "substituted_boranuide":
        return _render_substituted_boranuide(cls, mol, strategy, session, depth)
    if cls.suffix_hint in ("acidic_anion_olate", "acidic_anion_carboxylate"):
        return _render_acidic_anion(cls, mol, strategy, session, depth)
    if cls.suffix_hint == "carbamate_anion":
        return _render_carbamate_anion(cls, mol, strategy, session, depth)
    if cls.suffix_hint == "amine_anion":
        return _render_amine_anion(cls, mol, strategy, session, depth)
    if cls.suffix_hint == "amide_anion":
        return _render_amide_anion(cls, mol, strategy, session, depth)
    return None


def _render_acidic_anion(
    cls: ChargeClassification,
    mol,
    strategy,
    session,
    depth: int,
) -> str | None:
    """Re-protonate the deprotonated atom and drive the engine on the
    resulting neutral parent with ``OutputForm.ANION``.

    The standard suffix machinery (SUFFIX_VARIANT_TABLE) and the retained
    acid stem table do the work; this function just provides the
    structural pivot.
    """
    from iupac_namer.engine import name as _recursive_name
    from iupac_namer.assembly import assemble
    from iupac_namer.types import OutputForm

    # Re-protonate every claimed deprotonation site (1 site -> mono anion
    # such as ``benzenethiolate``; N sites -> multiplicative anion such as
    # ``benzene-1,2-dithiolate``).  Driving the engine on the fully neutral
    # parent with OutputForm.ANION lets the standard suffix machinery emit
    # the multiplied anion suffix.
    parent_smiles = _neutral_skeleton_smiles(
        mol,
        {a_idx: {"charge": 0, "no_implicit": False}
         for a_idx in cls.site_atom_indices},
    )
    if parent_smiles is None:
        return None
    from rdkit import Chem
    parent_mol = Chem.MolFromSmiles(parent_smiles)
    if parent_mol is None:
        return None
    try:
        tree = _recursive_name(
            parent_mol,
            strategy,
            OutputForm.ANION,
            free_valence=None,
            decision_ctx=None,
            _session=session,
            _depth=depth + 1,
        )
    except Exception:
        return None
    name = assemble(tree)
    if name is None or "NAMING ERROR" in name:
        return None
    return name


def _render_amine_anion(
    cls: ChargeClassification,
    mol,
    strategy,
    session,
    depth: int,
) -> str | None:
    """Render the deprotonated-amine PIN (``methanaminide`` etc.).

    Re-protonate the N⁻ to a neutral amine and drive the engine with
    ``OutputForm.ANION``.  The ``("amine", OutputForm.ANION) → "aminide"``
    suffix-variant entry, plus the retained-amine ANION skip in
    ``_generate_retained_plans``, produce the systematic ``-aminide`` PIN
    for primary, secondary, alkyl, aryl, and ring amine anions.
    """
    from iupac_namer.engine import name as _recursive_name
    from iupac_namer.assembly import assemble
    from iupac_namer.types import OutputForm
    from rdkit import Chem

    n_idx = cls.site_atom_indices[0]
    parent_smiles = _neutral_skeleton_smiles(
        mol,
        {n_idx: {"charge": 0, "no_implicit": False}},
    )
    if parent_smiles is None:
        return None
    parent_mol = Chem.MolFromSmiles(parent_smiles)
    if parent_mol is None:
        return None
    try:
        tree = _recursive_name(
            parent_mol,
            strategy,
            OutputForm.ANION,
            free_valence=None,
            decision_ctx=None,
            _session=session,
            _depth=depth + 1,
        )
    except Exception:
        return None
    name = assemble(tree)
    if name is None or "NAMING ERROR" in name:
        return None
    # Guard: only accept the result if the suffix transform actually fired
    # (the name must end in the anion form, not a stray neutral amine).
    if not name.endswith("aminide"):
        return None
    return name


def _render_amide_anion(
    cls: ChargeClassification,
    mol,
    strategy,
    session,
    depth: int,
) -> str | None:
    """Render the deprotonated primary-amide PIN (``acetylamide`` etc.).

    Carve the corresponding acid (replace the N⁻ with -OH on the acyl C),
    name its acyl group via the engine's standard acid→acyl machinery, and
    append ``amide``.  This produces the OPSIN-parseable ``{acyl}amide``
    form (``acetylamide``, ``benzoylamide``, ``formylamide``,
    ``propanoylamide`` …); the systematic ``-amidide`` promotion is not an
    OPSIN-parseable name.
    """
    from iupac_namer.engine import (
        name as _recursive_name,
        _acid_name_to_acyl,
    )
    from iupac_namer.assembly import assemble
    from iupac_namer.types import OutputForm
    from rdkit import Chem

    n_idx = cls.site_atom_indices[0]
    # Replace the N⁻ (and its H) with an -OH oxygen on the acyl carbon by
    # deleting the N and grafting an O: build the acid surrogate.
    rw = Chem.RWMol(mol)
    n_atom = rw.GetAtomWithIdx(n_idx)
    acyl_c_idx = None
    for nb in n_atom.GetNeighbors():
        if nb.GetAtomicNum() == 6:
            acyl_c_idx = nb.GetIdx()
            break
    if acyl_c_idx is None:
        return None
    # Add the hydroxyl oxygen, bond it to the acyl C, then remove the N.
    new_o = rw.AddAtom(Chem.Atom(8))
    rw.GetAtomWithIdx(new_o).SetNumExplicitHs(1)
    rw.GetAtomWithIdx(new_o).SetNoImplicit(True)
    rw.AddBond(acyl_c_idx, new_o, Chem.BondType.SINGLE)
    rw.RemoveAtom(n_idx)
    acid_mol = rw.GetMol()
    try:
        Chem.SanitizeMol(acid_mol)
    except Exception:
        return None
    try:
        acid_tree = _recursive_name(
            acid_mol,
            strategy,
            OutputForm.STANDALONE,
            free_valence=None,
            decision_ctx=None,
            _session=session,
            _depth=depth + 1,
        )
    except Exception:
        return None
    acid_name = assemble(acid_tree)
    if acid_name is None or "NAMING ERROR" in acid_name:
        return None
    acyl_name = _acid_name_to_acyl(acid_name)
    if acyl_name is None:
        return None
    return f"{acyl_name}amide"


def _render_carbamate_anion(
    cls: ChargeClassification,
    mol,
    strategy,
    session,
    depth: int,
) -> str | None:
    """Render ``N-propylcarbamate``, ``N,N-dimethylcarbamate``, etc.

    The classifier has already confirmed the R-NH-C(=O)-O⁻ (or R2N-C(=O)-O⁻)
    pattern.  This renderer carves the N-substituents from the N atom and
    names each one as a SUBSTITUENT, then assembles the name using the same
    ``_carbamic_n_subs_to_prefix`` logic as the ester carbamate FC path.
    """
    from rdkit import Chem
    from iupac_namer.engine import name as _recursive_name
    from iupac_namer.assembly import assemble, _carbamic_n_subs_to_prefix
    from iupac_namer.types import OutputForm, FreeValenceInfo, SubstituentMethod

    site_set = set(cls.site_atom_indices)

    # Locate the N atom in site_set.
    n_atom = None
    acyl_c = None
    for idx in site_set:
        a = mol.GetAtomWithIdx(idx)
        if a.GetAtomicNum() == 7:
            n_atom = a
        elif a.GetAtomicNum() == 6:
            acyl_c = a
    if n_atom is None or acyl_c is None:
        return None

    # Carve each N-substituent (heavy neighbours of N that are not acyl_c).
    n_sub_names: list[str] = []
    for nb in n_atom.GetNeighbors():
        if nb.GetAtomicNum() == 1:
            continue
        if nb.GetIdx() == acyl_c.GetIdx():
            continue
        nb_idx = nb.GetIdx()
        # BFS from nb_idx, excluding n_atom.
        visited: set[int] = {n_atom.GetIdx()}
        stack = [nb_idx]
        sub_atoms: list[int] = []
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            sub_atoms.append(cur)
            for nb2 in mol.GetAtomWithIdx(cur).GetNeighbors():
                stack.append(nb2.GetIdx())
        # Build sub-mol.
        rw = Chem.RWMol(mol)
        bond = rw.GetBondBetweenAtoms(n_atom.GetIdx(), nb_idx)
        if bond is None:
            return None
        rw.RemoveBond(n_atom.GetIdx(), nb_idx)
        keep = set(sub_atoms)
        delete_indices = sorted(
            (a.GetIdx() for a in rw.GetAtoms() if a.GetIdx() not in keep),
            reverse=True,
        )
        for d in delete_indices:
            rw.RemoveAtom(d)
        try:
            Chem.SanitizeMol(rw)
        except Exception:
            return None
        sub_mol = rw.GetMol()
        new_attachment_idx = sorted(keep).index(nb_idx)
        fv = FreeValenceInfo(
            bond_orders=(1,),
            method=SubstituentMethod.ALKYL,
            attachment_atoms_in_fragment=(new_attachment_idx,),
        )
        try:
            sub_tree = _recursive_name(
                sub_mol,
                strategy,
                OutputForm.SUBSTITUENT,
                free_valence=fv,
                decision_ctx=None,
                _session=session,
                _depth=depth + 1,
            )
        except Exception:
            return None
        sub_name = assemble(sub_tree)
        if sub_name is None or "NAMING ERROR" in sub_name:
            return None
        if sub_name.startswith("-"):
            sub_name = sub_name[1:]
        n_sub_names.append(sub_name)

    if not n_sub_names:
        return None

    n_prefix = _carbamic_n_subs_to_prefix(n_sub_names)
    if n_prefix:
        return f"{n_prefix}carbamate"
    return "carbamate"


def _render_substituted_boranuide(
    cls: ChargeClassification,
    mol,
    strategy,
    session,
    depth: int,
) -> str | None:
    """Render ``methylboranuide`` / ``dimethylboranuide`` etc.

    For each heavy neighbour of the B atom, carve the substituent fragment
    (everything reachable from the neighbour without crossing back through
    B) and recursively name it as a SUBSTITUENT.  Then alphabetise +
    multiply via the standard prefix machinery and emit
    ``<prefixes>boranuide``.
    """
    from rdkit import Chem
    from iupac_namer.engine import name as _recursive_name
    from iupac_namer.assembly import (
        assemble,
        merge_identical_prefixes,
        render_merged_prefixes,
    )
    from iupac_namer.types import OutputForm, FreeValenceInfo, SubstituentMethod

    b_idx = cls.site_atom_indices[0]
    b_atom = mol.GetAtomWithIdx(b_idx)
    heavy_nbs = [nb for nb in b_atom.GetNeighbors() if nb.GetAtomicNum() != 1]

    assembled_prefixes: list[tuple[str, tuple]] = []
    for nb in heavy_nbs:
        nb_idx = nb.GetIdx()
        # Walk the substituent: all atoms reachable from nb without
        # crossing through b_idx.
        visited: set[int] = {b_idx}
        stack = [nb_idx]
        sub_atoms: list[int] = []
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            sub_atoms.append(cur)
            for nb2 in mol.GetAtomWithIdx(cur).GetNeighbors():
                if nb2.GetIdx() == b_idx:
                    continue
                stack.append(nb2.GetIdx())
        # Build the substituent fragment as a fresh mol so atom indices
        # are dense; the Chem.RWMol path lets us strip the B link.
        rw = Chem.RWMol(mol)
        # Remove the B-nb bond so the fragment becomes its own connected
        # component; we keep b_idx for now and delete it after.
        bond = rw.GetBondBetweenAtoms(b_idx, nb_idx)
        if bond is None:
            return None
        rw.RemoveBond(b_idx, nb_idx)
        rw.GetAtomWithIdx(b_idx).SetFormalCharge(0)
        # Remove B and any other-fragment atoms; only keep the substituent
        # connected component containing nb_idx.
        keep = set(sub_atoms)
        delete_indices = sorted(
            (a.GetIdx() for a in rw.GetAtoms() if a.GetIdx() not in keep),
            reverse=True,
        )
        for d in delete_indices:
            rw.RemoveAtom(d)
        try:
            Chem.SanitizeMol(rw)
        except Exception:
            return None
        sub_mol = rw.GetMol()
        # The atom that was attached to B is the new attachment point.
        # After deletion its index is shifted; rebuild via canonical SMILES.
        # Simpler: remap the original nb_idx to its new index in sub_mol.
        # Since we kept only `keep` and deleted the rest in reverse order,
        # the ordering of atoms in sub_mol matches the order they appeared
        # in the original mol (filtered by keep).  So the new index of
        # nb_idx is its position in `sorted(keep)`.
        new_attachment_idx = sorted(keep).index(nb_idx)
        fv = FreeValenceInfo(
            bond_orders=(1,),
            method=SubstituentMethod.ALKYL,
            attachment_atoms_in_fragment=(new_attachment_idx,),
        )
        try:
            sub_tree = _recursive_name(
                sub_mol,
                strategy,
                OutputForm.SUBSTITUENT,
                free_valence=fv,
                decision_ctx=None,
                _session=session,
                _depth=depth + 1,
            )
        except Exception:
            return None
        sub_name = assemble(sub_tree)
        if sub_name is None or "NAMING ERROR" in sub_name:
            return None
        # Strip a leading hyphen if the substituent emitted one (free-
        # valence locant rendered with leading hyphen).
        if sub_name.startswith("-"):
            sub_name = sub_name[1:]
        assembled_prefixes.append((sub_name, ()))
    if not assembled_prefixes:
        return None
    merged = merge_identical_prefixes(assembled_prefixes)
    merged.sort(key=lambda m: m.sort_name)
    return f"{render_merged_prefixes(merged)}boranuide"


def _neutral_skeleton_smiles(mol, atom_changes: dict[int, dict]) -> str | None:
    """Build a canonical SMILES of the neutralized skeleton.

    ``atom_changes`` maps atom-idx -> dict of mutations:
        {"charge": int, "explicit_h": int, "delete": True}

    Atoms with ``delete: True`` are removed (along with all bonds to
    them).  Returns the canonical SMILES string, or ``None`` on
    sanitisation failure.
    """
    from rdkit import Chem

    rw = Chem.RWMol(mol)
    delete_indices = sorted(
        (idx for idx, ch in atom_changes.items() if ch.get("delete")),
        reverse=True,
    )
    # Apply non-delete mutations first so the indices are still valid.
    for idx, ch in atom_changes.items():
        if ch.get("delete"):
            continue
        atom = rw.GetAtomWithIdx(idx)
        if "charge" in ch:
            atom.SetFormalCharge(ch["charge"])
        if "explicit_h" in ch:
            atom.SetNumExplicitHs(ch["explicit_h"])
        if "no_implicit" in ch:
            atom.SetNoImplicit(ch["no_implicit"])
    for idx in delete_indices:
        rw.RemoveAtom(idx)
    try:
        Chem.SanitizeMol(rw)
    except Exception:
        return None
    return Chem.MolToSmiles(rw)


def _drive_engine(parent_smiles: str, strategy, session, depth: int) -> str | None:
    """Drive the engine recursively on a constructed neutral parent."""
    from rdkit import Chem

    parent_mol = Chem.MolFromSmiles(parent_smiles)
    if parent_mol is None:
        return None
    from iupac_namer.engine import name as _recursive_name
    from iupac_namer.assembly import assemble

    try:
        tree = _recursive_name(
            parent_mol,
            strategy,
            OutputForm.STANDALONE,
            free_valence=None,
            decision_ctx=None,
            _session=session,
            _depth=depth + 1,
        )
        return assemble(tree)
    except Exception:
        return None


# ---------- ylium / ide ----------


def _render_simple_carbon(
    cls: ChargeClassification,
    mol,
    strategy,
    session,
    depth: int,
) -> str | None:
    """Render ``methylium`` / ``methanide`` / ``ethan-1-ylium`` /
    ``cyclohexan-1-ide`` etc."""
    c_idx = cls.site_atom_indices[0]
    # Methyl special case: single-atom skeleton -> retained name.
    if mol.GetNumHeavyAtoms() == 1:
        return "methylium" if cls.is_cation else "methanide"
    parent_smiles = _neutral_skeleton_smiles(
        mol,
        {c_idx: {"charge": 0, "no_implicit": False}},
    )
    if parent_smiles is None:
        return None
    parent_name = _drive_engine(parent_smiles, strategy, session, depth)
    if parent_name is None:
        return None
    # We need to know the locant of c_idx in the parent's numbering.
    # For this module's scope (single charged C on a chain or
    # cyclohexane), locant 1 is always correct: the alpha carbon of
    # the chain is numbered 1, and a monosubstituted cyclohexane is
    # numbered with the substituent at 1.  The unit tests pin every
    # locant we emit through the OPSIN round-trip.
    locant = 1
    suffix = "ylium" if cls.is_cation else "ide"
    return _splice_alkane_suffix(parent_name, locant, suffix)


def _splice_alkane_suffix(parent_name: str, locant: int, suffix: str) -> str:
    """Splice ``parent_name + locant + suffix`` into IUPAC form.

    - ``ethane`` + 1 + ``ylium`` -> ``ethan-1-ylium``
    - ``cyclohexane`` + 1 + ``ide`` -> ``cyclohexan-1-ide``

    The trailing ``e`` of ``-ane`` elides before the vowel-initial
    suffixes ``ylium`` / ``ide``.
    """
    if parent_name.endswith("ane"):
        stem = parent_name[:-1]  # "ethane" -> "ethan"
        return f"{stem}-{locant}-{suffix}"
    # Aromatic / retained parents (benzene, naphthalene) don't elide
    # the trailing ``e`` quite the same way; we fall back to dash form.
    return f"{parent_name}-{locant}-{suffix}"


# ---------- carbocyclic ring ylium (tropylium etc.) ----------


def _render_aromatic_ring_cation(
    cls: ChargeClassification,
    mol,
    strategy,
    session,
    depth: int,
) -> str | None:
    """Render a monocyclic carbocyclic ring cation as ``cyclo<stem>-...-1-ylium``.

    The cationic carbon is fixed at locant 1 (the ``-ylium`` suffix outranks
    the ene unsaturation for low locants, per P-31.1.4 numbering); the ring
    is walked in both directions from the cation and the direction giving the
    lowest double-bond locant set is chosen.  Stems and multipliers come from
    the shared data tables so every ring size generalises (no per-size pins).

    The plan-search SUBSTITUENT path saturates unsaturated ring substituents
    (a separate, broader limitation), so this renderer composes the PIN
    directly from the kekulised ring graph rather than driving the engine.
    Every emitted name is validated by the OPSIN round-trip in the eval.
    """
    from rdkit import Chem
    from iupac_namer.data_loader import get_chain_stem, get_multiplier

    c_idx = cls.site_atom_indices[0]
    ri = mol.GetRingInfo()
    rings = ri.AtomRings()
    if len(rings) != 1:
        return None
    ring = rings[0]
    n = len(ring)
    stem = get_chain_stem(n)
    if stem is None:
        return None

    km = Chem.RWMol(mol)
    try:
        Chem.Kekulize(km, clearAromaticFlags=True)
    except Exception:
        return None
    ring_set = set(ring)

    def ring_neighbours(idx: int) -> list[int]:
        return [
            nb.GetIdx()
            for nb in km.GetAtomWithIdx(idx).GetNeighbors()
            if nb.GetIdx() in ring_set
        ]

    start = c_idx
    start_nbrs = ring_neighbours(start)
    if len(start_nbrs) != 2:
        return None  # not a simple ring (degree-2 expected at every ring atom)

    best_dbl: tuple[int, ...] | None = None
    for first in start_nbrs:
        # Walk the cycle, numbering the cation as position 1.
        order = [start]
        prev, cur = start, first
        while cur != start:
            order.append(cur)
            nxts = [x for x in ring_neighbours(cur) if x != prev]
            if not nxts:
                break
            prev, cur = cur, nxts[0]
        if len(order) != n:
            continue
        dbl: list[int] = []
        for k in range(n):
            a1 = order[k]
            a2 = order[(k + 1) % n]
            b = km.GetBondBetweenAtoms(a1, a2)
            if b is not None and b.GetBondTypeAsDouble() == 2.0:
                dbl.append(k + 1)
        key = tuple(sorted(dbl))
        if best_dbl is None or key < best_dbl:
            best_dbl = key
    if not best_dbl:
        return None

    locs = ",".join(str(x) for x in best_dbl)
    n_dbl = len(best_dbl)
    mult = "" if n_dbl == 1 else (get_multiplier(n_dbl) or "")
    if n_dbl > 1 and not mult:
        return None
    # The euphonic "a" interfix precedes a multiplied "-ene" (cyclohepta-,
    # cyclopenta-); a single "-ene" attaches directly (cycloprop-2-en-).
    link = "a" if mult else ""
    return f"cyclo{stem}{link}-{locs}-{mult}en-1-ylium"


# ---------- diazonium ----------


def _render_diazonium(
    cls: ChargeClassification,
    mol,
    strategy,
    session,
    depth: int,
) -> str | None:
    """Render ``ethane-1-diazonium`` / ``cyclohexane-1-diazonium`` etc.

    Strip the [N+]#N pair, rename the parent, then splice
    ``-<locant>-diazonium``.
    """
    n_plus_idx, n_terminal_idx = cls.site_atom_indices
    parent_smiles = _neutral_skeleton_smiles(
        mol,
        {
            n_plus_idx: {"delete": True},
            n_terminal_idx: {"delete": True},
        },
    )
    if parent_smiles is None:
        return None
    parent_name = _drive_engine(parent_smiles, strategy, session, depth)
    if parent_name is None:
        return None
    locant = 1
    return _splice_diazonium(parent_name, locant)


def _splice_diazonium(parent_name: str, locant: int) -> str:
    """Compose ``ethane-1-diazonium`` / ``naphthalene-1-diazonium`` etc.

    ``-diazonium`` is consonant-initial so the trailing ``e`` of
    ``-ane`` / ``benzene`` / ``naphthalene`` is preserved.
    """
    return f"{parent_name}-{locant}-diazonium"


# ---------- amidinium ----------


def _render_amidinium(
    cls: ChargeClassification,
    mol,
    strategy,
    session,
    depth: int,
) -> str | None:
    """Render ``ethanamidinium`` / ``cyclohexan-1-amidinium`` etc.

    The amidinium tail ``-C(=[NH2+])N`` is stripped to a carboxylic
    acid ``-C(=O)OH`` so the engine can name the parent acid; we then
    transform the acid name into the amidinium form.
    """
    c_idx, n_plus_idx, n_neutral_idx = cls.site_atom_indices
    # Convert =[NH2+] -> =O and -NH2 -> -OH (i.e. emulate the parent
    # carboxylic acid by mutating in place).  RDKit's RWMol supports
    # bond-order changes via GetBondBetweenAtoms.
    from rdkit import Chem

    rw = Chem.RWMol(mol)
    n_plus = rw.GetAtomWithIdx(n_plus_idx)
    n_plus.SetAtomicNum(8)
    n_plus.SetFormalCharge(0)
    n_plus.SetNumExplicitHs(0)
    n_plus.SetNoImplicit(True)
    n_neutral = rw.GetAtomWithIdx(n_neutral_idx)
    n_neutral.SetAtomicNum(8)
    n_neutral.SetFormalCharge(0)
    n_neutral.SetNumExplicitHs(1)
    n_neutral.SetNoImplicit(True)
    try:
        Chem.SanitizeMol(rw)
    except Exception:
        return None
    parent_smiles = Chem.MolToSmiles(rw.GetMol())
    parent_name = _drive_engine(parent_smiles, strategy, session, depth)
    if parent_name is None:
        return None
    return _acid_name_to_amidinium(parent_name)


def _acid_name_to_amidinium(acid_name: str) -> str | None:
    """Transform a rendered acid name to its amidinium form.

    - ``acetic acid`` -> ``acetamidinium`` (retained)
    - ``benzoic acid`` -> ``benzamidinium`` (retained)
    - ``-oic acid`` -> ``-amidinium`` (systematic)
    - ``-carboxylic acid`` -> ``-carboximidamidium`` is a different
      class; we fall back to ``-carbamidinium`` only when OPSIN parses
      it.  In practice the audit corpus pins ``cyclohexan-1-amidinium``
      which OPSIN derives from the cyclohexane parent + amidinium
      suffix.  We synthesise that surface form when the parent name
      ends in ``carboxylic acid`` by stripping the suffix and
      appending ``-1-amidinium``.
    """
    if not acid_name:
        return None
    # Retained
    if acid_name == "acetic acid":
        return "acetamidinium"
    if acid_name == "benzoic acid":
        return "benzamidinium"
    if acid_name == "formic acid":
        return "formamidinium"
    if acid_name.endswith("oic acid"):
        # ``ethanoic acid`` would be unusual (we render ``acetic``),
        # but be defensive: ``pentanoic acid`` -> ``pentanamidinium``.
        stem = acid_name[: -len("oic acid")]
        return stem + "amidinium"
    if acid_name.endswith("carboxylic acid"):
        # Drop the carboxylic suffix and append ``-1-amidinium``.
        stem = acid_name[: -len("carboxylic acid")].rstrip("-").rstrip()
        if stem.endswith("e"):
            stem = stem[:-1]
        return f"{stem}-1-amidinium"
    return None


# ---------- acylium ----------


def _render_acylium(
    cls: ChargeClassification,
    mol,
    strategy,
    session,
    depth: int,
) -> str | None:
    """Render ``acetylium`` / ``pentanoylium`` /
    ``cyclohexanecarbonylium`` etc.

    The [C+]=O carbon is mutated in-place to a carboxylic acid carbon
    (=O kept, -OH added), so the parent acid name comes out in the
    canonical form (``acetic acid`` / ``pentanoic acid`` /
    ``cyclohexanecarboxylic acid``).  We then map it to the acylium.
    """
    c_idx, o_idx = cls.site_atom_indices
    from rdkit import Chem

    rw = Chem.RWMol(mol)
    c_atom = rw.GetAtomWithIdx(c_idx)
    c_atom.SetFormalCharge(0)
    # Add an OH off the C+.
    oh = rw.AddAtom(Chem.Atom(8))
    rw.GetAtomWithIdx(oh).SetNumExplicitHs(1)
    rw.AddBond(c_idx, oh, Chem.BondType.SINGLE)
    try:
        Chem.SanitizeMol(rw)
    except Exception:
        return None
    parent_smiles = Chem.MolToSmiles(rw.GetMol())
    parent_name = _drive_engine(parent_smiles, strategy, session, depth)
    if parent_name is None:
        return None
    return _acid_name_to_acylium(parent_name)


def _acid_name_to_acylium(acid_name: str) -> str | None:
    """Transform a rendered carboxylic acid name to its acylium form.

    - retained: ``acetic acid`` -> ``acetylium``, ``benzoic acid`` ->
      ``benzoylium``, ``formic acid`` -> ``formylium``
    - systematic ``-oic acid`` -> ``-oylium``
    - systematic ``-carboxylic acid`` -> ``-carbonylium``

    Note: our engine prefers retained ``propionic`` over systematic
    ``propanoic``; OPSIN does parse ``propanoylium`` but does NOT
    parse ``propionylium``.  When we hit a retained-acid name that
    isn't in our small map we fall back to converting it to the
    systematic stem first.
    """
    if not acid_name:
        return None
    if acid_name in _RETAINED_ACID_TO_ACYLIUM:
        return _RETAINED_ACID_TO_ACYLIUM[acid_name]
    if acid_name.endswith("oic acid"):
        return acid_name[: -len("oic acid")] + "oylium"
    if acid_name.endswith("carboxylic acid"):
        return acid_name[: -len("carboxylic acid")] + "carbonylium"
    if acid_name.endswith("propionic acid"):
        # Retain-style; convert to systematic to keep OPSIN happy.
        stem = acid_name[: -len("propionic acid")]
        return stem + "propanoylium"
    if acid_name.endswith("butyric acid"):
        stem = acid_name[: -len("butyric acid")]
        return stem + "butanoylium"
    return None


# ---------------------------------------------------------------------------
# Stage 7 renderers
# ---------------------------------------------------------------------------


def _splice_diylium_suffix(
    parent_name: str,
    locants: tuple[int, ...],
    base_suffix: str,
) -> str:
    """Compose ``propane-1,3-diylium`` / ``cyclohexane-1,4-diide`` etc.

    ``base_suffix`` is one of the consonant-initial multi-charge
    suffixes (``diylium``, ``triylium``, ``tetraylium``, ``diide``).
    The trailing ``e`` of ``-ane`` / ``benzene`` / ``naphthalene`` is
    preserved before consonant-initial ``d`` / ``t``, matching the
    diazonium splice in R2-B.
    """
    locant_str = ",".join(str(loc) for loc in locants)
    return f"{parent_name}-{locant_str}-{base_suffix}"


def _multi_charge_suffix(count: int, sign: str) -> str | None:
    """Return ``diylium`` / ``triylium`` / ``tetraylium`` / ``diide`` etc.

    ``sign`` is ``"+"`` or ``"-"`` (``"-"`` selects ``-ide`` family).
    Returns ``None`` for unsupported counts (>4) so the caller defers
    to the engine's existing dispatch.
    """
    base = "ylium" if sign == "+" else "ide"
    multi = {2: "di", 3: "tri", 4: "tetra"}.get(count)
    if multi is None:
        return None
    return f"{multi}{base}"


def _atom_locants_in_parent_per_site(
    mol, site_atom_indices: tuple[int, ...]
) -> tuple[int, ...] | None:
    """Per-site locants in canonical numbering, preserving site order.

    Used by the polycarbon renderers.  Returns ``None`` when the
    parent skeleton is not a single-ring or linear chain (out-of-scope
    for Stage 7).
    """
    n_heavy = mol.GetNumHeavyAtoms()
    if not site_atom_indices:
        return None
    if n_heavy == 1:
        return tuple([1] * len(site_atom_indices))
    is_ring = any(atom.IsInRing() for atom in mol.GetAtoms())
    if is_ring:
        if not all(atom.IsInRing() for atom in mol.GetAtoms()):
            return None
        ring_info = mol.GetRingInfo()
        rings = ring_info.AtomRings()
        if len(rings) != 1:
            return None
        ring_atoms = list(rings[0])
        site_set = set(site_atom_indices)
        best_per_site: tuple[int, ...] | None = None
        best_sorted: tuple[int, ...] | None = None
        for start in range(len(ring_atoms)):
            for direction in (1, -1):
                ordered = [
                    ring_atoms[(start + direction * i) % len(ring_atoms)]
                    for i in range(len(ring_atoms))
                ]
                if ordered[0] not in site_set:
                    continue
                per_site = tuple(
                    ordered.index(idx) + 1 for idx in site_atom_indices
                )
                sorted_ = tuple(sorted(per_site))
                if best_sorted is None or sorted_ < best_sorted:
                    best_sorted = sorted_
                    best_per_site = per_site
        return best_per_site
    deg1 = [a.GetIdx() for a in mol.GetAtoms() if a.GetDegree() == 1]
    if len(deg1) != 2:
        return None
    start, end = deg1
    path: list[int] = [start]
    visited = {start}
    current = start
    while current != end:
        atom = mol.GetAtomWithIdx(current)
        next_atom: int | None = None
        for nb in atom.GetNeighbors():
            if nb.GetIdx() not in visited:
                next_atom = nb.GetIdx()
                break
        if next_atom is None:
            return None
        path.append(next_atom)
        visited.add(next_atom)
        current = next_atom
    forward = tuple(path.index(idx) + 1 for idx in site_atom_indices)
    n = len(path)
    reverse = tuple(n - path.index(idx) for idx in site_atom_indices)
    if tuple(sorted(forward)) <= tuple(sorted(reverse)):
        return forward
    return reverse


def _render_polycarbon(
    cls: ChargeClassification,
    mol,
    strategy,
    session,
    depth: int,
) -> str | None:
    """Render multi-charge carbon motifs (di / tri / tetra-ylium / -ide / mixed)."""
    # Single-atom |q|>=2 case — methanediylium / methanetriylium etc.
    if mol.GetNumHeavyAtoms() == 1:
        site_charge = cls.site_charges[0] if cls.site_charges else 0
        n = abs(site_charge)
        sign = "+" if site_charge > 0 else "-"
        if n == 2:
            return "methanediylium" if sign == "+" else "methanediide"
        if n == 3:
            return "methanetriylium" if sign == "+" else "methanetriide"
        if n == 4:
            return "methanetetraylium" if sign == "+" else "methanetetraide"
        return None
    # Build the neutral parent skeleton (charges -> 0, no_implicit off).
    atom_changes = {
        idx: {"charge": 0, "no_implicit": False}
        for idx in cls.site_atom_indices
    }
    parent_smiles = _neutral_skeleton_smiles(mol, atom_changes)
    if parent_smiles is None:
        return None
    parent_name = _drive_engine(parent_smiles, strategy, session, depth)
    if parent_name is None:
        return None
    # Per-site locants (parallel to site_atom_indices).
    per_site_locants = _atom_locants_in_parent_per_site(
        mol, cls.site_atom_indices
    )
    if per_site_locants is None:
        return None
    sorted_locants = tuple(sorted(per_site_locants))
    # Single-atom |q|>=2 inside a larger parent
    # (e.g. ``C[C+2]C`` -> ``propane-2,2-diylium``): the multiplicity
    # comes from the charge magnitude, not from multiple sites.  Repeat
    # the locant to produce ``2,2`` etc.
    if (
        len(cls.site_atom_indices) == 1
        and cls.suffix_hint in ("diylium", "diide")
        and cls.site_charges
        and abs(cls.site_charges[0]) >= 2
    ):
        n = abs(cls.site_charges[0])
        sign = "+" if cls.site_charges[0] > 0 else "-"
        suffix = _multi_charge_suffix(n, sign)
        if suffix is None:
            return None
        repeated_locants = tuple([per_site_locants[0]] * n)
        return _splice_diylium_suffix(parent_name, repeated_locants, suffix)
    if cls.suffix_hint == "mixed_id_ylium":
        # Lay out per-site tags in ascending-locant order:
        # ``butan-1-id-4-ylium`` form.
        ordered = sorted(
            zip(cls.site_atom_indices, cls.site_charges, per_site_locants),
            key=lambda t: t[2],
        )
        tags: list[str] = []
        for _idx, charge, loc in ordered:
            tag = "id" if charge < 0 else "ylium"
            tags.append(f"{loc}-{tag}")
        if parent_name.endswith("ane"):
            stem = parent_name[:-1]
        else:
            stem = parent_name
        return f"{stem}-" + "-".join(tags)
    site_charge = cls.site_charges[0] if cls.site_charges else 0
    sign = "+" if site_charge > 0 else "-"
    suffix = _multi_charge_suffix(len(cls.site_atom_indices), sign)
    if suffix is None:
        return None
    return _splice_diylium_suffix(parent_name, sorted_locants, suffix)


# ---- polyacylium ----


# Retained polycarboxylic-acid -> retained polyacylium surface mapping.
# OPSIN parses every retained name in this map; the systematic
# "<parent>dioylium" form is emitted when the engine yields the
# systematic acid name (``hexanedioic acid`` -> ``hexanedioylium``).
_RETAINED_DIACID_TO_DIACYLIUM: dict[str, str] = {
    "oxalic acid": "oxalylium",
    "malonic acid": "malonylium",
    "succinic acid": "succinylium",
    "glutaric acid": "glutarylium",
    "adipic acid": "adipoylium",
}


def _render_polyacylium(
    cls: ChargeClassification,
    mol,
    strategy,
    session,
    depth: int,
) -> str | None:
    """Render ``oxalylium`` / ``malonylium`` / ``butanedioylium`` etc.

    Strategy mirrors the R2-B acylium renderer but adds an OH off
    *every* claimed cation carbon, then maps the parent diacid name
    into its bis-acyl-cation form.
    """
    n_acyls = len(cls.site_atom_indices) // 2
    cation_carbons = cls.site_atom_indices[:n_acyls]
    from rdkit import Chem

    rw = Chem.RWMol(mol)
    for c_idx in cation_carbons:
        atom = rw.GetAtomWithIdx(c_idx)
        atom.SetFormalCharge(0)
        oh = rw.AddAtom(Chem.Atom(8))
        rw.GetAtomWithIdx(oh).SetNumExplicitHs(1)
        rw.AddBond(c_idx, oh, Chem.BondType.SINGLE)
    try:
        Chem.SanitizeMol(rw)
    except Exception:
        return None
    parent_smiles = Chem.MolToSmiles(rw.GetMol())
    parent_name = _drive_engine(parent_smiles, strategy, session, depth)
    if parent_name is None:
        return None
    return _diacid_name_to_polyacylium(parent_name)


def _diacid_name_to_polyacylium(diacid_name: str) -> str | None:
    """Transform a diacid name to its bis-acyl-cation form.

    Handles retained names (``oxalic acid`` -> ``oxalylium`` etc.) and
    the systematic ``-dioic acid`` -> ``-dioylium`` mapping.
    """
    if not diacid_name:
        return None
    if diacid_name in _RETAINED_DIACID_TO_DIACYLIUM:
        return _RETAINED_DIACID_TO_DIACYLIUM[diacid_name]
    if diacid_name.endswith("dioic acid"):
        return diacid_name[: -len("dioic acid")] + "dioylium"
    return None


# ---- aminylium / iminylium / amidylium ----


def _splice_consonant_suffix(
    parent_name: str, locant: int, suffix: str
) -> str:
    """Compose ``ethan-1-aminylium`` / ``naphthalen-1-iminylium`` etc.

    The radical-cation N suffixes (``aminylium``, ``iminylium``,
    ``amidylium``) all start with vowels (``a`` / ``i``), so the
    trailing ``e`` of ``-ane`` / ``naphthalene`` elides — same rule
    as R2-B's ``ylium`` / ``ide`` splice.
    """
    if parent_name.endswith("ane"):
        stem = parent_name[:-1]
        return f"{stem}-{locant}-{suffix}"
    if parent_name.endswith("ene"):
        stem = parent_name[:-1]
        return f"{stem}-{locant}-{suffix}"
    return f"{parent_name}-{locant}-{suffix}"


def _render_aminylium(
    cls: ChargeClassification,
    mol,
    strategy,
    session,
    depth: int,
) -> str | None:
    """Render ``ethan-1-aminylium`` / ``cyclohexan-1-aminylium`` etc.

    Strip the [NH+] (with its 2 radical electrons) and rename the
    parent.  The locant always comes out as 1 in our audit scope
    (single substituent on chain or ring); the OPSIN round-trip in
    the test suite pins the surface form.
    """
    n_idx = cls.site_atom_indices[0]
    parent_smiles = _neutral_skeleton_smiles(
        mol, {n_idx: {"delete": True}}
    )
    if parent_smiles is None:
        return None
    parent_name = _drive_engine(parent_smiles, strategy, session, depth)
    if parent_name is None:
        return None
    return _splice_consonant_suffix(parent_name, 1, "aminylium")


def _render_azaniumyl(
    cls: ChargeClassification,
    mol,
    strategy,
    session,
    depth: int,
) -> str | None:
    """Render ``methylazaniumyl`` etc. for ``R-[NH2+]`` (azaniumyl radical-cation).

    Strategy: strip the [NH2+] atom, name R as a substituent, then
    concatenate "azaniumyl".  For ``C[NH2+]`` (methyl group) this gives
    "methylazaniumyl".  OPSIN (allow_radicals=True) parses this back to
    ``C[NH2+]``.
    """
    from rdkit import Chem
    from iupac_namer.engine import name as _recursive_name, NamingSession
    from iupac_namer.assembly import assemble
    from iupac_namer.types import OutputForm, FreeValenceInfo

    n_idx = cls.site_atom_indices[0]
    n_atom = mol.GetAtomWithIdx(n_idx)
    bond = n_atom.GetBonds()[0]
    parent_atom = bond.GetOtherAtom(n_atom)
    parent_atom_idx = parent_atom.GetIdx()

    # Build mol without the N+ atom to name the R group as a substituent.
    rw = Chem.RWMol(mol)
    # Remove N+ atom (and its bond to parent).
    rw.RemoveAtom(n_idx)
    # Adjust parent_atom_idx if it was after n_idx.
    new_parent_idx = parent_atom_idx if parent_atom_idx < n_idx else parent_atom_idx - 1
    # Allow the parent atom to accept implicit H.
    rw.GetAtomWithIdx(new_parent_idx).SetNoImplicit(False)
    try:
        Chem.SanitizeMol(rw)
    except Exception:
        return None
    parent_mol = rw.GetMol()

    # Name the R group as a substituent (free-valence from parent_atom).
    try:
        from iupac_namer.engine import _select_substituent_method, _fvi_elide_locant_one
        sub_method = _select_substituent_method(parent_mol, new_parent_idx)
        sub_fv = FreeValenceInfo(
            bond_orders=(1,),
            method=sub_method,
            attachment_atoms_in_fragment=(new_parent_idx,),
            elide_locant_one=_fvi_elide_locant_one(parent_mol, new_parent_idx),
        )
        if session is None:
            session = NamingSession()
        sub_tree = _recursive_name(
            parent_mol, strategy, OutputForm.SUBSTITUENT,
            free_valence=sub_fv,
            _session=session, _depth=depth + 1,
        )
        sub_name = assemble(sub_tree)
    except Exception:
        return None
    if not sub_name or sub_name.startswith("[NAMING ERROR"):
        return None
    # Strip any trailing hyphen from the substituent form.
    sub_name = sub_name.rstrip("-")
    return f"{sub_name}azaniumyl"


def _render_iminylium(
    cls: ChargeClassification,
    mol,
    strategy,
    session,
    depth: int,
) -> str | None:
    """Render ``ethan-1-iminylium`` / ``pentan-2-iminylium`` etc.

    The =[N+] (radical-cation imino) is stripped from the parent C,
    and the parent C is reduced to a saturated CH (so the engine
    names it as the corresponding alkane).  We then splice
    ``-<locant>-iminylium``.

    The locant is computed by mapping the parent C's index in the
    original mol to the canonical numbering of the alkane parent.
    For ``CC(CCC)=[N+]`` the imine carbon is the middle of pentane
    so the locant is 2 (``pentan-2-iminylium``); for terminal cases
    (``CC=[N+]``) the locant is 1.
    """
    from rdkit import Chem

    n_idx = cls.site_atom_indices[0]
    n_atom = mol.GetAtomWithIdx(n_idx)
    bond = n_atom.GetBonds()[0]
    parent_c_idx = bond.GetOtherAtom(n_atom).GetIdx()
    rw = Chem.RWMol(mol)
    rw.RemoveAtom(n_idx)
    new_parent_c_idx = (
        parent_c_idx if parent_c_idx < n_idx else parent_c_idx - 1
    )
    pc = rw.GetAtomWithIdx(new_parent_c_idx)
    pc.SetNoImplicit(False)
    try:
        Chem.SanitizeMol(rw)
    except Exception:
        return None
    parent_mol = rw.GetMol()
    parent_smiles = Chem.MolToSmiles(parent_mol)
    parent_name = _drive_engine(parent_smiles, strategy, session, depth)
    if parent_name is None:
        return None
    locants = _atom_locants_in_parent_per_site(
        parent_mol, (new_parent_c_idx,)
    )
    locant = locants[0] if locants else 1
    return _splice_consonant_suffix(parent_name, locant, "iminylium")


def _render_amidylium(
    cls: ChargeClassification,
    mol,
    strategy,
    session,
    depth: int,
) -> str | None:
    """Render ``ethan-1-amidylium`` / ``cyclohexan-1-amidylium`` etc.

    Strip the ``-C(=O)[NH+]`` tail by mutating it into a carboxylic
    acid (replace [NH+] with -OH).  The parent acid name is then
    transformed into the amidylium form.
    """
    from rdkit import Chem

    central_c_idx, n_plus_idx, _o_idx = cls.site_atom_indices
    rw = Chem.RWMol(mol)
    n_plus = rw.GetAtomWithIdx(n_plus_idx)
    n_plus.SetAtomicNum(8)
    n_plus.SetFormalCharge(0)
    n_plus.SetNumExplicitHs(1)
    n_plus.SetNoImplicit(True)
    n_plus.SetNumRadicalElectrons(0)
    try:
        Chem.SanitizeMol(rw)
    except Exception:
        return None
    parent_smiles = Chem.MolToSmiles(rw.GetMol())
    parent_name = _drive_engine(parent_smiles, strategy, session, depth)
    if parent_name is None:
        return None
    return _acid_name_to_amidylium(parent_name)


def _acid_name_to_amidylium(acid_name: str) -> str | None:
    """Transform a rendered acid name to its amidylium form.

    OPSIN parses the following surface forms (probed):
    - ``acetamidylium`` (retained, from ``acetic acid``)
    - ``ethan-1-amidylium`` (from ``ethanoic acid`` -> ``ethan-`` stem)
    - ``pentan-1-amidylium`` (from ``pentanoic acid``)
    - ``cyclohexan-1-amidylium`` (from ``cyclohexanecarboxylic acid``)
    - ``naphthalen-1-amidylium`` (from ``naphthalene-1-carboxylic acid``)
    """
    if not acid_name:
        return None
    if acid_name == "acetic acid":
        return "acetamidylium"
    if acid_name == "benzoic acid":
        return "benzamidylium"
    if acid_name == "formic acid":
        return "formamidylium"
    if acid_name.endswith("oic acid"):
        # ``pentanoic acid`` -> ``pentan-1-amidylium``.
        stem = acid_name[: -len("oic acid")]
        if stem.endswith("-"):
            stem = stem[:-1]
        return f"{stem}-1-amidylium"
    if acid_name.endswith("carboxylic acid"):
        stem = acid_name[: -len("carboxylic acid")].rstrip("-").rstrip()
        if stem.endswith("e"):
            stem = stem[:-1]
        return f"{stem}-1-amidylium"
    return None


# ---------------------------------------------------------------------------
# P-72 / P-73 polycharged / zwitterion renderers
# ---------------------------------------------------------------------------


def _name_substituent_fragment(
    mol,
    backbone_idxs: set[int],
    attach_idx: int,
    nb_idx: int,
    strategy,
    session,
    depth: int,
) -> str | None:
    """Name one substituent fragment hanging off a backbone atom.

    ``attach_idx`` is the backbone atom; ``nb_idx`` is the first atom of the
    substituent (a heavy neighbour of ``attach_idx`` that is NOT a backbone
    atom).  The fragment is everything reachable from ``nb_idx`` without
    crossing back into ``backbone_idxs``.  Returns the SUBSTITUENT name with
    any leading hyphen stripped, or ``None`` on failure.

    The order of the bond joining the backbone atom to the substituent is
    honoured: a single bond yields an ``-yl`` prefix (``methyl``, ``phenyl``),
    a double bond an ``-ylidene`` prefix (``propan-2-ylidene``).  This is how a
    cation centre's ``=C(R)R`` substituent (e.g. in
    ``methyl(propan-2-ylidene)ammoniumyl``) is named correctly instead of as a
    saturated alkyl.

    Mirrors the carving logic already used by ``_render_substituted_boranuide``
    so the recursion path (and its OPSIN round-trip guarantees) is identical.
    """
    from rdkit import Chem
    from iupac_namer.engine import (
        name as _recursive_name,
        _select_substituent_method,
        _fvi_elide_locant_one,
    )
    from iupac_namer.assembly import assemble
    from iupac_namer.types import OutputForm, FreeValenceInfo, SubstituentMethod

    # BFS the substituent atoms (everything reachable from nb_idx without
    # crossing into the backbone).
    visited: set[int] = set(backbone_idxs)
    stack = [nb_idx]
    sub_atoms: list[int] = []
    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        sub_atoms.append(cur)
        for nb2 in mol.GetAtomWithIdx(cur).GetNeighbors():
            if nb2.GetIdx() in backbone_idxs:
                continue
            stack.append(nb2.GetIdx())

    rw = Chem.RWMol(mol)
    bond = rw.GetBondBetweenAtoms(attach_idx, nb_idx)
    if bond is None:
        return None
    attach_bond_order = int(round(bond.GetBondTypeAsDouble()))
    if attach_bond_order not in (1, 2):
        return None  # triple-bonded substituents (-ylidyne) out of scope here
    rw.RemoveBond(attach_idx, nb_idx)
    keep = set(sub_atoms)
    delete_indices = sorted(
        (a.GetIdx() for a in rw.GetAtoms() if a.GetIdx() not in keep),
        reverse=True,
    )
    for d in delete_indices:
        rw.RemoveAtom(d)
    # Clear any leftover formal charge / radical on the carved fragment so it
    # sanitises as a neutral substituent.
    for a in rw.GetAtoms():
        a.SetFormalCharge(0)
        a.SetNumRadicalElectrons(0)
        a.SetNoImplicit(False)
    try:
        Chem.SanitizeMol(rw)
    except Exception:
        return None
    sub_mol = rw.GetMol()
    new_attachment_idx = sorted(keep).index(nb_idx)
    try:
        method = _select_substituent_method(sub_mol, new_attachment_idx)
        elide = _fvi_elide_locant_one(sub_mol, new_attachment_idx)
    except Exception:
        method = SubstituentMethod.ALKYL
        elide = False
    fv = FreeValenceInfo(
        bond_orders=(attach_bond_order,),
        method=method,
        attachment_atoms_in_fragment=(new_attachment_idx,),
        elide_locant_one=elide,
    )
    try:
        sub_tree = _recursive_name(
            sub_mol,
            strategy,
            OutputForm.SUBSTITUENT,
            free_valence=fv,
            decision_ctx=None,
            _session=session,
            _depth=depth + 1,
        )
    except Exception:
        return None
    sub_name = assemble(sub_tree)
    if sub_name is None or "NAMING ERROR" in sub_name:
        return None
    if sub_name.startswith("-"):
        sub_name = sub_name[1:]
    return sub_name


def _merged_prefix_string(sub_names: list[str]) -> str:
    """Alphabetise + multiply a list of substituent names into a prefix block.

    e.g. ``["phenyl","phenyl"]`` -> ``"diphenyl"``;
         ``["methyl"]*4`` -> ``"tetramethyl"``.
    Reuses the standard assembly prefix machinery (P-16.3.3 / P-16.3.4).
    """
    from iupac_namer.assembly import (
        merge_identical_prefixes,
        render_merged_prefixes,
    )

    entries = [(name, ()) for name in sub_names]
    merged = merge_identical_prefixes(entries)
    merged.sort(key=lambda m: m.sort_name)
    return render_merged_prefixes(merged)


def _render_carbanion_zwitterion(
    cls: ChargeClassification,
    mol,
    strategy,
    session,
    depth: int,
) -> str | None:
    """Render ``2-(trimethylammoniumyl)propan-2-ide`` and family.

    The carbanion is the senior ``-ide`` suffix on the all-carbon parent; the
    onium cation is carved off as a ``...iumyl`` substituent prefix.
    """
    c_minus_idx, onium_idx = cls.site_atom_indices
    onium_atom = mol.GetAtomWithIdx(onium_idx)
    onium_base = _ONIUM_PREFIX_BASE.get(onium_atom.GetSymbol())
    if onium_base is None:
        return None

    # 1. Build the carbanion's all-carbon parent: delete the entire onium
    #    branch, neutralise the carbanion, and name the resulting hydrocarbon.
    #    Collect every atom reachable from the onium without crossing the
    #    carbanion (so we delete the cation and its own substituents).
    onium_branch: set[int] = set()
    stack = [onium_idx]
    while stack:
        cur = stack.pop()
        if cur in onium_branch:
            continue
        onium_branch.add(cur)
        for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
            if nb.GetIdx() == c_minus_idx:
                continue
            stack.append(nb.GetIdx())

    changes: dict[int, dict] = {idx: {"delete": True} for idx in onium_branch}
    changes[c_minus_idx] = {"charge": 0, "no_implicit": False}
    parent_smiles = _neutral_skeleton_smiles(mol, changes)
    if parent_smiles is None:
        return None
    parent_name = _drive_engine(parent_smiles, strategy, session, depth)
    if parent_name is None or "NAMING ERROR" in parent_name:
        return None

    # 2. Locant of the carbanion carbon in the parent's numbering.  Carve the
    #    parent the same way (delete the onium branch, neutralise the carbanion)
    #    but tag the carbanion so we can read its locant after canonicalisation.
    locant = _carbanion_locant_in_parent(mol, c_minus_idx, onium_branch)
    if locant is None:
        return None

    # 3. Build the onium substituent prefix (e.g. trimethylammoniumyl).
    onium_sub_names: list[str] = []
    for nb in onium_atom.GetNeighbors():
        if nb.GetIdx() == c_minus_idx:
            continue
        sub_name = _name_substituent_fragment(
            mol, {onium_idx}, onium_idx, nb.GetIdx(), strategy, session, depth
        )
        if sub_name is None:
            return None
        onium_sub_names.append(sub_name)
    onium_prefix = _merged_prefix_string(onium_sub_names) if onium_sub_names else ""
    cation_substituent = f"{onium_prefix}{onium_base}"

    # 4. Splice: <locant>-(<cation_substituent>)<parent stem>-<locant>-ide.
    #    The cation prefix is always a compound prefix (it carries its own
    #    substituents), so it is enclosed; escalate the bracket level when the
    #    prefix already contains parentheses (P-16.3.3).
    from iupac_namer.assembly import _choose_brackets
    open_b, close_b = _choose_brackets(cation_substituent)
    parent_stem = parent_name[:-1] if parent_name.endswith("ane") else parent_name
    return (
        f"{locant}-{open_b}{cation_substituent}{close_b}"
        f"{parent_stem}-{locant}-ide"
    )


def _carbanion_locant_in_parent(
    mol, c_minus_idx: int, onium_branch: set[int]
) -> int | None:
    """Locant of the carbanion carbon in the neutralised parent.

    Carve the carbon parent the same way ``_render_carbanion_zwitterion``
    does, but track the carbanion's identity through the deletion so we can
    read its canonical locant from the parent numbering.
    """
    from rdkit import Chem

    rw = Chem.RWMol(mol)
    # Tag the carbanion atom so we can find it after deletion + canonicalise.
    rw.GetAtomWithIdx(c_minus_idx).SetFormalCharge(0)
    rw.GetAtomWithIdx(c_minus_idx).SetNoImplicit(False)
    rw.GetAtomWithIdx(c_minus_idx).SetProp("_carbanion", "1")
    delete_indices = sorted(onium_branch, reverse=True)
    for d in delete_indices:
        rw.RemoveAtom(d)
    try:
        Chem.SanitizeMol(rw)
    except Exception:
        return None
    sub_mol = rw.GetMol()
    tagged = None
    for a in sub_mol.GetAtoms():
        if a.HasProp("_carbanion"):
            tagged = a.GetIdx()
            break
    if tagged is None:
        return None
    return _locant_from_parent_numbering(sub_mol, tagged)


def _locant_from_parent_numbering(parent_mol, atom_idx: int) -> int | None:
    """Drive the engine on ``parent_mol`` and read the locant of ``atom_idx``.

    Falls back to a chain/ring walk (mirroring
    ``_atom_locants_in_parent_per_site``) so we get the same numbering the
    surface name uses.
    """
    locs = _atom_locants_in_parent_per_site(parent_mol, (atom_idx,))
    if locs:
        return locs[0]
    return None


def _render_substituted_carbon_dianion(
    cls: ChargeClassification,
    mol,
    strategy,
    session,
    depth: int,
) -> str | None:
    """Render ``diphenylmethanediide`` and family.

    The lone ``[C-2]`` carbon is the methane parent; every heavy neighbour is
    a substituent prefix.  Suffix ``methanediide``.
    """
    c_idx = cls.site_atom_indices[0]
    c_atom = mol.GetAtomWithIdx(c_idx)
    sub_names: list[str] = []
    for nb in c_atom.GetNeighbors():
        if nb.GetAtomicNum() == 1:
            continue
        sub_name = _name_substituent_fragment(
            mol, {c_idx}, c_idx, nb.GetIdx(), strategy, session, depth
        )
        if sub_name is None:
            return None
        sub_names.append(sub_name)
    if not sub_names:
        return None
    prefix = _merged_prefix_string(sub_names)
    return f"{prefix}methanediide"


def _render_diazene_diium(
    cls: ChargeClassification,
    mol,
    strategy,
    session,
    depth: int,
) -> str | None:
    """Render ``tetramethyldiazene-1,2-diium`` and family.

    Backbone = the two N atoms (diazene, N=N); substituents off each N become
    prefixes; suffix ``-1,2-diium``.
    """
    n0_idx, n1_idx = cls.site_atom_indices
    backbone = {n0_idx, n1_idx}
    sub_names: list[str] = []
    for n_idx in (n0_idx, n1_idx):
        n_atom = mol.GetAtomWithIdx(n_idx)
        for nb in n_atom.GetNeighbors():
            if nb.GetIdx() in backbone:
                continue
            if nb.GetAtomicNum() == 1:
                continue
            sub_name = _name_substituent_fragment(
                mol, backbone, n_idx, nb.GetIdx(), strategy, session, depth
            )
            if sub_name is None:
                return None
            sub_names.append(sub_name)
    prefix = _merged_prefix_string(sub_names) if sub_names else ""
    return f"{prefix}diazene-1,2-diium"


def _render_diazane_diide(
    cls: ChargeClassification,
    mol,
    strategy,
    session,
    depth: int,
) -> str | None:
    """Render ``acetyldiazane-1,1-diide`` and family.

    Backbone = the two N atoms (diazane, N-N).  The dianionic N is locant 1;
    substituents off either N become prefixes; suffix ``-1,1-diide`` on the
    dianionic N.
    """
    nd_idx, n_other_idx = cls.site_atom_indices
    backbone = {nd_idx, n_other_idx}
    sub_names: list[str] = []
    for n_idx in (nd_idx, n_other_idx):
        n_atom = mol.GetAtomWithIdx(n_idx)
        for nb in n_atom.GetNeighbors():
            if nb.GetIdx() in backbone:
                continue
            if nb.GetAtomicNum() == 1:
                continue
            sub_name = _name_substituent_fragment(
                mol, backbone, n_idx, nb.GetIdx(), strategy, session, depth
            )
            if sub_name is None:
                return None
            sub_names.append(sub_name)
    prefix = _merged_prefix_string(sub_names) if sub_names else ""
    # The dianionic N is locant 1 (lowest locant to the principal suffix,
    # P-31.1.4); both charges sit on it so the suffix is ``-1,1-diide``.
    return f"{prefix}diazane-1,1-diide"


def _render_ring_polycarbanion(
    cls: ChargeClassification,
    mol,
    strategy,
    session,
    depth: int,
) -> str | None:
    """Render ``1,4-dihydronaphthalene-1,4-diide`` and family.

    Neutralise every charged ring carbon, name the resulting carbocyclic
    parent, then splice ``-<locants>-diide`` using the parent numbering.
    """
    atom_changes = {
        idx: {"charge": 0, "no_implicit": False}
        for idx in cls.site_atom_indices
    }
    parent_smiles = _neutral_skeleton_smiles(mol, atom_changes)
    if parent_smiles is None:
        return None
    parent_name = _drive_engine(parent_smiles, strategy, session, depth)
    if parent_name is None or "NAMING ERROR" in parent_name:
        return None
    per_site_locants = _atom_locants_in_parent_per_site(
        mol, cls.site_atom_indices
    )
    if per_site_locants is None:
        return None
    sorted_locants = tuple(sorted(per_site_locants))
    suffix = _multi_charge_suffix(len(cls.site_atom_indices), "-")
    if suffix is None:
        return None
    return _splice_diylium_suffix(parent_name, sorted_locants, suffix)
