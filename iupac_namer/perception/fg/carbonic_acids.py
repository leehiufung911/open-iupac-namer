"""iupac_namer.perception.fg.carbonic_acids

GENERATIVE namer for carbonic acid, the di-/tri-/tetra-/polycarbonic acid
chains, and their chalcogen / imido / hydrazono / peroxy functional-
replacement analogues (P-65.2).

Carbonic acid and the polynuclear di-, tri-, tetra-carbonic acids are
*functional parent compounds* with retained preferred-IUPAC names
(P-65.2): they carry no substitutable hydrogen and so are NOT named by
ordinary substitutive nomenclature.  Their structures are::

    carbonic acid    HO-CO-OH
    dicarbonic acid  HO-CO-O-CO-OH
    tricarbonic acid HO-CO-O-CO-O-CO-OH
    tetracarbonic    HO-CO-O-CO-O-CO-O-CO-OH

Functional replacement of the acid oxygens (P-65.2.1 / P-65.2.3.1.2) is
denoted by *prefixes*:

    thio      S  or =S      seleno    Se or =Se
    telluro   Te or =Te     imido     NH or =NH
    hydrazono =N-NH2        peroxy    -O-O-

cited in alphabetical order in front of the retained polyacid name, each
with its position locant (P-65.2.3.1.1: "the chain is numbered
consecutively from one end to the other, starting from and ending at a
carbon atom").

Locant scheme (P-65.2.3.1.2.1)
------------------------------
The oxygen *positions* along the chain are numbered 1, 2, 3, ...; carbon i
(1-indexed along the chain) owns the **odd** locant ``2*i - 1`` shared by
its doubly-bonded acid oxygen (=O) and, on a terminal carbon, its terminal
single-bonded acid oxygen (-OH); the **even** locant ``2*i`` is the
bridging -O- between carbon i and carbon i+1.  Thus an n-carbon chain has
``2*n - 1`` positions.  The direction is chosen to give the set of
replacement locants the lowest values (P-14.5).

OPSIN placement convention (used so our names round-trip)
--------------------------------------------------------
At an odd (carbon) locant the *first* chalcogen replacement falls on the
doubly-bonded =O; a second replacement at the same locant (``1,1``) then
falls on the terminal -OH.  ``imido``/``hydrazono`` always denote the
doubly-bonded =NH/=N-NH2 (the terminal -OH cannot become =NH; that would
be ``amido`` -NH2, P-65.2.3.1.4, out of scope here).  ``peroxy`` at an odd
locant is the terminal -O-OH; at an even locant it is the -O-O- bridge.

Scope
-----
This generator self-gates: it returns ``None`` for anything that is not a
clean carbonic-acid chain skeleton (rings, charge, genuine radicals, extra
substituents, branched chains, mixed/ambiguous chalcogen tautomers that
require the superscript letter-locant forms OPSIN cannot parse), leaving
those to the ordinary machinery.  Every decision branches on a structural
feature (chain length, per-site element, bond order, bridge topology), so
the namer extends automatically across the family — never a molecule-
specific short-circuit.

Citations
---------
IUPAC 2013 Recommendations (Blue Book):
* P-65.2          carbonic, cyanic, di- and polycarbonic acids
* P-65.2.1.2/.3   chalcogen / imido / hydrazono replacement (mononuclear)
* P-65.2.3.1.1    general methodology for the polynuclear acids
* P-65.2.3.1.2    replacement prefixes and the chain numbering
"""

from __future__ import annotations

from dataclasses import dataclass


# Multiplying prefixes (P-14.2).  Local copy keeps the module self-contained.
_MULT = {
    1: "", 2: "di", 3: "tri", 4: "tetra", 5: "penta",
    6: "hexa", 7: "hepta", 8: "octa", 9: "nona", 10: "deca",
}

# Retained polyacid stems by carbon count (P-65.2).  n == 1 is the
# mononuclear "carbonic"; n >= 5 ("polycarbonic") is named by skeletal
# replacement (P-65.2.3) and is out of this generator's scope.
_CHAIN_STEM = {1: "carbonic", 2: "dicarbonic", 3: "tricarbonic",
               4: "tetracarbonic"}

# Chalcogen replacement infix keyed by element symbol (P-65.2.3.1.2).
_CHALC_INFIX = {"S": "thio", "Se": "seleno", "Te": "telluro"}

# Elements that may appear as a doubly-bonded acid position (=X) or a
# terminal/bridge single-bonded acid position.
_CHALCOGENS = frozenset({"O", "S", "Se", "Te"})


class _Fail(Exception):
    """Raised internally when *mol* is not a clean carbonic-acid skeleton."""


@dataclass(frozen=True)
class _Site:
    """One numbered acid position and the prefix (if any) it contributes."""
    locant: int
    prefix: str | None   # None -> unchanged (plain -OH / =O / -O- bridge)


def _terminal_chalcogen_replacement(atom, mol, c_idx) -> str | None:
    """Prefix for a terminal single-bonded acid oxygen-position atom.

    Returns the replacement prefix ("thio"/"seleno"/"telluro"/"peroxy") or
    ``None`` if the position is an unchanged -OH.  Raises ``_Fail`` for any
    shape that is not a clean terminal acid position.
    """
    sym = atom.GetSymbol()
    if sym == "O":
        # plain -OH, or peroxy terminal -O-OH (degree 2 O bonded to another
        # terminal O).
        deg = atom.GetDegree()
        if deg == 1:
            if atom.GetTotalNumHs() == 1 and atom.GetFormalCharge() == 0:
                return None  # unchanged -OH
            raise _Fail("bad terminal O")
        if deg == 2:
            other = [n for n in atom.GetNeighbors() if n.GetIdx() != c_idx]
            if len(other) != 1 or other[0].GetSymbol() != "O":
                raise _Fail("bad terminal bridge")
            o2 = other[0]
            if (o2.GetDegree() == 1 and o2.GetTotalNumHs() == 1
                    and o2.GetFormalCharge() == 0):
                return "peroxy"  # -O-OH
            raise _Fail("bad terminal peroxy")
        raise _Fail("bad terminal O degree")
    if sym in ("S", "Se", "Te"):
        if (atom.GetDegree() == 1 and atom.GetTotalNumHs() == 1
                and atom.GetFormalCharge() == 0):
            return _CHALC_INFIX[sym]
        raise _Fail("bad terminal chalcogen")
    raise _Fail(f"unexpected terminal atom {sym}")


def _double_bond_replacement(atom) -> str | None:
    """Prefix for the doubly-bonded acid position (=X) of a carbon.

    Returns "thio"/"seleno"/"telluro"/"imido"/"hydrazono" or ``None`` for an
    unchanged =O.  Raises ``_Fail`` for any other shape.
    """
    sym = atom.GetSymbol()
    if sym == "O":
        if atom.GetDegree() != 1 or atom.GetFormalCharge() != 0:
            raise _Fail("bad =O")
        return None
    if sym in ("S", "Se", "Te"):
        if (atom.GetDegree() != 1 or atom.GetFormalCharge() != 0
                or atom.GetTotalNumHs() != 0):
            raise _Fail("bad =chalcogen")
        return _CHALC_INFIX[sym]
    if sym == "N":
        # =NH imido, or =N-NH2 hydrazono.
        if atom.GetFormalCharge() != 0:
            raise _Fail("charged imido N")
        nbrs = [n for n in atom.GetNeighbors()]  # excludes the carbon? no
        heavy = [n for n in nbrs if n.GetAtomicNum() != 1]
        # the carbon is one heavy neighbour; a second heavy neighbour (N)
        # makes it hydrazono.
        non_c = [n for n in heavy if n.GetAtomicNum() != 6]
        if not non_c:
            # =NH (only the carbon as heavy neighbour) -> imido
            if atom.GetTotalNumHs() == 1:
                return "imido"
            raise _Fail("bad imido")
        if len(non_c) == 1 and non_c[0].GetSymbol() == "N":
            n2 = non_c[0]
            if (n2.GetDegree() == 1 and n2.GetTotalNumHs() == 2
                    and n2.GetFormalCharge() == 0):
                return "hydrazono"  # =N-NH2
        raise _Fail("bad =N replacement")
    raise _Fail(f"unexpected double-bond atom {sym}")


def _bridge_replacement(bridge_atoms, mol) -> str | None:
    """Prefix for a bridging acid position between two carbons.

    ``bridge_atoms`` is a tuple of one atom index (single-atom bridge) or two
    (peroxy -O-O-).  Returns "thio"/"seleno"/"telluro"/"imido"/"peroxy" or
    ``None`` for an unchanged -O- bridge.  Raises ``_Fail`` otherwise.
    """
    if len(bridge_atoms) == 2:
        # peroxy -O-O- bridge (both atoms validated as O in _perceive).
        return "peroxy"
    atom = mol.GetAtomWithIdx(bridge_atoms[0])
    sym = atom.GetSymbol()
    if sym == "O":
        return None  # plain -O- bridge
    if sym in ("S", "Se", "Te"):
        if atom.GetFormalCharge() != 0 or atom.GetTotalNumHs() != 0:
            raise _Fail("bad bridge chalcogen")
        return _CHALC_INFIX[sym]
    if sym == "N":
        # -NH- amide-type bridge -> imido (P-65.2.3.1.2: NH).
        if (atom.GetTotalNumHs() == 1 and atom.GetFormalCharge() == 0):
            return "imido"
        raise _Fail("bad bridge N")
    raise _Fail(f"unexpected bridge atom {sym}")


def _perceive(mol):
    """Decompose *mol* into a carbonic-acid chain, or raise ``_Fail``."""
    from rdkit import Chem  # local import

    if mol.GetNumAtoms() == 0:
        raise _Fail("empty")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise _Fail("multi-fragment")
    if Chem.GetFormalCharge(mol) != 0:
        raise _Fail("charged")
    # Closed-shell only — never name a genuine open-shell radical (preserve
    # the engine's free-valence guard authority).
    for a in mol.GetAtoms():
        if a.GetNumRadicalElectrons():
            raise _Fail("radical")
        if a.IsInRing() or a.GetIsAromatic():
            raise _Fail("ring / aromatic")

    # --- Identify the "carbonic carbons": sp2 carbons each carrying exactly
    # one doubly-bonded acid position and the rest single-bonded acid /
    # bridge positions. ---
    carbons = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 6]
    if not carbons:
        raise _Fail("no carbon")
    carbon_idxs = {c.GetIdx() for c in carbons}

    # Per-carbon: the =X atom, the single-bonded neighbour atoms.
    dbl_atom: dict[int, object] = {}
    single_nbrs: dict[int, list] = {}
    for c in carbons:
        if c.GetDegree() != 3 or c.GetTotalNumHs() != 0:
            raise _Fail("non-carbonic carbon")
        dbl = None
        singles = []
        for nb in c.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(c.GetIdx(), nb.GetIdx())
            order = bond.GetBondTypeAsDouble()
            if order == 2.0:
                if dbl is not None:
                    raise _Fail("two double bonds on carbon")
                dbl = nb
            elif order == 1.0:
                singles.append(nb)
            else:
                raise _Fail("aromatic/odd bond on carbon")
        if dbl is None or len(singles) != 2:
            raise _Fail("carbon not C(=X)(-)(-)")
        # the doubly-bonded atom must be a terminal acid position (degree 1
        # chalcogen, or N for imido/hydrazono).
        if dbl.GetIdx() in carbon_idxs:
            raise _Fail("C=C double bond")
        dbl_atom[c.GetIdx()] = dbl
        single_nbrs[c.GetIdx()] = singles

    # --- Build the carbon chain through bridge atoms. ---
    # A bridge connects two carbons via one atom (-O-/-S-/-NH-/...) or two
    # atoms (peroxy -O-O-); a terminal acid atom connects only one carbon.
    # Map carbon -> list of (neighbour-carbon, bridge-atom-idx-tuple).
    adj: dict[int, list[tuple[int, tuple]]] = {ci: [] for ci in carbon_idxs}
    terminal_sites: dict[int, list] = {ci: [] for ci in carbon_idxs}
    for ci in carbon_idxs:
        for nb in single_nbrs[ci]:
            # single-atom bridge: nb links directly to a SECOND carbon.
            other_carbons = [
                x for x in nb.GetNeighbors()
                if x.GetIdx() != ci and x.GetIdx() in carbon_idxs
            ]
            if other_carbons:
                if len(other_carbons) != 1:
                    raise _Fail("bridge to multiple carbons")
                adj[ci].append((other_carbons[0].GetIdx(), (nb.GetIdx(),)))
                continue
            # peroxy bridge: nb is an O bonded to exactly one further O which
            # in turn bonds a SECOND carbon (-O-O- between two carbons).
            if nb.GetSymbol() == "O" and nb.GetDegree() == 2:
                far = [x for x in nb.GetNeighbors() if x.GetIdx() != ci]
                if (len(far) == 1 and far[0].GetSymbol() == "O"
                        and far[0].GetDegree() == 2
                        and far[0].GetFormalCharge() == 0
                        and far[0].GetTotalNumHs() == 0):
                    o2 = far[0]
                    c2 = [x for x in o2.GetNeighbors()
                          if x.GetIdx() != nb.GetIdx()]
                    if (len(c2) == 1 and c2[0].GetIdx() in carbon_idxs
                            and mol.GetBondBetweenAtoms(
                                o2.GetIdx(), c2[0].GetIdx()
                            ).GetBondTypeAsDouble() == 1.0):
                        adj[ci].append(
                            (c2[0].GetIdx(), (nb.GetIdx(), o2.GetIdx())))
                        continue
            # otherwise it is a terminal acid position on this carbon.
            terminal_sites[ci].append(nb)

    # Each carbon must have exactly 2 (terminal-acid + bridge) single
    # positions total; degree-2 bridges count once per carbon.
    # Find chain ends: carbons with exactly one bridge neighbour.
    ends = [ci for ci in carbon_idxs if len(adj[ci]) == 1]
    n = len(carbon_idxs)
    if n == 1:
        # Mononuclear carbonic acid and its replacement analogues
        # (carbonic / carbonimidic / carbonothioic / carbonoperoxoic) are
        # named by the distinct P-65.2.1 mononuclear scheme (carbono- infix,
        # S/O letter locants) and already round-trip via the engine's
        # ordinary machinery; decline so this polynuclear (P-65.2.3) generator
        # never overrides them.
        raise _Fail("mononuclear out of scope")
    else:
        if len(ends) != 2:
            raise _Fail("not a linear chain (ends != 2)")
        # Walk the chain from one end to the other.
        start = min(ends)
        chain = [start]
        bridges_in_order = []
        prev = None
        cur = start
        seen = {start}
        while True:
            nxts = [(c2, b) for (c2, b) in adj[cur] if c2 != prev]
            if not nxts:
                break
            if len(nxts) != 1:
                raise _Fail("branched chain")
            c2, b = nxts[0]
            if c2 in seen:
                raise _Fail("cyclic chain")
            bridges_in_order.append(b)
            chain.append(c2)
            seen.add(c2)
            prev, cur = cur, c2
        if len(chain) != n:
            raise _Fail("disconnected chain")

    # --- Verify terminal-site counts: chain ends carry exactly 1 terminal
    # acid site; internal carbons carry 0. ---
    for pos, ci in enumerate(chain):
        is_end = (pos == 0 or pos == n - 1)
        want = 1 if is_end else 0
        if len(terminal_sites[ci]) != want:
            raise _Fail("bad terminal-site count")

    return {
        "chain": chain,            # carbon idxs in chain order
        "bridges": bridges_in_order,  # bridge atom idxs between successive C
        "dbl_atom": dbl_atom,      # carbon idx -> doubly-bonded atom
        "terminal_sites": terminal_sites,  # carbon idx -> [terminal atom]
        "n": n,
    }


def _sites_for_direction(mol, per, reverse: bool):
    """Return the list of (_Site) for one numbering direction.

    Raises ``_Fail`` if a per-position shape is not nameable in this scheme.
    """
    chain = list(per["chain"])
    bridges = list(per["bridges"])
    if reverse:
        chain = chain[::-1]
        bridges = bridges[::-1]
    mol_dbl = per["dbl_atom"]
    term = per["terminal_sites"]
    n = per["n"]

    sites: list[_Site] = []
    for i, ci in enumerate(chain):  # i is 0-indexed; carbon locant = 2*i + 1
        odd = 2 * i + 1
        # doubly-bonded acid position at the odd locant.
        dpref = _double_bond_replacement(mol_dbl[ci])
        sites.append(_Site(odd, dpref))
        # terminal single-bonded acid position (only on the two chain ends).
        is_end = (i == 0 or i == n - 1)
        if is_end:
            t_atoms = term[ci]
            # _perceive guaranteed exactly one terminal site on an end.
            tpref = _terminal_chalcogen_replacement(t_atoms[0], mol, ci)
            sites.append(_Site(odd, tpref))
        # bridge to the next carbon at the even locant.
        if i < n - 1:
            sites.append(_Site(2 * i + 2,
                               _bridge_replacement(bridges[i], mol)))
    return sites


def _render(sites: list[_Site], n: int) -> str | None:
    """Render the final acid name from numbered replacement sites."""
    stem = _CHAIN_STEM.get(n)
    if stem is None:
        return None  # polycarbonic (n >= 5): skeletal-replacement scope

    # Collect replacement prefixes with their locants.
    by_prefix: dict[str, list[int]] = {}
    for s in sites:
        if s.prefix is None:
            continue
        by_prefix.setdefault(s.prefix, []).append(s.locant)

    if not by_prefix:
        # Unmodified parent acid (carbonic / dicarbonic / ...).
        return f"{stem} acid"

    # A single replacement prefix that covers EVERY acid position is cited
    # without locants (P-65.2.3.1.2 example "pentathiodicarbonic acid" for
    # the fully thio-replaced HS-CS-S-CS-SH): the all-positions count makes
    # the locants redundant.  ``n_positions`` = the doubly-bonded site of
    # every carbon (n) + every terminal -OH (2) + every bridge (n-1).
    n_positions = n + 2 + (n - 1)
    total_replaced = sum(len(v) for v in by_prefix.values())
    omit_locants = (len(by_prefix) == 1 and total_replaced == n_positions)

    # Build each prefix block: "<locants>-<mult><infix>" alphabetised by the
    # infix name (P-65.2.3.1.1: alphabetical order of the replacement
    # prefixes).
    blocks: list[tuple[str, str]] = []  # (sort_key, rendered)
    for pref, locs in by_prefix.items():
        locs = sorted(locs)
        cnt = len(locs)
        mult = _MULT.get(cnt)
        if mult is None:
            return None
        if omit_locants:
            rendered = f"{mult}{pref}"
        else:
            loc_str = ",".join(str(x) for x in locs)
            rendered = f"{loc_str}-{mult}{pref}"
        blocks.append((pref, rendered))

    blocks.sort(key=lambda b: b[0])
    # Join the prefix blocks: a hyphen separates a locant set (which opens the
    # next block) from the preceding alphabetic infix (P-16.3.3); the
    # omit-locant single-block form ("pentathio…") concatenates directly.
    rendered_blocks = [b[1] for b in blocks]
    if omit_locants:
        prefix_text = "".join(rendered_blocks)
    else:
        prefix_text = "-".join(rendered_blocks)
    return prefix_text + stem + " acid"


def _needs_letter_locants(per, sites: list[_Site]) -> bool:
    """True when the chalcogen tautomer cannot be expressed numerically.

    A terminal carbon whose chalcogen replacement sits on the -OH (single)
    position while its =O stays oxygen is ambiguous with the =O-replaced
    tautomer and requires the superscript letter locants (P-65.2.3.1.2.2),
    which OPSIN does not parse — decline so we never emit a non-round-tripping
    name.  Detected as: at some odd (carbon) locant, the terminal site is a
    chalcogen replacement but the doubly-bonded site is NOT.
    """
    # group sites by locant
    from collections import defaultdict
    odd_dbl: dict[int, str | None] = {}
    odd_term: dict[int, str | None] = {}
    seen_odd: defaultdict = defaultdict(list)
    for s in sites:
        if s.locant % 2 == 1:
            seen_odd[s.locant].append(s.prefix)
    chalc = {"thio", "seleno", "telluro"}
    for loc, prefs in seen_odd.items():
        # The double-bond site is appended first, terminal second (per
        # _sites_for_direction order) — but only ends carry a terminal.
        if len(prefs) == 2:
            dbl_p, term_p = prefs[0], prefs[1]
            if term_p in chalc and dbl_p not in chalc and dbl_p is None:
                return True
    return False


def _direction_key(sites: list[_Site]):
    """Lowest-locant key for a numbering direction (P-14.5)."""
    return sorted(s.locant for s in sites if s.prefix is not None)


def compute_carbonic_acid_name(mol) -> str | None:
    """Name a carbonic / di-/tri-/tetracarbonic acid or replacement analogue.

    Returns the IUPAC preferred name (e.g. ``"dicarbonic acid"``,
    ``"1-imidodicarbonic acid"``, ``"1,1,3,3-tetrathiodicarbonic acid"``,
    ``"1,2,3,4,5-pentaimidotricarbonic acid"``) or ``None`` when *mol* is not
    a clean carbonic-acid chain skeleton.

    P-65.2 (carbonic, cyanic, and di- and polycarbonic acids).
    """
    if mol is None:
        return None
    try:
        per = _perceive(mol)
    except _Fail:
        return None
    except Exception:  # pragma: no cover — never break the engine
        return None

    try:
        fwd = _sites_for_direction(mol, per, reverse=False)
        rev = _sites_for_direction(mol, per, reverse=True)
    except _Fail:
        return None
    except Exception:  # pragma: no cover
        return None
    # choose lowest-locant direction (P-14.5)
    chosen = min((fwd, rev), key=_direction_key)
    if _needs_letter_locants(per, chosen):
        return None
    return _render(chosen, per["n"])
