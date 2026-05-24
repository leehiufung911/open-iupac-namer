# open-iupac-namer

A from-scratch Python engine that generates **IUPAC names** for chemical
structures, following the 2013 IUPAC *Nomenclature of Organic Chemistry*
recommendations (the "Blue Book"). Give it a molecule (as a SMILES string) and
it returns a systematic — or, where one exists, retained — IUPAC name.

```python
from iupac_namer import name_smiles

name_smiles("CCO")                              # 'ethanol'
name_smiles("CC(=O)OCC")                        # 'ethyl acetate'
name_smiles("O=C(O)c1ccccc1")                   # 'benzoic acid'
name_smiles("c1ccc2ccccc2c1")                   # 'naphthalene'
name_smiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O")       # 'ibuprofen'

# ...and more complex structures:

# a thieno[2,3-d]pyrimidine bisphosphonate
name_smiles("O=C(C(C1)CCCN1C(N=C2NC(P(OCC)(OCC)=O)P(OCC)(OCC)=O)=NC3=C2C=CS3)OCC")
# -> 'ethyl 1-{4-{{bis[di(ethoxy)(oxo)phosphanyl]methyl}amino}thieno[2,3-d]pyrimidin-2-yl}piperidine-3-carboxylate'

# a chiral phosphinyl peptide
name_smiles("O=[P@](C[C@@H](CC1=CC=CC=C1)C(OC)=O)(OCC)[C@@H](NC(C2=CC(Cl)=NC=C2)=O)CC3=CC=CC=C3")
# -> 'methyl (2S)-3-phenyl-2-{{(1R)-[(1R)-1-(2-chloropyridine-4-carbonylamino)-2-phenylethyl](ethoxy)(oxo)phosphanyl}methyl}propanoate'
```

Every name above is generated from the molecular graph, not looked up — and each
one round-trips through [OPSIN](https://github.com/dan2097/opsin) back to the
original structure. The engine works *structure → name*; for the reverse
direction (name → structure) see OPSIN, which this project uses as an independent
oracle for testing.

---

## Why?

A non-commercial, non-proprietary structure → name engine does **not** exist.
The capable tools — ChemDraw, ACD/Labs, and the like — are all commercial and
closed. The one notable open effort, STOUTv2, is a neural translator: it is not
rule-based (which a from-first-principles IUPAC namer should be) and is no longer
publicly available. This project aims to build a **real, robust, and open**
structure → name engine focused on general organic chemistry.

Going structure → name means reproducing the Blue Book's decision procedure:
choosing the principal characteristic group, picking the parent hydride,
numbering it to satisfy a cascade of tie-breakers, choosing between retained and
systematic ring names, and assembling everything with the right elisions,
brackets, and alphabetical ordering — with an architecture designed so each of
those decisions lives in one well-defined place.

**Second reason:** this is an experiment in spec-driven agentic coding — given a
clear but complex spec (in this case the Blue Book), can a coding agent
([Claude Code](https://claude.com/claude-code)) build something as complex as an
IUPAC namer? (Subjective answer: not 100% yet — a human (me) had to step in at
many points — but maybe 50% of the way there in this case.)

## What it handles

> ⚠️ **Experimental.** The engine is still under active development.

The emphasis is on **structural correctness, not canonical correctness**: the
goal is a name that is unambiguous and parses back to the original structure
(verified throughout by OPSIN round-trip) — not necessarily the single
IUPAC-*preferred* name (PIN). A molecule usually has several valid IUPAC names;
the engine aims to always produce one that is correct, even when it isn't the
preferred form.

With that in mind, it should handle:

- **Chains and substitutive nomenclature** — alkanes/enes/ynes, principal
  characteristic group selection (P-65 seniority), parent selection (P-44),
  locant minimization.
- **Rings** — monocycles, fused polycyclics (von Baeyer, spiro, bridged),
  retained ring names, Hantzsch–Widman heterocycles, ring assemblies.
- **Functional groups** — full suffix/prefix handling, functional-class names
  (e.g. esters, acyl halides), acids and their derivatives, infix modifiers.
- **Stereochemistry** — CIP (*R/S*, *E/Z*) assignment and descriptor emission.
- **Isotopes** — isotopic labelling (e.g. deuterium, ¹³C).
- **Retained & natural-product names** — Blue-Book retained names, steroids,
  terpenes, and other scaffolds with curated stems.

## How it works

The engine is organised as a pipeline of layers with clear contracts. Naming is
treated as **a sequence of coupled choices**: choose a structural interpretation,
then a naming *plan* within it, then assemble a string.

```
Perception  →  Engine (plan search)  ⇄  Strategy (scoring)  →  Assembly
  what's          generates &              what's              deterministic
  structurally    executes plans,          preferred           string building
  valid           recurses on fragments
```

- **Perception** resolves structural facts (atoms, rings, functional groups,
  stereo, symmetry, chains) and yields candidate interpretations. It never makes
  naming decisions.
- **Strategy** scores plans — seniority, parent choice, numbering quality — and
  accepts or rejects them. Different strategies (canonical, CAS-like, aligned)
  reuse the same engine.
- **Engine** generates plans, executes the best, and **recurses** on
  sub-fragments (a substituent is named by naming its fragment). All mutable
  state is scoped to one `NamingSession`; data structures are immutable.
- **Assembly** turns the resulting name tree into a string: alphabetical prefix
  ordering, multiplier insertion, vowel elision, brackets, stereo descriptors.

The full design is documented in [`docs/`](docs/) — start with
[`ARCHITECTURE_OVERVIEW.md`](docs/ARCHITECTURE_OVERVIEW.md).

## Installation

Requires Python ≥ 3.11. The only runtime dependency is
[RDKit](https://www.rdkit.org/) (used for parsing SMILES and perceiving
structure).

```bash
git clone https://github.com/leehiufung911/open-iupac-namer.git
cd open-iupac-namer
pip install -e .          # editable install; keeps data/ discoverable
```

> **Note on data files.** The curated JSON tables in `data/` are loaded relative
> to the repository root, so use the engine from a clone or an **editable**
> install (`pip install -e .`). A plain wheel install would not bundle `data/`.

## Usage

```python
from iupac_namer import name_smiles

print(name_smiles("CC(C)O"))   # 'propan-2-ol'
```

`name_smiles(smiles: str) -> str` is the main entry point. The lower-level
`name(...)` function and the typed data structures (plans, name trees,
`OutputForm`, etc.) are also exported from the top-level package for advanced use.

## Testing

The unit-test suite (130+ files in [`tests/`](tests/)) covers individual
nomenclature features:

```bash
pip install -e ".[test]"
pytest tests/
```

Some tests round-trip generated names through OPSIN and therefore need a Java
runtime. They read `JAVA_HOME` from the environment if set; otherwise they rely
on `java` being on your `PATH`. Tests that need OPSIN are skipped gracefully when
`py2opsin`/Java are unavailable.

## Evaluation (OPSIN round-trip)

The engine is validated by an **independent round-trip**: name a structure, feed
the generated name back into OPSIN, and check that OPSIN's parsed structure
matches the original. This catches names that are wrong without needing a
hand-curated answer key.

```bash
# names every SMILES in eval/testset.json, round-trips through OPSIN
./eval/run_eval.sh

# or a single molecule
python eval/authoritative_eval.py --smiles "CCO"
```

`eval/testset.json` is a small public example. Point the `IUPAC_NAMER_EVAL_DATA`
environment variable at your own JSON (same shape — a `{"compounds": [{"smiles":
...}]}` list) to evaluate against a larger set. Running the full eval needs
`py2opsin` and a Java runtime.

## Project structure

```
open-iupac-namer/
├── iupac_namer/        the naming engine (Python package)
│   ├── perception/     structural perception (atoms, rings, FGs, stereo, chains)
│   ├── ring_naming/    retained + systematic ring naming
│   ├── natural_products/  steroid/terpene/etc. stems
│   ├── engine.py       plan search, recursion, path handlers
│   ├── strategy.py     plan scoring and accept/reject
│   ├── assembly.py     deterministic string building
│   └── types.py        all typed (frozen) data structures
├── data/               curated JSON tables (stems, prefixes, retained names, …)
├── docs/               architecture specification (7 documents)
├── tests/              unit tests
└── eval/               OPSIN round-trip eval harness + example test set
```

## Data sources & attribution

- **[RDKit](https://www.rdkit.org/)** — cheminformatics toolkit (BSD), used for
  structure perception.
- **[OPSIN](https://github.com/dan2097/opsin)** — name → structure parser (MIT).
  Some of the ring/retained-name reference tables in `data/opsin_extracted/`
  were derived from OPSIN's open vocabulary, and OPSIN is the round-trip oracle
  used for evaluation.
- **IUPAC 2013 Recommendations** ("Blue Book") — the nomenclature rules this
  engine implements. The JSON tables encode nomenclature *facts and rules*
  (stems, prefixes, seniority orderings); they are not reproductions of the
  Blue Book text.

## Status & limitations

This is an early-stage, independent implementation, not an official IUPAC
product. It covers a broad range of organic nomenclature but is not exhaustive —
very large biomolecules, some exotic ring systems, and certain edge cases may
produce an error or a non-preferred (though still valid) name. Names are
generated algorithmically and should be sanity-checked for critical use.
Contributions, bug reports, and counter-examples are welcome.

## License

[MIT](LICENSE) © 2026 leehiufung911 · leehiufung911@gmail.com
