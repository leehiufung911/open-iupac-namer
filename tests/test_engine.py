"""
tests/test_engine.py

Phase 1.6 engine skeleton tests.

Focus: no crashes, correct tree types, assemble() produces a name string,
pipeline works end-to-end.  Exact IUPAC correctness is a Phase 2 concern.
"""
from __future__ import annotations

import pytest

from rdkit import Chem

from iupac_namer.engine import name, name_smiles
from iupac_namer.strategy import IUPACCanonical
from iupac_namer.assembly import assemble
from iupac_namer.types import (
    OutputForm, NamingSession,
    SubstitutiveTree, LeafTree, SaltTree, ErrorTree,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _name(smiles: str) -> str:
    """Convenience: name a SMILES and return the assembled string."""
    return name_smiles(smiles)


def _tree(smiles: str) -> object:
    """Return the raw NameTree for a SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    return name(mol, IUPACCanonical())


# ---------------------------------------------------------------------------
# Basic hydrocarbons (no functional groups)
# ---------------------------------------------------------------------------

class TestHydrocarbons:
    def test_methane_name(self):
        result = _name("C")
        assert result == "methane", f"Expected 'methane', got {result!r}"

    def test_ethane_name(self):
        result = _name("CC")
        assert result == "ethane", f"Expected 'ethane', got {result!r}"

    def test_propane_name(self):
        result = _name("CCC")
        assert result == "propane", f"Expected 'propane', got {result!r}"

    def test_methane_tree_type(self):
        tree = _tree("C")
        assert isinstance(tree, SubstitutiveTree), f"Expected SubstitutiveTree, got {type(tree).__name__}"

    def test_ethane_tree_type(self):
        tree = _tree("CC")
        assert isinstance(tree, SubstitutiveTree), f"Expected SubstitutiveTree, got {type(tree).__name__}"

    def test_propane_tree_type(self):
        tree = _tree("CCC")
        assert isinstance(tree, SubstitutiveTree), f"Expected SubstitutiveTree, got {type(tree).__name__}"

    def test_methane_no_error_children(self):
        tree = _tree("C")
        assert not isinstance(tree, ErrorTree)

    def test_ethane_no_error_children(self):
        tree = _tree("CC")
        assert not isinstance(tree, ErrorTree)


# ---------------------------------------------------------------------------
# Alcohols
# ---------------------------------------------------------------------------

class TestAlcohols:
    def test_methanol_contains_ol(self):
        result = _name("CO")
        # Should contain "ol" (methanol)
        assert "ol" in result, f"Expected 'ol' in methanol name, got {result!r}"

    def test_ethanol_contains_ol(self):
        result = _name("CCO")
        assert "ol" in result, f"Expected 'ol' in ethanol name, got {result!r}"

    def test_methanol_tree_type(self):
        tree = _tree("CO")
        assert isinstance(tree, SubstitutiveTree), f"Expected SubstitutiveTree, got {type(tree).__name__}"

    def test_ethanol_tree_type(self):
        tree = _tree("CCO")
        assert isinstance(tree, SubstitutiveTree), f"Expected SubstitutiveTree, got {type(tree).__name__}"

    def test_methanol_not_error(self):
        tree = _tree("CO")
        assert not isinstance(tree, ErrorTree), f"Got error: {tree.message if isinstance(tree, ErrorTree) else ''}"

    def test_ethanol_not_error(self):
        tree = _tree("CCO")
        assert not isinstance(tree, ErrorTree), f"Got error: {tree.message if isinstance(tree, ErrorTree) else ''}"


# ---------------------------------------------------------------------------
# Carboxylic acids
# ---------------------------------------------------------------------------

class TestAcids:
    def test_acetic_acid_contains_acid(self):
        result = _name("CC(=O)O")
        # Should contain "acid" somewhere in the name
        assert "acid" in result, f"Expected 'acid' in acetic acid name, got {result!r}"

    def test_acetic_acid_not_error(self):
        tree = _tree("CC(=O)O")
        assert not isinstance(tree, ErrorTree), (
            f"Got error: {tree.message if isinstance(tree, ErrorTree) else ''}"
        )


# ---------------------------------------------------------------------------
# Substituents (branched molecules)
# ---------------------------------------------------------------------------

class TestSubstituents:
    def test_2_methylpropane_tree_type(self):
        # isobutane: CC(C)C — may be retained "isobutane" or substitutive "2-methylpropane"
        tree = _tree("CC(C)C")
        # Either a retained name (LeafTree) or a systematic substitutive tree is acceptable
        assert isinstance(tree, (SubstitutiveTree, LeafTree)), (
            f"Expected SubstitutiveTree or LeafTree, got {type(tree).__name__}"
        )
        # If systematic, the parent should be propane or butane based
        if isinstance(tree, SubstitutiveTree):
            parent_name = tree.named_parent.name
            assert "propane" in parent_name or "butane" in parent_name, (
                f"Expected propane or butane parent, got {parent_name!r}"
            )

    def test_2_methylpropane_produces_name(self):
        result = _name("CC(C)C")
        assert len(result) > 0, "Expected non-empty name"
        # Should be some form of butane or methylpropane
        assert "methyl" in result or "butane" in result, (
            f"Expected methyl or butane in result, got {result!r}"
        )

    def test_2_methylpropane_not_error(self):
        tree = _tree("CC(C)C")
        assert not isinstance(tree, ErrorTree)

    def test_2_methylbutane_no_hyphen_in_methyl(self):
        """Bug 1 regression: substituent should be 'methyl' not 'meth-yl'."""
        result = _name("CCC(C)C")
        # The assembled name must NOT contain "meth-yl" (spurious hyphen)
        assert "meth-yl" not in result, (
            f"Spurious hyphen in substituent name: {result!r}"
        )
        # And should produce the correct IUPAC name
        assert result == "2-methylbutane", (
            f"Expected '2-methylbutane', got {result!r}"
        )

    def test_3_ethylheptane_correct(self):
        """Bug 1 regression: substituent should be 'ethyl' not 'eth-yl'."""
        result = _name("CCCCC(CC)CC")
        assert "eth-yl" not in result, (
            f"Spurious hyphen in substituent name: {result!r}"
        )
        assert result == "3-ethylheptane", (
            f"Expected '3-ethylheptane', got {result!r}"
        )

    def test_no_parentheses_around_simple_substituent(self):
        """Bug 2 regression: simple substituents should NOT be wrapped in parentheses."""
        result = _name("CCC(C)C")
        assert "(" not in result, (
            f"Unexpected parentheses around simple substituent: {result!r}"
        )
        result2 = _name("CCCCC(CC)CC")
        assert "(" not in result2, (
            f"Unexpected parentheses around simple substituent: {result2!r}"
        )

    def test_plain_chains_still_work(self):
        """Basic chains should still be named correctly after the fix."""
        assert _name("CCCC") == "butane"
        assert _name("CC") == "ethane"
        assert _name("CCC") == "propane"


# ---------------------------------------------------------------------------
# Chain unsaturation stem format (Bug 1 fixes: P-31.1.2.1)
# ---------------------------------------------------------------------------

class TestChainUnsaturationStem:
    """Tests that unsaturated chains use the alkyl stem (drop -ane ending).

    IUPAC rule: when a chain has double/triple bonds, the parent stem
    drops the "-ane" ending.  "butane" -> "but-2-ene" (not "butan-2-ene").
    """

    def test_propene_no_propan(self):
        """Propene should be 'prop-X-ene', not 'propan-X-ene'."""
        result = _name("C=CC")
        assert "propan" not in result, (
            f"Expected 'prop-' stem (not 'propan-') for propene, got {result!r}"
        )
        assert "prop" in result, f"Expected 'prop' in propene name, got {result!r}"
        assert "ene" in result, f"Expected 'ene' in propene name, got {result!r}"

    def test_butene_no_butan(self):
        """But-X-ene should not contain 'butan-'."""
        result = _name("C=CCC")
        assert "butan" not in result, (
            f"Expected 'but-' stem (not 'butan-') for butene, got {result!r}"
        )
        assert "but" in result, f"Expected 'but' in butene name, got {result!r}"
        assert "ene" in result, f"Expected 'ene' in butene name, got {result!r}"

    def test_pentene_no_pentan(self):
        """Pent-X-ene should not contain 'pentan-'."""
        result = _name("C=CCCC")
        assert "pentan" not in result, (
            f"Expected 'pent-' stem (not 'pentan-') for pentene, got {result!r}"
        )

    def test_hexene_no_hexan(self):
        """Hex-X-ene should not contain 'hexan-'."""
        result = _name("C=CCCCC")
        assert "hexan" not in result, (
            f"Expected 'hex-' stem (not 'hexan-') for hexene, got {result!r}"
        )

    def test_butyne_no_butan(self):
        """But-X-yne should not contain 'butan-'."""
        result = _name("C#CCC")
        assert "butan" not in result, (
            f"Expected 'but-' stem for butyne, got {result!r}"
        )
        assert "yne" in result, f"Expected 'yne' in butyne name, got {result!r}"

    def test_unsaturated_with_suffix_no_butan(self):
        """But-3-en-1-ol should not contain 'butan-'."""
        result = _name("C(CC=C)O")  # but-3-en-1-ol
        assert "butan" not in result, (
            f"Expected 'but-' stem for but-3-en-1-ol, got {result!r}"
        )
        assert "but" in result, f"Expected 'but' in name, got {result!r}"
        assert "ol" in result, f"Expected 'ol' suffix, got {result!r}"

    def test_saturated_chain_unchanged(self):
        """Saturated chains should still use 'butan', 'hexan', etc. for stems."""
        # butane - no unsaturation
        assert _name("CCCC") == "butane", f"Expected 'butane', got {_name('CCCC')!r}"
        # butan-1-ol
        result_ol = _name("CCCCO")
        assert "butan" in result_ol, (
            f"Expected 'butan-' stem in butan-1-ol, got {result_ol!r}"
        )

    def test_saturated_with_suffix_unchanged(self):
        """Saturated chains with FG suffix should still use '-an' stem."""
        result = _name("CCCO")  # propan-1-ol
        assert "propan" in result, (
            f"Expected 'propan' in propan-1-ol, got {result!r}"
        )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_invalid_smiles_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid SMILES"):
            name_smiles("invalid_smiles_xyz")

    def test_empty_smiles_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid SMILES"):
            name_smiles("this_is_not_smiles")


# ---------------------------------------------------------------------------
# Salt naming
# ---------------------------------------------------------------------------

class TestSaltNaming:
    def test_sodium_chloride_produces_salt_tree(self):
        # NaCl: [Na+].[Cl-]
        tree = _tree("[Na+].[Cl-]")
        assert isinstance(tree, SaltTree), f"Expected SaltTree, got {type(tree).__name__}"

    def test_sodium_chloride_has_two_fragments(self):
        tree = _tree("[Na+].[Cl-]")
        if isinstance(tree, SaltTree):
            assert len(tree.ion_trees) == 2, f"Expected 2 ion trees, got {len(tree.ion_trees)}"

    def test_sodium_chloride_produces_name(self):
        result = _name("[Na+].[Cl-]")
        assert len(result) > 0, "Expected non-empty name for salt"

    def test_salt_not_error(self):
        tree = _tree("[Na+].[Cl-]")
        assert not isinstance(tree, ErrorTree)


# ---------------------------------------------------------------------------
# Session caching
# ---------------------------------------------------------------------------

class TestSessionCaching:
    def test_same_smiles_uses_cache(self):
        """Naming the same molecule twice in the same session returns the same tree."""
        mol = Chem.MolFromSmiles("CC")
        strategy = IUPACCanonical()
        session = NamingSession()

        tree1 = name(mol, strategy, _session=session)
        tree2 = name(mol, strategy, _session=session)

        assert tree1 is tree2, "Expected same tree object (cache hit)"

    def test_different_smiles_distinct_trees(self):
        """Different molecules produce different trees."""
        mol1 = Chem.MolFromSmiles("CC")
        mol2 = Chem.MolFromSmiles("CCC")
        strategy = IUPACCanonical()
        session = NamingSession()

        tree1 = name(mol1, strategy, _session=session)
        tree2 = name(mol2, strategy, _session=session)

        assert assemble(tree1) != assemble(tree2), "Expected different names for different molecules"


# ---------------------------------------------------------------------------
# Pipeline smoke test
# ---------------------------------------------------------------------------

class TestPipeline:
    """End-to-end: SMILES → tree → assembled string, no crash."""

    @pytest.mark.parametrize("smiles", [
        "C",        # methane
        "CC",       # ethane
        "CCC",      # propane
        "CCCC",     # butane
        "CCCCC",    # pentane
        "CO",       # methanol
        "CCO",      # ethanol
        "CC(C)C",   # isobutane
    ])
    def test_smoke_no_crash(self, smiles):
        result = name_smiles(smiles)
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.parametrize("smiles", [
        "C", "CC", "CCC",
    ])
    def test_smoke_no_error_brackets(self, smiles):
        """Result should not be an error message."""
        result = name_smiles(smiles)
        assert not result.startswith("[NAMING ERROR"), f"Got error for {smiles!r}: {result}"


# ---------------------------------------------------------------------------
# FG atom exclusion (Phase 2a-2)
# ---------------------------------------------------------------------------

class TestFGAtomExclusion:
    """FG-claimed atoms must not be carved as structural substituents.

    Before the fix, halogen atoms (Cl, F, Br, I) and other FG atoms were
    carved as tiny fragments like [H]Cl, which could not be named, producing
    NAMING_ERROR in the output.

    After the fix, the FG's prefix_form is used directly.
    """

    def test_chloropropane_no_naming_error(self):
        """Chloro substituent should use prefix_form 'chloro', not try to name [H]Cl."""
        result = _name("CCCCl")
        assert "NAMING ERROR" not in result, (
            f"Got NAMING ERROR for 1-chloropropane: {result!r}"
        )

    def test_chloropropane_contains_chloro(self):
        """Chloro prefix should appear in name."""
        result = _name("CCCCl")
        assert "chloro" in result, f"Expected 'chloro' in result, got {result!r}"

    def test_fluoropropane_no_naming_error(self):
        """Fluoro substituent should not produce a NAMING ERROR."""
        result = _name("CCCF")
        assert "NAMING ERROR" not in result, (
            f"Got NAMING ERROR for fluoropropane: {result!r}"
        )

    def test_fluoropropane_contains_fluoro(self):
        result = _name("CCCF")
        assert "fluoro" in result, f"Expected 'fluoro' in result, got {result!r}"

    def test_bromoethane_no_naming_error(self):
        result = _name("CCBr")
        assert "NAMING ERROR" not in result, (
            f"Got NAMING ERROR for bromoethane: {result!r}"
        )

    def test_bromoethane_contains_bromo(self):
        result = _name("CCBr")
        assert "bromo" in result, f"Expected 'bromo' in result, got {result!r}"

    def test_iodopropane_no_naming_error(self):
        result = _name("CCCI")
        assert "NAMING ERROR" not in result, (
            f"Got NAMING ERROR for iodopropane: {result!r}"
        )

    def test_iodopropane_contains_iodo(self):
        result = _name("CCCI")
        assert "iodo" in result, f"Expected 'iodo' in result, got {result!r}"

    def test_butanol_no_naming_error(self):
        """Alcohol FG as PCG should work — no NAMING ERROR."""
        result = _name("CCCCO")
        assert "NAMING ERROR" not in result, (
            f"Got NAMING ERROR for butan-1-ol: {result!r}"
        )

    def test_butanol_correct_name(self):
        """butan-1-ol should be named correctly."""
        result = _name("CCCCO")
        assert result == "butan-1-ol", f"Expected 'butan-1-ol', got {result!r}"

    def test_pure_hydrocarbons_unaffected(self):
        """Existing hydrocarbon naming must not be broken."""
        assert _name("CC") == "ethane"
        assert _name("CCC") == "propane"
        assert _name("CCC(C)C") == "2-methylbutane"

    def test_no_fg_atom_error_for_chlorobenzene(self):
        """Chlorobenzene ring compound: no NAMING ERROR from Cl carving."""
        result = _name("c1ccc(Cl)cc1")
        assert "NAMING ERROR" not in result, (
            f"Got NAMING ERROR for chlorobenzene: {result!r}"
        )
        assert "chloro" in result, f"Expected 'chloro' in result, got {result!r}"

    def test_amino_butane_no_error(self):
        """Amino group (suffix-eligible FG) as PCG: butan-1-amine."""
        result = _name("CCCCN")
        assert "NAMING ERROR" not in result, (
            f"Got NAMING ERROR for butan-1-amine: {result!r}"
        )

    def test_leaf_tree_for_fg_prefix(self):
        """FG-directed prefix must produce a LeafTree, not recurse into name()."""
        from iupac_namer.types import SubstitutiveTree, LeafTree
        mol = Chem.MolFromSmiles("CCCCl")
        tree = name(mol, IUPACCanonical())
        assert isinstance(tree, SubstitutiveTree), f"Expected SubstitutiveTree, got {type(tree)}"
        # Should have exactly one prefix
        assert len(tree.prefixes) == 1, f"Expected 1 prefix, got {len(tree.prefixes)}"
        # The prefix tree should be a LeafTree (fg_prefix path, not recursive)
        prefix_tree = tree.prefixes[0].tree
        assert isinstance(prefix_tree, LeafTree), (
            f"Expected LeafTree for FG prefix, got {type(prefix_tree).__name__}"
        )


# ---------------------------------------------------------------------------
# Phase 2a-4: Chain length and numbering fixes
# ---------------------------------------------------------------------------

class TestChainLengthAndNumbering:
    """Regression tests for Phase 2a-4 fixes:

    - Bug A: wrong parent chain (shorter chain chosen due to low-locant bonus)
    - Bug B: wrong numbering direction (PCG at wrong locant)
    - Bug C: wrong numbering for pure hydrocarbon substituents

    The fixes ensure:
    1. FG anchors point to the carbon bearing the FG (not the heteroatom)
    2. Chain length dominates over low-locant preference (IUPAC P-44.3)
    3. Lowest-locant numbering is applied after chain selection (P-14.5)
    4. Chain atoms are ordered by connectivity, not atom index
    """

    def test_butan_2_ol_chain_length(self):
        """CC(O)CC must use butane parent, not propane + methyl substituent."""
        result = _name("CC(O)CC")
        assert result == "butan-2-ol", (
            f"Expected 'butan-2-ol' (butane parent, locant 2), got {result!r}"
        )

    def test_butan_2_amine_chain_length(self):
        """CC(N)CC must use butane parent, not propane + methyl substituent."""
        result = _name("CC(N)CC")
        assert result == "butan-2-amine", (
            f"Expected 'butan-2-amine' (butane parent, locant 2), got {result!r}"
        )

    def test_butan_2_one_numbering(self):
        """CC(=O)CC: ketone must be at locant 2, not locant 1."""
        result = _name("CC(=O)CC")
        assert result == "butan-2-one", (
            f"Expected 'butan-2-one' (ketone at locant 2), got {result!r}"
        )

    def test_butan_2_one_symmetric_chain(self):
        """CCC(=O)C: ketone must be at locant 2 (lowest)."""
        result = _name("CCC(=O)C")
        assert result == "butan-2-one", (
            f"Expected 'butan-2-one', got {result!r}"
        )

    def test_2_2_dimethylpropane_numbering(self):
        """Neopentane CC(C)(C)C: methyls must be at locant 2, not locant 1 or 3."""
        result = _name("CC(C)(C)C")
        assert result == "2,2-dimethylpropane", (
            f"Expected '2,2-dimethylpropane', got {result!r}"
        )

    def test_2_methylbutane_locant(self):
        """CC(C)CC: methyl must be at locant 2, not locant 1 or 4."""
        result = _name("CC(C)CC")
        assert result == "2-methylbutane", (
            f"Expected '2-methylbutane', got {result!r}"
        )

    def test_existing_cases_still_work(self):
        """Previously-passing cases must not regress."""
        assert _name("CCC(C)C") == "2-methylbutane", "2-methylbutane regression"
        assert _name("CCCC") == "butane", "butane regression"
        assert _name("CCO") == "ethanol", "ethanol regression"
        assert _name("CCC(=O)C") == "butan-2-one", "butan-2-one regression"
        assert _name("CCCCC(CC)CC") == "3-ethylheptane", "3-ethylheptane regression"


# ---------------------------------------------------------------------------
# Phase 2a-5: Chain finding stops at heteroatoms; N-locant substituents
# ---------------------------------------------------------------------------

class TestChainDoesNotCrossHeteroatoms:
    """Regression tests for the heteroatom chain-crossing bug.

    Before the fix, chain finding traversed through N, O, S atoms as if they
    were carbons.  The longest path was computed across N, giving wrong parent
    chains like 'nonan-3-amine' for CCCCCCCN(CC)CC.

    After the fix:
    - Chain finding only traverses carbon atoms.
    - The correct parent chain is the longest carbon-only path.
    - N-substituents on an amine PCG get 'N-' locants.
    """

    def test_diethylamine_in_heptane_chain_not_crossed(self):
        """CCCCCCCN(CC)CC: chain must be 7C heptane, not 9-atom chain crossing N."""
        result = _name("CCCCCCCN(CC)CC")
        # Must NOT contain "nonan" (the incorrect 9-carbon chain through N)
        assert "nonan" not in result, (
            f"Chain incorrectly crossed N: {result!r}"
        )
        # Must contain "heptan" (the correct 7-carbon parent)
        assert "heptan" in result, (
            f"Expected 'heptan' parent chain, got {result!r}"
        )
        # Full correct name
        assert result == "N,N-diethylheptan-1-amine", (
            f"Expected 'N,N-diethylheptan-1-amine', got {result!r}"
        )

    def test_dimethylbutylamine_chain_not_crossed(self):
        """CCCCN(C)C: chain must be 4C butane, not 5-atom chain through N."""
        result = _name("CCCCN(C)C")
        assert "pentan" not in result, (
            f"Chain incorrectly crossed N: {result!r}"
        )
        assert "butan" in result, (
            f"Expected 'butan' parent chain, got {result!r}"
        )
        assert result == "N,N-dimethylbutan-1-amine", (
            f"Expected 'N,N-dimethylbutan-1-amine', got {result!r}"
        )

    def test_diethylamine_chain_not_crossed(self):
        """CCNCC: chain must be 2C ethane, not 4-atom chain through N."""
        result = _name("CCNCC")
        assert "butan" not in result, (
            f"Chain incorrectly crossed N: {result!r}"
        )
        assert result == "N-ethylethanamine", (
            f"Expected 'N-ethylethanamine', got {result!r}"
        )

    def test_primary_amine_unchanged(self):
        """CCCCN: primary amine; chain is 4C, N is the suffix only."""
        result = _name("CCCCN")
        assert result == "butan-1-amine", (
            f"Expected 'butan-1-amine', got {result!r}"
        )

    def test_non_amine_heteroatom_chain_still_correct(self):
        """CC(O)CC: alcohol O is NOT in chain; chain is 4C butane."""
        result = _name("CC(O)CC")
        assert result == "butan-2-ol", (
            f"Expected 'butan-2-ol' (chain does not include O), got {result!r}"
        )


class TestNLocantSubstituents:
    """Tests for N-locant prefix assignment on secondary/tertiary amines."""

    def test_dimethyl_butan_1_amine(self):
        """CCCCN(C)C: N,N-dimethylbutan-1-amine."""
        result = _name("CCCCN(C)C")
        assert result == "N,N-dimethylbutan-1-amine", (
            f"Expected 'N,N-dimethylbutan-1-amine', got {result!r}"
        )

    def test_diethyl_heptan_1_amine(self):
        """CCCCCCCN(CC)CC: N,N-diethylheptan-1-amine."""
        result = _name("CCCCCCCN(CC)CC")
        assert result == "N,N-diethylheptan-1-amine", (
            f"Expected 'N,N-diethylheptan-1-amine', got {result!r}"
        )

    def test_n_ethyl_ethanamine(self):
        """CCNCC: N-ethylethanamine."""
        result = _name("CCNCC")
        assert result == "N-ethylethanamine", (
            f"Expected 'N-ethylethanamine', got {result!r}"
        )

    def test_n_locant_prefix_contains_n_locant(self):
        """N-locant substituents must use 'N' locant, not a numeric one."""
        result = _name("CCCCN(C)C")
        # The name must contain 'N,' or 'N-' to show the N locant
        assert "N," in result or "N-" in result, (
            f"Expected N-locant in name, got {result!r}"
        )


# ---------------------------------------------------------------------------
# Ring substituent on chain-parent FG (regression for silent-drop bug)
# ---------------------------------------------------------------------------

class TestRingSubstituentOnChainFG:
    """Tests that ring atoms are NOT silently dropped when the FG SMARTS captures
    flanking carbons as context atoms (e.g. ketone [#6][CX3](=O)[#6]).

    Before the fix, the flanking ring carbon was placed in suffix_atoms, which
    prevented the remaining ring atoms from connecting to the parent chain.
    Result: the ring was silently dropped (e.g. CC(=O)c1ccccc1 → 'ethanone').
    """

    def test_phenyl_ketone_has_phenyl(self):
        """CC(=O)c1ccccc1: name must contain 'phenyl' (ring not dropped)."""
        result = _name("CC(=O)c1ccccc1")
        assert "phenyl" in result, (
            f"Expected 'phenyl' in name for CC(=O)c1ccccc1, got {result!r}"
        )

    def test_phenyl_ketone_reversed_smiles(self):
        """O=C(c1ccccc1)C: same molecule, different SMILES ordering."""
        result = _name("O=C(c1ccccc1)C")
        assert "phenyl" in result, (
            f"Expected 'phenyl' in name for O=C(c1ccccc1)C, got {result!r}"
        )

    def test_diphenyl_ketone_has_phenyl(self):
        """C(=O)(c1ccccc1)c1ccccc1: name must contain 'phenyl'."""
        result = _name("C(=O)(c1ccccc1)c1ccccc1")
        assert "phenyl" in result, (
            f"Expected 'phenyl' in name for diphenylmethanone, got {result!r}"
        )

    def test_alcohol_still_works(self):
        """CC(O)c1ccccc1: 1-phenylethanol (regression guard)."""
        result = _name("CC(O)c1ccccc1")
        assert result == "1-phenylethanol", (
            f"Expected '1-phenylethanol', got {result!r}"
        )

    def test_aldehyde_with_phenyl_still_works(self):
        """O=CCc1ccccc1: 2-phenylethanal (regression guard)."""
        result = _name("O=CCc1ccccc1")
        assert result == "2-phenylethanal", (
            f"Expected '2-phenylethanal', got {result!r}"
        )


# ---------------------------------------------------------------------------
# Multiplied suffix naming: terminal 'e' retention and locant correctness
# ---------------------------------------------------------------------------

class TestMultipliedSuffixes:
    """Tests for IUPAC elision rule with multiplied suffixes (P-31.1.2).

    When the suffix has a multiplier (di, tri, etc.), the rendered suffix
    starts with a consonant (e.g. 'diol', 'dione', 'diamine', 'triol').
    Per IUPAC rules, the parent stem retains its terminal 'e' in this case.

    Single suffixes that start with a vowel (ol, one, al, amine) elide the
    terminal 'e' as usual.
    """

    def test_ethane_1_2_diol(self):
        """OCCO: ethane-1,2-diol (two OH groups → retain 'e', both locants)."""
        result = _name("OCCO")
        assert result == "ethane-1,2-diol", f"Expected 'ethane-1,2-diol', got {result!r}"

    def test_propane_1_2_3_triol(self):
        """OCC(O)CO: propane-1,2,3-triol (three OH groups → retain 'e')."""
        result = _name("OCC(O)CO")
        assert result == "propane-1,2,3-triol", (
            f"Expected 'propane-1,2,3-triol', got {result!r}"
        )

    def test_benzene_1_2_diamine(self):
        """c1ccc(N)c(N)c1: benzene-1,2-diamine (ring diamine → retain 'e')."""
        result = _name("c1ccc(N)c(N)c1")
        assert result == "benzene-1,2-diamine", (
            f"Expected 'benzene-1,2-diamine', got {result!r}"
        )

    def test_cyclohexane_1_2_diol(self):
        """OC1CCCCC1O: cyclohexane-1,2-diol (ring diol → retain 'e')."""
        result = _name("OC1CCCCC1O")
        assert result == "cyclohexane-1,2-diol", (
            f"Expected 'cyclohexane-1,2-diol', got {result!r}"
        )

    def test_cyclohexane_1_2_diamine(self):
        """NC1CCCCC1N: cyclohexane-1,2-diamine (ring diamine → retain 'e')."""
        result = _name("NC1CCCCC1N")
        assert result == "cyclohexane-1,2-diamine", (
            f"Expected 'cyclohexane-1,2-diamine', got {result!r}"
        )

    # --- Single suffix still elides 'e' before vowel ---

    def test_ethanol_elision(self):
        """CCO: ethanol (single -ol still elides 'e')."""
        result = _name("CCO")
        assert result == "ethanol", f"Expected 'ethanol', got {result!r}"

    def test_butan_1_ol_locant(self):
        """CCCCO: butan-1-ol (single -ol with locant, no 'e' added before '-')."""
        result = _name("CCCCO")
        assert result == "butan-1-ol", f"Expected 'butan-1-ol', got {result!r}"

    def test_ethane_unchanged(self):
        """CC: ethane (no suffix, terminal 'e' supplied by terminal_vowel)."""
        result = _name("CC")
        assert result == "ethane", f"Expected 'ethane', got {result!r}"


# ---------------------------------------------------------------------------
# Phase 2b Bug 1: Ring locant omission for single substituent (P-31.1.3.4)
# ---------------------------------------------------------------------------

class TestRingLocantOmission:
    """Tests for IUPAC P-31.1.3.4: locant omitted for a single substituent
    on a fully symmetric (all-carbon monocyclic) ring.

    Benzene and cycloalkanes have all positions equivalent when there is only
    one substituent and no FG suffix.  The locant '1' must be omitted.
    """

    def test_chlorobenzene_no_locant(self):
        """c1ccc(Cl)cc1 → 'chlorobenzene', not '1-chlorobenzene'."""
        result = _name("c1ccc(Cl)cc1")
        assert result == "chlorobenzene", (
            f"Expected 'chlorobenzene' (no locant), got {result!r}"
        )

    def test_methylcyclohexane_no_locant(self):
        """CC1CCCCC1 → 'methylcyclohexane', not '1-methylcyclohexane'."""
        result = _name("CC1CCCCC1")
        assert result == "methylcyclohexane", (
            f"Expected 'methylcyclohexane' (no locant), got {result!r}"
        )

    def test_methylcyclopentane_no_locant(self):
        """CC1CCCC1 → 'methylcyclopentane', not '1-methylcyclopentane'."""
        result = _name("CC1CCCC1")
        assert result == "methylcyclopentane", (
            f"Expected 'methylcyclopentane' (no locant), got {result!r}"
        )

    def test_multiple_substituents_keep_locants(self):
        """Two substituents: locants must be kept to distinguish positions."""
        result = _name("CC1CCCCC1C")
        # 1,2-dimethylcyclohexane — locants required
        assert "1" in result and "2" in result, (
            f"Expected locants for disubstituted ring, got {result!r}"
        )

    def test_ring_with_suffix_keeps_locant(self):
        """Ring with FG suffix (cyclohexan-1-ol): prefix locant NOT affected here."""
        result = _name("OC1CCCCC1")
        # cyclohexan-1-ol: suffix locant is kept (it indicates position of OH)
        assert "cyclohexan" in result and "ol" in result, (
            f"Expected cyclohexanol name, got {result!r}"
        )

    def test_pyridine_with_one_substituent_keeps_locant(self):
        """Methylpyridine: locant kept because pyridine is not all-carbon."""
        result = _name("Cc1ccccn1")
        assert result == "2-methylpyridine", (
            f"Expected '2-methylpyridine' (locant kept for heterocycle), got {result!r}"
        )


# ---------------------------------------------------------------------------
# Phase 2b Bug 2: Heterocycle numbering (P-31.1.2.2)
# ---------------------------------------------------------------------------

class TestHeterocycleNumbering:
    """Tests for IUPAC P-31.1.2.2: heteroatoms get the lowest locant set.

    In monocyclic heterocyclic rings, N, O, S etc. atoms must be numbered
    before substituents.  This fixes the pyrazin-1-amine → pyrazin-2-amine
    regression.
    """

    def test_pyrazin_2_amine(self):
        """Nc1cnccn1 → 'pyrazin-2-amine' (N ring atoms at 1,4; C-amino at 2)."""
        result = _name("Nc1cnccn1")
        assert result == "pyrazin-2-amine", (
            f"Expected 'pyrazin-2-amine', got {result!r}"
        )

    def test_2_methylpyrazine(self):
        """Cc1cnccn1 → '2-methylpyrazine' (N at 1,4; methyl at 2)."""
        result = _name("Cc1cnccn1")
        assert result == "2-methylpyrazine", (
            f"Expected '2-methylpyrazine', got {result!r}"
        )

    def test_2_methylpyridine(self):
        """Cc1ccccn1 → '2-methylpyridine' (N at 1; methyl at adjacent C)."""
        result = _name("Cc1ccccn1")
        assert result == "2-methylpyridine", (
            f"Expected '2-methylpyridine', got {result!r}"
        )

    def test_4_methylpyridine(self):
        """Cc1ccncc1 → '4-methylpyridine' (methyl para to N)."""
        result = _name("Cc1ccncc1")
        assert result == "4-methylpyridine", (
            f"Expected '4-methylpyridine', got {result!r}"
        )


# ---------------------------------------------------------------------------
# Phase 2b Bug 3: Isopropyl named as propyl (substituent method)
# ---------------------------------------------------------------------------

class TestSubstituentMethod:
    """Tests for P-29.2: substituent method choice (ALKYL vs ALKANYL).

    When the free valence (attachment point) of a substituent is NOT at a
    terminal carbon, Method 2 (ALKANYL) must be used: "propan-2-yl" not
    "propyl" for isopropyl groups.
    """

    def test_isopropylbenzene_systematic(self):
        """CC(C)c1ccccc1 → '(propan-2-yl)benzene' (isopropyl at C2)."""
        result = _name("CC(C)c1ccccc1")
        assert "propan-2-yl" in result, (
            f"Expected 'propan-2-yl' in isopropylbenzene name, got {result!r}"
        )
        assert "propylbenzene" not in result, (
            f"Got 'propylbenzene' (propan-1-yl, wrong): {result!r}"
        )

    def test_isopropylcyclohexane_systematic(self):
        """C1CCC(CC1)C(C)C → '(propan-2-yl)cyclohexane'."""
        result = _name("C1CCC(CC1)C(C)C")
        assert "propan-2-yl" in result, (
            f"Expected 'propan-2-yl' in isopropylcyclohexane name, got {result!r}"
        )

    def test_propylbenzene_normal(self):
        """CCCc1ccccc1 → 'propylbenzene' (n-propyl, attachment at C1)."""
        result = _name("CCCc1ccccc1")
        # n-propyl: attachment at C1 (terminal) → should be "propyl" (Method 1)
        assert "propyl" in result, (
            f"Expected 'propyl' in propylbenzene name, got {result!r}"
        )
        # Must NOT contain the ALKANYL form for n-propyl
        assert "propan-1-yl" not in result, (
            f"Got 'propan-1-yl' for n-propyl (unnecessary locant): {result!r}"
        )

    def test_ethylbenzene_method1(self):
        """CCc1ccccc1 → 'ethylbenzene' (ethyl, attachment at C1, Method 1)."""
        result = _name("CCc1ccccc1")
        # ethyl is terminal → Method 1 → "ethyl" not "ethan-1-yl"
        assert "ethyl" in result, (
            f"Expected 'ethyl' in ethylbenzene name, got {result!r}"
        )
        assert "ethan-1-yl" not in result, (
            f"Got 'ethan-1-yl' for ethyl (unnecessary locant): {result!r}"
        )


# ---------------------------------------------------------------------------
# N-substituent flood-fill with ring-attached FGs (atom-drop regression tests)
# ---------------------------------------------------------------------------

class TestNSubstituentFloodFillRingFG:
    """Tests for the Pass 2.5 N-substituent flood-fill straddling check.

    When an N-substituent is a ring that carries its own FG (hydroxy, amino,
    halo, etc.), the flood-fill must include the ring AND the ring-attached FG
    atoms instead of rejecting the whole component.  Previously the code used
    a simple ``component & blocker_atoms`` check that incorrectly rejected
    components containing any non-PCG FG atoms — even FGs fully enclosed within
    the component (like a phenol -OH on the aryl ring).  The fix uses a
    straddling check: only reject if the FG has atoms BOTH inside AND outside
    the component.
    """

    def test_paracetamol_no_atom_drop(self):
        """CC(=O)Nc1ccc(O)cc1 (paracetamol) must not produce an atom-drop error."""
        result = _name("CC(=O)Nc1ccc(O)cc1")
        assert "NAMING ERROR" not in result, (
            f"Atom-drop error naming paracetamol: {result!r}"
        )

    def test_paracetamol_contains_ethanamide(self):
        """Paracetamol name should contain 'ethanamide' (the acyl parent)."""
        result = _name("CC(=O)Nc1ccc(O)cc1")
        assert "ethanamide" in result or "acetamide" in result, (
            f"Expected 'ethanamide' or 'acetamide' in paracetamol name, got {result!r}"
        )

    def test_paracetamol_contains_hydroxyphenyl(self):
        """Paracetamol name should contain 'hydroxyphenyl'."""
        result = _name("CC(=O)Nc1ccc(O)cc1")
        assert "hydroxyphenyl" in result, (
            f"Expected 'hydroxyphenyl' in paracetamol name, got {result!r}"
        )

    def test_n_aminophenyl_amide_no_atom_drop(self):
        """CC(=O)Nc1ccc(N)cc1 (amino on ring) must not produce an atom-drop error."""
        result = _name("CC(=O)Nc1ccc(N)cc1")
        assert "NAMING ERROR" not in result, (
            f"Atom-drop error: {result!r}"
        )

    def test_n_chlorophenyl_amide_no_atom_drop(self):
        """CC(=O)Nc1ccc(Cl)cc1 (chloro on ring) must not produce an atom-drop error."""
        result = _name("CC(=O)Nc1ccc(Cl)cc1")
        assert "NAMING ERROR" not in result, (
            f"Atom-drop error: {result!r}"
        )

    def test_n_methylphenyl_amide_still_works(self):
        """CC(=O)Nc1ccc(C)cc1 must still produce a non-error name (regression)."""
        result = _name("CC(=O)Nc1ccc(C)cc1")
        assert "NAMING ERROR" not in result, (
            f"Regression: methylphenyl amide broke: {result!r}"
        )
        assert "phenyl" in result, (
            f"Expected 'phenyl' in N-(methylphenyl)ethanamide name, got {result!r}"
        )


# ---------------------------------------------------------------------------
# Ring substituent locants (P-31.1.2.4 / P-31.1.3.4 interaction)
# ---------------------------------------------------------------------------

class TestRingSubstituentLocants:
    """Ring substituents used as radicals must emit locants for their own substituents.

    P-31.1.3.4 omits locants when a STANDALONE ring has a single substituent
    (e.g. chlorobenzene).  That rule must NOT apply when the ring is itself
    being named as a substituent of another parent (the attachment atom takes
    locant 1, so other substituents need their non-1 locants cited explicitly).
    """

    def test_4_hydroxyphenyl_locant_present(self):
        """CC(=O)Nc1ccc(O)cc1 → N-(4-hydroxyphenyl)ethanamide (locant '4-' required)."""
        result = _name("CC(=O)Nc1ccc(O)cc1")
        assert "4-hydroxy" in result, (
            f"Expected '4-hydroxy' in name, got {result!r}"
        )

    def test_standalone_chlorobenzene_no_locant(self):
        """Clc1ccccc1 → chlorobenzene (no locant — P-31.1.3.4 applies in standalone)."""
        result = _name("Clc1ccccc1")
        assert result == "chlorobenzene", (
            f"Expected 'chlorobenzene', got {result!r}"
        )

    def test_standalone_toluene_no_locant(self):
        """Cc1ccccc1 → toluene (retained name, no locant)."""
        result = _name("Cc1ccccc1")
        assert "toluene" in result or result == "methylbenzene", (
            f"Expected toluene or methylbenzene, got {result!r}"
        )

    def test_phenyl_no_substituents_no_locant(self):
        """Unsubstituted phenyl fragment should produce 'phenyl' (no locant)."""
        result = _name("c1ccccc1")
        # Standalone benzene — no locant needed
        assert "benzene" in result or result == "benzene", (
            f"Expected 'benzene', got {result!r}"
        )

    def test_4_aminophenyl_locant_present(self):
        """CCOC(=O)c1ccc(N)cc1 → ethyl 1-(4-aminophenyl)methanoate (locant '4-' required)."""
        result = _name("CCOC(=O)c1ccc(N)cc1")
        assert "4-amino" in result, (
            f"Expected '4-amino' in name, got {result!r}"
        )

    def test_4_hydroxyphenyl_ester(self):
        """CCOC(=O)c1ccc(O)cc1 → contains '4-hydroxy' (locant required)."""
        result = _name("CCOC(=O)c1ccc(O)cc1")
        assert "4-hydroxy" in result, (
            f"Expected '4-hydroxy' in name, got {result!r}"
        )
