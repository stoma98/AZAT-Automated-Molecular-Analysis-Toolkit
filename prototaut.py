# prototaut.py
from __future__ import annotations

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


# ----------------------------
# Helpers (robust parsing)
# ----------------------------
def _mol_from_smiles_robust(smiles: str) -> Chem.Mol | None:
    """
    Robust SMILES parsing for diverse ligands.
    - Try normal sanitization first (needed for aromaticity/tautomers).
    - If that fails (often for metal-containing or exotic valence), try sanitize=False.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None

    m = Chem.MolFromSmiles(smiles, sanitize=True)
    if m is not None:
        return m

    # Fallback: allow parsing without sanitization (keeps connectivity)
    m = Chem.MolFromSmiles(smiles, sanitize=False)
    return m


def _canonical_smiles(mol: Chem.Mol) -> str:
    """
    Canonical isomeric SMILES.
    If sanitization is incomplete, RDKit still usually can write SMILES.
    """
    try:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        # As a last resort: try to sanitize minimally then output
        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            return Chem.MolToSmiles(mol, isomericSmiles=True)
        except Exception:
            # If absolutely impossible, return original-like string
            return Chem.MolToSmiles(Chem.MolFromSmiles(""), isomericSmiles=True)  # empty molecule SMILES ""


def _contains_metal(mol: Chem.Mol) -> bool:
    """
    Detect metal atoms (including alkali/alkaline earth/transition/lanthanides/actinides).
    For such molecules, standard tautomer enumeration is generally not chemically meaningful
    in a general ligand-prep tool (coordination chemistry, variable valence/bond orders).
    """
    # Common non-metals we DO NOT consider as metals, including halogens
    non_metals = {
        1, 2, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 34, 35, 36, 53, 54, 85, 86
    }
    # Boron(5) included as non-metal; Silicon(14), Phosphorus(15), Sulfur(16), etc.

    for a in mol.GetAtoms():
        z = a.GetAtomicNum()
        if z == 0:
            continue
        if z not in non_metals:
            # This will treat Na, K, Mg, Ca, Fe, Zn, Pt, etc. as metals
            return True
    return False


# ----------------------------
# API
# ----------------------------
def protonate(smiles: str, method: str = "rdkit", pH: float = 7.4) -> str | None:
    """
    Charge normalization (neutralization where RDKit Uncharger can do it).
    NOTE: This is NOT a pH-dependent microstate model; pH parameter is kept for API compatibility.
    For metal-containing molecules we do NOT apply Uncharger by default.
    """
    mol = _mol_from_smiles_robust(smiles)
    if mol is None:
        return None

    # If metal present: avoid Uncharger side effects on coordination/charges
    if _contains_metal(mol):
        return _canonical_smiles(mol)

    try:
        uncharger = rdMolStandardize.Uncharger()
        mol2 = uncharger.uncharge(mol)
        return _canonical_smiles(mol2)
    except Exception:
        return _canonical_smiles(mol)


def enumerate_tautomers(
    smiles: str,
    max_tautomers: int = 20,
    preserve_amide: bool = True,
    preserve_aromatic: bool = True,
    preserve_sulfonyl: bool = True,
    preserve_phosphoryl: bool = True,
) -> list[str]:
    """
    RDKit tautomer enumeration + strict filtering of chemically rigid motifs.

    Why these constraints (chemically justified, not "heuristics"):
      - Aromatic systems should not be dearomatized in a general ligand-prep workflow.
      - Amide N–C(=O) should be preserved (amide tautomerism is generally irrelevant in drug-like context).
      - Sulfonyl S(=O)2 and phosphoryl P(=O) motifs should be preserved (avoid unrealistic proton shifts/bond order changes).
      - For metal-containing molecules: do not enumerate tautomers (coordination chemistry not covered by this model).
    """
    base_mol = _mol_from_smiles_robust(smiles)
    if base_mol is None:
        return []

    # Metals: return just one canonical form (no tautomer enumeration)
    if _contains_metal(base_mol):
        return [_canonical_smiles(base_mol)]

    # Ensure aromaticity perception works; if molecule was parsed unsanitized, try sanitize
    try:
        Chem.SanitizeMol(base_mol)
    except Exception:
        # If sanitization fails but it's still organic-ish, we can still return the canonical base form
        return [_canonical_smiles(base_mol)]

    base_arom = sum(1 for a in base_mol.GetAtoms() if a.GetIsAromatic())

    enumerator = rdMolStandardize.TautomerEnumerator()
    enumerator.SetMaxTautomers(int(max_tautomers))

    try:
        taut_mols = enumerator.Enumerate(base_mol)
    except Exception:
        return [_canonical_smiles(base_mol)]

    # Unique by SMILES
    unique = []
    seen = set()
    for t in taut_mols:
        try:
            smi = Chem.MolToSmiles(t, isomericSmiles=True)
        except Exception:
            continue
        if smi not in seen:
            seen.add(smi)
            unique.append(t)

    # Always keep at least base form if something goes wrong
    if not unique:
        return [_canonical_smiles(base_mol)]

    # 1) Preserve aromaticity (no decrease of aromatic atom count)
    if preserve_aromatic:
        filtered = []
        for m in unique:
            arom_count = sum(1 for a in m.GetAtoms() if a.GetIsAromatic())
            if arom_count >= base_arom:
                filtered.append(m)
        if filtered:
            unique = filtered

    # 2) Preserve amide motif N-C(=O) (covers primary/secondary/tertiary amides)
    if preserve_amide:
        amide = Chem.MolFromSmarts("[NX3][CX3](=O)")
        if amide is not None and base_mol.HasSubstructMatch(amide):
            filtered = [m for m in unique if m.HasSubstructMatch(amide)]
            if filtered:
                unique = filtered

    # 3) Preserve sulfonyl motif S(=O)(=O) (covers sulfonamides, sulfones, sulfonates, sulfates)
    if preserve_sulfonyl:
        sulfonyl = Chem.MolFromSmarts("[SX4](=O)(=O)")
        if sulfonyl is not None and base_mol.HasSubstructMatch(sulfonyl):
            filtered = [m for m in unique if m.HasSubstructMatch(sulfonyl)]
            if filtered:
                unique = filtered

    # 4) Preserve phosphoryl motif P(=O) (phosphates/phosphonates/phosphoramidates)
    if preserve_phosphoryl:
        phosphoryl = Chem.MolFromSmarts("[PX4](=O)")
        if phosphoryl is not None and base_mol.HasSubstructMatch(phosphoryl):
            filtered = [m for m in unique if m.HasSubstructMatch(phosphoryl)]
            if filtered:
                unique = filtered

    # Return SMILES list
    out = []
    seen2 = set()
    for m in unique:
        smi = Chem.MolToSmiles(m, isomericSmiles=True)
        if smi not in seen2:
            seen2.add(smi)
            out.append(smi)

    # Safety: never return empty
    if not out:
        out = [_canonical_smiles(base_mol)]
    return out
