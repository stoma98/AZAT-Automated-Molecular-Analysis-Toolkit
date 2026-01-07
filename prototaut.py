# prototaut.py
from __future__ import annotations
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

def _contains_metal(mol: Chem.Mol) -> bool:
    non_metals = {1, 2, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 34, 35, 36, 53, 54, 85, 86}
    return any(a.GetAtomicNum() != 0 and a.GetAtomicNum() not in non_metals for a in mol.GetAtoms())

def _get_critical_motifs(mol: Chem.Mol) -> list[Chem.Mol]:
    motifs = {
        "carboxyl": "[CX3](=O)[OX2H1,OX1-]",
        "sulfonyl": "[SX4](=O)(=O)",
        "phosphoryl": "[PX4](=O)",
        "amide": "[NX3][CX3](=O)"
    }
    present = []
    for smarts in motifs.values():
        pat = Chem.MolFromSmarts(smarts)
        if pat and mol.HasSubstructMatch(pat):
            present.append(pat)
    return present

def protonate(smiles: str, method: str = "rdkit", pH: float = 7.4) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if not mol or _contains_metal(mol): return smiles
    orig_double_bonds = smiles.count('=')
    try:
        uncharger = rdMolStandardize.Uncharger()
        mol_n = uncharger.uncharge(mol)
        res_smi = Chem.MolToSmiles(mol_n, isomericSmiles=True)
        if res_smi.count('=') != orig_double_bonds: return smiles
        return res_smi
    except: return smiles

def enumerate_tautomers(smiles: str, max_tautomers: int = 5) -> list[str]:
    mol = Chem.MolFromSmiles(smiles)
    if not mol or _contains_metal(mol): return [smiles]
    critical_pats = _get_critical_motifs(mol)
    orig_double_bonds = smiles.count('=')
    
    enumerator = rdMolStandardize.TautomerEnumerator()
    enumerator.SetMaxTautomers(max_tautomers)
    try:
        tauts = enumerator.Enumerate(mol)
        results = []
        for t in tauts:
            if any(not t.HasSubstructMatch(p) for p in critical_pats): continue
            tsmi = Chem.MolToSmiles(t, isomericSmiles=True)
            if tsmi.count('=') < orig_double_bonds: continue
            results.append(tsmi)
        return list(set(results)) if results else [smiles]
    except: return [smiles]