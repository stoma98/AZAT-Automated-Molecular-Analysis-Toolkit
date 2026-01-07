# stategen.py
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
import numpy as np
from typing import List

def generate_microstates(sdf_data: str, num_confs: int = 5, top_k: int = 3,
                         rmsd_threshold: float = 0.7) -> List[str]:
    # Reduced num_confs and top_k, increased rmsd_threshold for faster but less exhaustive search
    mol = Chem.MolFromMolBlock(sdf_data)
    if mol is None:
        return []
    
    mol = Chem.AddHs(mol)
    
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    success = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
    if not success:
        return []
    
    for conf_id in range(mol.GetNumConformers()):
        AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=100)  # Reduced maxIters
    
    energies = []
    for conf_id in range(mol.GetNumConformers()):
        ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
        energy = ff.CalcEnergy() if ff is not None else float('inf')
        energies.append(energy)
    
    top_indices = np.argsort(energies)[:top_k]
    
    unique_conformers = []
    for idx in top_indices:
        is_unique = True
        for unique_idx in unique_conformers:
            rmsd = rdMolAlign.GetBestRMS(mol, mol, prbId=int(idx), refId=int(unique_idx))
            if rmsd < rmsd_threshold:
                is_unique = False
                break
        if is_unique:
            unique_conformers.append(idx)
    
    results = []
    for conf_id in unique_conformers:
        new_mol = Chem.Mol(mol)
        for other_conf_id in range(new_mol.GetNumConformers()-1, -1, -1):
            if other_conf_id != conf_id:
                new_mol.RemoveConformer(other_conf_id)
        results.append(Chem.MolToMolBlock(new_mol))
    
    return results