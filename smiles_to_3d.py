# smiles_to_3d.py
from __future__ import annotations
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom, rdDepictor
import numpy as np
from collections import namedtuple
from dataclasses import dataclass

from prototaut import protonate, enumerate_tautomers
from stategen import generate_microstates # ВАЖНО: вернули импорт

BuildResult = namedtuple("BuildResult", ["sdf", "states_smi", "states_sdf"])

@dataclass
class BuildOptions:
    engine: str = "ETKDGv3"
    ff: str = "UFF"
    num_confs: int = 20
    seed: int = 42
    prot_method: str = "rdkit"
    ph: float = 7.4
    gen_tautomers: bool = True
    gen_microstates: bool = False
    micro_confs: int = 10
    micro_topk: int = 5
    micro_rmsd: float = 0.5
    tautomer_mode: str = "Balanced"

class BuildError(Exception):
    pass

def _check_geometry_sanity(mol: Chem.Mol, conf_id: int) -> bool:
    conf = mol.GetConformer(conf_id)
    pts = conf.GetPositions()
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if np.linalg.norm(pts[i] - pts[j]) < 0.75: return False
    return True

def _safe_set_params(p, opts):
    p.randomSeed = opts.seed
    p.enforceChirality = True
    for attr in ["useExpTorsions", "useBasicKnowledge", "ignoreSmoothingFailures"]:
        if hasattr(p, attr):
            try: setattr(p, attr, True)
            except: pass

def smiles_to_3d_sdf(smiles: str, opts: BuildOptions) -> BuildResult:
    # 1. Химическая стабилизация
    p_smi = protonate(smiles, opts.prot_method, opts.ph)
    tauts = enumerate_tautomers(p_smi, max_tautomers=5) if opts.gen_tautomers else [p_smi]
    
    states_sdf = []
    
    # 2. Построение 3D для каждого таутомера
    for s in tauts:
        m = Chem.MolFromSmiles(s)
        if not m: continue
        m = Chem.AddHs(m)
        params = rdDistGeom.ETKDGv3() if hasattr(rdDistGeom, "ETKDGv3") else rdDistGeom.ETKDG()
        _safe_set_params(params, opts)
        
        cids = rdDistGeom.EmbedMultipleConfs(m, numConfs=opts.num_confs, params=params)
        if not list(cids):
            params.useRandomCoords = True
            cids = rdDistGeom.EmbedMultipleConfs(m, numConfs=opts.num_confs, params=params)
        
        best_id, min_e = -1, float('inf')
        for cid in cids:
            if not _check_geometry_sanity(m, cid): continue
            try:
                AllChem.UFFOptimizeMolecule(m, confId=cid, maxIters=500)
                e = AllChem.UFFGetMoleculeForceField(m, confId=cid).CalcEnergy()
                if e < min_e: min_e = e; best_id = cid
            except: continue
            
        if best_id != -1:
            final_m = Chem.Mol(m)
            for i in range(final_m.GetNumConformers()-1, -1, -1):
                if i != best_id: final_m.RemoveConformer(i)
            states_sdf.append(Chem.MolToMolBlock(final_m))
        else:
            # Если 3D не вышло - даем 2D заглушку
            m2 = Chem.Mol(m); rdDepictor.Compute2DCoords(m2)
            states_sdf.append(Chem.MolToMolBlock(m2))

    # 3. ГЕНЕРАЦИЯ АНСАМБЛЯ (Sampling) - Чтобы не было 0
    if opts.gen_microstates and states_sdf:
        try:
            # Берем основной таутомер (первый) и генерируем для него набор поз
            extra_confs = generate_microstates(
                states_sdf[0], 
                num_confs=opts.micro_confs, 
                top_k=opts.micro_topk, 
                rmsd_threshold=opts.micro_rmsd
            )
            states_sdf.extend(extra_confs)
        except Exception as e:
            print(f"Sampling error: {e}")

    return BuildResult(states_sdf[0], tauts, states_sdf)