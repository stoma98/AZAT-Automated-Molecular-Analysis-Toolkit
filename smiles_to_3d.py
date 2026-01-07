# smiles_to_3d.py 
from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom, rdDepictor
from collections import namedtuple
from dataclasses import dataclass

from prototaut import protonate, enumerate_tautomers
from stategen import generate_microstates

BuildResult = namedtuple("BuildResult", ["sdf", "states_smi", "states_sdf"])


@dataclass
class BuildOptions:
    engine: str = "ETKDGv3"
    ff: str = "UFF"  # "MMFF" allowed, will auto-fallback to UFF if not applicable
    num_confs: int = 20
    seed: int = 42
    prot_method: str = "rdkit"
    ph: float = 7.4
    gen_tautomers: bool = True
    gen_microstates: bool = False
    micro_confs: int = 5
    micro_topk: int = 3
    micro_rmsd: float = 0.7
    tautomer_mode: str = "Balanced"


class BuildError(Exception):
    pass


# "Металл" в смысле ветвления пайплайна (таутомеризация/FF могут быть неприменимы)
_NON_METALS = {
    1, 2, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18,
    34, 35, 36, 53, 54, 85, 86
}


def _contains_metal(mol: Chem.Mol) -> bool:
    for a in mol.GetAtoms():
        z = a.GetAtomicNum()
        if z != 0 and z not in _NON_METALS:
            return True
    return False


def _mol_from_smiles_robust(smiles: str) -> Chem.Mol | None:
    """Сначала нормальный sanitize=True, если не получилось — sanitize=False ()."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    m = Chem.MolFromSmiles(smiles, sanitize=True)
    if m is not None:
        return m
    return Chem.MolFromSmiles(smiles, sanitize=False)


def _molblock_2d(mol: Chem.Mol) -> str:
    """Гарантированный 2D MolBlock (fallback, чтобы UI не падал)."""
    m = Chem.Mol(mol)
    try:
        rdDepictor.Compute2DCoords(m)
    except Exception:
        pass
    return Chem.MolToMolBlock(m)


def _aromatize_mol(mol: Chem.Mol) -> Chem.Mol:
    """Проставляет ароматичность, не меняя connectivity."""
    try:
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)
    except Exception:
        pass
    return mol


def _choose_embed_params(opts: BuildOptions):
    if opts.engine == "ETKDGv2":
        params = rdDistGeom.ETKDGv2()
    else:
        params = rdDistGeom.ETKDGv3()
    params.randomSeed = int(opts.seed)
    params.useRandomCoords = False
    return params


def _mmff_applicable(mol: Chem.Mol) -> bool:
    try:
        props = AllChem.MMFFGetMoleculeProperties(mol)
        return props is not None
    except Exception:
        return False


def _optimize_molecule(mol: Chem.Mol, ff_type: str):
    """Оптимизация всех конформеров; MMFF -> fallback UFF если надо. Никогда не бросает исключение наружу."""
    ff = (ff_type or "UFF").upper()

    for conf_id in range(mol.GetNumConformers()):
        try:
            if ff == "MMFF" and _mmff_applicable(mol):
                AllChem.MMFFOptimizeMolecule(mol, confId=conf_id, maxIters=200)
            else:
                AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=200)
        except Exception:
            # не падаем на плохих случаях
            pass

def _find_best_conformer(mol: Chem.Mol, ff_type: str) -> int:
    if mol.GetNumConformers() <= 1:
        return 0

    ff = (ff_type or "UFF").upper()
    energies = []

    for conf_id in range(mol.GetNumConformers()):
        e = float("inf")
        try:
            if ff == "MMFF" and _mmff_applicable(mol):
                props = AllChem.MMFFGetMoleculeProperties(mol)
                f = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id) if props else None
                e = f.CalcEnergy() if f is not None else float("inf")
            else:
                f = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
                e = f.CalcEnergy() if f is not None else float("inf")
        except Exception:
            e = float("inf")
        
        # Сохраняем энергию вместе с ID для стабильной сортировки
        energies.append((e, conf_id))

    # Сортируем: сначала по энергии (основной критерий), 
    # затем по ID (как "тай-брейкер" для 100% повторяемости)
    energies.sort()
    
    return energies[0][1]
    
# def _find_best_conformer(mol: Chem.Mol, ff_type: str) -> int:
#     if mol.GetNumConformers() <= 1:
#         return 0

#     ff = (ff_type or "UFF").upper()
#     energies = []

#     for conf_id in range(mol.GetNumConformers()):
#         e = float("inf")
#         try:
#             if ff == "MMFF" and _mmff_applicable(mol):
#                 props = AllChem.MMFFGetMoleculeProperties(mol)
#                 f = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id) if props else None
#                 e = f.CalcEnergy() if f is not None else float("inf")
#             else:
#                 f = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
#                 e = f.CalcEnergy() if f is not None else float("inf")
#         except Exception:
#             e = float("inf")

#         energies.append(e)

#     return min(range(len(energies)), key=energies.__getitem__)


def _enumerate_tautomers_safe(protonated_smiles: str, max_tautomers: int, preserve_amide: bool, preserve_aromatic: bool):
    """

    """
    try:
        return enumerate_tautomers(
            protonated_smiles,
            max_tautomers=max_tautomers,
            preserve_amide=preserve_amide,
            preserve_aromatic=preserve_aromatic,
            preserve_sulfonyl=True,
            preserve_phosphoryl=True,
        )
    except TypeError:
        # старая версия prototaut.enumerate_tautomers без новых аргументов
        return enumerate_tautomers(
            protonated_smiles,
            max_tautomers=max_tautomers,
            preserve_amide=preserve_amide,
            preserve_aromatic=preserve_aromatic,
        )
    except Exception:
        return []


def smiles_to_3d_sdf(smiles: str, opts: BuildOptions) -> BuildResult:
    """
    Устойчивый конвейер:
    - старается дать 3D,
    - если не получилось — отдаёт 2D MolBlock,
    """
    # 1) Нормализация (standardize/uncharge)
    protonated_smiles = protonate(smiles, opts.prot_method, opts.ph)
    if protonated_smiles is None:
        # абсолютный fallback: пробуем исходный smiles как 2D
        m0 = _mol_from_smiles_robust(smiles)
        if m0 is None:
            raise BuildError("Invalid SMILES")
        mb = _molblock_2d(m0)
        return BuildResult(mb, [smiles], [mb])

    # 2) ТАУТОМЕРЫ ()
    if opts.gen_tautomers:
        mode = (opts.tautomer_mode or "Balanced").lower()
        max_tautomers = opts.num_confs

        if mode.startswith("conservative"):
            max_tautomers = max(1, min(10, opts.num_confs))
        elif mode.startswith("aggressive"):
            max_tautomers = max(opts.num_confs, 30)

        # Всегда True: чтобы не ломать кольца/амиды/сульфонилы/фосфорилы
        tautomers = _enumerate_tautomers_safe(
            protonated_smiles,
            max_tautomers=max_tautomers,
            preserve_amide=True,
            preserve_aromatic=True,
        )
        if not tautomers:
            tautomers = [protonated_smiles]
    else:
        tautomers = [protonated_smiles]

    states_smi = tautomers
    states_sdf: list[str] = []

    # 3) 3D generation per tautomer + fallback to 2D
    for taut_smiles in tautomers:
        mol = _mol_from_smiles_robust(taut_smiles)
        if mol is None:
            continue

        # 
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            states_sdf.append(_molblock_2d(mol))
            continue

        has_metal = _contains_metal(mol)

        # 
        try:
            molH = Chem.AddHs(mol)
        except Exception:
            molH = Chem.Mol(mol)

        # Параметры DG
        params = _choose_embed_params(opts)

        # 
        embedded = False
        for attempt in range(3):
            try:
                if attempt == 1:
                    params.useRandomCoords = True
                if attempt == 2:
                    params.useRandomCoords = True
                    params.pruneRmsThresh = -1  # иногда помогает

                # для металлов уменьшаем число конформеров, чтобы меньше было фейлов/времени
                nconf = min(opts.num_confs, 10) if has_metal else opts.num_confs

                conf_ids = list(rdDistGeom.EmbedMultipleConfs(molH, numConfs=nconf, params=params))
                if conf_ids:
                    embedded = True
                    break
            except Exception:
                embedded = False

        if not embedded:
            states_sdf.append(_molblock_2d(mol))
            continue

        # Optimize
        ff = (opts.ff or "UFF").upper()
        if has_metal and ff == "MMFF":
            # MMFF обычно не применим к координационным комплексам
            ff = "UFF"

        _optimize_molecule(molH, ff)

        # Keep best conformer
        best_conf = _find_best_conformer(molH, ff)
        if best_conf >= 0:
            for conf_id in range(molH.GetNumConformers() - 1, -1, -1):
                if conf_id != best_conf:
                    molH.RemoveConformer(conf_id)

        molH = _aromatize_mol(molH)
        states_sdf.append(Chem.MolToMolBlock(molH))

    # 
    if not states_sdf:
        m0 = _mol_from_smiles_robust(protonated_smiles) or _mol_from_smiles_robust(smiles)
        if m0 is None:
            raise BuildError("Could not build any structure")
        mb = _molblock_2d(m0)
        states_sdf = [mb]

    # 4) Microstates/ensemble — не даём падать, просто пропускаем при ошибке
    if opts.gen_microstates and states_sdf:
        try:
            microstates = generate_microstates(
                states_sdf[0],
                num_confs=opts.micro_confs,
                top_k=opts.micro_topk,
                rmsd_threshold=opts.micro_rmsd,
            )
            fixed_micro = []
            for sdf in microstates:
                m = Chem.MolFromMolBlock(sdf, removeHs=False)
                if m is None:
                    continue
                try:
                    m = Chem.AddHs(m, addCoords=True)
                except Exception:
                    pass
                try:
                    AllChem.UFFOptimizeMolecule(m, maxIters=200)
                except Exception:
                    pass
                m = _aromatize_mol(m)
                fixed_micro.append(Chem.MolToMolBlock(m))
            if fixed_micro:
                states_sdf.extend(fixed_micro)
        except Exception:
            pass

    return BuildResult(states_sdf[0], states_smi, states_sdf)
