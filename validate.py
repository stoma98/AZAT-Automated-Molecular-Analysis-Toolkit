# validate.py
from __future__ import annotations

from rdkit import Chem
from collections import namedtuple
import numpy as np

ValidationResult = namedtuple("ValidationResult", ["is_ok", "warnings"])

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


def validate(mol, bond_tol=0.1, angle_tol=10.0, planarity_tol=0.1):
    warnings = []
    if mol is None:
        return ValidationResult(False, ["Invalid molecule"])

    conf = mol.GetConformer()
    if not conf.Is3D():
        warnings.append("Molecule doesn't have 3D coordinates")
        return ValidationResult(False, warnings)

    #
    if _contains_metal(mol):
        _check_aromatic_planarity(mol, conf, planarity_tol, warnings)
        return ValidationResult(len(warnings) == 0, warnings)

    # Expected lengths by (sorted atom symbols, bond type)
    # Values are typical organic bond lengths (Å); only used for common pairs.
    BT = Chem.BondType
    expected = {
        (("C", "C"), BT.SINGLE): 1.54,
        (("C", "C"), BT.DOUBLE): 1.34,
        (("C", "C"), BT.TRIPLE): 1.20,
        (("C", "C"), BT.AROMATIC): 1.39,

        (("C", "N"), BT.SINGLE): 1.47,
        (("C", "N"), BT.DOUBLE): 1.28,
        (("C", "N"), BT.TRIPLE): 1.16,
        (("C", "N"), BT.AROMATIC): 1.34,

        (("C", "O"), BT.SINGLE): 1.43,
        (("C", "O"), BT.DOUBLE): 1.23,
        (("C", "O"), BT.AROMATIC): 1.36,

        (("C", "S"), BT.SINGLE): 1.82,
        (("C", "S"), BT.DOUBLE): 1.60,

        # P–O: allow both typical P=O and P–O single 
        (("O", "P"), BT.DOUBLE): 1.48,
        (("O", "P"), BT.SINGLE): 1.60,

        (("C", "H"), BT.SINGLE): 1.09,
        (("N", "H"), BT.SINGLE): 1.01,
        (("O", "H"), BT.SINGLE): 0.96,

        (("C", "F"), BT.SINGLE): 1.35,
        (("C", "Cl"), BT.SINGLE): 1.77,
        (("C", "Br"), BT.SINGLE): 1.94,
        (("C", "I"), BT.SINGLE): 2.14,
    }

    # --- Bond length checks () ---
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        a1 = mol.GetAtomWithIdx(i).GetSymbol()
        a2 = mol.GetAtomWithIdx(j).GetSymbol()
        pair = tuple(sorted((a1, a2)))
        btype = bond.GetBondType()

        key = (pair, btype)

        # For aromatic bonds RDKit uses BondType.AROMATIC.
        # 
        if key not in expected:
            continue

        pos_i = conf.GetAtomPosition(i)
        pos_j = conf.GetAtomPosition(j)
        length = pos_i.Distance(pos_j)
        exp = expected[key]

        if abs(length - exp) > bond_tol:
            warnings.append(
                f"Bond {i}-{j} ({a1}-{a2}, {btype}): length {length:.3f} Å, expected ~{exp:.3f} Å"
            )

    # --- Angle checks (reduced false positives) ---

    for (i, j, k) in _get_angles(mol):
        central = mol.GetAtomWithIdx(j)

        # 
        if _in_small_ring(mol, j, max_size=5):
            continue

        pos_i = conf.GetAtomPosition(i)
        pos_j = conf.GetAtomPosition(j)
        pos_k = conf.GetAtomPosition(k)

        v1 = pos_i - pos_j
        v2 = pos_k - pos_j
        ang = v1.AngleTo(v2) * 180.0 / np.pi

        # Aromatic angle check (safe)
        ai = mol.GetAtomWithIdx(i)
        ak = mol.GetAtomWithIdx(k)
        if ai.GetIsAromatic() and central.GetIsAromatic() and ak.GetIsAromatic():
            exp = 120.0
            if abs(ang - exp) > angle_tol:
                warnings.append(f"Angle {i}-{j}-{k} (aromatic): {ang:.1f}°, expected ~{exp:.1f}°")
            continue

        # Linear check for SP centers ()
        if central.GetHybridization() == Chem.HybridizationType.SP and central.GetDegree() == 2:
            exp = 180.0
            if abs(ang - exp) > angle_tol:
                warnings.append(f"Angle {i}-{j}-{k} (sp): {ang:.1f}°, expected ~{exp:.1f}°")

    # --- Aromatic planarity check () ---
    _check_aromatic_planarity(mol, conf, planarity_tol, warnings)

    return ValidationResult(len(warnings) == 0, warnings)


def _get_angles(mol):
    angles = []
    for bond in mol.GetBonds():
        j = bond.GetBeginAtomIdx()
        k = bond.GetEndAtomIdx()
        for nbr in mol.GetAtomWithIdx(j).GetNeighbors():
            i = nbr.GetIdx()
            if i != k:
                angles.append((i, j, k))
        for nbr in mol.GetAtomWithIdx(k).GetNeighbors():
            l = nbr.GetIdx()
            if l != j:
                angles.append((j, k, l))
    return angles


def _check_aromatic_planarity(mol, conf, planarity_tol, warnings):
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) >= 5:
            atoms = [mol.GetAtomWithIdx(idx) for idx in ring]
            if all(a.GetIsAromatic() for a in atoms):
                pts = [conf.GetAtomPosition(idx) for idx in ring]
                dev = _check_planarity(pts)
                if dev > planarity_tol:
                    warnings.append(f"Aromatic ring {ring}: planarity deviation {dev:.3f} Å")


def _check_planarity(points):
    if len(points) < 3:
        return 0.0
    arr = np.array([(p.x, p.y, p.z) for p in points])
    centroid = arr.mean(axis=0)
    centered = arr - centroid
    _, _, vt = np.linalg.svd(centered)
    normal = vt[2]
    dist = np.abs(np.dot(centered, normal))
    return dist.max()


def _in_small_ring(mol: Chem.Mol, atom_idx: int, max_size: int = 5) -> bool:
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        if atom_idx in ring and len(ring) <= max_size:
            return True
    return False
