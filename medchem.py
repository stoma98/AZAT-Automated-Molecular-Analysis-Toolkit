# medchem.py
from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, QED
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams


def _contains_metal(mol: Chem.Mol) -> bool:
    non_metals = {1, 2, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 34, 35, 36, 53, 54, 85, 86}
    for a in mol.GetAtoms():
        z = a.GetAtomicNum()
        if z == 0:
            continue
        if z not in non_metals:
            return True
    return False


def _count_elements(mol: Chem.Mol) -> dict[str, int]:
    """
    Deterministic element counts (by atomic number).
    """
    counts = {"F": 0, "Cl": 0, "Br": 0, "I": 0, "metals": 0}
    for a in mol.GetAtoms():
        z = a.GetAtomicNum()
        if z == 9:
            counts["F"] += 1
        elif z == 17:
            counts["Cl"] += 1
        elif z == 35:
            counts["Br"] += 1
        elif z == 53:
            counts["I"] += 1
        else:
            # metals by same definition as in prototaut
            non_metals = {1, 2, 5, 6, 7, 8, 10, 14, 15, 16, 18, 34, 36, 54, 85, 86}
            if z != 0 and z not in non_metals and z not in (9, 17, 35, 53):
                counts["metals"] += 1
    return counts


def count_lipinski_violations(mol: Chem.Mol) -> int:
    violations = 0
    if Lipinski.NumHDonors(mol) > 5:
        violations += 1
    if Lipinski.NumHAcceptors(mol) > 10:
        violations += 1
    if Descriptors.MolWt(mol) > 500:
        violations += 1
    if Crippen.MolLogP(mol) > 5:
        violations += 1
    return violations


def count_aromatic_rings(mol: Chem.Mol) -> int:
    ring_info = mol.GetRingInfo()
    n_aromatic = 0
    for ring in ring_info.AtomRings():
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            n_aromatic += 1
    return n_aromatic


def count_charged_atoms(mol: Chem.Mol) -> int:
    return sum(1 for atom in mol.GetAtoms() if atom.GetFormalCharge() != 0)


def count_charged_clusters(mol: Chem.Mol) -> int:
    charged_idxs = [atom.GetIdx() for atom in mol.GetAtoms()
                    if atom.GetFormalCharge() != 0]
    if not charged_idxs:
        return 0
    charged_set = set(charged_idxs)
    visited = set()
    clusters = 0
    for start in charged_idxs:
        if start in visited:
            continue
        clusters += 1
        stack = [start]
        visited.add(start)
        while stack:
            i = stack.pop()
            atom = mol.GetAtomWithIdx(i)
            for nbr in atom.GetNeighbors():
                j = nbr.GetIdx()
                if j in charged_set and j not in visited:
                    visited.add(j)
                    stack.append(j)
    return clusters


def ring_size_and_special_atoms(mol: Chem.Mol) -> dict[str, int]:
    ring_info = mol.GetRingInfo()
    small_rings = 0
    hetero_small_rings = 0
    for ring in ring_info.AtomRings():
        size = len(ring)
        if size <= 4:
            small_rings += 1
            if any(mol.GetAtomWithIdx(i).GetAtomicNum() not in (1, 6) for i in ring):
                hetero_small_rings += 1
    num_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    num_bridge = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return {
        "small_rings_3_4": small_rings,
        "hetero_small_rings": hetero_small_rings,
        "spiro_atoms": num_spiro,
        "bridgehead_atoms": num_bridge,
    }


def complexity_features(mol: Chem.Mol) -> dict[str, float | int]:
    num_rings = rdMolDescriptors.CalcNumRings(mol)
    num_arom_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    num_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    num_bridge = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    num_hetero = rdMolDescriptors.CalcNumHeteroatoms(mol)
    frac_csp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    num_chiral = len(chiral_centers)
    return {
        "num_rings": num_rings,
        "num_aromatic_rings": num_arom_rings,
        "num_chiral_centers": num_chiral,
        "num_spiro_atoms": num_spiro,
        "num_bridgehead_atoms": num_bridge,
        "num_heteroatoms": num_hetero,
        "fraction_csp3": frac_csp3,
    }


def medchem_report_text(mol: Chem.Mol, lang: str = "en") -> str:
    if mol is None:
        return "Invalid molecule"

    report: list[str] = []

    elems = _count_elements(mol)
    has_metal = _contains_metal(mol)

    report.append("=== MOLECULE PROPERTIES ===")
    report.append(f"Molecular formula: {rdMolDescriptors.CalcMolFormula(mol)}")
    report.append(f"Molecular weight: {Descriptors.MolWt(mol):.2f}")
    report.append(f"Heavy atoms: {Lipinski.HeavyAtomCount(mol)}")
    report.append(f"Rotatable bonds: {Lipinski.NumRotatableBonds(mol)}")
    report.append(f"H-bond donors: {Lipinski.NumHDonors(mol)}")
    report.append(f"H-bond acceptors: {Lipinski.NumHAcceptors(mol)}")
    report.append(f"LogP: {Crippen.MolLogP(mol):.2f}")
    report.append(f"TPSA: {rdMolDescriptors.CalcTPSA(mol):.2f}")

    report.append("\n=== ELEMENT COUNTS ===")
    report.append(f"Halogens (F/Cl/Br/I): {elems['F']}/{elems['Cl']}/{elems['Br']}/{elems['I']}")
    report.append(f"Metal atoms: {elems['metals']}")

    report.append("\n=== DRUG-LIKENESS ===")
    report.append(f"Lipinski violations: {count_lipinski_violations(mol)}")

    # QED is defined for typical organic drug-like space; metal-containing structures are outside scope.
    if has_metal:
        report.append("QED drug-likeness: not applicable for metal-containing molecules")
    else:
        try:
            report.append(f"QED drug-likeness: {QED.qed(mol):.3f}")
        except Exception:
            report.append("QED drug-likeness: not computable for this structure")

    report.append("\n=== RING SYSTEM ===")
    ring_info = mol.GetRingInfo()
    report.append(f"Number of rings: {ring_info.NumRings()}")
    aromatic_ring_count = count_aromatic_rings(mol)
    aromatic_atom_count = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    report.append(f"Aromatic rings: {aromatic_ring_count}  (aromatic atoms: {aromatic_atom_count})")

    report.append("\n=== POLARITY & HETEROATOMS ===")
    num_hetero = rdMolDescriptors.CalcNumHeteroatoms(mol)
    frac_csp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    report.append(f"Heteroatoms (non C/H): {num_hetero}")
    report.append(f"Fraction Csp3: {frac_csp3:.2f}")
    charged_atom_count = count_charged_atoms(mol)
    charged_cluster_count = count_charged_clusters(mol)
    report.append(f"Atoms with formal charge: {charged_atom_count}")
    if charged_cluster_count:
        report.append(f"Charged atom clusters: {charged_cluster_count}")

    rs = ring_size_and_special_atoms(mol)
    report.append("\n=== RING SIZE & SPECIAL RING ATOMS ===")
    report.append(f"Small rings (3–4 members): {rs['small_rings_3_4']}")
    report.append(f"Heteroatom-containing small rings: {rs['hetero_small_rings']}")
    report.append(f"Spiro atoms: {rs['spiro_atoms']}")
    report.append(f"Bridgehead atoms: {rs['bridgehead_atoms']}")

    cx = complexity_features(mol)
    report.append("\n=== STRUCTURAL COMPLEXITY DESCRIPTORS ===")
    report.append(f"Chiral centers (incl. unassigned): {cx['num_chiral_centers']}")
    report.append(f"Ring count: {cx['num_rings']}")
    report.append(f"Aromatic rings: {cx['num_aromatic_rings']}")
    report.append(f"Spiro atoms: {cx['num_spiro_atoms']}")
    report.append(f"Bridgehead atoms: {cx['num_bridgehead_atoms']}")
    report.append(f"Heteroatoms: {cx['num_heteroatoms']}")
    report.append(f"Fraction Csp3: {cx['fraction_csp3']:.2f}")

    report.append("\n=== STRUCTURAL ALERTS ===")
    # PAINS/Brenk are designed for organic drug-like chemistry; keep running them, but be aware:
    # for metal-containing complexes they may be uninformative.
    try:
        params_pains = FilterCatalogParams()
        params_pains.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        catalog_pains = FilterCatalog(params_pains)
        matches_pains = catalog_pains.GetMatches(mol)
    except Exception:
        matches_pains = []

    try:
        params_brenk = FilterCatalogParams()
        params_brenk.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
        catalog_brenk = FilterCatalog(params_brenk)
        matches_brenk = catalog_brenk.GetMatches(mol)
    except Exception:
        matches_brenk = []

    if matches_pains:
        report.append("PAINS alerts:")
        for match in matches_pains:
            report.append(f"  - {match.GetDescription()}")

    if matches_brenk:
        report.append("Brenk alerts:")
        for match in matches_brenk:
            report.append(f"  - {match.GetDescription()}")

    if not (matches_pains or matches_brenk):
        report.append("No structural alerts detected")

    return "\n".join(report)
