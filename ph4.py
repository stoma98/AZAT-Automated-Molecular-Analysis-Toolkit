# ph4.py
from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from collections import namedtuple

PharmacophoreFeature = namedtuple(
    "PharmacophoreFeature",
    ["feature_type", "points", "radius", "color"]
)

# Build a feature factory once (RDKit standard definitions)
_FDEF_PATH = RDConfig.RDDataDir + "/BaseFeatures.fdef"
_FACTORY = ChemicalFeatures.BuildFeatureFactory(_FDEF_PATH)

# Map RDKit feature families -> AZAT feature type, radius, color
# (Keep your existing labels HBD/HBA/Aromatic/Hydrophobic/Charged)
_FAMILY_MAP = {
    "Donor": ("HBD", 1.0, "blue"),
    "Acceptor": ("HBA", 1.0, "red"),
    "Aromatic": ("Aromatic", 1.2, "yellow"),
    "Hydrophobe": ("Hydrophobic", 1.5, "green"),
    "PosIonizable": ("Charged", 1.0, "orange"),
    "NegIonizable": ("Charged", 1.0, "purple"),
}

def compute_features(mol: Chem.Mol):
    """
    Pharmacophore features based on RDKit ChemicalFeatures (BaseFeatures.fdef):
      - HBD/HBA via Donor/Acceptor families (chemically correct donor/acceptor rules)
      - Hydrophobic via Hydrophobe family
      - Aromatic via Aromatic family
      - Charged via PosIonizable/NegIonizable families

    Requires at least one conformer (3D or 2D coords). If none -> [].
    """
    features = []
    if mol is None or mol.GetNumConformers() == 0:
        return features

    # RDKit chemical features use the current conformer coordinates
    try:
        feats = _FACTORY.GetFeaturesForMol(mol)
    except Exception:
        return features

    conf = mol.GetConformer()

    # De-duplicate points (same family often returns multiple features, but can coincide)
    seen = set()

    for f in feats:
        fam = f.GetFamily()
        if fam not in _FAMILY_MAP:
            continue

        feat_type, radius, color = _FAMILY_MAP[fam]

        # f.GetPos() is Point3D in conformer coordinates
        try:
            pos = f.GetPos()
        except Exception:
            # fallback: center of involved atoms
            try:
                atom_ids = list(f.GetAtomIds())
                pts = [conf.GetAtomPosition(i) for i in atom_ids]
                x = sum(p.x for p in pts) / len(pts)
                y = sum(p.y for p in pts) / len(pts)
                z = sum(p.z for p in pts) / len(pts)
                pos = Chem.rdGeometry.Point3D(x, y, z)
            except Exception:
                continue

        key = (feat_type, round(pos.x, 3), round(pos.y, 3), round(pos.z, 3), color)
        if key in seen:
            continue
        seen.add(key)

        features.append(PharmacophoreFeature(feat_type, [pos], radius, color))

    return features


def features_to_js_shapes(features):
    if not features:
        return ""
    js_code = []
    for feat in features:
        for point in feat.points:
            js_code.append(f"""
                viewer.addSphere({{
                    center: {{x: {point.x}, y: {point.y}, z: {point.z}}},
                    radius: {feat.radius},
                    color: "{feat.color}",
                    alpha: 0.3
                }});
            """)
    return "\n".join(js_code)
