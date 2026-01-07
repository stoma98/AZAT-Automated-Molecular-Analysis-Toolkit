# AZAT: Automated Molecular Analysis Toolkit

**Interactive desktop platform for comprehensive ligand preparation, 3D structure generation, and conformational torsion analysis. Built with RDKit and PySide6.**

AZAT is a standalone desktop application designed for researchers working with small-molecule ligands. It unifies essential pre-processing steps — from SMILES input to 3D conformational ensembles — into a single, interactive workflow.

##  Key Features
- **3D Structure Generation**: Reliable conversion of SMILES to 3D using ETKDGv3 with automated UFF/MMFF94 optimization.
- **Smart Tautomer Enumeration**: Generates tautomers with "motif-preserving" filters to protect chemically rigid groups like amides, sulfonyls, and phosphoryls.
- **Interactive Torsion Scanning**: Manual and automated dihedral scans with real-time energy profiling and synchronized 3D visualization.
- **Property Profiling**: Instant generation of medicinal chemistry reports (Lipinski’s rules, LogP, TPSA, PAINS/Brenk structural alerts).
- **Geometry Validation**: Integrated sanity checks for bond lengths, valence angles, and aromatic ring planarity.
- **Ensemble Building**: Generation of low-energy conformer sets with RMSD-based deduplication.
  
## Tech Stack
- **Language**: Python 3.10+
- **Cheminformatics**: RDKit, OpenBabel
- **GUI**: PySide6 (Qt)
- **3D Graphics**: 3Dmol.js 
- **Plotting**: Matplotlib

## Installation
To ensure the AZAT platform runs correctly with all its features (cheminformatics, 3D visualization, and UI), follow these steps:

Environment Setup 
It is highly recommended to use Conda to manage the specific versions of RDKit and OpenBabel required by the project.

### 1. Create the environment
```bash
conda env create -f environment.yml
```
### 2. Activate the environment
```bash
conda activate azat
```
## Required File Structure
 For the application to launch and function properly, all module files and the application icon must be placed in the same working directory. Your project folder should look like this:
```
your-project-folder/
├── azat.py                 # Main application script
├── medchem.py              # Medicinal chemistry module
├── ph4.py                  # Pharmacophore module
├── prototaut.py            # Protonation & Tautomer module
├── smiles_to_3d.py         # 3D generation module
├── stategen.py             # Conformer ensemble module
├── validate.py             # Geometry validation module
├── icon.ico                # Application icon (Required)
└── environment.yml         # Conda configuration
```
## Run

Run the main script to start the toolkit:

```bash
python azat.py
```
## Authors

- **Mariia Y. Stoliarskaia** — Institute of Protein Research (RAS)
- **Oleg S. Nikonov** — Institute of Protein Research (RAS)

## License
This project is licensed under the MIT License.
