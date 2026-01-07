# AZAT.py
# -*- coding: utf-8 -*-
import sys
import json
import zipfile
import logging
import math
from pathlib import Path
from base64 import b64encode

from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtWebEngineWidgets import QWebEngineView

import warnings
warnings.filterwarnings(
    "ignore",
    message="to-Python converter for class boost::shared_ptr<class RDKit::FilterHierarchyMatcher>"
)

from rdkit import Chem
from rdkit.Chem import Draw, AllChem, rdMolTransforms
try:
    from rdkit.Chem import rdMolDraw2D, rdDepictor
    HAS_MOLDRAW = True
except ImportError:
    HAS_MOLDRAW = False
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.error')
RDLogger.DisableLog('rdApp.warning')
import rdkit
print("RDKit version:", rdkit.__version__)
from medchem import medchem_report_text
from ph4 import compute_features, features_to_js_shapes
from prototaut import protonate, enumerate_tautomers
from smiles_to_3d import smiles_to_3d_sdf, BuildOptions, BuildError
from stategen import generate_microstates
from validate import validate

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

ROOT = Path(__file__).parent
CFG = Path.home() / ".azat_config.json"
SESS = Path.home() / ".azat_sessions.json"


def rp(*parts):
    base = Path(getattr(sys, "_MEIPASS", ROOT))
    return str(base.joinpath(*parts))


def _escape_js_string(s):
    return s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\r", "\\r")


class MoleculeInfoPanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.conformers = []      # 
        self.energies = []        # 
        self.current_index = 0
        self.setup_ui()

    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        title = QtWidgets.QLabel("Molecule Information")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title)

        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)

        # Properties tab
        self.tab_properties = QtWidgets.QWidget()
        props_layout = QtWidgets.QVBoxLayout(self.tab_properties)
        self.props_text = QtWidgets.QTextEdit()
        self.props_text.setReadOnly(True)
        self.props_text.setFont(QtGui.QFont("Consolas", 9))
        props_layout.addWidget(self.props_text)
        self.tabs.addTab(self.tab_properties, "Properties")

        # Conformers tab - with list and navigation
        self.conformers = []
        self.energies = []
        self.tab_conformers = QtWidgets.QWidget()
        conf_layout = QtWidgets.QVBoxLayout(self.tab_conformers)

        self.list_conformers = QtWidgets.QListWidget()
        conf_layout.addWidget(self.list_conformers)

        nav_layout = QtWidgets.QHBoxLayout()
        self.btn_prev_conf = QtWidgets.QPushButton("\u25c0 Previous")
        self.btn_next_conf = QtWidgets.QPushButton("Next \u25b6")
        self.lbl_conf_info = QtWidgets.QLabel("State: - / -")
        self.lbl_conf_info.setAlignment(QtCore.Qt.AlignCenter)
        nav_layout.addWidget(self.btn_prev_conf)
        nav_layout.addWidget(self.lbl_conf_info)
        nav_layout.addWidget(self.btn_next_conf)
        conf_layout.addLayout(nav_layout)

        self.tab_conformers.setLayout(conf_layout)
        self.tabs.addTab(self.tab_conformers, "3D Conformers")

        # Tautomers tab
        self.tab_tautomers = QtWidgets.QWidget()
        taut_layout = QtWidgets.QVBoxLayout(self.tab_tautomers)
        self.taut_text = QtWidgets.QTextEdit()
        self.taut_text.setReadOnly(True)
        taut_layout.addWidget(self.taut_text)
        self.tabs.addTab(self.tab_tautomers, "Tautomers")

        # Microstates tab
        self.tab_microstates = QtWidgets.QWidget()
        micro_layout = QtWidgets.QVBoxLayout(self.tab_microstates)
        self.states_text = QtWidgets.QTextEdit()
        self.states_text.setReadOnly(True)
        micro_layout.addWidget(self.states_text)
        self.tabs.addTab(self.tab_microstates, "Conformational Ensemble")

        layout.addStretch()

        # Signal connections
        self.list_conformers.currentRowChanged.connect(self.on_conformer_selected)
        self.btn_prev_conf.clicked.connect(self.show_prev_conformer)
        self.btn_next_conf.clicked.connect(self.show_next_conformer)

    def update_info(self, mol, smiles, states_count=0, tautomers_count=0, conformers=None, energies=None):
        if mol is None:
            self.props_text.setPlainText("Invalid molecule")
            return

        try:
            report_text = medchem_report_text(mol, lang="en")
        except Exception as e:
            report_text = f"Error generating report: {str(e)}"
        self.props_text.setPlainText(report_text)

        self.taut_text.setPlainText(f"Number of tautomers: {tautomers_count}")

        self.states_text.setPlainText(
            f"Additional conformers generated by sampling: {states_count}\n"
            f"Total 3D conformers (tautomers + sampled): {tautomers_count + states_count}"
        )
        self.conformers = conformers if conformers is not None else []
        self.energies = energies if energies is not None else []
        self.list_conformers.blockSignals(True)
        self.list_conformers.clear()
        for i, energy in enumerate(self.energies):
            energy_str = f"{energy:.3f} kcal/mol" if energy is not None and not math.isnan(energy) else "N/A"
            self.list_conformers.addItem(f"State {i+1}: {energy_str}")
        self.list_conformers.blockSignals(False)

        if self.conformers:
            self.current_index = 0
            self.list_conformers.setCurrentRow(self.current_index)
            self._update_conformer_label()
        else:
            self.current_index = -1
            self.lbl_conf_info.setText("State: - / -")

    def on_conformer_selected(self, index):
        if 0 <= index < len(self.conformers):
            self.current_index = index
            self._update_conformer_label()
            # Could emit a signal here to update 3D view if needed

    def show_prev_conformer(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.list_conformers.setCurrentRow(self.current_index)

    def show_next_conformer(self):
        if self.current_index < len(self.conformers) - 1:
            self.current_index += 1
            self.list_conformers.setCurrentRow(self.current_index)

    def _update_conformer_label(self):
        total = len(self.conformers)
        if 0 <= self.current_index < total:
            self.lbl_conf_info.setText(f"State: {self.current_index + 1} / {total}")
        else:
            self.lbl_conf_info.setText("State: - / -")

class TorsionScanWorker(QtCore.QObject):
    finished = QtCore.Signal()
    progress = QtCore.Signal(int)
    result_ready = QtCore.Signal(list, list)
    conformer_ready = QtCore.Signal(int, str)
    error_occurred = QtCore.Signal(str)

    def __init__(self, mol, bond, method, angle_start, angle_end, angle_step):
        super().__init__()
        # ИСХОДНОЕ ИСПРАВЛЕНИЕ: addCoords=False, чтобы не менять геометрию случайно
        self.mol = Chem.AddHs(mol, addCoords=False)
        self.bond = bond
        self.method = method
        self.angle_start = float(angle_start)
        self.angle_end = float(angle_end)
        self.angle_step = float(angle_step)
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            angles, energies = [], []
            num_points = int((self.angle_end - self.angle_start) / self.angle_step) + 1
            
            for i in range(num_points):
                if not self._is_running: break
                angle = self.angle_start + i * self.angle_step
                angles.append(angle)
                
                mol_copy = Chem.Mol(self.mol)
                conf = mol_copy.GetConformer()
                rdMolTransforms.SetDihedralDeg(conf, *self.bond, angle)
                
                energy = float('nan')
                try:
                    if self.method == "UFF":
                        ff = AllChem.UFFGetMoleculeForceField(mol_copy, confId=conf.GetId())
                        if ff: ff.UFFAddTorsionConstraint(*self.bond, False, angle, angle, 1.0e6)
                    else:
                        prop = AllChem.MMFFGetMoleculeProperties(mol_copy)
                        ff = AllChem.MMFFGetMoleculeForceField(mol_copy, prop, confId=conf.GetId())
                        if ff: ff.MMFFAddTorsionConstraint(*self.bond, False, angle, angle, 1.0e6)
                    
                    if ff:
                        ff.Initialize()
                        # УВЕЛИЧЕННАЯ ТОЧНОСТЬ: 2000 итераций и контроль сходимости
                        ff.Minimize(maxIts=2000, forceTol=1e-4, energyTol=1e-6)
                        energy = ff.CalcEnergy()
                except: pass
                
                energies.append(energy)
                self.conformer_ready.emit(i, Chem.MolToMolBlock(mol_copy))
                self.progress.emit(int(((i + 1) / num_points) * 100))

            valid_e = [e for e in energies if not math.isnan(e)]
            if valid_e:
                min_e = min(valid_e)
                rel_energies = [(e - min_e) for e in energies]
            else: rel_energies = energies

            self.result_ready.emit(angles, rel_energies); self.finished.emit()
        except Exception as e:
            self.error_occurred.emit(str(e)); self.finished.emit()
            
# class TorsionScanWorker(QtCore.QObject):
#     finished = QtCore.Signal()
#     progress = QtCore.Signal(int)
#     result_ready = QtCore.Signal(list, list)
#     conformer_ready = QtCore.Signal(int, str)
#     error_occurred = QtCore.Signal(str)

#     def __init__(self, mol, bond, method, angle_start, angle_end, angle_step):
#         super().__init__()
#         # Для корректной торсии ОБЯЗАТЕЛЬНО нужны водороды (VdW силы)
#         self.mol = Chem.AddHs(mol, addCoords=True)
#         self.bond = bond
#         self.method = method
#         self.angle_start = float(angle_start)
#         self.angle_end = float(angle_end)
#         self.angle_step = float(angle_step)
#         self._is_running = True

#     def stop(self):
#         self._is_running = False

#     def run(self):
#         try:
#             angles, energies = [], []
#             num_points = int((self.angle_end - self.angle_start) / self.angle_step) + 1
            
#             for i in range(num_points):
#                 if not self._is_running: break
#                 angle = self.angle_start + i * self.angle_step
#                 angles.append(angle)
                
#                 mol_copy = Chem.Mol(self.mol)
#                 conf = mol_copy.GetConformer()
#                 rdMolTransforms.SetDihedralDeg(conf, *self.bond, angle)
                
#                 energy = float('nan')
#                 try:
#                     # ФИКСАЦИЯ УГЛА (Constraint) - чтобы не было "нулей"
#                     if self.method == "UFF":
#                         ff = AllChem.UFFGetMoleculeForceField(mol_copy, confId=conf.GetId())
#                         if ff: ff.UFFAddTorsionConstraint(*self.bond, False, angle, angle, 1.0e6)
#                     else:
#                         prop = AllChem.MMFFGetMoleculeProperties(mol_copy)
#                         ff = AllChem.MMFFGetMoleculeForceField(mol_copy, prop, confId=conf.GetId())
#                         if ff: ff.MMFFAddTorsionConstraint(*self.bond, False, angle, angle, 1.0e6)
                    
#                     if ff:
#                         ff.Initialize(); ff.Minimize(maxIts=500); energy = ff.CalcEnergy()
#                 except: pass
                
#                 energies.append(energy)
#                 self.conformer_ready.emit(i, Chem.MolToMolBlock(mol_copy))
#                 self.progress.emit(int(((i + 1) / num_points) * 100))

#             valid_e = [e for e in energies if not math.isnan(e)]
#             if valid_e:
#                 min_e = min(valid_e)
#                 rel_energies = [(e - min_e) for e in energies]
#             else: rel_energies = energies

#             self.result_ready.emit(angles, rel_energies); self.finished.emit()
#         except Exception as e:
#             self.error_occurred.emit(str(e)); self.finished.emit()


class TorsionScanPanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.mol = None
        self.worker = None
        self.thread = None
        self.conformers = []
        self.angles = []
        self.energies = []
        self.current_index = 0
        self.color_scheme = "By Atom Type" # По умолчанию
        self._setup_ui()

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        controls = QtWidgets.QHBoxLayout()
        self.cb_bond = QtWidgets.QComboBox()
        self.btn_refresh = QtWidgets.QPushButton("Refresh Bonds")
        self.cb_method = QtWidgets.QComboBox()
        self.cb_method.addItems(["MMFF94", "UFF"])
        
        controls.addWidget(QtWidgets.QLabel("Bond:"))
        controls.addWidget(self.cb_bond); controls.addWidget(self.btn_refresh)
        controls.addWidget(QtWidgets.QLabel("Method:")); controls.addWidget(self.cb_method)
        
        self.sb_start = QtWidgets.QSpinBox(); self.sb_start.setRange(0, 360)
        self.sb_end = QtWidgets.QSpinBox(); self.sb_end.setRange(0, 360); self.sb_end.setValue(360)
        self.sb_step = QtWidgets.QSpinBox(); self.sb_step.setRange(1, 90); self.sb_step.setValue(15)
        
        controls.addWidget(QtWidgets.QLabel("Start:")); controls.addWidget(self.sb_start)
        controls.addWidget(QtWidgets.QLabel("End:")); controls.addWidget(self.sb_end)
        controls.addWidget(QtWidgets.QLabel("Step:")); controls.addWidget(self.sb_step)
        layout.addLayout(controls)

        btns = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("Start Scan")
        self.btn_stop = QtWidgets.QPushButton("Stop"); self.btn_stop.setEnabled(False)
        self.btn_save_plot = QtWidgets.QPushButton("Save Plot") 
        self.btn_export = QtWidgets.QPushButton("Export CSV")
        btns.addWidget(self.btn_start); btns.addWidget(self.btn_stop)
        btns.addStretch(); btns.addWidget(self.btn_save_plot); btns.addWidget(self.btn_export)
        layout.addLayout(btns)

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        self.fig = Figure(figsize=(5, 4), facecolor='#121212')
        self.canvas = FigureCanvas(self.fig); self.ax = self.fig.add_subplot(111)
        self._format_ax(); self.splitter.addWidget(self.canvas)

        self.view3d = QWebEngineView(); self.view3d.setMinimumHeight(350); self.splitter.addWidget(self.view3d)
        layout.addWidget(self.splitter)

        nav = QtWidgets.QHBoxLayout()
        self.btn_prev = QtWidgets.QPushButton("◀"); self.btn_next = QtWidgets.QPushButton("▶")
        self.lbl_conf_info = QtWidgets.QLabel("Scan point: - / -")
        self.lbl_conf_info.setAlignment(QtCore.Qt.AlignCenter)
        nav.addWidget(self.btn_prev); nav.addWidget(self.lbl_conf_info); nav.addWidget(self.btn_next)
        layout.addLayout(nav)

        self.btn_start.clicked.connect(self.start_scan)
        self.btn_stop.clicked.connect(self.stop_scan)
        self.btn_refresh.clicked.connect(self._populate_bonds)
        self.btn_prev.clicked.connect(self.show_prev_conformer)
        self.btn_next.clicked.connect(self.show_next_conformer)
        self.btn_save_plot.clicked.connect(self.save_plot)
        self.btn_export.clicked.connect(self.export_results)

    def _format_ax(self):
        self.ax.set_facecolor('#121212')
        for spine in self.ax.spines.values(): spine.set_color('#444')
        self.ax.tick_params(colors='white')
        self.ax.xaxis.label.set_color('white'); self.ax.yaxis.label.set_color('white')
        self.ax.set_xlabel("Torsion Angle (°)"); self.ax.set_ylabel("Rel. Energy (kcal/mol)")
        self.ax.grid(True, alpha=0.1)

    def set_molecule(self, mol):
        self.mol = mol; self._populate_bonds()

    def _populate_bonds(self):
        self.cb_bond.clear()
        if not self.mol: return
        p = Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')
        matches = self.mol.GetSubstructMatches(p)
        for j, k in matches:
            aj, ak = self.mol.GetAtomWithIdx(j), self.mol.GetAtomWithIdx(k)
            nj = [n.GetIdx() for n in aj.GetNeighbors() if n.GetIdx() != k]
            nk = [n.GetIdx() for n in ak.GetNeighbors() if n.GetIdx() != j]
            if nj and nk:
                self.cb_bond.addItem(f"Bond {j}-{k} ({aj.GetSymbol()}-{ak.GetSymbol()})", (nj[0], j, k, nk[0]))

    def start_scan(self):
        if not self.mol: return
        bond = self.cb_bond.currentData()
        if not bond: return
        self._reset_results()
        self.btn_start.setEnabled(False); self.btn_stop.setEnabled(True)
        self.thread = QtCore.QThread()
        self.worker = TorsionScanWorker(self.mol, bond, self.cb_method.currentText(),
                                        self.sb_start.value(), self.sb_end.value(), self.sb_step.value())
        self.worker.moveToThread(self.thread)
        self.worker.conformer_ready.connect(self.handle_conformer)
        self.worker.result_ready.connect(self.handle_results)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(lambda: self.btn_start.setEnabled(True))
        self.worker.finished.connect(lambda: self.btn_stop.setEnabled(False))
        self.thread.started.connect(self.worker.run); self.thread.start()

    def stop_scan(self): # ТОТ САМЫЙ МЕТОД, КОТОРОГО НЕ ХВАТАЛО
        if self.worker: self.worker.stop()

    def handle_conformer(self, idx, mol_block):
        while len(self.conformers) <= idx: self.conformers.append(None)
        self.conformers[idx] = mol_block

    def handle_results(self, angles, energies):
        self.angles, self.energies = angles, energies
        self.show_conformer(0)

    def show_conformer(self, idx):
        if not (0 <= idx < len(self.conformers)) or self.conformers[idx] is None: return
        self.current_index = idx
        energy = self.energies[idx]
        
        # Логика окраски
        style_js = "{stick:{radius:0.2, colorscheme: 'default'}}"
        color_hex = "#4dabf7"
        if self.color_scheme == "By Energy" and self.energies:
            ve = [e for e in self.energies if not math.isnan(e)]
            if ve and not math.isnan(energy):
                norm = (energy - min(ve)) / (max(ve) - min(ve)) if (max(ve) - min(ve)) > 0 else 0
                color_hex = f"#{int(255*norm):02x}50{int(255*(1-norm)):02x}"
                style_js = f"{{stick:{{radius:0.2, color: '{color_hex}'}}}}"

        sdf_js = self.conformers[idx].replace('\\', '\\\\').replace('`', '\\`')
        html = f"""<html><head><script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script></head>
        <body style="margin:0; background:#121212; color:white; font-family:sans-serif;">
        <div style="position:absolute; top:10; left:10; z-index:10; background:rgba(0,0,0,0.6); padding:10px; border-left:4px solid {color_hex};">
            Angle: {self.angles[idx]:.1f}°<br>Rel. Energy: <b style="color:{color_hex}">{energy:.3f}</b> kcal/mol
        </div>
        <div id="c" style="width:100vw; height:100vh;"></div>
        <script>let v = $3Dmol.createViewer('c', {{backgroundColor:'#121212'}});
        v.addModel(`{sdf_js}`, 'sdf'); v.setStyle({{}}, {style_js}); v.zoomTo(); v.render();</script></body></html>"""
        self.view3d.setHtml(html)
        self.lbl_conf_info.setText(f"Scan point: {idx+1} / {len(self.conformers)}"); self._update_plot()

    def _update_plot(self):
        self.ax.clear(); self._format_ax()
        v = [(a, e) for a, e in zip(self.angles, self.energies) if not math.isnan(e)]
        if v:
            xs, ys = zip(*v)
            self.ax.plot(xs, ys, '-o', color='#4dabf7', markersize=4, alpha=0.8)
            if 0 <= self.current_index < len(self.angles):
                self.ax.plot(self.angles[self.current_index], self.energies[self.current_index], 'ro', markersize=8)
        self.canvas.draw()

    def save_plot(self):
        if not self.angles: return
        f, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Plot", "plot.png", "PNG (*.png)")
        if f: self.fig.savefig(f, dpi=300, facecolor='#121212')

    def export_results(self):
        if not self.angles: return
        f, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export CSV", "scan.csv", "CSV (*.csv)")
        if f:
            with open(f, 'w') as fh:
                fh.write("Angle,RelEnergy_kcal_mol\n")
                for a, e in zip(self.angles, self.energies): fh.write(f"{a},{e}\n")

    def show_prev_conformer(self): 
        if self.current_index > 0: self.show_conformer(self.current_index - 1)
    def show_next_conformer(self): 
        if self.current_index < len(self.conformers) - 1: self.show_conformer(self.current_index + 1)

    def _reset_results(self):
        self.angles = []; self.energies = []; self.conformers = []; self.current_index = 0
        self.ax.clear(); self._format_ax(); self.canvas.draw()
        self.lbl_conf_info.setText("Scan point: - / -")


class Main(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Automated Molecular Analysis Toolkit")
        self.setMinimumSize(1400, 900)

        try:
            icon_path = rp("icon.ico")
            if Path(icon_path).exists():
                self.setWindowIcon(QtGui.QIcon(icon_path))
        except Exception:
            pass

        self._last_svg = None
        self._last_png = None
        self._last_sdf = None

        self._states_smi = []
        self._states_sdf = []
        self._state_idx = -1

        self.sessions = []
        self.current_mol = None
        self.current_smiles = ""

        self._build_ui()
        self._build_menu()
        self._load_sessions()
        self._setup_dark_theme()

        self.statusBar().showMessage("Ready - Load a molecule to begin")

        self.setCorner(QtCore.Qt.TopLeftCorner, QtCore.Qt.LeftDockWidgetArea)
        self.setCorner(QtCore.Qt.BottomLeftCorner, QtCore.Qt.LeftDockWidgetArea)
        self.setCorner(QtCore.Qt.TopRightCorner, QtCore.Qt.RightDockWidgetArea)
        self.setCorner(QtCore.Qt.BottomRightCorner, QtCore.Qt.RightDockWidgetArea)
        self.setDockNestingEnabled(True)

    def _setup_dark_theme(self):
        style = """
        QMainWindow {
            background-color: #121212;
            color: #eee;
        }
        QTabWidget::pane {
            border: 1px solid #444;
            background-color: #222;
        }
        QTabBar::tab {
            background-color: #333;
            color: #eee;
            border: 1px solid #555;
            padding: 8px 16px;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background-color: #121212;
            border-bottom: 1px solid #121212;
        }
        QTabBar::tab:hover {
            background-color: #444;
        }
        QPushButton {
            background-color: #6c757d;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: 500;
        }
        QPushButton:hover {
            background-color: #5a6268;
        }
        QPushButton:pressed {
            background-color: #4e555b;
        }
        QPushButton:disabled {
            background-color: #555;
            color: #ccc;
        }
        QGroupBox {
            font-weight: 600;
            border: 2px solid #444;
            border-radius: 6px;
            margin: 8px 0px;
            padding-top: 10px;
            color: #eee;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            border: 1px solid #555;
            border-radius: 4px;
            padding: 6px 8px;
            background-color: #222;
            color: #eee;
        }
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
            border-color: #80bdff;
            background-color: #333;
        }
        QTextEdit {
            border: 1px solid #555;
            border-radius: 4px;
            background-color: #222;
            color: #eee;
        }
        QDockWidget {
            background-color: #222;
            color: #eee;
        }
        QDockWidget::title {
            background-color: #333;
            border: 1px solid #444;
            padding: 6px;
            font-weight: 600;
            color: #eee;
        }
        QProgressBar {
            border: 1px solid #555;
            border-radius: 4px;
            text-align: center;
            background-color: #333;
            color: #eee;
        }
        QProgressBar::chunk {
            background-color: #28a745;
            border-radius: 3px;
        }
        QStatusBar {
            background-color: #333;
            border-top: 1px solid #444;
            color: #eee;
        }
        QLabel {
            color: #eee;
        }
        """
        self.setStyleSheet(style)

    def _build_menu(self):
        mbar = self.menuBar()

        self.menu_file = mbar.addMenu("&File")

        # Create actions only once, prevent duplicates
        self.act_open = QtGui.QAction("&Open Molecule...", self)
        self.act_open.setShortcut(QtGui.QKeySequence.Open)
        self.act_open.setStatusTip("Open molecule from file")
        self.act_open.triggered.connect(self.on_open_file)
        self.menu_file.addAction(self.act_open)

        self.menu_file.addSeparator()
        self.menu_save = mbar.addMenu("&Save")

        self.act_save2d = QtGui.QAction("Save &2D Image...", self)
        self.act_save2d.setShortcut(QtGui.QKeySequence("Ctrl+2"))
        self.act_save2d.setStatusTip("Save 2D molecular structure image")
        self.act_save2d.triggered.connect(self.on_save_2d)
        self.menu_save.addAction(self.act_save2d)

        self.act_save3d = QtGui.QAction("Save &3D Structure...", self)
        self.act_save3d.setShortcut(QtGui.QKeySequence("Ctrl+3"))
        self.act_save3d.setStatusTip("Save 3D molecular structure")
        self.act_save3d.triggered.connect(self.on_save_3d)
        self.menu_save.addAction(self.act_save3d)

        self.act_snap3d = QtGui.QAction("3D &Snapshot...", self)
        self.act_snap3d.setShortcut(QtGui.QKeySequence("Ctrl+Shift+3"))
        self.act_snap3d.setStatusTip("Capture 3D viewer screenshot")
        self.act_snap3d.triggered.connect(self.on_snap_3d)
        self.menu_save.addAction(self.act_snap3d)

        self.act_savezip = QtGui.QAction("Save &All States...", self)
        self.act_savezip.setStatusTip("Export all conformers and states")
        self.act_savezip.triggered.connect(self.on_save_all_states)
        self.menu_save.addAction(self.act_savezip)

        self.act_savejob = QtGui.QAction("Save &Project...", self)
        self.act_savejob.setShortcut(QtGui.QKeySequence.Save)
        self.act_savejob.setStatusTip("Save current work as project")
        self.act_savejob.triggered.connect(self.on_save_job)
        self.menu_save.addAction(self.act_savejob)

        self.menu_analysis = mbar.addMenu("&Analysis")

        self.act_medchem = QtGui.QAction("&Drug-likeness Report", self)
        self.act_medchem.setShortcut(QtGui.QKeySequence("Ctrl+D"))
        self.act_medchem.setStatusTip("Generate medicinal chemistry report")
        self.act_medchem.triggered.connect(self.on_medchem)
        self.menu_analysis.addAction(self.act_medchem)

        self.menu_tools = mbar.addMenu("&Tools")

        self.act_torsion = QtGui.QAction("&Torsion Scan...", self)
        self.act_torsion.setShortcut(QtGui.QKeySequence("Ctrl+T"))
        self.act_torsion.setStatusTip("Perform conformational torsion scan")
        self.act_torsion.triggered.connect(self.on_torsion_scan)
        self.menu_tools.addAction(self.act_torsion)

        self.act_optimization = QtGui.QAction("Geometry &Optimization", self)
        self.act_optimization.setShortcut(QtGui.QKeySequence("Ctrl+G"))
        self.act_optimization.setStatusTip("Run geometry optimization")
        # Connect optimization action to a placeholder or actual method if implemented
        self.act_optimization.triggered.connect(self.on_geometry_optimization)
        self.menu_tools.addAction(self.act_optimization)

        self.menu_view = mbar.addMenu("&View")

        self.act_dock_left = self.menu_view.addAction("Input Panel")
        self.act_dock_left.setCheckable(True)
        self.act_dock_left.setChecked(True)
        self.act_dock_left.toggled.connect(self.toggle_left_dock)

        self.act_dock_right = self.menu_view.addAction("Parameters Panel")
        self.act_dock_right.setCheckable(True)
        self.act_dock_right.setChecked(True)
        self.act_dock_right.toggled.connect(self.toggle_right_dock)

        self.menu_view.addSeparator()

        self.act_fullscreen = QtGui.QAction("&Full Screen", self)
        self.act_fullscreen.setShortcut(QtGui.QKeySequence.FullScreen)
        self.act_fullscreen.setCheckable(True)
        self.act_fullscreen.triggered.connect(self.toggle_fullscreen)
        self.menu_view.addAction(self.act_fullscreen)

        self.menu_help = mbar.addMenu("&Help")

        self.act_about = QtGui.QAction("&About AZAT", self)
        self.act_about.triggered.connect(self.on_about_azat)
        self.menu_help.addAction(self.act_about)

    def _build_ui(self):
        self.tabs_main = QtWidgets.QTabWidget()
        self.tabs_main.setTabPosition(QtWidgets.QTabWidget.North)
        self.tabs_main.setMovable(True)

        # 3D Structure tab with embedded statebar below viewer
        self.view3d_main = QWebEngineView()
        self.view3d_main.setHtml(
            "<html><body style='background:#222; color:#eee; text-align:center; padding:50px;'><h3>Generate 3D structure to view</h3></body></html>")

        # Statebar for microstate navigation under 3D viewer
        self.statebar = QtWidgets.QWidget()
        self.statebar.setMaximumHeight(32)
        self.statebar.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.statebar.setStyleSheet("background: transparent; border: none;")

        hbox = QtWidgets.QHBoxLayout(self.statebar)
        # почти без отступов, снизу чуть-чуть
        hbox.setContentsMargins(0, 0, 0, 4)
        hbox.setSpacing(4)

        self.btn_state_prev = QtWidgets.QPushButton()
        self.btn_state_prev.setFixedSize(20, 20)
        self.btn_state_prev.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_ArrowLeft))
        self.btn_state_prev.setToolTip("Previous State")
        self.btn_state_prev.setFlat(True)

        self.btn_state_next = QtWidgets.QPushButton()
        self.btn_state_next.setFixedSize(20, 20)
        self.btn_state_next.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_ArrowRight))
        self.btn_state_next.setToolTip("Next State")
        self.btn_state_next.setFlat(True)

        self.lbl_state_info = QtWidgets.QLabel("- / -")
        self.lbl_state_info.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_state_info.setStyleSheet("color: #eee; font-size: 9pt;")

        # стрелки и счётчик компактно по центру
        hbox.addStretch()
        hbox.addWidget(self.btn_state_prev)
        hbox.addWidget(self.lbl_state_info)
        hbox.addWidget(self.btn_state_next)
        hbox.addStretch()

        self.btn_state_prev.clicked.connect(self.on_prev_state)
        self.btn_state_next.clicked.connect(self.on_next_state)

        # Добавляем statebar как нижний элемент панели 3D просмотра
        self.view3d_container = QtWidgets.QWidget()
        vlayout = QtWidgets.QVBoxLayout(self.view3d_container)
        vlayout.setContentsMargins(0, 0, 0, 0)
        vlayout.setSpacing(0)
        vlayout.addWidget(self.view3d_main)
        vlayout.addWidget(self.statebar)

        self.tabs_main.removeTab(self.tabs_main.indexOf(self.view3d_main))
        self.tabs_main.insertTab(0, self.view3d_container, "3D Structure")

        self._update_statebar_visibility()

        # 2D Structure tab
        self.view2d_main = QWebEngineView()
        self.view2d_main.setHtml(
            "<html><body style='background:#121212; color:#eee; text-align:center; padding:50px;'><h3>Load a molecule to view 2D structure</h3></body></html>")
        self.tabs_main.addTab(self.view2d_main, "2D Structure")

        # Analysis tab (MoleculeInfoPanel)
        self.info_panel = MoleculeInfoPanel(self)
        self.tabs_main.addTab(self.info_panel, "Analysis")

        # Torsion Scan tab
        self.torsion_scan_tab = TorsionScanPanel(self)
        self.tabs_main.addTab(self.torsion_scan_tab, "Torsion Scan")

        self.setCentralWidget(self.tabs_main)

        self._build_input_panel()
        self._build_parameters_panel()

        self.dock_left.visibilityChanged.connect(self._on_dock_visibility_changed)
        self.dock_right.visibilityChanged.connect(self._on_dock_visibility_changed)

    def _update_statebar_visibility(self):
        visible = len(self._states_sdf) > 1
        self.statebar.setVisible(visible)
        self.btn_state_prev.setEnabled(visible and self._state_idx > 0)
        self.btn_state_next.setEnabled(visible and (self._state_idx < len(self._states_sdf) - 1))

    def on_prev_state(self):
        if self._state_idx > 0:
            self._state_idx -= 1
            self._update_3d_view()
            self._update_statebar()
            self.statusBar().showMessage(f"Switched to state {_format_state_idx(self._state_idx)}")

    def on_next_state(self):
        if self._state_idx < len(self._states_sdf) - 1:
            self._state_idx += 1
            self._update_3d_view()
            self._update_statebar()
            self.statusBar().showMessage(f"Switched to state {_format_state_idx(self._state_idx)}")

    def _update_statebar(self):
        total = len(self._states_sdf)
        if total > 0 and 0 <= self._state_idx < total:
            # формат 1 / 6, без слова "State"
            self.lbl_state_info.setText(f"{self._state_idx + 1} / {total}")
            self.btn_state_prev.setEnabled(self._state_idx > 0)
            self.btn_state_next.setEnabled(self._state_idx < total - 1)
        else:
            self.lbl_state_info.setText("- / -")
            self.btn_state_prev.setEnabled(False)
            self.btn_state_next.setEnabled(False)

    # Rest of Main class unchanged, copied as-is...

    def _on_dock_visibility_changed(self, visible):
        self.setUpdatesEnabled(False)
        QtCore.QTimer.singleShot(100, lambda: self.setUpdatesEnabled(True))

    def on_job_double_clicked(self, item):
        row = self.jobs.row(item)
        if row < 0 or row >= len(self.sessions):
            return

        if self.current_smiles or self._last_sdf:
            reply = QtWidgets.QMessageBox.question(
                self, "Save Current Project",
                "Do you want to save the current project before opening another?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel
            )
            if reply == QtWidgets.QMessageBox.Yes:
                self.on_save_job()
            elif reply == QtWidgets.QMessageBox.Cancel:
                return

        job = self.sessions[row]
        self.ed_smiles.setText(job["smiles"])
        self.on_build_2d()
        self.statusBar().showMessage(f"Opened project '{job['name']}'")

    def _build_input_panel(self):
        left = QtWidgets.QWidget()
        vleft = QtWidgets.QVBoxLayout(left)
        vleft.setContentsMargins(10, 10, 10, 10)
        vleft.setSpacing(10)

        input_group = QtWidgets.QGroupBox("Molecule Input")
        form = QtWidgets.QFormLayout(input_group)
        form.setContentsMargins(12, 18, 12, 12)
        form.setSpacing(8)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)

        self.ed_smiles = QtWidgets.QLineEdit()
        self.ed_smiles.setPlaceholderText("Enter SMILES string (e.g., CCO for ethanol)")
        self.ed_smiles.setFont(QtGui.QFont("Consolas", 9))

        self.btn_2dimg = QtWidgets.QPushButton("Generate 2D")
        self.btn_2dimg.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogDetailedView))

        self.btn_build = QtWidgets.QPushButton("Generate 3D")
        self.btn_build.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_ComputerIcon))

        self.btn_new = QtWidgets.QPushButton("New Project")
        self.btn_new.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_FileIcon))

        form.addRow("SMILES:", self.ed_smiles)
        form.addRow(self.btn_2dimg)
        form.addRow(self.btn_build)
        form.addRow(self.btn_new)

        vleft.addWidget(input_group, 0)

        self.jobs_label = QtWidgets.QLabel("Saved Projects")
        self.jobs_label.setFont(QtGui.QFont("", 10, QtGui.QFont.Bold))

        self.jobs = QtWidgets.QListWidget()
        self.jobs.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.jobs.setAlternatingRowColors(True)

        vleft.addWidget(self.jobs_label, 0)
        vleft.addWidget(self.jobs, 1)

        dock_left = QtWidgets.QDockWidget("Molecule Input", self)
        dock_left.setWidget(left)
        dock_left.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable |
                             QtWidgets.QDockWidget.DockWidgetFloatable)
        dock_left.setMinimumWidth(300)
        dock_left.setMaximumWidth(400)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock_left)
        self.dock_left = dock_left

        self.ed_smiles.returnPressed.connect(self.on_build_2d)
        self.btn_2dimg.clicked.connect(self.on_build_2d)
        self.btn_build.clicked.connect(self.on_build_3d)
        self.btn_new.clicked.connect(self.on_new_job)
        self.jobs.currentRowChanged.connect(self.on_job_selected)
        self.jobs.itemDoubleClicked.connect(self.on_job_double_clicked)
        self.jobs.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.jobs.customContextMenuRequested.connect(self.on_jobs_context_menu)

    def on_jobs_context_menu(self, pos):
        item = self.jobs.itemAt(pos)
        if item is None:
            return

        menu = QtWidgets.QMenu()
        delete_action = menu.addAction("Delete Project")
        action = menu.exec(self.jobs.mapToGlobal(pos))
        if action == delete_action:
            row = self.jobs.row(item)
            if 0 <= row < len(self.sessions):
                reply = QtWidgets.QMessageBox.question(
                    self, "Delete Project",
                    f"Are you sure you want to delete project '{self.sessions[row]['name']}'?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
                )
                if reply == QtWidgets.QMessageBox.Yes:
                    del self.sessions[row]
                    self._save_sessions()
                    self._update_jobs_list()
                    self.statusBar().showMessage("Project deleted")

    def _build_parameters_panel(self):
        right_scroll = QtWidgets.QScrollArea()
        right_scroll.setWidgetResizable(True)
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        rv.setContentsMargins(10, 10, 10, 10)
        rv.setSpacing(14)

        # 3D Structure Generation group
        self.gb_method = QtWidgets.QGroupBox("3D Structure Generation")
        frm_method = QtWidgets.QFormLayout(self.gb_method)
        frm_method.setContentsMargins(12, 18, 12, 18)
        frm_method.setSpacing(8)

        self.cb_engine = QtWidgets.QComboBox()
        self.cb_engine.addItems(["ETKDGv3", "ETKDGv2", "ETKDGv1", "UFF", "MMFF94"])
        self.cb_engine.setCurrentText("ETKDGv3")

        self.cb_ff = QtWidgets.QComboBox()
        self.cb_ff.addItems(["UFF", "MMFF94"])
        self.cb_ff.setCurrentText("UFF")

        self.sb_confs = QtWidgets.QSpinBox()
        self.sb_confs.setRange(1, 1000)
        self.sb_confs.setValue(50)
        self.sb_confs.setSuffix(" conformers")

        self.sb_seed = QtWidgets.QSpinBox()
        self.sb_seed.setRange(0, 1000000)
        self.sb_seed.setValue(42)

        frm_method.addRow("Generation method:", self.cb_engine)
        frm_method.addRow("Force field:", self.cb_ff)
        frm_method.addRow("Conformers per tautomer:", self.sb_confs)
        frm_method.addRow("Random seed:", self.sb_seed)
        rv.addWidget(self.gb_method)

        # Protonation & Tautomers (updated with tautomer mode)
        self.gb_protaut = QtWidgets.QGroupBox("Charge & Tautomers")
        frm_protaut = QtWidgets.QFormLayout(self.gb_protaut)
        frm_protaut.setContentsMargins(12, 18, 12, 18)
        frm_protaut.setSpacing(8)

        self.cb_prot_method = QtWidgets.QComboBox()
        self.cb_prot_method.addItems(["rdkit"])
        self.cb_prot_method.setCurrentText("rdkit")

        self.sb_ph = QtWidgets.QDoubleSpinBox()
        self.sb_ph.setRange(0.0, 14.0)
        self.sb_ph.setValue(7.4)
        self.sb_ph.setSingleStep(0.1)
        self.sb_ph.setSuffix(" pH")

        # NEW: Tautomer mode selection
        self.cb_tautomer_mode = QtWidgets.QComboBox()
        self.cb_tautomer_mode.addItems(["Conservative", "Balanced", "Aggressive"])
        self.cb_tautomer_mode.setCurrentText("Balanced")
        self.cb_tautomer_mode.setToolTip(
            "Conservative: Fewer tautomers, more stable\n"
            "Balanced: Good for most molecules\n"
            "Aggressive: Maximum tautomer coverage"
        )

        self.cb_taut_gen = QtWidgets.QCheckBox("Generate tautomers")
        self.cb_taut_gen.setChecked(True)

        frm_protaut.addRow("Charge model:", self.cb_prot_method)
        frm_protaut.addRow("Target pH:", self.sb_ph)
        frm_protaut.addRow("Tautomer mode:", self.cb_tautomer_mode)
        frm_protaut.addRow("", self.cb_taut_gen)
        rv.addWidget(self.gb_protaut)

        # Conformational States group
        self.gb_micro = QtWidgets.QGroupBox("Conformer Sampling")
        frm_micro = QtWidgets.QFormLayout(self.gb_micro)
        frm_micro.setContentsMargins(12, 18, 12, 18)
        frm_micro.setSpacing(8)

        self.cb_micro = QtWidgets.QCheckBox("Enable conformer sampling")
        self.cb_micro.setChecked(False)

        self.sb_micro_confs = QtWidgets.QSpinBox()
        self.sb_micro_confs.setRange(1, 200)
        self.sb_micro_confs.setValue(10)
        self.sb_micro_confs.setSuffix(" conf./state")

        self.sb_micro_topk = QtWidgets.QSpinBox()
        self.sb_micro_topk.setRange(1, 100)
        self.sb_micro_topk.setValue(5)
        self.sb_micro_topk.setSuffix(" lowest-E conf.")

        self.sb_micro_rmsd = QtWidgets.QDoubleSpinBox()
        self.sb_micro_rmsd.setRange(0.0, 2.0)
        self.sb_micro_rmsd.setValue(0.5)
        self.sb_micro_rmsd.setSingleStep(0.1)
        self.sb_micro_rmsd.setSuffix(" \u00c5 RMSD")

        frm_micro.addRow("", self.cb_micro)
        frm_micro.addRow("Conformers to sample:", self.sb_micro_confs)
        frm_micro.addRow("Top K lowest-energy conf. :", self.sb_micro_topk)
        frm_micro.addRow("RMSD threshold:", self.sb_micro_rmsd)
        rv.addWidget(self.gb_micro)

        # Structure Validation group
        self.gb_val = QtWidgets.QGroupBox("Structure Validation")
        frm_val = QtWidgets.QFormLayout(self.gb_val)
        frm_val.setContentsMargins(12, 18, 12, 18)
        frm_val.setSpacing(8)

        self.btn_validate = QtWidgets.QPushButton("Validate Geometry")
        self.btn_validate.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogApplyButton))

        self.sb_bondtol = QtWidgets.QDoubleSpinBox()
        self.sb_bondtol.setRange(0.01, 0.5)
        self.sb_bondtol.setValue(0.1)
        self.sb_bondtol.setSingleStep(0.01)
        self.sb_bondtol.setSuffix(" \u00c5")

        self.sb_angletol = QtWidgets.QDoubleSpinBox()
        self.sb_angletol.setRange(1.0, 20.0)
        self.sb_angletol.setValue(10.0)
        self.sb_angletol.setSingleStep(1.0)
        self.sb_angletol.setSuffix("\u00b0")

        self.sb_planarity = QtWidgets.QDoubleSpinBox()
        self.sb_planarity.setRange(0.01, 0.5)
        self.sb_planarity.setValue(0.1)
        self.sb_planarity.setSingleStep(0.01)
        self.sb_planarity.setSuffix(" \u00c5")

        frm_val.addRow("", self.btn_validate)
        frm_val.addRow("Bond length tolerance:", self.sb_bondtol)
        frm_val.addRow("Angle tolerance:", self.sb_angletol)
        frm_val.addRow("Planarity tolerance:", self.sb_planarity)
        rv.addWidget(self.gb_val)

        # Torsion Scan Settings group (for color scheme)
        self.gb_torsion_scan = QtWidgets.QGroupBox("Torsion Scan Settings")
        frm_torsion = QtWidgets.QFormLayout(self.gb_torsion_scan)
        frm_torsion.setContentsMargins(12, 18, 12, 18)
        frm_torsion.setSpacing(8)

        self.cb_torsion_color_scheme = QtWidgets.QComboBox()
        self.cb_torsion_color_scheme.addItems(["By Atom Type", "By Energy"])
        self.cb_torsion_color_scheme.setCurrentIndex(0)
        frm_torsion.addRow("Color scheme:", self.cb_torsion_color_scheme)

        rv.addWidget(self.gb_torsion_scan)
        try:
            self.cb_torsion_color_scheme.currentTextChanged.connect(self._on_torsion_color_changed)
        except Exception:
            pass

        rv.addStretch(1)
        right_scroll.setWidget(right)

        dock_right = QtWidgets.QDockWidget("Generation Parameters", self)
        dock_right.setWidget(right_scroll)
        dock_right.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable |
                               QtWidgets.QDockWidget.DockWidgetFloatable)
        dock_right.setMinimumWidth(300)
        dock_right.setMaximumWidth(400)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock_right)
        self.dock_right = dock_right

        self.btn_validate.clicked.connect(self.on_validate)
        self.cb_torsion_color_scheme.currentIndexChanged.connect(self._on_torsion_color_changed)

    def _on_torsion_color_changed(self, text):
        try:
            if hasattr(self, "torsion_scan_tab") and self.torsion_scan_tab is not None:
                self.torsion_scan_tab.color_scheme = text
                idx = getattr(self, "torsion_scan_tab").current_index if hasattr(self.torsion_scan_tab, "current_index") else 0
                try:
                    self.torsion_scan_tab.show_conformer(idx)
                except Exception:
                    pass
        except Exception:
            pass

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def toggle_left_dock(self, checked):
        if checked:
            self.dock_left.show()
        else:
            self.dock_left.hide()

    def toggle_right_dock(self, checked):
        if checked:
            self.dock_right.show()
        else:
            self.dock_right.hide()

    def on_torsion_scan(self):
        # Проверяем, есть ли молекула и ее 3D данные
        sdf_to_use = None
        if self._state_idx >= 0 and self._state_idx < len(self._states_sdf):
            sdf_to_use = self._states_sdf[self._state_idx]
        else:
            sdf_to_use = self._last_sdf

        if not sdf_to_use:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please generate a 3D structure first.")
            return

        # Создаем молекулу из SDF-блока (с сохранением координат)
        mol_3d = Chem.MolFromMolBlock(sdf_to_use, removeHs=False)
        if not mol_3d:
            QtWidgets.QMessageBox.warning(self, "Error", "Could not parse 3D data for torsion scan.")
            return

        # Передаем молекулу в панель
        self.torsion_scan_tab.set_molecule(mol_3d)
        # Переключаемся на вкладку сканирования
        self.tabs_main.setCurrentWidget(self.torsion_scan_tab)
        self.statusBar().showMessage("Torsion scan panel ready.")

    def on_geometry_optimization(self):
        """
        Оптимизирует именно тот конформер (State), который сейчас отображается на панели.
        """
        # Проверяем, есть ли что оптимизировать
        current_sdf = None
        if self._state_idx >= 0 and self._state_idx < len(self._states_sdf):
            current_sdf = self._states_sdf[self._state_idx]
        else:
            current_sdf = self._last_sdf

        if not current_sdf:
            QtWidgets.QMessageBox.warning(self, "Warning", "No 3D structure visible to optimize.")
            return

        ff_type = self.cb_ff.currentText().upper()
        self.statusBar().showMessage(f"Optimizing current state ({ff_type})...")
        QtWidgets.QApplication.processEvents()

        try:
            # 1. Загружаем молекулу из текущего SDF-блока
            mol = Chem.MolFromMolBlock(current_sdf, removeHs=False)
            if mol is None: raise ValueError("Invalid molecular data.")

            # 2. Добавляем водороды (критично для правильных сил Ван-дер-Ваальса)
            mol = Chem.AddHs(mol, addCoords=True)
            
            # 3. Выбор и запуск силового поля
            success = False
            if ff_type == "MMFF94":
                if AllChem.MMFFHasAllMoleculeParams(mol):
                    # 1000 итераций достаточно для большинства лигандов
                    res = AllChem.MMFFOptimizeMolecule(mol, mmffVariant='MMFF94', maxIters=1000)
                    if res != -1: success = True
                else:
                    self.statusBar().showMessage("MMFF parameters missing. Using UFF fallback...")
                    res = AllChem.UFFOptimizeMolecule(mol, maxIters=1000)
                    if res != -1: success = True
            else:
                res = AllChem.UFFOptimizeMolecule(mol, maxIters=1000)
                if res != -1: success = True

            if success:
                # 4. Сохраняем результат обратно именно в этот индекс состояний
                optimized_sdf = Chem.MolToMolBlock(mol)
                if self._state_idx >= 0:
                    self._states_sdf[self._state_idx] = optimized_sdf
                
                # Обновляем основную переменную, если это была главная модель
                if self._state_idx == 0 or self._state_idx == -1:
                    self._last_sdf = optimized_sdf

                # 5. Обновляем визуализацию и список энергий в Analysis
                self._update_3d_view()
                
                # Пересчитываем энергию для вкладки Analysis
                try:
                    ff = AllChem.UFFGetMoleculeForceField(mol)
                    new_energy = ff.CalcEnergy() if ff else float('nan')
                    if hasattr(self, 'energies_list') and 0 <= self._state_idx < len(self.energies_list):
                        self.energies_list[self._state_idx] = new_energy
                except: pass

                self._update_info_panel()
                self.statusBar().showMessage(f"State {self._state_idx + 1} optimized successfully.")
            else:
                self.statusBar().showMessage("Optimization failed to converge.")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Optimization failed: {str(e)}")

    def on_build_2d(self):
        smi = self.ed_smiles.text().strip()
        if not smi:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please enter a SMILES string first.")
            return
        if self.current_smiles and smi != self.current_smiles:
            reply = QtWidgets.QMessageBox.question(
                self, "New Molecule",
                "Save current project before building a new molecule?",
                QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel
            )
            if reply == QtWidgets.QMessageBox.Save:
                self.on_save_job()
            elif reply == QtWidgets.QMessageBox.Cancel:
                return

            # сбрасываем всё, но текст SMILES в поле оставляем
            self._reset_workspace(clear_smiles_field=False)

        try:
            self.statusBar().showMessage("Parsing SMILES...")
            QtWidgets.QApplication.processEvents()

            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError("Invalid SMILES string")

            self.current_mol = mol
            self.current_smiles = smi

            self.statusBar().showMessage("Generating 2D structure...")
            QtWidgets.QApplication.processEvents()

            self._update_2d_view()
            self._update_info_panel()

            self.statusBar().showMessage("2D structure generated successfully")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to generate 2D structure:\n{str(e)}")
            self.statusBar().showMessage("Error generating 2D structure")
    def _reset_workspace(self, clear_smiles_field: bool = False):
        """
        Полностью очистить текущее состояние молекулы/конформеров/вьюверов.
        Используется перед началом работы с НОВОЙ молекулой.
        """
        if clear_smiles_field:
            self.ed_smiles.clear()

        self.current_mol = None
        self.current_smiles = ""
        self._last_svg = None
        self._last_png = None
        self._last_sdf = None
        self._states_smi = []
        self._states_sdf = []
        self._state_idx = -1

        self.view2d_main.setHtml(
            "<html><body style='background:#121212; color:#eee; text-align:center; padding:50px;'><h3>Load a molecule to view 2D structure</h3></body></html>"
        )
        self.view3d_main.setHtml(
            "<html><body style='background:#222; color:#eee; text-align:center; padding:50px;'><h3>Generate 3D structure to view</h3></body></html>"
        )
        self._update_statebar_visibility()
        self._update_statebar()

    def on_build_3d(self):
        smi = (self.ed_smiles.text() or "").strip()

        # Если нет молекулы и SMILES тоже пуст — нечего строить
        if not self.current_mol and not smi:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please enter a SMILES string first.")
            return

        # Если SMILES в поле изменился относительно текущего проекта —
        # считаем, что пользователь хочет НОВУЮ молекулу
        if smi and self.current_smiles and smi != self.current_smiles:
            reply = QtWidgets.QMessageBox.question(
                self, "New Molecule",
                "Save current project before generating 3D for a new molecule?",
                QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel
            )
            if reply == QtWidgets.QMessageBox.Save:
                self.on_save_job()
            elif reply == QtWidgets.QMessageBox.Cancel:
                return

            # Сбрасываем рабочее состояние, но SMILES в поле оставляем
            self._reset_workspace(clear_smiles_field=False)

        # Если после возможного сброса current_mol ещё нет — создаём его из SMILES
        if not self.current_mol:
            if not smi:
                QtWidgets.QMessageBox.warning(self, "Warning", "Please enter a SMILES string first.")
                return
            mol = Chem.MolFromSmiles(smi)
            if mol is None or mol.GetNumAtoms() == 0:
                QtWidgets.QMessageBox.warning(self, "Warning", "Invalid or empty SMILES. Please correct it.")
                return
            mol = Chem.AddHs(mol)
            try:
                AllChem.Compute2DCoords(mol)
            except Exception:
                pass
            self.current_mol = mol
            self.current_smiles = smi
            self._update_2d_view()

        # ---- Генерация 3D ----
        self.statusBar().showMessage("Generating 3D structure...")
        QtWidgets.QApplication.processEvents()

        try:
            opts = BuildOptions(
                engine=self.cb_engine.currentText(),
                ff=self.cb_ff.currentText(),
                num_confs=self.sb_confs.value(),
                seed=self.sb_seed.value(),
                prot_method=self.cb_prot_method.currentText(),
                ph=self.sb_ph.value(),
                gen_tautomers=self.cb_taut_gen.isChecked(),
                gen_microstates=self.cb_micro.isChecked(),
                micro_confs=self.sb_micro_confs.value(),
                micro_topk=self.sb_micro_topk.value(),
                micro_rmsd=self.sb_micro_rmsd.value(),
                tautomer_mode=self.cb_tautomer_mode.currentText() 
            )

            result = smiles_to_3d_sdf(self.current_smiles, opts)

            self._last_sdf = result.sdf
            self._states_smi = result.states_smi
            self._states_sdf = result.states_sdf
            self._state_idx = 0 if self._states_sdf else -1
           
            # --- NEW: собираем список "конформеров" и их энергий для вкладки Analysis ---
            self.conformers_list = []
            self.energies_list = []

            from rdkit.Chem import AllChem

            for sdf in (self._states_sdf or []):
                m = Chem.MolFromMolBlock(sdf, removeHs=False)
                if m is None:
                    continue

                # Добавим H, чтобы FF адекватно посчитал энергию
                try:
                    m = Chem.AddHs(m, addCoords=True)
                except Exception:
                    pass

                energy = float("nan")
                try:
                    ff = AllChem.UFFGetMoleculeForceField(m)
                    if ff is not None:
                        energy = ff.CalcEnergy()
                except Exception:
                    # если UFF не смог затипизировать (металлы и т.п.) — оставим NaN
                    pass

                self.conformers_list.append(sdf)
                self.energies_list.append(energy)
            # --- END NEW ---

            mol_3d = Chem.MolFromMolBlock(self._last_sdf, removeHs=False)
            if mol_3d is not None:
                mol_3d = Chem.AddHs(mol_3d, addCoords=True)
                self.current_mol = mol_3d

            self._update_3d_view()
            self._update_info_panel()
            self._update_statebar_visibility()
            self._update_statebar()

            self.tabs_main.setCurrentWidget(self.view3d_container)

            num_states = len(self._states_sdf) if self._states_sdf else 1
            self.statusBar().showMessage(f"3D structure generated with {num_states} conformational states")

        except BuildError as e:
            QtWidgets.QMessageBox.critical(self, "Build Error", str(e))
            self.statusBar().showMessage("3D generation failed")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Unexpected Error", f"3D generation failed:\n{str(e)}")
            self.statusBar().showMessage("Unexpected error during 3D generation")

    def on_new_job(self):
        if self.current_smiles or self._last_sdf:
            reply = QtWidgets.QMessageBox.question(
                self, "New Project",
                "This will clear the current work. Save project first?",
                QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel
            )

            if reply == QtWidgets.QMessageBox.Save:
                self.on_save_job()
            elif reply == QtWidgets.QMessageBox.Cancel:
                return

        # Полный сброс, SMILES-поле очищаем
        self._reset_workspace(clear_smiles_field=True)
        self.statusBar().showMessage("New project started")

    def on_job_selected(self, row):
        if row < 0 or row >= len(self.sessions):
            return

        job = self.sessions[row]
        self.ed_smiles.setText(job["smiles"])
        self.on_build_2d()

    def on_job_delete(self, item):
        row = self.jobs.row(item)
        if 0 <= row < len(self.sessions):
            del self.sessions[row]
            self._save_sessions()
            self._update_jobs_list()
            self.statusBar().showMessage("Project removed")

    def on_open_file(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Molecule File", "",
            "Molecule files (*.sdf *.mol *.mol2 *.smi *.smiles);;All files (*.*)"
        )

        if not fname:
            return

        if self.current_smiles or self._last_sdf:
            reply = QtWidgets.QMessageBox.question(
                self, "Open Molecule",
                "Save current project before opening a new molecule?",
                QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel
            )
            if reply == QtWidgets.QMessageBox.Save:
                self.on_save_job()
            elif reply == QtWidgets.QMessageBox.Cancel:
                return

        # Сбросить рабочее состояние, SMILES поле очистим —
        # новая молекула из файла его заполнит
        self._reset_workspace(clear_smiles_field=True)

        try:
            self.statusBar().showMessage(f"Loading {Path(fname).name}...")

            if fname.lower().endswith(('.smi', '.smiles')):
                with open(fname, 'r', encoding='utf-8') as f:
                    smi_line = f.readline().strip()
                    if smi_line:
                        smiles = smi_line.split()[0]
                        self.ed_smiles.setText(smiles)
                        self.on_build_2d()
            else:
                suppl = Chem.SDMolSupplier(fname)
                mol = None
                for m in suppl:
                    if m is not None:
                        mol = m
                        break

                if mol is None:
                    raise ValueError("No valid molecule found in file")

                smi = Chem.MolToSmiles(mol)
                self.ed_smiles.setText(smi)
                self.on_build_2d()

            self.statusBar().showMessage(f"Loaded {Path(fname).name}")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "File Error", f"Failed to open file:\n{str(e)}")
            self.statusBar().showMessage("Failed to load file")

    def on_job_selected(self, row):
        if row < 0 or row >= len(self.sessions):
            return

        job = self.sessions[row]
        self.ed_smiles.setText(job["smiles"])
        self.on_build_2d()

    def on_job_delete(self, item):
        row = self.jobs.row(item)
        if 0 <= row < len(self.sessions):
            del self.sessions[row]
            self._save_sessions()
            self._update_jobs_list()
            self.statusBar().showMessage("Project removed")

    def on_validate(self):
        if not self.current_mol:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please load a molecule first.")
            return

        try:
            result = validate(
                self.current_mol,
                bond_tol=self.sb_bondtol.value(),
                angle_tol=self.sb_angletol.value(),
                planarity_tol=self.sb_planarity.value()
            )

            if result.is_ok:
                QtWidgets.QMessageBox.information(
                    self, "Validation Results",
                    "\u2713 Structure validation passed\n\nGeometry is within specified tolerances."
                )
            else:
                warnings_text = "\n".join([f"\u2022 {w}" for w in result.warnings])
                QtWidgets.QMessageBox.warning(
                    self, "Validation Results",
                    f"\u26a0 Structure validation found issues:\n\n{warnings_text}\n\nConsider geometry optimization."
                )

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Validation Error", f"Validation failed:\n{str(e)}")

    def on_open_file(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Molecule File", "",
            "Molecule files (*.sdf *.mol *.mol2 *.smi *.smiles);;All files (*.*)"
        )

        if not fname:
            return

        if self.current_smiles or self._last_sdf:
            reply = QtWidgets.QMessageBox.question(
                self, "Open Molecule",
                "Save current project before opening a new molecule?",
                QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel
            )
            if reply == QtWidgets.QMessageBox.Save:
                self.on_save_job()
            elif reply == QtWidgets.QMessageBox.Cancel:
                return

        # Сбросить рабочее состояние, SMILES поле очистим —
        # новая молекула из файла его заполнит
            self._reset_workspace(clear_smiles_field=True)

        try:
            self.statusBar().showMessage(f"Loading {Path(fname).name}...")

            if fname.lower().endswith(('.smi', '.smiles')):
                with open(fname, 'r', encoding='utf-8') as f:
                    smi_line = f.readline().strip()
                    if smi_line:
                        smiles = smi_line.split()[0]
                        self.ed_smiles.setText(smiles)
                        self.on_build_2d()
            else:
                suppl = Chem.SDMolSupplier(fname)
                mol = None
                for m in suppl:
                    if m is not None:
                        mol = m
                        break

                if mol is None:
                    raise ValueError("No valid molecule found in file")

                smi = Chem.MolToSmiles(mol)
                self.ed_smiles.setText(smi)
                self.on_build_2d()

            self.statusBar().showMessage(f"Loaded {Path(fname).name}")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "File Error", f"Failed to open file:\n{str(e)}")
            self.statusBar().showMessage("Failed to load file")

    def on_save_2d(self):
        if self._last_png is None and self._last_svg is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "No 2D image available to save.")
            return

        fname, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save 2D Image", f"{self.current_smiles[:20] if self.current_smiles else 'molecule'}_2d",
            "PNG Image (*.png);;SVG Image (*.svg);;All files (*.*)"
        )

        if not fname:
            return

        try:
            if selected_filter.startswith("PNG") or fname.lower().endswith('.png'):
                if self._last_png:
                    with open(fname, 'wb') as f:
                        f.write(self._last_png)
                else:
                    raise ValueError("PNG image not available")

            elif selected_filter.startswith("SVG") or fname.lower().endswith('.svg'):
                if self._last_svg:
                    with open(fname, 'w', encoding='utf-8') as f:
                        f.write(self._last_svg)
                else:
                    raise ValueError("SVG image not available")
            else:
                if self._last_png:
                    with open(fname + '.png', 'wb') as f:
                        f.write(self._last_png)
                else:
                    raise ValueError("No image data available")

            self.statusBar().showMessage(f"2D image saved to {Path(fname).name}")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save Error", f"Failed to save 2D image:\n{str(e)}")

    def on_save_3d(self):
        if not self._last_sdf:
            QtWidgets.QMessageBox.warning(self, "Warning", "No 3D structure available to save.")
            return

        fname, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save 3D Structure", f"{self.current_smiles[:20] if self.current_smiles else 'molecule'}_3d",
            "SDF files (*.sdf);;PDB files (*.pdb);;MOL2 files (*.mol2);;MOL files (*.mol);;All files (*.*)"
        )

        if not fname:
            return

        try:
            mol = Chem.MolFromMolBlock(self._last_sdf)
            if mol is None:
                raise ValueError("Invalid 3D structure data")

            if selected_filter.startswith("SDF") or fname.lower().endswith('.sdf'):
                with Chem.SDWriter(fname) as writer:
                    writer.write(mol)

            elif selected_filter.startswith("PDB") or fname.lower().endswith('.pdb'):
                pdb_block = Chem.MolToPDBBlock(mol)
                with open(fname, 'w', encoding='utf-8') as f:
                    f.write(pdb_block)

            elif selected_filter.startswith("MOL2") or fname.lower().endswith('.mol2'):
                mol2_block = Chem.MolToMol2Block(mol)
                with open(fname, 'w', encoding='utf-8') as f:
                    f.write(mol2_block)

            else:
                with Chem.SDWriter(fname + '.sdf') as writer:
                    writer.write(mol)

            self.statusBar().showMessage(f"3D structure saved to {Path(fname).name}")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save Error", f"Failed to save 3D structure:\n{str(e)}")

    def on_snap_3d(self):
        """Save a screenshot of the entire AZAT window."""
        try:
            # Capture full window instead of only the 3D viewer
            pixmap = self.grab()
            if pixmap.isNull():
                raise ValueError("Failed to capture screenshot")

            fname, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save Window Snapshot",
                f"{self.current_smiles[:20] if self.current_smiles else 'window'}_snapshot",
                "PNG Image (*.png);;All files (*.*)"
            )

            if fname and pixmap.save(fname):
                self.statusBar().showMessage(
                    f"Window snapshot saved to {Path(fname).name}"
                )
            elif fname:
                raise ValueError("Failed to save screenshot")

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Snapshot Error",
                f"Failed to save window screenshot:\n{e}"
            )

    def on_save_all_states(self):
        if not self._states_sdf:
            QtWidgets.QMessageBox.warning(self, "Warning", "No conformational states available to save.")
            return

        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save All States", f"{self.current_smiles[:20] if self.current_smiles else 'molecule'}_states",
            "ZIP Archive (*.zip);;All files (*.*)"
        )

        if not fname:
            return

        try:
            with zipfile.ZipFile(fname, 'w', zipfile.ZIP_DEFLATED) as zf:
                for i, sdf in enumerate(self._states_sdf):
                    zf.writestr(f'conformer_{i+1:03d}.sdf', sdf)

                if self._states_smi:
                    smiles_data = "\n".join([f"{smi}\t{i+1}" for i, smi in enumerate(self._states_smi)])
                    zf.writestr('tautomers.smi', smiles_data)

            self.statusBar().showMessage(f"All states saved to {Path(fname).name}")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Error", f"Failed to save states ZIP:\n{str(e)}")

    def on_save_job(self):
        if not self.current_smiles:
            QtWidgets.QMessageBox.warning(self, "Warning", "No current work to save.")
            return

        name, ok = QtWidgets.QInputDialog.getText(
            self, "Save Project", "Enter project name:",
            QtWidgets.QLineEdit.Normal, self.current_smiles[:30] + "..." if len(self.current_smiles) > 30 else self.current_smiles
        )

        if not ok or not name.strip():
            return

        existing_names = [job["name"] for job in self.sessions]
        if name.strip() in existing_names:
            reply = QtWidgets.QMessageBox.question(
                self, "Duplicate Name", "Project name already exists. Overwrite?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return
            self.sessions = [job for job in self.sessions if job["name"] != name.strip()]

        job = {
            "name": name.strip(),
            "smiles": self.current_smiles,
            "timestamp": QtCore.QDateTime.currentDateTime().toString()
        }

        self.sessions.append(job)
        self._save_sessions()
        self._update_jobs_list()
        self.statusBar().showMessage(f"Project '{name.strip()}' saved")

    def on_medchem(self):
        if not self.current_mol:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please load a molecule first.")
            return

        try:
            report = medchem_report_text(self.current_mol, lang="en")

            dialog = QtWidgets.QDialog(self)
            dialog.setWindowTitle("Drug-likeness Analysis")
            dialog.resize(700, 500)
            dialog.setModal(True)

            layout = QtWidgets.QVBoxLayout(dialog)

            text_edit = QtWidgets.QTextEdit()
            text_edit.setReadOnly(True)
            text_edit.setFont(QtGui.QFont("Consolas", 9))
            text_edit.setPlainText(report)
            layout.addWidget(text_edit)

            buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
            buttons.rejected.connect(dialog.accept)
            layout.addWidget(buttons)

            dialog.exec()

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Analysis Error", f"Drug-likeness analysis failed:\n{str(e)}")

    def _update_2d_view(self):
        if not self.current_mol:
            return

        try:
            if HAS_MOLDRAW:
                drawer = rdMolDraw2D.MolDraw2DSVG(800, 600)
                rdDepictor.Compute2DCoords(self.current_mol)
                drawer.DrawMolecule(self.current_mol)
                drawer.FinishDrawing()
                self._last_svg = drawer.GetDrawingText()

                img = Draw.MolToImage(self.current_mol, size=(800, 600))
                import io
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                self._last_png = buf.getvalue()

                html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>2D Structure</title>
                    <style>
                        body {{ margin: 0; padding: 20px; background: #121212; color: #eee; text-align: center; }}
                        .structure {{ max-width: 100%; height: auto; }}
                    </style>
                </head>
                <body>
                    <div class="structure">{self._last_svg if isinstance(self._last_svg, str) else ''}</div>
                </body>
                </html>
                """
                self.view2d_main.setHtml(html)

            else:
                img = Draw.MolToImage(self.current_mol, size=(800, 600))
                import io
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                data = b64encode(buf.getvalue()).decode('ascii')

                html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>2D Structure</title>
                    <style>
                        body {{ margin: 0; padding: 20px; background: #121212; color: #eee; text-align: center; }}
                        img {{ max-width: 100%; height: auto; }}
                    </style>
                </head>
                <body>
                    <img src="data:image/png;base64,{data}" alt="2D Structure" />
                </body>
                </html>
                """
                self.view2d_main.setHtml(html)
                self._last_png = buf.getvalue()
                self._last_svg = None

        except Exception as e:
            error_html = f"""
            <!DOCTYPE html>
            <html>
            <body style="padding: 50px; text-align: center; color: #dc3545;">
                <h3>2D Structure Generation Failed</h3>
                <p>Error: {str(e)}</p>
            </body>
            </html>
            """
            self.view2d_main.setHtml(error_html)
            self._last_png = None
            self._last_svg = None

    def _update_3d_view(self):
        sdf_data = self._last_sdf
        if self._state_idx >= 0 and self._state_idx < len(self._states_sdf):
            sdf_data = self._states_sdf[self._state_idx]

        if not sdf_data:
            default_html = """
            <!DOCTYPE html>
            <html>
            <body style="background: #222; color: #eee; text-align: center; padding: 50px; font-family: Arial, sans-serif;">
                <h3>Generate 3D Structure</h3>
                <p>Click "Generate 3D" to create a 3D molecular structure</p>
            </body>
            </html>
            """
            self.view3d_main.setHtml(default_html)
            return

        try:
            sdf_js = sdf_data.replace('\\', '\\\\').replace('`', '\\`').replace('${', '\\${')

            html_template = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>3D Molecular Viewer</title>
                <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
                <style>
                    body {{
                        margin: 0;
                        padding: 0;
                        background: #121212;
                        font-family: Arial, sans-serif;
                        overflow: hidden;
                        color: #eee;
                    }}
                    #container {{
                        width: 100%;
                        height: 100vh;
                        position: relative;
                    }}
                </style>
            </head>
            <body>
                <div id="container"></div>
                <script>
                    let sdfData = `{sdf_js}`;
                    if (typeof sdfData !== 'string' || sdfData.trim() === '') {{
                        document.body.innerHTML = '<h3 style="color:red; text-align:center; margin-top:50px;">Error: No valid molecular data to display.</h3>';
                    }} else {{
                        let viewer = $3Dmol.createViewer('container', {{
                            backgroundColor: '#121212'
                        }});
                        let model = viewer.addModel(sdfData, 'sdf');
                        viewer.setStyle({{}}, {{
                            stick: {{radius: 0.15, colorscheme: 'default'}}
                        }});
                        viewer.zoomTo();
                        viewer.render();
                        viewer.zoom(0.8, 1000);
                    }}
                </script>
            </body>
            </html>
            """

            self.view3d_main.setHtml(html_template)

        except Exception as e:
            error_html = f"""
            <!DOCTYPE html>
            <html>
            <body style="background: #222; color: #dc3545; text-align: center; padding: 50px; font-family: Arial, sans-serif;">
                <h3>3D Viewer Error</h3>
                <p>Error: {str(e)}</p>
            </body>
            </html>
            """
            self.view3d_main.setHtml(error_html)

    def _update_info_panel(self):
        smiles = self.ed_smiles.text().strip()
        total_states = len(self._states_sdf) if self._states_sdf else 0
        tautomers_count = len(self._states_smi) if self._states_smi else 0

        # микросостояния = все 3D-состояния минус базовые таутомеры
        micro_count = max(0, total_states - tautomers_count)

        self.info_panel.update_info(
            self.current_mol,
            smiles,
            states_count=micro_count,         # только микросостояния
            tautomers_count=tautomers_count,
            conformers=getattr(self, 'conformers_list', None),
            energies=getattr(self, 'energies_list', None),
        )
    def _load_sessions(self):
        try:
            if SESS.exists():
                with open(SESS, 'r', encoding='utf-8') as f:
                    self.sessions = json.load(f)
                self._update_jobs_list()
        except Exception as e:
            logger.warning(f"Failed to load sessions: {e}")
            self.sessions = []

    def _save_sessions(self):
        try:
            with open(SESS, 'w', encoding='utf-8') as f:
                json.dump(self.sessions, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save sessions: {e}")

    def _update_jobs_list(self):
        self.jobs.clear()
        for sess in self.sessions:
            item = QtWidgets.QListWidgetItem(sess["name"])
            item.setToolTip(f"SMILES: {sess['smiles']}")
            if "timestamp" in sess:
                item.setToolTip(f"SMILES: {sess['smiles']}\nSaved: {sess['timestamp']}")
            self.jobs.addItem(item)

    def closeEvent(self, event):
        if self.current_smiles or self._last_sdf:
            reply = QtWidgets.QMessageBox.question(
                self, "Exit Application",
                "Save current work before exiting?",
                QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel
            )

            if reply == QtWidgets.QMessageBox.Save:
                self.on_save_job()
            elif reply == QtWidgets.QMessageBox.Cancel:
                event.ignore()
                return

        event.accept()

    def on_about_azat(self):
        import platform
        python_version = platform.python_version()
        system = platform.system()
        release = platform.release()

        try:
            import rdkit
            rdkit_version = rdkit.__version__
        except Exception:
            rdkit_version = "Unknown"

        try:
            import PySide6
            pyside_version = PySide6.__version__
        except Exception:
            pyside_version = "Unknown"

        about_text = f"""
Automated Molecular Analysis Toolkit

This is a specialized desktop application for building, visualizing, and analyzing molecular structures based on SMILES data.
Designed for chem-pharma research, 3D conformer generation, drug-likeness evaluation, and geometry validation.

System Information:
- OS: {system} {release}
- Python: {python_version}
- RDKit: {rdkit_version}
- PySide6: {pyside_version}

Additional modules included:
- medchem
- ph4
- prototaut
- smiles_to_3d
- stategen
- validate

© AZAT Team 2024
        """

        dlg = QtWidgets.QMessageBox(self)
        dlg.setWindowTitle("About AZAT")
        dlg.setText(about_text)
        dlg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        dlg.exec()


def _format_state_idx(idx):
    return f"{idx + 1}"

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setApplicationName("Automated Molecular Analysis Toolkit")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("AZAT Team")

    try:
        icon_path = rp("icon.ico")
        if Path(icon_path).exists():
            app.setWindowIcon(QtGui.QIcon(icon_path))
    except Exception:
        pass

    window = Main()
    window.show()

    screen = app.primaryScreen().availableGeometry()
    window.move((screen.width() - window.width()) // 2,
                (screen.height() - window.height()) // 2)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
