from __future__ import annotations

import copy
import traceback
from pathlib import Path

import yaml
import numpy as np

from PySide6.QtCore import Qt, QObject, QThread, Signal, QTimer
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QFileDialog,
    QLabel,
    QPushButton,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QLineEdit,
    QTextEdit,
    QGroupBox,
    QFormLayout,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QMessageBox,
    QTabWidget,
    QScrollArea,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from core.scenarios import load_scenario
from core.geometry_builder import (
    build_anomalies_from_targets,
    anomaly_to_target_spec,
    raster_model_preview,
)
from solvers.dispatch import run_forward


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_DIR = PROJECT_ROOT / "scenarios"


class ForwardWorker(QObject):
    finished = Signal(str, object)
    failed = Signal(str)
    log = Signal(str)

    def __init__(self, method, scenario):
        super().__init__()
        self.method = method
        self.scenario = scenario

    def run(self):
        try:
            self.scenario["_progress_callback"] = lambda text: self.log.emit(str(text))
            result = run_forward(self.method, self.scenario)
            self.finished.emit(self.method, result)
        except Exception:
            self.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Forward Modelling GUI")
        self.resize(1750, 980)

        self.elevation_csv_path = ""
        self.dat_file_path = ""
        self.output_dir_path = str(PROJECT_ROOT / "outputs")

        self.target_specs = [self._default_target_spec()]
        self._loading_target_controls = False

        self._build_ui()
        self._connect_signals()
        self._populate_scenarios()
        self._refresh_layer_table()
        self._sync_target_selector()
        self._load_target_spec_to_controls(self.target_specs[0])
        self._refresh_position_options()
        self._refresh_parameter_panel()

    def _default_target_spec(self):
        return {
            "target_type": "Circle / sphere-like body",
            "position_mode": "Between layer 1 and 2",
            "horizontal_position": "Centre",
            "size": "Medium",
            "property_preset": "Air-filled void",
            "custom_x": 0.0,
            "custom_depth": 2.2,
            "radius": 0.8,
            "width": 1.6,
            "height": 1.0,
            "angle": 0.0,
            "resistivity": 100000.0,
            "conductivity": 0.00001,
            "vp": 340.0,
            "epsilon_r": 1.0,
            "susceptibility": 1e-6,
        }

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)

        self.splitter = QSplitter(Qt.Horizontal)
        splitter = self.splitter
        main_layout.addWidget(splitter)

        self.left_scroll = QScrollArea()
        self.left_scroll.setWidgetResizable(True)

        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        self.left_scroll.setWidget(self.left_panel)

        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)

        splitter.addWidget(self.left_scroll)
        splitter.addWidget(self.right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)

        self.method_group = QGroupBox("Method and scenario")
        self.method_layout = QFormLayout(self.method_group)

        self.method_combo = QComboBox()
        self.method_combo.addItems(["GPR", "Seismic acoustic",
                "Seismic elastic", "ERT", "EM 1D", "EM 2D"])

        self.scenario_combo = QComboBox()
        self.load_scenario_button = QPushButton("Load scenario into controls")

        self.method_layout.addRow("Method:", self.method_combo)
        self.method_layout.addRow("Template:", self.scenario_combo)
        self.method_layout.addRow("", self.load_scenario_button)

        self.left_layout.addWidget(self.method_group)

        self.files_group = QGroupBox("Input files")
        self.files_layout = QFormLayout(self.files_group)

        self.elevation_line = QLineEdit()
        self.elevation_line.setReadOnly(True)
        self.elevation_button = QPushButton("Load elevation CSV")

        elev_row = QHBoxLayout()
        elev_row.addWidget(self.elevation_line)
        elev_row.addWidget(self.elevation_button)

        self.dat_line = QLineEdit()
        self.dat_line.setReadOnly(True)
        self.dat_button = QPushButton("Load .dat / field file")

        dat_row = QHBoxLayout()
        dat_row.addWidget(self.dat_line)
        dat_row.addWidget(self.dat_button)

        self.output_line = QLineEdit(self.output_dir_path)
        self.output_button = QPushButton("Select output folder")

        out_row = QHBoxLayout()
        out_row.addWidget(self.output_line)
        out_row.addWidget(self.output_button)

        self.files_layout.addRow("Elevation:", elev_row)
        self.files_layout.addRow("Data file:", dat_row)
        self.files_layout.addRow("Output:", out_row)

        self.left_layout.addWidget(self.files_group)

        self.common_group = QGroupBox("Common model")
        self.common_layout = QFormLayout(self.common_group)

        self.model_length = QDoubleSpinBox()
        self.model_length.setRange(1.0, 10000.0)
        self.model_length.setValue(8.0)
        self.model_length.setSuffix(" m")

        self.model_depth = QDoubleSpinBox()
        self.model_depth.setRange(1.0, 10000.0)
        self.model_depth.setValue(3.5)
        self.model_depth.setSuffix(" m")

        self.grid_dx = QDoubleSpinBox()
        self.grid_dx.setRange(0.01, 100.0)
        self.grid_dx.setValue(0.02)
        self.grid_dx.setSuffix(" m")

        self.noise = QDoubleSpinBox()
        self.noise.setRange(0.0, 100.0)
        self.noise.setValue(2.0)
        self.noise.setSuffix(" %")

        self.common_layout.addRow("Model length:", self.model_length)
        self.common_layout.addRow("Model depth:", self.model_depth)
        self.common_layout.addRow("Grid spacing:", self.grid_dx)
        self.common_layout.addRow("Noise:", self.noise)

        self.left_layout.addWidget(self.common_group)

        self.layer_group = QGroupBox("Layers")
        self.layer_layout = QVBoxLayout(self.layer_group)

        self.layer_count = QSpinBox()
        self.layer_count.setRange(1, 8)
        self.layer_count.setValue(3)

        self.layer_form = QFormLayout()

        self.layer_thickness_boxes = []
        self.layer_resistivity_boxes = []
        self.layer_conductivity_boxes = []
        self.layer_vp_boxes = []
        self.layer_epsilon_boxes = []
        self.layer_sus_boxes = []

        self.layer_layout.addWidget(QLabel("Number of layers:"))
        self.layer_layout.addWidget(self.layer_count)
        self.layer_layout.addLayout(self.layer_form)

        self.left_layout.addWidget(self.layer_group)

        self.target_group = QGroupBox("Geological target builder")
        self.target_layout = QVBoxLayout(self.target_group)

        self.target_count = QSpinBox()
        self.target_count.setRange(0, 12)
        self.target_count.setValue(1)

        self.target_selector = QComboBox()

        self.target_type_combo = QComboBox()
        self.target_type_combo.addItems(
            [
                "Circle / sphere-like body",
                "Ellipse / elongated lens",
                "Rectangle / block",
                "Tilted rectangle / dyke",
                "Buried channel",
                "Interface bump",
                "Tilted layer",
            ]
        )

        self.target_position_combo = QComboBox()
        self.target_horizontal_combo = QComboBox()
        self.target_horizontal_combo.addItems(
            [
                "Far left",
                "Left",
                "Centre",
                "Right",
                "Far right",
                "Custom",
            ]
        )

        self.target_size_combo = QComboBox()
        self.target_size_combo.addItems(["Small", "Medium", "Large", "Custom"])

        self.target_property_combo = QComboBox()
        self.target_property_combo.addItems(
            [
                "Conductive clay",
                "Resistive boulder",
                "Air-filled void",
                "Water-filled void",
                "Saturated sand",
                "Bedrock high velocity",
                "Custom",
            ]
        )

        self.custom_x_box = QDoubleSpinBox()
        self.custom_x_box.setRange(-10000.0, 10000.0)
        self.custom_x_box.setValue(0.0)
        self.custom_x_box.setSuffix(" m")

        self.custom_depth_box = QDoubleSpinBox()
        self.custom_depth_box.setRange(0.0, 10000.0)
        self.custom_depth_box.setValue(2.2)
        self.custom_depth_box.setSuffix(" m")

        self.target_radius_box = QDoubleSpinBox()
        self.target_radius_box.setRange(0.01, 10000.0)
        self.target_radius_box.setValue(0.8)
        self.target_radius_box.setSuffix(" m")

        self.target_width_box = QDoubleSpinBox()
        self.target_width_box.setRange(0.01, 10000.0)
        self.target_width_box.setValue(1.6)
        self.target_width_box.setSuffix(" m")

        self.target_height_box = QDoubleSpinBox()
        self.target_height_box.setRange(0.01, 10000.0)
        self.target_height_box.setValue(1.0)
        self.target_height_box.setSuffix(" m")

        self.target_angle_box = QDoubleSpinBox()
        self.target_angle_box.setRange(-89.0, 89.0)
        self.target_angle_box.setValue(0.0)
        self.target_angle_box.setSuffix(" °")

        self.target_resistivity_box = QDoubleSpinBox()
        self.target_resistivity_box.setRange(1e-6, 1e9)
        self.target_resistivity_box.setValue(100000.0)
        self.target_resistivity_box.setSuffix(" ohm m")

        self.target_conductivity_box = QDoubleSpinBox()
        self.target_conductivity_box.setRange(0.0, 1e6)
        self.target_conductivity_box.setDecimals(8)
        self.target_conductivity_box.setValue(0.00001)
        self.target_conductivity_box.setSuffix(" S/m")

        self.target_vp_box = QDoubleSpinBox()
        self.target_vp_box.setRange(1.0, 10000.0)
        self.target_vp_box.setValue(340.0)
        self.target_vp_box.setSuffix(" m/s")

        self.target_epsilon_box = QDoubleSpinBox()
        self.target_epsilon_box.setRange(1.0, 1000.0)
        self.target_epsilon_box.setValue(1.0)

        self.target_sus_box = QDoubleSpinBox()
        self.target_sus_box.setRange(0.0, 10.0)
        self.target_sus_box.setDecimals(8)
        self.target_sus_box.setValue(1e-6)

        self.apply_target_button = QPushButton("Apply target settings")
        self.preview_button = QPushButton("Preview geological model")

        self.target_summary = QTextEdit()
        self.target_summary.setReadOnly(True)
        self.target_summary.setMaximumHeight(110)

        self.target_form = QFormLayout()
        self.target_form.addRow("Number of targets:", self.target_count)
        self.target_form.addRow("Edit target:", self.target_selector)
        self.target_form.addRow("Target type:", self.target_type_combo)
        self.target_form.addRow("Vertical position:", self.target_position_combo)
        self.target_form.addRow("Horizontal position:", self.target_horizontal_combo)
        self.target_form.addRow("Size preset:", self.target_size_combo)
        self.target_form.addRow("Property preset:", self.target_property_combo)
        self.target_form.addRow("Custom x:", self.custom_x_box)
        self.target_form.addRow("Custom depth:", self.custom_depth_box)
        self.target_form.addRow("Radius:", self.target_radius_box)
        self.target_form.addRow("Width:", self.target_width_box)
        self.target_form.addRow("Height/thickness:", self.target_height_box)
        self.target_form.addRow("Tilt angle:", self.target_angle_box)
        self.target_form.addRow("Custom resistivity:", self.target_resistivity_box)
        self.target_form.addRow("Custom conductivity:", self.target_conductivity_box)
        self.target_form.addRow("Custom Vp:", self.target_vp_box)
        self.target_form.addRow("Custom epsilon_r:", self.target_epsilon_box)
        self.target_form.addRow("Custom susceptibility:", self.target_sus_box)

        self.target_layout.addLayout(self.target_form)
        self.target_layout.addWidget(self.apply_target_button)
        self.target_layout.addWidget(self.preview_button)
        self.target_layout.addWidget(QLabel("Generated targets:"))
        self.target_layout.addWidget(self.target_summary)

        self.left_layout.addWidget(self.target_group)

        self.method_specific_group = QGroupBox("Method-specific parameters")
        self.method_specific_layout = QFormLayout(self.method_specific_group)

        self.ert_array = QComboBox()
        self.ert_array.addItems(
            [
                "wenner-alpha",
                "wenner-beta",
                "dipole-dipole",
                "schlumberger",
                "gradient",
            ]
        )

        self.ert_electrodes = QSpinBox()
        self.ert_electrodes.setRange(4, 256)
        self.ert_electrodes.setValue(41)

        self.ert_spacing = QDoubleSpinBox()
        self.ert_spacing.setRange(0.1, 100.0)
        self.ert_spacing.setValue(2.0)
        self.ert_spacing.setSuffix(" m")

        self.ert_invert = QCheckBox("Run inversion after forward model")
        self.ert_invert.setChecked(False)

        self.ert_layers_follow_topography = QCheckBox("ERT layers follow topography")
        self.ert_layers_follow_topography.setChecked(False)

        self.seis_source_freq = QDoubleSpinBox()
        self.seis_source_freq.setRange(1.0, 500.0)
        self.seis_source_freq.setValue(25.0)
        self.seis_source_freq.setSuffix(" Hz")

        self.seis_receiver_spacing = QDoubleSpinBox()
        self.seis_receiver_spacing.setRange(0.1, 100.0)
        self.seis_receiver_spacing.setValue(2.0)
        self.seis_receiver_spacing.setSuffix(" m")

        self.seis_time_ms = QDoubleSpinBox()
        self.seis_time_ms.setRange(10.0, 10000.0)
        self.seis_time_ms.setValue(1000.0)
        self.seis_time_ms.setSuffix(" ms")

        self.seis_layers_follow_topography = QCheckBox("Layers follow topography")
        self.seis_layers_follow_topography.setChecked(False)

        self.seis_background_difference = QCheckBox("Run background model and difference gather")
        self.seis_background_difference.setChecked(True)

        self.seis_first_arrivals = QCheckBox("Show first-arrival / refraction view")
        self.seis_first_arrivals.setChecked(True)

        self.seis_save_wavefield = QCheckBox("Save wavefield snapshot")
        self.seis_save_wavefield.setChecked(False)
        self.seis_apply_agc = QCheckBox("Apply AGC display gain")
        self.seis_apply_agc.setChecked(True)

        self.seis_agc_window_ms = QDoubleSpinBox()
        self.seis_agc_window_ms.setRange(5.0, 500.0)
        self.seis_agc_window_ms.setValue(80.0)
        self.seis_agc_window_ms.setSuffix(" ms")

        self.seis_mute_direct = QCheckBox("Mute direct/early arrivals in display")
        self.seis_mute_direct.setChecked(True)

        self.seis_mute_velocity = QDoubleSpinBox()
        self.seis_mute_velocity.setRange(100.0, 10000.0)
        self.seis_mute_velocity.setValue(900.0)
        self.seis_mute_velocity.setSuffix(" m/s")

        self.seis_mute_padding_ms = QDoubleSpinBox()
        self.seis_mute_padding_ms.setRange(0.0, 300.0)
        self.seis_mute_padding_ms.setValue(40.0)
        self.seis_mute_padding_ms.setSuffix(" ms")

        self.seis_clip_percentile = QDoubleSpinBox()
        self.seis_clip_percentile.setRange(90.0, 100.0)
        self.seis_clip_percentile.setValue(99.0)
        self.seis_clip_percentile.setSuffix(" %")


        self.seis_acquisition_mode = QComboBox()
        self.seis_acquisition_mode.addItems(["single_shot", "multi_shot", "cmp"])

        self.seis_show_pick_overlay = QCheckBox("Show first-arrival / refraction picks")
        self.seis_show_pick_overlay.setChecked(True)

        self.seis_show_apparent_velocity = QCheckBox("Show apparent velocity plot")
        self.seis_show_apparent_velocity.setChecked(False)

        self.seis_show_first_arrival_difference = QCheckBox("Show first-arrival difference")
        self.seis_show_first_arrival_difference.setChecked(True)
        self.seis_show_stacked_anomaly_response = QCheckBox("Show stacked anomaly-response map")
        self.seis_show_stacked_anomaly_response.setChecked(True)

        self.seis_show_refraction_summary = QCheckBox("Show refraction summary")
        self.seis_show_refraction_summary.setChecked(False)

        self.seis_nshots = QSpinBox()
        self.seis_nshots.setRange(1, 101)
        self.seis_nshots.setValue(1)

        self.seis_shot_spacing = QDoubleSpinBox()
        self.seis_shot_spacing.setRange(0.1, 1000.0)
        self.seis_shot_spacing.setValue(5.0)
        self.seis_shot_spacing.setSuffix(" m")

        self.gpr_freq = QDoubleSpinBox()
        self.gpr_freq.setRange(10.0, 3000.0)
        self.gpr_freq.setValue(100.0)
        self.gpr_freq.setSuffix(" MHz")

        self.gpr_trace_spacing = QDoubleSpinBox()
        self.gpr_trace_spacing.setRange(0.01, 10.0)
        self.gpr_trace_spacing.setValue(0.50)
        self.gpr_trace_spacing.setSuffix(" m")

        self.gpr_time_window = QDoubleSpinBox()
        self.gpr_time_window.setRange(1.0, 1000.0)
        self.gpr_time_window.setValue(60.0)
        self.gpr_time_window.setSuffix(" ns")

        self.gpr_run_actual = QCheckBox("Run actual gprMax FDTD")
        self.gpr_run_actual.setChecked(True)

        self.gpr_background_difference = QCheckBox("Run background model and difference B-scan")
        self.gpr_background_difference.setChecked(False)

        self.gpr_layers_follow_topography = QCheckBox("GPR layers follow topography")
        self.gpr_layers_follow_topography.setChecked(False)

        self.gpr_component = QComboBox()
        self.gpr_component.addItems(["Ey", "Hx", "Hz", "Ex", "Ez", "Hy"])

        self.em_frequencies = QLineEdit("475, 1525, 5625, 16025, 63025")

        self.em_spacing = QDoubleSpinBox()
        self.em_spacing.setRange(0.1, 200.0)
        self.em_spacing.setValue(1.66)
        self.em_spacing.setSuffix(" m")

        self.em_height = QDoubleSpinBox()
        self.em_height.setRange(0.0, 50.0)
        self.em_height.setValue(0.0)
        self.em_height.setSuffix(" m")

        self.em_orientation = QComboBox()
        self.em_orientation.addItems(["ZZ", "XX", "YY", "ZX", "XZ", "ZY", "YZ"])

        self.em_run_target_equivalent = QCheckBox("Run target-equivalent comparison")
        self.em_run_target_equivalent.setChecked(False)

        self.em_show_sensitivity = QCheckBox("Show layer sensitivity")
        self.em_show_sensitivity.setChecked(False)

        self.em_sensitivity_parameter = QComboBox()
        self.em_sensitivity_parameter.addItems(["con", "sus", "perm"])

        self.em_layers_follow_topography = QCheckBox("EM layers follow topography")
        self.em_layers_follow_topography.setChecked(False)

        self.left_layout.addWidget(self.method_specific_group)

        self.run_button = QPushButton("Run Forward Model")
        self.left_layout.addWidget(self.run_button)

        self.tabs = QTabWidget()
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(170)

        self.right_layout.addWidget(self.tabs, stretch=8)
        self.right_layout.addWidget(self.log_box, stretch=1)

        self._reset_tabs()
        QTimer.singleShot(0, self._fit_left_panel_width)
        QTimer.singleShot(250, self._fit_left_panel_width)


    def _fit_left_panel_width(self):
        """Keep the left control panel readable after startup."""
        splitter = getattr(self, "splitter", None)
        if splitter is None:
            return

        self.left_scroll.setMinimumWidth(500)
        self.left_panel.setMinimumWidth(500)
        self.left_scroll.setMaximumWidth(760)

        total_width = max(self.width(), 1400)
        left_width = min(max(560, int(total_width * 0.32)), 720)
        right_width = max(700, total_width - left_width)

        splitter.setChildrenCollapsible(False)
        splitter.setCollapsible(0, False)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([left_width, right_width])

    def _connect_signals(self):
        self.method_combo.currentTextChanged.connect(self._refresh_parameter_panel)
        self.layer_count.valueChanged.connect(self._refresh_layer_table)
        self.layer_count.valueChanged.connect(self._refresh_position_options)
        self.elevation_button.clicked.connect(self._load_elevation_csv)
        self.dat_button.clicked.connect(self._load_dat_file)
        self.output_button.clicked.connect(self._select_output_dir)
        self.run_button.clicked.connect(self._run)
        self.load_scenario_button.clicked.connect(self._load_selected_scenario_into_controls)

        self.target_count.valueChanged.connect(self._target_count_changed)
        self.target_selector.currentIndexChanged.connect(self._target_selector_changed)
        self.apply_target_button.clicked.connect(self._apply_current_target)
        self.preview_button.clicked.connect(self._preview_geological_model)
        self.target_size_combo.currentTextChanged.connect(self._size_preset_changed)
        self.target_property_combo.currentTextChanged.connect(self._property_preset_changed)

    def _populate_scenarios(self):
        self.scenario_combo.clear()

        if not SCENARIO_DIR.exists():
            SCENARIO_DIR.mkdir(parents=True, exist_ok=True)

        for p in sorted(SCENARIO_DIR.glob("*.yaml")):
            self.scenario_combo.addItem(p.stem, str(p))

    def _refresh_layer_table(self):
        n = self.layer_count.value()

        defaults = [
            [2.0, 1000.0, 0.001, 600.0, 6.0, 1e-6],
            [999.0, 500.0, 0.002, 1200.0, 9.0, 1e-6],
            [999.0, 1000.0, 0.001, 1600.0, 7.0, 1e-6],
            [999.0, 500.0, 0.002, 2200.0, 8.0, 1e-6],
            [999.0, 800.0, 0.00125, 2500.0, 7.0, 1e-6],
        ]

        existing = self._layers_from_boxes_safe()

        while self.layer_form.rowCount():
            self.layer_form.removeRow(0)

        self.layer_thickness_boxes = []
        self.layer_resistivity_boxes = []
        self.layer_conductivity_boxes = []
        self.layer_vp_boxes = []
        self.layer_epsilon_boxes = []
        self.layer_sus_boxes = []

        for i in range(n):
            row = existing[i] if i < len(existing) else None
            d = defaults[min(i, len(defaults) - 1)]

            thickness = QDoubleSpinBox()
            thickness.setRange(0.01, 10000.0)
            thickness.setValue(float(row.get("thickness", d[0])) if row else d[0])
            thickness.setSuffix(" m")

            resistivity = QDoubleSpinBox()
            resistivity.setRange(1e-6, 1e9)
            resistivity.setValue(float(row.get("resistivity", d[1])) if row else d[1])
            resistivity.setSuffix(" ohm m")

            conductivity = QDoubleSpinBox()
            conductivity.setRange(0.0, 1e6)
            conductivity.setDecimals(8)
            conductivity.setValue(float(row.get("conductivity", d[2])) if row else d[2])
            conductivity.setSuffix(" S/m")

            vp = QDoubleSpinBox()
            vp.setRange(1.0, 10000.0)
            vp.setValue(float(row.get("vp", d[3])) if row else d[3])
            vp.setSuffix(" m/s")

            eps = QDoubleSpinBox()
            eps.setRange(1.0, 1000.0)
            eps.setValue(float(row.get("epsilon_r", d[4])) if row else d[4])

            sus = QDoubleSpinBox()
            sus.setRange(0.0, 10.0)
            sus.setDecimals(8)
            sus.setValue(float(row.get("susceptibility", d[5])) if row else d[5])

            self.layer_thickness_boxes.append(thickness)
            self.layer_resistivity_boxes.append(resistivity)
            self.layer_conductivity_boxes.append(conductivity)
            self.layer_vp_boxes.append(vp)
            self.layer_epsilon_boxes.append(eps)
            self.layer_sus_boxes.append(sus)

            layer_box = QWidget()
            layer_layout = QVBoxLayout(layer_box)
            layer_layout.setContentsMargins(0, 0, 0, 0)

            form = QFormLayout()
            form.addRow("Thickness:", thickness)
            form.addRow("Resistivity:", resistivity)
            form.addRow("Conductivity:", conductivity)
            form.addRow("Vp:", vp)
            form.addRow("epsilon_r:", eps)
            form.addRow("Susceptibility:", sus)

            layer_layout.addLayout(form)

            self.layer_form.addRow(f"Layer {i + 1}", layer_box)

    def _layers_from_boxes_safe(self):
        try:
            return self._layers_from_boxes()
        except Exception:
            return []

    def _layers_from_boxes(self):
        layers = []

        for i in range(len(self.layer_thickness_boxes)):
            resistivity = float(self.layer_resistivity_boxes[i].value())
            conductivity = float(self.layer_conductivity_boxes[i].value())

            if conductivity <= 0.0 and resistivity > 0.0:
                conductivity = 1.0 / resistivity

            vp = float(self.layer_vp_boxes[i].value())
            epsilon_r = float(self.layer_epsilon_boxes[i].value())
            susceptibility = float(self.layer_sus_boxes[i].value())

            layers.append(
                {
                    "thickness": float(self.layer_thickness_boxes[i].value()),
                    "resistivity": resistivity,
                    "conductivity": conductivity,
                    "vp": vp,
                    "velocity": vp,
                    "epsilon_r": epsilon_r,
                    "permittivity": epsilon_r,
                    "susceptibility": susceptibility,
                }
            )

        return layers

    def _refresh_position_options(self):
        current = self.target_position_combo.currentText()

        self.target_position_combo.blockSignals(True)
        self.target_position_combo.clear()

        n = self.layer_count.value()

        for i in range(1, n + 1):
            self.target_position_combo.addItem(f"In layer {i}")

        for i in range(1, n):
            self.target_position_combo.addItem(f"Between layer {i} and {i + 1}")

        self.target_position_combo.addItems(
            [
                "Very shallow",
                "Shallow",
                "Middle depth",
                "Deep",
                "Very deep",
                "Custom depth",
            ]
        )

        idx = self.target_position_combo.findText(current)
        if idx >= 0:
            self.target_position_combo.setCurrentIndex(idx)
        else:
            self.target_position_combo.setCurrentIndex(min(1, self.target_position_combo.count() - 1))

        self.target_position_combo.blockSignals(False)

    def _sync_target_selector(self):
        self.target_selector.blockSignals(True)
        self.target_selector.clear()

        for i in range(len(self.target_specs)):
            self.target_selector.addItem(f"Target {i + 1}")

        self.target_selector.blockSignals(False)
        self._update_target_summary()

    def _target_count_changed(self):
        new_n = self.target_count.value()
        old_n = len(self.target_specs)

        if new_n > old_n:
            for _ in range(new_n - old_n):
                self.target_specs.append(self._default_target_spec())

        elif new_n < old_n:
            self.target_specs = self.target_specs[:new_n]

        self._sync_target_selector()

        if self.target_specs:
            idx = min(self.target_selector.currentIndex(), len(self.target_specs) - 1)
            self.target_selector.setCurrentIndex(idx)
            self._load_target_spec_to_controls(self.target_specs[idx])

        self._update_target_summary()

    def _target_selector_changed(self):
        if self._loading_target_controls:
            return

        idx = self.target_selector.currentIndex()
        if 0 <= idx < len(self.target_specs):
            self._load_target_spec_to_controls(self.target_specs[idx])

    def _apply_current_target(self):
        idx = self.target_selector.currentIndex()

        if idx < 0:
            return

        while idx >= len(self.target_specs):
            self.target_specs.append(self._default_target_spec())

        self.target_specs[idx] = self._target_spec_from_controls()
        self._update_target_summary()
        self._log(f"Applied settings for Target {idx + 1}")

    def _target_spec_from_controls(self):
        return {
            "target_type": self.target_type_combo.currentText(),
            "position_mode": self.target_position_combo.currentText(),
            "horizontal_position": self.target_horizontal_combo.currentText(),
            "size": self.target_size_combo.currentText(),
            "property_preset": self.target_property_combo.currentText(),
            "custom_x": self.custom_x_box.value(),
            "custom_depth": self.custom_depth_box.value(),
            "radius": self.target_radius_box.value(),
            "width": self.target_width_box.value(),
            "height": self.target_height_box.value(),
            "angle": self.target_angle_box.value(),
            "resistivity": self.target_resistivity_box.value(),
            "conductivity": self.target_conductivity_box.value(),
            "vp": self.target_vp_box.value(),
            "epsilon_r": self.target_epsilon_box.value(),
            "susceptibility": self.target_sus_box.value(),
        }

    def _load_target_spec_to_controls(self, spec):
        self._loading_target_controls = True

        self._set_combo_text(self.target_type_combo, spec.get("target_type", "Circle / sphere-like body"))
        self._set_combo_text(self.target_position_combo, spec.get("position_mode", "Between layer 1 and 2"))
        self._set_combo_text(self.target_horizontal_combo, spec.get("horizontal_position", "Centre"))
        self._set_combo_text(self.target_size_combo, spec.get("size", "Medium"))
        self._set_combo_text(self.target_property_combo, spec.get("property_preset", "Conductive clay"))

        self.custom_x_box.setValue(float(spec.get("custom_x", 0.0)))
        self.custom_depth_box.setValue(float(spec.get("custom_depth", 4.0)))
        self.target_radius_box.setValue(float(spec.get("radius", 2.0)))
        self.target_width_box.setValue(float(spec.get("width", 8.0)))
        self.target_height_box.setValue(float(spec.get("height", 2.0)))
        self.target_angle_box.setValue(float(spec.get("angle", 0.0)))
        self.target_resistivity_box.setValue(float(spec.get("resistivity", 20.0)))
        self.target_conductivity_box.setValue(float(spec.get("conductivity", 0.05)))
        self.target_vp_box.setValue(float(spec.get("vp", 1200.0)))
        self.target_epsilon_box.setValue(float(spec.get("epsilon_r", 25.0)))
        self.target_sus_box.setValue(float(spec.get("susceptibility", 1e-6)))

        self._loading_target_controls = False

    def _size_preset_changed(self, text):
        if self._loading_target_controls:
            return

        key = text.strip().lower()

        if key == "small":
            self.target_radius_box.setValue(1.0)
            self.target_width_box.setValue(3.0)
            self.target_height_box.setValue(1.0)

        elif key == "medium":
            self.target_radius_box.setValue(0.8)
            self.target_width_box.setValue(1.6)
            self.target_height_box.setValue(1.0)

        elif key == "large":
            self.target_radius_box.setValue(4.0)
            self.target_width_box.setValue(15.0)
            self.target_height_box.setValue(4.0)

    def _property_preset_changed(self, text):
        if self._loading_target_controls:
            return

        key = text.strip().lower()

        presets = {
            "conductive clay": [20.0, 0.05, 1200.0, 25.0, 1e-6],
            "resistive boulder": [2000.0, 0.0005, 3500.0, 5.0, 1e-5],
            "air-filled void": [1e5, 1e-5, 340.0, 1.0, 0.0],
            "water-filled void": [30.0, 0.033333, 1500.0, 80.0, 0.0],
            "saturated sand": [80.0, 0.0125, 1700.0, 25.0, 1e-6],
            "bedrock high velocity": [1000.0, 0.001, 3500.0, 6.0, 1e-5],
        }

        if key in presets:
            rho, sig, vp, eps, sus = presets[key]
            self.target_resistivity_box.setValue(rho)
            self.target_conductivity_box.setValue(sig)
            self.target_vp_box.setValue(vp)
            self.target_epsilon_box.setValue(eps)
            self.target_sus_box.setValue(sus)

    def _update_target_summary(self):
        if not hasattr(self, "target_summary"):
            return

        layers = self._layers_from_boxes_safe()
        domain = {
            "length": self.model_length.value() if hasattr(self, "model_length") else 80.0,
            "depth": self.model_depth.value() if hasattr(self, "model_depth") else 20.0,
        }

        try:
            anomalies = build_anomalies_from_targets(self.target_specs, layers, domain)
            text = yaml.safe_dump(anomalies, sort_keys=False)
        except Exception as exc:
            text = f"Could not build targets: {exc}"

        self.target_summary.setPlainText(text)

    def _detach_layout_item(self, item):
        if item is None:
            return

        widget = item.widget()
        if widget is not None:
            widget.setParent(None)
            return

        layout = item.layout()
        if layout is not None:
            while layout.count():
                child = layout.takeAt(0)
                self._detach_layout_item(child)

    def _clear_method_specific(self):
        while self.method_specific_layout.rowCount():
            row = self.method_specific_layout.takeRow(0)
            self._detach_layout_item(row.labelItem)
            self._detach_layout_item(row.fieldItem)

    def _refresh_parameter_panel(self):
        self._clear_method_specific()

        method = self.method_combo.currentText()

        if method == "ERT":
            self.method_specific_layout.addRow("Array type:", self.ert_array)
            self.method_specific_layout.addRow("Electrodes:", self.ert_electrodes)
            self.method_specific_layout.addRow("Electrode spacing:", self.ert_spacing)
            self.method_specific_layout.addRow("", self.ert_invert)
            self.method_specific_layout.addRow("", self.ert_layers_follow_topography)

        elif method in {"Seismic", "Seismic acoustic", "Seismic elastic"}:
            self.method_specific_layout.addRow("Source frequency:", self.seis_source_freq)
            self.method_specific_layout.addRow("Receiver spacing:", self.seis_receiver_spacing)
            self.method_specific_layout.addRow("Recording time:", self.seis_time_ms)
            self.method_specific_layout.addRow("Acquisition mode:", self.seis_acquisition_mode)
            self.method_specific_layout.addRow("Number of shots:", self.seis_nshots)
            self.method_specific_layout.addRow("Shot spacing:", self.seis_shot_spacing)
            self.method_specific_layout.addRow("", self.seis_layers_follow_topography)
            self.method_specific_layout.addRow("", self.seis_background_difference)
            self.method_specific_layout.addRow("", self.seis_show_pick_overlay)
            self.method_specific_layout.addRow("", self.seis_show_apparent_velocity)
            self.method_specific_layout.addRow("", self.seis_show_first_arrival_difference)
            self.method_specific_layout.addRow("", self.seis_show_stacked_anomaly_response)
            self.method_specific_layout.addRow("", self.seis_show_refraction_summary)
            self.method_specific_layout.addRow("", self.seis_save_wavefield)
            self.method_specific_layout.addRow("", self.seis_apply_agc)
            self.method_specific_layout.addRow("AGC window:", self.seis_agc_window_ms)
            self.method_specific_layout.addRow("", self.seis_mute_direct)
            self.method_specific_layout.addRow("Mute velocity:", self.seis_mute_velocity)
            self.method_specific_layout.addRow("Mute padding:", self.seis_mute_padding_ms)
            self.method_specific_layout.addRow("Display clip:", self.seis_clip_percentile)

        elif method == "GPR":
            self.method_specific_layout.addRow("Antenna frequency:", self.gpr_freq)
            self.method_specific_layout.addRow("Trace spacing:", self.gpr_trace_spacing)
            self.method_specific_layout.addRow("Time window:", self.gpr_time_window)
            self.method_specific_layout.addRow("Output component:", self.gpr_component)
            self.method_specific_layout.addRow("", self.gpr_run_actual)
            self.method_specific_layout.addRow("", self.gpr_background_difference)
            self.method_specific_layout.addRow("", self.gpr_layers_follow_topography)

        elif method in {"EM", "EM 1D", "EM 2D"}:
            self.method_specific_layout.addRow("Frequencies Hz:", self.em_frequencies)
            self.method_specific_layout.addRow("Coil spacing:", self.em_spacing)
            self.method_specific_layout.addRow("Sensor height:", self.em_height)
            self.method_specific_layout.addRow("Orientation:", self.em_orientation)
            self.method_specific_layout.addRow("", self.em_run_target_equivalent)
            self.method_specific_layout.addRow("", self.em_show_sensitivity)
            self.method_specific_layout.addRow("Sensitivity:", self.em_sensitivity_parameter)
            self.method_specific_layout.addRow("", self.em_layers_follow_topography)

    def _load_elevation_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select elevation CSV",
            str(PROJECT_ROOT),
            "CSV files (*.csv);;Text files (*.txt);;All files (*)",
        )

        if path:
            self.elevation_csv_path = path
            self.elevation_line.setText(path)
            self._log(f"Loaded elevation file: {path}")

    def _load_dat_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select data file",
            str(PROJECT_ROOT),
            "Data files (*.dat *.txt *.csv);;All files (*)",
        )

        if path:
            self.dat_file_path = path
            self.dat_line.setText(path)
            self._log(f"Loaded data file: {path}")

    def _select_output_dir(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Select output directory",
            str(PROJECT_ROOT),
        )

        if path:
            self.output_dir_path = path
            self.output_line.setText(path)
            self._log(f"Selected output folder: {path}")

    def _runtime_overrides(self):
        method = self.method_combo.currentText()
        layers = self._layers_from_boxes()

        domain = {
            "length": self.model_length.value(),
            "depth": self.model_depth.value(),
            "dx": self.grid_dx.value(),
            "dz": self.grid_dx.value(),
        }

        self._apply_current_target()

        anomalies = build_anomalies_from_targets(self.target_specs, layers, domain)

        overrides = {
            "domain": domain,
            "noise": {
                "relative_percent": self.noise.value(),
            },
            "layers": layers,
            "anomalies": anomalies,
            "target_specs": copy.deepcopy(self.target_specs),
            "files": {
                "elevation_csv": self.elevation_csv_path,
                "data_file": self.dat_file_path,
                "output_dir": self.output_line.text(),
            },
            "survey": {},
        }

        if method == "ERT":
            overrides["survey"]["ert"] = {
                "array": self.ert_array.currentText(),
                "scheme": self.ert_array.currentText(),
                "electrodes": self.ert_electrodes.value(),
                "spacing": self.ert_spacing.value(),
                "run_inversion": self.ert_invert.isChecked(),
                "topography_mode": ("parallel_to_topography" if self.ert_layers_follow_topography.isChecked() else "horizontal_interfaces"),
            }

        elif method in {"Seismic", "Seismic acoustic", "Seismic elastic"}:
            overrides["survey"]["seismic"] = {
                "source_frequency": self.seis_source_freq.value(),
                "receiver_spacing": self.seis_receiver_spacing.value(),
                "recording_time_ms": self.seis_time_ms.value(),
                "topography_mode": (
                    "parallel_to_topography"
                    if self.seis_layers_follow_topography.isChecked()
                    else "horizontal_interfaces"
                ),
                "run_background_difference": self.seis_background_difference.isChecked(),
                "show_first_arrivals": self.seis_first_arrivals.isChecked(),
                "save_wavefield_snapshot": self.seis_save_wavefield.isChecked(),
                "display_agc": self.seis_apply_agc.isChecked(),
                "display_agc_window_ms": self.seis_agc_window_ms.value(),
                "display_mute_direct": self.seis_mute_direct.isChecked(),
                "display_mute_velocity": self.seis_mute_velocity.value(),
                "display_mute_padding_ms": self.seis_mute_padding_ms.value(),
                "display_clip_percentile": self.seis_clip_percentile.value(),
                "acquisition_mode": self.seis_acquisition_mode.currentText(),
                "show_refraction_overlay": self.seis_show_pick_overlay.isChecked(),
                "show_apparent_velocity": self.seis_show_apparent_velocity.isChecked(),
                "show_first_arrival_difference": self.seis_show_first_arrival_difference.isChecked(),
                "show_stacked_anomaly_response": self.seis_show_stacked_anomaly_response.isChecked(),
                "show_refraction_summary": self.seis_show_refraction_summary.isChecked(),
                "nshots": self.seis_nshots.value(),
                "shot_spacing": self.seis_shot_spacing.value(),
            }

        elif method == "GPR":
            overrides["survey"]["gpr"] = {
                "frequency_mhz": self.gpr_freq.value(),
                "trace_spacing": self.gpr_trace_spacing.value(),
                "time_window_ns": self.gpr_time_window.value(),
                "component": self.gpr_component.currentText(),
                "run_actual_gprmax": self.gpr_run_actual.isChecked(),
                "run_background_difference": self.gpr_background_difference.isChecked(),
                "topography_mode": ("parallel_to_topography" if self.gpr_layers_follow_topography.isChecked() else "horizontal_interfaces"),
            }

        elif method in {"EM", "EM 1D", "EM 2D"}:
            freqs = []

            for token in self.em_frequencies.text().replace(";", ",").split(","):
                token = token.strip()
                if token:
                    freqs.append(float(token))

            overrides["survey"]["em"] = {
                "frequencies": freqs,
                "coil_spacing": self.em_spacing.value(),
                "height": self.em_height.value(),
                "orientation": self.em_orientation.currentText(),
                "run_target_equivalent": self.em_run_target_equivalent.isChecked(),
                "show_sensitivity": self.em_show_sensitivity.isChecked(),
                "sensitivity_parameter": self.em_sensitivity_parameter.currentText(),
                "topography_mode": ("parallel_to_topography" if self.em_layers_follow_topography.isChecked() else "horizontal_interfaces"),
            }

        return overrides

    def _deep_merge(self, base, update):
        for k, v in update.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                self._deep_merge(base[k], v)
            else:
                base[k] = v

        return base

    def _build_scenario(self):
        scenario_path = self.scenario_combo.currentData()

        if scenario_path:
            scenario = load_scenario(scenario_path)
        else:
            scenario = {}

        scenario = copy.deepcopy(scenario)
        scenario = self._deep_merge(scenario, self._runtime_overrides())

        return scenario

    def _load_selected_scenario_into_controls(self):
        scenario_path = self.scenario_combo.currentData()

        if not scenario_path:
            return

        try:
            scenario = load_scenario(scenario_path)

            domain = scenario.get("domain", {})
            if domain:
                self.model_length.setValue(float(domain.get("length", self.model_length.value())))
                self.model_depth.setValue(float(domain.get("depth", self.model_depth.value())))
                self.grid_dx.setValue(float(domain.get("dx", self.grid_dx.value())))

            noise = scenario.get("noise", {})
            if noise:
                self.noise.setValue(float(noise.get("relative_percent", self.noise.value())))

            layers = scenario.get("layers", [])
            if layers:
                self.layer_count.setValue(len(layers))
                self._refresh_layer_table()

                for i, layer in enumerate(layers):
                    self.layer_thickness_boxes[i].setValue(float(layer.get("thickness", 999.0)))
                    self.layer_resistivity_boxes[i].setValue(float(layer.get("resistivity", 100.0)))
                    self.layer_conductivity_boxes[i].setValue(float(layer.get("conductivity", 0.01)))
                    self.layer_vp_boxes[i].setValue(float(layer.get("vp", layer.get("velocity", 1000.0))))
                    self.layer_epsilon_boxes[i].setValue(float(layer.get("epsilon_r", layer.get("permittivity", 10.0))))
                    self.layer_sus_boxes[i].setValue(float(layer.get("susceptibility", 1e-6)))

            if "target_specs" in scenario:
                self.target_specs = copy.deepcopy(scenario["target_specs"])
            else:
                anomalies = scenario.get("anomalies", [])
                if anomalies:
                    self.target_specs = [anomaly_to_target_spec(a, i) for i, a in enumerate(anomalies)]
                else:
                    self.target_specs = [self._default_target_spec()]

            self.target_count.blockSignals(True)
            self.target_count.setValue(len(self.target_specs))
            self.target_count.blockSignals(False)

            self._sync_target_selector()

            if self.target_specs:
                self.target_selector.setCurrentIndex(0)
                self._load_target_spec_to_controls(self.target_specs[0])

            survey = scenario.get("survey", {})

            ert_s = survey.get("ert", {})
            if ert_s:
                self._set_combo_text(self.ert_array, ert_s.get("array", ert_s.get("scheme", "wenner-alpha")))
                self.ert_electrodes.setValue(int(ert_s.get("electrodes", ert_s.get("n_electrodes", 41))))
                self.ert_spacing.setValue(float(ert_s.get("spacing", ert_s.get("electrode_spacing", 2.0))))

            seismic_s = survey.get("seismic", {})
            if seismic_s:
                self.seis_source_freq.setValue(float(seismic_s.get("source_frequency", seismic_s.get("freq", 25.0))))
                self.seis_receiver_spacing.setValue(float(seismic_s.get("receiver_spacing", 2.0)))
                self.seis_time_ms.setValue(float(seismic_s.get("recording_time_ms", 1000.0)))
                mode = str(seismic_s.get("topography_mode", "horizontal_interfaces"))
                self.seis_layers_follow_topography.setChecked(mode == "parallel_to_topography")


            if seismic_s:
                if hasattr(self, "seis_show_pick_overlay"):
                    self.seis_show_pick_overlay.setChecked(bool(seismic_s.get("show_refraction_overlay", True)))
                if hasattr(self, "seis_show_apparent_velocity"):
                    self.seis_show_apparent_velocity.setChecked(bool(seismic_s.get("show_apparent_velocity", True)))
                if hasattr(self, "seis_show_first_arrival_difference"):
                    self.seis_show_first_arrival_difference.setChecked(bool(seismic_s.get("show_first_arrival_difference", True)))
                self.seis_show_stacked_anomaly_response.setChecked(bool(seismic_s.get("show_stacked_anomaly_response", True)))
                if hasattr(self, "seis_show_refraction_summary"):
                    self.seis_show_refraction_summary.setChecked(bool(seismic_s.get("show_refraction_summary", True)))

            gpr_s = survey.get("gpr", {})
            if gpr_s:
                self.gpr_freq.setValue(float(gpr_s.get("frequency_mhz", self.gpr_freq.value())))
                self.gpr_trace_spacing.setValue(float(gpr_s.get("trace_spacing", self.gpr_trace_spacing.value())))
                self.gpr_time_window.setValue(float(gpr_s.get("time_window_ns", self.gpr_time_window.value())))

                if hasattr(self, "gpr_component"):
                    self._set_combo_text(self.gpr_component, gpr_s.get("component", "Ey"))

                if hasattr(self, "gpr_run_actual"):
                    self.gpr_run_actual.setChecked(bool(gpr_s.get("run_actual_gprmax", True)))

                if hasattr(self, "gpr_background_difference"):
                    self.gpr_background_difference.setChecked(bool(gpr_s.get("run_background_difference", False)))

            em_s = survey.get("em", {})
            if em_s:
                freqs = em_s.get("frequencies", [475, 1525, 5625, 16025, 63025])
                self.em_frequencies.setText(", ".join(str(f) for f in freqs))
                self.em_spacing.setValue(float(em_s.get("coil_spacing", 1.66)))
                self.em_height.setValue(float(em_s.get("height", 0.0)))
                self._set_combo_text(self.em_orientation, em_s.get("orientation", "ZZ"))

            self._update_target_summary()
            self._log(f"Loaded scenario into controls: {scenario_path}")

        except Exception as exc:
            QMessageBox.critical(self, "Scenario load error", str(exc))
            self._log(traceback.format_exc())

    def _set_combo_text(self, combo, text):
        idx = combo.findText(str(text))

        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _preview_geological_model(self):
        try:
            self._apply_current_target()

            layers = self._layers_from_boxes()
            domain = {
                "length": self.model_length.value(),
                "depth": self.model_depth.value(),
                "dx": self.grid_dx.value(),
                "dz": self.grid_dx.value(),
            }

            anomalies = build_anomalies_from_targets(self.target_specs, layers, domain)

            resistivity_model, extent = raster_model_preview(
                layers=layers,
                anomalies=anomalies,
                domain=domain,
                property_name="resistivity",
            )

            velocity_model, _ = raster_model_preview(
                layers=layers,
                anomalies=anomalies,
                domain=domain,
                property_name="vp",
            )

            epsilon_model, _ = raster_model_preview(
                layers=layers,
                anomalies=anomalies,
                domain=domain,
                property_name="epsilon_r",
            )

            plots = {
                "Preview resistivity": {
                    "type": "image",
                    "array": np.log10(np.maximum(resistivity_model, 1e-12)),
                    "extent": extent,
                    "origin": "lower",
                    "title": "Preview resistivity model (log10 scale)",
                    "xlabel": "x [m]",
                    "ylabel": "z [m]",
                    "colorbar": True,
                    "clabel": "log10 resistivity [ohm m]",
                },
                "Preview velocity": {
                    "type": "image",
                    "array": velocity_model,
                    "extent": extent,
                    "origin": "lower",
                    "title": "Preview seismic velocity model",
                    "xlabel": "x [m]",
                    "ylabel": "z [m]",
                    "colorbar": True,
                },
                "Preview permittivity": {
                    "type": "image",
                    "array": epsilon_model,
                    "extent": extent,
                    "origin": "lower",
                    "title": "Preview GPR permittivity model",
                    "xlabel": "x [m]",
                    "ylabel": "z [m]",
                    "colorbar": True,
                },
            }

            self._plot_result(
                "Preview",
                {
                    "plots": plots,
                    "info": "Preview generated from target builder.",
                },
            )

            self._log("Preview generated.")

        except Exception as exc:
            self._log("PREVIEW ERROR:")
            self._log(str(exc))
            self._log(traceback.format_exc())
            QMessageBox.critical(self, "Preview error", str(exc))

    def _run(self):
        method = self.method_combo.currentText()

        try:
            scenario = self._build_scenario()

            self._log("=" * 70)
            self._log(f"Selected method: {method}")
            if method in {"Seismic acoustic", "Seismic elastic"}:
                self._log(f"Selected seismic solver: {method}")
            self._log(f"Selected scenario template: {self.scenario_combo.currentText()}")
            self._log("Runtime scenario from left-panel controls:")
            self._log(yaml.safe_dump(scenario, sort_keys=False))

            if method == "GPR":
                gpr_s = scenario.get("survey", {}).get("gpr", {})
                domain = scenario.get("domain", {})
                eps_values = []
                for layer in scenario.get("layers", []):
                    eps_values.append(float(layer.get("epsilon_r", layer.get("permittivity", 1.0))))
                for anomaly in scenario.get("anomalies", []):
                    eps_values.append(float(anomaly.get("epsilon_r", anomaly.get("permittivity", 1.0))))

                eps_max = max(eps_values) if eps_values else 1.0
                freq_mhz = float(gpr_s.get("frequency_mhz", 100.0))
                fmax_hz = 2.8 * freq_mhz * 1e6
                c0 = 299792458.0
                min_lambda = c0 / (fmax_hz * eps_max ** 0.5)
                dx = float(domain.get("dx", 0.05))
                dz = float(domain.get("dz", dx))
                rec_dx = min_lambda / 10.0

                self._log("GPR sampling diagnostics:")
                self._log(f"  frequency_mhz = {freq_mhz}")
                self._log(f"  estimated_fmax_mhz = {fmax_hz / 1e6:.3f}")
                self._log(f"  eps_max = {eps_max}")
                self._log(f"  dx = {dx}, dz = {dz}")
                self._log(f"  min_lambda_m = {min_lambda:.6g}")
                self._log(f"  cells_per_lambda_x = {min_lambda / dx:.3f}")
                self._log(f"  cells_per_lambda_z = {min_lambda / dz:.3f}")
                self._log(f"  recommended_dx_dz <= {rec_dx:.6g}")
                self._log(f"  trace_spacing = {gpr_s.get('trace_spacing')}")
                self._log(f"  time_window_ns = {gpr_s.get('time_window_ns')}")
                self._log(f"  run_actual_gprmax = {gpr_s.get('run_actual_gprmax')}")
                self._log(f"  run_background_difference = {gpr_s.get('run_background_difference')}")

            self._log("Starting forward model in background worker...")

            self.run_button.setEnabled(False)
            self.run_button.setText("Running...")

            self.worker_thread = QThread(self)
            self.worker = ForwardWorker(method, scenario)
            self.worker.moveToThread(self.worker_thread)

            self.worker_thread.started.connect(self.worker.run)
            self.worker.log.connect(self._log)
            self.worker.finished.connect(self._worker_finished)
            self.worker.failed.connect(self._worker_failed)

            self.worker.finished.connect(self.worker_thread.quit)
            self.worker.failed.connect(self.worker_thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.worker.failed.connect(self.worker.deleteLater)
            self.worker_thread.finished.connect(self.worker_thread.deleteLater)

            self.worker_thread.start()

        except Exception as exc:
            self.run_button.setEnabled(True)
            self.run_button.setText("Run Forward Model")
            self._log("ERROR:")
            self._log(str(exc))
            self._log(traceback.format_exc())
            QMessageBox.critical(self, "Forward modelling error", str(exc))

    def _worker_finished(self, method, result):
        self.run_button.setEnabled(True)
        self.run_button.setText("Run Forward Model")

        self._plot_result(method, result)

        info = result.get("info", "")
        if info:
            self._log(str(info))

        self._log("Run completed.")

    def _worker_failed(self, tb):
        self.run_button.setEnabled(True)
        self.run_button.setText("Run Forward Model")

        self._log("ERROR:")
        self._log(tb)

        last_line = tb.strip().splitlines()[-1] if tb.strip() else "Unknown error"
        QMessageBox.critical(self, "Forward modelling error", last_line)

    def _reset_tabs(self):
        self.tabs.clear()
        self._add_empty_tab("Model")
        self._add_empty_tab("Response / data")
        self._add_empty_tab("Inversion / extra")

    def _add_empty_tab(self, title):
        fig = Figure(figsize=(10, 7))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No result yet", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        self.tabs.addTab(canvas, title)

    def _plot_result(self, method, result):
        self.tabs.clear()

        plots = result.get("plots", None)

        if isinstance(plots, dict) and plots:
            for title, spec in plots.items():
                self._add_plot_tab(title, spec)
            return

        if result.get("model", None) is not None:
            self._add_plot_tab(
                "Model",
                {
                    "type": "image",
                    "array": result["model"],
                    "title": f"{method} model",
                    "xlabel": "x / index",
                    "ylabel": "z / index",
                    "colorbar": True,
                },
            )

        if result.get("data", None) is not None:
            self._add_plot_tab(
                "Data",
                {
                    "type": "line",
                    "array": result["data"],
                    "title": f"{method} data",
                    "xlabel": "index / x / frequency",
                    "ylabel": "response",
                },
            )

    def _add_plot_tab(self, title, spec):
        fig = Figure(figsize=(10, 7))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        kind = spec.get("type", "image")

        if kind == "image":
            arr = np.asarray(spec.get("array"))
            extent = spec.get("extent", None)

            im = ax.imshow(
                arr,
                aspect=spec.get("aspect", "auto"),
                origin=spec.get("origin", "upper"),
                interpolation=spec.get("interpolation", "nearest"),
                extent=extent,
                vmin=spec.get("vmin", None),
                vmax=spec.get("vmax", None),
            )

            if spec.get("colorbar", True):
                cbar = fig.colorbar(im, ax=ax)
                if "clabel" in spec:
                    cbar.set_label(spec["clabel"])

            ax.set_title(spec.get("title", title))
            ax.set_xlabel(spec.get("xlabel", ""))
            ax.set_ylabel(spec.get("ylabel", ""))

        elif kind == "line":
            arr = np.asarray(spec.get("array"))

            if arr.ndim == 1:
                ax.plot(arr)

            elif arr.ndim == 2 and arr.shape[1] >= 2:
                for i in range(1, arr.shape[1]):
                    ax.plot(
                        arr[:, 0],
                        arr[:, i],
                        marker=spec.get("marker", None),
                        label=spec.get("labels", {}).get(i, f"column {i}"),
                    )
                ax.legend()

            else:
                ax.plot(arr.ravel())

            ax.set_title(spec.get("title", title))
            ax.set_xlabel(spec.get("xlabel", ""))
            ax.set_ylabel(spec.get("ylabel", ""))

        elif kind == "scatter":
            x = np.asarray(spec.get("x"))
            y = np.asarray(spec.get("y"))
            c = np.asarray(spec.get("c"))

            sc = ax.scatter(
                x,
                y,
                c=c,
                s=spec.get("s", 25),
            )

            if spec.get("colorbar", True):
                cbar = fig.colorbar(sc, ax=ax)
                if "clabel" in spec:
                    cbar.set_label(spec["clabel"])

            ax.set_title(spec.get("title", title))
            ax.set_xlabel(spec.get("xlabel", ""))
            ax.set_ylabel(spec.get("ylabel", ""))

        elif kind == "ert_show":
            from pygimli.physics import ert

            data = spec["data"]

            ert.show(
                data,
                ax=ax,
                cMap=spec.get("cmap", "Spectral_r"),
                logScale=spec.get("logScale", True),
            )

            ax.set_title(spec.get("title", title))

        elif kind == "pg_show":
            import pygimli as pg

            mesh = spec["mesh"]
            values = spec["values"]

            pg.show(
                mesh,
                values,
                ax=ax,
                hold=True,
                cMap=spec.get("cmap", "Spectral_r"),
                logScale=spec.get("logScale", True),
                orientation=spec.get("orientation", "vertical"),
                label=spec.get("label", pg.unit("res")),
                cMin=spec.get("cMin", None),
                cMax=spec.get("cMax", None),
            )

            ax.set_title(spec.get("title", title))
            ax.set_xlabel(spec.get("xlabel", "x [m]"))
            ax.set_ylabel(spec.get("ylabel", "z [m]"))

            if "xlim" in spec:
                ax.set_xlim(*spec["xlim"])

            if "ylim" in spec:
                ax.set_ylim(*spec["ylim"])

        elif kind == "text":
            ax.set_axis_off()
            text = str(spec.get("text", ""))
            ax.text(
                0.01,
                0.99,
                text,
                ha="left",
                va="top",
                family="monospace",
                fontsize=9,
                transform=ax.transAxes,
            )
            ax.set_title(spec.get("title", title))

        else:
            ax.text(
                0.5,
                0.5,
                f"Unknown plot type: {kind}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        if spec.get("invert_yaxis", False):
            ax.invert_yaxis()

        fig.tight_layout()
        self.tabs.addTab(canvas, title)

    def _log(self, text):
        self.log_box.append(str(text))