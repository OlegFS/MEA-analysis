import sys
import importlib.util
import inspect
import numpy as np
import h5py
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QLabel, QDialog, QFormLayout,
    QLineEdit, QMainWindow, QMessageBox, QSplitter, QGridLayout,
    QScrollArea, QSizePolicy
)
from PyQt5.QtCore import Qt


def na(x):
    return np.array(x)


def electrode_label(row, col):
    letters = ['A','B','C','D','E','F','G','H','J','K','L','M','N','O','P','R']
    return f"{letters[col]}{row+1}"


class ParameterDialog(QDialog):
    def __init__(self, parameters):
        super().__init__()
        self.setWindowTitle("Function Parameters")
        self.layout = QFormLayout(self)
        self.fields = {}

        for name, value in parameters.items():
            line_edit = QLineEdit(str(value))
            self.layout.addRow(QLabel(name), line_edit)
            self.fields[name] = line_edit

        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        self.layout.addRow(ok_button)

    def get_parameters(self):
        return {name: self._parse_value(field.text()) for name, field in self.fields.items()}

    def _parse_value(self, value):
        try:
            return eval(value)
        except:
            return value


class TimeSeriesViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Event Detection Validator")
        self.setGeometry(100, 100, 1400, 800)

        self.signal = None
        self.full_data = None
        self.events = []
        self.event_func = None
        self.func_params = {}
        self.subsampling_rate = 1

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setBackground('w')
        self.plot_item = self.plot_widget.getPlotItem()

        load_button = QPushButton("Load Time Series")
        load_button.clicked.connect(self.load_timeseries)

        script_button = QPushButton("Select Detection Script")
        script_button.clicked.connect(self.load_script)

        settings_button = QPushButton("Settings")
        settings_button.clicked.connect(self.configure_parameters)

        run_button = QPushButton("Run Detection")
        run_button.clicked.connect(self.run_detection)

        save_button = QPushButton("Save Events")
        save_button.clicked.connect(self.save_events)

        button_layout = QHBoxLayout()
        for btn in [load_button, script_button, settings_button, run_button, save_button]:
            button_layout.addWidget(btn)

        main_plot_layout = QVBoxLayout()
        main_plot_layout.addLayout(button_layout)
        main_plot_layout.addWidget(self.plot_widget)

        # Electrode Grid
        grid_container = QWidget()
        self.grid_layout = QGridLayout()
        self.electrode_buttons = {}

        for i in range(16):
            for j in range(16):
                label = electrode_label(i, j)
                btn = QPushButton(label)
                btn.setFixedSize(30, 30)
                btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
                btn.setStyleSheet("background-color: white")
                btn.clicked.connect(lambda _, ch=(i, j): self.load_channel(ch))
                self.electrode_buttons[label] = btn
                self.grid_layout.addWidget(btn, i, j)

        grid_container.setLayout(self.grid_layout)
        grid_container.setMinimumSize(600, 600)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(grid_container)
        scroll_area.setMinimumWidth(350)

        # Splitter
        splitter = QSplitter(Qt.Horizontal)
        left_widget = QWidget()
        left_widget.setLayout(main_plot_layout)

        splitter.addWidget(left_widget)
        splitter.addWidget(scroll_area)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)

    def load_timeseries(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Time Series", "", "h5 Files (*.h5)")
        if file_path:
            with h5py.File(file_path, 'r') as f:
                self.subsampling_rate = float(f['resampled_rate'][()])
                adc_step = float(f['scale'][()])
                mv_scale_factor = 1_000_000
                self.full_data = f['data'][:, :] * adc_step * mv_scale_factor
                self.signal = self.full_data[:, 163]  # default channel
            self.plot_signal()

    def load_channel(self, ch):
        row, col = ch
        ch_index = row * 16 + col
        if self.full_data is not None and ch_index < self.full_data.shape[1]:
            self.signal = self.full_data[:, ch_index]
            self.plot_signal()

    def load_script(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Python Script", "", "Python Files (*.py)")
        if file_path:
            spec = importlib.util.spec_from_file_location("event_script", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            funcs = [obj for name, obj in inspect.getmembers(module) if inspect.isfunction(obj)]
            if funcs:
                self.event_func = funcs[0]
                sig = inspect.signature(self.event_func)
                self.func_params = {k: v.default for k, v in sig.parameters.items()
                                    if v.default is not inspect.Parameter.empty}

    def configure_parameters(self):
        if self.event_func is None:
            QMessageBox.warning(self, "No Script", "Please load a detection script first.")
            return

        dialog = ParameterDialog(self.func_params)
        if dialog.exec_():
            self.func_params = dialog.get_parameters()

    def run_detection(self):
        if self.signal is None or self.event_func is None:
            QMessageBox.warning(self, "Missing Data", "Load both time series and detection function.")
            return

        try:
            result = self.event_func(self.signal, **self.func_params)
            self.events = result[1]
            self.update_electrode_counts()
            self.plot_signal()
        except Exception as e:
            QMessageBox.critical(self, "Error Running Detection", str(e))

    def plot_signal(self):
        self.plot_item.clear()
        if self.signal is None:
            return

        dt = self.func_params.get('dt', 1.0 / self.subsampling_rate)
        x = np.arange(len(self.signal)) * dt
        self.plot_item.plot(x, self.signal, pen=pg.mkPen('k'), name='Signal')

        for start, end in self.events:
            region = pg.LinearRegionItem(values=(start, end), brush=(255, 0, 0, 80))
            region.setZValue(-10)
            region.setMovable(False)
            self.plot_item.addItem(region)

    def update_electrode_counts(self):
        if not isinstance(self.full_data, np.ndarray) or self.event_func is None:
            return

        for i in range(16):
            for j in range(16):
                ch_index = i * 16 + j
                if ch_index < self.full_data.shape[1]:
                    sig = self.full_data[:, ch_index]
                    try:
                        result = self.event_func(sig, **self.func_params)
                        event_count = len(result[1])
                        color = f"rgb({255 - min(event_count, 20)*10}, {255}, {255 - min(event_count, 20)*10})"
                        self.electrode_buttons[electrode_label(i, j)].setStyleSheet(f"background-color: {color}")
                    except:
                        self.electrode_buttons[electrode_label(i, j)].setStyleSheet("background-color: lightgray")

    def save_events(self):
        if not self.events:
            QMessageBox.warning(self, "No Events", "No events to save.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Events", "events.npy", "NumPy Files (*.npy)")
        if file_path:
            np.save(file_path, np.array(self.events))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = TimeSeriesViewer()
    viewer.show()
    sys.exit(app.exec_())