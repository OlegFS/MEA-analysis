import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton
import pyqtgraph as pg
from pyqtgraph import RectROI

class ScatterPlotSelection(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Scatter Plot - ROI Selection")
        self.setGeometry(100, 100, 600, 500)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout()
        self.central_widget.setLayout(layout)

        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        self.num_points = 50
        self.x_data = np.random.rand(self.num_points) * 10
        self.y_data = np.random.rand(self.num_points) * 10

        # Create scatter plot
        self.scatter = pg.ScatterPlotItem(size=10, brush=pg.mkBrush('k'))
        self.scatter.setData(self.x_data, self.y_data)
        self.plot_widget.addItem(self.scatter)

        # ROI Selection Tool (Rectangle)
        self.roi = RectROI([2, 2], [4, 4], pen=pg.mkPen('r'))
        self.plot_widget.addItem(self.roi)

        # Buttons
        self.toggle_roi_btn = QPushButton("Toggle ROI Selection")
        self.toggle_roi_btn.clicked.connect(self.toggle_roi)
        layout.addWidget(self.toggle_roi_btn)

        self.confirm_roi_btn = QPushButton("Confirm ROI Selection")
        self.confirm_roi_btn.clicked.connect(self.confirm_roi_selection)
        layout.addWidget(self.confirm_roi_btn)

        self.reset_btn = QPushButton("Reset Selection")
        self.reset_btn.clicked.connect(self.reset_selection)
        layout.addWidget(self.reset_btn)

        self.roi_active = True  # Start with ROI enabled
        self.selected_indices = np.zeros(self.num_points, dtype=bool)  # Keep track of selected points

    def toggle_roi(self):
        """Enable or disable ROI selection."""
        if self.roi_active:
            self.plot_widget.removeItem(self.roi)
            self.roi_active = False
        else:
            self.plot_widget.addItem(self.roi)
            self.roi_active = True

    def confirm_roi_selection(self):
        """Confirm points inside the ROI rectangle and add them to the selection."""
        if not self.roi_active:
            return

        roi_bounds = self.roi.parentBounds()
        x_min, x_max = roi_bounds.left(), roi_bounds.right()
        y_min, y_max = roi_bounds.top(), roi_bounds.bottom()

        new_selected = (self.x_data >= x_min) & (self.x_data <= x_max) & (self.y_data >= y_min) & (self.y_data <= y_max)
        self.selected_indices |= new_selected  # Add to existing selection

        brushes = [pg.mkBrush('r') if selected else pg.mkBrush('k') for selected in self.selected_indices]
        self.scatter.setBrush(brushes)
        # print(self.selected_indices)

    def reset_selection(self):
        """Reset all selections and restore default point color."""
        self.selected_indices[:] = False  # Clear all selections
        self.scatter.setBrush([pg.mkBrush('k')] * self.num_points)  # Reset all points to black

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ScatterPlotSelection()
    window.show()
    sys.exit(app.exec_())
