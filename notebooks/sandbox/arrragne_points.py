import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton
import pyqtgraph as pg
from pyqtgraph import GraphicsLayoutWidget

class InteractiveScatterPlot(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Interactive Scatter Plot")
        self.setGeometry(100, 100, 600, 500)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout()
        self.central_widget.setLayout(layout)

        # Plot Widget
        self.plot_widget = GraphicsLayoutWidget()
        layout.addWidget(self.plot_widget)
        self.plot = self.plot_widget.addPlot()

        # Generate random points
        self.num_points = 10
        self.x_data = np.random.rand(self.num_points) * 10
        self.y_data = np.random.rand(self.num_points) * 10

        # Create scatter plot
        self.scatter = pg.ScatterPlotItem(size=10, brush=pg.mkBrush('b'))
        self.scatter.setData(self.x_data, self.y_data)
        self.plot.addItem(self.scatter)

        # Create a GraphicsItemGroup to apply transformations
        self.group = pg.GraphItem()
        self.plot.addItem(self.group)
        self.group.setData(pos=np.column_stack((self.x_data, self.y_data)))

        # Transformations (Default: Disabled)
        self.transform_enabled = False
        self.transform = pg.QtGui.QTransform()

        # Button to Toggle Transformation Mode
        self.toggle_btn = QPushButton("Activate Transform Mode")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.clicked.connect(self.toggle_transform_mode)
        layout.addWidget(self.toggle_btn)

        # Mouse Drag Event
        self.plot.scene().sigMouseMoved.connect(self.on_mouse_drag)

        # Variables for transformation
        self.last_mouse_pos = None
        self.scale_factor = 1.0
        self.rotation_angle = 0.0

    def toggle_transform_mode(self):
        """Enables or disables transform mode."""
        self.transform_enabled = self.toggle_btn.isChecked()
        self.toggle_btn.setText("Deactivate Transform Mode" if self.transform_enabled else "Activate Transform Mode")

    def on_mouse_drag(self, event):
        """Handles movement, scaling, and rotation of points when transform mode is active."""
        if not self.transform_enabled:
            return

        pos = self.plot.getViewBox().mapSceneToView(event)

        if self.last_mouse_pos is None:
            self.last_mouse_pos = pos
            return

        dx = pos.x() - self.last_mouse_pos.x()
        dy = pos.y() - self.last_mouse_pos.y()

        # Move points
        self.x_data += dx
        self.y_data += dy

        # Apply scaling based on vertical movement
        self.scale_factor += dy * 0.01
        self.scale_factor = max(0.5, min(2.0, self.scale_factor))  # Limit scaling

        # Apply rotation based on horizontal movement
        self.rotation_angle += dx * 2  # Scale rotation sensitivity

        # Compute transformation matrix
        self.transform.reset()
        self.transform.translate(np.mean(self.x_data), np.mean(self.y_data))
        self.transform.rotate(self.rotation_angle)
        self.transform.scale(self.scale_factor, self.scale_factor)
        self.transform.translate(-np.mean(self.x_data), -np.mean(self.y_data))

        # Apply transformation
        transformed_pos = self.transform.map(np.column_stack((self.x_data, self.y_data)))
        self.scatter.setData(transformed_pos[:, 0], transformed_pos[:, 1])

        self.last_mouse_pos = pos

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InteractiveScatterPlot()
    window.show()
    sys.exit(app.exec_())
