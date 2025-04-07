import sys
import os
import random
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, 
                             QFileDialog, QListWidget, QHBoxLayout, QLineEdit)
from PyQt5.QtGui import QImage
from PyQt5.QtCore import Qt
import cv2
import numpy as np
from pyqtgraph import ImageView, ScatterPlotItem
import pyqtgraph as pg
from pyqtgraph import RectROI

na = np.array 

def coregister_electrodes(points,tempate_points,template_e_position):
    matrix = cv2.getAffineTransform(na(tempate_points,dtype=np.float32), na(points,dtype=np.float32))
    e_position = na(template_e_position['pos'])
    mapped_e_position = np.dot(matrix, np.vstack((e_position.T, np.ones((1, e_position.shape[0])))))
    return mapped_e_position

import pickle 

class ImageViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.coregister =True
        
        # self.template_coordinates = np.array([[ 780.97939785,  466.21565591],
        #                                     [3695.36873118,  595.48292473],
        #                                     [ 651.71212903, 2793.02649462]])

        # Paramters 
        self.template_coordinates = np.array([(756.5106670552236, 479.26966703280675), (3680.59153281476, 618.9704003228011), (640.8962670910904, 2781.9231329851264)])
        with open('data/electrodes_template2.pickle', 'rb') as handle:
            b = pickle.load(handle,encoding='latin1')
        self.num_points = 252#standart MEA
        self.selected_indices = np.zeros(self.num_points, dtype=bool)  # Keep track of selected points
        self.tempate_electrodes =b
        self.select_mode = False  # Flag to track selection mode
        self.roi_active = False  # Start with ROI enabled 

        self.saved_labels = []
        self.saved_position = []
        self.saved_rgn = []


        self.setWindowTitle("PyQt Image Viewer")
        self.setGeometry(100, 100, 800, 500)
        
        self.image_view = ImageView()
        self.scatter_plot = ScatterPlotItem()
        self.image_view.getView().addItem(self.scatter_plot)
        # self.image_view.getView().scene().sigMouseClicked.connect(self.mouse_clicked)

        # if self.coregister==True:
        self.image_view.getView().scene().sigMouseClicked.connect(self.get_click_position)


        self.open_list_btn = QPushButton("Open list")
        self.open_list_btn.clicked.connect(self.open_list)
        
        self.list_btn = QPushButton("List")
        self.list_btn.clicked.connect(self.show_list)
        
        self.coregister_btn = QPushButton("Coregister")
        self.coregister_btn.setCheckable(True)  # Makes it toggleable
        self.coregister_btn.clicked.connect(self.toggle_selection)

        # self.coregister_btn.clicked.connect(self.add_coregister_point)
        
        self.show_electrodes_btn = QPushButton("Show Electrodes")
        self.show_electrodes_btn.clicked.connect(self.display_electrodes)

        self.show_label_btn = QPushButton("Show Labels")
        self.show_label_btn.clicked.connect(self.show_labels)
        
        self.prev_btn = QPushButton("Prev")
        self.prev_btn.clicked.connect(self.load_prev)
        
        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.load_next)
        
        self.roi = RectROI([2, 2], [4, 4], pen=pg.mkPen('r'))
        # self.image_view.addItem(self.roi) 

        self.select_points_btn = QPushButton("Select points")
        self.select_points_btn.clicked.connect(self.toggle_roi)

        self.add_points_btn = QPushButton("Add points")
        self.add_points_btn.clicked.connect(self.confirm_roi_selection)


        self.remove_points_btn = QPushButton("Remove points")
        self.remove_points_btn.clicked.connect(self.remove_points)

        self.reset_points_btn = QPushButton("Reset selection")
        self.reset_points_btn.clicked.connect(self.reset_selection)

        self.enter_name_field = QLineEdit()
        self.enter_name_field.setPlaceholderText("Enter region name here")

        self.add_region_btn = QPushButton("Add region")
        self.add_region_btn.clicked.connect(self.add_region)

        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.save_data)
        
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.open_list_btn)
        left_layout.addWidget(self.list_btn)
        left_layout.addWidget(self.coregister_btn)
        left_layout.addWidget(self.show_electrodes_btn)
        left_layout.addWidget(self.show_label_btn)
        left_layout.addWidget(self.prev_btn)
        left_layout.addWidget(self.next_btn)
        left_layout.addStretch()
        
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.select_points_btn)
        right_layout.addWidget(self.add_points_btn)
        right_layout.addWidget(self.remove_points_btn)
        right_layout.addWidget(self.reset_points_btn)
        right_layout.addWidget(self.enter_name_field)
        right_layout.addWidget(self.add_region_btn)
        right_layout.addWidget(self.save_btn)
        right_layout.addStretch()
        
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addWidget(self.image_view, stretch=1)
        main_layout.addLayout(right_layout)
        
        self.setLayout(main_layout)
        
        self.file_list = []
        self.current_index = -1
        self.points = []
        self.data = pd.DataFrame(columns=["X", "Y", "Label", "Filename"])
    
    def open_list(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File List", "", "Text Files (*.txt)")
        if file_path:
            with open(file_path, 'r') as f:
                self.file_list = [line.strip() for line in f.readlines()]
            if self.file_list:
                self.current_index = 0
                self.load_image()
    
    def show_list(self):
        self.list_window = QWidget()
        self.list_window.setWindowTitle("File List")
        self.list_window.setGeometry(200, 200, 300, 400)
        layout = QVBoxLayout()
        
        self.list_widget = QListWidget()
        for file in self.file_list:
            self.list_widget.addItem(file)
        layout.addWidget(self.list_widget)
        
        self.list_window.setLayout(layout)
        self.list_window.show()
    
    def load_image(self):

        # reset saved 
        self.saved_labels = []
        self.saved_position = []
        self.saved_rgn = []


        if self.current_index < 0 or self.current_index >= len(self.file_list):
            return
        file_path = self.file_list[self.current_index]
        print('file',file_path)
        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.image_view.setImage(image.transpose(1, 0, 2))
    
    def load_prev(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_image()

        
    
    def load_next(self):
        if self.current_index < len(self.file_list) - 1:
            self.current_index += 1
            self.load_image()
    # def mouse_clicked(self,event):
    #     pos = self.image_view.getView().mapSceneToView(event.scenePos())


    def get_click_position(self, event):
        if not self.select_mode:  # Only process clicks when selection mode is active
            return


        if len(self.points) >= 3:
            self.coregister =True
            self.points = []
            self.scatter_plot.clear()
            # return
        else:
            pos = self.image_view.getView().mapSceneToView(event.scenePos())
            # if 0 <= x < self.image_data.shape[1] and 0 <= y < self.image_data.shape[0]:
            #     print(f"Selected position: ({pos[0]}, {pos[1]})")
            # else:
            #     print("Clicked outside the image bounds")
            self.points.append((pos.x(), pos.y()))
            print(self.points)
            spots = [{'pos': point, 'size': 5, 'brush': 'r','symbol':'+'} for point in self.points]
            self.scatter_plot.setData(spots)
            if len(self.points)==3:
                self.coregister =False
        

    def toggle_selection(self):
        self.select_mode = self.coregister_btn.isChecked()
        print(f"Selection Mode: {'ON' if self.select_mode else 'OFF'}")



    def add_coregister_point(self, event):
        # self.coregister=True
        if len(self.points) >= 3:
            self.coregister =True
            self.points = []
            self.scatter_plot.clear()
            # return
        else:
            pos = self.image_view.getView().mapSceneToView(event.scenePos())
            self.points.append((pos.x(), pos.y()))
            print(self.points)
            spots = [{'pos': point, 'size': 5, 'brush': 'r','symbol':'+'} for point in self.points]
            self.scatter_plot.setData(spots)
            if len(self.points)==3:
                self.coregister =False
    
        
    
    def display_electrodes(self):
        print("Displaying electrodes at:")
        # Function implementation to be added separately
        self.electrode_postitions = na(coregister_electrodes(self.points,self.template_coordinates, self.tempate_electrodes)).T
        print(self.electrode_postitions)
        spots = [{'pos': point, 'size': 5, 'brush': 'b'} for point in self.electrode_postitions]
        self.scatter_plot.setData(spots)

    def show_labels(self):

        # Add text labels
        for i in range(self.num_points):
            text = pg.TextItem(self.tempate_electrodes['e_labels'][i], anchor=(0.5, +0.5), color='r')
            text.setPos(self.electrode_postitions[i,0],self.electrode_postitions[i,1])
            self.image_view.addItem(text)


    def toggle_roi(self):
        """Enable or disable ROI selection."""
        if self.roi_active:
            self.image_view.removeItem(self.roi)
            self.roi_active = False
        else:
            self.image_view.addItem(self.roi)
            self.roi_active = True

    def confirm_roi_selection(self):
        """Confirm points inside the ROI rectangle and add them to the selection."""
        if not self.roi_active:
            return

        roi_bounds = self.roi.parentBounds()
        x_min, x_max = roi_bounds.left(), roi_bounds.right()
        y_min, y_max = roi_bounds.top(), roi_bounds.bottom()
        self.x_data = self.electrode_postitions[:,0]
        self.y_data = self.electrode_postitions[:,1]

        new_selected = (self.x_data >= x_min) & (self.x_data <= x_max) & (self.y_data >= y_min) & (self.y_data <= y_max)
        self.selected_indices |= new_selected  # Add to existing selection

        brushes = [pg.mkBrush('r') if selected else pg.mkBrush('k') for selected in self.selected_indices]
        self.scatter_plot.setBrush(brushes)
        print(self.selected_indices)

    def remove_points(self):
        if not self.roi_active:
            return
        roi_bounds = self.roi.parentBounds()
        x_min, x_max = roi_bounds.left(), roi_bounds.right()
        y_min, y_max = roi_bounds.top(), roi_bounds.bottom()
        self.x_data = self.electrode_postitions[:,0]
        self.y_data = self.electrode_postitions[:,1]

        new_selected = (self.x_data >= x_min) & (self.x_data <= x_max) & (self.y_data >= y_min) & (self.y_data <= y_max)
        self.selected_indices[new_selected]= False  # remove selected points

        brushes = [pg.mkBrush('r') if selected else pg.mkBrush('k') for selected in self.selected_indices]
        self.scatter_plot.setBrush(brushes)




    def reset_selection(self):
        """Reset all selections and restore default point color."""
        self.selected_indices[:] = False  # Clear all selections
        self.scatter_plot.setBrush([pg.mkBrush('k')] * self.num_points)  # Reset all points to black


    def add_region(self):
        region_name = self.enter_name_field.text()
        if region_name:
            # self.list_widget.addItem(text)  # Add text to list
            self.enter_name_field.clear()  # Clear input field after adding

        e_label = na(self.tempate_electrodes['e_labels'])[self.selected_indices]
        e_pos = self.electrode_postitions[self.selected_indices]

        self.saved_labels.extend(e_label)
        self.saved_position.extend(e_pos)
        self.saved_rgn.extend([region_name]*len(e_pos))
        print('region added')
        self.reset_selection()





    def save_data(self):
        if self.current_index < 0:
            return
        # collect data
        data=pd.DataFrame({'e_label':self.saved_labels,
                           'e_pos_x':na(self.saved_position)[:,0],
                        'e_pos_y':na(self.saved_position)[:,0],
                        'region':self.saved_rgn})
        data['file'] = self.file_list[self.current_index]

        print(data)

        file_path = self.file_list[self.current_index]
        directory = os.path.dirname(file_path)
        save_path = os.path.join(directory, "labeled_electrodes.csv")

        data.to_csv(save_path)
        print(f"Data saved to {save_path}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())