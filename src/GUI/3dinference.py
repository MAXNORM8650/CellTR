import sys
import os
from glob import glob
import numpy as np
import pandas as pd
import imageio
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QFileDialog, QAction, QMessageBox,
    QProgressBar, QInputDialog, QTabWidget, QGraphicsRectItem
)
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph as pg
import pyqtgraph.opengl as gl

import warnings
warnings.filterwarnings("ignore") 


class CellTrackingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Cell Tracking Visualization')
        self.init_ui()
        self.images = []
        self.masks = []
        self.tracking_data = {}
        self.frame_count = 0
        self.current_frame = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

    def init_ui(self):
        # Create main layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Create a tab widget for 2D and 3D views
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # 2D Visualization Widget
        self.visualization_widget = pg.GraphicsLayoutWidget()
        self.tab_widget.addTab(self.visualization_widget, '2D View')

        # 3D Visualization Widget
        self.visualization_widget_3d = gl.GLViewWidget()
        self.tab_widget.addTab(self.visualization_widget_3d, '3D View')

        # Right panel: Controls and info
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)
        main_layout.addWidget(right_panel)

        # Controls
        self.play_button = QPushButton('Play')
        self.pause_button = QPushButton('Pause')
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_label = QLabel('Frame: 0')
        right_layout.addWidget(self.play_button)
        right_layout.addWidget(self.pause_button)
        right_layout.addWidget(self.frame_label)
        right_layout.addWidget(self.frame_slider)

        # Cell information
        self.cell_info_label = QLabel('Cell Info')
        right_layout.addWidget(self.cell_info_label)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        right_layout.addWidget(self.progress_bar)

        # Connections
        self.play_button.clicked.connect(self.play)
        self.pause_button.clicked.connect(self.pause)
        self.frame_slider.valueChanged.connect(self.update_frame)

        # Menu bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')

        open_images_action = QAction('Open Images', self)
        open_images_action.triggered.connect(self.open_images)
        file_menu.addAction(open_images_action)

        open_masks_action = QAction('Open Masks', self)
        open_masks_action.triggered.connect(self.open_masks)
        file_menu.addAction(open_masks_action)

        open_tracking_action = QAction('Open Tracking Data', self)
        open_tracking_action.triggered.connect(self.open_tracking_data)
        file_menu.addAction(open_tracking_action)

        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Enable cell selection
        self.enable_cell_selection()

    def open_images(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        image_dir = QFileDialog.getExistingDirectory(self, 'Select Image Directory', options=options)
        if image_dir:
            image_files = sorted(glob(os.path.join(image_dir, '*.tif')))
            if not image_files:
                QMessageBox.warning(self, 'No Images Found', 'No .tif images found in the selected directory.')
                return

            # Ask user to specify the number of images
            num_images, ok = QInputDialog.getInt(
                self, 'Number of Images', 'Enter the number of images to load:', min=1, max=len(image_files), value=len(image_files)
            )
            if not ok:
                return  # User cancelled the dialog

            # Limit the image files
            image_files = image_files[:num_images]

            self.images = []
            num_files = len(image_files)
            self.progress_bar.setMaximum(num_files)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat('Loading Images: %p%')
            self.progress_bar.show()

            for i, f in enumerate(image_files, 1):
                img = imageio.imread(f)
                self.images.append(img)
                self.progress_bar.setValue(i)
                QApplication.processEvents()  # Update the GUI

            self.progress_bar.hide()
            self.frame_count = len(self.images)
            self.frame_slider.setMaximum(self.frame_count - 1)
            self.frame_slider.setValue(0)
            self.frame_slider.setEnabled(True)
            self.current_frame = 0
            self.update_frame(0)
            self.check_data_loaded()

    def open_masks(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        mask_dir = QFileDialog.getExistingDirectory(self, 'Select Mask Directory', options=options)
        if mask_dir:
            mask_files = sorted(glob(os.path.join(mask_dir, '*.tif')))
            if not mask_files:
                QMessageBox.warning(self, 'No Masks Found', 'No .tif masks found in the selected directory.')
                return

            # Ask user to specify the number of masks
            num_masks, ok = QInputDialog.getInt(
                self, 'Number of Masks', 'Enter the number of masks to load:', min=1, max=len(mask_files), value=len(mask_files)
            )
            if not ok:
                return  # User cancelled the dialog

            # Limit the mask files
            mask_files = mask_files[:num_masks]

            self.masks = []
            num_files = len(mask_files)
            self.progress_bar.setMaximum(num_files)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat('Loading Masks: %p%')
            self.progress_bar.show()

            for i, f in enumerate(mask_files, 1):
                mask = imageio.imread(f)
                self.masks.append(mask)
                self.progress_bar.setValue(i)
                QApplication.processEvents()  # Update the GUI

            self.progress_bar.hide()
            self.check_data_loaded()

    def open_tracking_data(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        tracking_file, _ = QFileDialog.getOpenFileName(self, 'Open Tracking Data File', '', 'Text Files (*.txt)', options=options)
        if tracking_file:
            self.progress_bar.setMaximum(0)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat('Loading Tracking Data...')
            self.progress_bar.show()

            self.tracking_data = self.parse_tracking_data(tracking_file)

            self.progress_bar.hide()
            self.check_data_loaded()

    def check_data_loaded(self):
        if self.images and self.masks and self.tracking_data:
            self.update_frame(self.current_frame)
        else:
            QMessageBox.information(self, 'Data Loading', 'Please ensure that images, masks, and tracking data are loaded.')

    def parse_tracking_data(self, tracking_data_file):
        df = pd.read_csv(tracking_data_file, delim_whitespace=True, header=None)
        df.columns = ['label', 't_start', 't_end', 'parent']

        # Build a dictionary: label -> {t_start, t_end, parent}
        tracking_dict = {}
        num_rows = len(df)
        self.progress_bar.setMaximum(num_rows)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat('Parsing Tracking Data: %p%')
        QApplication.processEvents()

        for i, (_, row) in enumerate(df.iterrows(), 1):
            label = int(row['label'])
            tracking_dict[label] = {
                't_start': int(row['t_start']),
                't_end': int(row['t_end']),
                'parent': int(row['parent'])
            }
            self.progress_bar.setValue(i)
            QApplication.processEvents()

        return tracking_dict

    def update_frame(self, frame_index):
        if not self.images or not self.masks:
            return

        self.current_frame = frame_index
        self.frame_label.setText(f'Frame: {frame_index}')

        # Clear previous plots
        self.visualization_widget.clear()

        # Get current image and mask
        image = self.images[frame_index]
        mask = self.masks[frame_index]

        # Display image
        img_view = self.visualization_widget.addViewBox()
        img_view.setAspectLocked(True)
        img_view.invertY(True)  # Invert Y-axis to match image coordinate system
        img_item = pg.ImageItem(image)
        img_view.addItem(img_item)

        # Overlay masks and trajectories
        self.draw_masks_and_trajectories(frame_index, img_view)

        # Enable interactive view
        img_view.setMouseEnabled(x=True, y=True)

        # Update 3D visualization
        self.update_3d_view()

    def draw_masks_and_trajectories(self, frame_index, img_view):
        mask = self.masks[frame_index]
        labels_in_frame = np.unique(mask)
        labels_in_frame = labels_in_frame[labels_in_frame != 0]  # Exclude background

        for label in labels_in_frame:
            cell_info = self.tracking_data.get(label, None)
            if cell_info is None:
                continue

            # Check if cell is present in current frame
            if not (cell_info['t_start'] <= frame_index <= cell_info['t_end']):
                continue

            # Get cell mask
            cell_mask = (mask == label)
            positions = np.argwhere(cell_mask)
            if positions.size == 0:
                continue

            # Calculate bounding rectangle
            y_min, x_min = positions.min(axis=0)
            y_max, x_max = positions.max(axis=0) + 1  # +1 to include the max index

            # Draw rectangle around the cell
            color = pg.intColor(label)
            rect = QGraphicsRectItem(
                x_min, y_min, x_max - x_min, y_max - y_min
            )
            rect.setPen(pg.mkPen(color, width=2))
            img_view.addItem(rect)

            # Draw trajectory
            positions = self.get_cell_trajectory(label, frame_index)
            if positions is not None:
                positions = np.array(positions)
                # Swap x and y positions for correct plotting
                curve = pg.PlotCurveItem(
                    positions[:, 0], positions[:, 1],
                    pen=pg.mkPen(color, width=2)
                )
                img_view.addItem(curve)

    def get_cell_trajectory(self, label, frame_index):
        positions = []
        cell_info = self.tracking_data.get(label, None)
        if cell_info is None:
            return None

        for t in range(cell_info['t_start'], frame_index + 1):
            if t >= len(self.masks):
                break
            mask = self.masks[t]
            cell_mask = (mask == label)
            if np.any(cell_mask):
                y_coords, x_coords = np.where(cell_mask)
                y_mean = y_coords.mean()
                x_mean = x_coords.mean()
                positions.append((x_mean, y_mean))  # x, y

        if positions:
            return positions
        else:
            return None

    def update_3d_view(self):
        self.visualization_widget_3d.clear()
        self.visualization_widget_3d.setCameraPosition(distance=200)

        # Plot trajectories in 3D
        for label, cell_info in self.tracking_data.items():
            positions = []
            for t in range(cell_info['t_start'], cell_info['t_end'] + 1):
                if t >= len(self.masks):
                    break
                mask = self.masks[t]
                cell_mask = (mask == label)
                if np.any(cell_mask):
                    y_coords, x_coords = np.where(cell_mask)
                    y_mean = y_coords.mean()
                    x_mean = x_coords.mean()
                    positions.append([x_mean, y_mean, t])  # x, y, z (time)

            if positions:
                positions = np.array(positions)
                color = pg.glColor(pg.intColor(label))
                plt = gl.GLLinePlotItem(
                    pos=positions, color=color, width=2, antialias=True
                )
                self.visualization_widget_3d.addItem(plt)

    def play(self):
        if self.images:
            self.timer.start(200)  # Adjust interval as needed

    def pause(self):
        self.timer.stop()

    def next_frame(self):
        next_frame = (self.current_frame + 1) % self.frame_count
        self.frame_slider.setValue(next_frame)

    def enable_cell_selection(self):
        self.visualization_widget.scene().sigMouseClicked.connect(self.on_mouse_click)

    def on_mouse_click(self, event):
        if not self.images or not self.masks:
            return
        pos = event.scenePos()
        x = int(pos.x())
        y = int(pos.y())
        label = self.get_label_at_position(x, y)
        if label is not None:
            self.display_cell_info(label)

    def get_label_at_position(self, x, y):
        mask = self.masks[self.current_frame]
        if x >= 0 and x < mask.shape[1] and y >= 0 and y < mask.shape[0]:
            label = mask[y, x]
            if label != 0:
                return label
        return None

    def display_cell_info(self, label):
        cell_info = self.tracking_data.get(label, {})
        info_text = f'Cell ID: {label}\nStart Frame: {cell_info.get("t_start")}\nEnd Frame: {cell_info.get("t_end")}\nParent ID: {cell_info.get("parent")}'
        self.cell_info_label.setText(info_text)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = CellTrackingGUI()
    gui.show()
    sys.exit(app.exec_())
