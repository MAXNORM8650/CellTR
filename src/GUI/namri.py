import sys
import os
import numpy as np
import pandas as pd
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QListWidget, QListWidgetItem,
    QFileDialog, QCheckBox, QMessageBox
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.animation as animation
print(animation.writers.list())

# Data loading functions (unchanged)
def read_image(image_path):
    image = Image.open(image_path)
    image_array = np.array(image)
    return image_array

def read_segmentation_mask(mask_path):
    mask = Image.open(mask_path)
    mask_array = np.array(mask)
    return mask_array

def parse_man_track_file(man_track_file_path):
    df = pd.read_csv(man_track_file_path, sep='\s+', comment='#', header=None)
    df.columns = ['label', 't1', 't2', 'parent']
    tracking_data = {}
    for _, row in df.iterrows():
        label = int(row['label'])
        t1 = int(row['t1'])
        t2 = int(row['t2'])
        parent = int(row['parent'])
        tracking_data[label] = {'t1': t1, 't2': t2, 'parent': parent}
    return tracking_data

def extract_cells_from_sequence(sequence_dir, annotation_dir, tracking_data):
    image_files = sorted([f for f in os.listdir(sequence_dir) if f.endswith('.tif')])
    annotation_files = sorted([f for f in os.listdir(annotation_dir) if f.endswith('.tif')])

    frames = []
    masks = []
    cell_positions = {}

    for img_file, ann_file in zip(image_files, annotation_files):
        image_path = os.path.join(sequence_dir, img_file)
        annotation_path = os.path.join(annotation_dir, ann_file)

        frame_number_str = os.path.splitext(img_file)[0].lstrip('t')
        frame_number = int(frame_number_str)

        image = read_image(image_path)
        mask = read_segmentation_mask(annotation_path)

        frames.append(image)
        masks.append(mask)

        cell_labels = np.unique(mask)
        cell_labels = cell_labels[cell_labels != 0]

        for label in cell_labels:
            cell_info = tracking_data.get(label)
            if cell_info is None:
                continue

            if not (cell_info['t1'] <= frame_number <= cell_info['t2']):
                continue

            cell_mask = (mask == label)
            positions = np.argwhere(cell_mask)
            if positions.size == 0:
                continue

            y_coords, x_coords = positions[:, 0], positions[:, 1]
            y_centroid = np.mean(y_coords)
            x_centroid = np.mean(x_coords)

            if label not in cell_positions:
                cell_positions[label] = {'frames': [], 'positions': []}
            cell_positions[label]['frames'].append(frame_number)
            cell_positions[label]['positions'].append((x_centroid, y_centroid))

    return frames, masks, cell_positions

# Main GUI class
class CellTrackingGUI(QMainWindow):
    def __init__(self, frames, masks, cell_positions):
        super().__init__()
        self.setWindowTitle('Cell Tracking Visualization')
        self.frames = frames
        self.masks = masks
        self.cell_positions = cell_positions
        self.num_frames = len(frames)
        self.current_frame = 0
        self.selected_cell_ids = []
        self.is_recording = False
        self.recording_frames = []
        self.init_ui()

    def init_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Left panel: Matplotlib canvas
        self.canvas = FigureCanvas(Figure(figsize=(8, 6)))
        main_layout.addWidget(self.canvas)
        self.ax = self.canvas.figure.subplots()

        # Right panel: Controls
        control_panel = QWidget()
        control_layout = QVBoxLayout()
        control_panel.setLayout(control_layout)
        main_layout.addWidget(control_panel)

        # Frame slider
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(self.num_frames - 1)
        self.frame_slider.setValue(0)
        self.frame_slider.valueChanged.connect(self.update_frame)
        control_layout.addWidget(QLabel('Frame Slider'))
        control_layout.addWidget(self.frame_slider)

        # Cell selection list
        control_layout.addWidget(QLabel('Select Cells'))
        self.cell_list_widget = QListWidget()
        self.cell_list_widget.setSelectionMode(QListWidget.MultiSelection)
        for cell_id in sorted(self.cell_positions.keys()):
            item = QListWidgetItem(f'Cell {cell_id}')
            self.cell_list_widget.addItem(item)
        self.cell_list_widget.itemSelectionChanged.connect(self.update_selected_cells)
        control_layout.addWidget(self.cell_list_widget)

        # Play and Pause buttons
        play_pause_layout = QHBoxLayout()
        self.play_button = QPushButton('Play')
        self.pause_button = QPushButton('Pause')
        self.play_button.clicked.connect(self.play)
        self.pause_button.clicked.connect(self.pause)
        play_pause_layout.addWidget(self.play_button)
        play_pause_layout.addWidget(self.pause_button)
        control_layout.addLayout(play_pause_layout)

        # Recording options
        recording_layout = QHBoxLayout()
        self.record_checkbox = QCheckBox('Record')
        self.record_checkbox.stateChanged.connect(self.toggle_recording)
        recording_layout.addWidget(self.record_checkbox)

        self.save_video_button = QPushButton('Save Video')
        self.save_video_button.clicked.connect(self.save_video)
        self.save_video_button.setEnabled(False)
        recording_layout.addWidget(self.save_video_button)

        control_layout.addLayout(recording_layout)

        # Timer for animation
        self.timer = self.canvas.new_timer(interval=200)
        self.timer.add_callback(self.next_frame)

        # Initial display
        self.update_display()

    def update_frame(self, value):
        self.current_frame = value
        self.update_display()

    def update_selected_cells(self):
        selected_items = self.cell_list_widget.selectedItems()
        self.selected_cell_ids = [int(item.text().split()[1]) for item in selected_items]
        self.update_display()

    def update_display(self):
        self.ax.clear()
        image = self.frames[self.current_frame]
        mask = self.masks[self.current_frame]

        # Display the image
        self.ax.imshow(image, cmap='gray')

        # Overlay segmentation masks for selected cells
        overlay = np.zeros_like(image, dtype=np.float32)
        for cell_id in self.selected_cell_ids:
            cell_mask = (mask == cell_id)
            overlay[cell_mask] = 1.0  # You can adjust the intensity

        if np.any(overlay):
            self.ax.imshow(overlay, cmap='jet', alpha=0.5)

        # Plot trajectories
        for cell_id in self.selected_cell_ids:
            data = self.cell_positions.get(cell_id, None)
            if data:
                frames = np.array(data['frames'])
                positions = np.array(data['positions'])
                # Select positions up to current frame
                indices = frames <= self.current_frame
                x_traj = positions[indices, 0]
                y_traj = positions[indices, 1]
                self.ax.plot(x_traj, y_traj, linewidth=2, label=f'Cell {cell_id}')
        # self.ax.legend(loc='upper right')

        self.ax.set_title(f'Frame {self.current_frame}')
        self.canvas.draw()

        # If recording, store the current frame
        if self.is_recording:
            self.recording_frames.append(self.canvas.copy_from_bbox(self.ax.bbox))

    def play(self):
        self.timer.start()

    def pause(self):
        self.timer.stop()

    def next_frame(self):
        self.current_frame = (self.current_frame + 1) % self.num_frames
        self.frame_slider.setValue(self.current_frame)

    def toggle_recording(self, state):
        if state == Qt.Checked:
            self.is_recording = True
            self.recording_frames = []
            self.save_video_button.setEnabled(True)
        else:
            self.is_recording = False

    def save_video(self):
        if not self.recording_frames:
            QMessageBox.warning(self, 'No Recording', 'No frames were recorded.')
            return

        # Ask user for save location
        save_path, _ = QFileDialog.getSaveFileName(self, 'Save Video', '', 'MP4 Video (*.mp4);;GIF Animation (*.gif)')
        if not save_path:
            return

        # Determine format based on file extension
        if save_path.lower().endswith('.gif'):
            format = 'gif'
        else:
            format = 'mp4'

        # Create animation
        fig = self.canvas.figure

        # Define animation function
        def animate(i):
            self.ax.clear()
            image = self.frames[i]
            mask = self.masks[i]

            self.ax.imshow(image, cmap='gray')

            overlay = np.zeros_like(image, dtype=np.float32)
            for cell_id in self.selected_cell_ids:
                cell_mask = (mask == cell_id)
                overlay[cell_mask] = 1.0

            if np.any(overlay):
                self.ax.imshow(overlay, cmap='jet', alpha=0.5)

            for cell_id in self.selected_cell_ids:
                data = self.cell_positions.get(cell_id, None)
                if data:
                    frames = np.array(data['frames'])
                    positions = np.array(data['positions'])
                    indices = frames <= i
                    x_traj = positions[indices, 0]
                    y_traj = positions[indices, 1]
                    self.ax.plot(x_traj, y_traj, linewidth=2, label=f'Cell {cell_id}')
            self.ax.legend(loc='upper right')
            self.ax.set_title(f'Frame {i}')

        anim = animation.FuncAnimation(fig, animate, frames=self.num_frames, interval=200, blit=False)

        # Save the animation
        try:
            if format == 'mp4':
                anim.save(save_path, writer='ffmpeg')
            elif format == 'gif':
                anim.save(save_path, writer='imagemagick')
            QMessageBox.information(self, 'Success', f'Video saved to {save_path}')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to save video: {e}')

        # Reset recording
        self.record_checkbox.setChecked(False)
        self.recording_frames = []
        self.save_video_button.setEnabled(False)

# Function to automatically find directories (unchanged)
def find_data_directories(root_dir):
    sequence_dir = None
    annotation_dir = None
    man_track_file = None

    # Search for sequence directories (containing image files)
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if any(f.endswith('.tif') for f in filenames):
            if 'SEG' not in dirpath and 'TRA' not in dirpath:
                sequence_dir = dirpath
                break

    # Search for annotation directories (containing segmentation masks)
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if 'SEG' in dirpath and any(f.endswith('.tif') for f in filenames):
            annotation_dir = dirpath
            break

    # Search for man_track.txt file
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for f in filenames:
            if f == 'man_track.txt':
                man_track_file = os.path.join(dirpath, f)
                break
        if man_track_file:
            break

    return sequence_dir, annotation_dir, man_track_file

# Main execution (unchanged)
if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Prompt user to select the root directory
    root_dir = QFileDialog.getExistingDirectory(None, 'Select Root Directory', os.getcwd())

    if not root_dir:
        print('No directory selected. Exiting.')
        sys.exit()

    # Automatically find the directories
    sequence_dir, annotation_dir, man_track_file = find_data_directories(root_dir)

    if not sequence_dir or not annotation_dir or not man_track_file:
        print('Could not find all required directories/files.')
        print(f'Sequence directory: {sequence_dir}')
        print(f'Annotation directory: {annotation_dir}')
        print(f'Tracking file: {man_track_file}')
        sys.exit()

    print(f'Found sequence directory: {sequence_dir}')
    print(f'Found annotation directory: {annotation_dir}')
    print(f'Found tracking file: {man_track_file}')

    # Parse tracking data
    tracking_data = parse_man_track_file(man_track_file)

    # Extract frames, masks, and cell positions
    frames, masks, cell_positions = extract_cells_from_sequence(sequence_dir, annotation_dir, tracking_data)

    # Create and show the GUI
    gui = CellTrackingGUI(frames, masks, cell_positions)
    gui.show()
    sys.exit(app.exec_())
