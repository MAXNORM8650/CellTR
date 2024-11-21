import matplotlib
matplotlib.use('TkAgg')  # Or 'Qt5Agg', 'Qt4Agg', depending on your system
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import TextBox, Slider
from PIL import Image
import numpy as np
import os
import pandas as pd

def read_image(image_path):
    image = Image.open(image_path)
    image_array = np.array(image)
    return image_array

def read_segmentation_mask(mask_path):
    mask = Image.open(mask_path)
    mask_array = np.array(mask)
    return mask_array

def parse_man_track_file(man_track_file_path):
    # Read the file into a pandas DataFrame
    df = pd.read_csv(man_track_file_path, sep='\s+', comment='#', header=None)
    df.columns = ['label', 't1', 't2', 'parent']

    # Convert DataFrame to a dictionary for easy access
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

    # Initialize data structures
    cell_positions = {}  # {cell_id: {frame_number: (x_centroid, y_centroid)}}
    frames = []  # List of image arrays
    masks = []  # List of mask arrays

    for img_file, ann_file in zip(image_files, annotation_files):
        image_path = os.path.join(sequence_dir, img_file)
        annotation_path = os.path.join(annotation_dir, ann_file)

        # Extract frame number from image file name (assuming format 'tXXX.tif')
        frame_number_str = os.path.splitext(img_file)[0].lstrip('t')
        frame_number = int(frame_number_str)

        image = read_image(image_path)
        mask = read_segmentation_mask(annotation_path)

        frames.append(image)
        masks.append(mask)

        # Extract individual cells based on unique labels in the mask
        cell_labels = np.unique(mask)
        cell_labels = cell_labels[cell_labels != 0]  # Exclude background

        for label in cell_labels:
            # Check if the cell is present in the current frame based on tracking data
            cell_info = tracking_data.get(label)
            if cell_info is None:
                continue  # Skip if cell info is not available

            if not (cell_info['t1'] <= frame_number <= cell_info['t2']):
                continue  # Skip if the cell is not present in this frame

            cell_mask = (mask == label).astype(np.uint8)
            positions = np.argwhere(cell_mask)
            if positions.size == 0:
                continue  # Skip if the mask is empty

            # Compute centroid
            y_coords, x_coords = positions[:, 0], positions[:, 1]
            y_centroid = np.mean(y_coords)
            x_centroid = np.mean(x_coords)

            # Store position
            if label not in cell_positions:
                cell_positions[label] = {}
            cell_positions[label][frame_number] = (x_centroid, y_centroid)

    return frames, masks, cell_positions


# Visualization Class
class CellTrackingVisualizer:
    def __init__(self, frames, masks, cell_positions):
        self.frames = frames
        self.masks = masks
        self.cell_positions = cell_positions
        self.num_frames = len(frames)
        self.current_frame = 0
        self.selected_cell_id = None

        # Set up the figure and axes
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        plt.subplots_adjust(bottom=0.25)

        # Display the first frame
        self.im_display = self.ax.imshow(self.frames[self.current_frame], cmap='gray')

        # Initialize overlay
        self.overlay = None
        self.trajectory_line, = self.ax.plot([], [], 'r-', linewidth=2)

        # Cell ID input box
        axbox = plt.axes([0.1, 0.1, 0.2, 0.05])
        self.text_box = TextBox(axbox, 'Cell ID:', initial='')

        # Frame slider
        axframe = plt.axes([0.4, 0.1, 0.5, 0.03])
        self.frame_slider = Slider(axframe, 'Frame', 0, self.num_frames - 1, valinit=0, valfmt='%d')

        # Register event handlers
        self.text_box.on_submit(self.update_cell_id)
        self.frame_slider.on_changed(self.update_frame)

        # Show the initial state
        self.update_display()

        plt.show()

    def update_cell_id(self, text):
        try:
            cell_id = int(text)
            if cell_id in self.cell_positions:
                self.selected_cell_id = cell_id
                self.update_display()
            else:
                print(f'Cell ID {cell_id} not found.')
        except ValueError:
            print('Invalid Cell ID')

    def update_frame(self, val):
        self.current_frame = int(self.frame_slider.val)
        self.update_display()

    def update_display(self):
        # Update image
        self.im_display.set_data(self.frames[self.current_frame])

        # Remove previous overlay
        if self.overlay is not None:
            self.overlay.remove()
            self.overlay = None

        # Overlay segmentation mask
        mask = self.masks[self.current_frame]
        if self.selected_cell_id is not None:
            # Highlight selected cell
            cell_mask = (mask == self.selected_cell_id)
            if np.any(cell_mask):
                # Create an RGBA overlay
                overlay_array = np.zeros((*cell_mask.shape, 4), dtype=np.float32)
                overlay_array[cell_mask, 0] = 1.0  # Red channel
                overlay_array[cell_mask, 3] = 0.5  # Alpha channel
                self.overlay = self.ax.imshow(overlay_array, extent=self.im_display.get_extent())

            # Update trajectory
            self.update_trajectory()
        else:
            # Clear trajectory
            self.trajectory_line.set_data([], [])

        self.fig.canvas.draw_idle()

    def update_trajectory(self):
        # Get positions up to current frame
        frames = sorted(self.cell_positions[self.selected_cell_id].keys())
        x_traj = []
        y_traj = []
        for frame in frames:
            if frame > self.current_frame:
                break
            x, y = self.cell_positions[self.selected_cell_id][frame]
            x_traj.append(x)
            y_traj.append(y)
        self.trajectory_line.set_data(x_traj, y_traj)
        
# Paths to your data
sequence_dir = '/home/komal.kumar/Documents/Cell/datasets/data/CTC/Training/Fluo-N2DL-HeLa/01/'
annotation_dir = '/home/komal.kumar/Documents/Cell/datasets/data/CTC/Training/Fluo-N2DL-HeLa/01_ST/SEG'
man_track_file = "/home/komal.kumar/Documents/Cell/datasets/data/CTC/Training/Fluo-N2DL-HeLa/01_GT/TRA/man_track.txt"

# Parse tracking data
tracking_data = parse_man_track_file(man_track_file)

# Extract frames, masks, and cell positions
frames, masks, cell_positions = extract_cells_from_sequence(sequence_dir, annotation_dir, tracking_data)

# Create the visualizer
visualizer = CellTrackingVisualizer(frames, masks, cell_positions)
