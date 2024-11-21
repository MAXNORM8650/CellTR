import os
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw

class DrawPlane(tk.Tk):
    def __init__(self, width: int, height: int, w_offset: int, h_offset: int, scale_factor: float = 1.0):
        super().__init__()
        self.canvas = tk.Canvas(
            self,
            bg='black',
            height=(height - h_offset) * scale_factor,
            width=(width - w_offset) * scale_factor
        )
        self.canvas.pack()
        self.w_offset = w_offset
        self.h_offset = h_offset
        self.scale_factor = scale_factor
        self.trajectories = {}  # Dictionary to store trajectories for each cell

    def remove_offset(self, location: tuple) -> np.ndarray:
        return np.array(location) - np.array((self.w_offset, self.h_offset))

    def draw_cell(self, bounding_box: tuple, cell_id: int, parent_id: int, centroid: tuple = None):
        min_y, min_x, max_y, max_x = bounding_box
        min_y, min_x, max_y, max_x = [coord * self.scale_factor for coord in [min_y, min_x, max_y, max_x]]

        # Draw the cell bounding box
        self.canvas.create_rectangle(
            min_x, min_y, max_x, max_y,
            outline='red', width=2
        )

        # Draw cell ID and parent ID
        label_position = (min_x, min_y - 10)
        self.canvas.create_text(
            label_position, text=f"{cell_id}, {parent_id}", fill="yellow", anchor='nw'
        )

        # Draw trajectory if available
        if centroid:
            if cell_id not in self.trajectories:
                self.trajectories[cell_id] = []
            self.trajectories[cell_id].append(centroid)

            # Scale centroid for drawing
            scaled_centroid = tuple(c * self.scale_factor for c in centroid)
            for i in range(1, len(self.trajectories[cell_id])):
                prev_point = tuple(c * self.scale_factor for c in self.trajectories[cell_id][i - 1])
                curr_point = tuple(c * self.scale_factor for c in self.trajectories[cell_id][i])
                self.canvas.create_line(prev_point, curr_point, fill="blue", width=2)

            # Draw current centroid
            self.canvas.create_oval(
                scaled_centroid[0] - 3, scaled_centroid[1] - 3,
                scaled_centroid[0] + 3, scaled_centroid[1] + 3,
                fill="green"
            )

    def draw_cells_from_masks(self, directory: str, lineage_info: dict):
        images = sorted([f for f in os.listdir(directory) if f.endswith('.tif')])
        for idx, file in enumerate(images):
            image_path = os.path.join(directory, file)
            image = Image.open(image_path)
            image_array = np.array(image)
            for cell_id in lineage_info.keys():
                mask = (image_array == cell_id)
                if mask.any():
                    x, y = np.where(mask)
                    min_x, min_y = np.min(x), np.min(y)
                    max_x, max_y = np.max(x), np.max(y)
                    parent_id = lineage_info[cell_id].get('parent', 0)
                    centroid = (np.mean(y), np.mean(x))
                    self.draw_cell((min_y, min_x, max_y, max_x), cell_id, parent_id, centroid)

if __name__ == '__main__':
    lineage_file_path = "/home/komal.kumar/Documents/Cell/datasets/data/CTC/Training/Fluo-N2DL-HeLa/01_GT/TRA/man_track.txt"

    # Parse lineage information from the file
    def parse_lineage_file(file_path):
        lineage_info = {}
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 4:
                    cell_id, start_time, end_time, parent_id = map(int, parts)
                    lineage_info[cell_id] = {'start': start_time, 'end': end_time, 'parent': parent_id}
        return lineage_info

    lineage_info = parse_lineage_file(lineage_file_path)

    # Directory containing the sequence of cell images
    directory = "/home/komal.kumar/Documents/Cell/datasets/data/CTC/Training/Fluo-N2DL-HeLa/01_GT/TRA"

    draw_plane = DrawPlane(200, 200, 10, 10, 1.5)
    draw_plane.draw_cells_from_masks(directory, lineage_info)
    draw_plane.mainloop()