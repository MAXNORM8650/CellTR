import numpy as np
import sys
import tkinter as tk

class DrawPlane(tk.Tk):
    def __init__(self, width: int, height: int, w_offset: int, h_offset: int, scale_factor: float = 1.0):
        super().__init__()
        self.canvas = tk.Canvas(
            self,
            bg='white',
            height=(height - h_offset) * scale_factor,
            width=(width - w_offset) * scale_factor
        )
        self.canvas.pack()
        self.w_offset = w_offset
        self.h_offset = h_offset
        self.scale_factor = scale_factor

    def remove_offset(self, location: tuple) -> np.ndarray:
        return np.array(location) - np.array((self.w_offset, self.h_offset))

    def draw_cell(self, center: tuple, radius: float, cell_type: str, level: int):
        center = self.remove_offset(center) * self.scale_factor
        radius *= self.scale_factor

        cell = self.canvas.create_oval(
            center[0] - radius,
            center[1] - radius,
            center[0] + radius,
            center[1] + radius
        )

        fill_colors = {
            'AI': 'red',
            'NUMB': '#64d45c',  # green
            'STATE': 'yellow',
            'DEST': 'cyan',
            'GOAL': 'yellow'
        }
        fill_color = fill_colors.get(cell_type, 'black')  # Default to black if type is unknown
        self.canvas.itemconfig(cell, fill=fill_color)
        return cell

    def draw_target(self, target: tuple, cell_radius: float, tolerance: float):
        target = self.remove_offset(target) * self.scale_factor
        cell_radius *= self.scale_factor
        tolerance *= self.scale_factor

        self.canvas.create_oval(
            target[0] - cell_radius - tolerance,
            target[1] - cell_radius - tolerance,
            target[0] + cell_radius + tolerance,
            target[1] + cell_radius + tolerance
        )

if __name__ == '__main__':
    draw_plane = DrawPlane(800, 1200, 10, 10, 1.5)
    draw_plane.draw_cell((50, 50), 10, 'AI', 1)
    draw_plane.draw_cell((100, 100), 15, 'NUMB', 1)
    draw_plane.draw_cell((150, 150), 20, 'STATE', 1)
    draw_plane.mainloop()