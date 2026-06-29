import numpy as np

class HitMap:
    def __init__(self, width=4, height=4, resolution=0.02):
        self.width = width
        self.height = height
        self.resolution = resolution

        self.x_min = -width / 2
        self.y_min = -height / 2

        self.columns = int(width / resolution)
        self.rows = int(height / resolution)

        self.grid = np.full((self.rows, self.columns), -20, dtype=np.float32)

    def coord_to_cell(self, x, y):
        j = int((x - self.x_min) / self.resolution)
        i = int((y - self.y_min) / self.resolution)

        if 0 <= i < self.rows and 0 <= j < self.columns:
            return i, j
        return None

    def add_point(self, x, y, w):
        cell = self.coord_to_cell(x, y)
        if cell is None:
            return False

        i, j = cell
        self.grid[i, j] += w
        return True

    def add_points(self, points, w):
        for x, y in points:
            self.add_point(x, y, w)

    def decay(self, factor=0.96):
        self.grid *= factor

    def clamp(self, low=-20, high=20):
        self.grid = np.clip(self.grid, low, high)