import math

class Sensor:
    def __init__(self, ranges, width, xy, name, bearing):
        self.min, self.max = ranges
        self.width = width
        self.half_width = self.width / 2.0

        # Mount position in BOAT frame
        self.mount_x, self.mount_y = xy
        self.name = name
        self.mount_bearing = bearing

    def pose_in_world(self, boat_x, boat_y, boat_heading):
        cx = math.cos(boat_heading)
        sx = math.sin(boat_heading)

        sensor_x = boat_x + self.mount_x * cx - self.mount_y * sx
        sensor_y = boat_y + self.mount_x * sx + self.mount_y * cx
        sensor_bearing = boat_heading + self.mount_bearing

        return sensor_x, sensor_y, sensor_bearing

    def reading_translation(self, dist, boat_x=0.0, boat_y=0.0, boat_heading=0.0):
        if dist is None:
            return None

        if not (self.min <= dist <= self.max):
            return None

        sensor_x, sensor_y, sensor_bearing = self.pose_in_world(boat_x, boat_y, boat_heading)

        px = sensor_x + dist * math.cos(sensor_bearing)
        py = sensor_y + dist * math.sin(sensor_bearing)

        return round(px, 6), round(py, 6)

    def arc_points(self, dist, boat_x=0.0, boat_y=0.0, boat_heading=0.0,
                   resolution=0.02, min_samples=5, max_samples=100):

        if dist is None:
            return []

        if not (self.min <= dist <= self.max):
            return []

        sensor_x, sensor_y, sensor_bearing = self.pose_in_world(boat_x, boat_y, boat_heading)

        arc_len = max(1e-6, dist * self.width)
        n = int(arc_len / resolution) + 1
        n = max(min_samples, min(n, max_samples))

        a0 = sensor_bearing - self.half_width
        a1 = sensor_bearing + self.half_width

        points = []

        if n == 1:
            p = self.reading_translation(dist, boat_x, boat_y, boat_heading)
            return [p] if p else []

        for i in range(n):
            a = a0 + (a1 - a0) * (i / (n - 1))
            px = sensor_x + dist * math.cos(a)
            py = sensor_y + dist * math.sin(a)
            points.append((px, py))

        return points

    def free_space_points(self, dist, boat_x=0.0, boat_y=0.0, boat_heading=0.0,
                          resolution=0.02, margin=0.03, max_points=20000):

        if dist is None:
            return []

        sensor_x, sensor_y, sensor_bearing = self.pose_in_world(boat_x, boat_y, boat_heading)

        if dist > self.max:
            dist = self.max + 0.04

        if dist < self.min:
            return []

        end = max(0.0, dist - margin)

        if end <= resolution:
            return []

        angle_step = resolution / max(end, 1e-6)
        angle_step = max(angle_step, math.radians(0.5))
        angle_step = min(angle_step, math.radians(5))

        a0 = sensor_bearing - self.half_width
        a1 = sensor_bearing + self.half_width

        n_angles = max(2, int((a1 - a0) / angle_step) + 1)

        points = set()

        for i in range(n_angles):
            a = a0 + (a1 - a0) * (i / (n_angles - 1))

            r = resolution
            while r <= end:
                px = sensor_x + r * math.cos(a)
                py = sensor_y + r * math.sin(a)
                points.add((round(px, 3), round(py, 3)))

                if len(points) >= max_points:
                    return points

                r += resolution

        return points