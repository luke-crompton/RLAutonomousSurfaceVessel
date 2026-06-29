import os
import sys
import math
import time
from collections import deque

# allow imports from project root when running from this subdirectory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pygame
import serial
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from Lake_environment import LakeMapEnv
from Configuration import SimConfig
from serial_reader import SensorPacket, parse_csv_line
from sensor_class import Sensor
from mapping import HitMap

# ==========================================================
# User settings
# ==========================================================
LOG_DIR = "runs/ppo_lake"
MODEL_BEST_PATH = os.path.join(LOG_DIR, "best_model")
VECNORM_PATH = os.path.join(LOG_DIR, "vecnorm.1")

COM_PORT = "COM5"
BAUD_RATE = 115200
USE_ESP32_SERIAL = True

CONTROL_HZ = 20
AGENT_HZ = 5
TICKS_PER_ACTION = max(1, CONTROL_HZ // AGENT_HZ)

# Larger scale so motion is visible on screen
SCALE = 10
PANEL_W = 440
PANEL_GAP = 8

# Match training scale better
SENSOR_MAX_M = 4.0
FALLBACK_SENSOR_M = SENSOR_MAX_M

# World map memory resolution. 0.5 m is finer than training cells (1 m)
# but still light enough to run well.
WORLD_MAP_RES_M = 0.5
WORLD_MAP_FREE_W = -2.0
WORLD_MAP_HIT_W = +3.0
WORLD_MAP_DECAY = 0.998
WORLD_MAP_MIN = -20.0
WORLD_MAP_MAX = 20.0
FREE_THRESHOLD = -1.5
OCC_THRESHOLD = +1.5



# Local cone sampling resolution. Safe because the model only sees cone summaries,
# not the raw patch.
LOCAL_SAMPLE_RES_M = 0.20

DEBUG_HISTORY = deque(maxlen=16)
PATH_HISTORY = deque(maxlen=600)

SENSORS = {
    "US1": Sensor((0.02, SENSOR_MAX_M), math.radians(30), (-0.1665, 0.0),    "US1", math.pi),
    "US2": Sensor((0.02, SENSOR_MAX_M), math.radians(30), (-0.1476, 0.2217), "US2", math.radians(125)),
    "US3": Sensor((0.02, SENSOR_MAX_M), math.radians(30), (0.0, 0.25),       "US3", math.pi / 2),
    "US4": Sensor((0.02, SENSOR_MAX_M), math.radians(30), (0.1476, 0.2217),  "US4", math.radians(55)),
    "US5": Sensor((0.02, SENSOR_MAX_M), math.radians(30), (0.1665, 0.0),     "US5", 0.0),
}

EMPTY_PACKET = SensorPacket(
    rx_time=0.0,
    ranges_m={"US1": None, "US2": None, "US3": None, "US4": None, "US5": None},
)


def make_env():
    cfg = SimConfig(size=64, obstacle_p=0.0, algae_p=0.0, max_steps=int(1e9))
    return LakeMapEnv(cfg)


def load_vecnormalize(venv):
    venv = VecNormalize.load(VECNORM_PATH, venv)
    venv.training = False
    venv.norm_reward = False
    return venv


def load_model(venv):
    return PPO.load(MODEL_BEST_PATH, env=venv, device="cpu")


def mark_visited_patch(env):
    ix = int(np.floor(env.xf))
    iy = int(np.floor(env.yf))
    env.x = ix
    env.y = iy
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            nx, ny = ix + dx, iy + dy
            if 0 <= nx < env.size and 0 <= ny < env.size:
                env.known_visited[nx, ny] = True


def hard_reset_env(env):
    env.reset()
    env.obstacle[:, :] = False
    env.algae[:, :] = False

    env.xf = env.size * 0.5
    env.yf = env.size * 0.5
    env.x = int(np.floor(env.xf))
    env.y = int(np.floor(env.yf))
    env.heading = 0.0

    env.v = 0.0
    env.v_prev = 0.0
    env.yaw = 0.0
    env.applied_speed_cmd = 0.0
    env.applied_yaw_cmd = 0.0
    env.prev_throttle = 0.0
    env.prev_turn = 0.0
    env.t = 0
    env.current_cell = (env.x, env.y)
    env.disp_window.clear()

    env.known_free[:, :] = False
    env.known_visited[:, :] = False
    env.log_prob[:, :] = 0.0
    env.prev_coverage = float(env.known_visited.mean())
    mark_visited_patch(env)
    PATH_HISTORY.clear()
    PATH_HISTORY.append((env.xf, env.yf))


def reset_world_map(env):
    wm = HitMap(width=float(env.size), height=float(env.size), resolution=WORLD_MAP_RES_M)
    wm.grid[:, :] = 0.0
    return wm


def world_to_map_coords(env, wx, wy):
    return float(wx - env.size * 0.5), float(wy - env.size * 0.5)


def clean_distance_m(v, max_m=FALLBACK_SENSOR_M):
    if v is None:
        return max_m
    try:
        v = float(v)
    except Exception:
        return max_m
    if not np.isfinite(v):
        return max_m
    return float(np.clip(v, 0.0, max_m))


def packet_to_metric_channels(packet):
    p = packet if packet is not None else EMPTY_PACKET
    us1 = clean_distance_m(p.ranges_m.get("US1"), FALLBACK_SENSOR_M)
    us2 = clean_distance_m(p.ranges_m.get("US2"), FALLBACK_SENSOR_M)
    us3 = clean_distance_m(p.ranges_m.get("US3"), FALLBACK_SENSOR_M)
    us4 = clean_distance_m(p.ranges_m.get("US4"), FALLBACK_SENSOR_M)
    us5 = clean_distance_m(p.ranges_m.get("US5"), FALLBACK_SENSOR_M)
    return us1, us2, us3, us4, us5


def packet_to_normalised_direct_channels(env, packet):
    us1, us2, us3, us4, us5 = packet_to_metric_channels(packet)
    tof_l = float(np.clip(us1 / max(1e-6, env.tof_max), 0.0, 1.0))
    d_l   = float(np.clip(us2 / max(1e-6, env.us_ray_max), 0.0, 1.0))
    d_f   = float(np.clip(us3 / max(1e-6, env.radar_max), 0.0, 1.0))
    d_r   = float(np.clip(us4 / max(1e-6, env.us_ray_max), 0.0, 1.0))
    tof_r = float(np.clip(us5 / max(1e-6, env.tof_max), 0.0, 1.0))
    return tof_l, d_l, d_f, d_r, tof_r, us1, us2, us3, us4, us5


def write_world_point_to_env(env, world_x, world_y, value):
    ix = int(np.floor(world_x))
    iy = int(np.floor(world_y))
    if 0 <= ix < env.size and 0 <= iy < env.size:
        env.log_prob[ix, iy] = float(np.clip(value, WORLD_MAP_MIN, WORLD_MAP_MAX))
        if env.log_prob[ix, iy] <= FREE_THRESHOLD:
            env.known_free[ix, iy] = True


def update_world_hit_map(world_map, env, packet):
    if packet is None:
        return
    world_map.decay(WORLD_MAP_DECAY)
    # mild fading of displayed probability too
    env.log_prob[:, :] *= WORLD_MAP_DECAY
    env.known_free[:, :] = env.log_prob <= FREE_THRESHOLD

    for key, sensor in SENSORS.items():
        dist = packet.ranges_m.get(key, None)
        clear_dist = dist if dist is not None else sensor.max

        free_points_world = sensor.free_space_points(
            clear_dist,
            boat_x=env.xf,
            boat_y=env.yf,
            boat_heading=env.heading,
            resolution=world_map.resolution,
        )
        for xw, yw in free_points_world:
            xm, ym = world_to_map_coords(env, xw, yw)
            world_map.add_point(xm, ym, WORLD_MAP_FREE_W)
            cell = world_map.coord_to_cell(xm, ym)
            if cell is not None:
                i, j = cell
                write_world_point_to_env(env, xw, yw, world_map.grid[i, j])

        if dist is not None:
            arc_points_world = sensor.arc_points(
                dist,
                boat_x=env.xf,
                boat_y=env.yf,
                boat_heading=env.heading,
                resolution=world_map.resolution,
            )
            for xw, yw in arc_points_world:
                xm, ym = world_to_map_coords(env, xw, yw)
                world_map.add_point(xm, ym, WORLD_MAP_HIT_W)
                cell = world_map.coord_to_cell(xm, ym)
                if cell is not None:
                    i, j = cell
                    write_world_point_to_env(env, xw, yw, world_map.grid[i, j])

    world_map.clamp(low=WORLD_MAP_MIN, high=WORLD_MAP_MAX)
    env.log_prob[:, :] = np.clip(env.log_prob, WORLD_MAP_MIN, WORLD_MAP_MAX)


def build_cone_geometry(n_cones, radius_m, resolution_m):
    x_coords = np.arange(-radius_m, radius_m + 0.5 * resolution_m, resolution_m, dtype=np.float32)
    y_coords = np.arange(-radius_m, radius_m + 0.5 * resolution_m, resolution_m, dtype=np.float32)
    yy, xx = np.meshgrid(y_coords, x_coords, indexing="xy")
    rr = np.sqrt(xx**2 + yy**2)
    inside = rr <= (radius_m + 1e-9)
    angles = (np.arctan2(xx, yy) + 2.0 * np.pi) % (2.0 * np.pi)
    bins = ((angles / (2.0 * np.pi)) * n_cones).astype(np.int32) % n_cones

    masks = np.zeros((n_cones, yy.shape[0], yy.shape[1]), dtype=bool)
    for k in range(n_cones):
        masks[k] = inside & (bins == k)

    center_x = np.argmin(np.abs(x_coords))
    center_y = np.argmin(np.abs(y_coords))
    masks[:, center_y, center_x] = False

    counts = masks.reshape(n_cones, -1).sum(axis=1).astype(np.float32)
    counts = np.clip(counts, 1.0, None)
    return {
        "x_coords": x_coords,
        "y_coords": y_coords,
        "rr": rr.astype(np.float32),
        "masks": masks,
        "counts": counts,
        "radius_m": float(radius_m),
    }


def sample_world_map_local_patch(world_map, env, cone_geom):
    x_coords = cone_geom["x_coords"]
    y_coords = cone_geom["y_coords"]
    patch = np.zeros((len(y_coords), len(x_coords)), dtype=np.float32)

    cx = math.cos(env.heading)
    sx = math.sin(env.heading)

    for row_idx, y_local in enumerate(y_coords):
        for col_idx, x_local in enumerate(x_coords):
            wx = env.xf + x_local * cx - y_local * sx
            wy = env.yf + x_local * sx + y_local * cx
            mx, my = world_to_map_coords(env, wx, wy)
            cell = world_map.coord_to_cell(mx, my)
            if cell is not None:
                i, j = cell
                patch[row_idx, col_idx] = world_map.grid[i, j]
    return patch


def build_live_cone_features(world_map, env, cone_geom):
    patch = sample_world_map_local_patch(world_map, env, cone_geom)
    masks = cone_geom["masks"]
    counts = cone_geom["counts"]
    rr = cone_geom["rr"]
    radius_m = cone_geom["radius_m"]
    n_cones = masks.shape[0]

    free_mask = patch <= FREE_THRESHOLD
    occ_mask = patch >= OCC_THRESHOLD

    free_counts = (masks & free_mask[None, :, :]).reshape(n_cones, -1).sum(axis=1).astype(np.float32)
    cone_free = np.clip(free_counts / counts, 0.0, 1.0)

    cone_ob_d = np.ones(n_cones, dtype=np.float32)
    for k in range(n_cones):
        this_occ = masks[k] & occ_mask
        if np.any(this_occ):
            min_r = float(rr[this_occ].min())
            cone_ob_d[k] = float(np.clip(min_r / max(1e-6, radius_m), 0.0, 1.0))
    return patch, cone_free.astype(np.float32), cone_ob_d.astype(np.float32)


def build_live_observation(env, world_map, cone_geom, packet):
    tof_l, d_l, d_f, d_r, tof_r, *_ = packet_to_normalised_direct_channels(env, packet)
    v_norm = float(np.clip(env.v / max(1e-6, env.v_max), -1.0, 1.0))
    yaw_norm = float(np.clip(env.yaw / max(1e-6, env.yaw_cap), -1.0, 1.0))

    core = np.array([
        env.xf / (env.size - 1),
        env.yf / (env.size - 1),
        math.sin(env.heading),
        math.cos(env.heading),
        tof_l, d_l, d_f, d_r, tof_r,
        v_norm, yaw_norm,
        float(env.prev_throttle),
        float(env.prev_turn),
    ], dtype=np.float32)

    local_patch, cone_free, cone_ob_d = build_live_cone_features(world_map, env, cone_geom)
    obs = np.concatenate([core, cone_free, cone_ob_d]).astype(np.float32)
    return obs, local_patch, cone_free, cone_ob_d


def hybrid_substep(env, desired_speed, desired_yaw_rate):
    env._update_speed(desired_speed)
    env._yaw_update(desired_yaw_rate)
    heading_next = (env.heading + env.yaw * env.dt) % (2.0 * math.pi)
    nfx = env.xf + math.cos(heading_next) * env.v * env.dt
    nfy = env.yf + math.sin(heading_next) * env.v * env.dt

    if nfx < 0.0:
        nfx = 0.0
        env.v = 0.0
    elif nfx >= env.size:
        nfx = env.size - 1e-3
        env.v = 0.0

    if nfy < 0.0:
        nfy = 0.0
        env.v = 0.0
    elif nfy >= env.size:
        nfy = env.size - 1e-3
        env.v = 0.0

    env.xf = float(nfx)
    env.yf = float(nfy)
    env.heading = float(heading_next)
    mark_visited_patch(env)
    env.t += 1
    PATH_HISTORY.append((env.xf, env.yf))
    return False


def to_px(xf, yf):
    return int(yf * SCALE), int(xf * SCALE)


def draw_text(surface, text, pos, color=(255, 255, 255), size=16):
    font = pygame.font.SysFont(None, size)
    surface.blit(font.render(text, True, color), pos)


def draw_seen_map(screen, env):
    h = w = env.size
    img = np.zeros((h, w, 3), dtype=np.uint8)

    kv = env.known_visited.astype(bool)
    img[kv] = [220, 220, 220]

    kf = env.known_free.astype(bool) & (~kv)
    img[kf] = [70, 150, 170]

    prob = 1.0 / (1.0 + np.exp(-env.log_prob.astype(np.float32)))
    occ = prob > 0.55
    if np.any(occ):
        intensity = ((prob - 0.55) / 0.45).clip(0.0, 1.0)
        red = (40 + 215 * intensity).astype(np.uint8)
        img[occ] = np.stack([
            red[occ],
            np.zeros(np.count_nonzero(occ), dtype=np.uint8),
            np.zeros(np.count_nonzero(occ), dtype=np.uint8)
        ], axis=1)

    img[0, :, :] = [255, 0, 0]
    img[-1, :, :] = [255, 0, 0]
    img[:, 0, :] = [255, 0, 0]
    img[:, -1, :] = [255, 0, 0]

    img_whc = np.transpose(img, (1, 0, 2))
    surface = pygame.surfarray.make_surface(img_whc)
    surface = pygame.transform.scale(surface, (env.size * SCALE, env.size * SCALE))
    screen.blit(surface, (0, 0))


def draw_overlay(screen, env):
    if len(PATH_HISTORY) >= 2:
        pts = [to_px(x, y) for x, y in PATH_HISTORY]
        pygame.draw.lines(screen, (0, 255, 0), False, pts, 2)

    sx, sy = to_px(env.xf, env.yf)
    pygame.draw.circle(screen, (0, 120, 255), (sx, sy), max(3, SCALE // 2))
    hx = env.xf + math.cos(env.heading) * 3.0
    hy = env.yf + math.sin(env.heading) * 3.0
    px, py = to_px(hx, hy)
    pygame.draw.line(screen, (255, 255, 255), (sx, sy), (px, py), 2)


def draw_live_sensor_wedges(screen, env, packet):
    _, _, _, _, _, us1, us2, us3, us4, us5 = packet_to_normalised_direct_channels(env, packet)
    overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)

    metric = {"US1": us1, "US2": us2, "US3": us3, "US4": us4, "US5": us5}
    colors = {
        "US1": (255, 0, 255, 70),
        "US2": (0, 255, 255, 70),
        "US3": (255, 165, 0, 70),
        "US4": (0, 255, 255, 70),
        "US5": (255, 0, 255, 70),
    }

    for key, sensor in SENSORS.items():
        dist = max(0.25, float(metric[key]))
        sensor_x, sensor_y, sensor_bearing = sensor.pose_in_world(env.xf, env.yf, env.heading)
        sx, sy = to_px(sensor_x, sensor_y)
        pts = [(sx, sy)]
        steps = 18
        for i in range(steps + 1):
            a = sensor_bearing - sensor.half_width + (2.0 * sensor.half_width) * (i / steps)
            ex = sensor_x + math.cos(a) * dist
            ey = sensor_y + math.sin(a) * dist
            px, py = to_px(ex, ey)
            pts.append((px, py))
        if len(pts) >= 3:
            pygame.draw.polygon(overlay, colors[key], pts)

    screen.blit(overlay, (0, 0))
    return us3, us2, us4, us1, us5


def render_heatmap_surface(grid, target_size):
    clipped = np.clip(grid, WORLD_MAP_MIN, WORLD_MAP_MAX)
    norm = (clipped - WORLD_MAP_MIN) / (WORLD_MAP_MAX - WORLD_MAP_MIN)
    img = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
    img[..., 0] = (norm * 255).astype(np.uint8)
    img[..., 1] = (np.clip(1.0 - np.abs(norm - 0.5) * 2.0, 0.0, 1.0) * 120).astype(np.uint8)
    img[..., 2] = (np.clip(1.0 - norm, 0.0, 1.0) * 180).astype(np.uint8)
    img_screen = np.flipud(img)
    img_whc = np.transpose(img_screen, (1, 0, 2))
    surf = pygame.surfarray.make_surface(img_whc)
    return pygame.transform.scale(surf, target_size)


def draw_local_map_panel(panel, local_patch, cone_free, cone_ob_d):
    panel.fill((20, 20, 20))
    w, h = panel.get_size()
    draw_text(panel, "Live local occupancy map", (12, 10), size=22)
    map_h = min(h - 150, w - 24)
    map_w = w - 24
    map_surface = render_heatmap_surface(local_patch, (map_w, map_h))
    panel.blit(map_surface, (12, 40))
    cx = 12 + map_w // 2
    cy = 40 + map_h // 2
    pygame.draw.circle(panel, (255, 255, 255), (cx, cy), 4)
    pygame.draw.line(panel, (255, 255, 255), (cx, cy), (cx, cy - 30), 2)
    draw_text(panel, f"local radius: {local_patch.shape[0]//2 * LOCAL_SAMPLE_RES_M:.1f} m", (12, 50 + map_h), (200, 200, 200), 16)
    draw_text(panel, f"mean cone free: {float(np.mean(cone_free)):.3f}", (12, 74 + map_h), size=17)
    draw_text(panel, f"mean cone obstacle distance: {float(np.mean(cone_ob_d)):.3f}", (12, 98 + map_h), size=17)


def draw_side_panel(panel, env, packet, action, desired_speed, desired_yaw_rate, cone_free, cone_ob_d, serial_rx_count):
    panel.fill((18, 18, 18))
    pad = 16
    _, _, _, _, _, us1, us2, us3, us4, us5 = packet_to_normalised_direct_channels(env, packet)
    draw_text(panel, "Hybrid live-sensor test", (pad, pad), size=22)
    draw_text(panel, "World memory + local cone crop", (pad, pad + 24), (180, 180, 180), 16)
    y = pad + 70
    lines = [
        f"serial sensor packets: {serial_rx_count}",
        f"x: {env.xf:6.2f}   y: {env.yf:6.2f}",
        f"heading: {math.degrees(env.heading):6.1f} deg",
        f"u_meas: {env.v:+6.3f} m/s",
        f"r_meas: {env.yaw:+6.3f} rad/s",
        "",
        f"a_speed: {float(action[0]):+6.3f}",
        f"a_yaw:   {float(action[1]):+6.3f}",
        f"cmd_v:   {desired_speed:+6.3f} m/s",
        f"cmd_r:   {desired_yaw_rate:+6.3f} rad/s",
        "",
        f"US1 / TOF_L proxy: {us1:5.2f} m",
        f"US2 / LEFT:        {us2:5.2f} m",
        f"US3 / FRONT:       {us3:5.2f} m",
        f"US4 / RIGHT:       {us4:5.2f} m",
        f"US5 / TOF_R proxy: {us5:5.2f} m",
        "",
        f"cone_free min/max: {float(np.min(cone_free)):.2f} / {float(np.max(cone_free)):.2f}",
        f"cone_ob_d min/max: {float(np.min(cone_ob_d)):.2f} / {float(np.max(cone_ob_d)):.2f}",
    ]
    for line in lines:
        draw_text(panel, line, (pad, y), size=18)
        y += 22
    draw_text(panel, "Recent control samples:", (pad, y + 8), size=18)
    y += 34
    for item in DEBUG_HISTORY:
        draw_text(panel, item, (pad, y), (200, 200, 200), 16)
        y += 18


class SharedSerialLink:
    def __init__(self, port, baudrate=115200, timeout=0.0, unit_scale_to_m=0.01, min_m=0.02, max_m=SENSOR_MAX_M):
        self.ser = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
        self.unit_scale_to_m = unit_scale_to_m
        self.min_m = min_m
        self.max_m = max_m
        self.rx_count = 0
        time.sleep(2.0)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    def send_control(self, a_speed, a_yaw, u_meas, r_meas):
        line = f"{a_speed:.4f},{a_yaw:.4f},{u_meas:.4f},{r_meas:.4f}\n"
        self.ser.write(line.encode("utf-8"))

    def poll_latest_sensor_packet(self, max_lines=50):
        latest = None
        lines_done = 0
        while self.ser.in_waiting > 0 and lines_done < max_lines:
            raw = self.ser.readline()
            if not raw:
                break
            try:
                line = raw.decode("utf-8", errors="ignore").strip()
            except Exception:
                continue
            if not line:
                continue
            data_raw = parse_csv_line(line)
            if data_raw is None:
                continue
            ranges_m = {}
            for key, raw_value in data_raw.items():
                raw_m = raw_value * self.unit_scale_to_m
                if self.min_m <= raw_m <= self.max_m:
                    ranges_m[key] = raw_m
                else:
                    ranges_m[key] = None
            latest = SensorPacket(rx_time=time.time(), ranges_m=ranges_m)
            self.rx_count += 1
            lines_done += 1
        return latest

    def close(self):
        if self.ser is not None and self.ser.is_open:
            self.ser.close()


def main():
    base_env = make_env()
    dummy_venv = DummyVecEnv([lambda: base_env])
    venv = load_vecnormalize(dummy_venv)
    model = load_model(venv)

    hard_reset_env(base_env)
    world_map = reset_world_map(base_env)
    cone_geom = build_cone_geometry(int(base_env.N_cones), float(base_env.r_cone), LOCAL_SAMPLE_RES_M)

    pygame.init()
    map_px = base_env.size * SCALE
    screen_w = map_px + PANEL_GAP + PANEL_W + PANEL_GAP + PANEL_W
    screen = pygame.display.set_mode((screen_w, map_px))
    clock = pygame.time.Clock()
    pygame.display.set_caption("Hybrid RL single-port live-sensor viewer")

    link = SharedSerialLink(COM_PORT, BAUD_RATE) if USE_ESP32_SERIAL else None

    last_packet = EMPTY_PACKET
    tick_count = 0
    action = np.zeros(2, dtype=np.float32)
    desired_speed = 0.0
    desired_yaw_rate = 0.0
    cone_free = np.zeros(base_env.N_cones, dtype=np.float32)
    cone_ob_d = np.ones(base_env.N_cones, dtype=np.float32)
    local_patch = np.zeros((len(cone_geom["y_coords"]), len(cone_geom["x_coords"])), dtype=np.float32)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    hard_reset_env(base_env)
                    world_map = reset_world_map(base_env)
                    action[:] = 0.0
                    desired_speed = 0.0
                    desired_yaw_rate = 0.0

        if link is not None:
            packet = link.poll_latest_sensor_packet()
            if packet is not None:
                last_packet = packet
                update_world_hit_map(world_map, base_env, packet)

        if tick_count % TICKS_PER_ACTION == 0:
            raw_obs, local_patch, cone_free, cone_ob_d = build_live_observation(base_env, world_map, cone_geom, last_packet)
            norm_obs = venv.normalize_obs(raw_obs.reshape(1, -1))
            action, _ = model.predict(norm_obs, deterministic=True)
            action = np.asarray(action).reshape(-1).astype(np.float32)
            base_env.prev_throttle = float(np.clip(action[0], -1.0, 1.0))
            base_env.prev_turn = float(np.clip(action[1], -1.0, 1.0))
            desired_speed, desired_yaw_rate = base_env._translate_actions(action)
            desired_speed, desired_yaw_rate = base_env._apply_command_limits(desired_speed, desired_yaw_rate)

        hybrid_substep(base_env, desired_speed, desired_yaw_rate)
        DEBUG_HISTORY.appendleft(
            f"as={float(action[0]):+0.2f} ay={float(action[1]):+0.2f}  u={base_env.v:+0.2f} r={base_env.yaw:+0.2f}"
        )

        if link is not None:
            link.send_control(float(action[0]), float(action[1]), float(base_env.v), float(base_env.yaw))

        screen.fill((10, 10, 10))
        draw_seen_map(screen, base_env)
        draw_overlay(screen, base_env)
        d_f, d_l, d_r, _, _ = draw_live_sensor_wedges(screen, base_env, last_packet)

        side_panel = pygame.Surface((PANEL_W, map_px))
        draw_side_panel(side_panel, base_env, last_packet, action, desired_speed, desired_yaw_rate, cone_free, cone_ob_d, 0 if link is None else link.rx_count)
        screen.blit(side_panel, (map_px + PANEL_GAP, 0))

        local_panel = pygame.Surface((PANEL_W, map_px))
        draw_local_map_panel(local_panel, local_patch, cone_free, cone_ob_d)
        screen.blit(local_panel, (map_px + PANEL_GAP + PANEL_W + PANEL_GAP, 0))

        pygame.display.set_caption(
            f"Hybrid live test | a_speed:{float(action[0]):+0.3f} a_yaw:{float(action[1]):+0.3f} | "
            f"u:{base_env.v:+0.3f} yaw:{base_env.yaw:+0.3f} | "
            f"Front:{d_f:.2f} Left:{d_l:.2f} Right:{d_r:.2f}"
        )
        pygame.display.flip()
        clock.tick(CONTROL_HZ)
        tick_count += 1

    if link is not None:
        link.close()
    pygame.quit()


if __name__ == "__main__":
    main()
