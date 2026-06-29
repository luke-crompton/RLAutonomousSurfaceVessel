import os
import math
import time
import numpy as np
import pygame
import serial
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from Lake_environment import LakeMapEnv
from Configuration import SimConfig
from collections import deque

debug_history = deque(maxlen=12) #commanded yaw and speed

LOG_DIR = "runs/ppo_lake"
MODEL_BEST_PATH = os.path.join(LOG_DIR, "best_model")
VECNORM_PATH = os.path.join(LOG_DIR, "vecnorm.1")

SERIAL_PORT = "COM5"      # change this to your ESP32 port
BAUD_RATE = 115200
SEND_TO_ESP32 = False

CONTROL_HZ = 20
ACTION_HOLD_STEPS = 4     # 20 Hz control, new RL action every 4 ticks = 5 Hz action

SCALE   = 10    # pixels per grid cell
PANEL_W = 340   # right-side panel width

# -------------------
# Env / model
# -------------------
def make_env():
    # N_cones comes from SimConfig; change there if needed (e.g., 16)
    env = LakeMapEnv(SimConfig(size=64, obstacle_p=0.01, algae_p=0.00, max_steps=int(10000000)))
    return env

def load_VECNormalize(venv):
    venv = VecNormalize.load(VECNORM_PATH, venv)
    if isinstance(venv, VecNormalize):
        venv.training = False
        venv.norm_reward = False
    return venv

def load_model(venv):
    return PPO.load(MODEL_BEST_PATH, env=venv, device="cpu")

# -------------------
# Coord helpers
# -------------------
def to_px(xf, yf):
    """Convert (row=x, col=y) -> screen pixels (x,y)."""
    return int(yf * SCALE), int(xf * SCALE)

def ray_endpoint(env, start_xf, start_yf, angle, max_cells, stop_on_obstacle=True):
    """March 1 cell at a time; stop on obstacle/edge; return (end_xf, end_yf, steps)."""
    px, py = float(start_xf), float(start_yf)
    dx, dy = math.cos(angle), math.sin(angle)
    endx, endy = px, py
    steps = 0
    for _ in range(int(max_cells)):
        px += dx
        py += dy
        ix, iy = int(round(px)), int(round(py))
        if ix < 0 or iy < 0 or ix >= env.size or iy >= env.size:
            break
        if stop_on_obstacle and env.obstacle[ix, iy]:
            break
        endx, endy = px, py
        steps += 1
    return endx, endy, steps

def _draw_text(surface, text, pos, color=(255, 255, 255), size=16):
    font = pygame.font.SysFont(None, size)
    surface.blit(font.render(text, True, color), pos)

# -------------------
# "Seen-only" map layer with object probabilities
# -------------------
def draw_seen_map(screen, env):
    """
    Draw the boat's perceived world + object probabilities:

      - known_visited: white
      - known_free: cyan tint
      - known_al_visited: green
      - obstacle belief (from log_prob): faint red (low p) to full red (high p)

    NOTE: This does NOT draw ground-truth env.obstacle, only the belief.
    """
    H = W = env.size
    img = np.zeros((H, W, 3), dtype=np.uint8)

    # visited = white
    if getattr(env, "known_visited", None) is not None:
        kv = env.known_visited.astype(bool)
        img[kv] = [255, 255, 255]

    # free space the boat has inferred (overlay, additive cyan)
    if getattr(env, "known_free", None) is not None:
        kf = env.known_free.astype(bool)
        cyan = np.array([0, 120, 120], dtype=np.uint8)
        img[kf] = np.maximum(img[kf], cyan)

    #if getattr(env, "obstacle", None) is not None:
        #img[env.obstacle] = [255, 0 ,0]

    # --- Object probability from log-odds (log_prob) ---
    prob_map = None
    if hasattr(env, "log_prob") and env.log_prob is not None:
        # log-odds -> probability
        lp = np.asarray(env.log_prob, dtype=np.float32)
        prob_map = 1.0 / (1.0 + np.exp(-lp))

    if prob_map is not None:
        # clip 0..1, map to 0..255 intensity
        p = np.clip(prob_map, 0.0, 1.0)
        # optional: slight gamma to make mid values more visible
        # p = p**0.7
        red_layer = (p * 255.0).astype(np.uint8)

        # overlay into red channel so other colours still show
        img[..., 0] = np.maximum(img[..., 0], red_layer)

    # to screen
    img_whc = np.transpose(img, (1, 0, 2))
    surface = pygame.surfarray.make_surface(img_whc)
    surface = pygame.transform.scale(surface, (env.size * SCALE, env.size * SCALE))
    screen.blit(surface, (0, 0))


# -------------------
# Sensors on map
# -------------------
def draw_filled_fan(surface, origin, center_angle, radius_cells, half_angle_rad, rgba, step_deg=5):
    if radius_cells <= 0.1 or half_angle_rad <= 0.0:
        return
    steps = max(2, int((2 * half_angle_rad) / math.radians(step_deg)))
    ox, oy = origin
    pts = [(ox, oy)]
    # use global copies of the env position (set by caller)
    for i in range(steps + 1):
        a = center_angle - half_angle_rad + (2 * half_angle_rad) * (i / steps)
        ex = env_x + math.cos(a) * radius_cells
        ey = env_y + math.sin(a) * radius_cells
        px, py = to_px(ex, ey)
        pts.append((px, py))
    if len(pts) >= 3:
        pygame.draw.polygon(surface, rgba, pts)

def draw_overlay(screen, env):
    sx, sy = to_px(env.xf, env.yf)
    pygame.draw.circle(screen, (0, 120, 255), (sx, sy), max(2, SCALE // 2))

    # thin look-ahead line
    k = getattr(env.cfg, "look_ahead", 8)
    ex, ey, _ = ray_endpoint(env, env.xf, env.yf, env.heading, k)
    x2, y2 = to_px(ex, ey)
    pygame.draw.line(screen, (220, 0, 220), (sx, sy), (x2, y2), width=2)

def draw_sensors(screen, env):
    """
    Draw:
      - Side Ultrasonic wedges (±π/6) in cyan
      - TOF left/right wedges (±π/2) in magenta
      - Front radar wedge in orange
    Returns (d_f, d_l, d_r, tof_l, tof_r) as ints.
    """
    pygame.font.init()
    sx, sy = to_px(env.xf, env.yf)

    def _dist_only(v):
        # accept float/int, or tuple/list like (dist, ...); return numeric distance
        if isinstance(v, (list, tuple)) and len(v) > 0:
            return v[0]
        return v

    def _safe(v):
        try:
            v = float(_dist_only(v))
            return 0.0 if (math.isnan(v) or math.isinf(v)) else v
        except Exception:
            return 0.0

    global env_x, env_y
    env_x, env_y = env.xf, env.yf

    overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)

    # --- Side Ultrasonic (±π/6) as filled wedges (80° total) ---
    us_offset = math.pi / 6.0                # centerlines: ±30°
    us_half_width = math.radians(40.0)       # 80° cone width (half = 40°)

    left_angle  = env.heading - us_offset
    right_angle = env.heading + us_offset

    d_l  = _safe(getattr(env, "us_ray_fan", lambda *_: 0.0)(left_angle,  count_unvisited=False))
    d_r  = _safe(getattr(env, "us_ray_fan", lambda *_: 0.0)(right_angle, count_unvisited=False))

    draw_filled_fan(overlay, (sx, sy), left_angle,  max(1.0, d_l), us_half_width, (0, 255, 255, 80))
    draw_filled_fan(overlay, (sx, sy), right_angle, max(1.0, d_r), us_half_width, (0, 255, 255, 80))

    # outline US centerlines
    lx, ly, _ = ray_endpoint(env, env.xf, env.yf, left_angle,  max(1, int(d_l)))
    rx, ry, _ = ray_endpoint(env, env.xf, env.yf, right_angle, max(1, int(d_r)))
    lpx, lpy = to_px(lx, ly)
    rpx, rpy = to_px(rx, ry)
    pygame.draw.line(screen, (0, 255, 255), (sx, sy), (lpx, lpy), width=2)
    pygame.draw.line(screen, (0, 255, 255), (sx, sy), (rpx, rpy), width=2)
    _draw_text(screen, f"US L:{int(d_l)}", (lpx + 6, lpy - 14), (0, 255, 255), 16)
    _draw_text(screen, f"US R:{int(d_r)}", (rpx + 6, rpy - 14), (0, 255, 255), 16)

    # --- TOF sensors: directly left/right (±π/2) ---
    tof_offset = math.pi / 2.0  # 90°
    tof_left_angle  = env.heading - tof_offset
    tof_right_angle = env.heading + tof_offset

    tof_l = _safe(getattr(env, "tof_ray_fan", lambda *_: 0.0)(tof_left_angle,  count_unvisited=False))
    tof_r = _safe(getattr(env, "tof_ray_fan", lambda *_: 0.0)(tof_right_angle, count_unvisited=False))

    raw_tof_angle = getattr(env, "tof_angle", getattr(env, "TOF_angle", 10.0))
    try:
        tof_angle_rad = math.radians(float(raw_tof_angle))
    except Exception:
        tof_angle_rad = math.radians(10.0)
    tof_half_width = tof_angle_rad / 2.0

    draw_filled_fan(overlay, (sx, sy), tof_left_angle,  max(1.0, tof_l), tof_half_width, (255, 0, 255, 80))
    draw_filled_fan(overlay, (sx, sy), tof_right_angle, max(1.0, tof_r), tof_half_width, (255, 0, 255, 80))

    # outline TOF centerlines
    tlx, tly, _ = ray_endpoint(env, env.xf, env.yf, tof_left_angle,  max(1, int(tof_l)))
    trx, try_, _ = ray_endpoint(env, env.xf, env.yf, tof_right_angle, max(1, int(tof_r)))
    tlpx, tlpy = to_px(tlx, tly)
    trpx, trpy = to_px(trx, try_)
    pygame.draw.line(screen, (255, 0, 255), (sx, sy), (tlpx, tlpy), width=2)
    pygame.draw.line(screen, (255, 0, 255), (sx, sy), (trpx, trpy), width=2)
    _draw_text(screen, f"TOF L:{int(tof_l)}", (tlpx + 6, tlpy - 14), (255, 0, 255), 16)
    _draw_text(screen, f"TOF R:{int(tof_r)}", (trpx + 6, trpy - 14), (255, 0, 255), 16)

    # blit all wedges
    screen.blit(overlay, (0, 0))

    # --- Front Radar (wedge width from env.radar_angle) ---
    d_f = _safe(getattr(env, "radar_ray_fan", lambda *_: 0.0)(env.heading, count_unvisited=False))
    raw_angle = getattr(env, "radar_angle", 60.0)
    try:
        radar_angle_rad = math.radians(float(raw_angle))
    except Exception:
        radar_angle_rad = math.radians(60.0)

    cx = env.xf + math.cos(env.heading) * max(1.0, d_f)
    cy = env.yf + math.sin(env.heading) * max(1.0, d_f)
    cpx, cpy = to_px(cx, cy)
    pygame.draw.line(screen, (255, 165, 0), (sx, sy), (cpx, cpy), width=2)
    _draw_text(screen, f"RAD:{int(d_f)}", (cpx + 6, cpy - 14), (255, 165, 0), 16)

    if radar_angle_rad >= math.radians(1.0) and d_f >= 0.5:
        half = radar_angle_rad / 2.0
        steps = max(2, int(radar_angle_rad / (math.pi / 36)))  # ~5° step
        pts = [(sx, sy)]
        for i in range(steps + 1):
            a = env.heading - half + (radar_angle_rad * i / steps)
            ex = env.xf + math.cos(a) * max(1.0, d_f)
            ey = env.yf + math.sin(a) * max(1.0, d_f)
            px, py = to_px(ex, ey)
            pts.append((px, py))
        if len(pts) >= 3:
            wedge = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
            pygame.draw.polygon(wedge, (255, 165, 0, 64), pts)
            screen.blit(wedge, (0, 0))

    return int(d_f), int(d_l), int(d_r), int(tof_l), int(tof_r)

# -------------------
# Cone data from env.compute_cones()
# -------------------
def compute_cone_panel_data(env):
    """
    Ask the environment for its cone metrics:
      - unvisited_frac[k] in [0,1]
      - cone_dist_norm[k] in [0,1] (normalised distance to nearest obstacle)
    If anything fails, return zeros so the panel still draws.
    """
    N = int(getattr(env, "N_cones", 16))
    try:
        if hasattr(env, "compute_cones"):
            unvisited_frac, cone_dist_norm = env.compute_cones()
            unvisited_frac = np.asarray(unvisited_frac, dtype=np.float32)
            cone_dist_norm = np.asarray(cone_dist_norm, dtype=np.float32)
            if unvisited_frac.shape[0] != N:
                unvisited_frac = np.zeros(N, dtype=np.float32)
            if cone_dist_norm.shape[0] != N:
                cone_dist_norm = np.ones(N, dtype=np.float32)
        else:
            unvisited_frac = np.zeros(N, dtype=np.float32)
            cone_dist_norm = np.ones(N, dtype=np.float32)
    except Exception:
        unvisited_frac = np.zeros(N, dtype=np.float32)
        cone_dist_norm = np.ones(N, dtype=np.float32)

    # clip to 0..1 just in case
    unvisited_frac = np.clip(unvisited_frac, 0.0, 1.0)
    cone_dist_norm = np.clip(cone_dist_norm, 0.0, 1.0)
    return unvisited_frac, cone_dist_norm

# -------------------
# Sensor / cone panel
# -------------------
def draw_sensor_panel(panel, env, d_f, d_l, d_r, tof_l, tof_r, debug_history= None):
    panel.fill((18, 18, 18))
    W, H = panel.get_size()
    pad = 16
    _draw_text(panel, "Cones: Unvisited & Obstacles", (pad, pad), (255, 255, 255), 18)

    # Cones (allocentric)
    N = int(getattr(env, "N_cones", 16))
    dtheta = 2 * math.pi / N
    cx = W // 2
    cy = pad + 140
    outer_r = 110
    ring_t = 18
    gap = 3

    # Get cone unvisited + obstacle distance (from env.compute_cones)
    unvisited_frac, cone_dist_norm = compute_cone_panel_data(env)
    # Obstacle proximity = 1 - normalised distance
    obstacle_close = 1.0 - cone_dist_norm
    obstacle_close = np.clip(obstacle_close, 0.0, 1.0)

    # Two rings:
    #   outer: unvisited fraction (cyan)
    #   inner: obstacle proximity (red)
    rings = [
        (unvisited_frac, outer_r,                      (0, 200, 255, 150), "Unvisited"),
        (obstacle_close, outer_r - (ring_t + gap),     (255, 50, 50, 150), "Obstacle proximity"),
    ]

    surf = pygame.Surface(panel.get_size(), pygame.SRCALPHA)
    for values, radius, color, _name in rings:
        if radius <= 12 or radius - ring_t <= 6:
            continue
        for i, v in enumerate(values):
            a = max(0, min(255, int(25 + 230 * float(v))))
            col = (color[0], color[1], color[2], a)
            theta0 = 0 + i * dtheta
            theta1 = theta0 + dtheta

            pts = []
            steps = max(6, int(dtheta / (math.pi / 48)))
            for k in range(steps + 1):
                t = theta0 + (theta1 - theta0) * (k / steps)
                x = cx + radius * math.cos(t)
                y = cy - radius * math.sin(t)
                pts.append((x, y))
            inner_r = radius - ring_t
            for k in range(steps + 1):
                t = theta1 - (theta1 - theta0) * (k / steps)
                x = cx + inner_r * math.cos(t)
                y = cy - inner_r * math.sin(t)
                pts.append((x, y))
            if len(pts) >= 3:
                pygame.draw.polygon(surf, col, pts)

    panel.blit(surf, (0, 0))




    # Heading arrow (allocentric)
    head_len = outer_r + 8
    vx = math.sin(env.heading)
    vy = -math.cos(env.heading)
    hx = cx + head_len * vx
    hy = cy + head_len * vy
    pygame.draw.line(panel, (255, 255, 255), (cx, cy), (hx, hy), 2)
    pygame.draw.circle(panel, (255, 255, 255), (cx, cy), 3)
    _draw_text(panel, "heading", (int(hx) - 28, int(hy) - 18), (255, 255, 255), 14)

    # Legends
    legx = pad
    legy = cy + outer_r + 14
    _draw_text(panel, "Rings:", (legx, legy), (200, 200, 200), 16)
    _draw_text(panel, "Outer  = Unvisited fraction",   (legx + 72, legy),      (0, 200, 255), 16)
    _draw_text(panel, "Inner  = Obstacle proximity",   (legx + 72, legy + 18), (255, 80, 80), 16)

    # Distance bars
    bars_y = legy + 70
    _draw_text(panel, "Raw sensor distances (cells):", (pad, bars_y), (255, 255, 255), 16)
    meter_x = pad
    meter_w = W - 2 * pad
    meter_h = 14

    def bar(label, val, yoff, color, vmax):
        vmax = max(1, int(vmax))
        fill = int(meter_w * min(val, vmax) / vmax)
        pygame.draw.rect(panel, (60, 60, 60), pygame.Rect(meter_x, yoff, meter_w, meter_h))
        pygame.draw.rect(panel, color,        pygame.Rect(meter_x, yoff, fill,    meter_h))
        _draw_text(panel, f"{label}: {int(val)}", (meter_x, yoff - 18), (200, 200, 200), 14)

    us_max  = getattr(env, "us_ray_max", 15)
    radar_m = getattr(env, "radar_max",  30)
    tof_m   = getattr(env, "tof_max",    getattr(env, "TOF_max", 10))

    bar("US Left",   d_l,   bars_y + 30,  (0, 200, 255), us_max)
    bar("US Right",  d_r,   bars_y + 60,  (0, 200, 255), us_max)
    bar("TOF Left",  tof_l, bars_y + 90,  (255, 0, 255), tof_m)
    bar("TOF Right", tof_r, bars_y + 120, (255, 0, 255), tof_m)
    bar("Radar",     d_f,   bars_y + 150, (255, 165, 0), radar_m)

# -------------------
# Main loop
# -------------------
def main():
    base_env = make_env()
    venv = DummyVecEnv([lambda: base_env])
    venv = load_VECNormalize(venv)
    model = load_model(venv)
    obs = venv.reset()

    pygame.init()
    map_px = base_env.size * SCALE
    screen = pygame.display.set_mode((map_px + PANEL_W, map_px))
    clock = pygame.time.Clock()
    pygame.display.set_caption("RL viewing (prob map + cones)")

    ser = None
    if SEND_TO_ESP32:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
        time.sleep(2.0)  # allow ESP32 to reset after opening serial

    tick_count = 0
    action = None

    pending_samples = []
    last_info = None
    done_flag = False

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        if not pending_samples:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = venv.step(action)

            core_env = venv.venv if isinstance(venv, VecNormalize) else venv
            lake_env = core_env.envs[0]

            pending_samples = list(lake_env.substep_telemetry)
            last_info = info
            done_flag = done[0]

        # unwrap to raw env
        core_env = venv.venv if isinstance(venv, VecNormalize) else venv
        lake_env = core_env.envs[0]

        a_speed = float(action[0][0])
        a_yaw = float(action[0][1])

        if pending_samples:
            sample = pending_samples.pop(0)
            u_meas = float(sample["v"])
            r_meas = float(sample["yaw"])
        else:
            u_meas = float(lake_env.v)
            r_meas = float(lake_env.yaw)

        debug_history.appendleft(
            f"as={a_speed:+.2f} ay={a_yaw:+.2f} u={u_meas:+.2f} r={r_meas:+.2f}"
        )

        if ser is not None and ser.is_open:
            line = f"{a_speed:.4f},{a_yaw:.4f},{u_meas:.4f},{r_meas:.4f}\n"
            ser.write(line.encode("utf-8"))


        tick_count += 1

        # left: draw map with known areas + object probabilities
        draw_seen_map(screen, lake_env)

        # on-map overlay + sensors
        draw_overlay(screen, lake_env)
        d_f, d_l, d_r, tof_l, tof_r = draw_sensors(screen, lake_env)

        # right panel
        panel_surface = pygame.Surface((PANEL_W, map_px))
        draw_sensor_panel(panel_surface, lake_env, d_f, d_l, d_r, tof_l, tof_r, debug_history)
        screen.blit(panel_surface, (map_px, 0))

        # caption
        info_for_display = last_info[0] if last_info is not None else {}
        cov = info_for_display.get("coverage", 0.0)
        algae_seen = info_for_display.get("algae seen", 0)

        pygame.display.set_caption(
            f"Cov:{cov:.1%} | Algae:{algae_seen} | "
            f"a_speed:{a_speed:+.3f} a_yaw:{a_yaw:+.3f} | "
            f"u:{u_meas:+.3f} yaw:{r_meas:+.3f} | "
            f"Radar:{d_f} US(L:{d_l},R:{d_r}) TOF(L:{tof_l},R:{tof_r})"
        )

        pygame.display.flip()
        clock.tick(100)

        if done_flag and not pending_samples:
            obs = venv.reset()
            action = None
            last_info = None
            done_flag = False

    if ser is not None and ser.is_open:
        ser.close()
    pygame.quit()

if __name__ == "__main__":
    main()
