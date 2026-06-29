# Autonomous Boat Navigation — PPO Reinforcement Learning

A PPO-trained autonomous navigation agent for an unmanned surface vessel (USV), with a custom Gymnasium environment, real-time sensor integration via ESP32, and deployment onto physical hardware.

![Agent demo](assets/demo.gif)

---

## What it Does

The agent learns to navigate a simulated lake, avoiding obstacles and maximising area coverage using only onboard sensor readings — no GPS, no global map. Once trained, it runs live on a physical boat: an ESP32 reads ultrasonic and ToF sensor data over serial, and the model outputs thrust and yaw commands back to the motors in real time.

The full pipeline covers simulation design, reward shaping, PPO training, sensor modelling, and embedded hardware integration.

---

## Results

Two PPO training runs over **1 million environment steps** across 20 parallel environments.

### Coverage
![Coverage](assets/RLBoat_coverage.png)
Both runs converge to roughly **38–39% area coverage**, with steady improvement from ~3% at the start. The agent learns to systematically explore rather than circle the same area.

### Total Reward
![Total Reward](assets/RLBoat_Trew.png)
Mean per-step reward increases consistently across training, reflecting better exploration efficiency and fewer crashes over time.

### Crash Rate
![Crash Rate](assets/RLBoat_crash.png)
Crash penalty starts at around −0.008 early in training and trends toward zero by 500k steps, showing the agent learns reliable obstacle avoidance as exploration improves.

---

## Observation Space — Directional Cone Map

![Cone Spatial Map](assets/cone_spartial_map.png)

The agent observes its surroundings through **12 equal angular sectors** (cones) radiating from its current position. Each cone returns two values that form part of the observation vector:

- **Green shading (unvisited fraction):** How much unvisited free space lies in that direction — darker green means more unexplored area to the agent's knowledge. This drives the agent toward areas it has not yet covered.
- **Red arcs (obstacle proximity):** The normalised distance to the nearest detected obstacle within that cone. The arc is drawn at the obstacle's position — a red arc close to the centre means an obstacle is nearby in that direction.
- **Blue lines:** The raw normalised obstacle distance along each cone's centreline.

All cone values are rotated into the **agent's egocentric frame** at each step, so cone index 0 always points in the direction the boat is currently heading. This gives the agent a consistent directional reference regardless of world orientation. See `Lake_environment.py → compute_cones()` and `translate_egocentric_cone()` for the implementation.

---

## Simulation Viewer

Run `python View_model.py` to watch the trained agent navigate in real time.

![Viewer Color Code](assets/viewer_color_code.png)

| Colour | Meaning |
|---|---|
| White | Visited cells — areas the agent has physically passed through |
| Grey | Known free space — inferred clear from sensor ray-casting |
| Red | High obstacle probability — log-odds belief map |
| Cyan wedges | Ultrasonic sensor (US) field of view with measured range |
| Magenta wedges | Time-of-Flight (ToF) sensor left/right readings |
| Orange wedge | Forward radar cone |

---

## Real Sensor Integration

The `hardware/` folder contains the Python sensor interface and ESP32 firmware for running the agent on the physical boat.

### Sensor Heatmap
![Sensor Heatmap](assets/sensor_heatmap.png)

Live occupancy heatmap built from 5 ultrasonic sensors. Free space is pushed negative (dark), detected surfaces push positive (bright). The sensor wedges and boat outline are overlaid for reference. Run with `hardware/live_sensor_viewer.py`.

### Hardware Test Setup
![Hardware Test Setup](assets/sensor_test_setup.png)

Physical test bench: the boat surrounded by cardboard tube obstacles for sensor validation before water testing. The ESP32 and sensor array are visible on the deck.

---

## Project Structure

**Simulation / Training** (no hardware required)

| File | Purpose |
|---|---|
| `Lake_environment.py` | Gymnasium RL environment — lake map, ray-cast sensors, reward shaping |
| `Configuration.py` | Simulation config dataclass and ray-line precomputation |
| `mapping.py` | Occupancy grid (`HitMap`) with decay and clamp operations |
| `Training.py` | Train a PPO agent from scratch |
| `REtraining.py` | Resume training from a saved checkpoint |
| `View_model.py` | Visualise the trained agent running in simulation (pygame) |

**Hardware** (requires ESP32 + ultrasonic/ToF sensors)

| File | Purpose |
|---|---|
| `hardware/sensor_class.py` | Ultrasonic sensor geometry (arc/free-space point generation) |
| `hardware/serial_reader.py` | USB serial parser for ESP32 CSV packets |
| `hardware/live_sensor_viewer.py` | Hybrid viewer: real sensor data fused with RL model inference |
| `hardware/combined_sensor_motor/` | Arduino firmware for ESP32 (sensor reading + motor control) |

---

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Train from scratch:**
```bash
python Training.py
```

**Continue training from checkpoint:**
```bash
python REtraining.py
```

**Visualise trained agent (simulation):**
```bash
python View_model.py
```

**Run on hardware (ESP32 connected via USB):**
```bash
python hardware/live_sensor_viewer.py
```
Set `COM_PORT` in `hardware/live_sensor_viewer.py` to match your ESP32's serial port.

**View training curves:**
```bash
tensorboard --logdir runs/ppo_lake
```

---

## Hardware

- ESP32 microcontroller running `hardware/combined_sensor_motor/combined_sensor_motor.ino`
- 5× HC-SR04 ultrasonic sensors
- 2× VL53L0X ToF sensors
- Brushless motors with ESC

## Trained Model

A trained model is included in `runs/ppo_lake/` (`best_model.zip` + `vecnorm.1`). Load it with `View_model.py` or `hardware/live_sensor_viewer.py` without retraining.
