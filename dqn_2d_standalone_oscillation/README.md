# DQN 2D Standalone — Isaac-Lab-Free Training

A standalone, dependency-light version of this project's DQN training loop,
running against a pure-2D kinematic environment instead of the full Isaac
Lab simulation. No Isaac Sim install, no GPU, and no macOS/Linux-specific
setup required -- just Python and a few pip packages.

## Why this exists

The main Drone-Ranger-RL project trains the same DQN algorithm inside
Isaac Lab, which requires a full Isaac Sim installation and is Linux/GPU-only.
This 2D version exists to isolate the DQN algorithm, network, action
space, and reward design from any simulation-layer detail (physics
stepping, attitude control dynamics, rendering) -- useful both for
teammates without an Isaac Lab setup (e.g. on a Mac) and for quickly
iterating on reward/algorithm changes (a 1500-episode run takes under 10
minutes on CPU here, versus roughly an hour in Isaac Lab).

The environment (`dqn_2d_env.py`) uses simple kinematic integration
(position += velocity * dt, heading += yaw_rate * dt) with the exact same
control frequency, physical speed/turn-rate parameters, and reward
structure as the full Isaac Lab environment (Stage 0: pure navigation, no
obstacles) -- see the main README's "Known Issues & Findings" section for
the diagnosed problems this comparison uncovered.

## Installation

Requires Python 3.9+.

```bash
pip install -r requirements.txt
```

That's it -- no Isaac Sim, no CUDA required (training runs on CPU
automatically if no GPU is available).

## Usage

### Training

```bash
python3 train_dqn.py --num_envs=64 --num_episodes=1500 --run_name=my_run
```

Checkpoints are saved to `checkpoints/<run_name>_ep<N>.pt` and
`checkpoints/<run_name>_final.pt` (relative to this script's own
directory, not your current working directory -- so it works the same
regardless of where you run it from). TensorBoard logs go to
`tensorboard/<run_name>/`.

Any `DQNConfig` field can be overridden on the command line, e.g.
`--hidden_dim=64 --gamma=0.95`.

- `--resume_from=<path>`: resume from a saved checkpoint's weights.
- `--resume_optimizer_from=<path>`: also restore Adam's optimizer state.
- `--output_dir=<path>`: change where `checkpoints/` and `tensorboard/`
  are created (defaults to this script's own directory).
- `--num_obstacles=<k>` turns on k active obstacles per episode, from 0 (the default, pure navigation) to 5. See [Obstacles and ground-truth distances](#obstacles-and-ground-truth-distances).

### Batch evaluation

```bash
python3 dqn_diagnose_2d.py --checkpoint=checkpoints/my_run_final.pt \
  --num_envs=64 --num_episodes=200 --csv_out=results.csv
```

Runs the policy greedily for N episodes and reports the outcome
distribution (reached / contact_crash / out_of_bounds / timed_out).

### Reproducible evaluation

```bash
python3 eval_2d.py --checkpoint=checkpoints/my_run_final.pt --num_episodes=2000 --seed=0
```

Runs the policy greedily and scores every env on its first episode only.
With the same `--seed` and `--num_obstacles`, every checkpoint therefore faces the same starts, targets and obstacle layouts, which makes two checkpoints comparable episode for episode.
It prints each outcome with a 95% interval, the mean steps to reach, and the reach rate split by start-to-target distance.
`dqn_diagnose_2d.py` keeps its envs running instead, so its later episodes depend on how fast the policy finished the earlier ones.

### Step-by-step trace

```bash
python3 trace_2d.py --checkpoint=checkpoints/my_run_final.pt --max_steps=300 --seed=42
```

Prints every step's chosen action, position, distance, heading alignment,
and reward breakdown for one episode. `--seed` lets you reproduce a
specific start/target pair (useful for revisiting a case you've already
found interesting).
With obstacles it also lists where each obstacle starts, and adds `d_front`, `d_left` and `d_right` columns showing exactly what the network sees.
In coords mode those three columns show the nearest slot `s1_x`, `s1_y` and `v1` instead.
The last row shows where the episode really ended, read from the environment's pre-reset snapshot rather than from the next episode's spawn point.

### Trajectory plot

```bash
python3 plot_2d_trajectory.py --checkpoint=checkpoints/my_run_final.pt \
  --num_episodes=8 --max_steps=1000 --out=trajectory.png
```

Runs several episodes and plots the drone's top-down path with matplotlib
(circle = start, star = target).
Each episode's obstacles are drawn as shaded shapes in that episode's color.

### Q-value sweep

```bash
python3 q_sweep.py --checkpoint=checkpoints/my_run_final.pt \
  --distances 2.0 6.0 10.0 --angle_step_deg 10.0 --device cpu
```

Constructs synthetic states across a grid of heading angles and reports
the network's action choice and decision-margin at each point -- no
environment rollout needed. Useful for checking whether the policy's
left/right turn judgment is symmetric and confident.

## Obstacles and ground-truth distances

`--num_obstacles` adds obstacles to the environment, and `train_dqn.py`, `dqn_diagnose_2d.py`, `eval_2d.py`, `trace_2d.py` and `plot_2d_trajectory.py` all accept it, along with `--distance_mode` and `--k_nearest`.
The default of 0 is Stage 0, pure navigation, and behaves exactly as it did before obstacles existed, down to the random number stream.

The obstacle layout is ported from the Isaac Lab environment's `_reset_idx`.
- The same five obstacles, three upright cylinders and two square boxes, are strung along a random flight direction starting about 2.5 m from the drone.
- The target sits in a gap between two neighboring obstacles that is wide enough to fly through, or past the last obstacle when no gap qualifies.
- On half of the resets the first obstacle is parked far away (rehearsal mixing), so the replay buffer keeps pure-navigation experience.
- Touching an obstacle ends the episode as `contact_crash` with a reward of -50.

`d_front`, `d_left` and `d_right` are not simulated sensor readings.
`obstacle_geometry.py` computes each one exactly from the obstacle shapes, as the distance from the drone's center to the nearest obstacle surface inside a 73 degree horizontal sector facing forward, left or right, capped at 10 m.
The sectors copy the field of view of the Isaac Lab cameras, so bearings between 36.5 and 53.5 degrees off the front axis are blind in both environments.

`test_obstacle_geometry.py` checks these distances against a brute-force reference, and checks that the copy of the same functions inside the Isaac Lab environment file, and the constants both environments feed them, have not drifted.
It also covers the environment side, meaning the obstacle layout, crash detection, the outcome priority and the Stage 0 shortcuts.

```bash
python3 test_obstacle_geometry.py
```

### Camera emulation

`--distance_mode camera` replaces the exact distances with an emulation of the Isaac Lab depth cameras, implemented in `camera_emulation.py`.
- Each camera casts one ray per pixel column, 84 on the front camera and 8 on each side camera, from its mount point 0.25 m off the drone's center, and reports the nearest hit between 0.1 and 10 m.
- A thin obstacle can therefore sit between two side-camera rays and go unseen, which the exact mode never misses.
- Rendering noise, ground-filter errors, attitude tilt and self-occlusion by the airframe are not modeled.
- In Stage 0 both modes read 10 everywhere, so they only differ once obstacles are present.

A checkpoint does not record which mode trained it, so evaluate a camera-trained checkpoint with the same `--distance_mode camera`.
`test_camera_emulation.py` checks the emulation against the exact distances and checks that its camera parameters still match the Isaac Lab environment.

```bash
python3 test_camera_emulation.py
```

### Coordinate mode

`--distance_mode coords` is a diagnostic upper bound with no Isaac Lab counterpart.
It answers whether a failure to avoid obstacles comes from the sensor layout or from the DQN itself.
- The three readings are replaced by `--k_nearest` slots (3 by default) of `(sx, sy, visible)`, nearest obstacle first, followed by `dx` and `dy` as before.
- `sx` and `sy` are the nearest surface point of that obstacle in the body frame, so their length is the exact distance and its bearing is kept.
- There is no sector and so no blind spot.
- An obstacle farther than 10 m, or parked, leaves its slot all zeros with `visible` at 0.
- The state size is `3 * k_nearest + 2` whatever `--num_obstacles` is, so one checkpoint can move through obstacle counts.

A checkpoint does not record its distance mode, so evaluate a coords checkpoint with the same `--distance_mode coords --k_nearest`.
A mismatch usually fails on the input size, but `--k_nearest 1` gives 5 inputs like the other two modes, so a mismatch there loads silently and feeds the network the wrong inputs.
`q_sweep.py` builds 5-input synthetic states and does not support it.

## Files

| File | Purpose |
|---|---|
| `dqn_config.py` | `DQNConfig` dataclass: all network/action/hyperparameter settings, CLI-overridable |
| `dqn_action_space.py` | Shared `QNetwork` and discrete-to-continuous action mapping |
| `dqn_2d_env.py` | The pure-2D kinematic environment |
| `obstacle_geometry.py` | Exact ground-truth obstacle distances, shared with the Isaac Lab environment |
| `test_obstacle_geometry.py` | Tests for `obstacle_geometry.py` and for the obstacle handling in `dqn_2d_env.py`, run as a plain script |
| `camera_emulation.py` | Emulated depth-camera readings for `--distance_mode camera` |
| `test_camera_emulation.py` | Tests for `camera_emulation.py`, run as a plain script |
| `train_dqn.py` | Training loop (this standalone entry point) |
| `dqn_diagnose_2d.py` | Batch evaluation |
| `eval_2d.py` | Greedy evaluation on reproducible layouts, first episode per env |
| `trace_2d.py` | Step-by-step episode trace |
| `plot_2d_trajectory.py` | matplotlib trajectory visualization |
| `q_sweep.py` | Offline decision-boundary sweep |

## Relationship to the main project

This is a subset of the `dqn/` directory in the main Drone-Ranger-RL
repository, with `train_dqn.py` modified to drop the Isaac Lab code path
and use paths relative to the script's own location instead of a
hardcoded absolute path. Everything else (network, action space, reward,
config system) is identical. See the main repository's README for the
full project context, the Isaac Lab environment, and a detailed account
of diagnosed issues and findings.
