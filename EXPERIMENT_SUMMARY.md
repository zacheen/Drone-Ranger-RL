# DQN Drone Obstacle Avoidance, Experiment Summary

Compiled on 2026-10-22 and brought up to baseline_v6 on 2026-11-02. Every number comes from the 2D kinematic environment in `dqn_2d_standalone_oscillation`, and nothing has been tested in Isaac Lab yet.

## Research goals

### Why a 2D environment

- The Isaac Lab simulation includes physics, attitude control and camera rendering, so training is slow, and when something goes wrong it is hard to tell whether the cause is the DQN itself or the simulation layer.
- So I look for a working DQN design in a 2D kinematic environment that uses the same state, actions and control rate as the Isaac env (0.02 s per step). From baseline_v4 the control step is 0.1 s (see "Why a longer control step trains more stably").

### Design constraints

- The input uses only what a real drone can measure, meaning the obstacle distances within the depth-camera sectors and the target's relative position, never the true obstacle coordinates (see "How my environment differs from RipZou's").
- Training happens in simulation, so a curriculum can start from easy scenarios at no real-world cost.

### Main goals

1. Fix oscillation, meaning swinging left and right in front of an obstacle, spinning in place, or hovering stuck.
2. Fix training instability, meaning a policy that learns well midway and then gets worse.
3. Raise the success rate on the harder benchmark of 8 obstacles, where baseline_v4.2 and baseline_stop_v5 both pass 99% best and 98% final (see "The 8-obstacle benchmark").
4. Learn to turn around and leave a gap that is too narrow, which is why the stop action came back (see "Why the stop action came back").

## How my environment differs from RipZou's

RipZou's `dqn_2d_standalone` on `origin/main` feeds the true obstacle coordinates to the network. I consider direct coordinates information a real drone cannot obtain, so my environment still simulates the sensors, only without noise.

| Item | RipZou (`dqn_2d_standalone`) | Mine (`dqn_2d_standalone_oscillation`) |
|---|---|---|
| Obstacle input | Relative center coordinates and radius (ox, oy, r) of the K nearest visible obstacles | Distance to the nearest obstacle surface in each of three 73 degree sectors, front, left and right |
| State | [dx, dy, ox_1, oy_1, r_1, ...], divided by 10 | [d_front, d_left, d_right, dx, dy], with the front sector split into front-left and front-right and the previous action added from baseline_v4 (see "Why baseline_v4 looks the way it does"), and the sector readings replaced by 10 camera bins from baseline_v6 (see "Why baseline_v6 looks the way it does") |
| Field of view | Every direction within 10 m | Only the three camera sectors, with a blind spot behind, matching the three depth cameras of the Isaac env. From baseline_v6 two cameras cover the front 146 degrees and nothing to the sides |
| Noise | None | None (`ground_truth` mode), with a separate `camera` mode that emulates camera readings |
| Obstacle shape and layout | Circles of radius 0.5 m, rejection-sampled in a 10 m box, with the first one placed on the straight line from start to target | The same, deliberately reusing RipZou's commit 1643e0e so that both versions use the same obstacle environment |

I also tried `coords` mode, which feeds the coordinates of the nearest obstacle surface points directly, as an upper reference with complete information. With 3 obstacles it reached only 63.98% (3000 ep) and 80.10% (6000 ep), both below the 87.6% of the sensor-distance input. Each length has a single run, so it settles nothing, but it shows no sign that the policy fails without coordinates.

## Evaluation

- A checkpoint is saved every 100 episodes, and the one with the highest success on the validation set (seed 1000, 1,000 layouts) is kept. Ties are broken on seed 2000.
- The reported numbers come from 10,000 layouts of seed 777, which was never used to choose anything.
- **best** is the success rate of the selected checkpoint and stands for what the training can reach.
- **final** stands for stability. The larger the gap between best and final, the more the policy degraded late in training.
  - Up to 2026-10-31 it was the success rate of the last checkpoint alone.
  - From 2026-10-31 it is the mean of the last 3 checkpoints, for a 3000-episode stage `ep2800`, `ep2900` and `final`. V4LONGG99PROX r3 scored 99.5 on validation at ep2800, and its last checkpoint alone landed on a dip at 96.62, so one checkpoint could make a stable run look failed.
  - Arms started before that date keep the last checkpoint alone unless they were rescored for a comparison. Every table from "The 8-obstacle benchmark" on marks such a final with †.
- From baseline_v4 on an arm passes when its mean best reaches 99% and its mean final reaches 98% at 8 obstacles. Success already reaches 99%, so a control arm can no longer separate settings. This tells whether a setting is good enough, not whether it is better than another.
- Every comparison from "The hard group, flights through a gap" on also lists the success rate on the episodes that flew through a gap, because the overall rate averages those away.
- Success often swings by 30 to 40 percentage points between checkpoints, so every arm runs at least 3 times and comparisons use Welch's t-test.
- One episode in this document is one training block, in which 64 environments each run 100 steps. A real flight lasts at most 20 seconds, which is 1,000 steps at the 0.02 s control step and 200 steps at the 0.1 s step of baseline_v4. The stop line runs 60 s episodes, 600 steps, from baseline_stop_v5 (see "Why baseline_stop_v5 looks the way it does").

### Failure types

A flight that does not reach the target goes to the first type below that it matches.

| Type | CSV column | Rule |
|---|---|---|
| Crash | `fail crash (%)` | The drone's center comes within 0.2 m of an obstacle surface |
| Out of bounds | `fail out of bounds (%)` | More than 40 m from the target |
| Oscillation | `fail oscillation (%)` | At least 10% of steps turn the opposite way from the previous step (left after right or right after left) |
| Zigzag | `fail zigzag (%)` | At least 5% of steps turn against the previous turn, with straight flight allowed in between |
| Spinning in place | `fail spin in place (%)` | At least 90% of steps are turns |
| Timeout near the target | `fail near-target timeout (%)` | Time runs out within 2 m of the target |
| Far timeout | `fail far timeout (%)` | Time runs out 2 m or more from the target |

The figure below takes the best checkpoint of the first BASE run and, over 3,000 layouts with 5 obstacles, shows the first example of each type. The green dot is the start, the red cross the end and the star the target, and the path runs from dark to light as time goes on.

![A representative trajectory for each failure type](docs/failure_modes.png)

- Over these 3,000 layouts the failures split into 233 far timeouts, 105 oscillations, 42 crashes, 8 cases of spinning in place, 2 timeouts near the target and 1 zigzag.
- **Apart from the crash, every example spends its last 200 steps stopped.** Oscillation, spinning in place and far timeout differ only in how the drone turns while stopped, swinging left and right, turning one way, or not turning at all. The timeout near the target never even left its start.
- This supports the view that swinging and stopping share one cause. In some states the network rates stopping highest, and once stopped the turns score almost the same, so it stays in place and turns toward whichever side is slightly higher.

## Experiments from the original baseline to baseline_v2

These experiments all came before baseline_v2 was fixed. They start from the original setup on `main` and add one change at a time. Their evaluation differs from later sections, and the opening of each subsection states it. The full per-run numbers are in `docs/results/baseline_to_v2.csv`.

### Two bugs fixed first

| Fix | Before | After | Result (no obstacles, 10 runs per arm) |
|---|---|---|---|
| First transition of a new episode (A1) | Used the last observation of the previous episode | Uses the observation after reset | final median 82.3% to 97.3%, p=0.57 |
| Timeout no longer terminal (V1) | A timeout counted as terminal | Only reaching the target or crashing is terminal | Comparisons start after both fixes |

The same round showed that checkpoint selection has the largest effect of anything tried. Picking best on a separate 1,000 layouts lifted the median of almost every arm to 99% to 100%, while the final checkpoints of one arm ranged from about 20% to 100%, so from then on both best and final are reported.

### Ablation without obstacles

Each arm ran 10 times and was evaluated on 2,000 layouts of seed 1. An arm passed the screen when its final mean beat A1 (81.0%) by at least 5 percentage points without fewer runs at or above 90%.

| Change | final mean | Runs at or above 90% |
|---|---|---|
| R2, progress reward weight 0.05 to 0.15 | 93.2% | 7/10 |
| V9, two output heads (speed times turn) | 93.2% | 7/10 |
| E3, 25% of starts placed 0.8 to 2.0 m from the target | 92.6% | 8/10 |
| E1, learning rate decayed from ep500 to 1/10 | 90.3% | 7/10 |
| V7, Advantage Learning | 89.5% | 8/10 |
| E2, gamma 0.98 to 0.995 | 88.4% | 7/10 |

- No combination replicated. E1 + R2 scored 99.0% and only 92.3% on a rerun. V9 + R2 scored 95.9% and then 87.6%. V9 + V7 reached only 74.3%, so Advantage Learning and the two output heads interfere.
- Changes that hurt were a per-step penalty of -0.01 (R1, only 1/10 runs at or above 90%, p=0.020), oversampling starts with a large heading error (V8, final median 62.8%, p=0.031), and a reversal penalty on top of V2 (V3, oscillation from 2.0% to 5.1%).
- Changes that did not help were a larger reaching bonus (R3), Double DQN (V5), the Huber loss (V6) and a batch of 256 (E5), and V6 and E5 raised oscillation to 15% to 18%.
- Adding hysteresis at evaluation (a left and right reversal must win by delta before the action switches) brought the best checkpoint of all 10 A1 runs to 100%, but it only works at evaluation.
- With 10 runs per arm, no single change or combination beat A1 significantly.

### Ablation with 3 obstacles

Each arm ran 3 times for 1500 ep and was evaluated on 2,000 layouts of seed 1, adding one change at a time to the setup after the bug fixes.

| Change | final | best |
|---|---|---|
| Setup after the bug fixes, 1500 ep | 41.1% (59.2 / 43.8 / 20.4) | 60.0% (70.8 / 58.4 / 50.9) |
| progress reward weight 0.05 to 0.15 | 59.3% (66.3 / 60.6 / 51.0) | 61.8% (73.7 / 60.6 / 51.0) |
| Two output heads | 42.8% (73.7 / 31.0 / 23.7) | 69.8% (85.7 / 72.1 / 51.7) |
| replay buffer 500,000 | 53.1% (67.5 / 65.9 / 25.8) | 56.1% (67.5 / 65.9 / 35.0) |
| replay buffer 500,000, 3000 ep | 65.9% (81.7 / 65.7 / 50.3) | 72.9% (81.7 / 69.8 / 67.2) |
| Two output heads + reward 0.15 + buffer 500,000 | 73.5% (82.5 / 75.8 / 62.3) | 78.2% (86.5 / 75.8 / 72.4) |
| The same, 3000 ep | 74.3% (82.1 / 75.1 / 65.8) | 81.1% (83.6 / 82.1 / 77.5) |
| Plus an exploration floor of 0.03 | 65.5% (77.3 / 77.0 / 42.3) | 87.0% (91.1 / 88.5 / 81.3) |
| The 3000 ep arm with 2 gradient updates per environment step instead, without the 0.03 floor (2 runs) | 41.2% (44.7 / 37.6) | 90.3% (93.7 / 86.9) |
| The 3000 ep arm with the target copied every 20 ep instead, without the 0.03 floor (2 runs) | 80.8% (83.8 / 77.9) | 82.2% (83.8 / 80.5) |

- Changes that did not raise best were learning-rate decay, gamma 0.995, Advantage Learning, a previous-action input, a 256-unit network and a lower learning rate. Learning-rate decay and gamma 0.995 did raise final by about 10 percentage points (see "Getting worse with more training").
- The first 2 runs with the 0.03 floor averaged 89.8%, the third pulled the arm down to 87.0%, and 5 later runs on 10,000 layouts reached only 82.5%, so part of the first gain was luck in selection.

The final setup was then compared with two more changes on 10,000 layouts of seed 777, 3 runs each.

| 3 obstacles | final | best |
|---|---|---|
| The final setup at the time | 60.42% (75.66 / 71.79 / 33.81) | 83.10% (89.40 / 82.85 / 77.04) |
| Plus 2 gradient updates per environment step | 75.61% (85.74 / 71.43 / 69.66) | 86.36% (89.30 / 85.74 / 84.05) |
| Plus the target copied every 20 ep | 73.70% (83.01 / 74.57 / 63.53) | 87.58% (87.93 / 87.71 / 87.09) |

- Both changes mainly reduce late degradation, raising final by 13 to 15 percentage points.
- best rises by 3 to 4.5 percentage points, about the gap between two evaluation sets for one unchanged setup, so only the gain in final is clear.
- Copying the target every 20 ep gave the most consistent 3 runs (87.93%, 87.71%, 87.09%), so it went into baseline_v2.

## Why baseline_v2 looks the way it does

| Change | Before | After | Reason | Evidence (3 obstacles, seed 1, 2,000 layouts, best) |
|---|---|---|---|---|
| First transition of a new episode | The last observation of the previous episode | The observation after reset | A bug that put one wrong transition in every episode | Comparisons start after the fix |
| Timeout no longer terminal | A timeout counted as terminal | Only reaching the target or crashing is terminal | A timeout is not a property of the state, and treating it as terminal teaches wrong Q-values | Comparisons start after the fix |
| Output split into a speed head and a turn head | 3 actions | 2 speeds times 3 turns, 6 actions | The drone can turn while moving forward instead of stopping to turn | 60.0% to 69.8% |
| progress reward weight | 0.05 | 0.15 | The signal was too weak, since the reaching bonus of 100 dwarfs the per-step shaping | final 41.1% to 59.3% |
| replay buffer | 50,000 | 500,000 | With 64 environments running at once, the old buffer held only about the last 780 steps | 72.9% at 3000 ep |
| The three changes above combined (two output heads, progress reward 0.15, buffer 500,000) | | | | 81.1% (3000 ep) |
| Exploration floor | 0.05 | 0.03 | Too many random actions late in training disrupt a learned policy | 87.0% |
| Target copy interval | Every 10 ep | Every 20 ep | A slower-moving target makes the Q-values oscillate less | final 60.4% to 73.7% (seed 777) |
| Training length | 1500 ep | 3000 ep | Runs were often still improving at 1500 ep | See "Ablation with 3 obstacles" |
| Curriculum | Each obstacle count trained from scratch | 1 → 2 → 3 → 5, each stage starting from the previous stage's best | Learn the easy cases before the hard ones | See "New curriculum 1 → 2 → 3 → 5 against the old 3 → 5" |

These settings used to live only in the arguments that outside scripts passed, so leaving out one argument trained a different setup. All of them became code defaults, tagged `baseline_v2.0`.

### Training parameters of baseline_v2

Once the changes above were fixed, these are the defaults of `baseline_v2`.

| Parameter | Value | Note |
|---|---|---|
| `hidden_dim` | 128 | A two-layer MLP with 128 units per layer |
| `learning_rate` | 1e-3 | Adam |
| `gamma` | 0.98 | |
| `batch_size` | 64 | |
| `num_envs` | 64 | Environments run at once |
| `max_steps_per_episode` | 100 | Steps per training block |
| `num_episodes` | 3000 | Per stage, and 1500 for the 1 and 2 obstacle stages of the curriculum |
| `replay_buffer_size` | 500,000 | |
| `epsilon_start`, `epsilon_decay`, `epsilon_min` | 1.0, 0.995, 0.03 | decay is applied once per training block |
| `resume_epsilon` | 0.3 | epsilon when a stage resumes from the previous stage |
| `target_update_every` | 20 | The online network is copied to the target every 20 training blocks |
| `forward_speed`, `turn_rate` | 1.5 m/s, 0.8 rad/s | |
| progress reward weight `K_PROGRESS` | 0.15 | Each step earns 0.15 times forward speed times the cosine of the heading error |
| Reaching bonus, crash penalty | 100, -50 | |
| curriculum `--stages` | 1, 2, 3 and then 5 obstacles | Each stage starts from the previous stage's best weights, weights only |

Every later experiment changes only the following on top of these defaults.

| Experiment | Change |
|---|---|
| SOFT | New `target_tau` of 0.0005, a soft update after every gradient step that replaces the full target copy |
| DDQN | How the TD target is computed, with no new parameter |
| Time penalty | The environment adds a reward of -0.05 at every step |
| V2, V2S | The state gains a one-hot of the previous action (3 or 4 values) |
| No hover | Both speeds fly forward |
| Three forward speeds | The actions become 0.3, 0.75 or 1.5 m/s times 3 turns, with `num_actions` 9 |

## Experiments from baseline_v2 to baseline_v3

The full per-run numbers for every stage are in `docs/results/v2_to_v3.csv`.

### Results of single changes

The numbers are means (%) on 10,000 layouts of seed 777 with 5 obstacles unless stated otherwise.

| Experiment | Runs | best | final | Oscillation | Far timeout | Crash |
|---|---|---|---|---|---|---|
| BASE (baseline_v2) | 5 | 89.45 (92.21 / 91.83 / 90.68 / 86.77 / 85.78) | 64.15 (79.59 / 79.07 / 70.03 / 47.00 / 45.04) | 3.78 | 3.75 | 2.20 |
| Soft target update (SOFT) | 5 | **93.17** (94.39 / 94.33 / 92.80 / 92.39 / 91.96) | 68.18 (91.96 / 73.68 / 70.17 / 66.21 / 38.86) | **2.07** | 1.47 | 1.83 |
| Double DQN (DDQN) | 3 | 91.82 (93.21 / 92.25 / 90.01) | 62.55 (85.96 / 70.93 / 30.75) | 3.22 | 2.30 | 1.26 |
| Time penalty + DDQN | 3 | 90.64 (93.92 / 91.25 / 86.75) | 74.91 (87.00 / 74.71 / 63.01) | 5.00 | 3.07 | 0.88 |
| Per-step time penalty 0.05 (TP) | 3 | 88.41 (89.51 / 88.64 / 87.07) | 73.16 (73.50 / 73.08 / 72.91) | 6.54 | 1.35 | 2.79 |
| Previous turn and speed (V2S) | 3 | 86.54 (91.24 / 84.34 / 84.03) | 61.96 (86.30 / 67.03 / 32.56) | 7.45 | 3.80 | 0.65 |
| Previous turn (V2) | 3 | 86.11 (88.57 / 85.49 / 84.26) | 64.06 (72.64 / 60.06 / 59.47) | | | |
| No hover | 3 | 71.06 (73.20 / 71.51 / 68.47) | 62.46 (65.82 / 63.77 / 57.80) | 0.01 | 0.00 | 27.65 |

### Expectations, results and likely causes

**SOFT (the target moves 0.05% toward the online network at every step)**
- The expectation was that a target no longer jumping every 20 ep gives a smoother learning target and steadier training.
- best rose by 3.7 percentage points, p=0.046, the only significant improvement so far, with all 5 runs at or above 91.96%. Oscillation also roughly halved. final did not improve (68.18%, p=0.73), and the fifth run ended at only 38.86%.
- A likely cause is that a smooth target makes the Q-values more accurate, so better checkpoints can be found. Late collapse has causes beyond the jumping target, so final did not follow.

**DDQN**
- The expectation was less overestimation of Q-values.
- best rose by 2.4 percentage points, but p=0.20. With 3 obstacles oscillation fell from 4.94% to 2.05%. final did not improve.
- A likely cause is that overestimation is real but not the main source of late degradation.

**Time penalty (-0.05 per step)**
- The expectation was that stopping or spinning in place would no longer score 0, so hovering stuck would drop.
- Far timeouts fell from 5.90% (the first 3 BASE runs) to 1.35% and final rose by about 9 percentage points, but best did not rise, and oscillation and crashes increased.
- A likely cause is that the penalty forces the drone to keep moving, so it gets stuck less, but it pushes through or turns erratically in front of obstacles more often.
- This is not necessarily bad. The penalty keeps the network from settling in place and makes it keep trying to move forward, and the extra crashes may come partly from the three range readings missing obstacles, for example in the rear quarters. If the input later becomes camera images with more information, a policy that does not stop has a better chance of learning to avoid them. This is an untested hypothesis. Three forward speeds reach a similar effect by making a full stop impossible, with better results (see "Combinations"). The later crash analysis found other causes, a slowest speed whose turn radius is too large and an input that cannot tell left from right, while only about 17% to 26% of crashes hit an obstacle in the blind diagonals between the front and side sectors (see "The action-space limit" and "The observation limit").

**Previous action as input (V2, V2S)**
- The expectation was that knowing the previous action would let the network avoid swinging left and right.
- Both made things worse, and V2S doubled oscillation. Adding it to DDQN made things worse as well (see "Combinations").
- There are four likely causes. First, copying the previous action is right most of the time, so the network easily learns that shortcut and then fails to switch when it should, which imitation learning calls the copycat problem or causal confusion. Second, the action shapes the next input and the next input shapes the action, which can form a self-reinforcing loop. Third, the next state in the TD target contains the action just taken, which makes the learning target less stable. Fourth, the input size differs, so these runs cannot start from the shared obstacle-free checkpoint (although the experiment in "New curriculum 1 → 2 → 3 → 5 against the old 3 → 5" shows the starting point has no significant effect).

**No hover (every action flies forward)**
- The expectation was to remove hovering stuck and spinning in place at the root.
- With 3 obstacles crashes rose to 14% to 17% and best fell to 78.66%. With 5 obstacles crashes reached 27.65% and best 71.06%.
- A likely cause is that the drone needs to slow down or stop to reorient in front of an obstacle. This is why the design later moved to three forward speeds (see "Combinations").

### New curriculum 1 → 2 → 3 → 5 against the old 3 → 5

- I first expected the old curriculum to be better, because the first few arms seemed to score higher with it.
- To rule out the effect of the starting point, I used different obstacle-free starting checkpoints (A, B, C) and ran 5 times per cell.

best (%) with 5 obstacles, sorted from high to low in each cell.

| Start | New curriculum 1 → 2 → 3 → 5 | Old curriculum 3 → 5 |
|---|---|---|
| A | 92.21 / 91.83 / 90.68 / 86.77 / 85.78 | 91.98 / 90.81 / 89.02 / 88.49 / 86.17 |
| B | 94.08 / 91.42 / 88.36 / 84.26 / 82.40 | 94.23 / 92.33 / 91.79 / 89.13 / 84.70 |
| C | 89.40 / 88.62 / 87.72 / 84.33 / 83.36 | 89.80 / 89.45 / 88.63 / 81.08 / 77.27 |

- The two curricula do not differ. Pooled over the three starts, the new one scores 88.08% and the old one 88.33%, p=0.87. In a two-way ANOVA the curriculum gives p=0.87, the start p=0.11 and their interaction p=0.55.
- The gap seen earlier came from the first 3 BASE runs happening to score low, and it vanished once there were 5.
- With 3 obstacles the new curriculum was actually better for start C (92.10% against 85.98%, p=0.044), but that is 1 of 6 comparisons and it is not significant after correcting for multiple comparisons.
- The decision is to keep 1 → 2 → 3 → 5. The two reach the same success rate, while the new one is more intuitive and varies less with 3 obstacles.

### Combinations

Each combination is compared only with the arm that has one change fewer.
- baseline_v2 against DDQN, see "Results of single changes".
- DDQN against DDQN + SOFT.
- DDQN against DDQN + previous turn and speed.
- DDQN against DDQN + three forward speeds.
- DDQN + previous turn and speed against the same with three forward speeds added.

| Experiment (5 obstacles) | Runs | best | final | Oscillation | Far timeout | Crash |
|---|---|---|---|---|---|---|
| baseline_v2 | 5 | 89.45 (92.21 / 91.83 / 90.68 / 86.77 / 85.78) | 64.15 (79.59 / 79.07 / 70.03 / 47.00 / 45.04) | 3.78 | 3.75 | 2.20 |
| DDQN | 3 | 91.82 (93.21 / 92.25 / 90.01) | 62.55 (85.96 / 70.93 / 30.75) | 3.22 | 2.30 | 1.26 |
| **DDQN + SOFT** | 3 | 91.80 (93.90 / 91.07 / 90.43) | **83.12** (91.07 / 87.96 / 70.32) | 3.67 | 1.51 | 1.29 |
| DDQN + previous turn and speed | 3 | 84.98 (85.40 / 85.08 / 84.46) | 66.11 (76.17 / 63.84 / 58.31) | 4.56 | 8.54 | 0.29 |
| DDQN + three forward speeds | 3 | 92.26 (93.66 / 92.43 / 90.68) | 67.17 (86.03 / 62.87 / 52.61) | 0.02 | 0.08 | 7.63 |
| DDQN + previous turn and speed + three forward speeds | 3 | 91.56 (92.39 / 91.32 / 90.96) | 77.76 (86.48 / 84.72 / 62.09) | 0.29 | 0.00 | 7.80 |

**DDQN + SOFT**
- The expectation was that the two changes address different problems, so their effects could add up.
- best did not rise (91.80% against 91.82%), but final rose from 62.55% to 83.12%, with runs of 70.32%, 87.96% and 91.07%. It is the most stable combination late in training among these. Against baseline_v2 p=0.11, which is not yet significant with 3 runs.
- A likely cause is that DDQN reduces overestimation and SOFT smooths the target, and together they keep the Q-values from drifting late in training, so the policy degrades less. best did not rise further, which suggests the ceiling is set by something else, such as the action design or the state information.

**DDQN + previous turn and speed**
- The expectation was that once DDQN makes the Q-values more accurate, the previous-action input might stop doing harm.
- best fell from 91.82% to 84.98% and far timeouts rose from 2.30% to 8.54%, as bad as on baseline_v2.
- A likely cause is that none of the four causes listed in "Expectations, results and likely causes" has to do with Q-value overestimation, so DDQN cannot reach them.

**Three forward speeds (0.3, 0.75 or 1.5 m/s, times 3 turns)**
- The motivation is that removing hover leads to crashes, while hover is the source of hovering stuck, so the drone may slow down but can never fully stop.
- The speeds were chosen with the turning radius in mind. At a turn rate of 0.8 rad/s the radii are 0.38, 0.94 and 1.88 m, all inside the ±5 m arena. Fitting inside the arena turned out not to be enough. The later crash analysis showed that 0.375 m is too wide to steer off an obstacle that is already close, which led to lowering the speeds (see "The action-space limit").
- With three speeds added to DDQN, best is 92.26%, on par with DDQN's 91.82%, while oscillation falls from 3.22% to 0.02% and far timeouts from 2.30% to 0.08%. Stopping and swinging almost disappear, and nearly every failure becomes a crash (7.63%).
- final did not improve (67.17% against 62.55%, p=0.82), so three speeds fix swinging and stopping, not training instability.
- Adding three speeds to the arm with the previous-action input lifted best from 84.98% to 91.56%, back to the level of DDQN. Three speeds also remove the stopping that the previous-action input caused.

## Why baseline_v3 looks the way it does

`baseline_v3` is `baseline_v2` plus Double DQN and a soft target update, tagged `baseline_v3.0`. Every new experiment branches from it, and its control is DDQN + SOFT.

| Change | Before | After | Reason |
|---|---|---|---|
| Double DQN | The target network both selects and evaluates the action | The online network selects the action and the target network evaluates it | Taking the max in plain DQN picks overestimated actions, and separating selection from evaluation reduces overestimation |
| Soft target update | The online network is copied to the target every 20 ep | The target moves 0.05% toward the online network at every gradient step | The learning target no longer jumps at fixed intervals, so the Q-values swing less |

Evidence (5 obstacles, seed 777, 10,000 layouts)
- Double DQN alone raised best from 89.45% to 91.82%, p=0.20. final did not improve.
- With the soft target update added, best held at 91.80% and final rose from 62.55% to 83.12%. With 2 and 3 obstacles final also rose by 11 and 16 percentage points, in the same direction.
- Each arm has only 3 runs, and against DDQN the final at 5 obstacles gives p=0.34. It was chosen because the three stages point the same way and because it best serves the goal of training that converges.
- SOFT alone has a higher best (93.17%), but its final is only 68.18%, so it was not chosen.

Double DQN was tried once before baseline_v2 (V5, see "Ablation without obstacles"), when it did not help and scored below A1.

| No obstacles, 10 runs per arm | final median | final mean | final at or above 90% | best median | best at or above 90% |
|---|---|---|---|---|---|
| A1 (control) | 97.3% | 81.0% | 7/10 | 99.8% | 9/10 |
| V5 (Double DQN) | 85.0% | 73.4% | 4/10 | 96.1% | 6/10 |

- The difference is not significant, with p=0.43 for final and p=0.13 for best. It was mostly pulled down by a few very poor runs, such as one whose final was only 6.9%, with 77.9% of its episodes spinning in place.
- The conditions then differed from later ones. There were no obstacles, 3 actions, a buffer of 50,000, a target copied every 10 ep and a progress reward of 0.05.
- Without obstacles the best of A1 was already close to 100%, which left DDQN almost no room to improve. Reducing Q-value overestimation matters more in hard scenarios, which is why the gains in best and oscillation appeared only with obstacles on the new setup.

## Experiments from baseline_v3 to baseline_v4

These experiments are all built on baseline_v3, so each is compared with DDQN + SOFT. DDQN + SOFT was later extended to 5 runs, and this section uses those 5 runs, 92.20% best and 81.28% final. The full per-run numbers for every stage are in `docs/results/after_v3.csv`.

Every number is 5 obstacles, seed 777, 10,000 layouts, with each run in parentheses from highest to lowest. p-values are Welch t-tests against DDQN + SOFT.

### Single changes

| Experiment | Runs | best | final | Oscillation | Crash |
|---|---|---|---|---|---|
| DDQN + SOFT (baseline_v3) | 5 | 92.20 (94.34 / 93.90 / 91.25 / 91.07 / 90.43) | 81.28 (91.07 / 87.96 / 86.60 / 70.43 / 70.32) | 2.97 | 1.70 |
| Three forward speeds 0.3 / 0.75 / 1.5 m/s | 3 | 91.76 (94.38 / 92.04 / 88.86) | 87.60 (92.13 / 85.65 / 85.03) | 0.01 | 8.21 |
| Three forward speeds, crash penalty -50 to -100 | 3 | 91.00 (93.82 / 91.91 / 87.28) | 75.01 (88.59 / 80.37 / 56.07) | 0.00 | 8.98 |
| replay buffer 500,000 to 2,000,000 | 3 | 90.13 (92.90 / 88.80 / 88.70) | 73.71 (83.25 / 72.31 / 65.57) | 2.81 | 2.91 |
| gamma 0.98 to 0.996 | 3 | 81.87 (86.32 / 83.02 / 76.28) | 63.54 (69.42 / 64.80 / 56.40), p=0.025 | 14.03 | 0.80 |
| **Control step dt 0.02 s to 0.1 s** | 5 | **94.63** (95.15 / 95.10 / 94.76 / 94.64 / 93.49), p=0.035 | **92.59** (95.99 / 92.49 / 92.29 / 92.13 / 90.05), p=0.065 | 4.90 | 0.17 |
| Own obstacle-free stage, dt unchanged | 3 | 90.39 (91.92 / 91.74 / 87.52) | 71.67 (80.71 / 69.61 / 64.69) | 2.39 | 2.12 |

**Three forward speeds**
- The actions become 3 forward speeds times 3 turns, 9 in all, with no stop. The expectation was that moving slowly could replace stopping.
- Oscillation and far timeouts almost vanished, and final was the highest at the time at 87.60%, but best did not move, and nearly every failure became a crash (8.21%).

**Crash penalty doubled**
- The expectation was that the network would become more careful and crash less.
- Crashes did not drop (8.21% to 8.98%), the three runs of each overlap almost completely, and final fell from 87.60% to 75.01%.
- The analysis in "The action-space limit" shows these crashes were mostly impossible to avoid physically. A penalty only makes the network more afraid of crashing without giving it a new way to avoid one, and a larger value widens the range of the TD target.

**replay buffer enlarged to 2,000,000**
- The expectation was that hard states would stay in the buffer longer and late forgetting would shrink.
- final fell from 83.12% to 73.71% (against the 3-run DDQN + SOFT of the time), so the hypothesis that a short buffer causes forgetting is not supported. A likely cause is that a very large buffer holds a larger share of data from outdated policies.

**Control step dt 0.1 s**
- Someone else reported that changing dt from 0.02 to 0.1 made results much more stable. dt is the interval between decisions. Every dt seconds the network looks at the state and picks an action, so a 20 s flight goes from 1,000 steps to 200.
- best and final both rose, all 5 final values are above 90%, and the training instability is largely solved.
- The reason is in "Why a longer control step trains more stably".

**gamma 0.996**
- gamma discounts once per step, so dt 0.1 stretches the horizon from about 1 s to about 5 s. This arm raised gamma to 0.996 at dt 0.02 to give the same horizon, to tell whether the effect of dt 0.1 comes from the longer horizon alone.
- It got clearly worse, with oscillation at 14.03%. So the effect of dt 0.1 does not come from the horizon.

**Own obstacle-free stage**
- dt 0.1 cannot reuse the shared starting checkpoint trained at dt 0.02, so it trains its own stage with no obstacles. This arm also trains its own obstacle-free stage at dt 0.02, to rule out the different start.
- It is worse than DDQN + SOFT, so the gain of dt 0.1 does not come from training its own first stage, and against this arm with the same start the effect of dt 0.1 is even larger.

### Speed values of the three forward speeds and the control step

| Experiment | Speeds (m/s) | dt (s) | best | final | Oscillation | Crash |
|---|---|---|---|---|---|---|
| Three forward speeds | 0.3 / 0.75 / 1.5 | 0.02 | 91.76 (94.38 / 92.04 / 88.86) | 87.60 (92.13 / 85.65 / 85.03) | 0.01 | 8.21 |
| Slowest speed lowered | 0.15 / 0.45 / 1.5 | 0.02 | 94.42 (96.18 / 94.21 / 92.88) | 79.06 (94.23 / 89.18 / 53.77) | 0.08 | 5.37 |
| Slowest speed lowered further | 0.075 / 0.3 / 1.5 | 0.02 | 94.44 (95.02 / 94.35 / 93.95) | 86.66 (90.41 / 86.97 / 82.60) | 0.36 | 4.44 |
| Three forward speeds, gamma 0.996 | 0.3 / 0.75 / 1.5 | 0.02 | 90.38 (91.32 / 91.30 / 88.52) | 75.28 (86.07 / 81.65 / 58.11) | 0.09 | 9.41 |
| Three forward speeds | 0.3 / 0.75 / 1.5 | 0.1 | 95.17 (95.71 / 95.08 / 94.71) | 94.19 (94.65 / 94.44 / 93.48) | 0.11 | 4.68 |
| Q network with one output per action | 0.3 / 0.75 / 1.5 | 0.1 | 96.14 (97.42 / 95.61 / 95.40) | 95.89 (97.25 / 95.27 / 95.16) | 0.11 | 3.69 |
| **C, slowest speed lowered plus dt 0.1** | 0.075 / 0.3 / 1.5 | 0.1 | **98.94** (99.25 / 98.90 / 98.68) | **97.93** (98.72 / 97.81 / 97.26) | 0.37 | 0.65 |

- Lowering the slowest speed follows from the analysis in "The action-space limit". At dt 0.02 best rises by about 2.7 percentage points, but at p=0.23 against the original speeds and with no better final, so on its own it cannot be called better.
- At dt 0.1, 0.075 / 0.3 / 1.5 m/s is far better than 0.3 / 0.75 / 1.5 m/s, at p=0.001 for best and p=0.003 for final, and the three runs do not overlap. The two changes together do far more than either alone.
- This arm became the interim baseline, called C.
- A Q network with one output per action helped, but much less. Its outputs grow with every combination of speeds and turns, so it will not be tried again.

### The 2x2 on candidate C

With C as the interim baseline, this compares 3 sectors and 4 sectors (see "The observation limit"), with and without the previous-action input (V2S).

| | Without V2S | With V2S |
|---|---|---|
| 3 sectors | 98.94 (99.25 / 98.90 / 98.68) / 97.93 (98.72 / 97.81 / 97.26), crash 0.65 | 99.31 (99.59 / 99.30 / 99.05) / 98.43 (98.97 / 98.82 / 97.50), crash 0.61 |
| 4 sectors | **99.50** (99.65 / 99.44 / 99.40) / **98.96** (99.50 / 98.72 / 98.65), crash 0.43 | 99.44 (99.52 / 99.44 / 99.35) / 98.91 (99.18 / 98.92 / 98.63), crash 0.50 |

Each cell gives best, then final.

| Comparison | p for best | p for final |
|---|---|---|
| 4 against 3 sectors, without V2S | 0.061 | 0.12 |
| 4 against 3 sectors, with V2S | 0.52 | 0.42 |
| With against without V2S, 3 sectors | 0.18 | 0.47 |
| With against without V2S, 4 sectors | 0.56 | 0.89 |

- All four cells are near 99%, and three runs cannot separate them statistically.
- 4 sectors without V2S has the highest mean and the fewest crashes.
- V2S helps a little with 3 sectors and not at all with 4. What V2S can add is breaking the left-right dithering, and 4 sectors already removes it.
- With 4 sectors alone at the old speeds 0.3 / 0.75 / 1.5 m/s, best also rose from 95.17% to 99.26% (99.35 / 99.28 / 99.15), with 0.00% oscillation.

## Limits built into the design

Both problems in this section come from the action space and the input themselves. They have nothing to do with how well the network trains or how its parameters are set, and no amount of learning gets around them.
- The action-space limit is about what the drone cannot do. Once it is too close to an obstacle, no control avoids it.
- The observation limit is about what the drone cannot know. The input lacks the information it needs, so even the best policy can only guess, and a wrong guess crashes.

### The action-space limit, a minimum turn radius that is too large

With three forward speeds 0.3 / 0.75 / 1.5 m/s at dt 0.02, nearly every failure was a crash. The best checkpoint of the first training run was replayed on 2,000 layouts. Of its 121 crashes (6.05%), the last action was the slow speed in 97% and a left turn in 98%, and d_front on the step before impact had a median of 0.20 m.

![Every crash ends in a slow left turn](docs/crash_turn_radius.png)

- At 0.3 m/s and a turn rate of 0.8 rad/s, the smallest turn radius is 0.375 m.
- The last unbroken run of slow left turns before impact lasted a median of 52 steps and began with a median surface gap of 0.46 m. The network did start turning early, it just could not turn tightly enough.
- Holding a different action for a full circle from where that run began avoided every obstacle in the following share of crashes.

| Held from the same position | Turn radius | Avoids every obstacle |
|---|---|---|
| Left at 0.3 m/s, the action taken | 0.375 m | 1% |
| Left at 0.15 m/s | 0.19 m | 76% |
| Left at 0.075 m/s | 0.09 m | 95% |
| Right at 0.3 m/s | 0.375 m | 46% |

- So the slowest speed is too fast. Crashes that cannot turn away come from a physical limit, not from a network that is not afraid enough of crashing, which also explains why doubling the crash penalty failed.
- Lowering the speeds to 0.075 / 0.3 / 1.5 m/s brings the turn radius down to about 0.09 m. At dt 0.1, crashes fell from 4.68% to 0.65% and best rose from 95.17% to 98.94% (the C in "Speed values of the three forward speeds and the control step").

### The observation limit, three sectors cannot tell left from right

After dt 0.1, crashes looked different. In the arm with three speeds and dt 0.1 (95.17%), 91% of crashes happened during a right turn and the obstacle hit was on the left in 84%.

![Crashes at the 0.1 s control step](docs/crash_front_sector.png)

Replaying one of these crashes step by step showed the following.
- At the start it is 1.03 m from the obstacle surface, enough room to turn away.
- For the next 16 steps the network alternates left and right every step at the slow speed, while d_front shrinks from 0.88 m to 0.43 m. Meanwhile the right reading jumps between about 3.3 m and 10 m, because the heading moves only 4.6 degrees a step and another obstacle crosses the edge of the sector each time.
- By the time it commits to turning right it is too close, and its left side grazes the obstacle.

Across three training runs, 44%, 52% and 17% of crashes followed 4 or more left-right reversals in the last 2 s, and in 64% to 78% of those the surface gap at the first reversal was about 0.7 m, enough to turn away. At dt 0.02 only 21% of crashes were of this kind, with a gap of only 0.45 m at the first reversal.

The cause is the input design.
- The state is [d_front, d_left, d_right, dx, dy], and d_front is only the distance to the nearest obstacle within 36.5 degrees of the heading. An obstacle 10 degrees left or 10 degrees right of the heading gives the same d_front.
- So when an obstacle sits right on the line between the drone and the target, turning left and turning right give almost the same state and almost the same Q-values. Any small change, such as a jump in a side reading, flips the choice, and the drone dithers on that line.
- The right rule is to turn left when the obstacle is right of the heading and right when it is left of it. The network never receives which side the obstacle is on, so it cannot learn that rule.

The fix splits the front sector into a front-left half (0 to +36.5 degrees) and a front-right half (0 to -36.5 degrees). Coverage stays the same and the state grows from 5 to 6 values. A depth camera already sees which side of the image an obstacle is on, so this is still information a real drone has.
- This change alone raised best from 95.17% to 99.26%, with oscillation at 0.00%.
- On C it reaches 99.50% (see "The 2x2 on candidate C").

## Why a longer control step trains more stably

A shorter control interval usually means faster reactions, but here the network must learn the value of each action by Q-learning, and too short an interval makes that harder.
- At dt 0.02 an action lasts only 0.02 s, so even at 1.5 m/s the drone moves 3 cm, and turning left, turning right or going straight lead to almost the same outcome. The Q-value gap between actions scales roughly with dt, while the network's error in estimating Q-values does not shrink with it. Once the gap is smaller than the error, the argmax keeps flipping, which shows up as left-right swinging and as final falling. Tallec, Blier and Ollivier proved in *Making Deep Q-learning Methods Robust to Time Discretization* (ICML 2019) that Q-learning breaks down as dt goes to 0.
- The +100 for reaching the target has to propagate back one TD update at a time, over 1,000 steps at dt 0.02 but only 200 at dt 0.1.
- Only stretching the horizon made things worse (gamma 0.996 in "Single changes"), which fits the first point. With dt unchanged the Q-value gap stays just as small, and a longer horizon only makes the Q-values and their error larger together.
- The cost is slower reactions, since a full-speed step covers 0.15 m. On a real drone the network can decide every 0.1 s (10 Hz) while the low-level flight controller still runs at 50 Hz, the same idea as the Atari DQN repeating each action for 4 frames.

## Why baseline_v4 looks the way it does

`baseline_v4` was built from `baseline_v3` by merging in the 4-sector, previous-action cell of "The 2x2 on candidate C", plus two fixes that do not change results.

| Change | Before | After | Reason and evidence |
|---|---|---|---|
| Forward speeds | stop or 1.5 m/s | 0.075 / 0.3 / 1.5 m/s, no stop | Removes stopping and swinging ("Single changes"), and the smallest turn radius falls from 0.375 m to about 0.09 m ("The action-space limit") |
| Control step | 0.02 s | 0.1 s | best and final both rise, all 5 final values above 90% ("Single changes", "Why a longer control step trains more stably") |
| Front sector | 1 | front-left and front-right | Tells the network which side the obstacle is on, and oscillation falls to 0 ("The observation limit") |
| Previous action (V2S) | none | one-hots of the previous turn and speed | See the discussion below |
| Memory of the tie-break | curriculum.py held torch for the whole chain, about 1.3 GB | scored in a child process, about 11 MB | On two past tie-breaks the old and new code wrote identical records and checkpoints |
| Parameters of the action mapping | `--forward_speed` and three others had no effect or mapped wrongly | the mapping uses the run's own config | Default values give the same raw actions as before, and no past experiment changed these parameters |

On the 2x2 on C this setup reached 99.44% best and 98.91% final, against 92.20% and 81.28% for baseline_v3.

**Whether to add V2S**

This was my decision, and it differs from Claude's recommendation, so both sides' reasons are recorded here.

My reasons
- Adding V2S should be a safeguard. If the previous action is truly useless to the model, its weights should become small during training and do no harm.
- It did not hurt with 4 sectors (99.44% against 99.50%, p=0.56).

Claude's counterargument
- Useless inputs are not pushed to 0 on their own. Training has no weight decay or L2 penalty, so nothing drives a weight toward 0. DQN targets are noisy, and an input that happens to correlate with that noise gets used.
- Measurement agrees. In the three trained networks with 4 sectors and V2S, the norm of each first-layer weight column for a previous-action input grew from about 1.9 at initialization to 4.6 to 6.7, so the network does use these inputs.
- The previous action is highly correlated with the current one, so the network easily learns the shortcut of repeating what it just did, and the state changes with the network's own decisions, which makes the training distribution less stable.
- In this project V2S has caused harm before. Adding it to DDQN dropped best from 91.82% to 84.98%, and at best it made no difference.
- The left-right dithering it would address is already solved by 4 sectors. Its cost is a state of 12 values instead of 6.
- Where it may really be needed is an environment with inertia and delay, such as Isaac Lab or a real drone. In this 2D kinematic model every step switches straight to the new command, so the previous action carries no extra information.

The final decision was to include V2S, and to test weight decay separately to see whether it really pushes useless weights down.

## The 8-obstacle benchmark

With 5 obstacles every arm close to baseline_v4 reached about 99%, so success could no longer tell settings apart. Two changes made the task harder.

**A layout that always places the path-blocking obstacle.** The old placement code drew obstacle 0 on the straight line from start to target and dropped it after 30 rejected draws.
- On seed 3000 with 10,000 layouts per obstacle count, only about 73% of layouts placed every obstacle, and the one missing was always obstacle 0. So about 27% of layouts had nothing blocking the direct path.
- The new rule needs a start-to-target distance of at least 3.0 m when obstacles exist and draws obstacle 0 only from the part of the line where it fits, so every obstacle is placed at 5 and at 8 obstacles.
- With 8 obstacles, 58.9% of the new layouts have overlapping obstacles and 51.3% have a gap narrower than the drone. None is unreachable.
- Even if all the easy layouts had succeeded, 99% overall still means at least 98.6% on the layouts with obstacle 0, so the conclusions drawn on the old layout stand. Numbers from the two layouts are still never mixed in one table.

**8 obstacles as the last stage.** Every arm from here on trains 0, 1, 2, 3, 5 and then 8 obstacles, the last stage 3000 episodes, and is judged at 8 obstacles.

baseline_v4 itself on this procedure (V4_0TO8) reached 98.97% best and 97.87%† final, and continuing its 5-obstacle runs to 8 (V4_8) reached 99.02% and 98.56%†.

## Experiments from baseline_v4 to baseline_v4.2 and baseline_v6, without the stop action

All numbers below are at 8 obstacles on the new layout, 3 runs each. † marks a final that is still the last checkpoint alone.

| Arm | Change | best | final | Pass |
|---|---|---|---|---|
| V4_0TO8 | baseline_v4 on the new procedure | 98.97 | 97.87† | no |
| V4WD_8 | AdamW weight decay 1e-4 | 99.04 | 98.27† | yes |
| V4LRD | learning rate held for the first half of each stage, then lowered linearly to 1/10 | 98.76 | 98.62† | no |
| V4LRD8L | V4LRD with the 8-obstacle stage at 4500 episodes | 98.73 | 98.80† | no |
| V4TP | -0.01 every step | 98.47 | 98.30† | no |
| V4TO | -50 on the step an episode times out | 98.95 | 98.44† | no |
| **V4EZ** | temporally-extended epsilon-greedy, which became baseline_v4.2 | **99.13** | **98.58** | **yes** |
| V4LONG | V4EZ with 60 s episodes and the replay thinned past step 200 | 98.99 | 98.28† | no |
| V4LONGPER | V4LONG with prioritized replay in place of the thinning | 99.19 | 98.45† | yes |
| V4LONGG99 | V4LONG with gamma 0.99 | 99.38 | 99.19† | yes |
| V4LONGG99PROX | V4LONGG99 with a proximity penalty | 99.72 | 98.31† | yes |
| V4EZG99 | V4EZ with gamma 0.99 | 99.30 | 89.88† | no |
| V4EZPROX | V4EZ with a proximity penalty | 99.16 | 93.68 | no |
| V4EZG99PROX | V4EZ with gamma 0.99 and a proximity penalty | 99.51 | 98.17 | yes |

- **Weight decay passed**, and after it the user judged the weight-decay experiments done.
- **The stage-local learning-rate decay** led every stage up to 5 obstacles, 99.41% best and 99.32% final at 5, but missed at 8 with 1.12% crashes, and a longer last stage did not help.
- **A time or timeout penalty changed nothing.** At gamma 0.98 a -50 that arrives at the end of a 20 s episode is worth almost nothing at the start.
- **V4EZ was the first arm to pass** and became baseline_v4.2. Its final was first reported as 99.03 from the last checkpoint alone and is 98.58 as the mean of the last 3 checkpoints. The reason for temporally-extended exploration is in "Why the stop line stalls".
- **60 s episodes alone did not help.** They remove only the roughly 0.2% of failures that are timeouts, and crashes rose from 0.64% to 1.01%, which fits the replay holding a different mix of transitions. Together with gamma 0.99 they passed with the highest final of any arm. Over 3 runs that spread by 0.9 points this is "not worse" rather than "better".
- **Gamma 0.99 needs the 60 s episode.** On 20 s episodes V4EZG99 kept its best, but one run's final collapsed to 71.61%.
- **The proximity penalty** charges up to 0.2 per step, rising linearly as the clearance short of contact falls from 0.3 m to 0. 87% of the crashes it targets were side contacts at 60 to 100 degrees from the heading. It raised best on every line it was added to, and on V4EZ without gamma 0.99 one run fell to 56.22% at ep2800.

## Why the stop action came back

baseline_v4 has no stop, because removing it ended the stopping and swinging (see "Why baseline_v4 looks the way it does"). The user brought it back on 2026-10-27 and kept it through every round in which it scored lower. This is the most important design decision after baseline_v4, so the reasons are recorded in full.

**The user's reasons**
- Most baseline_v4 failures at 8 obstacles were attempts to squeeze through a gap. The user's view was that a drone in a gap it cannot pass needs to stop and turn around, and only the stop action makes that possible.
- On 2026-10-30, translated from the user's words. "Although success is lower after adding stop, only stop can turn around and get out of a tight gap. If it can learn this, success should be higher. The supporting measure is the similarity penalty to avoid jitter."
- On 2026-10-31, after the failure analysis of the stop arms. "The second most common failure is forcing into the narrow gap between two close obstacles. This still needs stopping to solve, so for now the experiments focus on the ones with stop."
- There is a path out of a dead end, which is to turn around and go back along the route the drone came by. If the drone has never turned around successfully, it can never learn that turning around has a high value. This is what led to temporally-extended exploration.
- Standing still costs less than a crash, and on the real drone safety matters more. So a policy that sometimes stalls instead of crashing is acceptable.

**Changes that only work together count as one change**
- Stop alone makes the drone stall and swing in place, so testing it alone and dropping it when it scores lower would reject an idea that needs its partner. In the user's words, "if a combination of two changes is reasonable, it can still be treated as a single change."
- So the stop line is tested as packages. The stop package is the stop action, the similarity penalty at -0.3 and temporally-extended exploration. The long-episode package is 60 s episodes with prioritized replay or gamma 0.99.
- Each package's contents are fixed before its results are seen. Only after a package passes is it taken apart to find which part mattered, so that it is first shown to be no worse and then explained.
- The stop experiments got their own base, `baseline_v4_stop`, because every experiment on that line exists to make the stop action work.

**Claude's counterargument, also recorded**
- Without stop the slowest speed, 0.075 m/s, already turns on a radius of about 9 cm, so a U-turn is possible without stopping. Stop adds only turning in place, which matters only for gaps narrower than about 19 cm.
- The narrow-gap crashes come from flying into a gap the drone cannot pass, not from failing to get out of it (see "The hard group, flights through a gap").
- At that time V4EZ without stop scored 99.34% best at 600-step evaluation against 98.76% for the best stop arm.

**The trade-off accepted.** The stop line is kept even though no stop arm beats the arm without stop at the same settings by more than the spread between runs, and its finals swing more. baseline_stop_v5 reached 99.45% and 98.35%, against 99.38% and 99.19%† for V4LONGG99 without stop.

## Experiments with the stop action, from baseline_v4_stop to baseline_stop_v5

`baseline_v4_stop` is baseline_v4.1 plus the new layout and a stop as a fourth speed, so the speeds are 0, 0.075, 0.3 and 1.5 m/s times 3 turns, 12 actions. All finals in this table are the mean of the last 3 checkpoints.

| Arm | Change | best | final | Pass |
|---|---|---|---|---|
| V4STOP | the stop action | 98.20 | 97.82 | no |
| V4FRONT5STOP | V4STOP with the front camera's 5 bins only | 98.29 | 96.91 | no |
| V4STOPTP | -0.01 every step | 98.04 | 97.04 | no |
| V4STOPTO | -50 on a timeout | 98.69 | 97.18 | no |
| V4STOPSIM | similarity penalty -0.1 against the pose 20 steps back | 97.89 | 97.31 | no |
| V4STOPSIMW | V4STOPSIM at -0.03 | 97.89 | 97.00 | no |
| V4STOPSIMOBS | V4STOPSIM with the offset from that pose in the state | 98.68 | 97.30 | no |
| V4STOPSIMOBSTO | V4STOPSIMOBS with -50 on a timeout | 98.67 | 97.62 | no |
| V4STOPSIMOBSEXP | heading similarity exp(-abs(heading change) / 0.2) | 98.59 | 97.83 | no |
| V4STOPSIMOBSEXP3 | V4STOPSIMOBSEXP at -0.3 | 98.71 | 97.80 | no |
| V4STOPEZ | V4STOP with temporally-extended exploration | 98.14 | 97.61 | no |
| V4STOPSIMOBSEXP3EZ | the stop package | 98.93 | 98.05 | no |
| V4STOPLONG | V4STOP with 60 s episodes | 98.23 | 96.91 | no |
| V4STOPPKGLONGPER | stop package, 60 s, prioritized replay | 99.32 | 98.31 | yes |
| V4STOPPKGLONGG99 | stop package, 60 s, gamma 0.99 | 99.44 | 98.44 | yes |
| **V4STOPPKGLONGG99PROX** | V4STOPPKGLONGG99 with the proximity penalty, which became baseline_stop_v5 | **99.45** | **98.35** | **yes** |

- **The stop action alone** lowered crashes from 0.85% to 0.61% but raised oscillation from 0.10% to 1.14%. 85% to 97% of successful episodes used the stop, so it is not idle.
- **A flat time penalty did not stop the stalling**, which points at the value estimate rather than at stopping being free (see "Why the stop line stalls").
- **The similarity penalty** was the user's idea. It charges -c * exp(-distance / 0.05 m) times a heading factor against the pose 20 steps back in the same episode, so standing or swinging in place costs every step while flying on costs almost nothing. It is a penalty and not a reward because a reward for moving can be farmed by orbiting. Without the offset in the state the network cannot see why it is charged, and adding it helped. The steeper heading factor and the coefficient of -0.3 helped only a little each.
- **The packages** passed once the long-episode package joined the stop package. Under the last checkpoint alone V4STOPPKGLONGPER would have failed at 97.63 and V4STOPPKGLONGG99 at 97.92, which is part of why the final became the mean of the last 3 checkpoints.

## Why baseline_stop_v5 looks the way it does

V4STOPPKGLONGG99 and V4STOPPKGLONGG99PROX are tied in simulation, 99.44% and 98.44% against 99.45% and 98.35%.
- Claude first recommended the arm without the proximity penalty, because a baseline should carry only changes that were shown to help.
- The user's reason for the penalty is the real drone. Sensing and control errors push the real airframe closer to obstacles than planned, and the penalty keeps a clearance margin that the noise-free simulation cannot reward. In simulation the penalty arm even had slightly more crashes, 106 against 90, so "safer" is an expectation for the real drone, not a measurement.
- A worry that the penalty would make the drone circle a target close to an obstacle was checked and dropped. Among timeouts, targets 1.0 to 1.5 m from an obstacle were 17.8% for the final checkpoints and 27.1% for the best, against 34.3% of all layouts.
- On 2026-10-31 the user chose V4STOPPKGLONGG99PROX as `baseline_stop_v5`.

So baseline_stop_v5 is baseline_v4_stop plus the similarity penalty at -0.3 with the steeper heading factor and the pose offset in the state, temporally-extended exploration, 60 s episodes with the replay thinned past step 200, gamma 0.99 and the proximity penalty of 0.2 within 0.3 m.

Four single changes on top of it all failed the final.

| Arm | Change | best | final |
|---|---|---|---|
| baseline_stop_v5 | | 99.45 | 98.35 |
| STOP5G995 | gamma 0.995 | 99.62 | 97.49 |
| STOP5PER | prioritized replay in place of the thinning | 99.62 | 95.99 |
| STOP5LR5E4 | learning rate 5e-4 | 99.47 | 97.92 |
| STOP5TAU25 | soft update tau 0.00025 | 99.37 | 97.96 |

- Gamma 0.995 and prioritized replay raised best but collapsed at single late checkpoints. STOP5PER r1 fell to 76.98% at ep2800. Over 90 checkpoints each, the baseline dipped below 95% 7 times, G995 14 times and PER 6 times. These dips are instability, not the model still learning.
- Halving the learning rate or tau, both meant to calm those dips, did not raise the final.

## Why the stop line stalls

**The value of stopping is almost the same as moving.** Replaying the timeouts of V4STOP's best checkpoints, 148, 72 and 139 per run, the drone spent 73% to 86% of its steps stopped and reversed its turn on 57% to 74% of steps, about 3.1 m from the target and with the nearest reading at 0.32 to 0.56 m.
- The largest Q-value was 19 to 30 and stopping beat moving by only 0.16 to 1.1, so a small error in the estimate keeps postponing the move.
- Without stop the slowest speed still moves 7.5 mm a step, so the drone cannot stay in this loop.
- Swapping the previous-turn input flipped the choice only 22.5% of the time, against 84.8% alternation, so the swinging comes from sitting on the boundary between two headings, not from copying the previous action.

**The policy knows how to leave but cannot hold a turn.** Stalled episodes were taken at step 80 and forced to turn in place for 40 steps.
- In open space 100 of 104 then reached the target, against 9 of 104 left alone. In narrow gaps 40 of 47 reached it, against 7 of 47.
- A U-turn needs about 39 turns the same way in a row, and per-step epsilon-greedy produces that with a probability of about (0.03 / 12)^39, so exploration never shows the network that turning around pays off.
- Temporally-extended epsilon-greedy (Dabney, Ostrovski and Barreto, *Temporally-Extended epsilon-Greedy Exploration*, ICLR 2021) holds each exploratory action for a number of steps drawn from a zeta distribution with mu 2, capped at 100, with the start probability scaled so the share of exploratory steps stays epsilon.

**The remaining stalls are a fixed point.** Of the 59 timeouts of baseline_stop_v5's best checkpoints, 56 did not move at all in their last 200 steps.
- 29 alternated stop-left and stop-right every step, 10 stood still, 16 cycled with periods of 4 to 28 steps, 1 spun one way and 3 actually moved.
- They already pay the full -0.3 similarity penalty every step and still stay, so a larger penalty or more comparison windows would only charge them more without changing the choice.
- The same checkpoints flew through 2,766 gaps narrower than 1.0 m when they succeeded and backed off at least 0.3 m in 1,053 successes, so the stalls do not come from fear of gaps.

**Why gamma 0.99 helps a U-turn**, as a worked example rather than a measurement. For a slow U-turn of 40 turning steps, 2 m back and 5 m on, the discounted +100 is worth about 5.8 at gamma 0.98 and about 24.2 at gamma 0.99. Standing forever is worth about -15 and -30. So gamma 0.99 makes the U-turn clearly worth more than standing, and the 60 s episode gives the slow route time to finish.

## The hard group, flights through a gap

The overall rate averages away the hard layouts, which swing two to three times as much, so from 2026-11-01 every comparison also lists success on the episodes in which the drone flew through a gap.

**Definition.** On the 8-obstacle layouts of seed 777, an episode counts when at some step two obstacles lie more than 90 degrees apart in bearing from the drone and the drone's distances to their two surfaces sum to under 1.0 m, the width of the gap it is in.
- It depends on how the model flew, so each arm meets a different set, and the count is always given next to the rate. An arm that avoids gaps meets fewer of them.
- A crash against a single obstacle never counts, since the rule needs an obstacle on each side.
- Two definitions based only on the layout were tried first and dropped, because the drone could simply fly around the gaps they found. The user wants gaps the drone actually met.
- `drone_runs/baseline_v2/tools/gap_pass.py` replays the best and the last 3 checkpoints of each run, and the results are under `drone_runs/baseline_v2/hard_cases/<arm>/`.

Success in % with the mean episode count over 3 runs in brackets. final is the mean of the last 3 checkpoints.

| Arm | Readings | Checkpoint | Overall | Gap under 1.0 m | 0.4 to 0.6 m | 0.6 to 0.8 m | 0.8 to 1.0 m |
|---|---|---|---|---|---|---|---|
| baseline_stop_v5 | 4 sectors | best | 99.45 | 95.95 (1038) | 84.41 (207) | 98.34 | 99.46 |
| | | final | 98.35 | 95.09 (1031) | 82.43 (183) | 97.40 | 98.90 |
| STOP5LR5E4 | 4 sectors | best | 99.47 | 96.86 (1050) | 87.10 (204) | 98.80 | 99.65 |
| | | final | 97.92 | 95.08 (1035) | 84.81 (205) | 96.97 | 98.59 |
| STOP5TAU25 | 4 sectors | best | 99.37 | 94.96 (916) | 77.58 (161) | 98.03 | 98.97 |
| | | final | 97.96 | 93.25 (950) | 78.25 (153) | 94.81 | 97.50 |
| V4EZ (baseline_v4.2) | 4 sectors | best | 99.13 | 93.41 (937) | 74.64 (160) | 96.05 | 98.92 |
| | | final | 98.57 | 89.95 (1083) | 63.94 (234) | 96.76 | 98.78 |
| V4CAM5WD | 3 cameras x 5 bins | best | 98.80 | 91.02 (950) | 66.41 (194) | 96.55 | 98.94 |
| | | final | 98.28 | 88.86 (993) | 60.95 (214) | 95.28 | 98.44 |
| V4FRONT2CAMWD | 2 cameras x 5 bins | best | 99.24 | 94.41 (976) | 78.77 (175) | 97.84 | 98.70 |
| | | final | 97.79 | 91.51 (1011) | 71.39 (207) | 96.47 | 97.99 |

- **The stop line handles narrow gaps best.** baseline_stop_v5 succeeds in 84.41% of the 0.4 to 0.6 m gaps at best and 82.43% at final, against 74.64% and 63.94% for baseline_v4.2, and it does not show in the overall rate. baseline_stop_v5 differs from baseline_v4.2 by the stop package and the long-episode package together, so this does not yet isolate the stop action.
- **Two cameras beat four sectors on narrow gaps**, 78.77% against 74.64% at best and 71.39% against 63.94% at final on the same settings apart from the sensor and weight decay, while its overall final is lower.
- **Three cameras did worst** on narrow gaps.

**Narrow-gap success does not grow with training.** Checkpoints of baseline_stop_v5 every 300 episodes of the 8-obstacle stage were replayed.
- On 0.4 to 0.6 m gaps the 3-run mean swings between 75% and 88% with no trend, and neighbouring checkpoints of one run differ by 20 to 30 points. r2 scored 98.70% at ep2700 and 66.41% at its last checkpoint.
- A checkpoint that enters more narrow gaps does worse on them, a correlation of -0.45, while its overall rate barely moves, a correlation of -0.04. These gaps are under 3% of the 10,000 layouts.
- On a fixed set of 112 narrow-gap layouts that at least half the checkpoints met, the first half of the stage averaged 83.9% and the second half 85.3%, inside the noise of about 2 points.
- So the swings come from unstable training, not from a model still learning to pass gaps.

**What the narrow-gap failures are.** The 3 best checkpoints failed 28 of the 112 narrow-gap layouts.
- 23 forced into the gap and crashed, 4 stood at the gap mouth until the time ran out, and 1 hit another obstacle.
- Of the 21 forced entries with a usable record, 4 gaps were under the drone's 0.40 m diameter, 7 were 0.40 to 0.45 m and 3 were 0.45 to 0.50 m. So 11 should never have been entered.
- A small classifier that sees only the 4 sector readings separates a passable gap of at least 0.5 m from one under 0.45 m with an AUC of about 0.98 at the gap mouth, but only 0.80 while still 0.3 to 0.6 m away. Each sector reports only its nearest distance and no bearing, so from a distance the readings cannot encode how wide a gap is.
- So part of the failure is the policy, such as r2#6749, which read about 0.32 m on both front sectors, stood at the mouth for 5 steps and then sped in. Part is the sensor, which cannot tell gap width in time to turn away early.

## Sensor experiments and the move to two cameras

| Arm | Readings | best | final | Pass |
|---|---|---|---|---|
| V4_8 | 4 sectors (baseline_v4) | 99.02 | 98.56† | yes |
| V4CAM5 | 3 cameras x 5 bins, 15 readings | 98.97 | 97.53† | no |
| V4FRONT5 | the front camera's 5 bins only | 98.69 | 97.03† | no |
| V4CAM5WD | V4CAM5 on baseline_v4.2 with weight decay 1e-4 | 98.80 | 98.28 | no |
| V4FRONT2CAMWD | 2 cameras x 5 bins on baseline_v4.2 with weight decay 1e-4 | 99.24 | 97.79 | no |

- **5 bins per camera** split each 73 degree camera into five 14.6 degree bins, each reporting the nearest depth. A real depth camera gives this directly, so it stays within what the drone can measure. The bins add the bearing that the sector readings lack.
- **The side cameras are needed when there is only one front camera.** V4FRONT5 crashed about 126 times against about 89, many of them in gaps of 0.8 m or more.
- **Two cameras turned to plus and minus 36.5 degrees** cover the front 146 degrees with no gap and nothing to the sides. The user wants fewer sensors on the real drone, and two cameras were enough to tell left from right and to get around obstacles.
- V4CAM5WD was the candidate for baseline_v4.3 and failed, so baseline_v4.3 was never made.

## Why baseline_v6 looks the way it does

On 2026-11-02 the user decided that the 5-bin camera readings belong in the baseline, so baseline_v4.2 became `baseline_v6` and baseline_stop_v5 became `baseline_stop_v6`, and every new experiment builds on these two. The user chose two cameras over three because they need fewer sensors and already handle left and right.

| Branch | Built from | Change | State |
|---|---|---|---|
| `baseline_v6` (`410134d`) | baseline_v4.2 | merge of `exp/v4-2cam-wd` | 18 inputs, 10 of them camera bins |
| `baseline_stop_v6` (`addd60d`) | baseline_stop_v5 | merge of `exp/v4-2cam-wd` | 22 inputs, 10 camera bins plus the previous action and the pose offset |

- The merge also brings the optional AdamW weight decay, which defaults to 0, so neither baseline uses it. This is the only difference from V4FRONT2CAMWD, which used 1e-4.
- **What argues against it.** On the old settings two cameras had the highest best of the three sensors but the lowest final, 97.79%, with 2 of 3 runs under 98%. So on the overall rate there is not yet evidence that the bins beat four sectors, and the choice rests on the narrow-gap results and the sensor count. The v6 runs answer whether two cameras hold up on the v5 settings.

**Results so far.** The chains run 0, 1, 2, 3, 5 and 8 obstacles. V6 r3 has not started yet, so V6 has 2 runs.

| Arm | 3 obstacles best / final | 5 obstacles best / final | 8 obstacles |
|---|---|---|---|
| V4EZ (baseline_v4.2) | 99.67 / 99.30 | 99.33 / 99.11 | 99.13 / 98.58 |
| V6 | 99.51 / 98.94 | 99.22 / 99.11 | running |
| baseline_stop_v5 | 99.73 / 99.28 | 99.62 / 98.19 | 99.45 / 98.35 |
| STOP6 | 99.58 / 99.16 | running | not started |
| STOP6CRASH100 | 99.45 / 98.73 | running | not started |

**Four experiments on baseline_stop_v6**, all started from the narrow-gap and stalling analyses above.

| Arm | Change | Reason |
|---|---|---|
| STOP6SIM3W | the similarity penalty compares the pose 20, 40 and 80 steps back, at most -0.1 each, with the 40 and 80-step windows judging position alone at 0.1 and 0.2 m | A drone turning in place one way changes its heading by 1.6 rad every 20 steps and so escaped the penalty, which also counts heading. The total stays at -0.3, so it tests the longer windows and not a larger charge. The state shows all three offsets, 9 values instead of 3 |
| STOP6CRASH100 | crash penalty -100 instead of -50 | A crash is dangerous on the real drone |
| STOP6PROG05 | progress weight 0.05 instead of 0.15 | Tests whether the progress reward discourages a detour |
| STOP6POT | potential-based progress shaping | The same question with a reward that provably does not change the best policy |

- **What Codex and Claude expected of STOP6CRASH100.** With only the terminal rewards the expected return is 150p - 50 for a success rate p, so the size of the crash penalty does not change which policy succeeds most. What makes the drone take risks is the discount, since a detour delays the +100. A larger penalty may turn crashes into timeouts, so this arm reports timeouts and crashes along with success.
- **The progress reward and a detour.** It is 0.15 x speed x cos(heading error) per step, which sums to about 1.5 per metre gained toward the target whatever the speed, so backing off 1 m costs 1.5 however slowly it is done. Turning in place costs nothing. Discounting costs a detour more, since arriving 10 steps later at gamma 0.99 loses about 10 of the +100. A reward for facing the target alone was tried long ago, and the network learned to face the target and stand still.
- **Potential-based shaping** (Ng, Harada and Russell, *Policy invariance under reward transformations*, ICML 1999) pays 1.5 x (previous distance - gamma x current distance). It sums to the same amount on every path, so it cannot make a detour look worse. The current reward equals this shaping minus 1.5 x (1 - gamma) x remaining distance on every step, a hidden charge of about 0.075 per step at 5 m. So STOP6POT removes only that charge, and if nothing changes, the reward shape is ruled out as a cause.

**Failures of STOP6 at 2 obstacles.** The weakest run, r2 at 97.65% final, failed 706 episodes over its last 3 checkpoints.

| Pattern | Share | What happens |
|---|---|---|
| Stopped until timeout | 66% | 267 stop about 1.2 m from the target with the target behind them at about 137 degrees and turn left and right in place. 199 stop within the first 15 steps 0.5 m from an obstacle on the line to the target |
| Crash into an obstacle in view | 14% | Median crash at step 23, at full speed, side-swiping an obstacle at about 71 degrees in the outermost bin |
| Circling near the target | 8% | Ends within 2 m of the target while still moving |
| Swinging | 4% | |
| Crash in a gap | 3% | |
| Crash on the blind side | 2% | |
| Far timeout | 2% | |

- With two cameras the drone sees nothing beyond 73 degrees to either side, against 219 degrees of coverage with three cameras. Turning toward a target behind it means turning toward space it cannot see, so this pattern may come from losing the side view rather than from the stop action. That needs a check against baseline_stop_v5 at the same stage.

## Evidence on the two core problems

This section also includes the ablations from before baseline_v2. The numbers come from five sources.
- "Stage 0" means no obstacles, 10 runs per arm.
- "3 obstacles, old" means 3 obstacles, seed 1, 2,000 layouts, 1500 ep, 3 runs per arm.
- "3 obstacles, old, 3000 ep" is the same evaluation with 3000 ep of training.
- "3 obstacles, seed 777" means 3 obstacles, seed 777, 10,000 layouts.
- "5 obstacles" is the main evaluation of this document, seed 777 with 10,000 layouts.

### Swinging and stopping

The trajectories in "Failure types" show that every failure other than a crash ends with the drone stopped, so the two problems can be treated together.

| Attempt | Setting | Effect on swinging and stopping |
|---|---|---|
| Hysteresis at evaluation (a left and right reversal must win by delta before the action switches) | 3 obstacles, old | best rises by 1 to 8 percentage points, only at evaluation, and stopping near the target is not fixed |
| Two output heads (V9) | 3 obstacles, old | Oscillation falls from 22.7% to 9.6%, but far timeouts rise from 1.5% to 15.2%, so swinging turns into stopping |
| progress reward 0.15 (R2) | 3 obstacles, old | Alone it raises oscillation from 22.7% to 37.9%, and it helps only together with V9 |
| Per-step penalty -0.01 (R1) | Stage 0 | Worse, with stopping near the target rising to 15.5% |
| Reversal penalty (V3, on top of V2) | Stage 0 | Oscillation rises from 2.0% to 5.1% |
| Advantage Learning (V7) | 3 obstacles, old | best only 34.8%, with 25.4% spinning in place |
| Oversampling large heading errors at the start (V8) | Stage 0 | final median falls to 62.8%, p=0.031 |
| Batch 256, Huber loss | Stage 0 | Oscillation rises to 15% to 18% |
| Previous-action input (V2, V2S) | 5 obstacles | Both worse, V2S oscillation 7.45% |
| Per-step time penalty -0.05 | 5 obstacles | Stopping falls from 5.90% to 1.35%, while oscillation and crashes rise |
| No hover | 3 and 5 obstacles | Stopping disappears, but crashes reach 14% to 17% with 3 obstacles and 27.65% with 5 |
| DDQN | 5 obstacles | Oscillation 3.78% to 3.22%, and with 3 obstacles 4.94% to 2.05% |
| SOFT | 5 obstacles | Oscillation 3.78% to 2.07%, far timeouts 3.75% to 1.47% |
| DDQN + three forward speeds | 5 obstacles | Oscillation 3.22% to 0.02%, far timeouts 2.30% to 0.08%, crashes 1.26% to 7.63% |
| DDQN + SOFT + three forward speeds | 5 obstacles | Oscillation 0.01%, far timeouts 0%, crashes 8.21% |
| Crash penalty doubled on top | 5 obstacles | Crashes do not drop (8.98%) |
| Three forward speeds plus dt 0.1 | 5 obstacles | Oscillation 0.11%, crashes 4.68%, but about 40% of crashes come from dithering on the line ("The observation limit") |
| **Slowest speed lowered to 0.075 m/s, plus dt 0.1 (C)** | 5 obstacles | Crashes fall to 0.65%, oscillation 0.37% |
| **Front sector split in two** | 5 obstacles | Oscillation 0.00%, crashes 0.74% |
| **C plus 4 sectors** | 5 obstacles | Oscillation 0.07%, crashes 0.43% |

My current understanding has three parts.
- A step spent stopped earns a progress reward of 0, while moving toward an obstacle risks a crash. So in front of an obstacle stopping is often the action with the highest Q-value, and once stopped the left and right turns score almost the same, so the drone swings in place.
- Once stopping is removed, swinging and crashing have two separate causes. One is a slowest speed whose turn radius is too large to turn away ("The action-space limit"). The other is an input that cannot tell whether an obstacle is on the left or the right, so the drone hesitates on the line ("The observation limit").
- Lowering the slowest speed and splitting the front sector each bring success to about 99%, and they handle different kinds of failure.

### Getting worse with more training (final far below best)

| Attempt | Setting | Change in final |
|---|---|---|
| Fixing the first transition of an episode (A1) | Stage 0 | median 82.3% to 97.3%, p=0.57 |
| replay buffer 50,000 to 500,000 | 3 obstacles, old | 41.1% to 53.1% |
| Learning rate decayed from ep500 to 1/10 (E1) | 3 obstacles, old | 41.1% to 52.2%, and 99.0% at Stage 0 when combined with R2, but only 92.3% on a rerun |
| gamma 0.995 (E2) | 3 obstacles, old | 41.1% to 51.3% |
| Exploration floor 0.05 to 0.03 | 3 obstacles, old, 3000 ep | 74.3% to 65.5%, while best rises |
| 2 gradient updates per environment step | 3 obstacles, seed 777 | 60.4% to 75.6% |
| Target copied every 20 ep | 3 obstacles, seed 777 | 60.4% to 73.7% |
| Learning rate 3e-4, 256-unit network | 3 obstacles, old | No help |
| DDQN | 5 obstacles | 64.15% to 62.55% |
| SOFT | 5 obstacles | 64.15% to 68.18%, p=0.73 |
| Per-step time penalty -0.05 | 5 obstacles | About 9 percentage points higher |
| **DDQN + SOFT** | 5 obstacles | **64.15% to 83.12%**, p=0.11 |
| DDQN + SOFT plus three forward speeds | 5 obstacles | 83.12% to 87.60%, p=0.57 |
| DDQN + SOFT with the buffer enlarged to 2,000,000 | 5 obstacles | 83.12% to 73.71%, p=0.32 |
| DDQN + SOFT + three speeds with the crash penalty doubled | 5 obstacles | 87.60% to 75.01%, p=0.33 |
| DDQN + SOFT with gamma 0.996 | 5 obstacles | 81.28% to 63.54%, p=0.025 |
| **DDQN + SOFT with dt 0.1** | 5 obstacles | **81.28% to 92.59%**, p=0.065, all 5 runs above 90% |
| C plus 4 sectors plus V2S (baseline_v4) | 5 obstacles | 98.91%, only 0.5 percentage points below best |

My current understanding has three parts.
- The changes that raised final the most all concern how stable the learning target is, such as a longer target copy interval and DDQN + SOFT. SOFT or DDQN alone did not help.
- dt 0.1 is what brings final up to best, for the reason in "Why a longer control step trains more stably". A longer horizon by itself does not help.
- With baseline_v4, best and final differ by only about 0.5 percentage points, so degrading with more training is largely solved.

## Conclusions so far

1. **Oscillation** is solved. Three forward speeds removed stopping, a lower slowest speed lets the drone turn away in time, and splitting the front sector tells the network which side an obstacle is on. baseline_v4 oscillates in 0.06% of layouts.
2. **Training instability** is largely solved. DDQN + SOFT brought final to 81% to 83%, dt 0.1 to 92.59%, and baseline_v4 reaches 98.91%.
3. **Success with 5 obstacles** rose from 92.20% for baseline_v3 to 99.44% for baseline_v4.
4. Doubling the crash penalty, enlarging the buffer to 2,000,000 and gamma 0.996 all made things worse and can be dropped. A Q network with one output per action helped only a little, and its outputs grow with every combination of actions, so it will not be tried again.
5. The new and old curricula do not differ, so 1 → 2 → 3 → 5 stays, now followed by 8 obstacles.
6. **At 8 obstacles** baseline_v4.2 passes at 99.13% and 98.58%, and the best arm without stop, V4LONGG99, reaches 99.38% and 99.19%†. Temporally-extended exploration, 60 s episodes together with gamma 0.99, and the proximity penalty are the changes that helped. Time and timeout penalties, a stage-local learning-rate decay, gamma 0.995 and prioritized replay on top of baseline_stop_v5 did not.
7. **The stop action** is kept for the U-turn out of a gap that is too narrow, even though it scores no better overall and its finals swing more. With the similarity penalty, temporally-extended exploration, 60 s episodes and gamma 0.99 it passes, and baseline_stop_v5 reaches 99.45% and 98.35%. On 0.4 to 0.6 m gaps it succeeds about 10 points more often than baseline_v4.2, although that comparison still mixes the stop action with the long-episode package.
8. **Late-stage instability is back at 8 obstacles.** Single checkpoints late in a stage can drop 20 to 40 points, which is why the final is the mean of the last 3 checkpoints. The swings are training instability, and narrow-gap success does not trend upward during a stage.
9. **The sensor is part of the narrow-gap problem.** Four sector readings carry no bearing, so the drone cannot tell a passable gap from a closed one until it is at the mouth. Two cameras of 5 bins each did better on narrow gaps, which is why they are in baseline_v6, but their overall final is not yet shown to be as good.

## Limitations

- Most arms have only 3 to 5 runs, and the four cells of the 2x2 on C cannot be told apart statistically.
- Every result comes from a noise-free 2D kinematic environment. Real sensor noise and Isaac Lab dynamics are still untested.
- dt 0.1 differs from the 0.02 s of the Isaac env. Moving to Isaac Lab means letting the network decide every 0.1 s while the low-level control holds the same action in between.
- Contact is only tested at the end of each step. At dt 0.1 a full-speed step covers 0.15 m and can graze up to about 4 mm into the contact zone without counting as a crash, which is an accepted error.
- 4 sectors and three forward speeds change the state and the action space, so these runs cannot start from the shared obstacle-free checkpoint and must train their own first stage with no obstacles.
- The pass rule of 99% and 98% says whether a setting is good enough, not whether it is better. Most arms that pass differ from each other by less than the spread between their own runs.
- Finals from before 2026-10-31 that were never rescored are the last checkpoint alone and can sit on a dip. They are marked †.
- The hard group differs between arms, because it depends on how each model flew. Its rates are compared together with the episode counts.
- Two cameras see nothing beyond 73 degrees to either side. The real drone has to carry the same camera layout for these results to transfer.

## Next steps

1. Finish the baseline_v6 runs, V6, STOP6 and the four experiments in "Why baseline_v6 looks the way it does", and compare them with baseline_v4.2 and baseline_stop_v5 on the overall rate and on the hard group.
2. Check whether the stopped timeouts with the target behind the drone in STOP6 come from the narrower field of view, by replaying baseline_stop_v5 at the same stage.
3. Separate the stop action from the long-episode package on narrow gaps, by running the hard group on V4LONGG99, which has the long-episode package without the stop.
4. Try three methods from the literature that bring final closer to best. The 8-obstacle stage now shows late dips of 20 to 40 points, so the gap between best and final is large enough to measure. They are listed in order of cost.
   - Evaluate with the soft target network (Nikishin et al., 2018, and Anschel et al., *Averaged-DQN*, ICML 2017). The soft target is already a moving average of the online network's weights, so evaluating it can reduce the jumps between checkpoints. Training needs no change, only saving the target network as well.
   - Add a LayerNorm after fc1 and fc2 of the Q network (Gallici et al., *Simplifying Deep Temporal Difference Learning*, ICLR 2025). The paper proves that LayerNorm with an L2 penalty makes TD learning converge, and the parallel sampling it relies on is close to the 64 environments run at once here.
   - Replace Q-value regression with classification over a histogram, the HL-Gauss loss (Farebrother et al., *Stop Regressing*, ICML 2024), to soften noisy targets and a shifting distribution. It adds two hyperparameters, the number of bins and sigma, so it comes last.
5. Before moving to Isaac Lab, run baseline_v6 in `camera` mode to confirm that emulated camera readings give the same results.
6. Memory of past steps for maze-like layouts, first stacked recent inputs, then a GRU, and an attention encoder last. The state already carries the pose offset from 20 steps back as a summary. A 1-layer attention encoder over the last 20 actions would add 11% to 91% parameters for an embedding of 8 to 32, but 2 to 14 times the compute per forward pass, which matters on the CPU.
