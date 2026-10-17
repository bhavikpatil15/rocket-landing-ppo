# Rocket Lander PPO — LunarLander-v3

Compact, CPU-only PPO experiment comparing Gymnasium's native rewards with additive reward shaping. Uses discrete actions (0: idle, 1/3: side engine, 2: main engine), `MlpPolicy`, one environment, and reproducible seeds. **Default: 1,000,000 steps per model (2,000,000 total, plus two 10k smoke tests).** Laptop runtime varies and is not guaranteed to be below 24 hours.

## Install (Python 3.11 or 3.12 recommended)

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The two core packages are pinned to Gymnasium 1.2.2 and Stable-Baselines3 2.7.0 to avoid API drift; the Box2D extra provides LunarLander physics. If Box2D must compile from source, SWIG and build tools may be necessary (`pip install swig`); Python 3.11/3.12 is usually simpler than newer releases. MP4 encoding uses `imageio-ffmpeg` and its ffmpeg backend. Nothing is fetched by the project scripts; pip installs packages and may fetch their dependencies.

## One-command full experiment

```bash
python run_all.py --steps 1000000 --seed 42 --checkpoint-every 100000
```

This runs **10k smoke training + 3-episode checks for both modes first**, then baseline training, shaped training, 100-episode fixed-seed evaluation of each, comparison plot, and an MP4 recording. It stops immediately if any step fails. A short local trial without the million-step training:

```bash
python run_all.py --steps 10000 --smoke-steps 10000 --seed 42
```

To run the full experiment after a separately verified smoke test, add `--skip-smoke`. **Do not interpret 10k-step metrics as final learning results.** A full run saves `runs/report.md` with numbers measured on your machine.

## Individual commands

```bash
python train.py --steps 1000000 --seed 42 --checkpoint-every 100000
python train.py --shaped --steps 1000000 --seed 42 --checkpoint-every 100000
python evaluate.py --model runs/baseline/model.zip --out runs/baseline/eval --seed 1000 --episodes 100
python evaluate.py --model runs/shaped/model.zip --out runs/shaped/eval --seed 1000 --episodes 100
python plot.py --runs runs --out runs/comparison.png
python record.py --model runs/shaped/model.zip --out runs/landing.mp4 --seed 1000 --find-landing 100
```

The baseline trains first and writes `runs/baseline/episodes.csv`; shaped training writes `runs/shaped/episodes.csv`. Both runs save `model.zip`, `training.json`, and `checkpoints/ppo_<step>_steps.zip`. CSV columns: global timesteps, episode, episode steps, raw reward, shaped reward, landing success. `plot.py` compares **raw** reward for both on one chart, with a 30-episode rolling mean (tunable via `--window`). Each evaluation produces `evaluation.json` and `episodes_ranked.csv` sorted best-to-worst by *raw* return; top and bottom five episodes are also shown in JSON. The video recorder scans up to 100 seeds for a genuine landing; if none succeeds it still writes a video and says so. Rendering and encoding may be slow.

## Reward shaping terms

All tunable constants live at the top of [`reward_shaping.py`](reward_shaping.py). On each step the wrapper reads observation `(x, y, vx, vy, angle, angular velocity, left-leg contact, right-leg contact)` and applies:

| Term | Default | Purpose |
|---|---:|---|
| `DISTANCE_PROGRESS` | 0.8 | Reward decrease in distance from lander to pad center; capped per step |
| `NEAR_PAD_SPEED` | 0.6 | Reward lower speed only near pad (distance < `NEAR_RADIUS`, default 1.0); capped |
| `UPRIGHT` | 0.03 | Small bonus scaled by how close absolute tilt is to zero |
| `TIME_PENALTY` | 0.02 | Subtract each environment step |
| `MAIN_FUEL_PENALTY` | 0.03 | Subtract when main engine fires (action 2) |
| `SIDE_FUEL_PENALTY` | 0.01 | Subtract when a side engine fires (actions 1/3) |
| `SAFE_LANDING_BONUS` | 40 | Additional bonus on the environment's +100 successful terminal transition |
| `CRASH_PENALTY` | 40 | Additional penalty on the environment's −100 failed terminal transition |

The per-step reward for shaped training is **raw environment reward + these additional terms**. Raw and shaped returns are both saved per episode. The unshaped baseline uses the exact original reward. Evaluation intentionally **turns shaping off for both**: mean reward and success rate are comparable. A successful landing is detected from the underlying LunarLander terminal +100, not merely touching a leg to the ground. The environment's episode time limit produces truncation, not success.

## Why might shaping help—or hurt?

Progress and slowing rewards can make the early sparse terminal landing signal easier to discover; tilt bonuses encourage controlled descent. However, native LunarLander rewards **already include** position, velocity, angle, leg contact, and motor penalties. Extra shaping can therefore duplicate incentives and *hurt* learning: a policy might hover to farm upright rewards, overuse fuel while chasing distance bonuses, or change risk preference due to additional terminal rewards. This implementation deliberately uses small, capped incremental bonuses, but it does **not** use guaranteed policy-invariant potential-based shaping. **Evidence that it helped requires higher fixed-seed success rate / raw reward at the same training budget; it must not be asserted in advance.** Read `runs/report.md` and `runs/comparison.png` to decide.

## Reproducibility and scope

Training uses one environment, fixed NumPy/PyTorch/SB3 environment seed and CPU. Evaluation checks the exact 100 seeds `1000..1099` in both models with deterministic policy actions. PPO's rollout collection rounds training timesteps up to its `n_steps=1024` boundaries, so actual steps can exceed the request slightly. Seed control improves repeatability, but CPU library/platform differences may still change outcomes. Success is not guaranteed within one million steps. For stronger conclusions repeat with multiple training seeds; a single-seed result is exploratory.

## Current execution status

The authoring environment used to construct this repo had Python 3.13 but lacked `gymnasium` and `stable_baselines3`. Its pip index could not be reached (DNS/network failure). Consequently **no 10k smoke run, million-step training, or 100-episode comparison was performed here**; the report values must be produced by running the commands above on a machine with pip access. Python syntax validation does not substitute for runtime testing.
