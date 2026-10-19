# 🚀 Rocket Lander AI — PPO Reinforcement Learning

**Teaching an AI agent to land a rocket using Proximal Policy Optimization (PPO).**  
A reproducible, CPU-based comparison of the standard **LunarLander-v3** reward function and a custom **reward-shaped** alternative.

**Python 3.11 · Gymnasium · Stable-Baselines3 · PPO · Box2D · NumPy · Matplotlib · ImageIO**



### 🚀 Baseline agent — successful landing

https://github.com/user-attachments/assets/b9020941-2e81-4fdd-8b74-363596d85547

<!-- OPTIONAL: Paste a GitHub-hosted video URL on its own line below to show the video inline if GitHub supports it. -->


### 🛬 Reward-shaped agent — landing demonstration (if want to download the video )

[▶ Watch reward-shaped rocket landing (MP4)](media/shaped_landing.mp4)

<!-- OPTIONAL: Paste your shaped video URL on its own line below. -->

**[Space for your reward-shaped video preview or screenshot]**

> Video files must actually be present at the linked paths. If you haven't recorded the shaped video yet, generate it with the command in the recording section below. A recording should be labelled successful only if the console reports `landing_success=True`.


**Learning curves:** [Baseline vs. shaped training chart](runs/comparison.png) (include the generated PNG at this path).

## 📊 Results

Both agents trained for **1,000,000 timesteps each**, then underwent **100 deterministic evaluation episodes** using the **same 100 environment seeds (1000–1099)**. Both were scored with the *original environment reward* for a fair comparison.





| Metric | Baseline PPO | Reward-shaped PPO |
|---|---:|---:|
| **Successful landings** | **69 / 100 (69%)** | **35 / 100 (35%)** |
| Mean environment reward | **98.01** | -9.97 |
| Average episode length | **564.4 steps** | 782.9 steps |
| CPU training time | 144.2 s | 142.3 s |



<img width="1596" height="801" alt="Screenshot 2026-10-17 at 11 34 10 PM" src="https://github.com/user-attachments/assets/253ffeb8-d78a-48df-82c0-1d5265de5d74" />


**Combined training time:** **286.5 seconds (~4 min 47 sec)** on a MacBook Air, excluding evaluation, rendering, and installation. Runtime will vary by machine.

### Main takeaway

**The standard reward outperformed the custom shaping in this experiment.** The shaped agent landed in 35% of test episodes versus 69% for baseline. Additional reward terms may have overemphasized stability, approach behavior, or prolonged flight, but that interpretation requires further trajectory analysis. LunarLander's original reward already incentivizes position, velocity, orientation, and fuel efficiency, so adding more reward is not automatically beneficial. This is a **single-training-seed comparison**, not proof that shaping is always worse.

## 🧠 How it works

- **Environment:** `LunarLander-v3`, discrete action space (do nothing, fire left/right side engine, fire main engine).
- **State:** 8 observations describing position, velocity, angle, angular velocity, and leg contact.
- **Algorithm:** Stable-Baselines3 PPO with `MlpPolicy` on CPU.
- **Training:** 1 million timesteps per policy; checkpoints and episode-level CSV logs.
- **Evaluation:** 100 fixed-seed episodes per policy, ranked from best to worst by raw reward.
- **Visualization:** training-reward comparison plot and recorded MP4 flight demonstrations.

### Custom reward shaping

`reward_shaping.py` implements a `gymnasium.Wrapper` with adjustable coefficients:

| Component | Intention |
|---|---|
| Distance progress | Encourage moving toward the landing pad |
| Speed reduction near pad | Encourage slower final approach |
| Upright bonus | Encourage a small tilt angle |
| Time penalty | Discourage unnecessarily long episodes |
| Main and side engine penalties | Discourage excessive fuel use |
| Safe landing bonus | Reinforce successful landings |
| Crash penalty | Discourage terminal crashes |

Each step logs both **raw reward** and **shaped reward** separately. The custom terms are additive and **not guaranteed to preserve the optimal policy**.

## 📁 Project structure

```text
rocket-lander-ppo/
├── train.py              # PPO training, checkpoints, episode CSV
├── reward_shaping.py     # Tunable reward wrapper
├── evaluate.py           # Fixed-seed evaluation + ranked episodes
├── plot.py               # Baseline vs shaped reward curves
├── record.py             # Generate landing MP4 videos
├── run_all.py            # End-to-end experiment pipeline
├── requirements.txt
├── README.md
├── media/                 # Add selected videos here
│   ├── baseline_landing.mp4
│   └── shaped_landing.mp4
└── runs/                  # Locally generated (keep selected outputs)
    ├── comparison.png
    ├── report.md
    ├── baseline/
    │   ├── model.zip
    │   ├── episodes.csv
    │   └── eval/
    │       ├── evaluation.json
    │       └── episodes_ranked.csv
    └── shaped/
        ├── model.zip
        ├── episodes.csv
        └── eval/
            ├── evaluation.json
            └── episodes_ranked.csv
```

## ⚙️ Setup

Python **3.11** recommended. On macOS, use a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

**macOS troubleshooting:** If the `box2d-py` build reports `swig: No such file or directory`, install it using `brew install swig` and retry. If pygame/Box2D compilation fails on newer Python versions, use Python 3.11.

## ▶️ Run the experiment

**Quick smoke test (10k timesteps per agent):**

```bash
python run_all.py --steps 10000 --seed 42
```

**Full experiment (1M timesteps per agent):**

```bash
python run_all.py --steps 1000000 --seed 42
```

The full command runs smoke checks, trains both agents, evaluates both on 100 matched seeds, plots the learning curves, writes `runs/report.md`, and records a sample flight. For a repeated full run without smoke checks, add `--skip-smoke`.

**Individual commands:**

```bash
python train.py --steps 1000000 --seed 42
python train.py --shaped --steps 1000000 --seed 42
python evaluate.py --model runs/baseline/model.zip --out runs/baseline/eval --episodes 100 --seed 1000
python evaluate.py --model runs/shaped/model.zip --out runs/shaped/eval --episodes 100 --seed 1000
python plot.py
```

**Generate the two videos for the demo section:**

```bash
mkdir -p media
python record.py --model runs/baseline/model.zip --out media/baseline_landing.mp4 --seed 1000 --find-landing 100
python record.py --model runs/shaped/model.zip --out media/shaped_landing.mp4 --seed 1000 --find-landing 100
```

`--find-landing 100` searches for a successful flight in 100 seeds. Check Terminal for `landing_success=True`; the recorder produces a video even if no success is found. The previous automatically generated `runs/landing.mp4` is a **shaped-policy** recording, not the baseline video.

## 🔬 Reproducibility & limitations

- Training used **seed 42**; evaluation used seeds **1000–1099** with deterministic actions.
- One training seed per condition is insufficient for strong statistical claims; repeat with multiple seeds for greater confidence.
- PPO can slightly exceed the requested timesteps due to rollout batch boundaries.
- Exact results may vary with hardware and package versions.
- **Future experiments:** tune shaping weights, evaluate across multiple training seeds, compare sample efficiency, and analyze failure modes from trajectories.

## 📈 What I learned

This project demonstrates an end-to-end RL workflow: environment integration, PPO training, custom reward design, reproducible evaluation, experiment logging, video generation, and evidence-based comparison. The key lesson is that a seemingly intuitive reward function can reduce performance when it conflicts with or duplicates existing incentives.

---

**Experiment:** LunarLander-v3 · PPO/MlpPolicy · CPU · October 2026
