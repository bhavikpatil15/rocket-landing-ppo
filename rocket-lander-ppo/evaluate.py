"""Evaluate a PPO policy with identical seeded episodes and RAW rewards."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from reward_shaping import ShapedReward


def run_episode(model, env, seed):
    obs, _ = env.reset(seed=seed)
    raw = shaped = 0.0
    for steps in range(1, 1002):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        raw += info['raw_reward']
        shaped += info['shaped_reward']
        if terminated or truncated:
            return dict(seed=seed, raw_reward=round(raw, 4),
                        shaped_reward=round(shaped, 4), steps=steps,
                        landing_success=int(info['landing_success']))
    raise RuntimeError('Unexpected episode longer than time limit')


def evaluate(model_path, out, seed=1000, episodes=100):
    model = PPO.load(str(model_path), device='cpu')
    # Disabled shaping ALWAYS: fair baseline-vs-shaped evaluation on raw reward.
    env = ShapedReward(gym.make('LunarLander-v3'), enabled=False)
    try:
        results = [run_episode(model, env, seed + i) for i in range(episodes)]
    finally:
        env.close()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fields = ['seed', 'raw_reward', 'shaped_reward', 'steps', 'landing_success']
    with (out / 'episodes_ranked.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda x: x['raw_reward'], reverse=True))
    summary = dict(model=str(model_path), episodes=episodes, seed_start=seed,
                   mean_raw_reward=float(np.mean([r['raw_reward'] for r in results])),
                   landing_success_rate=float(np.mean([r['landing_success'] for r in results])),
                   avg_steps=float(np.mean([r['steps'] for r in results])),
                   best_5=sorted(results, key=lambda r: -r['raw_reward'])[:5],
                   worst_5=sorted(results, key=lambda r: r['raw_reward'])[:5])
    (out / 'evaluation.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=1000)
    p.add_argument('--episodes', type=int, default=100)
    a = p.parse_args()
    evaluate(a.model, a.out, a.seed, a.episodes)


if __name__ == '__main__':
    main()
