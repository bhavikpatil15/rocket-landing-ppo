"""Record an RGB-array policy rollout as an MP4, optionally finding a landing."""
import argparse
from pathlib import Path
import gymnasium as gym
import imageio.v2 as imageio
from stable_baselines3 import PPO
from reward_shaping import ShapedReward


def rollout(model, env, seed, capture=False):
    obs, _ = env.reset(seed=seed)
    frames = [env.render()] if capture else []
    landed = False
    for _ in range(1001):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))
        if capture:
            frames.append(env.render())
        if terminated or truncated:
            landed = info['landing_success']
            break
    return frames, landed


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='runs/shaped/model.zip')
    p.add_argument('--out', default='runs/landing.mp4')
    p.add_argument('--seed', type=int, default=1000)
    p.add_argument('--find-landing', type=int, default=100,
                   help='Search this many seeds for a successful landing; 0 disables')
    a = p.parse_args()
    model = PPO.load(a.model, device='cpu')
    env = ShapedReward(gym.make('LunarLander-v3', render_mode='rgb_array'), enabled=False)
    try:
        seed = a.seed
        if a.find_landing:
            for test_seed in range(a.seed, a.seed + a.find_landing):
                _, landed = rollout(model, env, test_seed)
                if landed:
                    seed = test_seed
                    break
            else:
                print('No successful landing in seed search; recording the first seed.')
        frames, success = rollout(model, env, seed, capture=True)
    finally:
        env.close()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(a.out, fps=50, codec='libx264', macro_block_size=16) as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f'Saved {a.out} | seed={seed} | landing_success={success}')


if __name__ == '__main__':
    main()
