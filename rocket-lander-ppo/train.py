"""Train reproducible CPU PPO; save raw/shaped reward CSV and checkpoints."""
import argparse
import csv
import json
import time
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from reward_shaping import ShapedReward


class EpisodeCSV(BaseCallback):
    def __init__(self, filename):
        super().__init__()
        self.file = Path(filename)
        self.raw = self.shaped = self.steps = 0
        self.episodes = 0
        self.file.parent.mkdir(parents=True, exist_ok=True)
        with self.file.open('w', newline='') as f:
            csv.writer(f).writerow(['timesteps', 'episode', 'steps', 'raw_reward',
                                    'shaped_reward', 'landing_success'])

    def _on_step(self):
        # One environment, so there is one info dictionary per step.
        info = self.locals['infos'][0]
        self.raw += info['raw_reward']
        self.shaped += info['shaped_reward']
        self.steps += 1
        if self.locals['dones'][0]:
            self.episodes += 1
            with self.file.open('a', newline='') as f:
                csv.writer(f).writerow([self.num_timesteps, self.episodes,
                    self.steps, round(self.raw, 5), round(self.shaped, 5),
                    int(info['landing_success'])])
            self.raw = self.shaped = self.steps = 0
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--shaped', action='store_true')
    p.add_argument('--steps', type=int, default=1_000_000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--checkpoint-every', type=int, default=100_000)
    p.add_argument('--out', default='runs')
    args = p.parse_args()
    if args.steps <= 0 or args.checkpoint_every <= 0:
        p.error('--steps and --checkpoint-every must be positive')
    run = Path(args.out) / ('shaped' if args.shaped else 'baseline')
    run.mkdir(parents=True, exist_ok=True)
    set_random_seed(args.seed)
    env = Monitor(ShapedReward(gym.make('LunarLander-v3'), enabled=args.shaped))
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    model = PPO('MlpPolicy', env, device='cpu', seed=args.seed, verbose=1,
                n_steps=1024, batch_size=64, n_epochs=5, learning_rate=3e-4)
    callback = CallbackList([
        EpisodeCSV(run / 'episodes.csv'),
        CheckpointCallback(save_freq=args.checkpoint_every,
                           save_path=str(run / 'checkpoints'), name_prefix='ppo')
    ])
    start = time.perf_counter()
    try:
        model.learn(total_timesteps=args.steps, callback=callback,
                    progress_bar=False)
        model.save(str(run / 'model'))
    finally:
        env.close()
    elapsed = time.perf_counter() - start
    summary = dict(mode='shaped' if args.shaped else 'baseline', seed=args.seed,
                   requested_steps=args.steps, actual_steps=model.num_timesteps,
                   train_seconds=round(elapsed, 2), model=str(run / 'model.zip'))
    (run / 'training.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
