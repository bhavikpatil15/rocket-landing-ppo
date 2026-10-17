"""Plot both training curves on the SAME raw reward metric."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    return np.array([float(r['timesteps']) for r in rows]), np.array([float(r['raw_reward']) for r in rows])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', default='runs')
    p.add_argument('--window', type=int, default=30)
    p.add_argument('--out', default='runs/comparison.png')
    a = p.parse_args()
    fig, ax = plt.subplots(figsize=(10, 5))
    for label in ('baseline', 'shaped'):
        x, y = load(Path(a.runs) / label / 'episodes.csv')
        if not len(x):
            print(f'No completed episodes in {label}; skipping')
            continue
        ax.plot(x, y, alpha=0.15)
        window = min(max(a.window, 1), len(y))
        smoothed = np.convolve(y, np.ones(window) / window, 'valid')
        ax.plot(x[window - 1:], smoothed, label=f'{label} ({window}-episode avg)')
    ax.set(xlabel='Environment timesteps', ylabel='Raw episode reward',
           title='PPO LunarLander-v3 — baseline vs reward shaping')
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=160)
    plt.close(fig)
    print(f'Saved {a.out}')


if __name__ == '__main__':
    main()
