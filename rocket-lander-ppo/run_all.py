"""One command: smoke test both, train both, evaluate, plot and record."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def invoke(*args):
    print('\n$', sys.executable, *map(str, args), flush=True)
    subprocess.run([sys.executable, *map(str, args)], check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=1_000_000, help='per model')
    p.add_argument('--smoke-steps', type=int, default=10_000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--checkpoint-every', type=int, default=100_000)
    p.add_argument('--skip-smoke', action='store_true')
    args = p.parse_args()
    if not args.skip_smoke:
        for mode in ('baseline', 'shaped'):
            invoke('train.py', '--steps', args.smoke_steps, '--seed', args.seed,
                   '--checkpoint-every', args.smoke_steps, '--out', 'smoke',
                   *(['--shaped'] if mode == 'shaped' else []))
            invoke('evaluate.py', '--model', f'smoke/{mode}/model.zip',
                   '--out', f'smoke/{mode}/eval', '--episodes', 3)
    for mode in ('baseline', 'shaped'):
        invoke('train.py', '--steps', args.steps, '--seed', args.seed,
               '--checkpoint-every', args.checkpoint_every,
               *(['--shaped'] if mode == 'shaped' else []))
        invoke('evaluate.py', '--model', f'runs/{mode}/model.zip',
               '--out', f'runs/{mode}/eval')
    invoke('plot.py')
    invoke('record.py')
    reports = {m: json.loads(Path(f'runs/{m}/eval/evaluation.json').read_text())
               for m in ('baseline', 'shaped')}
    times = {m: json.loads(Path(f'runs/{m}/training.json').read_text())['train_seconds']
             for m in ('baseline', 'shaped')}
    lines = ['# Final comparison (100 identical evaluation seeds)', '',
             '| Metric | Baseline | Shaped |', '|---|---:|---:|',
             f"| Success rate | {reports['baseline']['landing_success_rate']:.1%} | {reports['shaped']['landing_success_rate']:.1%} |",
             f"| Mean raw reward | {reports['baseline']['mean_raw_reward']:.2f} | {reports['shaped']['mean_raw_reward']:.2f} |",
             f"| Average episode steps | {reports['baseline']['avg_steps']:.1f} | {reports['shaped']['avg_steps']:.1f} |",
             f"| CPU training seconds | {times['baseline']:.1f} | {times['shaped']:.1f} |", '',
             'See runs/comparison.png and runs/*/eval/episodes_ranked.csv.']
    Path('runs/report.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
