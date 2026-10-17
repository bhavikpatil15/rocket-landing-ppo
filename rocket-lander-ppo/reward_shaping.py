"""Small additive reward terms for discrete LunarLander-v3.

Observation: x, y, vx, vy, angle, angular velocity, left leg, right leg.
Coordinates are pad-centered and velocities are in Gymnasium's normalized units.
Terminal success is taken from LunarLander's +100 terminal raw reward.
"""
import numpy as np
import gymnasium as gym

# Tune each coefficient independently; positive means reward/bonus.
DISTANCE_PROGRESS = 0.8    # bonus per unit reduction in distance to pad
NEAR_PAD_SPEED = 0.6       # bonus for slowing down within NEAR_RADIUS
UPRIGHT = 0.03            # per-step bonus when angle is near zero
TIME_PENALTY = 0.02       # small negative reward every step
MAIN_FUEL_PENALTY = 0.03  # applied when main engine (action 2) fires
SIDE_FUEL_PENALTY = 0.01  # applied when either side engine (1 or 3) fires
SAFE_LANDING_BONUS = 40.0
CRASH_PENALTY = 40.0
NEAR_RADIUS = 1.0


def distance(obs):
    return float(np.linalg.norm(np.asarray(obs)[:2]))


def speed(obs):
    return float(np.linalg.norm(np.asarray(obs)[2:4]))


class ShapedReward(gym.Wrapper):
    """Provide `raw_reward`, `shaping_bonus`, `shaped_reward`, `landing_success` in info.

    If enabled=False, training uses original rewards but both signals are still logged.
    Do not infer landing from merely contacting the ground: Gymnasium awards
    +100 for settled/sleeping landers and -100 for crashes/out-of-bounds.
    """

    def __init__(self, env, enabled=True):
        super().__init__(env)
        if not isinstance(self.action_space, gym.spaces.Discrete):
            raise ValueError('Use the discrete LunarLander-v3 environment')
        self.enabled = enabled
        self.previous = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous = np.array(obs, copy=True)
        return obs, info

    def step(self, action):
        obs, raw, terminated, truncated, info = self.env.step(action)
        action = int(action)
        old, new = self.previous, np.asarray(obs)
        # Positive only for improvements: moving closer should be rewarded.
        progress = DISTANCE_PROGRESS * np.clip(distance(old) - distance(new), -0.2, 0.2)
        # Slow near the pad, but do not demand slow flight at high altitude.
        slowing = (NEAR_PAD_SPEED * np.clip(speed(old) - speed(new), -0.2, 0.2)
                   if distance(new) < NEAR_RADIUS else 0.0)
        # Bounded upright bonus favors stable low-angle descent.
        upright = UPRIGHT * max(0.0, 1.0 - abs(float(new[4])) / 0.35)
        # Discourage wasting time and propellant.
        time_cost = -TIME_PENALTY
        fuel_cost = -(MAIN_FUEL_PENALTY if action == 2 else
                      SIDE_FUEL_PENALTY if action in (1, 3) else 0.0)
        # Terminal bonuses reflect genuine environment success/failure.
        landed = bool(terminated and float(raw) >= 99.0)
        crashed = bool(terminated and float(raw) <= -99.0)
        terminal = SAFE_LANDING_BONUS if landed else (-CRASH_PENALTY if crashed else 0.0)
        extra = float(progress + slowing + upright + time_cost + fuel_cost + terminal) if self.enabled else 0.0
        self.previous = np.array(new, copy=True)
        info = dict(info)
        info.update(raw_reward=float(raw), shaping_bonus=extra,
                    shaped_reward=float(raw) + extra, landing_success=landed,
                    crash=crashed)
        return obs, float(raw) + extra, terminated, truncated, info
