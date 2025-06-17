try:
    import gym
    from gym import spaces
except Exception:  # pragma: no cover - gym may be missing
    class _Box:
        def __init__(self, low, high, shape, dtype):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    class _Env:
        pass

    class _gym:
        Env = _Env

    gym = _gym()
    spaces = type("spaces", (), {"Box": _Box})

import numpy as np

from moto import (
    Motorcycle,
    DT,
    SIMULATION_TIME,
    TARGET_SPEED,
    add_perturbation,
    calculate_energy_consumption,
    device,
)


class MotorcycleEnv(gym.Env):
    """Gym-compatible environment wrapping the Motorcycle simulation."""

    def __init__(self, energy_weight: float = 0.0, device: str = device):
        super().__init__()
        self.motorcycle = Motorcycle(device=device, mass_std=1e-3)
        self.dt = DT
        self.sim_time = SIMULATION_TIME
        self.energy_weight = energy_weight
        self.device = device
        self.steps = int(self.sim_time / self.dt)
        self.current_step = 0
        import torch
        self.rng = torch.Generator(device)

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )

    def reset(self, seed: int | None = None):
        if seed is not None:
            np.random.seed(seed)
            import random

            random.seed(seed)
            import torch

            torch.manual_seed(seed)
            self.rng.manual_seed(seed)
        self.motorcycle.reset()
        self.current_step = 0
        state = np.array([self.motorcycle.speed.item()], dtype=np.float32)
        return state, {}

    def step(self, action):
        throttle, brake = action
        add_perturbation(self.motorcycle, self.dt, generator=self.rng)
        speed = self.motorcycle(throttle, brake, self.dt, generator=self.rng)
        reward = -((speed - TARGET_SPEED) ** 2).item()
        energy = calculate_energy_consumption(self.motorcycle, throttle, brake, self.dt)
        reward -= self.energy_weight * energy
        self.current_step += 1
        done = self.current_step >= self.steps
        state = np.array([speed.item()], dtype=np.float32)
        return state, reward, done, {}
