import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from moto import Motorcycle, DT, SIMULATION_TIME, TARGET_SPEED, device
from rl_agent import RLAgent


def train_rl_agent(episodes: int = 200, seed: int = 0):
    """Train an RL agent to reach TARGET_SPEED."""
    torch.manual_seed(seed)
    motorcycle = Motorcycle(device=device, mass_std=1e-3)
    agent = RLAgent(device=device)
    writer = SummaryWriter()
    steps = int(SIMULATION_TIME / DT)

    for ep in range(episodes):
        motorcycle.reset()
        state = motorcycle.speed.item()
        log_probs = []
        rewards = []
        speeds = []
        for _ in range(steps):
            throttle, brake, log_prob = agent.select_action(state)
            speed = motorcycle(throttle, brake, DT)
            reward = -((speed - TARGET_SPEED) ** 2).item()
            log_probs.append(log_prob)
            rewards.append(reward)
            speeds.append(speed.item())
            state = speed.item()
        agent.update(log_probs, rewards)
        writer.add_scalar("Episode Reward", sum(rewards), ep)
        writer.add_scalar("Average Speed", np.mean(speeds), ep)
    writer.close()
    return agent


def evaluate_agent(agent: RLAgent, episodes: int = 1, seed: int = 0) -> float:
    """Return average final speed of the agent."""
    torch.manual_seed(seed)
    motorcycle = Motorcycle(device=agent.device, mass_std=1e-3)
    steps = int(SIMULATION_TIME / DT)
    final_speeds = []
    for _ in range(episodes):
        motorcycle.reset()
        state = motorcycle.speed.item()
        for _ in range(steps):
            throttle, brake, _ = agent.select_action(state)
            speed = motorcycle(throttle, brake, DT)
            state = speed.item()
        final_speeds.append(motorcycle.speed.item())
    return float(np.mean(final_speeds))


if __name__ == "__main__":
    train_rl_agent()

