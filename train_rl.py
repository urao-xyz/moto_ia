import argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from moto import device
from motorcycle_env import MotorcycleEnv
from rl_agent import RLAgent


def train_rl_agent(
    episodes: int = 200,
    seed: int = 0,
    hidden_size: int = 32,
    energy_weight: float = 0.0,
    save_path: str | None = None,
    load_path: str | None = None,
    use_value: bool = True,
):
    torch.manual_seed(seed)
    if load_path:
        agent = RLAgent.load(load_path, device=device)
    else:
        agent = RLAgent(hidden_size=hidden_size, use_value=use_value, device=device)
    env = MotorcycleEnv(energy_weight=energy_weight, device=device)
    writer = SummaryWriter()

    for ep in range(episodes):
        state, _ = env.reset(seed=seed + ep)
        done = False
        log_probs = []
        rewards = []
        states = []
        while not done:
            throttle, brake, log_prob, _ = agent.select_action(float(state[0]))
            next_state, reward, done, _ = env.step((throttle, brake))
            log_probs.append(log_prob)
            rewards.append(reward)
            states.append(float(state[0]))
            state = next_state
        agent.update(log_probs, rewards, states)
        writer.add_scalar("Episode Reward", sum(rewards), ep)
    writer.close()
    if save_path:
        agent.save(save_path)
    return agent


def evaluate_agent(
    agent: RLAgent,
    episodes: int = 1,
    seed: int = 0,
    energy_weight: float = 0.0,
) -> float:
    torch.manual_seed(seed)
    env = MotorcycleEnv(energy_weight=energy_weight, device=agent.device)
    final_rewards = []
    for ep in range(episodes):
        state, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            throttle, brake, _, _ = agent.select_action(float(state[0]))
            state, reward, done, _ = env.step((throttle, brake))
        final_rewards.append(reward)
    return float(np.mean(final_rewards))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--energy-weight", type=float, default=0.0)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--no-value", action="store_true")
    args = parser.parse_args()

    train_rl_agent(
        episodes=args.episodes,
        seed=args.seed,
        hidden_size=args.hidden_size,
        energy_weight=args.energy_weight,
        save_path=args.save,
        load_path=args.load,
        use_value=not args.no_value,
    )
