import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from moto import Motorcycle
from motorcycle_env import MotorcycleEnv
from train_rl import train_rl_agent, evaluate_agent
from rl_agent import RLAgent

def test_speed_increases_with_throttle():
    m = Motorcycle(mass_std=1e-3, device='cpu')
    m.reset()
    dt = 0.1
    initial_speed = m.speed.item()
    for _ in range(5):
        m.forward(throttle=1.0, brake=0.0, dt=dt)
    assert m.speed.item() > initial_speed


def test_speed_decreases_with_brake():
    m = Motorcycle(mass_std=1e-3, device='cpu')
    m.reset()
    dt = 0.1
    # accelerate first
    for _ in range(5):
        m.forward(throttle=1.0, brake=0.0, dt=dt)
    speed_before_brake = m.speed.item()
    for _ in range(5):
        m.forward(throttle=0.0, brake=1.0, dt=dt)
    assert m.speed.item() < speed_before_brake


def test_env_deterministic():
    env1 = MotorcycleEnv()
    env2 = MotorcycleEnv()
    s1, _ = env1.reset(seed=42)
    s2, _ = env2.reset(seed=42)
    speeds1 = []
    speeds2 = []
    for _ in range(5):
        s1, _, _, _ = env1.step((0.5, 0.0))
        s2, _, _, _ = env2.step((0.5, 0.0))
        speeds1.append(s1[0])
        speeds2.append(s2[0])
    assert speeds1 == speeds2


def test_agent_save_load(tmp_path):
    agent = RLAgent(device="cpu")
    path = tmp_path / "agent.pth"
    agent.save(path)
    loaded = RLAgent.load(path, device="cpu")
    torch.manual_seed(0)
    state = 0.0
    act1 = agent.select_action(state)[3]
    torch.manual_seed(0)
    act2 = loaded.select_action(state)[3]
    assert act1 == act2


def test_rl_agent_converges():
    untrained = RLAgent(device="cpu")
    initial_reward = evaluate_agent(untrained, seed=0)

    agent = train_rl_agent(episodes=20, seed=0)
    trained_reward = evaluate_agent(agent, seed=0)

    assert trained_reward != initial_reward
