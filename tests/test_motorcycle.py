import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from moto import Motorcycle
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


def test_rl_agent_converges():
    untrained = RLAgent(device='cpu')
    initial_speed = evaluate_agent(untrained, seed=0)

    agent = train_rl_agent(episodes=20, seed=0)
    trained_speed = evaluate_agent(agent, seed=0)

    assert trained_speed > initial_speed
