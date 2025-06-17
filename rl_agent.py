import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    """Policy network returning action probabilities."""

    def __init__(self, hidden_size: int = 32, num_actions: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_actions)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(state))
        return torch.softmax(self.fc2(x), dim=-1)


class RLAgent:
    def __init__(self, lr: float = 1e-3, gamma: float = 0.99, device: str = "cpu"):
        self.policy = PolicyNetwork().to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.device = device

    def select_action(self, state: float):
        state_tensor = torch.tensor([[state]], dtype=torch.float32, device=self.device)
        probs = self.policy(state_tensor).squeeze(0)
        dist = distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        if action.item() == 0:  # accelerate
            throttle, brake = 1.0, 0.0
        elif action.item() == 1:  # brake
            throttle, brake = 0.0, 1.0
        else:  # idle
            throttle, brake = 0.0, 0.0
        return throttle, brake, log_prob

    def update(self, log_probs, rewards):
        returns = []
        R = 0.0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        if returns.numel() > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        loss = -(torch.stack(log_probs) * returns).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



