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


class ValueNetwork(nn.Module):
    """Simple value network used by the agent when actor-critic is enabled."""

    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(state))
        return self.fc2(x)


class RLAgent:
    def __init__(
        self,
        lr: float = 1e-3,
        gamma: float = 0.99,
        hidden_size: int = 32,
        use_value: bool = True,
        device: str = "cpu",
    ):
        self.policy = PolicyNetwork(hidden_size=hidden_size).to(device)
        self.use_value = use_value
        if use_value:
            self.value_network = ValueNetwork(hidden_size=hidden_size).to(device)
            params = list(self.policy.parameters()) + list(self.value_network.parameters())
        else:
            self.value_network = None
            params = self.policy.parameters()
        self.optimizer = optim.Adam(params, lr=lr)
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
        return throttle, brake, log_prob, action.item()

    def update(self, log_probs, rewards, states):
        returns = []
        R = 0.0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device).unsqueeze(-1)

        if self.use_value:
            values = self.value_network(states_t).squeeze(-1)
            advantages = returns - values.detach()
            policy_loss = -(torch.stack(log_probs) * advantages).sum()
            value_loss = F.mse_loss(values, returns)
            loss = policy_loss + value_loss
        else:
            if returns.numel() > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-5)
            loss = -(torch.stack(log_probs) * returns).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path: str):
        data = {
            "policy": self.policy.state_dict(),
            "use_value": self.use_value,
        }
        if self.use_value:
            data["value"] = self.value_network.state_dict()
        torch.save(data, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu"):
        checkpoint = torch.load(path, map_location=device)
        agent = cls(use_value=checkpoint.get("use_value", False), device=device)
        agent.policy.load_state_dict(checkpoint["policy"])
        if agent.use_value and "value" in checkpoint:
            agent.value_network.load_state_dict(checkpoint["value"])
        return agent



