"""
Temporally-extended epsilon-greedy exploration (Dabney, Ostrovski and Barreto,
2020, arXiv 2006.01782).

An env that starts exploring draws one uniform random action and a duration n
from a zeta distribution P(n) ~ n^-mu on 1..max_hold, then repeats that action
for n steps. A hold runs on into the env's next episode, since cutting it there
would lower the share of exploratory steps below epsilon. Per-step epsilon-greedy cannot turn
a stopped drone through 180 degrees, which takes about 39 same-way turn steps.
"""
import torch


class HeldRandomExploration:
    def __init__(self, num_envs: int, num_actions: int, mu: float, max_hold: int, device):
        self.num_actions = num_actions
        self.device = device
        n = torch.arange(1, max_hold + 1, dtype=torch.float64)
        probs = n ** -mu
        probs /= probs.sum()
        self.mean_hold = float((n * probs).sum())
        self.hold_probs = probs.float().to(device)
        self.remaining = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.action = torch.zeros(num_envs, dtype=torch.long, device=device)

    def start_probability(self, epsilon: float) -> float:
        # A hold covers mean_hold steps on average, so starting one with plain epsilon would raise the
        # long-run share of exploratory steps to about epsilon * mean_hold. This keeps that share at epsilon.
        return epsilon / (self.mean_hold - epsilon * (self.mean_hold - 1))

    def select(self, greedy_action: torch.Tensor, epsilon: float) -> torch.Tensor:
        idle = self.remaining == 0
        start = idle & (torch.rand(idle.shape[0], device=self.device) < self.start_probability(epsilon))
        k = int(start.sum())
        if k:
            self.action[start] = torch.randint(0, self.num_actions, (k,), device=self.device)
            self.remaining[start] = torch.multinomial(self.hold_probs, k, replacement=True) + 1
        exploring = self.remaining > 0
        self.remaining[exploring] -= 1
        return torch.where(exploring, self.action, greedy_action)
