"""Per-env n-step return accumulation for vectorized envs that auto-reset inside step()."""
import torch


class NStepAccumulator:
    """Holds each env's last n raw transitions and turns them into n-step transitions.

    A window entry becomes (s_i, a_i, sum_j gamma^(j-i) r_j, s_next, terminal, gamma^m), with m the
    number of rewards summed, once the window is full or its episode ends. On an end, terminated or
    truncated, every entry is emitted and the window is cleared, so no return crosses a reset. A
    truncated window still bootstraps from s_next, the pre-reset observation.
    """

    def __init__(self, n, gamma, num_envs, state_dim, device):
        self.n = n
        self.gamma = gamma
        self.s = torch.zeros(n, num_envs, state_dim, device=device)
        self.a = torch.zeros(n, num_envs, dtype=torch.long, device=device)
        self.r = torch.zeros(n, num_envs, device=device)
        self.start_step = torch.zeros(n, num_envs, dtype=torch.long, device=device)
        self.length = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._envs = torch.arange(num_envs, device=device)

    def add(self, s, a, r, s_next, terminal, done, step_in_episode):
        """Append one vector step and return the emitted transitions.

        Returns (env, s, a, return, s_next, terminal, discount, start_step), one row per emitted
        transition, in env order and oldest first within an env.
        start_step is the step_in_episode of its first transition.
        """
        n = self.n
        self.s[self.length, self._envs] = s
        self.a[self.length, self._envs] = a
        self.r[self.length, self._envs] = r
        self.start_step[self.length, self._envs] = step_in_episode
        self.length += 1

        slot = torch.arange(n, device=s.device).unsqueeze(1)  # (n, 1)
        valid = slot < self.length  # (n, E)
        # Return from slot i to the latest entry, built backwards over the window.
        ret = torch.zeros_like(self.r)
        acc = torch.zeros_like(self.r[0])
        for i in reversed(range(n)):
            acc = torch.where(valid[i], self.r[i] + self.gamma * acc, acc)
            ret[i] = acc
        discount = self.gamma ** (self.length.unsqueeze(0) - slot).clamp(min=0).float()

        full = self.length == n
        emit = valid & (done.unsqueeze(0) | ((slot == 0) & full.unsqueeze(0)))
        env_idx, slot_idx = emit.t().nonzero(as_tuple=True)
        out = (env_idx, self.s[slot_idx, env_idx], self.a[slot_idx, env_idx], ret[slot_idx, env_idx],
               s_next[env_idx], terminal[env_idx], discount[slot_idx, env_idx],
               self.start_step[slot_idx, env_idx])

        shift = full & ~done
        if shift.any():
            for buf in (self.s, self.a, self.r, self.start_step):
                buf[:, shift] = torch.roll(buf[:, shift], -1, dims=0)
            self.length[shift] = n - 1
        self.length[done] = 0
        return out
