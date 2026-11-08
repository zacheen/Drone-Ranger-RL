import math
import random

import torch

from nstep import NStepAccumulator


def _reference(n, gamma, episodes_per_env):
    """Expected n-step rows per env from plain per-episode lists of (s, a, r, terminal_at_end)."""
    rows = {}
    for e, episodes in enumerate(episodes_per_env):
        out = []
        for ep, last_next, terminal in episodes:
            T = len(ep)
            for t in range(T):
                m = min(n, T - t)
                ret = sum(gamma ** k * ep[t + k][2] for k in range(m))
                end = t + m
                s_next = last_next if end == T else ep[end][0]
                out.append((ep[t][0], ep[t][1], ret, s_next, terminal and end == T, gamma ** m, t + 1))
        rows[e] = out
    return rows


def _run(n, gamma, num_envs, steps, seed):
    rng = random.Random(seed)
    acc = NStepAccumulator(n, gamma, num_envs, 1, "cpu")
    step_count = [0] * num_envs
    episodes = [[] for _ in range(num_envs)]
    current = [[] for _ in range(num_envs)]
    got = {e: [] for e in range(num_envs)}
    counter = 0
    s = torch.tensor([[float(e)] for e in range(num_envs)])
    for _ in range(steps):
        a = torch.tensor([rng.randrange(12) for _ in range(num_envs)])
        r = torch.tensor([rng.uniform(-1, 1) for _ in range(num_envs)])
        counter += 100
        s_next = torch.tensor([[float(counter + e)] for e in range(num_envs)])
        terminal = torch.tensor([rng.random() < 0.08 for _ in range(num_envs)])
        truncated = torch.tensor([rng.random() < 0.05 for _ in range(num_envs)]) & ~terminal
        done = terminal | truncated
        step_in_episode = torch.tensor([c + 1 for c in step_count])
        for e in range(num_envs):
            current[e].append((s[e, 0].item(), a[e].item(), r[e].item()))
        env, ss, aa, rr, sn, term, disc, start = acc.add(s, a, r, s_next, terminal, done, step_in_episode)
        for k in range(env.numel()):
            got[env[k].item()].append((ss[k, 0].item(), aa[k].item(), rr[k].item(), sn[k, 0].item(),
                                       bool(term[k]), disc[k].item(), start[k].item()))
        for e in range(num_envs):
            step_count[e] += 1
            if done[e]:
                episodes[e].append((current[e], s_next[e, 0].item(), bool(terminal[e])))
                current[e] = []
                step_count[e] = 0
        # A continuing env acts next from s_next, as the env's reset_obs equals obs_dict there.
        s = torch.where(done.unsqueeze(1), s_next + 0.25, s_next)
    # Episodes still open at the end only emitted their full windows, which a reference over closed
    # episodes cannot predict, so compare only rows from closed episodes.
    return got, _reference(n, gamma, episodes)


def _check(n):
    got, want = _run(n, 0.99, 5, 400, seed=n)
    for e in want:
        closed = got[e][:len(want[e])]
        assert len(closed) == len(want[e])
        # Within an episode a full window emits its oldest entry each step and an end flushes the rest
        # oldest first, so rows come out in time order.
        for g, w in zip(closed, want[e]):
            assert g[0] == w[0] and g[1] == w[1] and g[3] == w[3] and g[4] == w[4] and g[6] == w[6], (g, w)
            assert math.isclose(g[2], w[2], rel_tol=1e-5, abs_tol=1e-6)
            assert math.isclose(g[5], w[5], rel_tol=1e-6)


def test_matches_reference():
    for n in (1, 2, 3, 5):
        _check(n)


def test_n1_is_one_step():
    acc = NStepAccumulator(1, 0.99, 3, 2, "cpu")
    s = torch.randn(3, 2)
    out = acc.add(s, torch.tensor([1, 2, 3]), torch.tensor([0.1, 0.2, 0.3]), s + 1,
                  torch.tensor([False, True, False]), torch.tensor([False, True, True]), torch.tensor([4, 5, 6]))
    env, ss, aa, rr, sn, term, disc, start = out
    assert env.tolist() == [0, 1, 2]
    assert torch.equal(ss, s) and torch.equal(sn, s + 1) and aa.tolist() == [1, 2, 3]
    assert torch.equal(rr, torch.tensor([0.1, 0.2, 0.3])) and term.tolist() == [False, True, False]
    assert torch.allclose(disc, torch.full((3,), 0.99)) and start.tolist() == [4, 5, 6]


if __name__ == "__main__":
    test_matches_reference()
    test_n1_is_one_step()
    print("ok")
