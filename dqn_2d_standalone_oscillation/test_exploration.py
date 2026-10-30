"""
Runs as a plain script, no pytest:

    python test_exploration.py
"""
import torch

from exploration import HeldRandomExploration


def test_max_hold_one_is_plain_epsilon_greedy():
    ex = HeldRandomExploration(4, 12, 2.0, 1, "cpu")
    assert ex.mean_hold == 1.0
    assert abs(ex.start_probability(0.03) - 0.03) < 1e-12
    print("  max_hold 1: start probability equals epsilon")


def test_hold_repeats_one_action():
    torch.manual_seed(0)
    ex = HeldRandomExploration(1, 12, 0.0, 100, "cpu")  # mu 0, uniform 1..100, so the hold is long
    greedy = torch.tensor([-1])
    first = ex.select(greedy, 1.0)
    held = int(ex.remaining[0]) + 1
    seen = [int(first[0])] + [int(ex.select(greedy, 0.0)[0]) for _ in range(held + 2)]
    assert held > 1 and seen[:held] == [seen[0]] * held and seen[0] != -1
    assert seen[held:] == [-1] * 3  # epsilon 0 after the hold, so greedy again
    print(f"  one action held for its {held} drawn steps, then greedy")


def test_long_run_share_matches_epsilon():
    torch.manual_seed(2)
    n = 4096
    ex = HeldRandomExploration(n, 12, 2.0, 100, "cpu")
    greedy = torch.full((n,), -1)
    explored = 0
    steps = 2000
    for t in range(steps):
        a = ex.select(greedy, 0.03)
        if t >= 500:
            explored += int((a != -1).sum())
    share = explored / (n * (steps - 500))
    # A random draw equal to greedy would be miscounted, but greedy is -1 here, never a valid action.
    assert abs(share - 0.03) < 0.002, share
    print(f"  long-run exploratory share {share:.4f} at epsilon 0.03")


if __name__ == "__main__":
    for test in (test_max_hold_one_is_plain_epsilon_greedy, test_hold_repeats_one_action,
                 test_long_run_share_matches_epsilon):
        print(test.__name__)
        test()
    print("all tests passed")
