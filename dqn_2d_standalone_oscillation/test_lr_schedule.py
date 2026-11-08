import math

from lr_schedule import scheduled_lr


def test_holds_then_reaches_final():
    lrs = [scheduled_lr(1e-3, 0.1, 2 / 3, e, 3000) for e in range(3000)]
    assert all(lr == 1e-3 for lr in lrs[:2000])
    assert math.isclose(lrs[-1], 1e-4)
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))


def test_start_zero_decays_from_first_episode():
    assert scheduled_lr(1e-3, 0.1, 0.0, 0, 1500) == 1e-3
    assert scheduled_lr(1e-3, 0.1, 0.0, 1, 1500) < 1e-3
    assert math.isclose(scheduled_lr(1e-3, 0.1, 0.0, 1499, 1500), 1e-4)


if __name__ == "__main__":
    test_holds_then_reaches_final()
    test_start_zero_decays_from_first_episode()
    print("ok")
