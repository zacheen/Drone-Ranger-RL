import math


def scheduled_lr(base_lr, final_fraction, decay_start, episode, num_episodes):
    """base_lr until episode decay_start * (num_episodes - 1), then a half cosine down to
    base_lr * final_fraction on the run's last episode. Each curriculum stage is its own
    run, so the schedule restarts at base_lr in every stage.
    """
    start = decay_start * (num_episodes - 1)
    if episode <= start:
        return base_lr
    progress = (episode - start) / (num_episodes - 1 - start)
    final = base_lr * final_fraction
    return final + 0.5 * (base_lr - final) * (1 + math.cos(math.pi * progress))
