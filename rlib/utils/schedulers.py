import torch


def polynomial_sheduler(
    optimiser: torch.optim.Optimizer,
    lr_final: float,
    decay_steps: int,
    power: float = 1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Polynomial-decay LR scheduler from ``lr_init`` to ``lr_final``.

    ``lr_final == lr_init`` is allowed and yields a constant LR (effectively
    no scheduler) so callers that want to opt out of decay don't need a
    separate code path.
    """
    lr_init = optimiser.defaults["lr"]
    assert lr_init >= lr_final, f"lr_final ({lr_final}) must be <= initial lr ({lr_init})"

    def polylambda(current_step: int) -> float:
        if current_step > decay_steps:
            return lr_final / lr_init  # as LambdaLR multiplies by lr_init
        else:
            decay = (lr_init - lr_final) * (1 - current_step / decay_steps) ** power + lr_final
            return decay / lr_init  # as LambdaLR multiplies by lr_init

    return torch.optim.lr_scheduler.LambdaLR(optimiser, polylambda, last_epoch=-1)
