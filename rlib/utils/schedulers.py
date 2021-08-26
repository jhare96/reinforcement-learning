import torch

def polynomial_sheduler(optimiser, lr_final, decay_steps, power=1):
    lr_init = optimiser.defaults["lr"]
    assert lr_init > lr_final, f"lr_final ({lr_final}) must be be smaller than initial lr ({lr_init})"

    def polylambda(current_step: int):
        if current_step > decay_steps:
            return lr_final / lr_init  # as LambdaLR multiplies by lr_init
        else:
            decay = (lr_init - lr_final) * (1 - current_step / decay_steps) ** power + lr_final
            return decay / lr_init  # as LambdaLR multiplies by lr_init
    
    return torch.optim.lr_scheduler.LambdaLR(optimiser, polylambda, last_epoch=-1)