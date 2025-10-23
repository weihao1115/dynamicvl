from torch.optim.lr_scheduler import LRScheduler


class WarmUpPolyLR(LRScheduler):
    """
    Linear warmup + polynomial decay:
      - warmup: lr = base_lr * (step / warmup_steps)
      - decay:  lr = base_lr * (1 - progress) ** lr_power, progress in [0, 1]
    """
    def __init__(self, optimizer, lr_power: float, total_iters: int, warmup_steps: int = 0, last_epoch: int = -1):
        self.lr_power = float(lr_power)
        self.total_iters = int(total_iters)
        self.warmup_steps = int(max(0, warmup_steps))
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # warmup
        if self.warmup_steps > 0 and self.last_epoch <= self.warmup_steps:
            w = self.last_epoch / max(1, self.warmup_steps)  # in [0,1]
            return [base_lr * w for base_lr in self.base_lrs]

        # decay
        denom = max(1, self.total_iters - self.warmup_steps)
        progress = (self.last_epoch - self.warmup_steps) / denom
        progress = min(max(progress, 0.0), 1.0)
        factor = (1.0 - progress) ** self.lr_power
        return [base_lr * factor for base_lr in self.base_lrs]
