import torch


def build_scheduler(optimizer, epochs: int, warmup: int = 5):
    # warmup + cosine
    def lr_lambda(ep):
        if ep < warmup:
            return float(ep + 1) / float(max(1, warmup))
        t = (ep - warmup) / float(max(1, epochs - warmup))
        return 0.5 * (1.0 + torch.cos(torch.tensor(t * 3.1415926535))).item()

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
