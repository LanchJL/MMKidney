class EarlyStopper:
    def __init__(self, patience=15, mode="max"):
        self.patience = int(patience)
        self.mode = mode
        self.best = None
        self.num_bad = 0

    def step(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            return False
        better = value > self.best if self.mode == "max" else value < self.best
        if better:
            self.best = value
            self.num_bad = 0
        else:
            self.num_bad += 1
        return self.num_bad >= self.patience
