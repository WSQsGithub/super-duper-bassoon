from dataclasses import asdict, dataclass


@dataclass
class TrainConfig:
    style: str = "walk"
    seed: int = 0
    num_envs: int = 512
    iterations: int = 2000
    history: int = 6
    horizon: int = 16
    window: int = 10
    buffer_steps: int = 256
    collect_steps: int = 4
    warmup_steps: int = 64
    wm_warmup_iters: int = 100
    wm_hidden: str = "512,512,512"
    wm_batch: int = 256
    wm_window: int = 4
    wm_updates: int = 4
    pi_batch: int = 256
    disc_batch: int = 512
    disc_updates: int = 2
    lr: float = 4e-4
    pi_lr: float = 1e-3
    disc_lr: float = 1e-4
    amp_coef: float = .35
    grad_penalty: float = .5
    gamma: float = .99
    lam: float = .95
    exploration: float = .1
    ema: float = .995
    rsi: float = .8
    command_max: float = .9
    ckpt_every: int = 250
    log_every: int = 25
    compile: bool = True
    direction: str = "FW"

    def asdict(self):
        return asdict(self)

    def validate(self):
        if self.style not in ("walk", "highknees", "dinosaur"):
            raise ValueError("Unknown motion style")
        if min(self.num_envs, self.iterations, self.wm_batch, self.pi_batch, self.disc_batch,
               self.collect_steps, self.wm_updates, self.disc_updates, self.horizon,
               self.ckpt_every, self.log_every, self.history, self.wm_window) < 1:
            raise ValueError("Counts and batch sizes must be positive")
        if self.warmup_steps < max(self.wm_window, self.history, self.window):
            raise ValueError("Warmup must cover dynamics and AMP histories")
        if self.buffer_steps < self.warmup_steps or self.buffer_steps < self.wm_window:
            raise ValueError("Replay too short")
        if not 0 <= self.amp_coef < 1 or not 0 <= self.rsi <= 1:
            raise ValueError("Invalid AMP coefficient or RSI probability")
        if self.command_max < .3:
            raise ValueError("command_max must be >= .3 m/s")
