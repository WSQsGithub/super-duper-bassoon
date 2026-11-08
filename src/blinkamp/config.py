from dataclasses import asdict, dataclass


@dataclass
class TrainConfig:
    task_profile: str = "legacy"
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
    amp_reward: str = "quadratic"
    amp_when_standing: bool = True
    joint_limit_weight: float = 2.
    stand_fraction: float = .1
    command_encoding: str = "direct"
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
    reward_profile: str = "legacy"  # Old checkpoints retain their historical reward meaning.
    disc_normalize: bool = True
    disc_activation: str = "elu"
    torque_warmup_iters: int = 500

    def asdict(self):
        return asdict(self)

    def validate(self):
        if self.task_profile not in ('legacy', 'physhsi_dinosaur'):
            raise ValueError('Unknown task profile')
        if self.task_profile == 'physhsi_dinosaur' and (self.style != 'dinosaur' or self.direction != 'all'
                or self.window != 10 or self.rsi != .5 or self.amp_coef != .3 or self.reward_profile != 'physhsi'):
            raise ValueError('Dinosaur source profile requires all three clips, window10, RSI .5, AMP .3 and source rewards')
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
        if self.amp_reward not in ('quadratic', 'softplus') or self.command_encoding not in ('direct', 'mlp'):
            raise ValueError("Invalid reward or command encoding")
        if not 0 <= self.stand_fraction <= 1 or self.joint_limit_weight < 0:
            raise ValueError("Invalid standing fraction or limit penalty")
        if self.reward_profile not in ('legacy', 'physhsi') or self.disc_activation not in ('elu', 'relu'):
            raise ValueError('Invalid reward profile or discriminator activation')
        if self.torque_warmup_iters < 0:
            raise ValueError('Torque warmup must be nonnegative')
        if self.reward_profile == 'physhsi':
            if (self.amp_reward != 'quadratic' or not self.amp_when_standing or self.joint_limit_weight != 5.
                    or self.disc_normalize or self.disc_activation != 'relu'):
                raise ValueError('PhysHSI profile requires quadratic AMP, stop AMP enabled, limit weight 5, raw inputs and ReLU')
