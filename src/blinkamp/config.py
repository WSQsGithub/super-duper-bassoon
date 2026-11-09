from dataclasses import asdict, dataclass


@dataclass
class TrainConfig:
    utd: float = 0.  # 0 preserves historical fixed collect_steps.
    critic_epochs: int = 3
    critic_minibatches: int = 1
    pi_opt: str = 'adam'
    wm_opt: str = 'adam'
    pi_clip_value: float = 0.
    pi_lr_mult: float = 1.
    lr_decay_iters: int = 0
    lr_decay_by: int = 0
    lr_decay_len: int = 300
    actor_hidden: str = '512,256,128'
    precision: str = 'none'
    replay_device: str = 'same'
    pi_microbatch: int = 0
    critic_norm_after_actor: bool = False
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
    command_profile: str = 'source'
    actor_ckpt_milestones: str = ''
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

    def vector_steps(self, iteration):
        if self.utd == 0:
            return self.collect_steps
        return int(iteration / self.utd) - int((iteration - 1) / self.utd)

    def learning_rates(self, iteration):
        import math
        base = self.lr
        if self.lr_decay_iters:
            phase = min((iteration - 1) / self.lr_decay_iters, 1.)
            base *= .03 + .97 * .5 * (1 + math.cos(math.pi * phase))
        multiplier = self.pi_lr_mult
        if self.lr_decay_by:
            progress = min(max((iteration - 1 - self.lr_decay_by) / self.lr_decay_len, 0.), 1.)
            multiplier += (1 - multiplier) * progress
        return base, (base * multiplier if self.pi_lr_mult != 1 or self.lr_decay_iters else self.pi_lr)

    def validate(self):
        if self.command_profile not in ('source','motion_1s'):
            raise ValueError('Unknown command profile')
        if self.command_profile == 'motion_1s' and self.task_profile != 'physhsi_dinosaur':
            raise ValueError('Motion command ablation requires the Dinosaur task')
        if self.actor_ckpt_milestones and any(int(x)<1 for x in self.actor_ckpt_milestones.split(',')):
            raise ValueError('Actor checkpoint milestones must be positive')
        if self.utd < 0 or self.utd > 1:
            raise ValueError('utd must be 0 (legacy) or in (0,1]')
        if self.critic_epochs < 1 or self.critic_minibatches < 1 or self.pi_microbatch < 0:
            raise ValueError('Invalid optimization batch counts')
        if self.horizon * self.pi_batch % self.critic_minibatches:
            raise ValueError('Critic minibatches must divide imagined state count')
        if self.pi_opt not in ('adam', 'muon') or self.wm_opt not in ('adam', 'muon'):
            raise ValueError('Unknown optimizer')
        if self.precision not in ('none', 'bf16') or self.replay_device not in ('same', 'cpu'):
            raise ValueError('Invalid precision or replay storage')
        if self.pi_microbatch and self.pi_batch % self.pi_microbatch:
            raise ValueError('Actor microbatch must divide full batch')
        if self.pi_microbatch and not self.critic_norm_after_actor:
            raise ValueError('Microbatch requires original critic normalization timing')
        if min(self.lr_decay_iters, self.lr_decay_by, self.pi_clip_value) < 0 or self.lr_decay_len < 1:
            raise ValueError('Invalid schedule or gradient clipping')

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
