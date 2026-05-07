import datetime
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from rlib.networks import Model, PPOConfig
from rlib.networks.networks import UniverseCNN
from rlib.utils import TrainerConfig
from rlib.utils.SyncMultiEnvTrainer import SyncMultiEnvTrainer
from rlib.utils.utils import (
    fastsample,
    fold_many,
    stack_many,
    tonumpy,
    totorch,
    totorch_many,
)
from rlib.utils.VecEnv import BatchEnv, DummyBatchEnv
from rlib.utils.wrappers import AtariEnv, DummyEnv, apple_pickgame


class PPOModel(Model):
    """PPO-family base class: defines the clipped-objective policy loss.

    Concrete subclasses (single-critic PPO, twin-critic PPOIntrinsic,
    DAAC's policy head, ...) only need to implement ``forward``,
    ``evaluate`` and ``backprop``; they all share the same clipped
    policy loss + entropy bonus via :meth:`ppo_clipped_policy_loss`.
    """

    config: PPOConfig

    def __init__(self, action_size: int, config: PPOConfig) -> None:
        super().__init__(config=config)
        self.action_size = action_size
        self.entropy_coeff = config.entropy_coeff
        self.policy_clip = config.policy_clip

    def ppo_clipped_policy_loss(self, policy, old_policy, action_onehot, advantage):
        """PPO clipped-objective policy loss + entropy bonus.

        Returns the ``(policy_loss, entropy)`` pair so subclasses can
        combine them with whatever value-function loss(es) and
        coefficients their agent uses.
        """
        policy_actions = torch.sum(policy * action_onehot, dim=1)
        old_policy_actions = torch.sum(old_policy * action_onehot, dim=1)
        ratio = policy_actions / old_policy_actions
        unclipped = ratio * -advantage
        clipped = torch.clip(ratio, 1 - self.policy_clip, 1 + self.policy_clip) * -advantage
        policy_loss = torch.mean(torch.maximum(unclipped, clipped))
        entropy = torch.mean(torch.sum(policy * -torch.log(policy), dim=1))
        return policy_loss, entropy


class PPO(PPOModel):
    """Single-critic PPO actor-critic."""

    def __init__(
        self,
        model,
        input_shape,
        action_size,
        config: PPOConfig,
        *,
        value_coeff: float = 1.0,
        build_optimiser: bool = True,
        optim: type[torch.optim.Optimizer] = torch.optim.Adam,
        optim_args: dict | None = None,
        **model_args,
    ):
        super().__init__(action_size=action_size, config=config)
        self.value_coeff = value_coeff

        self.model = model(input_shape, **model_args).to(self.device)
        dense_size = self.model.dense_size
        self.policy = torch.nn.Sequential(
            torch.nn.Linear(dense_size, action_size), torch.nn.Softmax(dim=-1)
        ).to(self.device)
        self.V = torch.nn.Linear(dense_size, 1).to(self.device)

        if build_optimiser:
            self._build_optimiser(optim=optim, optim_args=optim_args)

    def forward(self, state):
        state_enc = self.model(state)
        policy = self.policy(state_enc)
        value = self.V(state_enc).view(-1)
        return policy, value

    def evaluate(self, state):
        with torch.no_grad():
            policy, value = self.forward(totorch(state, self.device))
        return tonumpy(policy), tonumpy(value)

    def loss(self, policy, R, V, Adv, action_onehot, old_policy):
        policy_loss, entropy = self.ppo_clipped_policy_loss(policy, old_policy, action_onehot, Adv)
        value_loss = self.value_loss(R, V)
        return policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy

    def backprop(self, state, R, Adv, action, old_policy):
        state, action, R, Adv, old_policy = totorch_many(
            state, action, R, Adv, old_policy, device=self.device
        )
        action_onehot = F.one_hot(action.long(), self.action_size)
        policy, value = self.forward(state)
        loss = self.loss(policy, R, value, Adv, action_onehot, old_policy)
        return self._train_step(loss)


class PPOTrainer(SyncMultiEnvTrainer):
    def __init__(
        self,
        envs,
        model,
        val_envs,
        config: TrainerConfig,
        *,
        num_epochs: int = 4,
        num_minibatches: int = 4,
    ):
        super().__init__(envs, model, val_envs, config=config)

        self.num_epochs = num_epochs
        self.num_minibatches = num_minibatches

        hyper_paras = {
            'learning_rate': model.lr,
            'learning_rate_final': model.lr_final,
            'lr_decay_steps': model.decay_steps,
            'grad_clip': model.grad_clip,
            'nsteps': self.nsteps,
            'num_workers': self.num_envs,
            'total_steps': self.total_steps,
            'entropy_coefficient': self.model.entropy_coeff,
            'value_coefficient': self.model.value_coeff,
            'gamma': self.gamma,
            'lambda': self.lambda_,
        }

        if config.log_scalars:
            filename = config.log_dir + '/hyperparameters.txt'
            self.save_hyperparameters(filename, **hyper_paras)

    def _train_nstep(self):
        batch_size = self.num_envs * self.nsteps
        num_updates = self.total_steps // batch_size
        s = 0
        mini_batch_size = self.nsteps // self.num_minibatches
        start = time.time()
        # main loop
        for t in range(1, num_updates + 1):
            # rollout_start = time.time()
            states, actions, rewards, values, last_values, old_policies, dones = self.rollout()
            # print('rollout time', time.time()-rollout_start)
            Adv = self.GAE(
                rewards, values, last_values, dones, gamma=self.gamma, lambda_=self.lambda_
            )
            R = Adv + values
            loss_value = 0

            # backprop_time = time.time()
            idxs = np.arange(len(states))
            for _epoch in range(self.num_epochs):
                np.random.shuffle(idxs)
                for batch in range(0, len(states), mini_batch_size):
                    batch_idxs = idxs[batch : batch + mini_batch_size]
                    # stack all states, actions and Rs across all workers into a single batch
                    mb_states, mb_actions, mb_R, mb_Adv, mb_old_policies = fold_many(
                        states[batch_idxs],
                        actions[batch_idxs],
                        R[batch_idxs],
                        Adv[batch_idxs],
                        old_policies[batch_idxs],
                    )

                    loss_value += self.model.backprop(
                        mb_states.copy(),
                        mb_R.copy(),
                        mb_Adv.copy(),
                        mb_actions.copy(),
                        mb_old_policies.copy(),
                    )

            # print('backprop time', time.time()-backprop_time)
            loss_value /= self.num_epochs

            if (
                self.render_freq > 0
                and t % ((self.validate_freq // batch_size) * self.render_freq) == 0
            ):
                render = True
            else:
                render = False

            if self.validate_freq > 0 and t % (self.validate_freq // batch_size) == 0:
                # val_time = time.time()
                self.validation_summary(t, loss_value, start, render)
                # print('validation time', time.time()-val_time)
                start = time.time()

            if self.save_freq > 0 and t % (self.save_freq // batch_size) == 0:
                s += 1
                self.save(s)
                print('saved model')

    def get_action(self, states):
        policies, values = self.model.evaluate(states)
        actions = fastsample(policies)
        return actions

    def rollout(self):
        rollout = []
        for _t in range(self.nsteps):
            policies, values = self.model.evaluate(self.states)
            actions = fastsample(policies)
            next_states, rewards, dones, infos = self.env.step(actions)
            rollout.append((self.states, actions, rewards, values, policies, dones))
            self.states = next_states

        states, actions, rewards, values, policies, dones = stack_many(*zip(*rollout))
        (
            policy,
            last_values,
        ) = self.model.evaluate(next_states)
        return states, actions, rewards, values, last_values, policies, dones


def main(env_id):
    num_envs = 32
    nsteps = 128

    classic_list = ['MountainCar-v0', 'Acrobot-v1', 'LunarLander-v2', 'CartPole-v0', 'CartPole-v1']
    if any(env_id in s for s in classic_list):
        print('Classic Control')
        val_envs = [gym.make(env_id) for i in range(10)]
        envs = BatchEnv(DummyEnv, env_id, num_envs, blocking=False)

    elif 'ApplePicker' in env_id:
        print('ApplePicker')
        make_args = {'num_objects': 300, 'default_reward': 0}
        if 'Deterministic' in env_id:
            envs = DummyBatchEnv(
                apple_pickgame,
                env_id,
                num_envs,
                max_steps=5000,
                auto_reset=True,
                k=4,
                grey_scale=True,
                make_args=make_args,
            )
            val_envs = DummyBatchEnv(
                apple_pickgame,
                env_id,
                num_envs,
                max_steps=5000,
                auto_reset=False,
                k=4,
                grey_scale=True,
                make_args=make_args,
            )
            for i in range(len(envs)):
                val_envs.envs[i].set_locs(envs.envs[i].item_locs_master, envs.envs[i].start_loc)
            val_envs.reset()
        else:
            # val_envs = [apple_pickgame(gym.make(env_id), max_steps=5000, auto_reset=False, k=1) for i in range(16)]
            val_envs = DummyBatchEnv(
                apple_pickgame,
                env_id,
                num_envs,
                max_steps=5000,
                auto_reset=False,
                k=4,
                grey_scale=True,
            )
            envs = DummyBatchEnv(
                apple_pickgame,
                env_id,
                num_envs,
                max_steps=5000,
                auto_reset=True,
                k=4,
                grey_scale=True,
            )
        print(val_envs.envs[0])
        print(envs.envs[0])

    else:
        print('Atari')
        env = gym.make(env_id)
        if env.unwrapped.get_action_meanings()[1] == 'FIRE':
            reset = True
            print('fire on reset')
        else:
            reset = False
            print('only stack frames')
        env.close()
        val_envs = [
            AtariEnv(gym.make(env_id), k=4, episodic=False, reset=reset, clip_reward=False)
            for i in range(16)
        ]
        envs = BatchEnv(
            AtariEnv,
            env_id,
            num_envs,
            blocking=False,
            k=4,
            reset=reset,
            episodic=False,
            clip_reward=True,
        )

    action_size = val_envs.envs[0].action_space.n
    input_size = val_envs.envs[0].reset().shape

    current_time = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    train_log_dir = 'logs/PPO/' + env_id + '/Adam/' + current_time
    model_dir = "models/PPO/" + env_id + '/' + current_time

    model = PPO(
        UniverseCNN,
        input_shape=input_size,
        action_size=action_size,
        lr=1e-4,
        lr_final=1e-5,
        decay_steps=200e6 // (num_envs * nsteps),
        grad_clip=0.5,
        value_coeff=1.0,
        entropy_coeff=0.01,
        device='cuda',
    ).cuda()

    ppo = PPOTrainer(
        envs=envs,
        model=model,
        model_dir=model_dir,
        log_dir=train_log_dir,
        val_envs=val_envs,
        train_mode='nstep',
        total_steps=200e6,
        nsteps=nsteps,
        num_epochs=2,
        num_minibatches=4,
        validate_freq=1e5,
        save_freq=0,
        render_freq=0,
        num_val_episodes=32,
        log_scalars=False,
    )
    ppo.train()


if __name__ == "__main__":
    # env_id_list = ['SpaceInvadersDeterministic-v4', 'FreewayDeterministic-v4']# 'SpaceInvadersDeterministic-v4',]# , ]
    # env_id_list = ['MountainCar-v0', 'Acrobot-v1', 'CartPole-v1', ]
    env_id_list = ['ApplePickerDeterministic-v0']
    for env_id in env_id_list:
        main(env_id)
