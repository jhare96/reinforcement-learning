import datetime
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from rlib.networks import Model, ModelConfig, PPOConfig
from rlib.networks.networks import NatureCNN
from rlib.PPO.PPO import PPOModel
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


class ValueModel(Model):
    def __init__(
        self,
        model,
        input_shape,
        action_size,
        config: ModelConfig,
        *,
        build_optimiser: bool = True,
        optim: type[torch.optim.Optimizer] = torch.optim.Adam,
        optim_args: dict | None = None,
        **model_args,
    ):
        super().__init__(config=config)
        self.action_size = action_size

        self.model = model(input_shape, **model_args).to(self.device)
        dense_size = self.model.dense_size
        self.V = torch.nn.Linear(dense_size, 1).to(self.device)

        if build_optimiser:
            self._build_optimiser(optim=optim, optim_args=optim_args)

    def forward(self, state):
        enc_state = self.model(state)
        value = self.V(enc_state).view(-1)
        return value

    def evaluate(self, state):
        with torch.no_grad():
            value = self.forward(totorch(state, self.device))
        return tonumpy(value)

    def loss(self, V, R):
        return self.value_loss(R, V)

    def backprop(self, state, R):
        state, R = totorch_many(state, R, device=self.device)
        value = self.forward(state)
        loss = self.loss(value, R)
        return self._train_step(loss)


class PolicyModel(PPOModel):
    """DAAC policy head: clipped PPO objective with a separate advantage prediction."""

    def __init__(
        self,
        model,
        input_shape,
        action_size,
        config: PPOConfig,
        *,
        adv_coeff: float = 0.25,
        build_optimiser: bool = True,
        optim: type[torch.optim.Optimizer] = torch.optim.Adam,
        optim_args: dict | None = None,
        **model_args,
    ):
        super().__init__(action_size=action_size, config=config)
        self.adv_coeff = adv_coeff

        self.model = model(input_shape, **model_args).to(self.device)
        dense_size = self.model.dense_size
        self.policy = torch.nn.Sequential(
            torch.nn.Linear(dense_size, action_size), torch.nn.Softmax(dim=-1)
        ).to(self.device)
        self.Adv = torch.nn.Linear(dense_size, 1).to(self.device)

        if build_optimiser:
            self._build_optimiser(optim=optim, optim_args=optim_args)

    def forward(self, state):
        enc_state = self.model(state)
        policy = self.policy(enc_state)
        Adv = self.Adv(enc_state).view(-1)
        return policy, Adv

    def evaluate(self, state):
        with torch.no_grad():
            policy, Adv = self.forward(totorch(state, self.device))
        return tonumpy(policy), tonumpy(Adv)

    def loss(self, policy, Adv_hat, Adv, action_onehot, old_policy):
        policy_loss, entropy = self.ppo_clipped_policy_loss(policy, old_policy, action_onehot, Adv)
        adv_loss = torch.square(Adv_hat - Adv).sum(dim=-1).mean()
        return policy_loss - self.entropy_coeff * entropy + self.adv_coeff * adv_loss

    def backprop(self, state, Adv, action, old_policy):
        state, action, Adv, old_policy = totorch_many(
            state, action, Adv, old_policy, device=self.device
        )
        policy, Adv_hat = self.forward(state)
        action_onehot = F.one_hot(action.long(), self.action_size)
        loss = self.loss(policy, Adv_hat, Adv, action_onehot, old_policy)
        return self._train_step(loss)


class DAAC(Model):
    # Decoupling Value and Policy for Generalization in Reinforcement Learning
    # https://arxiv.org/pdf/2102.10330.pdf
    def __init__(
        self,
        policy_model,
        value_model,
        input_shape,
        action_size,
        config: PPOConfig,
        *,
        adv_coeff: float = 0.25,
        policy_optim: type[torch.optim.Optimizer] = torch.optim.Adam,
        policy_optim_args: dict | None = None,
        policy_model_args: dict | None = None,
        value_optim: type[torch.optim.Optimizer] = torch.optim.Adam,
        value_optim_args: dict | None = None,
        value_model_args: dict | None = None,
    ):
        if value_model_args is None:
            value_model_args = {}
        if value_optim_args is None:
            value_optim_args = {}
        if policy_model_args is None:
            policy_model_args = {}
        if policy_optim_args is None:
            policy_optim_args = {}
        super().__init__(config=config)
        self.entropy_coeff = config.entropy_coeff
        self.adv_coeff = adv_coeff
        self.policy_clip = config.policy_clip

        self.value = ValueModel(
            value_model,
            input_shape,
            action_size,
            config=config,
            optim=value_optim,
            optim_args=value_optim_args,
            **value_model_args,
        )

        self.policy = PolicyModel(
            policy_model,
            input_shape,
            action_size,
            config=config,
            adv_coeff=adv_coeff,
            optim=policy_optim,
            optim_args=policy_optim_args,
            **policy_model_args,
        )

    def get_policy(self, state: np.ndarray):
        return self.policy.evaluate(state)

    def get_value(self, state: np.ndarray):
        return self.value.evaluate(state)

    def evaluate(self, state: np.ndarray):
        with torch.no_grad():
            policy, _ = self.policy.forward(totorch(state, self.policy.device))
            value = self.value.forward(totorch(state, self.value.device))
        return tonumpy(policy), tonumpy(value)

    def backprop(self, state, R, Adv, action, old_policy):
        # ``policy`` and ``value`` each own their own optimiser/scheduler,
        # so DAAC simply delegates and sums the two scalar losses.
        policy_loss = self.policy.backprop(state, Adv, action, old_policy)
        value_loss = self.value.backprop(state, R)
        return policy_loss + value_loss


class DAACTrainer(SyncMultiEnvTrainer):
    def __init__(
        self,
        envs,
        model,
        val_envs,
        config: TrainerConfig,
        *,
        policy_epochs: int = 1,
        value_epochs: int = 9,
        num_minibatches: int = 8,
    ):
        super().__init__(envs, model, val_envs, config=config)

        self.policy_epochs = policy_epochs
        self.value_epochs = value_epochs
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
            'advantage_coefficient': self.model.adv_coeff,
            'value_coefficient': 1.0,
            'policy_clip': self.model.policy_clip,
            'num_minibatches': self.num_minibatches,
            'policy_epochs': self.policy_epochs,
            'value_epochs': self.value_epochs,
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
            R = self.lambda_return(
                rewards, values, last_values, dones, gamma=self.gamma, lambda_=self.lambda_
            )
            loss_value = 0

            idxs = np.arange(len(states))
            value_loss = 0
            for _epoch in range(self.value_epochs):
                np.random.shuffle(idxs)
                for batch in range(0, len(states), mini_batch_size):
                    batch_idxs = idxs[batch : batch + mini_batch_size]
                    # stack all states, actions and Rs across all workers into a single batch
                    (
                        mb_states,
                        mb_Rs,
                    ) = fold_many(states[batch_idxs], R[batch_idxs])

                    value_loss += self.model.value.backprop(mb_states.copy(), mb_Rs.copy())

            value_loss /= self.value_epochs

            idxs = np.arange(len(states))
            policy_loss = 0
            for _epoch in range(self.policy_epochs):
                np.random.shuffle(idxs)
                for batch in range(0, len(states), mini_batch_size):
                    batch_idxs = idxs[batch : batch + mini_batch_size]
                    # stack all states, actions and Rs across all workers into a single batch
                    mb_states, mb_actions, mb_Adv, mb_old_policies = fold_many(
                        states[batch_idxs],
                        actions[batch_idxs],
                        Adv[batch_idxs],
                        old_policies[batch_idxs],
                    )

                    policy_loss += self.model.policy.backprop(
                        mb_states.copy(), mb_Adv.copy(), mb_actions.copy(), mb_old_policies.copy()
                    )

            policy_loss /= self.policy_epochs
            loss_value = policy_loss + value_loss

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

    model = DAAC(
        policy_model=NatureCNN,
        value_model=NatureCNN,
        input_shape=input_size,
        action_size=action_size,
        lr=5e-4,
        lr_final=1e-5,
        decay_steps=200e6 // (num_envs * nsteps),
        grad_clip=0.5,
        adv_coeff=0.25,
        entropy_coeff=0.01,
        policy_clip=0.1,
        device='cuda',
    )

    daac = DAACTrainer(
        envs=envs,
        model=model,
        model_dir=model_dir,
        log_dir=train_log_dir,
        val_envs=val_envs,
        train_mode='nstep',
        total_steps=200e6,
        nsteps=nsteps,
        policy_epochs=1,
        value_epochs=1,
        num_minibatches=8,
        validate_freq=1e5,
        save_freq=0,
        render_freq=0,
        num_val_episodes=32,
        log_scalars=False,
    )
    daac.train()


if __name__ == "__main__":
    # env_id_list = ['SpaceInvadersDeterministic-v4', 'FreewayDeterministic-v4']# 'SpaceInvadersDeterministic-v4',]# , ]
    # env_id_list = ['MountainCar-v0', 'Acrobot-v1', 'CartPole-v1', ]
    env_id_list = ['ApplePickerDeterministic-v0']
    for env_id in env_id_list:
        main(env_id)
