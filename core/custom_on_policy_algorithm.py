'''
Author: Jikun Kang
Date: 2021-11-24 09:24:48
LastEditTime: 2022-10-13 17:17:50
LastEditors: Jikun Kang
FilePath: /Learning-Multi-Objective-Curricula-for-Robotic-Policy-Learning/core/custom_on_policy_algorithm.py
'''
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from causal_world.intervention_actors import goal_actor
from stable_baselines3.common import logger
# from core.custom_base_class import BaseAlgorithm
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.vec_env import VecEnv

# from core.custom_buffers import RolloutBuffer
from core.custom_buffers import RolloutBuffer
from core.custom_policies import ActorCriticPolicy

# from src.core.custom_intervention import EXT_MEM


class OnPolicyAlgorithm(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
            self,
            policy: Union[str, Type[ActorCriticPolicy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Callable],
            n_steps: int,
            gamma: float,
            gae_lambda: float,
            ent_coef: float,
            vf_coef: float,
            max_grad_norm: float,
            use_sde: bool,
            sde_sample_freq: int,
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            monitor_wrapper: bool = True,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
            h_cell=None,
            initial_cell=None,
            reward_cell=None,
            memory_cell=None,
            shared_hypernet=None,
            meta=True,
            reward_shaping=False,
            initial_curriculum=False,
            goal_curriculum=False,
            all_curricula=True,
            memory_only=False,
            num_timesteps=0
    ):

        super(OnPolicyAlgorithm, self).__init__(
            policy=policy,
            env=env,
            policy_base=ActorCriticPolicy,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            create_eval_env=create_eval_env,
            support_multi_env=True,
            seed=seed,
            tensorboard_log=tensorboard_log,
        )
        self.num_timesteps = num_timesteps
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer = None
        self.state, self.hyper_state = None, None
        self.h_cell = h_cell
        self.initial_cell = initial_cell
        self.reward_cell = reward_cell
        self.memory_cell = memory_cell
        self.shared_hypernet = shared_hypernet
        self.meta = meta
        self.reward_shaping = reward_shaping
        self.initial_curriculum = initial_curriculum
        self.goal_curriculum = goal_curriculum
        self.all_curricula = all_curricula
        self.memory_only = memory_only

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.rollout_buffer = RolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            goal_curriculum=self.goal_curriculum,
            memory_only=self.memory_only,
            ** self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)
        self.h_cell = self.h_cell.to(self.device)
        self.initial_cell = self.initial_cell.to(self.device)

    def collect_rollouts(
            self, env: VecEnv, callback: BaseCallback, rollout_buffer: RolloutBuffer, n_rollout_steps: int
    ) -> bool:
        """
        Collect rollouts using the current policy and fill a `RolloutBuffer`.
        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor
                obs_tensor = th.as_tensor(self._last_obs).to(self.device)
                rule_subgoal = th.tensor(
                    [[1, 0, 0]*env.num_envs]).view(env.num_envs, 3)
                rule_init = th.tensor(
                    [[0, 1, 0]*env.num_envs]).view(env.num_envs, 3)
                rule_reward = th.tensor(
                    [[0, 0, 1]*env.num_envs]).view(env.num_envs, 3)
                obs_tensor_subgoal = th.cat(
                    (obs_tensor.float(), rule_subgoal.to(self.device)),
                    dim=1)
                obs_tensor_init = th.cat(
                    (obs_tensor.float(), rule_init.to(self.device)),
                    dim=1)
                obs_tensor_reward = th.cat(
                    (obs_tensor.float(), rule_reward.to(self.device)),
                    dim=1)
                if self.meta:
                    if self.all_curricula:
                        subgoal, state, hyper_state = self.shared_hypernet(x=obs_tensor_subgoal.float(),
                                                                           state=self.state,
                                                                           hyper_state=self.hyper_state,
                                                                           lstm_cell=self.h_cell,
                                                                           emit_mem=False)
                        shape_reward, state, hyper_state = self.shared_hypernet(x=obs_tensor_init.float(),
                                                                          state=self.state,
                                                                          hyper_state=self.hyper_state,
                                                                          lstm_cell=self.reward_cell,
                                                                          emit_mem=False)
                        init_states, state, hyper_state = self.shared_hypernet(x=obs_tensor_reward.float(),
                                                                               state=self.state,
                                                                               hyper_state=self.hyper_state,
                                                                               lstm_cell=self.initial_cell,
                                                                               emit_mem=False)
                        _, state, hyper_state, memory = self.shared_hypernet(x=obs_tensor_subgoal.float(),
                                                                             state=self.state,
                                                                             hyper_state=self.hyper_state,
                                                                             lstm_cell=self.memory_cell,
                                                                             emit_mem=True)

                        self.state = state
                        self.hyper_state = hyper_state
                else:
                    subgoal = self.h_cell(obs_tensor_.float())
                    subgoal = subgoal.detach().squeeze(0)

                if self.all_curricula:
                    actions, values, log_probs = self.policy.forward_all(
                        obs_tensor, subgoal, shape_reward, memory, init_states)
                elif self.goal_curriculum:
                    actions, values, log_probs = self.policy.forward_subgoal(
                        obs_tensor, subgoal)
                elif self.memory_only:
                    actions, values, log_probs = self.policy.forward_subgoal(
                        obs_tensor, memory)
                else:
                    actions, values, log_probs = self.policy.forward(
                        obs_tensor)
            if self.meta:
                memory = memory.cpu().numpy()
            actions = actions.cpu().numpy()
            subgoals = subgoal.cpu().numpy()
            shape_rewards = shape_reward.cpu().numpy()
            init_state = init_states.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(
                    actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # add sudo-reward
            if self.reward_shaping:
                rule = th.tensor([[0, 0, 1]*env.num_envs]
                                 ).view(env.num_envs, 3)
                obs_tensor_ = th.cat(
                    (obs_tensor.float(), rule.to(self.device)),
                    dim=1)
                shape_reward, _, _ = self.shared_hypernet(x=obs_tensor_.float(),
                                                          state=self.state,
                                                          hyper_state=self.hyper_state,
                                                          lstm_cell=self.reward_cell)
                new_obs_ = th.cat((th.from_numpy(new_obs).to(
                    self.device).float(), rule.to(self.device)), dim=1)
                shape_reward_next, _, _ = self.shared_hypernet(x=new_obs_.float(),
                                                               state=self.state,
                                                               hyper_state=self.hyper_state,
                                                               lstm_cell=self.reward_cell)
                shaped_rew = 0.9 * \
                    shape_reward_next[0].detach().numpy(
                    ) - shape_reward[0].detach().numpy()
                rewards += shaped_rew[0]
            if self.meta:
                if self.all_curricula:
                    rollout_buffer.add(self._last_obs, actions, subgoals, shape_rewards, init_state,
                                   rewards, self._last_dones, values, log_probs, memory)
                elif self.goal_curriculum:
                    rollout_buffer.add(self._last_obs, actions, subgoals,
                                    rewards, self._last_dones, values, log_probs, memory)
            else:
                rollout_buffer.add(self._last_obs, actions, subgoals,
                                   rewards, self._last_dones, values, log_probs, 0)
            self._last_obs = new_obs
            self._last_dones = dones

        with th.no_grad():
            # Compute value for the last timestep

            obs_tensor = th.as_tensor(new_obs).to(self.device)
            rule_subgoal = th.tensor(
                [[1, 0, 0]*env.num_envs]).view(env.num_envs, 3)
            rule_init = th.tensor(
                [[0, 1, 0]*env.num_envs]).view(env.num_envs, 3)
            rule_reward = th.tensor(
                [[0, 0, 1]*env.num_envs]).view(env.num_envs, 3)
            obs_tensor_subgoal = th.cat(
                (obs_tensor.float(), rule_subgoal.to(self.device)),
                dim=1)
            obs_tensor_init = th.cat(
                (obs_tensor.float(), rule_init.to(self.device)),
                dim=1)
            obs_tensor_reward = th.cat(
                (obs_tensor.float(), rule_reward.to(self.device)),
                dim=1)
            if self.meta:
                if self.all_curricula:
                    subgoal, state, hyper_state = self.shared_hypernet(x=obs_tensor_subgoal.float(),
                                                                       state=self.state,
                                                                       hyper_state=self.hyper_state,
                                                                       lstm_cell=self.h_cell,
                                                                       emit_mem=False)
                    shape_reward, state, hyper_state = self.shared_hypernet(x=obs_tensor_init.float(),
                                                                      state=self.state,
                                                                      hyper_state=self.hyper_state,
                                                                      lstm_cell=self.reward_cell,
                                                                      emit_mem=False)
                    init_states, state, hyper_state = self.shared_hypernet(x=obs_tensor_reward.float(),
                                                                           state=self.state,
                                                                           hyper_state=self.hyper_state,
                                                                           lstm_cell=self.initial_cell,
                                                                           emit_mem=False)
                    _, state, hyper_state, memory = self.shared_hypernet(x=obs_tensor_subgoal.float(),
                                                                         state=self.state,
                                                                         hyper_state=self.hyper_state,
                                                                         lstm_cell=self.memory_cell,
                                                                         emit_mem=True)

                    self.state = state
                    self.hyper_state = hyper_state
            else:
                subgoal = self.h_cell(obs_tensor_.float())
                subgoal = subgoal.detach().squeeze(0)

            if self.all_curricula:
                _, values, _ = self.policy.forward_all(
                    obs_tensor, subgoal, shape_reward, init_states, memory)
            elif self.goal_curriculum:
                _, values, _ = self.policy.forward_subgoal(obs_tensor, subgoal)
            elif self.memory_only:
                _, values, _ = self.policy.forward_subgoal(obs_tensor, memory)
            else:
                _, values, _ = self.policy.forward(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(
            last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "OnPolicyAlgorithm",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
    ) -> "OnPolicyAlgorithm":
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps,
            tb_log_name
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer,
                                                      n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(
                self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int(self.num_timesteps / (time.time() - self.start_time))
                logger.record("time/iterations", iteration,
                              exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    logger.record(
                        "rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    logger.record(
                        "rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                logger.record("time/fps", fps)
                logger.record("time/time_elapsed", int(time.time() -
                              self.start_time), exclude="tensorboard")
                logger.record("time/total_timesteps",
                              self.num_timesteps, exclude="tensorboard")
                logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []
