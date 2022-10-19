'''
Author: Jikun Kang
Date: 2021-11-24 09:24:48
LastEditTime: 2022-10-13 17:19:19
LastEditors: Jikun Kang
FilePath: /Learning-Multi-Objective-Curricula-for-Robotic-Policy-Learning/train_causal_world.py
'''
import argparse
import json
import os
import warnings
from pathlib import Path

import gym
import torch
from causal_world.envs import CausalWorld
from causal_world.intervention_actors import GoalInterventionActorPolicy
from causal_world.task_generators import generate_task
from causal_world.wrappers.curriculum_wrappers import CurriculumWrapper
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv

from core.custom_intervention import (EXT_MEM, CustomGoalIntervention,
                                      CustomInitialIntervention)
from core.custom_monitor import Monitor
from core.custom_policies import ActorCriticPolicy
from core.custom_ppo import PPO
from curriculum_module.models import HyperLSTMCell, LSTMCell, LSTMGenerator

warnings.filterwarnings('error')


def _make_env(rank, shared_net, initial_h_cell, goal_cell, task_name, random_goal=False):
    def _init():
        task = generate_task(task_generator_id=task_name)
        env = CausalWorld(task, skip_frame=skip_frame, enable_visualization=False,
                          max_episode_length=maximum_episode_length)
        if random_goal:
            print('======Random Goal State')
            env = CurriculumWrapper(env,
                                    intervention_actors=[
                                        GoalInterventionActorPolicy(),
                                        CustomInitialIntervention(
                                            shared_hypernet=shared_net,
                                            h_cell=initial_h_cell,
                                            task_name=task_name,
                                            device=device,
                                            no_hyper_net=meta),
                                    ],
                                    actives=[(0, total_episodes, 1, 0),
                                             ])
        else:
            env = CurriculumWrapper(env,
                                    intervention_actors=[
                                        CustomGoalIntervention(
                                            shared_hypernet=shared_net, h_cell=goal_cell, task_name=task_name,
                                            device=device,
                                            no_hyper_net=meta),
                                        CustomInitialIntervention(
                                            shared_hypernet=shared_net, h_cell=initial_h_cell, task_name=task_name,
                                            device=device,
                                            no_hyper_net=meta),
                                    ],
                                    actives=[(0, total_episodes, 1, 0),
                                             ])
        env.seed(seed_num + rank)
        env = Monitor(env)
        return env

    set_random_seed(seed_num)
    return _init


def train_policy(num_of_envs,
                 log_relative_path,
                 ppo_config,
                 total_time_steps,
                 validate_every_timesteps,
                 evaluate_dir,
                 shared_hypernet,
                 initial_h_cell,
                 goal_cell,
                 memory_cell,
                 reward_cell,
                 env_,
                 train=False,
                 evaluate=True,
                 load_dir=None,
                 timesteps=0):
    model = PPO(ActorCriticPolicy,
                env_,
                verbose=1,
                h_cell=goal_cell,
                initial_cell=initial_h_cell,
                reward_cell=reward_cell,
                memory_cell=memory_cell,
                shared_hypernet=shared_hypernet,
                num_timesteps=timesteps,
                **ppo_config)
    if load_dir is not "None":
        model.load(path='{}/saved_model'.format(load_dir))
        initial_h_cell.load_state_dict(torch.load(
            'logs/{}/initial_h_cell.pt'.format(load_dir))['model_state_dict'])
        goal_cell.load_state_dict(torch.load(
            'logs/{}/goal_cell.pt'.format(load_dir))['model_state_dict'])
        reward_cell.load_state_dict(torch.load(
            'logs/{}/reward_cell.pt'.format(load_dir))['model_state_dict'])
        memory_cell.load_state_dict(torch.load(
            'logs/{}/memory_cell.pt'.format(load_dir))['model_state_dict'])
        print("Model loads successfully")
    if train:
        for i in range(int(total_time_steps / validate_every_timesteps)):
            checkpoint_freq = int(validate_every_timesteps/num_of_envs)
            checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=log_relative_path,
                                                     name_prefix='rl_model')
            model.learn(total_timesteps=total_time_steps,
                        tb_log_name=log_relative_path,
                        reset_num_timesteps=False,
                        )
            model.save(os.path.join(
                log_relative_path.as_posix()+"_0", 'saved_model'))
            torch.save({
                'model_state_dict': initial_h_cell.state_dict(),
            }, log_relative_path.as_posix()+"_0/initial_cell.pt")
            torch.save({
                'model_state_dict': goal_cell.state_dict(),
            }, log_relative_path.as_posix()+"_0/goal_cell.pt")
            torch.save({
                'model_state_dict': reward_cell.state_dict(),
            }, log_relative_path.as_posix()+"_0/reward_cell.pt")
            torch.save({
                'model_state_dict': memory_cell.state_dict(),
            }, log_relative_path.as_posix()+"_0/memory_cell.pt")
    if evaluate:
        print("="*5, "Start evaluation!", "="*5)
        evaluate_trained_policy("logs/run{}".format(evaluate_dir), model=model)
    return


def save_config_file(ppo_config, env, file_path):
    task_config = env.env._task.get_task_params()
    for task_param in task_config:
        if not isinstance(task_config[task_param], str):
            task_config[task_param] = str(task_config[task_param])
    env_config = env.get_world_params()
    env.close()
    configs_to_save = [task_config, env_config, ppo_config]
    with open(file_path, 'w') as fout:
        json.dump(configs_to_save, fout)


def evaluate_trained_policy(log_relative_path, model):
    task = generate_task(task_generator_id=task_name)
    env = CausalWorld(task, skip_frame=skip_frame, enable_visualization=False,
                      max_episode_length=maximum_episode_length)
    env = CurriculumWrapper(env,
                            intervention_actors=[
                                GoalInterventionActorPolicy()],
                            actives=[(0, 1000000000, 1, 0)])
    env = Monitor(env)
    mean_reward, std_reward, mean_success = evaluate_policy(
        model, env, n_eval_episodes=10)
    print(mean_reward, std_reward, mean_success)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-num", required=False,
                    default=0, help="seed number")
    ap.add_argument("--skip-frame",
                    required=False,
                    default=10,
                    help="skip frame")
    ap.add_argument("--max_episode_length",
                    required=False,
                    default=2500,
                    help="maximum episode length")
    ap.add_argument("--total_time_steps_per_update",
                    required=False,
                    default=100000,
                    help="total time steps per update")
    ap.add_argument("--num-of-envs",
                    required=False,
                    default=1,
                    help="number of parallel environments")
    ap.add_argument("--task-name",
                    required=False,
                    default="reaching",
                    help="the task nam for training")
    ap.add_argument("--fixed-position",
                    required=False,
                    default=True,
                    help="define the reset intervention wrapper")
    ap.add_argument("--evaluate-dir", default=3,
                    required=False, help="Evaluation dir")
    ap.add_argument('--save-freq', default=10000,
                    help='Save model frequency')
    ap.add_argument('--elapsed-timesteps', default=0, type=int,
                    help='The total timesteps elapsed')
    ap.add_argument('--train', default=1, type=int, help="Training the agent")
    ap.add_argument('--eval', default=0, type=int, help="Evaluating the agent")

    # Env hyper
    ap.add_argument('--meta', default=1, type=int, help="Training meta loop")
    ap.add_argument('--shaping', default=0, type=int,
                    help="Reward shaping or not?")
    ap.add_argument('--initial-curriculum', default=0, type=int,
                    help="Use initial state curriculum?")
    ap.add_argument('--goal-curriculum', default=0,
                    type=int, help="Use goal curriculum?")
    ap.add_argument('--all-curricula', default=1,
                    type=int, help="Use all curriculum?")
    ap.add_argument('--memory-only', default=0,
                    type=int, help="Use memory only?")
    ap.add_argument('--device', default='cpu', type=str, help="cpu or cuda?")
    ap.add_argument('--train-time', default=1, type=int,
                    help='How many iterations you want to proceed')

    ap.add_argument('--load-dir', default="None",
                    type=str, help="model load directory")
    ap.add_argument('--machine-id', default='memory_only', type=str,
                    required=False, help='Machine ID')
    ap.add_argument('--total-times', default=10000000,
                    help='Total training time steps')
    ap.add_argument('--random-goal', default=0, type=bool)
    # HyperLSTM parameters
    ap.add_argument('--hidden-size', default=128, type=int)
    ap.add_argument('--hyper-hidden-size', default=128, type=int)
    ap.add_argument('--hyper-embedding-size', default=4, type=int)
    ap.add_argument('--use-layer-norm', default=False, action='store_true')
    ap.add_argument('--drop-prob', default=0.1, type=float)

    # PPO parameters
    ap.add_argument('--gamma', default=0.9995, type=float)
    ap.add_argument('--lr', default=0.00025, type=float)

    args = vars(ap.parse_args())
    for _ in range(int(args['train_time'])):

        model_dir = Path('./logs')

        if not model_dir.exists():
            run_num = 1
        else:
            exist_run_nums = [int(str(folder.name).split('_')[0].split('run')[1]) for folder in model_dir.iterdir() if
                              str(folder.name).startswith('run')]
            if len(exist_run_nums) == 0:
                run_num = 1
            else:
                run_num = max(exist_run_nums) + 1
        current_run = 'run{}_{}_{}'.format(run_num, str(
            args['machine_id']), str(args['task_name']))
        log_dir = model_dir / current_run

        total_time_steps_per_update = int(args['total_time_steps_per_update'])
        num_of_envs = int(args['num_of_envs'])
        log_relative_path = log_dir
        maximum_episode_length = int(args['max_episode_length'])
        skip_frame = int(args['skip_frame'])
        seed_num = int(args['seed_num'])
        task_name = str(args['task_name'])
        fixed_position = bool(args['fixed_position'])
        load_dir = str(args['load_dir'])
        device = str(args['device'])
        total_episodes = int(args['total_times'])
        save_freq = int(args['save_freq'])
        time_elapsed = int(args['elapsed_timesteps'])
        meta = bool(args['meta'])
        reward_shaping = bool(args['shaping'])
        initial_curriculum = bool(args['initial_curriculum'])
        goal_curriculum = bool(args['goal_curriculum'])
        all_curricula = bool(args['all_curricula'])
        memory_only = bool(args['memory_only'])

        assert (((float(total_time_steps_per_update) / num_of_envs) /
                 5).is_integer())

        ppo_config = {
            "gamma": 0.9995,
            "n_steps": 5000,
            "ent_coef": 0,
            "learning_rate": 0.00025,
            "vf_coef": 0.5,
            "max_grad_norm": 10,
            "device": device,
            "tensorboard_log": "./",
            "meta": meta,
            "reward_shaping": reward_shaping,
            "initial_curriculum": initial_curriculum,
            "goal_curriculum": goal_curriculum,
            "all_curricula": all_curricula,
            "memory_only": memory_only
        }
        evaluate_dir = str(args["evaluate_dir"])
        task_gen = generate_task(task_generator_id=task_name)
        env = CausalWorld(task_gen, skip_frame=10, enable_visualization=False)
        obs_dim = env.observation_space.shape[0]
        shared_net = HyperLSTMCell(input_size=64, hidden_size=64,
                                   hyper_hidden_size=64,
                                   hyper_embedding_size=32,
                                   use_layer_norm=False, drop_prob=0.9, obs_dim=obs_dim + 3,
                                   num_chars=11,
                                   device=device).to(device)
        if bool(args['meta']):
            initial_h_cell = LSTMCell(input_size=128, hidden_size=64)
            goal_cell = LSTMCell(input_size=128, hidden_size=64)
            reward_cell = LSTMCell(input_size=128, hidden_size=64)
            memory_cell = LSTMCell(input_size=128, hidden_size=64)
        else:

            initial_h_cell = LSTMGenerator(input_dim=obs_dim + 3, output_dim=6)
            goal_cell = LSTMGenerator(input_dim=obs_dim + 3, output_dim=11)
            reward_cell = LSTMGenerator(input_dim=obs_dim + 3, output_dim=1)
            memory_cell = LSTMGenerator(input_dim=obs_dim + 3, output_dim=1)
        # init env
        env = SubprocVecEnv(
            [_make_env(i,
                       shared_net,
                       initial_h_cell,
                       goal_cell,
                       task_name,
                       random_goal=bool(args['random_goal'])) for i in range(num_of_envs)])

        tensor_log_path = log_relative_path.as_posix()

        train_policy(num_of_envs=num_of_envs,
                     log_relative_path=log_relative_path,
                     ppo_config=ppo_config,
                     total_time_steps=total_episodes,
                     validate_every_timesteps=total_episodes,
                     evaluate_dir=evaluate_dir,
                     train=bool(args['train']),
                     evaluate=bool(args['eval']),
                     load_dir=load_dir,
                     shared_hypernet=shared_net,
                     initial_h_cell=initial_h_cell,
                     goal_cell=goal_cell,
                     memory_cell=memory_cell,
                     reward_cell=reward_cell,
                     env_=env,
                     timesteps=time_elapsed)
