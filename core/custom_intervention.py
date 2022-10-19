'''
Author: Jikun Kang
Date: 2021-11-24 09:24:48
LastEditTime: 2022-10-13 17:13:56
LastEditors: Jikun Kang
FilePath: /Learning-Multi-Objective-Curricula-for-Robotic-Policy-Learning/core/custom_intervention.py
'''
import numpy as np
import torch
from causal_world.intervention_actors.base_actor import \
    BaseInterventionActorPolicy
from causal_world.task_generators.task import generate_task

EXT_MEM = torch.ones((64, 64))


class CustomGoalIntervention(BaseInterventionActorPolicy):

    def __init__(self, shared_hypernet, h_cell, task_name, device='cpu', no_hyper_net=False):
        """
        This class indicates the goal intervention actor, which an
        intervention actor that intervenes by sampling a new goal.

        :param kwargs: (params) parameters for the construction of the actor.
        """
        super(CustomGoalIntervention, self).__init__()
        self.shared_hypernet = shared_hypernet
        self.h_cell = h_cell.to(device)
        self.state = None
        self.hyper_state = None
        self.h_state = None
        self.h_hyper_state = None
        self.task_name = task_name
        self.device = device
        self.no_shared_hypernet = no_hyper_net

    def initialize(self, env):
        """
        This functions allows the intervention actor to query things from the env, such
        as intervention spaces or to have access to sampling funcs for goals..etc

        :param env: (causal_world.env.CausalWorld) the environment used for the
                                                   intervention actor to query
                                                   different methods from it.

        :return:
        """
        self.state = None
        self.hyper_state = None
        self.h_state = None
        self.h_hyper_state = None
        self.env = env
        return

    def _act(self, variables_dict):
        """

        :param variables_dict:
        :return:
        {'goal_60': {'cylindrical_position'an: array([0.1268533 , 0.30670429, 0.18678756]),
        'euler_orientation': array([0.        , 0.        , 0.91667975])},
        'goal_120': {'cylindrical_position': array([ 0.1416506 , -0.39215112,  0.28964389]),
        'euler_orientation': array([0.        , 0.        , 0.18155214])},
        'goal_300': {'cylindrical_position': array([0.1443119 , 0.42753659, 0.03524528]),
        'euler_orientation': array([0.        , 0.        , 2.08991213])}}
        """
        obs = self.env.reset()
        input = torch.cat((torch.from_numpy(obs).float().unsqueeze(
            0), torch.tensor([1, 0, 0]).unsqueeze(0)), dim=1)
        # embed_state, h_state, h_hyper_state, out = self.shared_hypernet(x=input, state=self.h_state,
        #                                                                 hyper_state=self.h_hyper_state, emit_mem=True)
        # output, state, hyper_state = self.h_cell(x=input, state=self.state, hyper_state=self.hyper_state)
        if not self.no_shared_hypernet:
            output = self.h_cell(input.to(self.device))
            output = output.cpu().detach().numpy().squeeze(0).squeeze(0)
        else:
            output, h_state, h_hyper_state, out = self.shared_hypernet(x=input.to(self.device),
                                                                       state=self.h_state,
                                                                       hyper_state=self.h_hyper_state,
                                                                       lstm_cell=self.h_cell,
                                                                       emit_mem=True)
            EXT_MEM = out
            # output, state, hyper_state = self.h_cell(x=embed_state, state=self.state,
            #                                          hyper_state=self.hyper_state)
            output = output.cpu().detach().numpy().squeeze(0)

            # update states
            # self.state = state
            # self.hyper_state = hyper_state
            self.h_state = h_state
            self.h_hyper_state = h_hyper_state
        if self.task_name == "reaching":
            interventions_dict = {'goal_60': {'cylindrical_position': np.array((self.unscale_action(output[0], 0, 0.15),
                                                                                self.unscale_action(output[1],
                                                                                                    -3.14159265,
                                                                                                    3.14159265),
                                                                                self.unscale_action(output[2], 0.015,
                                                                                                    0.3))),
                                              'euler_orientation': np.array((0.,
                                                                             0.,
                                                                             self.unscale_action(output[3], -1, 1)))},
                                  'goal_120': {
                                      'cylindrical_position': np.array((self.unscale_action(output[4], 0, 0.15),
                                                                        self.unscale_action(output[5], -3.14159265,
                                                                                            3.14159265),
                                                                        self.unscale_action(output[6], 0.015,
                                                                                            0.3))),
                                      'euler_orientation': np.array((0.,
                                                                     0.,
                                                                     self.unscale_action(output[7], 0.02, 0.08)))},
                                  'goal_300': {
                                      'cylindrical_position': np.array((self.unscale_action(output[8], 0, 0.15),
                                                                        self.unscale_action(output[9], -3.14159265,
                                                                                            3.14159265),
                                                                        self.unscale_action(output[10], 0.015,
                                                                                            0.3))),
                                      'euler_orientation': np.array((0.,
                                                                     0.,
                                                                     self.unscale_action(output[2], 0.02, 0.08)))}}
        elif self.task_name == "picking":
            interventions_dict = {
                'goal_block': {'cylindrical_position': np.array((self.unscale_action(output[0], 0, 0.15),
                                                                 self.unscale_action(output[1], -3.14159265,
                                                                                     3.14159265),
                                                                 self.unscale_action(output[2], 0.08,
                                                                                     0.25))),
                               'euler_orientation': np.array((0.,
                                                              0.,
                                                              self.unscale_action(output[3], -1, 1)))}}
        elif self.task_name == "pushing":
            interventions_dict = {
                'goal_block': {'cylindrical_position': np.array((self.unscale_action(output[0], 0, 0.15),
                                                                 self.unscale_action(output[1], -3.14159265,
                                                                                     3.14159265),
                                                                 0.0325)),
                               'euler_orientation': np.array((0.,
                                                              0.,
                                                              self.unscale_action(output[3], -1, 1)))}}
        elif self.task_name == "pick_and_place":
            interventions_dict = {
                'goal_block': {'cylindrical_position': np.array((self.unscale_action(output[0], 0.07, 0.15),
                                                                 self.unscale_action(output[1], -2.61799388,
                                                                                     -0.52359878),
                                                                 0.0325)),
                               'euler_orientation': np.array((0.,
                                                              0.,
                                                              self.unscale_action(output[3], -1, 1)))}}
        elif self.task_name == "stacking2":
            interventions_dict = {
                'goal_tower': {'cylindrical_position': np.array((self.unscale_action(output[0], 0, 0.15),
                                                                 self.unscale_action(output[1], -3.14159265,
                                                                                     3.14159265),
                                                                 0.065))},
                # 'goal_block_2': {
                #     'cylindrical_position': np.array((self.unscale_action(output[3], 0, 0.15),
                #                                     self.unscale_action(output[4], -3.14159265,
                #                                                         3.14159265),
                #                                     self.unscale_action(output[5], 0.08,
                #                                                         0.25)))}
            }
        elif self.task_name == "stacked_blocks":
            interventions_dict = {
                'goal_block': {'cylindrical_position': np.array((self.unscale_action(output[0], 0, 0.15),
                                                                 self.unscale_action(output[1], -3.14159265,
                                                                                     3.14159265),
                                                                 self.unscale_action(output[2], 0.08,
                                                                                     0.25))),
                               'euler_orientation': np.array((0.,
                                                              0.,
                                                              self.unscale_action(output[3], -1, 1)))}}
        elif self.task_name == "towers":
            interventions_dict = {
                'goal_block': {'cylindrical_position': np.array((self.unscale_action(output[0], 0, 0.15),
                                                                 self.unscale_action(output[1], -3.14159265,
                                                                                     3.14159265),
                                                                 self.unscale_action(output[2], 0.08,
                                                                                     0.25))),
                               'euler_orientation': np.array((0.,
                                                              0.,
                                                              self.unscale_action(output[3], -1, 1)))}}
        elif self.task_name == "general":
            interventions_dict = {
                'goal_0': {'cylindrical_position': np.array((self.unscale_action(output[0], 0, 0.15),
                                                             self.unscale_action(output[1], -3.14159265,
                                                                                 3.14159265),
                                                             self.unscale_action(output[2], 0.08,
                                                                                 0.25)))},
                'goal_1': {'cylindrical_position': np.array((self.unscale_action(output[0], 0, 0.15),
                                                             self.unscale_action(output[1], -3.14159265,
                                                                                 3.14159265),
                                                             self.unscale_action(output[2], 0.08,
                                                                                 0.25)))},
                'goal_2': {'cylindrical_position': np.array((self.unscale_action(output[0], 0, 0.15),
                                                             self.unscale_action(output[1], -3.14159265,
                                                                                 3.14159265),
                                                             self.unscale_action(output[2], 0.08,
                                                                                 0.25)))}
            }
        elif self.task_name == "creative_stacked_blocks":
            interventions_dict = {
                'goal_block': {'cylindrical_position': np.array((self.unscale_action(output[0], 0, 0.15),
                                                                 self.unscale_action(output[1], -3.14159265,
                                                                                     3.14159265),
                                                                 self.unscale_action(output[2], 0.08,
                                                                                     0.25))),
                               'euler_orientation': np.array((0.,
                                                              0.,
                                                              self.unscale_action(output[3], -1, 1)))}}
        else:
            raise NotImplementedError(
                "The task name is incorrect, please check it again.")

        print("current goal is: {}".format(interventions_dict))
        # self.state = state
        # self.hyper_state = hyper_state

        return interventions_dict

    def unscale_action(self, scaled_action, low, high):
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)

        :param scaled_action: Action to un-scale
        """
        return low + (0.5 * (scaled_action + 1.0) * (high - low))

    def get_params(self):
        """
        returns parameters that could be used in recreating this intervention
        actor.

        :return: (dict) specifying paramters to create this intervention actor
                        again.
        """
        return {'goal_actor': dict()}


class CustomInitialIntervention(BaseInterventionActorPolicy):
    def __init__(self, shared_hypernet, h_cell, task_name, device='cpu', no_hyper_net=False):
        """
        This is a random intervention actor which intervenes randomly on
        all available state variables except joint positions since its a
        trickier space.
        :param kwargs:
        """
        super(CustomInitialIntervention, self).__init__()
        self.shared_hypernet = shared_hypernet
        self.h_cell = h_cell.to(device)
        self.state = None
        self.hyper_state = None
        self.task_name = task_name
        self.device = device
        self.no_shared_hypernet = no_hyper_net

    def initialize(self, env):
        """
        This functions allows the intervention actor to query things from the env, such
        as intervention spaces or to have access to sampling funcs for goals..etc

        :param env: (causal_world.env.CausalWorld) the environment used for the
                                                   intervention actor to query
                                                   different methods from it.

        :return:
        """
        self.state = None
        self.hyper_state = None
        self.env = env
        self.h_state = None
        self.h_hyper_state = None

        return

    def _act(self, variables_dict):
        """

        :param variables_dict:
        :return:
        """
        obs = self.env.reset()
        # embed_state = self.shared_hypernet(torch.from_numpy(obs).float().unsqueeze(0),
        #                                    torch.tensor([0, 1, 0]).unsqueeze(0))
        input = torch.cat((torch.from_numpy(obs).float().unsqueeze(
            0), torch.tensor([1, 0, 0]).unsqueeze(0)), dim=1)
        # embed_state, h_state, h_hyper_state, out = self.shared_hypernet(x=input, state=self.h_state,
        #                                                                 hyper_state=self.h_hyper_state, emit_mem=True)
        # EXT_MEM = out
        # output, state, hyper_state = self.h_cell(x=input, state=self.state, hyper_state=self.hyper_state)

        if not self.no_shared_hypernet:
            output = self.h_cell(input.to(self.device))
            output = output.detach().cpu().numpy().squeeze(0).squeeze(0)
        else:
            output, h_state, h_hyper_state, out = self.shared_hypernet(x=input.to(self.device),
                                                                       state=self.h_state,
                                                                       hyper_state=self.h_hyper_state,
                                                                       lstm_cell=self.h_cell,
                                                                       emit_mem=True)
            EXT_MEM = out
            # output, state, hyper_state = self.h_cell(x=embed_state, state=self.state, hyper_state=self.hyper_state)
            output = output.detach().cpu().numpy().squeeze(0)
            self.h_state = h_state
            self.h_hyper_state = h_hyper_state
            # self.env.set_starting_state(
            #     {'goal_block': {
            #         'cartesian_position': output
            #     }})

            # cartesian_position cylindrical_position
        if self.task_name == "reaching":
            interventions_dict = {'goal_60': {'cylindrical_position': np.array((self.unscale_action(output[0], 0, 0.15),
                                                                                self.unscale_action(output[1],
                                                                                                    -3.14159265,
                                                                                                    3.14159265),
                                                                                0.0325)),
                                              }}
        elif self.task_name == "picking":
            interventions_dict = {
                'goal_block': {'cylindrical_position': np.array((self.unscale_action(output[0], 0, 0.15),
                                                                 self.unscale_action(output[1], -3.14159265,
                                                                                     3.14159265),
                                                                 self.unscale_action(output[2], 0.08,
                                                                                     0.25))),
                               'euler_orientation': np.array((0.,
                                                              0.,
                                                              self.unscale_action(output[3], -1, 1)))}}
        elif self.task_name == "pushing":
            interventions_dict = {
                'goal_block': {'cylindrical_position': np.array((self.unscale_action(output[0], 0, 0.15),
                                                                 self.unscale_action(output[1], -3.14159265,
                                                                                     3.14159265),
                                                                 0.0325)),
                               'euler_orientation': np.array((0.,
                                                              0.,
                                                              self.unscale_action(output[2], -1, 1)))}}
        elif self.task_name == "pick_and_place":
            interventions_dict = {
                'goal_block': {'cylindrical_position': np.array((self.unscale_action(output[0], 0.07, 0.15),
                                                                 self.unscale_action(output[1], -0.52359878,
                                                                                     -2.61799388),
                                                                 0.0325)),
                               # 'euler_orientation': np.array((0.,
                               #                                0.,
                               #                                self.unscale_action(output[3], -1, 1)))
                               }}
        elif self.task_name == "stacking2":
            interventions_dict = {
                'goal_tower': {'cylindrical_position': np.array((self.unscale_action(output[0], 0, 0.15),
                                                                 self.unscale_action(output[1], -3.14159265,
                                                                                     3.14159265),
                                                                 0.065))},
                # 'goal_block_2': {
                #     'cylindrical_position': np.array((self.unscale_action(output[3], 0, 0.15),
                #                                     self.unscale_action(output[4], -3.14159265,
                #                                                         3.14159265),
                #                                     self.unscale_action(output[5], 0.08,
                #                                                         0.25)))}
            }
        elif self.task_name == "stacked_blocks":
            interventions_dict = {
                'goal_block': {'cylindrical_position': np.array((self.unscale_action(output[0], 0, 0.15),
                                                                 self.unscale_action(output[1], -3.14159265,
                                                                                     3.14159265),
                                                                 self.unscale_action(output[2], 0.08,
                                                                                     0.25))),
                               # 'euler_orientation': np.array((0.,
                               #                                0.,
                               #                                self.unscale_action(output[3], -1, 1)))
                               }}
        elif self.task_name == "towers":
            interventions_dict = {
                'goal_block': {'cylindrical_position': np.array((self.unscale_action(output[0], 0, 0.15),
                                                                 self.unscale_action(output[1], -3.14159265,
                                                                                     3.14159265),
                                                                 self.unscale_action(output[2], 0.08,
                                                                                     0.25))),
                               # 'euler_orientation': np.array((0.,
                               #                                0.,
                               #                                self.unscale_action(output[3], -1, 1)))
                               }}
        elif self.task_name == "general":
            interventions_dict = {
                'goal_0': {'cylindrical_position': np.array((self.unscale_action(output[0], 0, 0.15),
                                                             self.unscale_action(output[1], -3.14159265,
                                                                                 3.14159265),
                                                             self.unscale_action(output[2], 0.08,
                                                                                 0.25))),
                           # 'euler_orientation': np.array((0.,
                           #                                0.,
                           #                                self.unscale_action(output[3], -1, 1)))
                           }}
        elif self.task_name == "creative_stacked_blocks":
            interventions_dict = {
                'goal_block': {'cylindrical_position': np.array((self.unscale_action(output[0], 0, 0.15),
                                                                 self.unscale_action(output[1], -3.14159265,
                                                                                     3.14159265),
                                                                 self.unscale_action(output[2], 0.08,
                                                                 0.25))),
                               }}
# env = get_world(task_gen.get_task_name(),
#                 task_gen.get_task_params(),
#                 world_params,
#                 enable_visualization=False,
#                 env_wrappers=np.array([]),
#                 env_wrappers_args=np.array([]))
# obs = env.reset()             0.25))),
            # 'euler_orientation': np.array((0.,
            #                                0.,
            #                                self.unscale_action(output[3], -1, 1)))

        else:
            raise NotImplementedError(
                "The task name is incorrect, please check it again.")

        print("current initial state is: {}".format(output))
        # self.state = state
        # self.hyper_state = hyper_state

        return interventions_dict

    def unscale_action(self, scaled_action, low, high):
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)

        :param scaled_action: Action to un-scale
        """
        return low + (0.5 * (scaled_action + 1.0) * (high - low))

    def get_params(self):
        """
        returns parameters that could be used in recreating this intervention
        actor.

        :return: (dict) specifying paramters to create this intervention actor
                        again.
        """
        return {'initial_actor': dict()}
