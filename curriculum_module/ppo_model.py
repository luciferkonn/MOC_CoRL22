'''
Author: Jikun Kang
Date: 2021-11-24 09:24:48
LastEditTime: 2022-10-13 17:18:35
LastEditors: Jikun Kang
FilePath: /Learning-Multi-Objective-Curricula-for-Robotic-Policy-Learning/curriculum_module/ppo_model.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.distributions import MultivariateNormal, Normal
from torch.nn.parameter import Parameter
from torch.utils.data import BatchSampler, SubsetRandomSampler


class HyperNet(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_dim=64, device="cuda"):
        super(HyperNet, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim

        self.w1 = Parameter(torch.fmod(torch.randn(
            embedding_dim, (input_dim) * embedding_dim).to(device), 2))
        self.w2 = Parameter(torch.fmod(torch.randn(
            embedding_dim, output_dim).to(device), 2))

        self.b1 = Parameter(torch.fmod(torch.randn(
            input_dim * embedding_dim).to(device), 2))
        self.b2 = Parameter(torch.fmod(torch.randn(output_dim).to(device), 2))

    def forward(self, x):
        h_in = torch.matmul(x, self.w1) + self.b1
        h_in = h_in.view(self.input_dim, self.embedding_dim)
        h_final = torch.matmul(h_in, self.w2) + self.b2
        h_final = h_final.view(self.output_dim, self.input_dim)

        return h_final


class Embedding(nn.Module):
    def __init__(self, z_num, z_num2, z_dim=64, device='cuda'):
        super(Embedding, self).__init__()
        self.z_num = z_num
        self.z_list = nn.ParameterList()
        for i in range(z_num):
            self.z_list.append(
                Parameter(torch.fmod(torch.randn(z_dim).to(device), 2)))

    def forward(self, hyper_net):
        w = []
        for i in range(self.z_num):
            w.append(hyper_net(self.z_list[i]))
        out = torch.cat(w, dim=0)
        return out


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, device='cpu'):
        super(Actor, self).__init__()

        self.hyper_net = HyperNet(obs_dim+11, 1, device=device)
        self.embedding_net = nn.ModuleList([Embedding(obs_dim+11, obs_dim),
                                            Embedding(obs_dim+11, obs_dim),
                                            Embedding(action_dim, action_dim)])
        self.device = device

    def forward(self, x, subgoal):
        x = x.to(self.device)
        subgoal = subgoal.to(self.device)
        x = torch.cat((x, subgoal), 1)
        for i, module in enumerate(self.embedding_net):
            w = self.embedding_net[i](self.hyper_net)
            if i == 2:
                x = torch.tanh(F.linear(x, w))

            else:
                x = F.leaky_relu(F.linear(x, w))
        return x


class Critic(nn.Module):
    def __init__(self, obs_dim, device='cpu'):
        super(Critic, self).__init__()
        self.hyper_net = HyperNet(obs_dim+11, 1, device=device)
        self.embedding_net = nn.ModuleList([Embedding(obs_dim+11, obs_dim),
                                            Embedding(obs_dim+11, obs_dim),
                                            Embedding(1, 1)])
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        for i, module in enumerate(self.embedding_net):
            w = self.embedding_net[i](self.hyper_net)
            if i == 2:
                x = F.linear(x, w)

            else:
                x = F.leaky_relu(F.linear(x, w))
        return x


class PPOFastPolicy:
    def __init__(self, obs_dim,
                 action_dim,
                 clip_param=0.2,
                 max_grad_norm=0.5,
                 ppo_epoch=10,
                 buffer_capacity=1000,
                 batch_size=9,
                 gamma=0.9,
                 action_std_init=0.6,
                 device='cpu'):
        self.action_dim = action_dim
        self.clip_param = clip_param
        self.max_grad_norm = max_grad_norm
        self.ppo_epoch = ppo_epoch
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.gamma = gamma

        self.actor_net = Actor(obs_dim, action_dim, device).float().to(device)
        self.critic_net = Critic(obs_dim, device).float().to(device)
        self.buffer = []
        self.counter = 0
        self.training_step = 0

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 1e-3)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), 4e-3)

        self.action_std = action_std_init
        self.device = device
        self.params_history = []

    def choose_action(self, state, subgoal):
        state = state.float().unsqueeze(0)
        with torch.no_grad():
            action_mean = self.actor_net(state, subgoal)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)
        # dist = Normal(mu, sigma)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        action = action.clamp(-2, 2)
        return action.detach(), action_log_prob.detach()

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1
        return (self.counter % self.buffer_capacity == 0) & (len(self.buffer) > self.buffer_capacity)

    def get_outer_loss(self):
        def outer_loss(params, hparams):

            state = torch.tensor(
                [t.state for t in self.buffer], dtype=torch.float)
            action = torch.tensor(
                [t.action for t in self.buffer], dtype=torch.float).view(-1, 1)
            reward = torch.tensor(
                [t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
            next_state = torch.tensor(
                [t.next_state for t in self.buffer], dtype=torch.float)
            old_action_log_prob = torch.tensor(
                [t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)
            subgoal = torch.tensor([t.subgoal for t in self.buffer])

            with torch.no_grad():
                target_v = reward + self.gamma * \
                    self.critic_net(
                        torch.cat((next_state, subgoal), 1)).to('cpu')

            advantage = (
                target_v - self.critic_net(torch.cat((next_state, subgoal), 1))).detach()

            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, True):
                index = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                action_mean = self.actor_net(state, subgoal)
                cov_mat = torch.diag(self.action_var).unsqueeze(
                    dim=0).to(self.device)
                n = MultivariateNormal(action_mean, cov_mat)
                action_log_prob = n.log_prob(action.flatten()[index])
                ratio = torch.exp(action_log_prob - old_action_log_prob[index])
                l1 = ratio * advantage[index]
                l2 = torch.clamp(ratio, 1 - self.clip_param,
                                 1 + self.clip_param) * advantage[index]
                action_loss = -torch.min(l1, l2).mean()
                # self.actor_optimizer.zero_grad()
                # action_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_net.parameters(), self.max_grad_norm)
                # self.actor_optimizer.step()

                value_loss = F.smooth_l1_loss(self.critic_net(torch.cat((state[index], subgoal[index]), 1)),
                                              target_v[index])
                # self.critic_optimizer.zero_grad()
                # value_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.critic_net.parameters(), self.max_grad_norm)
                # self.critic_optimizer.step()
                value_loss = F.smooth_l1_loss(self.critic_net(torch.cat((state[index], subgoal[index]), 1)),
                                              target_v[index])

                return value_loss
        return self.params_history, outer_loss

    def update(self):
        self.training_step += 1

        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float)
        action = torch.tensor(
            [t.action for t in self.buffer], dtype=torch.float).view(-1, 1)
        reward = torch.tensor(
            [t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        next_state = torch.tensor(
            [t.next_state for t in self.buffer], dtype=torch.float)
        old_action_log_prob = torch.tensor(
            [t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)
        subgoal = torch.tensor([t.subgoal for t in self.buffer])

        with torch.no_grad():
            target_v = reward + self.gamma * \
                self.critic_net(torch.cat((next_state, subgoal), 1)).to('cpu')

        advantage = (
            target_v - self.critic_net(torch.cat((next_state, subgoal), 1))).detach()
        for _ in range(self.ppo_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, True):
                action_mean = self.actor_net(state, subgoal)
                cov_mat = torch.diag(self.action_var).unsqueeze(
                    dim=0).to(self.device)
                n = MultivariateNormal(action_mean, cov_mat)
                action_log_prob = n.log_prob(action.flatten()[index])
                ratio = torch.exp(action_log_prob - old_action_log_prob[index])

                l1 = ratio * advantage[index]
                l2 = torch.clamp(ratio, 1 - self.clip_param,
                                 1 + self.clip_param) * advantage[index]
                action_loss = -torch.min(l1, l2).mean()
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                value_loss = F.smooth_l1_loss(self.critic_net(torch.cat((state[index], subgoal[index]), 1)),
                                              target_v[index])

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.critic_net.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
            self.params_history.append(self.actor_net.parameters())
            del self.buffer[:]

    def set_action_std(self, new_action_std):

        self.action_std = new_action_std
        self.action_var = torch.full(
            (self.action_dim,), new_action_std * new_action_std)  # .to(device)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")

        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if (self.action_std <= min_action_std):
            self.action_std = min_action_std
            print("setting actor output action_std to min_action_std : ",
                  self.action_std)
        else:
            print("setting actor output action_std to : ", self.action_std)
        self.set_action_std(self.action_std)

        print("--------------------------------------------------------------------------------------------")
