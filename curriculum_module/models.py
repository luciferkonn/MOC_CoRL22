'''
Author: Jikun Kang
Date: 2021-11-24 09:24:48
LastEditTime: 2022-10-13 17:14:03
LastEditors: Jikun Kang
FilePath: /Learning-Multi-Objective-Curricula-for-Robotic-Policy-Learning/curriculum_module/models.py
'''
import math

import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from core.custom_intervention import EXT_MEM


class LayerNorm(nn.Module):

    def __init__(self, num_features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps

        self.gamma = nn.Parameter(torch.FloatTensor(num_features))
        self.beta = nn.Parameter(torch.FloatTensor(num_features))


    def reset_parameters(self):
        nn.init.constant_(self.gamma.data, val=1)
        nn.init.constant_(self.beta.data, val=0)

    def forward(self, input):
        mean = input.mean(dim=-1, keepdim=True)
        std = input.std(dim=-1, keepdim=True)
        return self.gamma * (input - mean) / (std + self.eps) + self.beta


class ParallelLayerNorm(nn.Module):

    def __init__(self, num_inputs, num_features, eps=1e-6):
        super(ParallelLayerNorm, self).__init__()
        self.num_inputs = num_inputs
        self.num_features = num_features
        self.eps = eps

        self.gamma = nn.Parameter(torch.FloatTensor(num_inputs, num_features))
        self.beta = nn.Parameter(torch.FloatTensor(num_inputs, num_features))

    def reset_parameters(self):
        nn.init.constant_(self.gamma.data, val=1)
        nn.init.constant_(self.beta.data, val=0)

    def forward(self, *inputs):
        inputs_stack = torch.stack(inputs, dim=-2)
        mean = inputs_stack.mean(dim=-1, keepdim=True)
        std = inputs_stack.std(dim=-1, keepdim=True)
        outputs_stacked = (self.gamma * (inputs_stack -
                           mean) / (std + self.eps) + self.beta)
        outputs = torch.unbind(outputs_stacked, dim=-2)
        return outputs


class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, use_layer_norm=False,
                 dropout_prob=0.90):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_layer_norm = use_layer_norm
        self.dropout_prob = dropout_prob

        self.linear_ih = nn.Linear(input_size, 4 * hidden_size)
        self.linear_hh = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout_prob)
        if use_layer_norm:
            self.ln_ifgo = ParallelLayerNorm(4, hidden_size)
            self.ln_c = LayerNorm(hidden_size)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_ih.weight.data)
        nn.init.constant_(self.linear_ih.bias.data, val=0)
        nn.init.orthogonal_(self.linear_hh.weight.data)
        if self.use_layer_norm:
            self.ln_ifgo.reset_parameters()
            self.ln_c.reset_parameters()

    def forward(self, x, state):
        if state is None:
            batch_size = x.size(0)
            zero_state = Variable(x.data.new(
                batch_size, self.hidden_size).zero_())
            state = (zero_state, zero_state)
        h, c = state
        lstm_vector = self.linear_ih(x) + self.linear_hh(h)
        i, f, g, o = lstm_vector.chunk(chunks=4, dim=1)
        if self.use_layer_norm:
            i, f, g, o = self.ln_ifgo(i, f, g, o)

        f = f + 1
        new_c = c * f.sigmoid() + i.sigmoid() * self.dropout(g.tanh())
        if self.use_layer_norm:
            new_c = self.ln_c(new_c)
        new_h = new_c.tanh() * o.sigmoid()
        new_state = (new_h, new_c)
        return new_h, new_state


def embed_uniform(e):
    bound = math.sqrt(3)
    e.uniform_(-bound, bound)
    return e


def hyperfaninWi_init(i):
    def hyperfanin_init(Wi):
        fan_out, fan_in = Wi.size(0), Wi.size(1)
        bound = math.sqrt(
            3 * 2 / (fan_in * hardcoded_hyperfanin[i]) / hardcoded_receptive(i))
        Wi.uniform_(-bound, bound)
        return Wi

    return hyperfanin_init


def hyperfanoutWi_init(i):
    def hyperfanout_init(Wi):
        fan_out, fan_in = Wi.size(0), Wi.size(1)
        bound = math.sqrt(
            3 * 2 / (fan_in * hardcoded_hyperfanout[i]) / hardcoded_receptive(i))
        Wi.uniform_(-bound, bound)
        return Wi

    return hyperfanout_init


def fanin_uniform(W):
    fan_out, fan_in = W.size(0), W.size(1)
    bound = math.sqrt(3 / fan_in)
    W.uniform_(-bound, bound)
    return W


class LSTMGenerator(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_layer=64):
        super(LSTMGenerator, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_layer)
        self.lstm = nn.LSTM(hidden_layer, output_dim)

    def forward(self, ipts):
        output = self.embedding(ipts)
        output, _ = self.lstm(output.unsqueeze(0))
        return output


class HyperLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, hyper_hidden_size, hyper_embedding_size, use_layer_norm, drop_prob,
                 num_chars, center=(2, 2), log_writter=None, obs_dim=37, hidden_layer=64, embed_dim=56, device='cpu'):
        super(HyperLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hyper_hidden_size = hyper_hidden_size
        self.hyper_embedding_size = hyper_embedding_size
        self.use_layer_norm = use_layer_norm
        self.dropout_prob = drop_prob
        self.center = center
        # self.hyper_cell = lstm_cell
        self.log_writter = log_writter
        self.embed_dim = embed_dim

        # external memory
        self.last_k = torch.ones((1, 64)).to(device)
        self.external_mem = torch.ones((1, 64)).to(device)
        self.cos_sim = nn.CosineSimilarity()

        # initialize LSTM linear parameters
        self.ih_weight = nn.Parameter(torch.zeros(4 * hidden_size, input_size))
        self.ih_bias = nn.Parameter(torch.zeros(4 * hidden_size))
        self.hh_weight = nn.Parameter(
            torch.zeros(4 * hidden_size, hidden_size))

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_layer),
            nn.ReLU(True),
            nn.Linear(hidden_layer, hidden_layer),
            nn.ReLU(True),
            nn.Linear(hidden_layer, hidden_layer)
        )

        # Hyper LSTM: Projection
        for y in ('i', 'f', 'g', 'o'):
            proj_h = nn.Linear(hyper_hidden_size, hyper_embedding_size)
            proj_x = nn.Linear(hyper_hidden_size, hyper_embedding_size)
            proj_b = nn.Linear(hyper_hidden_size,
                               hyper_embedding_size, bias=False)

            setattr(self, f'hyper_proj_{y}h', proj_h)
            setattr(self, f'hyper_proj_{y}x', proj_x)
            setattr(self, f'hyper_proj_{y}b', proj_b)

        # Hyper LSTM: Scaling
        for y in ('i', 'f', 'g', 'o'):
            scale_h = nn.Linear(hyper_embedding_size, hidden_size, bias=False)
            scale_x = nn.Linear(hyper_embedding_size, hidden_size, bias=False)
            scale_b = nn.Linear(hyper_embedding_size, hidden_size, bias=False)

            setattr(self, f'hyper_scale_{y}h', scale_h)
            setattr(self, f'hyper_scale_{y}x', scale_x)
            setattr(self, f'hyper_scale_{y}b', scale_b)

        self.linear_ih = nn.Linear(input_size, 4 * hidden_size, bias=False)
        self.linear_hh = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

        # Linear transformation for external memory
        self.linear_trans = nn.Linear(hidden_size, hidden_size, bias=True)

        self.bias = nn.Parameter(torch.FloatTensor(4 * hidden_size))
        if self.use_layer_norm:
            self.ln_ifgo = ParallelLayerNorm(4, hidden_size)
            self.ln_c = LayerNorm(hidden_size)

        self.output_proj = nn.Linear(in_features=hidden_size,
                                     out_features=num_chars)

        self.dropout = nn.Dropout(drop_prob)
        self.reset_parameters()

    def reset_parameters(self):
        # Hyper LSTM
        # self.hyper_cell.reset_parameters()
        # Hyper LSTM: Projection
        for y in ('i', 'g', 'f', 'o'):
            proj_h = getattr(self, f'hyper_proj_{y}h')
            proj_x = getattr(self, f'hyper_proj_{y}x')
            proj_b = getattr(self, f'hyper_proj_{y}b')
            nn.init.constant_(proj_h.weight.data, val=0)
            nn.init.constant_(proj_h.bias.data, val=1)
            nn.init.constant_(proj_x.weight.data, val=0)
            nn.init.constant_(proj_x.bias.data, val=1)
            nn.init.normal_(proj_b.weight.data, mean=0, std=0.01)

        # Hyper LSTM: Scaling
        for y in ('i', 'g', 'f', 'o'):
            scale_h = getattr(self, f'hyper_scale_{y}h')
            scale_x = getattr(self, f'hyper_scale_{y}x')
            scale_b = getattr(self, f'hyper_scale_{y}b')
            nn.init.constant_(scale_h.weight.data, val=0.1 /
                              self.hyper_embedding_size)
            nn.init.constant_(scale_x.weight.data, val=0.1 /
                              self.hyper_embedding_size)
            nn.init.constant_(scale_b.weight.data, val=0)

        # Main LSTM
        nn.init.xavier_uniform_(self.linear_ih.weight.data)
        nn.init.orthogonal_(self.linear_hh.weight.data)
        nn.init.constant_(self.bias.data, val=0)

        # LayerNorm
        if self.use_layer_norm:
            self.ln_ifgo.reset_parameters()
            self.ln_c.reset_parameters()

    def compute_hyper_vector(self, hyper_h, name):
        proj = getattr(self, f'hyper_proj_{name}')
        scale = getattr(self, f'hyper_scale_{name}')
        return scale(proj(hyper_h))

    def forward(self,
                x,
                state,
                hyper_state,
                lstm_cell,
                mask=None,
                emit_mem=False,
                emit_self_mem=False,
                read_mem=False):
        x = self.encoder(x)
        # x = x.squeeze(0)
        if state is None:
            batch_size = x.size(0)
            zero_state = Variable(x.data.new(
                batch_size, self.hidden_size).zero_())

            state = (zero_state, zero_state)
        h, c = state

        if read_mem:
            self.external_mem = EXT_MEM
        # Run a signle step of Hyper LSTM
        hyper_input = torch.cat([x, h], dim=1)
        new_hyper_h, new_hyper_state = lstm_cell(hyper_input, hyper_state)

        # Then, compute values for the main LSTM
        xh = self.linear_ih(x)
        hh = self.linear_hh(h)

        ix, fx, gx, ox = xh.chunk(chunks=4, dim=1)
        ix = ix * self.compute_hyper_vector(new_hyper_h, 'ix')
        fx = fx * self.compute_hyper_vector(new_hyper_h, 'fx')
        gx = gx * self.compute_hyper_vector(new_hyper_h, 'gx')
        ox = ox * self.compute_hyper_vector(new_hyper_h, 'ox')

        ih, fh, gh, oh = hh.chunk(chunks=4, dim=1)
        ih = ih * self.compute_hyper_vector(new_hyper_h, 'ih')
        fh = fh * self.compute_hyper_vector(new_hyper_h, 'fh')
        gh = gh * self.compute_hyper_vector(new_hyper_h, 'gh')
        oh = oh * self.compute_hyper_vector(new_hyper_h, 'oh')

        ib, fb, gb, ob = self.bias.chunk(chunks=4, dim=0)
        ib = ib * self.compute_hyper_vector(new_hyper_h, 'ib')
        fb = fb * self.compute_hyper_vector(new_hyper_h, 'fb')
        gb = gb * self.compute_hyper_vector(new_hyper_h, 'gb')
        ob = ob * self.compute_hyper_vector(new_hyper_h, 'ob')

        i = ix + ih + ib
        f = fx + fh + fb + 1  # Set the initial forget bias to 1
        g = gx + gh + gb
        o = ox + oh + ob

        if self.use_layer_norm:
            i, f, g, o = self.ln_ifgo(i, f, g, o)
        new_c = c * f.sigmoid() + self.dropout(g.tanh()) * i.sigmoid()
        if self.use_layer_norm:
            new_c = self.ln_c(new_c)
        new_h = new_c.tanh() * o.sigmoid()

        # Apply the mask vector
        if mask is not None:
            mask = mask.unsqueeze(1)
            new_h = new_h * mask + h * (1 - mask)
            new_c = new_c * mask + c * (1 - mask)

        # Compute k_t, e_t, a_t
        linear = self.linear_trans(new_hyper_h)
        k_t = torch.tanh(linear)
        e_t = torch.sigmoid(linear)
        a_t = k_t

        if read_mem:
            r_t = a_t*self.external_mem

        new_state = (new_h, new_c)
        output = self.output_proj(new_h)

        if emit_mem:
            # Write memory
            # Compute cosine similarity
            cos_sim_value = self.cos_sim(self.external_mem, self.last_k)
            weight = torch.softmax(cos_sim_value, dim=0).unsqueeze(1)
            self.external_mem = self.external_mem * \
                (1 - weight * e_t) + weight * a_t

            self.last_k = k_t
            return output, new_state, new_hyper_state, self.external_mem

        if emit_self_mem:
            return output, new_state, new_hyper_state, self.external_mem
        else:
            return output, new_state, new_hyper_state


class SharedHypernet(nn.Module):

    def __init__(self, obs_dim, role_dim=3, hidden_layer=64):
        super(SharedHypernet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + role_dim, hidden_layer),
            nn.ReLU(True),
            nn.Linear(hidden_layer, hidden_layer),
            nn.ReLU(True),
            nn.Linear(hidden_layer, hidden_layer)
        )

    def forward(self, state, role):
        out = self.encoder(torch.cat((state, role), dim=1))
        return out


class BaseNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_layer=64):
        super(BaseNet, self).__init__()
        self.linear = nn.Sequential(nn.Linear(obs_dim, hidden_layer),
                                    nn.ReLU(),
                                    nn.Linear(hidden_layer, hidden_layer),
                                    nn.ReLU(),
                                    nn.Linear(hidden_layer, act_dim))

    def forward(self, obs):
        action_prob = self.linear(obs)
        return action_prob


class DQNFastPolicy:
    def __init__(self,
                 obs_dim,
                 act_dim,
                 memory_capacity=10000,
                 hidden_layer=64,
                 lr=1e-3,
                 q_learn_iteration=100,
                 batch_size=64,
                 epsilon=0.9,
                 gamma=0.9):
        super(DQNFastPolicy, self).__init__()
        self.q_learn_iteration = q_learn_iteration
        self.memory_capacity = memory_capacity
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.gamma = gamma

        self.eval_net, self.target_net = BaseNet(obs_dim, act_dim, hidden_layer), BaseNet(obs_dim, act_dim,
                                                                                          hidden_layer)

        self.memory = np.zeros((memory_capacity, obs_dim * 2 + 2))
        self.memory_counter = 0
        self.learn_step_counter = 0

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()

    def store_transition(self, state, action, reward, next_state):
        transition = numpy.hstack((state, [action, reward], next_state))
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, state):
        if np.random.randn() <= self.epsilon:
            action_value = self.eval_net(state)
            action = torch.max(action_value).data.numpy()
            action = action
        else:
            action = np.random.randint(self.act_dim)
        if action > 9:
            print()
        return action

    def learn(self):
        if self.learn_step_counter % self.q_learn_iteration == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :self.obs_dim])
        batch_action = torch.LongTensor(
            batch_memory[:, self.obs_dim:self.obs_dim + 1].astype(int))
        batch_reward = torch.FloatTensor(
            batch_memory[:, self.obs_dim + 1:self.obs_dim + 2])
        batch_next_state = torch.FloatTensor(batch_memory[:, -self.obs_dim:])

        v_ = self.eval_net(batch_state)
        q_eval = v_.gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + self.gamma * \
            q_next.max(1)[0].view(self.batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
