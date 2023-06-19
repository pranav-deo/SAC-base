import torch
import torch.nn as nn
import torch.nn.functional as F
from ReplayBuffer import ReplayBuffer
import copy
import numpy as np
from mlflow import log_metric
import tqdm
from utils import do_torchviz_plots

class BaseNetwork(nn.Module):
    def __init__(self, input, output, hidden_dim=256) -> None:
        super().__init__()
        self.hidden_dims = hidden_dim
        self.h1 = nn.Linear(input, self.hidden_dims)
        self.h2 = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.h3 = nn.Linear(self.hidden_dims, output)

    def forward(self, x):
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = self.h3(x)
        return x

class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256) -> None:
        super().__init__()
        self.hidden_dims = hidden_dim
        self.h1 = nn.Linear(obs_dim, self.hidden_dims)
        self.h2 = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.mean_layer = nn.Linear(self.hidden_dims, act_dim)
        self.log_std_layer = nn.Linear(hidden_dim, act_dim)

    def forward(self, x):
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        mean, log_std = torch.tanh(self.mean_layer(x)), self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20, 2)
        dist = torch.distributions.Normal(mean, log_std.exp())
        return dist

    def logprob(self, obs, act):
        dist = self.forward(obs)
        return torch.sum(dist.log_prob(act), dim=-1, keepdim=True)

class DoubleCriticNetwork(nn.Module):
    def __init__(self, obs_act_dim, hidden_dim=256) -> None:
        super().__init__()
        self.q1 = BaseNetwork(obs_act_dim, 1, hidden_dim)
        self.q2 = BaseNetwork(obs_act_dim, 1, hidden_dim)

    def forward(self, obs, act):
        inp = torch.concat((obs, act), dim=-1)
        return self.q1(inp), self.q2(inp)

class ValueNetwork(BaseNetwork):
    def __init__(self, obs_dim, hidden_dim=256) -> None:
        super().__init__(obs_dim, 1, hidden_dim)

class SAC:
    def __init__(self, env, eval_env, args):
        
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.actor_net = ActorNetwork(obs_dim, act_dim).to(self.device)
        self.critic_net = DoubleCriticNetwork(obs_dim + act_dim).to(self.device)
        self.value_net = ValueNetwork(obs_dim).to(self.device)
        self.target_value = copy.deepcopy(self.value_net).to(self.device)
        for p in self.target_value.parameters():
            p.requires_grad = False

        self.buffer = ReplayBuffer(args.buffer_size, obs_dim, act_dim, self.device, args.seed)
        self.actor_optim = torch.optim.Adam(
            self.actor_net.parameters(), lr=args.lr_actor)
        self.critic_optim = torch.optim.Adam(
            self.actor_net.parameters(), lr=args.lr_critic)
        self.value_optim = torch.optim.Adam(
            self.actor_net.parameters(), lr=args.lr_value)

        self.env = env
        self.eval_env = eval_env
        self.tau = args.tau
        self.alpha = args.alpha
        self.discount = args.discount
        self.batch_size = args.batch_size
        self.num_env_steps = args.num_env_steps
        self.eval_interval = args.eval_interval
        self.num_grad_steps = args.num_grad_steps
        self.num_eval_epochs = args.num_eval_epochs
        self.num_steps = args.num_steps
        self.start_steps = args.start_steps
        self.update_after = args.update_after
        self.plot_computational_graph = args.plot_computational_graph

    def select_action(self, obs, stochastic=True):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        if stochastic:
            return self.actor_net(obs).rsample()
        return self.actor_net(obs).mean

    def update_actor(self, obs):
        act = self.select_action(obs)
        log_prob = self.actor_net.logprob(obs, act)
        with torch.no_grad():
            q_min = torch.minimum(*self.critic_net(obs, act))
        loss = (self.alpha*log_prob - q_min).mean()

        if self.plot_computational_graph:
            do_torchviz_plots(loss, self.actor_net, self.critic_net, 
                            self.target_value, self.value_net, 'actor_update')

        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()
        return loss

    def update_value(self, obs, act):
        value = self.value_net(obs)
        with torch.no_grad():
            q_min = torch.minimum(*self.critic_net(obs, act))
            log_prob = self.actor_net.logprob(obs, act)
            target_v = q_min - self.alpha * log_prob
        loss = F.mse_loss(value, target_v).mean()

        if self.plot_computational_graph:
            do_torchviz_plots(loss, self.actor_net, self.critic_net, 
                            self.target_value, self.value_net, 'value_update')


        self.value_optim.zero_grad()
        loss.backward()
        self.value_optim.step()
        return loss, value.detach().mean().item()

    def update_critic(self, obs, next_obs, act, rew, not_dones):
        with torch.no_grad():
            value = self.target_value(next_obs)
            target_q = rew + self.discount * not_dones * value
        q1, q2 = self.critic_net(obs, act)
        
        loss = (F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)).mean()

        if self.plot_computational_graph:
            do_torchviz_plots(loss, self.actor_net, self.critic_net, 
                            self.target_value, self.value_net, 'critic_update')


        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
        return loss, (q1.detach().mean().item(), q2.detach().mean().item())

    def soft_update(self):
        with torch.no_grad():
            for p, tp in zip(self.value_net.parameters(), self.target_value.parameters()):
                tp.data.copy_((1.0-self.tau) * tp.data + self.tau*p.data)

    def train(self):
        obs, _ = self.env.reset()
        ep_rew = 0
        for step in tqdm.tqdm(range(self.num_steps // self.num_env_steps)):
            for env_step in range(self.num_env_steps):
                if (step * self.num_env_steps + env_step) > self.start_steps:
                    act = self.select_action(obs).cpu().detach().numpy()
                else:
                    act = self.env.action_space.sample()
                next_obs, rew, terminated, truncated, info = self.env.step(act)
                ep_rew += rew
                not_done = 1.0 - (terminated or truncated)
                self.buffer.add_step(obs, next_obs, act, rew, not_done)
                if (terminated or truncated):
                    next_obs, _  = self.env.reset()
                    ep_rew = 0
                obs = next_obs
            log_metric('train_reward', ep_rew, step * self.num_env_steps)
            if (step * self.num_env_steps + env_step) > self.update_after:
                for grad_step in range(self.num_grad_steps):
                    b_obs, b_next_obs, b_act, b_rew, b_not_dones = self.buffer.sample(self.batch_size)
                    value_loss, val = self.update_value(b_obs, b_act)
                    critic_loss, q = self.update_critic(b_obs, b_next_obs, b_act, b_rew, b_not_dones)
                    actor_loss = self.update_actor(b_obs)
                    self.soft_update()

                curr_step = step * self.num_grad_steps + env_step
                log_metric('value', val, curr_step)
                log_metric('q1', q[0], curr_step)
                log_metric('q2', q[1], curr_step)
                log_metric('value_loss', value_loss.item(), curr_step)
                log_metric('critic_loss', critic_loss.item(), curr_step)
                log_metric('actor_loss', actor_loss.item(), curr_step)

            if ((step + 1) * self.num_env_steps) % self.eval_interval == 0:
                eval_rew = self.eval()
                log_metric('eval_reward', eval_rew, step * self.num_env_steps)

    def eval(self):
        epoch_rew = []
        for _ in range(self.num_eval_epochs):
            obs, done = self.eval_env.reset()[0], False
            ep_reward = 0
            while not done:
                act = self.select_action(obs, stochastic=False).cpu().detach().numpy()
                next_obs, rew, terminated, truncated, done = self.eval_env.step(act)
                ep_reward += rew
                done = terminated or truncated
                obs = next_obs
            epoch_rew.append(ep_reward)
        return np.mean(epoch_rew)
    

    def save_model(self, path):
        torch.save(self.actor_net.state_dict(), path + '_actor_dict.pth')

    def load_model(self, path):
        self.actor_net.load_state_dict(torch.load(path))