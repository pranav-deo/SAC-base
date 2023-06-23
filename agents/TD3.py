import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from ReplayBuffer import ReplayBuffer
import copy
import numpy as np
from mlflow import log_metric
import tqdm
# from utils import do_torchviz_plots

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
    def __init__(self, obs_dim, act_dim, max_action, hidden_dim=256) -> None:
        super().__init__()
        self.max_action = max_action
        self.actor = BaseNetwork(obs_dim, act_dim, hidden_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.max_action * self.tanh(self.actor(x))

class DoubleCriticNetwork(nn.Module):
    def __init__(self, obs_act_dim, hidden_dim=256) -> None:
        super().__init__()
        self.q1 = BaseNetwork(obs_act_dim, 1, hidden_dim)
        self.q2 = BaseNetwork(obs_act_dim, 1, hidden_dim)

    def forward(self, obs, act):
        inp = torch.concat((obs, act), dim=-1)
        return self.q1(inp), self.q2(inp)


class TD3:
    def __init__(self, env, eval_env, args):

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        self.max_action = env.action_space.high[0]

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.actor_net = ActorNetwork(obs_dim, act_dim, self.max_action).to(self.device)
        self.critic_net = DoubleCriticNetwork(
            obs_dim + act_dim).to(self.device)
        self.target_actor = copy.deepcopy(self.actor_net).to(self.device)
        self.target_critic = copy.deepcopy(self.critic_net).to(self.device)

        for p in self.target_actor.parameters():
            p.requires_grad = False

        for p in self.target_critic.parameters():
            p.requires_grad = False

        self.buffer = ReplayBuffer(
            args.buffer_size, obs_dim, act_dim, self.device, args.seed)
        self.actor_optim = torch.optim.Adam(
            self.actor_net.parameters(), lr=args.lr_actor)
        self.critic_optim = torch.optim.Adam(
            self.critic_net.parameters(), lr=args.lr_critic)

        self.env = env
        self.eval_env = eval_env
        self.args = args

    def select_action(self, obs, stochastic=True, 
                      do_clip_noise=False, target_actor=False):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        mean = self.actor_net(obs) if not target_actor else self.target_actor(obs)
        act_noise = Normal(0., self.args.normal_std*self.max_action).sample(mean.shape)

        if do_clip_noise:
            act_noise = torch.clamp(act_noise, 
                                    -self.args.noise_clip, self.args.noise_clip)
        act = mean + act_noise.to(self.device) if stochastic else mean
        return act

    def update_actor(self, obs):
        act = self.select_action(obs, stochastic=False)
        # with torch.no_grad():
        q_1 = self.critic_net(obs, act)[0]
        loss = -q_1.mean()

        # if self.args.plot_computational_graph:
        #     do_torchviz_plots(loss, self.actor_net, self.critic_net,
        #                       self.target_value, self.value_net, 'actor_update')

        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()
        return loss

    def update_critic(self, obs, next_obs, act, rew, not_dones):
        with torch.no_grad():
            next_act = self.select_action(next_obs, do_clip_noise=True, target_actor=True)
            next_act = torch.clamp(next_act, -self.max_action, self.max_action)
            target_q_min = torch.minimum(*self.target_critic(next_obs, next_act))
            target_q = rew + self.args.discount * not_dones * target_q_min

        q1, q2 = self.critic_net(obs, act)

        loss = (F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)).mean()

        # if self.args.plot_computational_graph:
        #     do_torchviz_plots(loss, self.actor_net, self.critic_net,
        #                       self.target_value, self.value_net, 'critic_update')

        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
        return loss, (q1.detach().mean().item(), q2.detach().mean().item())

    def soft_update(self):
        with torch.no_grad():
            for p, tp in zip(self.actor_net.parameters(), self.target_actor.parameters()):
                tp.data.copy_((1.0-self.args.tau) *
                              tp.data + self.args.tau*p.data)
                
            for p, tp in zip(self.critic_net.parameters(), self.target_critic.parameters()):
                tp.data.copy_((1.0-self.args.tau) *
                              tp.data + self.args.tau*p.data)

    def train(self):
        obs, _ = self.env.reset()
        ep_rew = 0
        step = 0
        with tqdm.tqdm(total=self.args.num_steps) as pbar:
            while step < self.args.num_steps:
                if step > self.args.start_steps:
                    act = self.select_action(obs).cpu().detach().numpy()
                else:
                    act = self.env.action_space.sample()

                act = np.clip(act, -self.max_action, self.max_action)
                next_obs, rew, terminated, truncated, info = self.env.step(act)
                ep_rew += rew
                not_done = 1.0 - terminated
                self.buffer.add_step(obs, next_obs, act, rew, not_done)
                if (terminated or truncated):
                    log_metric('train_reward', ep_rew, step)
                    next_obs, _ = self.env.reset()
                    ep_rew = 0
                obs = next_obs
                if step > self.args.update_after:
                    b_obs, b_next_obs, b_act, b_rew, b_not_dones = self.buffer.sample(
                        self.args.batch_size)
                    critic_loss, q = self.update_critic(b_obs, b_next_obs, b_act, b_rew, b_not_dones)
                    if step % self.args.update_actor_freq == 0:
                        actor_loss = self.update_actor(b_obs)
                        self.soft_update()
                        log_metric('actor_loss', actor_loss.item(), step)

                    log_metric('q1', q[0], step)
                    log_metric('q2', q[1], step)
                    log_metric('critic_loss', critic_loss.item(), step)

                if step % self.args.eval_interval == 0:
                    eval_rew = self.eval()
                    log_metric('eval_reward', eval_rew, step)
                step += 1
                pbar.update(1)

    def eval(self):
        epoch_rew = []
        for _ in range(self.args.num_eval_epochs):
            obs, done = self.eval_env.reset()[0], False
            ep_reward = 0
            while not done:
                act = self.select_action(obs, stochastic=False, 
                                         target_actor=True).cpu().detach().numpy()
                # act = np.clip(act, -self.max_action, self.max_action)
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
