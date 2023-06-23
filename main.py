import gymnasium as gym
from agents import SAC, TD3
import mlflow
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--agent', type=str, required=True)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--num_steps', type=int, default=int(1e6))
    parser.add_argument('--num_eval_epochs', type=int, default=10)
    parser.add_argument('--eval_interval', type=int, default=int(5e3))
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--update_after', type=int, default=int(1e3))
    parser.add_argument('--start_steps', type=int, default=int(25e3))
    parser.add_argument('--buffer_size', type=int, default=int(1e6))
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=5e-3)
    parser.add_argument('--lr_actor', type=float, default=1e-3)
    parser.add_argument('--lr_critic', type=float, default=1e-3)
    parser.add_argument('--plot_computational_graph', action='store_true', default=False)

    # SAC
    parser.add_argument('--num_grad_steps', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--lr_value', type=float, default=1e-3)

    # TD3
    parser.add_argument('--normal_std', type=float, default=0.1)
    parser.add_argument('--noise_clip', type=float, default=0.5)
    parser.add_argument('--update_actor_freq', type=int, default=2)

    return parser.parse_args()

def set_seed(env:gym.Env, seed:int):
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

def train(env, eval_env, args):
    mlflow.set_experiment(f'{args.agent}_{args.env}')
    with mlflow.start_run():
        mlflow.log_params(vars(args))
        if args.agent == 'SAC':
            agent = SAC.SAC(env, eval_env, args)
        elif args.agent == 'TD3':
            agent = TD3.TD3(env, eval_env, args)
        else:
            raise ValueError(f'{args.agent} not supported')
        agent.train()
        agent.save_model(f'models/SAC_{args.env}_{args.seed}_{time.time()}')

def play(env, eval_env, args):
    if args.agent == 'SAC':
        agent = SAC.SAC(env, eval_env, args)
    elif args.agent == 'TD3':
        agent = TD3.TD3(env, eval_env, args)
    else:
        raise ValueError(f'{args.agent} not supported')
    agent.load_model('models/SAC_Walker2d-v4_100_1686879020.7474368_actor_dict.pth')
    obs, _ = env.reset()
    env.render()
    while True:
        action = agent.select_action(obs, stochastic=False).detach().cpu().numpy()
        # action = env.action_space.sample()
        next_obs, rew, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            next_obs, _ = env.reset()
        obs = next_obs

if __name__=="__main__":
    args = parse_args()
    # env = gym.make(args.env, render_mode='human')
    env = gym.make(args.env)
    eval_env = gym.make(args.env)
    set_seed(env, args.seed)
    set_seed(eval_env, args.seed + 100)
    train(env, eval_env, args)
    # play(env, eval_env, args)