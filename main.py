import gymnasium as gym
from agents import SAC, TD3
import time
import argparse
from utils import Logger, set_seed


def parse_args():
    parser = argparse.ArgumentParser()

    # Common Hyperparams
    parser.add_argument('--env', type=str, default='HalfCheetah-v4')
    parser.add_argument('--agent', type=str, required=True)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--num_steps', type=int, default=int(1e6))
    parser.add_argument('--num_eval_epochs', type=int, default=10)
    parser.add_argument('--eval_interval', type=int, default=int(5e3))
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--update_after', type=int, default=int(1e3))
    parser.add_argument('--start_steps', type=int, default=int(10e3))
    parser.add_argument('--buffer_size', type=int, default=int(1e6))
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=5e-3)
    parser.add_argument('--lr_actor', type=float, default=1e-3)
    parser.add_argument('--lr_critic', type=float, default=1e-3)

    # Tools
    parser.add_argument('--play', action='store_true', default=False,
                        help='load and render trained model')
    parser.add_argument('--load_actor_path', type=str, default='')
    parser.add_argument('--plot_computational_graph',
                        type=bool, default=True)
    parser.add_argument('--logger', type=str,
                        choices=['wandb', 'mlflow', 'None'], default='None')
    parser.add_argument('--exp_name', type=str, default='RL-Base')

    # SAC
    parser.add_argument('--num_grad_steps', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--lr_value', type=float, default=1e-3)

    # TD3
    parser.add_argument('--normal_std', type=float, default=0.1)
    parser.add_argument('--noise_clip', type=float, default=0.5)
    parser.add_argument('--update_actor_freq', type=int, default=2)

    return parser.parse_args()


def train(env, eval_env, args, logger):
    if args.agent == 'SAC':
        agent = SAC.SAC(env, eval_env, args, logger)
    elif args.agent == 'TD3':
        agent = TD3.TD3(env, eval_env, args, logger)
    else:
        raise ValueError(f'{args.agent} not supported')
    agent.train()
    agent.save_model(f'models/SAC_{args.env}_{args.seed}_{time.time()}')
    logger.end()

def play(env, eval_env, args):
    if args.agent == 'SAC':
        agent = SAC.SAC(env, eval_env, args)
    elif args.agent == 'TD3':
        agent = TD3.TD3(env, eval_env, args)
    else:
        raise ValueError(f'{args.agent} not supported')

    random_selection = False
    if args.load_actor_path == '':
        import warnings
        warnings.warn("No actor model path added.\
                      Playing random actions.\
                      Path can be added using --load_actor_path [path]")
        random_selection = True
    else:
        agent.load_model(args.load_actor_path)
    obs, _ = env.reset()
    env.render()
    while True:
        if random_selection:
            action = env.action_space.sample()
        else:
            action = agent.select_action(
                obs, stochastic=False).detach().cpu().numpy()
        next_obs, rew, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            next_obs, _ = env.reset()
        obs = next_obs


if __name__ == "__main__":
    args = parse_args()
    logger = Logger(args.logger)
    logger.init(args.exp_name, args)
    if args.play:
        eval_env = gym.make(args.env, render_mode='human')
        set_seed(eval_env, args.seed)
        play(eval_env, eval_env, args)
    else:
        env = gym.make(args.env)
        eval_env = gym.make(args.env)
        set_seed(env, args.seed)
        set_seed(eval_env, args.seed + 100)
        try:
            train(env, eval_env, args, logger)
        except KeyboardInterrupt:
            logger.end(status='KILLED')
        except Exception as e:
            print('-'*30)
            print(e)
            print('-'*30)
            logger.end(status='FAILED')