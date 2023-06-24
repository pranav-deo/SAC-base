from torchviz import make_dot
import os

class Logger():
    def __init__(self, logger_name) -> None:
        self.supported_loggers = ['wandb', 'mlflow']
        self.logger_name = logger_name.lower()
        if self.logger_name == 'none':
            import warnings
            warnings.warn(f'''Not using any loggers for current experiment! 
                          Use --logger [logger_name] flag to use logger. 
                          Supported loggers are {self.supported_loggers}\n''')
        
        else:
            if self.logger_name not in self.supported_loggers:
                raise ValueError('Only loggers supported are mlflow and wandb')
            else:
                self.logger = __import__(logger_name)

        self.added_graphs = set()

    def init(self, exp_name, config):
        self.exp_name = exp_name
        if self.logger_name == 'none':
            pass
        elif self.logger_name == 'mlflow':
            self.logger.set_experiment(self.exp_name)
            self.logger.start_run()
            self.logger.log_params(vars(config))
        elif self.logger_name == 'wandb':
            self.logger.init(project=self.exp_name, config=config)

    def log(self, data, step):
        assert isinstance(data, dict), 'Only able to log metrics as a dict'
        if self.logger_name == 'none':
            pass
        elif self.logger_name == 'mlflow':
            self.logger.log_metrics(data, step)
        elif self.logger_name == 'wandb':
            self.logger.log(data, step)

    def end(self, status='FINISHED'):
        if self.logger_name == 'none':
            pass
        elif self.logger_name == 'mlflow':
            self.logger.end_run(status)
        elif self.logger_name == 'wandb':
            pass

    def do_torchviz_plots(self, loss, params, update_name):
        if update_name not in self.added_graphs:
            self.added_graphs.add(update_name)
            os.makedirs(f'torchviz_plots/{self.exp_name}/', exist_ok=True)
            make_dot(loss, params=params, show_attrs=True, 
                show_saved=True).render(f'torchviz_plots/{self.exp_name}/' + update_name, format="png")
            if self.logger_name == 'wandb':
                self.logger.log({update_name: self.logger.Image(f'torchviz_plots/{self.exp_name}/{update_name}.png')})
            else:
                pass

    
def set_seed(env, seed):
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)