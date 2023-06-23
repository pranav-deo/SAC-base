from torchviz import make_dot
import torch.nn.functional as F
from torch.distributions.transforms import Transform
from torch.distributions import constraints
import math

def do_torchviz_plots(loss, actor_net, critic_net, target_value, value_net, update_name):
    make_dot(loss, params=dict(**dict(actor_net.named_parameters(prefix='actor')), 
                                **dict(critic_net.named_parameters(prefix='critic')), 
                                **dict(target_value.named_parameters(prefix='target_value')), 
                                **dict(value_net.named_parameters(prefix='value'))), 
                                show_attrs=True, show_saved=True).render('torchviz_plots/' + update_name, format="svg")

class TanhTransform(Transform):
    r"""
    Transform via the mapping :math:`y = \tanh(x)`.
    It is equivalent to
    ```
    ComposeTransform([AffineTransform(0., 2.), SigmoidTransform(), AffineTransform(-1., 2.)])
    ```
    However this might not be numerically stable, thus it is recommended to use `TanhTransform`
    instead.
    Note that one should use `cache_size=1` when it comes to `NaN/Inf` values.
    """
    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L69-L80
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))