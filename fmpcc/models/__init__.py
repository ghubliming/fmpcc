from .diffusion import Diffusion
from .flow import FlowMatching
from .meanfm import AnalyticMeanFM, CIMeanFM
from .unet import TemporalUnet
from .vision import VisualEncoder

MODELS = ('diffusion', 'fm', 'analytic_meanfm', 'ci_meanfm')


def build(model, spec, observation_dim, action_dim, horizon, goal_dim=0, visual=False, train_steps=100000):
    """The objective `model` with its network; spec is the model section of the training configuration."""
    encoder = VisualEncoder() if visual else None
    network = TemporalUnet(
        observation_dim + action_dim, dim=spec.get('dim', 32), dim_mults=tuple(spec.get('dim_mults', (1, 2, 4, 8))),
        two_time=model in ('analytic_meanfm', 'ci_meanfm'), cond_dim=encoder.feature_dim if encoder is not None else 0)
    common = dict(network=network, horizon=horizon, observation_dim=observation_dim, action_dim=action_dim,
                  goal_dim=goal_dim, encoder=encoder)
    if model == 'diffusion':
        return Diffusion(steps=spec['steps'], action_weight=spec['action_weight'], **common)
    if model == 'fm':
        return FlowMatching(time_beta=spec['time_beta'], action_weight=spec['action_weight'], **common)
    if model == 'analytic_meanfm':
        return AnalyticMeanFM(
            time_logit_normal=spec['time_logit_normal'], fm_share=spec['fm_share'], adaptive_p=spec['adaptive_p'],
            adaptive_eps=spec['adaptive_eps'], **common)
    if model == 'ci_meanfm':
        return CIMeanFM(
            time_logit_normal=spec['time_logit_normal'], fm_share=spec['fm_share'], adaptive_eps=spec['adaptive_eps'],
            alpha_end=spec['alpha_end'], alpha_gamma=spec['alpha_gamma'], alpha_clamp=spec['alpha_clamp'],
            alpha_end_step=train_steps, target_clamp=spec['target_clamp'], **common)
    raise KeyError(f'unknown model {model!r}, expected one of {MODELS}')
