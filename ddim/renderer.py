from config import *


def render_uncondition(conf: TrainConfig,
                       model: None,
                       x_T,
                       sampler: Sampler):
    device = x_T.device
    if conf.train_mode == TrainMode.diffusion:
        assert conf.model_type.can_sample()
        return sampler.sample(model=model, noise=x_T)
    else:
        raise NotImplementedError()


def render_condition(
    conf: TrainConfig,
    model: None,
    x_T,
    sampler: Sampler,
    x_start=None,
    cond=None,
):
    if conf.train_mode == TrainMode.diffusion:
        assert conf.model_type.has_autoenc()
        # returns {'cond', 'cond2'}
        if cond is None:
            cond = model.encode(x_start)
        return sampler.sample(model=model,
                              noise=x_T,
                              model_kwargs={'cond': cond})
    else:
        raise NotImplementedError()
