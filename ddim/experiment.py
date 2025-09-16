import copy
import json
import os

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import *
from torch.cuda import amp
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataset import ConcatDataset, TensorDataset
from torchvision.utils import make_grid, save_image

from config import *
from dataset.dataset import *
from utils.dist_utils import *
from utils.lmdb_writer import *
from metrics import *
from renderer import *


class LitModel(pl.LightningModule):
    def __init__(self, conf: TrainConfig):
        super().__init__()
        if conf.seed is not None:
            pl.seed_everything(conf.seed)

        self.save_hyperparameters(conf.as_dict_jsonable())

        self.conf = conf

        self.model = conf.make_model_conf().make_model()
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()

        model_size = 0
        for param in self.model.parameters():
            model_size += param.data.nelement()
        print('Model params: %.2f M' % (model_size / 1024 / 1024))

        self.sampler = conf.make_diffusion_conf().make_sampler()
        
        self.eval_sampler = conf.make_eval_diffusion_conf().make_sampler()

        # this is shared for both model and latent
        self.T_sampler = conf.make_T_sampler()

        # initial variables for consistent sampling
        self.register_buffer(
            'x_T',
            torch.randn(conf.sample_size, 3, conf.img_size, conf.img_size))

        if conf.pretrain is not None:
            print(f'loading pretrain ... {conf.pretrain.name}')
            state = torch.load(conf.pretrain.path, map_location='cpu')
            print('step:', state['global_step'])
            self.load_state_dict(state['state_dict'], strict=False)

    def sample(self, N, device, T=None):
        if T is None:
            sampler = self.eval_sampler
        else:
            sampler = self.conf._make_diffusion_conf(T).make_sampler()

        noise = torch.randn(N,
                            3,
                            self.conf.img_size,
                            self.conf.img_size,
                            device=device)
        pred_img = render_uncondition(
            self.conf,
            self.ema_model,
            noise,
            sampler=sampler
        )
        pred_img = (pred_img + 1) / 2
        return pred_img

    def render(self, noise, cond=None, T=None):
        if T is None:
            sampler = self.eval_sampler
        else:
            sampler = self.conf._make_diffusion_conf(T).make_sampler()

        if cond is not None:
            pred_img = render_condition(self.conf,
                                        self.ema_model,
                                        noise,
                                        sampler=sampler,
                                        cond=cond)
        else:
            pred_img = render_uncondition(self.conf,
                                          self.ema_model,
                                          noise,
                                          sampler=sampler)
        pred_img = (pred_img + 1) / 2
        return pred_img


    def forward(self, noise=None, x_start=None, ema_model: bool = False):
        with amp.autocast(False):
            if ema_model:
                model = self.ema_model
            else:
                model = self.model
            gen = self.eval_sampler.sample(model=model,
                                           noise=noise,
                                           x_start=x_start)
            return gen

    def setup(self, stage=None) -> None:
        """
        make datasets & seeding each worker separately
        """
        ##############################################
        # NEED TO SET THE SEED SEPARATELY HERE
        if self.conf.seed is not None:
            seed = self.conf.seed * get_world_size() + self.global_rank + 1234
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            print('local seed:', seed)
        ##############################################

        self.train_data = self.conf.make_dataset()
        print('train data:', len(self.train_data))
        self.val_data = self.train_data
        print('val data:', len(self.val_data))

    def _train_dataloader(self, drop_last=True):
        """
        really make the dataloader
        """
        # make sure to use the fraction of batch size
        # the batch size is global!
        conf = self.conf.clone()
        conf.batch_size = self.batch_size

        dataloader = conf.make_loader(self.train_data,
                                      shuffle=True,
                                      drop_last=drop_last)

        return dataloader

    def train_dataloader(self):
        print('on train dataloader start ...')
        return self._train_dataloader()

    @property
    def batch_size(self):
        """
        local batch size for each worker
        """
        ws = get_world_size()
        assert self.conf.batch_size % ws == 0
        return self.conf.batch_size // ws

    @property
    def num_samples(self):
        """
        (global) batch size * iterations
        """
        # batch size here is global!
        # global_step already takes into account the accum batches
        return self.global_step * self.conf.batch_size_effective

    def is_last_accum(self, batch_idx):
        """
        is it the last gradient accumulation loop? 
        used with gradient_accum > 1 and to see if the optimizer will perform "step" in this iteration or not
        """
        return (batch_idx + 1) % self.conf.accum_batches == 0

    def training_step(self, batch, batch_idx):
        """
        given an input, calculate the loss function
        no optimization at this stage.
        """
        with amp.autocast(False):
            # batch size here is local!
            # forward
            imgs, idxs = batch['img'], batch['index']
            # print(f'(rank {self.global_rank}) batch size:', len(imgs))
            x_start = imgs

            if self.conf.train_mode == TrainMode.diffusion:
                """
                main training mode!!!
                """
                # with numpy seed we have the problem that the sample t's are related!
                t, weight = self.T_sampler.sample(len(x_start), x_start.device)
                losses = self.sampler.training_losses(model=self.model,
                                                      x_start=x_start,
                                                      t=t)
            else:
                raise NotImplementedError()
            
            """
            # For Classifier-Free training
            imgs, labels = batch
            x_start = imgs
            if self.conf.train_mode == TrainMode.diffusion:
            # with numpy seed we have the problem that the sample t's are related!
            t, weight = self.T_sampler.sample(len(x_start), x_start.device)
            losses = self.sampler.training_losses(model=self.model,
                                                    x_start=x_start,
                                                    t=t,
                                                    labels=labels
                                                    )
            """        

            loss = losses['loss'].mean()
            # divide by accum batches to make the accumulated gradient exact!
            for key in ['loss', 'vae', 'mmd', 'chamfer', 'arg_cnt']:
                if key in losses:
                    losses[key] = self.all_gather(losses[key]).mean()

            if self.global_rank == 0:
                self.logger.experiment.add_scalar('loss', losses['loss'],
                                                  self.num_samples)
                for key in ['vae', 'mmd', 'chamfer', 'arg_cnt']:
                    if key in losses:
                        self.logger.experiment.add_scalar(
                            f'loss/{key}', losses[key], self.num_samples)

        return {'loss': loss}

    def on_train_batch_end(self, outputs, batch, batch_idx: int,
                           dataloader_idx: int) -> None:
        """
        after each training step ...
        """
        if self.is_last_accum(batch_idx):
            # only apply ema on the last gradient accumulation step,
            # if it is the iteration that has optimizer.step()
            ema(self.model, self.ema_model, self.conf.ema_decay)

            # logging
            imgs = batch['img']
            self.log_sample(x_start=imgs)
            self.evaluate_scores()

    def on_before_optimizer_step(self, optimizer: Optimizer,
                                 optimizer_idx: int) -> None:
        # fix the fp16 + clip grad norm problem with pytorch lightinng
        # this is the currently correct way to do it
        if self.conf.grad_clip > 0:
            # from trainer.params_grads import grads_norm, iter_opt_params
            params = [
                p for group in optimizer.param_groups for p in group['params']
            ]
            # print('before:', grads_norm(iter_opt_params(optimizer)))
            torch.nn.utils.clip_grad_norm_(params,
                                           max_norm=self.conf.grad_clip)
            # print('after:', grads_norm(iter_opt_params(optimizer)))

    def log_sample(self, x_start):
        """
        put images to the tensorboard
        """
        def do(model,
               postfix):
            model.eval()
            with torch.no_grad():
                all_x_T = self.split_tensor(self.x_T)
                batch_size = min(len(all_x_T), self.conf.batch_size_eval)
                # allow for superlarge models
                loader = DataLoader(all_x_T, batch_size=batch_size)

                Gen = []
                for x_T in loader:
                    gen = self.eval_sampler.sample(model=model,
                                                    noise=x_T)
                    Gen.append(gen)

                gen = torch.cat(Gen)
                gen = self.all_gather(gen)
                if gen.dim() == 5:
                    # (n, c, h, w)
                    gen = gen.flatten(0, 1)

                if self.global_rank == 0:
                    # save samples to the tensorboard
                    grid = (make_grid(gen) + 1) / 2
                    sample_dir = os.path.join(self.conf.logdir,
                                              f'sample{postfix}')
                    if not os.path.exists(sample_dir):
                        os.makedirs(sample_dir)
                    path = os.path.join(sample_dir,
                                        '%d.png' % self.num_samples)
                    save_image(grid, path)
                    self.logger.experiment.add_image(f'sample{postfix}', grid,
                                                     self.num_samples)
            model.train()

        if self.conf.sample_every_samples > 0 and is_time(
                self.num_samples, self.conf.sample_every_samples,
                self.conf.batch_size_effective):
            
                do(self.model, '')
                do(self.ema_model, '_ema')

    def evaluate_scores(self):
        """
        evaluate FID and other scores during training (put to the tensorboard)
        For, FID. It is a fast version with 5k images (gold standard is 50k).
        Don't use its results in the paper!
        """
        def fid(model, postfix):
            score = evaluate_fid(self.eval_sampler,
                                 model,
                                 self.conf,
                                 device=self.device,
                                 train_data=self.train_data,
                                 val_data=self.val_data)
            if self.global_rank == 0:
                self.logger.experiment.add_scalar(f'FID{postfix}', score,
                                                  self.num_samples)
                if not os.path.exists(self.conf.logdir):
                    os.makedirs(self.conf.logdir)
                with open(os.path.join(self.conf.logdir, 'eval.txt'),
                          'a') as f:
                    metrics = {
                        f'FID{postfix}': score,
                        'num_samples': self.num_samples,
                    }
                    f.write(json.dumps(metrics) + "\n")
        
        def lpips(model, postfix):
            if self.conf.model_type.has_autoenc(
            ) and self.conf.train_mode.is_autoenc():
                # {'lpips', 'ssim', 'mse'}
                score = evaluate_lpips(self.eval_sampler,
                                       model,
                                       self.conf,
                                       device=self.device,
                                       val_data=self.val_data,
                                       latent_sampler=self.eval_latent_sampler)

                if self.global_rank == 0:
                    for key, val in score.items():
                        self.logger.experiment.add_scalar(
                            f'{key}{postfix}', val, self.num_samples)             
                    
        if self.conf.eval_every_samples > 0 and self.num_samples > 0 and is_time(
                self.num_samples, self.conf.eval_every_samples,
                self.conf.batch_size_effective):
            print(f'eval fid @ {self.num_samples}')
            lpips(self.model, '')
            fid(self.model, '')

        if self.conf.eval_ema_every_samples > 0 and self.num_samples > 0 and is_time(
                self.num_samples, self.conf.eval_ema_every_samples,
                self.conf.batch_size_effective):
            print(f'eval fid ema @ {self.num_samples}')
            fid(self.ema_model, '_ema')
            # it's too slow
            # lpips(self.ema_model, '_ema')

    def configure_optimizers(self):
        out = {}
        if self.conf.optimizer == OptimizerType.adam:
            optim = torch.optim.Adam(self.model.parameters(),
                                     lr=self.conf.lr,
                                     weight_decay=self.conf.weight_decay)
        elif self.conf.optimizer == OptimizerType.adamw:
            optim = torch.optim.AdamW(self.model.parameters(),
                                      lr=self.conf.lr,
                                      weight_decay=self.conf.weight_decay)
        else:
            raise NotImplementedError()
        out['optimizer'] = optim
        if self.conf.warmup > 0:
            sched = torch.optim.lr_scheduler.LambdaLR(optim,
                                                      lr_lambda=WarmupLR(
                                                          self.conf.warmup))
            out['lr_scheduler'] = {
                'scheduler': sched,
                'interval': 'step',
            }
        return out

    def split_tensor(self, x):
        """
        extract the tensor for a corresponding "worker" in the batch dimension

        Args:
            x: (n, c)

        Returns: x: (n_local, c)
        """
        n = len(x)
        rank = self.global_rank
        world_size = get_world_size()
        # print(f'rank: {rank}/{world_size}')
        per_rank = n // world_size
        return x[rank * per_rank:(rank + 1) * per_rank]

    def test_step(self, batch, *args, **kwargs):
        """
        for the "eval" mode. 
        We first select what to do according to the "conf.eval_programs". 
        test_step will only run for "one iteration" (it's a hack!).
        
        We just want the multi-gpu support. 
        """
        # make sure you seed each worker differently!
        self.setup()

        # it will run only one step!
        print('global step:', self.global_step)

        # evals those "fidXX"
        """
        "fid<T>" = unconditional generation (conf.train_mode = diffusion).
            Note:   Diff. autoenc will still receive real images in this mode.
        """
        for each in self.conf.eval_programs:
            if each.startswith('fid'):
                # evalT
                _, T = each.split('fid')
                T = int(T)
                print(f'evaluating FID T = {T}...')

                self.train_dataloader()
                sampler = self.conf._make_diffusion_conf(T=T).make_sampler()

                conf = self.conf.clone()
                conf.eval_num_images = 10000
                score = evaluate_fid(
                    sampler,
                    self.ema_model,
                    conf,
                    device=self.device,
                    train_data=self.train_data,
                    val_data=self.val_data,
                    remove_cache=False
                )
                
                self.log(f'fid_ema_T{T}', score)


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(target_dict[key].data * decay +
                                    source_dict[key].data * (1 - decay))


class WarmupLR:
    def __init__(self, warmup) -> None:
        self.warmup = warmup

    def __call__(self, step):
        return min(step, self.warmup) / self.warmup


def is_time(num_samples, every, step_size):
    closest = (num_samples // every) * every
    return num_samples - closest < step_size


def train(conf: TrainConfig, gpus, nodes=1, mode: str = 'train'):
    print('conf:', conf.name)
    # assert not (conf.fp16 and conf.grad_clip > 0
    #             ), 'pytorch lightning has bug with amp + gradient clipping'
    model = LitModel(conf)

    if not os.path.exists(conf.logdir):
        os.makedirs(conf.logdir)
    checkpoint = ModelCheckpoint(dirpath=f'{conf.logdir}',
                                 save_last=True,
                                 # TODO original save_top_k=1
                                 save_top_k=-1,
                                 # TODO every_n_train_steps=conf.save_every_samples // conf.batch_size_effective
                                 every_n_epochs=50)
    checkpoint_path = f'{conf.logdir}/last.ckpt'
    print('ckpt path:', checkpoint_path)
    if os.path.exists(checkpoint_path):
        resume = checkpoint_path
        print('resume!')
    else:
        if conf.continue_from is not None:
            # continue from a checkpoint
            resume = conf.continue_from.path
        else:
            resume = None

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=conf.logdir,
                                             name=None,
                                             version='')

    # from pytorch_lightning.

    plugins = []
    if len(gpus) == 1 and nodes == 1:
        accelerator = None
    else:
        accelerator = 'ddp'
        from pytorch_lightning.plugins import DDPPlugin

        # important for working with gradient checkpoint
        plugins.append(DDPPlugin(find_unused_parameters=False))

    trainer = pl.Trainer(
        max_steps=conf.total_samples // conf.batch_size_effective,
        resume_from_checkpoint=resume,
        gpus=gpus,
        num_nodes=nodes,
        accelerator=accelerator,
        precision=16 if conf.fp16 else 32,
        callbacks=[
            checkpoint,
            LearningRateMonitor(),
        ],
        # clip in the model instead
        # gradient_clip_val=conf.grad_clip,
        replace_sampler_ddp=True,
        logger=tb_logger,
        accumulate_grad_batches=conf.accum_batches,
        plugins=plugins,
    )

    if mode == 'train':
        trainer.fit(model)
    elif mode == 'eval':
        # load the latest checkpoint
        # perform lpips
        # dummy loader to allow calling "test_step"
        dummy = DataLoader(TensorDataset(torch.tensor([0.] * conf.batch_size)),
                           batch_size=conf.batch_size)
        eval_path = conf.eval_path or checkpoint_path
        # conf.eval_num_images = 50
        print('loading from:', eval_path)
        state = torch.load(eval_path, map_location='cpu')
        print('step:', state['global_step'])
        model.load_state_dict(state['state_dict'])
        # trainer.fit(model)
        out = trainer.test(model, dataloaders=dummy)
        # first (and only) loader
        out = out[0]
        print(out)

        if get_rank() == 0:
            # save to tensorboard
            for k, v in out.items():
                tb_logger.experiment.add_scalar(
                    k, v, state['global_step'] * conf.batch_size_effective)

            # # save to file
            # # make it a dict of list
            # for k, v in out.items():
            #     out[k] = [v]
            tgt = f'evals/{conf.name}.txt'
            dirname = os.path.dirname(tgt)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            with open(tgt, 'a') as f:
                f.write(json.dumps(out) + "\n")
            # pd.DataFrame(out).to_csv(tgt)
    else:
        raise NotImplementedError()
