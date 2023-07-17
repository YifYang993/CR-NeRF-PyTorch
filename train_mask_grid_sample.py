import os
import wandb
from numpy.lib.utils import who
from opt import get_opts
import torch
from collections import defaultdict

from torch.utils.data import DataLoader
from datasets import dataset_dict

from math import sqrt

# models
from models.nerf import *
from models.rendering import *
from models.networks import E_attr, implicit_mask

# optimizer, scheduler, visualization
from utils import *

# losses
from losses import loss_dict

# metrics
from metrics import *

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer


from datasets import global_val

import random

class NeRFSystem(LightningModule):
    def __init__(self, hparams_):
        super().__init__()
        self.hparams_ = hparams_
        self.loss = loss_dict['hanerf'](hparams_, coef=1)

        self.models_to_train = []
        self.embedding_xyz = PosEmbedding(hparams_.N_emb_xyz-1, hparams_.N_emb_xyz)
        self.embedding_dir = PosEmbedding(hparams_.N_emb_dir-1, hparams_.N_emb_dir)
        self.embedding_uv = PosEmbedding(10-1, 10)

        self.embeddings = {'xyz': self.embedding_xyz,
                           'dir': self.embedding_dir}


        self.nerf_coarse = NeRF('coarse',
                                in_channels_xyz=6*hparams_.N_emb_xyz+3,
                                in_channels_dir=6*hparams_.N_emb_dir+3)
        self.models = {'coarse': self.nerf_coarse}

        if hparams_.N_importance > 0:
            self.nerf_fine = NeRF('fine',
                                  in_channels_xyz=6*hparams_.N_emb_xyz+3,
                                  in_channels_dir=6*hparams_.N_emb_dir+3,
                                  )

            self.models['fine'] = self.nerf_fine
        self.models_to_train += [self.models]


        
    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def forward(self, rays, ts, whole_img, W, H, rgb_idx, uv_sample, test_blender,val_mode=False):
        results = defaultdict(list)
        kwargs ={}
        

        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        chunk_temp=self.hparams_.chunk
        if val_mode==True:
            chunk_temp=2048
        for i in range(0, B, chunk_temp):
            rendered_ray_chunks = \
                render_rays(self.models,
                            self.embeddings,
                            rays[i:i+chunk_temp],
                            ts[i:i+chunk_temp],
                            self.hparams_.N_samples,
                            self.hparams_.use_disp,
                            self.hparams_.perturb,
                            self.hparams_.noise_std,
                            self.hparams_.N_importance,
                            chunk_temp, # chunk size is effective in val mode
                            self.train_dataset.white_back,
                            **kwargs)

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)

       

        return results

    def setup(self, stage):
        dataset = dataset_dict[self.hparams_.dataset_name]
        kwargs = {'root_dir': self.hparams_.root_dir}
        if self.hparams_.dataset_name == 'phototourism':
            kwargs['img_downscale'] = self.hparams_.img_downscale
            kwargs['val_num'] = self.hparams_.num_gpus
            kwargs['use_cache'] = self.hparams_.use_cache
            kwargs['batch_size'] = self.hparams_.batch_size
            kwargs['scale_anneal'] = self.hparams_.scale_anneal
            kwargs['min_scale'] = self.hparams_.min_scale
        elif self.hparams_.dataset_name == 'blender':
            kwargs['img_wh'] = tuple(self.hparams_.img_wh)
            kwargs['perturbation'] = self.hparams_.data_perturb
            kwargs['batch_size'] = self.hparams_.batch_size
            kwargs['scale_anneal'] = self.hparams_.scale_anneal
            kwargs['min_scale'] = self.hparams_.min_scale
            if self.hparams_.useNeuralRenderer:
                kwargs['NeuralRenderer_downsampleto'] = (self.hparams_.NRDS, self.hparams_.NRDS)
        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams_, self.models_to_train)
        scheduler = get_scheduler(self.hparams_, self.optimizer)
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # self.hparams_.batch_size a time
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)
    
    def training_step(self, batch, batch_nb):
        rays, ts = batch['rays'].squeeze(), batch['ts'].squeeze()
        rgbs = batch['rgbs'].squeeze()
        uv_sample = batch['uv_sample'].squeeze()
        if self.hparams_.encode_a or self.hparams_.use_mask:
            whole_img = batch['whole_img']
            rgb_idx = batch['rgb_idx']
        else:
            whole_img = None
            rgb_idx = None
        H = int(sqrt(rgbs.size(0)))
        W = int(sqrt(rgbs.size(0)))

        test_blender = False
        results = self(rays, ts, whole_img, W, H, rgb_idx, uv_sample, test_blender)
        loss_d, AnnealingWeight = self.loss(results, rgbs, self.hparams_, self.global_step)
        loss = sum(l for l in loss_d.values())

        with torch.no_grad():
            typ = 'fine' if 'rgb_fine' in results else 'coarse'
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)

        self.log('lr', get_learning_rate(self.optimizer))
        self.log('train/loss', loss)
        self.log('train/AnnealingWeight', AnnealingWeight)
        self.log('train/min_scale_cur', batch['min_scale_cur'])
        for k, v in loss_d.items():
            self.log(f'train/{k}', v)
        self.log('train/psnr', psnr_)

        if (self.global_step + 1) % 5000 == 0 or (self.global_step + 1)==2:
            img = results[f'rgb_{typ}'].detach().view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            img_gt = rgbs.detach().view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            depth = visualize_depth(results[f'depth_{typ}'].detach().view(H, W)) # (3, H, W)
            if self.hparams_.use_mask:
                mask = results['out_mask'].detach().view(H, W, 1).permute(2, 0, 1).repeat(3, 1, 1).cpu() # (3, H, W)
                if 'rgb_fine_random' in results:
                    img_random = results[f'rgb_fine_random'].detach().view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
                    stack = [img_gt, img, depth, img_random, mask]# (4, 3, H, W)
                    self.logger.experiment.log({    "samples": [wandb.Image(img) for img in stack]})
                else:
                    stack = [img_gt, img, depth, mask] # (3, 3, H, W)
                    self.logger.experiment.log({    "samples": [wandb.Image(img) for img in stack]})
            elif 'rgb_fine_random' in results:
                img_random = results[f'rgb_fine_random'].detach().view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
                stack = [img_gt, img, depth, img_random] # (4, 3, H, W)
                self.logger.experiment.log({    "samples": [wandb.Image(img) for img in stack]})
            else:
                stack = [img_gt, img, depth] # (3, 3, H, W)
                self.logger.experiment.log({    "samples": [wandb.Image(img) for img in stack]})
        
        return loss

    def validation_step(self, batch, batch_nb):
        if self.current_epoch!=self.hparams_.num_epochs -1:
            return 0
        rays, ts = batch['rays'].squeeze(), batch['ts'].squeeze()
        rgbs =  batch['rgbs'].squeeze()
        if self.hparams_.dataset_name == 'phototourism':
            uv_sample = batch['uv_sample'].squeeze()
            WH = batch['img_wh']
            W, H = WH[0, 0].item(), WH[0, 1].item()
        else:
            W, H = self.hparams_.img_wh
            uv_sample = None

        if self.hparams_.encode_a or self.hparams_.use_mask :#or self.hparams_.deocclusion:
            if self.hparams_.dataset_name == 'phototourism':
                whole_img = batch['whole_img']
            else:
                whole_img = rgbs.view(1, H, W, 3).permute(0, 3, 1, 2) * 2 - 1
            rgb_idx = batch['rgb_idx']
        else:
            whole_img = None
            rgb_idx = None

        test_blender = (self.hparams_.dataset_name == 'blender')
        results = self(rays, ts, whole_img, W, H, rgb_idx, uv_sample, test_blender, val_mode=True)
        loss_d, AnnealingWeight = self.loss(results, rgbs, self.hparams, self.global_step)
        loss = sum(l for l in loss_d.values())
        log = {'val_loss': loss}
        for k, v in loss_d.items():
            log[k] = v
        
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        img = results[f'rgb_{typ}'].view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
        img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
        if batch_nb == 0:
            depth = visualize_depth(results[f'depth_{typ}'].view(H, W)) # (3, H, W)
            if self.hparams_.use_mask:
                mask = results['out_mask'].detach().view(H, W, 1).permute(2, 0, 1).repeat(3, 1, 1).cpu() # (3, H, W)
                if 'rgb_fine_random' in results:
                    img_random = results[f'rgb_fine_random'].detach().view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
                    stack = [img_gt, img, depth, img_random, mask] # (5, 3, H, W)
                    self.logger.experiment.log({    "samples": [wandb.Image(img) for img in stack]})
                else:
                    stack = torch.stack[img_gt, img, depth, mask] # (4, 3, H, W)
                    self.logger.experiment.log({    "samples": [wandb.Image(img) for img in stack]})
            elif 'rgb_fine_random' in results:
                img_random = results[f'rgb_fine_random'].detach().view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
                stack = [img_gt, img, depth, img_random] # (4, 3, H, W)
                self.logger.experiment.log({    "samples": [wandb.Image(img) for img in stack]})
            else:
                stack = [img_gt, img, depth] # (3, 3, H, W)
                self.logger.experiment.log({    "samples": [wandb.Image(img) for img in stack]})

        psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
        ssim_ = ssim(img[None,...], img_gt[None,...])
        log['val_psnr'] = psnr_
        log['val_ssim'] = ssim_

        return log

    def validation_epoch_end(self, outputs):
        if self.current_epoch!=self.hparams_.num_epochs -1:
            return 0
        if len(outputs) == 1:
            global_val.current_epoch = self.current_epoch
        else:
            global_val.current_epoch = self.current_epoch + 1
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()
        mean_ssim = torch.stack([x['val_ssim'] for x in outputs]).mean()
        self.log('val/loss', mean_loss)
        self.log('val/psnr', mean_psnr, prog_bar=True)
        self.log('val/ssim', mean_ssim, prog_bar=True)

        if self.hparams_.use_mask:
            self.log('val/c_l', torch.stack([x['c_l'] for x in outputs]).mean())
            self.log('val/f_l', torch.stack([x['f_l'] for x in outputs]).mean())
            self.log('val/r_ms', torch.stack([x['r_ms'] for x in outputs]).mean())
            self.log('val/r_md', torch.stack([x['r_md'] for x in outputs]).mean())


def main(hparams__):

    system = NeRFSystem(hparams__)
    checkpoint_callback = \
        ModelCheckpoint(dirpath=os.path.join(hparams__.save_dir,  ###change3 filepath->dirpath
                                              f'ckpts/{hparams__.exp_name}'), save_last=True) ####replace dirpath


    # ###saving code####

    from pytorch_lightning.loggers import WandbLogger
    wandb_logger = WandbLogger(name=hparams__.exp_name, project=hparams__.proj_name, save_dir=hparams__.wandbsavepath)

    logger1=wandb_logger
    trainer = Trainer(max_epochs=hparams__.num_epochs,
                      checkpoint_callback=checkpoint_callback,
                      resume_from_checkpoint=hparams__.ckpt_path,
                      logger=logger1,
                      weights_summary=None,
                      progress_bar_refresh_rate=hparams__.refresh_every,
                      gpus= hparams__.num_gpus,
                      accelerator='ddp' if hparams__.num_gpus>1 else None,
                      num_sanity_val_steps=-1,
                      benchmark=True,
                      profiler="simple" if hparams__.num_gpus==1 else None)

    trainer.fit(system)

    # trainer = Trainer(max_epochs=hparams__.num_epochs,
    #                   callbacks=[checkpoint_callback], ##change4 checkpoint_callback -> callbacks
    #                   resume_from_checkpoint=hparams__.ckpt_path,
    #                   logger=logger1,
    #                   strategy='ddp',
    #                 #   distributed_backend= 'ddp',
    #                 #   weights_summary=None,     ##change5 zhushi
    #                 #   progress_bar_refresh_rate=hparams__.refresh_every, ##change6 zhushi
    #                   gpus= hparams__.num_gpus,
    #                   accelerator='cuda',
    #                   num_sanity_val_steps=-1,
    #                   benchmark=True,
    #                   profiler="simple" if hparams__.num_gpus==1 else None) #callbacks

    # trainer.fit(system)
    # wandb.finish()
# def main1(hparams):
#     system = NeRFSystem(hparams)
#     checkpoint_callback = \
#         ModelCheckpoint(filepath=os.path.join(hparams.save_dir,
#                                               f'ckpts/{hparams.exp_name}'),
#                         # monitor='val/psnr',
#                         # mode='max',
#                         # save_top_k=1,
#                         save_last=True)

#     logger = TestTubeLogger(save_dir=os.path.join(hparams.save_dir,"logs"),
#                             name=hparams.exp_name,
#                             debug=False,
#                             create_git_tag=False,
#                             log_graph=False)

#     trainer = Trainer(max_epochs=hparams.num_epochs,
#                       checkpoint_callback=checkpoint_callback,
#                       resume_from_checkpoint=hparams.ckpt_path,
#                       logger=logger,
#                       weights_summary=None,
#                       progress_bar_refresh_rate=hparams.refresh_every,
#                       gpus= hparams.num_gpus,
#                       accelerator='ddp' if hparams.num_gpus>1 else None,
#                       num_sanity_val_steps=-1,
#                       benchmark=True,
#                       profiler="simple" if hparams.num_gpus==1 else None)

#     trainer.fit(system)

from pytorch_lightning.utilities.distributed import rank_zero_only
@rank_zero_only
def save_code(hparams):
    import datetime
    import shutil
    from distutils.dir_util import copy_tree
    now = datetime.datetime.now()
    # timestr=str(now.month)+str(now.day)+str(now.hour)+str(now.minute)+str(now.second)
    experiment_dir = os.path.join(hparams.save_dir,'logs',hparams.exp_name,"codes")
    copy_tree('models/', experiment_dir+"/models")
    copy_tree('datasets/', experiment_dir+"/datasets")
    copy_tree('utils/', experiment_dir+"/utils")
    # shutil.copy('datasets/phototourism_mask_grid_sample.py', experiment_dir)
    shutil.copy('train_mask_grid_sample.py', experiment_dir)
    shutil.copy('losses.py', experiment_dir)
    shutil.copy('eval.py',experiment_dir)
    shutil.copy('eval_metric.py', experiment_dir)
    shutil.copy('opt.py', experiment_dir)
    shutil.copy('metrics.py', experiment_dir)
    logstr=str(hparams)
    with open(experiment_dir+"/command.txt",'w') as f:
        f.writelines(logstr)

if __name__ == '__main__':
    hparams = get_opts()
    save_code(hparams)
    main(hparams)