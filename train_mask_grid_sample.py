import os
from pytorch_lightning.utilities.distributed import rank_zero_only
from numpy.lib.utils import who
from models.rendering import render_rays_cross_ray
from opt import get_opts
import torch
import torch.nn as nn
from collections import defaultdict
from einops import rearrange
from torch.utils.data import DataLoader
from datasets import dataset_dict
from math import sqrt
import wandb
from models.nerf_decoder_stylenerf import get_renderer
from models.esrgan import get_esrgan_decoder
from models.nerf import *
from models.rendering import *
from models.linearStyleTransfer import encoder_sameoutputsize
from models.linearStyleTransfer import style_net
from models.lightweight_seg import Context_Guided_Network
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

def get_model(hparams_):
        nerf_coarse = NeRF_sigma(typ='coarse', args=hparams_,
                                        in_channels_xyz=6*hparams_.N_emb_xyz+3,
                                        in_channels_dir=6*hparams_.N_emb_dir+3).cuda()
                         
                                
        if hparams_.encode_c:
            enc_cont = encoder_sameoutputsize(out_channel=hparams_.nerf_out_dim).cuda()
            models_to_train += [enc_cont]
        if hparams_.encode_a:
                decoder=style_net(args=hparams_, residual_blocks=hparams_.decoder_num_res_blocks).cuda()
        else:
            decoder=get_renderer(hparams_).cuda()

        models = {'coarse': nerf_coarse,"decoder":decoder}

        if hparams_.N_importance > 0:
           
            nerf_fine = NeRF_sigma('fine',args=hparams_,
                                        in_channels_xyz=6*hparams_.N_emb_xyz+3,
                                        in_channels_dir=6*hparams_.N_emb_dir+3,
                                        encode_appearance=hparams_.encode_a,
                                        in_channels_a=hparams_.N_a,
                                        encode_random=hparams_.encode_random).cuda()

            models['fine'] = nerf_fine
        return models


class NeRFSystem(LightningModule):
    

    def __init__(self, hparams_):
        super().__init__()
        self.hparams_ = hparams_ 
        self.define_transforms()
        self.loss = loss_dict['crnerf'](hparams_, coef=1)

        self.models_to_train = []
        self.embedding_xyz = PosEmbedding(hparams_.N_emb_xyz-1, hparams_.N_emb_xyz)
        self.embedding_dir = PosEmbedding(hparams_.N_emb_dir-1, hparams_.N_emb_dir)
        self.embedding_uv = PosEmbedding(10-1, 10)

        self.embeddings = {'xyz': self.embedding_xyz,
                           'dir': self.embedding_dir}
        if hparams_.encode_c:
            self.enc_cont = encoder_sameoutputsize(out_channel=hparams_.nerf_out_dim)
            self.models_to_train += [self.enc_cont]
        if hparams_.encode_a:
            self.enc_a = encoder_sameoutputsize(out_channel=hparams_.nerf_out_dim)
            self.models_to_train += [self.enc_a]
            self.embedding_a_list = [None] * hparams_.N_vocab
        self.nerf_coarse = NeRF_sigma(typ='coarse', args=hparams_,
                                        in_channels_xyz=6*hparams_.N_emb_xyz+3,
                                        in_channels_dir=6*hparams_.N_emb_dir+3).cuda()
       
        if hparams_.encode_a:
                self.decoder=style_net(args=hparams_, residual_blocks=hparams_.decoder_num_res_blocks)
        else:
            self.decoder=get_renderer(hparams_).cuda()
        self.models = {'coarse': self.nerf_coarse,"decoder":self.decoder}



        if hparams_.N_importance > 0:
            self.nerf_fine = NeRF_sigma('fine',args=hparams_,
                                        in_channels_xyz=6*hparams_.N_emb_xyz+3,
                                        in_channels_dir=6*hparams_.N_emb_dir+3,
                                        encode_appearance=hparams_.encode_a,
                                        in_channels_a=hparams_.N_a,
                                        encode_random=hparams_.encode_random).cuda()
            self.models['fine'] = self.nerf_fine
        self.models_to_train += [self.models]
        self.best_psnr=0
        self.best_ssim=0
        if hparams_.use_mask:
            self.implicit_mask = Context_Guided_Network(classes= 1, M= 2, N= 2, input_channel=3)
            self.models_to_train += [self.implicit_mask]

        
    def define_transforms(self):
        self.transform = T.ToTensor()
        self.normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def decode(self, results, type, **kwargs):
        if type!='content':
            feature = results['feature_'+type] #torch.Size([699008, 4])
        else:
            feature = results['feature_fine'] #torch.Size([699008, 4])
        lastdim = feature.size(-1)
        feature = rearrange(feature, 'n1 n3 -> n3 n1', n3=lastdim)
        feature = rearrange(feature, ' n3 (h w) ->  1 n3 h w',  h=int(kwargs['H']), w=int(kwargs['W']),n3=lastdim)  ##torch.Size([1, 64, 340, 514])
        if type=="fine_random":
            rgbs_pred = self.models['decoder'](feature, kwargs['a_embedded_random'])
        elif type=="coarse":
            rgbs_pred = self.models['decoder'](feature, kwargs['a_embedded_from_img'])
            rgbs_pred = rearrange(rgbs_pred, ' 1 n1 h w ->  (h w) n1',  h=int(kwargs['H']), w=int(kwargs['W']), n1=3)
        elif type=="fine":
            rgbs_pred = self.models['decoder'](feature, kwargs['a_embedded_from_img'])
            results['rgb_fine_img'] = rgbs_pred
            rgbs_pred = rearrange(rgbs_pred, ' 1 n1 h w ->  (h w) n1',  h=int(kwargs['H']), w=int(kwargs['W']), n1=3)
        elif type=="content":
            rgbs_pred = self.models['decoder'](feature, None, type="content")
            results['rgb_content_img'] = rgbs_pred
            rgbs_pred = None
        results['rgb_'+type] = rgbs_pred
        return results

    def forward(self, rays, ts, whole_img, W, H, rgb, rgb_idx, test_blender, args, hw_whole=None,val_mode=False):
        results = defaultdict(list)
        kwargs ={}
        kwargs['args']=args
        if self.hparams_.encode_a:
            whole_img=(whole_img+1)/2 #convert range of whole_img from [-1,1] to [0,1] to match with sigmoid operation in style_net
            if test_blender:
                kwargs['a_embedded_from_img'] = self.embedding_a_list[0] if self.embedding_a_list[0] != None else self.enc_a(whole_img)
            else:
                kwargs['a_embedded_from_img'] = self.enc_a(whole_img)
            if self.hparams_.encode_random:
                idexlist = [k for k,v in enumerate(self.embedding_a_list) if v != None]
                if len(idexlist) == 0:
                    kwargs['a_embedded_random'] = kwargs['a_embedded_from_img']
                else:
                    kwargs['a_embedded_random'] = self.embedding_a_list[random.choice(idexlist)]
        else: 
            kwargs['a_embedded_from_img']=None
            kwargs['a_embedded_random']=None
        if self.hparams_.use_mask:     
            pred_mask=self.implicit_mask(whole_img)
            pred_mask= nn.functional.interpolate(pred_mask, size=hw_whole, mode='bilinear', align_corners=False) 
            pred_mask=rearrange(pred_mask,'1 n h w -> (h w) n')
            if val_mode==False:
                pred_mask=pred_mask[rgb_idx] ##grid sample strategy, refer to datasets/phototourism_mask_grid_sample for details of choosing rgb_idx
            kwargs['mask_embedded_from_img']= pred_mask
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        chunk_temp=self.hparams_.chunk
        if val_mode==True:
            chunk_temp=2048
        kwargs["H"]=H
        kwargs["W"]=W
        for i in range(0, B, chunk_temp):
            rendered_ray_chunks = \
                render_rays_cross_ray(self.models,
                            self.embeddings,
                            rays[i:i+chunk_temp],
                            ts[i:i+chunk_temp],
                            self.hparams_.N_samples,
                            self.hparams_.use_disp,
                            self.hparams_.perturb,
                            self.hparams_.noise_std,
                            self.hparams_.N_importance,
                            chunk_temp, 
                            self.train_dataset.white_back,
                            **kwargs) #obtaining cross ray feature

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]
            

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        results = self.decode(results,"coarse",**kwargs) ##cross ray appearance transfer and decode coarse rgb map of novel view
        if self.hparams_.N_importance>0:
            results = self.decode(results,"fine",**kwargs) ##cross ray appearance transfer and decode fine rgb map of novel view
        if self.hparams_.encode_c:
            results = self.decode(results,"content",**kwargs) 

        if self.hparams_.use_mask:
            results['out_mask'] = kwargs['mask_embedded_from_img']
        if self.hparams_.encode_a:
            results['a_embedded'] = kwargs['a_embedded_from_img']
            results['whole_img'] = whole_img
            if self.hparams_.encode_random:
                results['a_embedded_random'] = kwargs['a_embedded_random']
                results = self.decode(results,"fine_random",**kwargs) ##decode fine rgb map of novel view from random embedding
                results['a_embedded_random_rec'] = self.enc_a(results['rgb_fine_random']) ##encode random embedding from fine random rgb map, constrain the encoded embedding to be close to the embedding of the randomly choosen embedding->kwargs['a_embedded_random']
                results['rgb_fine_random'] = rearrange(results['rgb_fine_random'], ' 1 n1 h w ->  (h w) n1',  h=int(kwargs['H']), w=int(kwargs['W']), n1=3)
                self.embedding_a_list[ts[0]] = kwargs['a_embedded_from_img'].clone().detach()
        if hparams_.encode_c: 
            results['content_with_a_embed']=self.enc_cont(results['rgb_fine_img'])
            results['content_wo_a_embed']=self.enc_cont(results['rgb_content_img'])

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
        self.train_dataset = dataset(args=self.hparams_,split='train', **kwargs)
        self.val_dataset = dataset(args=self.hparams_,split='val', **kwargs)

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
        rays, ts = batch['rays'].squeeze(), batch['ts'].squeeze()##image id in .tsv all id in list ts are the same
        rgbs = batch['rgbs'].squeeze()
        rgb_idx = batch['rgb_idx'].squeeze()
        w_whole, h_whole=batch['img_wh'].squeeze()
        if self.hparams_.encode_a or self.hparams_.use_mask:
            whole_img = batch['whole_img']
            rgb_idx = batch['rgb_idx']
        else:
            whole_img = None
            rgb_idx = None
        H = int(sqrt(rgbs.size(0)))
        W = int(sqrt(rgbs.size(0)))

        test_blender = False
        results = self(rays, ts, whole_img, H ,W, rgbs, rgb_idx, test_blender, self.hparams_, hw_whole=(int(h_whole),int(w_whole)))
        loss_d, AnnealingWeight = self.loss(results, rgbs, self.hparams_, self.global_step)#, masks_rcnn)
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

        if (self.global_step + 1) % 5000 == 0 or (self.global_step + 1)==2 :

            img = results[f'rgb_{typ}'].view(H, W, 3).permute(2, 0, 1).detach().cpu() # (3, H, W)
            img_gt = rgbs.detach().view(H, W, 3).permute(2, 0, 1).squeeze().cpu() # (3, H, W)

            if self.hparams_.use_mask:
                mask = results['out_mask'].detach().view(H, W, 1).permute(2, 0, 1).repeat(3, 1, 1).cpu() # (3, H, W)
                if 'rgb_fine_random' in results:
                    img_random = results[f'rgb_fine_random'].detach().view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
                    stack = [img_gt, img,  img_random, mask]# (4, 3, H, W)
                    if 'rgb_content_img' in results:
                        img_content=results[f'rgb_content_img'].detach().view(H, W, 3).permute(2, 0, 1).cpu() 
                        stack+=[img_content]
                    self.logger.experiment.log({    "samples": [wandb.Image(img) for img in stack]})



                else:
                    stack = [img_gt, img,  mask] # (3, 3, H, W)
                    if 'rgb_content_img' in results:
                        img_content=results[f'rgb_content_img'].detach().view(H, W, 3).permute(2, 0, 1).cpu() 
                        stack+=[img_content]
                    self.logger.experiment.log({"samples": [wandb.Image(img) for img in stack]})

            elif 'rgb_fine_random' in results:
                img_random = results[f'rgb_fine_random'].detach().view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
                stack = [img_gt, img, img_random] # (4, 3, H, W)
                if 'rgb_content_img' in results:
                        img_content=results[f'rgb_content_img'].detach().view(H, W, 3).permute(2, 0, 1).cpu() 
                        stack+=[img_content]
                self.logger.experiment.log({    "samples": [wandb.Image(img) for img in stack]})
            else:
                stack = [img_gt, img] # (3, 3, H, W)
                if 'rgb_content_img' in results:
                        img_content=results[f'rgb_content_img'].detach().view(H, W, 3).permute(2, 0, 1).cpu() 
                        stack+=[img_content]
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

        if self.hparams_.encode_a or self.hparams_.use_mask: 
            if self.hparams_.dataset_name == 'phototourism':
                whole_img = batch['whole_img']
            else:
                whole_img = rgbs.view(1, H, W, 3).permute(0, 3, 1, 2) * 2 - 1
            rgb_idx = batch['rgb_idx']
        else:
            whole_img = None
            rgb_idx = None
        test_blender = (self.hparams_.dataset_name == 'blender')
        results = self(rays, ts, whole_img, W, H, rgbs, uv_sample, test_blender, self.hparams_, hw_whole=(int(H),int(W)),val_mode=True)
        loss_d, AnnealingWeight = self.loss(results, rgbs, self.hparams_, self.global_step)
        loss = sum(l for l in loss_d.values())
        log = {'val_loss': loss}
        for k, v in loss_d.items():
            log[k] = v
        
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        img = results[f'rgb_{typ}'].view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
        
        img_gt = rgbs.view(H, W, 3).permute(2, 0, 1) # (3, H, W)
        if batch_nb == 0 :
            if self.hparams_.use_mask:
                mask = results['out_mask'].detach().view(H, W, 1).permute(2, 0, 1).repeat(3, 1, 1).cpu() # (3, H, W)
                if 'rgb_fine_random' in results:
                    img_random = results[f'rgb_fine_random'].detach().view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
                    stack = [img_gt.cpu(), img, img_random, mask] # (5, 3, H, W)
                    self.logger.experiment.log({    "samples": [wandb.Image(img) for img in stack]})
                else:
                    stack = [img_gt.cpu(), img, mask]
                    self.logger.experiment.log({    "samples": [wandb.Image(img) for img in stack]})
    

            elif 'rgb_fine_random' in results:
                img_random = results[f'rgb_fine_random'].detach().view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
                stack = [img_gt.cpu(), img,  img_random] # (4, 3, H, W)
                self.logger.experiment.log({
    "samples": [wandb.Image(img) 
    for img in stack]
})
            
            else:
                stack = [img_gt.cpu(), img]# (3, 3, H, W)
                self.logger.experiment.log({    "samples": [wandb.Image(img) for img in stack]})
        psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
        gt_cpu=img_gt[None,...].cpu()
        ssim_ = ssim(img[None,...], gt_cpu)
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
        self.log('epoch',self.current_epoch)

        if self.hparams_.use_mask:
            self.log('val/c_l', torch.stack([x['c_l'] for x in outputs]).mean())
            self.log('val/f_l', torch.stack([x['f_l'] for x in outputs]).mean())

def main(hparams_):

    system = NeRFSystem(hparams_)
    checkpoint_callback = \
        ModelCheckpoint(dirpath=os.path.join(hparams_.save_dir,  
                                              f'ckpts/{hparams_.exp_name}'),
                        save_last=True)  


    if hparams_.testit:
        from pytorch_lightning.loggers import WandbLogger
        wandb_logger = WandbLogger(name=hparams_.exp_name,project=hparams_.proj_name, save_dir=hparams_.wandbsavepath,offline=False)
        logger1=wandb_logger
    else:
        from pytorch_lightning.loggers import WandbLogger
        wandb_logger = WandbLogger(name=hparams_.exp_name,project=hparams_.proj_name, save_dir=hparams_.wandbsavepath,offline=False)
        logger1=wandb_logger

    trainer = Trainer(max_epochs=hparams_.num_epochs,
                      callbacks=checkpoint_callback,
                      resume_from_checkpoint=hparams_.ckpt_path,
                      logger=logger1,
                      strategy="ddp_find_unused_parameters_false",
                      gpus= hparams_.num_gpus,
                      accelerator='cuda',
                      num_sanity_val_steps=-1,
                      benchmark=True,
                      profiler="simple" if hparams_.num_gpus==1 else None) 

    trainer.fit(system)
    wandb.finish()

@rank_zero_only
def save_code(hparams_):
    import datetime
    import shutil
    from distutils.dir_util import copy_tree
    experiment_dir = os.path.join(hparams_.save_dir,'logs',hparams_.exp_name,"codes")
    copy_tree('models/', experiment_dir+"/models")
    copy_tree('datasets/', experiment_dir+"/datasets")
    copy_tree('utils/', experiment_dir+"/utils")
    try:shutil.copy('command/train.sh', experiment_dir)
    except:shutil.copy('train.sh', experiment_dir)
    shutil.copy('train_mask_grid_sample.py', experiment_dir)
    shutil.copy('losses.py', experiment_dir)
    shutil.copy('eval.py',experiment_dir)
    shutil.copy('eval_metric.py', experiment_dir)
    shutil.copy('opt.py', experiment_dir)
    shutil.copy('metrics.py', experiment_dir)
    logstr=str(hparams_)
    with open(experiment_dir+"/command.txt",'w') as f:
        f.writelines(logstr)



if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    hparams_ = get_opts()
    print(hparams_.exp_name)
    if hparams_.testit:
        hparams_.num_epochs=1
    save_code(hparams_)
    main(hparams_)