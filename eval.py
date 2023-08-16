import torch
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import imageio
from argparse import ArgumentParser

from models.rendering import render_rays_cross_ray
from models.nerf import *
from models.nerf_decoder_stylenerf import get_renderer
from utils import load_ckpt
import metrics
from einops import rearrange
from datasets import dataset_dict
from datasets.depth_utils import *
from models.linearStyleTransfer import encoder3, encoder_sameoutputsize
from models.networks import E_attr
from math import sqrt
import math
import json
from PIL import Image
from torchvision import transforms as T
from opt import get_opts
from train_mask_grid_sample import get_model
torch.backends.cudnn.benchmark = True


@torch.no_grad()
def batched_inference(models, embeddings,
                      rays, ts, N_samples, N_importance, use_disp,
                      chunk,
                      white_back,
                      **kwargs):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays_cross_ray(models,
                        embeddings,
                        rays[i:i+chunk],
                        ts[i:i+chunk] if ts is not None else None,
                        N_samples,
                        use_disp,
                        0,
                        0,
                        N_importance,
                        chunk,
                        white_back,
                        test_time=True,
                        **kwargs)

        for k, v in rendered_ray_chunks.items():
            results[k] += [v]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results

def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R

if __name__ == "__main__":
    args = get_opts()

    kwargs = {'root_dir': args.root_dir,
              'split': args.split}
    if args.dataset_name == 'blender':
        kwargs['img_wh'] = tuple(args.img_wh)
    else:
        kwargs['img_downscale'] = args.img_downscale
        kwargs['use_cache'] = args.use_cache
    dataset = dataset_dict[args.dataset_name](args=args,**kwargs)
    scene = os.path.basename(args.root_dir.strip('/'))

    embedding_xyz = PosEmbedding(args.N_emb_xyz-1, args.N_emb_xyz)
    embedding_dir = PosEmbedding(args.N_emb_dir-1, args.N_emb_dir)
    embeddings = {'xyz': embedding_xyz, 'dir': embedding_dir}
    if args.encode_a:
        # enc_a
        enc_a = encoder_sameoutputsize(out_channel=args.nerf_out_dim).cuda()
        load_ckpt(enc_a, args.ckpt_path, model_name='enc_a')
        kwargs = {}
        if args.dataset_name == 'blender':
            with open(os.path.join(args.root_dir, f"transforms_train.json"), 'r') as f:
                meta_train = json.load(f)
            frame = meta_train['frames'][0]
            image_path = os.path.join(args.root_dir, f"{frame['file_path']}.png")
            img = Image.open(image_path)
            img = img.resize(args.img_wh, Image.LANCZOS)
            toTensor = T.ToTensor()
            normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            img = toTensor(img) # (4, h, w)
            img = img[:3, :, :]*img[-1:, :, :] + (1-img[-1:, :, :]) # blend A to RGB (3, h, w)
            whole_img = normalize(img).unsqueeze(0).cuda()
            kwargs['a_embedded_from_img'] = enc_a(whole_img)
 

   
    models=get_model(args)
    nerf_coarse=models['coarse']
    nerf_fine=models['fine']
    decoder=models['decoder']
    load_ckpt(nerf_coarse, args.ckpt_path, model_name='nerf_coarse')
    load_ckpt(nerf_fine, args.ckpt_path, model_name='nerf_fine')
    load_ckpt(decoder, args.ckpt_path, model_name='decoder')


    imgs, psnrs, ssims = [], [], []
    dir_name = os.path.join(args.save_dir, f'results/{args.dataset_name}/{args.scene_name}')
    os.makedirs(dir_name, exist_ok=True)

    # enc_a
    # define testing poses and appearance index for phototourism
    kwargs['args']=args
    if args.dataset_name == 'phototourism' and args.split == 'test':
        # define testing camera intrinsics (hard-coded, feel free to change)
        dataset.test_img_w, dataset.test_img_h = args.img_wh
        dataset.test_focal = dataset.test_img_w/2/np.tan(np.pi/6) # fov=60 degrees
        dataset.test_K = np.array([[dataset.test_focal, 0, dataset.test_img_w/2],
                                   [0, dataset.test_focal, dataset.test_img_h/2],
                                   [0,                  0,                    1]])
        if scene == 'brandenburg_gate':
            # select appearance embedding, hard-coded for each scene
            img = Image.open(os.path.join(args.root_dir, 'dense/images',
                                          dataset.image_paths[dataset.img_ids_train[314]])).convert('RGB') # 111 159 178 208 252 314
            img_downscale = 8
            img_w, img_h = img.size
            img_w = img_w//img_downscale
            img_h = img_h//img_downscale
            img = img.resize((img_w, img_h), Image.LANCZOS)
            toTensor = T.ToTensor()
            normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            img = toTensor(img) # (3, h, w)
            whole_img = normalize(img).unsqueeze(0).cuda()
            kwargs['a_embedded_from_img'] = enc_a(whole_img)
            

            dataset.test_appearance_idx = 314 # 85572957_6053497857.jpg
            N_frames = 30*8

            dx1 = np.linspace(-0.25, 0.25, N_frames)
            dx2 = np.linspace(0.25, 0.38, N_frames - N_frames//2)
            dx = np.concatenate((dx1, dx2))

            dy1 = np.linspace(0.05, -0.1, N_frames//2)
            dy2 = np.linspace(-0.1, 0.05, N_frames - N_frames//2)
            dy = np.concatenate((dy1, dy2))

            dz1 = np.linspace(0.1, 0.3, N_frames//2)
            dz2 = np.linspace(0.3, 0.1, N_frames - N_frames//2)
            dz = np.concatenate((dz1, dz2))

            theta_x1 = np.linspace(math.pi/30, 0, N_frames//2)
            theta_x2 = np.linspace(0, math.pi/30, N_frames - N_frames//2)
            theta_x = np.concatenate((theta_x1, theta_x2))

            theta_y = np.linspace(math.pi/10, -math.pi/10, N_frames)

            theta_z = np.linspace(0, 0, N_frames)
            # define poses
            dataset.poses_test = np.tile(dataset.poses_dict[1123], (N_frames, 1, 1))
            for i in range(N_frames):
                dataset.poses_test[i, 0, 3] += dx[i]
                dataset.poses_test[i, 1, 3] += dy[i]
                dataset.poses_test[i, 2, 3] += dz[i]
                dataset.poses_test[i, :, :3] = np.dot(eulerAnglesToRotationMatrix([theta_x[i],theta_y[i],theta_z[i]]), dataset.poses_test[i, :, :3])
        elif scene == 'trevi_fountain':
            # select appearance embedding, hard-coded for each scene
            img = Image.open(os.path.join(args.root_dir, 'dense/images',
                                          dataset.image_paths[dataset.img_ids_train[1548]])).convert('RGB') # 10 1336 1548 296 420 1570 1662
            img_downscale = 8
            img_w, img_h = img.size
            img_w = img_w//img_downscale
            img_h = img_h//img_downscale
            img = img.resize((img_w, img_h), Image.LANCZOS)
            toTensor = T.ToTensor()
            normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            img = toTensor(img) # (3, h, w)
            whole_img = normalize(img).unsqueeze(0).cuda()
            kwargs['a_embedded_from_img'] = enc_a(whole_img)

            dataset.test_appearance_idx = dataset.img_ids_train[1548] # 85572957_6053497857.jpg
            N_frames = 30*8
            dx = np.linspace(-0.8, 0.7, N_frames)   # + right
            dy1 = np.linspace(-0., 0.05, N_frames//2)    # + down
            dy2 = np.linspace(0.05, -0., N_frames - N_frames//2)
            dy = np.concatenate((dy1, dy2))

            dz1 = np.linspace(0.4, 0.1, N_frames//4)  # + foaward
            dz2 = np.linspace(0.1, 0.5, N_frames//4)  # + foaward
            dz3 = np.linspace(0.5, 0.1, N_frames//4)
            dz4 = np.linspace(0.1, 0.4, N_frames - 3*(N_frames//4))
            dz = np.concatenate((dz1, dz2, dz3, dz4))

            theta_x1 = np.linspace(-0, 0, N_frames//2)
            theta_x2 = np.linspace(0, -0, N_frames - N_frames//2)
            theta_x = np.concatenate((theta_x1, theta_x2))

            theta_y = np.linspace(math.pi/6, -math.pi/6, N_frames)

            theta_z = np.linspace(0, 0, N_frames)
            # define poses
            dataset.poses_test = np.tile(dataset.poses_dict[dataset.img_ids_train[1548]], (N_frames, 1, 1))
            for i in range(N_frames):
                dataset.poses_test[i, 0, 3] += dx[i]
                dataset.poses_test[i, 1, 3] += dy[i]
                dataset.poses_test[i, 2, 3] += dz[i]
                dataset.poses_test[i, :, :3] = np.dot(eulerAnglesToRotationMatrix([theta_x[i],theta_y[i],theta_z[i]]), dataset.poses_test[i, :, :3])
        elif scene == 'sacre_coeur':
            # select appearance embedding, hard-coded for each scene
            img = Image.open(os.path.join(args.root_dir, 'dense/images',
                                          dataset.image_paths[dataset.img_ids_train[58]])).convert('RGB') # 10 1336 1548 296 420 1570 1662
            img_downscale = 8
            img_w, img_h = img.size
            img_w = img_w//img_downscale
            img_h = img_h//img_downscale
            img = img.resize((img_w, img_h), Image.LANCZOS)
            toTensor = T.ToTensor()
            normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            img = toTensor(img) # (3, h, w)
            whole_img = normalize(img).unsqueeze(0).cuda()
            kwargs['a_embedded_from_img'] = enc_a(whole_img)

            dataset.test_appearance_idx = dataset.img_ids_train[58] # 85572957_6053497857.jpg
            N_frames = 30*8
            dx = np.linspace(-2, 2, N_frames)   # + right

            dy1 = np.linspace(-0., 2, N_frames//2)    # + down
            dy2 = np.linspace(2, -0., N_frames - N_frames//2)
            dy = np.concatenate((dy1, dy2))

            dz1 = np.linspace(0, -3, N_frames//2)  # + foaward
            dz2 = np.linspace(-3, 0, N_frames - N_frames//2)  # + foaward

            dz = np.concatenate((dz1, dz2))

            theta_x1 = np.linspace(-0, 0, N_frames//2)
            theta_x2 = np.linspace(0, -0, N_frames - N_frames//2)
            theta_x = np.concatenate((theta_x1, theta_x2))

            theta_y = np.linspace(math.pi/6, -math.pi/6, N_frames)

            theta_z = np.linspace(0, 0, N_frames)
            # define poses
            dataset.poses_test = np.tile(dataset.poses_dict[dataset.img_ids_train[99]], (N_frames, 1, 1))
            for i in range(N_frames):
                dataset.poses_test[i, 0, 3] += dx[i]
                dataset.poses_test[i, 1, 3] += dy[i]
                dataset.poses_test[i, 2, 3] += dz[i]
                dataset.poses_test[i, :, :3] = np.dot(eulerAnglesToRotationMatrix([theta_x[i],theta_y[i],theta_z[i]]), dataset.poses_test[i, :, :3])
        else:
            raise NotImplementedError
        kwargs['output_transient'] = False

    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        rays = sample['rays']
        ts = sample['ts']
        if args.split == 'test_test' and args.encode_a:
            whole_img = sample['whole_img'].unsqueeze(0).cuda()
            whole_img=(whole_img+1)/2
            kwargs['a_embedded_from_img'] = enc_a(whole_img)
        results = batched_inference(models, embeddings, rays.cuda(), ts.cuda(),
                                    args.N_samples, args.N_importance, args.use_disp,
                                    args.chunk,
                                    dataset.white_back,
                                    **kwargs)
        if args.dataset_name == 'blender':
            w, h = args.img_wh
        else:
            w, h = sample['img_wh']
        feature=results['feature_fine'] #torch.Size([699008, 4])
        print("using fine feature")
        lastdim=feature.size(-1)
        feature = rearrange(feature, 'n1 n3 -> n3 n1', n3=lastdim)
        feature = rearrange(feature, ' n3 (h w) ->  1 n3 h w',  h=int(h), w=int(w),n3=lastdim)  ##torch.Size([1, 64, 340, 514])
        rgbs_pred=models['decoder'](feature, kwargs['a_embedded_from_img'])
        rgbs_pred=rearrange(rgbs_pred, ' 1 n1 h w ->  (h w) n1',  h=int(h), w=int(w),n1=3)
        results['rgb_fine']=rgbs_pred.cpu()
        img_pred = np.clip(results['rgb_fine'].view(h, w, 3).detach().numpy(), 0, 1)
        img_pred_ = (img_pred*255).astype(np.uint8)
        imgs += [img_pred_]
        imageio.imwrite(os.path.join(dir_name, f'{i:03d}.png'), img_pred_)
        
    if args.dataset_name == 'blender' or \
      (args.dataset_name == 'phototourism' and args.split == 'test'):
        imageio.mimsave(os.path.join(dir_name, f'{args.scene_name}.{args.video_format}'),
                        imgs, fps=30)
    print('Done')