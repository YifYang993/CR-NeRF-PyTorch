import torch
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import imageio
from argparse import ArgumentParser
from einops import rearrange
from models.rendering import render_rays_cross_ray
from models.linearStyleTransfer import encoder_sameoutputsize 
from models.linearStyleTransfer import style_net
from train_mask_grid_sample import get_model
from models.networks import *
from models.nerf import *

from utils import load_ckpt
from datasets.PhototourismDataset import *
import math
from PIL import Image
from torchvision import transforms as T

torch.backends.cudnn.benchmark = True

def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--example_image', type=str,
                        default='example_imgs_cross_datasets/97851507_2113931340.jpg',
                        help='directory of example image')
    parser.add_argument('--scene_name', type=str, default='fountain_2_gate_exp1',
                        help='scene name, used as output folder name')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[320, 240],
                        help='resolution (img_w, img_h) of the image')
    parser.add_argument('--ckpt_path', type=str, default="ckpt/epoch19.ckpt",
                        help='pretrained checkpoint path to load')
    parser.add_argument('--save_dir', type=str, default="./",
                        help='path to save')

    # original NeRF parameters  decoder_num_res_blocks=1 
    parser.add_argument('--decoder_num_res_blocks', type=int, default=1,
                        help='number of decoder_num_res_blocks')
    parser.add_argument('--nerf_out_dim', type=int, default=64,
                        help='number of output dimension in nerf')
    parser.add_argument('--N_emb_xyz', type=int, default=15,
                        help='number of xyz embedding frequencies')
    parser.add_argument('--N_emb_dir', type=int, default=4,
                        help='number of direction embedding frequencies')
    parser.add_argument('--N_samples', type=int, default=256,
                        help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=256,
                        help='number of additional fine samples')
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument('--chunk', type=int, default=16384,
                        help='chunk size to split the input to avoid OOM')

    # CR-NeRF parameters
    parser.add_argument('--pertubeCord', default=False, action="store_true",
                        help='whether to encode appearance')
    parser.add_argument('--encode_a', default=False, action="store_true",
                        help='whether to encode appearance')
    parser.add_argument('--encode_c', default=False, action="store_true",
                        help='whether to encode content')
    parser.add_argument('--encode_random', default=False, action="store_true",
                        help='whether to encode appearance')
    parser.add_argument('--N_a', type=int, default=48,
                        help='number of embeddings for appearance')

    return parser.parse_args()


@torch.no_grad()
def batched_inference(models, embeddings,
                      rays, N_samples, N_importance, use_disp,
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
                        None,
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

def define_poses_brandenburg_gate(dataset):
    N_frames = 30 * 8
    pose_init = np.array([[ 0.99702646,  0.00170214, -0.07704115,  0.03552477], \
                          [ 0.01082206, -0.99294089,  0.11811554,  0.02343685], \
                          [-0.07629626, -0.11859807, -0.99000676,  0.12162088]])

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
    dataset.poses_test = np.tile(pose_init, (N_frames, 1, 1))
    for i in range(N_frames):
        dataset.poses_test[i, 0, 3] += dx[i]
        dataset.poses_test[i, 1, 3] += dy[i]
        dataset.poses_test[i, 2, 3] += dz[i]
        dataset.poses_test[i, :, :3] = np.dot(eulerAnglesToRotationMatrix([theta_x[i],theta_y[i],theta_z[i]]), dataset.poses_test[i, :, :3])

def define_poses_trevi_fountain(dataset):
    N_frames = 30 * 8

    pose_init = np.array([[ 9.99719757e-01, -4.88717623e-03, -2.31629550e-02, -2.66316808e-02],
                          [-6.52512819e-03, -9.97442504e-01, -7.11749546e-02, -6.68793042e-04],
                          [-2.27558713e-02,  7.13061496e-02, -9.97194867e-01, 7.93278041e-04]])

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

    dataset.poses_test = np.tile(pose_init, (N_frames, 1, 1))
    for i in range(N_frames):
        dataset.poses_test[i, 0, 3] += dx[i]
        dataset.poses_test[i, 1, 3] += dy[i]
        dataset.poses_test[i, 2, 3] += dz[i]
        dataset.poses_test[i, :, :3] = np.dot(eulerAnglesToRotationMatrix([theta_x[i],theta_y[i],theta_z[i]]), dataset.poses_test[i, :, :3])


def define_camera(dataset):
    # define testing camera intrinsics (hard-coded, feel free to change)
    dataset.test_img_w, dataset.test_img_h = args.img_wh
    dataset.test_focal = dataset.test_img_w/2/np.tan(np.pi/6) # fov=60 degrees
    dataset.test_K = np.array([[dataset.test_focal, 0, dataset.test_img_w/2],
                                [0, dataset.test_focal, dataset.test_img_h/2],
                                [0,                  0,                    1]])

if __name__ == "__main__":
    args = get_opts()
    dataset = PhototourismDataset()

    embedding_xyz = PosEmbedding(args.N_emb_xyz-1, args.N_emb_xyz)
    embedding_dir = PosEmbedding(args.N_emb_dir-1, args.N_emb_dir)
    embeddings = {'xyz': embedding_xyz, 'dir': embedding_dir}
 
    enc_a = encoder_sameoutputsize(out_channel=args.nerf_out_dim).cuda()
    load_ckpt(enc_a, args.ckpt_path, model_name='enc_a')
    models=get_model(args)
    nerf_coarse=models['coarse']
    nerf_fine=models['fine']
    decoder=models['decoder']
    load_ckpt(nerf_coarse, args.ckpt_path, model_name='nerf_coarse')
    load_ckpt(nerf_fine, args.ckpt_path, model_name='nerf_fine')
    load_ckpt(decoder, args.ckpt_path, model_name='decoder')

    imgs = []
    dir_name = os.path.join(args.save_dir, f'appearance_modification/{args.scene_name}')
    os.makedirs(dir_name, exist_ok=True)

    define_camera(dataset)
    if dir_name.split('_')[-1] == 'gate':
        define_poses_brandenburg_gate(dataset)
    elif dir_name.split('_')[-1] == 'fountain':
        define_poses_trevi_fountain(dataset)
    else:
        input("Pose not defined")

    kwargs = {}
    kwargs['args']=args

    files= os.listdir(args.example_image)
    for file_name in tqdm(files):
        org_img = Image.open(os.path.join(args.example_image, file_name)).convert('RGB')
        imageio.imwrite(os.path.join(dir_name, file_name), org_img)
        fig_name = file_name.split('.')[0]
        img_downscale = 8
        img_w, img_h = org_img.size
        img_w = img_w//img_downscale
        img_h = img_h//img_downscale
        img = org_img.resize((img_w, img_h), resample=Image.LANCZOS)
        toTensor = T.ToTensor()
        normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        img = toTensor(img) # (3, h, w)
        whole_img = normalize(img).unsqueeze(0).cuda()
        whole_img=(whole_img+1)/2
        kwargs['a_embedded_from_img'] = enc_a(whole_img)

        imgs = []
        for i in range(len(dataset)):
            sample = dataset[i]
            rays = sample['rays']
            results = batched_inference(models, embeddings, rays.cuda(),
                                        args.N_samples, args.N_importance, args.use_disp,
                                        args.chunk,
                                        dataset.white_back,
                                        **kwargs)
            w, h = sample['img_wh']
            feature=results['feature_fine'] #torch.Size([699008, 4])
            lastdim=feature.size(-1)
            feature = rearrange(feature, 'n1 n3 -> n3 n1', n3=lastdim)
            feature = rearrange(feature, ' n3 (h w) ->  1 n3 h w',  h=int(h), w=int(w),n3=lastdim)  ##torch.Size([1, 64, 340, 514])
            rgbs_pred=models['decoder'](feature, kwargs['a_embedded_from_img'])
            rgbs_pred=rearrange(rgbs_pred, ' 1 n1 h w ->  (h w) n1',  h=int(h), w=int(w),n1=3)
            results['rgb_fine']=rgbs_pred.cpu()
            img_pred = np.clip(results['rgb_fine'].view(h, w, 3).cpu().detach().numpy(), 0, 1)
            img_pred_ = (img_pred*255).astype(np.uint8)
            imgs += [img_pred_]

            imageio.imwrite(os.path.join(dir_name, f'{fig_name}_{i:03d}.png'), img_pred_)
        imageio.mimsave(os.path.join(dir_name, f'{fig_name}.gif'), imgs, fps=30)