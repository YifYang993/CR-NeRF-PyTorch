import torch
from torch import nn
import os
class PosEmbedding(nn.Module):
    def __init__(self, max_logscale, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        """
        super().__init__()
        self.funcs = [torch.sin, torch.cos]

        if logscale:
            self.freqs = 2**torch.linspace(0, max_logscale, N_freqs)
        else:
            self.freqs = torch.linspace(1, 2**max_logscale, N_freqs)

    def forward(self, x):
        """
        Inputs:
            x: (B, 3)

        Outputs:
            out: (B, 6*N_freqs+3)
        """
        out = [x]
        for freq in self.freqs:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)


class NeRF(nn.Module):
    def __init__(self, typ,args,
                 D=8, W=256, skips=[4],
                 in_channels_xyz=63, in_channels_dir=27,
                 encode_appearance=False, in_channels_a=48,
                 encode_random=False):

        super().__init__()
        self.typ = typ
        self.D = D
        self.W = W
        self.skips = skips
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir

        # self.encode_appearance = encode_appearance
        self.encode_appearance = False if typ=='coarse' else encode_appearance
        self.in_channels_a = in_channels_a if encode_appearance else 0
        self.encode_random = False if typ=='coarse' else encode_random

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # static output layers
        self.static_sigma = nn.Sequential(nn.Linear(W, 1), nn.Softplus())

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
                        nn.Linear(W+in_channels_dir+self.in_channels_a, W//2), nn.ReLU(True))
        self.static_rgb = nn.Sequential(nn.Linear(W//2, args.nerf_out_dim), nn.Sigmoid())


    def forward(self, x, sigma_only=False, output_random=True):

        if sigma_only:
            input_xyz = x
        elif output_random:
            input_xyz, input_dir, input_a, input_random_a = \
                  torch.split(x, [self.in_channels_xyz,
                                  self.in_channels_dir,
                                  self.in_channels_a,
                                  self.in_channels_a], dim=-1)
            input_dir_a = torch.cat((input_dir, input_a), dim=-1)
            input_dir_a_random = torch.cat((input_dir, input_random_a), dim=-1)
        else:
            input_xyz, input_dir_a = \
                torch.split(x, [self.in_channels_xyz,
                                self.in_channels_dir+self.in_channels_a], dim=-1)

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], 1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        # static_sigma = self.static_sigma(xyz_) # (B, 1)
        # if sigma_only:
        #     return static_sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)
        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir_a], 1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        static_rgb = self.static_rgb(dir_encoding) # (B, 3)
        # static = torch.cat([static_rgb, static_sigma], 1) # (B, 4)
        static=static_rgb
        if output_random:
            dir_encoding_input_random = torch.cat([xyz_encoding_final.detach(), input_dir_a_random.detach()], 1)
            dir_encoding_random = self.dir_encoding(dir_encoding_input_random)
            static_rgb_random = self.static_rgb(dir_encoding_random) # (B, 3)
            return torch.cat([static, static_rgb_random], 1) # (B, 7)

        return static

class NeRF_sigma(nn.Module):
    def __init__(self, typ, args,
                 D=8, W=256, skips=[4],
                 in_channels_xyz=63, in_channels_dir=27,
                 encode_appearance=False, in_channels_a=48,
                 encode_random=False):

        super().__init__()
        
        self.typ = typ
        self.D = D
        self.W = W
        self.skips = skips
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir

        # self.encode_appearance = encode_appearance
        self.encode_appearance = False if typ=='coarse' else encode_appearance
        self.in_channels_a = in_channels_a if encode_appearance else 0
        self.encode_random = False if typ=='coarse' else encode_random

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(inplace=True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # static output layers
        self.static_sigma = nn.Sequential(nn.Linear(W, 1), nn.Softplus())

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
                        nn.Linear(W+in_channels_dir, W//2), nn.ReLU(inplace=True))
        self.static_rgb = nn.Sequential(nn.Linear(W//2, args.nerf_out_dim), nn.Sigmoid())


    def forward(self, x, sigma_only=False, output_random=True):
        # print("coordinate", torch.max(x),torch.min(x))
        if sigma_only:
            input_xyz = x                                
        else:
            input_xyz, input_dir = \
                torch.split(x, [self.in_channels_xyz,
                                self.in_channels_dir], dim=-1)

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], 1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        static_sigma = self.static_sigma(xyz_) # (B, 1)
        if sigma_only:
            return static_sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)
        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], 1)
        dir_encoding = self.dir_encoding(dir_encoding_input)

        static_rgb = self.static_rgb(dir_encoding) # (B, 3)
        static = torch.cat([static_rgb, static_sigma], -1) # (B, 4)
        return static

class NeRF_sigma_tanh(nn.Module):
    def __init__(self, typ, args,
                 D=8, W=256, skips=[4],
                 in_channels_xyz=63, in_channels_dir=27,
                 encode_appearance=False, in_channels_a=48,
                 encode_random=False):

        super().__init__()
        
        self.typ = typ
        self.D = D
        self.W = W
        self.skips = skips
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir

        # self.encode_appearance = encode_appearance
        self.encode_appearance = False if typ=='coarse' else encode_appearance
        self.in_channels_a = in_channels_a if encode_appearance else 0
        self.encode_random = False if typ=='coarse' else encode_random

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.LeakyReLU(0.2, inplace=True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # static output layers
        self.static_sigma = nn.Sequential(nn.Linear(W, 1), nn.Softplus())

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
                        nn.Linear(W+in_channels_dir, W//2), nn.LeakyReLU(0.2, inplace=True))
        self.static_rgb = nn.Sequential(nn.Linear(W//2, args.nerf_out_dim), nn.Tanh())


    def forward(self, x, sigma_only=False, output_random=True):
        # print("coordinate", torch.max(x),torch.min(x))
        if sigma_only:
            input_xyz = x        
        elif output_random:
            input_xyz, input_dir = \
                  torch.split(x, [self.in_channels_xyz,
                                  self.in_channels_dir]
                                  , dim=-1)

        else:
            input_xyz, input_dir = \
                torch.split(x, [self.in_channels_xyz,
                                self.in_channels_dir], dim=-1)

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], 1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        static_sigma = self.static_sigma(xyz_) # (B, 1)
        if sigma_only:
            return static_sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)
        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], 1)
        dir_encoding = self.dir_encoding(dir_encoding_input)

        static_rgb = self.static_rgb(dir_encoding) # (B, 3)
        static = torch.cat([static_rgb, static_sigma], -1) # (B, 4)
        # static=static_rgb
        # if output_random:
        #     dir_encoding_input_random = torch.cat([xyz_encoding_final.detach(), input_dir.detach()], 1)
        #     dir_encoding_random = self.dir_encoding(dir_encoding_input_random)
        #     static_rgb_random = self.static_rgb(dir_encoding_random) # (B, 3)
        #     print(static_rgb_random==static_rgb,11111111111111111111111111)
        #     return torch.cat([static, static_rgb_random], 1) # (B, 7)
        return static