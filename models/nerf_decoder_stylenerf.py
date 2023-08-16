
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from numpy import pi
import torch.nn as nn
import torch
from math import log2



def _xavier_init(net_layer):
    """
    Performs the Xavier weight initialization of the net layer.
    """
    torch.nn.init.xavier_uniform_(net_layer.weight.data)


def get_decoder(args1):
    # decoder = cfg['model']['decoder']
    decoder = Decoder(
       args=args1
    )
    return decoder
def _xavier_init(net_layer):
    """
    Performs the Xavier weight initialization of the net layer.
    """
    torch.nn.init.xavier_uniform_(net_layer.weight.data)


class Decoder(nn.Module):
    def __init__(self, args, pos_in_dims=63, dir_in_dims=27, D=8):
        """
        :param pos_in_dims: scalar, number of channels of encoded positions
        :param dir_in_dims: scalar, number of channels of encoded directions
        :param D:           scalar, number of hidden dimensions
        """
        super(Decoder, self).__init__()

        self.pos_in_dims = pos_in_dims
        self.dir_in_dims = dir_in_dims

        self.layers0 = nn.Sequential(
            nn.Linear(pos_in_dims, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
        )

        self.layers1 = nn.Sequential(
            nn.Linear(D + pos_in_dims, D), nn.ReLU(),  # shortcut
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
        )

        self.fc_density = nn.Linear(D, 1)
        # _xavier_init(self.fc_density)
        # self.fc_density.bias.data[:] = 0.0
        self.fc_feature = nn.Linear(D, D)
        self.use_dirmlp=args.use_dirmlp
        self.nerfoutdim_dim=args.nerf_out_dim
        if self.use_dirmlp:
            print("using dir")
            self.rgb_layers = nn.Sequential(nn.Linear(D + dir_in_dims, D//2), nn.ReLU())
        
        else:
            print("deprecate dir")
            self.rgb_layers = nn.Sequential(nn.Linear(D, D//2), nn.ReLU())
        self.fc_rgb = nn.Linear(D//2, self.nerfoutdim_dim)

        self.fc_density.bias.data = torch.tensor([0.2]).float()
        self.fc_rgb.bias.data = torch.tensor([0.02]*self.nerfoutdim_dim).float()

    def forward(self, pos_enc, dir_enc):
        """
        :param pos_enc: (H, W, N_sample, pos_in_dims) encoded positions
        :param dir_enc: (H, W, N_sample, dir_in_dims) encoded directions
        :return: rgb_density (H, W, N_sample, 4)
        """
        # print(torch.max(pos_enc), torch.min(pos_enc),"pos")
        # print(torch.max(dir_enc), torch.min(dir_enc),"dir")
        x = self.layers0(pos_enc)  # (H, W, N_sample, D)
        x = torch.cat([x, pos_enc], dim=-1)  # (H, W, N_sample, D+pos_in_dims)
        x = self.layers1(x)  # (H, W, N_sample, D)

        density = self.fc_density(x)  # (H, W, N_sample, 1)

        feat = self.fc_feature(x)  # (H, W, N_sample, D)
        if self.use_dirmlp: x = torch.cat([feat, dir_enc], dim=-1)  # (H, W, N_sample, D+dir_in_dims)
        else:x=feat
        
        x = self.rgb_layers(x)  # (H, W, N_sample, D/2)
        rgb = self.fc_rgb(x)  # (H, W, N_sample, 3)

        # rgb_den = torch.cat([rgb, density], dim=-1)  # (H, W, N_sample, 4)
        # print(torch.max(rgb), torch.min(rgb),"rgb")
        # print(torch.max(density), torch.min(density),"density")
        return rgb, density

#######renderer
from kornia.filters import filter2d
class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)


    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f[None, :, None]
        return filter2d(x, f, normalized=True)


class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_feature):
        super().__init__()
        self.in_feature = in_feature
        # self.out_feature = out_feature
        self._make_layer()
        

    def _make_layer(self):
        self.layer_1 = nn.Conv2d(self.in_feature, self.in_feature * 2, 1, 1, padding=0)
        self.layer_2 = nn.Conv2d(self.in_feature * 2, self.in_feature * 4, 1, 1, padding=0)
        self.blur_layer = Blur()
        self.actvn = nn.LeakyReLU(0.2, inplace=True)


    def forward(self, x:torch.Tensor):
        y = x.repeat(1, 4, 1, 1)
        out = self.actvn(self.layer_1(x))
        out = self.actvn(self.layer_2(out))
        
        out = out + y
        out = F.pixel_shuffle(out, 2)
        out = self.blur_layer(out)
        
        return out

class NeuralRenderer_11(nn.Module):

    def __init__(
            self, args_here, bg_type = "white", feat_nc=16, out_dim=3, final_actvn=True, min_feat=32, featmap_size=(32,32), img_size=(256, 256),  **kwargs):
            
        super().__init__()
        # assert n_feat == input_dim
        # self.sigma_dropout_rate=args_here.sigma_dropout_rate
        self.bg_type = bg_type
        self.featmap_size = featmap_size
        self.final_actvn = final_actvn
        # self.input_dim = input_dim
        self.n_feat = feat_nc
        self.out_dim = out_dim
        self.n_blocks = int(log2(img_size[0]/featmap_size[0]))
        self.min_feat = min_feat
        self._make_layer()
        # self._initialize_weights()

        
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def _make_layer(self):
        self.feat_upsample_list = nn.ModuleList(
            [PixelShuffleUpsample(max(self.n_feat // (2 ** (i)), self.min_feat)) for i in range(self.n_blocks)]
        )
        
        self.rgb_upsample = nn.Sequential(nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False), Blur())

        self.feat_2_rgb_list = nn.ModuleList(
                [nn.Conv2d(self.n_feat, self.out_dim, 1, 1, padding=0)] +
                [nn.Conv2d(max(self.n_feat // (2 ** (i + 1)), self.min_feat),
                           self.out_dim, 1, 1, padding=0) for i in range(0, self.n_blocks)]
            )

        self.feat_layers = nn.ModuleList(
            [nn.Conv2d(max(self.n_feat // (2 ** (i)), self.min_feat),
                       max(self.n_feat // (2 ** (i + 1)), self.min_feat), 1, 1,  padding=0)
                for i in range(0, self.n_blocks)]
        )
        
        self.actvn = nn.LeakyReLU(0.2, inplace=True)

        
        
    def forward(self, x):
        # res = []
        x_=self.feat_2_rgb_list[0](x)
        print(x_.size())
        rgb = self.rgb_upsample(x_)
        print(rgb.size())
        # res.append(rgb)
        net = x
        for idx in range(self.n_blocks):
            hid = self.feat_layers[idx](self.feat_upsample_list[idx](net))
            net = self.actvn(hid)
            
            rgb = rgb + self.feat_2_rgb_list[idx + 1](net)
            if idx < self.n_blocks - 1:
                rgb = self.rgb_upsample(rgb)
                # res.append(rgb)
        
        if self.final_actvn:

            rgbs = torch.sigmoid(rgb)

        return rgbs



##codes are from headnerf https://github.com/CrisHY1995/headnerf.git
class NeuralRenderer(nn.Module):  ##the decoder of CR-NeRF

    def __init__(
            self, bg_type = "white", feat_nc=128, out_dim=3, final_actvn=True, min_feat=32, featmap_size=(32,32), img_size=(256, 256), 
            **kwargs):
        super().__init__()
        
        self.bg_type = bg_type
        self.featmap_size = featmap_size
        self.final_actvn = final_actvn
        self.n_feat = feat_nc
        self.out_dim = out_dim
        self.n_blocks = int(log2(img_size[0]/featmap_size[0]))
        self.min_feat = min_feat
        self._make_layer()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def _make_layer(self):
        self.feat_upsample_list = nn.ModuleList(
            [PixelShuffleUpsample(max(self.n_feat // (2 ** (i)), self.min_feat)) for i in range(self.n_blocks)]
        )
        
        self.rgb_upsample = nn.Sequential(nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False), Blur())

        self.feat_2_rgb_list = nn.ModuleList(
                [nn.Conv2d(self.n_feat, self.out_dim, 1, 1, padding=0)] +
                [nn.Conv2d(max(self.n_feat // (2 ** (i + 1)), self.min_feat),
                           self.out_dim, 1, 1, padding=0) for i in range(0, self.n_blocks)]
            )

        self.feat_layers = nn.ModuleList(
            [nn.Conv2d(max(self.n_feat // (2 ** (i)), self.min_feat),
                       max(self.n_feat // (2 ** (i + 1)), self.min_feat), 1, 1,  padding=0)
                for i in range(0, self.n_blocks)]
        )
        
        self.actvn = nn.LeakyReLU(0.2, inplace=True)

        
        
    def forward(self, x):
        rgb=self.feat_2_rgb_list[0](x)
        for idx in range(self.n_blocks):
            hid = self.feat_layers[idx](self.feat_upsample_list[idx](net))
            net = self.actvn(hid)
            
            rgb = rgb + self.feat_2_rgb_list[idx + 1](net)
            if idx < self.n_blocks - 1:
                rgb = self.rgb_upsample(rgb)
        
        if self.final_actvn:
            rgbs = torch.sigmoid(rgb)
        return rgbs

class NeuralRenderer_11v1(nn.Module):

    def __init__(
            self, args_here, bg_type = "white", feat_nc=16, out_dim=3, final_actvn=True, min_feat=16, featmap_size=(32,32), img_size=(256, 256),  **kwargs):
            
        super().__init__()
        # assert n_feat == input_dim
        # self.sigma_dropout_rate=args_here.sigma_dropout_rate
        self.bg_type = bg_type
        self.featmap_size = featmap_size
        self.final_actvn = final_actvn
        # self.input_dim = input_dim
        self.n_feat = feat_nc
        self.out_dim = out_dim
        # self.n_blocks = int(log2(img_size[0]/featmap_size[0]))
        self.n_blocks = 2 
        print(self.n_blocks)
        self.min_feat = min_feat
        self._make_layer()
        # self._initialize_weights()

        
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def _make_layer(self):
        self.feat_upsample_list = nn.ModuleList(
            [PixelShuffleUpsample(max(self.n_feat // (2 ** (i)), self.min_feat)) for i in range(self.n_blocks)]
        )
        
        self.rgb_upsample = nn.Sequential(nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False), Blur())

        self.feat_2_rgb_list = nn.ModuleList(
                [nn.Conv2d(self.n_feat, self.out_dim, 1, 1, padding=0)] +
                [nn.Conv2d(max(self.n_feat // (2 ** (i + 1)), self.min_feat),
                           self.out_dim, 1, 1, padding=0) for i in range(0, self.n_blocks)]
            )

        self.feat_layers = nn.ModuleList(
            [nn.Conv2d(max(self.n_feat // (2 ** (i)), self.min_feat),
                       max(self.n_feat // (2 ** (i + 1)), self.min_feat), 1, 1,  padding=0)
                for i in range(0, self.n_blocks)]
        )
        
        self.actvn = nn.LeakyReLU(0.2, inplace=True)
        # self.actvn = nn.ReLU()
        self.rgb_downsample = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=int(self.n_blocks+2), stride=int(self.n_blocks+2), padding=0, bias=False))
        
    def forward(self, x):
        # print(torch.max(x), torch.min(x))
        # res = []
        rgb = self.rgb_upsample(self.feat_2_rgb_list[0](x))
        # res.append(rgb)
        net = x
        for idx in range(self.n_blocks):
            hid0=self.feat_upsample_list[idx](net)
            hid = self.feat_layers[idx](hid0)
            net = self.actvn(hid)
            
            rgb = rgb + self.feat_2_rgb_list[idx + 1](net)
            if idx < self.n_blocks - 1:
                rgb = self.rgb_upsample(rgb)
        rgb=self.rgb_downsample(rgb)
        if self.final_actvn:rgbs = torch.sigmoid(rgb)
        return rgbs

class NeuralRenderer_11_tanh(nn.Module):

    def __init__(
            self, args_here, bg_type = "white", feat_nc=16, out_dim=3, final_actvn=True, min_feat=16, featmap_size=(32,32), img_size=(256, 256),  **kwargs):
            
        super().__init__()
        # assert n_feat == input_dim
        # self.sigma_dropout_rate=args_here.sigma_dropout_rate
        self.bg_type = bg_type
        self.featmap_size = featmap_size
        self.final_actvn = final_actvn
        # self.input_dim = input_dim
        self.n_feat = feat_nc
        self.out_dim = out_dim
        # self.n_blocks = int(log2(img_size[0]/featmap_size[0]))
        self.n_blocks = 2 
        print(self.n_blocks)
        self.min_feat = min_feat
        self._make_layer()
        # self._initialize_weights()

        
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def _make_layer(self):
        self.feat_upsample_list = nn.ModuleList(
            [PixelShuffleUpsample(max(self.n_feat // (2 ** (i)), self.min_feat)) for i in range(self.n_blocks)]
        )
        
        self.rgb_upsample = nn.Sequential(nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False), Blur())

        self.feat_2_rgb_list = nn.ModuleList(
                [nn.Conv2d(self.n_feat, self.out_dim, 1, 1, padding=0)] +
                [nn.Conv2d(max(self.n_feat // (2 ** (i + 1)), self.min_feat),
                           self.out_dim, 1, 1, padding=0) for i in range(0, self.n_blocks)]
            )

        self.feat_layers = nn.ModuleList(
            [nn.Conv2d(max(self.n_feat // (2 ** (i)), self.min_feat),
                       max(self.n_feat // (2 ** (i + 1)), self.min_feat), 1, 1,  padding=0)
                for i in range(0, self.n_blocks)]
        )
        
        self.actvn = nn.LeakyReLU(0.2, inplace=True)
        # self.actvn = nn.ReLU()
        self.rgb_downsample = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=int(self.n_blocks+2), stride=int(self.n_blocks+2), padding=0, bias=False))
        
    def forward(self, x):
        # print(torch.max(x), torch.min(x))
        # res = []
        rgb = self.rgb_upsample(self.feat_2_rgb_list[0](x))
        # res.append(rgb)
        net = x
        for idx in range(self.n_blocks):
            hid0=self.feat_upsample_list[idx](net)
            hid = self.feat_layers[idx](hid0)
            net = self.actvn(hid)
            
            rgb = rgb + self.feat_2_rgb_list[idx + 1](net)
            if idx < self.n_blocks - 1:
                rgb = self.rgb_upsample(rgb)
        rgb=self.rgb_downsample(rgb)
        if self.final_actvn:
            rgbs = torch.tanh(rgb)
            rgbs=(rgbs+1)/2
        # if self.final_actvn:rgbs = torch.sigmoid(rgb)
        return rgbs

def get_renderer(args):
    if args.model_mode=='1-1':
        renderer = NeuralRenderer(img_size=(args.img_wh[0],args.img_wh[1]) , featmap_size=(args.img_wh[0],args.img_wh[1]), feat_nc=args.nerf_out_dim, out_dim=3, args_here=args ) 
    elif args.model_mode=='1-4-1':
        # renderer = NeuralRenderer_11v1(img_size=(args.img_wh[0],args.img_wh[1]) , featmap_size=(args.img_wh[0],args.img_wh[1]), feat_nc=args.nerf_out_dim, out_dim=3, args_here=args ) 
        renderer = NeuralRenderer_11_tanh(img_size=(args.img_wh[0],args.img_wh[1]) , featmap_size=(args.img_wh[0],args.img_wh[1]), feat_nc=args.nerf_out_dim, out_dim=3, args_here=args ) 
    return renderer

if __name__ == '__main__':
    import sys
    sys.path.append('/mnt/cephfs/home/yangyifan/yangyifan/code/learnToSyLf/CR-NeRF')
    from opt import get_opts
    args=get_opts()
    from torchview import draw_graph
    import graphviz
    graphviz.set_jupyter_format('png')
    enc= NeuralRenderer(img_size=(args.img_wh[0],args.img_wh[1]) , featmap_size=(args.img_wh[0],args.img_wh[1]), feat_nc=args.nerf_out_dim, out_dim=3, args_here=args ) 

    model_graph = draw_graph(enc, input_size=(1,64,32,32), device='meta', save_graph=True,graph_dir="images/encoder.png")
    model_graph.visual_graph.render(format='pdf')