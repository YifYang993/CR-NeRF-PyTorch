import torch
import torch.nn as nn
import sys
sys.path.append('/mnt/cephfs/home/yangyifan/yangyifan/code/learnToSyLf/CR-NeRF')
from models.nerf_decoder_stylenerf import NeuralRenderer
class CNN(nn.Module):
    def __init__(self,matrixSize=32,in_channel=64):
        super(CNN,self).__init__()
        # if(layer == 'r31'):
            # 256x64x64
        self.convs = nn.Sequential(nn.Conv2d(in_channel,128,1,1,0),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Conv2d(128,64,1,1,0),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Conv2d(64,matrixSize,1,1,0))
        # elif(layer == 'r41'):
        #     # 512x32x32
        #     self.convs = nn.Sequential(nn.Conv2d(512,256,3,1,1),
        #                                nn.ReLU(inplace=True),
        #                                nn.Conv2d(256,128,3,1,1),
        #                                nn.ReLU(inplace=True),
        #                                nn.Conv2d(128,matrixSize,3,1,1))

        # 32x8x8
        self.fc = nn.Linear(matrixSize*matrixSize,matrixSize*matrixSize)
        #self.fc = nn.Linear(32*64,256*256)

    def forward(self,x):
        out = self.convs(x)
        # 32x8x8
        b,c,h,w = out.size()
        out = out.view(b,c,-1)
        # 32x64
        out = torch.bmm(out,out.transpose(1,2)).div(h*w)
        # 32x32
        out = out.view(out.size(0),-1)
        return self.fc(out)


"""
The code below are from linear style transfer
""" 
class MulLayer(nn.Module):
    def __init__(self,matrixSize=32,in_channel=64):
        super(MulLayer,self).__init__()
        self.snet = CNN(matrixSize)
        self.cnet = CNN(matrixSize)
        self.matrixSize = matrixSize

        # if(layer == 'r41'):
        #     self.compress = nn.Conv2d(512,matrixSize,1,1,0)
        #     self.unzip = nn.Conv2d(matrixSize,512,1,1,0)
        # elif(layer == 'r31'):
        self.compress = nn.Conv2d(in_channel,matrixSize,1,1,0)
        self.unzip = nn.Conv2d(matrixSize,in_channel,1,1,0)
        self.transmatrix = None

    def forward(self,cF,sF,trans=True):
        cFBK = cF.clone()
        cb,cc,ch,cw = cF.size()
        cFF = cF.view(cb,cc,-1)
        cMean = torch.mean(cFF,dim=2,keepdim=True)
        cMean = cMean.unsqueeze(3)
        cMean = cMean.expand_as(cF)
        cF = cF - cMean

        sb,sc,sh,sw = sF.size()
        sFF = sF.view(sb,sc,-1)
        sMean = torch.mean(sFF,dim=2,keepdim=True)
        sMean = sMean.unsqueeze(3)
        sMeanC = sMean.expand_as(cF)
        sMeanS = sMean.expand_as(sF)
        sF = sF - sMeanS


        compress_content = self.compress(cF)
        b,c,h,w = compress_content.size()
        compress_content = compress_content.view(b,c,-1)

        if(trans):
            cMatrix = self.cnet(cF)
            sMatrix = self.snet(sF)

            sMatrix = sMatrix.view(sMatrix.size(0),self.matrixSize,self.matrixSize)
            cMatrix = cMatrix.view(cMatrix.size(0),self.matrixSize,self.matrixSize)
            transmatrix = torch.bmm(sMatrix,cMatrix)
            transfeature = torch.bmm(transmatrix,compress_content).view(b,c,h,w)
            out = self.unzip(transfeature.view(b,c,h,w))
            out = out + sMeanC
            return out, transmatrix
        else:
            out = self.unzip(compress_content.view(b,c,h,w))
            out = out + cMean
            return 


class decoder3(nn.Module):
    def __init__(self,in_channel=64):
        super(decoder3,self).__init__()
        # decoder
        self.reflecPad7 = nn.ReflectionPad2d((1,1,1,1))
        self.conv7 = nn.Conv2d(in_channel,128,3,1,0)
        self.relu7 = nn.ReLU(inplace=True)
        # 56 x 56

        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        # 112 x 112

        self.reflecPad8 = nn.ReflectionPad2d((1,1,1,1))
        self.conv8 = nn.Conv2d(128,128,3,1,0)
        self.relu8 = nn.ReLU(inplace=True)
        # 112 x 112

        self.reflecPad9 = nn.ReflectionPad2d((1,1,1,1))
        self.conv9 = nn.Conv2d(128,64,3,1,0)
        self.relu9 = nn.ReLU(inplace=True)

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        # 224 x 224

        self.reflecPad10 = nn.ReflectionPad2d((1,1,1,1))
        self.conv10 = nn.Conv2d(64,64,3,1,0)
        self.relu10 = nn.ReLU(inplace=True)

        self.reflecPad11 = nn.ReflectionPad2d((1,1,1,1))
        self.conv11 = nn.Conv2d(64,3,3,1,0)

    def forward(self,x):
        output = {}
        out = self.reflecPad7(x)
        out = self.conv7(out)
        out = self.relu7(out)
        out = self.unpool(out)
        out = self.reflecPad8(out)
        out = self.conv8(out)
        out = self.relu8(out)
        out = self.reflecPad9(out)
        out = self.conv9(out)
        out_relu9 = self.relu9(out)
        out = self.unpool2(out_relu9)
        out = self.reflecPad10(out)
        out = self.conv10(out)
        out = self.relu10(out)
        out = self.reflecPad11(out)
        out = self.conv11(out)
        return out


class encoder3(nn.Module):
    def __init__(self,out_channel=64):
        super(encoder3,self).__init__()
        # vgg
        # 224 x 224
        self.conv1 = nn.Conv2d(3,3,1,1,0)
        self.reflecPad1 = nn.ReflectionPad2d((1,1,1,1))
        # 226 x 226

        self.conv2 = nn.Conv2d(3,64,3,1,0)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        # 224 x 224

        self.reflecPad3 = nn.ReflectionPad2d((1,1,1,1))
        self.conv3 = nn.Conv2d(64,64,3,1,0)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        # 224 x 224

        self.maxPool = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = True)
        # 112 x 112

        self.reflecPad4 = nn.ReflectionPad2d((1,1,1,1))
        self.conv4 = nn.Conv2d(64,128,3,1,0)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)
        # 112 x 112

        self.reflecPad5 = nn.ReflectionPad2d((1,1,1,1))
        self.conv5 = nn.Conv2d(128,128,3,1,0)
        self.relu5 = nn.LeakyReLU(0.2, inplace=True)
        # 112 x 112

        self.maxPool2 = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = True)
        # 56 x 56

        self.reflecPad6 = nn.ReflectionPad2d((1,1,1,1))
        self.conv6 = nn.Conv2d(128,out_channel,3,1,0)
        self.relu6 = nn.LeakyReLU(0.2, inplace=True)
        # 56 x 56
    def forward(self,x):
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.reflecPad3(out)
        out = self.conv3(out)
        pool1 = self.relu3(out)
        out,pool_idx = self.maxPool(pool1)
        out = self.reflecPad4(out)
        out = self.conv4(out)
        out = self.relu4(out)
        out = self.reflecPad5(out)
        out = self.conv5(out)
        pool2 = self.relu5(out)
        out,pool_idx2 = self.maxPool2(pool2)
        out = self.reflecPad6(out)
        out = self.conv6(out)
        out = self.relu6(out)
        return out

class encoder_sameoutputsize(nn.Module):
    def __init__(self,out_channel=64):
        super(encoder_sameoutputsize,self).__init__()
        # vgg
        # 224 x 224
        self.conv1 = nn.Conv2d(3,3,1,1,0)
        self.reflecPad1 = nn.ReflectionPad2d((1,1,1,1))
        # 226 x 226

        self.conv2 = nn.Conv2d(3,64,3,1,0)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        # 224 x 224

        self.reflecPad3 = nn.ReflectionPad2d((1,1,1,1))
        self.conv3 = nn.Conv2d(64,64,3,1,0)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        # 224 x 224

        self.maxPool = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = True)
        # 112 x 112

        self.reflecPad4 = nn.ReflectionPad2d((1,1,1,1))
        self.conv4 = nn.Conv2d(64,128,3,1,0)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)
        # 112 x 112

        self.reflecPad5 = nn.ReflectionPad2d((1,1,1,1))
        self.conv5 = nn.Conv2d(128,128,3,1,0)
        self.relu5 = nn.LeakyReLU(0.2, inplace=True)
        # 112 x 112

        self.maxPool2 = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = True)
        # 56 x 56

        self.reflecPad6 = nn.ReflectionPad2d((1,1,1,1))
        self.conv6 = nn.Conv2d(128,128,3,1,0)
        self.relu6 = nn.LeakyReLU(0.2, inplace=True)
        self.adppool=nn.AdaptiveAvgPool2d(32)
        self.conv7 = nn.Conv2d(128,out_channel,1,1,0)
        self.relu7 = nn.LeakyReLU(0.2, inplace=True)

        # 56 x 56
    def forward(self,x):
        # print(torch.min(x),torch.max(x),248)
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.reflecPad3(out)
        out = self.conv3(out)
        # print(torch.min(out),torch.max(out),254)
        pool1 = self.relu3(out)
        out,pool_idx = self.maxPool(pool1)
        out = self.reflecPad4(out)
        out = self.conv4(out)
        out = self.relu4(out)
        out = self.reflecPad5(out)
        out = self.conv5(out)
        # print(torch.min(out),torch.max(out),262)
        pool2 = self.relu5(out)
        out,pool_idx2 = self.maxPool2(pool2)
        out = self.reflecPad6(out)
        out = self.conv6(out)
        out = self.relu6(out)
        # print(torch.min(out),torch.max(out),267)
        out=self.adppool(out)
        out=self.conv7(out)
        out=self.relu7(out)
        return out

class style_net(nn.Module):
    def __init__(self, args, residual_blocks=2):
        super(style_net,self).__init__()
        nerf_channel=args.nerf_out_dim
        self.multi_net=MulLayer(in_channel=nerf_channel)
        self.decoder=NeuralRenderer(img_size=(args.img_wh[0],args.img_wh[1]) , featmap_size=(args.img_wh[0],args.img_wh[1]), feat_nc=args.nerf_out_dim, out_dim=3, args_here=args ) 
    def forward(self, content_feature, style_feature,type=None):
        if style_feature==None and type=="content":
            gen_img=self.decoder(content_feature)
            return gen_img
        else:
            fused_feature,_=self.multi_net(content_feature,style_feature)
            gen_img=self.decoder(fused_feature)
            return gen_img
        

if __name__=='__main__':
  input2=torch.randn(1,3,340,512)
  enc=encoder_sameoutputsize()
  feature=enc(input2)



