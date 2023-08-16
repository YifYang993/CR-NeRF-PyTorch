import torch

weightMS=0.01
inputs={}
inputs["rgb_fine"]=torch.randn(1024,3)
inputs['rgb_fine_random']=torch.randn(1024,3)
inputs['a_embedded']=torch.randn(1,48)
inputs['a_embedded_random']=torch.randn(1,48)
res=weightMS * 1 / ((torch.mean(torch.abs(inputs['rgb_fine'].detach() - inputs['rgb_fine_random'])) / torch.mean(torch.abs(inputs['a_embedded'].detach() - inputs['a_embedded_random'].detach()))) + 1 * 1e-5)
print(res)
res2= weightMS * 1 / \
                  ((torch.mean(torch.abs(inputs['rgb_fine'].detach() - inputs['rgb_fine_random'])) / \
                  torch.mean(torch.abs(inputs['a_embedded'].detach() - inputs['a_embedded_random'].detach()))) + 1 * 1e-5)
print(res2==res)
                 
                  