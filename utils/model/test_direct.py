
# from direct.nn.jointicnet.jointicnet_engine import JointICNetEngine
# from direct.nn.jointicnet.jointicnet import JointICNet
# from direct.nn.jointicnet.config import JointICNetConfig

from direct.nn.multidomainnet.multidomainnet import MultiDomainNet 
# from direct.nn.multidomainnet.config import MultiDomainNetConfig

# from direct.nn.recurrentvarnet.recurrentvarnet import RecurrentVarNet 
# from direct.nn.recurrentvarnet.config import RecurrentVarNetConfig
# from direct.nn.recurrentvarnet.recurrentvarnet_engine import RecurrentVarNetEngine

from torch import nn 
import torch 
import  direct.data.transforms as T
from varnet import SensitivityModel
# jointicnetengine = JointICNetEngine(cfg = JointICNetConfig(), model = JointICNet(forward_operator= T.fft2, backward_operator= T.ifft2), device = 'cuda:0') 


# jointicnet = JointICNet(forward_operator= T.fft2, backward_operator= T.ifft2, use_norm_unet=True) 

class CustomMutiDomainNet(nn.Module): 

    def __init__(self):
        super().__init__()
        self.sens_net = SensitivityModel(4, 4) 
        self.multidomainnet = MultiDomainNet(forward_operator= T.fft2, backward_operator= T.ifft2, num_filters=12) 
        
    def forward(self, masked_kspace, mask) : 
        
        sensitivity_map = self.sens_net(masked_kspace, mask)
        x = self.multidomainnet(masked_kspace, mask, sensitivity_map) 
        result = T.root_sum_of_squares(x , 1, -1) # 1 H W
        
        # crop 
        height, width = result.shape[-2:]
        return result[..., (height - 384) // 2: 384 + (height - 384) // 2, (width - 384) // 2: 384 + (width - 384) // 2]



if __name__ == "__main__" :
    
    
    masked_kspace_sample = torch.randn(1, 16, 768, 396, 2).to('cuda:0')
    sens_net = SensitivityModel(4, 4).to('cuda:0') 
    mask = torch.randn(1, 1, 768, 396, 2).byte().float().to('cuda:0')
    
    target = torch.randn(1, 768, 396).to('cuda:0')
    sentivity_map = sens_net(masked_kspace_sample, mask).contiguous()
    print(sentivity_map.shape )
    
    # model = Myjointicnet().to('cuda:0')
    # model = RecurrentVarNetEngine(cfg = RecurrentVarNetConfig(), model = RecurrentVarNet(forward_operator= T.fft2, backward_operator= T.ifft2), device = 'cuda:0')
    
    model = MultiDomainNet(forward_operator= T.fft2, backward_operator= T.ifft2, num_filters=12).to('cuda:0') 
    
    print(sum(p.numel() for p in model.parameters() if p.requires_grad) + sum(p.numel() for p in sens_net.parameters() if p.requires_grad))
    
    
    
    x = model(masked_kspace_sample, sentivity_map)
    
    output_image = T.root_sum_of_squares(x , 1, -1)
    
    loss = torch.nn.MSELoss()(output_image, target) 
    loss.backward()
    print(x.shape)