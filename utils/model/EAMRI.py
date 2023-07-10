"""
official network architecture for EAMRI
author: huihui
modified by: jinpil Choi
"""

import sys
sys.path.insert(0, '..')

import torch
from torch import nn
import transforms_simple_eamri as T
from eamri_networkutil import dilatedConvBlock, DC_multicoil, rearrange, LayerNorm, SensitivityModel, default_conv



class rdn_convBlock(nn.Module):
    def __init__(self, convNum=3, recursiveTime=3, inChannel=2, midChannel=16, shift=False):
        super(rdn_convBlock, self).__init__()
        self.rTime = recursiveTime
        self.LRelu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(inChannel, midChannel, 3, padding=1)
        self.dilateBlock = dilatedConvBlock(convNum, midChannel)
        self.conv2 = nn.Conv2d(midChannel, inChannel, 3, padding=1)
        self.dc = DC_multicoil(shift=shift)

    def forward(self, x1, y, m, sens_map):
        x2 = self.conv1(x1)
        x2 = self.LRelu(x2)
        xt = x2
        for i in range(self.rTime):
            x3 = self.dilateBlock(xt)
            xt = x3 + x2
        x4 = self.conv2(xt)
        x4 = self.LRelu(x4)
        x5 = x4 + x1  # (B, 2, H, W)

        # dc
        x5 = self.dc(x5, y, m, sens_map)  # (B, coils, H, W, 2)

        return x5


class MSRB(nn.Module):
    def __init__(self, n_feats):
        """
        n_feats: input dimension
        """
        super(MSRB, self).__init__()

        kernel_size_1 = 3
        kernel_size_2 = 5

        # stage 1
        self.conv_3_1 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, bias=True)
        self.conv_5_1 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=2, bias=True, dilation=2)
        self.fuse1 = nn.Conv2d(2 * n_feats, n_feats, kernel_size=1, bias=True)

        # stage 2
        self.conv_3_2 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, bias=True)
        self.conv_5_2 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=2, dilation=2, bias=True)  # 7*7 conv

        self.confusion = nn.Conv2d(n_feats * 2, n_feats, 1, padding=0, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        input_1 = x
        output_3_1 = self.conv_3_1(input_1)  # 3*3 conv
        output_5_1 = self.conv_5_1(input_1)  # 3*3 conv
        input_2 = self.relu(torch.cat([output_3_1, output_5_1], 1))
        input_2 = self.fuse1(input_2)  # 1*1 conv

        output_3_2 = self.conv_3_2(input_2)
        output_5_2 = self.conv_5_2(input_2)

        input_3 = self.relu(torch.cat([output_3_2, output_5_2], 1))
        output = self.confusion(input_3)

        output += x

        return output


class Edge_Net(nn.Module):
    def __init__(self, indim, hiddim, conv=default_conv, n_MSRB=3):

        super(Edge_Net, self).__init__()

        kernel_size = 3
        self.n_MSRB = n_MSRB

        modules_head = [conv(2, hiddim, kernel_size)]  # 3*3 conv
        # body
        modules_body = nn.ModuleList()
        for i in range(n_MSRB):
            modules_body.append(MSRB(hiddim))

        # tail
        modules_tail = [nn.Conv2d(hiddim * (self.n_MSRB + 1), hiddim, 1, padding=0, stride=1),
                        conv(hiddim, 1, kernel_size)]

        self.Edge_Net_head = nn.Sequential(*modules_head)
        self.Edge_Net_body = nn.Sequential(*modules_body)
        self.Edge_Net_tail = nn.Sequential(*modules_tail)

    def forward(self, x):

        """
        x: (B, 2, H, W)
        """
        x = self.Edge_Net_head(x)  # (B, hiddim, H, W)
        res = x

        MSRB_out = []
        for i in range(self.n_MSRB):
            x = self.Edge_Net_body[i](x)
            MSRB_out.append(x)
        MSRB_out.append(res)

        res = torch.cat(MSRB_out, 1)  # (B, hiddim*(self.n_MSRB+1), H, W)
        x = self.Edge_Net_tail(res)

        return x


# ==============================

class attHead_image(nn.Module):
    """
    norm + 1*1 conv + 3*3 dconv
    """

    def __init__(self, C, C1, layernorm_type='BiasFree', bias=False):
        """
        C: input channel
        C1: output channel after 1*1 conv

        """
        super(attHead_image, self).__init__()
        self.norm = LayerNorm(C, layernorm_type)
        self.conv1 = nn.Conv2d(C, C1, kernel_size=1, bias=bias)
        self.conv2 = nn.Conv2d(C1, C1, kernel_size=3, stride=1, padding=1, groups=C1, bias=bias)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        x1 = self.conv1(x)  # (B, C1, H, W)
        x1 = self.conv2(x1)  # (B, C1, H, W)
        return x1


class EEM(nn.Module):
    def __init__(self, C, C1, C2, num_heads, bias):
        """
        edge enhance module (EEM)
        C: input channel of image
        C1: input channel of edge
        C2 : output channel of imhead/ehead
        """

        super(EEM, self).__init__()

        self.imhead = attHead_image(2, 2 * C2)
        self.ehead = nn.Conv2d(1, C2, kernel_size=3, stride=1, padding=1, groups=1, bias=bias)

        self.num_heads = num_heads

        self.a1 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.project_out = nn.Conv2d(C2, 2, kernel_size=1, bias=bias)

    def edge_att(self, x, e):
        """
        edge attention
        x: input image (B, C, H, W)
        e: input edge (B, C1, H, W)
        """

        _, _, H, W = x.shape

        # =================================
        # high freq, edge
        # split into q, k, v
        q1 = self.imhead(x)  # (B, 2*C2, H, W)
        k_eg = self.ehead(e)  # (B, C2, H, W)
        q_im, v_im = q1.chunk(2, dim=1)

        q_im = rearrange(q_im, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_eg = rearrange(k_eg, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_im = rearrange(v_im, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # (B, head, C, H*W)

        q_im = torch.nn.functional.normalize(q_im, dim=-1)
        k_eg = torch.nn.functional.normalize(k_eg, dim=-1)

        attn = (q_im @ k_eg.transpose(-2, -1)) * self.a1  # (B, head, C, C)
        attn = attn.softmax(dim=-1)
        out = (attn @ v_im)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)

        # skip connection
        out = x + self.project_out(out)  # (B, 2, H, W)

        return out.contiguous()

    def forward(self, x, e):
        return self.edge_att(x, e)


class EAM(nn.Module):
    def __init__(self, C, C1, C2, num_heads, bias, shift=False):
        super(EAM, self).__init__()

        self.b1 = EEM(C, C1, C2, num_heads, bias)
        self.dc = DC_multicoil(shift=shift)

    def forward(self, x, e, y, m, sens_map=None):
        xout = self.b1(x, e)  # (B, 2, H, W)
        xout = self.dc(xout, y, m, sens_map)  # (B, coils, H, W, 2)

        return xout


class EAMRI(nn.Module):
    """
    12 DAM + transformer block
    """
    def __init__(self,
                 indim=2,
                 edgeFeat=16,
                 attdim=4,
                 num_head=4,
                 fNums=[16, 16, 16, 16, 16],
                 num_iters=[3, 3, 1, 3, 3],
                 n_MSRB=1,
                 shift=False):
        """
        input:
            C: number of conv in RDB, default 3
            G0: input_channel, default 2
            G1: output_channel, default 2
            n_RDB: number of RDBs

        """
        super(EAMRI, self).__init__()

        # sens_net
        self.sens_net = SensitivityModel(chans=8, num_pools=4, shift=shift)

        # image module
        self.imHead = rdn_convBlock(inChannel=indim, midChannel=fNums[0], recursiveTime=num_iters[0], shift=shift)
        self.net1 = rdn_convBlock(inChannel=indim, midChannel=fNums[1], recursiveTime=num_iters[1], shift=shift)
        self.net2 = rdn_convBlock(inChannel=indim, midChannel=fNums[2], recursiveTime=num_iters[2], shift=shift)
        self.net3 = rdn_convBlock(inChannel=indim, midChannel=fNums[3], recursiveTime=num_iters[3], shift=shift)
        self.net4 = rdn_convBlock(inChannel=indim, midChannel=fNums[4], recursiveTime=num_iters[4], shift=shift)

        # edge module
        self.edgeNet = Edge_Net(indim=indim, hiddim=edgeFeat, n_MSRB=n_MSRB)  # simple edge block

        # edgeatt module
        self.fuse1 = EAM(indim, 1, attdim, num_head, bias=False, shift=shift)
        self.fuse2 = EAM(indim, 1, attdim, num_head, bias=False, shift=shift)
        self.fuse3 = EAM(indim, 1, attdim, num_head, bias=False, shift=shift)
        self.fuse4 = EAM(indim, 1, attdim, num_head, bias=False, shift=shift)

    def reduce(self, x, sens_map):
        x1 = T.reduce_operator(x, sens_map, dim=1)  # (B, H, W, 2)
        x1 = x1.permute(0, 3, 1, 2).contiguous()  # (B,2,H,W)
        return x1

    def forward(self, masked_kspace, mask):  # (image, kspace, mask)
        """
        :param masked_kspace: (B, coils, H, W, 2)
        :param mask: (B, 1, 1, W, 1)
        """
        # data consistency
        x1 = T.ifft2(masked_kspace)  # (B, coils, H, W, 2)      # zero-filled image (B, coils, H, W, 2)

        # stack mask to be (B, 1, H, W, 1)
        m = mask.repeat(1, 1, x1.shape[-3], 1, 1)  # (B, 1, H, W, 1)

        # estimated sens map
        sens_map = self.sens_net(masked_kspace, m)

        x1 = self.reduce(x1, sens_map)  # (B, 2, H, W)

        # image head
        x1 = self.imHead(x1, masked_kspace, m, sens_map)  # (B, 2, H, W)
        

        # first stage
        x2 = self.net1(x1, masked_kspace, m, sens_map)  # (B, 2, H, W)
        e2 = self.edgeNet(x1)  # (B, 1, H, W)
        x1 = self.fuse1(x2, e2, masked_kspace, m, sens_map)
        

        # second stage
        x2 = self.net2(x1, masked_kspace, m, sens_map)  # (B, 2, H, W)
        e3 = self.edgeNet(x1)
        x1 = self.fuse2(x2, e3, masked_kspace, m, sens_map)
         

        # third stage
        x2 = self.net3(x1, masked_kspace, m, sens_map)  # (B, 2, H, W)
        e4 = self.edgeNet(x1)
        x1 = self.fuse3(x2, e4, masked_kspace, m, sens_map)
        
        x2 = self.net4(x1, masked_kspace, m, sens_map)  # (B, 2, H, W)
        e5 = self.edgeNet(x1)
        result = self.fuse4(x2, e5, masked_kspace, m, sens_map)
        height = result.shape[-2]
        width = result.shape[-1]
        
        result = result[..., (height - 384) // 2: 384 + (height - 384) // 2,
                 (width - 384) // 2: 384 + (width - 384) // 2]  # (B, 2, 384, 384)

        result = (result ** 2).sum(dim=1).sqrt().unsqueeze(dim = 1) # (B, H, W)

        return torch.stack([e2,e3,e4,e5], dim = 0)[..., (height - 384) // 2: 384 + (height - 384) // 2,
                 (width - 384) // 2: 384 + (width - 384) // 2], result


if __name__ == "__main__":

    ...

    # original flow input

    masked_kspace = torch.randn(1, 14, 768, 396, 2) # b coils h w 2
    mask = torch.randn(1, 1, 1, 396, 1).byte() # b 1 1 w 1

    from fastmri import fft2c
    image_complex = fft2c(masked_kspace)

    # simply inverse fft of masked_kspace
    # image_complex = torch.randn(1, 14, 768, 396, 2) # b 1 h w



    # (zf, subF, mask_var) as input


    # model

    model = EAMRI(indim=2, edgeFeat=24, attdim=32, num_head=4, num_iters=[1,3,3,3,3], fNums=[48,96,96,96,96], n_MSRB=3, shift=True)

    # print model number of parameters

    print(sum(p.numel() for p in model.parameters()))

    output = model(masked_kspace, mask)

    for i in range(len(output)):
        print(output[i].shape)

