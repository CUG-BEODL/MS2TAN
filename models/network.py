import torch
import torch.nn as nn
import utils.pytorch_ssim as pssim
from utils.util_image import *
from utils.util_metrics import *
# from utils.util_tiff import *
# from torchvision import models, transforms

from .attention import MFE


def init_weights(net, init_type='kaiming', gain=0.02):
    from torch.nn import init

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0.2, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


class MS2TAN(nn.Module):
    def __init__(
        self,
        # 180
        dim_list=[768, 640, 512],
        num_frame=12,
        image_size=180,
        patch_list=[15, 12, 10],
        in_chans=6,
        out_chans=6,
        depth_list=[2, 2, 2],
        heads_list=[4, 6, 8],
        dim_head_list=[8, 16, 16],
        attn_dropout=0.0,
        ff_dropout=0.0,
        optim_input=False,
        missing_mask=True,
        enable_model=True,
        enable_conv=False,
        enable_mse=True,
        enable_struct=False,
        enable_percept=False,
    ):
        super().__init__()
        assert (enable_mse or enable_struct or enable_percept)
        self.num_block = len(dim_list)
        self.in_chans = in_chans
        self.blocks = nn.ModuleList(
            [
                MFE(
                    dim=dim_list[i],
                    num_frames=num_frame,
                    image_size=image_size,
                    patch_size=patch_list[i],
                    in_channels=in_chans,
                    out_channels=out_chans,
                    depth=depth_list[i],
                    heads=heads_list[i],
                    dim_head=dim_head_list[i],
                    attn_dropout=attn_dropout,
                    ff_dropout=ff_dropout,
                    missing_mask=(i == 0 and missing_mask),
                    diag_mask=False,
                    # 第一个block用missing_mask
                ) if enable_model else None
                for i in range(self.num_block)
            ]
        )
        self.out_chans = out_chans
        self.optim_input = optim_input
        self.enable_conv = enable_conv
        self.enable_model = enable_model
        self.enable_mse = enable_mse
        self.enable_struct = enable_struct
        self.enable_percept = enable_percept

        if self.enable_conv:
            # 维度变换
            in_ch, out_ch, cnum = out_chans, out_chans, 32
            self.first_conv = nn.Sequential(
                nn.Flatten(0, 1),
                nn.Conv2d(in_ch, cnum, 5, 1, 2),
                nn.BatchNorm2d(cnum),
                nn.PReLU(cnum),
                nn.Conv2d(cnum, 2*cnum, 3, 1, 1),
                nn.PReLU(2*cnum),
                nn.Conv2d(2*cnum, 2*cnum, 1, 1, 0),
                nn.BatchNorm2d(2*cnum),
                nn.PReLU(2*cnum),
                nn.Conv2d(2*cnum, 2*cnum, 3, 1, 1),
                nn.BatchNorm2d(2*cnum),
                nn.PReLU(2*cnum),
                nn.Conv2d(2*cnum, out_ch, 5, 1, 2),
                nn.PReLU(out_ch),
                nn.Unflatten(0, (-1, num_frame)),
            )

            conv_inner_dim = 32
            self.after_conv = nn.ModuleList([nn.Sequential(
                nn.Flatten(0, 1),
                nn.Conv2d(out_chans, conv_inner_dim, 3, 1, 1),
                nn.BatchNorm2d(conv_inner_dim),
                nn.PReLU(conv_inner_dim),
                nn.Conv2d(conv_inner_dim, conv_inner_dim, 1, 1, 0),
                nn.BatchNorm2d(conv_inner_dim),
                nn.PReLU(conv_inner_dim),
                nn.Conv2d(conv_inner_dim, out_chans, 3, 1, 1),
                nn.Unflatten(0, (-1, num_frame)),
            ) for _ in range(self.num_block)])

        self.SSIM = pssim.SSIM()
        # self.TV = TVLoss()

    def forward(self, X, extend_layers, y, mode="val", return_attn=False):
        b, t, c, h, w = X.shape
        block_out = []
        # extend_layers = X[:, :, -2:, :, :]
        obs_mask, art_mask = extend_layers
        # get mean_face
        mean_face = calc_mean_face(X, obs_mask)
        y_mean_face = calc_mean_face(y, obs_mask+art_mask)
        # expand
        obs_mask = obs_mask.expand(-1, -1, c, -1, -1)
        art_mask = art_mask.expand(-1, -1, c, -1, -1)
        # opt_X = 观测值 + mean_face
        opt_X = X.clone()
        # 将缺失值替换
        opt_X[obs_mask == 0] = mean_face[obs_mask == 0]
        # print(opt_X.shape)
        opt_y = y.clone()
        opt_y[(obs_mask+art_mask) == 0] = y_mean_face[(obs_mask+art_mask) == 0]

        out = opt_X if self.optim_input else X

        if self.enable_conv:
            out = out + self.first_conv(out)

        block_attn = []

        for idx, block in enumerate(self.blocks):
            if self.enable_model:
                merge = out

                if idx != -1:
                    if return_attn:
                        res_out, attn_list = block(
                            merge, extend_layers[0], return_attn=True)
                        out = out + res_out
                        block_attn.append(attn_list)
                    else:
                        out = out + block(merge, extend_layers[0])
                else:
                    out = block(merge, extend_layers[0])
                    raise NotImplementedError("idx == -1")

            if self.enable_conv:
                out = out + self.after_conv[idx](out)

            block_out.append(out)

        raw_out = block_out[-1]

        if self.enable_mse:
            C_obs, C_art = 1, 3
        else:
            C_obs, C_art = 0, 0

        C_ssim = 0.05
        C_percept = 0.05

        if mode == "train":
            loss_obs = 0
            loss_art = 0
            for idx, view in enumerate(block_out):
                loss_obs += masked_rmse_cal(view, y, obs_mask)
                if idx != self.num_block-1:
                    loss_art += masked_rmse_cal(view, y, art_mask)
                else:
                    # loss_art += masked_rmse_cal(view, y, art_mask) * self.num_block
                    loss_art += masked_rmse_cal(view, y, art_mask)
            loss_obs /= self.num_block
            # loss_art /= (self.num_block*2 - 1)
            loss_art /= self.num_block
            # loss_art = masked_rmse_cal(raw_out, y, art_mask)
            # loss_tv = self.TV(raw_out.flatten(0, 1)/y.max())
            # 去除从未观测到的像素
            raw_out[(obs_mask+art_mask) == 0] = opt_y[(obs_mask+art_mask) == 0]
            if self.enable_percept:
                loss_percept = self.percept_loss(
                    raw_out[:, :, :], opt_y[:, :, :])
            else:
                loss_percept = torch.as_tensor(0.).cuda()
            if self.enable_struct:
                loss_ssim = self.SSIM(
                    raw_out.reshape(-1, c, h, w), opt_y.reshape(-1, c, h, w))
            else:
                loss_ssim = torch.as_tensor(0.1).cuda()

            loss_all = (loss_obs * C_obs + loss_art * C_art) / (C_obs + C_art)
            loss_all += loss_percept * C_percept + C_ssim * (1. - loss_ssim)
            return {
                'raw_out': raw_out,
                'loss_all': loss_all,
                'loss_obs': loss_obs,
                'loss_art': loss_art,
                'loss_ssim': loss_ssim,
                'loss_percept': loss_percept,
                'block_attn': block_attn,
            }
        elif mode == 'val':
            loss_obs = 0
            loss_art = 0
            for idx, view in enumerate(block_out):
                loss_obs += masked_rmse_cal(view, y, obs_mask)
                if idx != self.num_block-1:
                    loss_art += masked_rmse_cal(view, y, art_mask)
                else:
                    # loss_art += masked_rmse_cal(view, y, art_mask) * self.num_block
                    loss_art += masked_rmse_cal(view, y, art_mask)
            loss_obs /= self.num_block
            loss_art /= self.num_block

            cmp_out = raw_out.clone()
            # 计算指标时忽略标签缺失的像素
            cmp_out[(obs_mask+art_mask) == 0] = 0
            if self.enable_percept:
                loss_percept = self.percept_loss(
                    cmp_out[:, :, :], opt_y[:, :, :])
            else:
                loss_percept = torch.as_tensor(0.).cuda()
            if self.enable_struct:
                loss_ssim = self.SSIM(
                    cmp_out.reshape(-1, c, h, w), opt_y.reshape(-1, c, h, w))
            else:
                loss_ssim = torch.as_tensor(0.1).cuda()

            loss_all = (loss_obs * C_obs + loss_art * C_art) / (C_obs + C_art)
            loss_all += loss_percept * C_percept + C_ssim * (1. - loss_ssim)

            replace_out = raw_out.clone()
            replace_out[obs_mask != 0] = opt_X[obs_mask != 0]
            return {
                'hist_list': block_out,
                'raw_out': raw_out,
                # 'cmp_out': cmp_out,
                "replace_out": replace_out,
                # "block_out": block_out,
                'loss_all': loss_all,
                'loss_obs': loss_obs,
                'loss_art': loss_art,
                'loss_ssim': loss_ssim,
                'loss_percept': loss_percept,
                'opt_X': opt_X,
                'opt_y': opt_y,
                'mean_face': mean_face,
                'y_mean_face': y_mean_face,
                'block_attn': block_attn,
            }
        else:
            assert False, f'Error mode {mode}'
