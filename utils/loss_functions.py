from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
import os

import models
import torch
from torch import nn
import torch.nn.functional as F

class HiddenPrints:
    """
    Suppress all print statements

    > with HiddenPrints():
    >     print('hi') # nothing happens
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def check_input(x):
    oob_msg = 'image is outside the range [-1, 1] but got {} {}'
    assert x.min() >= -1.0 or x.max() <= 1.0, oob_msg.format(x.min(), x.max())
    assert x.size(0) == 1, 'only supports batch size 1 {}'.format(x.size())
    assert len(list(x.size())) == 4
    return


def check_loss_input(im0, im1, w):
    """ im0 is out and im1 is target and w is mask"""
    assert list(im0.size())[2:] == list(im1.size())[2:], 'spatial dim mismatch'
    assert (w is None) or (list(im0.size())[2:] == list(w.size())[2:]), 'spatial dim mismatch'

    if im1.size(0) != 1:
        assert im0.size(0) == im1.size(0), print(im0.size(0), im1.size(0))

    if w is not None and w.size(0) != 1:
        assert im0.size(0) == w.size(0)
    return


## -- Boolean statements -- ##

def is_single_1d(z):
    """ checks whether the vector has the form [1 x N] """
    if type(z) is not torch.Tensor:
        return False

    if z.size(0) == 1 and len(z.size()) == 2:
        return True
    return False


def l1_loss(out, target):
    """ computes loss = | x - y |"""
    return torch.abs(target - out)


def l2_loss(out, target):
    """ computes loss = (x - y)^2 """
    return ((target - out) ** 2)


def invertibility_loss(ims, target_transform, transform_params, mask=None):
    """ Computes invertibility loss MSE(ims - T^{-1}(T(ims))) """
    if ims.size(0) == 1:
        ims = ims.repeat(len(transform_params), 1, 1, 1)
    transformed = target_transform(ims, transform_params)
    inverted = target_transform(transformed, transform_params, invert=True)
    if mask is None:
        return torch.mean((ims - inverted) ** 2, [1, 2, 3])
    return masked_l2_loss(ims, inverted, mask)


def masked_l1_loss(out, target, mask):
    check_loss_input(out, target, mask)
    if mask.size(0) == 1:
        mask = mask.repeat(out.size(0), 1, 1, 1)
    if target.size(0) == 1:
        target = target.repeat(out.size(0), 1, 1, 1)

    loss = l1_loss(out, target)
    n = torch.sum(loss * mask, [1, 2, 3])
    d = torch.sum(mask, [1, 2, 3])
    return (n / d)


def masked_l2_loss(out, target, mask):
    check_loss_input(out, target, mask)
    if mask.size(0) == 1:
        mask = mask.repeat(out.size(0), 1, 1, 1)
    if target.size(0) == 1:
        target = target.repeat(out.size(0), 1, 1, 1)
    loss = l2_loss(out, target)
    n = torch.sum(loss * mask, [1, 2, 3])
    d = torch.sum(mask, [1, 2, 3])
    return (n / d)


class ReconstructionLoss(nn.Module):
    """ Reconstruction loss with spatial weighting """

    def __init__(self, loss_type='l1'):
        super(ReconstructionLoss, self).__init__()
        if loss_type in ['l1', 1]:
            self.loss_fn = l1_loss
        elif loss_type in ['l2', 2]:
            self.loss_fn = l2_loss
        else:
            raise ValueError('Unknown loss_type {}'.format(loss_type))
        return

    def __call__(self, im0, im1, w=None):
        check_loss_input(im0, im1, w)
        loss = self.loss_fn(im0, im1)
        if w is not None:
            n = torch.sum(loss * w, [1, 2, 3])
            d = torch.sum(w, [1, 2, 3])
            loss = n / d
        return loss


class PerceptualLoss(nn.Module):
    def __init__(self, net='vgg', use_gpu=True, precision='float'):
        """ LPIPS loss with spatial weighting """
        super(PerceptualLoss, self).__init__()
        with HiddenPrints():
            self.lpips = models.PerceptualLoss(model='net-lin',
                                               net=net,
                                               spatial=True,
                                               use_gpu=use_gpu)
        if use_gpu:
            self.lpips = nn.DataParallel(self.lpips).cuda()
        if precision == 'half':
            self.lpips.half()
        elif precision == 'float':
            self.lpips.float()
        elif precision == 'double':
            self.lpips.double()
        return

    def forward(self, im0, im1, w=None):
        """ ims have dimension BCHW while mask is 1HW """
        check_loss_input(im0, im1, w)
        # lpips takes the sum of each spatial map
        loss = self.lpips(im0, im1)
        if w is not None:
            n = torch.sum(loss * w, [1, 2, 3])
            d = torch.sum(w, [1, 2, 3])
            loss = n / d
        return loss

    def __call__(self, im0, im1, w=None):
        return self.forward(im0, im1, w)

    
def total_variation_loss(img, weight=1.0):
    variation_loss = weight * (
                    torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) +
                    torch.mean(torch.abs(img[:, :,  :-1, :] - img[:, :, 1:, :]))
                    )
    return variation_loss

def trimap_alpha_loss(trimap, alpha_map, mean=False):
    b1 = (trimap==1.0).double()
    b0 = (trimap==0).double()
    if (mean):
        b1_loss = torch.sum(torch.abs(b1 * (1-alpha_map))) / torch.sum(2*torch.abs(b1))
        b0_loss = torch.sum(torch.abs(b0 * alpha_map)) / torch.sum(2*torch.abs(b0))
    else:
        b1_loss = torch.sum(torch.abs(b1 * (1-alpha_map))) / (2*torch.sum(torch.abs(b1)))
        b0_loss = torch.sum(torch.abs(b0 * alpha_map)) / (2*torch.sum(torch.abs(b0)))
    t_a_loss = torch.sum(b1_loss + b0_loss)  # (1, 1, 256, 256)
    return t_a_loss

def adv_loss(frames, D, cv):
    """
    frames: shape(num_frames, 3, 256, 256), range [0,1]
    D: discriminator
    cv: class
    """
    # downsample because D is trained on 128x128
    frames = torch.nn.functional.interpolate(frames, size=128).float()
    # pass to the discriminator
    dis_fake = D(frames, cv)  # shape (num_frames, 1000)
    # loss
    adv_loss = -torch.mean(dis_fake)
    return adv_loss

# Crecit: https://github.com/simonmeister/UnFlow/blob/master/src/e2eflow/core/losses.py
# https://github.com/lliuz/ARFlow/blob/e92a8bbe66f0ced244267f43e3e55ad0fe46ff3e/losses/loss_blocks.py#L7
def TernaryLoss(im, im_warp, max_distance=1):
    """
    Note: this implementation does not include the occlusion mask.
    """
    patch_size = 2 * max_distance + 1

    def _rgb_to_grayscale(image):
        grayscale = image[:, 0, :, :] * 0.2989 + \
                    image[:, 1, :, :] * 0.5870 + \
                    image[:, 2, :, :] * 0.1140
        return grayscale.unsqueeze(1)

    def _ternary_transform(image):
        intensities = _rgb_to_grayscale(image) * 255
        out_channels = patch_size * patch_size
        w = torch.eye(out_channels).view((out_channels, 1, patch_size, patch_size))
        weights = w.type_as(im)
        patches = F.conv2d(intensities, weights, padding=max_distance)
        transf = patches - intensities
        transf_norm = transf / torch.sqrt(0.81 + torch.pow(transf, 2))
        return transf_norm

    def _hamming_distance(t1, t2):
        dist = torch.pow(t1 - t2, 2)
        dist_norm = dist / (0.1 + dist)
        dist_mean = torch.mean(dist_norm, 1, keepdim=True)  # instead of sum
        return dist_mean

    def _valid_mask(t, padding):
        n, _, h, w = t.size()
        inner = torch.ones(n, 1, h - 2 * padding, w - 2 * padding).type_as(t)
        mask = F.pad(inner, [padding] * 4)
        return mask

    t1 = _ternary_transform(im)
    t2 = _ternary_transform(im_warp)
    dist = _hamming_distance(t1, t2)
    mask = _valid_mask(im, max_distance)

    return dist * mask

def occlusion(flow_fw, flow_bw, thresh=1.0):
    """
    Use forward-backward consistency to get occlusion.
    flow_fw: forward flow t -> t + 1, shape (N, 2, H, W)
    flow_bw: backward flow t + 1 -> t, shape (N, 2, H, W)
    """

    
    coords0 = coords_grid(flow_fw.shape[0], flow_fw.shape[2], flow_fw.shape[3])
    if torch.cuda.is_available():
        coords0 = coords0.cuda()
    coords1 = coords0 + flow_fw
    coords2 = coords1 + bilinear_sampler(flow_bw, coords1.permute(0,2,3,1))

    err = (coords0 - coords2).norm(dim=1, keepdim=True)
    occ = (err > thresh).float()
    
#     del coords1, coords2
    
    return occ, err

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)

def SSIM(x, y, md=1):
    patch_size = 2 * md + 1
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(patch_size, 1, 0)(x)
    mu_y = nn.AvgPool2d(patch_size, 1, 0)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(patch_size, 1, 0)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(patch_size, 1, 0)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(patch_size, 1, 0)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d
    dist = torch.clamp((1 - SSIM) / 2, 0, 1)
    return dist

"""
Below from
https://github.com/lliuz/ARFlow/blob/e92a8bbe66f0ced244267f43e3e55ad0fe46ff3e/utils/warp_utils.py
"""
import inspect


def mesh_grid(B, H, W):
    # mesh grid
    x_base = torch.arange(0, W).repeat(B, H, 1)  # BHW
    y_base = torch.arange(0, H).repeat(B, W, 1).transpose(1, 2)  # BHW

    base_grid = torch.stack([x_base, y_base], 1)  # B2HW
    return base_grid


def norm_grid(v_grid):
    _, _, H, W = v_grid.size()

    # scale grid to [-1,1]
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / (W - 1) - 1.0
    v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / (H - 1) - 1.0
    return v_grid_norm.permute(0, 2, 3, 1)  # BHW2

def get_corresponding_map(data):
    """
    :param data: unnormalized coordinates Bx2xHxW
    :return: Bx1xHxW
    """
    B, _, H, W = data.size()

    # x = data[:, 0, :, :].view(B, -1).clamp(0, W - 1)  # BxN (N=H*W)
    # y = data[:, 1, :, :].view(B, -1).clamp(0, H - 1)

    x = data[:, 0, :, :].view(B, -1)  # BxN (N=H*W)
    y = data[:, 1, :, :].view(B, -1)

    # invalid = (x < 0) | (x > W - 1) | (y < 0) | (y > H - 1)   # BxN
    # invalid = invalid.repeat([1, 4])

    x1 = torch.floor(x)
    x_floor = x1.clamp(0, W - 1)
    y1 = torch.floor(y)
    y_floor = y1.clamp(0, H - 1)
    x0 = x1 + 1
    x_ceil = x0.clamp(0, W - 1)
    y0 = y1 + 1
    y_ceil = y0.clamp(0, H - 1)

    x_ceil_out = x0 != x_ceil
    y_ceil_out = y0 != y_ceil
    x_floor_out = x1 != x_floor
    y_floor_out = y1 != y_floor
    invalid = torch.cat([x_ceil_out | y_ceil_out,
                         x_ceil_out | y_floor_out,
                         x_floor_out | y_ceil_out,
                         x_floor_out | y_floor_out], dim=1)

    # encode coordinates, since the scatter function can only index along one axis
    corresponding_map = torch.zeros(B, H * W).type_as(data)
    indices = torch.cat([x_ceil + y_ceil * W,
                         x_ceil + y_floor * W,
                         x_floor + y_ceil * W,
                         x_floor + y_floor * W], 1).long()  # BxN   (N=4*H*W)
    values = torch.cat([(1 - torch.abs(x - x_ceil)) * (1 - torch.abs(y - y_ceil)),
                        (1 - torch.abs(x - x_ceil)) * (1 - torch.abs(y - y_floor)),
                        (1 - torch.abs(x - x_floor)) * (1 - torch.abs(y - y_ceil)),
                        (1 - torch.abs(x - x_floor)) * (1 - torch.abs(y - y_floor))],
                       1)
    # values = torch.ones_like(values)

    values[invalid] = 0

    corresponding_map.scatter_add_(1, indices, values)
    # decode coordinates
    corresponding_map = corresponding_map.view(B, H, W)

    return corresponding_map.unsqueeze(1)

def flow_warp(x, flow12, pad='border', mode='bilinear'):
    B, _, H, W = x.size()

    base_grid = mesh_grid(B, H, W).type_as(x)  # B2HW

    v_grid = norm_grid(base_grid + flow12)  # BHW2
    if 'align_corners' in inspect.getfullargspec(torch.nn.functional.grid_sample).args:
        im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad, align_corners=True)
    else:
        im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad)
    return im1_recons


def get_occu_mask_bidirection(flow12, flow21, scale=0.01, bias=0.5):
    flow21_warped = flow_warp(flow21, flow12, pad='zeros')
    flow12_diff = flow12 + flow21_warped
    mag = (flow12 * flow12).sum(1, keepdim=True) + \
          (flow21_warped * flow21_warped).sum(1, keepdim=True)
    occ_thresh = scale * mag + bias
    occ = (flow12_diff * flow12_diff).sum(1, keepdim=True) > occ_thresh
    return occ.float()


def get_occu_mask_backward(flow21, th=0.2):
    B, _, H, W = flow21.size()
    base_grid = mesh_grid(B, H, W).type_as(flow21)  # B2HW

    corr_map = get_corresponding_map(base_grid + flow21)  # BHW
    occu_mask = corr_map.clamp(min=0., max=1.) < th
    return occu_mask.float()

if __name__ == '__main__':
    im = torch.rand(5, 3, 256, 256) * 255
    im_warp = torch.rand(5, 3, 256, 256) * 255
    census_loss = TernaryLoss(im, im_warp)
    
#     print(census_loss)
    print(census_loss.shape)
    