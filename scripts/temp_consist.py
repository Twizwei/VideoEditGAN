"""
This code is for temporal consistency with a hierarchical connection.
Each batch we focous on two frames, to mitigate computation we only pick some special pairs following a hierarchical connection.
Note this approach has no anchor so regularization is added.
Examples:
In-Domain:
CUDA_VISIBLE_DEVICES=0 python -W ignore scripts/temp_consist_hier.py --edit_root /work/cascades/yiranx/codes/PTI/exps/Actor_08-01-02-03-01-02-01-08/StyleCl
ip --metadata_root /work/cascades/yiranx/codes/PTI/exps/Actor_08-01-02-03-01-02-01-08/ --checkpoint_path /work/cascades/yiranx/codes/PTI/checkpoints/model_Actor_08-01-02-03-01-02-01-08_multi_id.p
t --original_root /work/cascades/yiranx/datasets/RAVDESS/frames/Actor_08/01-02-03-01-02-01-08 --batch_size 1 --reg_frame 100.0 --weight_cycle 0.0 --weight_tv_flow 0.0 --lr 1e-3 --weight_photo 1.0
 --reg_G 100.0 --lr_G 1e-4 --weight_out_mask 0.0 --tune_G --epochs_G 10 --scale_factor 4 --in_domain
Out-of-domain:
CUDA_VISIBLE_DEVICES=0 python -W ignore scripts/temp_consist_hier_multivideo.py --checkpoint_path /work/cascades/yiranx/codes/StyleGAN-nada/pretrain_directions --batch_size 2 --reg_frame 100.0 --weight_cycle 0.0 --weight_tv_flow 0.0 --lr 1e-5 --weight_photo 1.0 --reg_G 200.0 --lr_G 2e-4 --weight_out_mask 0.0 --tune_G --epochs_G 10 --tune_w --epochs_w 5 --scale_factor 4
"""
import os
import glob

import argparse
from argparse import Namespace
from tqdm import tqdm
import numpy as np
import torch
from torchvision.transforms.functional import perspective
from torchvision import transforms
from PIL import Image
import lpips
import matplotlib.pyplot as plt 


import sys
import imageio
import cv2

sys.path.append(".")
sys.path.append("..")

from options.test_options import TestOptions
from models.stylegan2.model import Generator
from models.StyleCLIP.mapper.styleclip_mapper import StyleCLIPMapper
from utils.latent_data import LatentDataset, collate_fn
from utils.flow_utils import flow_warp
import utils.loss_functions as LF
import utils.flow_viz as flow_viz

import pdb
import time

from RAFT import RAFT
# from RAFT.utils import InputPadder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_tuned_G(model_path, need_grad=False):
    new_G_path = model_path
    with open(new_G_path, 'rb') as f:
        new_G = torch.load(f).to(device).eval()
    new_G = new_G.float()
    # toogle_grad(new_G, need_grad)
    return new_G

def toogle_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def read(im_path, as_transformed_tensor=False, im_size=256, transform_style=None):
    im = np.array(Image.open(im_path).convert("RGB"))
    h, w = im.shape[:2]

    if np.max(im) <= 1. + 1e-6:
        im = (im * 255).astype(np.uint8)

    im = Image.fromarray(im)

    if as_transformed_tensor:

        if transform_style == 'biggan':
            transform = transforms.Compose(
                [
                    transforms.Resize(im_size),
                    transforms.CenterCrop(im_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )

        elif transform_style in ['stylegan', 'stylegan2']:
            if h < w:
                pad_top = (w - h) // 2
                pad_bot = w - h - pad_top
                pad_left, pad_right = 0, 0
            else:
                pad_left = (h - w) // 2
                pad_right = h - w - pad_left
                pad_top, pad_bot = 0, 0

            transform = transforms.Compose([
                    transforms.Pad((pad_left, pad_top, pad_right, pad_bot)),
                    transforms.Resize(im_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                    ])

        elif transform_style == 'original':
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )

        elif transform_style is None:
            transform = transforms.Compose([
                    transforms.Resize(im_size),
                    transforms.CenterCrop(im_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                    ])

        else:
            raise ValueError(f'unknown transformation style {transform_style}')

    return transform(im)

def to_image(output, to_cpu=True, denormalize=True, jpg_format=True,
             to_numpy=True, cv2_format=True):
    """ Formats torch tensor in the form BCHW -> BHWC """
    is_batched = True
    if len(list(output.size())) == 3:
        output = output.unsqueeze(0)
        is_batched = False
    tmp = output.detach().float()
    if to_cpu:
        tmp = tmp.cpu()

    tmp = tmp.permute(0, 2, 3, 1)
    if denormalize:
        tmp = (tmp + 1.0) / 2.0
    if jpg_format:
        tmp = (tmp * 255).int()
    if cv2_format and output.size(1) > 1:
        tmp = tmp[:, :, :, [2, 1, 0]]
    if to_numpy:
        tmp = tmp.numpy()
    if not is_batched:
        return tmp.squeeze(0)
    return tmp


def output_frame(output_dir, image, itr, filename='mask', t=None, transform_fn=None, frame_id=5):
    image = image[frame_id]
    os.makedirs(os.path.join(output_dir, 'vis'), exist_ok=True)
    if t is not None and transform_fn is not None:
        inv_im = to_image(transform_fn(image.cpu().unsqueeze(0), t.cpu()[frame_id].unsqueeze(0), invert=True))[0]
    if image.shape[0] > 1:
        inv_im = to_image(image.cpu().unsqueeze(0))[0]
    else:
        # mask
        inv_im = to_image(image.cpu().unsqueeze(0), denormalize=False)[0]
        inv_im = np.repeat(inv_im, 3, axis=2)
    jpg_quality = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
        
    cv2.imwrite(os.path.join(output_dir, 'vis', filename + '-{:05}-niter{:05}.jpg'.format(frame_id, itr)), inv_im, jpg_quality)
    return inv_im

def output_flow(output_dir, itr, flo, frame_id=1, forward=True):
    flo = flo[frame_id:frame_id+1][0].permute(1,2,0).clone().detach().cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = flo

    import matplotlib.pyplot as plt
    if forward:
        plt.imsave(os.path.join(output_dir, 'vis', 'flow_for' + '-{:05}-niter{:05}.jpg'.format(frame_id, itr)), img_flo/255.0)
    else:
        plt.imsave(os.path.join(output_dir, 'vis', 'flow_back' + '-{:05}-niter{:05}.jpg'.format(frame_id, itr)), img_flo/255.0)
    return img_flo    

def initialize_RAFT(args):
    """Initializes the RAFT model.
    """
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to('cuda')
    model.eval()

    return model


def calculate_flow(model, video, mode='forward'):
    """Calculates optical flow.
    """
    if mode not in ['forward', 'backward']:
        raise NotImplementedError

    nFrame, _, imgH, imgW = video.shape
    Flow = []
    for i in range(video.shape[0] - 1):
        if mode == 'forward':
            # Flow i -> i + 1
            image1 = video[i, None]
            image2 = video[i + 1, None]
        elif mode == 'backward':
            # Flow i + 1 -> i
            image1 = video[i + 1, None]
            image2 = video[i, None]
        else:
            raise NotImplementedError

        _, flow = model(image1, image2, iters=20, test_mode=True)
        Flow.append(flow)
        
    return Flow

def calc_reg_G(new_G, old_G, fixed_w, ploss_fn, num_of_sampled_latents=1, regulizer_l2_lambda=1.0, regulizer_lpips_lambda=1.0, in_domain=False):
    loss = 0.0
    
    if in_domain:
        z_samples = np.random.randn(num_of_sampled_latents, old_G.z_dim)
        w_samples = old_G.mapping(torch.from_numpy(z_samples).to(device), None,
                                        truncation_psi=0.5)
    else:
        w_samples = old_G([torch.randn(num_of_sampled_latents, old_G.style_dim, dtype=torch.float).to(device)], 
                            truncation=1.0, randomize_noise=False, only_return_latents=True)   # TODO
    territory_indicator_ws = [get_morphed_w_code(w_code.unsqueeze(0), fixed_w) for w_code in w_samples]
    
    for w_code in territory_indicator_ws:
        if in_domain:
            new_img = new_G.synthesis(w_code, noise_mode='none', force_fp32=True)
            with torch.no_grad():
                old_img = old_G.synthesis(w_code, noise_mode='none', force_fp32=True)
        else:
            new_img, _ = new_G([w_code], input_is_latent=True, randomize_noise=False)
            with torch.no_grad():
                old_img, _ = old_G([w_code], input_is_latent=True, randomize_noise=False)

        if regulizer_l2_lambda > 0:
            l2_loss_val = (old_img - new_img) ** 2
            l2_loss_val = torch.mean(torch.squeeze(l2_loss_val))
            loss += l2_loss_val * regulizer_l2_lambda

        if regulizer_lpips_lambda > 0:
            loss_lpips = ploss_fn(old_img, new_img).mean(dim=2,keepdim=True).mean(dim=3,keepdim=True)
            loss_lpips = torch.mean(torch.squeeze(loss_lpips))
            loss += loss_lpips * regulizer_lpips_lambda

    return loss / len(territory_indicator_ws)

def get_morphed_w_code(new_w_code, fixed_w, morphing_regulizer_alpha=30.0):
    interpolation_direction = new_w_code.repeat(fixed_w.shape[0], 1, 1) - fixed_w
    interpolation_direction_norm = torch.norm(interpolation_direction, p=2)
    direction_to_move = morphing_regulizer_alpha * interpolation_direction / interpolation_direction_norm
    result_w = fixed_w + direction_to_move
    return result_w

def to_original(frames_aligned, masks_aligned, cropped_padded_imgs, quads, quad_masks, ori_imgs, crop_coords):
    """
    Map aligned faces back to the original input space.
    Inputs:
    :param frames_aligned (BS, 3, H, W)
    :param masks_aligned (BS, 1, H, W)
    :param cropped_padded_imgs [[pair0], [pair1], ..., [pairBS-1]], each pair: [img0, img1]
    :param quads (BS, 2, 4, 2)
    :param quad_masks [[pair0], [pair1], ..., [pairBS-1]], each pair: [mask0, mask1]
    :param ori_imgs (BS, 2, 3, H_ori, W_ori)
    :param crop_coords (BS, 2, 4)
    """
    frames_unaligned = []
    masks_unaligned = []

    # ori_imgs = ori_imgs.reshape(-1, *ori_imgs.shape[2:])  # (BS*2, 3, H_ori, W_ori)
    # quads = quads.reshape(-1, *quads.shape[1:])
    # crop_coords = crop_coords.reshape(-1, *crop_coords.shape[1:])
    
    frames_aligned = frames_aligned.reshape(-1, 2, *frames_aligned.shape[1:])
    masks_aligned = masks_aligned.reshape(-1, 2, *masks_aligned.shape[1:])

    BS = frames_aligned.shape[0]
    for batch_itr in range(BS):
        for frame_itr in range(2):
            quad_coords = quads[batch_itr][frame_itr]
            quad_mask = quad_masks[batch_itr][frame_itr].to(device)
            cropped_padded = cropped_padded_imgs[batch_itr][frame_itr].to(device)
            crop_coord = crop_coords[batch_itr][frame_itr]
            ori_img = ori_imgs[batch_itr][frame_itr:frame_itr+1].to(device)
            
            frame_out = perspective(frames_aligned[batch_itr][frame_itr:frame_itr+1], [(0, 0), (0, 1024), (1024, 1024), (1024, 0)], quad_coords)[:,:,:quad_mask.shape[2], :quad_mask.shape[3]]
            frame_out = quad_mask * frame_out + (1.0 - quad_mask) * cropped_padded  # cropped and padded space
            mask_ = perspective(masks_aligned[batch_itr][frame_itr:frame_itr+1], [(0, 0), (0, 1024), (1024, 1024), (1024, 0)], quad_coords)[:,:,:quad_mask.shape[2], :quad_mask.shape[3]]
            
            # make a paste mask
            paste_mask = torch.zeros((1, 1, ori_img.shape[-2], ori_img.shape[-1]))
            paste_mask[:, :, crop_coord[1]:crop_coord[3], crop_coord[0]:crop_coord[2]] = 1.0
            paste_mask = paste_mask.to(device)
            # if crop_coord[0] + (ori_img.shape[-1] - crop_coord[2]) + frame_out.shape[-1] < 1280:
            padding = (1280 - frame_out.shape[-1] - (ori_img.shape[-1] - crop_coord[2]), ori_img.shape[-1]-crop_coord[2], 720 - frame_out.shape[-2] - (ori_img.shape[-2]-crop_coord[3]), ori_img.shape[-2]-crop_coord[3])
            frame_out = torch.nn.functional.pad(frame_out, padding, 'constant', 0)
            frame_out_ = paste_mask * frame_out + (1.0 - paste_mask) * ori_img
            mask_ = torch.nn.functional.pad(mask_, padding, 'constant', 0)
            frames_unaligned.append(frame_out_)
            masks_unaligned.append(mask_)

    frames_unaligned = torch.stack(frames_unaligned).squeeze(1)
    masks_unaligned = torch.stack(masks_unaligned).squeeze(1)
    return frames_unaligned, masks_unaligned

def latents_adjust(
                    RAFT_model, latent_anchor, anchor_id, data_loader, G, mapper, lr, epochs, optimizer, output_dir, latent_code_update,
                    weight_photo, weight_out_mask, weight_cycle, weight_tv_flow, reg_frame, ploss_fn, 
                    scale_factor=2, in_domain=False, delta_w_step=0.12,
                    ):
    photo_losses = []
    n_iter = 0
    

    with torch.no_grad():
        if in_domain:
            frame_anchor = G.synthesis(latent_anchor, noise_mode='const', force_fp32=True)
            aligned_ori_anchor = read(sorted(glob.glob(os.path.join(data_loader.dataset.edit_imgs_root, '*.png')))[anchor_id], as_transformed_tensor=True, transform_style='original').to(device)
            mask_anchor = ploss_fn(frame_anchor, aligned_ori_anchor)
            mask_anchor = torch.nn.functional.interpolate(mask_anchor, (mask_anchor.shape[-2]//scale_factor, mask_anchor.shape[-1]//scale_factor))
        else:
            frame_anchor, _ = G([latent_anchor], input_is_latent=True, truncation=1.0, randomize_noise=False)
            aligned_ori_anchor = read(sorted(glob.glob(os.path.join(data_loader.dataset.edit_imgs_root, '*.png')))[anchor_id], as_transformed_tensor=True, transform_style='original').to(device)
        
        frame_anchor_256 = (frame_anchor + 1.0)/2.0 * 255
        frame_anchor_256 = torch.nn.functional.interpolate(frame_anchor_256, (frame_anchor_256.shape[-2]//scale_factor, frame_anchor_256.shape[-1]//scale_factor))
        frame_anchor_ = torch.nn.functional.interpolate(frame_anchor, (frame_anchor.shape[-2]//scale_factor, frame_anchor.shape[-1]//scale_factor))
        
        # aligned_ori_anchor_256 = (aligned_ori_anchor.unsqueeze(0) + 1.0)/2.0 * 255
        # aligned_ori_anchor_256 = torch.nn.functional.interpolate(aligned_ori_anchor_256, (aligned_ori_anchor_256.shape[-2]//scale_factor, aligned_ori_anchor_256.shape[-1]//scale_factor))

    optim_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    for epoch in range(epochs):
        photo_losses_epoch = []
        # if (epoch + 1) % 200 == 0:
        #     weight_out_mask *= 10.0
        #     print("Increasing out of mask weight...")
        epoch_iter = 0
        for itr, data in enumerate(data_loader):
            optimizer.zero_grad()

            # get frames
            # latent_code_pair, mask_pair, cropped_padded_pair, quad_pair, quad_mask_pair, ori_img_pair, aligned_ori_img_pair, crop_coord_pair = data
            latents, _, _, _, _, _, _, _, aligned_ori_imgs, _ = data
            latents = latents.to(device)
            if in_domain:
                # get new latents
                # mapper_input = torch.cat((latent_anchor, latents), dim=-1)
                # latents_ = latents + delta_w_step * mapper.mapper(mapper_input)
                delta_latents = mapper.mapper(latents.float())
                latents_ = latents + delta_w_step * delta_latents
                frames_out = G.synthesis(latents_, noise_mode='const', force_fp32=True)  # generate pair
                aligned_ori_imgs = aligned_ori_imgs.to(device)
                masks = ploss_fn(frames_out, aligned_ori_imgs)
                masks = torch.nn.functional.interpolate(masks, (masks.shape[-2]//scale_factor, masks.shape[-1]//scale_factor)).to(device)

                delta_anchor = mapper.mapper(latent_anchor.float())
                latent_anchor_ = latent_anchor + delta_w_step * delta_anchor
                frame_anchor = G.synthesis(latent_anchor_, noise_mode='const', force_fp32=True)
                aligned_ori_anchor = read(sorted(glob.glob(os.path.join(data_loader.dataset.edit_imgs_root, '*.png')))[anchor_id], as_transformed_tensor=True, transform_style='original').to(device)
                mask_anchor = ploss_fn(frame_anchor, aligned_ori_anchor)
                mask_anchor = torch.nn.functional.interpolate(mask_anchor, (mask_anchor.shape[-2]//scale_factor, mask_anchor.shape[-1]//scale_factor))

            else:
                frames_out, _ = G([latents], input_is_latent=True, truncation=1.0, randomize_noise=False)
                aligned_ori_imgs = aligned_ori_imgs.to(device)

            frames_out_ = torch.nn.functional.interpolate(frames_out, (frames_out.shape[-2]//scale_factor, frames_out.shape[-1]//scale_factor))
            aligned_ori_imgs = torch.nn.functional.interpolate(aligned_ori_imgs, (aligned_ori_imgs.shape[-2]//scale_factor, aligned_ori_imgs.shape[-1]//scale_factor))
                
            
            
            frames_256 = (frames_out_ + 1.0) / 2.0
            frames_256 = (frames_256 * 255)
            forward_flows = []
            backward_flows = []

            # aligned_ori_img_256 = (aligned_ori_imgs + 1.0) / 2.0
            # aligned_ori_img_256 = (aligned_ori_img_256 * 255)

            # we consider an order: anchor -> others
            for itr in range(frames_256.shape[0]):
                forward_flows.append(RAFT_model(frame_anchor_256, frames_256[itr:itr+1], iters=20, test_mode=True)[1])
                backward_flows.append(RAFT_model(frames_256[itr:itr+1], frame_anchor_256, iters=20, test_mode=True)[1])
                # forward_flows.append(RAFT_model(aligned_ori_anchor_256, aligned_ori_img_256[itr:itr+1], iters=20, test_mode=True)[1])
                # backward_flows.append(RAFT_model(aligned_ori_img_256[itr:itr+1], aligned_ori_anchor_256, iters=20, test_mode=True)[1])
                # forward_flows += calculate_flow(RAFT_model, aligned_ori_img_256[pair_itr * 2:pair_itr * 2 + 2], 'forward')
                # backward_flows += calculate_flow(RAFT_model, aligned_ori_img_256[pair_itr * 2:pair_itr * 2 + 2], 'backward') 
            
            forward_flows = torch.cat(forward_flows, dim=0)
            backward_flows = torch.cat(backward_flows, dim=0)
            # forward_flows = torch.zeros(1, 2, 256, 256).to(device)
            # backward_flows = torch.zeros(1, 2, 256, 256).to(device)
            backwarped_frames, _ = flow_warp(frames_out_, forward_flows) # (BS*2, 3, 256, 256), baseline 2, I0 - IN-1
            forwardwarped_frames, _ = flow_warp(frame_anchor_, backward_flows)
            

            if weight_photo > 0.0:
                loss_photo = 0.0

                _, err_forward = LF.occlusion(backward_flows, forward_flows, thresh=1.0)


                valid_mask_forward = torch.exp(-5.0 * err_forward)
                forwardwarped_frames_ = (forwardwarped_frames + 1.0)/2
                forwardwarped_frames_256 = forwardwarped_frames_ * 255

                if in_domain:
                    forwardwarped_mask_anchor, _ = flow_warp(mask_anchor, forward_flows)
                    valid_mask_forward = (valid_mask_forward + forwardwarped_mask_anchor).clamp(0.0, 1.0)

                loss_photo += (ploss_fn(frames_256, forwardwarped_frames_256)*valid_mask_forward).sum()/valid_mask_forward.sum()

                _, err_backward = LF.occlusion(forward_flows, backward_flows, thresh=1.0)
                valid_mask_backward = torch.exp(-5.0 * err_backward)

                backwarped_frames_ = (backwarped_frames + 1.0)/2
                backwarped_frames_256 = backwarped_frames_ * 255
                

                if in_domain:
                    valid_mask_backward = (valid_mask_backward + mask_anchor).clamp(0.0, 1.0)

                loss_photo += (ploss_fn(frame_anchor_256, backwarped_frames_256)*valid_mask_backward).sum()/valid_mask_backward.sum()

            else:
                loss_photo = torch.tensor(0.0)
            
            
            if weight_out_mask > 0.0 and in_domain:
                # loss_out_mask = LF.l1_loss(frames_out_ * ((1.0 - masks)**2) * 255, aligned_ori_imgs * ((1.0 - masks)**2) * 255).sum()/((1.0 - masks)**2).sum()
                # loss_out_mask = LF.l1_loss(frames_out_ * (1.0 - valid_mask_forward) * 255, aligned_ori_imgs * (1.0 - valid_mask_forward) * 255).sum()/(1.0 - valid_mask_forward).sum()
                loss_out_mask = (ploss_fn(frames_out_, aligned_ori_imgs)*((1.0 - forwardwarped_mask_anchor)**4)).sum()/((1.0 - forwardwarped_mask_anchor)**4).sum()
            else:
                loss_out_mask = torch.tensor(0.0)
            
            # forward-backward consistency error
            if weight_cycle > 0.0:
                loss_cycle = (valid_mask_backward * err_backward.abs()).sum()/valid_mask_backward.sum() + (valid_mask_forward * err_forward.abs()).sum()/valid_mask_forward.sum() # TODO: need to be fixed, right now only takes one direction
            else:
                loss_cycle = torch.tensor(0.0)
            
            # tv loss for flows
            if weight_tv_flow > 0.0:
                loss_tv_flow = LF.total_variation_loss(forward_flows) + LF.total_variation_loss(backward_flows)
            else:
                loss_tv_flow = torch.tensor(0.0)

            # regularize latents
            if reg_frame > 0.0:
                loss_reg_frame = torch.mean(torch.abs(delta_latents)) + torch.mean(torch.abs(delta_anchor))
            else:
                loss_reg_frame = torch.tensor(0.0) 
            # sum them up
            loss = weight_photo * loss_photo + reg_frame * loss_reg_frame + weight_cycle * loss_cycle + weight_tv_flow * loss_tv_flow + weight_out_mask * loss_out_mask
            loss.backward()
            optimizer.step()
            photo_losses_epoch.append(loss_photo.item())
            
            if epoch % 1 == 0 and n_iter % 10 == 0:
                print(
                    'Step: {}/{}, reg loss:{:.7f}, \
                    photometric loss:{:.7f}, \
                    out-of-mask loss:{:.7f}, \
                    forward-backward consistency loss:{:.7f}, \
                    total variational flow loss:{:.7f}, \
                    total loss:{:.7f}'.format(epoch, epochs, loss_reg_frame.item(), loss_photo.item(), loss_out_mask.item(), loss_cycle.item(), loss_tv_flow.item(), loss.item())
                    )
                
            if epoch % 1 == 0 and n_iter % 50 == 0:
                torch.save({'latents': latent_code_update.clone().cpu(), 
                               }, 
                               os.path.join(output_dir, 'variables.pth'))

            n_iter += 1
            epoch_iter += 1
        photo_losses.append(np.mean(photo_losses_epoch))
        print('Average photometric loss at Epoch {}: {:.4f}'.format(epoch, photo_losses[-1]))

        plt.figure()
        plt.plot(photo_losses)
        plt.savefig(os.path.join(output_dir, 'photo_loss.jpg'))
        plt.show() 
        plt.close('all')
        optim_scheduler.step()

    # update latent codes
    with torch.no_grad():
        latent_anchor = latent_anchor + delta_w_step * mapper.mapper(latent_anchor.float())
        for itr in range(latent_code_update.shape[0]):
            latent_code_input = latent_code_update[itr:itr+1].to(device)
            latent_code_update[itr:itr+1] = latent_code_input + delta_w_step * mapper.mapper(latent_code_input.float())
    
    # latent_code_update = torch.cat((latent_code_update[:anchor_id], latent_anchor, latent_code_update[anchor_id:]), dim=0)
    return optimizer, photo_losses, latent_code_update, latent_anchor

def generator_adjust(RAFT_model, latent_anchor, data_loader, anchor_id, old_G, new_G, lr_G, optimizer, output_dir, latent_code_update, reg_G, 
                    weight_photo, weight_out_mask, weight_in_mask, weight_cycle, weight_tv_flow, ploss_fn, 
                    scale_factor=2,
                    num_of_sampled_latents=1, regulizer_l2_lambda=0.1, regulizer_lpips_lambda=0.1, epochs=100,
                    in_domain=False,
                    ):
    
    photo_losses = []

    n_iter = 0
    weight_out_mask *= 15.0
    with torch.no_grad():
        if in_domain:
            frame_anchor_old = old_G.synthesis(latent_anchor, noise_mode='const', force_fp32=True)
            aligned_ori_anchor = read(sorted(glob.glob(os.path.join(data_loader.dataset.edit_imgs_root, '*.png')))[anchor_id], as_transformed_tensor=True, transform_style='original').unsqueeze(0).to(device)
            mask_anchor = ploss_fn(frame_anchor_old, aligned_ori_anchor)
            mask_anchor = torch.nn.functional.interpolate(mask_anchor, (mask_anchor.shape[-2]//scale_factor, mask_anchor.shape[-1]//scale_factor))
        else:
            frame_anchor_old, _ = new_G([latent_anchor], input_is_latent=True, truncation=1.0, randomize_noise=False)
            aligned_ori_anchor = read(sorted(glob.glob(os.path.join(data_loader.dataset.edit_imgs_root, '*.png')))[anchor_id], as_transformed_tensor=True, transform_style='original').unsqueeze(0).to(device)
            aligned_ori_anchor = aligned_ori_anchor.to(device)


        frame_anchor_256_old = (frame_anchor_old + 1.0)/2.0 * 255
        frame_anchor_256_old = torch.nn.functional.interpolate(frame_anchor_256_old, (frame_anchor_256_old.shape[-2]//scale_factor, frame_anchor_256_old.shape[-1]//scale_factor))
        

        frame_anchor_old = torch.nn.functional.interpolate(frame_anchor_old, (frame_anchor_old.shape[-2]//scale_factor, frame_anchor_old.shape[-1]//scale_factor))
        aligned_ori_anchor = torch.nn.functional.interpolate(aligned_ori_anchor, (aligned_ori_anchor.shape[-2]//scale_factor, aligned_ori_anchor.shape[-1]//scale_factor))

    
    optim_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    for epoch in range(epochs):
        photo_losses_epoch = []
        out_mask_loss_epoch = []
        in_mask_loss_epoch = []
        epoch_iter = 0
        for itr, data in enumerate(data_loader):
            optimizer.zero_grad()

            # get frames
            # latent_code_pair, mask_pair, cropped_padded_pair, quad_pair, quad_mask_pair, ori_img_pair, aligned_ori_img_pair, crop_coord_pair = data
            latents, _, _, _, _, _, _, _, aligned_ori_imgs, _ = data
            latents = latents.to(device)
            if in_domain:
                frame_anchor = new_G.synthesis(latent_anchor, noise_mode='const', force_fp32=True)

                aligned_ori_anchor = read(sorted(glob.glob(os.path.join(data_loader.dataset.edit_imgs_root, '*.png')))[anchor_id], as_transformed_tensor=True, transform_style='original').unsqueeze(0).to(device)
                
                frame_anchor_256 = (frame_anchor + 1.0)/2.0 * 255
                frame_anchor_256 = torch.nn.functional.interpolate(frame_anchor_256, (frame_anchor_256.shape[-2]//scale_factor, frame_anchor_256.shape[-1]//scale_factor))
                frame_anchor_ = torch.nn.functional.interpolate(frame_anchor, (frame_anchor.shape[-2]//scale_factor, frame_anchor.shape[-1]//scale_factor))
                aligned_ori_anchor = torch.nn.functional.interpolate(aligned_ori_anchor, (aligned_ori_anchor.shape[-2]//scale_factor, aligned_ori_anchor.shape[-1]//scale_factor))
                
                frames_out = new_G.synthesis(latents, noise_mode='const', force_fp32=True)  # generate pair
                aligned_ori_imgs = aligned_ori_imgs.to(device)
                masks = ploss_fn(frames_out, aligned_ori_imgs)
                masks = torch.nn.functional.interpolate(masks, (masks.shape[-2]//scale_factor, masks.shape[-1]//scale_factor)).to(device)
            else:
                frames_out, _ = new_G([latents], input_is_latent=True, truncation=1.0, randomize_noise=False)
                aligned_ori_imgs = aligned_ori_imgs.to(device)

                frame_anchor, _ = new_G([latent_anchor], input_is_latent=True, truncation=1.0, randomize_noise=False)
                aligned_ori_anchor = read(sorted(glob.glob(os.path.join(data_loader.dataset.edit_imgs_root, '*.png')))[anchor_id], as_transformed_tensor=True, transform_style='original').unsqueeze(0).to(device)
                aligned_ori_anchor = aligned_ori_anchor.to(device)

                frame_anchor_256 = (frame_anchor + 1.0)/2.0 * 255
                frame_anchor_256 = torch.nn.functional.interpolate(frame_anchor_256, (frame_anchor_256.shape[-2]//scale_factor, frame_anchor_256.shape[-1]//scale_factor))
                frame_anchor_ = torch.nn.functional.interpolate(frame_anchor, (frame_anchor.shape[-2]//scale_factor, frame_anchor.shape[-1]//scale_factor))

            frames_out_ = torch.nn.functional.interpolate(frames_out, (frames_out.shape[-2]//scale_factor, frames_out.shape[-1]//scale_factor))
            aligned_ori_imgs = torch.nn.functional.interpolate(aligned_ori_imgs, (aligned_ori_imgs.shape[-2]//scale_factor, aligned_ori_imgs.shape[-1]//scale_factor))
            
            frames_256 = (frames_out_ + 1.0) / 2.0
            frames_256 = (frames_256 * 255)
            forward_flows = []
            backward_flows = []

            
            # we consider an order: anchor -> others
            for itr in range(frames_256.shape[0]):
                forward_flows.append(RAFT_model(frame_anchor_256, frames_256[itr:itr+1], iters=20, test_mode=True)[1])
                backward_flows.append(RAFT_model(frames_256[itr:itr+1], frame_anchor_256, iters=20, test_mode=True)[1])
            
            forward_flows = torch.cat(forward_flows, dim=0)
            backward_flows = torch.cat(backward_flows, dim=0)

            backwarped_frames, _ = flow_warp(frames_out_, forward_flows) # (BS*2, 3, 256, 256), baseline 2, I0 - IN-1
            forwardwarped_frames, _ = flow_warp(frame_anchor_, backward_flows)
            
            if weight_photo > 0.0:
                # loss_photo = torch.tensor(0.0)
                loss_photo = 0.0

                _, err_forward = LF.occlusion(backward_flows, forward_flows, thresh=1.0)

                
                valid_mask_forward = torch.exp(-5.0 * err_forward)

                forwardwarped_frames_ = (forwardwarped_frames + 1.0)/2
                forwardwarped_frames_256 = forwardwarped_frames_ * 255
                valid_mask_forward = (valid_mask_forward + masks).clamp(0.0, 1.0)
                loss_photo += 1.0 * (ploss_fn(frames_256, forwardwarped_frames_256)*valid_mask_forward).sum()/valid_mask_forward.sum()

                _, err_backward = LF.occlusion(forward_flows, backward_flows, thresh=1.0)

                valid_mask_backward = torch.exp(-5.0 * err_backward)

                backwarped_frames_ = (backwarped_frames + 1.0)/2
                backwarped_frames_256 = backwarped_frames_ * 255

                if in_domain:
                    valid_mask_backward = (valid_mask_backward + mask_anchor).clamp(0.0, 1.0)

                loss_photo += 1.0 * (ploss_fn(frame_anchor_256, backwarped_frames_256)*valid_mask_backward).sum()/valid_mask_backward.sum()
            else:
                loss_photo = torch.tensor(0.0)
            
            
            if weight_out_mask > 0.0 and in_domain:
                forwardwarped_mask_anchor, _ = flow_warp(mask_anchor, forward_flows)
                loss_out_mask = (ploss_fn(frames_out_, aligned_ori_imgs)*((1.0 - forwardwarped_mask_anchor)**4)).sum()/((1.0 - forwardwarped_mask_anchor)**4).sum()
                loss_out_mask += (ploss_fn(frame_anchor_, aligned_ori_anchor)*((1.0 - mask_anchor)**4)).sum()/((1.0 - mask_anchor)**4).sum()
                
            else:
                loss_out_mask = torch.tensor(0.0)
            
            if weight_in_mask > 0.0 and in_domain:
                frames_out_old = old_G.synthesis(latents, noise_mode='const', force_fp32=True)
                frames_out_old_ = torch.nn.functional.interpolate(frames_out_old, (frames_out_old.shape[-2]//scale_factor, frames_out_old.shape[-1]//scale_factor))
                loss_in_mask = (ploss_fn(frames_out_, frames_out_old_)*(forwardwarped_mask_anchor**4)).sum()/(forwardwarped_mask_anchor**4).sum()
                loss_in_mask += (ploss_fn(frame_anchor_, frame_anchor_old)*(mask_anchor**4)).sum()/(mask_anchor**4).sum()
            else:
                loss_in_mask = torch.tensor(0.0)

            # forward-backward consistency error
            if weight_cycle > 0.0:
                loss_cycle = (valid_mask_backward * err_backward.abs()).sum()/valid_mask_backward.sum() + (valid_mask_forward * err_forward.abs()).sum()/valid_mask_forward.sum() # TODO: need to be fixed, right now only takes one direction
            else:
                loss_cycle = torch.tensor(0.0)
            
            # tv loss for flows
            if weight_tv_flow > 0.0:
                loss_tv_flow = LF.total_variation_loss(forward_flows) + LF.total_variation_loss(backward_flows)
            else:
                loss_tv_flow = torch.tensor(0.0)

            # sum them up
            loss = weight_photo * loss_photo + weight_cycle * loss_cycle + weight_tv_flow * loss_tv_flow + weight_out_mask * loss_out_mask + weight_in_mask * loss_in_mask
            # tic = time.time()
            if reg_G > 0.0:
                loss_reg_G = calc_reg_G(new_G, old_G, latent_anchor, ploss_fn, num_of_sampled_latents, regulizer_l2_lambda, regulizer_lpips_lambda, in_domain)
            else:
                loss_reg_G = torch.tensor(0.0)
            loss += reg_G * loss_reg_G

            loss.backward()
            optimizer.step()
            photo_losses_epoch.append(loss_photo.item())
            out_mask_loss_epoch.append(loss_out_mask.item())
            in_mask_loss_epoch.append(loss_in_mask.item())
            if epoch % 1 == 0 and n_iter % 10 == 0:
                print(
                    'Step: {}/{}, \
                    photometric loss:{:.4f}, \
                    out-of-mask loss:{:.4f}, \
                    in-mask loss:{:.4f}, \
                    forward-backward consistency loss:{:.4f}, \
                    total variational flow loss:{:.4f}, \
                    regularization for G:{:.4f},\
                    total loss:{:.4f}'.format(epoch, epochs, loss_photo.item(), loss_out_mask.item(), loss_in_mask.item(), loss_cycle.item(), loss_tv_flow.item(), loss_reg_G.item(), loss.item())
                    )
            if epoch % 5 == 0 and n_iter % 20 == 0:
                torch.save(new_G.state_dict(), os.path.join(output_dir, 'G.pth'))
            n_iter += 1
            epoch_iter += 1
        photo_losses.append(np.mean(photo_losses_epoch))


        plt.figure()
        plt.plot(photo_losses)
        plt.savefig(os.path.join(output_dir, 'photo_loss.jpg'))
        plt.show() 
        plt.close('all')


        optim_scheduler.step()
    latent_anchor = latent_anchor.detach().cpu()
    latent_code_update = latent_code_update.detach().cpu()  # TODO:
    latent_code_update = torch.cat((latent_code_update[:anchor_id], latent_anchor, latent_code_update[anchor_id:]), dim=0)
    return optimizer, photo_losses, new_G, latent_code_update



def temp_adjust(RAFT_model, old_G, checkpoint_path, edit_direction,
                latent_path, anchor_id, mask_path, output_dir, 
                num_frames, batch_size, epochs_w, hierarchicy,
                lr_w, tune_w, resume_w_from,
                lr_G, reg_G, tune_G, epochs_G,
                weight_photo, weight_out_mask, weight_in_mask, weight_cycle, weight_tv_flow, reg_frame,
                cropped_padded_path=None, quads_path=None, quad_mask_path=None, original_frame_path=None, aligned_ori_frame_path=None, crop_coords_path=None,
                scale_factor=2,
                in_domain=False,
                ):

    num_frames = torch.load(latent_path).shape[0]
    assert num_frames > 0, 'No frames!'
    if num_frames % 2 == 0:
        anchor_id = num_frames // 2 - 1
    else:
        anchor_id = num_frames // 2
    # load latent codes
    with torch.no_grad():
        if in_domain:
            
            latent_anchor = torch.load(latent_path)[anchor_id:anchor_id+1].clone().to(device)
            latents_update = torch.cat((torch.load(latent_path)[0:anchor_id], torch.load(latent_path)[anchor_id+1:num_frames]), dim=0).clone()
            latents_ori = torch.load(latent_path)[:num_frames]

            mapper_weight_path = os.path.join('./pretrained_models', edit_direction+'.pt')
            if os.path.exists(mapper_weight_path):
                ckpt = torch.load(mapper_weight_path, map_location='cpu')
            else:
                ckpt = torch.load('./PTI/pretrained_models/eyeglasses.pt', map_location='cpu')
            opts = ckpt['opts']
            # opts['mapper_type'] = 'LevelsMapper_1024'
            if edit_direction not in ["afro", "angry", "Jhonny Depp", "beard", "depp", "surprised", "eyeglasses", "smile", "beyonce", "elsa", "red_hair", "heavy_makeup"]:
                opts['checkpoint_path'] = None  # means we don't load any weights
            else:
                opts['checkpoint_path'] = mapper_weight_path
            print(mapper_weight_path)
            opts = Namespace(**opts)

            mapper = StyleCLIPMapper(opts).to(device)
            
        else:
            latents_update = torch.from_numpy(np.array(list(np.load(latent_path, allow_pickle=True).item().values())))
            latent_anchor = latents_update[anchor_id:anchor_id+1].clone().to(device)
            latents_update = torch.cat((latents_update[:anchor_id], latents_update[anchor_id+1:num_frames]), dim=0)
            latents_ori = torch.from_numpy(np.array(list(np.load(latent_path, allow_pickle=True).item().values())))

            # init a MLP to predict residual
            mapper_weight_path = './PTI/pretrained_models/eyeglasses.pt'
            ckpt = torch.load(mapper_weight_path, map_location='cpu')
            opts = ckpt['opts']
            # opts['mapper_type'] = 'LevelsMapper_1024'
            opts['checkpoint_path'] = None
            opts = Namespace(**opts)

            mapper = StyleCLIPMapper(opts).to(device)

    optimizer_w = torch.optim.Adam([
                                {'params': mapper.parameters(), 'lr': float(lr_w)},
                                ])
    
    ploss_fn = lpips.LPIPS(net='alex', spatial=True).to(device)
    # data loader, preprocessing
    latent_dataset = LatentDataset(latents_update, 
                                    'mask_path', 
                                    cropped_padded_path, 
                                    quads_path, 
                                    quad_mask_path, 
                                    original_frame_path, 
                                    aligned_ori_frame_path,
                                    crop_coords_path, 
                                    anchor_id=anchor_id)
    latent_dataloader = torch.utils.data.DataLoader(latent_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    print(len(latent_dataloader))

    if tune_w:
        print('Starting tuning w!')
        if edit_direction in ["eyeglasses"]:
            delta_w_step = 0.05
        else:
            delta_w_step = 0.04
        _, photo_losses, latents_update, latent_anchor = latents_adjust(RAFT_model, latent_anchor, anchor_id, latent_dataloader, old_G, mapper, lr_w, epochs_w, optimizer_w, output_dir, latents_update,
                        weight_photo, weight_out_mask, weight_cycle, weight_tv_flow, reg_frame, ploss_fn, scale_factor, in_domain, delta_w_step=delta_w_step)
        new_G = old_G

        torch.cuda.empty_cache()
        with torch.no_grad():
            latent_anchor = latent_anchor.cpu()
            latents_out = torch.cat((latents_update[:anchor_id], latent_anchor, latents_update[anchor_id:]), dim=0).detach()
            latents = latents_out.detach()
            torch.save({'latents': latents.clone().cpu()}, os.path.join(output_dir, 'variables.pth'))
            if tune_G:
                torch.save(new_G, os.path.join(output_dir, 'G.pth'))

            # frames = []
            # direct_edit_frames = []
            # quads = torch.from_numpy(np.load(quads_path))
            # crop_coords = torch.from_numpy(np.load(crop_coords_path))
            # os.makedirs(os.path.join(output_dir, 'frames'), exist_ok=True)
            # os.makedirs(os.path.join(output_dir, 'aligned_frames'), exist_ok=True)

            # edited_writer = imageio.get_writer(os.path.join(output_dir, 'edited.mp4'), fps=30)
            # de_writer = imageio.get_writer(os.path.join(output_dir, 'direct_edited.mp4'), fps=30)

            # print('Baking video...')
            # for frame_itr in tqdm(range(latents.shape[0])):

            #     cropped_padded_path_ = sorted(glob.glob(os.path.join(cropped_padded_path, '*.png')))[frame_itr]
            #     quad_mask_path_  = sorted(glob.glob(os.path.join(quad_mask_path, '*.png')))[frame_itr]
            #     original_frame_path_ =  sorted(glob.glob(os.path.join(original_frame_path, '*.png')))[frame_itr]

            #     cropped_padded = read(cropped_padded_path_, as_transformed_tensor=True, transform_style='original').unsqueeze(0).to(device)
            #     quad_mask = (torch.from_numpy(cv2.imread(quad_mask_path_)[:,:,[2,1,0]]).permute(2,0,1).unsqueeze(0)/255).to(device)
            #     ori_img = read(original_frame_path_, as_transformed_tensor=True, transform_style='original').to(device)

            #     quad_coords = quads[frame_itr]
            #     crop_coord = crop_coords[frame_itr]
                
            #     latent_curr = latents[frame_itr:frame_itr+1].to(device)
            #     latent_curr_ori = latents_ori[frame_itr:frame_itr+1].to(device)

            #     if in_domain:
            #         out_ = new_G.synthesis(latent_curr, noise_mode='const', force_fp32=True)
            #         direct_edit_ = old_G.synthesis(latent_curr_ori, noise_mode='const', force_fp32=True)
            #     else:
            #         out_, _ = new_G([latent_curr], input_is_latent=True, truncation=1.0, randomize_noise=False)
            #         direct_edit_, _ = old_G([latent_curr_ori], input_is_latent=True, truncation=1.0, randomize_noise=False)
                
            #     cv2.imwrite(os.path.join(output_dir, 'aligned_frames','output-{:05}.png'.format(frame_itr)), to_image(out_.cpu())[0])

            #     out_ = perspective(out_, [(0, 0), (0, 1024), (1024, 1024), (1024, 0)], quad_coords)[:,:,:quad_mask.shape[2], :quad_mask.shape[3]]
            #     out_ = quad_mask * out_ + (1.0 - quad_mask) * cropped_padded  # cropped and padded space
            #     # make a paste mask
            #     paste_mask = torch.zeros((1, 1, ori_img.shape[-2], ori_img.shape[-1]))
            #     paste_mask[:, :, crop_coord[1]:crop_coord[3], crop_coord[0]:crop_coord[2]] = 1.0
            #     paste_mask = paste_mask.to(device)
            #     # if crop_coord[0] + (ori_img.shape[-1] - crop_coord[2]) + frame_out.shape[-1] < 1280:
            #     padding = (1280 - out_.shape[-1] - (ori_img.shape[-1] - crop_coord[2]), ori_img.shape[-1]-crop_coord[2], 720 - out_.shape[-2] - (ori_img.shape[-2]-crop_coord[3]), ori_img.shape[-2]-crop_coord[3])
            #     out_ = torch.nn.functional.pad(out_, padding, 'constant', 0)
            #     out_ = paste_mask * out_ + (1.0 - paste_mask) * ori_img

            #     # out_im = to_image(out_.cpu())[0]
            #     out_im = (out_.cpu().permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()[0]
            #     cv2.imwrite(os.path.join(output_dir, 'frames','output-{:05}.png'.format(frame_itr)), out_im[:, :, [2, 1, 0]])
            #     # frames.append(out_im[:, :, [2, 1, 0]])
            #     edited_writer.append_data(out_im)

            #     direct_edit_ = perspective(direct_edit_, [(0, 0), (0, 1024), (1024, 1024), (1024, 0)], quad_coords)[:,:,:quad_mask.shape[2], :quad_mask.shape[3]]
            #     direct_edit_ = quad_mask * direct_edit_ + (1.0 - quad_mask) * cropped_padded  # cropped and padded space
            #     # make a paste mask
            #     paste_mask = torch.zeros((1, 1, ori_img.shape[-2], ori_img.shape[-1]))
            #     paste_mask[:, :, crop_coord[1]:crop_coord[3], crop_coord[0]:crop_coord[2]] = 1.0
            #     paste_mask = paste_mask.to(device)
            #     padding = (1280 - direct_edit_.shape[-1] - (ori_img.shape[-1] - crop_coord[2]), ori_img.shape[-1]-crop_coord[2], 720 - direct_edit_.shape[-2] - (ori_img.shape[-2]-crop_coord[3]), ori_img.shape[-2]-crop_coord[3])
            #     direct_edit_ = torch.nn.functional.pad(direct_edit_, padding, 'constant', 0)

            #     direct_edit_ = paste_mask * direct_edit_ + (1.0 - paste_mask) * ori_img
            #     direct_edit_im = (direct_edit_.cpu().permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()[0]
            #     de_writer.append_data(direct_edit_im)

            
            plt.figure()
            plt.plot(photo_losses)
            plt.savefig(os.path.join(output_dir, 'photo_loss.jpg'))
            plt.show() 
        
    else:
        print('Skip latent code adjustment!')
        resume_w_from = os.path.join(output_dir, 'variables.pth')
        if os.path.exists(resume_w_from):
            # latents_update = torch.load(resume_w_from)['latents'].to(device)  # TODO: beta version, this only works for single video.
            new_G = old_G
            latent_anchor = torch.load(resume_w_from)['latents'][anchor_id:anchor_id+1].clone().to(device)
            latents_update = torch.cat((torch.load(resume_w_from)['latents'][0:anchor_id], torch.load(resume_w_from)['latents'][anchor_id+1:num_frames]), dim=0).clone().to(device).requires_grad_(True)

    if tune_G:
        print('Start tuning G!')
        torch.cuda.empty_cache()
        # data loader, preprocessing
        latent_dataset = LatentDataset(latents_update, 
                                    'mask_path', 
                                    cropped_padded_path, 
                                    quads_path, 
                                    quad_mask_path, 
                                    original_frame_path, 
                                    aligned_ori_frame_path,
                                    crop_coords_path, 
                                    anchor_id=anchor_id)
        latent_dataloader = torch.utils.data.DataLoader(latent_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)

        print(len(latent_dataloader))
        # # Tune G
        if in_domain:
            new_G = load_tuned_G(checkpoint_path)
        else:
            new_G = Generator(
                            size=1024, style_dim=512, n_mlp=8,
                            ).to(device)   # StyleGAN-Nada
            checkpoint = torch.load(checkpoint_path)
            new_G.load_state_dict(checkpoint['g_ema'])
        new_G.train()
        new_G.to(device)
        for param in new_G.parameters():
            param.requires_grad = True
        for param in old_G.parameters():
            param.requires_grad = False
        optimizer_G = torch.optim.Adam([
                                        {'params': new_G.parameters(), 'lr': float(lr_G)},
                                        # {'params': latents_update, 'lr': float(lr_w)},
                                        ])

        output_dir = os.path.join(output_dir, 'tune_G')
        os.makedirs(output_dir, exist_ok=True)
        latent_anchor = latent_anchor.to(device)
        optimizer_G, photo_losses, new_G, latents_update = generator_adjust(RAFT_model, latent_anchor, latent_dataloader, anchor_id, old_G, new_G, lr_G, optimizer_G, output_dir, latents_update, reg_G,
                                                            weight_photo, weight_out_mask, weight_in_mask, weight_cycle, weight_tv_flow, ploss_fn=ploss_fn, epochs=epochs_G, scale_factor=scale_factor,
                                                            in_domain=in_domain,
                                                            )
    else:
        print('Skip generator adjustment!')


    torch.cuda.empty_cache()
    # save results
    
    edited_writer = imageio.get_writer(os.path.join(output_dir, 'edited.mp4'), fps=30)
    de_writer = imageio.get_writer(os.path.join(output_dir, 'direct_edited.mp4'), fps=30)

    with torch.no_grad():
        latents = latents_update.detach()
        torch.save({'latents': latents.clone().cpu()}, os.path.join(output_dir, 'variables.pth'))
        if tune_G:
            torch.save(new_G, os.path.join(output_dir, 'G.pth'))

        frames = []
        direct_edit_frames = []
        quads = torch.from_numpy(np.load(quads_path))
        crop_coords = torch.from_numpy(np.load(crop_coords_path))
        os.makedirs(os.path.join(output_dir, 'frames'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'aligned_frames'), exist_ok=True)
        print('Baking video...')
        for frame_itr in tqdm(range(latents.shape[0])):

            cropped_padded_path_ = sorted(glob.glob(os.path.join(cropped_padded_path, '*.png')))[frame_itr]
            quad_mask_path_  = sorted(glob.glob(os.path.join(quad_mask_path, '*.png')))[frame_itr]
            original_frame_path_ =  sorted(glob.glob(os.path.join(original_frame_path, '*.png')))[frame_itr]

            cropped_padded = read(cropped_padded_path_, as_transformed_tensor=True, transform_style='original').unsqueeze(0).to(device)
            quad_mask = (torch.from_numpy(cv2.imread(quad_mask_path_)[:,:,[2,1,0]]).permute(2,0,1).unsqueeze(0)/255).to(device)
            ori_img = read(original_frame_path_, as_transformed_tensor=True, transform_style='original').to(device)
            # ori_img = -torch.ones_like(ori_img).to(device)

            quad_coords = quads[frame_itr]
            crop_coord = crop_coords[frame_itr]
            
            latent_curr = latents[frame_itr:frame_itr+1].to(device)
            latents_curr_ori = latents_ori[frame_itr:frame_itr+1].to(device)
            if in_domain:
                out_ = new_G.synthesis(latent_curr, noise_mode='const', force_fp32=True)
                direct_edit_ = old_G.synthesis(latents_curr_ori, noise_mode='const', force_fp32=True)
            else:
                out_, _ = new_G([latent_curr], input_is_latent=True, truncation=1.0, randomize_noise=False)
                direct_edit_, _ = old_G([latents_curr_ori], input_is_latent=True, truncation=1.0, randomize_noise=False)
            
            cv2.imwrite(os.path.join(output_dir, 'aligned_frames','output-{:05}.png'.format(frame_itr)), to_image(out_.cpu())[0])

            out_ = perspective(out_, [(0, 0), (0, 1024), (1024, 1024), (1024, 0)], quad_coords)[:,:,:quad_mask.shape[2], :quad_mask.shape[3]]
            out_ = quad_mask * out_ + (1.0 - quad_mask) * cropped_padded  # cropped and padded space
            # make a paste mask
            paste_mask = torch.zeros((1, 1, ori_img.shape[-2], ori_img.shape[-1]))
            paste_mask[:, :, crop_coord[1]:crop_coord[3], crop_coord[0]:crop_coord[2]] = 1.0
            paste_mask = paste_mask.to(device)
            # if crop_coord[0] + (ori_img.shape[-1] - crop_coord[2]) + frame_out.shape[-1] < 1280:
            padding = (1280 - out_.shape[-1] - (ori_img.shape[-1] - crop_coord[2]), ori_img.shape[-1]-crop_coord[2], 720 - out_.shape[-2] - (ori_img.shape[-2]-crop_coord[3]), ori_img.shape[-2]-crop_coord[3])
            out_ = torch.nn.functional.pad(out_, padding, 'constant', 0)
            out_ = paste_mask * out_ + (1.0 - paste_mask) * ori_img

            out_im = (out_.cpu().permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()[0]
            cv2.imwrite(os.path.join(output_dir, 'frames','output-{:05}.png'.format(frame_itr)), out_im[:, :, [2, 1, 0]])
            edited_writer.append_data(out_im)

            direct_edit_ = perspective(direct_edit_, [(0, 0), (0, 1024), (1024, 1024), (1024, 0)], quad_coords)[:,:,:quad_mask.shape[2], :quad_mask.shape[3]]
            direct_edit_ = quad_mask * direct_edit_ + (1.0 - quad_mask) * cropped_padded  # cropped and padded space
            # make a paste mask
            paste_mask = torch.zeros((1, 1, ori_img.shape[-2], ori_img.shape[-1]))
            paste_mask[:, :, crop_coord[1]:crop_coord[3], crop_coord[0]:crop_coord[2]] = 1.0
            paste_mask = paste_mask.to(device)
            padding = (1280 - direct_edit_.shape[-1] - (ori_img.shape[-1] - crop_coord[2]), ori_img.shape[-1]-crop_coord[2], 720 - direct_edit_.shape[-2] - (ori_img.shape[-2]-crop_coord[3]), ori_img.shape[-2]-crop_coord[3])
            direct_edit_ = torch.nn.functional.pad(direct_edit_, padding, 'constant', 0)

            direct_edit_ = paste_mask * direct_edit_ + (1.0 - paste_mask) * ori_img
            direct_edit_im = (direct_edit_.cpu().permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()[0]
            de_writer.append_data(direct_edit_im)

        
        plt.figure()
        plt.plot(photo_losses)
        plt.savefig(os.path.join(output_dir, 'photo_loss.jpg'))
        plt.show() 

if __name__ == '__main__':
     # args
    test_opts = TestOptions()

    # running
    test_opts.parser.add_argument('--in_domain', action='store_true', help='call it if we are handling in-domain editing')
    test_opts.parser.add_argument('--encoder', type=str, default='psp')

    # paths
    test_opts.parser.add_argument('--edit_root', type=str, required=True, help='the direct editing variable root')
    test_opts.parser.add_argument('--metadata_root', type=str, required=True, help='root ofg metadata')
    test_opts.parser.add_argument('--original_root', type=str, required=True, help='root of the original frames')
    test_opts.parser.add_argument('--aligned_ori_frame_root', type=str, required=True, help='root of aligned original frames')
    test_opts.parser.add_argument('--mask_path', type=str, default=None, help='path to the mask')
    test_opts.parser.add_argument('--exp_name', type=str, default=None)
    test_opts.parser.add_argument('--run_name', type=str)

    # training parameters
    test_opts.parser.add_argument('--scale_factor', type=int, default=2)
    test_opts.parser.add_argument('--num_frames', type=int, default=None)
    test_opts.parser.add_argument('--anchor_id', type=int, default=0)
    test_opts.parser.add_argument('--epochs_w', type=int, default=100, help='number of epochs')
    test_opts.parser.add_argument('--epochs_G', type=int, default=100, help='number of epochs')
    test_opts.parser.add_argument('--batch_size', type=int, default=2, help='number of batch')
    test_opts.parser.add_argument('--hierarchicy', type=int, default=3, help='number of hierarchicy')
    test_opts.parser.add_argument('--lr', type=float, default=1e-1, help='the learning rate of w')
    test_opts.parser.add_argument('--lr_G', type=float, default=1e-5, help='the learning rate for G')
    test_opts.parser.add_argument('--tune_G', action="store_true", help='if tune G')
    test_opts.parser.add_argument('--tune_w', action="store_true", help='if tune w')
    test_opts.parser.add_argument('--resume_w_from', type=str, default='./nowheretofind.pt', )
    
    # loss weights
    test_opts.parser.add_argument('--reg_frame', type=float, default=1e2, help='reg strength for w')
    test_opts.parser.add_argument('--reg_anchor', type=float, default=5000.0, help='reg stregth for anchor')
    test_opts.parser.add_argument('--reg_G', type=float, default=0.0, help='reg strength for G')
    test_opts.parser.add_argument('--weight_photo', type=float, default=1.0, help='the weight of photometric loss')
    test_opts.parser.add_argument('--weight_out_mask', type=float, default=1.0, help='the weight of out of mask loss')
    test_opts.parser.add_argument('--weight_in_mask', type=float, default=1.0, help='the weight of in mask loss')
    test_opts.parser.add_argument('--weight_cycle', type=float, default=1.0, help='the weight of forward-backward consistency loss')
    test_opts.parser.add_argument('--weight_tv_flow', type=float, default=1.0, help='the weight of total variational loss')
    
    # RAFT
    test_opts.parser.add_argument('--if_raft', action='store_true', help='include this flag if you want to use raft on-the-fly')
    test_opts.parser.add_argument('--model', default='./pretrained_models/raft-things.pth', help="restore checkpoint")
    test_opts.parser.add_argument('--small', action='store_true', help='use small model')
    test_opts.parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    test_opts.parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    
    test_opts = test_opts.parse()
    
    # RAFT model for flow estimation
    RAFT_model = initialize_RAFT(test_opts)
    print('Not update RAFT_model')
    for param in RAFT_model.parameters():
        param.requires_grad = False
    
    video_name = test_opts.run_name
    print('Dealing with: ', video_name)
    if test_opts.in_domain:
        edit_root = os.path.join(test_opts.edit_root, 'StyleCLIP')
    else:
        edit_root = os.path.join(test_opts.edit_root, 'StyleGAN-Nada')  # debugging

    if not test_opts.in_domain:
        edit_directions = [
                                'disney_princess',  
                                'elf',   
                                'pixar',   
                                'sketch_hq',   
                                'vintage_comics',  
                                'zombie',  
                            ]
    else:
        edit_directions = [                
                                'eyeglasses', 
                            ]

    print('Dealing with in-domain editing? ', test_opts.in_domain)

    for edit_direction in edit_directions:  # debugging
        if test_opts.exp_name is not None:
            output_dir = os.path.join(edit_root, edit_direction, test_opts.exp_name)  # debugging
        else:
            output_dir = os.path.join(edit_root, edit_direction, 'refined')  # debugging
        if test_opts.force or (not os.path.exists(os.path.join(output_dir, 'tune_G','edited.mp4')) and not os.path.exists(os.path.join(output_dir,'edited.mp4'))):  # TODO
            print('------------Woring on {} ------------'.format(edit_direction))
            if test_opts.in_domain:
                checkpoint_path = os.path.join(test_opts.checkpoint_path, 'model_multi_id.pt')
                old_G = load_tuned_G(checkpoint_path)  # fixed G
            else:
                checkpoint_path = os.path.join(test_opts.checkpoint_path, edit_direction+'.pt')
            
                old_G =  Generator(
                        size=1024, style_dim=512, n_mlp=8,
                        ).to(device)# StyleGAN-Nada
                checkpoint = torch.load(checkpoint_path)
                old_G.load_state_dict(checkpoint['g_ema'])
            # old_G.eval()
            for param in old_G.parameters():
                param.requires_grad = False
            
            if test_opts.in_domain:
                latent_path = os.path.join(edit_root, edit_direction, 'latents.pth')
            else:
                latent_path = os.path.join(edit_root, '..','latents.npy')  # stylegan-nada
            
            os.makedirs(output_dir, exist_ok=True)
            f = open(os.path.join(output_dir, 'training_param.txt'), "w")
            f.write(str(test_opts))
            f.close()

            mask_path = test_opts.mask_path
            
            
            original_frame_path = test_opts.original_root
            aligned_ori_frame_path = test_opts.aligned_ori_frame_root

            metadata_root = test_opts.metadata_root
            cropped_padded_path = os.path.join(metadata_root, 'cropped_padded')
            quads_path = os.path.join(metadata_root, 'quads.npy')
            crop_coords_path = os.path.join(metadata_root, 'crop_coords.npy')
            quad_mask_path = os.path.join(metadata_root, 'quad_masks')
        
            temp_adjust(RAFT_model, old_G, checkpoint_path, edit_direction,
                        latent_path, test_opts.anchor_id, mask_path, output_dir, 
                        test_opts.num_frames, test_opts.batch_size, test_opts.epochs_w, test_opts.hierarchicy,
                        test_opts.lr, test_opts.tune_w, test_opts.resume_w_from,
                        test_opts.lr_G, test_opts.reg_G, test_opts.tune_G, test_opts.epochs_G,
                        test_opts.weight_photo, test_opts.weight_out_mask, test_opts.weight_in_mask, test_opts.weight_cycle, test_opts.weight_tv_flow, test_opts.reg_frame,
                        cropped_padded_path=cropped_padded_path, quads_path=quads_path, quad_mask_path=quad_mask_path, crop_coords_path=crop_coords_path,
                        original_frame_path=original_frame_path,
                        aligned_ori_frame_path=aligned_ori_frame_path,
                        scale_factor=test_opts.scale_factor,
                        in_domain=test_opts.in_domain,
                        )
            
