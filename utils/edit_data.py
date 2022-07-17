"""
Data loader for editing optimization.
"""
import os
import glob
import pdb

import torch
from torchvision import transforms
from torch.utils.data import Dataset

import numpy as np
from PIL import Image
import cv2

def read(im_path, as_transformed_tensor=False, im_size=1024, transform_style=None):
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

# class LatentDataset(Dataset):
#     def __init__(self, 
#                 latents, 
#                 masks=None, 
#                 cropped_padded_imgs=None, 
#                 quads=None, 
#                 quad_masks=None, 
#                 ori_imgs=None, 
#                 crop_coords=None,
#                 hierarchicy=1,
#                 ):
#         """
#         Initialization.
#         Load latent code
#         """
#         self.latents = latents
        
#         # get consecutive pairs, t and t+1 make a pair
#         self.latent_pair = []
#         self.masks = None
#         self.cropped_padded_imgs = None
#         self.quads = None
#         self.quad_masks = None
#         self.ori_imgs = None
#         self.crop_coords = None
        
#         if masks is not None:
#             self.masks = masks
#             self.mask_pair = []
#         if cropped_padded_imgs is not None:
#             self.cropped_padded_imgs = cropped_padded_imgs
#             self.cropped_padded_img_pair = []
#         if quads is not None:
#             self.quads = quads
#             self.quad_pair = []
#         if quad_masks is not None:
#             self.quad_masks = quad_masks
#             self.quad_mask_pair = []
#         if ori_imgs is not None:
#             self.ori_imgs = ori_imgs
#             self.ori_img_pair = []
#         if crop_coords is not None:
#             self.crop_coords = crop_coords
#             self.crop_coord_pair = []


#         for k in range(hierarchicy):  # k = 0, 1, 2, 4, ..., K - 1
#             d = 2 ** k
#             for itr in range(0, self.latents.shape[0] - 1, d):
#                 if itr + d + 1 <= self.latents.shape[0]:
#                     self.latent_pair.append(self.latents[itr:itr+d+1:d])
#                     if masks is not None:
#                         self.mask_pair.append(self.masks[itr:itr+d+1:d])
#                     if cropped_padded_imgs is not None:
#                         self.cropped_padded_img_pair.append(self.cropped_padded_imgs[itr:itr+d+1:d])
#                     if quads is not None:
#                         self.quad_pair.append(self.quads[itr:itr+d+1:d])
#                     if quad_masks is not None:
#                         self.quad_mask_pair.append(self.quad_masks[itr:itr+d+1:d])
#                     if ori_imgs is not None:
#                         self.ori_img_pair.append(self.ori_imgs[itr:itr+d+1:d])
#                     if crop_coords is not None:
#                         self.crop_coord_pair.append(self.crop_coords[itr:itr+d+1:d])
    
#     def __len__(self):
#         'Denotes the total number of samples'
#         return len(self.latent_pair)
    
#     def __getitem__(self, idx):
#         """
#         Get batched variables. 
#         """
#         outputs = (self.latent_pair[idx], )
#         if self.masks is not None:
#             outputs += (self.mask_pair[idx], )
#         if self.cropped_padded_imgs is not None:
#             outputs += (self.cropped_padded_img_pair[idx], )
#         if self.quads is not None:
#             outputs += (self.quad_pair[idx], )
#         if self.quad_masks is not None:
#             outputs += (self.quad_mask_pair[idx], )
#         if self.ori_imgs is not None:
#             outputs += (self.ori_img_pair[idx], )
#         if self.crop_coords is not None:
#             outputs += (self.crop_coord_pair[idx], )

#         return outputs

class LatentDataset(Dataset):
    def __init__(self, 
                latents, 
                masks_root=None, 
                cropped_padded_imgs_root=None, 
                quads_root=None, 
                quad_masks_root=None, 
                ori_imgs_root=None, 
                aligned_ori_frame_root=None,
                crop_coords_root=None,
                hierarchicy=1,
                H=1024,
                W=1024,
                ):
        """
        Initialization.
        Load latent code
        """
        self.H = H
        self.W = W

        self.latents = latents
        
        self.latent_pair = []
        
        self.masks_root = masks_root

        self.cropped_padded_imgs_root = cropped_padded_imgs_root
        self.quads_root = quads_root  # quads coordinates
        self.quad_masks_root = quad_masks_root  # quad mask
        self.ori_imgs_root = ori_imgs_root  # original image
        self.aligned_ori_frame_root = aligned_ori_frame_root  # aligned frame
        self.crop_coords_root = crop_coords_root  # crop coordinates

        self.pair_index = []
        for k in range(hierarchicy):  # k = 0, 1, 2, 4, ..., K - 1
            d = 2 ** k
            
            for itr in range(0, self.latents.shape[0] - 1, d):
                if itr + d + 1 <= self.latents.shape[0]:
                    self.latent_pair.append(self.latents[itr:itr+d+1:d])

                    self.pair_index.append((itr, itr+d+1, d))  # start, end, interval
                    # if masks is not None:
                    #     self.mask_pair.append(self.masks[itr:itr+d+1:d])
                    # if cropped_padded_imgs is not None:
                    #     self.cropped_padded_img_pair.append(self.cropped_padded_imgs[itr:itr+d+1:d])
                    # if quads is not None:
                    #     self.quad_pair.append(self.quads[itr:itr+d+1:d])
                    # if quad_masks is not None:
                    #     self.quad_mask_pair.append(self.quad_masks[itr:itr+d+1:d])
                    # if ori_imgs is not None:
                    #     self.ori_img_pair.append(self.ori_imgs[itr:itr+d+1:d])
                    # if crop_coords is not None:
                    #     self.crop_coord_pair.append(self.crop_coords[itr:itr+d+1:d])
        # long-term
        # for itr in range(8, self.latents.shape[0]):
        #     self.latent_pair.append(self.latents[0:itr:itr-1])
        #     self.pair_index.append((0, itr, itr-1))
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.latent_pair)
    
    def __getitem__(self, idx):
        """
        Get batched variables. 
        """
        outputs = (self.latent_pair[idx], )
        start, end, interval = self.pair_index[idx]


        if self.masks_root is not None:
            mask_list = sorted(glob.glob(os.path.join(self.masks_root, '*.png')))[start:end:interval]
            # read masks
            mask_pair = []
            for mask_path in mask_list:
                mask_pair.append(torch.from_numpy(cv2.imread(mask_path)[:,:,[2,1,0]]).permute(2,0,1)/255.)
            mask_pair = torch.stack(mask_pair)
            outputs += (mask_pair, )
        else:
            outputs += (torch.ones(2, 1, self.H, self.W), )
        
        if self.cropped_padded_imgs_root is not None:
            cropped_padded_img_list = sorted(glob.glob(os.path.join(self.cropped_padded_imgs_root, '*.png')))[start:end:interval]
            # read imgs
            cropped_padded_img_pair = []
            for cropped_padded_img_path in cropped_padded_img_list:
                cropped_padded_img_pair.append(read(cropped_padded_img_path, as_transformed_tensor=True, transform_style='original').unsqueeze(0))
            outputs += (cropped_padded_img_pair, )
        else:
            outputs += (None, )

        if self.quads_root is not None:
            quad_pair = torch.from_numpy(np.load(self.quads_root)[start:end:interval])
            outputs += (quad_pair, )
        else:
            outputs += (None, )

        if self.quad_masks_root is not None:
            quad_mask_list = sorted(glob.glob(os.path.join(self.quad_masks_root, '*.png')))[start:end:interval]
            # read masks
            quad_mask_pair = []
            for quad_mask_path in quad_mask_list:
                quad_mask_pair.append(torch.from_numpy(cv2.imread(quad_mask_path)[:,:,[2,1,0]]).permute(2,0,1).unsqueeze(0)/255.)
            outputs += (quad_mask_pair, )
        else:
            outputs += (None, )

        if self.ori_imgs_root is not None:
            ori_img_list = sorted(glob.glob(os.path.join(self.ori_imgs_root, '*.jpg')) + glob.glob(os.path.join(self.ori_imgs_root, '*.png')))[start:end:interval]
            # read images
            ori_img_pair = []
            for ori_img_path in ori_img_list:
                ori_img_pair.append(read(ori_img_path, as_transformed_tensor=True, transform_style='original'))
            ori_img_pair = torch.stack(ori_img_pair)
            outputs += (ori_img_pair, )
        else:
            outputs += (None, )
        
        if self.aligned_ori_frame_root is not None:
            aligned_ori_list = sorted(glob.glob(os.path.join(self.aligned_ori_frame_root, '*.jpg')) + glob.glob(os.path.join(self.aligned_ori_frame_root, '*.png')))[start:end:interval]
            # read images
            aligned_ori_img_pair = []
            for aligned_ori_img_path in aligned_ori_list:
                aligned_ori_img_pair.append(read(aligned_ori_img_path, as_transformed_tensor=True, transform_style='original'))
            aligned_ori_img_pair = torch.stack(aligned_ori_img_pair)
            outputs += (aligned_ori_img_pair, )
        else:
            outputs += (None, )
        

        if self.crop_coords_root is not None:
            crop_coord_pair = torch.from_numpy(np.load(self.crop_coords_root)[start:end:interval])
            outputs += (crop_coord_pair, )
        else:
            outputs += (None, )

        return outputs

def collate_fn(batch):
    """
    Batch contains (latent_pair, mask_pair, cropped_padded_img_pair, quad_pair, quad_mask_pair, ori_img_pair, aligned_ori_frame_paircrop_coord_pair)
    where cropped_padded_img_pair and quad_mask_pair may have different image sizes, 
    but corresponded cropped_padded_img and quad_mask_pair have the same shape.
    Note that each 'pair' contains two variables.
    """
    # latent_pair, mask_pair, cropped_padded_img_pair, quad_pair, quad_mask_pair, ori_img_pair, aligned_ori_img_pair, crop_coord_pair = batch

    latent_pair = []
    mask_pair = []
    cropped_padded_img_pair = []
    quad_pair = []
    quad_mask_pair = []
    ori_img_pair = []
    aligned_ori_img_pair = []
    crop_coord_pair = []
    for item in batch:
        latent_pair.append(item[0])
        mask_pair.append(item[1])
        cropped_padded_img_pair.append(item[2])
        quad_pair.append(item[3])
        quad_mask_pair.append(item[4])
        ori_img_pair.append(item[5])
        aligned_ori_img_pair.append(item[6])
        crop_coord_pair.append(item[7])

    latent_pair = torch.stack(latent_pair)
    # latent_pair = torch.cat(latent_pair, dim=0) 
    mask_pair = torch.stack(mask_pair)
    if quad_pair[0] is not None:
        quad_pair = torch.stack(quad_pair)
    ori_img_pair = torch.stack(ori_img_pair)
    aligned_ori_img_pair = torch.stack(aligned_ori_img_pair)
    if crop_coord_pair[0] is not None:
        crop_coord_pair = torch.stack(crop_coord_pair)

    return latent_pair, mask_pair, cropped_padded_img_pair, quad_pair, quad_mask_pair, ori_img_pair, aligned_ori_img_pair, crop_coord_pair
    

class LatentDatasetAnchor(Dataset):
    def __init__(self, latents, masks=None, cropped_padded_imgs=None, quads=None, quad_masks=None, ori_imgs=None, crop_coords=None):
        """
        Initialization.
        Load latent code
        """
        self.latents = latents
        
        # get consecutive pairs, t and t+1 make a pair
        if masks is not None:
            self.masks = masks
        if cropped_padded_imgs is not None:
            self.cropped_padded_imgs = cropped_padded_imgs
        if quads is not None:
            self.quads = quads
        if quad_masks is not None:
            self.quad_masks = quad_masks
        if ori_imgs is not None:
            self.ori_imgs = ori_imgs
        if crop_coords is not None:
            self.crop_coords = crop_coords
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.latents)
    
    def __getitem__(self, idx):
        """
        Get batched variables. 
        """
        outputs = (self.latents[idx:idx+1], )
        if self.masks is not None:
            outputs += (self.masks[idx:idx+1], )
        if self.cropped_padded_imgs is not None:
            outputs += (self.cropped_padded_imgs[idx:idx+1], )
        if self.quads is not None:
            outputs += (self.quads[idx:idx+1], )
        if self.quad_masks is not None:
            outputs += (self.quad_masks[idx:idx+1], )
        if self.ori_imgs is not None:
            outputs += (self.ori_imgs[idx:idx+1], )
        if self.crop_coords is not None:
            outputs += (self.crop_coords[idx:idx+1], )

        return outputs
    