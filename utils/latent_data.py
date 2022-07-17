"""
Data loader for stitch optimization.
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

def create_boundary(mask, kernel=np.ones((40,40), np.uint8), erode=False):
    mask_dilated = cv2.dilate(mask, kernel, iterations=1)
    mask_eroded = cv2.erode(mask, kernel, iterations=1)
    if erode:
        boundary = mask_dilated - mask_eroded
    else:
        boundary = mask_dilated - mask.squeeze()
    
    return boundary

class LatentDataset(Dataset):
    def __init__(self, 
                latents, 
                masks_root, 
                cropped_padded_imgs_root, 
                quads_root, 
                quad_masks_root, 
                ori_imgs_root, 
                edit_imgs_root,
                crop_coords_root,
                boundary_kernel=(21, 21),
                H=1024,
                W=1024,
                anchor_id=None,
                ):
        """
        Initialization.
        Load latent code
        """
        self.H = H
        self.W = W
        self.boundary_kernel = np.ones(boundary_kernel, np.uint8)

        self.latents = latents
            
        self.masks_root = masks_root
        self.cropped_padded_imgs_root = cropped_padded_imgs_root
        self.quads_root = quads_root  # quads coordinates
        self.quad_masks_root = quad_masks_root  # quad mask
        self.ori_imgs_root = ori_imgs_root  # original image
        self.edit_imgs_root = edit_imgs_root
        self.crop_coords_root = crop_coords_root  # crop coordinates


        self.mask_list = sorted(glob.glob(os.path.join(self.masks_root, '*.png')))
        self.cropped_padded_img_list = sorted(glob.glob(os.path.join(self.cropped_padded_imgs_root, '*.png')))
        self.quads = torch.from_numpy(np.load(self.quads_root))
        self.quad_mask_list = sorted(glob.glob(os.path.join(self.quad_masks_root, '*.png')))
        self.ori_img_list = sorted(glob.glob(os.path.join(self.ori_imgs_root, '*.jpg')) + glob.glob(os.path.join(self.ori_imgs_root, '*.png')))
        self.edit_img_list = sorted(glob.glob(os.path.join(self.edit_imgs_root, '*.jpg')) + glob.glob(os.path.join(self.edit_imgs_root, '*.png')))
        self.crop_coords = torch.from_numpy(np.load(self.crop_coords_root))

        if anchor_id is not None:
            self.mask_list = self.mask_list[:anchor_id] + self.mask_list[anchor_id+1:]
            self.cropped_padded_img_list =  self.cropped_padded_img_list[:anchor_id] +  self.cropped_padded_img_list[anchor_id+1:]
            self.quads = torch.cat((self.quads[:anchor_id], self.quads[anchor_id+1:]), dim=0)
            self.quad_mask_list = self.quad_mask_list[:anchor_id] + self.quad_mask_list[anchor_id+1:]
            self.ori_img_list = self.ori_img_list[:anchor_id] + self.ori_img_list[anchor_id+1:]
            self.edit_img_list = self.edit_img_list[:anchor_id] + self.edit_img_list[anchor_id+1:]
            self.crop_coords = torch.cat((self.crop_coords[:anchor_id], self.crop_coords[anchor_id+1:]), dim=0)

    def __len__(self):
        return self.latents.shape[0]

    def __getitem__(self, index):
        latent = self.latents[index]
        # read all the stuff
        # read segmentation mask
        
        if len(self.mask_list) > 0:
            mask = cv2.imread(self.mask_list[index])[:, :, 0:1]/255.
        else:
            mask = np.ones((self.H, self.W, 1)) / 1.0
        boundary = create_boundary(mask, self.boundary_kernel)
        mask = torch.from_numpy(mask).permute(2, 0, 1)
        boundary = torch.from_numpy(boundary).unsqueeze(0)

        # read cropped_padded_image
        cropped_padded_img = read(self.cropped_padded_img_list[index], as_transformed_tensor=True, transform_style='original')

        # read quads coords
        quad_coords = self.quads[index]

        # read quad mask
        # quad_mask = cv2.imread(self.quad_mask_list[index])[:,:,[2,1,0]]/255.
        quad_mask = cv2.imread(self.quad_mask_list[index])[:, :, 0:1]/255.
        quad_boundary = create_boundary(quad_mask, np.ones((45,45), np.uint8), erode=True)
        quad_mask = torch.from_numpy(quad_mask).permute(2,0,1)
        quad_boundary = torch.from_numpy(quad_boundary).unsqueeze(0)

        # read original image
        ori_img = read(self.ori_img_list[index], as_transformed_tensor=True, transform_style='original')

        # read edited image
        edit_img = read(self.edit_img_list[index], as_transformed_tensor=True, transform_style='original')

        # read crop coords
        crop_coords = self.crop_coords[index]

        return latent, mask, boundary, cropped_padded_img, quad_coords, quad_mask, quad_boundary, ori_img, edit_img, crop_coords


def collate_fn(batch):
    latents = []
    masks = []
    boundaries = []
    cropped_padded_imgs = []
    quad_coords = []
    quad_masks = []
    quad_boundaries = []
    ori_imgs = []
    edit_imgs = []
    crop_coords = []

    for item in batch:
        latents.append(item[0])
        masks.append(item[1])
        boundaries.append(item[2])
        cropped_padded_imgs.append(item[3])
        quad_coords.append(item[4])
        quad_masks.append(item[5])
        quad_boundaries.append(item[6])
        ori_imgs.append(item[7])
        edit_imgs.append(item[8])
        crop_coords.append(item[9])
    
    latents = torch.stack(latents)
    masks = torch.stack(masks)
    boundaries = torch.stack(boundaries)
    # cropped_padded_imgs = torch.stack(cropped_padded_imgs)
    quad_coords = torch.stack(quad_coords)
    # quad_masks = torch.stack(quad_masks)
    ori_imgs = torch.stack(ori_imgs)
    edit_imgs = torch.stack(edit_imgs)
    crop_coords = torch.stack(crop_coords)

    return latents, masks, boundaries, cropped_padded_imgs, quad_coords, quad_masks, quad_boundaries, ori_imgs, edit_imgs, crop_coords