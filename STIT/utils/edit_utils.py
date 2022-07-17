import argparse
import math
import os
import pickle
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import configs.paths_config
from configs import paths_config
from training.networks import SynthesisBlock

import pdb


def add_texts_to_image_vertical(texts, pivot_images):
    images_height = pivot_images.height
    images_width = pivot_images.width

    text_height = 256 + 16 - images_height % 16
    num_images = len(texts)
    image_width = images_width // num_images
    text_image = Image.new('RGB', (images_width, text_height), (255, 255, 255))
    draw = ImageDraw.Draw(text_image)
    font_size = int(math.ceil(24 * image_width / 256))

    try:
        font = ImageFont.truetype("truetype/freefont/FreeSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    for i, text in enumerate(texts):
        draw.text((image_width // 2 + i * image_width, text_height // 2), text, fill='black', anchor='ms', font=font)

    out_image = Image.new('RGB', (pivot_images.width, pivot_images.height + text_image.height))
    out_image.paste(text_image, (0, 0))
    out_image.paste(pivot_images, (0, text_image.height))
    return out_image


def get_affine_layers(synthesis):
    blocks: List[SynthesisBlock] = [getattr(synthesis, f'b{res}') for res in synthesis.block_resolutions]
    affine_layers = []
    for block in blocks:
        if hasattr(block, 'conv0'):
            affine_layers.append((block.conv0.affine, True))
        affine_layers.append((block.conv1.affine, True))
        affine_layers.append((block.torgb.affine, False))
    return affine_layers


def load_stylespace_std():
    with open(paths_config.stylespace_mean_std, 'rb') as f:
        _, s_std = pickle.load(f)
        s_std = [torch.from_numpy(s).cuda() for s in s_std]

    return s_std


def to_styles(edit: torch.Tensor, affine_layers):
    idx = 0
    styles = []
    for layer, is_conv in affine_layers:
        layer_dim = layer.weight.shape[0]
        if is_conv:
            styles.append(edit[idx:idx + layer_dim].clone())
            idx += layer_dim
        else:
            styles.append(torch.zeros(layer_dim, device=edit.device, dtype=edit.dtype))

    return styles


def w_to_styles(w, affine_layers):
    w_idx = 0
    styles = []
    for affine, is_conv in affine_layers:
        styles.append(affine(w[:, w_idx]))
        if is_conv:
            w_idx += 1

    return styles


def paste_image_mask(inverse_transform, image, dst_image, mask, radius=0, sigma=0.0):
    image_masked = image.copy().convert('RGBA')
    pasted_image = dst_image.copy().convert('RGBA')
    if radius != 0:
        mask_np = np.array(mask)
        kernel_size = (radius * 2 + 1, radius * 2 + 1)
        kernel = np.ones(kernel_size)
        eroded = cv2.erode(mask_np, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        blurred_mask = cv2.GaussianBlur(eroded, kernel_size, sigmaX=sigma)
        blurred_mask = Image.fromarray(blurred_mask)
        image_masked.putalpha(blurred_mask)
    else:
        # mask_np = np.array(mask)
        # dilated = cv2.dilate(mask_np, np.ones((55, 55)), borderType=cv2.BORDER_CONSTANT, borderValue=0)
        # dilated = Image.fromarray(dilated)
        # image_masked.putalpha(dilated)

        image_masked.putalpha(mask)

    projected = image_masked.transform(dst_image.size, Image.PERSPECTIVE, inverse_transform,
                                       Image.BILINEAR)
    pasted_image.alpha_composite(projected)
    return pasted_image

def binarize(mask, min=0.0, max=1.0, eps=1e-3):
    """ used to convert continuous valued mask to binary mask """
    if type(mask) is torch.Tensor:
        assert mask.max() <= 1 + 1e-6, mask.max()
        assert mask.min() >= -1 - 1e-6, mask.min()
        mask = (mask > 1.0 - eps).float()
        return mask.clamp_(min, max)
    elif type(mask) is np.ndarray:
        mask = (mask > 1.0 - eps).astype(float)
        return np.clip(mask, min, max, out=mask)
    return False

def compute_stat_from_mask(mask):
    """ Given a binarized mask 0, 1. Compute the object size and center """
    st_h, st_w, en_h, en_w = bbox_from_mask(mask)
    obj_size = obj_h, obj_w = en_h - st_h, en_w - st_w
    obj_center = (st_h + obj_h // 2, st_w + obj_w // 2)

    obj_size = (obj_size[0] / mask.size(1), obj_size[1] / mask.size(2))
    obj_center = (obj_center[0] / mask.size(1), obj_center[1] / mask.size(2))
    return obj_center, obj_size

def bbox_from_mask(mask):
    assert len(list(mask.size())) == 3, \
        'expected 3d tensor but got {}'.format(len(list(mask.size())))

    try:
        tlc_h = (mask.mean(0).sum(1) != 0).nonzero()[0].item()
        brc_h = (mask.mean(0).sum(1) != 0).nonzero()[-1].item()
    except:
        tlc_h, brc_h = 0, mask.size(1)  # max range if failed

    try:
        tlc_w = (mask.mean(0).sum(0) != 0).nonzero()[0].item()
        brc_w = (mask.mean(0).sum(0) != 0).nonzero()[-1].item()
    except:
        tlc_w, brc_w = 0, mask.size(2)
    return tlc_h, tlc_w, brc_h, brc_w

def seamless_paste_image_mask(inverse_transform, image, dst_image, mask):
    """
    image - PIL Image, size (1024, 1024)
    dst_image - PIL Image, size (H, W)
    mask - PIL Image, size (1024, 1024)
    """
    mask_projected = mask.transform(dst_image.size, Image.PERSPECTIVE, inverse_transform,
                                       Image.BILINEAR)
    image_projected = image.transform(dst_image.size, Image.PERSPECTIVE, inverse_transform,
                                       Image.BILINEAR)

    image_masked = np.array(image_projected.copy())
    pasted_image = np.array(dst_image.copy())

    mask_np = np.array(mask_projected.copy())/255.

    obj_center, _ = compute_stat_from_mask(
        binarize(torch.Tensor(mask_np).unsqueeze(0)))
    
    obj_center = (int(obj_center[1] * pasted_image.shape[1]), \
                  int(obj_center[0] * pasted_image.shape[0]))

    mask_np = (mask_np > 0.5).astype(np.float)

    blended_result = cv2.seamlessClone(
                                image_masked.astype(np.uint8),
                                pasted_image.astype(np.uint8),
                                (255 * mask_np).astype(np.uint8),
                                obj_center,
                                cv2.NORMAL_CLONE,
                                # cv2.MIXED_CLONE
                                )
    
    blended_result = Image.fromarray(blended_result)

    return blended_result

def paste_image(inverse_transform, img, orig_image):
    pasted_image = orig_image.copy().convert('RGBA')
    projected = img.convert('RGBA').transform(orig_image.size, Image.PERSPECTIVE, inverse_transform, Image.BILINEAR)
    pasted_image.paste(projected, (0, 0), mask=projected)
    return pasted_image

