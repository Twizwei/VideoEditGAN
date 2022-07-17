"""
Do Unalignment and put cropped image back.
Right now it doesn't support paddding!!
"""

from argparse import ArgumentParser
import time
import numpy as np
import torch
import PIL
import PIL.Image
from PIL import ImageDraw
import os
import scipy
import scipy.ndimage

import pdb
from tqdm import tqdm
import imageio
import glob


def align_face(filepath, lm=None, output_size=1024, transform_size=1024, output_quad=False):
    """
	:param filepath: str
	:return: PIL Image
	"""
    
    lm_chin = lm[0: 17]  # left-right
    lm_eyebrow_left = lm[17: 22]  # left-right
    lm_eyebrow_right = lm[22: 27]  # left-right
    lm_nose = lm[27: 31]  # top-down
    lm_nostrils = lm[31: 36]  # top-down
    lm_eye_left = lm[36: 42]  # left-clockwise
    lm_eye_right = lm[42: 48]  # left-clockwise
    lm_mouth_outer = lm[48: 60]  # left-clockwise
    lm_mouth_inner = lm[60: 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    quad_ori = quad.copy()
    qsize = np.hypot(*x) * 2

    # read image
    img = PIL.Image.open(filepath)
    enable_padding = False

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]
    
    cropped_padded_img = img.copy()
    crop_pad_size = img.size
    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    # Save aligned image.
    if not output_quad:
        return img
    else:
        return img, quad + 0.5, quad_ori, crop_pad_size, crop, cropped_padded_img

def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)

def single_unalign(ori_filepath, aligned_path, output_dir, filename, lm=None, save_imgs=False):
    # Try only one image
    _, quad, quad_ori, crop_pad_size, crop_coord, cropped_padded_img = align_face(ori_filepath, lm=lm, output_quad=True) 
    # _, quad, crop_pad_size, crop_coord, cropped_padded_img = align_face(ori_filepath, predictor, lm=lm, output_quad=True)  # RAVDESS
    # _, quad, crop_pad_size, crop_coord, cropped_padded_img = align_face(ori_filepath.replace('png', 'jpg'), predictor, output_quad=True)
    if save_imgs:
        cropped_padded_img.save(os.path.join(output_dir, 'cropped_padded', filename.replace('jpg', 'png')))
    # put the aligned image back
    coeffs = find_coeffs(quad, [(0, 0), (0, 1024), (1024, 1024), (1024,0)])

    if os.path.exists(aligned_path):  # hard-coding here...
        aligned = PIL.Image.open(aligned_path)
    else:
        aligned = PIL.Image.open(aligned_path.replace('png', 'jpg'))
    
    img_inv = aligned.transform(crop_pad_size, PIL.Image.PERSPECTIVE, coeffs, PIL.Image.BILINEAR)  # size is crop_pad_size

    quad_mask = PIL.Image.new('L', crop_pad_size, 0)
    ImageDraw.Draw(quad_mask).polygon(quad.astype(np.int).flatten().tolist(), outline=1, fill=255)
    quad_mask_save_dir = os.path.join(output_dir, 'quad_masks', filename.replace('jpg', 'png'))  # TODO
    if save_imgs:
        quad_mask.save(quad_mask_save_dir)
    
    quad_mask = np.array(quad_mask)/255.
    quad_mask = np.expand_dims(quad_mask, axis=-1)

    filled_crop_pad_img = quad_mask * np.array(img_inv)[:,:,:3] + (1.0 - quad_mask) * np.array(cropped_padded_img)[:,:,:3]
    # filled_crop_pad_img = quad_mask * np.array(img_inv)[:,:,:3] + (1.0 - quad_mask) * 0.0  # black gb
    filled_crop_pad_img = PIL.Image.fromarray(filled_crop_pad_img.astype(np.uint8))
    
    input_img = PIL.Image.open(ori_filepath)

    # paste via PIL
    pasted_img = input_img.copy()
    pasted_img.paste(filled_crop_pad_img, (crop_coord[0], crop_coord[1]))

    ori_save_dir = os.path.join(output_dir, 'ori_back', filename.replace('jpg', 'png'))  # TODO
    if save_imgs:
        pasted_img.save(ori_save_dir)

    return input_img, pasted_img, quad, quad_ori, crop_coord

def multi_unalign(ori_imgs_path, aligned_imgs_path, output_path, num_frames=32, output_meta=True, output_name='ori_fitted'):
    """
    Multi-frames.
    """
    ori_frames = []
    pasted_frames = []
    quads = []
    quads_ori = []
    crop_coords = []
    frames_list = sorted(glob.glob(os.path.join(aligned_imgs_path, '*.png')))
    ori_frames_list = sorted(glob.glob(os.path.join(ori_imgs_path, '*.png')))

    lm = np.load(os.path.join(ori_imgs_path.replace('frames', 'landmarks'), 'landmarks.npy'))  # Internet video
    for frame_itr in tqdm(range(num_frames)):
    
        ori_filepath = ori_frames_list[frame_itr]
        filename_ori = os.path.basename(ori_filepath)
        aligned_path = frames_list[frame_itr]

        if output_meta:
            os.makedirs(os.path.join(output_path, 'quad_masks'), exist_ok=True)
            os.makedirs(os.path.join(output_path, 'ori_back'), exist_ok=True)
            os.makedirs(os.path.join(output_path, 'cropped_padded'), exist_ok=True)
        ori_frame, fit_to_ori_frame, quad, quad_ori, crop_coord = single_unalign(ori_filepath, aligned_path, output_path, filename_ori, lm=lm[frame_itr], save_imgs=output_meta)
        # ori_frame, fit_to_ori_frame, quad, crop_coord = single_unalign(ori_filepath, aligned_path, output_path, filename_ori)
        ori_frames.append(np.array(ori_frame))
        pasted_frames.append(np.array(fit_to_ori_frame))
        quads.append(quad)
        quads_ori.append(quad_ori)
        crop_coords.append(np.array(crop_coord))
    ori_frames = np.stack(ori_frames)
    pasted_frames = np.stack(pasted_frames)
    quads = np.stack(quads)
    quads_ori = np.stack(quads_ori)
    crop_coords = np.stack(crop_coords)
    if output_meta:
        np.save(os.path.join(output_path, 'quads.npy'), quads)
        np.save(os.path.join(output_path, 'quads_ori.npy'), quads_ori)
        np.save(os.path.join(output_path, 'crop_coords.npy'), crop_coords)

    imageio.mimwrite(
        os.path.join(output_path, output_name + '.mp4'), pasted_frames, fps=30, quality=8
        )


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--ori_images_path', type=str, required=True)
    parser.add_argument('--aligned_images_path', type=str, required=True)
    parser.add_argument('--num_frames', type=int, default=None)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--output_name', type=str, default='ori_fitted')
    args = parser.parse_args()

    # single video
    if args.num_frames is None:
        num_frames = len(glob.glob(os.path.join(args.aligned_images_path, '*.jpg'))) + len(glob.glob(os.path.join(args.aligned_images_path, '*.png')))
    else:
        num_frames = args.num_frames
    os.makedirs(args.output_path, exist_ok=True)

    multi_unalign(args.ori_images_path, args.aligned_images_path, args.output_path, num_frames, output_meta=True, output_name=args.output_name)

