import numpy as np
import torch
# import cv2
from skimage import transform as tf

import matplotlib.pyplot as plt

def getTransformFlow(T, H=256, W=256):
    # all coordinates of HxW matrix
    # get base coordinates
    pi = np.expand_dims(np.where(np.ones((H, W)) == 1)[0], axis=1) # y, i
    pj = np.expand_dims(np.where(np.ones((H, W)) == 1)[1], axis=1) # x, j
    pij = np.concatenate((pi, pj), 1)
    tij = T(pij)

    flowij = tij-pij
    # flow = np.zeros((256, 256, 2))
    flow = np.zeros((H, W, 2))
    # fix the flow dy/dx
    flow[:, :, 0] = flowij[:, 1].reshape(H, W)
    flow[:, :, 1] = flowij[:, 0].reshape(H, W)

    return flow

def flow_warp(input_image, flow):
    B, C, H, W = input_image.size()
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float().to(flow)
    vgrid = grid + flow
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    if torch.cuda.is_available():
        vgrid = vgrid.cuda()
    
    vgrid = vgrid.permute(0, 2, 3, 1)

    warped = torch.nn.functional.grid_sample(input_image, vgrid)
    mask = torch.ones((B, 1, H, W)).to(flow)
    if torch.cuda.is_available():
        mask = mask.cuda()
    warped_mask = torch.nn.functional.grid_sample(mask, vgrid)
    warped_mask[warped_mask < 0.9999] = 0
    warped_mask[warped_mask > 0] = 1
    warped = warped * warped_mask
    
    del vgrid
    
    return warped, warped_mask


def transformImage(image, T):
    flow = getTransformFlow(T, image.shape[1], image.shape[2])
    # convert to Pytorch tensor and change shape h,w,c -> 1,c,h,w
    flow = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0)

    # convert to Pytorch tensor and change shape h,w,c -> 1,c,h,w
    image = image.unsqueeze(0)

    # warp
    out, out_mask = flow_warp(image.float(), flow.float())

    return out, out_mask, flow
    

def transform(im, scale=.75, translation=(0, 32)):
    """
    transform image im
    input: image to zoom into
    output: zoomed in image + zoom in flow
    """
    T = tf.SimilarityTransform(scale=scale, translation=translation)
    imageT, mask, flow = transformImage(im, T)

    return imageT, flow, mask

if __name__ == '__main__':
    from skimage import data
    # an example
    scale = .75
    translation = (0, 64)
    im = data.coffee()  # load some image
    im = np.transpose(im, (2, 0, 1))  # to PyTorch format (C, H, W)
    im = torch.from_numpy(im)  # to tensor
    transformed_im, transformed_flow, mask = transform(im, scale=scale, translation=translation)
    # inv back
    T = tf.SimilarityTransform(scale=scale, translation=translation)
    T_inv = np.linalg.inv(T.params.copy())
    T_inv = tf.SimilarityTransform(matrix=T_inv)
    im_rec, mask_rec, flow_rec = transformImage(transformed_im[0], T_inv)  # inverse


    # plot
    fig, ax = plt.subplots(5)
    ax[0].imshow(np.transpose(im.numpy(), (1, 2, 0)))  # original image
    ax[1].imshow(np.transpose(transformed_im.squeeze().numpy(), (1, 2, 0)).astype(np.uint8))  # transformed image
    ax[2].imshow(np.transpose(mask.squeeze(0).expand(3, im.shape[1], im.shape[2]).numpy(), (1, 2, 0)).astype(np.uint8) * 255)  # mask that could be used in backward warping
    ax[3].imshow(im_rec.squeeze().permute(1, 2, 0).numpy().astype(np.uint8))   # backward recovered image
    ax[4].imshow(np.transpose(mask_rec.squeeze(0).expand(3, im.shape[1], im.shape[2]).numpy(), (1, 2, 0)).astype(np.uint8) * 255) # mask that could be used in forward warping
    for a in ax:
        a.axis('off')

    plt.tight_layout()
    plt.show()
