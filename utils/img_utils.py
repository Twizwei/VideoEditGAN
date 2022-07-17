import numpy as np
import cv2
import torch

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


def poisson_blend(target, mask, generated):

    if np.max(target) <= 1.0:
        target = target * 255.
    if np.max(generated) <= 1.0:
        generated = generated * 255.
    if np.max(mask) > 1.0:
        mask = mask / 255.

    obj_center, _ = compute_stat_from_mask(
        binarize(torch.Tensor(mask).permute(2, 0, 1)))

    obj_center = (int(obj_center[1] * target.shape[1]), \
                  int(obj_center[0] * target.shape[0]))

    mask = (mask > 0.5).astype(np.float)

    blended_result = cv2.seamlessClone(
                                generated.astype(np.uint8),
                                target.astype(np.uint8),
                                (255 * mask[:, :, 0]).astype(np.uint8),
                                obj_center,
                                cv2.NORMAL_CLONE
                                )

    return blended_result