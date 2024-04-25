import os
import math
import numpy as np
import cv2
from torchvision.utils import make_grid
from skimage.metrics import peak_signal_noise_ratio as PSNR
from pathlib import Path
import lpips
from torchvision import transforms
from PIL import Image

def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / \
        (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(
            math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calc_metric(result_path:str, gt_path='datasets/F2_img'):

    result_path = Path(result_path)
    gt_path = Path(gt_path)
    gt_names = list(gt_path.glob('*.jpg'))
    re_names = list(result_path.glob('*.png'))
    gt_names.sort()
    re_names.sort()

    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_lpips = 0.0

    idx = 0
    loss_fn_vgg = lpips.LPIPS(net='alex').to('cuda')
    for rname, fname in zip(gt_names, re_names):
        idx += 1
        ridx = rname.stem
        fidx = fname.stem
        assert ridx == fidx, 'Image ridx:{ridx}!=fidx:{fidx}'.format(
            ridx, fidx)
        
        hr_img = np.array(Image.open(rname))
        sr_img = np.array(Image.open(fname))
        psnr = PSNR(sr_img, hr_img)
        ssim = calculate_ssim(sr_img, hr_img)
        transform = transforms.ToTensor()
        img1 = transform(hr_img).unsqueeze(0).to('cuda')
        img2 = transform(sr_img).unsqueeze(0).to('cuda')
        lpips_value = loss_fn_vgg(img1, img2).item()
        avg_psnr += psnr
        avg_ssim += ssim
        avg_lpips += lpips_value
        print('{} :{:.4f} :{:.4f} :{:.4f}'.format(ridx, psnr, ssim, lpips_value))

    avg_psnr = avg_psnr / idx
    avg_ssim = avg_ssim / idx
    avg_lpips = avg_lpips / idx


    print('PSNR: {}'.format(avg_psnr))
    print('SSIM: {}'.format(avg_ssim))
    print('LPIPS:{}'.format(avg_lpips))
    # print(metrics_dict)