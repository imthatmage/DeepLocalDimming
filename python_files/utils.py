import torch
import math


def get_transmittance(Image: torch.tensor, transmittance) -> torch.tensor:
    """
    here we should apply a linear transform 
    f(x) = ax + b, where
    f(0) = min_transmittance
    f(1) = 1.0
    trans(SIM2) -- 0.005
    trans(our) -- 0.00025
    now = 0.005
    """
    b = transmittance

    a = 1. - b
    def f(x): return a*x + b

    return f(Image)


def tensor_masking(b_h=37, b_w=60, img=None, img_h=1080, img_w=1920):
    if img is None:
        img = torch.ones((b_h, b_w))
    mask = torch.zeros((img_h, img_w))
    h_step = img_h/(b_h + 1)
    w_step = img_w/(b_w + 1)

    int_h = 0
    int_w = 0

    for i in range(b_h):
        h = int_h + int(h_step*(i+1) - int_h)
        int_h = h
        for j in range(b_w):
            w = int_w + int(w_step*(j+1) - int_w)
            int_w = w

            mask[h, w] = img[i, j]

    return mask


def gamma_correction(image):
    def gamma_trans(img, gamma):
        result = torch.clip(255*(img**gamma), 0.0, 255.0)
        return result/255.0
    # for each channel separately
    result = torch.zeros(image.shape).to(device)
    for i in range(3):
        # compute gamma = log(mid*255)/log(mean)
        mid = 0.5
        mean = torch.mean(image[0][i])*255.0
        gamma_val = math.log(mid*255.0)/math.log(mean)
        result[0][i] = gamma_trans(image[0][i], gamma_val).type(torch.double)
    return result
