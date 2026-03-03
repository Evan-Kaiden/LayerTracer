import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np
from scipy.ndimage import gaussian_filter

import os
import imageio
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch




class AverageMeter(object):
    def __init__(self):
        self.val = None
        self.sum = 0
        self.count = 0
        self.avg = None

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def make_gif(
    img: torch.Tensor,
    saliency: list[torch.Tensor],
    save_path: str,
    interp_factor: int = 8,
    fps: int = 20,
    blur_sigma: float = 1.0,
    alpha: float = 0.35,
    save_mp4: bool = False,
):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    img = img.clamp(0, 1)
    img_np = img.permute(1, 2, 0).cpu().numpy()

    saliency_tensor = torch.stack(saliency)
    L, H, W = saliency_tensor.shape

    saliency_tensor = saliency_tensor.unsqueeze(0).unsqueeze(0).cpu()

    smooth_saliency = F.interpolate(
        saliency_tensor,
        size=(L * interp_factor, H, W),
        mode="trilinear",
        align_corners=False,
    ).squeeze()

    total_frames = smooth_saliency.shape[0]

    global_min = smooth_saliency.min()
    global_max = smooth_saliency.max()
    smooth_saliency = (smooth_saliency - global_min) / (
        global_max - global_min + 1e-8
    )

    smooth_saliency = smooth_saliency.cpu().numpy()

    frames = []

    for i in range(total_frames):

        heat = smooth_saliency[i]
        heat = gaussian_filter(heat, sigma=blur_sigma)

        fig, ax = plt.subplots(figsize=(4, 4))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        ax.imshow(img_np)
        ax.imshow(
            heat,
            cmap="magma",
            alpha=alpha,
            interpolation="bicubic",
        )

        progress = (i + 1) / total_frames

        cx, cy = 0.92, 0.92
        r = 0.045

        theta_bg = np.linspace(0, 2 * np.pi, 400)
        x_bg = cx + r * np.cos(theta_bg)
        y_bg = cy + r * np.sin(theta_bg)

        ax.plot(
            x_bg,
            y_bg,
            transform=ax.transAxes,
            color="white",
            linewidth=2,
            alpha=0.5,
        )

        theta = np.linspace(-np.pi / 2, -np.pi / 2 + 2 * np.pi * progress, 400)
        x = cx + r * np.cos(theta)
        y = cy + r * np.sin(theta)

        ax.plot(
            x,
            y,
            transform=ax.transAxes,
            color="white",
            linewidth=3,
            alpha=0.95,
        )
        ax.text(
            cx,
            cy,
            f"{int(progress*100)}%",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="white",
            fontsize=8,
            weight="bold",
        )

        ax.axis("off")

        fig.canvas.draw()
        frame = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        frames.append(frame)
        plt.close(fig)

    if save_mp4:
        mp4_path = save_path.replace(".gif", ".mp4")
        imageio.mimsave(mp4_path, frames, fps=fps)
    else:
        imageio.mimsave(save_path, frames, fps=fps)

    print(f"Saved animation with {total_frames} frames.")



def get_device(verbose=False):
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.mps.is_available():
        device = 'mps'
    
    if verbose:
        print("Device:", device)

    return device