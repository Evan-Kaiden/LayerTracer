import torch
import torch.nn as nn

import random
import numpy as np

import os
import imageio
import matplotlib.pyplot as plt


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


def make_gif(img: torch.Tensor, saliency: torch.Tensor, save_path: str) -> None:
    frames = []

    img = img.clamp(0,1)
    img_np = img.permute(1,2,0).numpy()

    num_layers = len(saliency)

    for i in range(num_layers):
        fig, ax = plt.subplots(figsize=(4,4))

        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        heat = saliency[i].detach().cpu().numpy()

        ax.imshow(img_np)
        ax.imshow(heat, cmap='jet', alpha=0.2, interpolation='bicubic')

        progress = (i+1) / num_layers

        bar_x0 = 0.05
        bar_y0 = 0.92
        bar_w  = 0.9
        bar_h  = 0.04

        ax.add_patch(plt.Rectangle(
            (bar_x0, bar_y0), bar_w, bar_h,
            transform=ax.transAxes,
            color='black', alpha=0.3
        ))

        ax.add_patch(plt.Rectangle(
            (bar_x0, bar_y0), bar_w * progress, bar_h,
            transform=ax.transAxes,
            color='lime', alpha=0.9
        ))

        ax.text(
            0.5, 0.955,
            f"Depth {i+1}/{num_layers}",
            transform=ax.transAxes,
            ha='center', va='top',
            color='white', fontsize=8,
            weight="bold"
        )

        ax.axis("off")

        fig.canvas.draw()

        frame = np.array(fig.canvas.renderer.buffer_rgba())
        frame = frame[:, :, :3]

        frames.append(frame)
        plt.close(fig)

    frames_dir = save_path[:-4] + "_frames"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)

    for i, frame in enumerate(frames):
        imageio.imwrite(f"{frames_dir}/frame_{i}.png", frame)

    imageio.mimsave(
        save_path,
        frames,
        duration=3,
        loop=0
    )

def get_device(verbose=False):
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.mps.is_available():
        device = 'mps'
    
    if verbose:
        print("Device:", device)

    return device