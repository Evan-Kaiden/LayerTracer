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

class LayerSaliency():
    """the interface to generate saliency maps"""
    def __init__(self, image_size: int, patch_coverage: tuple[float, float]=(0.125, 0.2), repeats:int=1000, device:torch.device='cpu'):
        
        self.device = device
        
        self.image_shape = (3, image_size, image_size)
        self.repeats = repeats
        self.max_size = int(image_size * max(patch_coverage))
        self.min_size = int(image_size * min(patch_coverage))

    def _generate_masks(self) -> torch.Tensor:
        C, H, W = self.image_shape
        if self.max_size is None:
            self.max_size = min(H, W) // 2

        masks = torch.ones((self.repeats, C, H, W), dtype=torch.float32, device=self.device)

        for i in range(self.repeats):
            size = random.randint(self.min_size, self.max_size)
            top = random.randint(0, H - size)
            left = random.randint(0, W - size)

            masks[i, :, top:top+size, left:left+size] = 0.0

        return masks

    def _create_saliency_map(self, masks: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        scores = scores.view(-1, 1, 1, 1)
        weighted_masks = masks * scores
        saliency = weighted_masks.sum(dim=0)
        saliency = saliency.mean(dim=0)
        saliency /= scores.sum() + 1e-8
        return saliency

    def generate_visual(self, masks:torch.Tensor, baseline_dists:torch.Tensor, masked_dists:torch.Tensor) -> torch.Tensor:
        scores = baseline_dists - masked_dists
        results = []
        for idx in range(scores.size(0)):
            results.append(self._create_saliency_map(masks, scores[idx]))

        return results
    

def make_gif(img:torch.Tensor, saliency:torch.Tensor, save_path:str) -> None:
    frames = []

    img = img.clamp(0,1)
    img_np = img.permute(1,2,0).numpy()

    num_layers = len(saliency)

    for i in range(num_layers):
        fig, ax = plt.subplots(figsize=(4,4))
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

    dir_path = '/'.join(save_path.split('/')[:-1])
    os.makedirs(dir_path, exist_ok=True)
    imageio.mimsave(save_path, frames, duration=500.0)


MODEL_SOURCES = {
    "cifar10": [
        {
            "repo": "chenyaofo/pytorch-cifar-models",
            "name_format": "cifar10_{model}"
        },
    ],
    "cifar100": [
        {
            "repo": "chenyaofo/pytorch-cifar-models",
            "name_format": "cifar100_{model}"
        },
    ],
    "tinyimagenet": [
        {
            "repo": "chenyaofo/pytorch-cifar-models",
            "name_format": "{model}",   # try anyway
        },
    ],
}


def _try_torchhub(repo, model_name, pretrained=True):
    try:
        print(f"Trying {repo} : {model_name}")
        model = torch.hub.load(repo, model_name, pretrained=pretrained, verbose=False)
        print(f"Loaded from {repo}")
        return model
    except Exception as e:
        print(f"Failed {repo}: {e}")
        return None


def get_model(dataset, model):
    dataset = dataset.lower()
    model = model.lower()

    if dataset not in MODEL_SOURCES:
        raise ValueError(f"Unknown dataset '{dataset}'")

    sources = MODEL_SOURCES[dataset]

    for src in sources:
        repo = src["repo"]
        name = src["name_format"].format(model=model)

        loaded = _try_torchhub(repo, name, True)
        if loaded is not None:
            return loaded

    raise RuntimeError(
        f"Could not find pretrained model '{model}' for dataset '{dataset}'. "
        f"Tried {len(sources)} known sources."
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