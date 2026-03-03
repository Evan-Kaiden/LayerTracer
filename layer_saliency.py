import torch
import random

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
    
