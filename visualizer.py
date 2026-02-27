import torch
import torch.nn as nn

from utils import LayerSaliency, AverageMeter

class Visualizer():
    def __init__(self, net, dataloader, repeats=1000, device='cpu'):

        self.net = net
        self.dataloader = dataloader
        self.device = device
        self.usable_layer_types = (nn.Conv2d)
        self.tracked_layers = []

        for _, module in net.named_modules():
            if len(list(module.children())) > 0:
                continue
            if not isinstance(module, self.usable_layer_types):
                continue
            self.tracked_layers.append(module)

        self.usable_layers = len(self.tracked_layers)
        self.num_classes = [*net.named_children()][-1][1].out_features
        print(f"Current Model Has {self.usable_layers} Usable Layers For Visualization, And {self.num_classes} Classes")

        image_size = next(iter(self.dataloader))[0].size(-1)
        self.visualizer = LayerSaliency(image_size, repeats=repeats)
     
    def set_granularity(self, granularity):
        if isinstance(granularity, str):
            if granularity == "all":
                self.use_layer = [1] * self.usable_layers
            elif granularity == "half":
                n = max(1, self.usable_layers // 2)
                idx = torch.linspace(0, self.usable_layers - 1, n).round().long()
                self.use_layer = [1 if i in idx else 0 for i in range(self.usable_layers)]
            else:
                raise NotImplementedError

        elif isinstance(granularity, int):
            if granularity <= 0:
                raise ValueError("granularity must be > 0")

            n = min(granularity, self.usable_layers)
            idx = torch.linspace(0, self.usable_layers - 1, n).round().long()
            self.use_layer = [1 if i in idx else 0 for i in range(self.usable_layers)]

        elif isinstance(granularity, (list, tuple)):
            if len(granularity) != self.usable_layers:
                raise ValueError("granularity must be the same size as number of usable layers")
            self.use_layer = granularity
            
        else:
            raise NotImplementedError


        self.num_prototypes = sum(self.use_layer)
        self.prototypes = [[AverageMeter() for _ in range(self.num_classes)] for _ in range(self.num_prototypes)]
        print(
            f"Using {self.num_prototypes} prototypes "
            f"(requested {granularity})"
        )

    def gather_class_means(self, show_progress=False):
        self.net.eval()
        self.net.to(self.device)
        handles = []

        def make_hook(idx):
            def hook(module, inp, out):

                if not self.use_layer[idx]:
                    return

                labels = module._forward_labels
                proto_idx = sum(self.use_layer[:idx])

                for i in range(out.shape[0]):
                    cls = labels[i].item()
                    self.prototypes[proto_idx][cls].update(out[i].detach().cpu())

            return hook

        for i, layer in enumerate(self.tracked_layers):
            handles.append(layer.register_forward_hook(make_hook(i)))

        total = len(self.dataloader.dataset)
        seen = 0
        with torch.no_grad():
            for x, y in self.dataloader:
                x = x.to(self.device)
                y = y.to(self.device)

                for layer in self.tracked_layers:
                    layer._forward_labels = y

                self.net(x)

                seen += x.size(0)
                percent = int(100 * seen / total)
                if show_progress:
                    print(f"\r{percent}% ({seen}/{total})", end="")

        for h in handles:
            h.remove()

        for layer in self.tracked_layers:
            if hasattr(layer, "_forward_labels"):
                del layer._forward_labels
    
    def _get_probs(self, x, prototypes):
        means = torch.stack([cm.avg for cm in prototypes], dim=0).to(x.device)
        diff = x.unsqueeze(1) - means.unsqueeze(0)
        diff = diff.reshape(diff.shape[0], diff.shape[1], -1)
        dists = torch.linalg.norm(diff, dim=-1)
        return torch.softmax(-dists, dim=-1)
    
    def _forward_with_prototype_comparision(self, x, cls):
        self.net.eval()
        self.net.to(self.device)
        handles = []

        def make_hook(idx):
            def hook(module, inp, out):
                if not self.use_layer[idx]:
                    return
                
                target_cls = module._forward_labels
                proto_idx = sum(self.use_layer[:idx])
                module._result = self._get_probs(out, self.prototypes[proto_idx])[:, target_cls]
            return hook

        for i, layer in enumerate(self.tracked_layers):
            handles.append(layer.register_forward_hook(make_hook(i)))

        with torch.no_grad():
            x = x.to(self.device)
            for layer in self.tracked_layers:
                layer._forward_labels = cls
            self.net(x)

        for h in handles:
            h.remove()

        results = []
        for layer in self.tracked_layers:
            if hasattr(layer, "_forward_labels"):
                del layer._forward_labels
            if hasattr(layer, "_result"):
                results.append(layer._result)
                del layer._result

        return torch.stack(results, dim=0)

    def track_class_with_mask(self, x, batch_size):
        assert x.size(0) == 1, "Batch Size Must Be 1"
        x = x.to(self.device)
        masks = self.visualizer._generate_masks().to(self.device)
        pred_cls = self.net(x).argmax(-1).item()
        baseline_proto_dists = self._forward_with_prototype_comparision(x, pred_cls)

        masked_images = masks * x
        all_masked_proto_dists = []

        with torch.no_grad():
            for i in range(0, masked_images.size(0), batch_size):
                batch = masked_images[i:i+batch_size]
                dists = self._forward_with_prototype_comparision(batch, pred_cls)
                all_masked_proto_dists.append(dists)

        masked_proto_dists = torch.cat(all_masked_proto_dists, dim=1)
        masks = 1 - masks
        saliency = self.visualizer.generate_visual(masks, baseline_proto_dists, masked_proto_dists)
        return saliency
        

        