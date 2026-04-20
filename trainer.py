import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm


class PrototypeNet(nn.Module):
    def __init__(self, net, tracked_layers, use_layer, prototypes, chunk_size=10):
        super().__init__()
        self.net = net
        self.tracked_layers = tracked_layers
        self.use_layer = use_layer
        self.chunk_size = chunk_size
        self.prototypes = nn.ParameterList()

        for layer in range(len(prototypes)):
            layer_prototypes = nn.ParameterList()
            for cls in range(len(prototypes[0])):
                layer_prototypes.append(nn.Parameter(prototypes[layer][cls].avg.clone().detach()))
            self.prototypes.append(layer_prototypes)

    def _get_probs(self, x, prototypes):
        device = x.device
        means = torch.stack(list(prototypes), dim=0).to(device)
        diffs = []
        
        for i in range(0, means.size(0), self.chunk_size): 
            diffs.append((x.unsqueeze(1) - means[i:i+self.chunk_size].unsqueeze(0)))
        diff = torch.cat(diffs, dim=1)
        diff = diff.reshape(diff.shape[0], diff.shape[1], -1)
        dists = torch.linalg.norm(diff.to(device), dim=-1)
        return torch.softmax(-dists, dim=-1)
    

    def forward(self, x):
        handles = []

        def make_hook(idx):
            def hook(module, inp, out):
                if not self.use_layer[idx]:
                    return
                
                proto_idx = sum(self.use_layer[:idx])
                module._result = self._get_probs(out, self.prototypes[proto_idx])
            return hook
        
        
        for i, layer in enumerate(self.tracked_layers):
            handles.append(layer.register_forward_hook(make_hook(i)))

        try:
            self.net(x)
        finally:
            for h in handles:
                h.remove()

        results = []
        for layer in self.tracked_layers:
            if hasattr(layer, "_result"):
                results.append(layer._result)
                del layer._result

        return results

    def train_prototypes(self, epochs, loader, device):
        self.train()
        for p in self.net.parameters():
            p.requires_grad = False

        criterion = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(params=self.parameters(), lr=2e-4) # a conservative lr

        for epoch in range(epochs):
            correct, total = [0 for _ in range(sum(self.use_layer))], [0 for _ in range(sum(self.use_layer))]
            for x, y in tqdm(loader, leave=False, desc=f'Train Epoch {epoch + 1}'):
                x, y = x.to(device), y.to(device)

                results = self.forward(x)

                loss = 0
                for idx, res in enumerate(results):
                    batch_correct = res.argmax(dim=-1) == y
                    correct[idx] += batch_correct.sum()
                    total[idx] += y.size(0)
                    loss += criterion(res, y)

                opt.zero_grad()
                loss.backward()
                opt.step()

            print(f'Epoch {epoch+1}:\n  Prototype | Accuracy')

            active_indices = [i for i, used in enumerate(self.use_layer) if used]

            for idx in range(len(correct)):
                layer_idx = active_indices[idx]
                layer_name = self.tracked_layers[layer_idx]
                pad = " " * (13 - len(str(idx)))
                print(f"{pad}{idx} | {correct[idx] / total[idx] * 100:.2f}% | {layer_name}")
                
        return self.prototypes
    
    def test(self, loader, device):
        self.eval()
        correct, total = [0 for _ in range(sum(self.use_layer))], [0 for _ in range(sum(self.use_layer))]
        with torch.no_grad():
            for x, y in tqdm(loader, leave=False, desc='Testing'):
                x, y = x.to(device), y.to(device)

                results = self.forward(x)

                for idx, res in enumerate(results):
                    batch_correct = res.argmax(dim=-1) == y
                    correct[idx] += batch_correct.sum()
                    total[idx] += y.size(0)

        accs = [correct[idx].sum() / total[idx] for idx in range(len(correct))]
        
        return accs
        
        
    

            
    
            
    