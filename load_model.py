import torch
from torchvision import models

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
    "places365": [
        {
            "repo": "CSAILVision/places365",
            "name_format": "{model}"
        },
    ],
    "tinyimagenet": [
        {
            "repo": "chenyaofo/pytorch-cifar-models",
            "name_format": "{model}",   # try anyway
        },
    ],
}


def load_places365(model_name):

    urls = {
        "resnet18": "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar",
        "resnet50": "http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar",
        "densenet161": "http://places2.csail.mit.edu/models_places365/densenet161_places365.pth.tar",
        "alexnet": "http://places2.csail.mit.edu/models_places365/alexnet_places365.pth.tar",
        "vgg16": "http://places2.csail.mit.edu/models_places365/vgg16_places365.pth.tar",
    }

    if model_name not in urls:
        raise ValueError(f"Unsupported Places365 model: {model_name}")

    if model_name == "resnet18":
        model = models.resnet18(num_classes=365)
    elif model_name == "resnet50":
        model = models.resnet50(num_classes=365)
    elif model_name == "densenet161":
        model = models.densenet161(num_classes=365)
    elif model_name == "alexnet":
        model = models.alexnet(num_classes=365)
    elif model_name == "vgg16":
        model = models.vgg16(num_classes=365)

    checkpoint = torch.hub.load_state_dict_from_url(urls[model_name], map_location="cpu")

    state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(state_dict)

    return model

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


    if dataset == "places365":
        return load_places365(model)

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
