import torch
import torch.nn as nn
from torchvision import models
import timm


DATASET_NUM_CLASSES = {
    "cifar10": 10,
    "cifar100": 100,
    "places365": 365,
    "tinyimagenet": 200,
    "oxford-pets": 37,
}

# TorchHub sources for datasets other than Oxford Pets
MODEL_SOURCES = {
    "cifar10": [
        {"repo": "chenyaofo/pytorch-cifar-models", "name_format": "cifar10_{model}"},
    ],
    "cifar100": [
        {"repo": "chenyaofo/pytorch-cifar-models", "name_format": "cifar100_{model}"},
    ],
    "places365": [
        {"repo": "CSAILVision/places365", "name_format": "{model}"},
    ],
    "tinyimagenet": [
        {"repo": "chenyaofo/pytorch-cifar-models", "name_format": "{model}"},
    ],
}


def replace_classifier(model, num_classes):
    """
    Automatically replaces final classifier layer for common torchvision/timm models.
    Supports ResNet, DenseNet, VGG, AlexNet, SqueezeNet, MobileNet, EfficientNet, etc.
    """
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        return model

    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        if isinstance(model.classifier[-1], nn.Linear):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
            return model
        if isinstance(model.classifier[-1], nn.Conv2d):
            in_channels = model.classifier[-1].in_channels
            model.classifier[-1] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
            return model

    raise RuntimeError(f"Automatic classifier replacement not implemented for {type(model)}")

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

    print("Loading Places365 pretrained weights")
    model_fn = getattr(models, model_name)
    model = model_fn(num_classes=365)

    checkpoint = torch.hub.load_state_dict_from_url(urls[model_name], map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(state_dict)

    return model


def _try_torchhub(repo, model_name, pretrained=True):
    try:
        print(f"Trying TorchHub: {repo} -> {model_name}")
        model = torch.hub.load(repo, model_name, pretrained=pretrained, verbose=False)
        print("Loaded successfully from TorchHub")
        return model
    except Exception as e:
        print(f"TorchHub failed: {e}")
        return None

def load_imagenet_fallback(model_name, dataset):
    print("Falling back to ImageNet pretrained weights")
    if dataset not in DATASET_NUM_CLASSES:
        raise ValueError(f"No class metadata for dataset '{dataset}'")

    num_classes = DATASET_NUM_CLASSES[dataset]

    try:
        model_fn = getattr(models, model_name)
    except AttributeError:
        raise ValueError(f"Model '{model_name}' not found in torchvision.models")

    model = model_fn(weights="DEFAULT")
    model = replace_classifier(model, num_classes)
    return model

def load_oxford_pets(model_name):
    """
    Attempts to load a timm model pretrained on Oxford Pets from Hugging Face.
    Falls back to ImageNet if not available.
    """
    num_classes = DATASET_NUM_CLASSES["oxford-pets"]

    hf_model_id = f"hf-hub:nateraw/{model_name}-oxford-iiit-pet" 
    try:
        print(f"Trying timm HuggingFace Oxford Pets model: {hf_model_id}")
        model = timm.create_model(hf_model_id, pretrained=True)
        print("Loaded Oxford Pets pretrained model from HuggingFace via timm")
        return model
    except Exception as e:
        print(f"Could not load Oxford Pets timm model: {e}")
        print("Falling back to ImageNet pretrained weights")
        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        return model

def get_model(dataset, model):
    dataset = dataset.lower()
    model = model.lower()

    if dataset not in DATASET_NUM_CLASSES:
        raise ValueError(f"Unknown dataset '{dataset}'")

    # Special-cases
    if dataset == "places365":
        return load_places365(model)
    if dataset == "oxford-pets":
        return load_oxford_pets(model)

    # Try TorchHub sources
    if dataset in MODEL_SOURCES:
        for src in MODEL_SOURCES[dataset]:
            repo = src["repo"]
            name = src["name_format"].format(model=model)
            loaded = _try_torchhub(repo, name, pretrained=True)
            if loaded is not None:
                return loaded

    # Final fallback: ImageNet
    return load_imagenet_fallback(model, dataset)