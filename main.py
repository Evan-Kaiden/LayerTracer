import torch
import random
from PIL import Image

import argparse

from data import get_dataloader
from utils import get_device, make_gif
from load_model import get_model
from visualizer import Visualizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet20", choices=["resnet20", "resnet56", "resnet18", "resnet50"])
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "places365", "oxford-pets"])
    parser.add_argument("--frequency", default="half", help="Frequency of Prototype comparision. Values can be: [int] where we space out \
                                             comarison evenly across model [str] options are \"all\" or \"half\" corresponding to all layers or \
                                             every other layer [list/tuple] where each value must be 1 or 0 and total length must be the number of \
                                             usable exits in the model where 1 corresponds to usage of a specific layer and 0 corresponds to ignorning \
                                             that layer for example if a model has 10 usable layers a valid input would be: 1 0 1 1 0 0 1 0 1 0 ")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=10, help="Number of classes to process at a time, if program is running slow set chunk_size=NUM_CLASSES\
                                                                     for your dataset, if program is running out of memory reduce chunk size")
    parser.add_argument("--num_masks", type=int, default=5000, help="Number of random masks used to generate layer saliency, increasing \
                                                                     this number will improve visualization quality")
    parser.add_argument("--image_path_to_use", type=str, default=None, help="path/to/image if you would like to use a spcific image otherwise \
                                                                             a random image from the dataset will be used")
    parser.add_argument("--num_examples", type=int, default=1, help="sets the number of images to visualize if using random images from the selected dataset")
    parser.add_argument("--result_save_path", type=str, required=True)
    parser.add_argument("--save_prototypes", action="store_true", help="store the class means for the model")
    parser.add_argument("--prototype_save_path", type=str, help="location to save prototypes")
    parser.add_argument("--prototype_load_path", type=str, help="location to load prototypes from", default=None)
    args = parser.parse_args()

    device = get_device(verbose=True)
    model = get_model(args.dataset, args.model).eval().to(device)
    dset = get_dataloader(args.dataset, args.batch_size)

    if (args.dataset == "places365"):
        loader = dset.val_loader
    else:
        loader = dset.train_loader

    viz = Visualizer(model, loader, args.num_masks, args.chunk_size, device)


    if args.frequency.isdigit():
        frequency = int(args.frequency)
    elif args.frequency.count(" ") == 0:
        frequency = args.frequency
    else:
        frequency = []
        for use in args.frequency.split(" "):
            frequency.append(int(use))

    viz.set_granularity(frequency)

    if args.prototype_load_path is not None:
        viz.load_class_means(args.prototype_load_path, device=device)
        print(f"loaded class means from: {args.prototype_load_path}")
    else:
        viz.gather_class_means(show_progress=True)
        if args.save_prototypes:
            viz.save_class_means(args.prototype_save_path)
            print(f"Saved prototypes to {args.prototype_save_path}")

    if args.image_path_to_use is not None:
        img = Image.open(args.image_path_to_use).convert("RGB")
        img = dset.transform(img)
        img = img.unsqueeze(0) 
        saliency = viz.track_class_with_mask(img.to(device), args.batch_size)

        mean = torch.tensor(dset.mean).view(1,3,1,1).to(img.device)
        std  = torch.tensor(dset.std).view(1,3,1,1).to(img.device)
        img = img * std + mean
        img = img.squeeze(0)

        make_gif(img, saliency, args.result_save_path)
    else:
        dataset = dset.testset
        save_prefix, save_postfix = args.result_save_path.split('.')
        for i in range(args.num_examples):
            idx = random.randrange(len(dataset))
            img, _ = dataset[idx]
            img = img.unsqueeze(0)
            saliency = viz.track_class_with_mask(img.to(device), args.batch_size)

            mean = torch.tensor(dset.mean).view(1,3,1,1).to(img.device)
            std  = torch.tensor(dset.std).view(1,3,1,1).to(img.device)
            img = img * std + mean
            img = img.squeeze(0)

            make_gif(img, saliency, f"{save_prefix}_{i}.{save_postfix}")


    