from PIL import Image
import json
import os
import random
import argparse
import sys

sys.path.append("..")

from sklearn.cluster import KMeans

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from models import ResNet
from utils.clusting_utils import CoarseLeadingForest


# ============================================================================
# argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path",
    default="./checkpoints/epoch-99.pth",
    type=str,
    help="indicate the path of pretrain model for Places365.",
)
parser.add_argument(
    "--data_path",
    default="~/Datasets/places365/data_256/",
    type=str,
    help="indicate the path of train data for Places365.",
)
parser.add_argument(
    "--sub_path",
    type=str,
    nargs='+',
    required=True,
    help="indicate the sub path of train data for Places365.",
)
parser.add_argument(
    "--seed",
    default=25,
    type=int,
    help="Fix the random seed for reproduction. Default is 25.",
)
args = parser.parse_args()

if "~" in args.data_path:
    args.data_path = os.path.expanduser(args.data_path)

# ============================================================================
# fix random seed
print("=====> Using fixed random seed: " + str(args.seed))
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


# ============================================================================
# Data loader
class PlacesExtract(Dataset):
    def __init__(self, data_path, sub_path):
        super(PlacesExtract, self).__init__()
        if isinstance(sub_path, list):
            names = list()
            self.images = list()
            for sub in sub_path:
                tmp_names = os.listdir(os.path.join(data_path, sub))
                tmp_images = [os.path.join(data_path, sub, name) for name in tmp_names]
                names += tmp_names
                self.images += tmp_images
        else: 
            names = os.listdir(os.path.join(data_path, sub_path))
            self.images = [os.path.join(data_path, sub_path, name) for name in names]
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]
        with open(path, "rb") as f:
            sample = Image.open(f).convert("RGB")
            sample = self.transform(sample)
        return sample, index


def get_loader(sub_path, batch_size=128):
    dataset = PlacesExtract(args.data_path, sub_path)
    loader = data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=8
    )
    return loader, dataset


def load_module(module, module_state):
    x = module.state_dict()
    for key, _ in x.items():
        if key in module_state:
            x[key] = module_state[key]
            print("Load {:>50} from checkpoint.".format(key))
        elif "module." + key in module_state:
            x[key] = module_state["module." + key]
            print("Load {:>50} from checkpoint (rematch with module.).".format(key))
        else:
            print("WARNING: Key {} is missing in the checkpoint.".format(key))

    module.load_state_dict(x)
    return module


def load_model(path):
    print("Loading checkpoints...")
    checkpoint = torch.load(path, map_location="cpu")
    model_state = checkpoint["model"]
    model = ResNet.create_model(m_type="resnext50")
    model = load_module(model, model_state)
    model = nn.DataParallel(model).cuda()
    return model


# ============================================================================
# Get Features for Each Class
def get_outputs(model, sub_path):
    # show reconstructed image
    indexes = []
    features = []
    out_paths = []
    model.eval()
    val_loader, val_data = get_loader(sub_path=sub_path, batch_size=16)
    img_paths = val_data.images

    with torch.no_grad():
        for images, ids in val_loader:
            images = images.cuda()
            z = model(images)
            indexes.append(ids.view(-1).cpu())
            features.append(z.cpu())
        indexes = torch.cat(indexes, dim=0).numpy()
        features = torch.cat(features, dim=0).numpy()

    for idx in list(indexes):
        out_paths.append(img_paths[idx])
    return features, out_paths


# ============================================================================
# Main
def main(args):
    model = load_model(args.model_path)
    features, img_paths = get_outputs(model, args.sub_path)
    clf = CoarseLeadingForest(samples=features, min_dist_multiple=0.6, max_dist_multiple=1.6)
    paths, _ = clf.generate_path(detailed=True)
    for i, path in enumerate(paths):
        print("========================================")
        print(f"Path {str(i)} (total cn:{len(path)}):")
        for j, cn in enumerate(path):
            print(f"CoarseNode {str(j)} (total fn:{len(cn)}):")
            for n in cn:
                print(f"Index:{str(n)}, path:{img_paths[n]}")


if __name__ == "__main__":
    main(args)
