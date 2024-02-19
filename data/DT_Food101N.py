######################################
#         Jinyer Yang
######################################


import os
import json
import pandas as pd
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

from randaugment import RandAugment


class Food101N(data.Dataset):
    def __init__(
        self, phase, data_path, anno_path, rgb_mean, rgb_std, rand_aug, output_path, logger
    ):
        super(Food101N, self).__init__()
        valid_phase = ["train", "val", "test"]
        assert phase in valid_phase
        if phase == "train":
            full_phase = "train"
        else:
            full_phase = "test"
        logger.info("====== The Current Split is : {}".format(full_phase))
        if "~" in data_path:
            data_path = os.path.expanduser(data_path)
        if "~" in anno_path:
            anno_path = os.path.expanduser(anno_path)
        logger.info(
            "====== The data_path is : {}, the anno_path is {}.".format(
                data_path, anno_path
            )
        )
        self.logger = logger

        self.phase = phase
        self.rand_aug = rand_aug
        self.data_path = data_path
        self.anno_path = anno_path

        self.transform = self.get_data_transform(phase, rgb_mean, rgb_std)

        # load all image info
        logger.info("=====> Load image info")
        self.img_paths, self.labels, self.noises = self.load_img_info()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.img_paths[index]
        label = self.labels[index]

        with open(os.path.join(self.data_path, path), "rb") as f:
            sample = Image.open(f).convert("RGB")

        if self.transform is not None:
            sample = self.transform(sample)

        # It has no practical significance, just to align with other formats.
        align = {}
        if self.phase != "train":
            return sample, label, align, align, index
        else:
            return sample, label, align, index

    #######################################
    #  Load image info
    #######################################
    def load_img_info(self):
        img_paths = []
        labels = []
        noises = []

        categories = os.listdir(self.data_path)
        if self.phase == "train":
            anno_path = os.path.join(self.anno_path, "verified_train_selected.tsv")
        else:
            anno_path = os.path.join(self.anno_path, "verified_val.tsv")
        data = pd.read_csv(anno_path, sep='\t', header=0)
        for index, row in data.iterrows():
            if self.phase == "train" and row["selected"] == 0:
                continue
            tmp_path = row["class_name/key"]
            label_name = tmp_path.split("/")[0]
            img_paths.append(tmp_path)
            labels.append(categories.index(label_name))
            noises.append(row["verification_label"])

        return img_paths, labels, noises

    #######################################
    #  transform
    #######################################
    def get_data_transform(self, phase, rgb_mean, rgb_std):
        transform_info = {
            "rgb_mean": rgb_mean,
            "rgb_std": rgb_std,
        }

        if phase == 'train':
            if self.rand_aug:
                self.logger.info('============= Using Rand Augmentation in Dataset ===========')
                trans = transforms.Compose([
                            transforms.RandomResizedCrop(112),
                            transforms.RandomHorizontalFlip(),
                            RandAugment(),
                            transforms.ToTensor(),
                            transforms.Normalize(rgb_mean, rgb_std)
                        ])
                transform_info['operations'] = ['RandomResizedCrop(112)', 'RandomHorizontalFlip()', 
                                            'RandAugment()', 'ToTensor()', 'Normalize(rgb_mean, rgb_std)']
            else:
                self.logger.info('============= Using normal transforms in Dataset ===========')
                trans = transforms.Compose([
                            transforms.RandomResizedCrop(112),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(rgb_mean, rgb_std)
                        ])
                transform_info['operations'] = ['RandomResizedCrop(112)', 'RandomHorizontalFlip()', 
                                            'ToTensor()', 'Normalize(rgb_mean, rgb_std)']
        else:
            trans = transforms.Compose([
                            transforms.Resize(128),
                            transforms.CenterCrop(112),
                            transforms.ToTensor(),
                            transforms.Normalize(rgb_mean, rgb_std)
                        ])
            transform_info['operations'] = ['Resize(128)', 'CenterCrop(112)', 'ToTensor()', 'Normalize(rgb_mean, rgb_std)']

        return trans


import sys

sys.path.append("../")
from utils.logger_utils import custom_logger

if __name__ == "__main__":
    data = Food101N(
        "train",
        data_path="~/Datasets/Food-101N_release/images",
        anno_path="~/Datasets/Food-101N_release/meta",
        rgb_mean=[0.485, 0.456, 0.406],
        rgb_std=[0.210, 0.224, 0.225],
        output_path="",
        rand_aug=False,
        logger=custom_logger(output_path="/home/yjy/tmp"),
    )
    indics = [85, 114, 129, 152, 228]
    for index in indics:
        print(data.img_paths[index], data.labels[index], data.noises[index])
