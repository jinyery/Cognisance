######################################
#         Jinyer Yang
######################################


import os
import json
import pickle
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

from randaugment import RandAugment


class Places_GLT(data.Dataset):
    def __init__(
        self,
        phase,
        data_path,
        anno_path,
        testset,
        rgb_mean,
        rgb_std,
        rand_aug,
        output_path,
        logger,
    ):
        super(Places_GLT, self).__init__()
        valid_phase = ["train", "val", "test"]
        assert phase in valid_phase
        if phase == "train":
            full_phase = "train_set"
        elif phase == "val":
            full_phase = "val_set"
        else:
            full_phase = "test_set"
            if testset == "test_bl":
                full_phase += "_cbl"
            elif testset == "test_bbl":
                full_phase += "_gbl"
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

        self.dataset_info = {}
        self.phase = phase
        self.rand_aug = rand_aug
        self.data_path = data_path
        self.transform = self.get_data_transform(phase, rgb_mean, rgb_std)

        # load annotation
        with open(anno_path, "rb") as file:
            self.annotations = pickle.load(file)
        self.data = self.annotations[full_phase]
        self.cat_ratio = self.annotations["cat_ratio"]
        self.inst_cat = self.annotations["inst_cat"]
        self.inst_path = self.annotations["inst_path"]
        self.frequencies = self.load_frequencies()
        self.labels = self.load_labels()

        self.dataset_info = {
            "data": self.data,
            "inst_cat": self.inst_cat,
            "inst_path": self.inst_path,
            "frequencies": self.frequencies,
        }

        # # save dataset info
        # logger.info("=====> Save dataset info")
        # self.save_dataset_info(output_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        inst_id = self.data[index]
        path = self.inst_path[inst_id]
        label = self.inst_cat[inst_id]
        rarity = self.frequencies[inst_id]

        path = path[path.index("data_256"):]
        with open(os.path.join(self.data_path, path), "rb") as f:
            sample = Image.open(f).convert("RGB")

        if self.transform is not None:
            sample = self.transform(sample)

        # intra-class attribute SHOULD NOT be used during training
        if self.phase != "train":
            return sample, label, rarity, -1, index
        else:
            return sample, label, rarity, index

    def load_labels(self):
        labels = list()
        for idx, inst in enumerate(self.data):
            inst_id = self.data[idx]
            label = self.inst_cat[inst_id]
            labels.append(label)
        return labels
        

    #######################################
    #  Load image info
    #######################################
    def load_frequencies(self):
        frequencies = dict()
        for inst in self.data:
            if self.cat_ratio[self.inst_cat[inst]] < 0.002:
                frequencies[inst] = 0
            elif self.cat_ratio[self.inst_cat[inst]] < 0.005:
                frequencies[inst] = 1
            else:
                frequencies[inst] = 2
        return frequencies

    #######################################
    #  Save dataset info
    #######################################
    def save_dataset_info(self, output_path):
        with open(
            os.path.join(output_path, "dataset_info_{}.json".format(self.phase)), "w"
        ) as f:
            json.dump(self.dataset_info, f)

        del self.dataset_info

    #######################################
    #  transform
    #######################################
    def get_data_transform(self, phase, rgb_mean, rgb_std):
        transform_info = {
            "rgb_mean": rgb_mean,
            "rgb_std": rgb_std,
        }

        if phase == "train":
            if self.rand_aug:
                self.logger.info(
                    "============= Using Rand Augmentation in Dataset ==========="
                )
                trans = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(112),
                        transforms.RandomHorizontalFlip(),
                        RandAugment(),
                        transforms.ToTensor(),
                        transforms.Normalize(rgb_mean, rgb_std),
                    ]
                )
                transform_info["operations"] = [
                    "RandomResizedCrop(112)",
                    "RandomHorizontalFlip()",
                    "RandAugment()",
                    "ToTensor()",
                    "Normalize(rgb_mean, rgb_std)",
                ]
            else:
                self.logger.info(
                    "============= Using normal transforms in Dataset ==========="
                )
                trans = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(112),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(rgb_mean, rgb_std),
                    ]
                )
                transform_info["operations"] = [
                    "RandomResizedCrop(112)",
                    "RandomHorizontalFlip()",
                    "ToTensor()",
                    "Normalize(rgb_mean, rgb_std)",
                ]
        else:
            trans = transforms.Compose(
                [
                    transforms.Resize(128),
                    transforms.CenterCrop(112),
                    transforms.ToTensor(),
                    transforms.Normalize(rgb_mean, rgb_std),
                ]
            )
            transform_info["operations"] = [
                "Resize(128)",
                "CenterCrop(112)",
                "ToTensor()",
                "Normalize(rgb_mean, rgb_std)",
            ]

        # save dataset info
        self.dataset_info["transform_info"] = transform_info

        return trans
