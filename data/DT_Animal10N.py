######################################
#         Jinyery Yang
######################################


import os
import json

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

from randaugment import RandAugment


class Animal10N(data.Dataset):
    def __init__(
        self, phase, data_path, rgb_mean, rgb_std, rand_aug, output_path, logger
    ):
        super(Animal10N, self).__init__()
        valid_phase = ["train", "val", "test"]
        assert phase in valid_phase
        if phase == "train":
            full_phase = "train"
        else:
            full_phase = "test"
        logger.info("====== The Current Split is : {}".format(full_phase))
        if "~" in data_path:
            data_path = os.path.expanduser(data_path)
        # if "~" in anno_path:
        #     anno_path = os.path.expanduser(anno_path)
        # logger.info('====== The data_path is : {}, the anno_path is {}.'.format(data_path, anno_path))
        self.logger = logger

        self.phase = phase
        self.rand_aug = rand_aug
        self.data_path = data_path

        self.transform = self.get_data_transform(phase, rgb_mean, rgb_std)

        # load all image info
        logger.info("=====> Load image info")
        self.img_paths, self.labels = self.load_img_info()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.img_paths[index]
        label = self.labels[index]

        path = os.path.basename(path)
        if self.phase == "train":
            img_path = os.path.join(self.data_path, "training")
        else:
            img_path = os.path.join(self.data_path, "testing")
        with open(os.path.join(img_path, path), "rb") as f:
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

        if self.phase == "train":
            img_paths = os.listdir(os.path.join(self.data_path, "training_selected"))
        else:
            img_paths = os.listdir(os.path.join(self.data_path, "testing"))
        
        for path in img_paths:
            labels.append(int(path[0]))

        return img_paths, labels

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
                        transforms.RandomResizedCrop(64, scale=(0.5, 1.0)),
                        transforms.RandomHorizontalFlip(),
                        RandAugment(),
                        transforms.ToTensor(),
                        transforms.Normalize(rgb_mean, rgb_std),
                    ]
                )
                transform_info["operations"] = [
                    "RandomResizedCrop(64, scale=(0.5, 1.0)),",
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
                        transforms.RandomResizedCrop(64, scale=(0.5, 1.0)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(rgb_mean, rgb_std),
                    ]
                )
                transform_info["operations"] = [
                    "RandomResizedCrop(64, scale=(0.5, 1.0)),",
                    "RandomHorizontalFlip()",
                    "ToTensor()",
                    "Normalize(rgb_mean, rgb_std)",
                ]
        else:
            trans = transforms.Compose(
                [
                    transforms.Resize(81),
                    transforms.CenterCrop(64),
                    transforms.ToTensor(),
                    transforms.Normalize(rgb_mean, rgb_std),
                ]
            )
            transform_info["operations"] = [
                "Resize(81)",
                "CenterCrop(60)",
                "ToTensor()",
                "Normalize(rgb_mean, rgb_std)",
            ]

        return trans


import sys

sys.path.append("../")
from utils.logger_utils import custom_logger

if __name__ == "__main__":
    data = Animal10N(
        "train",
        data_path="~/Datasets/animal10n",
        rgb_mean=[0.485, 0.456, 0.406],
        rgb_std=[0.210, 0.224, 0.225],
        output_path="",
        rand_aug=False,
        logger=custom_logger(output_path="./"),
    )
    indics = [11659, 18258, 42169, 16178, 805, 12785, 38105, 9525, 48701, 22978, 12453, 33385, 12583, 48162, 1843, 15032, 41016, 5005, 14104, 22358, 47551, 27922, 47251, 36717, 794, 27785, 13840, 4346, 39420, 28380, 19322, 3835, 22124, 29211, 2498, 34188, 34862, 18565, 3634, 25809, 38972, 8253, 5395, 46363, 14381, 4537, 37442, 43062, 17073, 30976, 22164, 14114, 48156, 8168, 41918, 32955, 4669, 47227, 785, 28860, 42960, 17654, 34593, 10857, 26215, 41900, 11368, 167, 7366, 3943, 24484, 652, 37643, 13005, 27432, 42179, 7490, 33113, 39027, 16512, 36670, 30257, 11881, 45784, 552, 16712, 48940, 11637, 2303, 25637, 21245, 47603, 18204, 15244, 46797, 6118, 11257, 16305, 46476, 41589, 40852, 42992, 19093, 48418, 736, 10970, 6073, 10842, 45839, 6468, 35481, 2783, 33996, 30317, 1837, 18572, 13697, 38519, 46839, 18211, 42809, 45809, 25136, 10556, 49660, 4422, 6245, 11880, 22957, 37103, 13298, 985, 24356, 16789, 20015, 40869, 26704, 36548, 18580, 30107, 25390, 25126, 49743, 31627, 186, 7430, 1543, 44503, 28312, 7057, 5844, 3453, 45437, 47829, 12751, 25937, 21297, 19502, 28324, 2971, 10348, 26449, 10638, 12459, 8744, 2691, 35071, 18543, 27416, 9984, 5087, 1490, 45693, 23550, 41416, 27145, 14917, 39896, 46378, 29250, 41173, 48839, 41256, 4580, 46037, 38569, 27058, 14002, 22806, 7291, 11757, 30140, 42411, 48538, 7220, 2043, 4492, 34561, 15853, 3435, 45801, 42280, 36498, 30376, 34505, 32110, 23012, 25396, 7492, 32937, 46241, 37215, 44766, 42780, 28256, 47468, 8635, 2985, 32078, 18593, 25949, 23822, 45941, 16118, 28812, 6893, 29522, 28169, 13115, 32262, 1121, 19224, 17006, 37387, 45154, 26132, 18240, 14363, 37141, 36283, 47141, 20172, 38298, 41601, 24712, 11671, 43207, 36884, 17230, 29485, 45510, 49299, 18178, 46461, 27231, 5667, 17861, 37309, 11888, 15373, 1716, 32116, 12592, 30021, 1205, 21967, 29688, 27873, 34126, 28783, 43620, 39321, 19591, 26577, 25054, 42868, 18331, 42123, 31287, 17211, 30336, 45852, 31211, 3245, 31954, 11186, 4605, 7757, 4402, 32217, 13662, 31357, 41856, 16214, 13429, 11255, 26960, 14953, 49184, 16083, 33151, 20234, 34260, 31130, 27647, 45129, 6513, 17388, 5373, 32238, 49955, 10455, 33320, 11921, 48802, 4374, 1828, 10957, 33622, 32698, 41841, 36906, 27617, 25834, 19145, 12793, 17798, 24478, 40981, 14622, 45635, 35966, 16496, 11042, 24395, 18151, 1097, 39611, 39398, 44965, 35423, 45752, 16500, 8832, 1018, 29194, 13035, 20028, 46456, 40056, 6606, 33898, 5751, 544, 45670, 49499, 32215, 33581, 42610, 25270, 21435, 31639, 22471, 37984, 8836, 1213, 600, 10106, 47507, 6275, 49552, 43561, 30919, 32093, 12972, 29268, 31162, 27352, 22450, 25500, 39254, 47572, 21140, 49105, 30362, 20259, 12282, 17251, 4162, 7519, 46514, 8793, 40141, 36658, 866, 45673, 11433, 28683, 3413, 30634, 7467, 29407, 4907, 31193, 14080, 24622, 35414, 49026, 40704, 38860, 43688, 29104, 37272, 1952, 38436, 17492, 46981, 15880, 35427, 34112, 16675, 45879, 34637, 49304, 38198, 29781, 39493, 46668, 48183, 13189, 30793, 24920, 743, 28776, 4336, 9420, 1787, 48139, 5884, 40092, 4859, 46982, 16558, 13914, 6031, 15438, 30433, 13602, 31537, 5263, 14633, 27894, 9197, 29771, 33021, 39301, 42989, 45465, 6873, 14607, 39221, 29463, 10791, 22801, 20864, 7375, 8881, 25267, 1178, 46581, 29866, 33348, 5394, 43447, 48969, 13791, 6137, 16655, 24745, 30010, 17815, 41066, 43566, 1191, 27443, 31824, 21605, 43147, 17852, 45530, 41463, 15045, 16853, 21003, 34354, 42433, 48414, 12964, 20276, 41020, 48346, 30935, 21596, 8330, 37827, 35364, 4379, 31831, 10942, 43734, 22171, 19622, 19457, 48752, 5780, 49467, 25191, 14227, 40526, 14266, 21377, 25069, 40426, 23463, 40663, 29482, 3819, 32769, 48602, 26262, 37715, 26293, 44664, 32470, 14811, 245, 36241, 3732, 48335, 8244, 27491, 36640, 45060, 45664, 6223, 13018, 10990, 18605, 8899, 9235, 289, 26815, 10000, 39969, 28534, 28586, 10244, 38270, 33, 17469, 36205, 9110, 30557, 48064, 22787, 29534, 20529, 41487, 43309, 9701, 39982, 20039, 49118, 14658, 10587, 12516, 29296, 30369, 20559, 5381, 49398, 43383, 37978, 624, 27175, 45818, 14867, 15565, 29570, 28912, 11429, 3859, 11378, 38395, 26446, 9179, 4654, 46141, 3184, 48095, 21591, 23052, 43046, 4922, 5805, 26254, 29377, 11758, 191, 25615, 30801, 33314, 16679, 37800, 47908, 40331, 31926, 38495, 37473, 17860, 35747, 33929, 46010, 3626, 23915, 49722, 16573, 34199, 41089, 24473, 33160, 31412, 37907, 8778, 23322, 4430, 5259, 19927, 10468, 9966, 34067, 43922, 37, 26975, 20345, 14152, 28286, 49058, 41166, 3600, 10154, 18828, 10517, 9679, 40025, 16004, 23741, 8513, 9047, 41623, 28882, 30638, 3550, 15557, 3029, 12945, 42354, 19024, 4685, 49697, 21746, 9867, 37525, 39088, 18058, 48634, 39021, 13502, 14355, 15166, 39809, 7953, 46790, 9646, 5709, 20403, 938, 46906, 32256, 23331, 2041, 47981, 33671, 31825, 49218, 5601, 49670, 48240, 26584, 41704, 19417, 36943, 16232, 22708, 13329, 49004, 7827, 24973, 24715, 6596, 38855, 14989, 10235, 38909, 43003, 5850, 32860, 33954, 10215, 31930, 47786, 17250, 349, 27266, 27054, 44810, 5684, 34201, 25460, 43962, 35503, 15220, 26970, 206, 48177, 34626, 31041, 37254, 15157, 1623, 26114, 2371, 142, 23703, 37432, 40001, 23191, 15899, 30473, 11242, 26448, 8437, 10672, 43752, 17546, 11577, 29046, 6915, 47673, 37028, 28529, 29259, 21147, 39550, 44265, 33459, 32523, 41852, 34044, 14596, 44025, 47235, 39318, 6840, 13031, 4946, 501, 36211, 16900, 36073]
    for index in indics:
        print(data.img_paths[index])
