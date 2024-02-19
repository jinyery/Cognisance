import os
import pandas as pd
import os
import math
import torch
import random
import numpy as np

os.environ["TMPDIR"] = os.path.expanduser("~/tmp")

N_CLASSES = 10
POWER_EXPONENT = 0.6
SIZE_OF_TRAIN_SET = 60000
SIZE_OF_VAL_SET = 10000
SIZE_OF_TEST_SET = 30000
SIZE_OF_TEST_SET_CBL = 30000
SIZE_OF_TEST_SET_GBL = 30000


def seed_torch(seed=25):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def generate_long_tail_distribution(n_classes=N_CLASSES, power_exponent=POWER_EXPONENT):
    probabilities = np.arange(1, n_classes + 1) ** (-power_exponent)
    probabilities /= np.sum(probabilities)
    return probabilities


def generate_long_tail_cat_probabilities(
    cat_inst: dict, n_classes=N_CLASSES, power_exponent=POWER_EXPONENT
):
    probabilities = generate_long_tail_distribution(n_classes, power_exponent)
    cat_num = {cat: len(inst) for cat, inst in cat_inst.items()}
    sorted_cat = sorted(cat_num.items(), key=lambda x: x[1], reverse=True)
    activated_cat = list()
    cat_probabilities = dict()
    for i, probability in enumerate(probabilities):
        activated_cat.append(sorted_cat[i][0])
        cat_probabilities[sorted_cat[i][0]] = probability
    return cat_probabilities, activated_cat


def generate_long_tail_cat_sample_num(
    total_count, cat_inst: dict, n_classes=N_CLASSES, power_exponent=POWER_EXPONENT
):
    probabilities = generate_long_tail_distribution(n_classes, power_exponent)
    cat_num = {cat: len(inst) for cat, inst in cat_inst.items()}
    cat_num_list = probabilities * total_count
    sorted_cat = sorted(cat_num.items(), key=lambda x: x[1], reverse=True)
    cat_sample_num = dict()
    for i, num in enumerate(cat_num_list):
        cat_sample_num[sorted_cat[i][0]] = math.ceil(num)
    return cat_sample_num

# long-tail
def sampling_train_set(size_of_train_set, cat_inst: dict, used_inst: list, activated_cat:list=None):
    print("Sampling train set...")
    all_samples = list()
    cat_sample_num = generate_long_tail_cat_sample_num(size_of_train_set, cat_inst)
    print(cat_sample_num)
    for cat, insts in cat_inst.items():
        if activated_cat is not None and cat not in activated_cat:
            continue
        insts = list(set(insts) - set(used_inst))
        num_of_sample = cat_sample_num[cat]
        if num_of_sample > len(insts):
            print(
                f"Warning: Request {num_of_sample}, only {len(insts)} for category-{cat}!"
            )
            num_of_sample = len(insts)
        samples = np.random.permutation(insts)[:num_of_sample].tolist()
        all_samples += samples
        used_inst += samples
    return all_samples


# cat_inst = dict()
# data_path="/home/yjy/Datasets/Food-101N_release/images"
# anno_path="/home/yjy/Datasets/Food-101N_release/meta"
# categories = os.listdir(data_path)
# for cat in categories:
#     cat_inst[cat] = list()
# anno_path_in = os.path.join(anno_path, "verified_train.tsv")
# data = pd.read_csv(anno_path_in, sep='\t', header=0)
# for index, row in data.iterrows():
#     tmp_path = row["class_name/key"]
#     label_name = tmp_path.split("/")[0]
#     cat_inst[label_name].append(index)

# used_inst = list()
# selected = sampling_train_set(30000, cat_inst, used_inst)
# #print(l)
# data['selected'] = 0
# data.loc[selected,"selected"] = 1
# anno_path_out = os.path.join(anno_path, "verified_train_selected.tsv")
# data.to_csv(anno_path_out, sep='\t', index=None)

# dt = pd.read_csv("/home/yjy/Datasets/Food-101N_release/meta/verified_train_selected.tsv", sep='\t', header=0)
# print(dt)
# print(dt["selected"][0]==1)


cat_inst = dict()
for cat in range(10):
    cat_inst[cat] = list()
img_paths = []
labels = []

data_path="/home/yjy/Datasets/animal10n"
img_paths = os.listdir(os.path.join(data_path, "training"))
for index, path in enumerate(img_paths):
    cat_ind = int(path[0])
    cat_inst[cat_ind].append(index)

used_inst = list()
selected = sampling_train_set(24000, cat_inst, used_inst)

import shutil
from_dir = "/home/yjy/Datasets/animal10n/training"
to_dir = "/home/yjy/Datasets/animal10n/training_selected"
if not os.path.exists(to_dir):
    os.makedirs(to_dir)

img_paths = np.array(img_paths)
for path in img_paths[selected]:
    from_file = os.path.join(from_dir, path)
    to_file = os.path.join(to_dir, path)
    shutil.copyfile(from_file, to_file)