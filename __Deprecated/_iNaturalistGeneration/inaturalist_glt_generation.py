import os
import math
import torch
import random
import pickle
import argparse
import numpy as np
from inaturalist_train_forward import data_info


def seed_torch(seed=25):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sampling_train_set(
    size_of_train_set, cat_inst: dict, cat_ratio: dict, used_inst: list
):
    all_samples = list()
    for cat, insts in cat_inst.items():
        insts = np.array(set(insts) - set(used_inst))
        num_of_sample = math.ceil(cat_ratio[cat] * size_of_train_set)
        if num_of_sample > len(insts):
            print(
                f"Warning: Request {num_of_sample}, only {len(insts)} for category-{cat}!"
            )
            num_of_sample = len(insts)
        samples = np.random.permutation(insts)[:num_of_sample].tolist()
        all_samples += samples
        used_inst += samples
    return all_samples


def sampling_val_set(size_of_val_set, cat_inst: dict, cat_ratio: dict, used_inst: list):
    all_samples = list()
    for cat, insts in cat_inst.items():
        insts = np.array(set(insts) - set(used_inst))
        num_of_sample = math.ceil(cat_ratio[cat] * size_of_val_set)
        if num_of_sample > len(insts):
            print(
                f"Warning: Request {num_of_sample}, only {len(insts)} for category-{cat}!"
            )
            num_of_sample = len(insts)
        samples = np.random.permutation(insts)[:num_of_sample].tolist()
        all_samples += samples
        used_inst += samples
    return all_samples


def sampling_test_set(
    size_of_test_set, cat_inst: dict, cat_ratio: dict, used_inst: list
):
    all_samples = list()
    for cat, insts in cat_inst.items():
        insts = np.array(set(insts) - set(used_inst))
        num_of_sample = math.ceil(cat_ratio[cat] * size_of_test_set)
        if num_of_sample > len(insts):
            print(
                f"Warning: Request {num_of_sample}, only {len(insts)} for category-{cat}!"
            )
            num_of_sample = len(insts)
        samples = np.random.permutation(insts)[:num_of_sample].tolist()
        all_samples += samples
    return all_samples


def sampling_test_set_cbl(size_of_test_set_cbl, cat_inst: dict, used_inst: list):
    all_samples = list()
    for cat, insts in cat_inst.items():
        insts = np.array(set(insts) - set(used_inst))
        num_of_sample = int(size_of_test_set_cbl / len(cat_inst))
        if num_of_sample > len(insts):
            print(
                f"Warning: Request {num_of_sample}, only {len(insts)} for category-{cat}!"
            )
            num_of_sample = len(insts)
        samples = np.random.permutation(insts)[:num_of_sample].tolist()
        all_samples += samples
    return all_samples


def sampling_test_set_gbl(
    size_of_test_set_cbl, cat_inst: dict, cat_attr_inst: dict, used_inst: list
):
    all_samples = list()
    for cat, insts in cat_inst.items():
        insts = np.array(set(insts) - set(used_inst))
        num_of_sample = math.ceil(size_of_test_set_cbl / len(cat_inst))
        if num_of_sample > len(insts):
            print(
                f"Warning: Request {num_of_sample}, only {len(insts)} for category-{cat}!"
            )
            num_of_sample = len(insts)
            all_samples += insts
            continue

        inst_attr = {inst: [0] * len(cat_attr_inst[cat]) for inst in cat_inst[cat]}
        for i, attr in enumerate(cat_attr_inst[cat]):
            for inst in cat_attr_inst[cat][attr]:
                inst_attr[inst][i] += 1

        count = 0
        attr_vec = np.zeros(len(cat_attr_inst[cat]))
        while count < num_of_sample:
            min_idx = -1
            min_std = float("inf")
            min_inst = float("inf")
            for i, inst in enumerate(insts):
                tmp_vec = attr_vec.copy()
                tmp_vec += inst_attr[inst]
                tmp_std = np.std(tmp_vec)
                if tmp_std < min_std:
                    min_idx = i
                    min_std = tmp_std
                    min_inst = inst
            insts = np.delete(insts, min_idx)
            attr_vec += inst_attr[min_inst]
            all_samples.append(min_inst)
            count += 1
    return all_samples


if __name__ == "__main__":
    # ============================================================================
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="~/Datasets/inaturalist/",
        type=str,
        help="indicate the root path of train data for iNaturalist.",
    )
    parser.add_argument(
        "--anno_path",
        default="~/Datasets/inaturalist/train2018.json",
        type=str,
        help="indicate the anno path of train data for iNaturalist.",
    )
    parser.add_argument(
        "--out_path",
        default="./inaturalist_anno.pkl",
        type=str,
        help="indicate the output path of anno data for iNaturalist-LT.",
    )
    parser.add_argument(
        "--seed",
        default=25,
        type=int,
        help="Fix the random seed for reproduction. Default is 25.",
    )
    args = parser.parse_args()

    # ============================================================================
    # fix random seed
    print("=====> Using fixed random seed: " + str(args.seed))
    seed_torch(str(args.seed))

    size_of_train_set = 170000
    size_of_val_set = 30000
    size_of_test_set = 50000
    size_of_test_set_cbl = 30000
    size_of_test_set_gbl = 20000

    cat_attr_inst, cat_inst, cat_ratio, inst_cat, inst_path = data_info(
        args.data_path, args.anno_path
    )
    used_inst = list()
    train_set = sampling_train_set(size_of_train_set, cat_inst, cat_ratio, used_inst)
    val_set = sampling_val_set(size_of_val_set, cat_inst, cat_ratio, used_inst)
    test_set = sampling_test_set(size_of_test_set, cat_inst, cat_ratio, used_inst)
    test_set_cbl = sampling_test_set_cbl(size_of_test_set_cbl, cat_inst, used_inst)
    test_set_gbl = sampling_test_set_gbl(
        size_of_test_set_gbl, cat_inst, cat_attr_inst, used_inst
    )
    outputs = {
        "train_set": train_set,
        "val_set": val_set,
        "test_set": test_set,
        "test_set_cbl": test_set_cbl,
        "test_set_gbl": test_set_gbl,
        "cat_ratio": cat_ratio,
        "inst_cat": inst_cat,
        "inst_path": inst_path,
    }

    with open(args.out_path, "wb") as file:
        pickle.dump(outputs, file)
