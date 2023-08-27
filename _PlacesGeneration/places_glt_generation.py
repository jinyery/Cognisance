import os
import math
import torch
import random
import pickle
import argparse
import numpy as np
from places_train_forward import data_info

os.environ["TMPDIR"] = os.path.expanduser("~/tmp")

N_CLASSES = 344
POWER_EXPONENT = 0.8
SIZE_OF_TRAIN_SET = 120000
SIZE_OF_VAL_SET = 10000
SIZE_OF_TEST_SET = 40000
SIZE_OF_TEST_SET_CBL = 80000
SIZE_OF_TEST_SET_GBL = 60000


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
    cat_probabilities = dict()
    for i, probability in enumerate(probabilities):
        cat_probabilities[sorted_cat[i][0]] = probability
    return cat_probabilities


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
def sampling_train_set(size_of_train_set, cat_inst: dict, used_inst: list):
    print("Sampling train set...")
    all_samples = list()
    cat_sample_num = generate_long_tail_cat_sample_num(size_of_train_set, cat_inst)
    for cat, insts in cat_inst.items():
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


# class-wise balance
def sampling_val_set(size_of_val_set, cat_inst: dict, used_inst: list):
    print("Sampling val set...")
    all_samples = list()
    for cat, insts in cat_inst.items():
        insts = list(set(insts) - set(used_inst))
        num_of_sample = int(size_of_val_set / len(cat_inst))
        if num_of_sample > len(insts):
            print(
                f"Warning: Request {num_of_sample}, only {len(insts)} for category-{cat}!"
            )
            num_of_sample = len(insts)
        samples = np.random.permutation(insts)[:num_of_sample].tolist()
        all_samples += samples
        used_inst += samples
    return all_samples


# long-tail
def sampling_test_set(size_of_test_set, cat_inst: dict, used_inst: list):
    print("Sampling test set...")
    all_samples = list()
    cat_sample_num = generate_long_tail_cat_sample_num(size_of_test_set, cat_inst)
    for cat, insts in cat_inst.items():
        insts = list(set(insts) - set(used_inst))
        num_of_sample = cat_sample_num[cat]
        if num_of_sample > len(insts):
            print(
                f"Warning: Request {num_of_sample}, only {len(insts)} for category-{cat}!"
            )
            num_of_sample = len(insts)
        samples = np.random.permutation(insts)[:num_of_sample].tolist()
        all_samples += samples
    return all_samples


# class-wise balance
def sampling_test_set_cbl(size_of_test_set_cbl, cat_inst: dict, used_inst: list):
    print("Sampling train set cbl...")
    all_samples = list()
    for cat, insts in cat_inst.items():
        insts = list(set(insts) - set(used_inst))
        num_of_sample = int(size_of_test_set_cbl / len(cat_inst))
        if num_of_sample > len(insts):
            print(
                f"Warning: Request {num_of_sample}, only {len(insts)} for category-{cat}!"
            )
            num_of_sample = len(insts)
        samples = np.random.permutation(insts)[:num_of_sample].tolist()
        all_samples += samples
    return all_samples


# class-wise and attr-wise balance
def sampling_test_set_gbl(
    size_of_test_set_cbl, cat_inst: dict, cat_attr_inst: dict, used_inst: list
):
    print("Sampling train set gbl...")
    all_samples = list()
    cat_attr_vec = dict()
    for cat, insts in cat_inst.items():
        print("=====>cat:", cat)
        insts = list(set(insts) - set(used_inst))
        print("=====>insts_size:", len(insts))
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
        cat_attr_vec[cat] = list(attr_vec)
        print(
            f"=====>cat_attr_vec_std:{np.std(attr_vec)}, len(cat_attr_vec):{len(attr_vec)}"
        )
    return all_samples, cat_attr_vec


if __name__ == "__main__":
    # ============================================================================
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="~/Datasets/places365/data_256",
        type=str,
        help="Indicate the root path of train data for Places-GLT.",
    )
    parser.add_argument(
        "--anno_path",
        default="~/Datasets/places365/categories_places365_merge.txt",
        type=str,
        help="Indicate the anno path of train data for Places-GLT.",
    )
    parser.add_argument(
        "--out_dir",
        default="./checkpoints/",
        type=str,
        help="Indicate the output dir of anno_file/model_file for Places-GLT.",
    )
    parser.add_argument(
        "--seed",
        default=25,
        type=int,
        help="Fix the random seed for reproduction. Default is 25.",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        help="Indicate the checkpoints path of model for Places-GLT.",
    )
    parser.add_argument(
        "--remove_strange",
        default=False,
        type=bool,
        help="Remove strange samples from the dataset for Places-GLT.",
    )
    args = parser.parse_args()

    # ============================================================================
    # fix random seed
    print("=====> Using fixed random seed: " + str(args.seed))
    seed_torch(args.seed)

    save_info_path = os.path.join(args.out_dir, "save_info.pkl")
    if os.path.exists(save_info_path):
        with open(save_info_path, "rb") as file:
            save_info = pickle.load(file)
        cat_attr_inst, cat_inst, inst_cat, inst_path, cat_strange = save_info
    else:
        save_info = data_info(args.data_path, args.anno_path, args.model_path)
        cat_attr_inst, cat_inst, inst_cat, inst_path, cat_strange = save_info
        with open(save_info_path, "wb") as file:
            pickle.dump(save_info, file)

    used_inst = list()
    if args.remove_strange:
        for tmp_cat in cat_strange:
            used_inst.extend(cat_strange[tmp_cat])
        print("Removed_insts:\n", used_inst)

    train_set = sampling_train_set(SIZE_OF_TRAIN_SET, cat_inst, used_inst)
    val_set = sampling_val_set(SIZE_OF_VAL_SET, cat_inst, used_inst)
    test_set = sampling_test_set(SIZE_OF_TEST_SET, cat_inst, used_inst)
    test_set_cbl = sampling_test_set_cbl(SIZE_OF_TEST_SET_CBL, cat_inst, used_inst)
    test_set_gbl, cat_attr_vec = sampling_test_set_gbl(
        SIZE_OF_TEST_SET_GBL, cat_inst, cat_attr_inst, used_inst
    )
    cat_ratio = generate_long_tail_cat_probabilities(cat_inst)
    outputs = {
        "train_set": train_set,
        "val_set": val_set,
        "test_set": test_set,
        "test_set_cbl": test_set_cbl,
        "test_set_gbl": test_set_gbl,
        "cat_ratio": cat_ratio,
        "inst_cat": inst_cat,
        "inst_path": inst_path,
        "cat_attr_vec": cat_attr_vec,
    }

    print("Saving anno file...")
    anno_path = os.path.join(args.out_dir, "anno_file.pkl")
    with open(anno_path, "wb") as file:
        pickle.dump(outputs, file)
    print("Finished.")
