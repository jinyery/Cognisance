import os
import json
import pickle
import torch.utils.data as data
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import sys

sys.path.append("..")
from utils.clusting_utils import CoarseLeadingForest
from models import ResNet, ClassifierFC
import torch.optim as optim
from operator import add
from functools import reduce
from tqdm import tqdm

NUM_EPOCH = 100
SAVE_EPOCH = 10
BATCH_SIZE = 64
PRINT_STEPS = 100
NUM_CLASSES = 344
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


class Places365TrainSet(Dataset):
    def __init__(self, data_path, anno_path):
        if "~" in data_path:
            self.data_path = os.path.expanduser(data_path)
        if "~" in anno_path:
            self.anno_path = os.path.expanduser(anno_path)

        self.insts = list()
        self.cat_inst = dict()
        self.inst_cat = dict()
        self.inst_path = dict()
        self.cat_strange = dict()

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.init_data()

    def init_data(self):
        print("Initializing dataset...")
        with open(self.anno_path, "r") as file:
            lines = file.readlines()

        i = 0  # inst index
        j = 0  # cat index
        last_cat_dummy = ""
        for line in lines:
            cat_dummy = line.split(" ")[1]
            if last_cat_dummy == cat_dummy:
                j -= 1
            last_cat_dummy = cat_dummy
            cat_path = line.split(" ")[0][1:]
            cat_files = os.listdir(os.path.join(self.data_path, cat_path))
            if j not in self.cat_inst:
                self.cat_inst[j] = list()
            for cat_file in cat_files:
                img_path = os.path.join(self.data_path, cat_path, cat_file)
                self.insts.append(i)
                self.cat_inst[j].append(i)
                self.inst_cat[i] = j
                self.inst_path[i] = img_path
                i += 1
            j += 1

    def __len__(self):
        return len(self.insts)

    def __getitem__(self, index):
        path = self.inst_path[index]
        catgory = self.inst_cat[index]

        with open(os.path.join(self.data_path, path), "rb") as f:
            sample = Image.open(f).convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, catgory

    def get_loader(
        self,
        num_workers=8,
        batch_size=BATCH_SIZE,
    ):
        return data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
        )


class Places365ExtractSet(DataLoader):
    def __init__(
        self,
        category,
        all_set: Places365TrainSet,
    ):
        self.category = category
        self.data_path = all_set.data_path
        self.cat_inst = all_set.cat_inst
        self.inst_cat = all_set.inst_cat
        self.inst_path = all_set.inst_path
        self.insts = self.cat_inst[self.category]
        all_set.cat_strange[category] = list()
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
        return len(self.insts)

    def __getitem__(self, index):
        inst_id = self.insts[index]
        path = self.inst_path[inst_id]
        category = self.inst_cat[inst_id]

        with open(os.path.join(self.data_path, path), "rb") as f:
            sample = Image.open(f).convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, category

    def order_inst(self, order_list):
        insts = list()
        for order in order_list:
            insts.append(self.insts[order])
        return insts

    def get_loader(
        self,
        num_workers=8,
        batch_size=BATCH_SIZE,
    ):
        return data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )


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
    classifier_state = checkpoint["classifier"]

    model = ResNet.create_model(m_type="resnext50")
    classifier = ClassifierFC.create_model(feat_dim=2048, num_classes=NUM_CLASSES)
    model = load_module(model, model_state)
    classifier = load_module(classifier, classifier_state)
    model = nn.DataParallel(model).cuda()
    classifier = nn.DataParallel(classifier).cuda()
    return model, classifier


def train_model(train_set, num_epoch=NUM_EPOCH):
    train_loader = train_set.get_loader()
    model = ResNet.create_model(m_type="resnext50")
    classifier = ClassifierFC.create_model(feat_dim=2048, num_classes=NUM_CLASSES)
    model = nn.DataParallel(model).cuda()
    classifier = nn.DataParallel(classifier).cuda()
    loss_fn = nn.CrossEntropyLoss()

    all_params = []
    for _, val in model.named_parameters():
        if not val.requires_grad:
            continue
        all_params += [
            {"params": [val], "lr": 0.05, "weight_decay": 0.0005, "momentum": 0.9}
        ]
    for _, val in classifier.named_parameters():
        if not val.requires_grad:
            continue
        all_params += [
            {"params": [val], "lr": 0.05, "weight_decay": 0.0005, "momentum": 0.9}
        ]
    optimizer = optim.SGD(all_params)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epoch, eta_min=0.0)

    model.train()
    for epoch in range(num_epoch):
        print("------------ Start Epoch {} -----------".format(epoch))
        total_batch = len(train_loader)
        for step, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            inputs, labels = inputs.cuda(), labels.cuda()
            features = model(inputs)
            predictions = classifier(features)

            # calculate loss
            loss = loss_fn(predictions, labels)
            loss.backward()
            optimizer.step()

            # calculate accuracy
            accuracy = (
                predictions.max(1)[1] == labels
            ).sum().float() / predictions.shape[0]

            if step % PRINT_STEPS == 0:
                print(
                    f"epoch:{epoch:3}, step/total:{step:4}/{total_batch}, Acc:{accuracy.item():>6.4}, Loss:{loss.sum().item():>6.4}"
                )

        # checkpoint
        if epoch % SAVE_EPOCH == 0 or epoch == num_epoch - 1:
            output = {
                "model": model.state_dict(),
                "classifier": classifier.state_dict(),
                "epoch": epoch,
            }
            if not os.path.exists(OUTPUTS_DIR):
                os.makedirs(OUTPUTS_DIR)
            torch.save(output, os.path.join(OUTPUTS_DIR, f"epoch-{epoch}.pth"))

        # update scheduler
        scheduler.step()
    return model, classifier


def data_info(data_path, anno_path, model_path=None):
    cat_attr_inst = dict()
    train_set = Places365TrainSet(data_path, anno_path)
    print("model_path:", model_path)
    if model_path is None:
        model, _ = train_model(train_set)
    else:
        model, _ = load_model(model_path)

    model.eval()
    with torch.no_grad():
        processing_bar = tqdm(train_set.cat_inst.keys())
        for cat in processing_bar:
            cat_attr_inst[cat] = dict()
            sub_set = Places365ExtractSet(category=cat, all_set=train_set)
            all_feat = []
            for inputs, _ in sub_set.get_loader():
                inputs = inputs.cuda()
                features = model(inputs)
                all_feat.append(features.detach().clone().cpu())
            all_feat = torch.cat(all_feat, dim=0).tolist()
            processing_bar.set_description(
                f"Building CoarseLeadingForest (label:{cat}, label_size:{len(all_feat)})"
            )

            clf = CoarseLeadingForest(samples=all_feat, min_dist_multiple=0.6, max_dist_multiple=1.5)
            paths, _ = clf.generate_path(detailed=True)
            paths_flatten = [reduce(add, path) for path in paths]
            for i, path_flatten in enumerate(paths_flatten):
                insts_tmp = sub_set.order_inst(path_flatten)
                cat_attr_inst[cat][i] = insts_tmp
                if len(path_flatten) <= 3:
                    train_set.cat_strange[cat].extend(insts_tmp)
            clf_path = os.path.join(OUTPUTS_DIR, "clf")
            if not os.path.exists(clf_path):
                os.makedirs(clf_path)
            with open(os.path.join(clf_path, f"clf-{cat}.pkl"), "wb") as file:
                pickle.dump(clf, file)

    return (
        cat_attr_inst,
        train_set.cat_inst,
        train_set.inst_cat,
        train_set.inst_path,
        train_set.cat_strange
    )


# x, y, z, l = data_info(
#     "~/Datasets/places365/data_256", "~/Datasets/places365/categories_places365_merge.txt"
# )
