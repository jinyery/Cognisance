######################################
#         Jinye Yang
######################################

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F

import utils.general_utils as utils
from data.dataloader import get_loader
from utils.checkpoint_utils import Checkpoint
from utils.training_utils import *
from utils.test_loader import test_loader

from tqdm import tqdm
from operator import add
from functools import reduce
from utils.clusting_utils import CoarseLeadingForest


class train_multi_center_dual:
    def __init__(self, args, config, logger, eval=False):
        # ============================================================================
        # create model
        logger.info(
            "=====> Model construction from: " + str(config["networks"]["type"])
        )
        model_type = config["networks"]["type"]
        model_file = config["networks"][model_type]["def_file"]
        model_args = config["networks"][model_type]["params"]
        logger.info(
            "=====> Classifier construction from: " + str(config["classifiers"]["type"])
        )
        classifier_type = config["classifiers"]["type"]
        classifier_file = config["classifiers"][classifier_type]["def_file"]
        classifier_args = config["classifiers"][classifier_type]["params"]
        model = utils.source_import(model_file).create_model(**model_args)
        classifier = utils.source_import(classifier_file).create_model(
            **classifier_args
        )

        model = nn.DataParallel(model).cuda()
        classifier = nn.DataParallel(classifier).cuda()

        # other initialization
        self.algorithm_opt = config["algorithm_opt"]
        self.args = args
        self.config = config
        self.logger = logger
        self.model = model
        self.classifier = classifier
        self.optimizer = create_optimizer(model, classifier, logger, config)
        self.scheduler = create_scheduler(self.optimizer, logger, config)
        self.eval = eval
        self.training_opt = config["training_opt"]
        self.multi_type = self.algorithm_opt["multi_type"]

        self.checkpoint = Checkpoint(config)

        # get dataloader
        self.logger.info("=====> Get train dataloader")
        self.train_loader = get_loader(
            config, "train", config["dataset"]["testset"], logger
        )

        # get loss
        self.loss_fc = create_loss(logger, config, self.train_loader)
        if "cos_loss" in self.algorithm_opt and self.algorithm_opt["cos_loss"]:
            self.metric = "cosine"
            self.loss_center = MultiCenterCosLoss(
                num_classes=classifier_args["num_classes"],
                feat_dim=classifier_args["feat_dim"],
            )
        else:
            self.metric = "euclidean"
            self.loss_center = MultiCenterLoss(
                num_classes=classifier_args["num_classes"],
                feat_dim=classifier_args["feat_dim"],
            )
        self.center_optimizer = torch.optim.SGD(self.loss_center.parameters(), lr=0.5)

        # set eval
        if self.eval:
            test_func = test_loader(config)
            self.testing = test_func(config, logger, model, classifier, val=True)

        self.plain = False
        if "plain" in self.algorithm_opt and self.algorithm_opt["plain"]:
            self.plain = True

    def get_center_weight(self, epoch):
        center_weight = self.algorithm_opt["center_weights"][0]
        for i, ms in enumerate(self.algorithm_opt["center_milestones"]):
            if epoch >= ms:
                center_weight = self.algorithm_opt["center_weights"][i]
        self.logger.info("Center Weight: {}".format(center_weight))
        return center_weight

    def run(self):
        # Start Training
        self.logger.info("=====> Start Center Loss with Dual Env Training")

        # preprocess for each epoch
        env1_loader, env2_loader = self.train_loader
        assert len(env1_loader) == len(env2_loader)
        total_batch = len(env1_loader)
        total_image = len(env1_loader.dataset)

        # run epoch
        num_epoch = self.training_opt["num_epochs"]
        for epoch in range(num_epoch):
            self.logger.info("------------ Start Epoch {} -----------".format(epoch))
            self.logger.info(
                "--------------- Environment Type {} -----------".format(
                    self.algorithm_opt["env_type"]
                )
            )
            # saving training info for environments building
            all_ind = []
            all_lab = []
            all_prb = []
            all_feat = []

            center_weight = self.get_center_weight(epoch)

            for step, (
                (inputs1, labels1, _, indexs1),
                (inputs2, labels2, _, indexs2),
            ) in enumerate(zip(env1_loader, env2_loader)):
                iter_info_print = {}

                self.optimizer.zero_grad()

                # additional inputs
                inputs = torch.cat([inputs1, inputs2], dim=0).cuda()
                labels = torch.cat([labels1, labels2], dim=0).cuda()
                indexs = torch.cat([indexs1, indexs2], dim=0).cuda()
                add_inputs = {}

                features = self.model(inputs)
                predictions = self.classifier(features, add_inputs)

                # calculate loss
                loss_ce = self.loss_fc(predictions, labels)
                iter_info_print[self.training_opt["loss"]] = loss_ce.sum().item()

                # center loss
                self.center_optimizer.zero_grad()
                loss_ct = (
                    self.loss_center(
                        features, labels, self.get_label_center(labels, indexs)
                    )
                    * center_weight
                )
                iter_info_print["center_loss"] = loss_ct.sum().item()

                # backward
                loss = loss_ce + loss_ct
                loss.backward()
                self.optimizer.step()
                # multiple (1./alpha) in order to remove the effect of alpha on updating centers
                for param in self.loss_center.parameters():
                    param.grad.data *= 1.0 / (center_weight + 1e-12)
                self.center_optimizer.step()

                # calculate accuracy
                accuracy = (
                    predictions.max(1)[1] == labels
                ).sum().float() / predictions.shape[0]

                # save info for environment spliting
                predictions = predictions.softmax(-1)
                gt_score = torch.gather(
                    predictions, 1, torch.unsqueeze(labels, 1)
                ).view(-1)
                all_ind.append(indexs.detach().clone().cpu())
                all_lab.append(labels.detach().clone().cpu())
                all_prb.append(gt_score.detach().clone().cpu())
                all_feat.append(features.detach().clone().cpu())

                # log information
                iter_info_print.update(
                    {
                        "Accuracy": accuracy.item(),
                        "Loss": loss.sum().item(),
                        "Poke LR": float(self.optimizer.param_groups[0]["lr"]),
                    }
                )
                self.logger.info_iter(
                    epoch,
                    step,
                    total_batch,
                    iter_info_print,
                    self.config["logger_opt"]["print_iter"],
                )

                first_batch = (epoch == 0) and (step == 0)
                if (
                    first_batch
                    or self.config["logger_opt"]["print_grad"]
                    and step % 1000 == 0
                ):
                    utils.print_grad(self.classifier.named_parameters())
                    utils.print_grad(self.model.named_parameters())

            # evaluation on validation set
            if self.eval:
                val_acc = self.testing.run_val(epoch)
            else:
                val_acc = 0.0

            # save env score
            env_score_memo = {}

            if self.algorithm_opt["always_update"] or (
                epoch in self.algorithm_opt["update_milestones"]
            ):
                # update env mask
                self.all_ind = torch.cat(all_ind, dim=0)
                self.all_lab = torch.cat(all_lab, dim=0)
                self.all_prb = torch.cat(all_prb, dim=0)
                self.all_feat = torch.cat(all_feat, dim=0)

                # save env_score
                env_score_memo["label_{}".format(epoch)] = self.all_lab.tolist()
                env_score_memo["prob_{}".format(epoch)] = self.all_prb.tolist()
                env_score_memo["idx_{}".format(epoch)] = self.all_ind.tolist()

                if self.algorithm_opt["env_type"] == "clf":
                    self.update_env_by_clf(env1_loader, env2_loader, total_image)
                    if self.multi_type == 1 or (
                        self.multi_type == 2
                        and epoch == self.algorithm_opt["update_milestones"][-1]
                    ):
                        self.update_center_loss()
                else:
                    raise ValueError("Wrong Env Type")

            # checkpoint
            self.checkpoint.save(
                self.model,
                self.classifier,
                epoch,
                self.logger,
                acc=val_acc,
                add_dict=env_score_memo,
            )

            # update scheduler
            self.scheduler.step()

        # save best model path
        self.checkpoint.save_best_model_path(self.logger)

    def update_env_by_clf(self, env1_loader, env2_loader, total_image):
        # seperate environments by clf
        all_ind, all_lab, all_feat = (
            self.all_ind.tolist(),
            self.all_lab.tolist(),
            self.all_feat.tolist(),
        )
        all_cat = list(set(all_lab))
        all_cat.sort()
        cat_feat = {cat: {} for cat in all_cat}
        for ind, lab, feat in zip(all_ind, all_lab, all_feat):
            cat_feat[lab][ind] = feat

        # baseline distribution
        env1_score = torch.zeros(total_image).fill_(1.0)
        env2_score = torch.zeros(total_image).fill_(1.0)
        # inverse distribution
        clf_weight = self.generate_clf_weight(
            cat_feat, total_image, tg_scale=self.algorithm_opt["sample_scale"]
        )
        env2_score = env2_score * clf_weight

        env1_loader.sampler.set_parameter(env1_score)
        env2_loader.sampler.set_parameter(env2_score)

    def generate_clf_weight(self, cat_feat, total_image, tg_scale=4.0):
        self.cat_ind: dict[any, torch.LongTensor] = dict()
        self.cat_clf: dict[any, CoarseLeadingForest] = dict()
        # normalize
        clf_weight = torch.zeros(total_image).fill_(0.0)
        processing_bar = tqdm(cat_feat.items())
        for cat, cat_items in processing_bar:
            cat_size = len(cat_items)
            if cat_size < 5:
                for ind in list(cat_items.keys()):
                    clf_weight[ind] = 1.0 / max(cat_size, 1.0)
                continue
            processing_bar.set_description(
                f"Building CoarseLeadingForest (label:{cat}, label_size:{cat_size})"
            )

            ind = torch.LongTensor(list(cat_items.keys()))
            clf = CoarseLeadingForest(list(cat_items.values()), metric=self.metric)
            self.cat_ind[cat] = ind
            self.cat_clf[cat] = clf
            weights = torch.ones(len(ind))
            paths, repetitions = clf.generate_path(detailed=True)
            repetitions = torch.Tensor(repetitions)
            weights *= repetitions
            weights /= len(paths)
            paths_flatten = [reduce(add, path) for path in paths]
            for i, path in enumerate(paths):
                if len(path) == 1:
                    self.logger.info(
                        f"Warning:{ind[paths_flatten[i]]} maybe the noise!"
                    )
                weights[paths_flatten[i]] /= len(path)
                for nodes in path:
                    weights[nodes] /= len(nodes)
            weights /= repetitions

            if not self.plain:
                # use Pareto principle to determine the scale parameter
                weights += 1e-5
                head_mean = (
                    torch.topk(weights, k=int(cat_size * 0.8), largest=False)[0]
                    .mean()
                    .item()
                )
                tail_mean = (
                    torch.topk(weights, k=int(cat_size * 0.2), largest=True)[0]
                    .mean()
                    .item()
                )
                scale = tail_mean / head_mean + 1e-5
                exp_scale = (
                    torch.FloatTensor([tg_scale]).log() / torch.FloatTensor([scale]).log()
                )
                exp_scale = exp_scale.clamp(min=1, max=10)
                weights = weights**exp_scale
            weights = weights + 1e-12
            weights = weights / weights.sum()
            clf_weight[ind] = weights
        return clf_weight

    def update_center_loss(self):
        max_num_centers = 1
        label_center_list = list()
        for cat, clf in self.cat_clf.items():
            num_tree = clf.num_tree()
            if num_tree > max_num_centers:
                max_num_centers = num_tree
            default_centers = list()
            for root_id in clf.root_ids:
                fine_id = clf.coarse_nodes[root_id].agent
                index = self.cat_ind[cat][fine_id]
                feat = self.all_feat[index].tolist()
                default_centers.append(feat)
            label_center_list.append(default_centers)
        self.logger.info("=====> max_num_centers:" + str(max_num_centers))
        self.loss_center.update_center(max_num_centers, label_center_list)
        self.center_optimizer = torch.optim.SGD(self.loss_center.parameters(), lr=0.5)

    def get_label_center(self, labels, indexs):
        if self.loss_center.max_num_centers == 1:
            return None
        label_center = torch.zeros(len(labels), dtype=torch.long, device=labels.device)
        for i, label in enumerate(labels):
            if label.item() not in self.cat_clf:
                label_center[i] = 0
                continue
            clf = self.cat_clf[label.item()]
            ind = self.cat_ind[label.item()]
            idx = torch.where(ind == indexs[i].item())[0]
            tmp_idx = clf.where_is_fine_node(idx.item())
            tmp_idx = clf.where_is_coarse_node(tmp_idx)
            label_center[i] = tmp_idx
        return label_center
