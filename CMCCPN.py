import os.path
from collections import namedtuple
from pathlib import Path

import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F
from time import time

from data.pretrained_model import PretrainedModelsFeature
from model.InverseNetwork import InverseNetworkQuadChannel, inverseNetworkTrainingQuadChannel, \
    InverseNetworkMultiChannel, training_InverseNetwork_CMCCPN
from myutils import make_dir, today_str, rotate_images_CMCCPN
import pandas as pd
import sklearn
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import pandas as pd
import random
import copy

class CMCCPN(nn.Module):
    def __init__(self, n_class=20, gamma=0.5, feature_dim=240, input_feature=100,
                 distance_func=None, use_clf_for_openset=True, backbone=None, pretrained=True, fine_tune=False):
        # TODO implement backbone here

        super(CMCCPN, self).__init__()

        self.ucfo = use_clf_for_openset
        self.num_class = n_class
        self.num_features = feature_dim
        self.gamma = nn.Parameter(torch.Tensor([gamma]))
        self.classifier = nn.Linear(self.num_features*8, self.num_class)
        if distance_func is not None:
            self.distance_func = distance_func
        else:
            self.distance_func = lambda x: torch.sqrt(torch.sum(x ** 2, 1))
            # self.distance_func = lambda x: 0.5 * torch.sum(x ** 2, 1)

        self.backbone = None if backbone is None else PretrainedModelsFeature(backbone,
                                                                              pretrained_weights=pretrained,
                                                                              fine_tune=fine_tune)
        if self.backbone:
            if backbone == "mobilenet_v3_large":
                input_feature = 960
            elif backbone == "resnet50":
                input_feature = 2048
            elif backbone == "vit_b_16":
                input_feature = 768
            else:
                input_feature = 1024
            self.feature = nn.Linear(input_feature, self.num_features)
            self.feature_rotate = nn.Linear(input_feature, self.num_features)
        else:
            self.feature = nn.Linear(input_feature, self.num_features)
            self.feature_rotate = nn.Linear(input_feature, self.num_features)

        self.centers = torch.nn.Parameter(torch.randn(self.num_class, self.num_features*8))
        self.centers.requires_grad = True

    def grad_stage1(self):
        for param in self.classifier.parameters():
            param.requires_grad = False

        for param in self.feature.parameters():
            param.requires_grad = True

        for param in self.feature_rotate.parameters():
            param.requires_grad = True

        self.centers.requires_grad = True
        self.gamma.requires_grad = True

    def grad_stage2(self):
        for param in self.classifier.parameters():
            param.requires_grad = True

        for param in self.feature.parameters():
            param.requires_grad = False

        for param in self.feature_rotate.parameters():
            param.requires_grad = False

        self.centers.requires_grad = False
        self.gamma.requires_grad = False

    def forward(self, x, x_90, x_180, x_270, x_60, x_120, x_hflip, x_vflip, return_fea=False):
        x_all = self.forward_feature(x, x_90, x_180, x_270, x_60, x_120, x_hflip, x_vflip)
        distances = self.distance_centers(x_all) * -1 * self.gamma
        logits = self.classifier(x_all)
        if return_fea:
            return distances, x_all, logits
        return distances, logits


    def split_distance_centers_nobatch(self, x, labels, d):
        center = self.centers[labels]
        center = center.view(d, -1)
        x = self.forward_feature(torch.unsqueeze(x, 0))
        x = x.view(d, -1)
        dis = self.distance_func((x-center))
        return dis

    def forward_feature(self, x, x_90, x_180, x_270, x_60, x_120, x_hflip, x_vflip):
        if self.backbone:
            x = self.backbone(x)
            x_90 = self.backbone(x_90)
            x_180 = self.backbone(x_180)
            x_270 = self.backbone(x_270)
            x_60 = self.backbone(x_60)
            x_120 = self.backbone(x_120)
            x_hflip = self.backbone(x_hflip)
            x_vflip = self.backbone(x_vflip)

        x = F.relu(self.feature(x))
        x_90 = F.relu(self.feature_rotate(x_90))
        x_180 = F.relu(self.feature_rotate(x_180))
        x_270 = F.relu(self.feature_rotate(x_270))
        x_60 = F.relu(self.feature_rotate(x_60))
        x_120 = F.relu(self.feature_rotate(x_120))
        x_hflip = F.relu(self.feature_rotate(x_hflip))
        x_vflip = F.relu(self.feature_rotate(x_vflip))

        x_all = torch.cat((x, x_90, x_180, x_270, x_60, x_120, x_hflip, x_vflip), 1)
        return x_all

    def distance_centers(self, features):
        num_class = self.centers.size(0)
        batch_size = features.size(0)
        expand_features = features.repeat_interleave(num_class, dim=0)
        expand_centers = self.centers.repeat(batch_size, 1)
        x = self.distance_func((expand_features - expand_centers))
        x = x.view(batch_size, num_class)
        return x

    def calc_distance(self, features, others):
        num_others = others.size(0)
        batch_size = features.size(0)
        expand_features = features.repeat_interleave(num_others, dim=0)
        expand_centers = others.repeat(batch_size, 1)
        x = self.distance_func((expand_features - expand_centers))
        x = x.view(batch_size, num_others)
        return x

    def extend_prototypes_logits(self, n_class):
        print("n_class", n_class)
        old_num_class = self.num_class
        self.num_class += n_class

        newcenters = torch.nn.Parameter(torch.randn(self.num_class, self.num_features * 8))

        with torch.no_grad():
            newcenters[:old_num_class] = self.centers

        self.centers = newcenters
        print("new centers shape", self.centers.shape)

        new_linear = nn.Linear(self.num_features*8, self.num_class)
        with torch.no_grad():
            new_linear.bias[:old_num_class] = self.classifier.bias.clone()
            new_linear.weight[:old_num_class] = self.classifier.weight.clone()

        self.classifier = new_linear

class QCCPNLoss(nn.Module):

    def __init__(self, gamma, epsilon_contrast, dis_func=None, device=None, w1=1.0):
        super(QCCPNLoss, self).__init__()
        self.gamma = gamma
        self.epsilon_contrast = epsilon_contrast
        self.w1 = w1
        # TODO check in the paper carefully
        # self.dis_func = lambda x: 0.5 * torch.sum(x ** 2, 1) if dis_func is None else dis_func
        self.dis_func = lambda x: torch.sqrt(torch.sum(x ** 2, 1)) if dis_func is None else dis_func
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def forward(self, features, labels, centers):
        distance = self.distance_centers(features, centers)
        loss_dce = self.cross_entropy(self.gamma * -distance, labels)
        lcontrast = self.distance_contrast_batch(labels, centers)
        l2 = torch.clip(self.epsilon_contrast - lcontrast, min=0)
        l2 = torch.mean(l2)
        return loss_dce + self.w1 * l2

    def distance_centers(self, features, centers):
        num_class = centers.size(0)
        batch_size = features.size(0)

        expand_features = features.repeat_interleave(num_class, dim=0)
        expand_centers = centers.repeat(batch_size, 1)

        x = self.dis_func(expand_features - expand_centers)
        x = x.view(batch_size, num_class)
        return x
    def distance_contrast_batch(self, labels, centers):
        labels_size = labels.size(0)
        centers_size = centers.size(0)

        labels_centers = centers[labels]

        expand_labels = labels_centers.repeat_interleave(centers_size - 1, dim=0)
        expand_centers = centers.repeat(labels_size, 1)

        idx_label = torch.tensor(list(range(labels.size(0)))).to(self.device)
        idx_label = centers_size * idx_label + labels
        idx_select = torch.tensor(list(range(expand_centers.size(0)))).to(self.device)
        mask_idx_select = (idx_select != idx_label[0])

        for i in range(1, idx_label.size(0)):
            mask_idx_select &= (idx_select != idx_label[i])
        mask_idx_select = torch.nonzero(mask_idx_select).squeeze()
        expand_centers = torch.index_select(expand_centers, 0, mask_idx_select)

        x = self.dis_func(expand_labels - expand_centers)
        x = x.view(labels_size, centers_size - 1)
        x = torch.mean(x, 1)
        return x

class ClassIncrementalQCCPNLossOLD(QCCPNLoss):
    def __init__(self, gamma, epsilon_contrast, dis_func=None, device=None, w1=1.0, w2=1.0, uthreshold=0.95):
        super(ClassIncrementalQCCPNLossOLD, self).__init__(gamma, epsilon_contrast, dis_func=dis_func, device=device, w1=w1)
        self.w2 = w2
        # self.uthreshold = torch.tensor(uthreshold).to(device)
        self.uthreshold = uthreshold
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features, labels, centers, ufea):
        distance = self.distance_centers(features, centers)
        loss_dce = self.cross_entropy(self.gamma * -distance, labels)
        lcontrast = self.distance_contrast_batch(labels, centers)
        l2 = torch.clip(self.epsilon_contrast - lcontrast, min=0)
        l2 = torch.mean(l2)
        distance_ufea = self.distance_centers(ufea, centers)
        softmax_ufea = self.softmax(distance_ufea).data
        max_ufea, _ = torch.max(softmax_ufea, dim=1)
        l3 = torch.clip(max_ufea - self.uthreshold, min=0)
        l3 = torch.mean(l3)
        return loss_dce + self.w1 * l2 + self.w2 * l3

class ClassIncrementalQCCPNLoss(QCCPNLoss):
    def __init__(self, gamma, epsilon_contrast, dis_func=None, device=None, w1=1.0, w2=1.0, scale_contrast_unknown=0.5):
        super(ClassIncrementalQCCPNLoss, self).__init__(gamma, epsilon_contrast, dis_func=dis_func, device=device, w1=w1)
        self.w2 = w2
        self.softmax = nn.Softmax(dim=1)
        self.scu = scale_contrast_unknown

    def forward(self, features, labels, centers, ufea):
        distance = self.distance_centers(features, centers)
        loss_dce = self.cross_entropy(self.gamma * -distance, labels)
        lcontrast = self.distance_contrast_batch(labels, centers)
        l2 = torch.clip(self.epsilon_contrast - lcontrast, min=0)
        l2 = torch.mean(l2)
        if ufea is not None:
            distance_ufea = self.distance_centers(ufea, centers)
            l3 = torch.clip(self.epsilon_contrast*self.scu - distance_ufea, min=0)
            l3 = torch.mean(l3)
        else:
            l3 = 0.0
        return loss_dce + self.w1 * l2 + self.w2 * l3

def evalCMCCPN(device, model, task_id, test_task_dl, unknown_dl, verbose, return_threshold=False):
    model.eval()
    total = 0
    correct = 0
    prob_known = torch.Tensor([]).to(device)
    label_known = torch.Tensor([]).to(device)
    prob_unknown = torch.Tensor([]).to(device)
    label_unknown = torch.Tensor([]).to(device)
    predicted_known = torch.Tensor([]).to(device)
    predicted_unknown = torch.Tensor([]).to(device)
    with torch.no_grad():
        acc_t = []
        for t in range(task_id+1):
            total_t = 0
            correct_t = 0
            for fea, fea90, fea180, fea270, fea60, fea120, fea_hflip, fea_vflip, labels, task in test_task_dl[t]:
                fea = fea.to(device)
                fea90 = fea90.to(device)
                fea180 = fea180.to(device)
                fea270 = fea270.to(device)
                fea60 = fea60.to(device)
                fea120 = fea120.to(device)
                fea_hflip = fea_hflip.to(device)
                fea_vflip = fea_vflip.to(device)
                labels = labels.to(device)

                outputs, logits = model(fea, fea90, fea180, fea270, fea60, fea120, fea_hflip, fea_vflip)
                label_known = torch.cat((label_known, torch.ones(labels.size(0)).to(device)), 0)

                prob, predicted = torch.max(nn.functional.softmax(outputs, dim=1).data, 1)

                prob_known = torch.cat((prob_known, prob), 0)
                predicted_known = torch.cat((predicted_known, predicted), 0)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                correct_t += (predicted == labels).sum().item()
                total_t += labels.size(0)


            acc_t.append(correct_t / total_t)

        for fea, fea90, fea180, fea270, fea60, fea120, fea_hflip, fea_vflip, labels, _ in unknown_dl:
            fea = fea.to(device)
            fea90 = fea90.to(device)
            fea180 = fea180.to(device)
            fea270 = fea270.to(device)
            fea60 = fea60.to(device)
            fea120 = fea120.to(device)
            fea_hflip = fea_hflip.to(device)
            fea_vflip = fea_vflip.to(device)
            labels = labels.to(device)
            outputs, logits = model(fea, fea90, fea180, fea270, fea60, fea120, fea_hflip, fea_vflip)

            label_unknown = torch.cat((label_unknown, torch.zeros(labels.size(0)).to(device)), 0)
            prob, predicted = torch.max(nn.functional.softmax(outputs, dim=1).data, 1)
            predicted_unknown = torch.cat((predicted_unknown, predicted), 0)
            prob_unknown = torch.cat((prob_unknown, prob), 0)

        out_pred = torch.cat((prob_known, prob_unknown), 0).detach().cpu().numpy()
        out_label = torch.cat((label_known, label_unknown), 0).detach().cpu().numpy()
        fpr, tpr, threshold = sklearn.metrics.roc_curve(y_true=out_label, y_score=out_pred, pos_label=1)
        roc_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'j-index': tpr - fpr, 'threshold': threshold})
        # roc_df.to_csv('logs/figures/roc_detail_proposed.csv')
        roc_df_maxj = roc_df.sort_values('j-index', ascending=False)
        threshold = roc_df_maxj.iloc[0]['threshold']
        auroc = sklearn.metrics.auc(fpr, tpr)

        predicted_known = predicted_known.where(prob_known >= threshold, torch.tensor(0).to(device))
        predicted_known = predicted_known.where(prob_known < threshold, torch.tensor(1).to(device))
        predicted_unknown = predicted_unknown.where(prob_unknown >= threshold, torch.tensor(0).to(device))
        predicted_unknown = predicted_unknown.where(prob_unknown < threshold, torch.tensor(1).to(device))
        predicted_baccu = torch.cat((predicted_known, predicted_unknown), 0).detach().cpu().numpy()
        baccu = balanced_accuracy_score(out_label, predicted_baccu)

    if return_threshold:
        return 100 * correct / total, auroc, baccu, threshold, acc_t

    return 100 * correct / total, auroc, baccu, acc_t

def evalCMCCPNBackbone(device, model, task_id, test_task_dl, unknown_dl, verbose, return_threshold=False):
    model.eval()
    total = 0
    correct = 0
    prob_known = torch.Tensor([]).to(device)
    label_known = torch.Tensor([]).to(device)
    prob_unknown = torch.Tensor([]).to(device)
    label_unknown = torch.Tensor([]).to(device)
    predicted_known = torch.Tensor([]).to(device)
    predicted_unknown = torch.Tensor([]).to(device)
    with torch.no_grad():
        acc_t = []
        for t in range(task_id+1):
            total_t = 0
            correct_t = 0
            for img, labels, task in test_task_dl[t]:
                img = img.to(device)
                labels = labels.to(device)

                img90, img180, img270, img60, img120, img_hflip, img_vflip = rotate_images_CMCCPN(img, device=device)
                outputs, logits = model(img, img90, img180, img270, img60, img120, img_hflip, img_vflip)
                label_known = torch.cat((label_known, torch.ones(labels.size(0)).to(device)), 0)

                prob, predicted = torch.max(nn.functional.softmax(outputs, dim=1).data, 1)

                prob_known = torch.cat((prob_known, prob), 0)
                predicted_known = torch.cat((predicted_known, predicted), 0)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                correct_t += (predicted == labels).sum().item()
                total_t += labels.size(0)


            acc_t.append(correct_t / total_t)

        for img, labels, _ in unknown_dl:
            img = img.to(device)
            labels = labels.to(device)

            img90, img180, img270, img60, img120, img_hflip, img_vflip = rotate_images_CMCCPN(img, device=device)
            outputs, logits = model(img, img90, img180, img270, img60, img120, img_hflip, img_vflip)

            label_unknown = torch.cat((label_unknown, torch.zeros(labels.size(0)).to(device)), 0)
            prob, predicted = torch.max(nn.functional.softmax(outputs, dim=1).data, 1)
            predicted_unknown = torch.cat((predicted_unknown, predicted), 0)
            prob_unknown = torch.cat((prob_unknown, prob), 0)

        out_pred = torch.cat((prob_known, prob_unknown), 0).detach().cpu().numpy()
        out_label = torch.cat((label_known, label_unknown), 0).detach().cpu().numpy()
        fpr, tpr, threshold = sklearn.metrics.roc_curve(y_true=out_label, y_score=out_pred, pos_label=1)
        roc_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'j-index': tpr - fpr, 'threshold': threshold})
        # roc_df.to_csv('logs/figures/roc_detail_proposed.csv')
        roc_df_maxj = roc_df.sort_values('j-index', ascending=False)
        threshold = roc_df_maxj.iloc[0]['threshold']
        auroc = sklearn.metrics.auc(fpr, tpr)

        predicted_known = predicted_known.where(prob_known >= threshold, torch.tensor(0).to(device))
        predicted_known = predicted_known.where(prob_known < threshold, torch.tensor(1).to(device))
        predicted_unknown = predicted_unknown.where(prob_unknown >= threshold, torch.tensor(0).to(device))
        predicted_unknown = predicted_unknown.where(prob_unknown < threshold, torch.tensor(1).to(device))
        predicted_baccu = torch.cat((predicted_known, predicted_unknown), 0).detach().cpu().numpy()
        baccu = balanced_accuracy_score(out_label, predicted_baccu)

    if return_threshold:
        return 100 * correct / total, auroc, baccu, threshold, acc_t

    return 100 * correct / total, auroc, baccu, acc_t

def eval_classifier_CMCCPN(device, model, task_id, test_task_dl, unknown_dl=None):
    model.eval()
    total = 0
    correct = 0

    if unknown_dl:
        prob_known = torch.Tensor([]).to(device)
        label_known = torch.Tensor([]).to(device)
        prob_unknown = torch.Tensor([]).to(device)
        label_unknown = torch.Tensor([]).to(device)
        predicted_known = torch.Tensor([]).to(device)
        predicted_unknown = torch.Tensor([]).to(device)

    with torch.no_grad():
        acc_t = []
        for t in range(task_id + 1):
            total_t = 0
            correct_t = 0
            for fea, fea90, fea180, fea270, fea60, fea120, fea_hflip, fea_vflip, labels, task in test_task_dl[t]:
                fea = fea.to(device)
                fea90 = fea90.to(device)
                fea180 = fea180.to(device)
                fea270 = fea270.to(device)
                fea60 = fea60.to(device)
                fea120 = fea120.to(device)
                fea_hflip = fea_hflip.to(device)
                fea_vflip = fea_vflip.to(device)
                labels = labels.to(device)

                outputs, logits = model(fea, fea90, fea180, fea270, fea60, fea120, fea_hflip, fea_vflip)
                prob, predicted = torch.max(nn.functional.softmax(logits, dim=1).data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                correct_t += (predicted == labels).sum().item()
                total_t += labels.size(0)

                if unknown_dl:
                    label_known = torch.cat((label_known, torch.ones(labels.size(0)).to(device)), 0)
                    prob_known = torch.cat((prob_known, prob), 0)
                    predicted_known = torch.cat((predicted_known, predicted), 0)


            acc_t.append(correct_t/total_t)


        if unknown_dl:
            for fea, fea90, fea180, fea270, fea60, fea120, fea_hflip, fea_vflip, labels, _ in unknown_dl:
                fea = fea.to(device)
                fea90 = fea90.to(device)
                fea180 = fea180.to(device)
                fea270 = fea270.to(device)
                fea60 = fea60.to(device)
                fea120 = fea120.to(device)
                fea_hflip = fea_hflip.to(device)
                fea_vflip = fea_vflip.to(device)
                labels = labels.to(device)
                outputs, logits = model(fea, fea90, fea180, fea270, fea60, fea120, fea_hflip, fea_vflip)
                prob, predicted = torch.max(nn.functional.softmax(logits, dim=1).data, 1)

                label_unknown = torch.cat((label_unknown, torch.zeros(labels.size(0)).to(device)), 0)
                predicted_unknown = torch.cat((predicted_unknown, predicted), 0)
                prob_unknown = torch.cat((prob_unknown, prob), 0)

            out_pred = torch.cat((prob_known, prob_unknown), 0).detach().cpu().numpy()
            out_label = torch.cat((label_known, label_unknown), 0).detach().cpu().numpy()
            fpr, tpr, threshold = sklearn.metrics.roc_curve(y_true=out_label, y_score=out_pred, pos_label=1)
            roc_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'j-index': tpr - fpr, 'threshold': threshold})
            # roc_df.to_csv('logs/figures/roc_detail_proposed.csv')
            roc_df_maxj = roc_df.sort_values('j-index', ascending=False)
            threshold = roc_df_maxj.iloc[0]['threshold']
            auroc = sklearn.metrics.auc(fpr, tpr)

            predicted_known = predicted_known.where(prob_known >= threshold, torch.tensor(0).to(device))
            predicted_known = predicted_known.where(prob_known < threshold, torch.tensor(1).to(device))
            predicted_unknown = predicted_unknown.where(prob_unknown >= threshold, torch.tensor(0).to(device))
            predicted_unknown = predicted_unknown.where(prob_unknown < threshold, torch.tensor(1).to(device))
            predicted_baccu = torch.cat((predicted_known, predicted_unknown), 0).detach().cpu().numpy()
            baccu = balanced_accuracy_score(out_label, predicted_baccu)

            return 100 * correct / total, acc_t, auroc, baccu

    return 100 * correct / total, acc_t


def eval_classifier_CMCCPN_backbone(device, model, task_id, test_task_dl, unknown_dl=None):
    model.eval()
    total = 0
    correct = 0

    if unknown_dl:
        prob_known = torch.Tensor([]).to(device)
        label_known = torch.Tensor([]).to(device)
        prob_unknown = torch.Tensor([]).to(device)
        label_unknown = torch.Tensor([]).to(device)
        predicted_known = torch.Tensor([]).to(device)
        predicted_unknown = torch.Tensor([]).to(device)

    with torch.no_grad():
        acc_t = []
        for t in range(task_id + 1):
            total_t = 0
            correct_t = 0
            for img, labels, task in test_task_dl[t]:
                img = img.to(device)
                labels = labels.to(device)

                img90, img180, img270, img60, img120, img_hflip, img_vflip = rotate_images_CMCCPN(img, device=device)

                outputs, logits = model(img, img90, img180, img270, img60, img120, img_hflip, img_vflip)
                prob, predicted = torch.max(nn.functional.softmax(logits, dim=1).data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                correct_t += (predicted == labels).sum().item()
                total_t += labels.size(0)

                if unknown_dl:
                    label_known = torch.cat((label_known, torch.ones(labels.size(0)).to(device)), 0)
                    prob_known = torch.cat((prob_known, prob), 0)
                    predicted_known = torch.cat((predicted_known, predicted), 0)


            acc_t.append(correct_t/total_t)


        if unknown_dl:
            for img, labels, _ in unknown_dl:
                img = img.to(device)
                labels = labels.to(device)

                img90, img180, img270, img60, img120, img_hflip, img_vflip = rotate_images_CMCCPN(img, device=device)
                outputs, logits = model(img, img90, img180, img270, img60, img120, img_hflip, img_vflip)
                prob, predicted = torch.max(nn.functional.softmax(logits, dim=1).data, 1)

                label_unknown = torch.cat((label_unknown, torch.zeros(labels.size(0)).to(device)), 0)
                predicted_unknown = torch.cat((predicted_unknown, predicted), 0)
                prob_unknown = torch.cat((prob_unknown, prob), 0)

            out_pred = torch.cat((prob_known, prob_unknown), 0).detach().cpu().numpy()
            out_label = torch.cat((label_known, label_unknown), 0).detach().cpu().numpy()
            fpr, tpr, threshold = sklearn.metrics.roc_curve(y_true=out_label, y_score=out_pred, pos_label=1)
            roc_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'j-index': tpr - fpr, 'threshold': threshold})
            # roc_df.to_csv('logs/figures/roc_detail_proposed.csv')
            roc_df_maxj = roc_df.sort_values('j-index', ascending=False)
            threshold = roc_df_maxj.iloc[0]['threshold']
            auroc = sklearn.metrics.auc(fpr, tpr)

            predicted_known = predicted_known.where(prob_known >= threshold, torch.tensor(0).to(device))
            predicted_known = predicted_known.where(prob_known < threshold, torch.tensor(1).to(device))
            predicted_unknown = predicted_unknown.where(prob_unknown >= threshold, torch.tensor(0).to(device))
            predicted_unknown = predicted_unknown.where(prob_unknown < threshold, torch.tensor(1).to(device))
            predicted_baccu = torch.cat((predicted_known, predicted_unknown), 0).detach().cpu().numpy()
            baccu = balanced_accuracy_score(out_label, predicted_baccu)

            return 100 * correct / total, acc_t, auroc, baccu

    return 100 * correct / total, acc_t


def eval_openset_classifier_CMCCPN(device, model, task_id, test_task_dl, unknown_dl=None):
    model.eval()
    total = 0
    correct = 0

    if unknown_dl:
        prob_known = torch.Tensor([]).to(device)
        label_known = torch.Tensor([]).to(device)
        prob_unknown = torch.Tensor([]).to(device)
        label_unknown = torch.Tensor([]).to(device)
        predicted_known = torch.Tensor([]).to(device)
        predicted_unknown = torch.Tensor([]).to(device)

    with torch.no_grad():
        acc_t = []
        for t in range(task_id + 1):
            total_t = 0
            correct_t = 0
            for fea, fea90, fea180, fea270, fea60, fea120, fea_hflip, fea_vflip, labels, task in test_task_dl[t]:
                fea = fea.to(device)
                fea90 = fea90.to(device)
                fea180 = fea180.to(device)
                fea270 = fea270.to(device)
                fea60 = fea60.to(device)
                fea120 = fea120.to(device)
                fea_hflip = fea_hflip.to(device)
                fea_vflip = fea_vflip.to(device)
                labels = labels.to(device)


                outputs, logits = model(fea, fea90, fea180, fea270, fea60, fea120, fea_hflip, fea_vflip)
                prob, predicted = torch.max(nn.functional.softmax(logits, dim=1).data, 1)

                if unknown_dl:
                    label_known = torch.cat((label_known, torch.ones(labels.size(0)).to(device)), 0)
                    prob_known = torch.cat((prob_known, prob), 0)
                    predicted_known = torch.cat((predicted_known, predicted), 0)

                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                correct_t += (predicted == labels).sum().item()
                total_t += labels.size(0)

            acc_t.append(correct_t/total_t)

        if unknown_dl:
            for fea, fea90, fea180, fea270, fea60, fea120, fea_hflip, fea_vflip, labels, _ in unknown_dl:
                fea = fea.to(device)
                fea90 = fea90.to(device)
                fea180 = fea180.to(device)
                fea270 = fea270.to(device)
                fea60 = fea60.to(device)
                fea120 = fea120.to(device)
                fea_hflip = fea_hflip.to(device)
                fea_vflip = fea_vflip.to(device)
                labels = labels.to(device)
                outputs, logits = model(fea, fea90, fea180, fea270, fea60, fea120, fea_hflip, fea_vflip)
                prob, predicted = torch.max(nn.functional.softmax(logits, dim=1).data, 1)

                label_unknown = torch.cat((label_unknown, torch.zeros(labels.size(0)).to(device)), 0)
                predicted_unknown = torch.cat((predicted_unknown, predicted), 0)
                prob_unknown = torch.cat((prob_unknown, prob), 0)

            out_pred = torch.cat((prob_known, prob_unknown), 0).detach().cpu().numpy()
            out_label = torch.cat((label_known, label_unknown), 0).detach().cpu().numpy()
            fpr, tpr, threshold = sklearn.metrics.roc_curve(y_true=out_label, y_score=out_pred, pos_label=1)
            roc_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'j-index': tpr - fpr, 'threshold': threshold})
            # roc_df.to_csv('logs/figures/roc_detail_proposed.csv')
            roc_df_maxj = roc_df.sort_values('j-index', ascending=False)
            threshold = roc_df_maxj.iloc[0]['threshold']
            auroc = sklearn.metrics.auc(fpr, tpr)
            # print("threshold", threshold)
            # print("auroc", auroc)
            # print("prob known", prob_known)
            # print("prob unknown", prob_unknown)

            predicted_known = predicted_known.where(prob_known >= threshold, torch.tensor(0).to(device))
            predicted_known = predicted_known.where(prob_known < threshold, torch.tensor(1).to(device))
            predicted_unknown = predicted_unknown.where(prob_unknown >= threshold, torch.tensor(0).to(device))
            predicted_unknown = predicted_unknown.where(prob_unknown < threshold, torch.tensor(1).to(device))
            predicted_baccu = torch.cat((predicted_known, predicted_unknown), 0).detach().cpu().numpy()
            baccu = balanced_accuracy_score(out_label, predicted_baccu)

            return 100 * correct / total, acc_t, auroc, baccu

    return 100 * correct / total, acc_t

def eval_openset_classifier_CMCCPN_backbone(device, model, task_id, test_task_dl, unknown_dl=None):
    model.eval()
    total = 0
    correct = 0

    if unknown_dl:
        prob_known = torch.Tensor([]).to(device)
        label_known = torch.Tensor([]).to(device)
        prob_unknown = torch.Tensor([]).to(device)
        label_unknown = torch.Tensor([]).to(device)
        predicted_known = torch.Tensor([]).to(device)
        predicted_unknown = torch.Tensor([]).to(device)

    with torch.no_grad():
        acc_t = []
        for t in range(task_id + 1):
            total_t = 0
            correct_t = 0
            for img, labels, task in test_task_dl[t]:
                img = img.to(device)
                labels = labels.to(device)

                img90, img180, img270, img60, img120, img_hflip, img_vflip = rotate_images_CMCCPN(img, device=device)

                outputs, logits = model(img, img90, img180, img270, img60, img120, img_hflip, img_vflip)
                prob, predicted = torch.max(nn.functional.softmax(logits, dim=1).data, 1)

                if unknown_dl:
                    label_known = torch.cat((label_known, torch.ones(labels.size(0)).to(device)), 0)
                    prob_known = torch.cat((prob_known, prob), 0)
                    predicted_known = torch.cat((predicted_known, predicted), 0)

                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                correct_t += (predicted == labels).sum().item()
                total_t += labels.size(0)

            acc_t.append(correct_t/total_t)

        if unknown_dl:
            for img, labels, _ in unknown_dl:
                img = img.to(device)
                labels = labels.to(device)

                img90, img180, img270, img60, img120, img_hflip, img_vflip = rotate_images_CMCCPN(img, device=device)
                outputs, logits = model(img, img90, img180, img270, img60, img120, img_hflip, img_vflip)

                prob, predicted = torch.max(nn.functional.softmax(logits, dim=1).data, 1)

                label_unknown = torch.cat((label_unknown, torch.zeros(labels.size(0)).to(device)), 0)
                predicted_unknown = torch.cat((predicted_unknown, predicted), 0)
                prob_unknown = torch.cat((prob_unknown, prob), 0)

            out_pred = torch.cat((prob_known, prob_unknown), 0).detach().cpu().numpy()
            out_label = torch.cat((label_known, label_unknown), 0).detach().cpu().numpy()
            fpr, tpr, threshold = sklearn.metrics.roc_curve(y_true=out_label, y_score=out_pred, pos_label=1)
            roc_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'j-index': tpr - fpr, 'threshold': threshold})
            # roc_df.to_csv('logs/figures/roc_detail_proposed.csv')
            roc_df_maxj = roc_df.sort_values('j-index', ascending=False)
            threshold = roc_df_maxj.iloc[0]['threshold']
            auroc = sklearn.metrics.auc(fpr, tpr)
            # print("threshold", threshold)
            # print("auroc", auroc)
            # print("prob known", prob_known)
            # print("prob unknown", prob_unknown)

            predicted_known = predicted_known.where(prob_known >= threshold, torch.tensor(0).to(device))
            predicted_known = predicted_known.where(prob_known < threshold, torch.tensor(1).to(device))
            predicted_unknown = predicted_unknown.where(prob_unknown >= threshold, torch.tensor(0).to(device))
            predicted_unknown = predicted_unknown.where(prob_unknown < threshold, torch.tensor(1).to(device))
            predicted_baccu = torch.cat((predicted_known, predicted_unknown), 0).detach().cpu().numpy()
            baccu = balanced_accuracy_score(out_label, predicted_baccu)

            return 100 * correct / total, acc_t, auroc, baccu

    return 100 * correct / total, acc_t

def training_CMCCPN_noncontinual(device, model, save_model_path, log_path, train_task_dl, test_task_dl, unknown_dl, feature_memory=None, load_path=None, learning_rate=0.001, start_epoch=0, n_epoch=50, batch_size=32,
                                 gamma=0.1, contrastive=100, trial=5, initial_seed=1, return_model=False, save_last_model=False, record_epoch=True, n_epoch_classifier=30,
                                 dataset_name="", backbone_name="", idx_combin=""):
    if load_path is not None:
        model.load_state_dict(torch.load(load_path))

    make_dir(save_model_path)

    start_time = time()
    model.train()
    print("DEVICE:", device)
    model.to(device)
    n_epoch += start_epoch
    n_task = len(train_task_dl)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # default gamma=0.1, contrastive=1000
    # qccpn_loss = QCCPNLoss(gamma, contrastive, device=device)
    qccpn_loss = QCCPNLoss(gamma, contrastive, device=device)
    acc_pertask = []
    auroc_pertask = []
    baccu_pertask = []
    task_desc_list = []
    task_id_list = []
    current_seed = initial_seed

    lp_proto_list = []


    data_epoch = {
        "seed": [],
        "task_id": [],
        "epoch": [],
        "accuracy": [],
        "auroc": [],
        "baccu": [],
        "acc_baccu": []
    }


    data_epoch_clf = {
        "seed": [],
        "task_id": [],
        "epoch": [],
        "accuracy": [],
        "auroc": [],
        "baccu": []
    }

    dir_lp_proto = f"{save_model_path}/OSR/CMCCPN/{dataset_name}/{backbone_name}/C_{contrastive}/idx_{idx_combin}/"
    make_dir(dir_lp_proto)

    for seed_trial in range(trial):
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)
        random.seed(current_seed)
        seed = current_seed

        for task_id in range(n_task):

            total_step = len(train_task_dl[task_id])
            if task_id > 0:
                model.extend_prototypes(n_class_per_task)
                model.to(device)
                if feature_memory:
                    feature_memory.adjust(list(range(0, end_class)), batch_size)

            model.train()
            # Training stage 1
            model.grad_stage1()
            for epoch in range(start_epoch, n_epoch):
                total_loss = 0
                loss_step = 0
                total = 0
                correct = 0
                for i, (fea, fea90, fea180, fea270, fea60, fea120, fea_hflip, fea_vflip, labels, task) in enumerate(
                        train_task_dl[task_id]):
                    fea = fea.to(device)
                    fea90 = fea90.to(device)
                    fea180 = fea180.to(device)
                    fea270 = fea270.to(device)
                    fea60 = fea60.to(device)
                    fea120 = fea120.to(device)
                    fea_vflip = fea_vflip.to(device)
                    fea_hflip = fea_hflip.to(device)
                    labels = labels.to(device)

                    if task_id > 0 and feature_memory:
                        # TODO check for CMCCPN
                        lat_fea_mem, lat90_fea_mem, lat180_fea_mem, lat270_fea_mem, lbl_fea_mem, task_mem = feature_memory.take()
                        fea = torch.cat((fea, lat_fea_mem.to(device)), 0)
                        fea90 = torch.cat((fea90, lat90_fea_mem.to(device)), 0)
                        fea180 = torch.cat((fea180, lat180_fea_mem.to(device)), 0)
                        fea270 = torch.cat((fea270, lat270_fea_mem.to(device)), 0)
                        labels = torch.cat((labels, lbl_fea_mem.to(device)), 0)

                    output, features, logits = model(fea, fea90, fea180, fea270, fea60, fea120, fea_hflip, fea_vflip,
                                                     return_fea=True)

                    total += labels.size(0)
                    prob, predicted = torch.max(nn.functional.softmax(output, dim=1).data, 1)

                    correct += (predicted == labels).sum().item()
                    loss = qccpn_loss(features, labels, model.centers)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    loss_step += 1

                # Eval and record the results for each epoch
                test_avg_acc, auroc, baccu, acc_t = evalCMCCPN(device, model, task_id, test_task_dl,
                                                                       unknown_dl,
                                                                       verbose=1)
                data_epoch['seed'].append(seed_trial)
                data_epoch['task_id'].append(task_id)
                data_epoch['epoch'].append(epoch + 1)
                data_epoch['accuracy'].append(test_avg_acc)
                data_epoch['auroc'].append(auroc)
                data_epoch['baccu'].append(baccu)
                data_epoch['acc_baccu'].append((test_avg_acc / 100 + baccu) / 2)

                SAVE_PATH = f"epoch_{epoch}_acc_{test_avg_acc:.3f}_auroc_{auroc:.3f}_{today_str()}.pth"
                SAVE_PATH = os.path.join(dir_lp_proto, SAVE_PATH)
                lp_proto_list.append(SAVE_PATH)
                torch.save(model.state_dict(), SAVE_PATH)

                print(
                    f"TASK: {task_id} - {task[0]} | Epoch [{epoch + 1}/{n_epoch}]: Loss: {total_loss / loss_step}, Accuracy Train: {100 * correct / total}, Accuracy Test: {test_avg_acc}, AUROC: {auroc}, BACCU: {baccu}, ACC_BACCU: {(test_avg_acc / 100 + baccu) / 2}")


            max_index = data_epoch['acc_baccu'].index(max(data_epoch['acc_baccu']))
            auroc_dis = data_epoch['auroc'][max_index]
            baccu_dis = data_epoch['baccu'][max_index]
            LOAD_PATH = lp_proto_list[max_index]

            print(f"Loading the best auroc at index {max_index}")
            model.load_state_dict(torch.load(LOAD_PATH))

            model.grad_stage2()
            for epoch in range(n_epoch_classifier):
                total_loss = 0
                loss_step = 0
                total = 0
                correct = 0
                for i, (fea, fea90, fea180, fea270, fea60, fea120, fea_hflip, fea_vflip, labels, task) in enumerate(
                        train_task_dl[task_id]):
                    fea = fea.to(device)
                    fea90 = fea90.to(device)
                    fea180 = fea180.to(device)
                    fea270 = fea270.to(device)
                    fea60 = fea60.to(device)
                    fea120 = fea120.to(device)
                    fea_vflip = fea_vflip.to(device)
                    fea_hflip = fea_hflip.to(device)
                    labels = labels.to(device)

                    if task_id > 0 and feature_memory:
                        # TODO check for CMCCPN
                        lat_fea_mem, lat90_fea_mem, lat180_fea_mem, lat270_fea_mem, lbl_fea_mem, task_mem = feature_memory.take()
                        fea = torch.cat((fea, lat_fea_mem.to(device)), 0)
                        fea90 = torch.cat((fea90, lat90_fea_mem.to(device)), 0)
                        fea180 = torch.cat((fea180, lat180_fea_mem.to(device)), 0)
                        fea270 = torch.cat((fea270, lat270_fea_mem.to(device)), 0)
                        labels = torch.cat((labels, lbl_fea_mem.to(device)), 0)

                    output, features, logits = model(fea, fea90, fea180, fea270, fea60, fea120, fea_hflip,
                                                     fea_vflip,
                                                     return_fea=True)

                    total += labels.size(0)
                    prob, predicted = torch.max(nn.functional.softmax(logits, dim=1).data, 1)
                    correct += (predicted == labels).sum().item()
                    loss = nn.functional.cross_entropy(logits, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    loss_step += 1

                test_avg_acc = eval_openset_classifier_CMCCPN(device, model, task_id, test_task_dl,
                                                              unknown_dl=unknown_dl if model.ucfo else None)

                test_avg_acc, acc_t, auroc_clf, baccu_clf = test_avg_acc

                data_epoch_clf['auroc'].append(auroc_clf)
                data_epoch_clf['baccu'].append(baccu_clf)


                data_epoch_clf['seed'].append(seed_trial)
                data_epoch_clf['task_id'].append(task_id)
                data_epoch_clf['epoch'].append(epoch + 1)
                data_epoch_clf['accuracy'].append(test_avg_acc)

                # print(f"Loss: {total_loss / loss_step}")
                # print(f"Accuracy Train: {100 * correct / total}")
                # print(f"ACC_BACCU: {(test_avg_acc/100+baccu)/2}")
                print(
                    f"CLF TASK: {task_id} - {task[0]} | Epoch [{epoch + 1}/{n_epoch}]: Loss: {total_loss / loss_step}, Accuracy Train: {100 * correct / total}, Accuracy Test: {test_avg_acc}, AUROC: {auroc_clf}, BACCU: {baccu_clf}")

            acc_clf = max(data_epoch_clf['accuracy'])

            task_desc_list.append(task[0])
            task_id_list.append(task_id)
            start_class = int(task[0].split(',')[0].split(':')[1].lstrip().split("(")[-1])
            end_class = int(task[0].split(',')[1].lstrip().split(')')[0])
            n_class_per_task = end_class - start_class
            print("n_class_pertask", n_class_per_task)

            model.to(device)

            test_avg_acc, auroc, baccu, forget_t = evalCMCCPN(device, model, task_id, test_task_dl, unknown_dl,
                                                              verbose=1)
            acc_pertask.append(test_avg_acc)
            auroc_pertask.append(auroc)
            baccu_pertask.append(baccu)
            print(f"Test Average Acc from first task 0 to current task {task_id}: {test_avg_acc}")
            print(f"AUROC from first task 0 to current task {task_id}: {auroc}")
            print(f"BACCU from first task 0 to current task {task_id}: {baccu}")

        current_seed += 1

    endtime = time()

    print(f"Total running time: {endtime - start_time}")
    logdf = pd.DataFrame(
        {"task id": task_id_list, "task desc": task_desc_list, "test accuracy": acc_pertask, "auroc": auroc_pertask,
         "baccu": baccu_pertask, 'acc_clf': [acc_clf], 'auroc_dis': [auroc_dis], 'baccu_dis': [baccu_dis], 'opt_acc_baccu': [(acc_clf/100+baccu_dis)/2]})

    if log_path is not None:
        make_dir(log_path)
        logdf.to_csv(log_path)

    logdfepoch = pd.DataFrame(data_epoch)
    dirlog = os.path.dirname(log_path)
    dirpath = Path(dirlog)
    log_epoch = os.path.join(dirpath.parent, f"C_{int(contrastive)}", "CMCCPN", "epoch", os.path.basename(log_path))
    make_dir(log_epoch)
    logdfepoch.to_csv(log_epoch)

    logdfepoch_clf = pd.DataFrame(data_epoch_clf)
    log_epoch_clf = os.path.join(dirpath.parent, f"C_{int(contrastive)}", "CMCCPN", "epoch", "clf_" + os.path.basename(log_path))
    make_dir(log_epoch_clf)
    logdfepoch_clf.to_csv(log_epoch_clf)

    if save_last_model:
        make_dir(save_model_path)
        torch.save(model.state_dict(), save_model_path)

    if return_model:
        return logdf, model

    return logdf

def training_CMCCPN_noncontinual_backbone(device, model, save_model_path, log_path, train_task_dl, test_task_dl, unknown_dl, image_memory=None, load_path=None, learning_rate=0.001, start_epoch=0, n_epoch=50, batch_size=32,
                                          gamma=0.1, contrastive=100, trial=5, initial_seed=1, return_model=False, save_last_model=False, record_epoch=True, n_epoch_classifier=30,
                                          dataset_name="", backbone_name="", idx_combin=""):
    if load_path is not None:
        model.load_state_dict(torch.load(load_path))

    make_dir(save_model_path)

    start_time = time()
    model.train()
    print("DEVICE:", device)
    model.to(device)
    n_epoch += start_epoch
    n_task = len(train_task_dl)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # default gamma=0.1, contrastive=1000
    # qccpn_loss = QCCPNLoss(gamma, contrastive, device=device)
    qccpn_loss = QCCPNLoss(gamma, contrastive, device=device)
    acc_pertask = []
    auroc_pertask = []
    baccu_pertask = []
    task_desc_list = []
    task_id_list = []
    current_seed = initial_seed

    lp_proto_list = []


    data_epoch = {
        "seed": [],
        "task_id": [],
        "epoch": [],
        "accuracy": [],
        "auroc": [],
        "baccu": [],
        "acc_baccu": []
    }


    data_epoch_clf = {
        "seed": [],
        "task_id": [],
        "epoch": [],
        "accuracy": [],
        "auroc": [],
        "baccu": []
    }

    dir_lp_proto = f"{save_model_path}/OSR/CMCCPN/{dataset_name}/{backbone_name}/C_{contrastive}/idx_{idx_combin}/"
    make_dir(dir_lp_proto)

    for seed_trial in range(trial):
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)
        random.seed(current_seed)
        seed = current_seed

        for task_id in range(n_task):

            total_step = len(train_task_dl[task_id])
            if task_id > 0:
                model.extend_prototypes(n_class_per_task)
                model.to(device)
                if image_memory:
                    image_memory.adjust(list(range(0, end_class)), batch_size)

            model.train()
            # Training stage 1
            model.grad_stage1()
            for epoch in range(start_epoch, n_epoch):
                total_loss = 0
                loss_step = 0
                total = 0
                correct = 0
                for i, (img, labels, task) in enumerate(train_task_dl[task_id]):
                    img = img.to(device)
                    labels = labels.to(device)
                    img90, img180, img270, img60, img120, img_hflip, img_vflip = rotate_images_CMCCPN(img, device=device)

                    if task_id > 0 and image_memory:
                        # non continual, not used
                        img_mem, img90_mem, lat180_fea_mem, lat270_fea_mem, lbl_fea_mem, task_mem = image_memory.take()
                        fea = torch.cat((fea, img_mem.to(device)), 0)
                        fea90 = torch.cat((fea90, img90_mem.to(device)), 0)
                        fea180 = torch.cat((fea180, lat180_fea_mem.to(device)), 0)
                        fea270 = torch.cat((fea270, lat270_fea_mem.to(device)), 0)
                        labels = torch.cat((labels, lbl_fea_mem.to(device)), 0)

                    output, features, logits = model(img, img90, img180, img270, img60, img120, img_hflip, img_vflip,
                                                     return_fea=True)

                    total += labels.size(0)
                    prob, predicted = torch.max(nn.functional.softmax(output, dim=1).data, 1)

                    correct += (predicted == labels).sum().item()
                    loss = qccpn_loss(features, labels, model.centers)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    loss_step += 1

                # Eval and record the results for each epoch
                test_avg_acc, auroc, baccu, acc_t = evalCMCCPNBackbone(device, model, task_id, test_task_dl,
                                                                       unknown_dl,
                                                                       verbose=1)
                data_epoch['seed'].append(seed_trial)
                data_epoch['task_id'].append(task_id)
                data_epoch['epoch'].append(epoch + 1)
                data_epoch['accuracy'].append(test_avg_acc)
                data_epoch['auroc'].append(auroc)
                data_epoch['baccu'].append(baccu)
                data_epoch['acc_baccu'].append((test_avg_acc / 100 + baccu) / 2)

                SAVE_PATH = f"epoch_{epoch}_acc_{test_avg_acc:.3f}_auroc_{auroc:.3f}_{today_str()}.pth"
                SAVE_PATH = os.path.join(dir_lp_proto, SAVE_PATH)
                lp_proto_list.append(SAVE_PATH)
                torch.save(model.state_dict(), SAVE_PATH)

                print(
                    f"TASK: {task_id} - {task[0]} | Epoch [{epoch + 1}/{n_epoch}]: Loss: {total_loss / loss_step}, Accuracy Train: {100 * correct / total}, Accuracy Test: {test_avg_acc}, AUROC: {auroc}, BACCU: {baccu}, ACC_BACCU: {(test_avg_acc / 100 + baccu) / 2}")


            max_index = data_epoch['acc_baccu'].index(max(data_epoch['acc_baccu']))
            auroc_dis = data_epoch['auroc'][max_index]
            baccu_dis = data_epoch['baccu'][max_index]
            LOAD_PATH = lp_proto_list[max_index]

            print(f"Loading the best auroc at index {max_index}")
            model.load_state_dict(torch.load(LOAD_PATH))

            model.grad_stage2()
            for epoch in range(n_epoch_classifier):
                total_loss = 0
                loss_step = 0
                total = 0
                correct = 0
                for i, (img, labels, task) in enumerate(
                        train_task_dl[task_id]):
                    img = img.to(device)
                    labels = labels.to(device)

                    img90, img180, img270, img60, img120, img_hflip, img_vflip = rotate_images_CMCCPN(img,
                                                                                                      device=device)

                    if task_id > 0 and image_memory:
                        # non connitual, not used
                        img_mem, img90_mem, lat180_fea_mem, lat270_fea_mem, lbl_fea_mem, task_mem = image_memory.take()
                        fea = torch.cat((fea, img_mem.to(device)), 0)
                        fea90 = torch.cat((fea90, img90_mem.to(device)), 0)
                        fea180 = torch.cat((fea180, lat180_fea_mem.to(device)), 0)
                        fea270 = torch.cat((fea270, lat270_fea_mem.to(device)), 0)
                        labels = torch.cat((labels, lbl_fea_mem.to(device)), 0)

                    output, features, logits = model(img, img90, img180, img270, img60, img120, img_hflip,
                                                     img_vflip,
                                                     return_fea=True)

                    total += labels.size(0)
                    prob, predicted = torch.max(nn.functional.softmax(logits, dim=1).data, 1)
                    correct += (predicted == labels).sum().item()
                    loss = nn.functional.cross_entropy(logits, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    loss_step += 1

                test_avg_acc = eval_openset_classifier_CMCCPN_backbone(device, model, task_id, test_task_dl,
                                                              unknown_dl=unknown_dl if model.ucfo else None)

                test_avg_acc, acc_t, auroc_clf, baccu_clf = test_avg_acc

                data_epoch_clf['auroc'].append(auroc_clf)
                data_epoch_clf['baccu'].append(baccu_clf)


                data_epoch_clf['seed'].append(seed_trial)
                data_epoch_clf['task_id'].append(task_id)
                data_epoch_clf['epoch'].append(epoch + 1)
                data_epoch_clf['accuracy'].append(test_avg_acc)

                # print(f"Loss: {total_loss / loss_step}")
                # print(f"Accuracy Train: {100 * correct / total}")
                # print(f"ACC_BACCU: {(test_avg_acc/100+baccu)/2}")
                print(
                    f"CLF TASK: {task_id} - {task[0]} | Epoch [{epoch + 1}/{n_epoch}]: Loss: {total_loss / loss_step}, Accuracy Train: {100 * correct / total}, Accuracy Test: {test_avg_acc}, AUROC: {auroc_clf}, BACCU: {baccu_clf}")

            acc_clf = max(data_epoch_clf['accuracy'])

            task_desc_list.append(task[0])
            task_id_list.append(task_id)
            start_class = int(task[0].split(',')[0].split(':')[1].lstrip().split("(")[-1])
            end_class = int(task[0].split(',')[1].lstrip().split(')')[0])
            n_class_per_task = end_class - start_class
            print("n_class_pertask", n_class_per_task)

            model.to(device)

            test_avg_acc, auroc, baccu, forget_t = evalCMCCPNBackbone(device, model, task_id, test_task_dl, unknown_dl,
                                                              verbose=1)
            acc_pertask.append(test_avg_acc)
            auroc_pertask.append(auroc)
            baccu_pertask.append(baccu)
            print(f"Test Average Acc from first task 0 to current task {task_id}: {test_avg_acc}")
            print(f"AUROC from first task 0 to current task {task_id}: {auroc}")
            print(f"BACCU from first task 0 to current task {task_id}: {baccu}")

        current_seed += 1

    endtime = time()

    print(f"Total running time: {endtime - start_time}")
    logdf = pd.DataFrame(
        {"task id": task_id_list, "task desc": task_desc_list, "test accuracy": acc_pertask, "auroc": auroc_pertask,
         "baccu": baccu_pertask, 'acc_clf': [acc_clf], 'auroc_dis': [auroc_dis], 'baccu_dis': [baccu_dis], 'opt_acc_baccu': [(acc_clf/100+baccu_dis)/2]})

    if log_path is not None:
        make_dir(log_path)
        logdf.to_csv(log_path)

    logdfepoch = pd.DataFrame(data_epoch)
    dirlog = os.path.dirname(log_path)
    dirpath = Path(dirlog)
    log_epoch = os.path.join(dirpath.parent, f"C_{int(contrastive)}", "CMCCPN", "epoch", os.path.basename(log_path))
    make_dir(log_epoch)
    logdfepoch.to_csv(log_epoch)

    logdfepoch_clf = pd.DataFrame(data_epoch_clf)
    log_epoch_clf = os.path.join(dirpath.parent, f"C_{int(contrastive)}", "CMCCPN", "epoch", "clf_" + os.path.basename(log_path))
    make_dir(log_epoch_clf)
    logdfepoch_clf.to_csv(log_epoch_clf)

    if save_last_model:
        make_dir(save_model_path)
        torch.save(model.state_dict(), save_model_path)

    if return_model:
        return logdf, model

    return logdf

Feature8 = namedtuple('Feature8', 'normal rot90 rot180 rot270 rot60 rot120 hflip vflip')

def PUFS_CMCCPN(centers, modelCMCCPN, modelIN, threshold, device=None, max_ufea=20):
    with torch.no_grad():
        print("Searching unknown feature")
        feasize = centers.size(1)
        temp_ufea = torch.Tensor([]).to(device)
        # ufea = torch.tensor([]).to(device)
        # ufea90 = torch.tensor([]).to(device)
        # ufea180 = torch.tensor([]).to(device)
        # ufea270 = torch.tensor([]).to(device)

        ufea = Feature8(*tuple(torch.tensor(([])).to(device) for x in range(8)))
        nclass = centers.size(0)
        idxpermute = torch.randperm(nclass)
        ccenter = centers[idxpermute].clone()
        ccenter = (centers + ccenter)/2
        # Do genetic operation first

        temp_ufea = torch.concat((temp_ufea, ccenter), 0)
        # ivc, ivc90, ivc180, ivc270 = modelIN(temp_ufea)
        ivc = Feature8(*modelIN(temp_ufea))
        output, _ = modelCMCCPN(*ivc)
        prob, predict = torch.max(nn.functional.softmax(output, 1), 1)
        # print(prob)
        print("threshold", threshold)
        unknown_prob = prob < threshold
        indices = torch.flatten(unknown_prob.nonzero())

        values = []
        for field in ufea._fields:
            values.append(torch.concat((getattr(ufea, field), getattr(ivc, field)[indices])))

        ufea = Feature8(*values)
        # temp_ufea = torch.Tensor([]).to(device)
        takesize = centers.size(0)
        step = 0
        while ufea.normal.size(0) < max_ufea:
            current_size = ufea.normal.size(0)
            if step >= 100:
                break
            print("found unknown feature size: ", ufea.normal.size())
            idx = torch.randperm(temp_ufea.size(0))[:takesize]
            idx2 = torch.randperm(temp_ufea.size(0))[:takesize]
            candidate1 = temp_ufea[idx].clone()
            candidate2 = temp_ufea[idx2].clone()
            alpha = torch.rand(1)[0]
            candidate = alpha * candidate1 + (1-alpha) * candidate2
            temp_ufea = torch.concat((temp_ufea, candidate), 0)
            # ivc, ivc90, ivc180, ivc270 = modelIN(candidate)
            ivc = Feature8(*modelIN(candidate))
            output, _ = modelCMCCPN(*ivc)
            prob, predict = torch.max(nn.functional.softmax(output, 1), 1)
            # Less than confidence indicate the unknown
            unknown_prob = prob < threshold
            indices = torch.flatten(unknown_prob.nonzero())

            values = []
            for field in ufea._fields:
                values.append(torch.concat((getattr(ufea, field), getattr(ivc, field)[indices])))

            ufea = Feature8(*values)

            if ufea.normal.size(0) <= current_size:
                step += 1

        return ufea


def PUFS_CMCCPN_DIRECT_CROSS(centers, modelCMCCPN, modelIN, threshold, device=None, max_ufea=20, feature_dim=256):
    with (torch.no_grad()):
        # TODO you can just concat the final feature and fix the training without using QCCPN to get the feature
        print("Searching unknown feature")
        temp_ufea = torch.Tensor([]).to(device)

        ufea = Feature8(*tuple(torch.tensor(([])).to(device) for x in range(8)))
        nclass = centers.size(0)
        idxpermute = torch.randperm(nclass)
        ccenter = centers[idxpermute].clone()
        ccenter = (centers + ccenter) / 2

        temp_ufea = torch.concat((temp_ufea, ccenter), 0)
        # ivc, ivc90, ivc180, ivc270 = modelIN(temp_ufea)

        # ivc = Feature8(*modelIN(temp_ufea))
        # output, _ = modelCMCCPN(*ivc)
        # prob, predict = torch.max(nn.functional.softmax(output, 1), 1)
        # # print(prob)
        # print("threshold", threshold)
        # unknown_prob = prob < threshold
        # indices = torch.flatten(unknown_prob.nonzero())
        #
        # values = []
        # for field in ufea._fields:
        #     values.append(torch.concat((getattr(ufea, field), getattr(ivc, field)[indices])))
        #
        # ufea = Feature8(*values)
        # # temp_ufea = torch.Tensor([]).to(device)

        takesize = centers.size(0)
        step = 0


        while ufea.normal.size(0) < max_ufea:
            current_size = ufea.normal.size(0)
            if step >= 100:
                break
            print("heuristic unknown feature size: ", ufea.normal.size())
            idx = torch.randperm(temp_ufea.size(0))[:takesize]
            idx2 = torch.randperm(temp_ufea.size(0))[:takesize]
            candidate1 = temp_ufea[idx].clone()
            candidate2 = temp_ufea[idx2].clone()

            idxshuffle = list(range(8))
            random.shuffle(idxshuffle)
            # print("idxshuffle", idxshuffle)
            # print(f"candidate2 at index {idxshuffle[0]}", candidate2[0, idxshuffle[0]*feature_dim
            # : (idxshuffle[0]+1)*feature_dim])
            acombin = tuple(candidate2[:, feature_dim*x:feature_dim*(x+1)] for x in idxshuffle)
            candidate3 = torch.cat(acombin, dim=1)
            # print(f"candidate3 at index {0}", candidate3[0, 0 * feature_dim: (0 + 1) * feature_dim])


            alpha = torch.rand(1)[0]
            candidate = alpha * candidate1 + (1 - alpha) * candidate3
            temp_ufea = torch.concat((temp_ufea, candidate), 0)
            # ivc, ivc90, ivc180, ivc270 = modelIN(candidate)
            ivc = Feature8(*modelIN(candidate))
            # output, _ = modelCMCCPN(*ivc)
            # prob, predict = torch.max(nn.functional.softmax(output, 1), 1)
            # # Less than confidence indicate the unknown
            # unknown_prob = prob < threshold
            # indices = torch.flatten(unknown_prob.nonzero())

            values = []
            for field in ufea._fields:
                values.append(torch.concat((getattr(ufea, field), getattr(ivc, field))))

            ufea = Feature8(*values)

            print("heuristic unknown feature size: ", ufea.normal.size())

            if ufea.normal.size(0) <= current_size:
                step += 1

        return ufea

def PUFS_CMCCPN_CROSS(centers, modelCMCCPN, modelIN, threshold, device=None, max_ufea=20, feature_dim=256):
    with (torch.no_grad()):
        # TODO you can just concat the final feature and fix the training without using QCCPN to get the feature
        print("Searching unknown feature")
        temp_ufea = torch.Tensor([]).to(device)

        ufea = Feature8(*tuple(torch.tensor(([])).to(device) for x in range(8)))
        nclass = centers.size(0)
        idxpermute = torch.randperm(nclass)
        ccenter = centers[idxpermute].clone()
        ccenter = (centers + ccenter) / 2

        temp_ufea = torch.concat((temp_ufea, ccenter), 0)
        # ivc, ivc90, ivc180, ivc270 = modelIN(temp_ufea)

        ivc = Feature8(*modelIN(temp_ufea))
        output, _ = modelCMCCPN(*ivc)
        prob, predict = torch.max(nn.functional.softmax(output, 1), 1)
        # print(prob)
        print("threshold", threshold)
        unknown_prob = prob < threshold
        indices = torch.flatten(unknown_prob.nonzero())

        values = []
        for field in ufea._fields:
            values.append(torch.concat((getattr(ufea, field), getattr(ivc, field)[indices])))

        ufea = Feature8(*values)
        # temp_ufea = torch.Tensor([]).to(device)

        takesize = centers.size(0)
        step = 0


        while ufea.normal.size(0) < max_ufea:
            current_size = ufea.normal.size(0)
            if step >= 100:
                break
            print("heuristic unknown feature size: ", ufea.normal.size())
            idx = torch.randperm(temp_ufea.size(0))[:takesize]
            idx2 = torch.randperm(temp_ufea.size(0))[:takesize]
            candidate1 = temp_ufea[idx].clone()
            candidate2 = temp_ufea[idx2].clone()

            idxshuffle = list(range(8))
            random.shuffle(idxshuffle)
            # print("idxshuffle", idxshuffle)
            # print(f"candidate2 at index {idxshuffle[0]}", candidate2[0, idxshuffle[0]*feature_dim
            # : (idxshuffle[0]+1)*feature_dim])
            acombin = tuple(candidate2[:, feature_dim*x:feature_dim*(x+1)] for x in idxshuffle)
            candidate3 = torch.cat(acombin, dim=1)
            # print(f"candidate3 at index {0}", candidate3[0, 0 * feature_dim: (0 + 1) * feature_dim])


            alpha = torch.rand(1)[0]
            candidate = alpha * candidate1 + (1 - alpha) * candidate3
            temp_ufea = torch.concat((temp_ufea, candidate), 0)
            # ivc, ivc90, ivc180, ivc270 = modelIN(candidate)
            ivc = Feature8(*modelIN(candidate))
            output, _ = modelCMCCPN(*ivc)
            prob, predict = torch.max(nn.functional.softmax(output, 1), 1)
            # Less than confidence indicate the unknown
            unknown_prob = prob < threshold
            indices = torch.flatten(unknown_prob.nonzero())

            values = []
            for field in ufea._fields:
                values.append(torch.concat((getattr(ufea, field), getattr(ivc, field)[indices])))

            ufea = Feature8(*values)

            print("heuristic unknown feature size: ", ufea.normal.size())

            if ufea.normal.size(0) <= current_size:
                step += 1

        return ufea
def training_CMCCPN(device, model, save_model_path, log_path, train_task_dl, test_task_dl, unknown_dl,
                    feature_memory=None, load_path=None, learning_rate=0.001, start_epoch=0, n_epoch=50, n_epoch_continual=30, n_epoch_classifier=30,
                    batch_size=32,
                    dataset_name="None", backbone_name="",
                    gamma=0.1, contrastive=100, trial=5, initial_seed=1, backbone_dim=960, feature_dim=2048, NO_PUFS=False, DIRECT=False, CROSS=False, SIMILAR_PROTO=False, DISTILL_LOSS=False, track_best_open=True):

    if load_path is not None:
        model.load_state_dict(torch.load(load_path))


    if track_best_open:
        dir_lp_proto = f"{save_model_path}/Continual/CMCCPN/{dataset_name}/{backbone_name}/C_{contrastive}/"
        make_dir(dir_lp_proto)

    make_dir(save_model_path)
    make_dir(log_path)

    start_time = time()
    model.train()
    print("DEVICE:", device)
    model.to(device)
    n_epoch += start_epoch
    n_task = len(train_task_dl)

    # default gamma=0.1, contrastive=1000
    qccpn_loss = QCCPNLoss(gamma, contrastive, device=device)


    current_seed = initial_seed

    def init_data_epoch():
        return {
            "seed": [],
            "task_id": [],
            "epoch": [],
            "accuracy": [],
            "auroc": [],
            "baccu": [],
            "acc_baccu": []
        }

    for seed_trial in range(trial):
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)
        random.seed(current_seed)
        seed = current_seed

        inet = InverseNetworkMultiChannel(int(feature_dim/8), backbone_dim=backbone_dim)

        with torch.no_grad():
            ufea_mem = torch.Tensor([]).to(device)
            ufea90_mem = torch.Tensor([]).to(device)
            ufea180_mem = torch.Tensor([]).to(device)
            ufea270_mem = torch.Tensor([]).to(device)

            ufea60_mem = torch.Tensor([]).to(device)
            ufea120_mem = torch.Tensor([]).to(device)
            ufea_hflip_mem = torch.Tensor([]).to(device)
            ufea_vflip_mem = torch.Tensor([]).to(device)

            # ufea_mem = Feature8(tuple(torch.tensor(([])).to(device) for x in range(8)))

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        record_acc_t = []
        record_acc_dis_t = []
        acc_all_pertask = []
        auroc_pertask = []
        baccu_pertask = []
        acc_baccu_pertask = []
        task_desc_list = []
        task_id_list = []
        forget = []
        acc_dis_pertask = []

        auroc_clf_pertask = []
        baccu_clf_pertask = []
        for task_id in range(n_task):
            total_step = len(train_task_dl[task_id])
            lp_proto_list = []
            data_epoch_taskid = init_data_epoch()
            if task_id > 0:
                start_epoch = 0
                n_epoch = n_epoch_continual

                if DISTILL_LOSS:
                    prev_model = copy.deepcopy(model)
                    prev_model.eval()

                if SIMILAR_PROTO:
                    prev_proto = torch.clone(model.centers)
                    prev_proto.requires_grad = False
                    prev_class_num = model.centers.size(0)

                model.extend_prototypes_logits(n_class_per_task)
                model = model.to(device)


                # contrast_param = contrastive/2 if DIRECT else contrastive
                ci_qccpn_loss = ClassIncrementalQCCPNLoss(gamma, contrastive, device=device)
                # ci_qccpn_loss = ClassIncrementalQCCPNLoss(gamma, contrast_param, device=device)

                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                if feature_memory:
                    feature_memory.adjust(list(range(0, end_class)), batch_size)

            model.train()
            model.grad_stage1()
            for epoch in range(start_epoch, n_epoch):
                total_loss = 0
                loss_step = 0
                total = 0
                correct = 0
                for i, (fea, fea90, fea180, fea270, fea60, fea120, fea_hflip, fea_vflip, labels, task) in enumerate(train_task_dl[task_id]):
                    fea = fea.to(device)
                    fea90 = fea90.to(device)
                    fea180 = fea180.to(device)
                    fea270 = fea270.to(device)

                    fea60 = fea60.to(device)
                    fea120 = fea120.to(device)
                    fea_hflip = fea_hflip.to(device)
                    fea_vflip = fea_vflip.to(device)

                    labels = labels.to(device)

                    if task_id > 0 and feature_memory:
                        # TODO check for QCCPN
                        lat_fea_mem, lat90_fea_mem, lat180_fea_mem, lat270_fea_mem, lat60_fea_mem, lat120_fea_mem, lat_hflip_fea_mem, lat_vflip_fea_mem, lbl_fea_mem, task_mem = feature_memory.take()
                        fea_combin = torch.cat((fea, lat_fea_mem.to(device)), 0)
                        fea90_combin = torch.cat((fea90, lat90_fea_mem.to(device)), 0)
                        fea180_combin = torch.cat((fea180, lat180_fea_mem.to(device)), 0)
                        fea270_combin = torch.cat((fea270, lat270_fea_mem.to(device)), 0)

                        fea60_combin = torch.cat((fea60, lat60_fea_mem.to(device)), 0)
                        fea120_combin = torch.cat((fea120, lat120_fea_mem.to(device)), 0)
                        fea_hflip_combin = torch.cat((fea_hflip, lat_hflip_fea_mem.to(device)), 0)
                        fea_vflip_combin = torch.cat((fea_vflip, lat_vflip_fea_mem.to(device)), 0)

                        labels = torch.cat((labels, lbl_fea_mem.to(device)), 0)

                        output, features, logits = model(fea_combin, fea90_combin, fea180_combin, fea270_combin, fea60_combin, fea120_combin, fea_hflip_combin, fea_vflip_combin, return_fea=True)
                    else:
                        output, features, logits = model(fea, fea90, fea180, fea270, fea60, fea120, fea_hflip, fea_vflip, return_fea=True)
                    total += labels.size(0)
                    prob, predicted = torch.max(nn.functional.softmax(output, dim=1).data, 1)
                    correct += (predicted == labels).sum().item()

                    if task_id > 0:

                        # print("ufea_mem shape", ufea_mem.shape)
                        if NO_PUFS:
                            # print("entering no pufs")
                            unknown_features = None
                        else:
                            _, unknown_features, _ = model(ufea_mem, ufea90_mem, ufea180_mem, ufea270_mem, ufea60_mem, ufea120_mem, ufea_hflip_mem, ufea_vflip_mem, return_fea=True)




                        if SIMILAR_PROTO and DISTILL_LOSS:
                            _, features_distill, _ = prev_model(fea, fea90,
                                                                fea180,
                                                                fea270,
                                                                fea60,
                                                                fea120,
                                                                fea_hflip,
                                                                fea_vflip,
                                                                return_fea=True)
                            _, features_curr, _ = model(fea, fea90, fea180, fea270, fea60, fea120, fea_hflip,
                                                        fea_vflip, return_fea=True)
                            loss = ci_qccpn_loss(features, labels, model.centers,
                                                 unknown_features) + nn.functional.mse_loss(features_distill, features_curr) + nn.functional.mse_loss(prev_proto, model.centers[:prev_class_num])
                        elif SIMILAR_PROTO:
                            loss = ci_qccpn_loss(features, labels, model.centers, unknown_features) + nn.functional.mse_loss(prev_proto, model.centers[:prev_class_num])
                        elif DISTILL_LOSS:

                            with torch.no_grad():
                                _, features_distill, _ = prev_model(fea, fea90,
                                                                                              fea180,
                                                                                              fea270,
                                                                                              fea60,
                                                                                              fea120,
                                                                                              fea_hflip,
                                                                                              fea_vflip,
                                                                                              return_fea=True)
                                _, features_curr, _ = model(fea, fea90, fea180, fea270, fea60, fea120, fea_hflip,
                                                                 fea_vflip, return_fea=True)

                            loss = ci_qccpn_loss(features, labels, model.centers,
                                                 unknown_features) + nn.functional.mse_loss(features_distill, features_curr)
                        else:
                            # print("no pufs ci qccpn")
                            loss = ci_qccpn_loss(features, labels, model.centers, unknown_features)

                    else:
                        loss = qccpn_loss(features, labels, model.centers)

                    optimizer.zero_grad()

                    # loss.backward(retain_graph=True)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    loss_step += 1

                test_avg_acc, auroc, baccu, acc_t = evalCMCCPN(device, model, task_id, test_task_dl, unknown_dl,
                                                               verbose=1)
                data_epoch_taskid['seed'].append(seed_trial)
                data_epoch_taskid['task_id'].append(task_id)
                data_epoch_taskid['epoch'].append(epoch + 1)
                data_epoch_taskid['accuracy'].append(test_avg_acc)
                data_epoch_taskid['auroc'].append(auroc)
                data_epoch_taskid['baccu'].append(baccu)
                data_epoch_taskid['acc_baccu'].append((test_avg_acc / 100 + baccu) / 2)

                if track_best_open:
                    SAVE_PATH = f"task_id_{task_id}_C_{contrastive}_epoch_{epoch}_acc_{test_avg_acc:.3f}_auroc_{auroc:.3f}.pth"
                    SAVE_PATH = os.path.join(dir_lp_proto, SAVE_PATH)
                    lp_proto_list.append(SAVE_PATH)
                    torch.save(model.state_dict(), SAVE_PATH)

                print(
                    f"TASK: {task_id} - {task[0]} | Epoch [{epoch + 1}/{n_epoch}]: Loss: {total_loss / loss_step}, Accuracy Train: {100 * correct / total}, Accuracy Test: {test_avg_acc}, AUROC: {auroc}, BACCU: {baccu}, ACC_BACCU: {(test_avg_acc / 100 + baccu) / 2}")



            if track_best_open:
                max_index = data_epoch_taskid['acc_baccu'].index(max(data_epoch_taskid['acc_baccu']))
                LOAD_PATH = lp_proto_list[max_index]

                print(f"Loading the best auroc at index {max_index}")
                model.load_state_dict(torch.load(LOAD_PATH))

            model.grad_stage2()
            for epoch in range(n_epoch_classifier):
                total_loss = 0
                loss_step = 0
                total = 0
                correct = 0
                for i, (fea, fea90, fea180, fea270, fea60, fea120, fea_hflip, fea_vflip, labels, task) in enumerate(
                        train_task_dl[task_id]):
                    fea = fea.to(device)
                    fea90 = fea90.to(device)
                    fea180 = fea180.to(device)
                    fea270 = fea270.to(device)
                    fea60 = fea60.to(device)
                    fea120 = fea120.to(device)
                    fea_vflip = fea_vflip.to(device)
                    fea_hflip = fea_hflip.to(device)
                    labels = labels.to(device)

                    if task_id > 0 and feature_memory:

                        if DISTILL_LOSS:
                            prev_model = copy.deepcopy(model)
                            prev_model.eval()

                        # TODO check for CMCCPN
                        lat_fea_mem, lat90_fea_mem, lat180_fea_mem, lat270_fea_mem, lat60_fea_mem, lat120_fea_mem, lat_hflip_fea_mem, lat_vflip_fea_mem, lbl_fea_mem, task_mem = feature_memory.take()
                        fea_combin = torch.cat((fea, lat_fea_mem.to(device)), 0)
                        fea90_combin = torch.cat((fea90, lat90_fea_mem.to(device)), 0)
                        fea180_combin = torch.cat((fea180, lat180_fea_mem.to(device)), 0)
                        fea270_combin = torch.cat((fea270, lat270_fea_mem.to(device)), 0)

                        fea60_combin = torch.cat((fea60, lat60_fea_mem.to(device)), 0)
                        fea120_combin = torch.cat((fea120, lat120_fea_mem.to(device)), 0)
                        fea_hflip_combin = torch.cat((fea_hflip, lat_hflip_fea_mem.to(device)), 0)
                        fea_vflip_combin = torch.cat((fea_vflip, lat_vflip_fea_mem.to(device)), 0)

                        labels = torch.cat((labels, lbl_fea_mem.to(device)), 0)
                        output, features, logits = model(fea_combin, fea90_combin, fea180_combin, fea270_combin, fea60_combin, fea120_combin, fea_hflip_combin, fea_vflip_combin, return_fea=True)
                    else:
                        output, features, logits = model(fea, fea90, fea180, fea270, fea60, fea120, fea_hflip, fea_vflip, return_fea=True)

                    total += labels.size(0)
                    prob, predicted = torch.max(nn.functional.softmax(logits, dim=1).data, 1)
                    correct += (predicted == labels).sum().item()
                    # print("predicted", predicted, predicted.size())
                    # print("labels", labels, labels.size())
                    if DISTILL_LOSS and task_id > 0:
                        with torch.no_grad():
                            _, _, logits_distill = prev_model(fea, fea90,
                                                                fea180,
                                                                fea270,
                                                                fea60,
                                                                fea120,
                                                                fea_hflip,
                                                                fea_vflip,
                                                                return_fea=True)
                            _, _, logits_curr = model(fea, fea90, fea180, fea270, fea60, fea120, fea_hflip,
                                                        fea_vflip, return_fea=True)

                        loss = nn.functional.cross_entropy(logits, labels) + nn.functional.mse_loss(logits_distill, logits_curr)
                    else:
                        loss = nn.functional.cross_entropy(logits, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    loss_step += 1

                print(
                    f"CLF TASK: {task_id} - {task[0]} | Epoch [{epoch + 1}/{n_epoch_classifier}]: Loss: {total_loss / loss_step}, Accuracy Train: {100 * correct / total}")

            task_desc_list.append(task[0])
            task_id_list.append(task_id)

            start_class = int(task[0].split(',')[0].split(':')[1].lstrip().split("(")[-1])
            end_class = int(task[0].split(',')[1].lstrip().split(')')[0])
            n_class_per_task = end_class - start_class
            # print("n_class_pertask", n_class_per_task)
            model.to(device)
            model.eval()
            # if task_id == 0:

            acc_dis, auroc, baccu, threshold, acc_dis_t = evalCMCCPN(device, model, task_id, test_task_dl, unknown_dl, verbose=1, return_threshold=True)
            acc_dis_pertask.append(acc_dis)
            record_acc_dis_t.append(acc_dis_t)
            # else:
            #     test_avg_acc, baccu = evalBACCUQCCPN(device, threshold, model, task_id, test_task_dl, unknown_dl, verbose=1)



            auroc_pertask.append(auroc)
            baccu_pertask.append(baccu)


            test_avg_acc, acc_t, auroc_clf, baccu_clf = eval_classifier_CMCCPN(device, model, task_id, test_task_dl, unknown_dl=unknown_dl)
            acc_all_pertask.append(test_avg_acc)
            record_acc_t.append(acc_t)
            auroc_clf_pertask.append(auroc_clf)
            baccu_clf_pertask.append(baccu_clf)

            acc_baccu_pertask.append((test_avg_acc / 100 + baccu) / 2)
            # print("ACC_T", acc_t)
            # print("RECORD_ACC_T", record_acc_t)

            # if len(acc_t) == 1:
            #     forget.append(-99999)
            # else:
            #     list_forget = [v - acc_t[-1] for i, v in enumerate(acc_t) if i < len(acc_t)-1]
            #     print("list_forget", list_forget)
            #     max_forget = max(list_forget)
            #     print("MAX FORGET", max_forget)
            #     forget.append(max_forget)
            #     print("FORGET", forget)

            print(f"Test Average Acc from first task 0 to current task {task_id}: {test_avg_acc}")
            print(f"AUROC from first task 0 to current task {task_id}: {auroc}")
            print(f"BACCU from first task 0 to current task {task_id}: {baccu}")

            if not NO_PUFS:

                if task_id > 0 and feature_memory:
                    mse_inet = training_InverseNetwork_CMCCPN(device, inet, model, None, None, train_task_dl[task_id], unknown_dl,
                                                             feature_memory=feature_memory,
                                                             n_epoch=n_epoch, initial_seed=seed)
                else:
                    mse_inet = training_InverseNetwork_CMCCPN(device, inet, model, None, None, train_task_dl[task_id], unknown_dl,
                                                      feature_memory=None,
                                                      n_epoch=n_epoch, initial_seed=seed)


                if DIRECT:
                    print("DIRECT PUFS...")
                    ufea_t = PUFS_CMCCPN_DIRECT_CROSS(model.centers.detach(), model, inet, threshold, device=device, feature_dim=int(feature_dim / 8))
                elif CROSS:
                    ufea_t = PUFS_CMCCPN_CROSS(model.centers.detach(), model, inet, threshold, device=device,
                                               feature_dim=int(feature_dim / 8))
                else:
                    ufea_t = PUFS_CMCCPN(model.centers.detach(), model, inet, threshold, device=device)

                ufea_mem = torch.concat((ufea_mem, ufea_t.normal), 0)
                ufea90_mem = torch.concat((ufea90_mem, ufea_t.rot90), 0)
                ufea180_mem = torch.concat((ufea180_mem, ufea_t.rot180), 0)
                ufea270_mem = torch.concat((ufea270_mem, ufea_t.rot270), 0)

                ufea60_mem = torch.concat((ufea60_mem, ufea_t.rot60), 0)
                ufea120_mem = torch.concat((ufea120_mem, ufea_t.rot120), 0)
                ufea_hflip_mem = torch.concat((ufea_hflip_mem, ufea_t.hflip), 0)
                ufea_vflip_mem = torch.concat((ufea_vflip_mem, ufea_t.vflip), 0)


        current_seed += 1

    endtime = time()

    print(f"Total running time: {endtime - start_time}")


    # Calculate average forgetting
    list_forget = []
    for t in range(n_task-1):
        a_kj = record_acc_t[n_task-1][t]
        f_kj = max([record_acc_t[l][t] for l in range(t, n_task-1)]) - a_kj
        list_forget.append(f_kj)

    avg_forget = sum(list_forget)/(n_task-1)
    forget = [avg_forget] * n_task

    # TODO check this
    list_forget_dis = []
    for t in range(n_task - 1):
        a_kj = record_acc_dis_t[n_task - 1][t]
        f_kj = max([record_acc_dis_t[l][t] for l in range(t, n_task - 1)]) - a_kj
        list_forget_dis.append(f_kj)

    avg_forget_dis = sum(list_forget_dis) / (n_task - 1)
    forget_dis = [avg_forget_dis] * n_task

    logdf = pd.DataFrame(
        {"task id": task_id_list, "task desc": task_desc_list, "test accuracy": acc_all_pertask,  "test accuracy distance": acc_dis_pertask, "auroc": auroc_pertask, "baccu": baccu_pertask, "acc_baccu": acc_baccu_pertask,"auroc_clf": auroc_clf_pertask, "baccu_clf": baccu_clf_pertask, "forget": forget, "forget distance": forget_dis})
    logdf.to_csv(log_path)
    return logdf