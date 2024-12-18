import itertools

import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.sgd import SGD
from torchvision import transforms

from datasets import *
from utils import *


def train_bjan(model, args, source_loader, targetdatasets, device, best_model_path, mbu=False):
    target_loader, val_loader = get_data_loaders(targetdatasets, args)

    mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
        kernels=[GaussianKernel(sigma=k ** 0.5, track_running_stats=False) for k in [10, 15, 20, 50]],
        linear=False
    )

    optimizer_g = torch.optim.Adam(
        [v for k, v in model.named_parameters() if 'decoder' not in k],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    optimizer_f = torch.optim.Adam([
        {"params": [v for k, v in model.named_parameters() if 'decoder' in k]},
    ], lr=args.lr, weight_decay=args.weight_decay, )

    ## train
    best_f1 = 0.0
    criterion = nn.CrossEntropyLoss()
    source_iter, target_iter = iter(cycle(source_loader)), iter(cycle(target_loader))
    global_step = 0
    log = list()
    for epoch in range(args.epochs):
        progress_bar = tqdm(range(args.steps_per_epoch), desc=f'{args.method.upper()} Epoch {epoch + 1}/{args.epochs}')

        cls_losses = AverageMeter('CLS Loss', ':.4e')
        mcd_losses = AverageMeter('MCD Loss', ':.4e')
        mmd_losses = AverageMeter('MMD Loss', ':.4e')

        # switch to train mode
        model.train()
        mkmmd_loss.train()

        for _ in progress_bar:
            sample_s, sample_t = next(source_iter), next(target_iter)
            X_s, y_s = sample_s
            X_t, y_t = sample_t
            X = cat_samples([X_s, X_t])
            X = recursive_todevice(X, device)
            y_s = y_s.to(device)

            # Step A train all networks to minimize loss on source domain
            optimizer_g.zero_grad()
            optimizer_f.zero_grad()

            cls_loss = 0.0
            outputs_t = torch.zeros(args.batchsize, args.nclasses, args.num_ens).to(device)
            for i in range(args.num_ens):
                out, feat = model(X, return_feats=True)
                cls_loss += F.cross_entropy(out.chunk(2, dim=0)[0], y_s)
                outputs_t[:,:,i] = F.softmax(out.chunk(2, dim=0)[1], dim=1)

            if mbu:
                loss = cls_loss
            else:
                feat_s, feat_t = feat.chunk(2, dim=0)
                mmd_loss = mkmmd_loss(feat_s, feat_t) * args.num_ens
                loss = cls_loss + mmd_loss * args.trade_off_mmd
            loss.backward()
            optimizer_g.step()
            optimizer_f.step()

            # Step B train classifier to maximize discrepancy
            optimizer_f.zero_grad()

            outputs_t = torch.zeros(args.batchsize, args.nclasses, args.num_ens).to(device)
            cls_loss = 0.0
            for i in range(args.num_ens):
                out, _ = model(X, return_feats=True)
                outputs_t[:, :, i] = F.softmax(out.chunk(2, dim=0)[1], dim=1)
                cls_loss += F.cross_entropy(out.chunk(2, dim=0)[0], y_s)
            mcd_loss = multi_classifier_discrepancy2(outputs_t)
            loss = cls_loss - mcd_loss * args.trade_off
            loss.backward()
            optimizer_f.step()

            # Step C train genrator to minimize discrepancy
            for k in range(args.num_k):
                optimizer_g.zero_grad()

                outputs_t = torch.zeros(args.batchsize, args.nclasses, args.num_ens).to(device)
                for i in range(args.num_ens):
                    out, feat = model(X, return_feats=True)
                    outputs_t[:, :, i] = F.softmax(out.chunk(2, dim=0)[1], dim=1)
                mcd_loss = multi_classifier_discrepancy2(outputs_t)

                feat_s, feat_t = feat.chunk(2, dim=0)
                mmd_loss = mkmmd_loss(feat_s, feat_t) * args.num_ens
                if mbu:
                    loss = mcd_loss * args.trade_off
                else:
                    loss = mcd_loss * args.trade_off + mmd_loss * args.trade_off_mmd
                    # loss = mmd_loss * args.trade_off_mmd
                loss.backward()
                optimizer_g.step()

            cls_losses.update(cls_loss.item(), args.batchsize)
            mmd_losses.update(mmd_loss.item(), args.batchsize)
            mcd_losses.update(mcd_loss.item(), args.batchsize)

            progress_bar.set_postfix(
                class_loss=f"{cls_losses.avg:.3f}",
                mcd_loss=f"{mcd_losses.avg:.3f}",
                mmd_loss=f"{mmd_losses.avg:.3f}",
            )

            global_step += 1

        progress_bar.close()

        losses = {
            'class_loss': cls_losses.avg,
            'mcd_loss': mcd_losses.avg,
            'mmd_loss': mmd_losses.avg, }
        model.eval()
        best_f1 = validation(best_f1, model, criterion, val_loader, device, args.nclasses,
                             log=log, epoch=epoch, best_model_path=best_model_path, train_loss=losses)

    # save final model and use for evaluation
    print(f"saving model to {str(best_model_path)}\n")
    torch.save({"model_state": model.state_dict(), 'criterion': criterion}, best_model_path)


def get_data_loaders(targetdatasets, args):
    year = args.target.split('_')[1]
    def create_data_loader(dataset):
        return torch.utils.data.DataLoader(
            dataset,
            num_workers=args.workers,
            pin_memory=True,
            batch_size=args.batchsize,
            shuffle=True,
            drop_last=True
        )

    train_transform = transforms.Compose([
        RandomTempRemoval(),  # temp
        RandomTempShift(),  # temp
        RandomSampleTimeSteps(args.sequencelength, args.inseason),
        Normalize(year),
        RandomAddNoise(),  # spec
        ToTensor(),
    ])

    test_transform = transforms.Compose([
        RandomSampleTimeSteps(args.sequencelength, args.inseason),
        Normalize(year),
        ToTensor(),
    ])

    val_i = args.test_fold - 1
    train_is = list(range(5))
    train_is.pop(val_i)
    trainvalsets = [deepcopy(targetdatasets[i]) for i in range(5)]  # train_is
    trainvaldataset = CropConcatDataset(trainvalsets)
    trainvaldataset.update_transform(train_transform)

    num_train = len(trainvaldataset)
    indices = list(range(num_train))
    num_val = int(num_train * args.val_ratio)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[num_val:], indices[:num_val]

    traindataset = Subset(deepcopy(trainvaldataset), train_idx)
    traindataset.update_transform(train_transform)
    valdataset = Subset(deepcopy(trainvaldataset), valid_idx)
    valdataset.update_transform(test_transform)

    target_loader = create_data_loader(traindataset)
    val_loader = create_data_loader(valdataset)

    print(f"size of train target data: {num_train - num_val} ({len(target_loader)} batches)")
    print(f"size of val target data: {num_val} ({len(val_loader)} batches)")

    return target_loader, val_loader


def entropy(predictions):
    if len(predictions.shape) == 2:
        return -torch.mean(torch.log(torch.mean(predictions, 0) + 1e-6))  # ent1
        # return -(predictions*torch.log(predictions+1e-6)).sum(1).mean().item()  # ent2
    elif len(predictions.shape) == 3:
        ent = []
        for i in range(predictions.shape[-1]):
            ent.append(entropy(predictions[:, :, i]))
        return torch.tensor(ent).sum()


def classifier_discrepancy(predictions1: torch.Tensor, predictions2: torch.Tensor, weight=None) -> torch.Tensor:
    if weight is None:
        weight = torch.ones(predictions1.size(0), 1).to(predictions1.device)
    return torch.mean(torch.abs(predictions1 - predictions2) * weight)  # .sum(1)


def multi_classifier_discrepancy(predictions):
    num_ens = predictions.size(-1)
    dis = 0
    for n, (i, j) in enumerate(itertools.combinations(np.arange(num_ens), 2)):
        dis += classifier_discrepancy(predictions[:, :, i],  predictions[:, :, j])
    return dis


def multi_classifier_discrepancy1(predictions, reduction='mean'):
    """max entropy"""
    avg_pi = predictions.mean(-1)
    if reduction == 'mean':
        dis = -(avg_pi*torch.log(avg_pi+1e-5)).mean()
    elif reduction == 'sum':
        dis = -(avg_pi * torch.log(avg_pi + 1e-5)).mean(-1).sum()
    elif reduction == 'none':
        dis = -(avg_pi * torch.log(avg_pi + 1e-5)).mean(-1)
    # dis = -((avg_pi*torch.log(avg_pi + 1e-5)).sum(1)).mean()
    return dis


def multi_classifier_discrepancy2(predictions):
    """multual information"""
    avg_pi = predictions.mean(-1)
    G_x = -((avg_pi*torch.log(avg_pi + 1e-5)).sum(1)).mean()
    F_x = -(predictions*torch.log(predictions + 1e-5)).sum(1).mean()
    dis = (G_x - F_x)

    return dis


def multi_classifier_similarity(predictions):
    num_ens = predictions.size(-1)
    dis = 0
    for n, (i, j) in enumerate(itertools.combinations(np.arange(num_ens), 2)):
        dis += classifier_cos_similarity(predictions[:, :, i],  predictions[:, :, j])
    return -dis


def classifier_similarity(predictions1: torch.Tensor, predictions2: torch.Tensor, weight=None) -> torch.Tensor:
    if weight is None:
        weight = torch.ones(predictions1.size(0), 1).to(predictions1.device)
    return torch.mean((predictions1 * predictions2).sum(1) * weight)


def classifier_cos_similarity(predictions1: torch.Tensor, predictions2: torch.Tensor, weight=None) -> torch.Tensor:
    b, c = predictions1.shape
    if weight is None:
        weight = torch.ones(predictions1.size(0), 1).to(predictions1.device)
    return torch.mean((predictions1.view(b, 1, c) @ predictions2.view(b, c, 1)).flatten() / (
            torch.norm(predictions1, dim=1) * torch.norm(predictions2, dim=1)) * weight)


def binary_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Computes the accuracy for binary classification"""
    with torch.no_grad():
        batch_size = target.size(0)
        pred = (output >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100.0 / batch_size)
        return correct


def logmeanexp(x, dim=2, keepdim=False):
    """Stable computation of log(mean(exp(x))"""

    if dim is None:
        x, dim = x.view(-1), 0
    x_max, _ = torch.max(x, dim, keepdim=True)
    x = x_max + torch.log(torch.mean(torch.exp(x - x_max), dim, keepdim=True))
    return x if keepdim else x.squeeze(dim)


'''Components'''
class MultipleKernelMaximumMeanDiscrepancy(nn.Module):
    r"""The Multiple Kernel Maximum Mean Discrepancy (MK-MMD)
    """

    def __init__(self, kernels: Sequence[nn.Module], linear: Optional[bool] = False):
        super(MultipleKernelMaximumMeanDiscrepancy, self).__init__()
        self.kernels = kernels
        self.index_matrix = None
        self.linear = linear

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        features = torch.cat([z_s, z_t], dim=0)
        batch_size = int(z_s.size(0))
        self.index_matrix = _update_index_matrix(batch_size, self.index_matrix, self.linear).to(z_s.device)

        kernel_matrix = sum([kernel(features) for kernel in self.kernels])  # Add up the matrix of each kernel
        # Add 2 / (n-1) to make up for the value on the diagonal
        # to ensure loss is positive in the non-linear version
        loss = (kernel_matrix * self.index_matrix).sum() + 2. / float(batch_size - 1)

        return loss


def _update_index_matrix(batch_size: int, index_matrix: Optional[torch.Tensor] = None,
                         linear: Optional[bool] = True) -> torch.Tensor:
    r"""
    Update the `index_matrix` which convert `kernel_matrix` to loss.
    If `index_matrix` is a tensor with shape (2 x batch_size, 2 x batch_size), then return `index_matrix`.
    Else return a new tensor with shape (2 x batch_size, 2 x batch_size).
    """
    if index_matrix is None or index_matrix.size(0) != batch_size * 2:
        index_matrix = torch.zeros(2 * batch_size, 2 * batch_size)
        if linear:
            for i in range(batch_size):
                s1, s2 = i, (i + 1) % batch_size
                t1, t2 = s1 + batch_size, s2 + batch_size
                index_matrix[s1, s2] = 1. / float(batch_size)
                index_matrix[t1, t2] = 1. / float(batch_size)
                index_matrix[s1, t2] = -1. / float(batch_size)
                index_matrix[s2, t1] = -1. / float(batch_size)
        else:
            for i in range(batch_size):
                for j in range(batch_size):
                    # if i != j:
                    index_matrix[i][j] = 1. / float(batch_size * (batch_size - 1))
                    index_matrix[i + batch_size][j + batch_size] = 1. / float(batch_size * (batch_size - 1))
            for i in range(batch_size):
                for j in range(batch_size):
                    index_matrix[i][j + batch_size] = -1. / float(batch_size * batch_size)
                    index_matrix[i + batch_size][j] = -1. / float(batch_size * batch_size)
    return index_matrix


class GaussianKernel(nn.Module):
    r"""Gaussian Kernel Matrix

    Gaussian Kernel k is defined by

    .. math::
        k(x_1, x_2) = \exp \left( - \dfrac{\| x_1 - x_2 \|^2}{2\sigma^2} \right)

    where :math:`x_1, x_2 \in R^d` are 1-d tensors.

    Gaussian Kernel Matrix K is defined on input group :math:`X=(x_1, x_2, ..., x_m),`

    .. math::
        K(X)_{i,j} = k(x_i, x_j)

    Also by default, during training this layer keeps running estimates of the
    mean of L2 distances, which are then used to set hyperparameter  :math:`\sigma`.
    Mathematically, the estimation is :math:`\sigma^2 = \dfrac{\alpha}{n^2}\sum_{i,j} \| x_i - x_j \|^2`.
    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and use a fixed :math:`\sigma` instead.

    Args:
        sigma (float, optional): bandwidth :math:`\sigma`. Default: None
        track_running_stats (bool, optional): If ``True``, this module tracks the running mean of :math:`\sigma^2`.
          Otherwise, it won't track such statistics and always uses fix :math:`\sigma^2`. Default: ``True``
        alpha (float, optional): :math:`\alpha` which decides the magnitude of :math:`\sigma^2` when track_running_stats is set to ``True``

    Inputs:
        - X (tensor): input group :math:`X`

    Shape:
        - Inputs: :math:`(minibatch, F)` where F means the dimension of input features.
        - Outputs: :math:`(minibatch, minibatch)`
    """

    def __init__(self, sigma: Optional[float] = None, track_running_stats: Optional[bool] = True,
                 alpha: Optional[float] = 1.):
        super(GaussianKernel, self).__init__()
        assert track_running_stats or sigma is not None
        self.sigma_square = torch.tensor(sigma * sigma) if sigma is not None else None
        self.track_running_stats = track_running_stats
        self.alpha = alpha

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        l2_distance_square = ((X.unsqueeze(0) - X.unsqueeze(1)) ** 2).sum(2)

        if self.track_running_stats:
            self.sigma_square = self.alpha * torch.mean(l2_distance_square.detach())

        return torch.exp(-l2_distance_square / (2 * self.sigma_square))