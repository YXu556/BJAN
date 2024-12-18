"""
Credits to  github.com/clcarwin/focal_loss_pytorch
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from collections import defaultdict
from joblib import dump
from sklearn.ensemble import RandomForestClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

from datasets import *
from models import *


INSEASON_DICT = {4: 121, 5:152, 6:182, 7: 213, 8: 244, 9: 274, 10: 305, 11: 335}

# -------------------------------------- #
#           train/val utils              #
# -------------------------------------- #
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def MMD(x, y, kernel='rbf', device='cuda'):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz  # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a ** 2 * (a ** 2 + dxx) ** -1
            YY += a ** 2 * (a ** 2 + dyy) ** -1
            XY += a ** 2 * (a ** 2 + dxy) ** -1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)  # a is sigma^2 in guassian kernels
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2. * XY)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, num_classes=21):
    num = target.shape[0]

    confusion_matrix = get_confusion_matrix(output, target, num_classes)
    TP = confusion_matrix.diagonal()
    FP = confusion_matrix.sum(1) - TP
    FN = confusion_matrix.sum(0) - TP

    po = TP.sum() / num
    pe = (confusion_matrix.sum(0) * confusion_matrix.sum(1)).sum() / num ** 2
    if pe == 1:
        kappa = 1
    else:
        kappa = (po - pe) / (1 - pe)

    p = TP / (TP + FP + 1e-12)
    r = TP / (TP + FN + 1e-12)
    f1 = 2 * p * r / (p + r + 1e-12)

    oa = po
    kappa = kappa
    macro_f1 = f1.mean()
    weight = confusion_matrix.sum(0) / confusion_matrix.sum()
    weighted_f1 = (weight * f1).sum()
    class_f1 = f1

    return dict(
        oa=oa,
        kappa=kappa,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        class_f1=class_f1,
        confusion_matrix=confusion_matrix
    )


def get_confusion_matrix(y_pred, y_true, num_classes=21):
    idx = y_pred * num_classes + y_true
    return np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


@torch.no_grad()
def acc(outputs, targets):
    preds = outputs.argmax(dim=1)
    return preds.eq(targets).float().mean().item()


def entropy(predictions: torch.Tensor, reduction="none") -> torch.Tensor:
    epsilon = 1e-5
    prob = nn.Softmax(-1)(predictions)
    H = -prob * torch.log(prob + epsilon)
    H = H.sum(dim=1)
    if reduction == "mean":
        return H.mean()
    else:
        return H


def validation(best_f1, model, criterion, dataloader, device, num_class=16,
               log=None, epoch=None, best_model_path=None, train_loss=None):
    val_loss, val_scores = evaluation(model, criterion, dataloader, device, num_class)
    val_acc, val_f1 = val_scores['oa'], val_scores['macro_f1']
    print(f"Validation result: loss={val_loss:.4f}, acc={val_acc:.4f}, f1={val_f1:.4f}")
    if val_f1 > best_f1:
        print(f'Validation F1 improved from {best_f1:.4f} to {val_f1:.4f}!')
        best_f1 = val_f1
    else:
        print(f'Validation F1 did not improve from {best_f1:.4f}.')
    if log is not None:
        val_scores["epoch"] = epoch + 1
        val_scores.update(train_loss)
        log.append(val_scores)
        log_df = pd.DataFrame(log).set_index("epoch")
        log_df.to_csv(best_model_path.parent / "trainlog.csv")

    return best_f1


def evaluation(model, criterion, dataloader, device, num_class=16, F2_state_dict=None, save_fea=False, hard_sample=False):
    losses = AverageMeter('Loss', ':.4e')
    if F2_state_dict is not None:
        F2 = deepcopy(model.decoder)
        F2.load_state_dict(F2_state_dict)
        F2.eval()
    model.eval()
    with torch.no_grad():
        y_true_list = list()
        y_pred_list = list()
        y_pred_list2 = list()
        fea_list = list()
        loss_list = list()
        with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
            for idx, (X, y) in iterator:
                X = recursive_todevice(X, device)
                y = y.to(device)

                if F2_state_dict is not None:
                    logits, fea = model(X, return_feats=True)
                    logits2 = F2(fea)
                    loss = criterion(logits, y)
                else:
                    logits, fea = model(X, return_feats=True)
                    loss = criterion(logits, y)
                iterator.set_description(f"test loss={loss:.2f}")
                losses.update(loss.item(), X[0].size(0))

                y_true_list.append(y)
                y_pred_list.append(logits.argmax(-1))
                if F2_state_dict is not None:
                    y_pred_list2.append(logits2.argmax(-1))
                if isinstance(fea, tuple):
                    fea = fea[-1]
                fea_list.append(fea)
                # if hard_sample:
                #     loss_list.append(F.cross_entropy(logits2, y, reduction='none'))
    y_true = torch.cat(y_true_list).cpu().numpy()
    y_pred = torch.cat(y_pred_list).cpu().numpy()
    fea = torch.cat(fea_list).cpu().numpy()
    scores = accuracy(y_pred, y_true, num_class)
    if len(y_pred_list2) != 0:
        y_pred2 = torch.cat(y_pred_list2).cpu().numpy()
        scores2 = accuracy(y_pred2, y_true, num_class)

    if len(loss_list) != 0:
        loss_array = torch.cat(loss_list).cpu().numpy()
        percent_scores = list()
        for i in range(10):
            num = loss_array.shape[0] * (i + 1) // 10
            index = np.argsort(loss_array)[-num:]
            scores_tmp = accuracy(y_pred2[index], y_true[index], num_class)
            percent_scores.append(scores_tmp)

    if save_fea:
        return losses.avg, scores, fea, y_true
    elif hard_sample:
        return losses.avg, scores, 0#, percent_scores
    else:
        return losses.avg, scores


def cycle(iterable):  # Don't use itertools.cycle, as it repeats the same shuffle
    while True:
        for x in iterable:
            yield x


def cat_samples(Xs):
    out = [torch.cat([X[i] for X in Xs]) for i in range(len(Xs[0]))]
    return out


def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        return [recursive_todevice(c, device) for c in x]


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.learning_rate
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save(model, path="model.pth", **kwargs):
    print(f"saving model to {str(path)}\n")
    model_state = model.state_dict()
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    torch.save(dict(model_state=model_state, **kwargs), path)


def overall_performance(args):
    overall_metrics = defaultdict(list)

    cms = []
    for fold in range(1, 2):
        fold_dir = args.output_dir / f'Fold_{fold}'
        if fold_dir.exists():
            test_metrics = pd.read_csv(fold_dir / f'testlog_{args.target}.csv').iloc[0].to_dict()
            for metric, value in test_metrics.items():
                overall_metrics[metric].append(value)
            cm = np.load(fold_dir / f'test_conf_mat_{args.target}.npy')
            cms.append(cm)

    print(f'Overall result across 5 trials:')
    for metric, values in overall_metrics.items():
        values = np.array(values)
        if isinstance(values[0], (str)) or np.any(np.isnan(values)):
            continue
        if 'loss' in metric or 'f1' in metric:
            print(f"{metric}: {np.mean(values):.4f}")#±{np.std(values):.4f}")
        else:
            values *= 100
            print(f"{metric}: {np.mean(values):.2f}")#±{np.std(values):.2f}")


# -------------------------------------- #
#              data utils                #
# -------------------------------------- #
def get_target_test_dataloader(target, test_fold, datapath, args):
    year = target.split('_')[1]
    test_transform = transforms.Compose([
        RandomSampleTimeSteps(args.sequencelength),
        Normalize(year),
        ToTensor(),
    ])

    testdataset = USCrops(
        name=target,
        root=datapath,
        fold=test_fold,
        transform=test_transform,
    )

    testdataloader = torch.utils.data.DataLoader(
        testdataset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    print(f"evaluation dataset:", target)
    print(f"test target data: {len(testdataset)} ({len(testdataloader)} batches)")

    return testdataloader


def get_full_target_test_dataloader(targetdatasets, args):
    year = args.target.split('_')[1]
    test_transform = transforms.Compose([
        RandomSampleTimeSteps(args.sequencelength, args.inseason),
        Normalize(year),
        ToTensor(),
    ])

    testdataset = CropConcatDataset(targetdatasets)
    testdataset.update_transform(test_transform)

    testdataloader = torch.utils.data.DataLoader(
        testdataset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    print(f"evaluation dataset:", args.target)
    print(f"test target data: {len(testdataset)} ({len(testdataloader)} batches)")

    return testdataloader


def get_source_dataloader(datasets, args):
    year = args.source.split('_')[1]
    train_transform = transforms.Compose([
        RandomTempRemoval(),  # temp
        RandomTempShift(),  # temp
        RandomSampleTimeSteps(args.sequencelength, inseason=args.inseason),  # todo
        Normalize(year),
        RandomAddNoise(),  # spec
        ToTensor(),
    ])
    if args.method == 'timematch':
        train_transform = transforms.Compose([
            RandomTempRemoval(),  # temp
            RandomSampleTimeSteps(args.sequencelength, inseason=args.inseason),  # todo
            Normalize(year),
            RandomAddNoise(),  # spec
            ToTensor(),
        ])

    if args.year:
        num_year = len(args.year)
        val_i = args.test_fold - 1
        train_is = list(range(5*num_year))
        del train_is[val_i: val_i + num_year]
    else:
        val_i = args.test_fold - 1
        train_is = list(range(5))
        train_is.pop(val_i)
    trainvalsets = [deepcopy(datasets[i]) for i in train_is]
    sourcedataset = CropConcatDataset(trainvalsets)
    sourcedataset.update_transform(train_transform)

    sourcedataloader = torch.utils.data.DataLoader(
        sourcedataset, batch_size=args.batchsize,
        num_workers=args.workers, pin_memory=True,
        shuffle=True, drop_last=True,)

    print(f"source dataset:", args.source)
    print(f"size of source data: {len(sourcedataset)} ({len(sourcedataloader)} batches)")

    return sourcedataloader


def get_supervised_dataloader(datasets, args):
    year = args.source.split('_')[1]
    train_transform = transforms.Compose([
        RandomTempRemoval(),  # temp
        RandomTempShift(),  # temp
        RandomSampleTimeSteps(args.sequencelength),
        Normalize(year),
        RandomAddNoise(),  # spec
        ToTensor(),
    ])

    test_transform = transforms.Compose([
        RandomSampleTimeSteps(args.sequencelength),
        Normalize(year),
        ToTensor(),
    ])

    if args.year:
        num_year = len(args.year)
        val_i = args.test_fold - 1
        train_is = list(range(5*num_year))
        del train_is[val_i: val_i + num_year]
    else:
        val_i = args.test_fold - 1
        train_is = list(range(5))
        train_is.pop(val_i)
    trainvalsets = [deepcopy(datasets[i]) for i in train_is]
    trainvaldataset = CropConcatDataset(trainvalsets)

    # random sample val_dataset according to val_ratio
    num_train = len(trainvaldataset)
    indices = list(range(num_train))
    num_val = int(num_train * args.val_ratio)
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[num_val:], indices[:num_val]

    traindataset = Subset(deepcopy(trainvaldataset), train_idx)
    traindataset.update_transform(train_transform)
    valdataset = Subset(deepcopy(trainvaldataset), valid_idx)
    valdataset.update_transform(test_transform)

    train_loader = torch.utils.data.DataLoader(
        traindataset, batch_size=args.batchsize, drop_last=True,
        num_workers=args.workers, pin_memory=True,)

    val_loader = torch.utils.data.DataLoader(
        valdataset, batch_size=args.batchsize,
        num_workers=args.workers)

    print(f"supervised dataset:", args.source)
    print(f"size of train data: {len(traindataset)} ({len(train_loader)} batches)")
    print(f"size of val data: {len(valdataset)} ({len(val_loader)} batches)")

    return train_loader, val_loader


def get_rf_data(datasets, testsets, args):
    print('Load RF Dataset')
    train_transform = transforms.Compose([
        RandomTempRemoval(),  # temp
        RandomTempShift(),  # temp
        RandomSampleTimeSteps(args.sequencelength, inseason=args.inseason, rf=True),
        Normalize(args.source.split('_')[1]),
        RandomAddNoise(),  # spec
        ToTensor(),
    ])
    test_transform = transforms.Compose([
        RandomSampleTimeSteps(args.sequencelength, inseason=args.inseason, rf=True),
        Normalize(args.target.split('_')[1]),
        ToTensor(),
    ])

    if args.year:
        num_year = len(args.year)
        val_i = args.test_fold - 1
        train_is = list(range(5*num_year))
        del train_is[val_i: val_i + num_year]
    else:
        val_i = args.test_fold - 1
        train_is = list(range(5))
        train_is.pop(val_i)

    trainsets = [deepcopy(datasets[i]) for i in train_is]
    traindataset = CropConcatDataset(trainsets)
    testdataset = CropConcatDataset(testsets)

    X_train_np = traindataset.X_list
    y_train = traindataset.y_list
    X_train = list()
    for i, X in enumerate(X_train_np):
        sample = {
            'x': X[:, :10],
            'doy': X[:, -1],
            'y': y_train[i]
        }
        X_train.append(train_transform(sample)['x'].flatten())
    X_train = torch.stack(X_train)

    X_test_np = testdataset.X_list
    y_test = testdataset.y_list
    X_test = list()
    for i, X in enumerate(X_test_np):
        sample = {
            'x': X[:, :10],
            'doy': X[:, -1],
            'y': y_test[i]
        }
        X_test.append(test_transform(sample)['x'].flatten())
    X_test = torch.stack(X_test)

    print(f"supervised dataset:", args.source)
    print(f"size of train data: {len(traindataset)}")

    print(f"test dataset:", args.target)
    print(f"size of test data: {len(testdataset)}")

    return (X_train, y_train), (X_test, y_test)


def get_ntrainparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -------------------------------------- #
#              model utils                #
# -------------------------------------- #
def get_model(modelname, num_classes, args):
    modelname = modelname.lower()  # make case invariant
    if modelname == 'dltae':
        model = DLTAE(input_dim=args.input_dim, num_classes=num_classes, max_seq_len=args.sequencelength)
    elif modelname == 'rf':
        model = RandomForestClassifier(n_estimators=1000)#, max_depth=20)
    else:
        raise ValueError("invalid model argument.")

    return model


class ImageClassifierHead(nn.Module):
    r"""Classifier Head for MCD.

    Args:
        in_features (int): Dimension of input features
        num_classes (int): Number of classes
        bottleneck_dim (int, optional): Feature dimension of the bottleneck layer. Default: 1024

    Shape:
        - Inputs: :math:`(minibatch, F)` where F = `in_features`.
        - Output: :math:`(minibatch, C)` where C = `num_classes`.
    """

    def __init__(self, in_features, num_classes, bottleneck_dim=1024, pool_layer=None):
        super(ImageClassifierHead, self).__init__()
        self.num_classes = num_classes
        if pool_layer is None:
            self.pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
        else:
            self.pool_layer = pool_layer

        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, num_classes)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool_layer(inputs))



def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()
