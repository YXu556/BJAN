import copy
import json
import random
import argparse
import sklearn.metrics
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from joblib import dump
from sklearn.ensemble import RandomForestClassifier

import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from utils import *
from bjan import train_bjan


# path to dataset
DATAPATH = Path(r"data")

def parse_args():
    parser = argparse.ArgumentParser()

    # Setup parameters
    years = ['2019', '2020', '2021', '2022']
    sites = ['Garfield', 'Adams', 'Randolph', 'Harvey', 'Coahoma', 'Haskell', 'All', 'all']
    available_target = [f"{site}_{year}" for year in years for site in sites]
    parser.add_argument('--source', default='All_2019', help='source dataset')
    parser.add_argument('--target', default=None, help='target dataset')
    parser.add_argument('--inseason', default=0, type=int,
                        help='inseason doy (default to 0 - end-of-the-season)')
    parser.add_argument('-c', '--nclasses', type=int, default=16,
                        help='num of classes (default: 16)')
    parser.add_argument('--seed', default=1, type=int,
                        help='random seed')
    parser.add_argument('--year', nargs='+', type=str, default=None,
                        help='year of source dataset')
    parser.add_argument("--val_ratio", default=0.1, type=float,
                        help='Ratio of training data to use for validation. Default 10%.')
    parser.add_argument('--output_dir', default='results/checkpoints',
                        help='logdir to store progress and models (defaults to ./results)')
    parser.add_argument('-s', '--suffix', default=None,
                        help='suffix to output_dir')
    parser.add_argument('-j', '--workers', type=int, default=0,
                        help='number of CPU workers to load the next batch')
    parser.add_argument('-d', '--device', type=str, default=None,
                        help='torch.Device. either "cpu" or "cuda". default will check by torch.cuda.is_available() ')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--eval', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--overall', action='store_true',
                        help='print overall results, if exists')

    # Training configuration
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('-b', '--batchsize', type=int, default=512,
                        help='batch size (number of time series processed simultaneously)')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3,
                        help='optimizer learning rate (default 1e-3)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='optimizer weight_decay (default 1e-4)')
    parser.add_argument('--sequencelength', type=int, default=45,
                        help='Maximum length of time series data (default 70)')
    parser.add_argument('--model', type=str, default="DLTAE",  # choices=['pseltae', 'Transformer'],
                        help='select model architecture.')
    parser.add_argument('--input_dim', default=10, type=int, help='Number of channels of input sample')
    parser.add_argument('--hard_sample', action='store_true',
                        help='evaluate hard sample')

    # Specific parameters for each training method
    subparsers = parser.add_subparsers(dest='method')

    # BJAN
    bjan = subparsers.add_parser('bjan')
    bjan.add_argument('--use_default_optim', default=False, action='store_true', help="whether to use default optimizer")
    bjan.add_argument('--weights', type=str, help='path to source trained model weights')
    bjan.add_argument("--steps_per_epoch", type=int, default=30, help='n steps per epoch')
    bjan.add_argument('--epochs', default=20, type=int, help='Number of epochs per fold')
    bjan.add_argument("--trade_off", default=1, type=float, help='weight of adversarial loss')
    bjan.add_argument('--trade_off_mmd', default=10, type=float,
                        help='the trade-off hyper-parameter for mmd loss')
    bjan.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    bjan.add_argument('--num_k', type=int, default=2, metavar='K',
                        help='how many steps to repeat the generator update')
    bjan.add_argument('--num_ens', type=int, default=2,
                        help='how many equavelent classifiers')

    # MBU
    mbu = subparsers.add_parser('mbu')
    mbu.add_argument('--use_default_optim', default=False, action='store_true', help="whether to use default optimizer")
    mbu.add_argument('--weights', type=str, help='path to source trained model weights')
    mbu.add_argument("--steps_per_epoch", type=int, default=30, help='n steps per epoch')
    mbu.add_argument('--epochs', default=20, type=int, help='Number of epochs per fold')
    mbu.add_argument("--trade_off", default=1, type=float, help='weight of adversarial loss')
    mbu.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    mbu.add_argument('--num_k', type=int, default=2, metavar='K',
                        help='how many steps to repeat the generator update')
    mbu.add_argument('--num_ens', type=int, default=2,
                        help='how many equavelent classifiers')

    args = parser.parse_args()

    # Setup folders based on method and target
    if args.target is None:
        args.target = args.source
    if args.year:
        args.source = args.source.split('_')[0] + '_' + '_'.join(args.year)
    args.output_dir = Path(args.output_dir) / f"{args.method}_{args.model}_{args.source}-{args.target}"
    if args.suffix:
        args.output_dir = args.output_dir.parent / f"{args.output_dir.name}_{args.suffix}"
    if args.inseason:
        if args.inseason <= 12:
            args.output_dir = args.output_dir.parent / f"{args.output_dir.name}_mon{args.inseason}"
            args.inseason = INSEASON_DICT[args.inseason]
        else:
            mon = {v: k for k, v in INSEASON_DICT.items()}[args.inseason]
            args.output_dir = args.output_dir.parent / f"{args.output_dir.name}_mon{mon}"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for fold in range(1, 2):  # 5-fold cross val
        (args.output_dir / f'Fold_{fold}').mkdir(parents=True, exist_ok=True)

    # Setup device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # write training config to file
    if not args.eval and not args.overall:
        with open(str(args.output_dir / 'train_config.json'), 'w') as f:
            args.output_dir = str(args.output_dir)
            f.write(json.dumps(vars(args), indent=4))
            args.output_dir = Path(args.output_dir)
    args.datapath = DATAPATH
    print(args)

    return args


def main(args):

    # load source dataset
    print("pre-load all 5 source datasetsr")
    datasets = []
    for fold in range(1, 6):
        if args.year:
            for year in args.year:
                source = args.source.split('_')[0] + '_' + year
                datasets.append(USCrops(
                    name=source,
                    root=args.datapath,
                    fold=fold,
                    transform=None,
                ))
        else:
            datasets.append(USCrops(
                name=args.source,
                root=args.datapath,
                fold=fold,
                transform=None, ))

    # load target test dataset
    print("=> creating target test dataloader")
    testset = args.target
    testdatasets = []
    for fold in range(1, 6):
        testdatasets.append(USCrops(name=testset, root=args.datapath, fold=fold, transform=None,))
    testdataloader = get_full_target_test_dataloader(targetdatasets=testdatasets, args=args)

    # iter among 5 folds
    for fold in range(1, 2):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

        print(f'Starting fold {fold}... [test on fold {fold}]')
        args.test_fold = fold
        args.fold_dir = args.output_dir / f'Fold_{fold}'
        print(f"Logging results to {args.fold_dir}")

        print("=> creating model '{}'".format(args.model))
        device = torch.device(args.device)
        model = get_model(args.model, args.nclasses, args)
        model.to(device)
        print(f"Initialized {model.modelname}: Total trainable parameters: {get_ntrainparams(model)}")
        model.apply(weight_init)
        best_model_path = args.fold_dir / 'model_best.pth'

        print(f'=> creating sourcedataloader')
        sourcedataloader = get_source_dataloader(datasets, args=args)

        if args.weights:
            pretrained_path = f"{args.weights}/Fold_{args.test_fold}/model_best.pth"
            print("=> loaded checkpoint '{}'".format(str(pretrained_path)))
            model_dict = model.state_dict()
            checkpoint = torch.load(pretrained_path)
            pretrained_weights = checkpoint["model_state"]
            if 'mh' in args.weights and 'mh' not in args.model:
                state_dict = {k: v for k, v in pretrained_weights.items() if 'decoder2' not in k}
            else:
                state_dict = {k.replace('decoder1', 'decoder'): v for k, v in pretrained_weights.items()}
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)

        if not args.eval:
            if args.method == 'bjan':
                train_bjan(model, args, sourcedataloader, testdatasets, device, best_model_path)
            elif args.method == 'mbu':
                train_bjan(model, args, sourcedataloader, testdatasets, device, best_model_path, mbu=True)
            else:
                raise ValueError

        print(f'Restoring best model {best_model_path} weights for testing...')
        checkpoint = torch.load(best_model_path)
        state_dict = checkpoint['model_state']
        criterion = checkpoint['criterion']
        F2 = checkpoint.get('F2', None)
        model.load_state_dict(state_dict)
        test_loss, scores = evaluation(model, criterion, testdataloader, device, num_class=args.nclasses, F2_state_dict=F2, hard_sample=args.hard_sample)
        scores_msg = ", ".join([f"{k}={v:.4f}" for (k, v) in scores.items() if k not in ['class_f1', 'confusion_matrix']])
        print(f"Test results for {args.method}-{args.target}-fold{fold}: \n\n {scores_msg} \n\n")

        scores['epoch'] = 'test'
        scores['testloss'] = test_loss
        conf_mat = scores.pop('confusion_matrix')
        class_f1 = scores.pop('class_f1')

        log_df = pd.DataFrame([scores]).set_index("epoch")
        log_df.to_csv(args.fold_dir / f"testlog_{args.target}.csv")
        np.save(args.fold_dir / f"test_conf_mat_{args.target}.npy", conf_mat)
        np.save(args.fold_dir / f"test_class_f1_{args.target}.npy", class_f1)


def train_supervised(model, args, traindataloader, valdataloader, device, best_model_path):
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    log = list()
    val_loss_min = np.Inf
    print(f"Training {model.modelname} in {args.source}")
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, optimizer, criterion, traindataloader, device)
        val_loss, scores = evaluation(model, criterion, valdataloader, device, args.nclasses)
        scores_msg = ", ".join([f"{k}={v:.4f}" for (k, v) in scores.items() if k not in ['class_f1', 'confusion_matrix']])
        print(f"epoch {epoch + 1}: trainloss={train_loss:.4f}, valloss={val_loss:.4f} " + scores_msg)

        scores["epoch"] = epoch + 1
        scores["trainloss"] = train_loss
        scores["testloss"] = val_loss
        log.append(scores)

        log_df = pd.DataFrame(log).set_index("epoch")
        log_df.to_csv(best_model_path.parent / "trainlog.csv")

        if val_loss < val_loss_min:
            not_improved_count = 0
            print(f'Validation loss improved from {val_loss_min:.4f} to {val_loss:.4f}!')
            val_loss_min = val_loss
            if best_model_path is not None:
                save(model, path=best_model_path, criterion=criterion)
        else:
            not_improved_count += 1
            print(f'Validation loss did not improve from {val_loss_min:.4f} for {not_improved_count} epochs')

        if not_improved_count >= 10:
            print("\nValidation performance didn\'t improve for 10 epochs. Training stops.")
            break

    if epoch == args.epochs - 1:
        print(f"\n{args.epochs} epochs training finished.")


def train_epoch(model, optimizer, criterion, dataloader, device):
    losses = AverageMeter('Loss', ':.4e')
    model.train()
    with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
        for idx, (X, y) in iterator:
            X = recursive_todevice(X, device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(X)
            out = F.log_softmax(logits, dim=-1)

            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            iterator.set_description(f"train loss={loss:.2f}")

            losses.update(loss.item(), X[0].size(0))

    return losses.avg


if __name__ == '__main__':
    args = parse_args()
    if not args.overall:
        main(args)
    overall_performance(args)