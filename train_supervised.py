import copy
import json
import random
import argparse
import sklearn.metrics
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier

import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from utils import *


# path to dataset
DATAPATH = Path(r"data")


def parse_args():
    parser = argparse.ArgumentParser()

    # Setup parameters
    parser.add_argument('--source', default='All_2019', help='source dataset')
    parser.add_argument('--target', default='All_2019', help='test dataset')
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
                        help='logdir to store progress and models (defaults to ./checkpoints)')
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
    parser.add_argument('--save_fea', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--hard_sample', action='store_true',
                        help='evaluate hard sample')
    parser.add_argument('--overall', action='store_true',
                        help='print overall results, if exists')
    parser.add_argument('--weights', type=str, help='restore specific model to eval')

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
                        help='Maximum length of time series data (default 45)')
    parser.add_argument('--model', type=str, default="DLTAE",
                        help='select model architecture.')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout rate.')
    parser.add_argument('--input_dim', default=10, type=int, help='Number of channels of input sample')

    args = parser.parse_args()

    if args.year:
        args.source = args.source.split('_')[0] + '_' + '_'.join(args.year)
    # Setup folders based on method and target
    args.output_dir = Path(args.output_dir) / f"{args.model}_{args.source}"
    if args.suffix:
        args.output_dir = args.output_dir.parent / f"{args.output_dir.name}_{args.suffix}"

    if args.inseason:
        if args.inseason <= 12:
            args.output_dir = args.output_dir.parent / f"{args.output_dir.name}_mon{args.inseason}"
            args.inseason = INSEASON_DICT[args.inseason]
        else:
            mon = {v: k for k, v in INSEASON_DICT.items()}[args.inseason]
            args.output_dir = args.output_dir.parent / f"{args.output_dir.name}_mon{mon}"

    for fold in range(1, 2):  # 5-fold cross val
        (args.output_dir / f'Fold_{fold}').mkdir(parents=True, exist_ok=True)

    # Setup device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # write training config to file
    if not args.eval:
        with open(str(args.output_dir / 'train_config.json'), 'w') as f:
            args.output_dir = str(args.output_dir)
            f.write(json.dumps(vars(args), indent=4))
            args.output_dir = Path(args.output_dir)
    args.datapath = DATAPATH
    print(args)

    return args


def main(args):
    # load supervised datasets
    print('pre-load all 5 source datasets')
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
                    transform=None,
                ))

    # load test datasets
    print('pre-load all 5 target datasets')
    testset = args.target
    testdatasets = []
    for fold in range(1, 6):
        testdatasets.append(USCrops(name=testset, root=args.datapath, fold=fold, transform=None,))

    # iter among 5 folds
    feas, ys = list(), list()
    scores_percents = list()
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

        if args.model in ['RF', 'rf']:
            best_model_path = args.fold_dir / 'model_best.joblib'
            traindataset, testdataset = get_rf_data(datasets, testdatasets, args)
            if not args.eval:
                X_train, y_train = traindataset
                print('training Random Forest...')
                model.fit(X_train, y_train)
                dump(model, best_model_path)
            print('Restoring best model weights for testing...')
            model = load(best_model_path)
            X_test, y_test= testdataset

            y_pred = model.predict(X_test)
            scores = accuracy(y_pred, y_test, args.nclasses)

            scores_msg = ", ".join(
                [f"{k}={v:.4f}" for (k, v) in scores.items() if k not in ['class_f1', 'confusion_matrix']])
            print(f"Test results : \n\n {scores_msg} \n\n")

            scores['epoch'] = 'test'
            conf_mat = scores.pop('confusion_matrix')
            class_f1 = scores.pop('class_f1')

            log_df = pd.DataFrame([scores]).set_index("epoch")
            log_df.to_csv(args.fold_dir / f"testlog_{testset}.csv")
            np.save(args.fold_dir / f"test_conf_mat_{testset}.npy", conf_mat)
            np.save(args.fold_dir / f"test_class_f1_{testset}.npy", class_f1)

            continue

        # load source dataset
        print("=> creating train/val dataloader")
        traindataloader, valdataloader = get_supervised_dataloader(datasets, args=args)

        model.apply(weight_init)
        model.to(device)
        best_model_path = args.fold_dir / 'model_best.pth'  # end todo

        if not args.eval:
            train_supervised(model, args, traindataloader, valdataloader, device, best_model_path)

        print('Restoring best model weights for testing...')
        if args.weights is not None:
            best_model_path = Path(args.weights) / f'Fold_{fold}' / 'model_best.pth'
        checkpoint = torch.load(best_model_path)
        state_dict = {k.replace('decoder1', 'decoder'): v for k, v in checkpoint['model_state'].items()}# checkpoint['model_state']
        criterion = checkpoint['criterion']
        F2 = checkpoint.get('F2', None)
        model.load_state_dict(state_dict)

        testdataloader = get_full_target_test_dataloader(testdatasets, args)

        if args.save_fea:
            test_loss, scores, fea, y = evaluation(model, criterion, testdataloader, device,
                                                   num_class=args.nclasses, save_fea=args.save_fea, F2_state_dict=F2)
            feas.append(fea)
            ys.append(y)
        elif args.hard_sample:
            test_loss, scores, scores_percent = evaluation(model, criterion, testdataloader, device, num_class=args.nclasses,
                                           hard_sample = args.hard_sample, F2_state_dict=F2)
            scores_percents.append(scores_percent)
        else:
            test_loss, scores = evaluation(model, criterion, testdataloader, device, num_class=args.nclasses, F2_state_dict=F2)

        scores_msg = ", ".join([f"{k}={v:.4f}" for (k, v) in scores.items() if k not in ['class_f1', 'confusion_matrix']])
        print(f"Test results : \n\n {scores_msg} \n\n")

        scores['epoch'] = 'test'
        scores['testloss'] = test_loss
        conf_mat = scores.pop('confusion_matrix')
        class_f1 = scores.pop('class_f1')

        log_df = pd.DataFrame([scores]).set_index("epoch")
        args.fold_dir = best_model_path.parent
        args.fold_dir.mkdir(exist_ok=True, parents=True)

        if args.inseason:
            if args.inseason <= 12:
                mon = args.inseason
            else:
                mon = {v: k for k, v in INSEASON_DICT.items()}[args.inseason]

        if args.inseason and f"mon{mon}" not in str(args.fold_dir):
            log_df.to_csv(args.fold_dir / f"testlog_{testset}_mon{mon}.csv")
            np.save(args.fold_dir / f"test_conf_mat_{testset}_mon{mon}.npy", conf_mat)
            np.save(args.fold_dir / f"test_class_f1_{testset}_mon{mon}.npy", class_f1)
        else:
            log_df.to_csv(args.fold_dir / f"testlog_{testset}.csv")
            np.save(args.fold_dir / f"test_conf_mat_{testset}.npy", conf_mat)
            np.save(args.fold_dir / f"test_class_f1_{testset}.npy", class_f1)

    if args.save_fea:
        ys = np.hstack(ys).reshape(-1, 1)
        feas = np.vstack(feas).reshape(ys.shape[0], -1)
        fea_y = np.hstack([feas, ys])
        fea_out_dir = Path('results/features')
        fea_out_dir.mkdir(parents=True, exist_ok=True)
        if args.weights is not None:
            fea_out_fn = fea_out_dir / (args.weights.split('/')[-1]+'_to_' + args.target + '.npy')
        else:
            fea_out_fn = fea_out_dir / (best_model_path.parts[-3] + '_to_' + args.target + '.npy')
        np.save(fea_out_fn, fea_y)

    if args.hard_sample:
        avg_oa_per = np.hstack([s['oa'] for ss in scores_percents for s in ss]).reshape(5, 10).mean(0)
        for i, oa in enumerate(avg_oa_per):
            print(f"{(i + 1) * 10:d}%: {oa:.4f}")


def train_supervised(model, args, traindataloader, valdataloader, device, best_model_path):
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)

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
            if model.modelname.startswith('MH'):
                logits, logits2, fea = model(X, return_feats=True)
                loss = criterion(logits, y) + criterion(logits2, y)
            else:
                logits, fea = model(X, return_feats=True)
                loss = criterion(logits, y)
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