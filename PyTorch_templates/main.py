import argparse
import sys
import os
import shutil
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from random import sample

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from NNmodel.model import XXNet
from NNmodel.data import get_train_val_test_loader, XXNetDataset

parser = argparse.ArgumentParser(description='XXX Neural Networks')
parser.add_argument('--root', default='./data/', metavar='DATA_ROOT', 
                    help='path to data root dir')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=2, type=int, metavar='N',
                    help='number of total epochs to run (default: 10)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=2, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate (default: '
                    '0.01)')
parser.add_argument('--lr-milestones', default=[50, 100], nargs='+', type=int,
                    metavar='N', help='milestones for scheduler (default: '
                    '[50, 100])')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--train-ratio', default=0.75, type=float, metavar='n/N',
                    help='ratio of training data (default: 0.75)')
parser.add_argument('--val-ratio', default=0.25, type=float, metavar='n/N',
                    help='ratio of validation data (default: 0.25)')
parser.add_argument('--test-ratio', default=0., type=float, metavar='n/N',
                    help='ratio of test data (default: 0.)')
parser.add_argument('--optim', default='SGD', type=str, metavar='SGD',
                    help='choose an optimizer, SGD or Adam, (default: SGD)')

args = parser.parse_args(sys.argv[1:])

args.cuda = not args.disable_cuda and torch.cuda.is_available()

best_mae_error = 1e10

def main():
    global args, best_mae_error

    # load dataset:
    dataset = XXNetDataset(root=args.root)
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset, batch_size=args.batch_size,
        train_ratio=args.train_ratio, num_workers=args.workers,
        val_ratio=args.val_ratio, test_ratio=args.test_ratio,
        pin_memory=args.cuda)

    # obtain target value normalizer
    sample_target = torch.Tensor([dataset[i][1] for i in \
                                 sample(range(len(dataset)), 1000)])
    normalizer = Normalizer(sample_target)

    # build model
    model = XXNet()
    
    # pring number of trainable model parameters
    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
    print('=> number of trainable model parameters: {:d}'.format(trainable_params))

    if args.cuda:
        model.cuda()

    # define loss func and optimizer
    criterion = nn.MSELoss()
    
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), args.lr,
                               weight_decay=args.weight_decay)
    else:
        raise NameError('Only SGD or Adam is allowed as --optim')

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            best_mae_error = checkpoint['best_mae_error']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
#            normalizer.load_state_dict(checkpoint['normalizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # TensorBoard writer
    summary_root = './runs/'
    if not os.path.exists(summary_root):
        os.mkdir(summary_root)
    summary_file = summary_root 
    if os.path.exists(summary_file):
        shutil.rmtree(summary_file)
    writer = SummaryWriter(summary_file)

    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones,
                            gamma=0.1)

    for epoch in range(args.start_epoch, args.start_epoch+args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, normalizer, writer)

        # evaluate on validation set
        mae_error = validate(val_loader, model, criterion, epoch, normalizer, writer)

        if mae_error != mae_error:
            print('Exit due to NaN')
            sys.exit(1)

        scheduler.step()

        # remember the best mae_eror and save checkpoint
        is_best = mae_error < best_mae_error
        best_mae_error = min(mae_error, best_mae_error)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mae_error': best_mae_error,
            'optimizer': optimizer.state_dict(),
#            'normalizer': normalizer.state_dict(),
            'args': vars(args)
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, normalizer, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    running_loss = 0.0
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output = model(input)
        target_normed = normalizer.norm(target)
        loss = criterion(output, target_normed)

        # measure accuracy and record loss
        mae_error = mae(normalizer.denorm(output), target)
        losses.update(loss.item(), target.size(0))
        mae_errors.update(mae_error.item(), target.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # write to TensorBoard
        running_loss += loss.item()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, mae_errors=mae_errors)
                  , flush=True)
            writer.add_scalar('training loss',
                            running_loss / args.print_freq,
                            epoch * len(train_loader) + i)
            running_loss = 0.0


def validate(val_loader, model, criterion, epoch, normalizer, writer, test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        running_loss = 0.0
        for i, (input, target) in enumerate(val_loader):
            # compute output
            output = model(input)
            target_normed = normalizer.norm(target)
            loss = criterion(output, target_normed)
    
            # measure accuracy and record loss
            mae_error = mae(normalizer.denorm(output), target)
            losses.update(loss.item(), target.size(0))
            mae_errors.update(mae_error.item(), target.size(0))
    
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    
            # write to TensorBoard
            running_loss += loss.item()
            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       mae_errors=mae_errors), flush=True)
                writer.add_scalar('validation loss',
                                running_loss / args.print_freq,
                                epoch * len(val_loader) + i)
                running_loss = 0.0
 
    print(' * MAE {mae_errors.avg:.3f}'.format(mae_errors=mae_errors), flush=True)
    return mae_errors.avg


class Normalizer(object):
    """Normalize a Tensor and restore it later. """
    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def save_checkpoint(state, is_best):
    out_root = './checkpoints/'
    if not os.path.exists(out_root):
        os.mkdir(out_root)
    out_dir = out_root
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)
    torch.save(state, out_dir+'/checkpoint.pth.tar')
    if is_best:
        shutil.copyfile(out_dir+'/checkpoint.pth.tar', out_dir+'/model_best.pth.tar')


if __name__ == '__main__':
    main()


