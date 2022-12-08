import argparse
import os
import random
import time
import warnings
import functools

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel as DP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import torchvision.models as models
from torch.utils.data import Subset

from src.utils import Summary, AverageMeter, ProgressMeter, accuracy
from src import dataset, experiment

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

best_acc1 = 0

def parse_args():
    """Create main args."""
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--train_data_path', default='', type=str,
                        help='path to training dataset. For Webdataset, path to set as /path/to/filename-{000000..001000}.tar')
    parser.add_argument('--val_data_path', default='', type=str,
                        help='path to validation dataset. For Webdataset, path to set as /path/to/filename-{000000..001000}.tar'')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--webdataset', action='store_true', 
                        help='use webdataset data loader (default: False)')
    parser.add_argument('--distributed_strategy', default='ddp', type=str, 
                        help='Training distribution strategy. Valid values are: dp, ddp, fsdp')
    # to run the job using torchrun
    parser.add_argument("--hostip", default="localhost", type=str, 
                        help="setting for etcd host ip")
    parser.add_argument("--hostipport", default=2379, type=int, 
                        help="setting for etcd host ip port",)
    parser.add_argument('--data_size', default=50000, type=int, 
                        help='data size for training')
    args = parser.parse_args()
    return args


def set_ddp(args):
    if args.dist_url == "env://" and args.rank == -1:
        args.rank = int(os.environ["RANK"])
    if args.multiprocessing_distributed:
        # For multiprocessing distributed training, rank needs to be the
        # global rank among all the processes
        args.rank = args.rank * args.ngpus_per_node + args.gpu
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    return args.rank


def set_device(args):
    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def prepare_model(args):
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
        model_name = f'{args.arch}-{args.device}{"-wds" if args.webdataset else ""}'
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / args.ngpus_per_node)
                args.workers = int((args.workers + args.ngpus_per_node - 1) / args.ngpus_per_node)
                if args.distributed_strategy == 'ddp':
                    model = DDP(model, device_ids=[args.gpu])
                elif args.distributed_strategy == 'fsdp':
                    # wrap policy.
                    fsdp_auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=100)
                    model = FSDP(model, auto_wrap_policy=fsdp_auto_wrap_policy, device_id=args.gpu)
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                if args.distributed_strategy == 'ddp':
                    model = DDP(model)
                elif args.distributed_strategy == 'fsdp':
                    # wrap policy.
                    fsdp_auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=100)
                    model = FSDP(model, auto_wrap_policy=fsdp_auto_wrap_policy)
            model_name = f'{args.arch}-{args.device}-{args.distributed_strategy}{"-wds" if args.webdataset else ""}'
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        model_name = f'{args.arch}-{args.device}-{args.distributed_strategy}{"-wds" if args.webdataset else ""}'
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = DP(model.features)
            model.cuda()
        else:
            model = DP(model).cuda()
        model_name = f'{args.arch}-{args.device}-dp{"-wds" if args.webdataset else ""}.pt'
    return model, model_name


def resume_from_checkpoint(model, optimizer, scheduler, args):
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        if args.gpu is None:
            checkpoint = torch.load(args.resume)
        elif torch.cuda.is_available():
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        if args.gpu is not None:
            # best_acc1 may be from a checkpoint from a different GPU
            best_acc1 = best_acc1.to(args.gpu)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    return model, optimizer, scheduler


def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if torch.cuda.is_available():
        args.ngpus_per_node = torch.cuda.device_count()
    else:
        args.ngpus_per_node = 1
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = args.ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args,))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1

    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    device = set_device(args)
    args.device = device

    if args.distributed and args.distributed_strategy in ('ddp', 'fsdp'):
        rank = set_ddp(args)
        args.rank = rank
    else:
        args.rank = 0

    # prepare model
    model, model_name = prepare_model(args)
    args.model_name = model_name 
    print(f"=> Model name={model_name}")

    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)    
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # optionally resume from a checkpoint
    if args.resume:
        model, optimizer, scheduler = resume_from_checkpoint(model, optimizer, 
                                                             scheduler, args)

    # prepare dataloader
    # args.train_data_path = os.path.join(args.data, 'train' if not args.webdataset else 'train/imagenet-train-{000000..100000}.tar')
    # args.val_data_path = os.path.join(args.data, 'val' if not args.webdataset else 'val/imagenet-val-{000000..000000}.tar')
        
    if args.webdataset:
        train_loader = dataset.prepare_wds_dataloader('train', gpu, args)
        val_loader = dataset.prepare_wds_dataloader('val', gpu, args)
    else:
        train_loader = dataset.prepare_dataloader('train', gpu, args)
        val_loader = dataset.prepare_dataloader('val', gpu, args)

    # training loop
    trainer = experiment.Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        best_acc1=best_acc1,
        model_name=model_name
    )

    print(args)
    
    if args.evaluate:
        trainer.validate(args)
        return

    trainer.run(args)

if __name__ == '__main__':
    main()