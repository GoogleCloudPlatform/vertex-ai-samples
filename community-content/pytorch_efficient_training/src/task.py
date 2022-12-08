# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.\n",
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import shutil

import torch

from torch import distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel as DP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet50

from src import dataset, experiment

def distributed_is_initialized(run_distributed):
    print(f"run_distributed={run_distributed}")
    print(f"distributed.is_available={dist.is_available()}")
    print(f"distributed.is_initialized={dist.is_initialized()}")
    if (run_distributed and 
        dist.is_available() and 
        dist.is_initialized()):
        return True
    return False


def ddp_setup(rank, world_size):
    print(f'Initiating process {rank}')
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    print(f'[{os.getpid()}]: '
          f'world_size = {dist.get_world_size()}, '
          f'rank = {dist.get_rank()}, '
          f'backend={dist.get_backend()} \n', end='')


def makedirs(_dir):
    if os.path.exists(_dir) and os.path.isdir(_dir):
        shutil.rmtree(_dir)
    os.makedirs(_dir)
    return

def get_dir(_dir, local_dir):
    gs_prefix = 'gs://'
    gcsfuse_prefix = '/gcs/'
    local_dir = './tmp/model'
    _dir = _dir or local_dir
    if _dir and _dir.startswith(gs_prefix):
        _dir = _dir.replace(gs_prefix, gcsfuse_prefix)
    makedirs(_dir)
    return _dir

def parse_args():
    """Create main args."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpus', default=4, type=int, help='number of gpus to use')
    parser.add_argument('--epochs', default=1, type=int, help='number of total epochs to run')
    parser.add_argument('--dataloader_num_workers', default=2, type=int, help='number of workders for dataloader')
    parser.add_argument('--train_data_path', default='', type=str, help='path to training data')
    parser.add_argument('--train_batch_size', default=32, type=int, help='batch size for training per gpu')
    parser.add_argument('--train_data_size', default=50000, type=int, help='data size for training')
    parser.add_argument('--eval_data_path', default='', type=str, help='path to evaluation data')
    parser.add_argument('--eval_batch_size', default=32, type=int, help='batch size for evaluation per gpu')
    parser.add_argument('--eval_data_size', default=50000, type=int, help='data size for evaluation')
    parser.add_argument('--webdataset', action='store_true', help='use webdataset data loader (default: False)')
    parser.add_argument('--distributed', action='store_true', help='use distributed training. (default: False)')
    parser.add_argument('--distributed_strategy', default='ddp', type=str, help='Training distribution strategy. Valid values are: dp, ddp, fsdp')

    # Using environment variables for Cloud Storage directories
    # see more details in https://cloud.google.com/vertex-ai/docs/training/code-requirements
    parser.add_argument('--model-dir', default=os.getenv('AIP_MODEL_DIR'), type=str, help='Cloud Storage URI to write model artifacts')
    parser.add_argument('--tensorboard-log-dir', default=os.getenv('AIP_TENSORBOARD_LOG_DIR'), type=str, help='Cloud Storage URI to write to TensorBoard')
    parser.add_argument('--checkpoint-dir', default=os.getenv('AIP_CHECKPOINT_DIR'), type=str, help='Cloud Storage URI to save checkpoints')

    args = parser.parse_args()
    return args


def worker(rank, args):
    """Run training and evaluation."""
    # Init process group.
    if args.distributed and args.distributed_strategy in ('ddp', 'fsdp'):
        ddp_setup(rank, args.gpus)

    # check if 
    is_dist = distributed_is_initialized(args.distributed)
    print(f"is_dist={is_dist}")
    


    # set model artifact directory
    model_dir = get_dir(args.model_dir , './tmp/model')
    # set tensorboard log directory
    tensorboard_log_dir = get_dir(args.tensorboard_log_dir , './tmp/logs')
    writer = SummaryWriter(tensorboard_log_dir)
    # set checkpoint directory and patj
    checkpoint_dir = get_dir(args.checkpoint_dir , './tmp/checkpoints')
    if rank == 0:
        print(f'Checkpoints will be saved to {checkpoint_dir}')
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pt')
    print(f'checkpoint_path is {checkpoint_path}')
      
    # Create model
    model = resnet50(weights=None)
      
    if is_dist:
        model_name = f'resnet-{args.device}-{args.distributed_strategy}{"-wds" if args.webdataset else ""}.pt'
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
        model.to(args.device)

        if args.distributed_strategy == 'ddp':
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[rank])
        elif args.distributed_strategy == 'dp':
            model = DP(model)
        elif args.distributed_strategy == 'fsdp':
            # wrap policy.
            fsdp_auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=100)
            model = FSDP(model, auto_wrap_policy=fsdp_auto_wrap_policy)
    else:
        model_name = f'resnet-{args.device}{"-wds" if args.webdataset else ""}.pt'
        model = model.to(args.device)

    # get data loaders
    if args.webdataset:
        train_dataloader = dataset.prepare_wds_dataloader(rank, args, 'train')
        eval_dataloader = dataset.prepare_wds_dataloader(rank, args, 'eval')
    else:
        train_dataloader = dataset.prepare_dataloader(rank, args, 'train', is_dist)
        eval_dataloader = dataset.prepare_dataloader(rank, args, 'eval', is_dist)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), 0.1)

    trainer = experiment.Trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        test_dataloader=eval_dataloader,
        device=args.device,
        model_name=model_name,
        checkpoint_path=checkpoint_path)
    trainer.fit(args.epochs, rank, writer)

    if model_dir == local_model_dir:
        trainer.save(model_dir)
        print(f'Model is saved to {model_dir}')

    print(f'Tensorboard logs are saved to: {tensorboard_log_dir}')
    writer.close()

    if is_dist:
        dist.destroy_process_group()
      
    if rank == 0:
        print('Done')


def main():
    args = parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(args)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8888'

    print(f'Launch job on {args.gpus} GPUs with DDP')
    mp.spawn(worker, nprocs=args.gpus, args=(args,))


if __name__ == '__main__':
    main()