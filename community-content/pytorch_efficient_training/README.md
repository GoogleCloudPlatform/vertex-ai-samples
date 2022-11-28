# PyTorch Efficient Training Examples

This folder provides PyTorch efficient training examples using ResNet-50 and ImageNet data.

## Requirements

```shell
pip install --upgrade pip
pip install -r requirements.txt
```

## Description

* resnet.py - Train ResNet-50 on single GPU.
* resnet_dp.py - Train ResNet-50 on single node multiple GPUs with `DataParallel` strategy.
* resnet_ddp.py - Train ResNet-50 on single node multiple GPUs with `DistributedDataParallel` strategy.
* resnet_ddp_wds.py - Train ResNet-50 on single node multiple GPUs with `DistributedDataParallel` strategy and `Webdataset`.
* shard_imagenet.py - Shard ImagNet individual files into `tar` files.

## Benchmark

When run the benchmark on Nvidia T4 GPUs using ImageNet validation dataset, you can get the result like:
Strategy              | Seconds/Epoch - Local Data | Seconds/Epoch - Cloud Data
--------------------- | -------------------------- | --------------------------
On 1 GPU              | 489                        | 804 (2x slower)
On 4 GPUs (DP)        | 157                        | 738 (5x slower)
On 4 GPUs (DDP)       | 134                        | 432 (3x slower)
On 4 GPUs (DDP + WDS) | 131                        | 133 (same performance)

