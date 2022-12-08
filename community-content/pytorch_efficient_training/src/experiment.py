import time
import shutil

import torch
import torchmetrics

from src.utils import Summary, AverageMeter, ProgressMeter, accuracy

class Trainer():

    def __init__(self,
                 model,
                 optimizer,
                 criterion,
                 scheduler,
                 train_loader,
                 val_loader,
                 best_acc1,
                 model_name):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.best_acc1 = best_acc1
        self.model_name = model_name 

    def train(self, epoch, args):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        num_batches = int(args.data_size / (args.batch_size * args.ngpus_per_node))
        progress = ProgressMeter(
            num_batches if args.webdataset else len(self.train_loader) ,
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))

        # switch to train mode
        self.model.train()

        end = time.time()
        for i, (images, target) in enumerate(self.train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            # move data to the same device as model
            images = images.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)

            # compute output
            output = self.model(images)
            loss = self.criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i + 1)


    def validate(self, args):

        def run_validate(loader, base_progress=0):
            with torch.no_grad():
                end = time.time()
                for i, (images, target) in enumerate(loader):
                    i = base_progress + i
                    if args.gpu is not None and torch.cuda.is_available():
                        images = images.cuda(args.gpu, non_blocking=True)
                    if torch.cuda.is_available():
                        target = target.cuda(args.gpu, non_blocking=True)

                    # compute output
                    output = self.model(images)
                    loss = self.criterion(output, target)

                    # measure accuracy and record loss
                    acc1, acc5 = accuracy(output, target, topk=(1, 5))
                    losses.update(loss.item(), images.size(0))
                    top1.update(acc1[0], images.size(0))
                    top5.update(acc5[0], images.size(0))

                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()

                    if i % args.print_freq == 0:
                        progress.display(i + 1)

        batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
        losses = AverageMeter('Loss', ':.4e', Summary.NONE)
        top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
        top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
       
        if args.webdataset:
            num_batches = int(args.data_size / (args.batch_size * args.ngpus_per_node))
        else:
            num_batches = len(self.val_loader) + (args.distributed and (len(self.val_loader.sampler) * args.world_size < len(self.val_loader.dataset)))

        progress = ProgressMeter(
            num_batches,
            [batch_time, losses, top1, top5],
            prefix='Test: ')

        # switch to evaluate mode
        self.model.eval()

        run_validate(self.val_loader)
        if args.distributed:
            top1.all_reduce()
            top5.all_reduce()

        if not args.webdataset:
            if args.distributed and (len(self.val_loader.sampler) * args.world_size < len(self.val_loader.dataset)):
                aux_val_dataset = Subset(self.val_loader.dataset,
                                         range(len(self.val_loader.sampler) * args.world_size, len(self.val_loader.dataset)))
                aux_val_loader = torch.utils.data.DataLoader(
                    aux_val_dataset, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True)
                run_validate(aux_val_loader, len(self.val_loader))

        progress.display_summary()

        return top1.avg


    def save_checkpoint(self, state, is_best, filename='checkpoint.pt'):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.pt')

    def run(self, args):
        best_acc1 = self.best_acc1
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed and not args.webdataset:
                self.train_loader.sampler.set_epoch(epoch)

            # train for one epoch
            start = time.time()
            self.train(epoch, args)
            end = time.time()
            if args.rank % args.ngpus_per_node == 0:
                print(f'=> [Epoch {epoch}]: Training finished in {(end - start):>0.3f} seconds')

            # evaluate on validation set
            start = time.time()
            acc1 = self.validate(args)

            self.scheduler.step()
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            end = time.time()
            if args.rank % args.ngpus_per_node == 0:
                print(f'=> [Epoch {epoch}]:Evaluation finished in {(end - start):>0.3f} seconds')

            # save checkpoint
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank % args.ngpus_per_node == 0):
                self.save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': self.model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer' : self.optimizer.state_dict(),
                    'scheduler' : self.scheduler.state_dict()
                }, is_best, filename=f"{self.model_name}.pt")