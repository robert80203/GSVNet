from tqdm import tqdm

import os
from util.summarizer import TensorboardSummary
from util.metrics import Evaluator, generate_conf_matrix_torch
from util.saver import Saver
from util.general import init_random_seed, pretty_print, gettime
#from loss.criterion import CriterionDSN
#from trainer.dataset import build_trainer_dataset
from config.cityscapes import cityscapes_config
from config.camvid import camvid_config
from dataset.cityscapes import cityscape_dataset
from dataset.camvid import camvid_dataset
try:
    from apex.parallel import DistributedDataParallel, SyncBatchNorm
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex .")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from torchvision.utils import make_grid
from torchvision.transforms import transforms as standard_transforms

#from inplace_abn import InPlaceABN, InPlaceABNSync

from torch.utils.data import DataLoader
#from model.bn import build_bn


class Trainer(object):
    name = None

    def _img_input_transform(self, args):
        return standard_transforms.ToTensor()

    def _val_img_input_transform(self, args):
        return self._img_input_transform(args)

    def _joint_transform(self, args):
        return None

    def _val_joint_transform(self, args):
        return None

    def _label_transform(self, args):
        return standard_transforms.ToTensor()

    def init_network(self, args, num_classes):
        raise NotImplementedError("Model is not defined. Must implement network.")

    def init_loss(self, args, weights, ignore_index):
        raise NotImplementedError("Model is not defined. Must implement network.")

    def init_optim_scheduler(self, args, params):
        if args.optim == "adam":
            optimizer = optim.Adam(params, weight_decay=0.0001)
        elif args.optim == "sgd":  # SGD
            optimizer = optim.SGD(params, weight_decay=0.0001, momentum=0.9)
        else:
            raise NotImplementedError

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience= 500 ,verbose=True, min_lr = 1e-5, threshold=1e-4)
        return optimizer, scheduler


    def init_saver(self, args):
        saver = Saver(args, self.name)
        pretty_print(args.__dict__)
        saver.save_experiment_config(args.__dict__)
        return saver

    def init_logger(self, args, color_classes):
        return TensorboardSummary( os.path.join(args.log_folder, 'summary', args.dataset, self.name, gettime()), color_classes)

    def init_net_params(self, args):
        train_params = [{'params': self.network.parameters(), 'lr': args.lr}]
        return train_params

    def init_dataset(self, args):
        train_dataset = []
        if args.dataset == 'camvid':
            dataset_used = camvid_dataset
        else:
            dataset_used = cityscape_dataset
        for step in args.multi_step:
            train_dataset.append(
            dataset_used(config = self.dataset_config, mode = 'train', interval = step, \
            label_transform = standard_transforms.ToTensor(), img_transform = self.image_transform, bi_direction = self.bi_direction)
            )
        val_dataset = []
        if args.eval_single:
            #mode='video' or mode='val'
            val_dataset.append(
            dataset_used(config = self.dataset_config, mode = 'val', interval = 0, \
            label_transform = standard_transforms.ToTensor(), img_transform = self.val_image_transform, bi_direction = self.bi_direction)
            )
        for step in args.eval_multi_step:
            val_dataset.append(
            dataset_used(config = self.dataset_config, mode = 'val', interval = step, \
            label_transform = standard_transforms.ToTensor(), img_transform = self.val_image_transform, bi_direction = self.bi_direction)
            )
        return train_dataset, val_dataset
        
    def init_dataloader(self, args, train_dataset, val_dataset, world_size):
        batch_size = args.batch_size
        if self.distributed : batch_size = args.batch_size // world_size
        train_sampler = []
        train_loader = []
        for dataset in train_dataset:
            sampler = None
            if self.distributed: sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_worker, pin_memory=True, sampler=sampler)
            train_sampler.append(sampler)
            train_loader.append(loader)
        val_sampler = []
        val_loader = []
        for dataset in val_dataset:
            sampler = None
            if self.distributed: sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_worker, pin_memory=True, sampler=sampler)
            val_sampler.append(sampler)
            val_loader.append(loader)
        return train_loader, train_sampler, val_loader, val_sampler


    def __init__(self, args, name):
        torch.backends.cudnn.enabled = True

        try:
            print("Bi-direction : " , self.bi_direction)
        except:
            self.bi_direction = False

        self.args = args
        self.name = name

        # Set Cuda
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.distributed = False
        if 'WORLD_SIZE' in os.environ:
            self.distributed = int(os.environ['WORLD_SIZE']) > 1

        if self.distributed:
            self.local_rank = self.args.local_rank
            self.world_size = int(os.environ['WORLD_SIZE'])
            torch.cuda.set_device(self.local_rank)
            print("=> rank %i: enable distributed data parallel" % self.local_rank)
            dist.init_process_group(backend="nccl", init_method='env://')
            args.seed = self.local_rank
        else:
            self.world_size = 1; self.local_rank = 0

        # Initialize random seed
        if args.seed != -1:
            init_random_seed(args.seed)

        # Setup Dataset
        print("=> rank %i: generating dataset %s" % (self.local_rank, args.dataset))

        if args.dataset.find("cityscapes") >= 0:
            self.dataset_config = cityscapes_config()
        elif args.dataset_name == "camvid":
            self.dataset_config = camvid_config()
        else:
            raise NotImplementedError("Trainer dataset %s is not registered into the system" % args.dataset)
        
        # Dataset and model weight configuration
        self.color_classes = self.dataset_config.color_classes
        self.num_classes = self.dataset_config.num_classes
        self.swnet_weight_path = self.dataset_config.swnet_weight_path
        self.bsnet_weight_path = self.dataset_config.bsnet_weight_path
        self.resume_path = self.dataset_config.resume_path
        self.optical_flow_network_path = self.dataset_config.optical_flow_network_path
        self.data_path = self.dataset_config.data_path

        self.image_transform, self.val_image_transform = self._img_input_transform(args), self._val_img_input_transform(args)
        self.train_dataset, self.val_dataset = self.init_dataset(args)
        self.train_loader, self.train_sampler, self.val_loader, self.val_sampler = self.init_dataloader(args, self.train_dataset, self.val_dataset, world_size=self.world_size)

        if self.distributed: dist.barrier()

        # Produce Criterion
        print("=> rank %i: initializing criterion, optim, and scheduler" % self.local_rank)
        #self.class_weights = torch.tensor(self.dataset_config.weights).float().to(self.device)
        #self.loss = self.init_loss(args, self.class_weights, self.dataset_config.ignore_index)
        self.class_weights = torch.tensor(self.dataset_config.weights).float().to(self.device)
        self.loss = self.init_loss(args, self.class_weights, self.dataset_config.ignore_index)
        if self.distributed: dist.barrier()

        print("=> rank %i: initializing network" % self.local_rank)
        
        self.batchnorm = nn.BatchNorm2d#build_bn(self.get_bn(), 'identity')

        self.network = self.init_network(args, self.num_classes)
        train_params = self.init_net_params(args)

        # Resuming checkpoint
        self.best_pred = 0.0
        self.start_epoch = 0
        self.test_best_pred = 0.0
        self.epoch = args.epoch
        
        #if args.resume is not None:
        #    self.load_state(args, reset_all=False)
        if args.resume:
            self.load_state(args)

        if self.distributed:
            self.network = self.network.to(self.device)
            print("=> rank %i: initializing ddp" % self.local_rank)
            self.network = DistributedDataParallel(self.network)
            dist.barrier()
        else:
            self.network = nn.DataParallel(self.network.to(self.device))

        if self.bi_direction:
            if args.separate:
                if self.distributed:
                    self.network_back = self.network_back.to(self.device)
                    print("=> rank %i: initializing ddp" % self.local_rank)
                    self.network_back = DistributedDataParallel(self.network_back)
                    dist.barrier()
                else:
                    self.network_back = nn.DataParallel(self.network_back.to(self.device))
        # Logger and Evaluator
        if not self.distributed or (self.local_rank == 0 and self.distributed):
            print("=> rank %i: initializing experiment directory and parameters" % self.local_rank)
            self.saver = self.init_saver(args)
            print("=> rank %i: initializing tensorboard summarizer and evaluator" % self.local_rank)
            self.logger = self.init_logger(args, self.color_classes)  # with the
        else:
            self.logger, self.evaluator = None, None

        self.evaluator = Evaluator(self.num_classes)  # only evaluate on valid classes (19 classes)

        if self.distributed: dist.barrier()
        # Optim and Scheduler
        self.optimizer, self.scheduler = self.init_optim_scheduler(args, params=train_params)
        if self.distributed: dist.barrier()

    def criterion(self, input, target):
        return self.loss(input, target)

    def all_reduce_tensor(self, tensor, norm=True):
        if self.distributed:
            tensor = tensor.clone()
            dist.all_reduce(tensor, dist.ReduceOp.SUM)
            if norm:
                tensor.div_(self.world_size)
            return tensor
        else:
            if norm:
                return torch.mean(tensor)
            else:
                return tensor

    def all_gather_tensor(self, tensor):
        if self.distributed:
            gather_tensor = [torch.ones_like(tensor)] * self.world_size
            dist.all_gather(gather_tensor, tensor)
            return torch.cat(gather_tensor, dim=0)
        else:
            return tensor
    '''
    def get_bn(self):
        if self.distributed:
            return InPlaceABNSync
        else:
            return InPlaceABN
    '''
    def load_state(self, args):
        return NotImplemented
        '''if not os.path.isfile(args.resume):
            raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
        checkpoint = torch.load(args.resume,  map_location=self.device)
        try:
            warning = self.network.load_state_dict(checkpoint['state_dict'], strict=False)
            print("=> warning: ", warning)
        except RuntimeError:
            print("=> error in importing, using soft import")
            from util.general import init_weight_from_state
            init_weight_from_state(self.network, checkpoint['state_dict'], ['decoder.last_conv.6','dsn.3'])

        if reset_all:
            self.start_epoch = checkpoint['epoch']
            if not args.reset_optimizer:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                self.start_epoch = 0

            if not args.reset_best_pred:
                self.best_pred = checkpoint['best_pred']
                if 'test_best_pred' in checkpoint.keys():
                    self.test_best_pred = checkpoint['test_best_pred']

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))'''

    def save_state(self, epoch, current_pred, is_best, current_test_pred=None):
        dict_mem = {
            'epoch': epoch + 1,
            'state_dict': self.network.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred,
            'current_pred': current_pred,
            'args': self.args
        }
        if current_test_pred is not None:
            dict_mem['test_best_pred'] = self.test_best_pred
            dict_mem['current_test_pred'] = current_test_pred

        self.saver.save_checkpoint(dict_mem, is_best)

    def iter_training(self):
        if self.distributed: dist.barrier()

        torch.backends.cudnn.benchmark = False
        train_loss = 0.0
        self.network.train()

        if not self.distributed or (self.distributed and self.local_rank == 0):
            tbar = tqdm(self.train_loader[0], ncols=100, desc="Training  ")
        else:
            tbar = self.train_loader[0]

        for i, sample in enumerate(tbar):
            self.optimizer.zero_grad()
            image, target, _ = sample
            image = image.cuda(non_blocking=True)
            target = target.long().cuda(non_blocking=True)
            output = self.network(image)
            loss = self.criterion(output, target)

            # Update The Network
            loss.backward()
            self.optimizer.step()
            # Average loss over all GPU
            train_loss += self.all_reduce_tensor(loss, norm=True).item()

            # Average Loss over all batch
            if not self.distributed or (self.distributed and self.local_rank == 0):
                tbar.set_postfix({'Loss': (train_loss / (i + 1))}, refresh=True)

        if not self.distributed or (self.distributed and self.local_rank == 0):
            tbar.close()

        train_loss =  train_loss / len(self.train_loader)

        return train_loss

    def reset_evaluator(self):
        self.evaluator.reset()

    def iter_val(self, loader=None, name="Validation"):
        """
        One epoch of validation
        """
        if self.distributed: dist.barrier()

        torch.backends.cudnn.benchmark = True
        self.reset_evaluator()
        self.network.eval()
        if loader is None:
            loader = self.val_loader[0]

        if not self.distributed or (self.distributed and self.local_rank == 0):
            tbar = tqdm(loader, ncols=100, desc="\r" + name)
            tbar.set_description_str("\r" + name, refresh=True)
        else:
            tbar = loader

        test_loss = 0.0
        image_val = None
        with torch.no_grad():
            for i, sample in enumerate(tbar):
                image, target, idx = sample
                image, target = image.cuda(non_blocking=True), target.long().cuda(non_blocking=True)
                output = self.network(image)
                pred = torch.argmax(output, dim=1)

                # # Gather All Results and Prediction
                conf_matrix  = generate_conf_matrix_torch(target, pred, self.num_classes)
                red_conf_matrix = self.all_reduce_tensor(conf_matrix, norm=False)

                # Add batch sample into evaluator
                self.evaluator.add_conf_matrix(red_conf_matrix)

                if not self.distributed or (self.distributed and self.local_rank == 0):
                    tbar.set_postfix({
                        'mIoU': self.evaluator.Mean_Intersection_over_Union(),
                        'PA': self.evaluator.Pixel_Accuracy(),
                    }, refresh=True)
            if not self.distributed or (self.distributed and self.local_rank == 0):
                tbar.close()

        test_loss = test_loss / len(loader)
        return test_loss, image_val

    def _set_sampler_epoch(self, sampler, epoch):
        if self.distributed:
            if isinstance(sampler, list):
                for i in sampler:
                    i.set_epoch(epoch)
            else:
                sampler.set_epoch(epoch)

    def set_all_sampler_epoch(self, epoch):
        if self.distributed:
            dist.barrier()

        self._set_sampler_epoch(self.train_sampler, epoch)
        for sampler in self.val_sampler:
            self._set_sampler_epoch(sampler, epoch)

        '''if self.test_loader is not None:
            self._set_sampler_epoch(self.test_sampler[0], epoch)
            self._set_sampler_epoch(self.test_sampler[1], epoch)'''


    def run(self):
        """
        Start training the network
        :return:
        """

        print("\r=> start training the network")
        for epoch in range(self.start_epoch, self.start_epoch + self.epoch):
            self.set_all_sampler_epoch(epoch)
            # Reset Values
            if not self.distributed or (self.distributed and self.local_rank == 0):
                print("\r\n\rEpoch %i" % epoch)
            current_pred = -1
            is_best = False

            # Perform Training
            if not self.args.evaluate:
                train_loss = self.iter_training()
                self.scheduler_step(train_loss, epoch)

            # Perform Validation
            if epoch % self.args.eval_interval == (self.args.eval_interval - 1):
                val_loss, val_image = self.iter_val(self.val_loader[0])

                # Logs Validation
                if not self.distributed or (self.distributed and self.local_rank == 0):
                    self.logger.log_val_loss(val_loss, epoch)
                    self.logger.log_evaluation("val",
                                               self.evaluator.Pixel_Accuracy(),
                                               self.evaluator.Pixel_Accuracy_Class(),
                                               self.evaluator.Mean_Intersection_over_Union(),
                                               self.evaluator.Frequency_Weighted_Intersection_over_Union(),
                                               epoch)
                    if val_image is not None:
                        self.logger.log_image('val/result', make_grid(val_image, scale_each=True, nrow=3), epoch)

                current_pred = self.evaluator.Mean_Intersection_over_Union()
                # Perform Testing
                if self.best_pred < current_pred:
                    if (not self.distributed or (self.distributed and self.local_rank == 0)) and not self.args.evaluate:
                        print("=> best val prediction: %.3f" % current_pred)
                    self.best_pred = current_pred
                    is_best = True
                
                current_test_pred = None
                if (is_best or self.args.evaluate) and self.test_loader is not None:
                    test_loss, test_image = self.iter_val(self.test_loader[0], name="Test")
                    current_test_pred = self.evaluator.Mean_Intersection_over_Union()
                    if self.test_best_pred < current_test_pred:
                        self.test_best_pred = current_test_pred
                        if (not self.distributed or (
                                self.distributed and self.local_rank == 0)) and not self.args.evaluate:
                            print("=> test prediction: %.3f" % current_test_pred)
                    self.test_best_pred = test_current_pred

            if (not self.distributed or (self.distributed and self.local_rank == 0)) and not self.args.evaluate:
                self.logger.log_train_loss(train_loss, epoch)
                self.save_state(epoch, current_pred, is_best, current_test_pred=current_test_pred)

        print("=> network training ends.")

    def scheduler_step(self, loss, epoch):
        self.scheduler.step(loss, epoch)

    def multi_load_images(self,file_list_path, data_path,num_of_process = 2):
        with open(file_list_path, "r+") as f:
            text = []
            for line in f:
                text.append(line)
            interval = math.ceil(len(text) / num_of_process)
            processes = []
            return_dicts = []
            manager = Manager()
            for i in range(num_of_process):
                return_dicts.append(manager.dict())
                processes.append(Process(target=self.get_images_from_lines, args=(data_path, text[interval*i:interval*(i+1)],return_dicts[i]) ) )
            for i in range(num_of_process):                
                processes[i].start()
            for i in range(num_of_process):
                processes[i].join()
                global_var.append_image_dict(return_dicts[i])
    def single_load_images(self,file_list_path, data_path):
        text = []
        image_loader = default_loader
        with open(file_list_path, "r+") as f:
            for line in f:
                text.append(line)
        for line in text:
            index = line.find(" ")
            line = line[:index]
            gt_frame_num = int(line[-22:-16])
            for frame in range(gt_frame_num-4,gt_frame_num+1):
                frame_name = line[:-22] + ( "%06d" % (frame) ) + line[-16:]
                frame_path = os.path.join(data_path, frame_name)
                global_var.put_image(frame_path, image_loader(frame_path))

    def get_images_from_lines(self, path, lines, return_dict):
        image_loader = default_loader
        for line in lines:
            index = line.find(" ")
            line = line[:index]
            gt_frame_num = int(line[-22:-16])
            for frame in range(gt_frame_num-4,gt_frame_num+1):
                frame_name = line[:-22] + ( "%06d" % (frame) ) + line[-16:]
                frame_path = os.path.join(path, frame_name)
                return_dict[frame_path] = image_loader(frame_path)