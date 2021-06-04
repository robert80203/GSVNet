import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torchvision.transforms import transforms as standard_transforms
from torchvision.utils import make_grid
import numpy as np
from tqdm import tqdm
import os

from trainer.base import Trainer

from model.gsv.network import GuidedSpatiallyVaryingConv
from model.warp.mc_mod import MC_Module_Batch
from model.end2end import End2End
from model.flownet.networks import Flownets

#from util.transforms import single as extended_transforms
#from util.transforms.multi import 
from util.transforms.img_utils import tensor_flip_channel, generate_transform_point, transform_map, FlipChannels
from util.metrics import generate_conf_matrix_torch, Evaluator

try:
    from apex.parallel import DistributedDataParallel, SyncBatchNorm
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex .")

class GSVNet_Trainer(Trainer):
    def __init__(self, args):
        super().__init__(args, args.segnet + "_" + args.checkname)
        self.evaluator_single = Evaluator(self.num_classes)
        self.scale_factor_low, self.scale_factor_high = 0.5, 1.5

        if not isinstance(self.train_loader, list):
            self.train_loader = [self.train_loader]
        if not isinstance(self.val_loader, list):
            self.val_loader = [self.val_loader]
        self.early_stop = args.early_stop

    def _img_input_transform(self, args):
        if self.args.segnet == 'bisenet':
            mean_std = ([0.406, 0.456, 0.485], [0.225, 0.224, 0.229])
            img_transform = standard_transforms.Compose([
                standard_transforms.ColorJitter(),
                FlipChannels(),
                standard_transforms.ToTensor(),
                standard_transforms.Normalize(*mean_std)
            ])
        elif self.args.segnet == 'swiftnet':
            mean_std = ([72.3, 82.90, 73.15],[47.73, 48.49, 47.67])
            img_transform = standard_transforms.Compose([
                standard_transforms.ColorJitter(),
                FlipChannels(),
                standard_transforms.ToTensor(),
                standard_transforms.Lambda(lambda x: x.mul_(255)),
                standard_transforms.Normalize(*mean_std),
            ])
        self.mean_std = mean_std
        return img_transform

    def _val_img_input_transform(self, args):
        if self.args.segnet == 'bisenet':
            mean_std = ([0.406, 0.456, 0.485], [0.225, 0.224, 0.229])
            img_transform = standard_transforms.Compose([
                FlipChannels(),
                standard_transforms.ToTensor(),
                standard_transforms.Normalize(*mean_std)
            ])
        elif self.args.segnet == 'swiftnet':
            mean_std = ([72.3, 82.90, 73.15],[47.73, 48.49, 47.67])
            img_transform = standard_transforms.Compose([
                FlipChannels(),
                standard_transforms.ToTensor(),
                standard_transforms.Lambda(lambda x: x.mul_(255)),
                standard_transforms.Normalize(*mean_std),
            ])
        return img_transform

    def init_segnet(self, args, num_classes):
        if args.segnet == 'bisenet':
            from model.bisenet.bisenet_resnet18 import BiSeNet as BiSeNet_Resnet
            segnet = BiSeNet_Resnet(num_classes, is_training=False, criterion=None, ohem_criterion=None, real_time = False, norm_layer = self.batchnorm)
            segnet_checkpoint = torch.load(self.bsnet_weight_path, map_location=self.device)
        elif args.segnet == 'swiftnet':
            from model.swiftnet.configs.import_swiftnet import Swiftnet
            segnet = Swiftnet(num_classes, self.batchnorm)
            segnet_checkpoint = torch.load(self.swnet_weight_path, map_location=self.device)
        else:
            raise NotImplementedError("Segnet is not recognized.")

        if segnet_checkpoint is not None:
            if args.segnet == 'bisenet':
                try:
                    state = segnet_checkpoint['state_dict']
                except:
                    state = segnet_checkpoint['model']    
                print("=> initializing from %s" % self.bsnet_weight_path)            
                segnet.load_state_dict(state, strict=True)
            elif args.segnet == 'swiftnet':
                print("=> initializing from %s" % self.swnet_weight_path)
                segnet.load_state_dict(segnet_checkpoint, strict=True)
            else:
                raise NotImplementedError("Segnet checkpoint is not recognized.")
        return segnet

    def init_optical_flow(self, args):
        return Flownets(args.optical_flow_network, self.optical_flow_network_path, self.batchnorm)

    def init_network(self, args, num_classes):
        # Base semantic segmentation network
        self.segnet = self.init_segnet(args, num_classes)
        self.segnet.to(self.device)
        self.segnet.eval()

        # Motion-compensated layer
        self.mc_layer = MC_Module_Batch(train_noise_sigma=False, eval_noise_sigma=False)
        self.mc_layer.to(self.device)
        self.mc_layer.eval()

        # Optical flow network
        self.ofnet = self.init_optical_flow(args)
        self.ofnet.eval()

        if self.distributed:
            self.ofnet = self.ofnet.to(self.device)
            print("=> rank %i: initializing of" % self.local_rank)
            self.ofnet = DistributedDataParallel(self.ofnet)
            dist.barrier()
        else:
            self.ofnet = nn.DataParallel(self.ofnet.to(self.device))

        return End2End(GuidedSpatiallyVaryingConv(freeze_bn=args.freeze_bn, in_class=num_classes, out_class=num_classes,
                         num_filter=32, batch_norm=nn.BatchNorm2d, is_train_filter=args.train_filter))

    def criterion(self, input, target , cm_mask =None, balance=None):
        final_pred = input 
        gt_target_gin, gt_target_distillation = target
        gt_target_gin = gt_target_gin.to(final_pred.device)
        ce_loss = self.loss['ce'](final_pred, gt_target_gin.squeeze(1))

        if gt_target_distillation is not None:
            final_pred = F.log_softmax(final_pred, dim=1)
            gt_target_distillation = F.softmax(gt_target_distillation, dim=1)

            distillation_loss = F.kl_div(final_pred, gt_target_distillation, reduction="none").mean()
        else:
            distillation_loss = torch.tensor([0]).to(final_pred.device).float()

        return ce_loss + distillation_loss * self.args.param_ce_distil

    def init_loss(self, args, weights, ignore_index):
        return {"ce": nn.CrossEntropyLoss(weight=weights, ignore_index = ignore_index)}
    
    def init_net_params(self, args):
        train_params = [
            {'params': self.network.grm.ideal_dk.parameters(), 'lr': args.lr},
            {'params': self.network.grm.guide.parameters(), 'lr': args.guide_lr}
        ]
        train_params.append({'params': self.network.grm.intra_net.parameters(), 'lr': args.lr})
        if args.train_optical_flow:
            train_params.append({'params': self.ofnet.parameters(), 'lr': args.optical_flow_lr})
        return train_params

    def bilinear(self, x, size):
        return F.interpolate(x, size=size,mode='bilinear',align_corners=True)

    def forward(self, images, loader_idx, args, is_train = False, transform = None, img_id = 0):
        previous_pred = None
        for ref_idx, image in enumerate(images):
            image = image.cuda(non_blocking = True)

            if transform is not None:
                image = transform_map(image, transform, use_crop=self.args.use_crop)

            height, width = image.shape[-2:]

            if ref_idx == 0:
                if not is_train:
                    #use scale_factor = 0.75 while inference
                    low_res_image = self.bilinear(image, (height//4*3, width//4*3))
                else:
                    low_res_image = image
                
                low_res_image = tensor_flip_channel(low_res_image)
                
                with torch.no_grad():
                    pred_ref = self.segnet(low_res_image)
                #resize to 1/8
                pred_ref = self.bilinear(pred_ref, (height//8, width//8))
                
                previous_image = image
                previous_pred = pred_ref
                final_pred = pred_ref
                continue
            else:
                pred_ref = previous_pred
    
            target_image_lowres = self.bilinear(image, (height//8, width//8))
            previous_image_lowres = self.bilinear(previous_image, (height//8, width//8))
            
            if self.args.optical_flow_network == 'light':
                input_flow = torch.cat([target_image_lowres, previous_image_lowres], dim=1)
            else:
                input_flow = torch.cat([previous_image_lowres, target_image_lowres], dim=1)
            
            if not self.args.train_optical_flow:
                with torch.no_grad():
                    output_flow = self.ofnet(input_flow)
            else:
                output_flow = self.ofnet(input_flow)
            
            output_flow = self.bilinear(output_flow, (height//8, width//8))
            merge_input = torch.cat((pred_ref, previous_image_lowres),dim=1)
            mc_out = self.mc_layer(merge_input.unsqueeze(1), output_flow)
            mc_output = mc_out[:,:19,:,:]
            mc_image = mc_out[:,19:22,:,:]
            #input_images = [target_image_lowres, image, previous_image_lowres, mc_image]
            
            if is_train:
                if not self.args.finetune_all:#train only last two layers
                    if ref_idx < len(images) - 2: # 0
                        self.network.eval()
                        with torch.no_grad():
                            final_pred = self.network(mc_output, target_image_lowres)
                    else: # 1 2
                        self.network.train()
                        final_pred = self.network(mc_output, target_image_lowres)
                else:
                    self.network.train()
                    final_pred = self.network(mc_output, target_image_lowres)
            else:
                self.network.eval()
                with torch.no_grad():
                    final_pred = self.network(mc_output, target_image_lowres)

            final_pred , guide_feat, edge = final_pred
            previous_pred = final_pred
            previous_image = image
        return final_pred

    def iter_training(self, epoch, args):
        train_ce_loss = 0.0
        train_num = 0
        self.evaluator.reset()
        self.evaluator_single.reset()
        self.network.train()

        if self.args.train_optical_flow: 
            self.ofnet.train()
        else:
            self.ofnet.eval()

        for loader_idx, train_loader in enumerate(self.train_loader):
            self.evaluator_single.reset()

            if self.distributed: dist.barrier()

            if not self.distributed or (self.distributed and self.local_rank == 0):
                tbar = tqdm(train_loader, ncols=120, desc="Training  ")
            else:
                tbar = train_loader
            
            for i, sample in enumerate(tbar):
                images, target = sample
                target = target.cuda(non_blocking=True).float()

                # scale_factor_low, scale_factor_high, height, width, crop_height, crop_width, do_horizontal_flip
                if self.args.use_crop:
                    transform = generate_transform_point(self.scale_factor_low, self.scale_factor_high,\
                                                    1024, 2048, self.args.crop_height, self.args.crop_width, True)
                    target = transform_map(target.unsqueeze(1), transform, interpolation_mode = 'nearest', use_crop=self.args.use_crop).long().squeeze(1)
                else:
                    target = target.long()
                    transform = None
                height, width = target.shape[-2:]

                final_pred = self.forward(images, loader_idx, args = args, is_train = True, transform = transform)
                
                final_pred = self.bilinear(final_pred, (height, width))
                    
                target_distil = None
                if self.args.distil:
                    last_img = images[-1].cuda(non_blocking = True)
                    last_img = tensor_flip_channel(last_img)
                    if self.args.use_crop:
                        last_img = transform_map(last_img, transform, use_crop=self.args.use_crop)
                    with torch.no_grad():
                        target_distil = self.segnet(last_img)
                        target_distil = self.bilinear(target_distil, (height, width))
                
                ce_loss = self.criterion(final_pred, (target, target_distil))

                loss = ce_loss * self.args.param_ce
                loss.backward()
                train_ce_loss += self.all_reduce_tensor(ce_loss, norm=True).item()
                train_num += 1
                self.optimizer.step()
                self.optimizer.zero_grad()
                map_final_pred = torch.argmax(final_pred, dim=1)
                
                # Gather All Results and Prediction
                conf_matrix = generate_conf_matrix_torch(target.squeeze(1), map_final_pred, self.num_classes)
                red_conf_matrix = self.all_reduce_tensor(conf_matrix, norm=False)
                
                # Add batch sample into evaluator
                self.evaluator.add_conf_matrix(red_conf_matrix)
                self.evaluator_single.add_conf_matrix(red_conf_matrix)
                if not self.distributed or (self.distributed and self.local_rank == 0):
                    tbar.set_postfix({
                        'CE Loss': (train_ce_loss / train_num ),
                        'mIoU': self.evaluator.Mean_Intersection_over_Union(),
                        'single': self.evaluator_single.Mean_Intersection_over_Union(),
                        'PA': self.evaluator.Pixel_Accuracy(),
                        'sPA': self.evaluator_single.Pixel_Accuracy(),
                    }, refresh=True)

                if self.early_stop and (i  == 5):
                    print('early break')
                    break

                if self.args.optim == 'sgd':
                    self.scheduler.step()
                
            if not self.distributed or (self.distributed and self.local_rank == 0):
                tbar.close()
        train_ce_loss = train_ce_loss / train_num
        if self.distributed: dist.barrier()
        return train_ce_loss, 0

    def iter_val(self, loader=None, name="Validation", epoch = 0, args = None):
        self.network.eval()
        self.segnet.eval()
        self.ofnet.eval()
        self.evaluator.reset()
        self.evaluator_single.reset()
        self.mc_layer.eval()

        if loader == None:
            loader = self.val_loader

        val_image = []
        with torch.no_grad():
            for val_idx, val_loader in enumerate(loader):
                self.evaluator_single.reset()
                if self.distributed: dist.barrier()
                if not self.distributed or (self.distributed and self.local_rank == 0):
                    tbar = tqdm(val_loader, ncols=120, desc='\r%s %i' % (name, val_idx))
                else:
                    tbar = val_loader
                
                for i, sample in enumerate(tbar):
                    images, target = sample
                    target = target.long().cuda(non_blocking=True)

                    final_pred = self.forward(images, val_idx-1, args = args, is_train = False, img_id = i)
                    #evaluate on 1024x2048
                    final_pred = self.bilinear(final_pred, (1024, 2048))
                    map_final_pred = torch.argmax(final_pred, dim=1)
                               
                    # Gather All Results and Prediction
                    conf_matrix = generate_conf_matrix_torch(target.squeeze(1), map_final_pred, self.num_classes)
                    red_conf_matrix = self.all_reduce_tensor(conf_matrix, norm=False)
                    
                    # Add batch sample into evaluator
                    self.evaluator.add_conf_matrix(red_conf_matrix)
                    self.evaluator_single.add_conf_matrix(red_conf_matrix)
                    if not self.distributed or (self.distributed and self.local_rank == 0):
                        tbar.set_postfix({
                            'avg.mIoU': self.evaluator.Mean_Intersection_over_Union(),
                            'min.mIoU': self.evaluator_single.Mean_Intersection_over_Union(),
                            'PA': self.evaluator.Pixel_Accuracy(),
                            'sPA': self.evaluator_single.Pixel_Accuracy(),
                        }, refresh=True)
                    
                    if self.early_stop and (i == 10):
                        print('early break')
                        break
                if not self.distributed or (self.distributed and self.local_rank == 0):
                    tbar.close()
        return val_image
    
    def init_optim_scheduler(self, args, params):
        if args.optim == "adam":
            optimizer = optim.Adam(params, weight_decay=0.0001)
        elif args.optim == "sgd":  # SGD
            optimizer = optim.SGD(params, weight_decay=args.weight_decay, momentum=0.9, nesterov=True)
        else:
            raise NotImplementedError

        scheduler = optim.lr_scheduler.StepLR(optimizer, 100, gamma=args.lr_gamma)
        return optimizer, scheduler

    def scheduler_step(self, loss, epoch):
        self.scheduler.step(epoch)

    def run(self):
        print("\r=> start training the network")
        for epoch in range(self.start_epoch, self.start_epoch + self.epoch):
            self.set_all_sampler_epoch(epoch)

            if not self.distributed or (self.distributed and self.local_rank == 0):
                print("\r\n\rEpoch %i" % epoch)

            current_pred = -1
            is_best = False

            if not self.args.evaluate:
                train_loss = self.iter_training(epoch, args = self.args)

            if epoch % self.args.eval_interval == (self.args.eval_interval - 1):
                val_image = self.iter_val(self.val_loader, epoch = epoch, args = self.args)

                if not self.distributed or (self.distributed and self.local_rank == 0):
                    self.logger.log_evaluation("val",
                                               self.evaluator.Pixel_Accuracy(),
                                               self.evaluator.Pixel_Accuracy_Class(),
                                               self.evaluator.Mean_Intersection_over_Union(),
                                               self.evaluator.Frequency_Weighted_Intersection_over_Union(),
                                               epoch)
                    if len(val_image):
                        self.logger.log_image('val/result', make_grid(val_image[0], scale_each=True, nrow=3), epoch)

                current_pred = self.evaluator.Mean_Intersection_over_Union()
                if self.best_pred < current_pred:
                    if (not self.distributed or (self.distributed and self.local_rank == 0)) and not self.args.evaluate:
                        print("=> best prediction: %.3f" % current_pred)
                    self.best_pred = current_pred
                    is_best = True
            current_test_pred = 0.0
            if (not self.distributed or (self.distributed and self.local_rank == 0)) and not self.args.evaluate:
                self.logger.log_loss('train_ce_loss', train_loss[0], epoch)
                self.logger.log_loss('train_cm_loss', train_loss[1], epoch)
                self.save_state(epoch, current_pred, is_best, current_test_pred)

        print("=> network training ends.")

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
        #if self.args.optical_flow:
        dict_mem['state_dict_of'] = self.ofnet.module.state_dict()
        self.saver.save_checkpoint(dict_mem, is_best)

    def load_state(self, args):
        import os
        if not os.path.isfile(self.resume_path):
            raise RuntimeError("=> no checkpoint found at '{}'".format(self.resume_path))
        
        checkpoint = torch.load(self.resume_path,  map_location=self.device)
        warning = self.network.load_state_dict(checkpoint['state_dict'], strict=False)
        print("=> warning: ", warning)

        if 'state_dict_of' in checkpoint.keys() and args.load_flow:
            print("=> load optical flow")
            self.ofnet.module.load_state_dict(checkpoint['state_dict_of'])

        #if reset_all:
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
              .format(self.resume_path, checkpoint['epoch']))