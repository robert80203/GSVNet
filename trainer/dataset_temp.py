#from config import get_config

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as standard_transforms

import multiprocessing as mp


def build_trainer_dataset(dataset_name, distributed=False):
    if dataset_name.find("cityscapes") >= 0:
        return CityscapesDataset(dataset_name, distributed)
    elif dataset_name == "camvid":
        return CamvidDataset(dataset_name, distributed)
    else:
        raise NotImplementedError("Trainer dataset %s is not registered into the system" % dataset_name)

class TrainerDataset(object):
    def init_dataset(self, **args):
        raise NotImplementedError("Must implement trainer dataset.")

    def init_dataloader(self, **args):
        raise NotImplementedError("Must implement trainer dataloader.")

class CamvidDataset(TrainerDataset):
    def __init__(self, config="cityscapes", distributed=False):
        super().__init__()
        self.config = get_config(config)
        self.distributed = distributed

    def init_dataset(self, args, image_transform, val_image_transform, label_transform, joint_transform, val_joint_transform=None):
        self.config = get_config(args.dataset)

        key_intervals = [1]
        load_reference = False
        multi_step = [1]
        eval_multi_step = [1]
        use_decoded_rgb= True

        if "dataset_key_int" in args.__dict__.keys():
            key_intervals = args.dataset_key_int
        if "multi_step" in args.__dict__.keys():
            multi_step = args.multi_step
            load_reference = True

        if "use_decoded_rgb" in args.__dict__.keys():
            use_decoded_rgb = args.use_decoded_rgb
        compressed_ver = "p_qp%i_ref%i" % (args.dataset_compressed_qp, args.dataset_compressed_ref)

        # Training Set
        train_datasets = []
        for step in multi_step:
            temp_dataset = build_base_dataset(args.dataset, 'train', compressed_ver, key_intervals=key_intervals,
                                                     multi_step= step,
                                                     image_mode=args.dataset_image_mode,
                                                     image_transform=image_transform,
                                                     label_transform=label_transform,
                                                     joint_transform=joint_transform,
                                                     iter_include_ref=load_reference,
                                                     iter_include_ref_resi= not use_decoded_rgb)
            if temp_dataset is not None: train_datasets.append(temp_dataset)

        if "eval_dataset_key_int" in args.__dict__.keys():
            key_intervals = args.eval_dataset_key_int
        if "eval_multi_step" in args.__dict__.keys():
            eval_multi_step = args.eval_multi_step

        # Validation Set
        val_datasets = []
        for step in eval_multi_step:
            val_dataset = build_base_dataset(args.dataset, 'val', compressed_ver, key_intervals=key_intervals,
                                             multi_step = step,
                                             image_mode=args.dataset_image_mode,
                                             image_transform=val_image_transform,
                                             label_transform=label_transform,
                                             joint_transform=val_joint_transform,
                                             iter_include_ref=load_reference,
                                             iter_include_ref_resi= not use_decoded_rgb)
            val_datasets.append(val_dataset)

        frame_val_dataset = build_base_dataset(args.dataset, 'val', compressed_ver, key_intervals=[1],
                                               multi_step = eval_multi_step,
                                               image_mode=args.dataset_image_mode,
                                               image_transform=val_image_transform,
                                               label_transform=label_transform,
                                               joint_transform=val_joint_transform,
                                               iter_include_ref=False)

        # Test Set
        test_datasets = []
        for step in eval_multi_step:
            test_dataset = build_base_dataset(args.dataset, 'test', compressed_ver, key_intervals=key_intervals,
                                             multi_step = step,
                                             image_mode=args.dataset_image_mode,
                                             image_transform=val_image_transform,
                                             label_transform=label_transform,
                                             joint_transform=val_joint_transform,
                                             iter_include_ref=load_reference,
                                             iter_include_ref_resi= not use_decoded_rgb)
            test_datasets.append(test_dataset)

        frame_test_dataset = build_base_dataset(args.dataset, 'test', compressed_ver, key_intervals=[1],
                                               multi_step = eval_multi_step,
                                               image_mode=args.dataset_image_mode,
                                               image_transform=val_image_transform,
                                               label_transform=label_transform,
                                               joint_transform=val_joint_transform,
                                               iter_include_ref=False)

        return train_datasets, [frame_val_dataset, val_datasets], [frame_test_dataset, test_datasets]

    def init_dataloader(self, args, train_dataset, val_dataset, test_dataset=None, world_size=1):
        batch_size = args.batch_size
        if self.distributed : batch_size = args.batch_size // world_size

        # Training set
        train_loader = []
        train_sampler = []
        if train_dataset is not None:
            if isinstance(train_dataset, list):
                for dataset in train_dataset:
                    sampler = None
                    if self.distributed: sampler = torch.utils.data.distributed.DistributedSampler(dataset)
                    train_loader.append(DataLoader(dataset, batch_size=batch_size, shuffle= (sampler is None), num_workers=args.num_worker, pin_memory=True, sampler=sampler))
                    train_sampler.append(sampler)
            else:
                train_sampler = None
                if self.distributed: train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None), num_workers=args.num_worker,pin_memory=True, sampler=train_sampler)


        # Validation set
        if isinstance(val_dataset[1], list):
            val_loader = []
            val_sampler = []

            for dataset in val_dataset[1]:
                sampler = None
                if self.distributed: sampler = torch.utils.data.distributed.DistributedSampler(dataset)
                val_loader.append(DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_worker, pin_memory=True, sampler=sampler))
                val_sampler.append(sampler)
        else:
            val_sampler = None
            if self.distributed: val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset[1])
            val_loader = DataLoader(val_dataset[1], batch_size=batch_size, shuffle=False, num_workers=args.num_worker, pin_memory=True, sampler=val_sampler)

        frame_val_sampler = None
        if self.distributed: frame_val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset[0])
        frame_val_loader = DataLoader(val_dataset[0], batch_size=batch_size, shuffle=False, num_workers=args.num_worker, pin_memory=True, sampler=frame_val_sampler)

        # Test set
        if isinstance(test_dataset[1], list):
            test_loader = []
            test_sampler = []

            for dataset in test_dataset[1]:
                sampler = None
                if self.distributed: sampler = torch.utils.data.distributed.DistributedSampler(dataset)
                test_loader.append(DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_worker, pin_memory=True, sampler=sampler))
                test_sampler.append(sampler)
        else:
            test_sampler = None
            if self.distributed: test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset[1])
            test_loader = DataLoader(test_dataset[1], batch_size=batch_size, shuffle=False, num_workers=args.num_worker, pin_memory=True, sampler=test_sampler)

        frame_test_sampler = None
        if self.distributed: frame_test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset[0])
        frame_test_loader = DataLoader(test_dataset[0], batch_size=batch_size, shuffle=False, num_workers=args.num_worker, pin_memory=True, sampler=frame_test_sampler)

        return train_loader, train_sampler, [frame_val_loader, val_loader], [frame_val_sampler, val_sampler], [frame_test_loader, test_loader], [frame_test_sampler, test_sampler]



class CityscapesDataset(TrainerDataset):

    def __init__(self, config="cityscapes", distributed=False):
        super().__init__()
        self.config = get_config(config)
        self.distributed = distributed

    def init_dataset(self, args, image_transform, val_image_transform, label_transform, joint_transform, val_joint_transform=None):
        self.config = get_config(args.dataset)

        key_intervals = [1]
        load_reference = False
        multi_step = [1]
        eval_multi_step = [1]
        use_decoded_rgb= True

        if "dataset_key_int" in args.__dict__.keys():
            key_intervals = args.dataset_key_int
        if "multi_step" in args.__dict__.keys():
            multi_step = args.multi_step
            load_reference = True

        if "use_decoded_rgb" in args.__dict__.keys():
            use_decoded_rgb = args.use_decoded_rgb
        compressed_ver = "p_qp%i_ref%i" % (args.dataset_compressed_qp, args.dataset_compressed_ref)


        # Training Set
        train_datasets = []
        for step in multi_step:
            temp_dataset = build_base_dataset(args.dataset, 'train', compressed_ver, key_intervals=key_intervals,
                                                     multi_step= step,
                                                     image_mode=args.dataset_image_mode,
                                                     image_transform=image_transform,
                                                     label_transform=label_transform,
                                                     joint_transform=joint_transform,
                                                     iter_include_ref=load_reference,
                                                     iter_include_ref_resi= not use_decoded_rgb,
                                                     load_data_once = args.load_data_once,
                                                     local_rank = args.local_rank
                                                     )
            if temp_dataset is not None: train_datasets.append(temp_dataset)

        if "eval_dataset_key_int" in args.__dict__.keys():
            key_intervals = args.eval_dataset_key_int
        if "eval_multi_step" in args.__dict__.keys():
            eval_multi_step = args.eval_multi_step

        # Validation Set
        val_datasets = []
        for step in eval_multi_step:
            val_dataset = build_base_dataset(args.dataset, 'val', compressed_ver, key_intervals=key_intervals,
                                             multi_step = step,
                                             image_mode=args.dataset_image_mode,
                                             image_transform=val_image_transform,
                                             label_transform=label_transform,
                                             joint_transform=val_joint_transform,
                                             iter_include_ref=load_reference,
                                             iter_include_ref_resi= not use_decoded_rgb,
                                             load_data_once = args.load_data_once,
                                             local_rank = args.local_rank
                                            )
            val_datasets.append(val_dataset)

        frame_val_dataset = build_base_dataset(args.dataset, 'val', compressed_ver, key_intervals=key_intervals[:1],
                                               multi_step = eval_multi_step,
                                               image_mode=args.dataset_image_mode,
                                               image_transform=val_image_transform,
                                               label_transform=label_transform,
                                               joint_transform=val_joint_transform,
                                               iter_include_ref=False,
                                               load_data_once = args.load_data_once,
                                               local_rank = args.local_rank
                                             )

        return train_datasets, [frame_val_dataset, val_datasets]#, [frame_test_dataset, test_datasets]#None

    def init_dataloader(self, args, train_dataset, val_dataset, test_dataset=None, world_size=1):
        batch_size = args.batch_size
        if self.distributed : batch_size = args.batch_size // world_size

        # Training set
        train_loader = []
        train_sampler = []
        if train_dataset is not None:
            if isinstance(train_dataset, list):
                for dataset in train_dataset:
                    sampler = None
                    if self.distributed: sampler = torch.utils.data.distributed.DistributedSampler(dataset)
                    #sampler.shuffle = False
                    train_loader.append(DataLoader(dataset, batch_size=batch_size, shuffle= (sampler is None), num_workers=args.num_worker, pin_memory=True, sampler=sampler))
                    train_sampler.append(sampler)
            else:
                train_sampler = None
                if self.distributed: train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, None, None,   False)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None), num_workers=args.num_worker,pin_memory=True, sampler=train_sampler)


        # Validation set
        if isinstance(val_dataset[1], list):
            val_loader = []
            val_sampler = []

            for dataset in val_dataset[1]:
                sampler = None
                if self.distributed: sampler = torch.utils.data.distributed.DistributedSampler(dataset)
                val_loader.append(DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_worker, pin_memory=True, sampler=sampler))
                val_sampler.append(sampler)
        else:
            val_sampler = None
            if self.distributed: val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset[1])
            val_loader = DataLoader(val_dataset[1], batch_size=batch_size, shuffle=False, num_workers=args.num_worker, pin_memory=True, sampler=val_sampler)

        frame_val_sampler = None
        if self.distributed: frame_val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset[0])
        frame_val_loader = DataLoader(val_dataset[0], batch_size=batch_size, shuffle=False, num_workers=args.num_worker, pin_memory=True, sampler=frame_val_sampler)

        return train_loader, train_sampler, [frame_val_loader, val_loader], [frame_val_sampler, val_sampler]#, [frame_test_loader, test_loader], [frame_test_sampler, test_sampler]#None, None