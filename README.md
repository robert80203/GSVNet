# GSVNet

This is the official implementation of [GSVNet: Guided Spatially-Varying Convolution for Fast Semantic Segmentation on Video](https://arxiv.org/abs/2103.08834)

Please cite our ICME 2021 paper if our paper/implementation is helpful for your research:
```
@misc{lee2021gsvnet,
      title={GSVNet: Guided Spatially-Varying Convolution for Fast Semantic Segmentation on Video}, 
      author={Shih-Po Lee and Si-Cun Chen and Wen-Hsiao Peng},
      year={2021},
      eprint={2103.08834},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Installation

Create a conda environment under Python 3.7
```
conda create --name gsvnet python=3.7
conda activate gsvnet
``` 

Install python packages from requirements.txt
```
pip install -r requirements.txt
```

Install python package apex for distributed training
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./
```

## Supported model weights

### Image-based segmentation network pre-trained on [Cityscapes](https://arxiv.org/abs/1604.01685)

|**Model**|**Pre-trained weight**|
|:-----:|:-----:|
|[BiSeNet](https://arxiv.org/abs/1808.00897) | [Download](https://drive.google.com/file/d/11tr04lZCWtU1dQXLCteEOSYnUE2REt4V/view?usp=sharing)|
|[SwiftNet](https://arxiv.org/abs/1903.08469) | [Download](https://drive.google.com/file/d/1_FxwIwK52OEMCx5gOz3LgFJTgX2J_rG7/view?usp=sharing)|
|[FlowNet2S](https://arxiv.org/abs/1612.01925) | [Download](https://drive.google.com/file/d/1A8FKDbMKORz9U_swRxLaCUFcmY4eAZYj/view?usp=sharing)|

Move the downloaded weights to weights/

## Preparation of dataset - Cityscapes

Please download the dataset from the officit site - [Download](https://www.cityscapes-dataset.com/)

```
data_path = './data/leftImg8bit_sequence_trainvaltest_2K/'
```

Modify the data_path in config/cityscapes.py

## Training of GSVNet

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 1111 \ 
--nproc_per_node 2 python main.py --segnet <segnet_name> --dataset <dataset_name> \
--optical-flow-network <of_name> --checkname <SAVE_DIR>
```

<segnet_name> = swiftnet/bisenet
<dataset_name> = cityscapes_2k/camvid
<of_name> = light/flownet

## Inference of GSVNet on Cityscapes

```
python main.py --evaluate 1 --batch-size 1 --resume 1
```

## Performance and Benchmarks

The experimental results were conducted on Nvidia GTX 1080Ti. 
- Avg. mIoU: the average mIoU over the keyframe and non-keyframes. 
- Min. mIoU: the minimum mIoU among frames. (It should be the last non-keyframe) 
- Scale: The scaling factor of input resolution.
- Avg. Flops: the average floating-point operations per second (FLOPS) over the keyframe and non-keyframes.
- l=K: The keyframe interval.

### Accuracy vs. Throughput

#### Cityscapes

|**Model**|**Method**|**Scale**|**Avg. mIoU**|**Min. mIoU**|**FPS**|**Weight**|
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|Ours-SN-R18(l=3)|Video|0.75|72.5|70.3|125|[Download](https://drive.google.com/file/d/1VIfO-T0EWhdOiHrorAppLZQFuMUwcjC6/view?usp=sharing)|
|Ours-BN-R18(l=3)|Video|0.75|72.0|70.5|123|[Download](https://drive.google.com/file/d/16adWqQRmzpQFGP8-EmGKn_ctXP8PPQPO/view?usp=sharing)|
|[TDNet-BN-R18](https://arxiv.org/abs/2004.01800) |Video|0.75|75.0|75.0|approx. 61||
|[Accel-DL-R101-18](https://arxiv.org/abs/1807.06667)(l=5) |Video|1.0|72.1|None|2.2||
|BiSeNet-R18|Image|0.75|73.7|73.7|61||
|BiSeNet-R18|Image|0.75|69.0|69.0|105||
|SwiftNet-R18|Image|0.75|74.4|74.4|63||

### Complexity

|**Model**|**Scale**|**# of Parameters**|**Avg. FLOPS**|
|:-----:|:-----:|:-----:|:-----:|
|Ours-SN-R18(l=3)|0.75|50.4M|21.3G|
|SwiftNet-R18|0.75|47.2M|58.5G|
|SwiftNet-R18|0.5|47.2M|26.0G|
|BiSeNet-R18|0.75|49.0M|58.0G|
