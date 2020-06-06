# Guided Spatially-Varying Convolution for Fast Semantic Segmentation on Video (GSVNet)

## Supported Models
- [TDNet](https://arxiv.org/abs/2004.01800)
- [BiSeNet](https://arxiv.org/abs/1808.00897)
- [SwiftNet](https://arxiv.org/abs/1903.08469)
- [Accel](https://arxiv.org/abs/1807.06667)

## Performance and Benchmarks

The experimental results were conducted on Nvidia GTX 1080Ti. 
- Avg. mIoU: the average mIoU over the keyframe and non-keyframes. 
- Min. mIoU: the minimum mIoU among frames. (It should be the last non-keyframe) 
- Scale: The scaling factor of input resolution.
- Avg. Flops: the average floating-point operations per second (FLOPS) over the keyframe and non-keyframes.
- l=K: The number of non-keyframes.

### Accuracy vs. Throughput
#### Cityscapes



|**Model**|**Method**|**Backbone**|**Scale**|**Avg. mIoU**|**Min. mIoU**|**FPS**|
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|GSVNet(l=2)|Video|SwiftNet-ResNet18|0.75|72.5|70.5|125|
|TDNet|Video|BiSeNet-ResNet18|0.75|75.0|75.0|approx. 61|
|Accel-18(l=5)|Video|DeepLab-ResNet18|1.0|72.1|??|2.2|
|BiSeNet|Image|ResNet-18|0.75|73.7|73.7|61|
|BiSeNet|Image|Xception-39|0.75|69.0|69.0|105|
|SwiftNet|Image|ResNet-18|0.75|74.4|74.4|63|

### Complexity
|**Model**|**Backbone**|**Scale**|**# of Parameters**|**Avg. FLOPS**|
|:-----:|:-----:|:-----:|:-----:|:-----:|
|GSVNet(l=2)|SwiftNet-ResNet18|0.75|48.8M|21.3G|
|GSVNet(l=3)|SwiftNet-ResNet18|0.75|48.8M|16.7G|
|SwiftNet|ResNet18|0.75|47.2M|58.5G|
|SwiftNet|ResNet18|0.5|47.2M|26.0G|
|BiSeNet|ResNet18|0.75|49.0M|58.0G|
