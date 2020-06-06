# Guided Spatially-Varying Convolution for Fast Semantic Segmentation on Video (GSVNet)

## Performance and Benchmarks

The experimental results were conducted on Nvidia GTX 1080Ti. 
Avg. mIoU: the average mIoU over the keyframe and non-keyframes. 
Min. mIoU: the minimum mIoU among frames. (It should be the last non-keyframe) 
Scale: The scaling factor of input resolution. 

### Accuracy vs. Throughput
#### Cityscapes

|**Model**|**Method**|**Backbone**|**Scale**|**# of Non-keyframes**|**Avg. mIoU**|**Min. mIoU**|**FPS**|
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|GSVNet(ours)|Video|SwiftNet-ResNet18|0.75|2|72.5|70.5|125|
|TDNet|Video|BiSeNet-ResNet18|0.75|-|75.0|75.0|approx. 61|
|BiSeNet|Image|ResNet-18|0.75|-|73.7|73.7|61|
|BiSeNet|Image|Xception-39|0.75|-|69.0|69.0|105|
|SwiftNet|Image|ResNet-18|0.75|-|74.4|74.4|63|

### Complexity
