# Attribution

The alignment method is from Isma Hadji, Konstantinos G. Derpanis, and Allan D.
Jepson, *Representation Learning via Global Temporal Alignment and Cycle-Consistency*,
CVPR 2021. Cite that paper when using this method in research.

[VideoAlignment](https://github.com/hadjisma/VideoAlignment) was consulted to clarify
architecture and algorithm details. Its main code/model license is
CC BY-NC-SA 4.0; some network and training files carry Google Research's Apache 2.0
notice. This directory contains newly written PyTorch code and no copied weights,
datasets, TensorFlow implementation, or imports from that repository.

The ResNet-v2 architecture was checked against the
[Keras Applications architecture](https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet_common.py).
The alternative ResNet50-v1 backbone uses torchvision as a dependency. Those
dependencies retain their own licenses. Any checkpoints later selected retain
the licenses associated with their respective sources.
