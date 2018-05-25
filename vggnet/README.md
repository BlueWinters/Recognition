# Vggnet
The vggnet model contains several networks, but all of them is based on the classical paper [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556).


## Variants For VggNet
- **vggnet**: the original vggnet which is typically constructed by convolution, nax-pooling and full-connect layer.
- **vggnet_bn**: a improved version of vggnet which adds dropout layers and batch normalization.
- **vggnet_se**: a improved version of vggnet_bn which add squeeze-and-excitation building block.
- **vggnet_pool**: a improved version of vggnet_bn which add max-avg pooling.
- **vggnet_mix**: a improved version of vggnet_bn which add all possible good architecture together.


## Classification Performance
The train logs of each model can be found in [save/vggnet_*]() directory.
The figure on the left represents the training loss of different methods, and the other one represents the accuracy of its test dataset.
<p align="center"> <img src="train_loss.jpg" width="320", height="236"> <img src="accuracy.jpg" width="320", height="236"> </p>


## Summary
Actually, I like the VggNet very much, because it is simple and effective. Most of time, VggNet can be retreated as a base line for many vision tasks. That mean, if you want to verify some ideas, it is a good and a fast thing to do it with VggNet.
Also, if you have some ideas of training techniques, you use VggNet too. All of there express an idea, just like what I do for cifar10, good ideas should work at most of conditions, whatever its environment is simple or complex.

