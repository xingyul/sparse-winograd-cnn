## Efficient Sparse-Winograd Convolutional Neural Networks
This is the code and models for paper <a href="https://openreview.net/forum?id=HJzgZ3JCW" target="_blank">Efficient Sparse-Winograd Convolutional Neural Networks</a> by <a href="https://stanford.edu/~xyl" target="_blank">Xingyu Liu</a> et al.

![architecture](https://github.com/xingyul/sparse-winograd-cnn/blob/master/doc/teaser.png)

### Introduction
This work is based on our ICLR 2018 paper. We propose modifications to Winograd-based CNN architecture to enable operation savings from Winograd’s minimal filtering algorithm and network pruning to be combined. 

Convolutional Neural Networks (CNNs) are computationally intensive, which limits their application on mobile devices. Their energy is dominated by the number of multiplies needed to perform the convolutions. Winograd’s minimal filtering algorithm and network pruning can reduce the operation count, but these two methods cannot be straightforwardly combined — applying the Winograd transform fills in the sparsity in both the weights and the activations. We propose two modifications to Winograd-based CNNs to enable these methods to exploit sparsity. 

In this repository, we release code and data for training Winograd-ReLU CNN on ImageNet as well as pre-trained and iteratively pruned Winograd-ReLU models.

### Citation
If you find our work useful in your research, please cite:

    @article{liu:2018:Winograd,
      title={Efficient Sparse-Winograd Convolutional Neural Networks},
      author={Xingyu Liu and Jeff Pool and Song Han and William J. Dally},
      journal={International Conference on Learning Representations (ICLR)},
      year={2018}
    }
   
### Installation

Install <a href="https://www.tensorflow.org/get_started/os_setup" target="_blank">TensorFlow</a> and <a href="https://github.com/ppwwyyxx/tensorpack" target="_blank">Tensorpack</a>. The code has been tested with Python 2.7, TensorFlow 1.3.0, CUDA 8.0 and cuDNN 5.1 on Ubuntu 14.04.

Users may also need to download raw <a href="http://image-net.org/" target="_blank">ImageNet</a> dataset for ImageNet experiments. Users should ensure that the <a href="https://github.com/ppwwyyxx/tensorpack/tree/master/examples/ResNet" target="_blank">Tensorpack ResNet example</a> can run with ImageNet.

Install customized Tensorflow Op:

    cd /path/to/Sparse-Winograd-CNN/winograd2x2_cublas
    make
    export PYTHONPATH=/path/to/Sparse-Winograd-CNN/winograd2x2_cublas:$PYTHONPATH

Users may also change the ``-arch`` flag in ``winograd2x2_cublas/winograd2x2_imTrans/Makefile`` and ``winograd2x2_cublas/winograd2x2_conv/Makefile`` to suit their GPU computing capability.

Put ``ResNet-18-var/winograd_conv.py`` and ``ResNet-18-var/winograd_imtrans.py`` into the cloned ``tensorpack/models`` directory.

### Usage

To train the Winograd-ReLU CNN from scratch on ImageNet with GPU 0 and 1:

    ./imagenet-resnet-transWino-prune.py --gpu 0,1 --data /path/to/dataset/imagenet

To use pre-trained model or test with pruned model, download the <a href="https://drive.google.com/drive/folders/1YA3syxt5yzBiRiwW_dswc5YmRg4p4vdG?usp=sharing" target="_blank">models</a>. Then run with command:

    ./imagenet-resnet-transWino-prune.py --gpu 0,1 --data /path/to/dataset/imagenet --load /path/to/model-name.data-00000-of-00001

We also provided scripts for pruning, retraining and viewing the model: ``ResNet-18-var/prune_sh.sh``, ``retrain_sh.sh`` and ``view_sh.sh``.

### License
Our code is released under MIT License (see LICENSE file for details).

