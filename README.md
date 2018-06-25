# CRF-RNN for Semantic Image Segmentation - Keras/Tensorflow version

This repository contains Keras/Tensorflow code for the "CRF-RNN" semantic image segmentation method, published in the ICCV 2015 paper [Conditional Random Fields as Recurrent Neural Networks](http://www.robots.ox.ac.uk/~szheng/papers/CRFasRNN.pdf). This paper was initially described in an [arXiv tech report](http://arxiv.org/abs/1502.03240). The [online demo](http://crfasrnn.torr.vision) of this project won the Best Demo Prize at ICCV 2015. Original Caffe-based code of this project can be found [here](https://github.com/torrvision/crfasrnn). Results produced with this Keras/Tensorflow code are almost identical to that with the Caffe-based version.

## Installation Guide

### Step 1: Install dependencies

**Note**: If you are using a Python virtualenv, make sure it is activated before running each command in this guide.

Use the `requirements.txt` file (or `requirements_gpu.txt`, if you have a GPU device) in this repository to install all the dependencies via `pip`:
```
$ cd crfasrnn_keras
$ pip install -r requirements.txt  # If you have a GPU device, use requirements_gpu.txt instead
```
As you can notice from the contents of `requirements.txt`, we only depend on `tensorflow`, `keras`, and `h5py`. Additionally, `Pillow` is required for running the demo.
After installing the dependencies, run the following commands to make sure they are properly installed:
```
$ python
>>> import tensorflow
>>> import keras
```
You should not see any errors while importing `tensorflow` and `keras` above.


### Step 2: Build CRF-RNN custom op C++ code

Run `make` inside the `crfasrnn_keras/src/cpp` directory:
```
$ cd crfasrnn_keras/src/cpp
$ make
``` 
Note that the `python` command in the console should refer to the Python interpreter associated with your Tensorflow installation before running the `make` command above.

You will get a new file named `high_dim_filter.so` from this build. If it fails, refer to the official Tensorflow guide for [building a custom op](https://www.tensorflow.org/extend/adding_an_op#build_the_op_library) for help.

**Note**: This make script works on Linux and macOS, but not on Windows OS. If you are on Windows, please check [this issue](https://github.com/tensorflow/models/issues/1103) and the comments therein for build instructions. The official Tensorflow guide for building a custom op does not yet include build instructions for Windows.

### Step 3: Download the pre-trained model weights

Download the model weights from [here](https://goo.gl/ciEYZi) or [here](https://github.com/sadeepj/crfasrnn_keras/releases/download/v1.0/crfrnn_keras_model.h5) and place it in the `crfasrnn_keras` directory with the file name `crfrnn_keras_model.h5`.


## Run the Model End-to-End:
Run the demo with images from big_figure and compose with backgrounds from bg
```
python matting_main.py
```

Run on custom images
```
python matting_main.py image_dir
```


## Notes
1. Current implementation of the CrfRnnLayer only supports batch_size == 1
2. An experimental GPU version of the CrfRnnLayer that has been tested on CUDA 9 and Tensorflow 1.7 only, is available under the `gpu_support` branch. This code was contributed by [thwjoy](https://github.com/thwjoy).
