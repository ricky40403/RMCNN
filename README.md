# RMCNN
Detectron is Facebook AI Research's software system that implements state-of-the-art object detection algorithms, including [Mask R-CNN](https://arxiv.org/abs/1703.06870). It is written in Python and powered by the [Caffe2](https://github.com/caffe2/caffe2) deep learning framework.
At FAIR, Detectron has enabled numerous research projects, including: [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144), [Mask R-CNN](https://arxiv.org/abs/1703.06870), [Detecting and Recognizing Human-Object Interactions](https://arxiv.org/abs/1704.07333), [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002), [Non-local Neural Networks](https://arxiv.org/abs/1711.07971), [Learning to Segment Every Thing](https://arxiv.org/abs/1711.10370), [Data Distillation: Towards Omni-Supervised Learning](https://arxiv.org/abs/1712.04440), [DensePose: Dense Human Pose Estimation In The Wild](https://arxiv.org/abs/1802.00434), and [Group Normalization](https://arxiv.org/abs/1803.08494).

And RMCNN is basely on Mask RCNN to make the improvement.

The code in this repository is heavily copied from [Detectron](https://github.com/facebookresearch/Detectron) and modify the artecture.



## Introduction

<div align="center">
  <img src="demo_imgs/outputs/33823288584_1d21cf0a26_k_example_output.jpg" width="700px" />
  <p>Example RMCNN output.</p>
</div>

Thinking that the problem of object loss should cause by the information loss and the insufficient information that detector received. Thus,RMCNN uses some physical concepts to find the best way to improve the detecting work. 

Due to the limitation of the device, this repository using image size with [800,600], the detatil is defined in config file.

## Architecture

<div align="center">
  <img src="" width="700px" />
  <p>Architecture of RMCNN.</p>
</div>



## Installation

Please find installation instructions for Caffe2 and Detectron in [`INSTALL.md`](INSTALL.md).
## Geting Started

## Inference with Pretrained Models

#### 1. Directory of Image Files
To run inference on a directory of image files (`demo/*.jpg` in this example), you can use the `infer_simple.py` tool. In this example, we're using an end-to-end trained Mask R-CNN model with a ResNet-101-FPN backbone from the model zoo:

```
python tools/infer_simple.py \
    --cfg configs/config.yaml \
    --output-dir demo/output
    --image-ext jpg \
    --wts weights/RMCNN.pkl
    demo/images
```

#### 2. COCO Dataset
This example shows how to run an end-to-end trained Mask R-CNN model from the model zoo using a single GPU for inference. As configured, this will run inference on all images in `coco_2014_minival` (which must be properly installed).
Running inference with the same model using `$N` GPUs (e.g., `N=8`).
```
python tools/test_net.py \
    --cfg configs/config.yaml \
    TEST.WEIGHTS weights/RMCNN.pkl \
    NUM_GPUS N
```

## Training a Model with RMCNN
A tiny tutorial showing how to train model on MS COCO.
The model here is an end-to-end trained RMCNN, after trainning, the box and mask AP will show at the end using [MS COCO Metrics](http://cocodataset.org/#detection-eval)

#### 1. Training with 1 GPU

```
python tools/train_net.py \
    --cfg configs/config.yaml \
    OUTPUT_DIR detectron-output
```

### 2. Multi-GPU Training
```
python tools/train_net.py \
    --multi-gpu-testing \
    --cfg configs/config.yaml \
    OUTPUT_DIR detectron-output
```


## License

Detectron is released under the [Apache 2.0 license](https://github.com/facebookresearch/detectron/blob/master/LICENSE). See the [NOTICE](https://github.com/facebookresearch/detectron/blob/master/NOTICE) file for additional details.


