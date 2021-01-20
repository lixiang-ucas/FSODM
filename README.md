# Few-shot YOLOv3: Few-shot Object Detection on Optical Remote Sensing Images


Our code is based on  [https://github.com/marvis/pytorch-yolo2](https://github.com/marvis/pytorch-yolo2) and developed with  Python 2.7 & PyTorch 0.3.1.

## Detection Examples

<div align=center>
<img src="https://user-images.githubusercontent.com/47288017/85243067-5051d380-b473-11ea-9a8e-17e19692746c.png" width="740">
</div>

<div align=center>
Samples of novel class detection result on NWPU VHR-10 and DIOR with 10-shot training bounding boxes.
</div> 

## Model
<div align=center>
<img src="https://user-images.githubusercontent.com/47288017/85242838-ab36fb00-b472-11ea-9ec8-418d06d057a0.png" width="740">
</div>

The pipeline of the proposed method for few-shot object detection on remote sensing images. Our method consists of three main components: a Meta Feature Extractor, a Reweighting Module, and a Bounding Box Prediction Module. The Feature Extractor network takes a query image as input and produces meta feature maps at three different scales. The Reweighting Module takes as input N support images with labels, one for each class, and outputs three groups of N reweighting vectors. These reweighting vectors are used to recalibrate the meta-feature maps of the same scale through a channel-wise multiplication. The reweighted feature maps are then fed into three independent bounding box detection modules to predict the objectness scores (o), the bounding box locations and sizes (x, y, w, h) and class scores (c) at three different scales.


## Abstract
In this paper, we deal with the problem of object detection on remote sensing images. Previous methods have developed numerous deep CNN-based methods for object detection on remote sensing images and the report remarkable achievements in detection performance and efficiency. However, current CNN-based methods mostly require a large number of annotated samples to train deep neural networks and tend to have limited generalization abilities for unseen object categories. In this paper, we introduce a few-shot learning-based method for object detection on remote sensing images where only a few annotated samples are provided for the unseen object categories. More specifically, our model contains three main components: a meta feature extractor that learns to extract feature representations from input images, a reweighting module that learn to adaptively assign different weights for each feature representation from the support images, and a bounding box prediction module that carries out object detection on the reweighted feature maps. We build our few-shot object detection model upon YOLOv3 architecture and develop a multi-scale object detection framework. Experiments on two benchmark datasets demonstrate that with only a few annotated samples our model can still achieve a satisfying detection performance on remote sensing images and the performance of our model is significantly better than the well-established baseline models.


## Training our model on NWPU VHR-10

- ``` $PROJ_ROOT : project root ```
- ``` $DATA_ROOT : dataset root ```

### Prepare dataset
Get the NWPU VHR-10 data from https://1drv.ms/u/s!AmgKYzARBl5cczaUNysmiFRH4eE

Preprocess data
```
cd $PROJ_ROOT
python scripts/ImgSplit_nwpu.py $DATA_ROOT
```

Generate labels for NWPU VHR-10
```
python scripts/label_nwpu.py $DATA_ROOT
```

Generate per-class labels for NWPU VHR-10 (used for the Reweighting Module input)
```
python scripts/label_1c_nwpu.py $DATA_ROOT
```

Generate few-shot datasets   
Change the ''DROOT'' varibale in scripts/gen_fewlist_nwpu.py to $DATA_ROOT
```
python scripts/gen_fewlist_nwpu.py
```

Generate training dictionary
```
python scripts/gen_dict_file.py $DATA_ROOT nwpu
```

### Base Training
Modify Config for NWPU VHR-10 Data   
Change the cfg/fewyolov3_nwpu.data file 
```
metayolo = 1
metain_type = 2
data = nwpu
neg = 1
rand = 0
novel = data/nwpu_novels.txt
novelid = 0
learning_rate = 0.001
steps = -1,64000
scales = 1,0.1
meta = data/nwpu_traindict_full.txt
train = $DATA_ROOT/training.txt
valid = $DATA_ROOT/evaluation.txt
backup = backup/fewyolov3_nwpu
gpus = 0,1,2,3
```

Train the Model
```
python train.py cfg/fewyolov3_nwpu.data cfg/darknet_yolov3_spp.cfg cfg/reweighting_net.cfg
```

Evaluate the Model
```
python valid.py cfg/fewyolov3_nwpu.data cfg/darknet_yolov3_spp.cfg cfg/reweighting_net.cfg path/toweightfile
python scripts/voc_eval.py results/path/to/comp4_det_test_ cfg/metayolo.data
```

### Few-shot Tuning
Modify Config for NWPU VHR-10 Data   
Change the cfg/fewtunev3_nwpu_10shot.data file (change the shot number to try different shots)
```
metayolo = 1
metain_type = 2
data = nwpu
tuning = 1
neg = 0
rand = 0
novel = data/nwpu_novels.txt
novelid = 0
max_epoch = 2000
repeat = 200
dynamic = 0
scale = 1
learning_rate = 0.0001
steps = -1,64000
scales = 1,0.1
train = $DATA_ROOT/training.txt
meta = data/nwpu_traindict_bbox_10shot.txt
valid = $DATA_ROOT/evaluation.txt
backup = backup/fewtunetestv3_nwpu_10shot
gpus = 0,1,2,3
```


Train the Model with 10 shot
```
python train.py cfg/fewtunev3_nwpu_10shot.data cfg/darknet_yolov3_spp.cfg cfg/reweighting_net.cfg path/to/base/weightfile
```

Evaluate the Model
```
python valid.py cfg/fewtunev3_nwpu_10shot.data cfg/darknet_yolov3_spp.cfg cfg/reweighting_net.cfg path/to/tuned/weightfile
python scripts/voc_eval.py results/path/to/comp4_det_test_ cfg/fewtunev3_nwpu_10shot.data
```
