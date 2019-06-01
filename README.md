# FSA-Net
**[CVPR19] FSA-Net: Learning Fine-Grained Structure Aggregation for Head Pose Estimation from a Single Image**

**Code Author: Tsun-Yi Yang**

**Last update: 2019/04/15 (Demo uploaded)**

### Comparison video
(Baseline **Hopenet:** https://github.com/natanielruiz/deep-head-pose)
<img src="https://github.com/shamangary/FSA-Net/blob/master/Compare_AFLW2000_gt_Hopenet_FSA.gif" height="320"/>

### Webcam demo

| Signle person (LBP) | Multiple people (MTCNN)|
| --- | --- |
| <img src="https://github.com/shamangary/FSA-Net/blob/master/webcam_demo.gif" height="220"/> | <img src="https://github.com/shamangary/FSA-Net/blob/master/webcam_demo_cvlab_citi.gif" height="220"/> |


| Time sequence | Fine-grained structure|
| --- | --- |
| <img src="https://github.com/shamangary/FSA-Net/blob/master/time_demo.png" height="160"/> | <img src="https://github.com/shamangary/FSA-Net/blob/master/heatmap_demo.png" height="330"/> |



### Results
<img src="https://github.com/shamangary/FSA-Net/blob/master/FSANET_table1.png" height="220"/><img src="https://github.com/shamangary/FSA-Net/blob/master/FSANET_table2.png" height="220"/><img src="https://github.com/shamangary/FSA-Net/blob/master/FSANET_table3.png" height="220"/>


## Paper


### PDF
https://github.com/shamangary/FSA-Net/blob/master/0191.pdf


### Paper authors
**[Tsun-Yi Yang](https://scholar.google.com/citations?user=WhISCE4AAAAJ&hl=en), [Yi-Ting Chen](https://sites.google.com/media.ee.ntu.edu.tw/yitingchen/), [Yen-Yu Lin](https://www.citi.sinica.edu.tw/pages/yylin/index_zh.html), and [Yung-Yu Chuang](https://www.csie.ntu.edu.tw/~cyy/)**


## Abstract
This paper proposes a method for head pose estimation from a single image. Previous methods often predicts head poses through landmark or depth estimation and would require more computation than necessary. Our method is based on regression and feature aggregation. For having a compact model, we employ the soft stagewise regression scheme. Existing feature aggregation methods treat inputs as a bag of features and thus ignore their spatial relationship in a feature map. We propose to learn a fine-grained structure mapping for spatially grouping features before aggregation. The fine-grained structure provides part-based information and pooled values. By ultilizing learnable and non-learnable importance over the spatial location, different variant models as a complementary ensemble can be generated. Experiments show that out method outperforms the state-of-the-art methods including both the landmark-free ones and the ones based on landmark or depth estimation. Based on a single RGB frame as input, our method even outperforms methods utilizing multi-modality information (RGB-D, RGB-Time) on estimating the yaw angle. Furthermore, the memory overhead of the proposed model is 100× smaller than that of previous methods.

## Platform
+ Keras
+ Tensorflow
+ GTX-1080Ti
+ Ubuntu
```
python                    3.5.6                hc3d631a_0  
keras-applications        1.0.4                    py35_1    anaconda
keras-base                2.1.0                    py35_0    anaconda
keras-gpu                 2.1.0                         0    anaconda
keras-preprocessing       1.0.2                    py35_1    anaconda
tensorflow                1.10.0          mkl_py35heddcb22_0  
tensorflow-base           1.10.0          mkl_py35h3c3e929_0  
tensorflow-gpu            1.10.0               hf154084_0    anaconda
cudnn                     7.1.3                 cuda8.0_0  
cuda80                    1.0                           0    soumith
numpy                     1.15.2          py35_blas_openblashd3ea46f_0  [blas_openblas]  conda-forge
numpy-base                1.14.3           py35h2b20989_0  
```

## Dependencies
+ A guide for most dependencies. (in Chinese)
http://shamangary.logdown.com/posts/3009851
+ Anaconda
+ OpenCV
+ MTCNN
```
pip3 install mtcnn
```
+ Capsule: https://github.com/XifengGuo/CapsNet-Keras
+ Loupe_Keras: https://github.com/shamangary/LOUPE_Keras

## Codes

There are three different section of this project. 
1. Data pre-processing
2. Training and testing
3. Demo

We will go through the details in the following sections.

This repository is for 300W-LP, AFLW2000, and BIWI datasets.


### 1. Data pre-processing

#### [For lazy people just like me] 

If you don't want to re-download every dataset images and do the pre-processing again, or maybe you don't even care about the data structure in the folder. Just download the file **data.zip** from the following link, and replace the data folder.

[Google drive](https://drive.google.com/file/d/1j6GMx33DCcbUOS8J3NHZ-BMHgk7H-oC_/view?usp=sharing)

Now you can skip to the "Training and testing" stage.

#### [Details]

In the paper, we define **Protocol 1** and **Protocol 2**.

```

# Protocol 1

Training: 300W-LP (A set of subsets: {AFW.npz, AFW_Flip.npz, HELEN.npz, HELEN_Flip.npz, IBUG.npz, IBUG_Flip.npz, LFPW.npz, LFPW_Flip.npz})
Testing: AFLW2000.npz or BIWI_noTrack.npz


# Protocol 2

Training: BIWI(70%)-> BIWI_train.npz
Testing: BIWI(30%)-> BIWI_test.npz

```
(Note that type1 (300W-LP, AFLW2000) datasets have the same image arrangement, and I categorize them as **type1**. It is not about Protocal 1 or 2.)

If you want to do the pre-processing from the beginning, you need to download the dataset first.

#### Download the datasets

+ [300W-LP, AFLW2000](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)
+ [BIWI](https://data.vision.ee.ethz.ch/cvl/gfanelli/head_pose/head_forest.html)

Put 300W-LP and AFLW2000 folders under **data/type1/**, and put BIWI folder under **data/**

#### Run pre-processing

```
# For 300W-LP and AFLW2000 datasets

cd data/type1
sh run_created_db_type1.sh


# For BIWI dataset

cd data
python TYY_create_db_biwi.py
python TYY_create_db_biwi_70_30.py
```


### 2. Training and testing

```

# Training
sh run_fsanet_train.sh

# Testing
sh run_fsanet_test.sh

```

Just remember to check which model type you want to use in the shell script and you are good to go.


### 3. Demo

You need a **webcam** to correctly process the demo file.


Note the the center of the color axes is the detected face center.
Ideally, each frame should have new face detection results.
However, if the face detection fails, the previous detection results will be used to estimate poses.


LBP is fast enough for real-time face detection, while MTCNN is much more accurate but slow.
```
# LBP face detector (fast but often miss detecting faces)
cd demo
sh run_demo_FSANET.sh

# MTCNN face detector (slow but accurate)
cd demo
sh run_demo_FSANET_mtcnn.sh
```
