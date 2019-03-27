# FSA-Net
**[CVPR19] FSA-Net: Learning Fine-Grained Structure Aggregation for Head PoseEstimation from a Single Image**

**Code Author: Tsun-Yi Yang**

**Last update: 2019/03/27 (Upload training and testing. Not done yet!)**

### Comparison video
(Baseline **Hopenet:** https://github.com/natanielruiz/deep-head-pose)
<img src="https://github.com/shamangary/FSA-Net/blob/master/Compare_AFLW2000_gt_Hopenet_FSA.gif" height="320"/>


| Time sequence | Fine-grained structure|
| --- | --- |
| <img src="https://github.com/shamangary/FSA-Net/blob/master/time_demo.png" height="160"/> | <img src="https://github.com/shamangary/FSA-Net/blob/master/heatmap_demo.png" height="330"/> |



### Results
<img src="https://github.com/shamangary/FSA-Net/blob/master/FSANET_table1.png" height="220"/><img src="https://github.com/shamangary/FSA-Net/blob/master/FSANET_table2.png" height="220"/><img src="https://github.com/shamangary/FSA-Net/blob/master/FSANET_table3.png" height="220"/>


## Paper


### PDF
(Working on camera ready version)


### Paper authors
**[Tsun-Yi Yang](https://scholar.google.com/citations?user=WhISCE4AAAAJ&hl=en), [Yi-Ting Chen](https://sites.google.com/media.ee.ntu.edu.tw/yitingchen/), [Yen-Yu Lin](https://www.citi.sinica.edu.tw/pages/yylin/index_zh.html), and [Yung-Yu Chuang](https://www.csie.ntu.edu.tw/~cyy/)**


## Abstract
This paper proposes a method for head pose estimation from a single image. Previous methods often predicts head poses through landmark or depth estimation and would re- quire more computation than necessary. Our method is based on regression and feature aggregation. For having a compact model, we employ the soft stagewise regression scheme. Existing feature aggregation methods treat inputs as a bag of features and thus ignore their spatial relation- ship in a feature map. We propose to learn a fine-grained structure mapping for spatially grouping features before ag- gregation. The fine-grained structure provides part-based information and pooled values. By ultilizing learnable and non-learnable importance over the spatial location, differ- ent variant models as a complementary ensemble can be generated. Experiments show that out method outperforms the state-of-the-art methods including both the landmark- free ones and the ones based on landmark or depth esti- mation. Based on a single RGB frame as input, our method even outperforms methods utilizing multi-modality informa- tion (RGB-D, RGB-Time) on estimating the yaw angle. Fur- thermore, the memory overhead of the proposed model is 100× smaller than that of previous methods.

## Platform
+ Keras
+ Tensorflow
+ GTX-1080Ti
+ Ubuntu

## Dependencies
+ A guide for most dependencies. (in Chinese)
http://shamangary.logdown.com/posts/3009851
+ Anaconda
+ OpenCV
+ dlib
+ MTCNN
+ Capsule: https://github.com/XifengGuo/CapsNet-Keras
+ Loupe_Keras: https://github.com/shamangary/LOUPE_Keras

## Codes

There are three different section of this project. 
1. Data pre-processing
2. Training and testing
We will go through the details in the following sections.

This repository is for 300W-LP, AFLW2000, and BIWI datasets.


### 1. Data pre-processing
(Not done yet!)

### 2. Training and testing
(Not done yet!)
