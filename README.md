# LineCounter: Learning Handwritten Text Line Segmentation by Counting 

<div align="center">
    <img src="https://www.um.edu.mo/wp-content/uploads/2020/09/UM-Logo_V-Black-1024x813.png" width="30%">
</div>

***

This is the official repo for LineCounter (ICIP 2021). For details of LineCounter, please refer to 

```
@INPROCEEDINGS{9506664,  
  author={Li, Deng and Wu, Yue and Zhou, Yicong},  
  booktitle={2021 IEEE International Conference on Image Processing (ICIP)},   
  title={Linecounter: Learning Handwritten Text Line Segmentation By Counting},   
  year={2021},  
  volume={},  
  number={},  
  pages={929-933},  
  doi={10.1109/ICIP42928.2021.9506664}}
```

***

# Overview


<img src="https://github.com/Leedeng/LineCounter/blob/master/image/Structure.png" width="80%">

# Dependency

SauvolaNet is written in the TensorFlow.
  
  - TensorFlow-GPU: 1.15.0
  
Other versions might also work but are not tested.


# Demo

Download the repo and create the virtual environment by following commands

```
conda create --name LineCounter --file requirements.txt
```

Download [trained-model](https://drive.google.com/file/d/1fMUkyg67QLLzyDMkU1vgnsgDIKb9SPF9/view?usp=sharing)

***
Put trained-model in **LineCounter/expts/Allx768xDnCNN/models/BF32:BLK5:BN0,0:M8:LArelu:LCbefore_decoder:SC0:DSdrop:USbilinear:BD1:N0.00:P1/**   
***

Then play with the provided ipython notebook.

Alternatively, one may play with the inference code using this [google colab link](https://colab.research.google.com/drive/1v-h7eSNhxfCTqQZC_IPGEp_s-sfA6dxn?usp=sharing).

# Datasets
We do not own the copyright of the dataset used in this repo.

Below is a summary table of the datasets used in this work along with a link from which they can be downloaded:


| Dataset      | URL     |
| ------------ | ------- |
| ICDAR-HCS2013    | https://users.iit.demokritos.gr/~nstam/ICDAR2013HandSegmCont/   |
| HIT-MW | http://space.hit.edu.cn/article/2019/03/11/10660 (Chinese) |
| VML-AHTE | https://www.cs.bgu.ac.il/~berat/ |

# Concat

For any paper related questions, please feel free to contact leedengsh@gmail.com.
