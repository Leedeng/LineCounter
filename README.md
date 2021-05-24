# Line-Counter
# Step 1
Download trained-model
https://drive.google.com/file/d/1fMUkyg67QLLzyDMkU1vgnsgDIKb9SPF9/view?usp=sharing
***
Put it in **Line-Counter/expts/Allx768xDnCNN/models/BF32:BLK5:BN0,0:M8:LArelu:LCbefore_decoder:SC0:DSdrop:USbilinear:BD1:N0.00:P1/**   
***
# Step 2
Create virtual enviroment by requirment.txt
# Step 3
see example.ipynb


# LineCounter: Learning Handwritten Text Line Segmentation by Counting 

<div align="left">
    <img src="https://www.um.edu.mo/wp-content/uploads/2020/09/UM-Logo_V-Black-1024x813.png" width="30%"><img src="https://viplab.cis.um.edu.mo/images/logo_5.JPG" width="30%"><img src="https://2021.ieeeicip.org/images/ip21-logo.svg" width="30%">     
</div>

***

This is the official repo for the LineCounter (ICIP 2021). For details of LineCounter, please refer to 

```
@article{li2021sauvolanet,
  title={SauvolaNet: Learning Adaptive Sauvola Network for Degraded Document Binarization},
  author={Li, Deng and Wu, Yue and Zhou, Yicong},
  journal={arXiv preprint arXiv:2105.05521},
  year={2021}
}

```

***

# Overview

SauvolaNet is an end-to-end document binarization solutions. It optimal three hyper-parameters of classic Sauvola algriothim. Compare with exisiting solutions, SauvolaNet has follow advantages:

- **SauvolaNet do not have any Pre/Post-processing**
- **SauvolaNet has comparable performance with SoTA**
- **SauvolaNet has a super lightweight network stucture and faster than DNN-based SoTA**

<img src="https://github.com/Leedeng/SauvolaNet/blob/main/Image/FPS.png" width="50%">

More precisely, SauvolaNet consists of three modules namely, Multi-window Sauvola (MWS), Pixelwise Window Attention (PWA), and Adaptive Sauolva Threshold (AST).

- **MWS generates multiple windows of different size Sauvola with trainable parameters**
- **PWA generates pixelwise attention of window size**
- **AST generates pixelwise threshold by fusing the result of MWS and PWA.**

<img src="https://github.com/Leedeng/SauvolaNet/blob/main/Image/Structure2.png" width="50%">

# Dependency

SauvolaNet is written in the TensorFlow.
  
  - TensorFlow-GPU: 2.3.0
  
Other versions might also work, but are not tested.


# Demo

Donwload the repo and create virtual environment by follow commands

```
conda create --name Sauvola --file spec-env.txt
conda activate Sauvola
pip install tensorflow-gpu==2.3
pip install opencv-python
pip install pandas
pip install parse
```

Then play with the provided ipython notebook

Alternatively, one may play with the inference code using this [google colab link](https://colab.research.google.com/drive/1aGYXVRuTf1dhoKSsOCPcB4vKULtplFSA?usp=sharing).


# Concat

For any paper related questions, please feel free to contact leedengsh@gmail.com
