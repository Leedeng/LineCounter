# LineCounter: Learning Handwritten Text Line Segmentation by Counting 

<div align="left">
    <img src="https://www.um.edu.mo/wp-content/uploads/2020/09/UM-Logo_V-Black-1024x813.png" width="30%"><img src="https://viplab.cis.um.edu.mo/images/logo_5.JPG" width="30%"><img src="https://2021.ieeeicip.org/images/ip21-logo.svg" width="25%">     
</div>

***

This is the official repo for LineCounter (ICIP 2021). For details of LineCounter, please refer to 

```
@misc{li2021linecounter,
      title={LineCounter: Learning Handwritten Text Line Segmentation by Counting}, 
      author={Deng Li and Yue Wu and Yicong Zhou},
      year={2021},
      eprint={2105.11307},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

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
| DIBCO 2009   | http://users.iit.demokritos.gr/~bgat/DIBCO2009/benchmark/   |
| DIBCO 2010   | http://users.iit.demokritos.gr/~bgat/H-DIBCO2010/benchmark/ |
| DIBCO 2011   | http://utopia.duth.gr/~ipratika/DIBCO2011/benchmark/        |
| DIBCO 2012   | http://utopia.duth.gr/~ipratika/HDIBCO2012/benchmark/       |
| DIBCO 2013   | http://utopia.duth.gr/~ipratika/DIBCO2013/benchmark/        |
| DIBCO 2014   | http://users.iit.demokritos.gr/~bgat/HDIBCO2014/benchmark/  |
| DIBCO 2016   | http://vc.ee.duth.gr/h-dibco2016/benchmark/                 |
| DIBCO 2017   | https://vc.ee.duth.gr/dibco2017/                 |
| DIBCO 2018   | https://vc.ee.duth.gr/h-dibco2018/                 |
| PHIDB | http://www.iapr-tc11.org/mediawiki/index.php/Persian_Heritage_Image_Binarization_Dataset_(PHIBD_2012) |
| Bickely-diary dataset | https://www.comp.nus.edu.sg/~brown/BinarizationShop/dataset.htm |
| Synchromedia Multispectral dataset | http://tc11.cvc.uab.es/datasets/SMADI_1　|
| Monk Cuper Set | https://www.ai.rug.nl/~sheng/ |

# Concat

For any paper related questions, please feel free to contact leedengsh@gmail.com.
