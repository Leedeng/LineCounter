import os
import random
import argparse

parser = argparse.ArgumentParser(prog='LineCounterTrainer',
                                     description='this is the training script for the line counter segmenation')
parser.add_argument('--fontnum', dest='num', type=int, default=1,
                        help='the number of fonts')
parser.add_argument('--gpu', dest='gpu', type=int, default=0,
                        help='gpu device')
args = parser.parse_args()

file = open('ICDAR_valid.list', 'w')
img = []
gt=[]
for f in os.listdir("Dataset/Validation/source"):
    if f.endswith(".png"):
        img.append("Dataset/Validation/source/"+f)
        gt.append("Dataset/Validation/gt/"+f)
for i in range(len(img)):
    file.write(img[i]+" "+gt[i])
    if i < len(img)-1:
        file.write("\n")
file.close()

file = open('ICDAR_train.list', 'w')
datalist=[]
for f in os.listdir("Dataset/Training"):
    datalist.append(f)

datalist = random.sample(datalist, args.num)
img = []
gt=[]
count = 0
for f in os.listdir("Dataset/Training/Damion-Regular/source"):
    if f.endswith(".png"):
        index = count%len(datalist)
        img.append("Dataset/Training/"+datalist[index]+"/source/"+f)
        gt.append("Dataset/Training/"+datalist[index]+"/gt/"+f)
        count = count + 1
for i in range(len(img)):
    file.write(img[i]+" "+gt[i])
    if i < len(img)-1:
        file.write("\n")
file.close()

os.system('CUDA_VISIBLE_DEVICES='+str(args.gpu)+' python trainLineCounterV2.py ICDAR_train.list ICDAR_valid.list --learningRate 1e-4 --baseFilter 8 --counterMultiplier 8 --activation relu --numConvBlock 5 --downsampling drop --upsampling bilinear  --noiseRate 0.00  --useSympadding --target_size_height 1088 --target_size_width 768 --counterLocation before_decoder --exptDir ./expts/2024/ICDAR_1088x768x'+str(args.num)+' --patience 10')