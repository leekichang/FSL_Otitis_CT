import os
import cv2
import numpy as np

data_path = './dataset/'
f = open(f'./dataset/dataset_config.txt', 'r')
lines = f.readlines()
f.close()
label_dict = {}
image_dict = {}
for line in lines:
    cfgs = line.strip().split(' ')
    label_dict[cfgs[0]+'_'+cfgs[1]] = cfgs[2]
    imgs = []
    for image in os.listdir(f'{data_path}{cfgs[0]}/{cfgs[1]}'):
        img = cv2.imread(f'{data_path}{cfgs[0]}/{cfgs[1]}/{image}')      
        imgs.append(img.transpose((2, 0, 1)))
    image_dict[cfgs[0]+'_'+cfgs[1]] = np.array(imgs)

X = []
Y = []
keys = list(label_dict.keys())
for i in range(len(keys)):
    for j in range(i, len(keys)):
        pair = [image_dict[keys[i]], image_dict[keys[j]]]
        X.append(pair)
        
        if label_dict[keys[i]] == label_dict[keys[j]]:
            Y.append(1)
        else:
            Y.append(0)

X = np.array(X)
Y = np.array(Y).reshape(-1, 1)
np.save('./dataset/X.npy', X)
np.save('./dataset/Y.npy', Y)
    
    