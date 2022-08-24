import os
import cv2
import torch
import models
from FslImageSet import *
import numpy as np
import config as cfg
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torchvision.transforms as transforms

def get_imgs(path):
    imgs = []
    for file in os.listdir(path):
        imgs.append(cv2.imread(path+file).transpose(2, 0, 1))
    return imgs
def get_support_set(data_path):
    f = open('./datasets/dataset_config.txt', 'r')
    lines = f.readlines()
    f.close()
    configs = {}
    for line in lines:
        data_id, is_left, label = line.strip().split(' ')
        configs[f'{data_id}_{is_left}'] = label
    
    support_set = []
    for key in configs.keys():
        id, side = key.split('_')
        imgs = get_imgs(f'{data_path}{id}/{side}/')
        support_set.append(FslImageSet(f'{id}_{side}', configs[f'{id}_{side}'], torch.from_numpy(np.array(imgs)).float()))
    return support_set

def get_support_set_feature(model, support_set):
    support_set_features = {'normal':[], 'moderate':[], 'severe':[]}
    for data in support_set:
        X = data.imgs.to('cuda')
        feature = model.custom_forward(X)
        support_set_features[data.label].append(feature.flatten(0, 2).reshape(1, -1))
    return support_set_features
    
def get_query_set(data_path):
    folders = [folder for folder in os.listdir(data_path) if os.path.isdir(data_path+folder)]
    query_set = []
    for folder in folders:
        imgs = get_imgs(data_path+folder+f'/left/')
        query_set.append(FslImageSet(folder+'_left', None, torch.from_numpy(np.array(imgs)).float()))
        imgs = get_imgs(data_path+folder+f'/right/')
        query_set.append(FslImageSet(folder+'_right', None, torch.from_numpy(np.array(imgs)).float()))
    return query_set

def infer(model, query_set, support_set_feature):
    result = {}
    result_value = {}
    for query in query_set:
        X = query.imgs.to('cuda')
        query_feature = model.custom_forward(X)
        result_detail = {subject:[] for subject in support_set_features.keys()}
        max_sim = 0
        max_sim_label = None
        for label in support_set_features.keys():
            for support_feature in support_set_features[label]:
                sim = model.cosine_sim(query_feature.flatten(0, 2).reshape(1, -1), support_feature)
                result_detail[label].append(sim.detach().cpu().numpy())
            if np.mean(result_detail[label]) >= max_sim:
                max_sim = np.mean(result_detail[label])
                max_sim_label = label
        result[query.id] = max_sim_label
        result_value[query.id] = max_sim
    return result, result_value

if __name__ == '__main__':
    support_set_path = './datasets/dataset_crop_144/'
    query_set_path = './datasets/dataset_unlabeled_crop_144/'
    checkpoint = './checkpoints/resnet50_cosine_sim_144_bal_lr-1_scheduler/15.pth'
    model = models.Cosine_ResNet50()
    model.embedding_layer.load_state_dict(torch.load(checkpoint))
    support_set = get_support_set(support_set_path)
    support_set_features = get_support_set_feature(model, support_set)
    query_set = get_query_set(query_set_path)
    
    with torch.no_grad():
        result, result_value = infer(model, query_set, support_set_features)
    for key in result.keys():
        print(key, result[key], result_value[key])
    
    