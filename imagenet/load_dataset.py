import os 
import sys
import pickle
import numpy as np
import pandas as pd
import random
import torch
import pdb
from torch.utils.data import Dataset, DataLoader,RandomSampler
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import random
import skimage.util as ski_util
from sklearn.utils import shuffle
from copy import copy
import pdb


def load_video(anno_path,mode):

    # root_path = '/home/wuzongzhen/wzz-HarDvs/tools/data/HARD/HARDVS_eventIMGs_files'
    root_path = os.path.join(anno_path,'HARDVS_eventIMGs_files')
    labels = []
    rgb_samples = []
    anno_file = os.path.join(anno_path,'list','{}_label.txt'.format(mode))
    
    with open(anno_file, 'r') as fin:
        for line in fin:
           
            line_split = line.strip().split()
            idx = 0
            # frame_dir = line_split[idx]
            frame_dir = line_split[idx]+line_split[idx][10:]+'_dvs'
            img_list = os.listdir(os.path.join(root_path,frame_dir))
        
            
            # img_path = []
            label = line_split[idx+2]
            for img in img_list:
                
                rgb_samples.append(os.path.join(root_path,frame_dir,img))
                # img_path.sort()
                labels.append(label)

            # rgb_samples.append(img_path) 
    return rgb_samples, labels

# def load_video(annot_path, mode):
#     # mode: train, val, test
#     csv_file = os.path.join(annot_path, '{}.pkl'.format(mode))
#     annot_df = pd.read_pickle(csv_file)
#     rgb_samples = []
#     depth_samples = []
#     labels = []
#     for frame_i in range(annot_df.shape[0]):
#         rgb_list = annot_df['frame'].iloc[frame_i] # convert string in dataframe to list
#         rgb_samples.append(rgb_list)
#         labels.append(annot_df['label'].iloc[frame_i])
#     print('{}: {} videos have been loaded'.format(mode, len(rgb_samples)))
#     return rgb_samples, labels


class dataset_video(Dataset):
    def __init__(self, root_path, mode, spatial_transform=None):
        self.root_path = root_path
        self.rgb_samples, self.labels = load_video(root_path, mode)
        self.sample_num = len(self.rgb_samples)
        self.spatial_transform = spatial_transform

    def __getitem__(self, idx):
        rgb_name = self.rgb_samples[idx]
        label = self.labels[idx]
        # indices = [i for i in range(len(rgb_name))]
        # selected_indice = self.temporal_transform(indices)
        # clip_frames = []
        # for i, frame_name_i in enumerate(indices):
        rgb_cache = Image.open(rgb_name).convert("RGB")

        clip_frames = self.spatial_transform(rgb_cache)
        # n, h, w = clip_frames.size()
        return clip_frames, int(label)
    def __len__(self):
        return int(self.sample_num)
