import os
import cv2
import sys
import torch
import random
import itertools
import numpy as np
import pandas as pd
import ujson as json
from PIL import Image
from torchvision import transforms
from collections import defaultdict
from modules.basic_utils import load_json
from torch.utils.data import Dataset
from config.base_config import Config
from datasets.video_capture import VideoCapture


class Flickr30Kataset(Dataset):
    """
        videos_dir: directory where all videos are stored 
        config: AllConfig object
        split_type: 'train'/'test'
        img_transforms: Composition of transforms
    """
    def __init__(self, config: Config, split_type = 'train', img_transforms=None):
        self.videos_dir = config.videos_dir
        self.img_transforms = img_transforms
        self.split_type = split_type
        data_file = '/mnt/data10t/bakuphome20210617/zhangkun/data/f30k_precomp/dataset_flickr30k.json'

        self.data = load_json(data_file)
        self._construct_all_I_T_pairs()


    def __getitem__(self, index):
        image_path, caption, image_id = self._get_image_path_and_caption_by_index(index)
        imgs = (Image.open(image_path))
                                  

        # process images of video
        if self.img_transforms is not None:
            imgs = self.img_transforms(imgs).unsqueeze(0)

        return {
            'image_id': image_id,
            'image': imgs,
            'text': caption,
        }

    
    def __len__(self):
        if self.split_type == 'train':
            return len(self.all_train_pairs)
        return len(self.all_test_pairs)


    def _get_image_path_and_caption_by_index(self, index):
        # returns video path and caption as string
        if self.split_type == 'train':
            image_id, caption = self.all_train_pairs[index]
            image_path = os.path.join(self.videos_dir, image_id)
        else:
            image_id, caption = self.all_test_pairs[index]
            image_path = os.path.join(self.videos_dir, image_id)

        return image_path, caption, image_id
    


    def _construct_all_I_T_pairs(self):
        self.all_train_pairs = []
        self.all_test_pairs = []
        for i in range(len(self.data['images'])):
            if self.data['images'][i]['split'] == 'train':
                image_id = self.data['images'][i]['filename']
                for captions in self.data['images'][i]['sentences']:
                    caption = captions['raw'].strip('.').lower()
                    self.all_train_pairs.append([image_id, caption])
            else:
                if self.data['images'][i]['split'] == 'test':
                    image_id = self.data['images'][i]['filename']
                    for captions in self.data['images'][i]['sentences']:
                        caption = captions['raw'].strip('.').lower()
                        self.all_test_pairs.append([image_id, caption])
