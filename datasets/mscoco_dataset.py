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

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class MSCOCOdataset(Dataset):
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


        data_train = '/mnt/data2/zk/train_coco.json'
        data_dev = '/mnt/data2/zk/dev_coco.json'
        data_testall = '/mnt/data2/zk/testall_coco.json'

        self.data_train = load_json(data_train)
        self.data_dev = load_json(data_dev)
        self.data_testall = load_json(data_testall)

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
            videos_dir = self.videos_dir + 'train2014'
            image_path = os.path.join(videos_dir, image_id)
            if not os.path.exists(image_path):
                videos_dir = self.videos_dir + 'val2014'
                image_path = os.path.join(videos_dir, image_id)

        else:
            image_id, caption = self.all_test_pairs[index]
            videos_dir = self.videos_dir + 'val2014'
            image_path = os.path.join(videos_dir, image_id)

        return image_path, caption, image_id
    


    def _construct_all_I_T_pairs(self):
        self.all_train_pairs = []
        self.all_test_pairs = []

        if self.split_type == 'train':
            for i in range(len(self.data_train['images'])):
                image_id = self.data_train['images'][i]['file_name']
                for captions in self.data_train['images'][i]['sentences']:
                    caption = captions['raw'].lower() # .strip('.')
                    self.all_train_pairs.append([image_id, caption])

        else:
            if self.split_type == 'dev':
                for i in range(len(self.data_dev['images'])):
                    image_id = self.data_dev['images'][i]['file_name']
                    for captions in self.data_dev['images'][i]['sentences']:
                        caption = captions['raw'].lower() # .strip('.')
                        self.all_test_pairs.append([image_id, caption])
            else:
                for i in range(len(self.data_testall['images'])):
                    image_id = self.data_testall['images'][i]['file_name']
                    for captions in self.data_testall['images'][i]['sentences']:
                        caption = captions['raw'].lower() # .strip('.')
                        self.all_test_pairs.append([image_id, caption]) 

        print('done for constructing all I_T_pairs')  