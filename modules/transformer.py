import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.base_config import Config
import numpy as np
import os

def logging_func(log_file, message):
    with open(log_file,'a') as f:
        f.write(message)
    f.close()

def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


########################################################
### For the dimensional selective mask, we design both heuristic and adaptive strategies. 
### You can use this flag to control which strategy is selected. True -> heuristic strategy, False -> adaptive strategy

heuristic_strategy = True
########################################################


if heuristic_strategy:
    ### Heuristic Dimensional Selective Mask
    class Image_levels(nn.Module):
        def __init__(self, config: Config):
            super(Image_levels, self).__init__()
            self.sub_space = config.embed_dim
            self.kernel_size = int(config.kernel_size)

            self.kernel_img_1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=self.kernel_size, stride=self.kernel_size, bias=False)
            self.kernel_img_2 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=self.kernel_size, stride=self.kernel_size, bias=False)
            self.kernel_img_3 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=self.kernel_size, stride=self.kernel_size, bias=False)
            self.kernel_img_4 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=self.kernel_size, stride=self.kernel_size, bias=False)
            self.kernel_img_5 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=self.kernel_size, stride=self.kernel_size, bias=False)
            self.kernel_img_6 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=self.kernel_size, stride=self.kernel_size, bias=False)
            self.kernel_img_7 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=self.kernel_size, stride=self.kernel_size, bias=False)
            self.kernel_img_8 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=self.kernel_size, stride=self.kernel_size, bias=False)


        def get_image_levels(self, img_emb, batch_size):
            img_emb_1 = self.kernel_img_1(img_emb.unsqueeze(-2)).sum(1)
            img_emb_1 = l2norm(img_emb_1, -1)

            img_emb_2 = self.kernel_img_2(img_emb_1.unsqueeze(-2)).sum(1)
            img_emb_2 = l2norm(img_emb_2, -1)

            img_emb_3 = self.kernel_img_3(img_emb_2.unsqueeze(-2)).sum(1)
            img_emb_3 = l2norm(img_emb_3, -1)

            img_emb_4 = self.kernel_img_4(img_emb_3.unsqueeze(-2)).sum(1)
            img_emb_4 = l2norm(img_emb_4, -1)

            img_emb_5 = self.kernel_img_5(img_emb_4.unsqueeze(-2)).sum(1)
            img_emb_5 = l2norm(img_emb_5, -1)

            img_emb_6 = self.kernel_img_6(img_emb_5.unsqueeze(-2)).sum(1)
            img_emb_6 = l2norm(img_emb_6, -1)

            img_emb_7 = self.kernel_img_7(img_emb_6.unsqueeze(-2)).sum(1)
            img_emb_7 = l2norm(img_emb_7, -1)

            img_emb_8 = self.kernel_img_8(img_emb_7.unsqueeze(-2)).sum(1)
            img_emb_8 = l2norm(img_emb_8, -1)


            return torch.cat([img_emb, img_emb_1, img_emb_2, img_emb_3, img_emb_4, img_emb_5, img_emb_6, img_emb_7, img_emb_8], -1)

        def forward(self, img_emb):
        
            batch_size = img_emb.size(0)

            return self.get_image_levels(img_emb, batch_size)


    class Text_levels(nn.Module):
        
        def __init__(self, config: Config):
            super(Text_levels, self).__init__()
            self.sub_space = config.embed_dim
            self.kernel_size = int(config.kernel_size)

            self.kernel_txt_1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=self.kernel_size, stride=self.kernel_size, bias=False)
            self.kernel_txt_2 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=self.kernel_size, stride=self.kernel_size, bias=False)
            self.kernel_txt_3 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=self.kernel_size, stride=self.kernel_size, bias=False)
            self.kernel_txt_4 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=self.kernel_size, stride=self.kernel_size, bias=False)
            self.kernel_txt_5 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=self.kernel_size, stride=self.kernel_size, bias=False)
            self.kernel_txt_6 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=self.kernel_size, stride=self.kernel_size, bias=False)
            self.kernel_txt_7 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=self.kernel_size, stride=self.kernel_size, bias=False)
            self.kernel_txt_8 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=self.kernel_size, stride=self.kernel_size, bias=False)


        def get_text_levels(self, cap_i):
            cap_i_1 = self.kernel_txt_1(cap_i.unsqueeze(-2)).sum(1)
            cap_i_expand_1 = l2norm(cap_i_1, -1)

            cap_i_2 = self.kernel_txt_2(cap_i_1.unsqueeze(-2)).sum(1)
            cap_i_expand_2 = l2norm(cap_i_2, -1)

            cap_i_3 = self.kernel_txt_3(cap_i_2.unsqueeze(-2)).sum(1)
            cap_i_expand_3 = l2norm(cap_i_3, -1)

            cap_i_4 = self.kernel_txt_4(cap_i_3.unsqueeze(-2)).sum(1)
            cap_i_expand_4 = l2norm(cap_i_4, -1)

            cap_i_5 = self.kernel_txt_5(cap_i_4.unsqueeze(-2)).sum(1)
            cap_i_expand_5 = l2norm(cap_i_5, -1)

            cap_i_6 = self.kernel_txt_6(cap_i_5.unsqueeze(-2)).sum(1)
            cap_i_expand_6 = l2norm(cap_i_6, -1)

            cap_i_7 = self.kernel_txt_7(cap_i_6.unsqueeze(-2)).sum(1)
            cap_i_expand_7 = l2norm(cap_i_7, -1)

            cap_i_8 = self.kernel_txt_8(cap_i_7.unsqueeze(-2)).sum(1)
            cap_i_expand_8 = l2norm(cap_i_8, -1)

            return torch.cat([cap_i, cap_i_expand_1, cap_i_expand_2, cap_i_expand_3, cap_i_expand_4, cap_i_expand_5, cap_i_expand_6, cap_i_expand_7, cap_i_expand_8], -1)

        def forward(self, cap_i):
        
            return self.get_text_levels(cap_i)

else:

    #### Adaptive Dimensional Selective Mask
    class Image_levels(nn.Module):
        def __init__(self, config: Config):
            super(Image_levels, self).__init__()
            self.sub_space = config.embed_dim
            self.kernel_size = int(config.kernel_size)
            self.out_channels = 2

            self.masks_1 = torch.nn.Embedding(int(self.sub_space/math.pow(self.kernel_size, 1)), int(self.sub_space/math.pow(self.kernel_size, 0))) # num_embedding, dims_input
            self.masks_2 = torch.nn.Embedding(int(self.sub_space/math.pow(self.kernel_size, 2)), int(self.sub_space/math.pow(self.kernel_size, 1)))
            self.masks_3 = torch.nn.Embedding(int(self.sub_space/math.pow(self.kernel_size, 3)), int(self.sub_space/math.pow(self.kernel_size, 2)))
            self.masks_4 = torch.nn.Embedding(int(self.sub_space/math.pow(self.kernel_size, 4)), int(self.sub_space/math.pow(self.kernel_size, 3)))
            self.masks_5 = torch.nn.Embedding(int(self.sub_space/math.pow(self.kernel_size, 5)), int(self.sub_space/math.pow(self.kernel_size, 4)))
            self.masks_6 = torch.nn.Embedding(int(self.sub_space/math.pow(self.kernel_size, 6)), int(self.sub_space/math.pow(self.kernel_size, 5)))
            self.masks_7 = torch.nn.Embedding(int(self.sub_space/math.pow(self.kernel_size, 7)), int(self.sub_space/math.pow(self.kernel_size, 6)))
            self.masks_8 = torch.nn.Embedding(int(self.sub_space/math.pow(self.kernel_size, 8)), int(self.sub_space/math.pow(self.kernel_size, 7)))


        def get_image_levels(self, img_emb):

            sub_space_index = torch.tensor(torch.linspace(0, 1024, steps=1024, dtype=torch.int)).cuda()
            Dim_learned_mask_1 = l2norm(self.masks_1(sub_space_index[:int(self.sub_space/math.pow(self.kernel_size, 1))]), dim=-1)
            Dim_learned_mask_2 = l2norm(self.masks_2(sub_space_index[:int(self.sub_space/math.pow(self.kernel_size, 2))]), dim=-1)
            Dim_learned_mask_3 = l2norm(self.masks_3(sub_space_index[:int(self.sub_space/math.pow(self.kernel_size, 3))]), dim=-1)
            Dim_learned_mask_4 = l2norm(self.masks_4(sub_space_index[:int(self.sub_space/math.pow(self.kernel_size, 4))]), dim=-1)
            Dim_learned_mask_5 = l2norm(self.masks_5(sub_space_index[:int(self.sub_space/math.pow(self.kernel_size, 5))]), dim=-1)
            Dim_learned_mask_6 = l2norm(self.masks_6(sub_space_index[:int(self.sub_space/math.pow(self.kernel_size, 6))]), dim=-1)
            Dim_learned_mask_7 = l2norm(self.masks_7(sub_space_index[:int(self.sub_space/math.pow(self.kernel_size, 7))]), dim=-1)
            Dim_learned_mask_8 = l2norm(self.masks_8(sub_space_index[:int(self.sub_space/math.pow(self.kernel_size, 8))]), dim=-1)


            if Dim_learned_mask_1.size(1) < self.out_channels:
                select_nums = Dim_learned_mask_1.size(1)
            else:
                select_nums = self.out_channels
            Dim_learned_range = Dim_learned_mask_1.sort(1, descending =True)[0][:, select_nums -1].unsqueeze(-1)
            Dim_learned_mask_1 = (Dim_learned_mask_1 >= Dim_learned_range).float() * Dim_learned_mask_1


            if Dim_learned_mask_2.size(1) < self.out_channels:
                select_nums = Dim_learned_mask_2.size(1)
            else:
                select_nums = self.out_channels
            Dim_learned_range = Dim_learned_mask_2.sort(1, descending =True)[0][:, select_nums -1].unsqueeze(-1)
            Dim_learned_mask_2 = (Dim_learned_mask_2 >= Dim_learned_range).float() * Dim_learned_mask_2


            if Dim_learned_mask_3.size(1) < self.out_channels:
                select_nums = Dim_learned_mask_3.size(1)
            else:
                select_nums = self.out_channels
            Dim_learned_range = Dim_learned_mask_3.sort(1, descending =True)[0][:, select_nums -1].unsqueeze(-1)
            Dim_learned_mask_3 = (Dim_learned_mask_3 >= Dim_learned_range).float() * Dim_learned_mask_3


            if Dim_learned_mask_4.size(1) < self.out_channels:
                select_nums = Dim_learned_mask_4.size(1)
            else:
                select_nums = self.out_channels
            Dim_learned_range = Dim_learned_mask_4.sort(1, descending =True)[0][:, select_nums -1].unsqueeze(-1)
            Dim_learned_mask_4 = (Dim_learned_mask_4 >= Dim_learned_range).float() * Dim_learned_mask_4


            if Dim_learned_mask_5.size(1) < self.out_channels:
                select_nums = Dim_learned_mask_5.size(1)
            else:
                select_nums = self.out_channels
            Dim_learned_range = Dim_learned_mask_5.sort(1, descending =True)[0][:, select_nums -1].unsqueeze(-1)
            Dim_learned_mask_5 = (Dim_learned_mask_5 >= Dim_learned_range).float() * Dim_learned_mask_5


            if Dim_learned_mask_6.size(1) < self.out_channels:
                select_nums = Dim_learned_mask_6.size(1)
            else:
                select_nums = self.out_channels
            Dim_learned_range = Dim_learned_mask_6.sort(1, descending =True)[0][:, select_nums -1].unsqueeze(-1)
            Dim_learned_mask_6 = (Dim_learned_mask_6 >= Dim_learned_range).float() * Dim_learned_mask_6


            if Dim_learned_mask_7.size(1) < self.out_channels:
                select_nums = Dim_learned_mask_7.size(1)
            else:
                select_nums = self.out_channels
            Dim_learned_range = Dim_learned_mask_7.sort(1, descending =True)[0][:, select_nums -1].unsqueeze(-1)
            Dim_learned_mask_7 = (Dim_learned_mask_7 >= Dim_learned_range).float() * Dim_learned_mask_7


            if Dim_learned_mask_8.size(1) < self.out_channels:
                select_nums = Dim_learned_mask_8.size(1)
            else:
                select_nums = self.out_channels
            Dim_learned_range = Dim_learned_mask_8.sort(1, descending =True)[0][:, select_nums -1].unsqueeze(-1)
            Dim_learned_mask_8 = (Dim_learned_mask_8 >= Dim_learned_range).float() * Dim_learned_mask_8


            img_emb_1 = img_emb.reshape(-1, self.sub_space) @ Dim_learned_mask_1.t()
            img_emb_1 = l2norm(img_emb_1, -1)

            emb_size = img_emb_1.size(-1)
            img_emb_2 = img_emb_1.reshape(-1, emb_size) @ Dim_learned_mask_2.t()
            img_emb_2 = l2norm(img_emb_2, -1)

            emb_size = img_emb_2.size(-1)
            img_emb_3 = img_emb_2.reshape(-1, emb_size) @ Dim_learned_mask_3.t()
            img_emb_3 = l2norm(img_emb_3, -1)

            emb_size = img_emb_3.size(-1)
            img_emb_4 = img_emb_3.reshape(-1, emb_size) @ Dim_learned_mask_4.t()
            img_emb_4 = l2norm(img_emb_4, -1)

            emb_size = img_emb_4.size(-1)
            img_emb_5 = img_emb_4.reshape(-1, emb_size) @ Dim_learned_mask_5.t()
            img_emb_5 = l2norm(img_emb_5, -1)

            emb_size = img_emb_5.size(-1)
            img_emb_6 = img_emb_5.reshape(-1, emb_size) @ Dim_learned_mask_6.t()
            img_emb_6 = l2norm(img_emb_6, -1)

            emb_size = img_emb_6.size(-1)
            img_emb_7 = img_emb_6.reshape(-1, emb_size) @ Dim_learned_mask_7.t()
            img_emb_7 = l2norm(img_emb_7, -1)

            emb_size = img_emb_7.size(-1)
            img_emb_8 = img_emb_7.reshape(-1, emb_size) @ Dim_learned_mask_8.t()
            img_emb_8 = l2norm(img_emb_8, -1)


            return torch.cat([img_emb, img_emb_1, img_emb_2, img_emb_3, img_emb_4, img_emb_5, img_emb_6, img_emb_7, img_emb_8], -1)

        def forward(self, img_emb):
        
            return self.get_image_levels(img_emb)


    class Text_levels(nn.Module):

        def __init__(self, config: Config):
            super(Text_levels, self).__init__()
            self.sub_space = config.embed_dim
            self.kernel_size = int(config.kernel_size)
            self.out_channels = 2

            self.masks_1 = torch.nn.Embedding(int(self.sub_space/math.pow(self.kernel_size, 1)), int(self.sub_space/math.pow(self.kernel_size, 0))) # num_embedding, dims_input
            self.masks_2 = torch.nn.Embedding(int(self.sub_space/math.pow(self.kernel_size, 2)), int(self.sub_space/math.pow(self.kernel_size, 1)))
            self.masks_3 = torch.nn.Embedding(int(self.sub_space/math.pow(self.kernel_size, 3)), int(self.sub_space/math.pow(self.kernel_size, 2)))
            self.masks_4 = torch.nn.Embedding(int(self.sub_space/math.pow(self.kernel_size, 4)), int(self.sub_space/math.pow(self.kernel_size, 3)))
            self.masks_5 = torch.nn.Embedding(int(self.sub_space/math.pow(self.kernel_size, 5)), int(self.sub_space/math.pow(self.kernel_size, 4)))
            self.masks_6 = torch.nn.Embedding(int(self.sub_space/math.pow(self.kernel_size, 6)), int(self.sub_space/math.pow(self.kernel_size, 5)))
            self.masks_7 = torch.nn.Embedding(int(self.sub_space/math.pow(self.kernel_size, 7)), int(self.sub_space/math.pow(self.kernel_size, 6)))
            self.masks_8 = torch.nn.Embedding(int(self.sub_space/math.pow(self.kernel_size, 8)), int(self.sub_space/math.pow(self.kernel_size, 7)))

        def get_text_levels(self, cap_i):

            sub_space_index = torch.tensor(torch.linspace(0, 1024, steps=1024, dtype=torch.int)).cuda()
            Dim_learned_mask_1 = l2norm(self.masks_1(sub_space_index[:int(self.sub_space/math.pow(self.kernel_size, 1))]), dim=-1)
            Dim_learned_mask_2 = l2norm(self.masks_2(sub_space_index[:int(self.sub_space/math.pow(self.kernel_size, 2))]), dim=-1)
            Dim_learned_mask_3 = l2norm(self.masks_3(sub_space_index[:int(self.sub_space/math.pow(self.kernel_size, 3))]), dim=-1)
            Dim_learned_mask_4 = l2norm(self.masks_4(sub_space_index[:int(self.sub_space/math.pow(self.kernel_size, 4))]), dim=-1)
            Dim_learned_mask_5 = l2norm(self.masks_5(sub_space_index[:int(self.sub_space/math.pow(self.kernel_size, 5))]), dim=-1)
            Dim_learned_mask_6 = l2norm(self.masks_6(sub_space_index[:int(self.sub_space/math.pow(self.kernel_size, 6))]), dim=-1)
            Dim_learned_mask_7 = l2norm(self.masks_7(sub_space_index[:int(self.sub_space/math.pow(self.kernel_size, 7))]), dim=-1)
            Dim_learned_mask_8 = l2norm(self.masks_8(sub_space_index[:int(self.sub_space/math.pow(self.kernel_size, 8))]), dim=-1)


            if Dim_learned_mask_1.size(1) < self.out_channels:
                select_nums = Dim_learned_mask_1.size(1)
            else:
                select_nums = self.out_channels
            Dim_learned_range = Dim_learned_mask_1.sort(1, descending =True)[0][:, select_nums -1].unsqueeze(-1)
            Dim_learned_mask_1 = (Dim_learned_mask_1 >= Dim_learned_range).float() * Dim_learned_mask_1


            if Dim_learned_mask_2.size(1) < self.out_channels:
                select_nums = Dim_learned_mask_2.size(1)
            else:
                select_nums = self.out_channels
            Dim_learned_range = Dim_learned_mask_2.sort(1, descending =True)[0][:, select_nums -1].unsqueeze(-1)
            Dim_learned_mask_2 = (Dim_learned_mask_2 >= Dim_learned_range).float() * Dim_learned_mask_2


            if Dim_learned_mask_3.size(1) < self.out_channels:
                select_nums = Dim_learned_mask_3.size(1)
            else:
                select_nums = self.out_channels
            Dim_learned_range = Dim_learned_mask_3.sort(1, descending =True)[0][:, select_nums -1].unsqueeze(-1)
            Dim_learned_mask_3 = (Dim_learned_mask_3 >= Dim_learned_range).float() * Dim_learned_mask_3


            if Dim_learned_mask_4.size(1) < self.out_channels:
                select_nums = Dim_learned_mask_4.size(1)
            else:
                select_nums = self.out_channels
            Dim_learned_range = Dim_learned_mask_4.sort(1, descending =True)[0][:, select_nums -1].unsqueeze(-1)
            Dim_learned_mask_4 = (Dim_learned_mask_4 >= Dim_learned_range).float() * Dim_learned_mask_4


            if Dim_learned_mask_5.size(1) < self.out_channels:
                select_nums = Dim_learned_mask_5.size(1)
            else:
                select_nums = self.out_channels
            Dim_learned_range = Dim_learned_mask_5.sort(1, descending =True)[0][:, select_nums -1].unsqueeze(-1)
            Dim_learned_mask_5 = (Dim_learned_mask_5 >= Dim_learned_range).float() * Dim_learned_mask_5


            if Dim_learned_mask_6.size(1) < self.out_channels:
                select_nums = Dim_learned_mask_6.size(1)
            else:
                select_nums = self.out_channels
            Dim_learned_range = Dim_learned_mask_6.sort(1, descending =True)[0][:, select_nums -1].unsqueeze(-1)
            Dim_learned_mask_6 = (Dim_learned_mask_6 >= Dim_learned_range).float() * Dim_learned_mask_6


            if Dim_learned_mask_7.size(1) < self.out_channels:
                select_nums = Dim_learned_mask_7.size(1)
            else:
                select_nums = self.out_channels
            Dim_learned_range = Dim_learned_mask_7.sort(1, descending =True)[0][:, select_nums -1].unsqueeze(-1)
            Dim_learned_mask_7 = (Dim_learned_mask_7 >= Dim_learned_range).float() * Dim_learned_mask_7


            if Dim_learned_mask_8.size(1) < self.out_channels:
                select_nums = Dim_learned_mask_8.size(1)
            else:
                select_nums = self.out_channels
            Dim_learned_range = Dim_learned_mask_8.sort(1, descending =True)[0][:, select_nums -1].unsqueeze(-1)
            Dim_learned_mask_8 = (Dim_learned_mask_8 >= Dim_learned_range).float() * Dim_learned_mask_8

            

            cap_i_1 = cap_i.reshape(-1, self.sub_space) @ Dim_learned_mask_1.t()
            cap_i_expand_1 = l2norm(cap_i_1, -1)

            emb_size = cap_i_expand_1.size(-1)
            cap_i_2 = cap_i_1.reshape(-1, emb_size) @ Dim_learned_mask_2.t()
            cap_i_expand_2 = l2norm(cap_i_2, -1)

            emb_size = cap_i_expand_2.size(-1)
            cap_i_3 = cap_i_2.reshape(-1, emb_size) @ Dim_learned_mask_3.t()
            cap_i_expand_3 = l2norm(cap_i_3, -1)

            emb_size = cap_i_expand_3.size(-1)
            cap_i_4 = cap_i_3.reshape(-1, emb_size) @ Dim_learned_mask_4.t()
            cap_i_expand_4 = l2norm(cap_i_4, -1)

            emb_size = cap_i_expand_4.size(-1)
            cap_i_5 = cap_i_4.reshape(-1, emb_size) @ Dim_learned_mask_5.t()
            cap_i_expand_5 = l2norm(cap_i_5, -1)

            emb_size = cap_i_expand_5.size(-1)
            cap_i_6 = cap_i_5.reshape(-1, emb_size) @ Dim_learned_mask_6.t()
            cap_i_expand_6 = l2norm(cap_i_6, -1)

            emb_size = cap_i_expand_6.size(-1)
            cap_i_7 = cap_i_6.reshape(-1, emb_size) @ Dim_learned_mask_7.t()
            cap_i_expand_7 = l2norm(cap_i_7, -1)

            emb_size = cap_i_expand_7.size(-1)
            cap_i_8 = cap_i_7.reshape(-1, emb_size) @ Dim_learned_mask_8.t()
            cap_i_expand_8 = l2norm(cap_i_8, -1)

            return torch.cat([cap_i, cap_i_expand_1, cap_i_expand_2, cap_i_expand_3, cap_i_expand_4, cap_i_expand_5, cap_i_expand_6, cap_i_expand_7, cap_i_expand_8], -1)



        def forward(self, cap_i):

            return self.get_text_levels(cap_i)



class Image_Text_Encoders(nn.Module):
    def __init__(self, config: Config):

        super(Image_Text_Encoders, self).__init__()
        self.text_levels = Text_levels(config)
        self.image_levels = Image_levels(config)

    def forward(self, images, captions, return_type):

        if return_type ==  'image':
            img_embs = self.image_levels(images)
            return img_embs
        else:
            cap_embs = self.text_levels(captions)
            return cap_embs

class Image_Text_Processing(nn.Module):
    
    def __init__(self, config: Config):
        super(Image_Text_Processing, self).__init__()
        self.encoders_1 = Image_Text_Encoders(config)

    def forward(self, images, captions):

        image_processed = self.encoders_1(images, captions, 'image')
        text_processed = self.encoders_1(images, captions, 'text')

        return image_processed, text_processed











class sims_claculator(nn.Module):
    def __init__(self, opt: Config, name):
        super(sims_claculator, self).__init__()
        self.sub_space = opt.embed_dim
        self.kernel_size = int(opt.kernel_size)
        self.claculator_name = name
        self.opt = opt


        self.sim_eval = nn.Linear(9, 1, bias=False)
        self.temp_scale = nn.Linear(1, 1, bias=False)
        self.temp_scale_1 = nn.Linear(1, 1, bias=False)
        self.temp_scale_2 = nn.Linear(1, 1, bias=False)

        self.list_length = [int(opt.embed_dim/math.pow(self.kernel_size, 0)), int(opt.embed_dim/math.pow(self.kernel_size, 1)), int(opt.embed_dim/math.pow(self.kernel_size, 2)),
                            int(opt.embed_dim/math.pow(self.kernel_size, 3)), int(opt.embed_dim/math.pow(self.kernel_size, 4)), int(opt.embed_dim/math.pow(self.kernel_size, 5)), 
                            int(opt.embed_dim/math.pow(self.kernel_size, 6)), int(opt.embed_dim/math.pow(self.kernel_size, 7)), int(opt.embed_dim/math.pow(self.kernel_size, 8))]


        self.masks_0 = nn.Linear(int(opt.embed_dim/math.pow(self.kernel_size, 0)), int(opt.embed_dim/math.pow(self.kernel_size, 0)), bias=False)
        self.masks_1 = nn.Linear(int(opt.embed_dim/math.pow(self.kernel_size, 1)), int(opt.embed_dim/math.pow(self.kernel_size, 1)), bias=False)
        self.masks_2 = nn.Linear(int(opt.embed_dim/math.pow(self.kernel_size, 2)), int(opt.embed_dim/math.pow(self.kernel_size, 2)), bias=False)
        self.masks_3 = nn.Linear(int(opt.embed_dim/math.pow(self.kernel_size, 3)), int(opt.embed_dim/math.pow(self.kernel_size, 3)), bias=False)
        self.masks_4 = nn.Linear(int(opt.embed_dim/math.pow(self.kernel_size, 4)), int(opt.embed_dim/math.pow(self.kernel_size, 4)), bias=False)
        self.masks_5 = nn.Linear(int(opt.embed_dim/math.pow(self.kernel_size, 5)), int(opt.embed_dim/math.pow(self.kernel_size, 5)), bias=False)
        self.masks_6 = nn.Linear(int(opt.embed_dim/math.pow(self.kernel_size, 6)), int(opt.embed_dim/math.pow(self.kernel_size, 6)), bias=False)
        self.masks_7 = nn.Linear(int(opt.embed_dim/math.pow(self.kernel_size, 7)), int(opt.embed_dim/math.pow(self.kernel_size, 7)), bias=False)
        self.masks_8 = nn.Linear(int(opt.embed_dim/math.pow(self.kernel_size, 8)), int(opt.embed_dim/math.pow(self.kernel_size, 8)), bias=False)

        self.init_weights()

    def init_weights(self):
        self.temp_scale.weight.data.fill_(np.log(1 / 0.07)) 
        self.sim_eval.weight.data.fill_(0.1) 
        self.temp_scale_1.weight.data.fill_(-1) 
        self.temp_scale_2.weight.data.fill_(4) 

        for name, param in self.named_parameters():
            if 'mask' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)


    def get_sims_levels(self, X_0, X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8,
                        Y_0, Y_1, Y_2, Y_3, Y_4, Y_5, Y_6, Y_7, Y_8,
                        D_0, D_1, D_2, D_3, D_4, D_5, D_6, D_7, D_8):
        attn_0 = (X_0 @ D_0 @ Y_0.transpose(0, 1)) 
        attn_1 = (X_1 @ D_1 @ Y_1.transpose(0, 1))
        attn_2 = (X_2 @ D_2 @ Y_2.transpose(0, 1))
        attn_3 = (X_3 @ D_3 @ Y_3.transpose(0, 1))
        attn_4 = (X_4 @ D_4 @ Y_4.transpose(0, 1))
        attn_5 = (X_5 @ D_5 @ Y_5.transpose(0, 1))
        attn_6 = (X_6 @ D_6 @ Y_6.transpose(0, 1))
        attn_7 = (X_7 @ D_7 @ Y_7.transpose(0, 1))
        attn_8 = (X_8 @ D_8 @ Y_8.transpose(0, 1))

        attn = attn_0 + attn_1 + attn_2 + attn_3 + attn_4 + attn_5 + attn_6 + attn_7 + attn_8

        return attn /9


    def forward(self, img_emb, cap_emb):
        
        sigma_ = self.temp_scale_1.weight
        lambda_ = torch.exp(self.temp_scale_2.weight)
        threshold = (torch.abs(self.sim_eval.weight).max() - torch.abs(self.sim_eval.weight).min()) * torch.sigmoid(sigma_) + torch.abs(self.sim_eval.weight).min()

        weight_0 = (torch.exp((torch.abs(self.sim_eval.weight[0, 0]) - threshold) * lambda_) * self.sim_eval.weight[0, 0])
        weight_1 = (torch.exp((torch.abs(self.sim_eval.weight[0, 1]) - threshold) * lambda_) * self.sim_eval.weight[0, 1])
        weight_2 = (torch.exp((torch.abs(self.sim_eval.weight[0, 2]) - threshold) * lambda_) * self.sim_eval.weight[0, 2])
        weight_3 = (torch.exp((torch.abs(self.sim_eval.weight[0, 3]) - threshold) * lambda_) * self.sim_eval.weight[0, 3])
        weight_4 = (torch.exp((torch.abs(self.sim_eval.weight[0, 4]) - threshold) * lambda_) * self.sim_eval.weight[0, 4])
        weight_5 = (torch.exp((torch.abs(self.sim_eval.weight[0, 5]) - threshold) * lambda_) * self.sim_eval.weight[0, 5])
        weight_6 = (torch.exp((torch.abs(self.sim_eval.weight[0, 6]) - threshold) * lambda_) * self.sim_eval.weight[0, 6])
        weight_7 = (torch.exp((torch.abs(self.sim_eval.weight[0, 7]) - threshold) * lambda_) * self.sim_eval.weight[0, 7])
        weight_8 = (torch.exp((torch.abs(self.sim_eval.weight[0, 8]) - threshold) * lambda_) * self.sim_eval.weight[0, 8])

        Dim_learned_mask_0 = self.masks_0.weight * weight_0
        Dim_learned_mask_1 = self.masks_1.weight * weight_1
        Dim_learned_mask_2 = self.masks_2.weight * weight_2
        Dim_learned_mask_3 = self.masks_3.weight * weight_3
        Dim_learned_mask_4 = self.masks_4.weight * weight_4
        Dim_learned_mask_5 = self.masks_5.weight * weight_5
        Dim_learned_mask_6 = self.masks_6.weight * weight_6
        Dim_learned_mask_7 = self.masks_7.weight * weight_7
        Dim_learned_mask_8 = self.masks_8.weight * weight_8
             
        sim_rest = self.get_sims_levels(
        img_emb[:, :sum(self.list_length[:1])],
        img_emb[:, sum(self.list_length[:1]):sum(self.list_length[:2])],
        img_emb[:, sum(self.list_length[:2]):sum(self.list_length[:3])],
        img_emb[:, sum(self.list_length[:3]):sum(self.list_length[:4])],
        img_emb[:, sum(self.list_length[:4]):sum(self.list_length[:5])],
        img_emb[:, sum(self.list_length[:5]):sum(self.list_length[:6])],
        img_emb[:, sum(self.list_length[:6]):sum(self.list_length[:7])],
        img_emb[:, sum(self.list_length[:7]):sum(self.list_length[:8])],
        img_emb[:, sum(self.list_length[:8]):sum(self.list_length[:9])],
        cap_emb[:, :sum(self.list_length[:1])],
        cap_emb[:, sum(self.list_length[:1]):sum(self.list_length[:2])],
        cap_emb[:, sum(self.list_length[:2]):sum(self.list_length[:3])],
        cap_emb[:, sum(self.list_length[:3]):sum(self.list_length[:4])],
        cap_emb[:, sum(self.list_length[:4]):sum(self.list_length[:5])],
        cap_emb[:, sum(self.list_length[:5]):sum(self.list_length[:6])],
        cap_emb[:, sum(self.list_length[:6]):sum(self.list_length[:7])],
        cap_emb[:, sum(self.list_length[:7]):sum(self.list_length[:8])],
        cap_emb[:, sum(self.list_length[:8]):sum(self.list_length[:9])],
        Dim_learned_mask_0, Dim_learned_mask_1, Dim_learned_mask_2, Dim_learned_mask_3, Dim_learned_mask_4, Dim_learned_mask_5, Dim_learned_mask_6, Dim_learned_mask_7, Dim_learned_mask_8)
        return sim_rest


class Sims_Measuring(nn.Module):
    def __init__(self, config: Config):
        super(Sims_Measuring, self).__init__()
        self.calculator_1 = sims_claculator(config, 1)

    def forward(self, img_embs, cap_embs):

        sims = self.calculator_1(img_embs, cap_embs)
        return sims




class MultiHeadedAttention(nn.Module):
    def __init__(self, config: Config):
        super(MultiHeadedAttention, self).__init__()
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_mha_heads
        assert self.embed_dim % self.num_heads == 0
        self.head_dim = self.embed_dim // self.num_heads
        self.sim_dim = config.sim_dim

        
        # self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        # self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.tanh = nn.Tanh()
        self.opt = config
        self.plus_encoder = Image_Text_Processing(config)
        self.sims = Sims_Measuring(config)


    
    def forward(self, text_features, image_features):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            o: num_vids x num_texts x embed_dim
        """
        # # original CLIP
        # image_features = self.q_proj(image_features)
        # text_features = self.v_proj(text_features)
        # # # normalized features
        # image_features = image_features / image_features.norm(dim=1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # # cosine similarity as logits
        # sims = image_features @ text_features.t()

        ###################################################################################
        ### Our proposed ESL

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        ## image and text feature processing: constructing hierarchical levels of measure-units
        img_embs, cap_embs = self.plus_encoder(image_features, text_features)

        ## calculating image-text similarity: weighted measuring
        sims =  self.sims(img_embs, cap_embs)

        return sims


class Transformer(nn.Module):
    def __init__(self, config: Config):
        super(Transformer, self).__init__()
        self.embed_dim = config.embed_dim
        self.cross_attn = MultiHeadedAttention(config)

    def forward(self, text_features, image_features):

        sims = self.cross_attn(text_features, image_features)

        return sims
