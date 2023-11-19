import re
import torch
import torch.nn as nn
from config.base_config import Config
from modules.transformer import Transformer

class CLIPTransformer(nn.Module):
    def __init__(self, config: Config):
        super(CLIPTransformer, self).__init__()
        self.config = config
        
        from model.clip_model import load_clip
        self.clip = load_clip(config.clip_arch)

        config.pooling_type = 'transformer'
        self.pool_frames = Transformer(config)


    def forward(self, data, return_all_frames=False):
        batch_size = data['image'].shape[0]
        text_data = data['text']
        image_data = data['image']
        image_data = image_data.reshape(-1, 3, self.config.input_res, self.config.input_res)
        
        text_features = self.clip.encode_text(text_data)
        image_features = self.clip.encode_image(image_data)

        sims = self.pool_frames(text_features, image_features) 

        return sims
