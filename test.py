import os
import torch
import random
import numpy as np
from config.all_config import AllConfig
from torch.utils.tensorboard.writer import SummaryWriter
from datasets.data_factory import DataFactory
from model.model_factory import ModelFactory
from modules.metrics import t2v_metrics, v2t_metrics
from modules.loss import LossFactory
from trainer.trainer import Trainer


def main():
    config = AllConfig()

    ## load checkpoints-----------------------------------------------------------------------------------------------------------------------------
    ## for heuristic_strategy, note that you should set the flag varibale 'heuristic_strategy'  in line 32 of transformer.py as True
    # checkpoint_path = "../model_best_heuristic_coco_clip_based.pth"

    ## for adaptive_strategy, note that you should set the flag varibale 'heuristic_strategy'  in line 32 of transformer.py as False
    checkpoint_path = "../model_best_adaptive_coco_clip_based.pth"

    print("Loading checkpoint: {} ...".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    config = checkpoint['config']

    ## setting dataset path in your machine----------------------------------------------------------------------------------------------------------
    config.videos_dir = '../MS-COCO/'



    print("config loaded")
    for arg in vars(config):
        print(format(arg, '<20'), format(str(getattr(config, arg)), '<')) 


    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    if not config.no_tensorboard:
        writer = SummaryWriter(log_dir=config.tb_log_dir)
    else:
        writer = None


    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # config.huggingface = False

    if config.huggingface:
        print('huggingface')
        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16", TOKENIZERS_PARALLELISM=False)
    else:
        from modules.tokenization_clip import SimpleTokenizer
        tokenizer = SimpleTokenizer()

    test_data_loader  = DataFactory.get_data_loader(config, split_type='testall') # testall  dev test
    model = ModelFactory.get_model(config)
    
    if config.metric == 't2v':
        metrics = t2v_metrics
    elif config.metric == 'v2t':
        metrics = v2t_metrics
    else:
        raise NotImplemented
    
    loss = LossFactory.get_loss(config)

    trainer = Trainer(model, loss, metrics, None,
                      config=config,
                      train_data_loader=None,
                      valid_data_loader=test_data_loader,
                      lr_scheduler=None,
                      writer=writer,
                      tokenizer=tokenizer)

    
    trainer.load_checkpoint(checkpoint_path) 
    trainer.validate()




if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "4"
    main()

