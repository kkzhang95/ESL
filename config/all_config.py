import os
import argparse
from config.base_config import Config
from modules.basic_utils import mkdirp, deletedir


class AllConfig(Config):
    def __init__(self):
        super().__init__()

    def parse_args(self):
        description = 'Text-to-Image Retrieval'
        parser = argparse.ArgumentParser(description=description)
        
        # data parameters
        parser.add_argument('--dataset_name', type=str, default='MS-COCO', help="Dataset name") # MS-COCO   Flickr30K
        parser.add_argument('--videos_dir', type=str, default='/mnt/data10t/bakuphome20210617/lz/data/data1/I-T/MS-COCO/', help="Location of videos") # /mnt/data10t/bakuphome20210617/lz/data/I-T/MS-COCO/              /mnt/data10t/bakuphome20210617/lz/data/I-T/Flickr30K/flickr30k-images/
        parser.add_argument('--msrvtt_train_file', type=str, default='9k')
        parser.add_argument('--num_frames', type=int, default=12)
        parser.add_argument('--video_sample_type', default='uniform', help="'rand'/'uniform'")
        parser.add_argument('--input_res', type=int, default=224)

        # experiment parameters
        parser.add_argument('--exp_name', type=str, default = 'replicate', help="Name of the current experiment") # required=True ,
        parser.add_argument('--output_dir', type=str, default='/mnt/data10t/bakuphome20210617/zhangkun/ESL_clip/outputs')
        parser.add_argument('--save_every', type=int, default=1, help="Save model every n epochs")
        parser.add_argument('--log_step', type=int, default=100, help="Print training log every n steps")
        parser.add_argument('--evals_per_epoch', type=int, default=4, help="Number of times to evaluate per epoch")
        parser.add_argument('--load_epoch', type=int, default = 0, help="Epoch to load from exp_name, or -1 to load model_best.pth")
        parser.add_argument('--eval_window_size', type=int, default=5, help="Size of window to average metrics")
        parser.add_argument('--metric', type=str, default='t2v', help="'t2v'/'v2t'")

        # model parameters
        parser.add_argument('--huggingface', action='store_false', default=True)
        parser.add_argument('--arch', type=str, default='clip_transformer')
        parser.add_argument('--clip_arch', type=str, default='ViT-B/16', help="CLIP arch. only when not using huggingface") 
        parser.add_argument('--embed_dim', type=int, default=512, help="Dimensionality of the model embedding")
        parser.add_argument('--kernel_size', type=int, default=2, help="measure-unit")

        # training parameters
        parser.add_argument('--loss', type=str, default='clip')
        parser.add_argument('--clip_lr', type=float, default=1e-6, help='Learning rate used for CLIP params')
        parser.add_argument('--noclip_lr', type=float, default=3e-5, help='Learning rate used for new params')
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--num_epochs', type=int, default=15)
        parser.add_argument('--weight_decay', type=float, default=0.2, help='Weight decay')
        parser.add_argument('--warmup_proportion', type=float, default=0.1, help='Warmup proportion for learning rate schedule')

        # frame pooling parameters
        parser.add_argument('--pooling_type', type=str)
        parser.add_argument('--k', type=int, default=-1, help='K value for topk pooling')
        parser.add_argument('--attention_temperature', type=float, default=0.01, help='Temperature for softmax (used in attention pooling only)')
        parser.add_argument('--num_mha_heads', type=int, default=1, help='Number of parallel heads in multi-headed attention')
        parser.add_argument('--transformer_dropout', type=float, default=0, help='Dropout prob. in the transformer pooling')

        # system parameters
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--seed', type=int, default=24, help='Random seed')
        parser.add_argument('--no_tensorboard', action='store_true', default=False)
        parser.add_argument('--tb_log_dir', type=str, default='logs')
        parser.add_argument('--mean_neg', default=0, type=float,
                            help='Mean value of mismatched distribution.')                        
        parser.add_argument('--stnd_neg', default=0, type=float,
                            help='Standard deviation of mismatched distribution.')
        parser.add_argument('--mean_pos', default=0, type=float,
                            help='Mean value of matched distribution.')
        parser.add_argument('--stnd_pos', default=0, type=float,
                            help='Standard deviation of matched distribution.')
        parser.add_argument('--thres', default=0, type=float,
                            help='Optimal learning  boundary.')
        parser.add_argument('--thres_safe', default=0, type=float,
                            help='Optimal learning  boundary.')
        parser.add_argument('--alpha', default=1.0, type=float,
                            help='Initial penalty parameter.')
        parser.add_argument('--in_training', action='store_true',
                            help='wether in training.')

        parser.add_argument('--thres_entity', default=0, type=float,
                            help='Optimal learning  boundary.')
        parser.add_argument('--thres_safe_entity', default=0, type=float,
                            help='Optimal learning  boundary.')
        parser.add_argument('--center_entity', default=1, type=float,
                            help='Optimal learning  boundary.')
        parser.add_argument('--center_non_entity', default=0, type=float,
                            help='Optimal learning  boundary.')
        parser.add_argument('--sim_dim', default=256, type=int,
                            help='Dimensionality of the sim embedding.')


        args = parser.parse_args()

        args.model_path = os.path.join(args.output_dir, args.exp_name)
        args.tb_log_dir = os.path.join(args.tb_log_dir, args.exp_name)

        mkdirp(args.model_path)
        deletedir(args.tb_log_dir)
        mkdirp(args.tb_log_dir)

        return args
