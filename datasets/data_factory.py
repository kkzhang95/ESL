from config.base_config import Config
from datasets.model_transforms import init_transform_dict
from datasets.msrvtt_dataset import MSRVTTDataset
from datasets.msvd_dataset import MSVDDataset
from datasets.lsmdc_dataset import LSMDCDataset
from datasets.flickr30k_dataset import Flickr30Kataset
from datasets.mscoco_dataset import MSCOCOdataset
from torch.utils.data import DataLoader

class DataFactory:

    @staticmethod
    def get_data_loader(config: Config, split_type='train'):
        img_transforms = init_transform_dict(config.input_res)
        if config.dataset_name == 'Flickr30K' or config.dataset_name == 'MS-COCO':
            train_img_tfms = img_transforms['clip']
            test_img_tfms = img_transforms['clip']  
        else:
            train_img_tfms = img_transforms['clip_train']
            test_img_tfms = img_transforms['clip_test']

        if config.dataset_name == "MSRVTT":
            if split_type == 'train':
                dataset = MSRVTTDataset(config, split_type, train_img_tfms)
                return DataLoader(dataset, batch_size=config.batch_size,
                           shuffle=True, num_workers=config.num_workers)
            else:
                dataset = MSRVTTDataset(config, split_type, test_img_tfms)
                return DataLoader(dataset, batch_size=config.batch_size,
                           shuffle=False, num_workers=config.num_workers)

        elif config.dataset_name == "MSVD":
            if split_type == 'train':
                dataset = MSVDDataset(config, split_type, train_img_tfms)
                return DataLoader(dataset, batch_size=config.batch_size,
                        shuffle=True, num_workers=config.num_workers)
            else:
                dataset = MSVDDataset(config, split_type, test_img_tfms)
                return DataLoader(dataset, batch_size=config.batch_size,
                        shuffle=False, num_workers=config.num_workers)
            
        elif config.dataset_name == 'LSMDC':
            if split_type == 'train':
                dataset = LSMDCDataset(config, split_type, train_img_tfms)
                return DataLoader(dataset, batch_size=config.batch_size,
                            shuffle=True, num_workers=config.num_workers)
            else:
                dataset = LSMDCDataset(config, split_type, test_img_tfms)
                return DataLoader(dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=config.num_workers)

        elif config.dataset_name == 'Flickr30K':
            if split_type == 'train':
                dataset = Flickr30Kataset(config, split_type, train_img_tfms)
                return DataLoader(dataset, batch_size=config.batch_size,
                            shuffle=True, num_workers=config.num_workers)
            else:
                dataset = Flickr30Kataset(config, split_type, test_img_tfms)
                return DataLoader(dataset, batch_size=100, #config.batch_size
                            shuffle=False, num_workers=config.num_workers)
        elif config.dataset_name == 'MS-COCO':
            if split_type == 'train':
                dataset = MSCOCOdataset(config, split_type, train_img_tfms)
                return DataLoader(dataset, batch_size=config.batch_size,
                            shuffle=True, num_workers=config.num_workers)
            else:
                dataset = MSCOCOdataset(config, split_type, test_img_tfms)

                ### cross-datasets evaluation
                # config.videos_dir = '/mnt/data10t/bakuphome20210617/lz/data/data1/I-T/Flickr30K/flickr30k-images/'
                # dataset = Flickr30Kataset(config, split_type, test_img_tfms)

                return DataLoader(dataset, batch_size=100, #config.batch_size
                            shuffle=False, num_workers=config.num_workers)
        else:
            raise NotImplementedError
