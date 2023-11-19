from config.base_config import Config
import numpy as np
import torch
from collections import defaultdict, deque
from trainer.base_trainer import BaseTrainer
from modules.metrics import sim_matrix_training, sim_matrix_inference, generate_embeds_per_video_id
from tqdm import tqdm
import os
import sys
import time

def logging_func(logfile, message):
    with open(logfile, "a") as f:
        f.write(message)
    f.close()

def i2t(sims, npts, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)

    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]

        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(sims, npts, return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)

    # --> (5N(caption), N(image))
    sims = sims.T

    for index in range(npts):
        for i in range(5):
            inds = np.argsort(sims[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)
            

class Trainer(BaseTrainer):
    """
    Trainer class
    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, model, loss, metrics, optimizer, config: Config, train_data_loader, 
                 valid_data_loader, tokenizer, lr_scheduler=None, writer=None):

        super().__init__(model, loss, metrics, optimizer, config, writer)
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.tokenizer = tokenizer 

        self.pooling_type = config.pooling_type
        self.window_metric = defaultdict(lambda: deque(maxlen=config.eval_window_size))
        self.best_window = -1.0
        self.best = -1.0
        self.config = config



    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.
        """
        self.model.train()
        total_loss = 0.0
        num_steps = len(self.train_data_loader)
        eval_steps = np.linspace(0, num_steps-1, self.evals_per_epoch+1, dtype=int)[1:]
        
        for batch_idx, data in enumerate(self.train_data_loader):
            # then assume we must tokenize the input, e.g. its a string
            if self.tokenizer is not None:
                data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True,
                                              truncation=True)
            if isinstance(data['text'], torch.Tensor):
                data['text'] = data['text'].to(self.device)
            else:
                data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}
            
            data['image'] = data['image'].to(self.device)

            sims  = self.model(data)

            loss = self.loss(sims, self.model.clip.logit_scale)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

            torch.clamp_(self.model.clip.logit_scale.data, max=np.log(100))

            self.global_step += 1
            if self.writer is not None:
                self.writer.add_scalar('train/loss_train', loss.detach().item(), self.global_step)

            total_loss += loss.detach().item()

            if batch_idx % self.log_step == 0:
                print('Train Epoch: {} dl: {}/{} Loss: {:.6f}'.format(
                    epoch,
                    batch_idx,
                    num_steps-1,
                    loss.detach().item()))

            if batch_idx in eval_steps:
                self.config.in_training = False
                sum, sims = self._valid_epoch_step(epoch, batch_idx, num_steps-1)
                self.model.train()
                self.config.in_training = True

                if sum > self.best_window:
                    self.best_window = sum
                    self._save_checkpoint(epoch, self.config, save_best=True)
                    np.savetxt(r'/mnt/data10t/bakuphome20210617/zhangkun/ESL_clip/outputs/replicate/sim_best.txt', sims, fmt='%.5f') 

                print(" Current Best Window Average sum is {}".format(self.best_window))
                print(" Current Best R@1 is {}\n\n".format(self.best))



        res = {
            'loss_train':  total_loss / num_steps
        }

        return res



    def shard_attn_scores(self, all_image, all_text, all_image_id, shard_size=100):

        shard_size = 100
        n_im_shard = (all_image.size(0) - 1) // shard_size + 1
        n_cap_shard = (len(all_text) - 1) // shard_size + 1
        sims = np.zeros((all_image.size(0), len(all_text)))

        for i in range(n_im_shard):
            im_start, im_end = shard_size * i, min(shard_size * (i + 1), all_image.size(0))
            for j in range(n_cap_shard):
                sys.stdout.write('\r>> shard_attn_scores batch (%d,%d)' % (i, j))
                ca_start, ca_end = shard_size * j, min(shard_size * (j + 1), len(all_text))

                all_data ={'image_id':all_image_id[im_start:im_end], 'image':all_image[im_start:im_end], 'text':all_text[ca_start:ca_end]}

                if self.tokenizer is not None:
                    all_data['text'] = self.tokenizer(all_data['text'], return_tensors='pt', padding=True, truncation=True)
                if isinstance(all_data['text'], torch.Tensor):
                    all_data['text'] = all_data['text'].to(self.device)
                else:
                    all_data['text'] = {key: val.to(self.device) for key, val in all_data['text'].items()}
                all_data['image'] = all_data['image'].to(self.device)
            
                sim = self.model(all_data, return_all_frames=True)

                sims[im_start:im_end, ca_start:ca_end] = sim.data.cpu().numpy()

        sys.stdout.write('\n')
        return sims




    
        
    def _valid_epoch_step(self, epoch, step, num_steps):
        """
        Validate at a step when training an epoch at a certain step
        :return: A log that contains information about validation
        """
        #### This falg varible can be set as True, when using for average 1K test for MS-COCO dataset
        Average_1k_test = True

        
        if not Average_1k_test:

            # # ##############################Flick test or COCO 5K TEST###################################################################################
            self.model.eval()
            with torch.no_grad():
                all_image_id = []
                all_image = []
                all_text = []
                for _, data in tqdm(enumerate(self.valid_data_loader)):
                    image_id = data['image_id'][::5]
                    image = data['image'][::5]
                    text = data['text']

                    all_image_id = all_image_id + image_id
                    all_image.append(image.cpu())
                    all_text = all_text + text
                all_image = torch.cat(all_image)

                # # construct similarity matrix
                print("calculate similarity")
                
                shard_size = 100
                n_im_shard = (all_image.size(0) - 1) // shard_size + 1
                n_cap_shard = (len(all_text) - 1) // shard_size + 1
                sims = np.zeros((all_image.size(0), len(all_text)))

                sims2 = np.zeros((all_image.size(0), len(all_text)))
                sims3 = np.zeros((all_image.size(0), len(all_text)))

                for i in range(n_im_shard):
                    im_start, im_end = shard_size * i, min(shard_size * (i + 1), all_image.size(0))
                    for j in range(n_cap_shard):
                        sys.stdout.write('\r>> shard_attn_scores batch (%d,%d)' % (i, j))
                        ca_start, ca_end = shard_size * j, min(shard_size * (j + 1), len(all_text))

                        all_data ={'image_id':all_image_id[im_start:im_end], 'image':all_image[im_start:im_end], 'text':all_text[ca_start:ca_end]}

                        if self.tokenizer is not None:
                            all_data['text'] = self.tokenizer(all_data['text'], return_tensors='pt', padding=True, truncation=True)
                        if isinstance(all_data['text'], torch.Tensor):
                            all_data['text'] = all_data['text'].to(self.device)
                        else:
                            all_data['text'] = {key: val.to(self.device) for key, val in all_data['text'].items()}
                        all_data['image'] = all_data['image'].to(self.device)
                    
                        sim  = self.model(all_data, return_all_frames=True)

                        sims[im_start:im_end, ca_start:ca_end] = sim.data.cpu().numpy()

                sys.stdout.write('\n')
                # bi-directional retrieval

                r, rt = i2t(sims, all_image.size(0), return_ranks=True)
                ri, rti = t2i(sims, all_image.size(0), return_ranks=True)
                ar = (r[0] + r[1] + r[2]) / 3
                ari = (ri[0] + ri[1] + ri[2]) / 3
                rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
                print("rsum: %.1f" % rsum)
                print("Average i2t Recall: %.1f" % ar)
                print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
                print("Average t2i Recall: %.1f" % ari)
                print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)

                message = "Epoch: %d: Image to text: (%.1f, %.1f, %.1f) " % (epoch, r[0], r[1], r[2])
                message += "Text to image: (%.1f, %.1f, %.1f) " % (ri[0], ri[1], ri[2])
                message += "rsum: %.1f\n" % rsum

                log_file = os.path.join('/mnt/data10t/bakuphome20210617/zhangkun/ESL_clip/logs/replicate_1/', "performance.log")
                logging_func(log_file, message)


                return rsum, sims


        else:

            # ##############################COCO 1K TEST###################################################################################
            self.model.eval()

            with torch.no_grad():
                all_image_id = []
                all_image = []
                all_text = []
                for _, data in tqdm(enumerate(self.valid_data_loader)):
                    image_id = data['image_id'][::5]
                    image = data['image'][::5]
                    text = data['text']

                    all_image_id = all_image_id + image_id
                    all_image.append(image.cpu())
                    all_text = all_text + text
                all_image = torch.cat(all_image)

                # # construct similarity matrix
                print("calculate similarity")
                
                # 5fold cross-validation, only for MSCOCO
                results = []
                for i in range(5):
                    img_embs_shard = all_image[i * 1000:(i + 1) * 1000]
                    cap_embs_shard = all_text[i * 5000:(i + 1) * 5000]
                    all_id_shard = all_image_id[i * 5000:(i + 1) * 5000]

                    start = time.time()
                    sims = self.shard_attn_scores(img_embs_shard, cap_embs_shard, all_id_shard, shard_size=1000)
                    end = time.time()
                    print("calculate similarity time:", end-start)

                    r, rt0 = i2t(sims, img_embs_shard.size(0), return_ranks=True)
                    ri, rti0 = t2i(sims, img_embs_shard.size(0), return_ranks=True)

                    if i == 0:
                        rt, rti = rt0, rti0
                    ar = (r[0] + r[1] + r[2]) / 3
                    ari = (ri[0] + ri[1] + ri[2]) / 3
                    rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
                    print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
                    results += [list(r) + list(ri) + [ar, ari, rsum]]

                print("-----------------------------------")
                print("Mean metrics: ")
                mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
                print("rsum: %.1f" % (mean_metrics[12]))
                print("Average i2t Recall: %.1f" % mean_metrics[10])
                print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
                    mean_metrics[:5])
                print("Average t2i Recall: %.1f" % mean_metrics[12])
                print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
                    mean_metrics[5:10])



                message = "Epoch: %d: Image to text: (%.1f, %.1f, %.1f) " % (epoch, r[0], r[1], r[2])
                message += "Text to image: (%.1f, %.1f, %.1f) " % (ri[0], ri[1], ri[2])
                message += "rsum: %.1f\n" % rsum

                log_file = os.path.join('/mnt/data10t/bakuphome20210617/zhangkun/xpool_flik/logs/replicate_1/', "performance.log")
                logging_func(log_file, message)

                return rsum, sims