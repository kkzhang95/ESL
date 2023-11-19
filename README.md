# Enhanced Semantic Similarity Learning Framework for Image-Text Matching

<img src="https://github.com/CrossmodalGroup/ESL/blob/main/lib/pytorch-logo-dark.png" width="10%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

Official PyTorch implementation of the paper [Enhanced Semantic Similarity Learning Framework for Image-Text Matching](https://www.researchgate.net/publication/373318149_Enhanced_Semantic_Similarity_Learning_Framework_for_Image-Text_Matching).

Please use the following bib entry to cite this paper if you are using any resources from the repo.

```
@article{zhang2023enhanced,
  title={Enhanced Semantic Similarity Learning Framework for Image-Text Matching},
  author={Zhang, Kun and Hu, Bo and Zhang, Huatian and Li, Zhe and Mao, Zhendong},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2023},
  publisher={IEEE}
}
```


We referred to the implementations of [X-Pool](https://github.com/layer6ai-labs/xpool) to build up our codebase. 

## Motivation
<div align=center><img src="https://github.com/CrossmodalGroup/ESL/blob/main/motivation.png" width="50%" ></div>
  
Squares denote local dimension elements in a feature. Circles denote the measure-unit, i.e., the minimal basic component used to examine semantic similarity. Compared with (a) existing methods typically default to a static mechanism that only examines the single-dimensional cross-modal correspondence, (b) our key idea is to dynamically capture and learn multi-dimensional enhanced correspondence.  That is, the number of dimensions constituting the measure-units is changed from existing only one to hierarchical multi-levels, enabling their examining information granularity to be enriched and enhanced to promote a more comprehensive semantic similarity learning.

## Introduction
<img src="https://github.com/CrossmodalGroup/ESL/blob/main/overview.png" width="100%">
In this paper, different from the single-dimensional correspondence with limited semantic expressive capability, we propose a novel enhanced semantic similarity learning (ESL), which generalizes both measure-units and their correspondences into a dynamic learnable framework to examine the multi-dimensional enhanced correspondence between visual and textual features. Specifically, we first devise the intra-modal multi-dimensional aggregators with iterative enhancing mechanism, which dynamically captures new measure-units integrated by hierarchical multi-dimensions, producing diverse semantic combinatorial expressive capabilities to provide richer and discriminative information for similarity examination. Then, we devise the inter-modal enhanced correspondence learning with sparse contribution degrees, which comprehensively and efficiently determines the cross-modal semantic similarity. Extensive experiments verify its superiority in achieving state-of-the-art performance.

### Image-text Matching Results

The following tables show partial results of image-to-text retrieval on COCO and Flickr30K datasets. In these experiments, we use BERT-base as the text encoder for our methods. This branch provides our code and pre-trained models for **using CLIP Encoders as the backbone**. Please check out to [**the ```BERT-based``` branch**](https://github.com/CrossmodalGroup/ESL) for the code and pre-trained models.

The pre-trained models for MS-COCO can be found [model_best_heuristic_coco_clip_based.pth](https://drive.google.com/file/d/1Wk-dzIx04v9NXZJk4oFWVzaEPiRrsfeT/view?usp=sharing) and [model_best_adaptive_coco_clip_based.pth](https://drive.google.com/file/d/1gPM-9xppPh-RPMLm6GLJ8IUKO7GUFP5n/view?usp=sharing). 
The pre-trained models for Flick30K are lost due to not saving in time. You can train the model yourself to produce the results. 



## Preparation

### Environment

* The specific required environment can be found [here](https://drive.google.com/file/d/1tAv2xW9u2tFgr2EhV7bIw83tW51SuM5I/view?usp=sharing) Using **conda env create -f ESL_CLIP_based.yaml** to create the corresponding environments.

### Data
The required files dataset_flickr30k.json, train_coco.json, testall_coco.json, and dev_coco.json can be found [here](https://drive.google.com/drive/folders/1TKucwpCcKdPlby6JpAKgjxT8V0uIqwrK?usp=sharing). 

You can download the raw image dataset through [Flick30k](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset) and [MS-COCO](https://cocodataset.org/#home). 

## Training

```bash
sh  train_clip_based_f30k.sh
```

```bash
sh  train_clip_based_coco.sh
```
For the dimensional selective mask, we design both heuristic and adaptive strategies.  You can use the flag in [./modules/transformer.py](https://github.com/kkzhang95/ESL/blob/main/modules/transformer.py) (line 32) 
```bash
heuristic_strategy = False
```
to control which strategy is selected. True -> heuristic strategy, False -> adaptive strategy. 

## Evaluation

Test on Flickr30K and MSCOCO
```bash
python test.py
```


