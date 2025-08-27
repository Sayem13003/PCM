# ROBUST MULTI-LABEL LEARNING WITH HUMAN-GUIDED AND FOUNDATION MODEL-AIDED CROWD FRAMEWORK
This is the official Pytorch implementation paper "ROBUST MULTI-LABEL LEARNING WITH HUMAN-GUIDED AND FOUNDATION MODEL-AIDED CROWD FRAMEWORK".

Authors: Faizul Rakib Sayem, Shahana Ibrahim

### Abstract

Multi-label learning has emerged as a critical task in artificial intelligence (AI) for understanding data across diverse modalities. However, a significant challenge in this domain is the acquisition of accurate labels, which is often both time-consuming and resource-intensive. Assigning multiple labels to each data instance typically requires input from multiple annotators, each bringing their own expertise or mistakes. Recent advancements in foundation models have enabled the use of pseudo-labels to supplement human annotations, but these models are often not primarily designed for multi-label tasks, introducing additional label noise. In this work, we present a novel crowd framework for multi-label learning that integrates hybrid collaboration between human annotators and foundation models. By combining their responses in a robust manner and leveraging insights from modeling and factorization techniques, the proposed framework is accompanied by a regularized end-to-end learning criterion. Experiments using several real-world datasets showcase the promise of our
framework.

<div align="center">
<img src="images/PCM.png" title="VLPL" width="80%">
</div>

## 🛠️ Installation
1. Create a Conda environment for the code:
```
conda create --name SPML python=3.8.8
```
2. Activate the environment:
```
conda activate SPML
```
3. Install the dependencies:
```
pip install -r requirements.txt
```

## 📖 Preparing Datasets
### Downloading Data
#### PASCAL VOC

1. Run the following commands:

```
cd {PATH-TO-THIS-CODE}/data/pascal
curl http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar --output pascal_raw.tar
tar -xf pascal_raw.tar
rm pascal_raw.tar
```

#### MS-COCO

1. Run the following commands:

```
cd {PATH-TO-THIS-CODE}/data/coco
curl http://images.cocodataset.org/annotations/annotations_trainval2014.zip --output coco_annotations.zip
curl http://images.cocodataset.org/zips/train2014.zip --output coco_train_raw.zip
curl http://images.cocodataset.org/zips/val2014.zip --output coco_val_raw.zip
unzip -q coco_annotations.zip
unzip -q coco_train_raw.zip
unzip -q coco_val_raw.zip
rm coco_annotations.zip
rm coco_train_raw.zip
rm coco_val_raw.zip
```

#### CUB

1.  Download `CUB_200_2011.tgz` in [this website](https://data.caltech.edu/records/20098).
2.  Run the following commands:
```
mv {PATH-TO-DOWNLOAD-FILES}/CUB_200_2011.tgz {PATH-TO-THIS-CODE}/data/cub
tar -xf CUB_200_2011.tgz
rm CUB_200_2011.tgz
```

### Formatting Data
For PASCAL VOC, MS-COCO, and CUB, use Python code to format data:
```
cd {PATH-TO-THIS-CODE}
python preproc/format_pascal.py
python preproc/format_coco.py
python preproc/format_cub.py
```
```
`{DOWNLOAD-FILES}` should be replaced by `formatted_train_images.npy`, `formatted_train_labels.npy`, `formatted_val_images.npy`, or `formatted_train_labels.npy`.


### Generating Single Positive Annotations
In the last step, run `generate_observed_labels.py` to yield single positive annotations from full annotations of each dataset:
```
python preproc/generate_observed_labels.py --dataset {DATASET}
```
`{DATASET}` should be replaced by `pascal`, `coco`,  or `cub`.

## 🦍 Training and Evaluation
Run `main_clip_multiclip_1s.py` to train and evaluate a model:
```
python main.py -d {DATASET} -l {LOSS} -g {GPU} -m {model} -t {tempurature} -th {threshold}  -p {partical} -s {PYTORCH-SEED}
```
Command-line arguments are as follows:
1. `{DATASET}`: The adopted dataset. (*default*: `pascal` | *available*: `pascal`, `coco`, `or `cub`)
2. `{LOSS}`: The method used for training. (*default*: `EM_PL` | *available*: `bce`, `iun`, `an`, `EM`, `EM_APL`, or `EM_PL`)
3. `{GPU}`: The GPU index. (*default*: `0`)
4. `{PYTORCH-SEED}`: The seed of PyTorch. (*default*: `0`)
5. `{model}`: The model of backbone. (*default*: `resnet50`| *available*: `resnet50`, `vit_clip`, `convnext_xlarge_22k`, or `convnext_xlarge_1k`)
6. `{tempurature}`: the temperature scalar of the softmax function.
7. `{threshold}`: the threshold for the positive pseudo-label. (*default*: `0.3`)
8. `{partical}`: the percentage of the negative pseudo-label. (*default*: `0.0`)

'''
## Results:

## Acknowledgement:
Many thanks to the authors of [VLPL](https://github.com/mvrl/VLPL) [single-positive-multi-label](https://github.com/elijahcole/single-positive-multi-label), and [SPML-AckTheUnknown
](https://github.com/Correr-Zhou/SPML-AckTheUnknown). Our scripts are highly based on their scripts.

# PCM
