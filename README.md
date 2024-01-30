# SymTC

[**SymTC: A Symbiotic Transformer-CNN Net for Instance Segmentation of Lumbar Spine MRI**](https://arxiv.org/abs/2401.09627)<br/>
[Jiasong Chen](https://jiasongchen.github.io/),
[Linchen Qian](https://scholar.google.com/citations?user=cqAjbgQAAAAJ&hl=en&oi=sra),
[Linhai Ma](https://sarielma.github.io/),
[Timur Urakov](https://med.miami.edu/faculty/timur-urakov-md),
[Weiyong Gu](https://people.miami.edu/profile/9d2998936f27f2b0daa8828b8a709d0c),
[Liang Liang](https://liangbright.wordpress.com/)<br/>

## Overview

Intervertebral disc disease, a prevalent ailment, frequently leads to intermittent or persistent low back pain,
and diagnosing and assessing of this disease rely on accurate measurement of vertebral bone and intervertebral disc
geometries from lumbar MR images. Deep neural network (DNN) models may assist clinicians with more efficient image
segmentation of individual instances (disks and vertebrae) of the lumbar spine in an automated way, which is termed as instance image
segmentation. In this work, we proposed SymTC, an innovative lumbar spine MR image segmentation model that combines the strengths of
Transformer and Convolutional Neural Network (CNN). Specifically, we designed a parallel dual-path architecture to merge
CNN layers and Transformer layers, and we integrated a novel position embedding into the self-attention module of Transformer,
enhancing the utilization of positional information for more accurate segmentation. To further improves model performance, we
introduced a new data augmentation technique to create synthetic yet realistic MR image dataset, named SSMSpine, which is made publicly available. We
evaluated our SymTC and the other 15 existing image segmentation models on our private in-house dataset and the public SSMSpine dataset, using
two metrics, Dice Similarity Coefficient and 95% Hausdorff Distance. The results show that our SymTC has the best performance for
segmenting vertebral bones and intervertebral discs in lumbar spine MR images.

## Environment Setup

Please set up an environment with python=3.11and execute the provided command to install the necessary dependencies.

Please be aware that the hd95 function in the _medpy = 0.3.0_ specifically requires _numpy = 1.23.5_. 

```commandline
pip install -r requirments.txt
```

## Download Dataset

Please access [**SSMSpine Dataset**](https://github.com/jiasongchen/SSMSpine) and proceed to download the dataset:

## Prepare data

Please partition the contents of the downloaded **"Train"** folder into **Train** and **Vali** folder. 
Place these sets, along with the **"Test"** folder, within the "dataset" folder, maintaining the specified structure.


```bash
├── dataset
│     ├──Train
│     │   ├──p[*]
│     │   │  ├──aug[*].pt
│     │   │  ├──***
│     │   ├──***
│     │ 
│     ├──Vali.py
│     │   ├──p[**]
│     │   │   ├──aug[*].pt
│     │   │   ├──***
│     │   ├──***
│     │
│     ├──Test
│     │   ├──p[***]
│     │   │   ├──aug[*].pt
│     │   │   ├──***
│     │   ├──***
```

## Pretrained SymTC models

Below is the download link for the pretrained SymTC model checkpoint:

[**SymTC Checkpoint**](https://drive.google.com/drive/folders/1NLWaRFqM1L-d8jpd7KOVP3M_nK-S03ve?usp=sharing)

## Train/Evaluation

### Training SymTC Models

Execute the command to initiate the training of the model.

`python train.py --net_name SymTC --num_classes 12 --max_epochs 500 --batch_size_train 3 --batch_size_eval 3 --base_lr 0.0001 --device cuda:0`

Adjust the values for _max_epochs_, _batch_size_train_, _batch_size_eval_, _device_, and any other relevant parameters as necessary.

The optimal model checkpoint **(best.pth)** will be stored in the **result/SymTC** directory.

### Model Evaluation

Execute the command for segmentation evaluation

`python evaluation.py --net_name SymTC --device cuda:0 --num_classes 12 --batch_size_eval 1 --save_fig False`

Ensure that the results are stored in the **"result"** folder. Enable sample image generation by setting the parameter **save_fig** to _True_.

### Model Robust Evaluation

Execute the command to assess the robustness of the model.

`python robust_evaluation.py --net_name SymTC --device cuda:0 --num_classes 12 --batch_size_eval 1`

The results will be logged in a file within the **robust** folder, and the generated sample images 
will be saved under the **result/robustness_evaluation** directory.

The shift directions are indicated as follows:
0 -> Up, 1 -> Down, 2 -> Left, 3 -> Right


## Reference

* [TransUnet](https://github.com/Beckschen/TransUNet)
* [Google ViT](https://github.com/google-research/vision_transformer)
* [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
* [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)

## Citations

```
@misc{chen2024symtc,
      title={SymTC: A Symbiotic Transformer-CNN Net for Instance Segmentation of Lumbar Spine MRI}, 
      author={Jiasong Chen and Linchen Qian and Linhai Ma and Timur Urakov and Weiyong Gu and Liang Liang},
      year={2024},
      eprint={2401.09627},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```