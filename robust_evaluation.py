# -*- coding: utf-8 -*-
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import argparse
import torch
from utils.trainer_utils import robust_evaluation, model_selection, evaluation

parser = argparse.ArgumentParser()
parser.add_argument('--net_name', default='SymTC', type=str, help='model')
parser.add_argument('--snapshot_path', default=r'result/SymTC/', type=str, help='saved model path')
parser.add_argument('--test_dataset_path', type=str, default=r'dataset/Test', help="testing dataset path")
parser.add_argument('--device', type=str, default='cuda:0', help='device')
parser.add_argument('--direction', default=0, type=int, help='shifting direction for robustness')

parser.add_argument('--num_classes', default=12, type=int, help='output channel of network')
parser.add_argument('--aug_elastic', default=0, type=int, nargs='+', help='elastic for augmentation')
parser.add_argument('--aug_rotate', default=0, type=int, help='rotation for augmentation')
parser.add_argument('--aug_shift', default=0, type=int, nargs='+', help='shift for augmentation')
parser.add_argument('--num_workers', default=4, type=int, help='num_workers for data loader')
parser.add_argument('--batch_size_eval', default=1, type=int, help='batch_size for validation and test')
parser.add_argument('--seed', default=1, type=int, help='random seed')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
args = parser.parse_args()

if __name__ == '__main__':
    device = args.device
    model = model_selection(args.net_name, args)
    model = model.to(device)
    snapshot_path = args.snapshot_path
    direction = args.direction
    # %%Robustness Test
    robust_evaluation(args, model, snapshot_path, direction, 'best')
