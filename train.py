import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import argparse
import random
import numpy as np
import torch
from utils.trainer_utils import trainer, model_selection

# %%data path config
parser = argparse.ArgumentParser()
parser.add_argument('--train_dataset_path', type=str,
                    default=r"dataset/Train", help='training dataset path')
parser.add_argument('--vali_dataset_path', type=str,
                    default=r"dataset/Vali", help='validation dataset path')
parser.add_argument('--test_dataset_path', type=str,
                    default=r'dataset/Test', help="testing dataset path")
parser.add_argument('--result_path', default=r"result/", type=str, help='path to save results')

# %%Training config
parser.add_argument('--net_name', default='SymTC', type=str, help='model')
parser.add_argument('--num_classes', default=12, type=int, help='output channel of network')
parser.add_argument('--max_epochs', default=1000, type=int, help='maximum epoch number for training')
parser.add_argument('--batch_size_train', default=1, type=int, help='batch_size for training')
parser.add_argument('--batch_size_eval', default=1, type=int, help='batch_size for validation and test')
parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer')
parser.add_argument('--plot_training_epoch', default=1, type=int, help='epoch number to plot and save training process')
parser.add_argument('--base_lr', default=0.0001, type=float, help='initial learning rate')
parser.add_argument('--min_lr', default=0.0001, type=float, help='the minimum learning rate')
parser.add_argument('--decay_epoch', default=1, type=int, help='epoch interval to decay learning rate')
parser.add_argument('--dice_percentage', default=0.5, type=float, help='dice loss percentage for training models')
parser.add_argument('--grad_clip_max_norm', default=1, type=float, help='max_norm for grad clip')
parser.add_argument('--loss_attn_weight', default=0, type=float, help='attn loss weight for training models')

parser.add_argument('--aug_elastic', default=[9, 17], type=list, help='elastic for augmentation')
parser.add_argument('--aug_rotate', default=0, type=int, help='rotation for augmentation')
parser.add_argument('--aug_shift', default=16, type=list, help='shift for augmentation')
parser.add_argument('--resume_training', default=0, type=int, help='1: resume training: load model from "last.pth"')
parser.add_argument('--seed', default=1, type=int, help='random seed')
parser.add_argument('--num_workers', default=4, type=int, help='num_workers for data loader')
parser.add_argument('--device', type=str, default='cuda:0', help='cuda device')

args = parser.parse_args()
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# %%
if __name__ == '__main__':
    net_name = args.net_name
    model = model_selection(net_name, args)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    snapshot_path = net_name.replace(' ', '')  # remove empty space
    snapshot_path = args.result_path + snapshot_path
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    trainer(args, model, snapshot_path)
