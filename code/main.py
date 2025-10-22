import argparse
import os
import random
import warnings
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='fakett', help='fakett/fakesv')
parser.add_argument('--mode', default='train', help='train/inference_test')
parser.add_argument('--epoches', type=int, default=)
parser.add_argument('--batch_size', type = int, default=)
parser.add_argument('--early_stop', type=int, default=)
parser.add_argument('--seed', type=int, default=)

parser.add_argument('--gpu', default='')
parser.add_argument('--lr', type=float, default=)
parser.add_argument('--alpha',type=float, default=)
parser.add_argument('--beta',type=float, default=)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
os.environ['CUDA_LAUNCH_BLOCKING']='1'
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print (args)