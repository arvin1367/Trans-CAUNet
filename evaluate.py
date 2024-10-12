import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
#from networks.vision_transformer import SwinUnet 
#from networks.muti_attention_parallel_mathain import MapcUnet as MaswinUnet
from net  import SWCAtranunet
import sys
import os
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
import os
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainer import evaluate

#from config import get_config


parser = argparse.ArgumentParser()
parser.add_argument('--test_path', type=str,
                    default=r'\test', help='root dir for test data')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')               
parser.add_argument('--max_epochs', type=int,
                    default=20, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=12, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')


args = parser.parse_args()

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    net = SWCAtranunet(in_channels=11, num_classes=args.num_classes, base_c=64).cuda()
    evaluate(args, net,r'./Output Path\bestMoudle.pth')