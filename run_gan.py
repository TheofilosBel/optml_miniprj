from __future__ import print_function
#%matplotlib inline
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim

from src.args import parse_args
from src.dataset import create_dl, create_dataset
from src.modeling.gan import GAN
from src.modeling.optimizers.utils import update_optim_args
from src.modeling.trainig import train
from src.utils import plot_img

# Import all optimizers even if not used, so str_to_cls can work
from src.modeling.optimizers.extragradient import ExtraAdam, ExtraSGD
from torch.optim import Adam, SGD


def str_to_cls(cls_name:str):
    return getattr(sys.modules[__name__], cls_name)

def main():
    # Parse arguments
    args = parse_args()

    # Set random seed for reproducibility
    manualSeed = 999
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")

    # Create dataset and dataloader
    dataset = create_dataset(args)
    dataloader = create_dl(args, dataset)

    # Plot some training images
    real_batch = next(iter(dataloader))
    plot_img(args, real_batch, device)

    # Create GAN
    gan = GAN(args)
    gan.to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (args.ngpu > 1):
        gan.netD = nn.DataParallel(gan.netD, list(range(args.ngpu)))
        gan.netG = nn.DataParallel(gan.netG, list(range(args.ngpu)))

    # Loss and optimizer
    criterion = nn.BCELoss()

    optD_params = update_optim_args(args, {'params':gan.netD.parameters(), 'lr':args.lr_D})
    optimizerD = str_to_cls(args.optim)(**optD_params)

    optG_params = update_optim_args(args, {'params':gan.netG.parameters(), 'lr':args.lr_G})
    optimizerG = str_to_cls(args.optim)(**optG_params)

    #  Start training
    train(args, gan, dataloader, criterion, optimizerD, optimizerG, device)


if __name__ == '__main__':
    main()