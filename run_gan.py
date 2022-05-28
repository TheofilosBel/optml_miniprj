from __future__ import print_function
#%matplotlib inline
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim

from src.args import parse_args
from src.dataset import create_dl, create_dataset
from src.modeling.gan import GAN
from src.modeling.trainig import train
from src.utils import plot_img

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
    plot_img(real_batch, device)

    # Create GAN
    gan = GAN(args)
    gan.to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (args.ngpu > 1):
        gan.netD = nn.DataParallel(gan.netD, list(range(args.ngpu)))
        gan.netG = nn.DataParallel(gan.netG, list(range(args.ngpu)))

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(gan.netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(gan.netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    train(args, gan, dataloader, criterion, optimizerD, optimizerG, device)


if __name__ == '__main__':
    main()