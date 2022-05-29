from csv import writer
from typing import Any, List
import torch

import torch.optim as optim
import torchvision.utils as vutils
import functools
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from src.modeling.utils import save_checkpoint
from tqdm import tqdm

from src.modeling.utils import save_checkpoint
from src.modeling.gan import GAN
from src.modeling.train_logger import TBWritter, tb_write_fid, tb_write_metrics
from src.utils import plot_real_vs_fake_imgs
from src.inception_utils import sample_gema, prepare_inception_metrics

def train(
        args,
        gan: GAN,
        dataloader: DataLoader,
        criterion: _Loss,
        optimizerD: optim.Optimizer,
        optimizerG: optim.Optimizer,
        device):
    '''
        Trains a GAN network on a specific dataset loaded by dataloader
    '''
    # Lazily initialized tensorboard writter
    tbwriter = TBWritter(args)

    print_training_args(args)

    get_inception_metrics = prepare_inception_metrics(args.inception, False)
    sample_fn = functools.partial(sample_gema, g_ema=gan.netG, device=device, nz=args.nz, batch_size=args.batch_size)

    # Create batch of latent vectors that we will use to visualize the progression of the generator
    fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Lists to keep track of progress
    img_list = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in tqdm(range(args.num_epochs), desc='Epochs'):
        # For each batch in the dataloader
        for i, data in tqdm(enumerate(dataloader, 0), total=len(dataloader), desc='Batch', leave=False):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################

            ## Train with all-real batch
            gan.netD.zero_grad()

            # Format batch
            real = data[0].to(device)
            b_size = real.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            # Forward pass real batch through D
            output = gan.netD(real).view(-1)

            # Calculate loss on all-real batch
            errD_real = criterion(output, label)

            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, args.nz, 1, 1, device=device)

            # Generate fake image batch with G
            fake = gan.netG(noise)
            label.fill_(fake_label)

            # Classify all fake batch with D
            output = gan.netD(fake.detach()).view(-1)

            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)

            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake

            # Update D
            if "ExtraAdam" == args.optim:
                if i % 2 == 0:
                    optimizerD.extrapolation()
                else:
                    optimizerD.step()
            else:
                optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################

            gan.netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost

            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = gan.netD(fake).view(-1)

            # Calculate G's loss based on this output
            errG = criterion(output, label)

            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()

            # Update G
            if "ExtraAdam" == args.optim:
                if i % 2 == 0:
                    optimizerG.extrapolation()
                else:
                    optimizerG.step()
            else:
                optimizerG.step()

            # Output training stats
            if (iters % args.nb_log_steps == 0) or ((epoch == args.num_epochs-1) and (i == len(dataloader)-1)):
                tb_write_metrics(args, tbwriter, errD.item(), errG.item(), epoch, i, iters, len(dataloader), D_x, D_G_z1, D_G_z2)

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % args.nb_fid_log_steps == 0) or ((epoch == args.num_epochs-1) and (i == len(dataloader)-1)):
                IS_mean, IS_std, FID = get_inception_metrics(sample_fn, num_inception_images=10000, use_torch=False)

                tb_write_fid(args, tbwriter, IS_mean, IS_std, FID, epoch, i, iters, len(dataloader))
                with torch.no_grad():
                    fake = gan.netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

        # Save per epoch
        save_checkpoint(gan, args, epoch, iters)

    # Print real and fake images
    real_batch = next(iter(dataloader))
    plot_real_vs_fake_imgs(args, real_batch, img_list, device)


def print_training_args(args):
    print("~~ Training Argumnets ~~")
    print(f"\tEpochs:{args.num_epochs}")
    print(f"\tlr :{args.lr}")
    print(f"\tAdam beta:{args.beta1}")
    print(f"\tLogging every :{args.nb_log_steps} steps")
    print(f"\tGan args: z_size={args.nz}, g_feats_size={args.ngf}, d_feats_size={args.ndf}")